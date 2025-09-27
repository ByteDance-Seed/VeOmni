import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from tqdm import trange

from veomni.checkpoint import build_checkpointer
from veomni.data import build_dummy_dataset, build_streaming_dataloader
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce


"""
torchrun --nnodes=1 --nproc-per-node=8 --master-port=4321 tests/utils/test_trainer_saveload.py \
--model.model_path Qwen/Qwen3-4B \
--train.expert_parallel_size 1 \
--train.global_batch_size 8 \
--train.micro_batch_size 1 \
--data.max_seq_len 128 \
--data.train_path "dummy" \
--train.output_dir ./test_trainer_saveload \
--train.max_steps 5 \
--train.rmpad false \
--train.rmpad_with_pos_ids true \
--train.data_parallel_mode "fsdp2" \
--train.init_device "meta" \
--train.ckpt_manager "dcp"

torchrun --nnodes=1 --nproc-per-node=8 --master-port=4321 tests/utils/test_trainer_saveload.py \
--model.model_path /path/to/Qwen3-30B-A3B-Instruct-2507-merge \
--model.moe_implementation fused \
--model.attn_implementation flash_attention_2 \
--train.expert_parallel_size 4 \
--train.global_batch_size 8 \
--train.micro_batch_size 1 \
--data.max_seq_len 128 \
--data.train_path "dummy" \
--train.output_dir ./test_trainer_saveload \
--train.max_steps 5 \
--train.rmpad false \
--train.rmpad_with_pos_ids true \
--train.data_parallel_mode "fsdp2" \
--train.init_device "meta" \
--train.ckpt_manager "dcp"
"""

# To prevent DCP from complaining "too many open files"
# see: https://github.com/pytorch/pytorch/issues/11201
torch.multiprocessing.set_sharing_strategy("file_system")

logger = helper.create_logger(__name__)


def print_device_mem_info():
    current_memory_allocated = torch.cuda.memory_allocated() / (1024**2)
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**2)

    logger.info_rank0(f"current_memory:{current_memory_allocated:.2f} MB | max_memory:{max_memory_allocated:.2f} MB")


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def check_state_dict(lhs_dict, rhs_dict, need_flatten=False, tied_weight_key: Optional[list[str]] = None):
    if need_flatten:
        lhs_dict = flatten_dict(lhs_dict)
        rhs_dict = flatten_dict(rhs_dict)

    for k, v in rhs_dict.items():
        if "step" in k or "param_groups" in k:
            continue
        if tied_weight_key and k in tied_weight_key:
            logger.info_rank0(f"skipping tied_weights_key: {k}")
            continue

        lhs, rhs = lhs_dict[k], v
        logger.info_rank0(f"checking {k}...")
        # unwrap to local if available
        lhs_val = lhs.to_local() if hasattr(lhs, "to_local") else lhs
        rhs_val = rhs.to_local() if hasattr(rhs, "to_local") else rhs

        torch.testing.assert_close(rhs_val, lhs_val)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    helper.enable_high_precision_for_bf16()

    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
        ep_outside=args.train.ep_outside,
    )

    logger.info_rank0("Prepare data")

    train_data_size = 8192
    train_dataset = build_dummy_dataset(task_type="text", size=train_data_size, max_seq_len=args.data.max_seq_len)

    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
    train_dataloader = build_streaming_dataloader(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        dyn_bsz_runtime=args.train.dyn_bsz_runtime,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        collate_fn=None,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        infinity=True,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
        drop_resume_buffer=args.data.drop_resume_buffer,
    )

    logger.info_rank0("Prepare model")

    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )

    model_config = model.config

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)

    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )
    # a pre_hook which calls update_gate_ema of M8 has been registered on the optimizer
    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        no_decay_modules=args.train.no_decay_modules,
        no_decay_params=args.train.no_decay_params,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    # The reference state dicts to compare with state dicts restored from DCP later
    golden_model_sd, golden_optim_sd = None, None

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            torch.cuda.synchronize()

            for micro_batch in micro_batches:
                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }

                with model_fwd_context:
                    outputs = model(**micro_batch, use_cache=False)
                    loss: "torch.Tensor" = outputs.loss.mean() / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            _gn = model.clip_grad_norm_(args.train.max_grad_norm)
            grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            torch.cuda.synchronize()

            lr = max(lr_scheduler.get_last_lr())

            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        # if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
        # save after first epoch
        if epoch == 0:
            helper.empty_cache()
            golden_model_sd = model.state_dict()
            import copy

            golden_optim_sd = copy.deepcopy(optimizer.state_dict())
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                },
            }

            logger.info_rank0("testing DCP async saving")
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step, save_async=True)

            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    # resume states from checkpoints and compare them with the ones before saving
    state = {"model": model, "optimizer": optimizer, "extra_state": {}}
    subdirs = [d for d in os.listdir(save_checkpoint_path) if d.startswith("global_step_")]
    if subdirs:

        def step_id(s):
            return int(s.split("_")[-1])

        subdir = max(subdirs, key=step_id)
        load_path = os.path.join(save_checkpoint_path, subdir)
    else:
        load_path = save_checkpoint_path
    # wait saving to finish
    if Checkpointer.dcp_save_future is not None:
        logger.info_rank0("Waiting model saving to finish...")
        Checkpointer.dcp_save_future.result()

    Checkpointer.load(load_path, state)
    dist.barrier()

    tied_weights_keys = None
    if hasattr(model, "_tied_weights_keys"):
        tied_weights_keys = model._tied_weights_keys

    check_state_dict(golden_model_sd, model.state_dict(), tied_weights_keys)
    check_state_dict(golden_optim_sd, optimizer.state_dict(), need_flatten=True)

    torch.cuda.synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
