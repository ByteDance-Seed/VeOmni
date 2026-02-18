import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

from tasks.data.vlm_data_process import process_sample_qwen_omni
from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.checkpoint import build_checkpointer
from veomni.data import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
    build_dataloader,
    build_dataset,
)
from veomni.data.constants import IGNORE_INDEX
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_processor, save_model_assets
from veomni.models.transformers.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.model_utils import pretty_print_trainable_parameters
from veomni.utils.save_safetensor_utils import save_hf_safetensor
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids


logger = helper.create_logger(__name__)


@dataclass
class MyDataArguments(DataArguments):
    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())

    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    helper.enable_high_precision_for_bf16()

    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    # Gradient checkpointing debug
    torch.utils.checkpoint.set_checkpoint_debug_enabled(args.train.debug_gradient_checkpointing)

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
    )

    logger.info_rank0("Prepare model")
    model: Qwen2_5OmniForConditionalGeneration = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        init_device=args.train.init_device,
        attn_implementation=args.model.attn_implementation,
    )
    model.disable_talker()
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")
    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path)
    position_id_func = model.thinker.get_position_id_func()
    transform = partial(
        process_sample_qwen_omni,
        processor=processor,
        position_id_func=position_id_func,
        **args.data.mm_configs,
    )

    if args.train.rmpad:
        raise ValueError("Qwen2_5_omni does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []
    if args.train.rmpad_with_pos_ids:
        data_collate_fn.append(
            OmniDataCollatorWithPacking(
                packing_features=[
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "position_ids",
                    "image_mask",
                    "video_mask",
                    "audio_mask",
                ],
                concat_features=[
                    "pixel_values",
                    "image_grid_thw",
                    "pixel_values_videos",
                    "video_grid_thw",
                    "input_features",
                    "audio_feature_lengths",
                ],
            )
        )
    else:
        data_collate_fn.append(
            OmniDataCollatorWithPadding(
                concat_features={
                    "pixel_values": 0,
                    "image_grid_thw": 0,
                    "pixel_values_videos": 0,
                    "video_grid_thw": 0,
                    "input_features": 0,
                    "audio_feature_lengths": 0,
                },
                padding_features={
                    "input_ids": 0,
                    "attention_mask": 0,
                    "labels": IGNORE_INDEX,
                    "position_ids": 0,
                    "image_mask": False,
                    "video_mask": False,
                    "audio_mask": False,
                },
            )
        )
    if get_parallel_state().sp_enabled:
        data_collate_fn.append(
            OmniSequenceShardCollator(
                sp_slice_features={
                    "input_ids": -1,
                    "labels": -1,
                    "pixel_values": 0,
                    "pixel_values_videos": 0,
                    "input_features": 0,
                },
                padding_features={
                    "input_ids": 0,
                    "attention_mask": 0,
                    "labels": IGNORE_INDEX,
                    "position_ids": 0,
                    "pixel_values": 0,
                    "pixel_values_videos": 0,
                    "input_features": 0,
                    "image_mask": False,
                    "video_mask": False,
                    "audio_mask": False,
                },
                padding_scale={
                    "pixel_values": 4,
                    "pixel_values_videos": 4,
                    "input_features": 1,
                },
                rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            )
        )

    train_dataset = build_dataset(
        dataset_name=args.data.dataset_name,
        transform=transform,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        **asdict(args.data),
    )
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping" and dataset_length is not None:
        dataset_length = dataset_length / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    train_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader_type,
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        dyn_bsz=args.train.dyn_bsz,
        pad_packed_to_length=args.train.pad_packed_to_length,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
        collate_fn=data_collate_fn,
    )

    fsdp_kwargs = {}
    model.thinker.audio_tower.requires_grad_(False)
    model.thinker.audio_tower.proj.requires_grad_(True)
    model.thinker.visual.requires_grad_(False)
    model.thinker.visual.merger.requires_grad_(False)
    if args.train.data_parallel_mode == "fsdp1":
        fsdp_kwargs["use_orig_params"] = True

    pretty_print_trainable_parameters(model)

    model = build_parallelize_model(
        model,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_reshard_after_forward=args.train.enable_reshard_after_forward,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )
    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=False,
        optimizer_type=args.train.optimizer,
    )
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

    model_assets = None
    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        # save model_assets before training
        model_assets = [model_config, processor]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
        empty_cache_steps=args.train.empty_cache_steps,
        gc_steps=args.train.gc_steps,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        if args.train.global_rank == 0:
            helper.load_step2token(args.train.load_checkpoint_path)
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info_rank0("Start training")
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
            synchronize()
            start_time = time.time()
            num_micro_steps = len(micro_batches)

            for micro_step, micro_batch in enumerate(micro_batches):
                if (
                    args.train.data_parallel_mode == "fsdp2"
                    and not args.train.enable_reshard_after_backward
                    and num_micro_steps > 1
                ):
                    if micro_step == 0:
                        model.set_reshard_after_backward(False)
                    elif micro_step == num_micro_steps - 1:
                        model.set_reshard_after_backward(True)
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)
                    micro_batch.pop("source_name", None)

                # Prepare flash attention kwargs from position_ids for both Qwen2.5-VL and Qwen3-VL
                (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                    micro_batch["position_ids"][:, 0, :]
                )
                micro_batch.update(
                    dict(
                        cu_seq_lens_q=cu_seq_lens_q,
                        cu_seq_lens_k=cu_seq_lens_k,
                        max_length_q=max_length_q,
                        max_length_k=max_length_k,
                    )
                )

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss / len(micro_batches)
                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            grad_norm = veomni_clip_grad_norm(model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                if args.train.global_rank == 0:
                    helper.save_step2token(
                        args.train.step2token_path,
                        consumed_tokens=train_metrics["consume_tokens(B)"],
                        global_step=global_step,
                        save_checkpoint_path=save_checkpoint_path,
                    )
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            if args.train.global_rank == 0:
                helper.save_step2token(
                    args.train.step2token_path,
                    consumed_tokens=train_metrics["consume_tokens(B)"],
                    global_step=global_step,
                    save_checkpoint_path=save_checkpoint_path,
                )
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        save_hf_safetensor(
            save_hf_safetensor_path=hf_weights_path,
            ckpt_manager=args.train.ckpt_manager,
            model_assets=model_assets,
            train_architecture=args.train.train_architecture,
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            is_rank_0=args.train.global_rank == 0,
            model=model,
            fqn_to_index_mapping=args.model.fqn_to_index_mapping,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
