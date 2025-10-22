import json
import math
import os
import pickle as pk
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, Literal, Optional, Sequence

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_dataloader,
    build_mapping_dataset,
)
from veomni.data.data_collator import DataCollator
from veomni.data.multimodal.preprocess import conv_preprocess
from veomni.data.multimodal.video_utils import fetch_videos
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.dit_trainer import DiTBaseTrainer, DiTTrainerRegistry
from veomni.models import build_foundation_model
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


@dataclass
class MyModelArguments(ModelArguments):
    lora_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for lora."},
    )
    trainer_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for trainer."},
    )


@dataclass
class MyDataArguments(DataArguments):
    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input."},
    )
    offline_embedding_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save offline embeddings."},
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Whether or not to shuffle the dataset."},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    save_initial_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the initial model."},
    )
    hf_weights_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to hf weights."},
    )
    training_task: Literal["offline_training", "online_training", "offline_embedding"] = field(
        default="online_training",
        metadata={
            "help": "Training task. Offline_training: training offline_embeded data. Online training: training raw data online. Offline_embedding: embedding raw data for offline training."
        },
    )


@dataclass
class Arguments:
    model: "MyModelArguments" = field(default_factory=MyModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


def process_online_example(example, processor, source_name, **kwargs):
    prompts = example["inputs"]  # TODO: maybe image in inputs
    prompts = conv_preprocess(source=source_name, conversation=prompts, **kwargs)
    video_info = example["outputs"][0]  # TODO: multi video or sth else

    if kwargs.get("use_audio_in_video", True):
        raise NotImplementedError("Audio in video is not supported yet for dit training.")
    video_inputs, _ = fetch_videos([video_info["video_bytes"].encode("latin-1")], **kwargs)

    processed_example = processor.preprocess(prompts, video_inputs)
    return [processed_example]


def process_offline_example(example, **kwargs):
    processed_example = {key: pk.loads(value) for key, value in example.items()}
    return [processed_example]


@dataclass
class DiTDataCollator(DataCollator):
    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            batch[key] = torch.cat(batch[key], dim=0)

        return batch


def main():
    dist.init_process_group(backend="nccl")
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
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

    logger.info_rank0("Prepare trainier")
    trainer_config = args.model.trainer_config

    build_foundation_model_func = partial(
        build_foundation_model,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )

    build_parallelize_model_func = partial(
        build_parallelize_model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    if args.train.training_task == "offline_embedding":
        assert args.train.micro_batch_size == 1, "Micro batch size must be 1 for offline embedding."
        assert args.train.ulysses_parallel_size == 1, "Ulysses parallel size must be 1 for offline embedding."

    trainer: DiTBaseTrainer = DiTTrainerRegistry.create(
        model_path=args.model.model_path,
        lora_config=args.model.lora_config,
        build_foundation_model_func=build_foundation_model_func,
        build_parallelize_model_func=build_parallelize_model_func,
        training_task=args.train.training_task,
        **trainer_config,
    )

    # model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")

    if trainer.training_task == "offline_training":
        # TODO: do not drop last
        transform = process_offline_example
    else:
        transform = partial(
            process_online_example,
            processor=trainer.processor,
            **args.data.mm_configs,
        )

    if trainer.training_task == "offline_embedding":
        drop_last = False
        shuffle = False
        logger.info_rank0(f"Task offline_embedding. Drop last: {drop_last}, shuffle: {shuffle}")
    else:
        drop_last = args.data.drop_last
        shuffle = args.data.shuffle

    train_dataset = build_mapping_dataset(args.data.train_path, transform=transform, source_name=args.data.source_name)
    if not drop_last:
        dataset_length = (
            math.ceil(len(train_dataset) / (args.train.dataloader_batch_size * args.train.data_parallel_size))
            * args.train.dataloader_batch_size
        )
    else:
        dataset_length = len(train_dataset) / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    if trainer.training_task == "offline_embedding":
        assert args.train.micro_batch_size == 1, "Micro batch size must be 1 for offline embedding."
        assert args.train.ulysses_parallel_size == 1, "Ulysses parallel size must be 1 for offline embedding."

        if args.data.offline_embedding_save_dir is None:
            offline_embedding_save_dir = f"{args.data.train_path}_offline"
        else:
            offline_embedding_save_dir = args.data.offline_embedding_save_dir

        assert not drop_last
        assert not shuffle

        base = len(train_dataset) // args.train.data_parallel_size
        extra = len(train_dataset) % args.train.data_parallel_size
        extra_for_rank = max(0, min(1, extra - args.train.local_rank))  # unshuffled distributed sampler
        valid_data_length = base + extra_for_rank
        logger.info(f"Rank {args.train.global_rank} data length to save: {valid_data_length}")
        trainer.offline_embedding_saver.lazy_init(
            save_path=offline_embedding_save_dir,
            dataset_length=valid_data_length,
        )

        # pad dataset_len
        train_dataset.data_len = (
            math.ceil(train_dataset.data_len / (args.train.global_batch_size)) * args.train.global_batch_size
        )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=drop_last,  # TODO: offline embedding 不 droplast
        shuffle=shuffle,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
        collate_fn=[DiTDataCollator()],
    )
    if args.train.save_initial_model:
        if args.train.global_rank == 0:
            trainer.save_model_weights(args.train.output_dir)
        dist.barrier()
        return

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None

    if trainer.training_task != "offline_embedding":
        model = trainer.get_model_for_training()
        optimizer = build_optimizer(
            model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer,
            no_decay_modules=args.train.no_decay_modules,
            no_decay_params=args.train.no_decay_params,
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
            trainer.save_model_assets(args.train.model_assets_dir)

        if args.train.load_checkpoint_path is not None:
            state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
            Checkpointer.load(args.train.load_checkpoint_path, state)
            global_step = state["extra_state"]["global_step"]
            start_epoch = global_step // args.train.train_steps
            start_step = global_step % args.train.train_steps
            lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])

            if state["extra_state"].get("train_dataloader", None) is not None:
                train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])

            if start_step == 0:  # resume at the end of epoch
                iter(train_dataloader)  # clear resume state and prefetch data

            dist.barrier()
            logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

        helper.empty_cache()

        model.train()
        logger.info(
            f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
        )
    else:
        args.train.num_train_epochs = 1
        logger.info(
            f"rank{args.train.local_rank} Start offline embedding, steps with dummy data: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
        )

    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )

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
                micro_batches = next(data_iterator)
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
                    loss = trainer.forward(**micro_batch)

                if trainer.training_task != "offline_embedding":
                    loss = loss / len(micro_batches)
                    with model_bwd_context:
                        loss.backward()

                    total_loss += loss.item()

                del micro_batch

            if trainer.training_task != "offline_embedding":
                if args.train.data_parallel_mode == "fsdp1":
                    grad_norm = model.clip_grad_norm_(args.train.max_grad_norm).item()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.train.max_grad_norm, foreach=True
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if hasattr(grad_norm, "full_tensor"):
                    grad_norm = grad_norm.full_tensor().item()

                # collect mean loss across data parallel group
                total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
                torch.cuda.synchronize()
                lr = max(lr_scheduler.get_last_lr())
                train_metrics = {}

                data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")

            data_loader_tqdm.update()

            if trainer.training_task != "offline_embedding":
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

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if trainer.training_task != "offline_embedding":
            if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    if trainer.training_task != "offline_embedding":
        torch.cuda.synchronize()
        # release memory
        del optimizer, lr_scheduler
        helper.empty_cache()
        # save model in huggingface's format
        if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
            pass
            # TODO: trainer.save_hf_weights
            if args.train.hf_weights_path:
                hf_weights_path = args.train.hf_weights_path
            else:
                hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
            model_state_dict = ckpt_to_state_dict(
                save_checkpoint_path=save_checkpoint_path,
                output_dir=args.train.output_dir,
                ckpt_manager=args.train.ckpt_manager,
            )
            trainer.save_hf_model_weights(model_state_dict, hf_weights_path)
    else:
        trainer.offline_embedding_saver.save_last()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
