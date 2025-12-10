import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import wandb
from torch import distributed as dist
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
    build_dataloader,
    build_dataset,
    build_multimodal_chat_template,
)
from veomni.data.constants import IGNORE_INDEX
from veomni.data.multimodal.multimodal_transform import encode_multimodal_sample
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import save_model_assets, save_model_weights
from veomni.models.seed_omni import SeedOmniModel, build_omni_model, build_omni_processor
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.utils.device import get_device_type, get_torch_device, synchronize
from veomni.utils.dist_utils import all_reduce
from veomni.utils.model_utils import pretty_print_trainable_parameters


logger = helper.create_logger(__name__)

MAX_PIXELS = 768 * 28 * 28


@dataclass
class MyDataArguments(DataArguments):
    max_image_nums: Optional[int] = field(
        default=None,
        metadata={"help": "The max number of images in the sample."},
    )
    max_pixels: int = field(
        default=MAX_PIXELS,
        metadata={"help": "The max pixel numbers of the image."},
    )
    max_pixel_size: int = field(
        default=None,
        metadata={"help": "The max pixel size (height/width) of the image."},
    )
    scale_factor: int = field(
        default=None,
        metadata={"help": "Scale factor of the input image."},
    )
    generation_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for instruction tuning data change to generation data."},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the encoder parameters."},
    )
    freeze_encoder_all: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the whole encoder parameters."},
    )
    freeze_decoder: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the decoder parameters."},
    )
    freeze_foundation: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the foundation parameters."},
    )
    save_initial_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the initial model."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group()
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

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
    model: SeedOmniModel = build_omni_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        encoders=args.model.encoders,
        decoders=args.model.decoders,
        input_encoder=args.model.input_encoder,
        output_encoder=args.model.output_encoder,
        init_device=args.train.init_device,
    )

    logger.info_rank0("Prepare data")
    processor = build_omni_processor(
        config_path=args.model.config_path,
        tokenizer_path=args.model.tokenizer_path,
        encoders=args.model.encoders,
        decoders=args.model.decoders,
        input_encoder=args.model.input_encoder,
        output_encoder=args.model.output_encoder,
        encode_target=args.model.encode_target,
        max_pixels=args.data.max_pixels,
    )
    chat_template = build_multimodal_chat_template(args.data.chat_template, processor.tokenizer)
    position_id_func = model.get_position_id_func()
    modality_info = model.get_modality()
    transform = partial(
        encode_multimodal_sample,
        processor=processor,
        chat_template=chat_template,
        position_id_func=position_id_func,
        modality_info=modality_info,
        max_image_nums=args.data.max_image_nums,
        max_pixel_size=args.data.max_pixel_size,
        generation_ratio=args.data.generation_ratio,
        scale_factor=args.data.scale_factor,
    )

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []

    # TODO: config by model
    if args.train.rmpad_with_pos_ids:
        data_collate_fn.append(
            OmniDataCollatorWithPacking(
                packing_features=[
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "position_ids",
                    "image_input_mask",
                    "image_output_mask",
                ],
                concat_features=[
                    "image_input_features",
                    "image_input_grid_thw",
                    "image_output_features",
                    "image_output_grid_thw",
                ],
            )
        )
    else:
        data_collate_fn.append(
            OmniDataCollatorWithPadding(
                concat_features={
                    "image_input_features": 0,
                    "image_input_grid_thw": 0,
                    "image_output_features": 0,
                    "image_output_grid_thw": 0,
                },
                padding_features={
                    "input_ids": 0,
                    "attention_mask": 0,
                    "labels": IGNORE_INDEX,
                    "position_ids": 0,
                    "image_input_mask": False,
                    "image_output_mask": False,
                },
            )
        )
    if get_parallel_state().sp_enabled:
        data_collate_fn.append(
            OmniSequenceShardCollator(
                sp_slice_features={
                    "input_ids": -1,
                    "labels": -1,
                    "image_input_features": 0,
                    "image_output_features": 0,
                },
                padding_features={
                    "input_ids": 0,
                    "attention_mask": 0,
                    "labels": IGNORE_INDEX,
                    "position_ids": 0,
                    "image_input_features": 0,
                    "image_output_features": 0,
                    "image_input_mask": False,
                    "image_output_mask": False,
                },
                padding_scale={
                    "image_input_features": 4,
                    "image_output_features": 1,
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
    if args.data.datasets_type == "mapping":
        dataset_length = dataset_length / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    train_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader_type,
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        collate_fn=data_collate_fn,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )

    freeze_any = False
    if args.train.freeze_encoder:
        if args.train.freeze_encoder_all:
            model.encoder.requires_grad_(False)
        else:
            model.encoder.set_projector_trainable_only()
        freeze_any = True
    if args.train.freeze_decoder:
        model.decoder.set_projector_trainable_only()
        freeze_any = True
    if args.train.freeze_foundation:
        model.foundation.requires_grad_(False)
        freeze_any = True
    pretty_print_trainable_parameters(model)

    fsdp_kwargs = {}
    if freeze_any:
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    if args.train.save_initial_model:
        if args.train.global_rank == 0:
            save_model_weights(args.train.output_dir, model.state_dict(), model_assets=[model_config, processor])

        dist.barrier()
        return

    model = build_parallelize_model(
        model,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
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

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        model_assets = [model_config, processor]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler_state = state["extra_state"]["lr_scheduler"]
        lr_scheduler.load_state_dict(lr_scheduler_state)
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        del state
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
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
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            total_losses = defaultdict(int)
            synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    model_outputs = model(**micro_batch, use_cache=False)

                loss: "torch.Tensor" = model_outputs.loss / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                losses = model_outputs.losses
                for key, v in losses.items():
                    total_losses[key] += v / len(micro_batches)

                del micro_batch

            grad_norm = veomni_clip_grad_norm(model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            for key, v in total_losses.items():
                total_losses[key] = all_reduce((v), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time=delta_time, global_step=global_step)

            step_info = total_losses
            step_info.update(
                {
                    "loss": total_loss,
                    "grad_norm": grad_norm,
                    "lr": lr,
                }
            )
            data_loader_tqdm.set_postfix_str({k: f"{v:.2f}" for k, v in step_info.items() if k != "lr"})
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update({f"training/{k}": v for k, v in step_info.items()})
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()
                    helper.upload_trace(args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir)

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
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
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
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {args.train.output_dir} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
