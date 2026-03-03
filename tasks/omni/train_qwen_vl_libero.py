import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments, parse_args, save_args
from veomni.checkpoint import build_checkpointer
from veomni.data import (
    LiberoActionCollator,
    LiberoActionPackingCollator,
    build_dataloader,
)
from veomni.data.dataset import MappingDataset
from veomni.data.multimodal.data_transform import (
    load_libero_task_descriptions,
    process_libero_sample_qwen3_vl,
)
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_processor, save_model_assets
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.save_safetensor_utils import save_hf_safetensor
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids


if TYPE_CHECKING:
    pass

logger = helper.create_logger(__name__)


def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]


@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )
    rmpad: bool = field(
        default=False,
        metadata={"help": "Whether to remove padding tokens."},
    )
    rmpad_with_pos_ids: bool = field(
        default=False,
        metadata={"help": "Whether to remove padding using position IDs."},
    )
    pad_packed_to_length: bool = field(
        default=False,
        metadata={"help": "Whether to pad packed sequences to max length."},
    )
    dyn_bsz_margin: float = field(
        default=0.0,
        metadata={"help": "Margin for dynamic batch sizing."},
    )


@dataclass
class MyDataArguments(DataArguments):
    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input."},
    )
    libero_data_dir: str = field(
        default="",
        metadata={"help": "Root directory of the LIBERO dataset."},
    )
    libero_prompt_template: str = field(
        default="Predict the next actions for the robot task: {task}",
        metadata={"help": "Prompt template for LIBERO task descriptions. Must contain {task}."},
    )
    obs_len: int = field(
        default=1,
        metadata={"help": "Number of observation frames (including anchor)."},
    )
    pred_len: int = field(
        default=4,
        metadata={"help": "Number of future prediction frames."},
    )
    chunk_index: Optional[int] = field(
        default=None,
        metadata={"help": "Chunk index for multi-chunk datasets. None loads all chunks."},
    )
    libero_dataset_backend: str = field(
        default="youmu",
        metadata={"help": "Dataset backend for LIBERO data. Options: youmu, lerobot, lance."},
    )
    libero_lance_dir: str = field(
        default="",
        metadata={"help": "Path to Lance dataset directory (only used when libero_dataset_backend=lance)."},
    )


def build_libero_dataset(
    backend: str,
    data_dir: str,
    obs_len: int,
    pred_len: int,
    chunk_index: Optional[int] = None,
    lance_dir: str = "",
    meta_path: str = "",
):
    """Build a LIBERO dataset using the specified backend.

    Args:
        backend: One of "youmu", "lerobot", "lance".
        data_dir: Root directory of the LIBERO dataset.
        obs_len: Number of observation frames (including anchor).
        pred_len: Number of future prediction frames.
        chunk_index: Chunk index for multi-chunk datasets. None loads all chunks.
        lance_dir: Path to Lance dataset directory (lance backend only).
        meta_path: Path to episode metadata parquet file (lance backend only).

    Returns:
        A PyTorch Dataset instance.
    """
    if backend == "youmu":
        from youmu.libero_dataset import LiberoYoumuDataset

        return LiberoYoumuDataset(
            data_dir=data_dir,
            obs_len=obs_len,
            pred_len=pred_len,
            chunk_index=chunk_index,
        )
    elif backend == "lerobot":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        info_path = os.path.join(data_dir, "meta", "info.json")
        with open(info_path) as f:
            info = json.load(f)
        fps = info["fps"]

        obs_timestamps = [-(obs_len - 1 - i) / fps for i in range(obs_len)]
        pred_timestamps = [i / fps for i in range(pred_len)]

        return LeRobotDataset(
            repo_id="local/libero",
            root=data_dir,
            delta_timestamps={
                "observation.state": obs_timestamps,
                "action": pred_timestamps,
                "observation.images.image": obs_timestamps,
            },
            download_videos=False,
        )
    elif backend == "lance":
        import importlib.util

        if not lance_dir:
            raise ValueError("libero_lance_dir must be set when using lance backend.")

        # Import LiberoLanceDataset from youmu's benchmark scripts
        lance_baseline_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "submodules",
            "youmu",
            "scripts",
            "benchmarks",
            "lance_baseline.py",
        )
        spec = importlib.util.spec_from_file_location("lance_baseline", lance_baseline_path)
        lance_baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lance_baseline)

        return lance_baseline.LiberoLanceDataset(
            lance_dir=lance_dir,
            meta_path=meta_path,
            obs_len=obs_len,
            pred_len=pred_len,
        )
    else:
        raise ValueError(f"Unknown libero_dataset_backend: {backend}. Must be one of: youmu, lerobot, lance.")


@dataclass
class Arguments(VeOmniArguments):
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


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
        async_enabled=args.train.async_enabled,
    )

    logger.info_rank0("Prepare model")
    # Build Qwen3VLForConditionalGenerationAction directly instead of the
    # generic build_foundation_model, which would instantiate the base
    # Qwen3VLForConditionalGeneration (language-modelling head) instead of
    # the action-prediction variant.
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    from veomni.models.transformers.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGenerationAction,
        apply_veomni_qwen3vl_patch,
    )

    # Apply VeOmni monkey-patches (dummy_forward, attention, etc.) before model creation
    apply_veomni_qwen3vl_patch()

    model_config = AutoConfig.from_pretrained(
        args.model.config_path or args.model.model_path,
        trust_remote_code=True,
        attn_implementation=args.model.attn_implementation,
    )
    # Set action-prediction config attributes
    model_config.action_dim = args.data.pred_len and 7  # LIBERO action dim
    model_config.pred_len = args.data.pred_len

    with init_empty_weights():
        model = Qwen3VLForConditionalGenerationAction(model_config)

    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path)
    position_id_func = model.get_position_id_func()

    # Load LIBERO task descriptions — auto-detect parquet or JSONL metadata
    libero_dir = args.data.libero_data_dir
    meta_candidates = [
        os.path.join(libero_dir, "meta", "episodes.jsonl"),
        os.path.join(libero_dir, "meta", "episodes", "episodes.parquet"),
        os.path.join(libero_dir, "meta", "episodes", "chunk-000", "file-000.parquet"),
    ]
    meta_path = next((p for p in meta_candidates if os.path.exists(p)), meta_candidates[-1])
    task_descriptions = load_libero_task_descriptions(meta_path)
    logger.info_rank0(f"Loaded {len(task_descriptions)} LIBERO task descriptions from {meta_path}")

    transform = partial(
        process_libero_sample_qwen3_vl,
        processor=processor,
        position_id_func=position_id_func,
        task_descriptions=task_descriptions,
        prompt_template=args.data.libero_prompt_template,
    )

    if args.train.rmpad_with_pos_ids:
        data_collate_fn = LiberoActionPackingCollator()
    else:
        data_collate_fn = LiberoActionCollator()

    if args.train.rmpad:
        raise ValueError("QwenVL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    libero_dataset = build_libero_dataset(
        backend=args.data.libero_dataset_backend,
        data_dir=args.data.libero_data_dir,
        obs_len=args.data.obs_len,
        pred_len=args.data.pred_len,
        chunk_index=args.data.chunk_index,
        lance_dir=args.data.libero_lance_dir,
        meta_path=meta_path,
    )
    logger.info_rank0(f"Using LIBERO dataset backend: {args.data.libero_dataset_backend}")
    train_dataset = MappingDataset(data=libero_dataset, transform=transform)
    dataset_length = len(train_dataset) / args.train.data_parallel_size
    # Compute train steps: dataset_length / dataloader_batch_size, capped by max_steps
    import math

    computed_steps = math.floor(dataset_length / args.train.dataloader_batch_size)
    if args.train.max_steps is not None and computed_steps >= args.train.max_steps:
        computed_steps = args.train.max_steps
    args.train.train_steps = computed_steps

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
        dyn_bsz=args.train.dyn_bsz,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )

    fsdp_kwargs = {}
    if args.train.freeze_vit:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

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
        param_groups=get_param_groups(model, args.train.lr, args.train.vit_lr),
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
                settings=wandb.Settings(console="off"),
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
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
        empty_cache_steps=args.train.empty_cache_steps,
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
                    loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss

                # MSE loss is already a mean; scale for gradient accumulation only.
                loss = loss / num_micro_steps

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

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {lr:.2e}", refresh=False
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
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
