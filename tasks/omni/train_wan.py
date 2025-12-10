import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data.diffusion.data_loader import build_dit_dataloader
from veomni.data.diffusion.dataset import build_tensor_dataset
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import (
    build_foundation_model,
    save_model_assets,
)
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.schedulers.flow_match import FlowMatchScheduler
from veomni.utils import helper
from veomni.utils.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    parse_args,
    save_args,
)
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.dit_utils import EnvironMeter, save_model_weights
from veomni.utils.lora_utils import add_lora_to_model, freeze_parameters
from veomni.utils.recompute_utils import convert_ops_to_objects


logger = helper.create_logger(__name__)


@dataclass
class MyDataArguments(DataArguments):
    datasets_repeat: int = field(
        default=1,
        metadata={"help": "The number of times to repeat the datasets."},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    save_initial_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the initial model."},
    )
    ops_to_save: List[str] = field(
        default_factory=list,
        metadata={"help": "Ops to save."},
    )


@dataclass
class Arguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    data: MyDataArguments = field(default_factory=MyDataArguments)
    train: MyTrainingArguments = field(default_factory=MyTrainingArguments)


def get_param_groups(model: torch.nn.Module, default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual" in name:
            vit_params.append(param)
        else:
            other_params.append(param)
    return [
        {"params": vit_params, "lr": vit_lr},
        {"params": other_params, "lr": default_lr},
    ]


def main():
    args = parse_args(Arguments)
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode,
        ckpt_manager=args.train.ckpt_manager,
    )

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
    logger.info_rank0(
        f"Parallel state: dp:{args.train.data_parallel_mode}, tp:{args.train.tensor_parallel_size}, ep:{args.train.expert_parallel_size}, pp:{args.train.pipeline_parallel_size}, cp:{args.train.context_parallel_size}, ulysses:{args.train.ulysses_parallel_size}"
    )

    if args.data.data_type == "diffusion":
        train_dataset = build_tensor_dataset(
            base_path=args.data.train_path,
            metadata_path=os.path.join(args.data.train_path, "metadata.csv"),
            datasets_repeat=args.data.datasets_repeat,
        )
        args.train.compute_train_steps(
            args.data.max_seq_len,
            args.data.train_size,
            len(train_dataset) // args.train.data_parallel_size,
        )

        train_dataloader = build_dit_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            train_steps=args.train.train_steps,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        init_device=args.train.init_device,
        torch_dtype="bfloat16",
        attn_implementation=args.model.attn_implementation,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    model.micro_batch_size = args.train.micro_batch_size

    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    lora_target_modules_support = ["q", "k", "v", "o", "ffn.0", "ffn.2"]
    if args.train.train_architecture == "lora":
        logger.info_rank0("train_architecture is lora")
        _use_orig_params = True
        freeze_parameters(model)
        add_lora_to_model(
            model,
            lora_rank=args.model.lora_rank,
            lora_alpha=args.model.lora_alpha,
            lora_target_modules=args.model.lora_target_modules,
            init_lora_weights=args.model.init_lora_weights,
            pretrained_lora_path=args.model.pretrained_lora_path,
            lora_target_modules_support=lora_target_modules_support,
        )
        model.to(torch.bfloat16)
    else:
        logger.info_rank0("train_architecture is full")
        _use_orig_params = False

    logger.info_rank0(f"model: {model}")

    if args.train.save_initial_model:
        if args.train.global_rank == 0:
            state_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
            save_model_weights(args.train.output_dir, model.state_dict(), model_assets=[model_config])

        dist.barrier()
        return

    ops_to_save = convert_ops_to_objects(args.train.ops_to_save)
    model = build_parallelize_model(
        model,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        use_orig_params=_use_orig_params,
        ops_to_save=ops_to_save,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        param_groups=get_param_groups(model, args.train.lr, args.train.vit_lr),
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        model_assets = [model_config]
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

    # Build diffusion scheduler: FlowMatchScheduler
    flow_scheduler = FlowMatchScheduler(
        shift=5,
        sigma_min=0.0,
        extra_one_step=True,
    )

    total_train_steps = args.train.train_steps * args.train.num_train_epochs
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=total_train_steps,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
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
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()

    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload,
        args.train.enable_gradient_checkpointing,
        args.train.activation_gpu_limit,
    )

    helper.empty_cache()
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )

    flow_scheduler.set_timesteps(1000, training=True)

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
        epoch_start_time = time.time()
        data_iterator = iter(train_dataloader)

        epoch_loss = 0
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            synchronize()
            total_loss = 0
            start_time = time.time()
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch, model_type="wan")
                latents = micro_batch["latents"].to(model.device)
                prompt_emb = micro_batch["prompt_emb"]
                if args.train.micro_batch_size > 1:
                    prompt_emb["context"] = prompt_emb["context"].squeeze(1).to(model.device)
                    image_emb = micro_batch["image_emb"]
                    if "clip_feature" in image_emb:
                        image_emb["clip_feature"] = image_emb["clip_feature"].squeeze(1).to(model.device)
                    if "y" in image_emb:
                        image_emb["y"] = image_emb["y"].squeeze(1).to(model.device)
                else:
                    prompt_emb["context"] = prompt_emb["context"][0].to(model.device)
                    image_emb = micro_batch["image_emb"]
                    if "clip_feature" in image_emb:
                        image_emb["clip_feature"] = image_emb["clip_feature"][0].to(model.device)
                    if "y" in image_emb:
                        image_emb["y"] = image_emb["y"][0].to(model.device)

                noise = torch.randn_like(latents)
                timestep_id = torch.randint(0, flow_scheduler.num_train_timesteps, (latents.size(0),))
                timestep = flow_scheduler.timesteps[timestep_id].to(latents.dtype).to(latents.device)
                # noise and target
                noisy_latents = flow_scheduler.add_noise(
                    latents,
                    noise,
                    timestep,
                    args.train.micro_batch_size,
                    args.train.enable_mixed_precision,
                )
                training_target = flow_scheduler.training_target(latents, noise, timestep)
                # predict noise
                with model_fwd_context:
                    noise_pred = model.forward(
                        noisy_latents,
                        timestep=timestep,
                        **prompt_emb,
                        **image_emb,
                    )
                    # MSE loss with weights
                    loss = F.mse_loss(noise_pred.float(), training_target.float(), reduction="none")
                    weight = flow_scheduler.training_weight(timestep, args.train.micro_batch_size)
                    # shape: [B, ...], weight: [B]
                    loss = (loss.view(latents.size(0), -1).mean(dim=1) * weight).mean() / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            grad_norm = veomni_clip_grad_norm(model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            epoch_loss += total_loss
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}, step_time: {delta_time:.2f}s"
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
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                if args.train.global_rank == 0:
                    save_hf_weights(args, save_checkpoint_path, model_assets)

        data_loader_tqdm.close()
        epoch_time = time.time() - epoch_start_time
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.global_rank == 0:
            logger.info_rank0(
                f"Epoch {epoch + 1} completed, epoch_time={epoch_time:.4f}s, epoch_loss={epoch_loss / args.train.train_steps:.4f}"
            )
        if args.train.global_rank == 0:
            if args.train.use_wandb:
                wandb.log({"training/loss_per_epoch": epoch_loss / args.train.train_steps}, step=global_step)
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
            if args.train.global_rank == 0:
                save_hf_weights(args, save_checkpoint_path, model_assets)

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()

    dist.barrier()
    dist.destroy_process_group()


def save_hf_weights(args, save_checkpoint_path, model_assets):
    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=save_checkpoint_path,
        output_dir=args.train.output_dir,
        ckpt_manager=args.train.ckpt_manager,
    )
    if args.train.train_architecture == "lora":
        model_state_dict = {k: v for k, v in model_state_dict.items() if "lora" in k}
    save_model_weights(
        hf_weights_path,
        model_state_dict,
        model_assets=model_assets,
    )
    logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")


if __name__ == "__main__":
    main()
