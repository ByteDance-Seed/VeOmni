import json
import os
import pickle as pk
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import trange
from veomni_patch.models.seedream.dit.modules import na
from veomni_patch.models.seedream.dit.modules.config import (
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
    create_training_timesteps_from_config,
)
from veomni_patch.models.seedream.dit.video_nadit import MultiShotNaDiT

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_dataloader,
    build_mapping_dataset,
)
from veomni.data.data_collator import DataCollator
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce
from veomni.utils.model_utils import pretty_print_trainable_parameters


logger = helper.create_logger(__name__)
tasks = ["t2v", "i2v", "v2v", "i2v_last"]
tasks_prob = [0.05, 0.95, 0.0, 0.0]


@dataclass
class MyDataArguments(DataArguments):
    null_text_embedding: Optional[str] = field(
        default="",
        metadata={"help": "Path to null text embedding."},
    )
    text_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dropout rate for text."},
    )


@dataclass
class MyModelArguments(ModelArguments):
    lora_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for lora."},
    )
    sampler_config: Optional[Dict] = field(
        default=None,
        metadata={"help": "Config for sampler."},
    )
    schedule_config: Optional[Dict] = field(
        default=None,
        metadata={"help": "Config for schedule."},
    )
    training_timesteps_config: Optional[Dict] = field(
        default=None,
        metadata={"help": "Config for training timesteps."},
    )
    sampling_timesteps_config: Optional[Dict] = field(
        default=None,
        metadata={"help": "Config for sampling timesteps."},
    )
    loss: Optional[Dict] = field(
        default=None,
        metadata={"help": "Loss config."},
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


@dataclass
class Arguments:
    model: "MyModelArguments" = field(default_factory=MyModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


def process_seedream_offline_example(example, text_null_embedding: torch.Tensor, text_dropout: float = 0.0):
    def _dropout_or_not(emb):
        return emb if random.random() >= text_dropout else text_null_embedding + emb * 0.0

    def get_condition(latent: torch.Tensor, task: str, latent_last: torch.Tensor = None) -> torch.Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            return cond
        if task == "i2v" or task == "i2v_last":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            if task == "i2v_last":
                # i2v generation conditioned on last frame.
                cond[-1:, ..., :-1] = latent_last
                cond[-1:, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        raise NotImplementedError

    def sample_conditions(latents: List[torch.Tensor], latents_last: List[torch.Tensor] = None) -> List[torch.Tensor]:
        task_ids = random.choices(range(len(tasks)), tasks_prob, k=len(latents))
        if latents_last is None:
            latents_last = [None] * len(latents)

        # Rollback to i2v if there is no latent_last
        for i in range(len(task_ids)):
            if tasks[task_ids[i]] == "i2v_last" and latents_last[i] is None:
                task_ids[i] = tasks.index("i2v")

        conds = [
            get_condition(latent, tasks[task_id], latent_last)
            for latent, latent_last, task_id in zip(latents, latents_last, task_ids)
        ]
        return conds, torch.tensor(task_ids)

    assert example.get("is_valid", True)

    processed_example = {}

    latent = pk.loads(example["latent"])
    latent = latent[list(latent.keys())[0]]  # 数据定义好后优化这个 # c, f, h, w

    # TODO: move this to modeling
    latent = rearrange(latent, "c f h w -> f c h w")
    dist = DiagonalGaussianDistribution(latent)
    latent = dist.sample()
    latent = rearrange(latent, "f c h w -> f h w c")
    latent = (latent - torch.tensor(0.012)) * torch.tensor(1.0)
    f, h, w = latent.shape[:3]

    latents = [latent]  # wrap batch size
    latents_cond, task_ids = sample_conditions(latents, None)
    latents, latents_shapes = na.flatten(latents)
    latents_cond, _ = na.flatten(latents_cond)

    # Cast latents to fp32 to avoid error.
    latents = latents.float()
    latents_cond = latents_cond.float()
    processed_example.update(
        {
            "latents": latents,  # (f h w) c
            "latents_shapes": latents_shapes,  # 1, 3
            "latents_cond": latents_cond,  # (f h w) c+1
            "video_task_ids": task_ids,  # 1
        }
    )

    text_emb_dict = pk.loads(example["text_emb"])
    text_key = random.choice(list(text_emb_dict.keys()))
    text_embs, shot_latents_shapes, shot_text_shapes, num_shots = [], [], [], []

    for item in text_emb_dict[text_key]:
        text_emb = item["text_embeds"]
        n_latent_frames = item["n_latent_frames"]
        assert n_latent_frames > 0, f"Invalid shot with n_latent_frames: {n_latent_frames}."
        if len(text_emb) == 1:
            repeats = [n_latent_frames]  # t2v
        else:
            repeats = [1, n_latent_frames - 1]  # i2v
        text_embs.append(torch.cat([_dropout_or_not(item) for item in text_emb]))
        shot_text_shapes.append([len(item) for item in text_emb])
        shot_latents_shapes.extend([[f, h, w] for f in repeats])
        num_shots.append(len(text_emb))
    text_embeds, text_shapes = na.flatten(text_embs)
    num_shots = torch.tensor(num_shots)
    shot_text_shapes = torch.tensor(shot_text_shapes).transpose(0, 1)
    shot_latents_shapes = torch.tensor(shot_latents_shapes)

    processed_example.update(
        {
            "text_embeds": text_embeds,  # l, 3584
            "num_shots": num_shots,  # 1
            "text_shapes": text_shapes,  # 1, 1
            "shot_text_shapes": shot_text_shapes,  # num_shot, 1 (这个有点抽象，在rope里取的是txt_shape[:, 0])
            "shot_latents_shapes": shot_latents_shapes,  # 1, num_shot, 3
        }
    )

    # TODO: latent_last

    return [processed_example]


def timestep_transform(schedule, timesteps: torch.Tensor, latents_shapes: torch.Tensor):
    # Compute resolution.
    # TODO: move to modeling dit, config vae configs
    vt = 4
    vs = 16
    frames = (latents_shapes[:, 0] - 1) * vt + 1
    heights = latents_shapes[:, 1] * vs
    widths = latents_shapes[:, 2] * vs

    # Compute shift factor.
    def get_lin_function(x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    space_shift_fn = get_lin_function(x1=256 * 256, y1=0.5, x2=1024 * 1024, y2=1.15)
    temp_shift_fn = get_lin_function(x1=1, y1=0, x2=121, y2=1.64)
    shift = torch.exp(space_shift_fn(heights * widths) + temp_shift_fn(frames))
    # Shift timesteps.
    # math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    # shift / (shift + 1 / t - 1) = shift * t / (1 + (shift - 1) * t)
    timesteps = timesteps / schedule.T
    timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
    timesteps = timesteps * schedule.T
    return timesteps


def build_lora_model(model: torch.nn.Module, lora_config: Dict):
    lora_adapter_path = lora_config.get("lora_adapter", None)
    if lora_adapter_path is not None:
        logger.info_rank0(f"Load lora_adapter from {lora_adapter_path}.")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_adapter_path)
    else:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_config["rank"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["lora_modules"],
        )
        logger.info_rank0(f"Init lora: {lora_config.to_dict()}.")
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model


@dataclass
class SeedreamDataCollator(DataCollator):
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

    text_null_embedding_path = args.data.null_text_embedding
    text_null_embedding = torch.load(text_null_embedding_path, map_location="cpu", weights_only=True).detach()
    assert text_null_embedding.dtype == torch.bfloat16
    transform = partial(
        process_seedream_offline_example,
        text_null_embedding=text_null_embedding,
        text_dropout=args.data.text_dropout,
    )
    train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
    dataset_length = len(train_dataset) / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

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
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
        collate_fn=[SeedreamDataCollator()],
    )

    logger.info_rank0("Prepare model")
    model: MultiShotNaDiT = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )

    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    fsdp_kwargs = {}
    if args.model.lora_config:
        model = build_lora_model(model, args.model.lora_config)
        fsdp_kwargs["use_orig_params"] = True

    pretty_print_trainable_parameters(model)

    if args.train.save_initial_model:
        if args.train.global_rank == 0:
            if args.model.lora_config is not None:
                if args.model.lora_config.get("save_merge", False):
                    logger.info_rank0(f"Save initial lora_adapter to {args.train.output_dir}.")
                    model.save_pretrained(args.train.output_dir)
                else:
                    logger.info_rank0(f"Save initial lora merged model to {args.train.output_dir}.")
                    model = model.merge_and_unload()
                    model.save_pretrained(args.train.output_dir)
            else:
                save_model_weights(args.train.output_dir, model.state_dict(), model_assets=[model_config])
        dist.barrier()
        return
    # TODO: ema now oom
    # ema = deepcopy(model)
    # ema.requires_grad_(False)
    # pretty_print_trainable_parameters(ema)

    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    # ema = build_parallelize_model(
    #     model,
    #     init_device=args.train.init_device,
    #     weights_path=args.model.model_path,
    #     enable_full_shard=args.train.enable_full_shard,
    #     enable_mixed_precision=args.train.enable_mixed_precision,
    #     enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
    #     enable_fsdp_offload=args.train.enable_fsdp_offload,
    #     fsdp_kwargs=fsdp_kwargs,
    #     basic_modules=model._no_split_modules + args.model.basic_modules,
    #     enable_reentrant=args.train.enable_reentrant,
    #     enable_forward_prefetch=args.train.enable_forward_prefetch,
    # )

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

    schedule = create_schedule_from_config(
        config=OmegaConf.create(args.model.schedule_config),
        device="cuda",
    )
    training_timesteps = create_training_timesteps_from_config(
        config=OmegaConf.create(args.model.training_timesteps_config),
        schedule=schedule,
        device="cuda",
    )
    sampling_timesteps = create_sampling_timesteps_from_config(
        config=OmegaConf.create(args.model.sampling_timesteps_config),
        schedule=schedule,
        device="cuda",
    )

    sampler = create_sampler_from_config(
        config=OmegaConf.create(args.model.sampler_config),
        schedule=schedule,
        timesteps=sampling_timesteps,
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
        model_assets = [model_config]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None

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
        for step, micro_batches in enumerate(train_dataloader):
            global_step += 1

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            torch.cuda.synchronize()
            for micro_batch in micro_batches:
                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }
                # helper.print_example(example=micro_batch, rank=args.train.local_rank, print_tensor=False)
                with model_fwd_context:
                    noises = torch.randn_like(micro_batch["latents"])
                    batch_size = micro_batch["text_shapes"].shape[0]
                    timesteps = training_timesteps.sample([batch_size], device="cuda")
                    timesteps = timestep_transform(schedule, timesteps, micro_batch["latents_shapes"])
                    timesteps_repeated = timesteps.repeat_interleave(micro_batch["latents_shapes"].prod(-1))
                    latents_noised = schedule.forward(micro_batch["latents"], noises, timesteps_repeated)

                    pred = model(
                        vid=torch.cat([latents_noised, micro_batch["latents_cond"]], dim=-1),
                        txt=micro_batch["text_embeds"],
                        vid_shape=micro_batch["latents_shapes"],
                        shot_vid_shape=micro_batch["shot_latents_shapes"],
                        txt_shape=micro_batch["text_shapes"],
                        shot_txt_shape=micro_batch["shot_text_shapes"],
                        timestep=timesteps,
                        num_shots=micro_batch["num_shots"],
                    ).vid_sample
                    latents_pred, noises_pred = schedule.convert_from_pred(
                        pred=pred,
                        pred_type=sampler.prediction_type,
                        x_t=latents_noised,
                        t=timesteps_repeated,
                    )

                    # Compute mse per sample loss
                    loss = F.mse_loss(
                        input=schedule.convert_to_pred(
                            x_0=latents_pred,
                            x_T=noises_pred,
                            t=timesteps_repeated,
                            pred_type=args.model.loss["type"],
                        ),
                        target=schedule.convert_to_pred(
                            x_0=micro_batch["latents"],
                            x_T=noises,
                            t=timesteps_repeated,
                            pred_type=args.model.loss["type"],
                        ),
                        reduction="none",
                    )
                    loss = na.unflatten(loss, micro_batch["latents_shapes"])
                    loss = torch.stack([x.mean() for x in loss])
                loss = loss / len(micro_batches)
                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

                # logger.info(f"[Rank: {args.train.local_rank}] global_step: {global_step} bs_id: {bs_id}")
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm, foreach=True)

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

    torch.cuda.synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        if args.train.hf_weights_path:
            hf_weights_path = args.train.hf_weights_path
        else:
            hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        if args.model.lora_config:
            from peft import get_peft_model_state_dict

            from veomni.models.module_utils import _save_state_dict

            model_state_dict = get_peft_model_state_dict(model, model_state_dict)
            lora_adapter_save_path = os.path.join(hf_weights_path, "adapter_model.bin")
            os.makedirs(hf_weights_path, exist_ok=True)
            _save_state_dict(model_state_dict, lora_adapter_save_path, safe_serialization=False)
            model.peft_config["default"].save_pretrained(hf_weights_path)
            logger.info_rank0(f"Lora adapter saved at {hf_weights_path} successfully!")

            if args.model.lora_config.get("save_merge", False):
                from peft import PeftModel

                model = build_foundation_model(
                    config_path=args.model.config_path,
                    weights_path=args.model.model_path,
                    torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
                    attn_implementation=args.model.attn_implementation,
                    moe_implementation=args.model.moe_implementation,
                    init_device=args.train.init_device,
                    force_use_huggingface=args.model.force_use_huggingface,
                )
                model = PeftModel.from_pretrained(model, hf_weights_path)
                model = model.merge_and_unload()  # 合并 LoRA 权重到 base_model
                model.save_pretrained(hf_weights_path)
                logger.info_rank0(f"Lora merged model adapter saved at {hf_weights_path} successfully!")
        else:
            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
