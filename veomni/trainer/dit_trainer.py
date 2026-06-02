# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pickle as pk
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import build_data_transform, build_dataloader
from ..data.data_collator import DataCollator, MakeMicroBatchCollator
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.parallel_state import get_parallel_state
from ..models import build_foundation_model
from ..models.auto import build_config
from ..models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY
from ..ops import apply_ops_config
from ..utils import helper
from ..utils.device import (
    get_device_type,
    synchronize,
)
from .base import BaseTrainer, VeOmniIter


logger = helper.create_logger(__name__)


class OfflineEmbeddingSaver:
    def __init__(self, save_path: str, dataset_length: int = 0, shard_num: int = 1, max_shard=1000):
        from ..distributed.parallel_state import get_parallel_state

        self.dp_rank = get_parallel_state().dp_rank
        dp_size = get_parallel_state().dp_size
        if dp_size * shard_num > max_shard:
            shard_num = max_shard // dp_size
            logger.info_rank0(f"shard_num * dp_size must be smaller than max_shard, set shard_num = {shard_num}")
        self.shard_num = shard_num
        self.max_shard = max_shard
        self.index = 0
        self.buffer = []

        self.save_path = save_path
        self.dataset_length = dataset_length
        self.batch_len = math.ceil(dataset_length / self.shard_num)
        logger.info(f"Rank [{self.dp_rank}] save to [{self.save_path}] each batch_len [{self.batch_len}].")
        os.makedirs(self.save_path, exist_ok=True)
        self.rest_len = self.dataset_length

    @staticmethod
    def _cpu_recursive(obj):
        """Move tensors to CPU recursively, leave other types unchanged."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: OfflineEmbeddingSaver._cpu_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(OfflineEmbeddingSaver._cpu_recursive(v) for v in obj)
        return obj

    def to_save_bytes(self, save_item: Dict[str, torch.Tensor]):
        converted_dict = {}
        for key in list(save_item.keys()):
            converted_dict[key] = pk.dumps(self._cpu_recursive(save_item[key]))
            del save_item[key]
        return converted_dict

    def _append_item(self, save_item: Dict[str, torch.Tensor]):
        if self.rest_len > 0:  # 多余的dummy data buffer 不保存
            self.buffer.append(self.to_save_bytes(save_item))
            self.rest_len -= 1

    def save(self, save_item):
        self._append_item(save_item)
        if len(self.buffer) >= self.batch_len:
            ds = Dataset.from_list(self.buffer)
            ds.to_parquet(os.path.join(self.save_path, f"rank_{self.dp_rank}_shard_{self.index}.parquet"))
            self.buffer = []
            self.index += 1

    def save_last(self):
        if len(self.buffer) > 0:
            ds = Dataset.from_list(self.buffer)
            ds.to_parquet(os.path.join(self.save_path, f"rank_{self.dp_rank}_shard_{self.index}.parquet"))
            self.buffer = []
            self.index += 1


@dataclass
class DiTDataCollator(DataCollator):
    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        return batch


@dataclass
class DiTModelArguments(ModelArguments):
    condition_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to condition model."},
    )
    condition_model_cfg: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for condition model."},
    )


@dataclass
class DiTDataArguments(DataArguments):
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
class DiTTrainingArguments(TrainingArguments):
    training_task: Literal["offline_training", "online_training", "offline_embedding"] = field(
        default="online_training",
        metadata={
            "help": "Training task. offline_training: training offline embedded data. "
            "online_training: training raw data online. offline_embedding: embedding raw data."
        },
    )


@dataclass
class VeOmniDiTArguments(VeOmniArguments):
    model: DiTModelArguments = field(default_factory=DiTModelArguments)
    data: DiTDataArguments = field(default_factory=DiTDataArguments)
    train: DiTTrainingArguments = field(default_factory=DiTTrainingArguments)


class DiTTrainer:
    """
    DiT Trainer merging BaseTrainer infrastructure with DiT-specific model setup.
    Reuses BaseTrainer's callbacks, dataloader building (with MainCollator/DiTConcatCollator),
    and training loop; overrides model building and forward pass.
    """

    condition_model: PreTrainedModel
    training_task: Literal["offline_training", "online_training", "offline_embedding"]
    offline_embedding_save_dir: str = None
    offline_embedding_saver: OfflineEmbeddingSaver = None

    def __init__(self, args: VeOmniDiTArguments):
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        # rewrite _setup, setup arguments for dit training
        self._setup()

        # rewrite _build_model, build condition model & dit model
        self._build_model()

        # rewrite _freeze_model_module, freeze condition model
        self._freeze_model_module()

        # rewrite _build_model_assets to support processor of condition model
        self._build_model_assets()

        # rewrite _build_data_transform, build data transform for offline or online dit data
        self._build_data_transform()

        # rewrite _build_dataset, init offline_embedding_saver after build_dataset
        self._build_dataset()

        # Do not use maincollator in dit training
        # self.base._build_collate_fn()

        # rewrite _build_dataloader, build dataloader only on sp_rank_0 to save memory
        self._build_dataloader()

        if self.training_task != "offline_embedding":
            self.base._build_parallelized_model()
            self.base._build_optimizer()
            self.base._build_lr_scheduler()
            self.base._build_training_context()

        self.base._init_callbacks()

    def _setup(self):
        self.base._setup()
        args: VeOmniDiTArguments = self.base.args
        args.train.dyn_bsz = False
        args.train.micro_batch_size = 1
        # dataloader_batch_size was computed in __post_init__ when dyn_bsz was still True
        # (default), so it was set to 1. Recompute now that dyn_bsz=False.
        args.train.dataloader_batch_size = args.train.global_batch_size // get_parallel_state().dp_size
        if args.train.training_task == "offline_embedding":
            assert args.data.datasets_type == "mapping", "Datasets type must be mapping for offline embedding."
            if args.data.offline_embedding_save_dir is None:
                self.offline_embedding_save_dir = f"{args.data.train_path}_offline"
            else:
                self.offline_embedding_save_dir = args.data.offline_embedding_save_dir

            args.data.drop_last = False
            args.data.shuffle = False
            args.train.checkpoint.save_epochs = 0
            args.train.checkpoint.save_hf_weights = False
            # No gradient accumulation needed; process one sample per step to
            # avoid broadcast_object_list serialising all micro-batches at once
            # which can OOM CPU memory with large video data.
            args.train.global_batch_size = get_parallel_state().dp_size
            args.train.dataloader_batch_size = 1
            logger.info_rank0(
                f"Task offline_embedding. Drop last: {args.data.drop_last}, shuffle: {args.data.shuffle}"
            )
            args.train.num_train_epochs = 1

        self.training_task = args.train.training_task

    def _build_model(self):
        logger.info_rank0("Build model")
        args: VeOmniDiTArguments = self.base.args
        # Apply ops config eagerly so the condition model (built below via
        # ``model_class._from_config``, not ``build_foundation_model``) sees a
        # populated ops singleton / LOSS_MAPPING. ``build_foundation_model``
        # below will re-apply the same config — that call is idempotent.
        apply_ops_config(args.model.ops_implementation)
        model_config = args.model.model_config
        dit_config = build_config(args.model.config_path, **model_config)
        self.base.model_config = dit_config
        logger.info_rank0(f"Detected DiT model type: {dit_config.model_type}.")
        self._build_condition_model(
            condition_model_type=dit_config.condition_model_type,
        )
        if self.training_task == "offline_training" or self.training_task == "online_training":
            logger.info_rank0(f"Task: {self.training_task}, prepare dit model.")
            self.base.model = build_foundation_model(
                config_path=args.model.config_path,
                weights_path=args.model.model_path,
                torch_dtype="float32" if args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
                init_device=args.train.init_device,
                ops_implementation=args.model.ops_implementation,
                config_kwargs=model_config,
            )
            self.base.model_config = getattr(self.base.model, "config", None)
        else:
            self.base.model = None
            logger.info_rank0(f"Task: {self.training_task}, dit model is not prepared.")

    def _build_condition_model(
        self,
        condition_model_type: str,
    ) -> PreTrainedModel:
        args: VeOmniDiTArguments = self.base.args
        config_class = MODEL_CONFIG_REGISTRY[condition_model_type]()
        condition_cfg = config_class.from_pretrained(
            args.model.condition_model_path,
            seed=args.train.seed,  # seed for randn noise and scheduler
            **args.model.condition_model_cfg,
        )
        model_class = MODELING_REGISTRY[condition_model_type]()
        if self.training_task == "offline_training":
            self.condition_model = model_class._from_config(condition_cfg, meta_init=True)
            logger.info_rank0("Condition model loaded with empty weights.")
        else:
            self.condition_model = model_class._from_config(condition_cfg)
            self.condition_model.to(get_device_type())
            logger.info_rank0("Condition model loaded.")

    def _freeze_model_module(self):
        self.condition_model.requires_grad_(False)

        if self.training_task == "offline_training" or self.training_task == "online_training":
            self.base._freeze_model_module()

    def _build_model_assets(self):
        if self.training_task == "offline_training" or self.training_task == "online_training":
            self.base.model_assets = [self.base.model.config]
        else:
            self.base.model_assets = []

    def _build_data_transform(self):
        args: VeOmniDiTArguments = self.base.args
        if self.training_task == "offline_training":
            self.base.data_transform = build_data_transform("dit_offline")
        else:
            self.base.data_transform = build_data_transform(
                "dit_online",
                **args.data.mm_configs,
            )

    def _build_dataset(self):
        args: VeOmniDiTArguments = self.base.args
        self.base._build_dataset()
        if get_parallel_state().sp_enabled and get_parallel_state().sp_rank != 0:
            self.base.train_dataset = None

        if (
            not get_parallel_state().sp_enabled or get_parallel_state().sp_rank == 0
        ) and self.base.train_dataset is not None:
            inner = (
                self.base.train_dataset._data if hasattr(self.base.train_dataset, "_data") else self.base.train_dataset
            )
            if hasattr(inner, "__len__"):
                dataset_len = len(inner)
                corrected_steps = max(1, dataset_len // args.train.global_batch_size)
                if args.train.max_steps is not None:
                    corrected_steps = min(corrected_steps, args.train.max_steps)
                args._train_steps = corrected_steps
                self.base.train_steps = args.train_steps
                logger.info_rank0(
                    f"Corrected train_steps based on actual dataset size: "
                    f"dataset_len={dataset_len}, global_batch_size={args.train.global_batch_size}, "
                    f"train_steps={corrected_steps}."
                )

                dp_size = get_parallel_state().dp_size
                per_rank_count = max(1, math.ceil(dataset_len / dp_size))
                if args.train.dataloader_batch_size > per_rank_count:
                    old_bs = args.train.dataloader_batch_size
                    args.train.dataloader_batch_size = per_rank_count
                    logger.info_rank0(
                        f"Capped dataloader_batch_size from {old_bs} to {per_rank_count} "
                        f"(dataset_len={dataset_len}, dp_size={dp_size}, per_rank_count={per_rank_count}) "
                        f"to ensure DataLoader can form batches with drop_last=True."
                    )

        if self.training_task == "offline_embedding":
            if not get_parallel_state().sp_enabled or get_parallel_state().sp_rank == 0:
                dp_rank = get_parallel_state().dp_rank
                dp_size = get_parallel_state().dp_size
                dataset_len = len(self.base.train_dataset)
                base_count = dataset_len // dp_size
                extra = dataset_len % dp_size
                valid_data_length = base_count + (1 if dp_rank < extra else 0)
                logger.info(f"Rank {args.train.global_rank} data length to save: {valid_data_length}")
                self.offline_embedding_saver = OfflineEmbeddingSaver(
                    save_path=self.offline_embedding_save_dir,
                    dataset_length=valid_data_length,
                )
                padded_len = (
                    math.ceil(self.base.train_dataset.data_len / args.train.global_batch_size)
                    * args.train.global_batch_size
                )
                self.base.train_dataset.data_len = padded_len
                args._train_steps = padded_len // dp_size // args.train.dataloader_batch_size
                self.base.train_steps = args.train_steps
            else:
                self.offline_embedding_saver = None

        # Sync _train_steps across the DP group so every rank agrees on step count.
        # Iterable datasets (e.g. PreprocessedIterableDataset) may yield different sample counts per rank after
        # sharding, which would cause some ranks to exit the training loop earlier than others
        # and deadlock at the next FSDP collective.
        if get_parallel_state().dp_enabled and hasattr(args, "_train_steps"):
            steps_t = torch.tensor([args._train_steps], dtype=torch.long, device=torch.device(get_device_type()))
            dist.all_reduce(steps_t, op=dist.ReduceOp.MIN, group=get_parallel_state().dp_group)
            args._train_steps = int(steps_t.item())
            self.base.train_steps = args.train_steps

        # Sync _train_steps across the SP group AFTER padding so every rank
        # agrees on step count (required to avoid deadlocks in broadcast_object_list).
        if get_parallel_state().sp_enabled:
            steps_t = torch.zeros(1, dtype=torch.int64, device=torch.device(get_device_type()))
            if get_parallel_state().sp_rank == 0:
                steps_t[0] = args._train_steps
            dist.broadcast(
                steps_t,
                src=dist.get_global_rank(get_parallel_state().sp_group, 0),
                group=get_parallel_state().sp_group,
            )
            args._train_steps = int(steps_t.item())
            self.base.train_steps = args.train_steps

    def _build_dataloader(self):
        """Build dataloader with dyn_bsz=False for DiT (fixed batch)."""
        args = self.base.args
        if not get_parallel_state().sp_enabled or get_parallel_state().sp_rank == 0:
            self.base.train_dataloader = build_dataloader(
                dataloader_type=args.data.dataloader.type,
                dataset=self.base.train_dataset,
                micro_batch_size=args.train.micro_batch_size,
                global_batch_size=args.train.global_batch_size,
                dataloader_batch_size=args.train.dataloader_batch_size,
                max_seq_len=args.data.max_seq_len,
                train_steps=args.train_steps,
                bsz_warmup_ratio=args.train.bsz_warmup_ratio,
                bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
                dyn_bsz=args.train.dyn_bsz,
                dyn_bsz_runtime=args.train.dyn_bsz_runtime,
                dyn_bsz_count_mode=args.train.dyn_bsz_count_mode,
                dyn_bsz_physical_overflow_ratio=args.train.dyn_bsz_physical_overflow_ratio,
                dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
                num_workers=args.data.dataloader.num_workers,
                drop_last=args.data.dataloader.drop_last,
                pin_memory=args.data.dataloader.pin_memory,
                prefetch_factor=args.data.dataloader.prefetch_factor,
                seed=args.train.seed,
                collate_fn=DiTDataCollator(),
                save_steps=args.train.checkpoint.save_steps,
            )
        else:
            self.base.train_dataloader = None

        if self.base.train_dataloader is not None and not args.train.dyn_bsz:
            num_micro_batch = args.train.global_batch_size // (
                args.train.micro_batch_size * get_parallel_state().dp_size
            )
            if num_micro_batch > args.train.dataloader_batch_size:
                capped_nmb = max(1, args.train.dataloader_batch_size)
                logger.info_rank0(
                    f"Capping num_micro_batch from {num_micro_batch} to {capped_nmb} "
                    f"(dataloader_batch_size={args.train.dataloader_batch_size}) "
                    f"to avoid empty micro-batches."
                )
                collate_fn = self.base.train_dataloader.collate_fn
                if isinstance(collate_fn, MakeMicroBatchCollator):
                    collate_fn.num_micro_batch = capped_nmb

    def on_train_begin(self):
        self.base.on_train_begin()

    def on_train_end(self):
        self.base.on_train_end()

    def on_epoch_begin(self):
        self.base.on_epoch_begin()

    def on_epoch_end(self):
        self.base.on_epoch_end()

    def on_step_begin(self, micro_batches=None):
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def preforward(self, micro_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess micro batches before forward pass."""

        def _to_device(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                return v.to(self.base.device, non_blocking=True)
            if isinstance(v, dict):
                return {k: _to_device(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [_to_device(item) for item in v]
            return v

        micro_batch = {k: _to_device(v) for k, v in micro_batch.items()}
        if getattr(self.base, "LOG_SAMPLE", True):
            helper.print_example(example=micro_batch, rank=self.base.args.train.local_rank)
            self.base.LOG_SAMPLE = False
        return micro_batch

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Postprocess model outputs after forward pass."""
        loss_dict: Dict[str, torch.Tensor] = outputs.loss
        loss_dict = {k: v / self.base.args.train.micro_batch_size for k, v in loss_dict.items()}
        loss = torch.stack(list(loss_dict.values())).sum()
        return loss, loss_dict

    def _compute_dit_loss(
        self,
        predictions: list[torch.Tensor],
        training_targets: list[torch.Tensor],
        video_loss_masks: list[torch.Tensor] | None,
        audio_predictions: list[torch.Tensor] | None = None,
        audio_training_targets: list[torch.Tensor] | None = None,
        audio_loss_masks: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_sample_losses = []

        for i, (prediction, target) in enumerate(zip(predictions, training_targets)):
            prediction = prediction.to(dtype=torch.float32)
            target = target.to(dtype=torch.float32)
            per_element_loss = (prediction - target).pow(2)

            B, C, F, H, W = prediction.shape

            sample_vlm = None
            if video_loss_masks is not None and i < len(video_loss_masks):
                sample_vlm = video_loss_masks[i]

            if sample_vlm is not None:
                sample_vlm = sample_vlm.to(device=prediction.device, dtype=torch.bool)
                loss_mask = sample_vlm.view(1, 1, F, H, W).float()
                masked_loss = per_element_loss * loss_mask
                valid_count = loss_mask.reshape(1, 1, -1).sum(dim=-1).clamp(min=1e-8)
                per_sample_loss = masked_loss.reshape(B, C, -1).sum(dim=-1).mean(dim=-1) / valid_count
            else:
                per_sample_loss = per_element_loss.reshape(B, -1).mean(dim=1)

            if audio_predictions is not None and audio_training_targets is not None and i < len(audio_predictions):
                audio_pred = audio_predictions[i].to(dtype=torch.float32)
                audio_target = audio_training_targets[i].to(dtype=torch.float32)
                audio_loss = (audio_pred - audio_target).pow(2).mean(dim=tuple(range(1, audio_pred.dim())))
                per_sample_loss = per_sample_loss + audio_loss

            per_sample_losses.append(per_sample_loss)

        loss = torch.stack(per_sample_losses).mean()
        loss_dict = {"mse_loss": loss}
        total_loss = torch.stack(list(loss_dict.values())).sum()
        return total_loss, loss_dict

    @staticmethod
    def _unpack_dict_of_list(batch: Dict[str, Any]) -> list[Dict[str, Any]]:
        if not isinstance(batch, dict) or len(batch) == 0:
            return []
        keys = list(batch.keys())
        num_items = len(batch[keys[0]])
        return [{k: batch[k][idx] for k in keys} for idx in range(num_items)]

    def forward_backward_step(self, micro_batch: Dict[str, torch.Tensor]) -> tuple:
        micro_batch = self.preforward(micro_batch)
        device = get_device_type()

        is_ltx2_precomputed = "video_prompt_embeds" in micro_batch
        is_preprocessed = ("conditions" in micro_batch and "inputs" not in micro_batch) or is_ltx2_precomputed
        fps_list = None

        if is_ltx2_precomputed:
            micro_batch.pop("idx", None)
            latents_raw = micro_batch.pop("latents", None)
            video_features = micro_batch.pop("video_prompt_embeds")
            audio_features = micro_batch.pop("audio_prompt_embeds", None)
            prompt_mask = micro_batch.pop("prompt_attention_mask", None)
            for extra_key in list(micro_batch.keys()):
                micro_batch.pop(extra_key)

            if latents_raw is not None:
                if isinstance(latents_raw, list) and len(latents_raw) > 0 and isinstance(latents_raw[0], dict):
                    fps_list = [float(d.get("fps", 24)) for d in latents_raw]
                    micro_batch["latents"] = [
                        (d["latents"].unsqueeze(0) if d["latents"].dim() == 4 else d["latents"]).to(device)
                        for d in latents_raw
                    ]
                elif isinstance(latents_raw, list):
                    micro_batch["latents"] = [t.to(device) if isinstance(t, torch.Tensor) else t for t in latents_raw]
                else:
                    micro_batch["latents"] = (
                        latents_raw.to(device) if isinstance(latents_raw, torch.Tensor) else latents_raw
                    )

            if isinstance(video_features, list):
                micro_batch["context"] = [(f.unsqueeze(0) if f.dim() == 2 else f).to(device) for f in video_features]
            else:
                micro_batch["context"] = [video_features.to(device)]

            if audio_features is not None:
                if isinstance(audio_features, list):
                    micro_batch["audio_context"] = [
                        (f.unsqueeze(0) if f.dim() == 2 else f).to(device) for f in audio_features
                    ]
                else:
                    micro_batch["audio_context"] = [audio_features.to(device)]

            if prompt_mask is not None:
                if isinstance(prompt_mask, list):
                    micro_batch["context_mask"] = [
                        (m.unsqueeze(0) if m.dim() == 1 else m).to(device) for m in prompt_mask
                    ]
                else:
                    micro_batch["context_mask"] = [prompt_mask.to(device)]

        elif is_preprocessed:
            conditions_raw = micro_batch.pop("conditions")
            audio_latents_raw = micro_batch.pop("audio_latents", None)
            micro_batch.pop("idx", None)
            for extra_key in list(micro_batch.keys()):
                if extra_key not in ("latents",):
                    micro_batch.pop(extra_key)
            if micro_batch.get("latents") and isinstance(micro_batch["latents"][0], dict):
                fps_list = [float(d.get("fps", 24)) for d in micro_batch["latents"]]
                micro_batch["latents"] = [
                    (d["latents"].unsqueeze(0) if d["latents"].dim() == 4 else d["latents"]).to(device)
                    for d in micro_batch["latents"]
                ]
            elif micro_batch.get("latents"):
                micro_batch["latents"] = [
                    t.to(device) if isinstance(t, torch.Tensor) else t for t in micro_batch["latents"]
                ]
            first_cond = conditions_raw[0]
            if isinstance(first_cond, dict) and "video_prompt_embeds" in first_cond:
                video_features = [c["video_prompt_embeds"] for c in conditions_raw]
                micro_batch["context"] = [(f.unsqueeze(0) if f.dim() == 2 else f).to(device) for f in video_features]
                audio_features_list = [c.get("audio_prompt_embeds") for c in conditions_raw]
                if any(a is not None for a in audio_features_list):
                    micro_batch["audio_context"] = [
                        (f.unsqueeze(0) if f.dim() == 2 else f).to(device)
                        for f in audio_features_list
                        if f is not None
                    ]
                prompt_mask_list = [c.get("prompt_attention_mask") for c in conditions_raw]
                if any(m is not None for m in prompt_mask_list):
                    micro_batch["context_mask"] = [
                        (m.unsqueeze(0) if m.dim() == 1 else m).to(device) for m in prompt_mask_list if m is not None
                    ]
            elif isinstance(first_cond, dict):
                micro_batch["context"] = [
                    c.get("last_hidden_state", next(iter(c.values()))).unsqueeze(0).to(device) for c in conditions_raw
                ]
            elif isinstance(first_cond, torch.Tensor):
                micro_batch["context"] = [(c.unsqueeze(0) if c.dim() == 2 else c).to(device) for c in conditions_raw]

            if audio_latents_raw is not None:
                if isinstance(audio_latents_raw, list) and len(audio_latents_raw) > 0:
                    if isinstance(audio_latents_raw[0], dict):
                        micro_batch["audio_latents"] = [
                            (d["latents"].unsqueeze(0) if d["latents"].dim() == 3 else d["latents"]).to(device)
                            for d in audio_latents_raw
                        ]
                    else:
                        micro_batch["audio_latents"] = [
                            (t.to(device) if isinstance(t, torch.Tensor) else t) for t in audio_latents_raw
                        ]

        if (
            self.training_task == "online_training" or self.training_task == "offline_embedding"
        ) and not is_preprocessed:
            with torch.no_grad():
                micro_batch = self.condition_model.get_condition(**micro_batch)

        if self.training_task == "offline_embedding":
            if self.offline_embedding_saver is not None:  # sp_rank 0 save
                for item in self._unpack_dict_of_list(micro_batch):
                    self.offline_embedding_saver.save(item)
            del micro_batch
            return 0.0, {}

        first_frame_p = self.base.args.data.mm_configs.get("first_frame_conditioning_p", 0.5)
        timestep_sampling_mode = self.base.args.data.mm_configs.get("timestep_sampling_mode", "shifted_logit_normal")
        with_audio = self.base.args.data.mm_configs.get("with_audio", False)
        with torch.no_grad():
            if is_preprocessed:
                micro_batch = self.condition_model.process_condition(
                    latents=micro_batch["latents"],
                    context=micro_batch["context"],
                    context_mask=micro_batch.get("context_mask"),
                    audio_context=micro_batch.get("audio_context"),
                    audio_latents=micro_batch.get("audio_latents") if with_audio else None,
                    first_frame_conditioning_p=first_frame_p,
                    timestep_sampling_mode=timestep_sampling_mode,
                    fps=fps_list,
                )
            else:
                micro_batch = self.condition_model.process_condition(**micro_batch)

        if is_preprocessed:
            training_targets = micro_batch.pop("training_target", None)
            audio_training_targets = micro_batch.pop("audio_training_target", None)
            micro_batch.pop("latents", None)

        with self.base.model_fwd_context:
            outputs = self.base.model(**micro_batch)

        loss: torch.Tensor
        loss_dict: Dict[str, torch.Tensor]
        if is_preprocessed:
            loss, loss_dict = self._compute_dit_loss(
                outputs.predictions,
                training_targets,
                micro_batch.get("video_loss_mask"),
                audio_predictions=outputs.audio_predictions,
                audio_training_targets=audio_training_targets,
                audio_loss_masks=micro_batch.get("audio_loss_mask"),
            )
        else:
            loss, loss_dict = self.postforward(outputs, micro_batch)

        # Backward pass
        with self.base.model_bwd_context:
            loss.backward()

        del micro_batch
        return loss, loss_dict

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        args = self.base.args
        self.base.state.global_step += 1

        # broadcast micro_batches from sp_rank_0 to all ranks
        if get_parallel_state().sp_enabled:
            if get_parallel_state().sp_rank == 0:
                micro_batches = next(data_iterator)
            else:
                micro_batches = None

            obj_list = [micro_batches]
            dist.broadcast_object_list(
                obj_list,
                src=dist.get_global_rank(get_parallel_state().sp_group, 0),
                group=get_parallel_state().sp_group,
            )
            micro_batches = obj_list[0]
        else:
            micro_batches = next(data_iterator)

        self.on_step_begin(micro_batches=micro_batches)

        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(float)
        grad_norm = 0.0
        num_micro_batches = len(micro_batches)
        self.base.num_micro_batches = num_micro_batches

        for micro_step, micro_batch in enumerate(micro_batches):
            if self.training_task != "offline_embedding":
                self.base.model_reshard(micro_step, num_micro_batches)

            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]

            loss, loss_dict = self.forward_backward_step(micro_batch)

            if self.training_task != "offline_embedding":
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    total_loss_dict[k] += v.item()

        if self.training_task != "offline_embedding":
            grad_norm = veomni_clip_grad_norm(self.base.model, args.train.optimizer.max_grad_norm)
            self.base.optimizer.step()
            self.base.lr_scheduler.step()
            self.base.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=dict(total_loss_dict), grad_norm=grad_norm)

    def train(self):
        args = self.base.args
        self.on_train_begin()
        if self.training_task == "offline_embedding":
            args.train.num_train_epochs = 1

        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {self.base.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.base.start_epoch, args.train.num_train_epochs):
            if self.base.train_dataloader is not None and hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.base.state.epoch = epoch
            self.on_epoch_begin()

            if self.base.train_dataloader is not None:
                self.base.data_iterator = VeOmniIter(
                    self.base.train_dataloader,
                    use_background_prefetcher=args.data.dataloader.use_background_prefetcher,
                )
            else:
                self.base.data_iterator = None

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(self.base.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()
            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
            if args.data.dataloader.use_background_prefetcher:
                self.base.data_iterator.stop()

        self.on_train_end()

        if args.data.dataloader.use_background_prefetcher:
            self.base.data_iterator.stop()

        synchronize()

        if self.training_task == "offline_embedding" and self.offline_embedding_saver is not None:
            self.offline_embedding_saver.save_last()

        self.base.destroy_distributed()
