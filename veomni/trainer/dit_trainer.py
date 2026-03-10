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
from functools import partial
from typing import Any, Dict, Literal, Optional, Sequence

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import build_data_transform, build_dataloader
from ..data.data_collator import DataCollator
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.parallel_state import get_parallel_state
from ..models import build_foundation_model
from ..models.auto import build_config
from ..models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY
from ..utils import helper
from ..utils.device import (
    synchronize,
)
from ..utils.model_utils import pretty_print_trainable_parameters
from .base import BaseTrainer


logger = helper.create_logger(__name__)


def patch_parallel_load_safetensors(model: torch.nn.Module):
    def patch_parallel_load_safetensors(weights_path, func, model: torch.nn.Module):
        shard_states = func(weights_path)
        parameter_name = next(model.named_parameters())[0]
        if parameter_name.startswith("base_model."):  # using lora peft will add prefix "base_model"
            shard_states = {"base_model.model." + k: v for k, v in shard_states.items()}
        for fqn, module in model.named_modules():
            fqn = fqn + ("." if fqn else "")
            if hasattr(module, "base_layer"):  # using lora peft will insert "base_layer"
                for pname, _ in module.base_layer.named_parameters():
                    old_name = fqn + pname
                    if old_name in shard_states:
                        wrap_name = fqn + "base_layer." + pname
                        shard_states[wrap_name] = shard_states.pop(old_name)
        return shard_states

    from veomni.distributed import torch_parallelize

    torch_parallelize.parallel_load_safetensors = partial(
        patch_parallel_load_safetensors,
        func=torch_parallelize.parallel_load_safetensors,
        model=model,
    )


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
        self.rest_len = self.dataset_length

    def to_save_bytes(self, save_item: Dict[str, torch.Tensor]):
        converted_dict = {}
        for key in list(save_item.keys()):
            converted_dict[key] = pk.dumps(save_item[key].cpu())
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

        # rewrite _freeze_model_module, freeze condition model & add lora for dit model
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
        if args.train.training_task == "offline_embedding":
            assert args.train.ulysses_parallel_size == 1, "Ulysses parallel size must be 1 for offline embedding."
            assert args.data.datasets_type == "mapping", "Datasets type must be mapping for offline embedding."
            if args.data.offline_embedding_save_dir is None:
                self.offline_embedding_save_dir = f"{args.data.train_path}_offline"
            else:
                self.offline_embedding_save_dir = args.data.offline_embedding_save_dir

            args.data.drop_last = False
            args.data.shuffle = False
            args.train.save_epochs = 0
            args.train.save_hf_weights = False
            logger.info_rank0(
                f"Task offline_embedding. Drop last: {args.data.drop_last}, shuffle: {args.data.shuffle}"
            )
            args.train.num_train_epochs = 1

        self.training_task = args.train.training_task

    def _build_model(self):
        logger.info_rank0("Build model")
        args: VeOmniDiTArguments = self.base.args
        dit_config = build_config(args.model.config_path)
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
                torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
                attn_implementation=args.model.ops_implementation.attn_implementation,
                moe_implementation=args.model.ops_implementation.moe_implementation,
                init_device=args.train.init_device,
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
        condition_cfg = config_class.from_pretrained(args.model.condition_model_path, **args.model.condition_model_cfg)
        model_class = MODELING_REGISTRY[condition_model_type]()
        if self.training_task == "offline_training":
            self.condition_model = model_class._from_config(condition_cfg, meta_init=True)
            logger.info_rank0("Condition model loaded with empty weights.")
        else:
            self.condition_model = model_class._from_config(condition_cfg)
            self.condition_model.cuda()
            logger.info_rank0("Condition model loaded.")

    def _freeze_model_module(self):
        args: VeOmniDiTArguments = self.base.args
        lora_config = args.model.lora_config
        self.condition_model.requires_grad_(False)

        if self.training_task == "offline_training" or self.training_task == "online_training":
            if not bool(lora_config):
                self.base.lora = False
            else:
                lora_adapter_path = lora_config.get("lora_adapter", None)
                if lora_adapter_path is not None:
                    logger.info_rank0(f"Load lora_adapter from {lora_adapter_path}.")
                    from peft import PeftModel

                    self.base.model = PeftModel.from_pretrained(self.base.model, lora_adapter_path)
                else:
                    from peft import LoraConfig, get_peft_model

                    lora_config: LoraConfig = LoraConfig(
                        r=lora_config["rank"],
                        lora_alpha=lora_config["alpha"],
                        target_modules=lora_config["lora_modules"],
                    )
                    logger.info_rank0(f"Init lora: {lora_config.to_dict()}.")
                    self.base.model = get_peft_model(self.base.model, lora_config)

                self.base.model.print_trainable_parameters()
                self.base.lora = True

                if args.train.init_device == "meta":
                    patch_parallel_load_safetensors(self.base.model)

            pretty_print_trainable_parameters(self.base.model)
            helper.print_device_mem_info("VRAM usage after building model")

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
        self.base._build_dataset()
        args: VeOmniDiTArguments = self.base.args
        if self.training_task == "offline_embedding":
            dp_size = get_parallel_state().dp_size
            base = len(self.base.train_dataset) // dp_size
            extra = len(self.base.train_dataset) % dp_size
            extra_for_rank = max(0, min(1, extra - args.train.local_rank))
            valid_data_length = base + extra_for_rank
            logger.info(f"Rank {args.train.global_rank} data length to save: {valid_data_length}")
            self.offline_embedding_saver = OfflineEmbeddingSaver(
                save_path=self.offline_embedding_save_dir,
                dataset_length=valid_data_length,
            )

            # pad dataset_len
            self.base.train_dataset.data_len = (
                math.ceil(self.base.train_dataset.data_len / (args.train.global_batch_size))
                * args.train.global_batch_size
            )

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
                dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
                num_workers=args.data.dataloader.num_workers,
                drop_last=args.data.dataloader.drop_last,
                pin_memory=args.data.dataloader.pin_memory,
                prefetch_factor=args.data.dataloader.prefetch_factor,
                seed=args.train.seed,
                collate_fn=DiTDataCollator(),
            )
        else:
            self.base.train_dataloader = None

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
        micro_batch = {
            k: v.to(self.base.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in micro_batch.items()
        }
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

    @staticmethod
    def _unpack_dict_of_list(batch: Dict[str, Any]) -> list[Dict[str, Any]]:
        if not isinstance(batch, dict) or len(batch) == 0:
            return []
        keys = list(batch.keys())
        num_items = len(batch[keys[0]])
        return [{k: batch[k][idx] for k in keys} for idx in range(num_items)]

    def forward_backward_step(self, micro_batch: Dict[str, torch.Tensor]) -> tuple:
        micro_batch = self.preforward(micro_batch)

        if self.training_task == "online_training" or self.training_task == "offline_embedding":
            with torch.no_grad():
                micro_batch = self.condition_model.get_condition(**micro_batch)

        if self.training_task == "offline_embedding":
            for item in self._unpack_dict_of_list(micro_batch):
                self.offline_embedding_saver.save(item)
            del micro_batch
            return 0.0, {}

        with torch.no_grad():
            micro_batch = self.condition_model.process_condition(**micro_batch)
        with self.base.model_fwd_context:
            outputs = self.base.model(**micro_batch)

        loss: torch.Tensor
        loss_dict: Dict[str, torch.Tensor]
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
                data_iterator = iter(self.base.train_dataloader)
            else:
                data_iterator = None

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()
            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.on_train_end()

        synchronize()

        if self.training_task == "offline_embedding":
            self.offline_embedding_saver.save_last()

        self.base.destroy_distributed()
