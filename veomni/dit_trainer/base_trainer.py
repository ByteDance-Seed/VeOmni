import os
import pickle as pk
from typing import Callable, Dict, Literal

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import init_empty_weights

from ..models import save_model_assets, save_model_weights
from ..models.auto import build_processor
from ..utils import logging
from ..utils.model_utils import pretty_print_trainable_parameters


logger = logging.get_logger(__name__)


class OfflineEmbeddingSaver:
    def __init__(self, shard_num: int = 1, max_shard=1000):
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

    def lazy_init(self, save_path: str = None, dataset_length: int = 0):
        import math

        self.save_path = save_path
        self.dataset_length = dataset_length
        self.batch_len = math.ceil(dataset_length / self.shard_num)
        logger.info(f"Rank [{self.dp_rank}] save to [{self.save_path}] each batch_len [{self.batch_len}].")
        self.rest_len = self.dataset_length


class DiTBaseTrainer:
    def __init__(
        self,
        model_path: str,
        build_foundation_model_func: Callable,
        build_parallelize_model_func: Callable,
        training_task: Literal["online_training", "offline_training", "offline_embedding"] = "online_training",
        condition_model_path: str = None,
        condition_model_cfg: dict = {},
        lora_config: dict = None,
        offline_embedding_saver_cfg: dict = {},
        **kwargs,
    ):
        logger.info_rank0("Prepare condition model.")
        self.training_task = training_task

        condition_model_config = AutoConfig.from_pretrained(condition_model_path, **condition_model_cfg)
        if training_task == "offline_training":
            logger.info_rank0(f"Task: {training_task}, prepare condition model with empty weights.")
            with init_empty_weights():
                self.condition_model = AutoModel.from_pretrained(
                    condition_model_path,
                    torch_dtype=torch.bfloat16,
                    config=condition_model_config,
                )
        else:
            logger.info_rank0(f"Task: {training_task}, prepare condition model fully loaded.")
            self.condition_model = AutoModel.from_pretrained(
                condition_model_path,
                torch_dtype=torch.bfloat16,
                config=condition_model_config,
            )
            self.condition_model.cuda()
        self.processor = build_processor(condition_model_path)

        if training_task == "offline_training" or training_task == "online_training":
            logger.info_rank0("Prepare dit model.")
            self.build_foundation_model_func = build_foundation_model_func
            self.model_path = model_path
            self.dit_model = self.build_foundation_model_func(config_path=model_path, weights_path=model_path)

            self.lora_config = lora_config
            fsdp_kwargs = self.configure_lora_model()

            if build_parallelize_model_func.keywords["init_device"] == "meta":
                from functools import partial

                def patch_parallel_load_safetensors(weights_path, func, model):
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
                    model=self.dit_model,
                )

            self.dit_model = build_parallelize_model_func(
                model=self.dit_model, fsdp_kwargs=fsdp_kwargs, basic_modules=self.dit_model._no_split_modules
            )
            self.dit_model.train()
            pretty_print_trainable_parameters(self.dit_model)
        else:
            self.offline_embedding_saver = OfflineEmbeddingSaver(**offline_embedding_saver_cfg)

    def get_model_for_training(self):
        return self.dit_model

    def configure_lora_model(self):
        fsdp_kwargs = {}
        if self.lora_config is None:
            self.lora = False
        else:
            lora_adapter_path = self.lora_config.get("lora_adapter", None)
            if lora_adapter_path is not None:
                logger.info_rank0(f"Load lora_adapter from {lora_adapter_path}.")
                from peft import PeftModel

                self.dit_model = PeftModel.from_pretrained(self.dit_model, lora_adapter_path)
            else:
                from peft import LoraConfig, get_peft_model

                lora_config: LoraConfig = LoraConfig(
                    r=self.lora_config["rank"],
                    lora_alpha=self.lora_config["alpha"],
                    target_modules=self.lora_config["lora_modules"],
                )
                logger.info_rank0(f"Init lora: {lora_config.to_dict()}.")
                self.dit_model = get_peft_model(self.dit_model, lora_config)

            self.dit_model.print_trainable_parameters()
            self.lora = True
            fsdp_kwargs["use_orig_params"] = True
        return fsdp_kwargs

    def save_model_weights(self, save_path: str):
        # TODO: ema model save
        if self.lora:
            if self.lora_config.get("save_merge", False):
                logger.info_rank0(f"Save initial lora_adapter to {save_path}.")
                self.dit_model.save_pretrained(save_path)
            else:
                logger.info_rank0(f"Save initial lora merged model to {save_path}.")
                self.dit_model = self.dit_model.merge_and_unload()
                self.dit_model.save_pretrained(save_path)
        else:
            save_model_weights(save_path, self.dit_model.state_dict(), model_assets=[self.dit_model.config])

    def save_hf_model_weights(self, model_state_dict: dict, save_path: str):
        if self.lora:
            import os

            from peft import get_peft_model_state_dict

            from veomni.models.module_utils import _save_state_dict

            model_state_dict = get_peft_model_state_dict(self.dit_model, model_state_dict)
            lora_adapter_save_path = os.path.join(save_path, "adapter_model.bin")
            os.makedirs(save_path, exist_ok=True)
            _save_state_dict(model_state_dict, lora_adapter_save_path, safe_serialization=False)
            self.dit_model.peft_config["default"].save_pretrained(save_path)
            logger.info_rank0(f"Lora adapter saved at {save_path} successfully!")

            if self.lora_config.get("save_merge", False):
                from peft import PeftModel

                del self.dit_model
                model = self.build_foundation_model_func(
                    config_path=self.model_path,
                    weights_path=self.model_path,
                )
                model = PeftModel.from_pretrained(model, save_path)
                model = model.merge_and_unload()  # 合并 LoRA 权重到 base_model
                model.save_pretrained(save_path)
                logger.info_rank0(f"Lora merged model adapter saved at {save_path} successfully!")
        else:
            save_model_weights(save_path, model_state_dict, model_assets=[self.dit_model.config])
            logger.info_rank0(f"Huggingface checkpoint saved at {save_path} successfully!")

    def save_model_assets(self, save_path: str):
        model_assets = [self.dit_model.config]
        save_model_assets(save_path, model_assets)

    def forward(self, **condition_dict):
        if self.training_task == "online_training" or self.training_task == "offline_embedding":
            with torch.no_grad():
                condition_dict = self.condition_model.get_condition(**condition_dict)
        if self.training_task == "offline_embedding":
            self.offline_embedding_saver.save(condition_dict)
            return None
        with torch.no_grad():
            condition_dict = self.condition_model.process_condition(**condition_dict)
        output = self.dit_model(**condition_dict)
        return output
