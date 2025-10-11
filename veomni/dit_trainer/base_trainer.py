from typing import Callable, Literal
from transformers import AutoModel, AutoProcessor
from ..models.auto import build_processor
from transformers.modeling_utils import init_empty_weights
import torch
from ..utils import logging
from ..utils.model_utils import pretty_print_trainable_parameters
from ..models import save_model_weights, save_model_assets


logger = logging.get_logger(__name__)

class DiTTrainerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, value: Callable):
        if name in cls._registry:
            raise ValueError(f"Trainer '{name}' is already registered")
        cls._registry[name] = value

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise KeyError(f"Trainer '{name}' is not registered, available trainers: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        trainer_cls = cls.get(name)
        return trainer_cls(*args, **kwargs)

    @classmethod
    def available_trainers(cls):
        return list(cls._registry.keys())


class DiTBaseTrainer:
    def __init__(
        self,
        model_path: str,
        build_foundation_model_func: Callable,
        build_parallelize_model_func: Callable,
        training_task: Literal["online_training", "offline_training", "offline_embedding"] = "online_training",
        condition_model_path: str = None,
        lora_config: dict = None,
        **kwargs,
    ):
        logger.info_rank0("Prepare condition model.")
        self.training_task = training_task
        if training_task == "offline_training":
            logger.info_rank0(f"Task: {training_task}, prepare condition model with empty weights.")
            with init_empty_weights():
                self.condition_model = AutoModel.from_pretrained(
                    condition_model_path,
                    torch_dtype=torch.bfloat16,
                    
                )
        else:
            logger.info_rank0(f"Task: {training_task}, prepare condition model fully loaded.")
            self.condition_model = AutoModel.from_pretrained(
                condition_model_path,
                torch_dtype=torch.bfloat16,
                
            )
        self.processor = build_processor(condition_model_path)

        if training_task != "offline_embedding":
            logger.info_rank0("Prepare dit model.")
            self.dit_model = build_foundation_model_func(config_path=model_path, weights_path=model_path)

            self.lora_config = lora_config
            fsdp_kwargs = self.configure_lora_model()
            pretty_print_trainable_parameters(self.dit_model)
            self.dit_model = build_parallelize_model_func(
                model=self.dit_model,
                fsdp_kwargs=fsdp_kwargs,
                basic_modules=self.dit_model._no_split_modules
            )
    
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

    def save_model_assets(self, save_path: str):
        model_assets = [self.dit_model.config]
        save_model_assets(save_path, model_assets)

    
    def forward(self, **kwargs):
        import ipdb;ipdb.set_trace()
