from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import build_multimodal_chat_template
from ..data.data_transform import process_sample_qwen2_5_vl, process_sample_qwen3_vl
from ..models import build_processor
from ..utils import helper
from .base import BaseTrainer


logger = helper.create_logger(__name__)
MAX_PIXELS = 768 * 28 * 28


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


@dataclass
class MyDataArguments(DataArguments):
    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input."},
    )


@dataclass
class Arguments(VeOmniArguments):
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


class VLMTrainer(BaseTrainer):
    def build_model_assets(self):
        self.processor = build_processor(self.args.model.tokenizer_path, max_pixels=MAX_PIXELS)
        self.chat_template = build_multimodal_chat_template(self.args.data.chat_template, self.processor.tokenizer)
        return [self.chat_template]

    def freeze_module(self):
        args: Arguments = self.args
        if args.train.freeze_vit:
            self.model.visual.requires_grad_(False)

    def build_param_groups(self):
        args: Arguments = self.args
        vit_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "visual" in name:
                    vit_params.append(param)
                else:
                    other_params.append(param)

        return [{"params": vit_params, "lr": args.train.vit_lr}, {"params": other_params, "lr": args.train.lr}]

    def build_data_transform(self):
        args: Arguments = self.args
        position_id_func = self.model.get_position_id_func()
        if self.model_config.model_type in ("qwen3_vl", "qwen3_vl_moe"):
            process_function = process_sample_qwen3_vl
        elif self.model_config.model_type == "qwen2_5_vl":
            process_function = process_sample_qwen2_5_vl
        else:
            raise NotImplementedError(f"Unsupported model type: {self.model_config.model_type}.")
        data_transform = partial(
            process_function,
            processor=self.processor,
            chat_template=self.chat_template,
            position_id_func=position_id_func,
            **args.data.mm_configs,
        )
        return data_transform
