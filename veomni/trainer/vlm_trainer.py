from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import build_multimodal_chat_template
from ..data.multimodal.data_transform import (
    process_sample_qwen2_5_vl,
    process_sample_qwen3_vl,
    process_sample_qwen_omni,
)
from ..models import build_foundation_model, build_processor
from ..utils import helper
from ..utils.model_utils import pretty_print_trainable_parameters
from .base import BaseTrainer


logger = helper.create_logger(__name__)
MAX_PIXELS = 768 * 28 * 28


@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    freeze_audio_tower: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the audio tower parameters."},
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
class MyModelArguments(ModelArguments):
    encoder_data_balance: Optional[bool] = field(
        default=False, metadata={"help": "Whether to balance encoder data for qwen3-vl model"}
    )
    encoder_data_balance_sorting_algo: Optional[str] = field(
        default="post_mbs_balancing_greedy_without_pad",
        metadata={
            "help": "The sorting algorithm of encoder data balance. All viable algorithms are defined in "
            "veomni/utils/data_balance/balance_sorting_algo.py, SORTING_ALGO_FUNC"
        },
    )


@dataclass
class Arguments(VeOmniArguments):
    model: "MyModelArguments" = field(default_factory=MyModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


class VLMTrainer(BaseTrainer):
    def build_model_assets(self):
        self.processor = build_processor(self.args.model.tokenizer_path, max_pixels=MAX_PIXELS)
        if self.model_config.model_type not in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.chat_template = build_multimodal_chat_template(self.args.data.chat_template, self.processor.tokenizer)
            return [self.processor, self.chat_template]
        else:
            self.chat_template = None
            return [self.processor]

    def build_data_collate_info(self):
        if self.model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            return {
                "audio_feature_lengths": (0, False, None, None),
                "input_features": (0, True, 0, 1),
                "audio_mask": (-1, False, 0, 1),
            }
        else:
            return {}

    def freeze_module(self):
        args: Arguments = self.args
        if self.model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.model.disable_talker()

        if args.train.freeze_vit:
            if self.model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
                self.model.thinker.visual.requires_grad_(False)
                self.model.thinker.visual.merger.requires_grad_(True)
            else:
                self.model.visual.requires_grad_(False)

        if args.train.freeze_audio_tower and self.model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.model.thinker.audio_tower.requires_grad_(False)
            self.model.thinker.audio_tower.proj.requires_grad_(True)

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

        if self.model_config.model_type in ("qwen3_vl", "qwen3_vl_moe"):
            process_function = process_sample_qwen3_vl
            position_id_func = self.model.get_position_id_func()
        elif self.model_config.model_type in ("qwen2_5_vl", "qwen2_vl"):
            process_function = process_sample_qwen2_5_vl
            position_id_func = self.model.get_position_id_func()
        elif self.model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            process_function = process_sample_qwen_omni
            position_id_func = self.model.thinker.get_position_id_func()
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

    def _build_model(self):
        args: Arguments = self.args
        logger.info_rank0("Build model")
        self.model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
            attn_implementation=args.model.attn_implementation,
            moe_implementation=args.model.moe_implementation,
            init_device=args.train.init_device,
            config_kwargs=self.build_model_config_kwargs(),
            encoder_data_balance=args.model.encoder_data_balance,
            encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
        )
        self.model_config = self.model.config

        # model assets
        self.model_assets = [self.model_config]
        self.model_assets.extend(self.build_model_assets())

        # freeze module
        self.freeze_module()
        pretty_print_trainable_parameters(self.model)
        helper.print_device_mem_info("VRAM usage after building model")
