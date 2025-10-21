import contextlib
from typing import Callable

import torch
from transformers import AutoConfig, AutoModel

from ..models.auto import build_processor
from ..utils import logging
from ..utils.model_utils import pretty_print_trainable_parameters
from .base_trainer import DiTBaseTrainer


logger = logging.get_logger(__name__)


@contextlib.contextmanager
def null_inference_context(model, scale: float):
    yield


class DiTBaseGenerator(DiTBaseTrainer):
    def __init__(
        self,
        model_path: str,
        build_foundation_model_func: Callable = None,
        condition_model_path: str = None,
        condition_model_cfg: dict = {},
        num_samples_per_prompt: int = 1,
        attn_implementation: str = "eager",
        moe_implementation: str = "eager",
    ):
        logger.info_rank0("Prepare condition model.")
        condition_model_config = AutoConfig.from_pretrained(condition_model_path, **condition_model_cfg)

        self.condition_model = AutoModel.from_pretrained(
            condition_model_path,
            torch_dtype=torch.bfloat16,
            config=condition_model_config,
        ).cuda()
        self.condition_processor = build_processor(condition_model_path)

        logger.info_rank0("Prepare dit model.")
        self.dit_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation=attn_implementation,
        ).eval()

        self.num_samples_per_prompt = num_samples_per_prompt

        pretty_print_trainable_parameters(self.dit_model)

    def forward(self, raw_data):
        # only support online embedding for generation
        processed_data = self.condition_processor.preprocess_infer(raw_data)

        with torch.no_grad():
            condition_embed = self.condition_model.get_condition_infer(
                processed_data, num_samples_per_prompt=self.num_samples_per_prompt
            )
            processed_cond = self.condition_model.process_condition_infer(condition_embed)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            latents = self.dit_model.generate(**processed_cond, cfg_scale=3.5, ada_precompute=False)
            decoded_latents = self.condition_model.postprocess(latents)
        videos = self.condition_processor.postprocess(decoded_latents)
        return videos
