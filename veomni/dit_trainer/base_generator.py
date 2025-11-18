import torch
from transformers import AutoModel

from ..utils import logging
from .base_trainer import DiTBaseTrainer


logger = logging.get_logger(__name__)


class DiTBaseGenerator(DiTBaseTrainer):
    def __init__(
        self,
        model_path: str,
        condition_model_path: str = None,
        condition_model_cfg: dict = {},
        lora_config: dict = {},
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.training_task = "offline_embedding"
        self.condition_model_path = condition_model_path
        self.condition_model_cfg = condition_model_cfg
        logger.info_rank0("Prepare condition model.")
        self.configure_condition_model()

        logger.info_rank0("Prepare dit model.")
        self.dit_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation=attn_implementation,
        ).eval()
        self.lora_config = lora_config
        self.configure_lora_model()

    def forward(self, processed_data):
        with torch.no_grad():
            condition_embed = self.condition_model.get_condition_infer(processed_data)
            processed_cond = self.condition_model.process_condition_infer(condition_embed)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            latents = self.dit_model.generate(**processed_cond)
            decoded_latents = self.condition_model.postprocess(latents)
        outputs = self.processor.postprocess(decoded_latents)
        return outputs
