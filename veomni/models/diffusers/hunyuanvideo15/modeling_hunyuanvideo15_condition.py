"""Frozen native encoders feeding VeOmni's online DiT lifecycle."""

import re
from pathlib import Path

import torch
from transformers import PreTrainedModel

from ....distributed.parallel_state import get_parallel_state
from ....utils import logging
from ....utils.device import get_device_type
from .conditioning import FlowMatchingConditioner, prepare_video
from .configuration_hunyuanvideo15 import HunyuanVideo15ConditionConfig


logger = logging.get_logger(__name__)


class HunyuanVideo15ConditionModel(PreTrainedModel):
    config_class = HunyuanVideo15ConditionConfig

    def __init__(self, config, meta_init=False, **kwargs):
        super().__init__(config)
        if meta_init:
            raise NotImplementedError("Only online_training is supported; cached conditions are not validated")
        device = config.generator_device or get_device_type()
        self.noise = FlowMatchingConditioner(
            config.seed + get_parallel_state().dp_rank, device, config.train_timestep_shift, config.num_train_timesteps
        )
        self._load_components(device)
        self.requires_grad_(False)
        self.eval()

    @property
    def generator(self):
        return self.noise.generator

    def rng_state_dict(self):
        return self.noise.rng_state_dict()

    def load_rng_state_dict(self, state):
        self.noise.load_rng_state_dict(state)

    def train(self, mode=True):
        return super().train(False)

    def _load_components(self, device):
        from .native.glyph_encoder import load_glyph_byT5_v2
        from .native.text_encoder import PROMPT_TEMPLATE, TextEncoder
        from .native.vae import AutoencoderKLConv3D

        base = Path(self.config.base_model_path)
        glyph = base / "text_encoder/Glyph-SDXL-v2"
        required = [
            base / "vae",
            base / "text_encoder/llm",
            base / "text_encoder/byt5-small",
            glyph / "checkpoints/byt5_model.pt",
            glyph / "assets/color_idx.json",
            glyph / "assets/multilingual_10-lang_idx.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        for component in required[:3]:
            if component.is_dir() and not any(component.glob("*.safetensors")) and not any(component.glob("*.bin")):
                missing.append(f"{component}: pretrained weights")
        if missing:
            raise FileNotFoundError("Missing pretrained components: " + "; ".join(missing))
        self.text_encoder = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            text_encoder_path=str(base / "text_encoder/llm"),
            max_length=self.config.max_text_length,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=logger,
            device=device,
        )
        self.vae, info = AutoencoderKLConv3D.from_pretrained(
            str(base / "vae"), local_files_only=True, torch_dtype=torch.float32, output_loading_info=True
        )
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"VAE checkpoint mismatch: {info}")
        self.vae.to(device).eval()
        if self.vae.config.latent_channels != 32:
            raise ValueError("Expected the native 32-channel VAE")
        self.vae.enable_tiling()
        values = load_glyph_byT5_v2(
            {
                "byT5_google_path": str(base / "text_encoder/byt5-small"),
                "byT5_ckpt_path": str(glyph / "checkpoints/byt5_model.pt"),
                "multilingual_prompt_format_color_path": str(glyph / "assets/color_idx.json"),
                "multilingual_prompt_format_font_path": str(glyph / "assets/multilingual_10-lang_idx.json"),
                "byt5_max_length": self.config.byt5_max_length,
            },
            device,
        )
        self.byt5_model, self.byt5_tokenizer = values["byt5_model"], values["byt5_tokenizer"]

    def _encode_glyph(self, prompt):
        matches = re.findall(r'"(.*?)"|\u201c(.*?)\u201d', prompt)
        texts = list(dict.fromkeys(a or b for a, b in matches))
        device = self.generator.device
        if not texts:
            return (
                torch.zeros(1, self.config.byt5_max_length, 1472, device=device),
                torch.zeros(1, self.config.byt5_max_length, dtype=torch.bool, device=device),
            )
        # Official prompt formatting with no font/color style.
        formatted = "".join(f'Text "{text}". ' for text in texts)
        tokens = self.byt5_tokenizer(
            formatted,
            padding="max_length",
            max_length=self.config.byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        states = self.byt5_model(tokens.input_ids, attention_mask=tokens.attention_mask.float())[0]
        return states, tokens.attention_mask.bool()

    @torch.no_grad()
    def get_condition(self, inputs, videos, **kwargs):
        if len(inputs) != len(videos):
            raise ValueError("Caption/video counts differ")
        result = {
            k: [] for k in ("latents", "text_states", "encoder_attention_mask", "byt5_text_states", "byt5_text_mask")
        }
        for prompt, sample in zip(inputs, videos):
            if len(sample) != 1:
                raise ValueError("T2V requires exactly one video per caption")
            pixels = prepare_video(sample[0], self.config.height, self.config.width, self.config.num_frames)
            posterior = self.vae.encode(pixels.to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist
            latents = self.noise.sample_posterior(posterior.parameters)
            latents = (latents - (self.vae.config.shift_factor or 0)) * self.vae.config.scaling_factor
            tokens = self.text_encoder.text2tokens(prompt, data_type="video", max_length=self.config.max_text_length)
            text = self.text_encoder.encode(tokens, data_type="video", device=self.generator.device)
            glyph, glyph_mask = self._encode_glyph(prompt)
            for key, value in zip(result, [latents, text.hidden_state, text.attention_mask.bool(), glyph, glyph_mask]):
                result[key].append(value)
        return result

    @torch.no_grad()
    def process_condition(self, latents, text_states, encoder_attention_mask, byt5_text_states, byt5_text_mask):
        fields = [latents, text_states, encoder_attention_mask, byt5_text_states, byt5_text_mask]
        if not latents or any(len(x) != len(latents) for x in fields):
            raise ValueError("Empty or misaligned conditions")
        result = {
            k: []
            for k in (
                "hidden_states",
                "timestep",
                "training_target",
                "text_states",
                "encoder_attention_mask",
                "byt5_text_states",
                "byt5_text_mask",
            )
        }
        dtype = self.config.dtype
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        for latent, text, mask, glyph, glyph_mask in zip(*fields):
            sampled = self.noise(latent.to(device=self.generator.device, dtype=dtype))
            for key, value in sampled.items():
                result[key].append(value)
            for key, value, target_dtype in (
                ("text_states", text, dtype),
                ("encoder_attention_mask", mask, torch.bool),
                ("byt5_text_states", glyph, dtype),
                ("byt5_text_mask", glyph_mask, torch.bool),
            ):
                result[key].append(value.to(device=self.generator.device, dtype=target_dtype))
        return result
