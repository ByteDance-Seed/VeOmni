"""Configuration for the Cosmos3 condition model (Wan2.2 VAE + Qwen3-VL tokenizer + flow sampler).

Defaults are the open-source ``vision_sft_nano`` recipe's.
"""

from __future__ import annotations

import os
from typing import Optional

from veomni.models.configuration_base import VeOmniPretrainedConfig


class Cosmos3ConditionModelConfig(VeOmniPretrainedConfig):
    model_type = "Cosmos3ConditionModel"

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        vae_subfolder: str = "vae",
        text_tokenizer_subfolder: str = "text_tokenizer",
        num_train_timesteps: int = 1000,
        shift: Optional[dict] = None,
        train_time_video_distribution: str = "waver",
        resolution: str = "256",
        latent_patch_size: int = 2,
        max_sequence_length: int = 2048,
        append_generation_markers: bool = True,
        vae_dtype: str = "bfloat16",
        conditioning_config: Optional[dict] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path
        self.vae_subfolder = vae_subfolder
        self.text_tokenizer_subfolder = text_tokenizer_subfolder
        # The DiT sees ``sigma * num_train_timesteps``.
        self.num_train_timesteps = num_train_timesteps
        # ``{resolution bucket: shift}`` of the sigma warp ``shift * t / (1 + (shift - 1) * t)``.
        self.shift = shift if shift is not None else {"256": 3, "480": 5, "720": 10}
        self.train_time_video_distribution = train_time_video_distribution
        self.resolution = str(resolution)
        self.latent_patch_size = latent_patch_size
        self.max_sequence_length = max_sequence_length
        # Append ``[eos, <|vision_start|>]`` after truncation, as the reference's packer does.
        self.append_generation_markers = append_generation_markers
        # bf16 like the reference VAE: its bf16 rounding is part of the latents being matched.
        self.vae_dtype = vae_dtype
        # ``{clean prefix latent frames: weight}``: 70% T2V, 20% I2V (first latent frame), 10% V2V (first two).
        self.conditioning_config = conditioning_config if conditioning_config is not None else {0: 0.7, 1: 0.2, 2: 0.1}
        self.seed = seed
        self.shift_for_resolution()
        self.conditioning_probabilities()

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        """Inject the checkpoint root as ``base_model_path``; the root may carry no ``config.json``."""
        try:
            config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        except OSError:
            config_dict = {}
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs

    def component_dir(self, subfolder: str) -> str:
        """``base_model_path/subfolder``, or the root when that subfolder does not exist."""
        base = self.base_model_path or ""
        candidate = os.path.join(base, subfolder) if subfolder else base
        return candidate if os.path.isdir(candidate) else base

    # The two readers below normalize on every call: ``from_pretrained`` applies the run's
    # ``condition_model_cfg`` with ``setattr``, bypassing ``__init__``.

    def conditioning_probabilities(self) -> list[tuple[int, float]]:
        """Normalized ``[(clean_prefix_latent_frames, probability), ...]`` in configured order."""
        weights = [(int(str(frames).strip()), float(weight)) for frames, weight in self.conditioning_config.items()]
        total = sum(weight for _, weight in weights)
        if not weights or total <= 0 or any(frames < 0 or weight < 0 for frames, weight in weights):
            raise ValueError(f"invalid conditioning_config {self.conditioning_config!r}.")
        return [(frames, weight / total) for frames, weight in weights]

    def shift_for_resolution(self, resolution: Optional[str] = None) -> float:
        bucket = str(resolution if resolution is not None else self.resolution)
        shifts = {str(key): float(value) for key, value in self.shift.items()}
        if bucket not in shifts:
            raise ValueError(f"resolution {bucket!r} has no entry in shift={self.shift!r}.")
        return shifts[bucket]
