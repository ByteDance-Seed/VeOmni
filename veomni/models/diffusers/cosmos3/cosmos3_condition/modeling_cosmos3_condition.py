"""Cosmos3 condition model: frozen Wan2.2 VAE + Qwen3-VL tokenizer + rectified-flow targets.

``get_condition`` encodes each clip and tokenizes its prompt; ``process_condition``
draws sigma (``waver`` + resolution shift), the noise and the T2V/I2V/V2V prefix
mask, and returns ``x_t = noise * sigma + clean * (1 - sigma)``, the target
``noise - clean`` and ``timestep = sigma * num_train_timesteps``. All draws come
from one CPU generator, so the stream is device-independent and resumable.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer

from veomni.models.modeling_base import VeOmniPreTrainedModel as PreTrainedModel

from .....distributed.parallel_state import get_parallel_state
from .....utils import logging
from ..cosmos3_core import TrainTimeSampler, interpolate
from .configuration_cosmos3_condition import Cosmos3ConditionModelConfig
from .packing import cosmos3_packed_length
from .wan22_vae import Wan22VAE


logger = logging.get_logger(__name__)


class Cosmos3ConditionModel(PreTrainedModel):
    config_class = Cosmos3ConditionModelConfig
    supports_gradient_checkpointing = False

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # The condition model runs no attention of its own (VAE + tokenizer).
        return attn_implementation

    def __init__(self, config: Cosmos3ConditionModelConfig, meta_init: bool = False, **kwargs):
        super().__init__(config, **kwargs)
        self.meta_init = meta_init
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(config.seed if config.seed is not None else 42) + get_parallel_state().dp_rank)
        self.train_time_sampler = TrainTimeSampler(config.train_time_video_distribution)
        self._logged_geometry = False

        if not config.base_model_path:
            raise ValueError("Cosmos3ConditionModelConfig.base_model_path is empty; set model.condition_model_path.")
        self._init_vae()
        self._init_tokenizer()

    def _init_vae(self) -> None:
        config = self.config
        if self.meta_init:
            # offline_training never encodes; only the VAE config (scales, statistics) is needed.
            from diffusers import AutoencoderKLWan

            vae = AutoencoderKLWan.from_config(
                AutoencoderKLWan.load_config(config.base_model_path, subfolder=config.vae_subfolder or None)
            )
            vae.eval()
            vae.requires_grad_(False)
            self.vae = Wan22VAE(vae, component_name=config.component_dir(config.vae_subfolder))
        else:
            self.vae = Wan22VAE.from_pretrained(
                config.base_model_path, subfolder=config.vae_subfolder, torch_dtype=getattr(torch, config.vae_dtype)
            )

    def _init_tokenizer(self) -> None:
        """The checkpoint processor's tokenizer when it has one, else ``text_tokenizer/``."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.component_dir(self.config.text_tokenizer_subfolder)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the Cosmos3 text tokenizer from {self.config.base_model_path}."
            ) from exc
        try:
            processor = AutoProcessor.from_pretrained(self.config.base_model_path)
        except Exception:
            processor = None
        if getattr(processor, "tokenizer", None) is not None:
            self.tokenizer = processor.tokenizer

        # The reference appends ``[eos, <|vision_start|>]`` to the text span.
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        vision_start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.generation_marker_ids = torch.tensor([int(eos), int(vision_start)], dtype=torch.long)

    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        return {"generator": self.generator.get_state()}

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator"])

    # ── DiTTrainer hook: token-budget packing ────────────────────────────────────

    def get_dyn_bsz_length_fn(self):
        return partial(
            cosmos3_packed_length,
            spatial_compression_factor=self.vae.spatial_scale,
            temporal_compression_factor=self.vae.temporal_scale,
            latent_patch_size=self.config.latent_patch_size,
        )

    # ── get_condition: raw data -> clean latents and prompt tokens ────────────────

    @torch.no_grad()
    def get_condition(
        self,
        inputs: list[str],
        videos: list,
        resolution: Any = None,
        video_fps: Optional[list] = None,
        **kwargs: Any,
    ) -> dict[str, list]:
        """Per sample: normalized ``latents`` ``[C, T_lat, H_lat, W_lat]``, the shift ``resolution`` bucket,
        the prompt's ``input_ids`` / ``ar_attention_mask`` and the clip ``fps`` (``None`` when unknown)."""
        count = len(inputs)
        buckets = self._resolve_resolutions(resolution, count)
        fps = self._resolve_fps(video_fps, count)
        condition: dict[str, list] = {
            "latents": [],
            "resolution": [],
            "input_ids": [],
            "ar_attention_mask": [],
            "fps": [],
        }
        for index, prompt in enumerate(inputs):
            video = self._prepare_video(self._frames(videos[index]))
            condition["latents"].append(self.vae.encode_latents(video)[0])
            condition["resolution"].append(buckets[index])
            input_ids = self._tokenize_prompt(prompt)
            condition["input_ids"].append(input_ids)
            condition["ar_attention_mask"].append(torch.ones_like(input_ids))
            condition["fps"].append(fps[index])
        if not self._logged_geometry:
            self._logged_geometry = True
            reported = [value for value in fps if value is not None]
            logger.info_rank0(
                f"Cosmos3 condition: {len(reported)}/{count} samples report fps "
                f"{sorted(set(reported)) or 'none (mRoPE falls back to base_fps)'}, prompt tokens "
                f"{[int(ids.numel()) for ids in condition['input_ids']]}."
            )
        return condition

    @staticmethod
    def _resolve_fps(video_fps: Any, count: int) -> list[Optional[float]]:
        if video_fps is None:
            return [None] * count
        values = [video_fps] * count if isinstance(video_fps, (int, float)) else list(video_fps)
        resolved = []
        for value in values:
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"video_fps must be positive and finite, got {value!r}.")
            resolved.append(None if value is None else float(value))
        return resolved

    @staticmethod
    def _frames(entry: Any) -> torch.Tensor:
        """The collator's per-sample entry -> one ``[T, C, H, W]`` clip."""
        if isinstance(entry, (list, tuple)):
            if len(entry) != 1:
                raise ValueError(f"Cosmos3 accepts one video per sample, got {len(entry)}.")
            entry = entry[0]
        if not isinstance(entry, torch.Tensor) or entry.dim() != 4 or entry.shape[1] != 3:
            raise ValueError(
                f"expected one decoded [T, 3, H, W] clip per sample, got {getattr(entry, 'shape', entry)}."
            )
        return entry

    def _prepare_video(self, frames: torch.Tensor) -> torch.Tensor:
        """``[T, C, H, W]`` uint8 or float -> ``[1, 3, T, H, W]`` in ``[-1, 1]`` on the VAE grid."""
        source_dtype = frames.dtype
        video = frames.detach().to(dtype=torch.float32)
        if source_dtype == torch.uint8:
            video = video.div(255.0)
        elif video.numel():
            if float(video.max()) > 1.0 + 1e-3:
                video = video.div(255.0)
            elif float(video.min()) < -1e-3:
                video = video.add(1.0).div(2.0)
        video = video.mul(2.0).sub(1.0)

        # The latent must tile into ``latent_patch_size`` patches after the VAE's spatial stride.
        height, width = video.shape[2], video.shape[3]
        target_h, target_w = self._latent_grid_target(height, width)
        if (target_h, target_w) != (height, width):
            video = F.interpolate(
                video, size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=True
            )
        video = video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        return video.to(device=self.vae.device, dtype=self.vae.dtype)

    def _latent_grid_target(self, height: int, width: int) -> tuple[int, int]:
        grid = self.vae.spatial_scale * int(self.config.latent_patch_size)
        return max(grid, int(round(height / grid)) * grid), max(grid, int(round(width / grid)) * grid)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """The reference's chat-template tokenization, truncated, then the generation markers appended."""
        conversation = [{"role": "user", "content": prompt}]
        token_ids = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, add_vision_id=False, return_dict=False
        )
        input_ids = torch.tensor(list(token_ids), dtype=torch.long)[: self.config.max_sequence_length]
        if self.config.append_generation_markers:
            input_ids = torch.cat([input_ids, self.generation_marker_ids])
        return input_ids

    # ── process_condition: sigma, noise and conditioning -> transformer inputs ──────

    @torch.no_grad()
    def process_condition(
        self,
        latents: list[torch.Tensor],
        resolution: Any = None,
        input_ids: Optional[list[torch.Tensor]] = None,
        ar_attention_mask: Optional[list[torch.Tensor]] = None,
        fps: Optional[list] = None,
        **kwargs: Any,
    ) -> dict[str, list]:
        """The transformer's ``forward`` inputs; ``condition_mask`` is ``[T_lat, 1, 1]`` with ``1`` on clean frames."""
        buckets = self._resolve_resolutions(resolution, len(latents))
        shift = torch.tensor([self.config.shift_for_resolution(bucket) for bucket in buckets], dtype=torch.float32)
        sigmas = self.train_time_sampler(len(buckets), generator=self.generator, shift=shift).to(torch.float32)

        processed: dict[str, list] = {"hidden_states": [], "timestep": [], "training_target": [], "condition_mask": []}
        for index, clean in enumerate(latents):
            sigma = sigmas[index : index + 1]
            noise = torch.randn(clean.shape, generator=self.generator, dtype=torch.float32)
            noise = noise.to(device=clean.device, dtype=clean.dtype)
            # Drawn after the noise, so the noise stream does not depend on the mixture.
            condition_mask = self._sample_condition_mask(clean)
            # Clean frames get sigma 0, i.e. the clean latent; the target stays unmasked.
            effective_sigma = sigma.to(device=noise.device).reshape(1, 1, 1) * (
                1.0 - condition_mask.to(device=noise.device, dtype=sigma.dtype)
            )
            hidden_states, training_target = interpolate(noise, clean, effective_sigma)
            processed["hidden_states"].append(hidden_states)
            processed["timestep"].append((sigma * self.config.num_train_timesteps).to(torch.float32))
            processed["training_target"].append(training_target)
            processed["condition_mask"].append(condition_mask)
        if input_ids is not None:
            processed["input_ids"] = list(input_ids)
        if ar_attention_mask is not None:
            processed["ar_attention_mask"] = list(ar_attention_mask)
        if fps is not None:
            processed["fps"] = list(fps)
        return processed

    def _sample_condition_mask(self, clean: torch.Tensor) -> torch.Tensor:
        """Clean prefix of 0 (T2V), 1 (I2V) or 2 (V2V) latent frames, at least one frame left noisy."""
        latent_frames = int(clean.shape[1])
        options = self.config.conditioning_probabilities()
        draw = float(torch.rand((), generator=self.generator, dtype=torch.float32))
        cumulative = 0.0
        num_clean = options[-1][0]
        for frames, probability in options:
            cumulative += probability
            if draw < cumulative:
                num_clean = frames
                break
        num_clean = max(0, min(num_clean, latent_frames - 1))
        # float32: a bf16 mask would round ``1 - mask`` and leak sigma into clean frames.
        condition_mask = torch.zeros((latent_frames, 1, 1), device=clean.device, dtype=torch.float32)
        if num_clean:
            condition_mask[:num_clean] = 1.0
        return condition_mask

    def _resolve_resolutions(self, resolution: Any, count: int) -> list[str]:
        if resolution is None:
            return [self.config.resolution] * count
        if isinstance(resolution, str):
            return [resolution] * count
        return [str(bucket) for bucket in resolution]
