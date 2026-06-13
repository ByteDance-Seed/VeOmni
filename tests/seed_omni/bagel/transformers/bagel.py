"""Thin assembly helpers around the vendored official BAGEL implementation."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from transformers.initialization import no_init_weights
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tests.seed_omni.bagel.transformers.vendor.data.data_utils import add_special_tokens
from tests.seed_omni.bagel.transformers.vendor.modeling.autoencoder import AutoEncoderParams, load_ae
from tests.seed_omni.bagel.transformers.vendor.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from tests.seed_omni.bagel.transformers.vendor.modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from veomni.models.module_utils import init_empty_weights


class BagelOfficialReference(nn.Module):
    """Own the official BAGEL assembly so captures do not import vendor internals."""

    def __init__(
        self,
        model: Bagel,
        *,
        tokenizer: Qwen2Tokenizer | None = None,
        new_token_ids: dict[str, int] | None = None,
        vae_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids or {}
        self.config = model.config

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        *,
        llm_config: Qwen2Config,
        vit_config: SiglipVisionConfig | None = None,
        vae_config: Any | None = None,
        visual_gen: bool,
        visual_und: bool,
        tokenizer: Qwen2Tokenizer | None = None,
        new_token_ids: dict[str, int] | None = None,
        init_on_meta: bool = True,
        latent_patch_size: int | None = None,
        max_latent_size: int | None = None,
        timestep_shift: float | None = None,
        vit_max_num_patch_per_side: int | None = None,
        connector_act: str | None = None,
    ) -> BagelOfficialReference:
        _ensure_default_rope_init()
        _normalize_llm_config(llm_config)
        config = BagelConfig(
            visual_gen=visual_gen,
            visual_und=visual_und,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            **_optional_bagel_config_kwargs(
                latent_patch_size=latent_patch_size,
                max_latent_size=max_latent_size,
                timestep_shift=timestep_shift,
                vit_max_num_patch_per_side=vit_max_num_patch_per_side,
                connector_act=connector_act,
            ),
        )
        context = _empty_init_context() if init_on_meta else nullcontext()
        with context:
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config) if visual_und and vit_config is not None else None
            model = Bagel(language_model, vit_model, config)
            if visual_und and vit_model is not None:
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=init_on_meta)
        return cls(model, tokenizer=tokenizer, new_token_ids=new_token_ids)

    @classmethod
    def _from_checkpoint_root(
        cls,
        model_root: str | Path,
        *,
        visual_gen: bool,
        visual_und: bool,
        init_on_meta: bool = True,
        torch_dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        load_weights: bool = True,
        latent_patch_size: int = 2,
        max_latent_size: int = 64,
        timestep_shift: float = 3.0,
        vit_max_num_patch_per_side: int = 70,
        connector_act: str = "gelu_pytorch_tanh",
    ) -> BagelOfficialReference:
        root = Path(model_root)
        dtype = _resolve_dtype(torch_dtype)
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        llm_config = Qwen2Config.from_json_file(str(root / "llm_config.json"))
        _normalize_llm_config(llm_config)

        tokenizer = Qwen2Tokenizer.from_pretrained(str(root), local_files_only=True)
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        if num_new_tokens > 0:
            llm_config.vocab_size = len(tokenizer)

        vit_config = SiglipVisionConfig.from_json_file(str(root / "vit_config.json")) if visual_und else None
        if vit_config is not None:
            vit_config.rope = False
            vit_config.num_hidden_layers -= 1
        reference = cls.from_configs(
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=_default_vae_config() if visual_gen else None,
            visual_gen=visual_gen,
            visual_und=visual_und,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            init_on_meta=init_on_meta,
            latent_patch_size=latent_patch_size,
            max_latent_size=max_latent_size,
            timestep_shift=timestep_shift,
            vit_max_num_patch_per_side=vit_max_num_patch_per_side,
            connector_act=connector_act,
        )
        if load_weights:
            _load_weights(
                reference.model,
                root / "ema.safetensors",
                device=target_device,
                dtype=dtype,
                visual_gen=visual_gen,
                visual_und=visual_und,
            )
        reference.model.to(device=target_device)
        if visual_und:
            vae_model, _ = load_ae(str(root / "ae.safetensors"))
            reference.vae_model = vae_model.to(device=target_device, dtype=dtype).eval()
        reference.eval()
        return reference


def load_vendored_model(model_root: str | Path | None, **kwargs: Any) -> BagelOfficialReference:
    """Functional loader required by the shared vendored reference contract."""

    if model_root is None:
        raise ValueError("BAGEL vendored reference requires reference.checkpoint.")
    return BagelOfficialReference._from_checkpoint_root(model_root, **kwargs)


class ReferenceImageResize(nn.Module):
    """Official BAGEL resize transform used by vendored reference recipes."""

    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Image.Image | torch.Tensor, img_num: int = 1) -> Image.Image | torch.Tensor:
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height)
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height)
        return TVF.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)

    def _apply_scale(self, width: int, height: int, scale: float) -> tuple[int, int]:
        return (
            max(self.stride, int(round(round(width * scale) / self.stride) * self.stride)),
            max(self.stride, int(round(round(height * scale) / self.stride) * self.stride)),
        )


class ReferenceImageTransform:
    """Official BAGEL image transform callable expected by prepare_*_images."""

    def __init__(
        self,
        *,
        max_image_size: int,
        min_image_size: int,
        image_stride: int,
        max_pixels: int,
    ) -> None:
        self.resize_transform = ReferenceImageResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    def __call__(self, img: Image.Image, img_num: int = 1) -> torch.Tensor:
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        return self.normalize_transform(img)


def make_reference_image(width: int, height: int) -> Image.Image:
    """Create the deterministic reference-side input image used by edit parity."""

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    rgb = np.stack([xx, yy, ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(rgb)


@contextmanager
def _empty_init_context():
    with no_init_weights(), init_empty_weights():
        yield


def _normalize_llm_config(llm_config: Qwen2Config) -> None:
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = getattr(llm_config, "bos_token_id", 0)


def _ensure_default_rope_init() -> None:
    if "default" in ROPE_INIT_FUNCTIONS:
        return
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _resolve_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"fp32", "float32"}:
        return torch.float32
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported BAGEL reference dtype: {dtype!r}")


def _default_vae_config() -> AutoEncoderParams:
    return AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )


def _optional_bagel_config_kwargs(
    *,
    latent_patch_size: int | None,
    max_latent_size: int | None,
    timestep_shift: float | None,
    vit_max_num_patch_per_side: int | None,
    connector_act: str | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if latent_patch_size is not None:
        values["latent_patch_size"] = latent_patch_size
    if max_latent_size is not None:
        values["max_latent_size"] = max_latent_size
    if timestep_shift is not None:
        values["timestep_shift"] = timestep_shift
    if vit_max_num_patch_per_side is not None:
        values["vit_max_num_patch_per_side"] = vit_max_num_patch_per_side
    if connector_act is not None:
        values["connector_act"] = connector_act
    return values


def _load_weights(
    model: Bagel,
    weights_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    visual_gen: bool,
    visual_und: bool,
) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"BAGEL official reference weights not found: {weights_path}")
    prefixes = ("language_model",)
    if visual_gen:
        prefixes = ("language_model.", "vae2llm.", "llm2vae.", "time_embedder.", "latent_pos_embed.")
    if visual_und:
        prefixes = (
            *prefixes,
            "vit_model.",
            "connector.",
            "vit_pos_embed.",
        )
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefixes):
                state_dict[key] = handle.get_tensor(key).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    relevant_missing = [key for key in missing if key.startswith(prefixes)]
    relevant_unexpected = [key for key in unexpected if key.startswith(prefixes)]
    if relevant_missing:
        raise RuntimeError(f"Missing BAGEL reference weight keys: {relevant_missing[:20]}")
    if relevant_unexpected:
        raise RuntimeError(f"Unexpected BAGEL reference weight keys: {relevant_unexpected[:20]}")


def _compute_default_rope_parameters(config: Any, device: torch.device | None = None, **kwargs: Any):
    del kwargs
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


__all__ = ["BagelOfficialReference", "ReferenceImageTransform", "load_vendored_model", "make_reference_image"]
