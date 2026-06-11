"""Capture official BAGEL input-image VAE-context image-generation fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights


pytestmark = pytest.mark.skip(reason="BAGEL official capture helper; run explicitly to generate parity fixtures.")

DEFAULT_OUTPUT = Path("outputs/bagel_v2/parity/image_edit_vae_context_one_step.pt")
DEFAULT_PROMPT = "Turn the scene into a clean watercolor illustration."


def _log(message: str) -> None:
    print(f"[bagel-image-edit-fixture] {message}", flush=True)


class _MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        antialias: bool = True,
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value: float) -> int:
        return max(self.stride, int(round(value / self.stride) * self.stride))

    def _apply_scale(self, width: int, height: int, scale: float) -> tuple[int, int]:
        new_width = self._make_divisible(round(width * scale))
        new_height = self._make_divisible(round(height * scale))
        return new_width, new_height

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
            new_width, new_height = self._apply_scale(new_width, new_height, scale)
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)
        return TVF.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class _ImageTransform:
    def __init__(
        self,
        max_image_size: int,
        min_image_size: int,
        image_stride: int,
        max_pixels: int,
    ):
        self.resize_transform = _MaxLongEdgeMinShortEdgeResize(
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


def _compute_default_rope_parameters(
    config: Any, device: torch.device | None = None, **kwargs: Any
) -> tuple[torch.Tensor, float]:
    del kwargs
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _import_official(official_repo: Path) -> dict[str, Any]:
    _log(f"importing official BAGEL from {official_repo}")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    sys.path.insert(0, str(official_repo))
    from data.data_utils import add_special_tokens, pil_img2rgb
    from modeling.autoencoder import AutoEncoderParams, load_ae
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.bagel.qwen2_navit import NaiveCache
    from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    return {
        "add_special_tokens": add_special_tokens,
        "AutoEncoderParams": AutoEncoderParams,
        "Bagel": Bagel,
        "BagelConfig": BagelConfig,
        "load_ae": load_ae,
        "NaiveCache": NaiveCache,
        "pil_img2rgb": pil_img2rgb,
        "Qwen2Config": Qwen2Config,
        "Qwen2ForCausalLM": Qwen2ForCausalLM,
        "Qwen2Tokenizer": Qwen2Tokenizer,
        "SiglipVisionConfig": SiglipVisionConfig,
        "SiglipVisionModel": SiglipVisionModel,
    }


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast("cuda", enabled=True, dtype=dtype)


def _move_tensors(data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}


def _cpu_tensors(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in data.items()}


def _cache_to_tensors(cache: Any) -> dict[str, Any]:
    keys: list[torch.Tensor | None] = []
    values: list[torch.Tensor | None] = []
    for idx in range(cache.num_layers):
        key = cache.key_cache[idx]
        value = cache.value_cache[idx]
        keys.append(None if key is None else key.detach().cpu())
        values.append(None if value is None else value.detach().cpu())
    return {"num_layers": cache.num_layers, "key": keys, "value": values}


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _make_image(width: int, height: int) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    rgb = np.stack([xx, yy, ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(rgb)


def _vae_config(official: dict[str, Any]) -> Any:
    return official["AutoEncoderParams"](
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


def _load_state(path: Path, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    prefixes = (
        "language_model.",
        "vit_model.",
        "connector.",
        "vit_pos_embed.",
        "vae2llm.",
        "llm2vae.",
        "time_embedder.",
        "latent_pos_embed.",
    )
    _log(f"loading text/vision/flow weights from {path} to {device} as {dtype}")
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefixes):
                state_dict[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    _log(f"loaded {len(state_dict)} tensors")
    return state_dict


def _build_model(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, int], Any, Any, torch.device, torch.dtype]:
    official = _import_official(args.official_repo)
    device = torch.device(args.device)
    torch_dtype = _resolve_dtype(args.dtype)

    llm_config = official["Qwen2Config"].from_json_file(str(args.model_root / "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = llm_config.bos_token_id

    vit_config = official["SiglipVisionConfig"].from_json_file(str(args.model_root / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    tokenizer = official["Qwen2Tokenizer"].from_pretrained(str(args.model_root), local_files_only=True)
    tokenizer, new_token_ids, num_new_tokens = official["add_special_tokens"](tokenizer)
    if num_new_tokens > 0:
        llm_config.vocab_size = len(tokenizer)

    config = official["BagelConfig"](
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=_vae_config(official),
        latent_patch_size=args.latent_patch_size,
        max_latent_size=args.max_latent_size,
        vit_max_num_patch_per_side=args.vit_max_num_patch_per_side,
        connector_act="gelu_pytorch_tanh",
        timestep_shift=args.timestep_shift,
    )
    _log("constructing official full BAGEL model on meta parameters")
    with no_init_weights(), init_empty_weights():
        language_model = official["Qwen2ForCausalLM"](llm_config)
        vit_model = official["SiglipVisionModel"](vit_config)
        model = official["Bagel"](language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    state = _load_state(args.model_root / "ema.safetensors", device=device, dtype=torch_dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    relevant = (
        "language_model.",
        "vit_model.",
        "connector.",
        "vit_pos_embed.",
        "vae2llm.",
        "llm2vae.",
        "time_embedder.",
    )
    unexpected_relevant = [key for key in unexpected if key.startswith(relevant)]
    missing_relevant = [key for key in missing if key.startswith(relevant)]
    if unexpected_relevant:
        raise RuntimeError(f"Unexpected official BAGEL keys: {unexpected_relevant[:20]}")
    if missing_relevant:
        raise RuntimeError(f"Missing official BAGEL keys: {missing_relevant[:20]}")

    vae_model, _ = official["load_ae"](str(args.model_root / "ae.safetensors"))
    model.to(device=device).eval()
    vae_model.to(device=device, dtype=torch_dtype).eval()
    vae_transform = _ImageTransform(
        args.vae_max_image_size,
        args.vae_min_image_size,
        args.vae_image_stride,
        max_pixels=args.vae_max_pixels,
    )
    vit_transform = _ImageTransform(
        args.vit_max_image_size,
        args.vit_min_image_size,
        args.vit_image_stride,
        max_pixels=args.vit_max_pixels,
    )
    return model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform, device, torch_dtype


@torch.no_grad()
def _forward_vae_context(
    model: Any,
    vae_model: Any,
    *,
    past_key_values: Any,
    vae_input: dict[str, Any],
) -> dict[str, Any]:
    packed_text_ids = vae_input["packed_text_ids"]
    packed_text_indexes = vae_input["packed_text_indexes"]
    packed_vae_token_indexes = vae_input["packed_vae_token_indexes"]
    packed_seqlens = vae_input["packed_seqlens"]

    packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence = packed_text_embedding.new_zeros((int(packed_seqlens.sum().item()), model.hidden_size))
    packed_sequence[packed_text_indexes] = packed_text_embedding

    padded_latent = vae_model.encode(vae_input["padded_images"])
    packed_latent = []
    p = model.latent_patch_size
    for latent, (h, w) in zip(padded_latent, vae_input["patchified_vae_latent_shapes"]):
        latent = latent[:, : h * p, : w * p].reshape(model.latent_channel, h, p, w, p)
        latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * model.latent_channel)
        packed_latent.append(latent)
    packed_latent = torch.cat(packed_latent, dim=0)
    packed_timestep_embeds = model.time_embedder(vae_input["packed_timesteps"])
    packed_pos_embed = model.latent_pos_embed(vae_input["packed_vae_position_ids"])
    latent_embeds = model.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
    if latent_embeds.dtype != packed_sequence.dtype:
        latent_embeds = latent_embeds.to(packed_sequence.dtype)
    packed_sequence[packed_vae_token_indexes] = latent_embeds

    output = model.language_model.forward_inference(
        packed_query_sequence=packed_sequence,
        query_lens=packed_seqlens,
        packed_query_position_ids=vae_input["packed_position_ids"],
        packed_query_indexes=vae_input["packed_indexes"],
        past_key_values=past_key_values,
        key_values_lens=vae_input["key_values_lens"],
        packed_key_value_indexes=vae_input["packed_key_value_indexes"],
        update_past_key_values=True,
        is_causal=False,
        mode="gen",
        packed_vae_token_indexes=packed_vae_token_indexes,
        packed_text_indexes=packed_text_indexes,
    )
    return {
        "past_key_values": output.past_key_values,
        "packed_latents": packed_latent.detach(),
        "latent_embeds": latent_embeds.detach(),
        "packed_sequence": packed_sequence.detach(),
    }


@torch.no_grad()
def _forward_flow_base(
    model: Any,
    *,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    latent_input: dict[str, torch.Tensor],
    past_key_values: Any,
) -> dict[str, torch.Tensor]:
    packed_text_embedding = model.language_model.model.embed_tokens(latent_input["packed_text_ids"])
    packed_sequence = packed_text_embedding.new_zeros(
        (int(latent_input["packed_seqlens"].sum().item()), model.hidden_size)
    )
    packed_sequence[latent_input["packed_text_indexes"]] = packed_text_embedding
    packed_timestep = torch.full((x_t.shape[0],), float(timestep.item()), device=x_t.device, dtype=x_t.dtype)
    latent_embeds = (
        model.vae2llm(x_t)
        + model.time_embedder(packed_timestep)
        + model.latent_pos_embed(latent_input["packed_vae_position_ids"])
    )
    if latent_embeds.dtype != packed_sequence.dtype:
        latent_embeds = latent_embeds.to(packed_sequence.dtype)
    packed_sequence[latent_input["packed_vae_token_indexes"]] = latent_embeds
    output = model.language_model.forward_inference(
        packed_query_sequence=packed_sequence,
        query_lens=latent_input["packed_seqlens"],
        packed_query_position_ids=latent_input["packed_position_ids"],
        packed_query_indexes=latent_input["packed_indexes"],
        past_key_values=past_key_values,
        key_values_lens=latent_input["key_values_lens"],
        packed_key_value_indexes=latent_input["packed_key_value_indexes"],
        update_past_key_values=False,
        is_causal=False,
        mode="gen",
        packed_vae_token_indexes=latent_input["packed_vae_token_indexes"],
        packed_text_indexes=latent_input["packed_text_indexes"],
    )
    velocity = model.llm2vae(output.packed_query_sequence)[latent_input["packed_vae_token_indexes"]]
    return {
        "packed_sequence": packed_sequence.detach(),
        "latent_embeds": latent_embeds.detach(),
        "hidden_state": output.packed_query_sequence.detach(),
        "velocity": velocity.detach(),
    }


def _first_flow_timestep(num_timesteps: int, timestep_shift: float, device: torch.device) -> dict[str, torch.Tensor]:
    timesteps_full = torch.linspace(1, 0, num_timesteps, device=device)
    timesteps_shifted = timestep_shift * timesteps_full / (1 + (timestep_shift - 1) * timesteps_full)
    dts = timesteps_shifted[:-1] - timesteps_shifted[1:]
    return {"timesteps": timesteps_shifted[:-1], "dts": dts, "timestep": timesteps_shifted[:1], "dt": dts[:1]}


@torch.no_grad()
def capture(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform, device, torch_dtype = _build_model(args)
    official = _import_official(args.official_repo)
    raw_image = _make_image(args.input_width, args.input_height)
    context_image = vae_transform.resize_transform(official["pil_img2rgb"](raw_image))
    image_shape = context_image.size[::-1]

    past_key_values = official["NaiveCache"](model.config.llm_config.num_hidden_layers)
    curr_kvlens = [0]
    curr_rope = [0]

    vae_input, curr_kvlens, curr_rope = model.prepare_vae_images(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        images=[context_image],
        transforms=vae_transform,
        new_token_ids=new_token_ids,
    )
    kv_lens_after_vae = list(curr_kvlens)
    ropes_after_vae = list(curr_rope)
    vae_input = _move_tensors(vae_input, device)
    rng_state_before_vae = _rng_state()
    with _autocast_context(device, torch_dtype):
        vae_context = _forward_vae_context(model, vae_model, past_key_values=past_key_values, vae_input=vae_input)
    past_key_values = vae_context["past_key_values"]
    cache_after_vae = _cache_to_tensors(past_key_values)

    vit_input, curr_kvlens, curr_rope = model.prepare_vit_images(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        images=[context_image],
        transforms=vit_transform,
        new_token_ids=new_token_ids,
    )
    kv_lens_after_vit = list(curr_kvlens)
    ropes_after_vit = list(curr_rope)
    vit_input = _move_tensors(vit_input, device)
    with _autocast_context(device, torch_dtype):
        past_key_values = model.forward_cache_update_vit(past_key_values, **vit_input)
    cache_after_vit = _cache_to_tensors(past_key_values)

    prompt_input, kv_lens_after_prompt, ropes_after_prompt = model.prepare_prompts(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        prompts=[args.prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_input = _move_tensors(prompt_input, device)
    with _autocast_context(device, torch_dtype):
        past_key_values = model.forward_cache_update_text(past_key_values, **prompt_input)
    cache_after_prompt = _cache_to_tensors(past_key_values)

    latent_input = model.prepare_vae_latent(
        curr_kvlens=kv_lens_after_prompt,
        curr_rope=ropes_after_prompt,
        image_sizes=[image_shape],
        new_token_ids=new_token_ids,
    )
    latent_input = _move_tensors(latent_input, device)
    timestep_fields = _first_flow_timestep(args.num_timesteps, args.timestep_shift, device)
    x_t0 = latent_input["packed_init_noises"]
    with _autocast_context(device, torch_dtype):
        flow_output = _forward_flow_base(
            model,
            x_t=x_t0,
            timestep=timestep_fields["timestep"],
            latent_input=latent_input,
            past_key_values=past_key_values,
        )
    x_t1 = x_t0 - flow_output["velocity"].to(x_t0.device) * timestep_fields["dt"][0]

    return {
        "metadata": {
            "case_id": "image_edit_vae_context_one_step",
            "dtype": args.dtype,
            "seed": args.seed,
            "official_repo": str(args.official_repo),
            "model_root": str(args.model_root),
        },
        "raw_input": {
            "prompt": args.prompt,
            "input_image_size": [args.input_width, args.input_height],
            "context_image_size": list(context_image.size),
            "image_size": [int(image_shape[0]), int(image_shape[1])],
            "num_timesteps": args.num_timesteps,
            "timestep_shift": args.timestep_shift,
        },
        "rng_state": _rng_state(),
        "rng_state_before_vae": rng_state_before_vae,
        "prepared": {
            "vae_context": _cpu_tensors(vae_input)
            | {
                "kv_lens_after": torch.tensor(kv_lens_after_vae, dtype=torch.int32),
                "ropes_after": torch.tensor(ropes_after_vae, dtype=torch.long),
            },
            "vit": _cpu_tensors(vit_input)
            | {
                "kv_lens_after": torch.tensor(kv_lens_after_vit, dtype=torch.int32),
                "ropes_after": torch.tensor(ropes_after_vit, dtype=torch.long),
            },
            "prompt": _cpu_tensors(prompt_input),
            "kv_lens_after_prompt": torch.tensor(kv_lens_after_prompt, dtype=torch.int32),
            "ropes_after_prompt": torch.tensor(ropes_after_prompt, dtype=torch.long),
            "latent": _cpu_tensors(latent_input),
            "timesteps": _cpu_tensors(timestep_fields),
        },
        "cache_after_vae": cache_after_vae,
        "cache_after_vit": cache_after_vit,
        "cache_after_prompt": cache_after_prompt,
        "vae_context": {
            "packed_latents": vae_context["packed_latents"].detach().cpu(),
            "latent_embeds": vae_context["latent_embeds"].detach().cpu(),
            "packed_sequence": vae_context["packed_sequence"].detach().cpu(),
        },
        "one_step": {
            "x_t0": x_t0.detach().cpu(),
            "latent_embeds": flow_output["latent_embeds"].detach().cpu(),
            "packed_sequence": flow_output["packed_sequence"].detach().cpu(),
            "hidden_state": flow_output["hidden_state"].detach().cpu(),
            "velocity": flow_output["velocity"].detach().cpu(),
            "x_t1": x_t1.detach().cpu(),
        },
        "tolerances": {
            "bf16": {"max_abs_diff": 1e-2, "mean_abs_diff": 1e-4, "cosine_similarity_min": 0.9999},
            "fp16": {"max_abs_diff": 1e-2, "mean_abs_diff": 1e-4, "cosine_similarity_min": 0.9999},
            "fp32": {"max_abs_diff": 1e-5, "mean_abs_diff": 1e-6, "cosine_similarity_min": 0.999999},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--input-width", type=int, default=384)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20250611)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--latent-patch-size", type=int, default=2)
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--vit-max-num-patch-per-side", type=int, default=70)
    parser.add_argument("--num-timesteps", type=int, default=4)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--vae-max-image-size", type=int, default=1024)
    parser.add_argument("--vae-min-image-size", type=int, default=512)
    parser.add_argument("--vae-image-stride", type=int, default=16)
    parser.add_argument("--vae-max-pixels", type=int, default=14 * 14 * 9 * 1024)
    parser.add_argument("--vit-max-image-size", type=int, default=980)
    parser.add_argument("--vit-min-image-size", type=int, default=378)
    parser.add_argument("--vit-image-stride", type=int, default=14)
    parser.add_argument("--vit-max-pixels", type=int, default=14 * 14 * 9 * 1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = capture(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.output)
    print(json.dumps({"output": str(args.output), "case_id": fixture["metadata"]["case_id"]}, indent=2))


if __name__ == "__main__":
    main()
