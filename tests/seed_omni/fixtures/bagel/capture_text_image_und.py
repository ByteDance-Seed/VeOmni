"""Capture official BAGEL text+image understanding one-step inference fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights


pytestmark = pytest.mark.skip(reason="BAGEL official capture helper; run explicitly to generate parity fixtures.")

DEFAULT_OUTPUT = Path("outputs/bagel_v2/parity/text_image_understanding_one_step_logits.pt")
DEFAULT_PROMPT = "Describe the image in one short sentence."


def _log(message: str) -> None:
    print(f"[bagel-text-image-fixture] {message}", flush=True)


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
        max_pixels: int = 14 * 14 * 9 * 1024,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
    ):
        self.resize_transform = _MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
            mean=[0.5, 0.5, 0.5] if image_mean is None else image_mean,
            std=[0.5, 0.5, 0.5] if image_std is None else image_std,
            inplace=True,
        )

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
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.bagel.qwen2_navit import NaiveCache
    from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    return {
        "add_special_tokens": add_special_tokens,
        "Bagel": Bagel,
        "BagelConfig": BagelConfig,
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
    return {
        "num_layers": cache.num_layers,
        "key": keys,
        "value": values,
    }


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


def _load_state(path: Path, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    prefixes = ("language_model.", "vit_model.", "connector.", "vit_pos_embed.")
    _log(f"loading text+vision weights from {path} to {device} as {dtype}")
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefixes):
                state_dict[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    _log(f"loaded {len(state_dict)} text+vision tensors")
    return state_dict


def _build_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, int], Any, torch.device]:
    official = _import_official(args.official_repo)
    device = torch.device(args.device)
    torch_dtype = _resolve_dtype(args.dtype)

    _log("reading official configs")
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

    _log(f"loading tokenizer from {args.model_root}")
    tokenizer = official["Qwen2Tokenizer"].from_pretrained(str(args.model_root), local_files_only=True)
    tokenizer, new_token_ids, num_new_tokens = official["add_special_tokens"](tokenizer)
    if num_new_tokens > 0:
        llm_config.vocab_size = len(tokenizer)

    config = official["BagelConfig"](
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=None,
        vit_max_num_patch_per_side=args.vit_max_num_patch_per_side,
        connector_act="gelu_pytorch_tanh",
    )
    _log("constructing official text+vision BAGEL model on meta parameters")
    with no_init_weights(), init_empty_weights():
        language_model = official["Qwen2ForCausalLM"](llm_config)
        vit_model = official["SiglipVisionModel"](vit_config)
        model = official["Bagel"](language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    state = _load_state(args.model_root / "ema.safetensors", device=device, dtype=torch_dtype)
    _log("assigning text+vision weights")
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    unexpected_relevant = [
        key for key in unexpected if key.startswith(("language_model.", "vit_model.", "connector.", "vit_pos_embed."))
    ]
    if unexpected_relevant:
        raise RuntimeError(f"Unexpected text+vision keys while loading official BAGEL: {unexpected_relevant[:20]}")
    missing_relevant = [
        key for key in missing if key.startswith(("language_model.", "vit_model.", "connector.", "vit_pos_embed."))
    ]
    if missing_relevant:
        raise RuntimeError(f"Missing text+vision keys while loading official BAGEL: {missing_relevant[:20]}")

    _log("moving buffers to target device")
    model.to(device=device)
    model.eval()
    transform = _ImageTransform(
        args.max_image_size,
        args.min_image_size,
        args.image_stride,
        max_pixels=args.max_pixels,
    )
    _log("model ready")
    return model, tokenizer, new_token_ids, transform, device


@torch.no_grad()
def _official_image_embeds(model: Any, image_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | int]:
    vit_token_seqlens = image_input["vit_token_seqlens"]
    cu_seqlens = F.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
    max_seqlen = int(torch.max(vit_token_seqlens).item())
    packed_vit_token_embed = model.vit_model(
        packed_pixel_values=image_input["packed_vit_tokens"],
        packed_flattened_position_ids=image_input["packed_vit_position_ids"],
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    packed_vit_token_embed = model.connector(packed_vit_token_embed)
    packed_vit_token_embed = packed_vit_token_embed + model.vit_pos_embed(image_input["packed_vit_position_ids"])
    return {
        "image_embeds": packed_vit_token_embed.detach().cpu(),
        "cu_seqlens": cu_seqlens.detach().cpu(),
        "max_seqlen": max_seqlen,
    }


def _packed_key_value_indexes_for_step(start_input: dict[str, torch.Tensor]) -> torch.Tensor:
    key_values_lens = start_input["key_values_lens"]
    packed_key_value_indexes = start_input["packed_key_value_indexes"]
    unpacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
    for idx in range(len(unpacked)):
        unpacked[idx] += idx
    return torch.cat(unpacked, dim=0)


@torch.no_grad()
def capture(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, new_token_ids, transform, device = _build_model(args)
    official = _import_official(args.official_repo)

    image = official["pil_img2rgb"](_make_image(args.image_width, args.image_height))
    past_key_values = official["NaiveCache"](model.config.llm_config.num_hidden_layers)
    curr_kvlens = [0]
    curr_rope = [0]

    image_input, kv_lens_after_image, ropes_after_image = model.prepare_vit_images(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        images=[image],
        transforms=transform,
        new_token_ids=new_token_ids,
    )
    image_input = _move_tensors(image_input, device)
    image_input["packed_vit_tokens"] = image_input["packed_vit_tokens"].to(dtype=_resolve_dtype(args.dtype))
    official_image = _official_image_embeds(model, image_input)
    past_key_values = model.forward_cache_update_vit(past_key_values, **image_input)
    cache_after_image = _cache_to_tensors(past_key_values)

    prompt_input, kv_lens_after_prompt, ropes_after_prompt = model.prepare_prompts(
        curr_kvlens=kv_lens_after_image,
        curr_rope=ropes_after_image,
        prompts=[args.prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_input = _move_tensors(prompt_input, device)
    past_key_values = model.forward_cache_update_text(past_key_values, **prompt_input)
    cache_after_prompt = _cache_to_tensors(past_key_values)

    start_input = model.prepare_start_tokens(kv_lens_after_prompt, ropes_after_prompt, new_token_ids)
    start_input = _move_tensors(start_input, device)
    curr_tokens = start_input["packed_start_tokens"]
    key_values_lens = start_input["key_values_lens"]
    packed_query_position_ids = start_input["packed_query_position_ids"]
    packed_text_embedding = model.language_model.model.embed_tokens(curr_tokens)
    query_lens = torch.ones_like(curr_tokens)
    packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
        0,
        len(key_values_lens),
        device=key_values_lens.device,
        dtype=key_values_lens.dtype,
    )
    packed_key_value_indexes_for_step = _packed_key_value_indexes_for_step(start_input)

    extra_inputs = {"mode": "und"} if model.use_moe else {}
    output = model.language_model.forward_inference(
        packed_query_sequence=packed_text_embedding,
        query_lens=query_lens,
        packed_query_position_ids=packed_query_position_ids,
        packed_query_indexes=packed_query_indexes,
        past_key_values=past_key_values,
        key_values_lens=key_values_lens,
        packed_key_value_indexes=packed_key_value_indexes_for_step,
        update_past_key_values=True,
        is_causal=True,
        **extra_inputs,
    )
    logits = model.language_model.lm_head(output.packed_query_sequence)
    greedy_token = torch.argmax(logits, dim=-1)

    return {
        "metadata": {
            "schema_version": 1,
            "case_id": "text_image_understanding_one_step_logits",
            "boundary": "official.prepare_vit_images.forward_cache_update_vit.prepare_prompts.forward_cache_update_text.one_step_logits",
            "dtype": args.dtype,
            "seed": args.seed,
            "official_repo": str(args.official_repo),
            "official_checkpoint": str(args.model_root),
            "device": str(device),
        },
        "raw_input": {
            "prompt": args.prompt,
            "image_size": [args.image_width, args.image_height],
            "do_sample": False,
            "temperature": 1.0,
            "max_new_tokens": 1,
        },
        "rng_state": _rng_state(),
        "tokenizer": {
            "new_token_ids": dict(new_token_ids),
            "encoded_prompt_ids": tokenizer.encode(args.prompt),
        },
        "prepared": {
            "image": _cpu_tensors(image_input),
            "image_embeds": official_image,
            "kv_lens_after_image": list(kv_lens_after_image),
            "ropes_after_image": list(ropes_after_image),
            "prompt": _cpu_tensors(prompt_input),
            "kv_lens_after_prompt": list(kv_lens_after_prompt),
            "ropes_after_prompt": list(ropes_after_prompt),
            "start": _cpu_tensors(start_input),
            "packed_query_indexes": packed_query_indexes.detach().cpu(),
            "packed_key_value_indexes_for_step": packed_key_value_indexes_for_step.detach().cpu(),
            "query_lens": query_lens.detach().cpu(),
        },
        "cache_after_image": cache_after_image,
        "cache_after_prompt": cache_after_prompt,
        "one_step": {
            "hidden_state": output.packed_query_sequence.detach().cpu(),
            "logits": logits.detach().cpu(),
            "greedy_token": greedy_token.detach().cpu(),
            "cache_after_step": _cache_to_tensors(output.past_key_values),
        },
        "tolerances": {
            "bf16": {
                "v2_parity": {
                    "max_abs_diff": 1.0e-2,
                    "mean_abs_diff": 1.0e-4,
                    "cosine_similarity_min": 0.9999,
                    "source": "agreed V2-vs-official bf16 parity gate",
                },
            }
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--image-height", type=int, default=336)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-image-size", type=int, default=980)
    parser.add_argument("--min-image-size", type=int, default=378)
    parser.add_argument("--image-stride", type=int, default=14)
    parser.add_argument("--max-pixels", type=int, default=14 * 14 * 9 * 1024)
    parser.add_argument("--vit-max-num-patch-per-side", type=int, default=70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_root.exists():
        raise FileNotFoundError(f"BAGEL checkpoint root does not exist: {args.model_root}")
    if not args.official_repo.exists():
        raise FileNotFoundError(f"Official BAGEL repo does not exist: {args.official_repo}")

    fixture = capture(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.output)
    print(json.dumps({"output": str(args.output), "case_id": fixture["metadata"]["case_id"]}, indent=2))


if __name__ == "__main__":
    main()
