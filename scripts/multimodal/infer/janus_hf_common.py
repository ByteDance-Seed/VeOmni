"""Shared helpers for HuggingFace Janus reference inference scripts.

These scripts mirror the official DeepSeek Janus repo
(``inference.py`` / ``generation_inference.py``) but call
``transformers.models.janus.modeling_janus.JanusForConditionalGeneration.generate``
(``generation_mode="text"`` or ``"image"``) — the baseline for SeedOmni V2
OmniModel alignment.

Usage
-----
Both sibling scripts accept ``--model_path`` pointing at either:

* a HuggingFace hub id (``deepseek-ai/Janus-1.3B``), or
* a local directory from ``scripts/download_hf_model.py`` / ``split_janus.py`` input.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import JanusForConditionalGeneration, JanusProcessor


# Allow ``from tests.tools.hf_paths import hf_local_or_remote`` when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.tools.hf_paths import hf_local_or_remote  # noqa: E402


DEFAULT_MODEL_ID = "deepseek-ai/Janus-1.3B"
DEFAULT_DTYPE = "bfloat16"
# Same COCO sample as HF Janus docs (used when no --image_path / --image_url).
DEFAULT_IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def resolve_model_path(model_path: str | None) -> str:
    """Map hub id → local NFS mirror when available."""
    path = model_path or DEFAULT_MODEL_ID
    return hf_local_or_remote(path)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(mapping)}")
    return mapping[name]


def load_janus(
    model_path: str, *, dtype: str = DEFAULT_DTYPE
) -> tuple[JanusProcessor, JanusForConditionalGeneration, torch.device]:
    """Load ``JanusProcessor`` + ``JanusForConditionalGeneration``.

    Uses ``device_map="auto"`` on CUDA so accelerate can pick / shard.
    Multi-device sharding is safe for image generation only when
    ``janus_image_generate_fix`` is active — see its docstring for why.
    """
    resolved = resolve_model_path(model_path)
    device = pick_device()
    torch_dtype = pick_dtype(dtype, device)

    processor = JanusProcessor.from_pretrained(resolved)
    model = JanusForConditionalGeneration.from_pretrained(
        resolved,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return processor, model, device


def move_inputs_to_device(inputs, device: torch.device, dtype: torch.dtype | None = None):
    """Move a processor ``BatchFeature`` / dict of tensors to ``device``."""
    out = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if dtype is not None and value.is_floating_point():
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_rgb_image(source: str):
    """Load a local path or HTTP(S) URL into an RGB PIL image (no httpx — avoids broken proxy env)."""
    from io import BytesIO
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    from PIL import Image

    if source.startswith(("http://", "https://")):
        req = Request(source, headers={"User-Agent": "veomni-janus-infer/1.0"})
        try:
            with urlopen(req, timeout=60) as resp:
                data = resp.read()
        except URLError as exc:
            raise RuntimeError(f"Failed to download image URL {source!r}: {exc}") from exc
        return Image.open(BytesIO(data)).convert("RGB")

    path = Path(source).expanduser()
    if path.is_file():
        return Image.open(path).convert("RGB")
    raise FileNotFoundError(f"Image not found: {path}")


def default_demo_image_path(model_path: str | None = None) -> Path | None:
    """Pick a local demo image so und inference works offline (no URL fetch)."""
    candidates: list[Path] = []
    if model_path:
        root = Path(model_path).expanduser()
        candidates.extend(
            [
                root / "teaser.png",
                root / "arch.jpg",
                root.parent / "Janus-1.3B" / "teaser.png",
            ]
        )
    candidates.extend(
        [
            Path("/mnt/hdfs/veomni/models/Janus-1.3B/teaser.png"),
            Path("/mnt/hdfs/user_dir/veomni_omni/models/transformers/Janus-1.3B/teaser.png"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def prepare_und_inputs(
    processor: JanusProcessor,
    *,
    prompt: str,
    image_path: str | None = None,
    image_url: str | None = None,
    model_path: str | None = None,
):
    """Build processor inputs for image understanding (``generation_mode=\"text\"``).

    Always preload the image to PIL and pass ``{\"type\": \"image\", \"image\": ...}``
    into ``apply_chat_template`` — never raw ``url``/path strings.
    """
    if image_path is not None:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        source, image_label = str(path), str(path)
    elif image_url is not None:
        source, image_label = image_url, image_url
    else:
        demo = default_demo_image_path(model_path)
        if demo is None:
            raise FileNotFoundError(
                "No image provided. Pass --image_path (recommended) or --image_url. "
                "Default URL fetch is disabled when no local demo image is found "
                "(tried teaser.png next to model_path)."
            )
        source, image_label = str(demo), str(demo)

    image_block = {"type": "image", "image": load_rgb_image(source)}

    messages = [
        {
            "role": "user",
            "content": [
                image_block,
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        generation_mode="text",
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs, image_label


def decode_new_text(processor: JanusProcessor, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    """Decode only tokens generated after the prompt (HF ``generate`` returns full sequence)."""
    prompt_len = int(input_ids.shape[-1])
    new_ids = output_ids[0, prompt_len:] if output_ids.ndim == 2 else output_ids[prompt_len:]
    return processor.decode(new_ids, skip_special_tokens=True)


def prepare_gen_inputs(
    processor: JanusProcessor,
    model: JanusForConditionalGeneration,
    *,
    prompt: str,
    num_images: int,
    guidance_scale: float,
) -> tuple[dict, dict]:
    """Tokenize a T2I prompt and build generate() kwargs for image mode.

    Returns ``(processor_inputs, generate_kwargs)``. ``generate_kwargs`` is meant
    to be unpacked into ``model.generate(**processor_inputs, **generate_kwargs)``.

    We pass ``num_return_sequences`` / ``guidance_scale`` / ``max_length`` as
    kwargs rather than mutating ``model.generation_config`` directly: starting
    in transformers 5.9.0 the merged generation config is ``validate()``-d
    eagerly, so a pre-set ``num_return_sequences=2`` clashes with the loaded
    ``do_sample=False`` default before our ``do_sample=True`` kwarg can land.
    Passing everything as one kwarg payload lets GenerationMixin merge them
    atomically and pass validation.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, generation_mode="image", return_tensors="pt")

    num_image_tokens = model.model.vision_model.config.num_image_tokens
    seq_len = int(inputs["input_ids"].shape[-1])
    # Janus image branch sizes the static KV cache as
    # ``max(generation_config.max_length, num_image_tokens + seq_len)`` —
    # ``max_length`` from converted checkpoints can be ``None``, so set it here.
    generate_kwargs = {
        "num_return_sequences": num_images,
        "guidance_scale": guidance_scale,
        "max_length": num_image_tokens + seq_len,
    }
    return inputs, generate_kwargs


@contextmanager
def janus_image_generate_fix():
    """Monkey-patches needed to make Janus image-gen work across transformers versions.

    Both patches are surgical, applied via a single hook on
    ``JanusForConditionalGeneration.prepare_inputs_for_generation`` and scoped
    to this context manager.

    Patch 1 — inject ``is_first_iteration=True``
    (transformers <=5.2.0; PR #45044):
        Upstream ``JanusForConditionalGeneration.generate`` image branch forgot
        to pass ``is_first_iteration=True``. The base class then sliced
        ``inputs_embeds`` down to one token at step 0, the full prompt never
        got embedded, ``position_ids`` / KV-cache content desynchronized from
        ``attention_mask``, and the model emitted garbage VQ tokens that
        decoded to blurry color noise. ``setdefault`` keeps this a no-op once
        transformers includes #45044 (the fixed code already passes ``True``).

    Patch 2 — inject ``position_ids`` on ``inputs_embeds.device``
    (transformers 5.9.x; needed for ``device_map="auto"`` sharding):
        ``StaticLayer.get_seq_length`` is declared ``-> int`` but actually
        returns ``self.cumulative_length`` — a 1-elem tensor pinned to the
        first decoder layer's device (set during ``lazy_initialization`` as
        ``key_states.device``). With ``device_map="auto"`` accelerate may put
        ``embed_tokens`` and decoder layer 0 on different GPUs.
        ``LlamaModel.forward`` then does
        ``torch.arange(..., device=inputs_embeds.device) + past_seen_tokens``
        with two operands on different devices → ``RuntimeError: Expected
        all tensors to be on the same device``.

        ``masking_utils.create_causal_mask`` already handles this exact case
        with ``q_offset.to(inputs_embeds.device)`` (``masking_utils.py:858``).
        ``LlamaModel.forward`` is the missing twin of that fix upstream.
        Rather than patch the base llama forward (would affect every model
        using llama as language backbone) or coerce ``get_seq_length`` to
        host int (one ``.item()`` D2H sync per AR step — 575 syncs per image,
        ~14s overhead), we precompute ``position_ids`` ourselves: pull
        ``past_seen_tokens = past_key_values.get_seq_length()``, async-copy
        it to ``inputs_embeds.device`` (non-blocking D2D, no host sync), and
        replicate Llama's formula ``torch.arange(seq_len, device=...) +
        past_seen_tokens``. Llama then enters its forward with
        ``position_ids != None`` and skips the cross-device branch entirely.

        Note: in 5.9.0 Janus image-gen dropped the ``cache_position`` kwarg
        path (compare to 5.2.0 ``modeling_janus.py:1315``), so we have to
        compute position_ids from the cache directly — we can't just reuse a
        ``cache_position`` returned by ``prepare_inputs_for_generation``.

    Both patches are no-ops on transformers versions that have the upstream
    fix (``setdefault`` and the ``"position_ids" not in out`` guard).
    """
    orig_prep = JanusForConditionalGeneration.prepare_inputs_for_generation

    def patched_prep(self, *args, **kwargs):
        kwargs.setdefault("is_first_iteration", True)
        inputs_embeds = kwargs.get("inputs_embeds")
        past_key_values = kwargs.get("past_key_values")
        out = orig_prep(self, *args, **kwargs)
        if inputs_embeds is not None and past_key_values is not None and "position_ids" not in out:
            past_seen = past_key_values.get_seq_length()
            if isinstance(past_seen, torch.Tensor):
                past_seen = past_seen.to(inputs_embeds.device, non_blocking=True)
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            out["position_ids"] = position_ids.unsqueeze(0)
        return out

    JanusForConditionalGeneration.prepare_inputs_for_generation = patched_prep
    try:
        yield
    finally:
        JanusForConditionalGeneration.prepare_inputs_for_generation = orig_prep


def save_images_from_tokens(
    model: JanusForConditionalGeneration,
    processor: JanusProcessor,
    image_token_ids: torch.Tensor,
    output_dir: Path,
    *,
    prefix: str = "janus_gen",
) -> list[Path]:
    """VQ-decode image token grid and write PNG files.

    Matches the official DeepSeek ``generation_inference.py`` post-process
    (``(x + 1) / 2 * 255`` clip).  We avoid ``processor.postprocess(...,
    return_tensors=\"PIL.Image.Image\")`` because JanusImageProcessorFast only
    supports ``return_tensors=\"pt\"`` on the fast path.
    """
    import numpy as np
    from PIL import Image

    del processor  # kept for API symmetry with HF docs
    with torch.inference_mode():
        decoded = model.decode_image_tokens(image_token_ids)  # (B, H, W, C)
    arr = decoded.detach().float().clamp(-1.0, 1.0).cpu().numpy()
    arr = np.clip((arr + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)

    paths: list[Path] = []
    for i in range(arr.shape[0]):
        path = output_dir / f"{prefix}_{i:02d}.png"
        Image.fromarray(arr[i]).save(path)
        paths.append(path)
    return paths
