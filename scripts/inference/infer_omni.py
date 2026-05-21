"""Multi-modal native-PyTorch inference CLI for VeOmni-trained DiT models.

Auto-detects the diffusers pipeline class from ``model_index.json`` and
dispatches the right modality (T2I / T2V / I2V) through a tiny registry.
Optionally swaps in VeOmni-trained weights — either by replacing the entire
transformer state dict or by attaching a LoRA adapter — without touching the
VAE / text encoder / scheduler of the base pipeline.

This is intentionally a thin wrapper around ``DiffusionPipeline.from_pretrained``
plus the standard diffusers call. No bespoke sampling loop, no scheduler
overrides; the script's value is uniform CLI access (timestamped filenames,
strict-determinism toggle, per-prompt reseeding, batch over prompts) across
the diffusion models that VeOmni currently trains.

Supported pipelines
-------------------

| Modality | Pipeline class                | Example base model          |
|----------|-------------------------------|-----------------------------|
| T2I      | ``QwenImagePipeline``         | ``Qwen/Qwen-Image``         |
| T2V      | ``WanPipeline``               | ``Wan-AI/Wan2.1-T2V-1.3B``  |
| I2V      | ``WanImageToVideoPipeline``   | ``Wan-AI/Wan2.1-I2V-14B``   |

Adding another pipeline is a one-line entry in ``_MODALITIES`` keyed by a
substring of ``type(pipeline).__name__``.

Examples
--------

Baseline T2I (pretrained Qwen-Image):

    python scripts/inference/infer_omni.py \\
        --model_path Qwen/Qwen-Image \\
        --output_dir ./inference_outputs/baseline \\
        --prompts "a corgi wearing a tiny astronaut helmet, studio lighting" \\
        --num_inference_steps 50 --height 1024 --width 1024 \\
        --enable_cpu_offload

T2I with a VeOmni-fine-tuned transformer swapped in:

    python scripts/inference/infer_omni.py \\
        --model_path Qwen/Qwen-Image \\
        --transformer_path ./qwen-image-sft/checkpoints/global_step_200/hf_ckpt \\
        --output_dir ./inference_outputs/ft_step200 \\
        --prompts "a corgi wearing a tiny astronaut helmet, studio lighting" \\
        --enable_cpu_offload

T2V (Wan2.1 T2V) with a VeOmni-trained LoRA adapter:

    python scripts/inference/infer_omni.py \\
        --model_path ./Wan2.1-T2V-1.3B-Diffusers \\
        --lora_path ./exp/Wan_lora/checkpoints/global_step_200 \\
        --lora_weight 1.0 \\
        --output_dir ./inference_outputs/wan_lora \\
        --prompts "Tom the cat is sprawled out on a vibrant red pillow" \\
        --height 480 --width 832 --num_frames 81 --fps 15 \\
        --enable_cpu_offload

I2V (Wan2.1 I2V) with the same fine-tuned LoRA and a conditioning frame:

    python scripts/inference/infer_omni.py \\
        --model_path ./Wan2.1-I2V-14B-480P \\
        --lora_path ./exp/Wan_i2v_lora/checkpoints/global_step_500 \\
        --input_image ./first_frame.png \\
        --output_dir ./inference_outputs/wan_i2v_lora \\
        --prompts "the cat slowly stretches and yawns" \\
        --height 480 --width 832 --num_frames 81 --fps 15 \\
        --enable_cpu_offload
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file


_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ============================ small helpers ============================


def _apply_strict_determinism() -> None:
    """Lock PyTorch to deterministic kernels for paper-grade reproducibility.

    Trades a small amount of throughput for cross-run / cross-PyTorch-version
    stability. ``warn_only=True`` keeps the run alive when an op lacks a
    deterministic implementation (e.g. Wan's 3D VAE interpolation); otherwise
    ``torch.use_deterministic_algorithms`` would raise instead of fall back.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def slugify(text: str, max_len: int = 60) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in text).strip("_")
    safe = safe[:max_len].strip("_")
    return safe or "prompt"


def _save_image(image: Any, path: Path, fps: int) -> None:  # noqa: ARG001 - fps kept for uniform signature
    image.save(path)


def _save_video(frames: Any, path: Path, fps: int) -> None:
    # Lazy import: diffusers' export_to_video pulls in imageio-ffmpeg and the
    # T2I-only user doesn't need that on PYTHONPATH.
    from diffusers.utils import export_to_video

    export_to_video(frames, str(path), fps=fps)


# ============================ modality registry ============================


@dataclass(frozen=True)
class ModalityConfig:
    """Per-pipeline-class dispatch parameters.

    Adding a new diffusion model means adding one entry to ``_MODALITIES``;
    the rest of the script reads from this dataclass without further special-
    casing. Pipeline matching is by class-name substring so the script does
    not need to import ``WanPipeline`` etc. at the top — it works even when
    only one of the wrappers is installed.
    """

    name: str  # "t2i" / "t2v" / "i2v"
    pipeline_class_substrings: tuple[str, ...]
    cfg_arg: str  # "true_cfg_scale" | "guidance_scale"
    cfg_default: float
    requires_input_image: bool  # I2V only
    output_attr: str  # "images" | "frames"
    file_ext: str  # ".png" | ".mp4"
    save_fn: Callable[[Any, Path, int], None]
    default_height: int
    default_width: int
    default_num_frames: int | None  # None for T2I
    default_fps: int | None  # None for T2I
    # T2I pipelines accept num_images_per_prompt; Wan video pipelines do not
    # (they emit exactly one video per prompt). Gate the kwarg on this flag so
    # we don't hand WanPipeline a kwarg it raises TypeError on.
    supports_batch_per_prompt: bool


# I2V is matched before T2V because ``WanImageToVideoPipeline`` would otherwise
# also match ``WanPipeline`` (substring). The dict is ordered (Py3.7+) and
# ``detect_modality`` iterates insertion order.
_MODALITIES: dict[str, ModalityConfig] = {
    "i2v": ModalityConfig(
        name="i2v",
        pipeline_class_substrings=("ImageToVideoPipeline",),
        cfg_arg="guidance_scale",
        cfg_default=5.0,
        requires_input_image=True,
        output_attr="frames",
        file_ext=".mp4",
        save_fn=_save_video,
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        supports_batch_per_prompt=False,
    ),
    "t2v": ModalityConfig(
        name="t2v",
        pipeline_class_substrings=("WanPipeline",),
        cfg_arg="guidance_scale",
        cfg_default=5.0,
        requires_input_image=False,
        output_attr="frames",
        file_ext=".mp4",
        save_fn=_save_video,
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        supports_batch_per_prompt=False,
    ),
    "t2i": ModalityConfig(
        name="t2i",
        pipeline_class_substrings=("QwenImagePipeline",),
        cfg_arg="true_cfg_scale",
        cfg_default=4.0,
        requires_input_image=False,
        output_attr="images",
        file_ext=".png",
        save_fn=_save_image,
        default_height=1024,
        default_width=1024,
        default_num_frames=None,
        default_fps=None,
        supports_batch_per_prompt=True,
    ),
}


def detect_modality(pipeline: Any) -> ModalityConfig:
    cls_name = type(pipeline).__name__
    for cfg in _MODALITIES.values():
        if any(sub in cls_name for sub in cfg.pipeline_class_substrings):
            return cfg
    raise NotImplementedError(
        f"Unsupported pipeline class {cls_name!r}. Add an entry to _MODALITIES keyed by a substring of the class name."
    )


# ============================ weight injection ============================


def _resolve_transformer_shards(transformer_path: Path) -> list[Path]:
    """Return safetensors shard paths under ``transformer_path``.

    Accepts both naming conventions found in the wild:

    * VeOmni ``save_hf_safetensor`` writes ``model.safetensors[.index.json]``.
    * Diffusers ``save_pretrained`` writes
      ``diffusion_pytorch_model.safetensors[.index.json]``.

    Index files are preferred over a single file because VeOmni can emit a
    sharded index even when the model fits in a single shard (e.g. the 20B
    Qwen-Image transformer at bf16 lands as ``model-00001-of-00001.safetensors``
    plus an index pointing only at that one shard).
    """
    if transformer_path.is_file():
        if transformer_path.suffix != ".safetensors":
            raise ValueError(f"--transformer_path file must be .safetensors, got {transformer_path}")
        return [transformer_path]

    if not transformer_path.is_dir():
        raise FileNotFoundError(transformer_path)

    for idx_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        idx_file = transformer_path / idx_name
        if idx_file.is_file():
            mapping = json.loads(idx_file.read_text())["weight_map"]
            shards = sorted({transformer_path / fname for fname in mapping.values()})
            missing = [p for p in shards if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"index references missing shards: {missing}")
            return shards

    for single_name in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
        f = transformer_path / single_name
        if f.is_file():
            return [f]

    raise FileNotFoundError(
        f"No safetensors found under {transformer_path}. Expected model.safetensors[.index.json] "
        f"or diffusion_pytorch_model.safetensors[.index.json]."
    )


def _override_transformer_weights(pipeline: Any, transformer_path: str, dtype: torch.dtype) -> None:
    shard_paths = _resolve_transformer_shards(Path(transformer_path))
    print(f"[info] loading transformer weights from {len(shard_paths)} shard(s) under {transformer_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shard_paths:
        state_dict.update(load_file(str(shard)))
    for k, v in state_dict.items():
        if v.is_floating_point() and v.dtype != dtype:
            state_dict[k] = v.to(dtype)

    missing, unexpected = pipeline.transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    if missing:
        print(f"[warn] {len(missing)} missing key(s); first 5: {missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected key(s); first 5: {unexpected[:5]}", file=sys.stderr)
    if not missing and not unexpected:
        print(f"[info] all transformer weights matched from {transformer_path}")


def _load_lora_adapter(
    pipeline: Any,
    lora_path: str,
    adapter_name: str,
    weight: float,
    prefix: str,
) -> None:
    """Attach a VeOmni-trained LoRA adapter to the pipeline's transformer.

    VeOmni's ``save_lora_adapter_with_dcp`` writes
    ``adapter_config.json + adapter_model.safetensors`` with weight keys
    prefixed by ``base_model.model.``; the default ``--lora_prefix`` matches
    that. Mirrors ``docs/examples/wan2.1_I2V_1.3B.md`` §6.2.
    """
    print(f"[info] loading LoRA adapter from {lora_path} (prefix={prefix!r}, weight={weight})")
    pipeline.transformer.load_lora_adapter(lora_path, prefix=prefix, adapter_name=adapter_name)
    pipeline.set_adapters(adapter_name, adapter_weights=weight)


# ============================ CLI ============================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Native-torch DiT inference (T2I / T2V / I2V; with optional VeOmni transformer or LoRA swap).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- common ---
    p.add_argument(
        "--model_path",
        required=True,
        help="Path to a diffusers pipeline directory (must contain model_index.json).",
    )
    p.add_argument(
        "--prompts",
        nargs="+",
        default=["A futuristic neon-lit city skyline at dusk, ultra-detailed cinematic photo."],
        help="One or more text prompts.",
    )
    p.add_argument(
        "--negative_prompt",
        default=" ",
        help="Negative prompt applied to every prompt (single space disables CFG-side negative).",
    )
    p.add_argument("--output_dir", default="./inference_outputs")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help=(
            "T2I only. Wan T2V / I2V always emit one video per prompt; pass "
            "multiple --prompts entries instead of bumping this for video."
        ),
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height. Defaults: 1024 for T2I; 480 for T2V/I2V.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. Defaults: 1024 for T2I; 832 for T2V/I2V.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base seed. Prompt at index N uses ``seed + N`` so any individual "
            "prompt is reproducible regardless of how many other prompts appear "
            "in the same run."
        ),
    )
    p.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPE.keys()))
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device (ignored when --enable_cpu_offload is set).",
    )
    p.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Use diffusers model_cpu_offload to fit larger pipelines on a single GPU.",
    )
    p.add_argument(
        "--strict_determinism",
        action="store_true",
        help=(
            "Force deterministic kernels (torch.use_deterministic_algorithms + "
            "cudnn.deterministic + CUBLAS_WORKSPACE_CONFIG). Adds a small "
            "latency overhead; required for bit-identical output across "
            "different PyTorch sub-versions."
        ),
    )

    # --- modality-gated (always declared; validated after pipeline detection) ---
    p.add_argument(
        "--true_cfg_scale",
        type=float,
        default=None,
        help="T2I only (e.g. Qwen-Image). Default 4.0 when applicable.",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="T2V / I2V only (e.g. Wan2.1). Default 5.0 when applicable.",
    )
    p.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="T2V / I2V only. Default 81 when applicable.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help="T2V / I2V only. Default 15 when applicable.",
    )
    p.add_argument(
        "--input_image",
        nargs="+",
        default=None,
        help=(
            "I2V only. Path(s) to first-frame conditioning image(s). Either pass "
            "one path (broadcast to every prompt) or one path per prompt."
        ),
    )

    # --- weight injection (mutually exclusive) ---
    p.add_argument(
        "--transformer_path",
        default=None,
        help=(
            "Optional VeOmni-trained full transformer checkpoint dir (config.json "
            "+ model.safetensors[.index.json]) or a single .safetensors file. "
            "Mutually exclusive with --lora_path."
        ),
    )
    p.add_argument(
        "--lora_path",
        default=None,
        help=(
            "Optional VeOmni-trained LoRA adapter directory (adapter_config.json "
            "+ adapter_model.safetensors). Mutually exclusive with "
            "--transformer_path."
        ),
    )
    p.add_argument("--lora_adapter_name", default="veomni_lora")
    p.add_argument("--lora_weight", type=float, default=1.0)
    p.add_argument(
        "--lora_prefix",
        default="base_model.model",
        help="LoRA weight-key prefix (matches VeOmni's save_lora_adapter_with_dcp default).",
    )

    args = p.parse_args(argv)
    if args.transformer_path and args.lora_path:
        p.error("--transformer_path and --lora_path are mutually exclusive.")
    return args


def _resolve_modality_args(args: argparse.Namespace, modality: ModalityConfig) -> None:
    """Fill in modality-default values for unset fields and warn on irrelevant ones.

    Mutates ``args`` in place. Called after ``detect_modality`` so we know
    which CFG / video kwargs are relevant.
    """
    # Fill defaults from the modality config.
    if args.height is None:
        args.height = modality.default_height
    if args.width is None:
        args.width = modality.default_width

    if modality.cfg_arg == "true_cfg_scale":
        if args.true_cfg_scale is None:
            args.true_cfg_scale = modality.cfg_default
        if args.guidance_scale is not None:
            print(
                f"[warn] --guidance_scale is ignored for modality={modality.name} (uses --true_cfg_scale).",
                file=sys.stderr,
            )
    else:  # "guidance_scale"
        if args.guidance_scale is None:
            args.guidance_scale = modality.cfg_default
        if args.true_cfg_scale is not None:
            print(
                f"[warn] --true_cfg_scale is ignored for modality={modality.name} (uses --guidance_scale).",
                file=sys.stderr,
            )

    if modality.output_attr == "frames":
        if args.num_frames is None:
            args.num_frames = modality.default_num_frames
        if args.fps is None:
            args.fps = modality.default_fps
    else:
        if args.num_frames is not None:
            print(
                f"[warn] --num_frames is ignored for modality={modality.name} (not a video pipeline).",
                file=sys.stderr,
            )
        if args.fps is not None:
            print(f"[warn] --fps is ignored for modality={modality.name} (not a video pipeline).", file=sys.stderr)

    if not modality.supports_batch_per_prompt and args.num_images_per_prompt != 1:
        # Video pipelines (Wan T2V / I2V) don't accept num_images_per_prompt;
        # passing it would raise TypeError downstream. Force it back to 1 and
        # tell the user to re-run with multiple --prompts entries instead.
        print(
            f"[warn] --num_images_per_prompt is not supported for modality={modality.name}; "
            f"forcing to 1. Use multiple --prompts entries to generate several videos.",
            file=sys.stderr,
        )
        args.num_images_per_prompt = 1

    if modality.requires_input_image:
        if not args.input_image:
            raise SystemExit(f"[error] modality={modality.name} requires --input_image (first-frame conditioning).")
        if len(args.input_image) not in (1, len(args.prompts)):
            raise SystemExit(
                f"[error] --input_image must be either 1 path (broadcast) or "
                f"{len(args.prompts)} paths (one per prompt); got {len(args.input_image)}."
            )
    else:
        if args.input_image:
            print(
                f"[warn] --input_image is ignored for modality={modality.name} (not an image-to-video pipeline).",
                file=sys.stderr,
            )


def build_call_kwargs(
    args: argparse.Namespace,
    modality: ModalityConfig,
    pil_image: Any | None = None,
) -> dict[str, Any]:
    """Assemble the diffusers ``pipeline(**kwargs)`` payload for one prompt.

    ``pil_image`` is set only for I2V; the per-prompt prompt / negative_prompt
    / generator are filled by the caller (``main``) since they vary per prompt.
    """
    kw: dict[str, Any] = dict(
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
    )
    if modality.supports_batch_per_prompt:
        kw["num_images_per_prompt"] = args.num_images_per_prompt
    if modality.cfg_arg == "true_cfg_scale":
        # Negative prompt is only meaningful when the user keeps CFG > 1.0;
        # mirrors the deleted T2I behavior so single-pass Qwen-Image runs
        # (true_cfg_scale=1.0) skip the negative branch entirely.
        if args.true_cfg_scale is not None and args.true_cfg_scale <= 1.0:
            kw["negative_prompt"] = None
        kw["true_cfg_scale"] = args.true_cfg_scale
    else:
        kw["guidance_scale"] = args.guidance_scale

    if modality.output_attr == "frames":
        kw["num_frames"] = args.num_frames
    if modality.requires_input_image:
        kw["image"] = pil_image
    return kw


def extract_outputs(result: Any, modality: ModalityConfig) -> list[Any]:
    """Normalize diffusers' per-modality result objects to a flat list.

    * T2I: ``result.images`` -> ``list[PIL.Image]`` (one per image).
    * T2V / I2V: ``result.frames`` -> ``list[list[PIL.Image]]`` (one
      frame-sequence per generated video).
    """
    if modality.output_attr == "images":
        return list(result.images)
    return list(result.frames)


# ============================ main ============================


def _resolve_i2v_image_for_prompt(args: argparse.Namespace, idx: int) -> Any | None:
    if not args.input_image:
        return None
    from PIL import Image

    if len(args.input_image) == 1:
        path = args.input_image[0]
    else:
        path = args.input_image[idx]
    return Image.open(path).convert("RGB")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dtype = _DTYPE[args.dtype]

    if args.strict_determinism:
        _apply_strict_determinism()
        print("[info] strict determinism enabled")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # One timestamp per invocation: groups all outputs of a single run and
    # prevents silent overwrite when the same prompt is regenerated later
    # with different seed / steps / CFG into the same --output_dir.
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"[info] loading pipeline from {args.model_path} (dtype={args.dtype})")
    t0 = time.time()
    pipeline = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    print(f"[info] {type(pipeline).__name__} ready in {time.time() - t0:.1f}s")

    modality = detect_modality(pipeline)
    print(f"[info] detected modality={modality.name} (cfg_arg={modality.cfg_arg}, output={modality.output_attr})")
    _resolve_modality_args(args, modality)

    # Weight injection BEFORE any device move: ``assign=True`` replaces
    # parameter tensors in place with CPU tensors from disk, so a prior
    # .to(cuda) on the pipeline would just be undone.
    if args.transformer_path:
        t0 = time.time()
        _override_transformer_weights(pipeline, args.transformer_path, dtype)
        print(f"[info] transformer overridden in {time.time() - t0:.1f}s")
    elif args.lora_path:
        t0 = time.time()
        _load_lora_adapter(
            pipeline,
            args.lora_path,
            adapter_name=args.lora_adapter_name,
            weight=args.lora_weight,
            prefix=args.lora_prefix,
        )
        print(f"[info] LoRA adapter attached in {time.time() - t0:.1f}s")

    if args.enable_cpu_offload:
        if not torch.cuda.is_available():
            raise RuntimeError("--enable_cpu_offload requires a CUDA device.")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(args.device)

    gen_device = "cuda" if (args.enable_cpu_offload or args.device == "cuda") else args.device

    cfg_summary = f"{modality.cfg_arg}={getattr(args, modality.cfg_arg)}" + (
        f", num_frames={args.num_frames}, fps={args.fps}" if modality.output_attr == "frames" else ""
    )
    print(
        f"[info] generating {len(args.prompts)} prompt(s) x {args.num_images_per_prompt} sample(s) "
        f"at {args.height}x{args.width}, steps={args.num_inference_steps}, {cfg_summary}, base_seed={args.seed}"
    )

    for idx, prompt in enumerate(args.prompts):
        prompt_seed = args.seed + idx
        generator = torch.Generator(device=gen_device).manual_seed(prompt_seed)
        pil_image = _resolve_i2v_image_for_prompt(args, idx)

        print(f"[{idx + 1}/{len(args.prompts)}] seed={prompt_seed} | {prompt}")
        t0 = time.time()
        call_kwargs = build_call_kwargs(args, modality, pil_image=pil_image)
        result = pipeline(prompt=prompt, generator=generator, **call_kwargs)

        outputs = extract_outputs(result, modality)
        slug = slugify(prompt)
        for k, item in enumerate(outputs):
            out_path = output_dir / f"{idx:03d}_{slug}_{k:02d}_{run_ts}{modality.file_ext}"
            modality.save_fn(item, out_path, args.fps or 0)
            print(f"    saved {out_path} ({time.time() - t0:.1f}s)")

    print(f"[info] all outputs in {output_dir}")


if __name__ == "__main__":
    main()
