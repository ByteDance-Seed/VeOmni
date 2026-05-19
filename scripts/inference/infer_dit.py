"""Minimal native-PyTorch text-to-image inference for VeOmni-supported DiT models.

Loads a diffusers pipeline directory (class auto-detected from
``model_index.json``), optionally swaps in a VeOmni-trained transformer
checkpoint, then generates one or more images for the given prompts and
writes them to ``--output_dir``. No torchrun, no FSDP, no VeOmni trainer
wrappers — intended for quick sanity checks alongside ``train_dit.py``.

The CLI options follow ``QwenImagePipeline.__call__`` semantics
(``true_cfg_scale``, ``negative_prompt``, ``height``/``width``). Other DiT
pipelines with a compatible signature will also work; if a pipeline uses a
different kwarg layout, extend ``main``.

Example (baseline, pretrained weights only)
-------------------------------------------

python inference/infer_dit.py \
    --model_path /mnt/hdfs/veomni/models/Qwen/Qwen-Image \
    --output_dir ./inference_outputs/baseline \
    --prompts "a corgi wearing a tiny astronaut helmet, studio lighting, ultra-detailed" \
    --num_inference_steps 50 \
    --height 512 --width 512 \
    --true_cfg_scale 4.0 \
    --enable_cpu_offload

Example (fine-tuned, swap in a VeOmni-trained transformer)
----------------------------------------------------------

python inference/infer_dit.py \
    --model_path /mnt/hdfs/veomni/models/Qwen/Qwen-Image \
    --transformer_path ./qwen-image-sft/checkpoints/global_step_200/hf_ckpt \
    --output_dir ./inference_outputs/ft_step200 \
    --prompts "hiyori (blue archive), playboy bunny, solo, 1girl, aqua leotard" \
    --num_inference_steps 50 \
    --height 1024 --width 1024 \
    --true_cfg_scale 4.0 \
    --enable_cpu_offload
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file


_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _apply_strict_determinism() -> None:
    """Lock PyTorch to deterministic kernels for paper-grade reproducibility.

    Trades a small amount of throughput for cross-run / cross-PyTorch-version
    stability. ``warn_only=True`` keeps the run alive when an op (e.g. some
    VAE interpolation) lacks a deterministic implementation; otherwise
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


def _override_transformer_weights(pipeline, transformer_path: str, dtype: torch.dtype) -> None:
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Native-torch DiT text-to-image inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_path",
        required=True,
        help="Path to a diffusers pipeline directory (must contain model_index.json).",
    )
    p.add_argument(
        "--transformer_path",
        default=None,
        help=(
            "Optional VeOmni-trained transformer checkpoint dir (config.json + "
            "model.safetensors[.index.json]) or a single .safetensors file. When "
            "set, only the transformer weights inside --model_path are replaced; "
            "VAE / text encoder / scheduler stay from the base pipeline."
        ),
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
        help="Negative prompt for classifier-free guidance (applied to every prompt).",
    )
    p.add_argument("--output_dir", default="./inference_outputs")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale; <=1.0 disables guidance.",
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num_images_per_prompt", type=int, default=1)
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
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(_DTYPE.keys()),
    )
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
    return p.parse_args(argv)


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

    # Swap transformer weights BEFORE any device move: assign=True replaces
    # parameter tensors in-place with CPU tensors from disk, so a prior
    # .to(cuda) would just be undone.
    if args.transformer_path:
        t0 = time.time()
        _override_transformer_weights(pipeline, args.transformer_path, dtype)
        print(f"[info] transformer overridden in {time.time() - t0:.1f}s")

    if args.enable_cpu_offload:
        if not torch.cuda.is_available():
            raise RuntimeError("--enable_cpu_offload requires a CUDA device.")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(args.device)

    gen_device = "cuda" if (args.enable_cpu_offload or args.device == "cuda") else args.device
    print(
        f"[info] generating {len(args.prompts)} prompt(s) x {args.num_images_per_prompt} "
        f"image(s) at {args.height}x{args.width}, steps={args.num_inference_steps}, "
        f"cfg={args.true_cfg_scale}, base_seed={args.seed}"
    )

    for idx, prompt in enumerate(args.prompts):
        prompt_seed = args.seed + idx
        generator = torch.Generator(device=gen_device).manual_seed(prompt_seed)
        print(f"[{idx + 1}/{len(args.prompts)}] seed={prompt_seed} | {prompt}")
        t0 = time.time()
        result = pipeline(
            prompt=prompt,
            negative_prompt=args.negative_prompt if args.true_cfg_scale > 1.0 else None,
            true_cfg_scale=args.true_cfg_scale,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.num_images_per_prompt,
            generator=generator,
        )
        for k, image in enumerate(result.images):
            out_path = output_dir / f"{idx:03d}_{slugify(prompt)}_{k:02d}_{run_ts}.png"
            image.save(out_path)
            print(f"    saved {out_path} ({time.time() - t0:.1f}s)")

    print(f"[info] all outputs in {output_dir}")


if __name__ == "__main__":
    main()
