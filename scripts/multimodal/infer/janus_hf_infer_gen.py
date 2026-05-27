#!/usr/bin/env python3
"""Janus text-to-image reference inference — HuggingFace transformers 5.2+.

Baseline for SeedOmni V2 ``infer_gen.yaml`` alignment.

Official DeepSeek script (pre-transformers):
  https://github.com/deepseek-ai/Janus/blob/main/generation_inference.py
  → manual AR loop over ``gen_head`` + CFG + ``gen_vision_model.decode_code``

Transformers equivalent (``generation_mode="image"``):
  ``JanusForConditionalGeneration.generate`` custom loop in
  ``transformers/models/janus/modeling_janus.py`` (~L1211) with CFG +
  ``decode_image_tokens`` for VQ decode.

Example
-------
  python scripts/multimodal/infer/janus_hf_infer_gen.py \\
      --model_path deepseek-ai/Janus-1.3B \\
      --prompt "A close-up high-contrast photo of Sydney Opera House..." \\
      --num_images 4 \\
      --guidance_scale 5.0 \\
      --output_dir ./janus_hf_gen_outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


_INFER_DIR = Path(__file__).resolve().parent
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))

from janus_hf_common import (  # noqa: E402
    ensure_dir,
    janus_image_generate_fix,
    load_janus,
    move_inputs_to_device,
    prepare_gen_inputs,
    save_images_from_tokens,
)


# Same default prompt as DeepSeek ``generation_inference.py``.
DEFAULT_PROMPT = (
    "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, "
    "under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Janus HF text-to-image reference script")
    p.add_argument("--model_path", type=str, default=None, help="Hub id or local dir")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--num_images", type=int, default=4, help="``num_return_sequences`` (official parallel_size)")
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance weight (official cfg_weight=5)",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", help="Greedy decoding instead of sampling")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output_dir", type=str, default="./janus_hf_gen_outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    do_sample = not args.greedy
    torch.manual_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    processor, model, device = load_janus(args.model_path, dtype=args.dtype)

    inputs, gen_kwargs = prepare_gen_inputs(
        processor,
        model,
        prompt=args.prompt,
        num_images=args.num_images,
        guidance_scale=args.guidance_scale,
    )
    inputs = move_inputs_to_device(inputs, device, dtype=model.dtype)

    # ``janus_image_generate_fix`` patches the transformers <=5.2.0 bug where
    # the image-gen AR loop misses ``is_first_iteration=True`` (PR #45044).
    # Without the patch ``model.generate`` returns garbage tokens that decode
    # to blurry noise. Becomes a no-op on >=5.9.0 thanks to ``setdefault``.
    with janus_image_generate_fix():
        image_token_ids = model.generate(
            **inputs,
            **gen_kwargs,
            generation_mode="image",
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            use_cache=True,
        )

    paths = save_images_from_tokens(model, processor, image_token_ids, out_dir, prefix="janus_gen")
    print("=== Janus text-to-image (HF baseline) ===")
    print(f"prompt         : {args.prompt}")
    print(f"num_images     : {args.num_images}")
    print(f"guidance_scale : {args.guidance_scale}")
    print(f"token shape    : {tuple(image_token_ids.shape)}")
    for path in paths:
        print(f"saved → {path}")

    (out_dir / "prompt.txt").write_text(args.prompt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
