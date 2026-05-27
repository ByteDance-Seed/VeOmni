#!/usr/bin/env python3
"""Janus image understanding (I2T) reference inference — HuggingFace transformers 5.2+.

Baseline for SeedOmni V2 ``infer_und.yaml`` alignment.

Official DeepSeek script (pre-transformers):
  https://github.com/deepseek-ai/Janus/blob/main/inference.py
  → ``language_model.generate(inputs_embeds=..., ...)``

Transformers equivalent (``generation_mode="text"``):
  ``JanusForConditionalGeneration.generate`` in
  ``transformers/models/janus/modeling_janus.py`` (delegates to ``GenerationMixin``).

Example
-------
  python scripts/multimodal/infer/janus_hf_infer_und.py \\
      --model_path deepseek-ai/Janus-1.3B \\
      --image_path /path/to/image.png \\
      --prompt "Describe this image in detail." \\
      --output_dir ./janus_hf_und_outputs
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
    decode_new_text,
    ensure_dir,
    load_janus,
    move_inputs_to_device,
    prepare_und_inputs,
)


DEFAULT_PROMPT = "What do you see in this image?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Janus HF image-understanding (text generation) reference script")
    p.add_argument(
        "--model_path", type=str, default=None, help="Hub id or local dir (default: deepseek-ai/Janus-1.3B)"
    )
    p.add_argument("--image_path", type=str, default=None, help="Local image path (recommended; avoids URL fetch)")
    p.add_argument("--image_url", type=str, default=None, help="Remote image URL (requires network)")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--do_sample", action="store_true", help="Sample; default is greedy (matches official und demo)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output_dir", type=str, default="./janus_hf_und_outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    processor, model, device = load_janus(args.model_path, dtype=args.dtype)

    inputs, image_label = prepare_und_inputs(
        processor,
        prompt=args.prompt,
        image_path=args.image_path,
        image_url=args.image_url,
        model_path=args.model_path,
    )
    inputs = move_inputs_to_device(inputs, device, dtype=model.dtype)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        generation_mode="text",
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        use_cache=True,
    )

    reply = decode_new_text(processor, inputs["input_ids"], output_ids)
    print("=== Janus image understanding (HF baseline) ===")
    print(f"prompt : {args.prompt}")
    print(f"image  : {image_label}")
    print(f"reply  : {reply}")

    (out_dir / "reply.txt").write_text(reply + "\n", encoding="utf-8")
    meta = out_dir / "run_meta.txt"
    meta.write_text(
        "\n".join(
            [
                f"model_path={args.model_path}",
                f"image={image_label}",
                f"prompt={args.prompt!r}",
                f"max_new_tokens={args.max_new_tokens}",
                f"do_sample={args.do_sample}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved reply → {out_dir / 'reply.txt'}")


if __name__ == "__main__":
    main()
