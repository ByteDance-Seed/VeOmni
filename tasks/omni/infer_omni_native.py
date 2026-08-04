"""SeedOmni V2 native eager inference — simple process, then generate.

Loads a split checkpoint with ``OmniModel.from_pretrained`` and preprocesses
requests through :class:`~veomni.models.seed_omni.processing.OmniProcessor`
(HF ``AutoProcessor``-style API).

Examples
--------
Janus image understanding:

    python tasks/omni/infer_omni_native.py \\
        --model_path /mnt/hdfs/.../Janus-1.3B-hf \\
        --infer_type infer_und \\
        --prompt "Describe this image briefly." \\
        --image /path/to/image.jpg
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import torch

from veomni.models.seed_omni import OmniModel, OmniProcessor
from veomni.utils import helper


logger = helper.create_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SeedOmni V2 native inference (split checkpoint, checkpoint-driven load).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Split-checkpoint root (config.json + one subfolder per module).",
    )
    parser.add_argument(
        "--infer_type",
        default=None,
        help="Generation scenario key. Defaults to the checkpoint's ``infer_type``.",
    )
    parser.add_argument("--prompt", required=True, help="User text prompt.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Reference image path or URL. Repeat for multiple images.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="Directory to write reply.txt and generated images.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    helper.set_seed(args.seed)
    model = OmniModel.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
        infer_type=args.infer_type,
    ).eval()
    processor = OmniProcessor.from_model(model)
    model_input = processor(
        text=args.prompt,
        images=args.image or None,
    )

    model.reset()
    with torch.no_grad():
        generated = model.generate(
            model_input,
            generation_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample or args.temperature > 0.0,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        )

    reply = "\n".join(item["value"] for item in generated if item["type"] == "text")
    print(reply)

    os.makedirs(args.output_dir, exist_ok=True)
    reply_path = os.path.join(args.output_dir, "reply.txt")
    with open(reply_path, "w", encoding="utf-8") as handle:
        handle.write(reply)
    logger.info_rank0(f"reply → {reply_path}")

    for idx, image in enumerate(item["value"] for item in generated if item["type"] == "image"):
        out_path = os.path.join(args.output_dir, f"generated_image_{idx}.png")
        image.save(out_path)
        logger.info_rank0(f"image #{idx} → {out_path}")


if __name__ == "__main__":
    main()
