#!/usr/bin/env python3
"""Unified entry point for SeedOmni V2 checkpoint conversion.

Reads ``model_type`` from the upstream HuggingFace ``config.json`` at
``--model_path`` and dispatches to the matching family converter registered
under ``veomni/models/seed_omni/modules/<family>/convert_model.py``.

Usage::

    python scripts/convert_model.py \\
        --model_path /mnt/hdfs/veomni/models/transformers/Janus-1.3B \\
        --output_dir /mnt/hdfs/veomni/models/seed_omni/janus_1.3b
"""

from __future__ import annotations

import argparse

from veomni.models.seed_omni import read_hf_model_type
from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a monolithic HF checkpoint into SeedOmni V2 modules")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Upstream HuggingFace checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write split module sub-checkpoints",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "native"],
        help=(
            "Dequantize to bf16 for training, or keep the released dtypes for inference. "
            "Only families with quantized releases accept this"
        ),
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        help="Convert only these module subfolders instead of all of them",
    )
    args = parser.parse_args()

    # Only forward what was asked for. These are per-family options, and a
    # converter that does not take them should not have to grow a signature to
    # ignore a default nobody set.
    options = {name: value for name, value in (("precision", args.precision), ("modules", args.modules)) if value}

    model_type = read_hf_model_type(args.model_path)
    print(f"Detected model_type={model_type!r} from {args.model_path}")
    convert_checkpoint(args.model_path, args.output_dir, **options)
    print(f"Conversion complete → {args.output_dir}")


if __name__ == "__main__":
    main()
