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

from veomni.models.seed_omni.convert_registry import convert_checkpoint, read_hf_model_type


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
    args = parser.parse_args()

    model_type = read_hf_model_type(args.model_path)
    print(f"Detected model_type={model_type!r} from {args.model_path}")
    convert_checkpoint(args.model_path, args.output_dir)
    print(f"Conversion complete → {args.output_dir}")


if __name__ == "__main__":
    main()
