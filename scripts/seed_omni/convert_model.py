#!/usr/bin/env python3
"""Unified entry point for SeedOmni checkpoint conversion.

Reads ``model_type`` from the upstream HuggingFace ``config.json`` at
``--model_path``, runs the matching family converter, and writes the split
checkpoint through
:func:`~veomni.models.seed_omni.utils.convert_registry.save_converted_omni`
(module subfolders plus ``training_graph.yaml`` / ``generation_graph.yaml``).

Usage::

    python scripts/seed_omni/convert_model.py \\
        --model_path /path/to/hf_checkpoint \\
        --output_dir /path/to/split_modules
"""

from __future__ import annotations

import argparse

from veomni.models.seed_omni import read_hf_model_type
from veomni.models.seed_omni.utils.convert_registry import OMNI_CONVERT_REGISTRY, save_converted_omni


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a monolithic HF checkpoint into SeedOmni modules")
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
    converted = OMNI_CONVERT_REGISTRY[model_type]()(args.model_path)
    save_converted_omni(args.output_dir, **converted)
    print(f"Conversion complete → {args.output_dir}")


if __name__ == "__main__":
    main()
