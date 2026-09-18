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
        --output_dir /path/to/split_modules \\
        --training_graph configs/seed_omni/fake_model/graph_train.yaml \\
        --generation_graph configs/seed_omni/fake_model/graph_infer.yaml
"""

from __future__ import annotations

import argparse

from veomni.models.seed_omni import read_hf_model_type
from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint


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
        help="Directory to write the split omni checkpoint (modules + graph YAML sidecars)",
    )
    parser.add_argument(
        "--training_graph",
        default=None,
        help=(
            "YAML for the training DAG (a list, or `{training_graph: [...]}`). "
            "Overrides the family converter's default; written as training_graph.yaml."
        ),
    )
    parser.add_argument(
        "--generation_graph",
        default=None,
        help=(
            "YAML for generation FSMs (`generation_graphs:` keyed by infer_type). "
            "Overrides the family converter's default; written as generation_graph.yaml."
        ),
    )
    args = parser.parse_args()

    model_type = read_hf_model_type(args.model_path)
    print(f"Detected model_type={model_type!r} from {args.model_path}")
    convert_checkpoint(
        args.model_path,
        args.output_dir,
        training_graph=args.training_graph,
        generation_graph=args.generation_graph,
    )
    print(f"Conversion complete → {args.output_dir}")


if __name__ == "__main__":
    main()
