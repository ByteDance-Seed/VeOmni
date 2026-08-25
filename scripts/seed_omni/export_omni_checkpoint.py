#!/usr/bin/env python3
"""Build a bare :class:`OmniModel` from module weights and re-export a self-contained omni checkpoint.

The launcher YAML — the same ``base.yaml`` used for a training or inference trial —
is the source of truth for graphs, ``infer_type``, ``generation_kwargs``, and module
layout.  ``model.model_path`` in that YAML (or a CLI override) points at the weight
root: a split checkpoint from ``scripts/convert_model.py`` (module subfolders only),
an assembled training step, or an existing omni root whose weights you want to
re-package under an updated config projection.

Usage::

    python scripts/seed_omni/export_omni_checkpoint.py \\
        configs/seed_omni/Janus/janus_1.3b/base.yaml \\
        --export.output_dir /path/to/Janus-1.3B-hf \\
        --model.model_path /mnt/hdfs/.../Janus-1.3B \\
        --export.verify
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from veomni.arguments import OmniArguments, parse_omni_args
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel


@dataclass
class ExportCheckpointArguments:
    """``export.*`` — one-shot omni checkpoint export knobs."""

    output_dir: str = field(default="", metadata={"help": "Destination omni checkpoint root."})
    save_module_weights: bool = field(
        default=True,
        metadata={"help": "Write module weights; set false to export config/assets only."},
    )
    verify: bool = field(
        default=False,
        metadata={"help": "Reload the exported root with OmniModel.from_pretrained."},
    )


@dataclass
class Arguments(OmniArguments):
    """Root config for ``export_omni_checkpoint`` — extends the omni launcher schema."""

    export: ExportCheckpointArguments = field(default_factory=ExportCheckpointArguments)


def build_config(args: Arguments) -> OmniConfig:
    """Project the launcher YAML onto the checkpoint-shaped :class:`OmniConfig`."""
    config = args.resolve_model().to_hf_config()

    model_path = args.model.model_path
    print(f"OmniConfig: {len(config.module_names)} module(s) from launcher YAML")
    for name in config.module_names:
        print(f"  {name:24s} subfolder={config.module_checkpoint_subfolder(name)!r}")
        print(f"  {'':24s} path={config.resolve_module_path(model_path, name)}")
    print(f"  training_graph edges: {len(config.training_edges)}")
    print(f"  generation scenarios: {', '.join(config.infer_types)}")
    for infer_type in config.infer_types:
        print(f"    {infer_type:20s} initial={config.generation_graphs[infer_type].get('initial')!r}")
    print(f"  active infer_type: {config.infer_type or config.infer_types[0]!r}")
    print(f"  generation_kwargs: {config.generation_kwargs}")
    return config


def build_model(model_path: str, config: OmniConfig) -> OmniModel:
    """Load every module from ``model_path`` into a bare :class:`OmniModel`."""
    model = OmniModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    for name, module in model.modules_dict.items():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:24s} {type(module).__name__:32s} {n_params / 1e6:8.2f}M params")
    return model


def verify_reload(output_dir: str) -> OmniModel:
    """Prove the exported root is self-contained (no launcher YAML needed)."""
    config = OmniConfig.from_pretrained(output_dir)
    print(f"Config: {config}")
    reloaded = OmniModel.from_pretrained(output_dir)
    print(f"Reloaded {type(reloaded).__name__} with modules: {list(reloaded.modules_dict)}")
    return reloaded


def main() -> None:
    args = parse_omni_args(
        Arguments,
        preload_path_fields=("model.model_config.modules",),
    )
    if not args.export.output_dir:
        raise SystemExit("--export.output_dir is required.")
    if not args.model.model_path:
        raise SystemExit("`model.model_path` is required (set it in the launcher YAML or via CLI).")

    config = build_config(args)
    model = build_model(args.model.model_path, config)

    model.save_pretrained(args.export.output_dir, save_module_weights=args.export.save_module_weights)
    print(f"Exported omni checkpoint → {args.export.output_dir}")
    for entry in sorted(os.listdir(args.export.output_dir)):
        path = os.path.join(args.export.output_dir, entry)
        if os.path.isdir(path):
            print(f"  {entry}/  ({', '.join(sorted(os.listdir(path)))})")
        else:
            print(f"  {entry}")

    if args.export.verify:
        verify_reload(args.export.output_dir)


if __name__ == "__main__":
    main()
