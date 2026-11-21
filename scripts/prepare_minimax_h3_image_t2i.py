"""Prepare the native MiniMax-H3-Image transformer for offline T2I training."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer import (
    MiniMaxH3DiTModelConfig,
)
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
    MiniMaxH3DiTModel,
)


_PDD_HEADS = 6
_PDD_PREFIX = "final_layer.pdd_out."
_PDD_SHARD = "model-pdd-heads.safetensors"


def _validate_transformer(transformer_dir, config_dict, weight_map):
    config = MiniMaxH3DiTModelConfig(**config_dict)
    config._set_ops_implementation(
        {
            "attn_implementation": "eager",
            "rms_norm_implementation": "eager",
            "rotary_pos_emb_implementation": "eager",
        }
    )
    with torch.device("meta"):
        dit = MiniMaxH3DiTModel(config).dit
    required = {name for name in dit.state_dict() if not name.startswith(_PDD_PREFIX)}
    buffers = dict(dit.named_buffers())
    shapes = {name: tensor.shape for name, tensor in (dict(dit.named_parameters()) | buffers).items()}

    missing = sorted(required - weight_map.keys())
    unexpected = sorted(weight_map.keys() - required - buffers.keys())
    if missing or unexpected:
        raise ValueError(
            f"Native transformer index does not match the image DiT: missing {missing}; unexpected {unexpected}."
        )

    shards = defaultdict(set)
    for name, filename in weight_map.items():
        if (
            Path(filename).name != filename
            or not filename.endswith(".safetensors")
            or filename in (SAFE_WEIGHTS_NAME, _PDD_SHARD)
        ):
            raise ValueError(f"Invalid native shard filename: {filename!r}")
        shards[filename].add(name)
    for filename, names in shards.items():
        with safe_open(transformer_dir / filename, framework="pt", device="cpu") as shard:
            if set(shard.keys()) != names:
                raise ValueError(f"Native shard {filename} keys do not match its index entries.")
            for name in names:
                shape = tuple(shard.get_slice(name).get_shape())
                if shape != shapes[name]:
                    raise ValueError(f"Native tensor {name} has shape {shape}; expected {tuple(shapes[name])}.")
    return shards.keys(), shapes


def _read_pdd_heads(model_dir, shapes):
    state = load_file(model_dir / "pdd_fused_heads.safetensors", device="cpu")
    if state.keys() != {"video_out.weight", "video_out.bias"}:
        raise ValueError("Expected only video_out.weight and video_out.bias in pdd_fused_heads.safetensors.")

    heads = {}
    for suffix in ("weight", "bias"):
        name = f"video_out.{suffix}"
        stacked = state[name]
        expected = (_PDD_HEADS, *shapes[f"{_PDD_PREFIX}0.{suffix}"])
        if tuple(stacked.shape) != expected or not stacked.is_floating_point():
            raise ValueError(
                f"PDD {name} must be floating point with shape {expected}; got {stacked.dtype} {stacked.shape}."
            )
        for step, tensor in enumerate(stacked.unbind()):
            heads[f"{_PDD_PREFIX}{step}.{suffix}"] = tensor.clone()
    return heads


def prepare_checkpoint(model_dir: Path, out_dir: Path) -> None:
    """Link the release's transformer shards and add its six PDD heads as one shard."""
    model_dir = model_dir.resolve()
    if out_dir.is_symlink() or (out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir()))):
        raise FileExistsError(f"Output must be a new or empty directory: {out_dir}")
    out_dir = out_dir.resolve()
    if out_dir.is_relative_to(model_dir):
        raise ValueError("Output must be outside the source release directory.")

    transformer_dir = model_dir / "transformer"
    config = json.loads((transformer_dir / "config.json").read_text())
    if (
        config.get("model_type", config.get("_class_name")) != "MiniMaxH3DiTModel"
        or config.get("audio_enabled") is not False
    ):
        raise ValueError("Expected a native MiniMax-H3-Image transformer config (MiniMaxH3DiTModel without audio).")
    config.update(
        model_type="MiniMaxH3DiTModel",
        architectures=["MiniMaxH3DiTModel"],
        condition_model_type="MiniMaxH3ImageConditionModel",
        image_training=True,
        num_pdd_heads=_PDD_HEADS,
    )
    index = json.loads((transformer_dir / SAFE_WEIGHTS_INDEX_NAME).read_text())
    shard_names, shapes = _validate_transformer(transformer_dir, config, index["weight_map"])
    heads = _read_pdd_heads(model_dir, shapes)
    index["weight_map"].update(dict.fromkeys(heads, _PDD_SHARD))
    index["metadata"]["total_size"] += sum(tensor.numel() * tensor.element_size() for tensor in heads.values())

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out_dir.name}.", dir=out_dir.parent) as staging_dir:
        staging = Path(staging_dir)
        for filename in shard_names:
            (staging / filename).symlink_to(transformer_dir / filename)
        save_file(heads, staging / _PDD_SHARD, metadata={"format": "pt"})
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
        staging.rename(out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to the native H3-Image release directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New or empty output transformer directory.")
    args = parser.parse_args()
    try:
        prepare_checkpoint(args.model_dir, args.out_dir)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(
        f"Prepared {args.out_dir}; keep the source transformer shards at {args.model_dir.resolve() / 'transformer'}."
    )


if __name__ == "__main__":
    main()
