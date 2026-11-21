# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Prepare the small random-weight checkpoint used by an eight-device model example."""

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["minimax_h3_ref2va", "qwen3_8_flash_next"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(f"{args.model}-small")
    root = Path(__file__).resolve().parents[1]
    minimax = args.model == "minimax_h3_ref2va"
    fixture = root / ("configs/models/minimax_h3_ref2va/small" if minimax else "tests/toy_config/qwen4_exp_toy")
    ops = OpsImplementationConfig(
        attn_implementation="sdpa",
        rotary_pos_emb_implementation="eager",
        rms_norm_implementation="eager",
        gated_mlp_implementation="eager",
        cross_entropy_loss_implementation="eager",
        moe_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
    )
    torch.manual_seed(71 if minimax else 42)
    model = build_foundation_model(
        config_path=str(fixture / "config.json"), init_device="cpu", torch_dtype="float32", ops_implementation=ops
    )
    output.mkdir(parents=True, exist_ok=False)
    if minimax:
        # MiniMax's published shards use bare DiT keys. The VeOmni loader adds the wrapper prefix.
        state = {name.removeprefix("dit."): value.contiguous() for name, value in model.state_dict().items()}
        save_file(state, output / "model.safetensors", metadata={"format": "pt"})
        shutil.copyfile(fixture / "config.json", output / "config.json")
    else:
        model.save_pretrained(output)


if __name__ == "__main__":
    main()
