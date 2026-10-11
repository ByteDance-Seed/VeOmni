# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A ~2M-parameter DiffusionGemma checkpoint in HuggingFace layout, for parity runs.

The cross-stack comparison needs both stacks holding the *same weights*, and the
only honest way to get there is for each to load through the loader it already
uses in training. Hand-writing a key mapping between the two would put the thing
under test -- how each stack interprets the HF layout -- into the test harness.

So the checkpoint is written by ``transformers``' own reference
``DiffusionGemmaForBlockDiffusion``. Its ``save_pretrained`` *defines* the layout
both loaders claim to read, which makes the artifact a third party to the
comparison rather than either side's idea of it.

Run this under the NeMo AutoModel venv: the reference model ships in
``transformers>=5.15``. VeOmni 5.16.1 also ships it, but vendors a training
fork, so the artifact that defines the HF layout is still written here.

    /opt/tiger/diffusion-gemma-automodel/.venv/bin/python \
        scripts/parity/make_tiny_diffusion_gemma.py --out /tmp/dg_parity/tiny_ckpt

Everything is fp32 and seeded. fp32 because bf16 would put a ~1e-2 rounding floor
under a comparison that is trying to resolve 1e-5; seeded because a parity failure
is only worth chasing if the inputs can be reproduced exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch


# Small enough to run on a CPU in a second, structured enough to exercise the parts
# that actually differ: both layer types (so the sliding rope/window path runs), a
# real MoE with top-k routing, GQA with a distinct global head dim, and logit
# softcapping. Shrinking any of these to a degenerate value would hide a whole path.
TINY_TEXT_CONFIG = {
    "model_type": "diffusion_gemma_text",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    # Deliberately not ``hidden_size / 2``: that would make the stacked gate/up expert
    # weight square, and a stack storing it transposed would then be invisible to a
    # shape check and show up only as a scrambled gradient.
    "moe_intermediate_size": 24,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 16,
    "global_head_dim": 32,
    "num_experts": 4,
    "top_k_experts": 2,
    # A sliding layer downstream of a full one, so a bug that only shows up after a
    # full layer has run is reachable. The reference rewrites a non-full last layer,
    # so the stack ends on one rather than being silently corrected.
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ],
    "sliding_window": 12,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "hidden_activation": "gelu_pytorch_tanh",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "final_logit_softcapping": 30.0,
    # Mirrors the released checkpoint. It selects which *input* spans are bidirectional
    # and is not what makes the canvas bidirectional, so it stays as shipped even though
    # this config is text-only.
    "use_bidirectional_attention": "vision",
    "rope_parameters": {
        "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
    "tie_word_embeddings": True,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "initializer_range": 0.02,
    "dtype": "float32",
}

# The reference model builds a vision tower unconditionally, so one has to exist for
# it to be constructed at all. It is stripped from the artifact afterwards (see
# ``strip_vision``) -- the parity steps are text-only, VeOmni skips the tower when
# ``vision_config`` is null, and NeMo's implementation has no tower to put it in.
TINY_VISION_CONFIG = {
    "model_type": "gemma4_vision",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "global_head_dim": 16,
    "patch_size": 16,
    "position_embedding_size": 64,
    "pooling_kernel_size": 2,
    "default_output_length": 4,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "hidden_activation": "gelu_pytorch_tanh",
    "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
    "standardize": True,
    "use_clipped_linears": False,
    "initializer_range": 0.02,
}

TINY_CONFIG = {
    "architectures": ["DiffusionGemmaForBlockDiffusion"],
    "model_type": "diffusion_gemma",
    "vision_config": TINY_VISION_CONFIG,
    "canvas_length": 8,
    "eos_token_id": [1],
    "tie_word_embeddings": True,
    "initializer_range": 0.02,
    "dtype": "float32",
    "text_config": TINY_TEXT_CONFIG,
}


def fill_deterministically(model: torch.nn.Module, seed: int, unit_expert_scale: bool = False) -> None:
    """Overwrite every parameter from one seeded stream, ignoring the model's own init.

    The reference ``_init_weights`` varies between transformers releases, so letting
    it run would tie the artifact to whichever release happened to write it. Drawing
    every tensor from a single generator in a fixed name order makes the checkpoint a
    function of the seed alone.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            # Gemma's RMSNorm computes ``x * (1 + weight)``, so a norm weight is a
            # residual around zero like everything else here; 0.02 keeps the forward
            # in a sane range rather than saturating the logit softcap.
            draw = torch.empty(parameter.shape, dtype=torch.float32).normal_(0.0, 0.02, generator=generator)
            # Drawn either way so the rest of the stream is unaffected and the two
            # variants differ in exactly this tensor.
            if unit_expert_scale and name.endswith("per_expert_scale"):
                draw = torch.ones_like(draw)
            parameter.copy_(draw)


VISION_PREFIXES = ("model.encoder.vision_tower.", "model.encoder.embed_vision.", "vision_tower.", "embed_vision.")


def strip_vision(out: Path) -> list[str]:
    """Drop the vision tower from the saved artifact and null the config that built it.

    The text weights keep the names ``save_pretrained`` gave them -- this removes
    tensors, it does not rename any -- so the artifact is still the reference's
    layout, restricted to the half both stacks implement.
    """
    from safetensors.torch import load_file, save_file

    weights_path = out / "model.safetensors"
    if not weights_path.is_file():
        raise SystemExit(
            f"expected a single unsharded {weights_path.name}; the tiny model got large enough to shard, "
            "so the shard index needs rewriting too"
        )

    tensors = load_file(weights_path)
    for key in [k for k in tensors if k.startswith(VISION_PREFIXES)]:
        del tensors[key]
    save_file(tensors, weights_path, metadata={"format": "pt"})
    return sorted(tensors)


def rewrite_config_in_released_schema(out: Path) -> None:
    """Express the full-attention geometry the way the released checkpoint does.

    The two stacks pin transformers versions that disagree about *how* this config is
    spelled, not about what it means. The released checkpoint and VeOmni's vendored
    modeling read ``global_head_dim`` / ``num_global_key_value_heads`` as plain
    text-config fields. Newer transformers pops them into a ``per_layer_config``
    override map and serializes the originals as ``null``, which the released
    spelling then reads as "no global head dim" and dies dividing by ``None``.

    The released ``diffusiongemma-26B-A4B-it`` config.json uses the flat spelling, so that
    is the interop contract and the one to write. This translates the override map back
    into it, and refuses anything it cannot translate rather than writing a config that
    describes different weights than the ones next to it.
    """
    config_path = out / "config.json"
    config = json.loads(config_path.read_text())
    text = config["text_config"]
    config["vision_config"] = None

    overrides = text.pop("per_layer_config", None) or {}
    full_layers = {index for index, kind in enumerate(text["layer_types"]) if kind == "full_attention"}
    if {int(index) for index in overrides} != full_layers:
        raise SystemExit(
            f"per_layer_config covers layers {sorted(int(i) for i in overrides)} but the full-attention "
            f"layers are {sorted(full_layers)}; the released schema has no way to say that"
        )
    distinct = {tuple(sorted(override.items())) for override in overrides.values()}
    if len(distinct) > 1:
        raise SystemExit(f"per_layer_config is not uniform across full-attention layers: {overrides}")
    if distinct:
        override = dict(next(iter(distinct)))
        if set(override) - {"head_dim", "num_key_value_heads"}:
            raise SystemExit(f"per_layer_config carries keys the released schema cannot express: {override}")
        text["global_head_dim"] = override.get("head_dim")
        text["num_global_key_value_heads"] = override.get("num_key_value_heads")

    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="directory to write the checkpoint into")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--config-override",
        type=str,
        default=None,
        help=(
            "JSON merged into the tiny config, `text_config` merged one level deep. "
            "Lets a single knob be moved to its released value and the two stacks re-compared "
            "in seconds, which is how a divergence that only appears at the 26B config gets "
            "attributed to a field rather than to scale"
        ),
    )
    parser.add_argument(
        "--unit-expert-scale",
        action="store_true",
        help=(
            "write per_expert_scale as ones, which is what makes the two stacks' gradients comparable: "
            "VeOmni keeps the scale as a parameter and NeMo folds it into the expert down-projection, so "
            "for any other value the same expert weight means a different tensor on each side"
        ),
    )
    args = parser.parse_args()

    try:
        from transformers.models.diffusion_gemma import (
            DiffusionGemmaConfig,
            DiffusionGemmaForBlockDiffusion,
        )
    except ModuleNotFoundError as error:
        import transformers

        raise SystemExit(
            f"transformers {transformers.__version__} has no reference `diffusion_gemma` model "
            "(added in 5.15). Run this under the NeMo AutoModel venv so the artifact is defined "
            "by HuggingFace's class, not VeOmni's vendored training fork."
        ) from error

    spec = copy.deepcopy(TINY_CONFIG)
    if args.config_override:
        override = json.loads(args.config_override)
        spec["text_config"].update(override.pop("text_config", {}))
        spec.update(override)

    config = DiffusionGemmaConfig(**spec)
    config.dtype = torch.float32

    torch.manual_seed(args.seed)
    model = DiffusionGemmaForBlockDiffusion(config).to(torch.float32)
    fill_deterministically(model, args.seed, unit_expert_scale=args.unit_expert_scale)
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    keys = strip_vision(args.out)
    rewrite_config_in_released_schema(args.out)

    import transformers

    manifest = {
        "seed": args.seed,
        "unit_expert_scale": args.unit_expert_scale,
        "written_by": f"transformers {transformers.__version__} DiffusionGemmaForBlockDiffusion",
        "text_parameters": sum(
            p.numel() for name, p in model.named_parameters() if not name.startswith(VISION_PREFIXES)
        ),
        "tensor_keys": keys,
    }
    (args.out / "parity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {manifest['text_parameters'] / 1e6:.2f}M text parameters ({len(keys)} tensors) to {args.out}")
    for name in sorted(p.name for p in args.out.iterdir()):
        print(f"  {name}")


if __name__ == "__main__":
    main()
