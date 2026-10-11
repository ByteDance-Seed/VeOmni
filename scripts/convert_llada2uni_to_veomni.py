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
"""Convert an upstream LLaDA2.0-Uni checkpoint (``model_type: llada2_moe``) for VeOmni's ``llada2_uni``.

The backbone weights and the image tokenizer's (``image_tokenizer/model.safetensors``,
keys ``model.visual.*`` / ``model.vqmodel.*``) already sit where
`LLaDA2UniForConditionalGeneration` expects them, so nothing is renamed; the per-expert
weights are fused at load time. The converter

  1. writes ``config.json`` (``model_type: llada2_uni``, the backbone fields, the image
     tokenizer's ``vision_config`` / ``vq_config``, the ``<|mask|>`` id);
  2. symlinks the backbone shards and the image tokenizer as one more shard
     (``model-00014-of-00014.safetensors`` after 13 backbone shards) and writes one index
     covering both;
  3. copies the GlmImage ``preprocessor_config.json`` and the tokenizer files.

``decoder/`` and ``vae/`` (image generation) are not used.

Usage:
    python scripts/convert_llada2uni_to_veomni.py --src /path/to/LLaDA2.0-Uni --dst /path/to/LLaDA2.0-Uni-veomni
    python scripts/convert_llada2uni_to_veomni.py --self-test
"""

import argparse
import json
import os
import shutil


_MASK_TOKEN = "<|mask|>"
_BACKBONE_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "use_qkv_bias",
    "use_qk_norm",
    "use_bias",
    "rms_norm_eps",
    "attention_dropout",
    "initializer_range",
    "max_position_embeddings",
    "tie_word_embeddings",
    "pad_token_id",
    "num_experts",
    "num_shared_experts",
    "num_experts_per_tok",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
    "moe_intermediate_size",
    "first_k_dense_replace",
    "image_token_offset",
)


def resolve_mask_token_id(src: str) -> int:
    with open(os.path.join(src, "tokenizer.json")) as f:
        added = {token["content"]: int(token["id"]) for token in json.load(f).get("added_tokens", [])}
    if _MASK_TOKEN not in added:
        raise ValueError(f"The tokenizer has no `{_MASK_TOKEN}` token to use as the diffusion mask.")
    return added[_MASK_TOKEN]


def convert_config(src_cfg: dict, image_tokenizer_cfg: dict, mask_token_id: int) -> dict:
    if src_cfg.get("model_type") != "llada2_moe":
        raise ValueError(f"Expected a `llada2_moe` backbone, got `{src_cfg.get('model_type')}`.")
    if image_tokenizer_cfg.get("model_type") != "glm_image":
        raise ValueError(f"Expected a `glm_image` image tokenizer, got `{image_tokenizer_cfg.get('model_type')}`.")
    rope = src_cfg.get("rope_parameters") or {
        "rope_type": "default",
        "rope_theta": src_cfg.get("rope_theta", 10000.0),
        "partial_rotary_factor": src_cfg.get("partial_rotary_factor", 0.5),
    }
    if rope.get("rope_type", "default") != "default" or src_cfg.get("rope_scaling") is not None:
        raise ValueError(f"Only the default RoPE is ported, got {rope} / rope_scaling={src_cfg.get('rope_scaling')}.")
    if src_cfg.get("num_shared_experts") is None:
        raise ValueError("The port assumes a shared expert; `num_shared_experts` is null.")
    src_cfg = {"tie_word_embeddings": False, **src_cfg}
    missing = [field for field in _BACKBONE_FIELDS if field not in src_cfg]
    if missing:
        raise ValueError(f"The source config lacks {missing}.")
    out = {
        "model_type": "llada2_uni",
        "architectures": ["LLaDA2UniForConditionalGeneration"],
        **{field: src_cfg[field] for field in _BACKBONE_FIELDS},
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(rope["rope_theta"]),
            "partial_rotary_factor": float(rope.get("partial_rotary_factor", 1.0)),
        },
        "eos_token_id": src_cfg.get("eos_token_id", src_cfg["pad_token_id"]),
        "mask_token_id": mask_token_id,
        "vision_config": image_tokenizer_cfg["vision_config"],
        "vq_config": image_tokenizer_cfg["vq_config"],
        "torch_dtype": "bfloat16",
    }
    # Without it, transformers' Mistral tokenizer heuristic warns about a wrong regex on load.
    if src_cfg.get("transformers_version"):
        out["transformers_version"] = src_cfg["transformers_version"]
    return out


def _safetensors_keys(path: str) -> list:
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        return list(f.keys())


def image_tokenizer_shard(backbone_index: dict) -> str:
    """The shard name after the backbone's, in the ``model-XXXXX-of-YYYYY`` form VeOmni parses.

    VeOmni reads each weight's shard number from that name to lay out the checkpoints it
    saves, so a free-form name such as ``image_tokenizer.safetensors`` fails at startup.
    """
    last = max(int(filename.split("-")[1]) for filename in backbone_index["weight_map"].values())
    return f"model-{last + 1:05d}-of-{last + 1:05d}.safetensors"


def merge_index(backbone_index: dict, image_tokenizer_keys: list) -> dict:
    weight_map = dict(backbone_index["weight_map"])
    clashes = [key for key in image_tokenizer_keys if key in weight_map]
    if clashes:
        raise ValueError(f"The image tokenizer and the backbone both store {clashes[:5]}.")
    unexpected = [key for key in image_tokenizer_keys if not key.startswith(("model.visual.", "model.vqmodel."))]
    if unexpected:
        raise ValueError(
            f"The image tokenizer stores weights outside `model.visual` / `model.vqmodel`: {unexpected[:5]}."
        )
    weight_map.update(dict.fromkeys(image_tokenizer_keys, image_tokenizer_shard(backbone_index)))
    return {"metadata": backbone_index.get("metadata", {}), "weight_map": weight_map}


def _symlink(target: str, link: str) -> None:
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(target), link)


def convert_checkpoint(src: str, dst: str, copy_tokenizer: bool = True) -> None:
    os.makedirs(dst, exist_ok=True)
    image_tokenizer_dir = os.path.join(src, "image_tokenizer")
    with open(os.path.join(src, "config.json")) as f:
        src_cfg = json.load(f)
    with open(os.path.join(image_tokenizer_dir, "config.json")) as f:
        image_tokenizer_cfg = json.load(f)
    out_cfg = convert_config(src_cfg, image_tokenizer_cfg, resolve_mask_token_id(src))
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    print(f"[config] wrote {dst}/config.json (model_type=llada2_uni, mask_token_id={out_cfg['mask_token_id']})")

    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    for shard in shards:
        _symlink(os.path.join(src, shard), os.path.join(dst, shard))
    image_tokenizer_weights = os.path.join(image_tokenizer_dir, "model.safetensors")
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        backbone_index = json.load(f)
    shard = image_tokenizer_shard(backbone_index)
    if shard in shards:
        raise ValueError(f"The backbone already has a shard named {shard}.")
    _symlink(image_tokenizer_weights, os.path.join(dst, shard))
    index = merge_index(backbone_index, _safetensors_keys(image_tokenizer_weights))
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[weights] symlinked {len(shards)} backbone shard(s) and the image tokenizer (no rename)")

    shutil.copy2(
        os.path.join(image_tokenizer_dir, "preprocessor_config.json"), os.path.join(dst, "preprocessor_config.json")
    )
    print("[aux] copied the GlmImage preprocessor_config.json")
    if copy_tokenizer:
        for fn in os.listdir(src):
            if fn.startswith(("tokenizer", "special_tokens", "generation_config", "vocab", "merges", "chat_template")):
                shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
                print(f"[aux] copied {fn}")
    print(f"\nDone. Converted checkpoint at: {dst}")


def self_test() -> None:
    src = {
        "model_type": "llada2_moe",
        "vocab_size": 173568,
        "hidden_size": 2048,
        "intermediate_size": 5120,
        "num_hidden_layers": 20,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_act": "silu",
        "use_qkv_bias": False,
        "use_qk_norm": True,
        "use_bias": False,
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "max_position_embeddings": 8192,
        "rope_theta": 600000,
        "rope_parameters": {"rope_type": "default", "rope_theta": 600000, "partial_rotary_factor": 0.5},
        "pad_token_id": 156892,
        "num_experts": 256,
        "num_shared_experts": 1,
        "num_experts_per_tok": 8,
        "n_group": 8,
        "topk_group": 4,
        "routed_scaling_factor": 2.5,
        "moe_intermediate_size": 512,
        "first_k_dense_replace": 1,
        "image_token_offset": 157184,
    }
    image_tokenizer = {
        "model_type": "glm_image",
        "vision_config": {"depth": 40},
        "vq_config": {"num_embeddings": 16384},
    }
    cfg = convert_config(src, image_tokenizer, 156895)
    assert cfg["model_type"] == "llada2_uni" and cfg["mask_token_id"] == 156895 and cfg["eos_token_id"] == 156892
    assert cfg["rope_parameters"] == {"rope_type": "default", "rope_theta": 600000.0, "partial_rotary_factor": 0.5}
    assert cfg["vision_config"] == {"depth": 40} and cfg["vq_config"] == {"num_embeddings": 16384}
    for bad in ({"rope_parameters": {"rope_type": "yarn", "rope_theta": 1.0}}, {"num_shared_experts": None}):
        try:
            convert_config({**src, **bad}, image_tokenizer, 156895)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must be refused")
    backbone = {
        "weight_map": {"lm_head.weight": "model-00002-of-00002.safetensors", "a": "model-00001-of-00002.safetensors"}
    }
    index = merge_index(backbone, ["model.visual.x", "model.vqmodel.y"])
    assert index["weight_map"]["model.visual.x"] == "model-00003-of-00003.safetensors"
    for keys in (["lm_head.weight"], ["model.language_model.z"]):
        try:
            merge_index(backbone, keys)
        except ValueError:
            continue
        raise AssertionError(f"{keys} must be refused")
    print("self-test OK: LLaDA2.0-Uni config mapping and index merge valid.")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a LLaDA2.0-Uni checkpoint for VeOmni's llada2_uni model.")
    p.add_argument("--src", type=str, default=None, help="Source LLaDA2.0-Uni checkpoint dir.")
    p.add_argument("--dst", type=str, default=None, help="Output dir (model_type=llada2_uni).")
    p.add_argument("--no-tokenizer", action="store_true", help="Do not copy tokenizer/aux files.")
    p.add_argument("--self-test", action="store_true", help="Run config-mapping self-test (no checkpoint needed).")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.src or not args.dst:
        p.error("--src and --dst are required (or use --self-test)")
    convert_checkpoint(args.src, args.dst, copy_tokenizer=not args.no_tokenizer)


if __name__ == "__main__":
    main()
