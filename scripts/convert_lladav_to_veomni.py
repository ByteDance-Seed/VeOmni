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
"""Convert an upstream LLaDA-V checkpoint (``model_type: llava_llada``) for VeOmni's ``llada_v``.

The weights already use the LLaVA names VeOmni's `LLaDAVForConditionalGeneration`
mirrors, so nothing is renamed:

  1. rewrite ``config.json`` (``model_type: llada_v``, a SigLIP ``vision_config`` with the
     layer count actually stored, the ``[MASK]`` id from the tokenizer);
  2. symlink the safetensors shards and copy the index and tokenizer files;
  3. write the SigLIP ``preprocessor_config.json`` (384 px, mean/std 0.5) the upstream
     repo does not ship.

Usage:
    python scripts/convert_lladav_to_veomni.py --src /path/to/LLaDA-V --dst /path/to/LLaDA-V-veomni
    python scripts/convert_lladav_to_veomni.py --self-test
"""

import argparse
import json
import os
import shutil


_DEFAULT_MASK_TOKEN_ID = 126336
_MASK_TOKEN_NAMES = ("<|mdm_mask|>", "<|mask|>")
_VISION_PREFIX = "model.vision_tower.vision_tower.vision_model.encoder.layers."
# google/siglip2-so400m-patch14-384; its depth is what `mm_vision_select_layer` counts from.
_SO400M_VISION = {
    "model_type": "siglip_vision_model",
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "image_size": 384,
    "patch_size": 14,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
    "vision_use_head": False,
}
# Fields whose other values select code paths the port does not implement.
_REQUIRED_SOURCE_VALUES = {
    "mm_projector_type": "mlp2x_gelu",
    "mm_vision_select_feature": "patch",
    "mm_newline_position": "grid",
    "mm_use_im_start_end": False,
}


def _added_tokens(src: str) -> dict:
    path = os.path.join(src, "tokenizer.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {token["content"]: int(token["id"]) for token in json.load(f).get("added_tokens", [])}


def resolve_mask_token_id(src: str) -> int:
    added = _added_tokens(src)
    return next((added[name] for name in _MASK_TOKEN_NAMES if name in added), _DEFAULT_MASK_TOKEN_ID)


def resolve_bos_token_id(src: str):
    """The tokenizer's BOS; the upstream ``config.json`` says 128000, past the end of the vocabulary."""
    path = os.path.join(src, "tokenizer_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        bos = json.load(f).get("bos_token")
    if isinstance(bos, dict):
        bos = bos.get("content")
    return _added_tokens(src).get(bos)


def count_vision_layers(weight_names) -> int:
    layers = {
        int(name[len(_VISION_PREFIX) :].split(".")[0]) for name in weight_names if name.startswith(_VISION_PREFIX)
    }
    if not layers:
        raise ValueError(f"No `{_VISION_PREFIX}*` weights: this is not a LLaDA-V checkpoint.")
    return max(layers) + 1


def convert_config(src_cfg: dict, num_vision_layers: int, mask_token_id: int, bos_token_id=None) -> dict:
    for field, expected in _REQUIRED_SOURCE_VALUES.items():
        if src_cfg.get(field, expected) != expected:
            raise ValueError(f"`{field}: {src_cfg[field]!r}` is not supported (expected {expected!r}).")
    if src_cfg.get("mm_hidden_size", _SO400M_VISION["hidden_size"]) != _SO400M_VISION["hidden_size"]:
        raise ValueError(
            f"Expected the SigLIP so400m tower (width 1152), got mm_hidden_size={src_cfg['mm_hidden_size']}."
        )

    # LLaVA drops the tower's layers after the one it reads, so the index into the
    # stored tower shifts by the number of layers removed.
    select_layer = src_cfg.get("mm_vision_select_layer", -2) + (
        _SO400M_VISION["num_hidden_layers"] - num_vision_layers
    )
    if not -num_vision_layers <= select_layer < 0:
        raise ValueError(
            f"`mm_vision_select_layer: {src_cfg.get('mm_vision_select_layer')}` does not address one of the "
            f"{num_vision_layers} stored vision layers."
        )
    n_heads = src_cfg["num_attention_heads"]
    out = {
        "model_type": "llada_v",
        "architectures": ["LLaDAVForConditionalGeneration"],
        "hidden_size": src_cfg["hidden_size"],
        "intermediate_size": src_cfg["intermediate_size"],
        "num_hidden_layers": src_cfg["num_hidden_layers"],
        "num_attention_heads": n_heads,
        "num_key_value_heads": src_cfg.get("num_key_value_heads") or n_heads,
        "head_dim": src_cfg["hidden_size"] // n_heads,
        "hidden_act": src_cfg.get("hidden_act", "silu"),
        "vocab_size": src_cfg["vocab_size"],
        "max_position_embeddings": src_cfg.get("max_position_embeddings", 4096),
        "rms_norm_eps": src_cfg.get("rms_norm_eps", 1e-5),
        "rope_theta": src_cfg.get("rope_theta", 500000.0),
        "tie_word_embeddings": bool(src_cfg.get("tie_word_embeddings", False)),
        "attention_bias": bool(src_cfg.get("attention_bias", False)),
        "mlp_bias": False,
        "attention_dropout": src_cfg.get("attention_dropout", 0.0),
        "mask_token_id": mask_token_id,
        "eos_token_id": src_cfg.get("eos_token_id", 126081),
        "pad_token_id": src_cfg.get("pad_token_id", src_cfg.get("eos_token_id", 126081)),
        "mm_hidden_size": _SO400M_VISION["hidden_size"],
        "mm_vision_select_layer": select_layer,
        "vision_config": {**_SO400M_VISION, "num_hidden_layers": num_vision_layers},
        "torch_dtype": "bfloat16",
    }
    if bos_token_id is not None:
        out["bos_token_id"] = bos_token_id
    elif src_cfg.get("bos_token_id") is not None and src_cfg["bos_token_id"] < src_cfg["vocab_size"]:
        out["bos_token_id"] = src_cfg["bos_token_id"]
    if src_cfg.get("rope_scaling") is not None:
        raise ValueError(f"`rope_scaling: {src_cfg['rope_scaling']}` is not supported.")
    return out


def _reject_identical_src_dst(src: str, dst: str) -> None:
    """Refuse a destination that resolves to the source, which the symlinking below would destroy."""
    if os.path.realpath(src) == os.path.realpath(dst):
        raise ValueError(f"--dst resolves to --src ({os.path.realpath(src)}); converting in place deletes the source.")


def convert_checkpoint(src: str, dst: str, copy_tokenizer: bool = True) -> None:
    from transformers import SiglipImageProcessor

    _reject_identical_src_dst(src, dst)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "config.json")) as f:
        src_cfg = json.load(f)
    if src_cfg.get("image_aspect_ratio", "pad") != "pad":
        print(
            f"[note] image_aspect_ratio={src_cfg['image_aspect_ratio']!r}: VeOmni trains on one "
            f"{_SO400M_VISION['image_size']}px tile per image, not on AnyRes tiles."
        )
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        weight_names = json.load(f)["weight_map"].keys()
    out_cfg = convert_config(
        src_cfg, count_vision_layers(weight_names), resolve_mask_token_id(src), resolve_bos_token_id(src)
    )
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    print(
        f"[config] wrote {dst}/config.json (model_type=llada_v, {out_cfg['vision_config']['num_hidden_layers']} "
        f"vision layers, select_layer={out_cfg['mm_vision_select_layer']}, mask_token_id={out_cfg['mask_token_id']})"
    )

    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    for shard in shards:
        link = os.path.join(dst, shard)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(os.path.join(src, shard)), link)
    shutil.copy2(os.path.join(src, "model.safetensors.index.json"), os.path.join(dst, "model.safetensors.index.json"))
    print(f"[weights] symlinked {len(shards)} safetensors shard(s) (no rename)")

    size = _SO400M_VISION["image_size"]
    SiglipImageProcessor(
        size={"height": size, "width": size}, image_mean=[0.5] * 3, image_std=[0.5] * 3
    ).save_pretrained(dst)
    print("[aux] wrote preprocessor_config.json")

    if copy_tokenizer:
        for fn in os.listdir(src):
            if fn.startswith(("tokenizer", "special_tokens", "generation_config", "vocab", "merges", "chat_template")):
                shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
                print(f"[aux] copied {fn}")
    print(f"\nDone. Converted checkpoint at: {dst}")


def self_test() -> None:
    src = {
        "model_type": "llava_llada",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 126349,
        "max_position_embeddings": 16384,
        "rope_theta": 500000.0,
        "eos_token_id": 126081,
        "bos_token_id": 128000,
        "mm_hidden_size": 1152,
        "mm_vision_select_layer": -2,
        "mm_projector_type": "mlp2x_gelu",
        "mm_vision_select_feature": "patch",
        "mm_newline_position": "grid",
    }
    names = [f"{_VISION_PREFIX}{i}.mlp.fc1.weight" for i in range(26)]
    cfg = convert_config(src, count_vision_layers(names), _DEFAULT_MASK_TOKEN_ID)
    assert cfg["vision_config"]["num_hidden_layers"] == 26 and cfg["mm_vision_select_layer"] == -1
    assert cfg["model_type"] == "llada_v" and cfg["head_dim"] == 128 and cfg["pad_token_id"] == 126081
    assert "bos_token_id" not in cfg, "an out-of-vocabulary BOS must not be copied"
    assert convert_config(src, 26, _DEFAULT_MASK_TOKEN_ID, 126080)["bos_token_id"] == 126080
    untruncated = convert_config(src, 27, _DEFAULT_MASK_TOKEN_ID)
    assert untruncated["mm_vision_select_layer"] == -2
    for field, value in (("mm_projector_type", "linear"), ("mm_vision_select_feature", "cls_patch")):
        try:
            convert_config({**src, field: value}, 26, _DEFAULT_MASK_TOKEN_ID)
        except ValueError:
            continue
        raise AssertionError(f"`{field}: {value}` must be refused")
    print("self-test OK: LLaDA-V config mapping valid.")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a LLaDA-V checkpoint for VeOmni's llada_v model.")
    p.add_argument("--src", type=str, default=None, help="Source LLaDA-V HF checkpoint dir.")
    p.add_argument("--dst", type=str, default=None, help="Output dir (model_type=llada_v).")
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
