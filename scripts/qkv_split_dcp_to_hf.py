"""
Reverse process of the FusedQKVLinear attention fusion - splits the v5
fused ``qkv_proj`` weight back to HuggingFace's per-Linear ``q_proj`` /
``k_proj`` / ``v_proj`` keys.

This script is the QKV counterpart of ``scripts/moe_ckpt_merge/moe_split.py``:
VeOmni training saves attention weights in v5 fused layout (the model
parameter is ``self.qkv_proj.weight``); for HuggingFace ``from_pretrained()``
/ vLLM / SGLang compatibility, those keys must be split back to the original
HF three-Linear convention.

Input format (VeOmni v5 fused — what `save_hf_weights=True` produces for
models that install ``FusedQKVLinear`` via patchgen ``modify_init``, e.g.
qwen3_vl / qwen3_vl_moe):

    <prefix>.self_attn.qkv_proj.weight  [(n_q + 2*n_kv) * head_dim, hidden]
    <prefix>.self_attn.qkv_proj.bias    [(n_q + 2*n_kv) * head_dim]   (optional)

Output format (HuggingFace per-Linear):

    <prefix>.self_attn.q_proj.weight   [n_q  * head_dim, hidden]
    <prefix>.self_attn.k_proj.weight   [n_kv * head_dim, hidden]
    <prefix>.self_attn.v_proj.weight   [n_kv * head_dim, hidden]
    (+ matching .bias keys when ``config.attention_bias=True``)

The split sizes ``(n_q*hd, n_kv*hd, n_kv*hd)`` match
``FusedQKVLinear._q_out`` / ``_kv_out`` exactly, and round-trip bit-exact
through ``Qwen3VLAttentionCheckpointTensorConverter`` on the load side.

Usage:
    python scripts/qkv_split.py \\
        --merge_hf_path <fused_checkpoint_dir> \\
        --split_hf_path <output_dir>
"""

from __future__ import annotations

import os
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob
from typing import Generator

import torch
from safetensors.torch import safe_open
from tqdm import tqdm
from transformers import AutoConfig

from veomni.models import build_tokenizer, save_model_weights


# Matches the fused weight/bias keys emitted by ``FusedQKVLinear`` when
# mounted as ``self.qkv_proj`` on an HF v5 attention module — anchored on
# ``self_attn.qkv_proj`` to avoid colliding with any unrelated ``qkv_proj``
# (e.g. some vision-tower modules historically named theirs ``self.qkv``,
# which does not match).
_QKV_PATTERN = re.compile(r"^(?P<prefix>.+\.self_attn)\.qkv_proj\.(?P<kind>weight|bias)$")


@dataclass
class StateDictIterator:
    filepath: str

    def __iter__(self) -> Generator[tuple[str, torch.Tensor]]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]


def _resolve_qkv_sizes(config) -> tuple[int, int, int]:
    """Resolve (n_q, n_kv, head_dim) from a model config, handling both flat
    text configs and nested VL configs (where the attention shapes live under
    ``config.text_config``).
    """
    # qwen3_vl / qwen3_vl_moe — attention dims live under text_config.
    # qwen3 (text-only) — attention dims at the top level.
    text_config = getattr(config, "text_config", config)

    n_q = getattr(text_config, "num_attention_heads", None)
    n_kv = getattr(text_config, "num_key_value_heads", None)
    head_dim = getattr(text_config, "head_dim", None)
    hidden_size = getattr(text_config, "hidden_size", None)

    if n_q is None:
        raise ValueError("config missing `num_attention_heads` (also not under `text_config`)")
    if n_kv is None:
        # MHA models sometimes omit num_key_value_heads, falling back to MHA.
        n_kv = n_q
    if head_dim is None:
        if hidden_size is None:
            raise ValueError("config missing both `head_dim` and `hidden_size`; cannot derive head_dim")
        if hidden_size % n_q != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) not divisible by num_attention_heads ({n_q}); cannot derive head_dim"
            )
        head_dim = hidden_size // n_q

    return int(n_q), int(n_kv), int(head_dim)


def _split_qkv_state_dict(
    state_dict: dict[str, torch.Tensor],
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Pure function: rewrite a state_dict so every ``...self_attn.qkv_proj.{weight,bias}``
    is replaced by three ``q_proj/k_proj/v_proj.{weight,bias}`` keys split
    along dim-0. Non-matching keys are kept verbatim.

    The split sizes use ``head_dim * (n_q, n_kv, n_kv)`` which is identical
    to ``FusedQKVLinear._q_out`` / ``_kv_out``, so the operation is the
    exact inverse of the load-side ``torch.cat([q, k, v], dim=0)``.
    """
    q_out = n_q * head_dim
    kv_out = n_kv * head_dim
    split_sizes = [q_out, kv_out, kv_out]

    new_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        match = _QKV_PATTERN.match(name)
        if match is None:
            new_state_dict[name] = tensor
            continue

        prefix = match.group("prefix")
        kind = match.group("kind")  # "weight" or "bias"

        if tensor.shape[0] != q_out + 2 * kv_out:
            raise RuntimeError(
                f"qkv_split: tensor `{name}` has dim-0 size {tensor.shape[0]} but expected "
                f"{q_out + 2 * kv_out} from config (n_q={n_q}, n_kv={n_kv}, head_dim={head_dim}); "
                "config likely does not match the checkpoint."
            )

        q, k, v = tensor.split(split_sizes, dim=0)
        new_state_dict[f"{prefix}.q_proj.{kind}"] = q.contiguous()
        new_state_dict[f"{prefix}.k_proj.{kind}"] = k.contiguous()
        new_state_dict[f"{prefix}.v_proj.{kind}"] = v.contiguous()

    return new_state_dict


def main(merge_hf_path: str, split_hf_path: str) -> None:
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(split_hf_path, exist_ok=True)

    config = AutoConfig.from_pretrained(merge_hf_path, trust_remote_code=True)
    n_q, n_kv, head_dim = _resolve_qkv_sizes(config)
    print(f"qkv_split: resolved n_q={n_q}, n_kv={n_kv}, head_dim={head_dim}")

    tokenizer: object | None
    try:
        tokenizer = build_tokenizer(merge_hf_path)
    except Exception as e:
        # Some VL / multimodal model dirs don't ship a tokenizer (processor
        # only) — that's fine, save_model_weights accepts a list without one.
        print(f"qkv_split: no tokenizer at {merge_hf_path} ({e}); proceeding without it")
        tokenizer = None

    safetensor_files = sorted(glob(os.path.join(merge_hf_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"qkv_split: no *.safetensors files under {merge_hf_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in tqdm(safetensor_files, desc="Loading checkpoint shards"):
        for name, tensor in StateDictIterator(shard_file):
            state_dict[name] = tensor.cpu()

    print("qkv_split: splitting fused qkv_proj keys")
    new_state_dict = _split_qkv_state_dict(state_dict, n_q=n_q, n_kv=n_kv, head_dim=head_dim)

    n_fused_before = sum(1 for k in state_dict if _QKV_PATTERN.match(k))
    n_fused_after = sum(1 for k in new_state_dict if _QKV_PATTERN.match(k))
    print(f"qkv_split: rewrote {n_fused_before} fused key(s); remaining fused: {n_fused_after}")
    if n_fused_before == 0:
        print(
            "qkv_split: WARNING — no `*.self_attn.qkv_proj.{weight,bias}` keys matched; "
            "the input checkpoint is likely already in HF per-Linear layout."
        )

    model_assets = [config]
    if tokenizer is not None:
        model_assets.append(tokenizer)

    print("Saving to safetensors")
    save_model_weights(split_hf_path, new_state_dict, model_assets=model_assets)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--merge_hf_path", type=str, required=True)
    parser.add_argument("--split_hf_path", type=str, required=True)
    args = parser.parse_args()
    main(args.merge_hf_path, args.split_hf_path)
