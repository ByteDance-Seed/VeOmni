# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Where each tensor of the released DeepSeek-V4.1 checkpoint lands in the four modules.

Two readers apply this rename: the offline splitter (``convert_model.py``) and
the direct load through ``hf_layout.py``. Both take it from here, so a split
checkpoint and a direct load hold the same module keys.

The release names are the reference inference stack's (``layers.N.attn.wq_a``,
``ffn.gate.bias``, ``hc_attn_fn``), not HuggingFace's. A quantized weight
``P.weight`` ships its block scale as ``P.scale``; the modules hold it as
``<target>_scale`` beside the weight.

Routed experts are the one family not renamed key by key: the release stores
each expert separately (``layers.N.ffn.experts.E.w1.weight``) and the module
holds them fused and stacked, so :func:`expert_prefixes` only moves the prefix
and the module's checkpoint tensor converter fuses them.
"""

from __future__ import annotations

from typing import Iterable


VISION = "deepseek_v41_vision"
TEXT_ENCODER = "deepseek_v41_text_encoder"
NGRAM = "deepseek_v41_ngram"
LLM = "deepseek_v41_llm"

#: The subfolders a full conversion writes, in the order it writes them.
MODULES = (VISION, TEXT_ENCODER, NGRAM, LLM)

#: The vision tower's names match :class:`DeepseekV41Vision` exactly.
VISION_PREFIXES = ("vision.", "aligner.", "image_")

TEXT_ENCODER_NAMES = {"embed.weight": "embed_tokens.weight", "head.weight": "lm_head.weight"}


def released_scale_name(weight_name: str) -> str:
    """The release's scale key for a weight: ``P.weight`` -> ``P.scale``, anything else ``+ ".scale"``."""
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")] + ".scale"
    return weight_name + ".scale"


def module_scale_name(target: str) -> str:
    """The module's scale key for a weight: ``P`` -> ``P_scale``."""
    return f"{target}_scale"


def llm_attention_names(layer: int) -> dict[str, str]:
    """Released -> module, for one layer's attention block."""
    prefix = f"layers.{layer}.attn"
    target = f"model.layers.{layer}.self_attn"
    names = {
        f"{prefix}.wq_a.weight": f"{target}.wq_a.weight",
        f"{prefix}.wq_b.weight": f"{target}.wq_b.weight",
        f"{prefix}.wkv.weight": f"{target}.wkv.weight",
        f"{prefix}.wo_a.weight": f"{target}.wo_a.weight",
        f"{prefix}.wo_b.weight": f"{target}.wo_b.weight",
        f"{prefix}.q_norm.weight": f"{target}.q_norm.weight",
        f"{prefix}.kv_norm.weight": f"{target}.kv_norm.weight",
        f"{prefix}.attn_sink": f"{target}.attn_sink",
    }
    for suffix in (
        "compressor.norm.weight",
        "compressor.wkv.weight",
        "compressor.wgate.weight",
        "indexer.wq_b.weight",
        "indexer.wk.weight",
        "indexer.k_norm.weight",
        "indexer.weights_proj.weight",
    ):
        names[f"{prefix}.{suffix}"] = f"{target}.{suffix}"
    return names


def llm_layer_names(layer: int) -> dict[str, str]:
    """Released -> module, for everything in one layer except the experts.

    The two norms are renamed rather than passed through: the released
    ``attn_norm`` / ``ffn_norm`` are HF's ``input_layernorm`` /
    ``post_attention_layernorm``, which is what the decoder layer declares.
    """
    names = llm_attention_names(layer)
    names[f"layers.{layer}.attn_norm.weight"] = f"model.layers.{layer}.input_layernorm.weight"
    names[f"layers.{layer}.ffn_norm.weight"] = f"model.layers.{layer}.post_attention_layernorm.weight"
    names[f"layers.{layer}.ffn.gate.weight"] = f"model.layers.{layer}.mlp.gate.weight"
    # The aux-loss-free controllers. The release calls them `bias`; every
    # DeepSeek-family module in this repo calls them e_score_correction_bias and
    # holds them as buffers, not parameters -- they are updated by a sign rule on
    # expert load, not by the objective.
    names[f"layers.{layer}.ffn.gate.bias"] = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    names[f"layers.{layer}.ffn.gate.bias_vl"] = f"model.layers.{layer}.mlp.gate.e_score_correction_bias_vl"
    for site in ("attn", "ffn"):
        for part in ("fn", "base", "scale"):
            names[f"layers.{layer}.hc_{site}_{part}"] = f"model.layers.{layer}.{site}_hc.{part}"
    for source_name, target in (
        ("w1", "gate_proj"),
        ("w2", "down_proj"),
        ("w3", "up_proj"),
    ):
        names[f"layers.{layer}.ffn.shared_experts.{source_name}.weight"] = (
            f"model.layers.{layer}.mlp.shared_experts.{target}.weight"
        )
    return names


def mtp_layer_names(stage: int) -> dict[str, str]:
    """Released ``mtp.{stage}.*`` -> ``model.mtp.{stage}.*``, same inner mapping as a backbone layer."""
    return {
        released.replace(f"layers.{stage}", f"mtp.{stage}", 1): target.replace(
            f"model.layers.{stage}", f"model.mtp.{stage}", 1
        )
        for released, target in llm_layer_names(stage).items()
    }


def mtp_extra_names(num_stages: int) -> dict[str, str]:
    """The DSpark tensors outside a stage's layer: the main-stream projection and the last stage's heads."""
    if num_stages <= 0:
        return {}
    last = num_stages - 1
    return {
        "mtp.0.main_proj.weight": "model.mtp.0.main_proj.weight",
        "mtp.0.main_norm.weight": "model.mtp.0.main_norm.weight",
        f"mtp.{last}.norm.weight": f"model.mtp.{last}.norm.weight",
        f"mtp.{last}.markov_head.embed.weight": f"model.mtp.{last}.markov_head.embed.weight",
        f"mtp.{last}.markov_head.head.weight": f"model.mtp.{last}.markov_head.head.weight",
        f"mtp.{last}.confidence_head.proj.weight": f"model.mtp.{last}.confidence_head.proj.weight",
    }


def tied_mtp_names(num_stages: int) -> dict[str, str]:
    """``mtp.*.embed`` / ``mtp.*.head`` alias the text encoder's tensors when a release stores them."""
    tied = {}
    for stage in range(num_stages):
        tied[f"mtp.{stage}.embed.weight"] = "embed.weight"
        tied[f"mtp.{stage}.head.weight"] = "head.weight"
    return tied


def engram_connection_names(layer: int) -> dict[str, str]:
    """The Engram connection's half: ``q``/``k`` gates and ``wkv`` read the mHC stream, so they stay in the llm."""
    connection = f"model.engram.{layer}"
    return {
        f"layers.{layer}.engram.q_weight": f"{connection}.q_weight",
        f"layers.{layer}.engram.k_weight": f"{connection}.k_weight",
        f"layers.{layer}.engram.wkv.weight": f"{connection}.wkv.weight",
    }


def ngram_table_names(layer_ids: Iterable[int]) -> dict[str, str]:
    """The tables' half of the Engram: one table per Engram layer, in ``layer_ids`` order."""
    return {
        f"layers.{layer_id}.engram.embed.weight": f"tables.{index}.weight" for index, layer_id in enumerate(layer_ids)
    }


def expert_prefixes(num_layers: int, num_stages: int) -> dict[str, str]:
    """Released per-expert prefix -> the module prefix the fused experts live under."""
    prefixes = {f"layers.{layer}.ffn.experts.": f"model.layers.{layer}.mlp.experts." for layer in range(num_layers)}
    prefixes.update({f"mtp.{stage}.ffn.experts.": f"model.mtp.{stage}.mlp.experts." for stage in range(num_stages)})
    return prefixes


def llm_dense_names(num_layers: int, engram_layer_ids: Iterable[int], num_stages: int) -> dict[str, str]:
    """Every llm tensor but the routed experts: layers, Engram connections, DSpark stages, final norm."""
    engram_layer_ids = set(engram_layer_ids)
    names: dict[str, str] = {}
    for layer in range(num_layers):
        names.update(llm_layer_names(layer))
        if layer in engram_layer_ids:
            names.update(engram_connection_names(layer))
    for stage in range(num_stages):
        names.update(mtp_layer_names(stage))
    names.update(mtp_extra_names(num_stages))
    names["norm.weight"] = "model.norm.weight"
    return names


def _with_scales(names: dict[str, str]) -> list[tuple[str, str]]:
    pairs = []
    for released, target in names.items():
        pairs.append((released, target))
        pairs.append((released_scale_name(released), module_scale_name(target)))
    return pairs


def release_key_prefixes(text_config: dict) -> dict[str, tuple[tuple[str, str], ...]]:
    """``{module: ((released_prefix, module_prefix), ...)}`` covering every tensor of a release.

    Dense tensors are listed key by key -- most of their renames sit in the
    middle of the key, where a prefix cannot reach -- each with its scale, so a
    quantized weight and its scale land on the module names the split checkpoint
    uses. ``text_config`` is the release's ``text_config``, which fixes the
    layer count, the Engram layers and the DSpark stage count.
    """
    num_layers = text_config["num_hidden_layers"]
    num_stages = text_config.get("num_nextn_predict_layers", 0) or 0
    engram_layer_ids = list(text_config.get("engram_layer_ids") or [])

    llm = _with_scales(llm_dense_names(num_layers, engram_layer_ids, num_stages))
    llm.extend(expert_prefixes(num_layers, num_stages).items())
    return {
        VISION: tuple((prefix, prefix) for prefix in VISION_PREFIXES),
        TEXT_ENCODER: tuple(_with_scales(TEXT_ENCODER_NAMES)),
        NGRAM: tuple(_with_scales(ngram_table_names(engram_layer_ids))),
        LLM: tuple(llm),
    }


__all__ = [
    "LLM",
    "MODULES",
    "NGRAM",
    "TEXT_ENCODER",
    "TEXT_ENCODER_NAMES",
    "VISION",
    "VISION_PREFIXES",
    "engram_connection_names",
    "expert_prefixes",
    "llm_attention_names",
    "llm_dense_names",
    "llm_layer_names",
    "module_scale_name",
    "mtp_extra_names",
    "mtp_layer_names",
    "ngram_table_names",
    "release_key_prefixes",
    "released_scale_name",
    "tied_mtp_names",
]
