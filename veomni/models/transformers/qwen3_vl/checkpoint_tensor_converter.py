"""
Runtime checkpoint tensor converter for Qwen3-VL models.

Converts the HuggingFace attention layout (three separate ``q_proj`` /
``k_proj`` / ``v_proj`` ``nn.Linear`` modules) to VeOmni's fused layout
(a single ``FusedQKVLinear`` whose row-concatenated ``weight`` lives
under ``<prefix>.qkv_proj.weight``).  Mirrors the Qwen3-MoE converter
pattern (``qwen3_moe/checkpoint_tensor_converter.py``).

    HF checkpoint layout (per-attention-layer):
        <prefix>.self_attn.q_proj.weight  [n_q  * head_dim, hidden]
        <prefix>.self_attn.k_proj.weight  [n_kv * head_dim, hidden]
        <prefix>.self_attn.v_proj.weight  [n_kv * head_dim, hidden]
        (+ matching .bias keys when ``config.attention_bias=True``)

    Target VeOmni v5 fused layout:
        <prefix>.self_attn.qkv_proj.weight  [(n_q + 2*n_kv) * head_dim, hidden]
        (+ matching .bias key)
"""

import re
from typing import Dict, List, Optional

import torch

from ....utils import logging
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


logger = logging.get_logger(__name__)

# Matches HF attention projection keys, e.g.
#   model.language_model.layers.0.self_attn.q_proj.weight
#   model.language_model.layers.0.self_attn.q_proj.bias
_QKV_PATTERN = re.compile(r"^(?P<prefix>.+\.self_attn)\.(?P<proj>q_proj|k_proj|v_proj)\.(?P<kind>weight|bias)$")

# Concatenation order — must match FusedQKVLinear._q_out / _kv_out slicing:
#   weight[:n_q*hd]                 -> q
#   weight[n_q*hd : (n_q+n_kv)*hd]  -> k
#   weight[(n_q+n_kv)*hd :]         -> v
_QKV_ORDER = ("q_proj", "k_proj", "v_proj")


class Qwen3VLAttentionCheckpointTensorConverter:
    """Stream three per-projection tensors per attention layer and emit the fused one.

    For every ``(prefix, kind)`` pair (where ``kind`` is ``"weight"`` or
    ``"bias"``) we buffer at most three tensors keyed by ``proj``. Once all
    three projections arrive, we ``torch.cat([q, k, v], dim=0)`` and emit
    a single ``ConvertedCheckpointTensor`` whose name is
    ``f"{prefix}.qkv_proj.{kind}"``.

    The converter is stateless across runs but maintains per-load buffering;
    the loader (``module_utils.py``) instantiates one per model.
    """

    def __init__(self) -> None:
        # {(prefix, kind): {proj_name: tensor}}
        self._buffer: Dict[tuple, Dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return bool(_QKV_PATTERN.match(name))

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        match = _QKV_PATTERN.match(name)
        if not match:
            return None

        prefix = match.group("prefix")
        proj = match.group("proj")
        kind = match.group("kind")
        buf_key = (prefix, kind)

        slot = self._buffer.setdefault(buf_key, {})
        if proj in slot:
            raise RuntimeError(
                f"Qwen3VL checkpoint converter: duplicate {proj}.{kind} for prefix '{prefix}' "
                "(checkpoint contains the same key twice — refusing to silently overwrite)"
            )
        slot[proj] = tensor

        if len(slot) < len(_QKV_ORDER):
            return None

        merged = torch.cat([slot[p] for p in _QKV_ORDER], dim=0)
        del self._buffer[buf_key]
        return ConvertedCheckpointTensor(f"{prefix}.qkv_proj.{kind}", merged)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        """Validate that all per-attention buffers were flushed.

        Any leftover entry means a checkpoint shard never delivered all three
        of ``q/k/v_proj`` for some attention layer — a corrupted or partial
        checkpoint that would silently produce missing parameters.
        """
        if not self._buffer:
            return []
        unflushed = {f"{prefix}({kind})": sorted(slot.keys()) for (prefix, kind), slot in self._buffer.items()}
        raise RuntimeError(
            f"Qwen3VL checkpoint converter: incomplete checkpoint detected. "
            f"Unflushed QKV buffers (missing at least one of {_QKV_ORDER}): {unflushed}"
        )


def create_qwen3_vl_checkpoint_tensor_converter(model):
    """Factory registered on model classes via ``_create_checkpoint_tensor_converter``.

    Takes the model only to keep the call-site shape uniform with peers
    (Qwen3MoE / Qwen3VLMoe converters do consult ``model.config``); QKV
    fusion does not need any config-derived sizes because the converter
    cats along ``dim=0`` and shape correctness is implied by the source
    Linears in the HF checkpoint.
    """
    del model  # unused
    return Qwen3VLAttentionCheckpointTensorConverter()
