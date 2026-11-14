"""How a Qwen3-MoE checkpoint splits into SeedOmni modules.

Registered under ``OMNI_HF_LAYOUT_REGISTRY["qwen3_moe"]``. Point
``model.model_path`` at an upstream Qwen3-MoE checkpoint to train / infer from it
directly, or split it offline with ``scripts/seed_omni/convert_model.py``. Graphs
come from ``configs/seed_omni/Qwen/moe/train/`` (``train/`` DAG,
``infer/`` FSM).

Modules::

    qwen3_text_encoder   # model.embed_tokens (+ lm_head if untied) + tokenizer
    qwen3_moe_llm        # the rest of model.* (decoder layers + fused experts + final norm)

The vocabulary half is MoE-agnostic, so it reuses the dense ``qwen3_text_encoder``
module and its config. This layout is the single description of the split: the
offline `scripts/seed_omni/convert_model.py` path has no family converter for
``qwen3_moe``, so it calls ``convert_with_hf_layout`` and writes out exactly these
modules with exactly this key map.

An upstream release stores experts **per expert** on disk
(``model.layers.N.mlp.experts.{j}.{gate,up,down}_proj.weight``). The layout
renames the keys onto the backbone's module-local names and the converter
attached to ``Qwen3MoeLlm`` fuses them into ``experts.gate_up_proj`` /
``experts.down_proj`` as they stream.

Export runs that mapping backwards: :class:`Qwen3MoeHFLayout` overrides
``extract_module_save_state`` and splits the fused tensors into
``experts.{j}.{gate,up,down}_proj.weight`` again, so an upstream-root run writes an
``hf_ckpt/`` in the source's own layout. The same hook shape
``bagel/hf_layout.py`` uses for its fused QKV.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import torch

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from ..qwen3.text_encoder.configuration import Qwen3TextEncoderConfig
from .llm.configuration import Qwen3MoeLlmConfig


def _text_encoder_config(hf_config: Any) -> Qwen3TextEncoderConfig:
    return Qwen3TextEncoderConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        lm_head_bias=False,
    )


def _llm_config(hf_config: Any) -> Qwen3MoeLlmConfig:
    return Qwen3MoeLlmConfig(text_config=hf_config.to_dict())


def _tokenizer(model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(model_path)}


class Qwen3MoeHFLayout(OmniHFLayout):
    """The MoE layout, plus the reverse of the per-expert fuse for export.

    Loading goes forward: the upstream file stores ``experts.{j}.{gate,up,down}_proj``
    and ``Qwen3MoeLlm``'s converter stacks them into ``experts.gate_up_proj`` /
    ``experts.down_proj``. Export has to run the same mapping backwards, or the
    fused tensor reaches :func:`save_hf_source_checkpoint` with no source key to
    land on. Same shape as ``bagel/hf_layout.py``: override
    :meth:`OmniHFLayout.extract_module_save_state` and hand back the split keys.
    """

    def extract_module_save_state(
        self,
        module_name: str,
        model: torch.nn.Module,
        parallel_state: Any = None,
    ) -> Mapping[str, torch.Tensor]:
        """Split fused ``experts.gate_up_proj`` / ``down_proj`` back to per-expert keys."""
        state = super().extract_module_save_state(module_name, model, parallel_state)
        if module_name != "qwen3_moe_llm":
            return state
        return _split_fused_experts(state)


def _split_fused_experts(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``experts.{gate_up,down}_proj`` → ``experts.{j}.{gate,up,down}_proj.weight``.

    The inverse of ``PerExpertFusedCheckpointTensorConverter``: it stacked
    per-expert tensors on dim 0 and concatenated gate with up on dim 1, so
    ``gate_up_proj[e]`` is ``[gate_e; up_e]`` along dim 0.
    """
    out: dict[str, torch.Tensor] = {}
    fused_pattern = re.compile(r"^(?P<prefix>.+\.mlp)\.experts\.(?P<proj>gate_up_proj|down_proj)$")
    for name, tensor in state.items():
        match = fused_pattern.match(name)
        if match is None:
            out[name] = tensor
            continue
        prefix, proj = match.group("prefix"), match.group("proj")
        num_experts = tensor.shape[0]
        if proj == "down_proj":
            for expert in range(num_experts):
                out[f"{prefix}.experts.{expert}.down_proj.weight"] = tensor[expert].clone()
            continue
        # ``gate_up_proj`` is ``[E, 2 * moe_intermediate, hidden]``: gate then up.
        gate, up = tensor.chunk(2, dim=1)
        for expert in range(num_experts):
            out[f"{prefix}.experts.{expert}.gate_proj.weight"] = gate[expert].clone()
            out[f"{prefix}.experts.{expert}.up_proj.weight"] = up[expert].clone()
    return out


QWEN3_MOE_HF_LAYOUT = Qwen3MoeHFLayout(
    modules={
        "qwen3_text_encoder": OmniHFModuleLayout(
            key_prefixes=(("model.embed_tokens.", "embed_tokens."), ("lm_head.", "lm_head.")),
            build_config=_text_encoder_config,
            build_assets=_tokenizer,
        ),
        "qwen3_moe_llm": OmniHFModuleLayout(
            key_prefixes=(("model.", "language_model."),),
            build_config=_llm_config,
        ),
    },
    # A tied release may still store ``lm_head.weight`` beside
    # ``model.embed_tokens.weight``.
    tied_source_keys={"lm_head.weight": "model.embed_tokens.weight"},
    graph_dir="configs/seed_omni/Qwen/moe/train",
    training_graphs="graph_train.yaml",
    generation_graphs={"infer_text": "graph_infer.yaml"},
    infer_type="infer_text",
)


@OMNI_HF_LAYOUT_REGISTRY.register("qwen3_moe")
def _register_qwen3_moe_hf_layout(source_path: str | None = None) -> OmniHFLayout:
    return QWEN3_MOE_HF_LAYOUT


__all__ = ["QWEN3_MOE_HF_LAYOUT"]
