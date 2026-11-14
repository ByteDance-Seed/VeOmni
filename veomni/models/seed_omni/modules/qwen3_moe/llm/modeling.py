"""Qwen3-MoE AR backbone (no wte / lm_head)."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from veomni.lora.target_mapping import convert_fused_moe_lora_targets
from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import (
    Qwen3MoeCheckpointTensorConverter,
    convert_qwen3_moe_fqn_to_index_mapping,
)
from veomni.utils.device import IS_NPU_AVAILABLE

from ...base.llm_packing import SimpleArGenerationMixin
from ...module_modeling_base import PretrainedOmniModule
from .configuration import Qwen3MoeLlmConfig


if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_npu import Qwen3MoeModel
else:
    from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import Qwen3MoeModel


class Qwen3MoeLlm(SimpleArGenerationMixin, PretrainedOmniModule):
    """Qwen3-MoE backbone (no wte, no lm_head).

    Multi-modal inputs are already embedded by the sibling text encoder and live
    on the ``conversation_list`` items.  :meth:`pre_forward` concatenates every
    non-dummy item's ``value`` in order into one packed bs=1 sequence.  Experts
    are stored in the v5 fused layout (``experts.gate_up_proj`` /
    ``experts.down_proj``); Expert Parallel is applied via ``get_parallel_plan``
    on the ``VeOmniMixin`` in the sibling ``accelerated.py``.

    ``generate()`` / inference-state reset come from :class:`SimpleArGenerationMixin`
    — shared verbatim with the dense Qwen3 backbone (:class:`~...qwen3.llm.modeling.Qwen3Llm`).
    """

    config_class = Qwen3MoeLlmConfig
    base_model_prefix = "qwen3_moe_llm"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    supports_gradient_checkpointing = True
    # Same capability flags the dense Qwen3 backbone declares: the patched
    # attention takes the selected backend explicitly, so dense backends
    # (``sdpa`` / ``eager``) must not be gated out at construction.
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = Qwen3MoeModel._supports_attention_backend

    def __init__(self, config: Qwen3MoeLlmConfig):
        super().__init__(config)
        self.config = config
        self.language_model = Qwen3MoeModel._from_config(self.config.text_config)
        self.language_model.set_input_embeddings(nn.Identity())
        self._past_key_values: Any = None
        self.post_init()

    def freeze_model(self) -> None:
        """Freeze the whole backbone when ``config.freeze`` (bootstrap setups).

        Mirrors ``qwen3/llm/modeling.py``: ``VeOmniModelRuntime._freeze_model_module``
        calls this only when it exists, so without it a module override of
        ``freeze: true`` would be stored but never applied, and the backbone would
        stay trainable and enter the optimizer.
        """
        if getattr(self.config, "freeze", False):
            self.language_model.requires_grad_(False)

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Router logits are only collected when the router-balancing objective is
        # configured; passing the flag through unconditionally would make every
        # layer build the list for nothing.
        output_router_logits = self._router_aux_loss_enabled()
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        return {
            "hidden_states": outputs.last_hidden_state,
            "past_key_values": outputs.past_key_values,
            # ``None`` unless the objective is on, and never a graph output: the
            # forward hook turns it into an ``_loss`` entry instead of moving
            # per-layer logits through the DAG.
            "router_logits": outputs.router_logits if output_router_logits else None,
            "router_attention_mask": attention_mask if output_router_logits else None,
        }

    def _router_aux_loss_enabled(self) -> bool:
        """Whether the upstream router-balancing objective is configured for this module.

        Mirrors ``Qwen3MoeForCausalLM.forward``: ``output_router_logits`` defaults to
        the config field. Off by default, so the common path collects nothing.
        """
        return bool(getattr(self.config.text_config, "output_router_logits", False))


def _create_qwen3_moe_llm_checkpoint_tensor_converter(model: "Qwen3MoeLlm"):
    """Per-expert -> fused experts converter for the omni backbone.

    The split checkpoint stores experts in HF per-expert layout
    (``experts.{j}.{gate,up,down}_proj.weight``), but the model uses the v5 fused
    layout (``experts.gate_up_proj`` / ``experts.down_proj``). HF
    ``from_pretrained`` (eager inference) fuses natively, but the veomni meta-init
    / FSDP weight loader needs the converter explicitly. ``num_experts`` lives on
    ``config.text_config`` for this wrapper (not ``config.num_experts``), so we
    can't reuse the transformers factory verbatim.
    """
    return Qwen3MoeCheckpointTensorConverter(num_experts=model.config.text_config.num_experts)


# Wire the per-expert -> fused conversion onto the omni wrapper class so the
# veomni loader (build_parallelize_model meta-init / FSDP) fuses experts at load.
# Without this, FSDP-loaded experts stay at their fused-shape init (unloaded) and
# the model emits garbage, while eager (HF from_pretrained) loads fine.
Qwen3MoeLlm._create_checkpoint_tensor_converter = staticmethod(_create_qwen3_moe_llm_checkpoint_tensor_converter)
Qwen3MoeLlm._convert_fqn_to_index_mapping = staticmethod(convert_qwen3_moe_fqn_to_index_mapping)


def _convert_qwen3_moe_llm_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    """Map semantic MoE LoRA target names onto this wrapper's fused expert parameters.

    ``gate_proj`` / ``up_proj`` / ``down_proj`` name per-expert ``nn.Linear`` modules
    in an unfused model, but this backbone holds the v5 fused ``experts.gate_up_proj``
    / ``experts.down_proj``. Without this hook the LoRA resolver leaves the names as
    ordinary module targets, they match nothing, and the experts silently get no
    adapters while attention does.

    Same hook the foundation-model path already registers on ``Qwen3MoeModel``
    (``veomni/models/transformers/qwen3_moe/__init__.py``); SeedOmni builds the
    patched ``Qwen3MoeModel`` directly rather than through
    ``MODELING_REGISTRY``, so that registration never runs and this wrapper has to
    carry it. The prefixes are ``language_model``-relative, matching where the
    wrapper keeps the backbone. The resolver's own pattern check
    (``_assert_patterns_match``) then fails loudly instead of silently no-op.
    """
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "language_model.layers.*.mlp.experts.gate_up_proj",
        "language_model.layers.*.mlp.experts.down_proj",
    )


Qwen3MoeLlm._convert_lora_targets_to_parameters = staticmethod(_convert_qwen3_moe_llm_lora_targets_to_parameters)
