"""Qwen3-MoE AR backbone (no wte / lm_head)."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import (
    Qwen3MoeCheckpointTensorConverter,
    convert_qwen3_moe_fqn_to_index_mapping,
)

# Re-export the patched module's OpSlots into THIS module's namespace.
# ``build_foundation_model`` binds kernels by walking ``model_cls.__module__``
# (here = this omni wrapper module) for ``OpSlot`` objects — the real OpSlots
# live in ``patched_modeling_qwen3_moe_gpu``. Without this re-export the fused,
# EP-aware MoE kernel is never bound, so the eager experts loop runs and
# crashes under Expert Parallel (it indexes the EP-sharded experts weight with
# global expert ids). These are the same singleton OpSlots, so binding them
# here activates the fused path globally.
from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import (  # noqa: F401
    Qwen3MoeModel,
    veomni_apply_rotary_pos_emb,
    veomni_moe_experts_forward,
    veomni_rms_norm,
    veomni_swiglu_mlp,
)

from .configuration import Qwen3MoeLlmConfig
from .modulemixin import Qwen3MoeLlmModuleMixin, Qwen3MoeLlmTraceMixin


class Qwen3MoeLlm(Qwen3MoeLlmModuleMixin, Qwen3MoeLlmTraceMixin, PreTrainedModel):
    """Qwen3-MoE backbone (no wte, no lm_head).

    Multi-modal inputs are already embedded by the sibling text encoder and live
    on the ``conversation_list`` items.  :meth:`pre_forward` concatenates every
    non-dummy item's ``value`` in order into one packed bs=1 sequence.  Experts
    are stored in the v5 fused layout (``experts.gate_up_proj`` /
    ``experts.down_proj``); Expert Parallel is applied via
    :meth:`Qwen3MoeLlmModuleMixin.get_parallel_plan`.
    """

    config_class = Qwen3MoeLlmConfig
    base_model_prefix = "qwen3_moe_llm"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3MoeLlmConfig):
        super().__init__(config)
        self.config = config
        self.language_model = Qwen3MoeModel._from_config(self.config.text_config)
        self.language_model.set_input_embeddings(nn.Identity())
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        return {
            "hidden_states": outputs.last_hidden_state,
            "past_key_values": outputs.past_key_values,
        }


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
