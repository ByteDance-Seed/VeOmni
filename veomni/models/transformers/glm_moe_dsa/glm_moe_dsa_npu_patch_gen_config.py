import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.models.transformers.deepseek_v3.deepseek_v3_gpu_patch_gen_config import (
    PatchedDeepseekV3Experts,
    deepseek_v3_get_parallel_plan_patched,
    deepseek_v3_moe_forward_patched,
    deepseek_v3_topk_router_forward_patched,
)
from veomni.patchgen.patch_spec import PatchConfig

from .glm_moe_dsa_gpu_patch_gen_config import _DEEPSEEK_V3_NAME_MAP


config = PatchConfig(
    source_module="transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    target_file="patched_modeling_glm_moe_dsa_npu.py",
    description="GLM-5 with NPU replacements",
)

config.add_import("veomni.ops", names=["fused_moe_forward"])
config.add_import("veomni.utils.moe_monitor", names=["record_router_indices"])

# Surface ``CausalLMOutputWithLogProbs`` so the patched ``forward`` can
# return per-token log-probs in the unified output dataclass.
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)

# This config is much smaller than the GPU sibling: besides
# `GlmMoeDsaForCausalLM.forward` it only shares the routed-expert block and
# parallel-plan patches, so the GPU indexer / attention ports do not reach the NPU build.
# The DSA top-k selection therefore relies on upstream's `indices=` hand-off
# here; VeOmni's `flash_attention_forward` rejects that kwarg rather than
# silently running dense attention.
config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    """
)

# Same routed-expert block and parallel-plan patches as the GPU config, so GPU
# and NPU runs share one expert layout, router numerics and EP plan.
config.replace_class(
    "GlmMoeDsaExperts",
    replacement=PatchedDeepseekV3Experts,
    name_map=_DEEPSEEK_V3_NAME_MAP,
    description="Use v5 gate_up_proj expert layout with OpSlot-guarded VeOmni fused-MoE path",
)
config.override_method(
    "GlmMoeDsaTopkRouter.forward",
    replacement=deepseek_v3_topk_router_forward_patched,
    name_map=_DEEPSEEK_V3_NAME_MAP,
    description="Disable autocast around fp32 router linear for VeRL actor/rollout parity",
)
config.override_method(
    "GlmMoeDsaMoE.forward",
    replacement=deepseek_v3_moe_forward_patched,
    name_map=_DEEPSEEK_V3_NAME_MAP,
    description="Report top-k indices to the MoE load-balance monitor",
)
config.override_method(
    "GlmMoeDsaForCausalLM.get_parallel_plan",
    replacement=deepseek_v3_get_parallel_plan_patched,
    name_map=_DEEPSEEK_V3_NAME_MAP,
    description="Register GlmMoeDsa expert parallel plan for v5 generated modeling",
)


@config.override_method(
    "GlmMoeDsaForCausalLM.forward",
    description="Support fused cross entropy path in GlmMoeDsaForCausalLM.forward",
)
def glm_moe_dsa_forcausallm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    r"""
    cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
        Indices depicting the position of the input sequence tokens in the sequence. Retained in the
        signature for callers that pass it positionally; transformers 5.16 moved it into `**kwargs`.
    """
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                # fused_linear_aux path empties loss/logits slots; clear the local 3D
                # logits so output mirrors the OpSlot branch's contract.
                logits = None
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    return CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
