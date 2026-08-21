"""Graph hooks for the Qwen3-MoE AR backbone.

The packing / scatter / generate logic is **identical** to the dense Qwen3
backbone, so :class:`VeOmniMixin` subclasses
:class:`~veomni.models.seed_omni.modules.qwen3.llm.accelerated.Qwen3LlmVeOmniMixin`
and only adds the Expert-Parallel (``ep``) plan for the fused experts.  The
metric meter mixin overrides the FLOPs estimate with the MoE (sparse-MLP) cost.
"""

from typing import List

from torch.distributed._tensor import Shard

# Re-export the patched module's OpSlots into THIS module's namespace too.
# ``build_foundation_model`` (the distributed/FSDP path — ``ModuleRuntime.
# _build_module_model``) resolves the model class for ``model_type ==
# "qwen3_moe_llm"`` via ``OMNI_ACCELERATED_MODEL_REGISTRY``, i.e. THIS
# accelerated class, and binds OpSlots by walking `sys.modules[model_cls.
# __module__]` — this file, not ``modeling.py``. ``modeling.py`` already
# re-exports these for the eager/native ``OmniModel.from_pretrained`` path
# (which resolves the class via ``OMNI_MODEL_REGISTRY`` instead), but that
# re-export alone is invisible here: without duplicating it, the fused
# EP-aware MoE kernel is never bound under FSDP/EP training or distributed
# inference, so the eager experts loop runs and crashes (it indexes the
# EP-sharded experts weight with un-translated global expert ids).
from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import (  # noqa: F401
    veomni_apply_rotary_pos_emb,
    veomni_moe_experts_forward,
    veomni_rms_norm,
    veomni_swiglu_mlp,
)

from ......distributed.parallel_plan import ParallelPlan
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ...qwen3.llm.accelerated import VeOmniMixin as Qwen3LlmVeOmniMixin
from .configuration import Qwen3MoeLlmConfig
from .modeling import Qwen3MoeLlm


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for the Qwen3-MoE backbone (transformer layers only)."""

    config: Qwen3MoeLlmConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        # Transformer layers only (no wte / lm_head — those live in text_encoder).
        # MoE MLP cost counts only the *activated* experts per token
        # (``num_experts_per_tok`` × ``moe_intermediate_size``) plus the router.
        # fwd+bwd ⇒ 6x for the linear params, 12x for the quadratic attention.
        cfg = self.config.text_config
        hidden = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", hidden // num_heads)

        # Sparse SwiGLU MLP: only top-k experts run per token (gate/up/down).
        moe_inter = cfg.moe_intermediate_size
        topk = cfg.num_experts_per_tok
        mlp_n = hidden * moe_inter * 3 * topk
        # Router gate: hidden -> num_experts logits (runs for every token).
        router_n = hidden * cfg.num_experts
        attn_linear_n = hidden * (num_heads * head_dim * 2 + num_kv_heads * head_dim * 2)
        dense_n = (mlp_n + router_n + attn_linear_n) * num_layers

        tokens = sum(seqlens)
        seqlen_sq = sum(s * s for s in seqlens)
        dense_flops = 6 * dense_n * tokens
        attn_flops = 12 * seqlen_sq * head_dim * num_heads * num_layers
        return (dense_flops + attn_flops) / 1e12


class VeOmniMixin(MeterMixin, Qwen3LlmVeOmniMixin):
    """Qwen3-MoE backbone hooks (dense backbone behaviour + Expert Parallel)."""

    def get_parallel_plan(self) -> ParallelPlan:
        # fqn is module-local: ``self.language_model`` is the bare ``Qwen3MoeModel``
        # (layers directly under it), so no ``model.`` prefix (unlike the
        # transformers ``Qwen3MoeForCausalLM`` parallel plan).
        ep_plan = {
            "language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
            "language_model.layers.*.mlp.experts.down_proj": Shard(0),
        }
        return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


class Qwen3MoeLlmAccelerated(VeOmniMixin, Qwen3MoeLlm):
    pass


__all__ = ["Qwen3MoeLlmAccelerated"]
