"""Graph hooks for the Qwen3-MoE AR backbone.

The packing / scatter / generate logic is **identical** to the dense Qwen3
backbone, so :class:`Qwen3MoeLlmModuleMixin` subclasses
:class:`~veomni.models.seed_omni.modules.qwen3.llm.modulemixin.Qwen3LlmModuleMixin`
and only adds the Expert-Parallel (``ep``) plan for the fused experts.  The
metric meter mixin overrides the FLOPs estimate with the MoE (sparse-MLP) cost.
"""

from typing import Any, Dict, List

from torch.distributed._tensor import Shard

from ......distributed.parallel_plan import ParallelPlan
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ...qwen3.llm.modulemixin import Qwen3LlmModuleMixin
from .configuration import Qwen3MoeLlmConfig


class Qwen3MoeLlmModuleMixin(Qwen3LlmModuleMixin):
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


class Qwen3MoeLlmMetricMeterMixin(MetricMeterMixin):
    """Per-module training meter for the Qwen3-MoE backbone (transformer layers only)."""

    config: Qwen3MoeLlmConfig

    def metric_meter_token_lengths(self, method: str, data: Dict[str, Any]) -> List[int]:
        from veomni.utils.seqlen_pos_transform_utils import valid_seqlens_from_cu_seqlens

        cu_seq_lens_q = data.get("cu_seq_lens_q")
        if cu_seq_lens_q is None:
            return []
        return [int(s) for s in valid_seqlens_from_cu_seqlens(cu_seq_lens_q).tolist()]

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


__all__ = ["Qwen3MoeLlmModuleMixin", "Qwen3MoeLlmMetricMeterMixin"]
