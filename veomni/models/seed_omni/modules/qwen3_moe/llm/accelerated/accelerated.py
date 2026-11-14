"""Graph hooks for the Qwen3-MoE AR backbone.

The packing / scatter / generate logic is **identical** to the dense Qwen3
backbone, so :class:`VeOmniMixin` subclasses
:class:`~veomni.models.seed_omni.modules.qwen3.llm.accelerated.Qwen3LlmVeOmniMixin`
and only adds the Expert-Parallel (``ep``) plan for the fused experts.  The
metric meter mixin overrides the FLOPs estimate with the MoE (sparse-MLP) cost.
"""

from typing import Any, Dict, List

import torch
from torch.distributed._tensor import Shard

from .......distributed.parallel_plan import ParallelPlan
from .......utils.device import IS_NPU_AVAILABLE
from .....mixins.metric_meter_mixin import MetricMeterMixin
from .....mixins.training_module_mixin import post_forward
from ....qwen3.llm.accelerated import VeOmniMixin as Qwen3LlmVeOmniMixin
from ..configuration import Qwen3MoeLlmConfig
from ..modeling import Qwen3MoeLlm


if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_npu import (
        load_balancing_loss_func,
    )
else:
    from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import (
        load_balancing_loss_func,
    )


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
        attn_linear_n = hidden * (num_heads * head_dim * 2 + num_kv_heads * head_dim * 2)

        # Not every layer is sparse: `Qwen3MoeDecoderLayer` builds a dense MLP when
        # the layer is in `mlp_only_layers`, or when `decoder_sparse_step` skips it
        # (`patched_modeling_qwen3_moe_gpu.py`, `Qwen3MoeDecoderLayer.__init__`).
        # Charging those layers the router + top-k cost would over-report FLOPs and
        # under-report MFU for any config that uses those two knobs.
        moe_inter = cfg.moe_intermediate_size
        topk = cfg.num_experts_per_tok
        sparse_mlp_n = hidden * moe_inter * 3 * topk + hidden * cfg.num_experts
        dense_mlp_n = hidden * cfg.intermediate_size * 3

        def _is_sparse(layer_idx: int) -> bool:
            return (layer_idx not in cfg.mlp_only_layers) and (
                cfg.num_experts > 0 and (layer_idx + 1) % cfg.decoder_sparse_step == 0
            )

        sparse_layers = sum(1 for i in range(num_layers) if _is_sparse(i))
        dense_layers = num_layers - sparse_layers
        dense_n = (sparse_mlp_n * sparse_layers + dense_mlp_n * dense_layers) + attn_linear_n * num_layers

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

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> Dict[str, Any]:
        """Dense scatter, plus the router-balancing loss when it is configured.

        ``OmniModel.forward`` sums every node's ``_loss``, so a mid-graph module
        contributes by returning one — this is not the graph's *only* loss, the
        decode head still produces its own. The aux term is only built when
        ``output_router_logits`` is on; otherwise this is exactly the dense hook.
        """
        router_logits = outputs.pop("router_logits", None)
        router_mask = outputs.pop("router_attention_mask", None)
        result = super().forward_post(**outputs)

        if router_logits is not None:
            cfg = self.config.text_config
            aux_loss = load_balancing_loss_func(
                router_logits,
                cfg.num_experts,
                cfg.num_experts_per_tok,
                router_mask,
            )
            if isinstance(aux_loss, torch.Tensor):
                # ``router_aux_loss_coef`` is applied here rather than inside the
                # loss so a zero coefficient still reports the raw term.
                coef = getattr(cfg, "router_aux_loss_coef", 0.0)
                result["_loss"] = aux_loss.to(device=result["conversation_list"][0][0].value.device) * coef
        return result


class Qwen3MoeLlmAccelerated(VeOmniMixin, Qwen3MoeLlm):
    pass


__all__ = ["Qwen3MoeLlmAccelerated"]
