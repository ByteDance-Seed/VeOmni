"""Unit tests for the LLaDA-MoE Omni module.

The batching and sample-counting behaviour is inherited from the dense module and
covered by ``tests/seed_omni/modules/llada/test_llada_omni_batching.py``; what is
specific to this family is the Expert Parallel plan and the fact that the router
term joins *after* the sample-count division rather than inside it.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.distributed._tensor import Shard

from veomni.models.seed_omni.modules.llada_moe.accelerated import LladaMoeOmniAccelerated


def test_the_expert_weights_are_sharded_over_the_ep_group():
    """EP shards the fused expert tensors on their leading (expert) axis. The fqn
    keeps the `model.` prefix because `LLaDAMoEForCausalLM` holds the backbone as
    `self.model`, unlike Qwen3-MoE whose module is the bare backbone."""
    plan = LladaMoeOmniAccelerated.get_parallel_plan(LladaMoeOmniAccelerated.__new__(LladaMoeOmniAccelerated))

    ep_plan = plan.extra_parallel_plan["ep"]
    assert ep_plan == {
        "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
    }


def test_the_router_term_joins_after_the_sample_count_division():
    """`aux_loss` is a per-token mean, so folding it into the sample-mean CE term
    before the division would rescale it with the batch composition. The module
    therefore divides first and adds `coef * aux_loss` afterwards.

    The hook itself is inherited from the dense module, so the patch targets that
    definition -- which also pins the reuse: a copy of `normalize_loss` here would
    make this test fail to find the symbol.
    """
    from veomni.models.seed_omni.modules.llada.accelerated import accelerated as mod

    module = LladaMoeOmniAccelerated.__new__(LladaMoeOmniAccelerated)
    module.config = SimpleNamespace(router_aux_loss_coef=0.5)
    module.micro_batches_num_samples = 4
    module._num_micro_batches = 2

    ce_sum = torch.tensor(8.0)
    aux_loss = torch.tensor(1.0)

    captured = {}

    def fake_global_loss(loss, num_samples, **kwargs):
        captured["ce"] = float(loss)
        captured["samples"] = num_samples
        captured["kwargs"] = kwargs
        return loss / num_samples

    original = mod.dllm_mean_global_loss
    mod.dllm_mean_global_loss = fake_global_loss
    try:
        out = module.normalize_loss(_loss=ce_sum, aux_loss=aux_loss)
    finally:
        mod.dllm_mean_global_loss = original

    # CE is divided by the sample count, then the router term is added on top,
    # divided only by the micro-batch count.
    assert captured["ce"] == 8.0
    assert captured["samples"] == 4
    assert "sum_spans_sp_group" not in captured["kwargs"]
    assert float(out["_loss"]) == 8.0 / 4 + 0.5 * 1.0 / 2
    assert "aux_loss" not in out


def test_the_module_declares_that_its_loss_spans_the_step():
    """`dllm_mean_global_loss` divides by the whole accumulation window's sample
    count, so the trainer must not apply `1 / num_micro_steps` on top."""
    assert LladaMoeOmniAccelerated.loss_is_step_mean is True
