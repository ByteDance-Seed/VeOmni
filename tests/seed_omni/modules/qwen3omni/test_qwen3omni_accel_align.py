"""Qwen3-Omni module-level alignment: the router objective under sequence parallelism.

The sibling families use an `accel_align` file to compare whole-graph losses
across SP sizes. Qwen3-Omni needs a narrower and more targeted version, for one
specific reason: it is the only family whose **router balancing loss is computed
inside a graph hook** rather than on the lm-head path, and that term is a product
of two batch-level means — so it is not linear in the rows, and it reads
per-forward state that the same hook also clears.

Three properties are pinned here, all without a checkpoint:

1. **The term is not the mean of its parts.** A direct demonstration on the shared
   loss: splitting a batch in half and averaging the halves does not reproduce the
   full-batch value. That is *why* the hook gathers rather than reducing per-rank.
2. **The gather uses the full row count.** The per-forward length must still be set
   when the loss runs; clearing it with the other per-forward fields left `None`
   reaching `.narrow`, which is the exact bug that shipped.
3. **The single-rank path is unchanged.** With `sp_size == 1` the gather is the
   identity, so the same call returns the same tensor — the baseline the gathered
   path is meant to reproduce.

What is deliberately **not** here: a full SP=2-vs-SP=1 loss and parameter
comparison. That needs the modules FSDP2-sharded, and the patched thinker's
RMSNorm op multiplies a DTensor weight by a plain tensor, which torch rejects
(`aten.mul.Tensor got mixed torch.Tensor and DTensor`). That is an op-level
limitation shared with the modelling layer rather than something this family's
test can route around, so it is recorded here instead of being worked around with
a harness that would not exercise the real path.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.training_utils import make_eager_ops_config
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY


HIDDEN = 32
NUM_EXPERTS = 4
TOP_K = 2
ROWS = 6


_TEXT_CONFIG = dict(
    num_hidden_layers=2,
    hidden_size=HIDDEN,
    intermediate_size=32,
    moe_intermediate_size=16,
    num_experts=NUM_EXPERTS,
    num_experts_per_tok=TOP_K,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=64,
    output_router_logits=True,
    router_aux_loss_coef=0.01,
    dtype="float32",
)


def _build_llm(*, device: torch.device, dtype: torch.dtype):
    ops = make_eager_ops_config()
    return (
        OMNI_ACCELERATED_MODEL_REGISTRY["qwen3omni_llm"]()(
            OMNI_CONFIG_REGISTRY["qwen3omni_llm"]()(
                text_config=_TEXT_CONFIG,
                spatial_merge_size=2,
                ops_implementation=ops,
            )
        )
        .to(device=device, dtype=dtype)
        .eval()
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_the_router_term_is_not_the_mean_of_its_per_rank_parts():
    """Why the hook gathers instead of averaging: the product form is non-linear.

    `load_balancing_loss` is `sum_e(tokens_e / N * prob_e / N) * E`. Splitting the
    rows across ranks and averaging each rank's value therefore does *not*
    reproduce the value over the combined rows — the whole reason the hook has to
    see every row before the product forms.
    """
    from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import load_balancing_loss_func

    torch.manual_seed(0)
    logits = torch.randn(ROWS, NUM_EXPERTS)
    full = float(load_balancing_loss_func((logits,), NUM_EXPERTS, TOP_K, None))

    halves = [
        float(load_balancing_loss_func((logits[start:stop],), NUM_EXPERTS, TOP_K, None))
        for start, stop in ((0, ROWS // 2), (ROWS // 2, ROWS))
    ]
    averaged = sum(halves) / len(halves)

    # Not merely "different", but different by enough that a loss averaged across
    # ranks would be visibly wrong rather than a rounding difference.
    assert abs(averaged - full) > 0.05, (full, halves, averaged)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_the_gather_keeps_its_length_until_the_loss_has_run():
    """`_sp_router_own_len` must survive the per-forward clear.

    `forward_post` clears its per-forward state after scattering the hidden
    states, and the router block runs *after* that. Clearing the router length
    with the rest left `None` reaching `.narrow(..., None)` under Ulysses — a
    `TypeError` on every SP step with the objective on. A single-device test cannot
    see it, which is what let it through.
    """
    module = _build_llm(device=torch.device("cpu"), dtype=torch.float32)
    module._pack_inputs_embeds_shape = torch.tensor([[ROWS]])
    module._conversation_carrier = [
        [
            type(
                "_Item",
                (),
                {"value": torch.zeros(ROWS, HIDDEN), "meta": {}, "is_dummy": False, "role": "user", "type": "text"},
            )()
        ]
    ]

    outputs = {
        "hidden_states": torch.randn(1, ROWS, HIDDEN),
        "router_logits": (torch.randn(ROWS, NUM_EXPERTS),),
        "router_attention_mask": None,
        "all_hidden_states": None,
    }

    with mock.patch(
        "veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.gather_outputs",
        side_effect=lambda t, **kw: t,
    ):
        with mock.patch(
            "veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state"
        ) as ps:
            ps.return_value.sp_size = 2
            ps.return_value.sp_group = object()
            # `forward_pre` stores these; the ordering bug is what cleared the
            # router one too early.
            module._sp_own_len = ROWS
            module._sp_router_own_len = ROWS
            result = module.forward_post(**outputs)

    assert "_loss" in result, "the router objective contributed nothing"
    assert module._sp_router_own_len is None, "per-forward state must still be cleared"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_the_single_rank_path_is_the_identity():
    """`sp_size == 1` must return the very same tensor, not a copy.

    The gathered path is meant to reproduce this one, so it has to be a no-op here
    — a gather that also ran at size 1 would silently change the single-device
    result while still "passing" a same-shape check.
    """
    module = _build_llm(device=torch.device("cpu"), dtype=torch.float32)
    logits = (torch.randn(ROWS, NUM_EXPERTS),)
    assert module._full_sequence_router_logits(logits) is logits
