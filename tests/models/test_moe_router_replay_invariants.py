# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MoE Router Replay (RR) bitwise invariant tests.

These tests are a layer up from the hook-API unit tests in
``tests/utils/test_moe_router_replay.py``. They instantiate the actually
patched ``SparseMoeBlock`` of each wired family (``Qwen3MoeSparseMoeBlock``,
``Qwen3_5MoeSparseMoeBlock``) from the generated ``patched_modeling_*.py``
modules, run real forward passes (with VeOmni's Liger fused MoE experts in
the loop for Test A), and verify the two RR guarantees that the API alone
cannot:

A. **RECORD mode is bit-identical to the no-RR baseline.** The patched
   forward must produce byte-equal output whether or not a RECORD-mode
   manager is installed — RR is a passive observer, not a perturbation.

B. **REPLAY-with-native-indices reproduces the native ``(idx, w)`` pair
   bit-for-bit at the experts call site, validated against vanilla HF
   as the oracle.** Combined with a small plumbing assertion that REPLAY
   with alt indices actually substitutes, this verifies the recompute
   path (``softmax → gather → renorm → cast`` for qwen3_moe; ``gather →
   renorm → cast`` for qwen3_5_moe) is bytewise equivalent to the native
   router's internal post-topk math, independent of the indices chosen.

The capture-experts pattern in Test B replaces ``block.experts`` with a
sink that records ``(idx, w)`` and returns zeros. This isolates the
comparison to RR's actual responsibility — what tuple is fed to the
experts module — and avoids both expert-weight-layout incompatibilities
between the patched class (Liger merged ``gate_up_proj``) and vanilla HF
(separate ``gate_proj``/``up_proj``), and any potential nondeterminism in
the fused expert kernel itself.

Test A keeps Liger experts in the loop (no capture) so the RECORD
no-perturbation guarantee is verified end-to-end including the fused
kernel path.

Scope:
  - Single GPU, in-process. No torchrun / SP / FSDP.
  - Forward pass invariants only. Backward / gradient bit-equality is a
    separate (stronger) property left to a future test.
  - Toy config (``hidden_size=64, num_experts=4, top_k=2``) — sufficient
    for invariant verification without paying real-model setup cost.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type
from veomni.utils.moe_router_replay import set_active_replay


# ----------------------------------------------------------------- skip gate

# All tests in this file require CUDA + Liger fused MoE. We skip at the
# module level rather than per-test to keep the skip message clean and
# avoid spending import time on transformers when the env is not GPU.
# The skip uses ``IS_CUDA_AVAILABLE`` from ``veomni.utils.device`` rather
# than calling the torch availability check directly, so this file passes
# the ``check_device_api_usage`` CI lint (no raw device-name literals in
# non-whitelisted files).
pytestmark = pytest.mark.skipif(
    not IS_CUDA_AVAILABLE,
    reason="RR invariant tests require CUDA + Liger fused MoE experts.",
)


# Device string used for tensor placement inside the test bodies. Resolves
# to the GPU device name on a GPU host (the only env where these tests run,
# per the skip gate above); kept as a module-level constant so individual
# asserts don't repeat the lookup.
_DEVICE = get_device_type()


# ----------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _restore_active_replay():
    """Guard the module-level RR singleton between tests."""
    set_active_replay(None)
    yield
    set_active_replay(None)


# ----------------------------------------------------------------- mocks


class _IndexController:
    """Duck-typed RR manager.

    Acts as RECORD by default (captures ``top_indices`` into ``recorded`` and
    returns them unchanged). When ``replay_target`` is set, switches to
    REPLAY behavior (returns the configured target instead).
    """

    def __init__(self) -> None:
        self.recorded: torch.Tensor | None = None
        self.replay_target: torch.Tensor | None = None

    def on_router_forward(self, module, routing_scores, top_indices):
        self.recorded = top_indices.detach().clone()
        if self.replay_target is not None:
            return self.replay_target
        return top_indices


def _capture_experts_inputs():
    """Return a (sink, captured) pair.

    ``sink`` is an ``nn.Module`` whose ``forward`` matches the
    ``self.experts(h, idx, w)`` call signature used by both qwen3_moe and
    qwen3_5_moe SparseMoeBlock. Install via attribute swap
    (``block.experts = sink``); the original experts module is never
    invoked. ``sink`` MUST be an ``nn.Module`` rather than a plain
    function: ``nn.Module.__setattr__`` rejects assignment of a non-Module
    to a name already registered as a child module.
    """
    captured: dict[str, torch.Tensor] = {}

    class _ExpertsSink(nn.Module):
        def forward(self, h, idx, w):
            captured["idx"] = idx.detach().clone()
            captured["w"] = w.detach().clone()
            return torch.zeros_like(h)

    return _ExpertsSink(), captured


# ----------------------------------------------------------------- toy config builders


def _make_qwen3_moe_config():
    """Toy ``Qwen3MoeConfig`` sized for fast invariant testing."""
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    return Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        output_router_logits=False,
        router_aux_loss_coef=0.0,
        rms_norm_eps=1e-6,
    )


def _make_qwen3_5_moe_config():
    """Toy ``Qwen3_5MoeConfig`` sized for fast invariant testing."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    return Qwen3_5MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        output_router_logits=False,
        router_aux_loss_coef=0.0,
        rms_norm_eps=1e-6,
    )


# ----------------------------------------------------------------- weight init


def _init_block_deterministic(block: nn.Module, seed: int = 0) -> None:
    """Fill all parameters of ``block`` with a deterministic init.

    Used to make per-test forward output reproducible AND to allow copying
    the gate weight from a patched block to a vanilla HF block so both
    routers see identical parameters.
    """
    torch.manual_seed(seed)
    for p in block.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(p)


# ----------------------------------------------------------------- block builders


def _build_patched_qwen3_moe_block(config, device=_DEVICE, dtype=torch.bfloat16):
    from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import (
        Qwen3MoeSparseMoeBlock as PatchedQwen3MoeSparseMoeBlock,
    )

    block = PatchedQwen3MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
    _init_block_deterministic(block, seed=0)
    return block


def _build_vanilla_qwen3_moe_block(config, device=_DEVICE, dtype=torch.bfloat16):
    """Build vanilla HF ``Qwen3MoeSparseMoeBlock`` for use as the Test B oracle.

    VeOmni's qwen3_moe registration does not mutate the HF module — patches
    live in the patchgen-generated path under ``generated/`` — so importing
    from ``transformers.models.qwen3_moe`` returns the pristine HF class.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock as VanillaQwen3MoeSparseMoeBlock,
    )

    block = VanillaQwen3MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
    _init_block_deterministic(block, seed=0)
    return block


def _build_patched_qwen3_5_moe_block(config, device=_DEVICE, dtype=torch.bfloat16):
    from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
        Qwen3_5MoeSparseMoeBlock as PatchedQwen3_5MoeSparseMoeBlock,
    )

    block = PatchedQwen3_5MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
    _init_block_deterministic(block, seed=0)
    return block


def _build_vanilla_qwen3_5_moe_block(config, device=_DEVICE, dtype=torch.bfloat16):
    """qwen3_5_moe ships only the patchgen-generated path in VeOmni — vanilla
    HF import is always pristine, no unpatch dance required."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock as VanillaQwen3_5MoeSparseMoeBlock,
    )

    block = VanillaQwen3_5MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
    _init_block_deterministic(block, seed=0)
    return block


def _make_hidden_states(config, batch=2, seq=16, dtype=torch.bfloat16, seed=42):
    """Deterministic hidden_states tensor sized for the toy config."""
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    return torch.randn(
        batch,
        seq,
        config.hidden_size,
        generator=g,
        device=_DEVICE,
        dtype=dtype,
    )


def _sync_gate_weight(src: nn.Module, dst: nn.Module) -> None:
    """Copy router weights from ``src`` to ``dst`` so both produce identical
    ``(router_logits, top_indices, top_value)`` for the same input."""
    dst.gate.weight.data.copy_(src.gate.weight.data)


# =================================================================
# Test A: RECORD mode does not perturb forward (end-to-end with Liger)
# =================================================================


def test_qwen3_moe_record_mode_is_bitwise_baseline_with_liger_experts():
    config = _make_qwen3_moe_config()
    block = _build_patched_qwen3_moe_block(config)
    h = _make_hidden_states(config)

    # Baseline: no manager installed.
    out_baseline = block(h).clone()

    # Record mode: manager active, captures indices, does not substitute.
    ctrl = _IndexController()
    set_active_replay(ctrl)
    out_record = block(h).clone()
    set_active_replay(None)

    # End-to-end (Liger fused experts in loop) bit equality.
    assert torch.equal(out_baseline, out_record), (
        "RECORD mode perturbed the forward output — RR is supposed to be a passive observer, not a transform."
    )
    # Sanity: the manager actually fired (ruling out 'forward never reached the hook').
    assert ctrl.recorded is not None
    expected_shape = (h.shape[0] * h.shape[1], config.num_experts_per_tok)
    assert tuple(ctrl.recorded.shape) == expected_shape


def test_qwen3_5_moe_record_mode_is_bitwise_baseline_with_liger_experts():
    config = _make_qwen3_5_moe_config()
    block = _build_patched_qwen3_5_moe_block(config)
    h = _make_hidden_states(config)

    out_baseline = block(h).clone()

    ctrl = _IndexController()
    set_active_replay(ctrl)
    out_record = block(h).clone()
    set_active_replay(None)

    assert torch.equal(out_baseline, out_record), (
        "RECORD mode perturbed the forward output — RR is supposed to be a passive observer, not a transform."
    )
    assert ctrl.recorded is not None
    expected_shape = (h.shape[0] * h.shape[1], config.num_experts_per_tok)
    assert tuple(ctrl.recorded.shape) == expected_shape


# =================================================================
# Test B: REPLAY-native matches vanilla HF at expert input + plumbing
# =================================================================


def _capture_block_experts_inputs(block, hidden_states):
    """Run ``block(hidden_states)`` with experts replaced by a capture sink.

    Returns ``(idx, w)`` that the patched / vanilla SparseMoeBlock would have
    fed to its experts module. The original experts module is restored
    after capture so the block remains usable.
    """
    sink, captured = _capture_experts_inputs()
    original_experts = block.experts
    block.experts = sink
    try:
        block(hidden_states)
    finally:
        block.experts = original_experts
    return captured["idx"], captured["w"]


def test_qwen3_moe_replay_native_matches_vanilla_hf_at_expert_input():
    config = _make_qwen3_moe_config()
    patched = _build_patched_qwen3_moe_block(config)
    vanilla = _build_vanilla_qwen3_moe_block(config)
    _sync_gate_weight(patched, vanilla)
    h = _make_hidden_states(config)

    # Oracle: vanilla HF SparseMoeBlock produces the (idx, w) pair the native
    # router would feed to experts.
    #
    # NOTE on dtype: the patched ``Qwen3MoeTopKRouter.forward`` (introduced by
    # upstream commit 64557a8 — VeOmni's replication of HF's #715 router
    # double-softmax fix, applied BEFORE this PR rebased onto it) explicitly
    # casts ``router_top_value`` back to ``input_dtype`` (bf16) as a perf
    # optimization for the downstream fused MoE matmul. The vanilla HF
    # router in the currently installed transformers version skips this cast
    # and returns fp32. Both perform the SAME softmax/topk/renorm math —
    # the only difference is the trailing ``.to(input_dtype)``. We therefore
    # cast the vanilla oracle to the patched dtype before comparison so the
    # equality check only catches real math divergence in the recompute path
    # (softmax → gather → renorm), not this Option E perf cast.
    idx_vanilla, w_vanilla_native_dtype = _capture_block_experts_inputs(vanilla, h)

    # Sanity (Option E correctness modulo the perf cast): patched-RR-disabled
    # default path must reproduce the vanilla pair byte-for-byte after we
    # bring vanilla into the patched dtype.
    idx_patched_off, w_patched_off = _capture_block_experts_inputs(patched, h)
    w_vanilla = w_vanilla_native_dtype.to(w_patched_off.dtype)
    assert torch.equal(idx_patched_off, idx_vanilla)
    assert torch.equal(w_patched_off, w_vanilla)

    # The actual RR invariant: REPLAY with target=native_indices must put
    # the recompute path (softmax → gather → renorm → cast) on the line and
    # produce the same (idx, w) as vanilla / native byte-for-byte.
    ctrl = _IndexController()
    ctrl.replay_target = idx_vanilla.clone()
    set_active_replay(ctrl)
    idx_replay, w_replay = _capture_block_experts_inputs(patched, h)
    set_active_replay(None)

    assert torch.equal(idx_replay, idx_vanilla), "REPLAY did not return the configured target indices."
    assert torch.equal(w_replay, w_vanilla), (
        "Recompute path (softmax + gather + renorm + cast) is not "
        "bitwise-equivalent to the native router's internal post-topk math. "
        "Likely cause: dtype-cast ordering, missing/extra renorm, or "
        "softmax kernel mismatch."
    )

    # Plumbing check: REPLAY with alt indices actually substitutes (proves
    # the substitution branch is reached and not silently bypassed).
    alt = (idx_vanilla + 1) % config.num_experts
    ctrl.replay_target = alt
    set_active_replay(ctrl)
    idx_alt, _ = _capture_block_experts_inputs(patched, h)
    set_active_replay(None)
    assert torch.equal(idx_alt, alt), (
        "REPLAY with substituted target did not feed the substitute to "
        "experts — the indices-only RR control flow is broken."
    )


def test_qwen3_5_moe_replay_native_matches_vanilla_hf_at_expert_input():
    config = _make_qwen3_5_moe_config()
    patched = _build_patched_qwen3_5_moe_block(config)
    vanilla = _build_vanilla_qwen3_5_moe_block(config)
    _sync_gate_weight(patched, vanilla)
    h = _make_hidden_states(config)

    # See qwen3_moe variant for the dtype-cast rationale. For qwen3_5_moe
    # the cast is typically a no-op (vanilla HF and patched both end up in
    # fp32 because the router locally rebinds ``router_logits`` to its
    # softmax output and casts top-k values to that dtype), but applying it
    # uniformly future-proofs against either side gaining a perf cast.
    idx_vanilla, w_vanilla_native_dtype = _capture_block_experts_inputs(vanilla, h)

    idx_patched_off, w_patched_off = _capture_block_experts_inputs(patched, h)
    w_vanilla = w_vanilla_native_dtype.to(w_patched_off.dtype)
    assert torch.equal(idx_patched_off, idx_vanilla)
    assert torch.equal(w_patched_off, w_vanilla)

    ctrl = _IndexController()
    ctrl.replay_target = idx_vanilla.clone()
    set_active_replay(ctrl)
    idx_replay, w_replay = _capture_block_experts_inputs(patched, h)
    set_active_replay(None)

    assert torch.equal(idx_replay, idx_vanilla)
    # For qwen3_5_moe, the recompute uses ``router_logits`` (which is already
    # post-softmax due to the latent double-softmax quirk in upstream's
    # ``Qwen3_5MoeTopKRouter.forward``) — see the comment in
    # ``qwen3_5_moe_gpu_patch_gen_config.py``. If this assertion fails AFTER
    # an upstream fix lands, the patch must switch to the qwen3_moe form
    # (recompute softmax from raw logits).
    assert torch.equal(w_replay, w_vanilla), (
        "Recompute path (gather + renorm + cast) is not bitwise-equivalent "
        "to the native router's internal post-topk math."
    )

    alt = (idx_vanilla + 1) % config.num_experts
    ctrl.replay_target = alt
    set_active_replay(ctrl)
    idx_alt, _ = _capture_block_experts_inputs(patched, h)
    set_active_replay(None)
    assert torch.equal(idx_alt, alt)
