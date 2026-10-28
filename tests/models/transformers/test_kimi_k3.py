# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Kimi K3 multimodal packing tests: merge label alignment and varlen metadata.

``KimiK3ForConditionalGeneration.forward`` undoes the SP collator's label
pre-shift before the multimodal merge, rebuilds segment-aware position_ids
after the placeholder expansion, and publishes the refreshed FlashAttention
cu_seq_lens kwargs (plus the ``cu_seqlens`` alias the KDA layers read) to the
inner model. These tests pin both halves without a multi-process SP group:

* the merge scatter semantics are checked directly -- labels sitting on a
  media placeholder slot are dropped and the expanded vision span is
  IGNORE-filled, which is exactly why a pre-shifted label stream must be
  un-shifted before the merge;
* the packed varlen metadata is checked end-to-end through a spy on the inner
  ``KimiLinearModel`` call, on the single-rank (non-SP) packed path where the
  rebuild is not SP-gated;
* a two-rank ulysses (gloo) run drives the real SP data path end to end --
  the collator's per-rank sequence shards (pixel data replicated), the
  gather / merge / re-slice cycle, the KDA a2a head-shard route -- and pins
  both the published slice-pad segment and loss/gradient parity with the
  single-rank reference.
"""

from __future__ import annotations

import math
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.tiny_configs import tiny_kimi_k3_config as _tiny_config
from veomni.arguments import AcceleratorConfig
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state_from_config
from veomni.models.auto import build_foundation_model
from veomni.ops.registry import VeomniOp


def _build_model(ops_cfg=None):
    cfg = ops_cfg if ops_cfg is not None else eager_ops_config()
    with ops_config_scope(None):
        return build_foundation_model(
            _tiny_config(),
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=cfg,
        )


@pytest.fixture
def dense_packed_debug(monkeypatch):
    """Explicitly opt into the documented dense-attention packed debug path.

    The packed forward rejects the dense eager/sdpa attention when packed
    varlen metadata is present; these tests deliberately exercise the
    packing machinery through the debug-only dense path because no varlen
    attention kernel runs on CPU.
    """
    monkeypatch.setenv("K3_ALLOW_DENSE_PACKED", "1")


def _packed_batch(model):
    """One packed row of two samples with a mid-sample image.

    Pre-merge layout ``[T0 T1 PH T2 T3 T4][B0 B1 B2 B3]``; the single image
    (grid 1x4x4, 2x2 tpool merge) expands the placeholder to 4 merged tokens,
    so the merged row is ``[T0 T1 V0 V1 V2 V3 T2 T3 T4][B0 B1 B2 B3]`` (13).
    Sample B starts at merged coordinate 9.
    """
    ignore = model.config.ignore_index
    ph = model.config.media_placeholder_token_id
    input_ids = torch.tensor([[10, 11, ph, 13, 14, 15, 20, 21, 22, 23]])
    labels = torch.tensor([[ignore, 101, ignore, 103, 104, 105, ignore, 203, 204, 205]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    grid_thws = torch.tensor([[1, 4, 4]])
    pixel_values = torch.randn(16, 3, 14, 14)
    return input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values


def _single_image_batch(model):
    """One un-packed sample containing one image.

    Pre-merge layout ``[T0 T1 PH T2 T3]``; the image (grid 1x4x4,
    2x2 tpool merge) expands the placeholder to 4 merged tokens, so
    the merged row is ``[T0 T1 V0 V1 V2 V3 T2 T3]`` (8) -- a single
    segment, like the collator emits for an un-packed sequence.
    """
    ignore = model.config.ignore_index
    ph = model.config.media_placeholder_token_id
    input_ids = torch.tensor([[10, 11, ph, 13, 14]])
    labels = torch.tensor([[ignore, 101, ignore, 103, 104]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    grid_thws = torch.tensor([[1, 4, 4]])
    pixel_values = torch.randn(16, 3, 14, 14)
    return input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values


def _spy_on_inner_model(model, recorded):
    inner_forward = model.language_model.model.forward

    def _spy(*args, **kwargs):
        recorded.update(kwargs)
        return inner_forward(*args, **kwargs)

    model.language_model.model.forward = _spy
    return inner_forward


def test_merge_scatters_raw_labels_and_maps_boundaries():
    model = _build_model()
    model.eval()
    input_ids, labels, _, attention_mask, _, _ = _packed_batch(model)
    ignore = model.config.ignore_index
    hidden = model.config.text_config.hidden_size
    inputs_embeds = torch.zeros(1, input_ids.shape[1], hidden)
    image_features = [torch.randn(4, hidden)]

    with torch.no_grad():
        _, _, merged_labels, _, first_positions = model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels
        )

    # Raw labels scatter onto text slots; the expanded vision span (merged
    # coords 2..5) is IGNORE-filled.
    expected_labels = torch.tensor(
        [[ignore, 101, ignore, ignore, ignore, ignore, 103, 104, 105, ignore, 203, 204, 205]]
    )
    assert torch.equal(merged_labels, expected_labels)
    # Text slots map to their own coordinate; the placeholder maps to the
    # FIRST slot of its expanded span (2), and T2 lands after the span (6).
    assert first_positions.tolist() == [[0, 1, 2, 6, 7, 8, 9, 10, 11, 12]]
    # A single causal shift over the merged labels supervises the first
    # post-image text token (T2, label 103) from the last vision position.
    shifted = torch.nn.functional.pad(expected_labels, (0, 1), value=ignore)[..., 1:]
    assert shifted.tolist() == [[101, ignore, ignore, ignore, ignore, 103, 104, 105, ignore, 203, 204, 205, ignore]]


def test_merge_drops_labels_sitting_on_placeholder_slots():
    """Why the forward un-shifts before merging: a pre-shifted label stream
    lands its first post-image supervision (103) on the placeholder slot,
    which the merge scatters away -- the supervision is stranded."""
    model = _build_model()
    model.eval()
    input_ids, _, _, attention_mask, _, _ = _packed_batch(model)
    ignore = model.config.ignore_index
    hidden = model.config.text_config.hidden_size
    inputs_embeds = torch.zeros(1, input_ids.shape[1], hidden)
    image_features = [torch.randn(4, hidden)]
    # labels[..., 1:] + tail pad (the SP collator's pre-shift of the raw labels)
    pre_shifted = torch.tensor([[101, ignore, 103, 104, 105, ignore, 203, 204, 205, ignore]])

    with torch.no_grad():
        _, _, merged_labels, _, _ = model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, pre_shifted
        )

    assert 103 not in merged_labels.tolist()[0]


def test_packed_forward_publishes_varlen_metadata(dense_packed_debug):
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values = _packed_batch(model)

    recorded = {}
    inner_forward = _spy_on_inner_model(model, recorded)
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
            )
    finally:
        model.language_model.model.forward = inner_forward

    assert out.loss is not None and torch.isfinite(out.loss)
    # Packed boundaries survive the merge: sample B starts at merged coord 9.
    expected_cu = torch.tensor([0, 9, 13], dtype=torch.int32)
    assert torch.equal(recorded["cu_seq_lens_q"], expected_cu)
    assert torch.equal(recorded["cu_seq_lens_k"], expected_cu)
    assert torch.equal(recorded["cu_seqlens"], expected_cu)
    # Segment-aware position_ids rebuilt after the placeholder expansion.
    expected_pos = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3]])
    assert torch.equal(recorded["position_ids"], expected_pos)


def test_packed_forward_rejects_dense_attention():
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values = _packed_batch(model)

    # The packed forward rebuilds the varlen metadata from the packed batch
    # and must reject the dense eager attention instead of silently leaking
    # across packed samples (K3_ALLOW_DENSE_PACKED is unset here).
    with torch.no_grad():
        with pytest.raises(ValueError, match="cannot run packed batches"):
            model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
            )


def test_kda_rejects_cached_single_token_decode():
    model = _build_model()
    model.eval()
    attn = model.language_model.model.layers[0].self_attn
    hidden = torch.randn(1, 1, model.config.text_config.hidden_size)

    # Cached single-token decode selects the fused_recurrent KDA branch,
    # which is a stub in this build: it must be rejected up front instead of
    # crashing inside the kernel shim mid-forward.
    with torch.no_grad():
        with pytest.raises(NotImplementedError, match="cached single-token generation"):
            attn(hidden_states=hidden, cache_params=object())


def test_packed_forward_honors_collator_all_ones_flag(dense_packed_debug):
    """The collator precomputes the all-ones check host-side and publishes it
    as ``attention_mask_is_all_ones``; when present it gates the packed-varlen
    rebuild without the device-side ``.all()`` sync and is authoritative over
    the mask's own check (False keeps the attention-mask path)."""
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values = _packed_batch(model)

    recorded = {}
    inner_forward = _spy_on_inner_model(model, recorded)
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
                attention_mask_is_all_ones=True,
            )
            assert out.loss is not None and torch.isfinite(out.loss)
            assert "cu_seq_lens_q" in recorded

            recorded.clear()
            out = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
                attention_mask_is_all_ones=False,
            )
    finally:
        model.language_model.model.forward = inner_forward

    assert out.loss is not None and torch.isfinite(out.loss)
    assert "cu_seq_lens_q" not in recorded


def test_text_only_forward_forwards_collator_kwargs():
    """Text-only batches keep the collator's cu_seq_lens (no expansion ran)."""
    model = _build_model()
    model.eval()
    ignore = model.config.ignore_index
    input_ids = torch.tensor([[5, 6, 7, 8]])
    cu = torch.tensor([0, 4], dtype=torch.int32)

    recorded = {}
    inner_forward = _spy_on_inner_model(model, recorded)
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                position_ids=torch.tensor([[0, 1, 2, 3]]),
                labels=torch.tensor([[ignore, 61, 62, ignore]]),
                use_cache=False,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=4,
                max_length_k=4,
                linear_attn_cu_seq_lens_q=cu,
            )
    finally:
        model.language_model.model.forward = inner_forward

    assert out.loss is not None and torch.isfinite(out.loss)
    assert torch.equal(recorded["cu_seq_lens_q"], cu)
    assert torch.equal(recorded["cu_seq_lens_k"], cu)
    assert torch.equal(recorded["cu_seqlens"], cu)
    assert recorded["max_length_q"] == 4


def test_single_sample_image_merge_metadata_runs_dense():
    """Single-segment varlen metadata must not trip the dense-attention guard.

    The collator emits ``cu_seq_lens_q`` = [0, length] for un-packed
    sequences too, and the image-merge path rebuilds the same
    single-segment shape over the expanded row; a dense causal mask is
    exact for one contiguous sequence, so eager attention runs without
    the K3_ALLOW_DENSE_PACKED escape.
    """
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values = _single_image_batch(model)
    cu = torch.tensor([0, input_ids.shape[1]], dtype=torch.int32)

    recorded = {}
    inner_forward = _spy_on_inner_model(model, recorded)
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=5,
                max_length_k=5,
                linear_attn_cu_seq_lens_q=cu,
            )
    finally:
        model.language_model.model.forward = inner_forward

    assert out.loss is not None and torch.isfinite(out.loss)
    # The merge rebuilt single-segment metadata over the expanded row:
    # [T0 T1 PH T2 T3] -> [T0 T1 V0 V1 V2 V3 T2 T3] (8), boundary at 0.
    assert recorded["cu_seq_lens_q"].tolist() == [0, 8]


def test_packed_forward_coalesces_collator_tail_pads(dense_packed_debug):
    """Tail pads (SP / pad-to-length) arrive as per-token ``position_ids == 0``
    boundaries; the rebuilt cu_seq_lens must coalesce them into one segment
    exactly like the collator does, using the collator's published
    ``tail_padding_length`` -- not a trailing-zeros heuristic, which would
    also swallow a genuine 1-token final sample."""
    model = _build_model()
    model.eval()
    input_ids, labels, _, attention_mask, grid_thws, pixel_values = _packed_batch(model)
    ignore = model.config.ignore_index
    pad = model.config.pad_token_id
    # [T0 T1 PH T2 T3 T4][B0 B1 B2 B3][P P]: append two SP tail pads.
    input_ids = torch.cat([input_ids, torch.tensor([[pad, pad]])], dim=1)
    labels = torch.cat([labels, torch.full((1, 2), ignore, dtype=labels.dtype)], dim=1)
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 0]])
    attention_mask = torch.ones_like(input_ids)
    tail_padding_length = torch.tensor(2, dtype=torch.int32)

    recorded = {}
    inner_forward = _spy_on_inner_model(model, recorded)
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
                tail_padding_length=tail_padding_length,
            )
    finally:
        model.language_model.model.forward = inner_forward

    assert out.loss is not None and torch.isfinite(out.loss)
    # Merged row is 15 slots; the two pad tokens form ONE independent 2-token
    # tail segment (13..15) instead of two 1-token segments -- an independent
    # segment, not merged into sample B's, matching the collator-side helper.
    expected_cu = torch.tensor([0, 9, 13, 15], dtype=torch.int32)
    assert torch.equal(recorded["cu_seq_lens_q"], expected_cu)
    assert torch.equal(recorded["cu_seqlens"], expected_cu)
    # The real samples keep their boundaries; each pad token is its own
    # 1-token segment in position_ids.
    expected_pos = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 0, 0]])
    assert torch.equal(recorded["position_ids"], expected_pos)


def test_kda_module_matches_source_layout():
    """The patched KDA ``__init__`` is a full-body override (patchgen has no
    init-append mechanism); pin that it still constructs the same parameters
    as the vendored source module so silent source drift is caught."""
    from veomni.models.transformers.kimi_k3.source.modeling_kimi_k3 import (
        KimiDeltaAttention as SourceKDA,
    )

    model = _build_model()
    patched_cls = type(model.language_model.model.layers[1].self_attn)
    assert patched_cls is not SourceKDA
    config = model.config.text_config
    with ops_config_scope(eager_ops_config()):
        source = SourceKDA(config, layer_idx=1)
        patched = patched_cls(config, layer_idx=1)
    source_state = source.state_dict()
    patched_state = patched.state_dict()
    assert set(source_state) == set(patched_state)
    for key, tensor in source_state.items():
        assert tensor.shape == patched_state[key].shape, key


def test_moe_gate_router_is_initialized():
    """The router weight and e_score_correction_bias survive to_empty re-init."""
    model = _build_model()
    gate = model.language_model.model.layers[1].block_sparse_moe.gate
    assert torch.equal(gate.e_score_correction_bias, torch.zeros_like(gate.e_score_correction_bias))
    assert torch.isfinite(gate.weight).all()
    assert gate.weight.abs().sum() > 0


def test_chunk_kda_handles_are_instance_local():
    model = _build_model()
    attn_1 = model.language_model.model.layers[1].self_attn
    attn_2 = model.language_model.model.layers[2].self_attn
    handle = attn_1.veomni_chunk_kda
    assert isinstance(handle, VeomniOp)
    assert handle.op == "chunk_kda"
    assert handle.impl == "eager"
    # VeomniOp handles are interned per (op, variant, impl) triple, but the
    # binding is per-instance: rebinding one layer's handle must not leak into
    # another layer (the pre-fix code swapped a shared module-level default).
    sentinel = object()
    attn_1.veomni_chunk_kda = sentinel
    try:
        assert attn_2.veomni_chunk_kda is handle
    finally:
        attn_1.veomni_chunk_kda = handle


# ---------------------------------------------------------------------------
# Two-rank ulysses sequence parallelism
# ---------------------------------------------------------------------------

_SP2_NPROC = 2


def _sp2_batch(model, *, pre_shifted):
    """The full packed row for the single-rank reference (deterministic pixels)."""
    ignore = model.config.ignore_index
    ph = model.config.media_placeholder_token_id
    input_ids = torch.tensor([[10, 11, ph, 13, 14, 15, 20, 21, 22, 23]])
    labels = torch.tensor([[ignore, 101, ignore, 103, 104, 105, ignore, 203, 204, 205]])
    if pre_shifted:
        labels = torch.nn.functional.pad(labels, (0, 1), value=ignore)[..., 1:]
    pixel_values = torch.arange(16 * 3 * 14 * 14, dtype=torch.float32).div_(1000).reshape(16, 3, 14, 14)
    return (
        input_ids,
        labels,
        torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3]]),
        torch.ones_like(input_ids),
        torch.tensor([[1, 4, 4]]),
        pixel_values,
    )


def _sp2_shard_batch(model, rank):
    """This rank's shard under the SP collator contract.

    The collator slices the pre-merge row evenly across ranks (the pre-merge
    length is kept sp-divisible; pixel data is replicated) and pre-shifts
    labels on the FULL row before slicing. The forward gathers the shards
    back, un-shifts, runs the multimodal merge on the full sequence, then
    slices the merged embeds to the local shard again -- the slice-pad
    handling under test lives in that second slicing.
    """
    ignore = model.config.ignore_index
    ph = model.config.media_placeholder_token_id
    input_ids = torch.tensor([[10, 11, ph, 13, 14, 15, 20, 21, 22, 23]])
    labels = torch.tensor([[ignore, 101, ignore, 103, 104, 105, ignore, 203, 204, 205]])
    pre_shifted = torch.nn.functional.pad(labels, (0, 1), value=ignore)[..., 1:]
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3]])
    shard = slice(rank * 5, (rank + 1) * 5)
    pixel_values = torch.arange(16 * 3 * 14 * 14, dtype=torch.float32).div_(1000).reshape(16, 3, 14, 14)
    return (
        input_ids[:, shard].contiguous(),
        pre_shifted[:, shard].contiguous(),
        position_ids[:, shard].contiguous(),
        torch.ones(1, 5, dtype=torch.long),
        torch.tensor([[1, 4, 4]]),
        pixel_values,
    )


def _sp2_worker(rank, rendezvous, result_dir, seed):
    os.environ["WORLD_SIZE"] = str(_SP2_NPROC)
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=_SP2_NPROC,
        timeout=timedelta(minutes=5),
    )
    try:
        init_parallel_state_from_config(AcceleratorConfig(ulysses_size=_SP2_NPROC), name=None)
        assert get_parallel_state().sp_enabled

        torch.manual_seed(seed)
        model = _build_model()
        model.train()
        input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values = _sp2_shard_batch(model, rank)

        recorded = {}
        inner_forward = _spy_on_inner_model(model, recorded)
        try:
            out = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
            )
        finally:
            model.language_model.model.forward = inner_forward

        # The merged row is 13 (odd): slice_input_tensor pads to 14 so both
        # shards split evenly, and the pad must be covered by the last cu
        # segment so every varlen consumer sees cu[-1] == T. Without it the
        # KDA conv / chunk_kda drop the pad token and the shapes diverge (the
        # NPU kernel asserts outright). The pad EXTENDS the last segment
        # instead of forming a new boundary: two adjacent trailing 1-token
        # segments (collator tail pad + slice pad) make the AscendC chunk_kda
        # kernel return non-finite values for earlier real segments.
        expected_cu = torch.tensor([0, 9, 14], dtype=torch.int32)
        assert torch.equal(recorded["cu_seq_lens_q"], expected_cu)
        assert torch.equal(recorded["cu_seq_lens_k"], expected_cu)
        assert torch.equal(recorded["cu_seqlens"], expected_cu)
        # Each rank holds half of the padded row; rank 1's trailing slot is
        # the slice pad.
        assert recorded["inputs_embeds"].shape[1] == 7
        assert out.loss is not None and torch.isfinite(out.loss)

        out.loss.backward()
        grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
        for key, tensor in grads.items():
            assert torch.isfinite(tensor).all(), key

        torch.save({"loss": float(out.loss.detach()), "grads": grads}, os.path.join(result_dir, f"sp2_rank{rank}.pt"))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=_SP2_NPROC))
def test_sp2_packed_forward_matches_single_rank(tmp_path, dense_packed_debug):
    """End-to-end SP(2) forward/backward over an odd merged row.

    The regression this pins: a merged length that is odd makes
    slice_input_tensor pad the row so the shards split evenly, and the pad
    used to reach the KDA a2a path as a token beyond cu_seqlens[-1] (kernel
    shape assert on NPU; on CPU the eager conv / chunk_kda reference drops
    the token and the shapes diverge). The fix publishes the slice pad as one
    final cu segment.

    Everything runs on plain eager ops: the KDA a2a head-shard route is
    gated on SP + ``K3_KDA_HEAD_SHARD`` (default on), not on the attention
    implementation; only the MLA layers pick their route from the impl and
    take the legacy full-sequence gather under eager. Beyond running clean,
    the SP run must reproduce the single-rank loss (the SP loss reducer
    reports the global mean on every rank) and reconstruct the single-rank
    gradients when the per-rank gradients are summed and rescaled by the sp
    world size (the reducer's backward compensates for ranks only seeing
    their own shard's tokens). The trailing pad cannot perturb the real
    tokens: it sits in its own cu segment and causal attention never looks
    forward.
    """
    seed = 1234
    torch.manual_seed(seed)
    model = _build_model()
    model.train()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values = _sp2_batch(model, pre_shifted=False)
    out = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        grid_thws=grid_thws,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        use_cache=False,
    )
    ref_loss = float(out.loss.detach())
    out.loss.backward()
    ref_grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}

    mp.spawn(
        _sp2_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path), seed),
        nprocs=_SP2_NPROC,
    )

    shards = [torch.load(tmp_path / f"sp2_rank{rank}.pt") for rank in range(_SP2_NPROC)]
    for rank, shard in enumerate(shards):
        assert math.isclose(shard["loss"], ref_loss, rel_tol=1e-5, abs_tol=1e-5), rank
    for key, ref in ref_grads.items():
        summed = shards[0]["grads"][key] + shards[1]["grads"][key]
        # fp32 reduction orders differ (per-shard CE means, a2a head exchange),
        # so the comparison carries a small absolute floor.
        torch.testing.assert_close(summed, _SP2_NPROC * ref, rtol=1e-3, atol=1e-3, msg=key)


# ---------------------------------------------------------------------------
# _init_weights: DTensor-safe padding row zeroing (FSDP2 Shard(0) embeddings)


_PAD_NPROC = 2


def _shard_ones(vocab, mesh, rank):
    from torch.distributed.tensor import DTensor, Shard

    base, remainder = divmod(vocab, _PAD_NPROC)
    length = base + (1 if rank < remainder else 0)
    offset = rank * base + min(rank, remainder)
    local = torch.ones(length, 4)
    shard = DTensor.from_local(local, mesh, [Shard(0)], shape=(vocab, 4), stride=(4, 1), run_check=False)
    return shard, offset, length


def _pad_row_worker(rank, rendezvous):
    from torch.distributed.device_mesh import DeviceMesh

    from veomni.models.transformers.kimi_k3.source.modeling_kimi_k3 import _zero_embedding_padding_row

    os.environ["WORLD_SIZE"] = str(_PAD_NPROC)
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=_PAD_NPROC,
        timeout=timedelta(minutes=5),
    )
    try:
        init_parallel_state_from_config(AcceleratorConfig(ulysses_size=_PAD_NPROC), name=None)
        mesh = DeviceMesh("cpu", torch.arange(_PAD_NPROC))

        # (vocab, padding_idx) pairs covering even and uneven chunk splits plus
        # both shard boundaries: the padding row must be zeroed on the owning
        # rank only, and every other row must stay untouched.
        for vocab, padding_idx in ((10, 7), (10, 4), (11, 5), (11, 6)):
            shard, offset, length = _shard_ones(vocab, mesh, rank)
            _zero_embedding_padding_row(shard, padding_idx)
            local = shard.to_local()
            for row in range(length):
                global_row = offset + row
                if global_row == padding_idx:
                    assert local[row].eq(0).all(), (vocab, padding_idx, rank)
                else:
                    assert local[row].eq(1).all(), (vocab, padding_idx, rank, global_row)

        # The real ``_init_weights`` must wire the helper: on a DTensor-sharded
        # embedding the padding row is zeroed on the owning rank while the
        # element-wise ``normal_`` fill (shard-local by construction) keeps
        # every other row non-zero.
        model = _build_model()
        embedding = torch.nn.Embedding(11, 4, padding_idx=6)
        shard, offset, length = _shard_ones(11, mesh, rank)
        embedding.weight = torch.nn.Parameter(shard)
        model._init_weights(embedding)
        local = embedding.weight.to_local()
        for row in range(length):
            global_row = offset + row
            if global_row == 6:
                assert local[row].eq(0).all(), rank
            else:
                assert local[row].abs().sum() > 0, (rank, global_row)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=_PAD_NPROC))
def test_init_weights_padding_row_zeroed_on_shard(tmp_path):
    """``weight.data[padding_idx].zero_()`` is a no-op (and, when params are
    CPU-materialized under fsdp offload, a crash) on ``Shard(0)`` DTensors:
    the indexed read redistributes to a discarded Replicate copy.  The fix
    zeros the owning rank's local shard without any collective; this pins the
    chunk-offset math on even and uneven splits plus the ``_init_weights``
    wiring on a real (tiny) model.
    """
    mp.spawn(_pad_row_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=_PAD_NPROC)


def test_zero_embedding_padding_row_plain_tensor():
    from veomni.models.transformers.kimi_k3.source.modeling_kimi_k3 import _zero_embedding_padding_row

    weight = torch.ones(10, 4)
    _zero_embedding_padding_row(weight, 3)
    assert weight[3].eq(0).all()
    assert weight.sum() == 36.0


def test_kda_o_norm_affine_matches_fla_semantics():
    from veomni.models.transformers.kimi_k3.source.kda_fallback import FusedRMSNormGated

    # fla's FusedRMSNormGated defaults to elementwise_affine=True with a
    # ones-initialised learnable scale; the vendored fallback must mirror it
    # so released checkpoints carry (and load) the o_norm scale.
    norm = FusedRMSNormGated(8)
    assert norm.elementwise_affine
    assert norm.weight.shape == (8,) and norm.weight.eq(1).all()

    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
    g = torch.randn(4, 8, generator=torch.Generator().manual_seed(1))
    x_f = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.eps)
    ref = (x_f * torch.sigmoid(g.float())).to(x.dtype)
    torch.testing.assert_close(norm(x, g), ref)  # weight=1 is an exact no-op

    with torch.no_grad():
        norm.weight.mul_(2.0)
    torch.testing.assert_close(norm(x, g), ref * 2.0)  # scale applies post-norm

    plain = FusedRMSNormGated(8, elementwise_affine=False)
    assert not hasattr(plain, "weight")
    torch.testing.assert_close(plain(x, g), ref)


def test_init_weights_reinits_o_norm_and_vision_pos_emb():
    from veomni.models.transformers.kimi_k3.source.kda_fallback import FusedRMSNormGated

    model = _build_model()

    # every KDA layer must expose the affine o_norm weight in the state dict
    assert any(n.endswith("o_norm.weight") for n, _ in model.named_parameters())

    # to_empty leaves allocator garbage behind; _init_weights must restore
    # construction defaults for module types outside the Linear/Embedding
    # chain. The pos-emb is taken from the built model itself so the
    # isinstance dispatch matches the modeling backend under test.
    pos_embs = [m for m in model.modules() if m.__class__.__name__ == "Learnable2DInterpPosEmbDivided_fixed"]
    assert pos_embs, "vision pos-emb module missing from the built model"
    pos_emb = pos_embs[0]
    pos_emb.weight.data.fill_(float("nan"))
    model._init_weights(pos_emb)
    assert torch.isfinite(pos_emb.weight).all() and pos_emb.weight.abs().sum() > 0

    # FusedRMSNormGated is shared with the generated modules (they import it
    # from the vendored source module), so a standalone instance dispatches.
    o_norm = FusedRMSNormGated(8)
    o_norm.weight.data.fill_(float("nan"))
    model._init_weights(o_norm)
    assert o_norm.weight.eq(1).all()
