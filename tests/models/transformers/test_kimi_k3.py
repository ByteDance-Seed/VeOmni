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

"""Kimi K3 multimodal packing tests: pre-expanded merge and varlen metadata.

The kimi_k3 data transform pre-expands every media placeholder into one
``<|media_pad|>`` per merged vision token, so the multimodal merge is
length-preserving: the collator's packed position_ids / cu_seq_lens stay
valid through it and are forwarded to the inner model as-is (exactly like
the text-only forward does), instead of being rebuilt after an in-model
expansion. These tests pin the merge dispatch, the metadata forwarding,
and the SP data path end to end:

* the merge's three forms -- pre-expanded scatter (production), legacy
  HF one-placeholder expansion (vendored compat, rejected under SP where
  the collator's metadata is only valid for a length-preserving merge),
  and the text-only passthrough -- plus the loud mismatch raises;
* the forward forwards the collator's varlen metadata (cu_seq_lens_q/k,
  max_length_q/k, the ``cu_seqlens`` alias the KDA layers read) unchanged
  to the inner model, rejects genuinely packed rows under dense
  attention, and keeps single-segment / known-tail-pad rows on the dense
  path;
* a two-rank ulysses (gloo) run drives the real SP data path end to end --
  the collator's per-rank shards (pixel data replicated, labels pre-shifted
  on the full row before slicing, full-row cu metadata published per rank),
  the gather / merge / re-slice cycle, the KDA a2a head-shard route -- and
  pins loss/gradient parity with the single-rank reference.
"""

from __future__ import annotations

import math
import os
from datetime import timedelta

import pytest
import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, stamp_attn_implementation
from tests.models.tiny_configs import (
    tiny_kimi_k3_config as _tiny_config,
)
from tests.models.tiny_configs import (
    tiny_kimi_linear_config as _tiny_linear_config,
)
from veomni.arguments import AcceleratorConfig
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state_from_config
from veomni.distributed.sequence_parallel.data import sp_pad_and_slice
from veomni.models.auto import build_foundation_model
from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import (
    _shard_dim0_offset_length,
)
from veomni.ops.registry import VeomniOp


def _sp2_build_model():
    """Build the SP(2) numerics model with every attention row on veomni_sdpa.

    The round-9 SP contract: raw eager/sdpa rows refuse the Ulysses
    exchange, so the SP(2) workers (and the single-rank reference they
    are compared against) must run a ``veomni_*`` attention row.
    Construction still goes through the all-eager ops config -- the
    flat ``ops_implementation`` override stamps root and children
    alike, and an accelerator-flavored root selection dies in HF's
    construction check (the wrapper class supports neither sdpa nor
    flash dispatch) -- and the rows are then rebound: every K3
    attention layer holds an instance-local ``veomni_attn`` VeomniOp
    bound at construction, so swap the handles and stamp the configs
    (the forward-time SP routing reads ``config._attn_implementation``)
    -- the same surgery ``pin_eager_attn_implementation`` performs, in
    the other direction. veomni_sdpa leases on any device and its
    all_to_all_single-based exchange is gloo-compatible, so the
    sharded vision tower and the attention-internal MLA exchange -- the
    production SP path -- run end to end on plain CPU tensors over the
    gloo group.
    """
    model = _build_model()
    stamp_attn_implementation(getattr(model, "config", None), "veomni_sdpa")
    for module in model.modules():
        stamp_attn_implementation(getattr(module, "config", None), "veomni_sdpa")
        handle = getattr(module, "veomni_attn", None)
        if getattr(handle, "op", None) == "attention":
            module.veomni_attn = VeomniOp("attention", "standard", "veomni_sdpa")
    return model


def _build_model(ops_cfg=None):
    cfg = ops_cfg if ops_cfg is not None else eager_ops_config()
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


def _expanded_packed_batch(model):
    """One pre-expanded packed row of two samples with a mid-sample image.

    The kimi_k3 data transform already materialized the merged vision span:
    ``[T0 T1 V0 V1 V2 V3 T2 T3 T4][B0 B1 B2 B3]`` (13 tokens, V slots carry
    the media placeholder id). The collator publishes the raw labels (media
    span IGNORE-filled by the transform), segment-aware position_ids, and
    the full-row cu metadata: sample B starts at merged coordinate 9.
    """
    ignore = model.config.ignore_index
    ph = model.config.media_placeholder_token_id
    input_ids = torch.tensor([[10, 11, ph, ph, ph, ph, 13, 14, 15, 20, 21, 22, 23]])
    labels = torch.tensor([[ignore, 101, ignore, ignore, ignore, ignore, 103, 104, 105, ignore, 203, 204, 205]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    grid_thws = torch.tensor([[1, 4, 4]])
    pixel_values = torch.randn(16, 3, 14, 14)
    cu = torch.tensor([0, 9, 13], dtype=torch.int32)
    return input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values, cu


def _expanded_single_image_batch(model):
    """One pre-expanded un-packed sample containing one image.

    ``[T0 T1 V0 V1 V2 V3 T2 T3]`` (8 tokens) -- a single segment, like the
    collator emits for an un-packed sequence (cu = [0, 8]).
    """
    ignore = model.config.ignore_index
    ph = model.config.media_placeholder_token_id
    input_ids = torch.tensor([[10, 11, ph, ph, ph, ph, 13, 14]])
    labels = torch.tensor([[ignore, 101, ignore, ignore, ignore, ignore, 103, 104]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    attention_mask = torch.ones_like(input_ids)
    grid_thws = torch.tensor([[1, 4, 4]])
    pixel_values = torch.randn(16, 3, 14, 14)
    cu = torch.tensor([0, 8], dtype=torch.int32)
    return input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values, cu


def _collator_fa_kwargs(cu):
    """The varlen metadata the collator publishes alongside the packed row."""
    max_len = int(cu[-1].item())
    return dict(
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=max_len,
        max_length_k=max_len,
        linear_attn_cu_seq_lens_q=cu,
    )


def _spy_on_inner_model(model, recorded):
    inner_forward = model.language_model.model.forward

    def _spy(*args, **kwargs):
        recorded.update(kwargs)
        return inner_forward(*args, **kwargs)

    model.language_model.model.forward = _spy
    return inner_forward


def test_merge_scatters_raw_labels_and_maps_boundaries():
    """The legacy one-placeholder merge keeps the vendored HF scatter semantics.

    Raw labels scatter onto the surviving text slots; the expanded vision
    span is IGNORE-filled, which is what a single causal shift relies on to
    supervise the first post-image text token from the last vision
    position. Positions are rebuilt continuous across the expanded row.
    """
    model = _build_model()
    model.eval()
    ph = model.config.media_placeholder_token_id
    ignore = model.config.ignore_index
    hidden = model.config.text_config.hidden_size
    # Legacy HF form: ONE placeholder for the whole image (4 merged tokens).
    input_ids = torch.tensor([[10, 11, ph, 13, 14, 15, 20, 21, 22, 23]])
    labels = torch.tensor([[ignore, 101, ignore, 103, 104, 105, ignore, 203, 204, 205]])
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds = torch.zeros(1, input_ids.shape[1], hidden)
    image_features = [torch.randn(4, hidden)]

    with torch.no_grad():
        merged_embeds, merged_mask, merged_labels, merged_position_ids = model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels
        )

    assert merged_embeds.shape == (1, 13, hidden)
    assert torch.equal(merged_mask, torch.ones(1, 13, dtype=merged_mask.dtype))
    # Raw labels scatter onto text slots; the expanded vision span
    # (merged coords 2..5) is IGNORE-filled.
    expected_labels = torch.tensor(
        [[ignore, 101, ignore, ignore, ignore, ignore, 103, 104, 105, ignore, 203, 204, 205]]
    )
    assert torch.equal(merged_labels, expected_labels)
    # The vendored expansion rebuilds continuous positions over the row.
    torch.testing.assert_close(merged_position_ids, torch.arange(13, dtype=torch.long).unsqueeze(0), rtol=0, atol=0)
    # A single causal shift over the merged labels supervises the first
    # post-image text token (T2, label 103) from the last vision position.
    shifted = torch.nn.functional.pad(expected_labels, (0, 1), value=ignore)[..., 1:]
    assert shifted.tolist() == [[101, ignore, ignore, ignore, ignore, 103, 104, 105, ignore, 203, 204, 205, ignore]]


def test_merge_pre_expanded_scatters_features_and_preserves_metadata():
    """The pre-expanded form is length-preserving: metadata passes through.

    With one ``<|media_pad|>`` per merged vision token the merge only
    scatters the projected features onto the pad slots; attention_mask,
    labels, and position_ids must be returned untouched (the transform and
    collator built them for the expanded row), so the collator's packed
    cu_seq_lens stay valid without any rebuild.
    """
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, _, _ = _expanded_packed_batch(model)
    hidden = model.config.text_config.hidden_size
    ph = model.config.media_placeholder_token_id
    inputs_embeds = torch.zeros(1, input_ids.shape[1], hidden)
    features = torch.randn(4, hidden)
    image_features = [features]

    with torch.no_grad():
        merged_embeds, merged_mask, merged_labels, merged_position_ids = model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels, position_ids
        )

    # Length-preserving: no rebuild, metadata passed through by identity.
    assert merged_embeds.shape == (1, 13, hidden)
    assert merged_mask is attention_mask
    assert merged_labels is labels
    assert merged_position_ids is position_ids
    # Features land row-major on the pad slots in order; text slots untouched.
    pad_slots = (input_ids[0] == ph).nonzero().flatten()
    for i, slot in enumerate(pad_slots.tolist()):
        torch.testing.assert_close(merged_embeds[0, slot], features[i], rtol=0, atol=0)
    text_slots = (input_ids[0] != ph).nonzero().flatten()
    assert merged_embeds[0, text_slots].eq(0).all()


def test_merge_rejects_placeholder_count_mismatch():
    """Ambiguous placeholder/feature counts must fail loudly, not mislabel."""
    model = _build_model()
    model.eval()
    ph = model.config.media_placeholder_token_id
    hidden = model.config.text_config.hidden_size
    input_ids = torch.tensor([[10, 11, ph, ph, 13, 14]])
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds = torch.zeros(1, 6, hidden)
    image_features = [torch.randn(4, hidden)]

    # 2 placeholders for 1 image of 4 merged tokens: neither the
    # pre-expanded nor the legacy form.
    with pytest.raises(ValueError, match="Cannot merge image features"):
        model._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask)

    # Features but no placeholder at all: nowhere to merge them.
    input_ids_no_ph = torch.tensor([[10, 11, 12, 13, 14, 15]])
    with pytest.raises(ValueError, match="nowhere to merge"):
        model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids_no_ph, torch.ones_like(input_ids_no_ph)
        )


def test_merge_rejects_legacy_form_under_sp():
    """Under SP the collator's metadata assumes a length-preserving merge.

    The SP collator pre-shifts labels and slices the packed position_ids /
    cu_seq_lens on the full row; the legacy expansion changes the row
    length and would silently invalidate all of them, so it must be
    rejected up front.
    """
    model = _build_model()
    model.eval()
    ph = model.config.media_placeholder_token_id
    ignore = model.config.ignore_index
    hidden = model.config.text_config.hidden_size
    input_ids = torch.tensor([[10, 11, ph, 13, 14]])
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds = torch.zeros(1, 5, hidden)
    image_features = [torch.randn(4, hidden)]

    with pytest.raises(ValueError, match="pre-expanded"):
        model._merge_input_ids_with_image_features(
            image_features,
            inputs_embeds,
            input_ids,
            attention_mask,
            torch.full_like(input_ids, ignore),
            require_pre_expanded=True,
        )


def test_text_only_merge_passthrough():
    """A row without media placeholders is returned unchanged."""
    model = _build_model()
    model.eval()
    hidden = model.config.text_config.hidden_size
    input_ids = torch.tensor([[5, 6, 7, 8]])
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds = torch.randn(1, 4, hidden)
    labels = torch.tensor([[1, 2, 3, 4]])
    position_ids = torch.tensor([[0, 1, 2, 3]])

    out_embeds, out_mask, out_labels, out_position_ids = model._merge_input_ids_with_image_features(
        [], inputs_embeds, input_ids, attention_mask, labels, position_ids
    )
    assert out_embeds is inputs_embeds
    assert out_mask is attention_mask
    assert out_labels is labels
    assert out_position_ids is position_ids

    _, _, out_labels_none, _ = model._merge_input_ids_with_image_features([], inputs_embeds, input_ids, attention_mask)
    assert out_labels_none is None


def test_packed_forward_forwards_collator_varlen_metadata(dense_packed_debug):
    """The collator's packed cu metadata flows to the inner model untouched.

    The pre-expanded merge is length-preserving, so the forward must not
    rebuild anything: the collator's cu_seq_lens_q/k, max_length_q/k, the
    ``cu_seqlens`` alias for the KDA layers, and the packed position_ids
    all pass through exactly as published.
    """
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values, cu = _expanded_packed_batch(model)

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
                **_collator_fa_kwargs(cu),
            )
    finally:
        model.language_model.model.forward = inner_forward

    assert out.loss is not None and torch.isfinite(out.loss)
    assert torch.equal(recorded["cu_seq_lens_q"], cu)
    assert torch.equal(recorded["cu_seq_lens_k"], cu)
    assert torch.equal(recorded["cu_seqlens"], cu)
    assert recorded["max_length_q"] == 13
    assert torch.equal(recorded["position_ids"], position_ids)


def test_packed_forward_rejects_dense_attention():
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values, cu = _expanded_packed_batch(model)

    # The collator's varlen metadata is forwarded as-is; a row with a real
    # boundary between two samples must be rejected under the dense eager
    # attention instead of silently leaking across packed samples
    # (K3_ALLOW_DENSE_PACKED is unset here).
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
                **_collator_fa_kwargs(cu),
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


def test_single_sample_image_forward_runs_dense():
    """Single-segment varlen metadata must not trip the dense-attention guard.

    The collator emits ``cu_seq_lens_q`` = [0, length] for un-packed
    sequences too; a dense causal mask is exact for one contiguous
    sequence, so eager attention runs without the K3_ALLOW_DENSE_PACKED
    escape and the metadata still flows through.
    """
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values, cu = _expanded_single_image_batch(model)

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
                **_collator_fa_kwargs(cu),
            )
    finally:
        model.language_model.model.forward = inner_forward

    assert out.loss is not None and torch.isfinite(out.loss)
    assert torch.equal(recorded["cu_seq_lens_q"], cu)
    assert torch.equal(recorded["cu_seqlens"], cu)


def test_single_sample_tail_pads_run_dense():
    """Known tail padding must not count as a packed sample.

    The collator coalesces trailing pads (SP / pad-to-length) into one
    final segment and publishes ``tail_padding_length``: a single-image row
    with two tail pads carries cu=[0, 8, 10] -- one real sample plus the
    pad segment. Under eager attention that must still run, like an
    un-packed sequence (no dense-attention rejection).
    """
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values, _ = _expanded_single_image_batch(model)
    ignore = model.config.ignore_index
    pad = model.config.pad_token_id
    input_ids = torch.cat([input_ids, torch.tensor([[pad, pad]])], dim=1)
    labels = torch.cat([labels, torch.full((1, 2), ignore, dtype=labels.dtype)], dim=1)
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 0]])
    attention_mask = torch.ones_like(input_ids)
    tail_padding_length = torch.tensor(2, dtype=torch.int32)
    cu = torch.tensor([0, 8, 10], dtype=torch.int32)

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
                **_collator_fa_kwargs(cu),
            )
    finally:
        model.language_model.model.forward = inner_forward

    assert out.loss is not None and torch.isfinite(out.loss)
    # [T0 T1 V V V V T2 T3][P P]: the two pads form ONE coalesced 2-token
    # tail segment (8..10) -- one real sample, so dense attention runs.
    assert recorded["cu_seq_lens_q"].tolist() == [0, 8, 10]


def test_shard_dim0_offset_length_matches_torch_chunk():
    """The padding-row shard math must mirror DTensor's Shard(0) even split.

    DTensor shards via torch.chunk semantics (ceil-sized chunks, the
    last shard absorbing the remainder); the offset/length math must
    match it exactly or the padding row is zeroed on the wrong rank /
    out of bounds.
    """
    for global_size, num_shards in [(163840, 3), (163840, 6), (163840, 8), (93, 4), (10, 3), (7, 3), (5, 5)]:
        chunks = torch.empty(global_size).chunk(num_shards)
        for rank in range(num_shards):
            offset, length = _shard_dim0_offset_length(global_size, num_shards, rank)
            expected_offset = sum(c.numel() for c in chunks[:rank])
            expected_length = chunks[rank].numel() if rank < len(chunks) else 0
            assert (offset, length) == (expected_offset, expected_length)


def test_shard_dim0_offset_length_native_vocab_padding():
    """The native vocab padding row stays in bounds on uneven shards.

    vocab 163840 / 3 shards -> [54614, 54614, 54612]; padding id 163839
    is local row 54611 of the last shard. The old floor + remainder
    math computed local row 54612 -- one past the end.
    """
    offset, length = _shard_dim0_offset_length(163840, 3, 2)
    assert (offset, length) == (109228, 54612)
    assert 163839 - offset == 54611
    assert 0 <= 163839 - offset < length


def test_packed_forward_forwards_collator_coalesced_tail_pads(dense_packed_debug):
    """The collator's coalesced tail-pad cu survives the merge untouched.

    Trailing pads (SP / pad-to-length) are coalesced into one final cu
    segment by the collator's ``coalesce_tail_padding_cu_seqlens`` (an
    uncoalesced per-token tail SIGSEGVs the Ascend npu_fusion_attention);
    the pre-expanded merge is length-preserving, so the published cu --
    including the coalesced pad segment -- must reach the inner model
    exactly as published.
    """
    model = _build_model()
    model.eval()
    input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values, _ = _expanded_packed_batch(model)
    ignore = model.config.ignore_index
    pad = model.config.pad_token_id
    # [T0 T1 V V V V T2 T3 T4][B0 B1 B2 B3][P P]: append two SP tail pads.
    input_ids = torch.cat([input_ids, torch.tensor([[pad, pad]])], dim=1)
    labels = torch.cat([labels, torch.full((1, 2), ignore, dtype=labels.dtype)], dim=1)
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 0, 0]])
    attention_mask = torch.ones_like(input_ids)
    tail_padding_length = torch.tensor(2, dtype=torch.int32)
    cu = torch.tensor([0, 9, 13, 15], dtype=torch.int32)

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
                **_collator_fa_kwargs(cu),
            )
    finally:
        model.language_model.model.forward = inner_forward

    assert out.loss is not None and torch.isfinite(out.loss)
    # Merged row is 15 slots; the two pad tokens form ONE coalesced 2-token
    # tail segment (13..15) -- exactly what the collator published.
    assert torch.equal(recorded["cu_seq_lens_q"], cu)
    assert torch.equal(recorded["cu_seqlens"], cu)
    # The real samples keep their segment-aware positions; each pad token
    # is its own boundary in the collator's position_ids.
    assert torch.equal(recorded["position_ids"], position_ids)


def test_use_cache_forward_matches_no_cache_logits():
    """A one-step ``use_cache=True`` forward goes through the 5.16.1 cache
    contract (the mask builder calls ``get_query_offset`` /
    ``get_mask_sizes`` with a Python int; the vendored KimiDynamicCache
    predates the contract and raised AttributeError) and must stay
    logit-identical to the no-cache path (round-6 review; training runs
    use_cache=False, which is why the broken path survived CI)."""
    model = _build_model()
    model.eval()
    input_ids = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        plain = model(input_ids=input_ids, use_cache=False).logits
        cached = model(input_ids=input_ids, use_cache=True).logits
    torch.testing.assert_close(plain, cached)


def test_packed_guard_reads_the_text_config():
    """The packed-batch guard must consult the decoder's own selection, not
    the root's (round-6 review): VeOmniPretrainedConfig permits per-child
    attention selections that rank above the root value, so the two can
    diverge. Both divergences break the guard -- a root varlen / text eager
    pair must still be rejected (the decoder's dense eager mask would
    attend across samples), and a root eager / text varlen pack is valid
    and must not be refused. The selections are stamped post-construction
    (the layer handles stay bound to the constructed eager rows: this pins
    the guard's verdict -- a diverged build is not constructible on CPU
    leases). CPU leases cannot run the varlen side either (no varlen
    kernel exists to bind), so the accepted direction runs behind a spy
    that stands in for the varlen backend the text config selected: it
    records the decoder call the guard let through and delegates an
    unpacked view to the real decoder."""
    model = _build_model()
    model.eval()
    cu = torch.tensor([0, 5, 10], dtype=torch.int32)
    input_ids = torch.randint(0, 100, (1, 10))
    packed = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(10).unsqueeze(0),
        use_cache=False,
        **_collator_fa_kwargs(cu),
    )

    model.config._attn_implementation = "veomni_flash_attention_2"
    model.config.text_config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="cannot run packed batches"):
        model(**packed)

    model.config._attn_implementation = "eager"
    model.config.text_config._attn_implementation = "veomni_flash_attention_2"
    # The guard must let this batch reach the decoder (the decoder's own
    # selection is varlen, so packing is legal here). The old root-reading
    # guard refused it before the decoder call, so the recorded kwargs
    # below fail on the unfixed code.
    recorded = {}
    inner_forward = model.language_model.model.forward

    def _varlen_backend_spy(*args, **kwargs):
        recorded.update(kwargs)
        unpacked = {
            key: value
            for key, value in kwargs.items()
            if not (key.startswith("cu_") or key.startswith("max_length") or key == "tail_padding_length")
        }
        return inner_forward(*args, **unpacked)

    model.language_model.model.forward = _varlen_backend_spy
    try:
        with torch.no_grad():
            out = model(**packed)
    finally:
        model.language_model.model.forward = inner_forward
    assert torch.equal(recorded["cu_seq_lens_q"], cu)
    assert torch.isfinite(out.logits).all()


def test_position_id_func_is_picklable():
    """Dataloader workers (spawn start method) pickle the VLM trainer's
    dataset transform, which captures the position-id func returned by
    ``KimiK3ForConditionalGeneration.get_position_id_func``. A closure
    defined inside the patched method fails with "Can't pickle local
    object" (round-3 review); the module-level helper must survive a
    pickle round-trip and keep the arange-stub behaviour."""
    import pickle

    from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import (
        _kimi_k3_position_ids,
    )

    input_ids = torch.randint(0, 100, (2, 7))
    position_ids = _kimi_k3_position_ids(input_ids=input_ids)
    assert position_ids.shape == input_ids.shape
    assert torch.equal(position_ids, torch.arange(7).expand(2, 7))
    restored = pickle.loads(pickle.dumps(_kimi_k3_position_ids))
    assert torch.equal(restored(input_ids=input_ids), position_ids)
    assert restored(input_ids=None) is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_vlm_attention_override_preserves_child_selections():
    """A root attention override must not clobber explicit per-child
    selections (round-3 review): ``VeOmniPretrainedConfig``'s
    ``_attn_implementation`` mapping form records per-child choices that
    rank above the root value, and the patched ``__init__`` used to
    re-stamp both sub-configs with the root selection. The tower's attention
    selection is validated by the attention/standard row binding at
    construction (round-9 review) -- unsupported names raise there, not
    KeyError at the first image forward.

    The construction is CPU-native end to end (round-4 review): both
    children pin 'eager' explicitly -- the toy config's text default is
    the accelerator-bound 'veomni_flash_attention_2' -- and the
    discriminating root selection is a non-flash veomni name,
    'veomni_sage_attention', which HF passes through for the root class
    while the vision validation rejects it for the tower: the re-stamp
    this test pins fails loudly instead of silently degrading to dense
    eager. A flash-flavored root cannot play that role here at all: the
    suite's cpu_model_attention_kernel_guard forbids FlashAttention
    preloading under CPU launch plans, and HF's preload path fires before
    any registry row, so even a mocked row would not construct."""
    from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import (
        KimiK3ForConditionalGeneration,
    )

    # explicit per-child selections (HF mapping form -- the setter writes
    # only the named children) survive a root selection embedded in the
    # config itself, the production shape: the launch yaml embeds
    # ops_config in the model config and VeOmniPretrainedConfig's
    # post-init keeps per-child entries on top of it. The old patch
    # re-stamped both sub-configs with the root value at __init__ -- with
    # the root at 'veomni_sage_attention' that re-stamp now fails the
    # vision-tower validation instead of quietly training dense eager.
    cfg = _tiny_config()
    cfg._attn_implementation = {"text_config": "eager", "vision_config": "eager"}
    # root selection written without the setter's tree-wide recursion
    cfg.ops_implementation["attn_implementation"] = "veomni_sage_attention"
    model = build_foundation_model(cfg, torch_dtype="float32", init_device="cpu")
    assert model.config.text_config._attn_implementation == "eager"
    assert model.config.vision_config._attn_implementation == "eager"

    # an sdpa-flavored selection on the vision child itself (raw dict
    # spelling -- both the normalizing setter and HF's own 'sdpa'
    # construction guard live elsewhere) is honored as a real
    # attention/standard row (round-9 review: the tower binds the standard
    # family, so 'veomni_sdpa' no longer silently falls back to 'eager').
    # The text side pins an explicit eager default instead of the toy
    # config's accelerator-bound 'veomni_flash_attention_2' (round-4
    # review).
    cfg = _tiny_config()
    cfg.text_config.ops_implementation["attn_implementation"] = "eager"
    cfg.vision_config.ops_implementation["attn_implementation"] = "veomni_sdpa"
    # the toy config's root default is the flash-flavored
    # 'veomni_flash_attention_2', which HF's root-class check would try to
    # preload (forbidden under CPU launch plans): pin the root to the same
    # eager default explicitly
    cfg.ops_implementation["attn_implementation"] = "eager"
    model = KimiK3ForConditionalGeneration(cfg)
    assert model.config.text_config._attn_implementation == "eager"
    assert model.config.vision_config._attn_implementation == "veomni_sdpa"

    # unsupported vision selection raises at construction: the row binding
    # in MoonViTEncoderLayer.__init__ refuses it (veomni_sage_attention is a
    # CUDA-SM80 attention/standard row, so a non-CUDA lease dies at the
    # device-registration check). Written per-child -- the scenario-2
    # spelling, root pinned to eager -- because the root mapping form hands
    # HF's own construction check a non-eager root value first, which drags
    # the optional flash preload into a CPU launch plan before any
    # registry row gets to refuse the selection.
    cfg = _tiny_config()
    cfg.text_config.ops_implementation["attn_implementation"] = "eager"
    cfg.ops_implementation["attn_implementation"] = "eager"
    cfg.vision_config.ops_implementation["attn_implementation"] = "veomni_sage_attention"
    with pytest.raises((ValueError, RuntimeError)):
        KimiK3ForConditionalGeneration(cfg)


def test_short_convolution_reuses_hoisted_host_boundaries():
    """ShortConvolution.forward accepts the hoisted host copy of the
    varlen boundaries (round-3 review): reusing it must be output-identical
    to re-deriving cu_seqlens.tolist() inside, for both the plain and
    packed varlen forms (CPU tensor or plain sequence)."""
    from veomni.ops.kernels.kda.kimi_k3.eager import ShortConvolution

    torch.manual_seed(0)
    conv = ShortConvolution(hidden_size=8, kernel_size=4)
    x = torch.randn(1, 10, 8)

    y_plain, _ = conv(x)
    assert y_plain.shape == x.shape

    cu = torch.tensor([0, 4, 10])
    y_ref, _ = conv(x=x, cu_seqlens=cu)
    # the packed form resets causality at the segment boundary, so the
    # varlen output must differ from the plain causal run
    assert not torch.allclose(y_ref, y_plain)
    y_host_list, _ = conv(x=x, cu_seqlens=cu, cu_seqlens_cpu=cu.tolist())
    y_host_tensor, _ = conv(x=x, cu_seqlens=cu, cu_seqlens_cpu=cu.cpu())
    torch.testing.assert_close(y_host_list, y_ref)
    torch.testing.assert_close(y_host_tensor, y_ref)


def test_decoder_layer_forward_runs():
    """The KimiDecoderLayer runs end-to-end standalone. The
    attention-residual block packing is threaded with the production shape the
    model uses -- a (num_tokens, 0, hidden) zero tensor that each
    block-boundary layer appends to -- spanning both attention families and
    both feed paths: KDA+dense-MLP, KDA+MoE, and MLA+MoE layers."""
    from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import KimiDecoderLayer as DecoderLayer

    config = _tiny_config().text_config
    # All-eager rows: the raw module is exercised outside build_foundation_model,
    # so stamp the eager selection the builder would have applied.
    config.ops_implementation.update(vars(eager_ops_config()))
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    hidden_states = torch.randn(1, 32, config.hidden_size)
    # the model threads block_residual as a (num_tokens, n_blocks, hidden)
    # chain starting from an empty block list (KimiLinearModel.forward)
    block_residual = hidden_states.new_zeros(hidden_states.shape[0] * hidden_states.shape[1], 0, config.hidden_size)
    for layer_idx in (0, 1, 3):
        module = DecoderLayer(config, layer_idx=layer_idx).eval()
        for p in module.parameters():
            torch.nn.init.normal_(p, std=0.02)
        with torch.no_grad():
            out, block_residual = module(hidden_states, block_residual=block_residual)
        assert out.shape == hidden_states.shape
        assert torch.isfinite(out).all()
    # layer 0 sits on the attn_res_block_size boundary and appends one block
    assert block_residual.shape == (32, 1, config.hidden_size)


def test_moe_gate_correction_bias_is_a_buffer():
    """The router's e_score_correction_bias feeds only the top-k selection,
    so autograd cannot reach it: as an nn.Parameter every gate bias sat
    without a gradient and broke DDP's reducer on the second step (VeOmni
    runs ddp without find_unused_parameters). It is registered as a buffer
    like the DeepSeek families (round-6 review), and a training step must
    leave every trainable language-model parameter with a gradient. The
    vision tower is outside this batch's graph (no pixel_values) and thus
    excluded from the no-unused assertion, alongside layer 0's
    attention-side res pair (architecturally unused; see below)."""
    model = _build_model()
    gate = model.language_model.model.layers[1].block_sparse_moe.gate
    assert "e_score_correction_bias" in dict(gate.named_buffers())
    assert "e_score_correction_bias" not in dict(gate.named_parameters())
    assert torch.equal(gate.e_score_correction_bias, torch.zeros_like(gate.e_score_correction_bias))

    model.train()
    input_ids = torch.randint(0, 100, (1, 12))
    out = model(input_ids=input_ids, labels=input_ids.clone())
    out.loss.backward()
    unused = [n for n, p in model.language_model.named_parameters() if p.requires_grad and p.grad is None]
    # The block_residual chain starts empty (new_zeros(n_tokens, 0, H)):
    # layer 0's attention-side res projection only ever processes the
    # accumulated blocks, and its own boundary append stores the raw
    # prefix sum rather than a res-param output, so
    # layers.0.self_attention_res_{norm,proj} are architecturally unused
    # in any forward; every other parameter must participate in the
    # backward.
    allowed = {name for name, _ in model.language_model.named_parameters() if ".layers.0.self_attention_res_" in name}
    leftover = [name for name in unused if name not in allowed]
    assert not leftover, f"trainable parameters without a gradient after backward: {leftover}"


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


def _sp2_full_row(model):
    """The pre-expanded full row both ranks reconstruct: 12 tokens (SP-even).

    ``[T0 T1 V0 V1 V2 V3 T2 T3 T4][B0 B1 B2]`` -- 9 + 3 = 12 tokens, already
    expanded by the data transform (V slots carry the media placeholder id).
    The collator keeps packed rows sp_size-divisible, so the shards split
    evenly (6 + 6) and no slice pad is introduced; full-row cu = [0, 9, 12].
    Deterministic pixels keep the ranks' sharded tower halves in lockstep
    with the single-rank tower after the gather.
    """
    ignore = model.config.ignore_index
    ph = model.config.media_placeholder_token_id
    input_ids = torch.tensor([[10, 11, ph, ph, ph, ph, 13, 14, 15, 20, 21, 22]])
    labels = torch.tensor([[ignore, 101, ignore, ignore, ignore, ignore, 103, 104, 105, ignore, 203, 204]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2]])
    pixel_values = torch.arange(16 * 3 * 14 * 14, dtype=torch.float32).div_(1000).reshape(16, 3, 14, 14)
    return input_ids, labels, position_ids, pixel_values


def _sp2_collator_shard(model, rank):
    """This rank's shard under the SP collator contract.

    The collator pre-shifts labels on the FULL row before slicing (the
    VeOmni SP loss contract: the CE consumes pre-shifted labels as-is),
    slices input_ids / labels / position_ids to the rank's even shard,
    slices pixel_values to the rank's even chunk of the padded global
    token rows (the (0, True, 0, 4) contract -- dim 0, pad 0, pad_scale 4
    keeps every 2x2 merge window whole), keeps grid_thws global, and
    publishes the FULL-row varlen metadata (cu_seq_lens / max_length) on
    every rank. The forward gathers the text shards back, runs the
    length-preserving merge on the full sequence, then slices the merged
    embeds to the local shard again -- the text slice-handling under test
    lives in that second slicing. The tower derives the global cu / pos /
    rope tables from the global grid and sp-slices them to the rank's
    local pixel rows.
    """
    input_ids, labels, _, pixel_values = _sp2_full_row(model)
    pre_shifted = torch.nn.functional.pad(labels, (0, 1), value=model.config.ignore_index)[..., 1:]
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2]])
    shard = slice(rank * 6, (rank + 1) * 6)
    pixel_values = sp_pad_and_slice(pixel_values, dim=0, pad_value=0, pad_scale=4)
    return (
        input_ids[:, shard].contiguous(),
        pre_shifted[:, shard].contiguous(),
        position_ids[:, shard].contiguous(),
        torch.ones(1, 6, dtype=torch.long),
        torch.tensor([[1, 4, 4]]),
        pixel_values,
    )


def _sp2_worker(rank, rendezvous, result_dir, seed):
    # The vendored vision tower keeps an @torch.compile on the ViT position
    # embedding; on the NPU runner a CPU-launch worker reaches the
    # torch_npu-registered inductor backend at that compile and dies at
    # aclInit under device isolation. This worker is a numerics reference, so
    # run it fully eagerly.
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
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
        model = _sp2_build_model()
        model.train()
        input_ids, labels, position_ids, attention_mask, grid_thws, pixel_values = _sp2_collator_shard(model, rank)
        cu = torch.tensor([0, 9, 12], dtype=torch.int32)

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
                **_collator_fa_kwargs(cu),
            )
        finally:
            model.language_model.model.forward = inner_forward

        # The pre-expanded merge is length-preserving and the row is
        # sp-even, so the collator's full-row cu reaches the inner model
        # untouched: two real samples, no slice pad, cu[-1] == gathered
        # full length (12) -- every varlen consumer sees exact metadata.
        expected_cu = cu
        assert torch.equal(recorded["cu_seq_lens_q"], expected_cu)
        assert torch.equal(recorded["cu_seq_lens_k"], expected_cu)
        assert torch.equal(recorded["cu_seqlens"], expected_cu)
        # position_ids were gathered to the full row at entry and stay
        # full-length (the decoder slices the causal mask to the shard,
        # layers re-gather to full length internally).
        assert torch.equal(recorded["position_ids"], torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2]]))
        # Each rank holds half of the merged row after the re-slice.
        assert recorded["inputs_embeds"].shape[1] == 6
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
    """End-to-end SP(2) forward/backward over a pre-expanded packed row.

    The attention rows bind veomni_sdpa on every config level (the
    round-9 SP contract: raw eager/sdpa rows refuse the Ulysses
    exchange), so the SP run exercises the production sharded path --
    the tower computes 1/sp of the pixel rows with the
    attention-internal exchange, and the MLA layers exchange instead
    of the legacy full-sequence gather. The KDA a2a head-shard route
    is gated on SP + ``K3_KDA_HEAD_SHARD`` (default on), not on the
    attention implementation. The SP run consumes
    the collator's pre-shifted labels (the VeOmni SP loss contract: CE
    uses them as-is) while the single-rank reference uses raw labels (CE
    applies its own causal shift), so both supervise the same pairs and
    the SP run must reproduce the single-rank loss (the SP loss reducer
    reports the global mean on every rank). The per-rank gradients,
    summed and rescaled by the sp world size, must reconstruct the
    single-rank gradients (the reducer's backward compensates for ranks
    only seeing their own shard's tokens).
    """
    # Keep the single-rank reference and the SP workers on the exact same code
    # path: the vendored vision tower's @torch.compile (ViT position embedding)
    # reaches the torch_npu-registered inductor backend in CPU-launch workers
    # on the NPU runner and dies at aclInit under device isolation. This is a
    # numerics parity test, so run everything eagerly.
    with torch._dynamo.config.patch(disable=True, suppress_errors=True):
        seed = 1234
        torch.manual_seed(seed)
        model = _sp2_build_model()
        model.train()
        input_ids, labels, position_ids, pixel_values = _sp2_full_row(model)
        cu = torch.tensor([0, 9, 12], dtype=torch.int32)
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=torch.tensor([[1, 4, 4]]),
            attention_mask=torch.ones_like(input_ids),
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            **_collator_fa_kwargs(cu),
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

    from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import (
        _zero_embedding_padding_row,
    )

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
    from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import (
        _zero_embedding_padding_row,
    )

    weight = torch.ones(10, 4)
    _zero_embedding_padding_row(weight, 3)
    assert weight[3].eq(0).all()
    assert weight.sum() == 36.0


def test_kda_o_norm_affine_matches_fla_semantics():
    from veomni.ops.kernels.kda.kimi_k3.eager import FusedRMSNormGated

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
    from veomni.ops.kernels.kda.kimi_k3.eager import FusedRMSNormGated

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

    # FusedRMSNormGated is shared with the modeling module (it imports it
    # from the ops-tree eager reference), so a standalone instance dispatches.
    o_norm = FusedRMSNormGated(8)
    o_norm.weight.data.fill_(float("nan"))
    model._init_weights(o_norm)
    assert o_norm.weight.eq(1).all()


def test_causal_lm_packed_forward_rejects_dense_attention():
    """The text-only causal-LM entry forwards the collator's varlen metadata
    to the inner model, so a packed row must be rejected under eager/sdpa
    attention exactly like the multimodal forward (K3_ALLOW_DENSE_PACKED is
    unset here)."""
    model = build_foundation_model(
        _tiny_linear_config(),
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=eager_ops_config(),
    )
    model.eval()
    input_ids = torch.randint(3, model.config.vocab_size, (1, 8))
    labels = input_ids.clone()
    with torch.no_grad():
        with pytest.raises(ValueError, match="cannot run packed batches"):
            model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=torch.ones(1, 8, dtype=torch.long),
                cu_seq_lens_q=torch.tensor([0, 4, 8]),
                cu_seq_lens_k=torch.tensor([0, 4, 8]),
                max_length_q=8,
                max_length_k=8,
                linear_attn_cu_seq_lens_q=torch.tensor([0, 4, 8]),
                use_cache=False,
            )


def test_single_sample_causal_lm_forward_runs_dense():
    """A single-segment cu ([0, length]) is not packing: the text-only forward
    keeps running it through the dense path without the debug escape hatch."""
    model = build_foundation_model(
        _tiny_linear_config(),
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=eager_ops_config(),
    )
    model.eval()
    input_ids = torch.randint(3, model.config.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            cu_seq_lens_q=torch.tensor([0, 8]),
            cu_seq_lens_k=torch.tensor([0, 8]),
            max_length_q=8,
            max_length_k=8,
            linear_attn_cu_seq_lens_q=torch.tensor([0, 8]),
            use_cache=False,
        )
    assert out.logits.shape == (1, 8, model.config.vocab_size)


def test_moe_rejects_unsupported_activation():
    """The standard MoE expert path always applies F.silu; a hidden_act the
    experts do not implement must fail construction instead of silently
    training with the wrong activation."""
    cfg = _tiny_config()
    cfg.text_config.hidden_act = "gelu"
    with pytest.raises(ValueError, match="activations only"):
        build_foundation_model(
            cfg,
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=eager_ops_config(),
        )


def test_get_seq_length_with_no_full_attention_layers():
    """A configuration with no full-attention layer has nothing cached:
    get_seq_length must return 0 instead of indexing an empty list."""
    from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import KimiDynamicCache

    cache = KimiDynamicCache.__new__(KimiDynamicCache)
    cache.transformer_layers = []
    cache.key_cache = []
    assert cache.get_seq_length() == 0


def test_vision_attention_family_removed_tower_binds_standard_row():
    """Round-9 review: the kimi_k3 vision_attention op family is gone -- the
    MoonViT3d tower binds the attention/standard family row like every other
    VL model in the repo, and no stale kimi_k3 vision rows remain."""
    from veomni.ops.registry import OP_REGISTRY

    try:
        stale = OP_REGISTRY.list_registered("vision_attention", "kimi_k3")
    except (KeyError, ValueError):
        stale = None
    assert not stale

    model = _build_model()
    for blk in model.vision_tower.encoder.blocks:
        assert isinstance(blk.veomni_attn, VeomniOp)
        assert blk.veomni_attn.impl == "eager"


def test_vision_tower_dispatches_through_bound_op_handle():
    """The patched tower dispatches through the bound attention/standard
    VeomniOp and is output-identical to the module-local packed eager
    reference (the vendored math the dedicated family used to dispatch to)
    across multimodal shapes (multi-image rows, mixed grids, video-style
    T>1) -- round-9 review: no kernel is lost by the family removal. The
    two paths run the same eager math with different kernels (the row goes
    through the wrapper's per-segment split of the module-resolved
    ``eager_attention_forward``, the reference stays packed-varlen), so the
    comparison carries a tolerance instead of bit-equality."""
    model = _build_model()
    model.eval()
    for blk in model.vision_tower.encoder.blocks:
        assert isinstance(blk.veomni_attn, VeomniOp)
        assert blk.veomni_attn.impl == "eager"

    from veomni.models.transformers.kimi_k3 import modeling_kimi_k3 as k3_modeling

    grids = [[1, 4, 4], [1, 2, 2], [2, 2, 2]]
    grid_thws = torch.tensor(grids)
    pixel_values = torch.randn(sum(t * h * w for t, h, w in grids), 3, 14, 14)
    with torch.no_grad():
        out_handle = model.vision_tower(pixel_values, grid_thws)

    def local_eager_adapter(module, q, k, v, **kwargs):
        # [1, H, T, D] -> the module-local packed [T, H, D] eager reference
        q3 = q.squeeze(0).transpose(0, 1)
        k3 = k.squeeze(0).transpose(0, 1)
        v3 = v.squeeze(0).transpose(0, 1)
        cu_q = kwargs["cu_seq_lens_q"]
        max_len = (cu_q[1:] - cu_q[:-1]).max()
        out = k3_modeling.eager_attention(
            q3,
            k3,
            v3,
            q_cu_seqlens=cu_q,
            k_cu_seqlens=kwargs["cu_seq_lens_k"],
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
        )
        return out, None

    saved = [blk.veomni_attn for blk in model.vision_tower.encoder.blocks]
    try:
        for blk in model.vision_tower.encoder.blocks:
            blk.veomni_attn = local_eager_adapter
        with torch.no_grad():
            out_dict = model.vision_tower(pixel_values, grid_thws)
    finally:
        for blk, handle in zip(model.vision_tower.encoder.blocks, saved):
            blk.veomni_attn = handle
    assert len(out_handle) == len(out_dict) == len(grids)
    for a, b in zip(out_handle, out_dict):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)


def test_vision_flash_row_fails_at_construction_without_flash_attn(monkeypatch):
    """A 'flash_attention_2' vision selection on a host without flash-attn
    fails at construction with an attributable error -- the standard-family
    row's flash_attn package requirement, or HF's tower availability guard
    for the raw spelling -- never an ImportError raised from inside the
    modeling function at the first image forward (round-3 review). The
    tower has bound the same attention/standard rows as every other VL
    model since the round-9 family removal.

    The dependency attribution is pinned by mocking a supported platform
    and hiding only the flash_attn package (round-4 review): the mocked
    cuda lease with the GPU requirement forced to match isolates the
    package requirement, and the model build below must fail the same way
    before any forward."""
    # The conftest's device-lease spelling of the GPU device name: the
    # device-api sanity gate forbids the literal in non-whitelisted files.
    from veomni.ops.platform import GpuKernelRequirement

    monkeypatch.setattr("veomni.ops.registry.get_device_type", lambda: GpuKernelRequirement.device)
    monkeypatch.setattr(
        "veomni.ops.platform.requirement.GpuKernelRequirement.matches",
        lambda self: True,
    )
    monkeypatch.setattr(
        "veomni.ops.registry.is_package_available",
        lambda package: package != "flash_attn",
    )
    # the standard family's flash_attention_2 cuda row declares the flash_attn
    # package requirement for the mocked lease
    with pytest.raises(RuntimeError, match="flash_attn"):
        VeomniOp("attention", "standard", "flash_attention_2")
    from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import (
        KimiK3ForConditionalGeneration,
    )

    # and a model carrying the raw spelling never reaches forward: the tower
    # row binding (or HF's construction guard) refuses it at construction.
    # The text side and the root pin explicit eager defaults so the refusal
    # cannot be masked by the toy config's accelerator-bound flash-flavored
    # defaults under the CPU construction guard.
    cfg = _tiny_config()
    cfg.text_config.ops_implementation["attn_implementation"] = "eager"
    cfg.ops_implementation["attn_implementation"] = "eager"
    cfg.vision_config.ops_implementation["attn_implementation"] = "flash_attention_2"
    with pytest.raises((ValueError, RuntimeError)):
        KimiK3ForConditionalGeneration(cfg)


def _npu_available():
    try:
        return torch.npu.is_available()
    except AttributeError:
        return False


@pytest.mark.skipif(not _npu_available(), reason="needs the torch_npu fused vision kernel")
def test_npu_fused_vision_attention_matches_eager_across_multimodal_shapes():
    """The fused vision path (attention/standard row dispatch through the
    veomni_flash_attention_2 NPU adapter) stays numerically equivalent to the
    eager reference across multimodal shapes: single image, multi-image packed
    rows, mixed grids, video-style T>1 -- with grid_thws staged on host and on
    device. Repointed at the standard row in round-9 (the dedicated
    vision_attention family is gone; the qwen families exercise the same
    adapter on packed vision varlen)."""
    cfg = _tiny_config()
    cfg._attn_implementation = {"vision_config": "veomni_flash_attention_2"}
    fused = build_foundation_model(cfg, torch_dtype="float32", init_device="cpu")
    eager = _build_model()
    # same weights on both towers, or the comparison is meaningless
    eager.vision_tower.load_state_dict(fused.vision_tower.state_dict())
    fused.vision_tower.to("npu").eval()
    eager.vision_tower.to("npu").eval()
    # the comparison must actually exercise the two different kernels
    fused_impls = {blk.veomni_attn.impl for blk in fused.vision_tower.encoder.blocks}
    eager_impls = {blk.veomni_attn.impl for blk in eager.vision_tower.encoder.blocks}
    assert fused_impls == {"veomni_flash_attention_2"}
    assert eager_impls == {"eager"}

    scenarios = [
        [[1, 4, 4]],
        [[1, 4, 4], [1, 2, 2]],
        [[2, 2, 2]],
        [[1, 4, 4], [2, 2, 2], [1, 2, 2], [1, 4, 4]],
    ]
    for grids in scenarios:
        n_pix = sum(t * h * w for t, h, w in grids)
        pixel_values = torch.randn(n_pix, 3, 14, 14, device="npu")
        for grid_device in ("cpu", "npu"):
            grid_thws = torch.tensor(grids, device=grid_device)
            with torch.no_grad():
                out_fused = fused.vision_tower(pixel_values, grid_thws)
                out_eager = eager.vision_tower(pixel_values, grid_thws)
            assert len(out_fused) == len(out_eager) == len(grids)
            for a, b in zip(out_fused, out_eager):
                torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
