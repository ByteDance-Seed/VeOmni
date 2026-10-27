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

"""HF mask builders registered on ``ALL_MASK_ATTENTION_FUNCTIONS``."""

from __future__ import annotations

import contextlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers import PreTrainedConfig
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, create_causal_mask

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.ops.attention.attention_cases import (
    dense_2d_mask,
    flex_2d_mask,
    flex_visible,
    magi_2d_mask,
    materialize_magi_mask,
)
from veomni.ops.kernels.attention import ulysses as ulysses_mask
from veomni.ops.kernels.attention.mask import flex as flex_mask
from veomni.ops.kernels.attention.mask import magi as magi_mask
from veomni.ops.kernels.attention.mask import sdpa as sdpa_mask
from veomni.ops.mask import (
    MagiAttentionMask,
    flash_attention_mask_builder,
    flex_attention_mask_builder,
    magi_attention_mask_builder,
    sdpa_attention_mask_builder,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_mask_builder_return_types():
    assert flash_attention_mask_builder(1, 4, 4) is None
    sdpa = sdpa_attention_mask_builder(1, 4, 4, device="cpu", allow_is_causal_skip=False)
    assert isinstance(sdpa, torch.Tensor)
    assert sdpa.dtype == torch.bool
    assert sdpa.shape == (1, 1, 4, 4)
    flex = flex_attention_mask_builder(1, 4, 4, device="cpu")
    assert isinstance(flex, BlockMask)
    magi = magi_attention_mask_builder(1, 4, 4, device="cpu")
    assert isinstance(magi, MagiAttentionMask)
    assert magi.q_ranges.dtype == torch.int32
    assert magi.q_ranges.shape == (1, 2)


@pytest.mark.parametrize("allow_skip", (None, False, True))
def test_sdpa_canonical_causal_skip_remains_optional(allow_skip):
    """Ordinary square causal attention retains its mask-elision optimization."""
    kwargs = {} if allow_skip is None else {"allow_is_causal_skip": allow_skip}
    mask = sdpa_attention_mask_builder(1, 4, 4, device="cpu", skip_ulysses=True, **kwargs)
    if allow_skip is False:
        torch.testing.assert_close(mask[0, 0], torch.ones(4, 4, dtype=torch.bool).tril())
    else:
        assert mask is None


def test_flash_mask_builder_preserves_padding_and_elides_all_valid_mask():
    padded = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    torch.testing.assert_close(
        flash_attention_mask_builder(1, 4, 4, attention_mask=padded),
        padded,
    )
    assert flash_attention_mask_builder(1, 4, 4, attention_mask=torch.ones_like(padded)) is None


def test_hf_create_causal_mask_uses_registered_builders():
    embeds = torch.randn(1, 8, 16)
    position_ids = torch.arange(8).unsqueeze(0)

    flex_config = PreTrainedConfig()
    flex_config._attn_implementation = "veomni_flex_attention"
    flex = create_causal_mask(flex_config, embeds, None, None, position_ids)
    assert isinstance(flex, BlockMask)
    assert flex.shape == (1, 1, 8, 8)

    magi_config = PreTrainedConfig()
    magi_config._attn_implementation = "veomni_magi_attention"
    magi = create_causal_mask(magi_config, embeds, None, None, position_ids)
    assert isinstance(magi, MagiAttentionMask)
    assert magi.q_ranges.tolist() == [[0, 8]]
    assert magi.k_ranges.tolist() == [[0, 8]]


def test_hf_create_causal_mask_preserves_flash_like_padding():
    config = PreTrainedConfig()
    config._attn_implementation = "veomni_flash_attention_2"
    embeds = torch.randn(1, 4, 16)
    position_ids = torch.arange(4).unsqueeze(0)
    attention_2d = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    mask = create_causal_mask(config, embeds, attention_2d, None, position_ids)

    torch.testing.assert_close(mask, attention_2d)


def test_flash_like_hf_names_share_the_flash_mask_builder():
    from veomni.ops.kernels.attention.mask.flash import flash_attention_mask_builder

    for name in (
        "veomni_flash_attention_2",
        "veomni_flash_attention_2_hub",
        "veomni_flash_attention_3",
        "veomni_flash_attention_3_hub",
        "veomni_flash_attention_4",
        "veomni_sage_attention",
    ):
        assert ALL_MASK_ATTENTION_FUNCTIONS[name] is flash_attention_mask_builder


def test_flex_and_sdpa_2d_padding_masks_align():
    attention_2d = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    flex = flex_attention_mask_builder(1, 4, 4, attention_mask=attention_2d, device="cpu")
    sdpa = sdpa_attention_mask_builder(
        1,
        4,
        4,
        attention_mask=attention_2d,
        device="cpu",
        allow_is_causal_skip=False,
    )
    assert sdpa is not None
    torch.testing.assert_close(flex_visible(flex, 4, 4), sdpa[0, 0])
    assert not bool(sdpa[0, 0, :, -1].any())


@pytest.mark.parametrize(
    "attention_2d",
    (
        torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
    ),
)
def test_magi_hf_builder_rejects_implicit_2d_visibility(attention_2d):
    with pytest.raises(ValueError, match="cannot recover packed boundaries"):
        magi_attention_mask_builder(1, 4, 4, attention_mask=attention_2d, device="cpu")


def test_magi_hf_builder_rejects_incompatible_causal_offsets():
    with pytest.raises(ValueError, match="bottom-right aligned"):
        magi_attention_mask_builder(1, 2, 4, device="cpu")


def _sync_ulysses_state(*, size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(ulysses_size=size, async_enabled=False)


def _patch_mask_ulysses(monkeypatch, *modules, apply: bool, size: int = 2) -> None:
    state = _sync_ulysses_state(size=size)
    for module in (*modules, ulysses_mask):
        if hasattr(module, "should_apply_ulysses"):
            monkeypatch.setattr(
                module,
                "should_apply_ulysses",
                lambda *, skip_ulysses=False: apply and not skip_ulysses,
            )
        if hasattr(module, "get_parallel_state"):
            monkeypatch.setattr(module, "get_parallel_state", lambda: state)


def test_flex_ulysses_2d_mask_length_aligns(monkeypatch):
    _patch_mask_ulysses(monkeypatch, flex_mask, apply=True)
    full_2d = torch.ones(1, 8, dtype=torch.bool)
    flex = flex_attention_mask_builder(1, 4, 4, attention_mask=full_2d, device="cpu")
    assert tuple(flex.shape[-2:]) == (8, 8)


@pytest.mark.parametrize(("impl", "module"), (("veomni_flex_attention", flex_mask), ("veomni_sdpa", sdpa_mask)))
def test_hf_create_causal_mask_with_ulysses_uses_global_lengths(monkeypatch, impl, module):
    """HF takes ``kv_length`` from the unsliced 2D mask, while Q stays the local shard."""
    _patch_mask_ulysses(monkeypatch, module, apply=True)
    config = PreTrainedConfig()
    config._attn_implementation = impl
    local_embeds = torch.randn(1, 4, 16)
    full_2d = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)

    mask = create_causal_mask(config, local_embeds, full_2d, None, torch.arange(4).unsqueeze(0))

    visible = flex_visible(mask, 8, 8) if impl == "veomni_flex_attention" else mask[0, 0]
    torch.testing.assert_close(visible, torch.ones(8, 8, dtype=torch.bool).tril() & full_2d)


@pytest.mark.parametrize(
    ("builder", "module", "attention_mask", "match"),
    [
        (
            flex_attention_mask_builder,
            flex_mask,
            torch.ones(1, 4, dtype=torch.bool),
            "post-Ulysses key length",
        ),
        (
            sdpa_attention_mask_builder,
            sdpa_mask,
            torch.ones(1, 4, dtype=torch.bool),
            "post-Ulysses key length",
        ),
    ],
)
def test_flex_and_sdpa_ulysses_reject_incomplete_2d_mask(monkeypatch, builder, module, attention_mask, match):
    _patch_mask_ulysses(monkeypatch, module, apply=True)
    with pytest.raises(ValueError, match=match):
        builder(1, 4, 4, attention_mask=attention_mask, device="cpu")


@pytest.mark.parametrize(
    ("builder", "module"),
    (
        (flex_attention_mask_builder, flex_mask),
        (sdpa_attention_mask_builder, sdpa_mask),
    ),
)
def test_flex_and_sdpa_ulysses_allow_canonical_mask_without_metadata(monkeypatch, builder, module):
    _patch_mask_ulysses(monkeypatch, module, apply=True)
    mask = builder(1, 4, 4, device="cpu", allow_is_causal_skip=False)
    assert tuple(mask.shape[-2:]) == (8, 8)


@pytest.mark.parametrize(
    ("builder", "module"),
    (
        (flex_attention_mask_builder, flex_mask),
        (sdpa_attention_mask_builder, sdpa_mask),
    ),
)
def test_flex_and_sdpa_ulysses_reject_custom_mask_without_metadata(monkeypatch, builder, module):
    _patch_mask_ulysses(monkeypatch, module, apply=True)
    with pytest.raises(ValueError, match="full-sequence metadata for a custom mask function"):
        builder(1, 4, 4, mask_function=lambda *args: True, device="cpu")


@pytest.mark.parametrize(
    ("builder", "module"),
    (
        (flex_attention_mask_builder, flex_mask),
        (sdpa_attention_mask_builder, sdpa_mask),
    ),
)
def test_flex_and_sdpa_ulysses_allow_sliding_window_without_metadata(monkeypatch, builder, module):
    _patch_mask_ulysses(monkeypatch, module, apply=True)
    mask = builder(1, 4, 4, sliding_window=2, device="cpu")
    assert tuple(mask.shape[-2:]) == (8, 8)
    visible = flex_visible(mask, 8, 8) if builder is flex_attention_mask_builder else mask[0, 0]
    positions = torch.arange(8)
    distance = positions[:, None] - positions[None, :]
    torch.testing.assert_close(visible, (distance >= 0) & (distance < 2))


def test_magi_hf_builder_expands_ulysses_local_lengths(monkeypatch):
    _patch_mask_ulysses(monkeypatch, magi_mask, apply=True)
    mask = magi_attention_mask_builder(1, 4, 4, device="cpu")
    assert mask.q_ranges.tolist() == [[0, 8]]
    assert mask.k_ranges.tolist() == [[0, 8]]


def test_magi_hf_builder_keeps_cached_offsets_unsupported_with_ulysses(monkeypatch):
    _patch_mask_ulysses(monkeypatch, magi_mask, apply=True)
    with pytest.raises(ValueError, match="with Ulysses does not support cached mask offsets"):
        magi_attention_mask_builder(1, 2, 4, q_offset=2, device="cpu")


@pytest.mark.parametrize(
    ("builder", "module", "kwargs"),
    [
        (flex_attention_mask_builder, flex_mask, {"attention_mask": torch.ones(1, 4, dtype=torch.bool)}),
        (
            sdpa_attention_mask_builder,
            sdpa_mask,
            {"attention_mask": torch.ones(1, 4, dtype=torch.bool), "allow_is_causal_skip": False},
        ),
        (magi_attention_mask_builder, magi_mask, {}),
    ],
)
def test_mask_builders_keep_local_lengths_when_async_or_skipped(monkeypatch, builder, module, kwargs):
    _patch_mask_ulysses(monkeypatch, module, apply=False)
    local = builder(1, 4, 4, device="cpu", **kwargs)
    _patch_mask_ulysses(monkeypatch, module, apply=True)
    skipped = builder(1, 4, 4, device="cpu", skip_ulysses=True, **kwargs)
    if builder is magi_attention_mask_builder:
        assert local.q_ranges.tolist() == [[0, 4]]
        assert local.k_ranges.tolist() == [[0, 4]]
        assert skipped.q_ranges.tolist() == [[0, 4]]
        assert skipped.k_ranges.tolist() == [[0, 4]]
    elif builder is flex_attention_mask_builder:
        assert tuple(local.shape[-2:]) == (4, 4)
        assert tuple(skipped.shape[-2:]) == (4, 4)
    else:
        assert local.shape[-2:] == (4, 4)
        assert skipped.shape[-2:] == (4, 4)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"batch_size": 2}, "physical batch size 1"),
        ({"q_offset": 1}, "bottom-right aligned"),
        ({"mask_function": lambda *args: True}, "canonical causal or bidirectional"),
    ],
)
def test_magi_hf_builder_rejects_unsupported_registry_inputs(override, match):
    arguments = {"batch_size": 1, "q_length": 4, "kv_length": 4, "device": "cpu", **override}
    with pytest.raises(ValueError, match=match):
        magi_attention_mask_builder(**arguments)


def test_magi_hf_builder_does_not_recover_packed_visibility_from_position_ids():
    config = PreTrainedConfig()
    config._attn_implementation = "veomni_magi_attention"
    embeds = torch.randn(1, 8, 16)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    packed_position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    with pytest.raises(ValueError, match="cannot recover packed boundaries"):
        create_causal_mask(config, embeds, attention_mask, None, packed_position_ids)


def test_magi_hf_builder_ignores_packed_boundary_kwargs():
    """The HF builder is unpacked-only; packed isolation needs from_cu_seqlens."""
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    mask = magi_attention_mask_builder(
        1,
        4,
        4,
        device="cpu",
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        q_ranges=torch.tensor([[0, 2], [2, 4]], dtype=torch.int32),
        k_ranges=torch.tensor([[0, 2], [2, 4]], dtype=torch.int32),
    )
    assert mask.q_ranges.tolist() == [[0, 4]]
    assert mask.k_ranges.tolist() == [[0, 4]]
    packed = MagiAttentionMask.from_cu_seqlens(cu_seqlens)
    assert packed.q_ranges.tolist() == [[0, 2], [2, 4]]
    assert packed.k_ranges.tolist() == [[0, 2], [2, 4]]


def test_magi_from_ranges_casts_ffa_contract():
    mask = MagiAttentionMask.from_ranges(
        torch.tensor([[0, 4]]),
        torch.tensor([[0, 4]]),
        torch.tensor([1]),
    )
    assert mask.q_ranges.dtype == torch.int32
    assert mask.q_ranges.tolist() == [[0, 4]]
    assert mask.attn_type_map is not None
    assert mask.attn_type_map.tolist() == [1]


@pytest.mark.parametrize("name", ("q_ranges", "k_ranges", "attn_type_map"))
def test_magi_from_ranges_rejects_non_integer_metadata(name):
    values = {
        "q_ranges": torch.tensor([[0, 4]]),
        "k_ranges": torch.tensor([[0, 4]]),
        "attn_type_map": torch.tensor([1]),
    }
    values[name] = values[name].to(torch.float32)
    with pytest.raises(TypeError, match="dtype int32 or int64"):
        MagiAttentionMask.from_ranges(**values)


def test_magi_from_ranges_preserves_mixed_visibility():
    q_ranges = torch.tensor([[0, 4], [4, 8], [4, 8]], dtype=torch.int32)
    k_ranges = torch.tensor([[0, 4], [0, 4], [4, 8]], dtype=torch.int32)
    attn_type_map = torch.tensor([1, 0, 0], dtype=torch.int32)
    mask = MagiAttentionMask.from_ranges(q_ranges, k_ranges, attn_type_map)
    torch.testing.assert_close(mask.q_ranges, q_ranges)
    torch.testing.assert_close(mask.k_ranges, k_ranges)
    torch.testing.assert_close(mask.attn_type_map, attn_type_map)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"q_ranges": torch.tensor([[0, 8]], dtype=torch.int64)}, "dtype int32"),
        ({"q_ranges": torch.tensor([0, 8], dtype=torch.int32)}, "shape \\[num_ranges, 2\\]"),
        ({"q_ranges": torch.empty(0, 2, dtype=torch.int32)}, "shape \\[num_ranges, 2\\]"),
        ({"q_ranges": torch.tensor([[-1, 8]], dtype=torch.int32)}, "0 <= start < end"),
        ({"k_ranges": torch.tensor([[4, 4]], dtype=torch.int32)}, "0 <= start < end"),
        ({"attn_type_map": torch.tensor([4], dtype=torch.int32)}, "attn_type_map values must be in \\[0, 3\\]"),
    ],
)
def test_magi_mask_rejects_invalid_metadata(overrides, match):
    values = {
        "q_ranges": torch.tensor([[0, 8]], dtype=torch.int32),
        "k_ranges": torch.tensor([[0, 8]], dtype=torch.int32),
        "attn_type_map": None,
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        MagiAttentionMask(**values)


def test_2d_masks_match_dense_visibility():
    sequence_length = 16
    device = torch.device("cpu")
    dense = dense_2d_mask(sequence_length, device)
    magi = magi_2d_mask(sequence_length, device)
    flex = flex_2d_mask(sequence_length, device)
    torch.testing.assert_close(materialize_magi_mask(magi, sequence_length), dense)
    torch.testing.assert_close(flex_visible(flex, sequence_length, sequence_length), dense[0, 0])


def _reset_flex_compile_cache(monkeypatch):
    """Isolate the builder singleton so tests do not inherit a compiled callable."""
    monkeypatch.setattr(flex_mask, "_COMPILED_CREATE_BLOCK_MASK", None)


def _spy_torch_compile(monkeypatch):
    """Observe compile requests; this spy does not establish Dynamo graph reuse."""
    compiled = []

    def fake_compile(fn, *args, **kwargs):
        """Leave execution eager while recording which builder was selected."""
        compiled.append(fn)
        return fn

    monkeypatch.setattr(flex_mask.torch, "compile", fake_compile)
    return compiled


def test_flex_mask_builder_compiles_create_block_mask_by_default(monkeypatch):
    """The default path should cache its compile wrapper across builder calls."""
    _reset_flex_compile_cache(monkeypatch)
    compiled = _spy_torch_compile(monkeypatch)
    mask = flex_attention_mask_builder(1, 4, 4, device="cpu")
    assert isinstance(mask, BlockMask)
    assert compiled == [flex_mask.create_block_mask]
    flex_attention_mask_builder(1, 4, 4, device="cpu")
    assert compiled == [flex_mask.create_block_mask]


@pytest.mark.parametrize(
    ("device", "compute_capability", "expected"),
    [
        (None, 0, 128),
        ("cpu", 0, 128),
        ("cuda", 80, 128),
        ("cuda", 90, 128),
        ("cuda", 100, (256, 128)),
        ("cuda", 110, (256, 128)),
    ],
)
def test_flex_mask_block_size_follows_nvidia_architecture(monkeypatch, device, compute_capability, expected):
    monkeypatch.setattr(flex_mask.torch.version, "hip", None)
    monkeypatch.setattr(flex_mask, "get_gpu_compute_capability", lambda selected_device: compute_capability)
    assert flex_mask._flex_block_size(device) == expected


def test_flex_mask_block_size_keeps_rocm_default(monkeypatch):
    monkeypatch.setattr(flex_mask.torch.version, "hip", "test-rocm")
    monkeypatch.setattr(flex_mask, "get_gpu_compute_capability", lambda selected_device: 100)
    assert flex_mask._flex_block_size("cuda") == 128


def test_flex_mask_builder_passes_sm100_block_size_to_pytorch(monkeypatch):
    captured = []
    real = flex_mask.create_block_mask

    def spy(*args, **kwargs):
        captured.append(kwargs["BLOCK_SIZE"])
        return real(*args, **kwargs)

    monkeypatch.setattr(flex_mask, "create_block_mask", spy)
    monkeypatch.setattr(flex_mask, "_flex_block_size", lambda device: (256, 128))
    mask = flex_attention_mask_builder(1, 256, 128, device="cpu", compile_block_mask=False)
    assert isinstance(mask, BlockMask)
    assert mask.BLOCK_SIZE == (256, 128)
    assert captured == [(256, 128)]


def test_flex_mask_builder_compile_block_mask_false_skips_compile(monkeypatch):
    """Explicit eager mode must also suppress HF's deprecated compile flag."""
    _reset_flex_compile_cache(monkeypatch)
    compiled = _spy_torch_compile(monkeypatch)
    captured = []
    real = flex_mask.create_block_mask

    def spy(*args, **kwargs):
        captured.append(kwargs.get("_compile"))
        return real(*args, **kwargs)

    monkeypatch.setattr(flex_mask, "create_block_mask", spy)
    mask = flex_attention_mask_builder(1, 4, 4, device="cpu", compile_block_mask=False)
    assert isinstance(mask, BlockMask)
    assert compiled == []
    assert captured == [False]


def test_flex_mask_builder_compile_flag_preserves_causal_visibility(monkeypatch):
    """Real compilation must preserve the causal predicate, not just mask type."""
    _reset_flex_compile_cache(monkeypatch)
    compiled = flex_attention_mask_builder(1, 4, 4, device="cpu", compile_block_mask=True)
    eager = flex_attention_mask_builder(1, 4, 4, device="cpu", compile_block_mask=False)
    torch.testing.assert_close(flex_visible(compiled, 4, 4), flex_visible(eager, 4, 4))
    expected = torch.ones(4, 4, dtype=torch.bool).tril()
    torch.testing.assert_close(flex_visible(compiled, 4, 4), expected)


def _packed_builder_inputs(length, segments, rows, cross_attention, boundary_shift):
    """Construct packed inputs and an independent per-document visibility oracle."""
    q_cuts = [0] + [length * i // segments for i in range(1, segments)] + [length]
    if segments > 1:
        q_cuts[1] += boundary_shift
    k_cuts = [0]
    for i, (start, end) in enumerate(zip(q_cuts, q_cuts[1:])):
        k_cuts.append(k_cuts[-1] + end - start + (3 * (i + 1) if cross_attention else 0))
    kv_length = k_cuts[-1]
    expected = torch.zeros(length, kv_length, dtype=torch.bool)
    for (qs, qe), (ks, ke) in zip(zip(q_cuts, q_cuts[1:]), zip(k_cuts, k_cuts[1:]), strict=True):
        # Cached Q tokens are the suffix of the paired K document.
        expected[qs:qe, ks:ke] = torch.ones(qe - qs, ke - ks, dtype=torch.bool).tril(diagonal=(ke - ks) - (qe - qs))
    q_flat = [0] + [row * length + end for row in range(rows) for end in q_cuts[1:]]
    k_flat = [0] + [row * kv_length + end for row in range(rows) for end in k_cuts[1:]]
    kwargs = {
        "batch_size": rows,
        "q_length": length,
        "kv_length": kv_length,
        "device": "cpu",
        "skip_ulysses": True,
        "cu_seqlens": torch.tensor(q_flat, dtype=torch.int32),
    }
    if cross_attention:
        kwargs["cu_seqlens_k"] = torch.tensor(k_flat, dtype=torch.int32)
    return kwargs, expected


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None, reason="Compiled masks require the compiler environment"
)
@pytest.mark.parametrize(("rows", "cross_attention"), [(1, False), (2, False), (1, True)])
def test_packed_flex_builder_reuses_real_graphs_and_preserves_visibility(
    _scoped_flex_compile_state, rows, cross_attention
):
    """Exercise the public cu_seqlens path; attention-only graph tests bypass it."""
    lengths = (65, 129, 193, 259, 389, 517, 645, 773, 901, 1027, 1283, 1541)
    cases = [(length, 3, 0) for length in lengths]
    cases += [(389, 3, shift) for shift in (1, 3, 5, 11)]
    if not cross_attention:
        cases += [(389, segments, 0) for segments in (1, 2, 4, 7, 11, 17)]
    cases += [cases[i] for i in (0, 4, 1, 0)]
    fields = (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
    )
    initial_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    # A regression must raise rather than silently fall back to eager when
    # new inputs exhaust the graph budget. Production limits stay unchanged.
    with torch._dynamo.config.patch(recompile_limit=4, fail_on_recompile_limit_hit=True):
        for case_index, (length, segments, shift) in enumerate(cases):
            if case_index == len(cases) - 4:
                repeat_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            kwargs, expected = _packed_builder_inputs(length, segments, rows, cross_attention, shift)
            compiled = flex_attention_mask_builder(**kwargs)
            eager = flex_attention_mask_builder(**kwargs, compile_block_mask=False)
            for field in fields:
                torch.testing.assert_close(getattr(compiled, field), getattr(eager, field))
            q = torch.arange(length)[:, None]
            k = torch.arange(kwargs["kv_length"])[None, :]
            for batch in range(rows):
                torch.testing.assert_close(compiled.mask_mod(batch, 0, q, k), expected)
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"] - initial_graphs
        assert 1 <= graphs <= 4
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == repeat_graphs


@pytest.mark.parametrize("fails", [False, True])
def test_flex_builder_keeps_hf_functions_during_predicate_evaluation(fails):
    """HF helpers must remain original throughout successful and failing calls."""
    from transformers import masking_utils

    original_builder = masking_utils.create_block_mask
    original_offsets = masking_utils.add_offsets_to_mask_function

    def predicate(batch_idx, head_idx, q_idx, kv_idx):
        """Observe HF globals while this invocation still evaluates its mask."""
        assert masking_utils.create_block_mask is original_builder
        assert masking_utils.add_offsets_to_mask_function is original_offsets
        if fails:
            raise RuntimeError("predicate failed")
        return q_idx >= kv_idx

    context = pytest.raises(RuntimeError, match="predicate failed") if fails else contextlib.nullcontext()
    with context:
        flex_attention_mask_builder(
            1, 4, 6, q_offset=2, mask_function=predicate, device="cpu", compile_block_mask=False
        )
    assert masking_utils.create_block_mask is original_builder
    assert masking_utils.add_offsets_to_mask_function is original_offsets


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("second_fails", [False, True])
def test_flex_builder_overlapping_calls_leave_native_hf_independent(monkeypatch, compiled, second_fails):
    """Interleave A-enter/B-enter/A-exit/B-exit while a native HF consumer runs."""
    from transformers import masking_utils

    original_builder = masking_utils.create_block_mask
    original_offsets = masking_utils.add_offsets_to_mask_function
    # Isolate routing and lifetime, rather than concurrent compiler startup.
    # Real compilation and graph reuse are exercised in separate regressions.
    monkeypatch.setattr(flex_mask, "_compiled_create_block_mask", lambda: original_builder)
    # Quarantine state even when this regression is run against the old patch.
    monkeypatch.setattr(masking_utils, "create_block_mask", original_builder)
    monkeypatch.setattr(masking_utils, "add_offsets_to_mask_function", original_offsets)
    first_entered, second_entered, native_finished, first_finished = (Event() for _ in range(4))
    observed = []

    def first_predicate(batch_idx, head_idx, q_idx, kv_idx):
        """Keep A live until B and the native HF consumer have entered."""
        first_entered.set()
        assert second_entered.wait(20), "second invocation never entered"
        assert native_finished.wait(20), "native HF invocation never finished"
        observed.append((masking_utils.create_block_mask, masking_utils.add_offsets_to_mask_function))
        return q_idx >= kv_idx

    def second_predicate(batch_idx, head_idx, q_idx, kv_idx):
        """Make B exit after A, including when user predicate evaluation fails."""
        second_entered.set()
        assert first_finished.wait(20), "first invocation never finished"
        observed.append((masking_utils.create_block_mask, masking_utils.add_offsets_to_mask_function))
        if second_fails:
            raise RuntimeError("second predicate failed")
        return q_idx >= kv_idx

    def build_first():
        """Publish A completion even when construction raises."""
        try:
            return flex_attention_mask_builder(
                1,
                65,
                69,
                q_offset=7,
                kv_offset=3,
                mask_function=first_predicate,
                device="cpu",
                compile_block_mask=compiled,
            )
        finally:
            first_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(build_first)
        try:
            assert first_entered.wait(20), "first invocation never entered"
            second = pool.submit(
                flex_attention_mask_builder,
                1,
                129,
                133,
                q_offset=9,
                kv_offset=5,
                mask_function=second_predicate,
                device="cpu",
                compile_block_mask=compiled,
            )
            assert second_entered.wait(20), "second invocation never entered"
            # Native HF helpers must not inherit either call's lengths/device.
            native_predicate = masking_utils.add_offsets_to_mask_function(
                masking_utils.causal_mask_function, q_offset=2, kv_offset=0
            )
            native = masking_utils.create_block_mask(
                native_predicate, B=1, H=None, Q_LEN=193, KV_LEN=195, device="cpu", _compile=False
            )
            torch.testing.assert_close(flex_visible(native, 193, 195), torch.ones(193, 195, dtype=torch.bool).tril(2))
        finally:
            native_finished.set()
        first_mask = first.result(timeout=20)
        context = (
            pytest.raises(RuntimeError, match="second predicate failed") if second_fails else contextlib.nullcontext()
        )
        with context:
            second_mask = second.result(timeout=20)
        if not second_fails:
            q = torch.arange(129)[:, None]
            k = torch.arange(133)[None, :]
            torch.testing.assert_close(second_mask.mask_mod(0, 0, q, k), q + 9 >= k + 5)
    q = torch.arange(65)[:, None]
    k = torch.arange(69)[None, :]
    torch.testing.assert_close(first_mask.mask_mod(0, 0, q, k), q + 7 >= k + 3)
    assert observed and all(pair == (original_builder, original_offsets) for pair in observed)
    assert masking_utils.create_block_mask is original_builder
    assert masking_utils.add_offsets_to_mask_function is original_offsets


@pytest.fixture
def _scoped_flex_compile_state(monkeypatch):
    """Reset Dynamo/Inductor and the builder singleton once around a compiled-mask test."""
    torch.compiler.reset()
    _reset_flex_compile_cache(monkeypatch)
    try:
        yield
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("compiled", [False, True])
def test_flex_local_builder_matches_pinned_hf_padding_and_offsets(_scoped_flex_compile_state, compiled):
    """Preserve native HF visibility and all eight sparse tables across window/padding combinations."""
    from transformers import masking_utils

    if compiled and importlib.util.find_spec("triton") is None:
        pytest.skip("Compiled masks require the compiler environment")

    for window in (None, 3):
        for padding in ("none", "all_valid_short", "short", "long"):
            padding_length = 6 if padding in ("short", "all_valid_short") else 15
            attention_mask = None
            if padding != "none":
                attention_mask = torch.ones(2, padding_length, dtype=torch.bool)
                if padding != "all_valid_short":
                    attention_mask[0, 4] = False
                    attention_mask[1, 5] = False
            native_predicate = masking_utils.causal_mask_function
            if window is not None:
                native_predicate = masking_utils.and_masks(
                    native_predicate, masking_utils.sliding_window_overlay(window)
                )
            kwargs = dict(
                batch_size=2,
                q_length=5,
                kv_length=9,
                q_offset=7,
                kv_offset=3,
                attention_mask=attention_mask,
                device="cpu",
            )
            with torch.compiler.set_stance("force_eager"):
                expected = masking_utils.flex_attention_mask(**kwargs, mask_function=native_predicate)
            # CPU Inductor's vectorized bool-padding lowering currently emits
            # incompatible VecMask types. Keep real compilation, but use scalar
            # CPU codegen here; FLASH integration tests retain normal GPU codegen.
            with torch._inductor.config.patch({"cpp.simdlen": 1}):
                actual = flex_attention_mask_builder(**kwargs, sliding_window=window, compile_block_mask=compiled)
            q, k = torch.arange(5)[:, None], torch.arange(9)[None, :]
            for batch in range(2):
                torch.testing.assert_close(
                    actual.mask_mod(batch, 0, q, k),
                    expected.mask_mod(batch, 0, q, k),
                    msg=f"compiled={compiled}, window={window}, padding={padding}, batch={batch}",
                )
            for field in (
                "kv_num_blocks",
                "kv_indices",
                "full_kv_num_blocks",
                "full_kv_indices",
                "q_num_blocks",
                "q_indices",
                "full_q_num_blocks",
                "full_q_indices",
            ):
                torch.testing.assert_close(
                    getattr(actual, field),
                    getattr(expected, field),
                    msg=f"compiled={compiled}, window={window}, padding={padding}, field={field}",
                )
            assert actual.seq_lengths == expected.seq_lengths
            assert actual.BLOCK_SIZE == expected.BLOCK_SIZE
