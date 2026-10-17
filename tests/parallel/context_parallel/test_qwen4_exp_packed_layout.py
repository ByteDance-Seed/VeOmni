"""Integer layout and host-metadata regressions; no accelerator is required."""

import ast
from pathlib import Path

import pytest
import torch

from veomni.distributed.context_parallel.dsa_cp import all_gather_compressed_rows
from veomni.models.transformers.qwen4_exp.packed_utils import _packed_segments, get_packed_qsa_layout


@pytest.mark.parametrize(
    "batch_size,seq_len,lengths,ratio,world",
    [
        (1, 32, (3, 5, 11, 13), 4, 4),
        (2, 16, (5, 11, 3, 13), 4, 4),
        (2, 16, (16, 16), 1, 8),
        (1, 16, (3, 3, 3, 3, 4), 4, 4),
        (1, 16, (1,) * 16, 4, 4),
        (1, 5, (2, 3), 8, 1),
    ],
)
def test_layout_matches_original_owner_major_order(batch_size, seq_len, lengths, ratio, world):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    segments = _packed_segments(batch_size, seq_len, cu)
    blocks = [
        (sample, batch, start)
        for sample, (batch, begin, end) in enumerate(segments)
        for start in range(begin, end - ratio + 1, ratio)
    ]
    local_len = seq_len // world
    gathered = [block for owner in range(world) for block in blocks if block[2] // local_len == owner]
    counts = tuple(sum(start // local_len == owner for _, _, start in blocks) for owner in range(world))
    for rank in range(world):
        layout = get_packed_qsa_layout(batch_size, seq_len, cu, ratio, rank, world, "cpu", packed_seq_lens=lengths)
        owned = [block for block in blocks if block[2] // local_len == rank]
        assert layout.counts == counts
        assert layout.block_starts.tolist() == [start for _, _, start in gathered]
        assert layout.block_segment_ids.tolist() == [sample for sample, _, _ in gathered]
        assert layout.block_batch_ids.tolist() == [batch for _, batch, _ in gathered]
        assert layout.block_owners.tolist() == [start // local_len for _, _, start in gathered]
        assert layout.local_batch_ids.tolist() == [batch for _, batch, _ in owned]
        assert layout.local_block_starts.tolist() == [start for _, _, start in owned]
        assert layout.gather_keep_indices.tolist() == [
            owner * max(counts) + row for owner, count in enumerate(counts) for row in range(count)
        ]
        for sample, _batch, start in owned:
            _, begin, end = segments[sample]
            assert begin <= start and start + ratio <= end
        for field in vars(layout).values():
            if isinstance(field, torch.Tensor):
                assert field.dtype == torch.long and not field.requires_grad


def test_cache_uses_boundaries_and_geometry_not_shape(monkeypatch):
    cache = {}
    first = get_packed_qsa_layout(1, 16, None, 4, 0, 2, "cpu", packed_seq_lens=(3, 13), layout_cache=cache)
    # No tensor-to-host access is needed on a validated host-length path.
    monkeypatch.setattr(
        torch.Tensor, "tolist", lambda self: (_ for _ in ()).throw(AssertionError("unexpected readback"))
    )
    again = get_packed_qsa_layout(1, 16, None, 4, 0, 2, "cpu", packed_seq_lens=(3, 13), layout_cache=cache)
    assert again is first
    monkeypatch.undo()
    changed = get_packed_qsa_layout(1, 16, None, 4, 0, 2, "cpu", packed_seq_lens=(5, 11), layout_cache=cache)
    other_rank = get_packed_qsa_layout(1, 16, None, 4, 1, 2, "cpu", packed_seq_lens=(3, 13), layout_cache=cache)
    ratio = get_packed_qsa_layout(1, 16, None, 2, 0, 2, "cpu", packed_seq_lens=(3, 13), layout_cache=cache)
    assert all(item is not first for item in (changed, other_rank, ratio))
    assert len(cache) == 4


def test_compressed_gather_host_counts_and_empty_gradient(monkeypatch):
    import veomni.distributed.context_parallel.dsa_cp as comm

    rows = torch.randn(1, 2, 3, requires_grad=True)
    monkeypatch.setattr(comm, "gather_outputs", lambda x, **kwargs: torch.cat((x, x), dim=1))
    expected = all_gather_compressed_rows(rows, torch.tensor([2, 1]), None)
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self: (_ for _ in ()).throw(AssertionError("device readback")))
    actual = all_gather_compressed_rows(rows, (2, 1), None, keep_indices=torch.tensor([0, 1, 2]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), rows)[0], torch.autograd.grad(expected.sum(), rows)[0]
    )
    empty = rows[:, :0]
    all_gather_compressed_rows(empty, (0, 0), None).sum().backward()
    assert rows.grad is not None and torch.count_nonzero(rows.grad) == 0


def _length_helpers():
    path = (
        Path(__file__).resolve().parents[3] / "veomni/models/transformers/qwen4_exp/qwen4_exp_gpu_patch_gen_config.py"
    )
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_qwen4_exp_validate_packed_seq_lens", "qwen4_exp_prepare_packed_lengths")
    ]
    for node in functions:
        node.decorator_list = []
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["qwen4_exp_prepare_packed_lengths"]


def test_host_lengths_reused_and_distinct_linear_boundaries(monkeypatch):
    prepare = _length_helpers()
    main = torch.tensor([0, 3, 16])
    linear = torch.tensor([0, 5, 16])
    kwargs = {"cu_seq_lens_q": main, "linear_attn_cu_seq_lens_q": linear}
    prepare(kwargs, 1, 16, refresh=True)
    assert kwargs["packed_seq_lens"] == (3, 13)
    assert kwargs["linear_attn_packed_seq_lens"] == (5, 11)
    main_tuple = kwargs["packed_seq_lens"]
    monkeypatch.setattr(
        torch.Tensor, "tolist", lambda self: (_ for _ in ()).throw(AssertionError("duplicate readback"))
    )
    prepare(kwargs, 1, 16)
    assert kwargs["packed_seq_lens"] is main_tuple
    monkeypatch.undo()
    kwargs["cu_seq_lens_q"] = torch.tensor([0, 7, 16])
    prepare(kwargs, 1, 16, refresh=True)
    assert kwargs["packed_seq_lens"] == (7, 9)
    assert kwargs["linear_attn_packed_seq_lens"] == (5, 11)


def test_direct_entry_rejects_stale_lengths_and_changed_boundaries():
    prepare = _length_helpers()
    main = torch.tensor([0, 5, 16])
    linear = torch.tensor([0, 7, 16])
    kwargs = {
        "cu_seq_lens_q": main,
        "linear_attn_cu_seq_lens_q": linear,
        "packed_seq_lens": (3, 13),
        "linear_attn_packed_seq_lens": (3, 13),
    }
    prepare(kwargs, 1, 16)
    assert kwargs["packed_seq_lens"] == (5, 11)
    assert kwargs["linear_attn_packed_seq_lens"] == (7, 9)
    main[1] = 6
    prepare(kwargs, 1, 16)
    assert kwargs["packed_seq_lens"] == (6, 10)
    kwargs["linear_attn_cu_seq_lens_q"] = torch.tensor([0, 8, 16])
    prepare(kwargs, 1, 16)
    assert kwargs["linear_attn_packed_seq_lens"] == (8, 8)


def test_inference_boundary_tensors_refresh_without_version_counter():
    prepare = _length_helpers()
    with torch.inference_mode():
        kwargs = {"cu_seq_lens_q": torch.tensor([0, 5, 16]), "packed_seq_lens": (3, 13)}
        prepare(kwargs, 1, 16)
        assert kwargs["packed_seq_lens"] == (5, 11)
        kwargs["cu_seq_lens_q"][1] = 7
        prepare(kwargs, 1, 16)
        assert kwargs["packed_seq_lens"] == (7, 9)


@pytest.mark.parametrize("sp_enabled", [False, True])
def test_gdn_consumes_independent_host_tuple_without_readback(monkeypatch, sp_enabled):
    from types import SimpleNamespace

    path = (
        Path(__file__).resolve().parents[3] / "veomni/models/transformers/qwen4_exp/qwen4_exp_gpu_patch_gen_config.py"
    )
    names = {
        "_qwen4_exp_validate_packed_seq_lens",
        "qwen4_exp_gdn_non_sp_forward",
        "qwen4_exp_gated_deltanet_forward_patched",
    }
    functions = [
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for node in functions:
        node.decorator_list = []
    namespace = {
        "get_parallel_state": lambda: SimpleNamespace(sp_enabled=sp_enabled, sp_size=4 if sp_enabled else 1),
        "apply_mask_to_padding_states": lambda x, _: x,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    observed = []
    validate = namespace["_qwen4_exp_validate_packed_seq_lens"]

    def record(lengths, batch, size):
        observed.append(tuple(lengths))
        return validate(lengths, batch, size)

    class ProjectionReached(Exception):
        pass

    def project(_):
        raise ProjectionReached

    namespace["_qwen4_exp_validate_packed_seq_lens"] = record
    main = torch.tensor([0, 3, 16])
    linear = torch.tensor([0, 5, 16])
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self: (_ for _ in ()).throw(AssertionError("GDN readback")))
    with pytest.raises(ProjectionReached):
        namespace["qwen4_exp_gated_deltanet_forward_patched"](
            SimpleNamespace(in_proj_qkv=project),
            torch.zeros(1, 4 if sp_enabled else 16, 2),
            cu_seq_lens_q=main,
            linear_attn_cu_seq_lens_q=linear,
            packed_seq_lens=(3, 13),
            linear_attn_packed_seq_lens=(5, 11),
        )
    assert observed == [(5, 11)]


@pytest.mark.parametrize("boundary", [[5, 8, 21], [0, 3, 15], [0, 0, 16], [[0, 16]]])
@pytest.mark.parametrize("linear", [False, True])
def test_prepare_rejects_invalid_cumulative_boundaries(boundary, linear):
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 3, 16])}
    kwargs["linear_attn_cu_seq_lens_q" if linear else "cu_seq_lens_q"] = torch.tensor(boundary)
    with pytest.raises(ValueError):
        _length_helpers()(kwargs, 1, 16)
