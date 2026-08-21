from dataclasses import replace

import pytest
import torch

from veomni.distributed.context_parallel.gdn_lossless import (
    align_gdn_varlen_chunks,
    aligned_gdn_cu_seqlens,
    make_state_participation,
    make_state_template,
    unpad_gdn_varlen_output,
)
from veomni.distributed.context_parallel.gdn_ownership import (
    GdnCopySpan,
    build_gdn_lossless_plan,
    validate_gdn_lossless_plan,
)


_LENGTHS = [0, 1, 63, 64, 65, 72, 128, 256]


def _copy_rows(source: torch.Tensor, spans: tuple[GdnCopySpan, ...], rows: int) -> torch.Tensor:
    output = source.new_zeros((rows, source.size(1)))
    for span in spans:
        output[span.destination_start : span.destination_start + span.length].copy_(
            source[span.source_start : span.source_start + span.length]
        )
    return output


def _transpose_wire(
    sends: list[torch.Tensor],
    input_splits: list[tuple[int, ...]],
) -> list[torch.Tensor]:
    world_size = len(sends)
    receives: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
    for source_rank in range(world_size):
        chunks = sends[source_rank].split(input_splits[source_rank], dim=0)
        for destination_rank, chunk in enumerate(chunks):
            receives[destination_rank].append(chunk)
    return [torch.cat(chunks, dim=0) for chunks in receives]


def _physical_reference(plan, rank: int) -> torch.Tensor:
    rows: list[int] = []
    valid_offset = 0
    for valid_length, ring_length in zip(plan.valid_lengths, plan.ring_physical_lengths):
        half = ring_length // (2 * plan.cp_size)
        physical = list(range(valid_offset + 1, valid_offset + valid_length + 1))
        physical.extend([0] * (ring_length - valid_length))
        rows.extend(physical[rank * half : (rank + 1) * half])
        second = 2 * plan.cp_size - 1 - rank
        rows.extend(physical[second * half : (second + 1) * half])
        valid_offset += valid_length
    return torch.tensor(rows, dtype=torch.int64).unsqueeze(-1)


def _owned_reference(plan, rank: int) -> torch.Tensor:
    rows: list[int] = []
    valid_offset = 0
    for valid_length, sample in zip(plan.valid_lengths, plan.ranks[rank].samples):
        rows.extend(range(valid_offset + sample.global_start + 1, valid_offset + sample.global_end + 1))
        valid_offset += valid_length
    return torch.tensor(rows, dtype=torch.int64).unsqueeze(-1)


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("ulysses_size", [1, 4])
def test_lossless_plan_is_deterministic_and_handles_edges(cp_size: int, ulysses_size: int):
    first = build_gdn_lossless_plan(_LENGTHS, cp_size=cp_size, ulysses_size=ulysses_size)
    second = build_gdn_lossless_plan(_LENGTHS, cp_size=cp_size, ulysses_size=ulysses_size)

    assert first == second
    assert first.plan_hash == second.plan_hash
    assert len(first.ranks) == cp_size
    assert all(len(rank.samples) == len(_LENGTHS) for rank in first.ranks)
    validate_gdn_lossless_plan(first)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_forward_and_inverse_wire_match_token_reference(cp_size: int):
    plan = build_gdn_lossless_plan(_LENGTHS, cp_size=cp_size, ulysses_size=4)
    physical = [_physical_reference(plan, rank) for rank in range(cp_size)]

    forward_sends = [
        _copy_rows(tensor, rank.forward_pack_spans, sum(rank.forward_input_splits))
        for tensor, rank in zip(physical, plan.ranks)
    ]
    forward_receives = _transpose_wire(forward_sends, [rank.forward_input_splits for rank in plan.ranks])
    owned = [
        _copy_rows(tensor, rank.forward_unpack_spans, rank.owned_token_count)
        for tensor, rank in zip(forward_receives, plan.ranks)
    ]
    for rank, tensor in enumerate(owned):
        torch.testing.assert_close(tensor, _owned_reference(plan, rank), rtol=0, atol=0)

    inverse_sends = [
        _copy_rows(tensor, rank.inverse_pack_spans, sum(rank.inverse_input_splits))
        for tensor, rank in zip(owned, plan.ranks)
    ]
    inverse_receives = _transpose_wire(inverse_sends, [rank.inverse_input_splits for rank in plan.ranks])
    restored = [
        _copy_rows(tensor, rank.inverse_unpack_spans, rank.source_token_count)
        for tensor, rank in zip(inverse_receives, plan.ranks)
    ]
    for expected, actual in zip(physical, restored):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"valid_lengths": [64.0], "cp_size": 2}, TypeError),
        ({"valid_lengths": [64], "cp_size": True}, TypeError),
        ({"valid_lengths": [64], "cp_size": 2, "ulysses_size": 0}, ValueError),
        ({"valid_lengths": [-1], "cp_size": 2}, ValueError),
        ({"valid_lengths": [64], "cp_size": 2, "chunk_size": 32}, ValueError),
    ],
)
def test_planner_rejects_invalid_typed_inputs(kwargs, exception):
    with pytest.raises(exception):
        build_gdn_lossless_plan(**kwargs)


def test_validator_rejects_forged_physical_coordinate():
    plan = build_gdn_lossless_plan([65, 128], cp_size=4, ulysses_size=2)
    route = plan.routes[0]
    forged_route = replace(route, source_start=route.source_start + 1, source_end=route.source_end + 1)
    forged = replace(plan, routes=(forged_route,) + plan.routes[1:])

    with pytest.raises(ValueError, match="physical source coordinate"):
        validate_gdn_lossless_plan(forged)


def test_planner_handles_512k_without_token_sized_route_objects():
    plan = build_gdn_lossless_plan([524288], cp_size=8, ulysses_size=4)

    assert sum(rank.owned_token_count for rank in plan.ranks) == 524288
    assert len(plan.routes) <= 2 * plan.cp_size * plan.cp_size


def test_chunk_alignment_preserves_state_inputs_and_unpads_outputs():
    assert aligned_gdn_cu_seqlens([0, 3, 5], chunk_size=4) == [0, 4, 8]
    query = torch.arange(20, dtype=torch.float32).reshape(1, 5, 2, 2).requires_grad_()
    key = (query.detach() + 1).requires_grad_()
    value = torch.ones(1, 5, 2, 3, requires_grad=True)
    g = torch.ones(1, 5, 2, requires_grad=True)
    beta = torch.ones(1, 5, 2, requires_grad=True)
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)

    q_pad, k_pad, v_pad, g_pad, beta_pad, padded_cu, real_indices = align_gdn_varlen_chunks(
        query,
        key,
        value,
        g,
        beta,
        cu,
        cu_seqlens_list=[0, 3, 5],
        chunk_size=4,
    )

    assert padded_cu.tolist() == [0, 4, 8]
    assert real_indices.tolist() == [0, 1, 2, 4, 5]
    assert torch.count_nonzero(v_pad[:, [3, 6, 7]]) == 0
    assert torch.count_nonzero(g_pad[:, [3, 6, 7]]) == 0
    assert torch.count_nonzero(beta_pad[:, [3, 6, 7]]) == 0
    torch.testing.assert_close(q_pad[:, 3], query[:, 2])
    torch.testing.assert_close(k_pad[:, 6], key[:, 4])
    torch.testing.assert_close(unpad_gdn_varlen_output(q_pad, real_indices), query)
    assert make_state_template(q_pad, v_pad, padded_cu).shape == (2, 2, 2, 3)
    participation = make_state_participation(q_pad, k_pad, v_pad, g_pad, beta_pad)
    participation.backward()
    assert query.grad is not None
    torch.testing.assert_close(query.grad, torch.zeros_like(query), rtol=0, atol=0)
