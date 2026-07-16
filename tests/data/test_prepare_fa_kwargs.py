import torch

from veomni.data.data_collator import add_flash_attention_kwargs_from_position_ids
from veomni.utils.seqlen_pos_transform_utils import (
    coalesce_tail_padding_cu_seqlens,
    prepare_fa_kwargs_from_position_ids,
    valid_seqlens_from_cu_seqlens,
)


def make_pos_ids_concat(lengths):
    seqs = [torch.arange(L, dtype=torch.long) for L in lengths]
    if len(seqs) == 0:
        return torch.tensor([], dtype=torch.long)
    return torch.cat(seqs, dim=0)


def run_test(name, func):
    try:
        func()
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")


def expect_cu_from_lengths(lengths):
    s, cu = 0, [0]
    for L in lengths:
        s += L
        cu.append(s)
    return torch.tensor(cu, dtype=torch.int32), max(lengths) if lengths else 0


def assert_monotonic_per_seq(pos_1d, lengths):
    """Verify each segment is exactly [0..L-1]."""
    offset = 0
    for i, L in enumerate(lengths):
        seg = pos_1d[offset : offset + L]
        assert torch.equal(seg, torch.arange(L)), f"Seq {i} not 0..{L - 1}"
        offset += L


def test_basic():
    lengths = [8, 6, 10]
    pos = make_pos_ids_concat(lengths)
    assert_monotonic_per_seq(pos, lengths)

    (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(pos)
    expected_cu, expected_max = expect_cu_from_lengths(lengths)

    assert cu_q.dtype == torch.int32 and cu_k.dtype == torch.int32
    assert torch.equal(cu_q, expected_cu)
    assert torch.equal(cu_k, expected_cu)
    assert max_q == expected_max and max_k == expected_max


def test_randomized():
    torch.manual_seed(42)
    lengths = torch.randint(5, 20, (5,)).tolist()
    pos = make_pos_ids_concat(lengths)
    assert_monotonic_per_seq(pos, lengths)

    (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(pos)
    expected_cu, expected_max = expect_cu_from_lengths(lengths)

    assert torch.equal(cu_q, expected_cu)
    assert torch.equal(cu_k, expected_cu)
    assert max_q == expected_max and max_k == expected_max


def test_random_batch():
    torch.manual_seed(7)
    B = 32
    lengths = torch.randint(50, 200, (B,)).tolist()
    pos = make_pos_ids_concat(lengths)
    assert_monotonic_per_seq(pos, lengths)

    (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(pos)
    expected_cu, expected_max = expect_cu_from_lengths(lengths)

    assert torch.equal(cu_q, expected_cu)
    assert torch.equal(cu_k, expected_cu)
    assert max_q == expected_max and max_k == expected_max


def test_linear_attn_cu_seqlens_coalesces_tail_padding():
    real_lengths = [4, 3]
    tail_padding_length = 3
    pos = torch.cat(
        (
            make_pos_ids_concat(real_lengths),
            torch.zeros(tail_padding_length, dtype=torch.long),
        )
    ).unsqueeze(0)

    (cu_seq_lens_q, _), _ = prepare_fa_kwargs_from_position_ids(pos)
    coalesced = coalesce_tail_padding_cu_seqlens(cu_seq_lens_q, tail_padding_length)

    # Raw prepare_fa_kwargs still encodes one segment per pad token.
    assert torch.equal(cu_seq_lens_q, torch.tensor([0, 4, 7, 8, 9, 10], dtype=torch.int32))
    assert torch.equal(coalesced, torch.tensor([0, 4, 7, 10], dtype=torch.int32))


def test_linear_attn_cu_seqlens_preserves_real_one_token_sequence():
    real_lengths = [4, 1]
    tail_padding_length = 3
    pos = torch.cat(
        (
            make_pos_ids_concat(real_lengths),
            torch.zeros(tail_padding_length, dtype=torch.long),
        )
    ).unsqueeze(0)

    (cu_seq_lens_q, _), _ = prepare_fa_kwargs_from_position_ids(pos)
    coalesced = coalesce_tail_padding_cu_seqlens(cu_seq_lens_q, tail_padding_length)

    assert torch.equal(cu_seq_lens_q, torch.tensor([0, 4, 5, 6, 7, 8], dtype=torch.int32))
    assert torch.equal(coalesced, torch.tensor([0, 4, 5, 8], dtype=torch.int32))


def test_add_flash_attention_kwargs_coalesces_fa_and_linear_cu_seqlens():
    real_lengths = [4, 3]
    tail_padding_length = 3
    pos = torch.cat(
        (
            make_pos_ids_concat(real_lengths),
            torch.zeros(tail_padding_length, dtype=torch.long),
        )
    ).unsqueeze(0)
    batch = {"position_ids": pos}

    cu_q, cu_k, max_q, max_k = add_flash_attention_kwargs_from_position_ids(batch, tail_padding_length)
    expected = torch.tensor([0, 4, 7, 10], dtype=torch.int32)

    assert torch.equal(cu_q, expected)
    assert torch.equal(cu_k, expected)
    assert torch.equal(batch["cu_seq_lens_q"], expected)
    assert torch.equal(batch["cu_seq_lens_k"], expected)
    assert torch.equal(batch["linear_attn_cu_seq_lens_q"], expected)
    assert int(batch["tail_padding_length"]) == tail_padding_length
    assert max_q == 4 and max_k == 4
    assert torch.equal(
        valid_seqlens_from_cu_seqlens(cu_q, tail_padding_length=tail_padding_length),
        torch.tensor([4, 3], dtype=torch.int32),
    )


def test_valid_seqlens_strips_coalesced_pad_segment():
    coalesced = torch.tensor([0, 4, 7, 10], dtype=torch.int32)
    expected = torch.tensor([4, 3], dtype=torch.int32)

    assert torch.equal(
        valid_seqlens_from_cu_seqlens(coalesced, tail_padding_length=3),
        expected,
    )


def test_valid_seqlens_with_exact_tail_padding_preserves_real_one_token_sequence():
    cu_seq_lens = torch.tensor([0, 4, 5, 6, 7, 8], dtype=torch.int32)

    assert torch.equal(valid_seqlens_from_cu_seqlens(cu_seq_lens), torch.tensor([4], dtype=torch.int32))
    assert torch.equal(
        valid_seqlens_from_cu_seqlens(cu_seq_lens, tail_padding_length=3),
        torch.tensor([4, 1], dtype=torch.int32),
    )


if __name__ == "__main__":
    run_test("basic", test_basic)
    run_test("randomized", test_randomized)
    run_test("large_random_batch_stress", test_random_batch)
    run_test("linear_attn_cu_seqlens_coalesces_tail_padding", test_linear_attn_cu_seqlens_coalesces_tail_padding)
    run_test(
        "linear_attn_cu_seqlens_preserves_real_one_token_sequence",
        test_linear_attn_cu_seqlens_preserves_real_one_token_sequence,
    )
    run_test(
        "add_flash_attention_kwargs_coalesces_fa_and_linear_cu_seqlens",
        test_add_flash_attention_kwargs_coalesces_fa_and_linear_cu_seqlens,
    )
    run_test("valid_seqlens_strips_coalesced_pad_segment", test_valid_seqlens_strips_coalesced_pad_segment)
    run_test(
        "valid_seqlens_with_exact_tail_padding_preserves_real_one_token_sequence",
        test_valid_seqlens_with_exact_tail_padding_preserves_real_one_token_sequence,
    )
