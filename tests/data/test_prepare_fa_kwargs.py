import torch

from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids


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


if __name__ == "__main__":
    run_test("basic", test_basic)
    run_test("randomized", test_randomized)
    run_test("large_random_batch_stress", test_random_batch)
