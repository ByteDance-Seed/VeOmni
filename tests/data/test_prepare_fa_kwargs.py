import torch

def prepare_fa_kwargs_from_position_ids(position_ids):
    """
    Copy from transformers/modeling_flash_attention_utils.py 354567d955fbc5fbd70fc841b7a7bcc654bea3f1
    This function returns all the necessary kwargs to call `flash_attn_varlen_func` extracted from position_ids.

    Arguments:
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into
            ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query,
            `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.view(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )
    cu_seq_lens_k = cu_seq_lens_q

    # https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/flash_attn/flash_attn_interface.py#L1423-L1424
    # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always increasing
    # for some models (e.g. qwen2-vl).
    max_length_q = cu_seq_lens_q.diff().max()
    # NOTE: With torch compile, this will cause a graph break if you don't set
    # `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` in the environment or call
    # `torch._dynamo.config.capture_scalar_outputs = True` before doing the forward pass.
    # This is a limitation of flash attention API, as the function `flash_attn_varlen_func`
    # requires `max_length_q`, `max_length_k` to be passed as `int` and not `torch.Tensor`.
    max_length_q = max_length_q.item()
    max_length_k = max_length_q

    return (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k)

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
        seg = pos_1d[offset:offset+L]
        assert torch.equal(seg, torch.arange(L)), f"Seq {i} not 0..{L-1}"
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
