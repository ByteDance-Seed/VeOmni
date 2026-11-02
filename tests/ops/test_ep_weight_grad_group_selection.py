"""Expert selection must preserve each selected GEMM's exact arithmetic."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Triton kernel test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(3, 5), (129, 63), (512, 2048)])
@pytest.mark.parametrize("selected", [[], [0], [1, 3], [3, 1, 0, 2]])
@pytest.mark.parametrize("accumulate", [False, True])
def test_selected_groups_exact_and_unselected_untouched(dtype, shape, selected, accumulate):
    from veomni.distributed.moe.ep_grad import _launch_weight_grad

    torch.manual_seed(1038)
    sizes = [0, 32, 65, 129]
    m, n = shape
    a = torch.randn(sum(sizes), m, device="cuda", dtype=dtype)
    b = torch.randn(sum(sizes), n, device="cuda", dtype=dtype)
    ends = torch.tensor(sizes, device="cuda", dtype=torch.long).cumsum(0)
    seed = torch.randn(4, m, n, device="cuda", dtype=torch.float32)
    expected, actual = seed.clone(), seed.clone()
    _launch_weight_grad(a, b, expected, ends, sum(sizes), accumulate)
    _launch_weight_grad(
        a,
        b,
        actual,
        ends,
        sum(sizes),
        accumulate,
        group_indices=torch.tensor(selected, device="cuda", dtype=torch.long),
    )
    for index in range(4):
        reference = expected[index] if index in selected else seed[index]
        assert torch.equal(actual[index].view(torch.int32), reference.view(torch.int32))
