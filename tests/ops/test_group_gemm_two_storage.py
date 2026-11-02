"""Two-storage grouped GEMM: CPU launch metadata and actual CUDA parity are separate."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch


def _cpu_launcher():
    # Load only the real Python launcher when Triton itself cannot import on CPU.
    # This seam proves pointer/geometry validation, never Triton arithmetic.
    path = Path(__file__).parents[2] / "veomni/ops/kernels/moe/_kernels/kernel/group_gemm.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "group_gemm_same_nk"
    )
    calls = []

    class Launch:
        def __getitem__(self, grid):
            def launch(**kwargs):
                calls.append((grid(dict(kwargs, BLOCK_M=128, BLOCK_N=128)), kwargs))

            return launch

    namespace = dict(
        torch=torch,
        Optional=Optional,
        ActivationType=object,
        triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        get_torch_device=lambda: SimpleNamespace(device=lambda _: nullcontext()),
        group_gemm_same_nk_kernel=Launch(),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["group_gemm_same_nk"], calls


@pytest.mark.parametrize("transpose", [False, True])
def test_two_storage_launcher_preserves_total_geometry_and_original_pointers(transpose):
    launch, calls = _cpu_launcher()
    a = torch.zeros(137, 33, dtype=torch.bfloat16)
    shape = (65, 33) if transpose else (33, 65)
    b, extra = torch.zeros(2, *shape, dtype=a.dtype), torch.zeros(3, *shape, dtype=a.dtype)
    out = launch(a, b, torch.tensor([0, 5, 5, 8, 137]), 137, transpose_b=transpose, b_extra=extra)
    assert out.shape == (137, 65)
    assert len(calls) == 1
    grid, kwargs = calls[0]
    assert grid == (2, 5)
    assert kwargs["b_ptr"] is b and kwargs["b_extra_ptr"] is extra
    assert (kwargs["G"], kwargs["LOCAL_G"], kwargs["total_M"], kwargs["max_M"], kwargs["N"], kwargs["K"]) == (
        5,
        2,
        137,
        137,
        65,
        33,
    )


@pytest.mark.parametrize("bad", ["shape", "dtype", "device", "strides", "groups"])
def test_two_storage_launcher_rejects_incompatible_extra_before_launch(bad):
    launch, calls = _cpu_launcher()
    a, b = torch.zeros(3, 4, dtype=torch.bfloat16), torch.zeros(2, 4, 6, dtype=torch.bfloat16)
    extra = torch.zeros(1, 4, 6, dtype=b.dtype)
    cumsum = torch.tensor([1, 2, 3])
    if bad == "shape":
        extra = extra[:, :3]
    elif bad == "dtype":
        extra = extra.float()
    elif bad == "device":
        extra = extra.to("meta")
    elif bad == "strides":
        extra = torch.zeros(1, 6, 4, dtype=b.dtype).transpose(1, 2)
    else:
        cumsum = cumsum[:2]
    with pytest.raises((ValueError, AssertionError)):
        launch(a, b, cumsum, 3, b_extra=extra)
    assert calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required for real Triton kernel parity")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("counts", [(0, 5, 0, 131, 1), (3, 0, 1, 0, 0), (0, 0, 0, 0, 0)])
def test_cuda_two_storage_group_gemm_matches_concat(dtype, transpose, counts):
    from veomni.ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_nk

    torch.manual_seed(71)
    shape = (65, 33) if transpose else (33, 65)
    a = torch.randn(sum(counts), 33, device="cuda", dtype=dtype)
    b = torch.randn(2, *shape, device="cuda", dtype=dtype)
    extra = torch.randn(3, *shape, device="cuda", dtype=dtype)
    combined = torch.cat((b, extra))
    cumsum = torch.tensor(counts, device="cuda", dtype=torch.int64).cumsum(0)
    actual = group_gemm_same_nk(a, b, cumsum, max(counts), transpose_b=transpose, b_extra=extra)
    baseline = group_gemm_same_nk(a, combined, cumsum, sum(counts), transpose_b=transpose)
    assert torch.equal(actual, baseline)
    expected = []
    for index, chunk in enumerate(a.split(counts)):
        weight = combined[index].float()
        expected.append(chunk.float() @ (weight.T if transpose else weight))
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.004
    torch.testing.assert_close(actual.float(), torch.cat(expected), rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required for grouped grid parity")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transpose", [False, True])
def test_cuda_tight_group_bound_preserves_partial_tile_groups(dtype, transpose):
    from veomni.ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_nk

    torch.manual_seed(73)
    counts = (1031, 5, 129, 0, 257)
    shape = (1025, 33) if transpose else (33, 1025)
    a = torch.randn(sum(counts), 33, device="cuda", dtype=dtype)
    original = torch.randn(2, *shape, device="cuda", dtype=dtype)
    replica = torch.randn(3, *shape, device="cuda", dtype=dtype)
    combined = torch.cat((original, replica))
    cumsum = torch.tensor(counts, device="cuda", dtype=torch.int64).cumsum(0)
    reference = group_gemm_same_nk(a, combined, cumsum, sum(counts), transpose_b=transpose)
    for extra in (False, True):
        result = group_gemm_same_nk(
            a,
            original if extra else combined,
            cumsum,
            max(counts),
            transpose_b=transpose,
            **({"b_extra": replica} if extra else {}),
        )
        assert torch.equal(result, reference)
