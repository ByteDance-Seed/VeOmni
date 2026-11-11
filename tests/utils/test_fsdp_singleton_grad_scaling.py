from types import SimpleNamespace

import pytest
import torch

from veomni.distributed.fsdp2.grad_scaling import configure_singleton_gradient_scaling


class HookModule:
    def __init__(self):
        self.hook = None

    def set_all_reduce_hook(self, hook):
        self.hook = hook


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("factor", [1, 2, 8, 3.0])
def test_singleton_owned_output_matches_native_copy_division_without_modifying_input(dtype, factor):
    module = HookModule()
    assert configure_singleton_gradient_scaling(module, mesh_size=1, reduce_dtype=dtype, gradient_divide_factor=factor)
    source = torch.cat(
        (torch.linspace(-1, 1, 129), torch.tensor([0.0, -0.0, torch.finfo(dtype).tiny, float("inf"), float("nan")]))
    ).to(dtype)
    original = source.clone()
    expected = torch.empty_like(source)
    expected.copy_(source / factor)
    owned = torch.empty_like(source)
    owned.copy_(source)
    pointer = owned.data_ptr()
    if module.hook is not None:
        assert module.hook(owned) is None
    assert owned.data_ptr() == pointer
    torch.testing.assert_close(owned, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(source, original, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_each_accumulation_contribution_is_normalized_once(dtype):
    module = HookModule()
    configure_singleton_gradient_scaling(module, mesh_size=1, reduce_dtype=dtype, gradient_divide_factor=8)
    first, second = torch.tensor([1.0, 2.0], dtype=dtype), torch.tensor([3.0, 4.0], dtype=dtype)
    a, b = first.clone(), second.clone()
    module.hook(a)
    saved = a.clone()
    module.hook(b)
    torch.testing.assert_close(a, saved, rtol=0, atol=0)
    torch.testing.assert_close(a + b, first / 8 + second / 8, rtol=0, atol=0)


@pytest.mark.parametrize("mesh_size,dtype", [(2, torch.float32), (8, torch.bfloat16), (1, torch.float16), (1, None)])
def test_non_singleton_or_unvalidated_dtype_keeps_native_fallback(mesh_size, dtype):
    module = HookModule()
    assert not configure_singleton_gradient_scaling(
        module, mesh_size=mesh_size, reduce_dtype=dtype, gradient_divide_factor=8
    )
    assert module.hook is None


def test_missing_public_hook_api_keeps_legacy_fallback():
    assert not configure_singleton_gradient_scaling(
        SimpleNamespace(), mesh_size=1, reduce_dtype=torch.float32, gradient_divide_factor=8
    )
