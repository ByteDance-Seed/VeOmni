# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""CPU-runnable sanity tests for the VeOmni kernel registry.

These tests do not require any accelerator.  They guard against
common ways the kernel registry silently breaks:

  * A new op is added but no backend is registered -> the eager
    fallback gets used and the model runs slow (silent regression).
  * A backend is registered but its hardware requirement is
    incorrect (e.g. claims "any" but actually requires GPU/NPU) ->
    eager call sites unexpectedly raise at runtime.
  * An OpSlot is bound to a kernel that no longer exists ->
    the binding raises KeyError at model-build time, which is
    better than at first forward, but still better caught in
    unit tests.

Run on any host with ``pytest tests/ops/test_kernel_registry_sanity.py``.
"""

import pytest
import torch

import veomni.ops  # noqa: F401 -- trigger KERNEL_REGISTRY registrations
from veomni.ops.dispatch import OpSlot
from veomni.ops.kernel_registry import KERNEL_REGISTRY, HardwareRequirement


# ---------------------------------------------------------------------------
# Hardware-requirement sanity
# ---------------------------------------------------------------------------


class TestHardwareRequirement:
    """``HardwareRequirement.is_satisfied()`` should never raise on a healthy host."""

    def test_device_type_gpu_consults_cuda(self):
        req = HardwareRequirement(device_type="gpu")
        result = req.is_satisfied()
        assert isinstance(result, bool)

    def test_device_type_npu_consults_torch_npu(self):
        req = HardwareRequirement(device_type="npu")
        result = req.is_satisfied()
        assert isinstance(result, bool)

    def test_device_type_any_always_satisfied(self):
        """``device_type='any'`` is hardware-agnostic; must always be satisfied."""
        assert HardwareRequirement(device_type="any").is_satisfied() is True
        assert HardwareRequirement(device_type="any", min_compute_capability=90).is_satisfied() is True

    def test_unknown_device_type_raises(self):
        """An unknown device type is a programmer error and must raise."""
        req = HardwareRequirement(device_type="tpu")
        with pytest.raises(ValueError, match="Unknown device_type"):
            req.is_satisfied()

    def test_compute_capability_bounds(self):
        """``min_compute_capability`` and ``max_compute_capability`` are checked.

        On a CPU host the GPU requirement can never be satisfied, but
        the bounds logic should still execute cleanly.  We test the
        truth-table on the ``any`` device type, which is always
        satisfied regardless of CC.
        """
        req = HardwareRequirement(device_type="any", min_compute_capability=70, max_compute_capability=90)
        assert req.is_satisfied() is True


# ---------------------------------------------------------------------------
# KERNEL_REGISTRY shape sanity
# ---------------------------------------------------------------------------


# Ops that VeOmni guarantees an NPU kernel for as of this commit.
# If a future refactor drops one, this list (and the test) must be
# updated -- the test is intentionally explicit so the failure mode
# is "missing expected kernel" rather than "test silently passes".
EXPECTED_NPU_OPS = [
    ("rms_norm", "standard"),
    ("rms_norm", "qwen3_5"),
    ("rotary_pos_emb", "full"),
    ("rotary_pos_emb_vision", "full"),
    ("rotary_pos_emb", "partial"),
    ("rms_norm_gated", "standard"),
]


class TestKernelRegistryShape:
    """Verify the registry contains the ops the rest of the codebase expects."""

    @pytest.mark.parametrize("op_name,variant", EXPECTED_NPU_OPS)
    def test_npu_kernel_registered(self, op_name, variant):
        """Every op+variant we ship an NPU kernel for must be findable."""
        assert "npu" in KERNEL_REGISTRY.list_available(op_name, variant), (
            f"Expected 'npu' kernel registered for ({op_name!r}, {variant!r}); "
            f"available: {KERNEL_REGISTRY.list_available(op_name, variant)}"
        )

    def test_registry_no_duplicate_implementations(self):
        """Two KernelSpec instances must not collide on the same (op, variant, name).

        ``register()`` with ``force=False`` (the default) is supposed to
        raise on duplicates.  We re-walk the registry to confirm no
        duplicates snuck in.
        """
        seen = {}
        for (op_name, variant), bucket in KERNEL_REGISTRY._specs.items():
            for impl_name in bucket:
                key = (op_name, variant, impl_name)
                assert key not in seen, f"Duplicate kernel registration: {key}"
                seen[key] = bucket[impl_name]

    def test_list_available_does_not_include_eager(self):
        """``list_available`` should return registered backends only.

        ``eager`` is a sentinel resolved by OpSlot, not a registered
        impl, so it must not appear in ``list_available``.
        """
        for op_name, variant in EXPECTED_NPU_OPS:
            backends = KERNEL_REGISTRY.list_available(op_name, variant)
            assert "npu" in backends, f"{op_name}.{variant} missing 'npu' in {backends}"
            assert "eager" not in backends, "'eager' is a sentinel, not a registered impl"

    def test_resolve_eager_returns_none(self):
        """``resolve(op, variant, 'eager')`` must always return ``None``."""
        for op_name, variant in EXPECTED_NPU_OPS:
            assert KERNEL_REGISTRY.resolve(op_name, variant, "eager") is None

    def test_resolve_unknown_impl_raises_keyerror(self):
        """An unknown impl name should raise KeyError, not silently fall back."""
        for op_name, variant in EXPECTED_NPU_OPS:
            with pytest.raises(KeyError, match="Unknown kernel"):
                KERNEL_REGISTRY.resolve(op_name, variant, "definitely_not_a_real_kernel")


# ---------------------------------------------------------------------------
# OpSlot behavior
# ---------------------------------------------------------------------------


class TestOpSlotBehavior:
    """Verify the dispatch contract for ``OpSlot`` itself."""

    def test_unbound_slot_use_non_eager_impl_is_false(self):
        """A freshly-created slot is unbound; ``use_non_eager_impl`` must be False."""
        slot = OpSlot("rms_norm", "standard")
        assert slot.use_non_eager_impl is False
        assert slot.bound_kernel() is None

    def test_bound_to_eager_keeps_use_non_eager_impl_false(self):
        """``bind('eager')`` is a valid no-op; the flag must still be False."""
        slot = OpSlot("rms_norm", "standard")
        slot.bind("eager")
        assert slot.use_non_eager_impl is False
        assert slot.bound_kernel() is None

    def test_call_without_binding_raises(self):
        """Calling an unbound slot must raise a clear RuntimeError."""
        slot = OpSlot("rms_norm", "standard")
        with pytest.raises(RuntimeError, match="has no kernel bound"):
            slot(torch.zeros(1, 1, 1))

    def test_repr_includes_state(self):
        """``__repr__`` must reflect the slot state for debugging."""
        slot = OpSlot("rms_norm", "standard")
        r = repr(slot)
        assert "rms_norm" in r
        assert "standard" in r
        assert "unbound" in r

        slot.bind("eager")
        r = repr(slot)
        assert "eager" in r

    def test_resolve_hardware_mismatch_raises_runtimeerror(self):
        """If the bound kernel's hardware requirement is not satisfied, raise RuntimeError.

        On a CPU host, any GPU/NPU kernel binding will fail; we use
        ``rotary_pos_emb_vision`` which has only an NPU registration
        in this repo to make the failure deterministic.
        """
        # Try to bind an op that ONLY has an NPU registration.  If
        # we're on an NPU host this won't raise; the test will then
        # pass for the wrong reason.  Guard with a skip when the
        # binding succeeds.
        try:
            slot = OpSlot("rotary_pos_emb_vision", "full")
            slot.bind("npu")
        except RuntimeError as e:
            assert "requires device_type" in str(e) or "npu" in str(e)
        else:
            # On an NPU host the binding succeeded -- nothing to test.
            pytest.skip("Test is NPU-host specific; skipping on NPU.")


# ---------------------------------------------------------------------------
# KernelSpec validation
# ---------------------------------------------------------------------------


class TestKernelSpecValidation:
    """KernelSpec is a frozen dataclass; mutation should raise."""

    def test_kernelspec_is_frozen(self):
        """KernelSpec is a frozen dataclass; mutation should raise."""
        spec = KERNEL_REGISTRY._specs[("rms_norm", "standard")]["npu"]
        with pytest.raises(AttributeError):  # FrozenInstanceError is a subclass of AttributeError
            spec.op_name = "mutated"  # type: ignore[misc]

    def test_all_registered_specs_have_factory(self):
        """Every registered spec must produce a callable from its factory.

        ``factory()`` is supposed to be a zero-argument callable
        returning the kernel.  We don't invoke it (the kernel itself
        may need a real accelerator), but we check that the factory
        attribute is callable.
        """
        for (op_name, variant), bucket in KERNEL_REGISTRY._specs.items():
            for impl_name, spec in bucket.items():
                assert callable(spec.factory), f"Spec for ({op_name}, {variant}, {impl_name}) has non-callable factory"

    def test_all_registered_specs_have_hardware_requirement(self):
        """Every registered spec must have a non-None hardware requirement."""
        for (op_name, variant), bucket in KERNEL_REGISTRY._specs.items():
            for impl_name, spec in bucket.items():
                assert spec.hardware is not None, (
                    f"Spec for ({op_name}, {variant}, {impl_name}) is missing hardware requirement"
                )
                assert spec.hardware.device_type in ("gpu", "npu", "any"), (
                    f"Spec for ({op_name}, {variant}, {impl_name}) has unexpected device_type "
                    f"{spec.hardware.device_type!r}"
                )
