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

"""Hardware and import-boundary tests for Qwen3.5 gated delta-rule selection.

Covers the ``causal_conv1d`` and ``chunk_gated_delta_rule`` OpSlots, which now
each expose a GPU (``fla``) and an NPU (``npu``) backend. The point of the
registry refactor is that selecting a backend whose ``HardwareRequirement`` is
not met raises at ``OpSlot.bind()`` time (via ``KERNEL_REGISTRY.resolve``)
rather than silently binding the wrong kernel — the exact guarantee the old
hard-coded path bypassed.

Only the *failure* direction is exercised (npu-on-GPU, fla-on-NPU): the
hardware check fires inside ``resolve`` before ``spec.factory()`` runs, so
these tests never import the Triton kernels and run on any CI host without
``triton-ascend`` / ``flash-linear-attention``.
"""

import builtins
import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

import veomni.ops  # noqa: F401 — trigger KERNEL_REGISTRY registrations
from veomni.ops.dispatch import OpSlot
from veomni.ops.kernel_registry import KERNEL_REGISTRY
from veomni.utils.import_utils import is_torch_npu_available


_REGISTRY_MODULE = "veomni.ops.kernel_registry"

_GDN_OPS = ["causal_conv1d", "chunk_gated_delta_rule"]

_QWEN3_5_NPU_MODELING_MODULES = [
    (
        "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_npu",
        "Qwen3_5TextModel",
    ),
    (
        "veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_npu",
        "Qwen3_5MoeTextModel",
    ),
]


# ---------------------------------------------------------------------------
# npu backend requested on a GPU host → device_type='npu' gate fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", False)
def test_opslot_npu_backend_on_gpu_raises(op_name):
    slot = OpSlot(op_name, "standard")
    with pytest.raises(RuntimeError, match="device_type='npu'"):
        slot.bind("npu")


# ---------------------------------------------------------------------------
# fla (GPU) backend requested on an NPU host → device_type='gpu' gate fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", False)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", True)
def test_opslot_fla_backend_on_npu_raises(op_name):
    slot = OpSlot(op_name, "standard")
    with pytest.raises(RuntimeError, match="device_type='gpu'"):
        slot.bind("fla")


# ---------------------------------------------------------------------------
# eager path never touches HardwareRequirement (resolves to None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
def test_opslot_eager_skips_hw_check(op_name):
    slot = OpSlot(op_name, "standard")
    slot.bind("eager")
    assert not slot.use_non_eager_impl


# ---------------------------------------------------------------------------
# unknown backend name is a KeyError, listing the available options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
def test_opslot_unknown_backend_raises(op_name):
    slot = OpSlot(op_name, "standard")
    with pytest.raises(KeyError, match="Unknown kernel 'bogus'"):
        slot.bind("bogus")


# ---------------------------------------------------------------------------
# Registry presence — a future reshuffle that drops a backend trips this early.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", _GDN_OPS)
def test_gdn_registry_has_fla_and_npu(op_name):
    available = KERNEL_REGISTRY.list_available(op_name, "standard")
    assert "fla" in available
    assert "npu" in available


# ---------------------------------------------------------------------------
# npu_ascendc — the second NPU backend on chunk_gated_delta_rule only
# ---------------------------------------------------------------------------


def test_chunk_gdr_registry_has_npu_ascendc():
    available = KERNEL_REGISTRY.list_available("chunk_gated_delta_rule", "standard")
    assert "npu_ascendc" in available
    # npu_ascendc is scoped to chunk_gated_delta_rule; causal_conv1d keeps a single npu backend.
    assert "npu_ascendc" not in KERNEL_REGISTRY.list_available("causal_conv1d", "standard")


@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", False)
def test_opslot_npu_ascendc_backend_on_gpu_raises():
    slot = OpSlot("chunk_gated_delta_rule", "standard")
    with pytest.raises(RuntimeError, match="device_type='npu'"):
        slot.bind("npu_ascendc")


@pytest.mark.skipif(
    is_torch_npu_available(),
    reason="guard only fires when torch_npu/fla_npu/triton are absent; on NPU the factory imports the real kernel",
)
def test_npu_ascendc_factory_missing_dep_raises_actionable():
    """Absent ``fla_npu`` / ``torch_npu`` / ``triton-ascend`` surfaces a RuntimeError with
    install guidance, not a bare ModuleNotFoundError from the transitive imports."""
    from veomni.ops.kernels.gated_delta_rule import _npu_ascendc_chunk_gated_delta_rule_factory

    with pytest.raises(RuntimeError, match="npu_ascendc"):
        _npu_ascendc_chunk_gated_delta_rule_factory()


def test_npu_ascendc_factory_triton_error_uses_locked_installer():
    from veomni.ops.kernels.gated_delta_rule import _npu_ascendc_chunk_gated_delta_rule_factory

    real_import = builtins.__import__

    def import_without_triton(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "npu_ascendc_gated_delta_rule" and level == 1:
            raise ModuleNotFoundError("No module named 'triton'", name="triton")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=import_without_triton):
        with pytest.raises(RuntimeError) as exc_info:
            _npu_ascendc_chunk_gated_delta_rule_factory()

    message = str(exc_info.value)
    assert "scripts/ci/install_triton_ascend.py" in message
    assert "--extra-index-url" not in message


@pytest.mark.parametrize("module_name,model_class_name", _QWEN3_5_NPU_MODELING_MODULES)
@pytest.mark.parametrize("implementation", ["npu", "npu_ascendc"])
def test_ascendc_metadata_precompute_follows_bound_backend(module_name, model_class_name, implementation):
    """Only the bound AscendC backend may import and inject ``fla_npu`` metadata."""
    from unittest.mock import MagicMock

    import torch

    from veomni.models.auto import _bind_veomni_ops

    module = importlib.import_module(module_name)
    ops_config = SimpleNamespace(
        moe_implementation="eager",
        chunk_gated_delta_rule_implementation=implementation,
    )
    with patch.object(OpSlot, "bind"):
        assert _bind_veomni_ops(module, ops_config)

    config_slot = module.veomni_chunk_gated_delta_rule_implementation
    assert config_slot.value == implementation

    metadata_module_name = "veomni.ops.kernels.gated_delta_rule._ascend.flash_gated_delta_rule"
    metadata_module = ModuleType(metadata_module_name)
    metadata = ([0, 2], {4: torch.tensor([0])}, {4: [0]})
    precompute = MagicMock(return_value=metadata)
    metadata_module.precompute_varlen_metadata = precompute

    inputs_embeds = torch.zeros(1, 2, 4)
    decoder_layer = MagicMock(side_effect=lambda hidden_states, **kwargs: hidden_states)
    model = SimpleNamespace(
        config=SimpleNamespace(
            linear_num_value_heads=4,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
        ),
        _update_linear_attn_mask=MagicMock(return_value=None),
        rotary_emb=MagicMock(return_value=None),
        layers=[decoder_layer],
        norm=MagicMock(side_effect=lambda hidden_states: hidden_states),
    )
    raw_forward = inspect.unwrap(getattr(module, model_class_name).forward)
    with (
        patch.dict(sys.modules, {metadata_module_name: metadata_module}),
        patch.object(module, "create_causal_mask", return_value=None),
        patch.object(module, "get_parallel_state", return_value=SimpleNamespace(sp_enabled=False)),
    ):
        raw_forward(
            model,
            inputs_embeds=inputs_embeds,
            position_ids=torch.zeros(4, 1, 2, dtype=torch.long),
            cu_seq_lens_q=torch.tensor([0, 2], dtype=torch.long),
        )

    decoder_kwargs = decoder_layer.call_args.kwargs
    if implementation == "npu_ascendc":
        precompute.assert_called_once()
        assert decoder_kwargs["cu_seqlens_list_q"] is metadata[0]
        assert decoder_kwargs["chunk_indices_q"] is metadata[1]
        assert decoder_kwargs["chunk_indices_list_q"] is metadata[2]
    else:
        precompute.assert_not_called()
        assert "cu_seqlens_list_q" not in decoder_kwargs
        assert "chunk_indices_q" not in decoder_kwargs
        assert "chunk_indices_list_q" not in decoder_kwargs
