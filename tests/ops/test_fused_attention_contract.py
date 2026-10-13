# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from transformers import PreTrainedConfig
from transformers.integrations.flex_attention import flex_attention_forward as hf_flex_attention_forward
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.transformers.attention_utils import VARLEN_ATTENTION_TYPES
from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import flex as flex_backend
from veomni.ops.kernels.attention import magi as magi_backend
from veomni.ops.kernels.attention.magi import mask as magi_mask_backend


_FLASH_IMPLEMENTATIONS = (
    "veomni_flash_attention_2_with_sp",
    "veomni_flash_attention_2_hub_with_sp",
    "veomni_flash_attention_3_with_sp",
    "veomni_flash_attention_3_hub_with_sp",
    "veomni_flash_attention_4_with_sp",
)
_HUB_FLASH_IMPLEMENTATIONS = (
    "flash_attention_2_hub",
    "flash_attention_3_hub",
    "veomni_flash_attention_2_hub_with_sp",
    "veomni_flash_attention_3_hub_with_sp",
)
_FLEX_IMPLEMENTATION = "veomni_flex_attention_with_sp"
_MAGI_IMPLEMENTATION = "veomni_magi_attention_with_sp"


class _FakeAttentionModule(nn.Module):
    def __init__(self, implementation: str):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)


@pytest.mark.parametrize("version", (2, 3))
def test_hub_flash_is_classified_as_varlen_attention(version):
    assert f"flash_attention_{version}_hub" in VARLEN_ATTENTION_TYPES
    assert f"veomni_flash_attention_{version}_hub_with_sp" in VARLEN_ATTENTION_TYPES


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
def test_hub_flash_config_maps_to_veomni_sp_backend(monkeypatch, implementation):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_npu_available", lambda: False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)

    config = OpsImplementationConfig(attn_implementation=implementation)

    expected = implementation if implementation.startswith("veomni_") else f"veomni_{implementation}_with_sp"
    assert config.attn_implementation == expected


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
def test_hub_flash_config_rejects_huggingface_modeling_backend(monkeypatch, implementation):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_npu_available", lambda: False)

    with pytest.raises(ValueError, match="requires MODELING_BACKEND=veomni"):
        OpsImplementationConfig(attn_implementation=implementation)


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
def test_hub_flash_config_rejects_npu(monkeypatch, implementation):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_npu_available", lambda: True)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)

    with pytest.raises(ValueError, match="not supported on Ascend NPU"):
        OpsImplementationConfig(attn_implementation=implementation)


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
@pytest.mark.parametrize("explicit_ops", (False, True))
@pytest.mark.parametrize(
    "on_npu,modeling_backend,error", [(True, "veomni", "Ascend NPU"), (False, "hf", "MODELING_BACKEND")]
)
def test_hub_flash_model_build_rejects_before_preload(
    monkeypatch, implementation, explicit_ops, on_npu, modeling_backend, error
):
    from veomni.models import auto

    monkeypatch.setenv("MODELING_BACKEND", modeling_backend)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_npu_available", lambda: on_npu)
    config = SimpleNamespace(attn_implementation=implementation if explicit_ops else "eager")
    monkeypatch.setattr("veomni.ops.config.singleton.get_ops_config", lambda: config)
    apply_ops = Mock()
    build_config = Mock(side_effect=AssertionError("Model loading must not start"))
    preload = Mock(side_effect=AssertionError("HF preloading must not start"))
    hub_load = Mock(side_effect=AssertionError("Hub loading must not start"))
    monkeypatch.setattr("veomni.ops.apply_ops_config", apply_ops)
    monkeypatch.setattr(auto, "build_config", build_config)
    monkeypatch.setattr("transformers.modeling_utils.lazy_import_flash_attention", preload)
    monkeypatch.setattr("transformers.integrations.hub_kernels.load_and_register_attn_kernel", hub_load)

    with pytest.raises(ValueError, match=error):
        if explicit_ops:
            auto.build_foundation_model("unused", ops_implementation=config)
        else:
            auto.build_foundation_model("unused", attn_implementation=implementation)

    apply_ops.assert_not_called()
    build_config.assert_not_called()
    preload.assert_not_called()
    hub_load.assert_not_called()


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
@pytest.mark.parametrize("entry", ("ops", "override", "installed"))
def test_hub_flash_model_build_passes_normalized_name_to_loader(monkeypatch, implementation, entry):
    from veomni.models import auto

    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_npu_available", lambda: False)
    monkeypatch.setattr(auto, "is_torch_npu_available", lambda: False)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=False, global_rank=0))
    config = SimpleNamespace(attn_implementation=implementation if entry != "override" else "eager")
    monkeypatch.setattr("veomni.ops.config.singleton.get_ops_config", lambda: config)
    monkeypatch.setattr("veomni.ops.apply_ops_config", Mock())
    model = nn.Linear(1, 1)
    load_model = Mock(return_value=model)
    monkeypatch.setattr(auto, "get_loader", lambda config: SimpleNamespace(load_model=load_model))
    kwargs = {"ops_implementation": config} if entry == "ops" else {}
    if entry == "override":
        kwargs["attn_implementation"] = implementation

    assert auto.build_foundation_model(PreTrainedConfig(), init_device="cpu", **kwargs) is model

    expected = implementation if implementation.startswith("veomni_") else f"veomni_{implementation}_with_sp"
    assert load_model.call_args.kwargs["init_kwargs"]["attn_implementation"] == expected
    assert ALL_ATTENTION_FUNCTIONS[expected] is veomni_attention.fused_attention_forward


@pytest.mark.parametrize("on_npu", (False, True))
@pytest.mark.parametrize("version", (2, 3))
def test_local_flash_config_is_unchanged(monkeypatch, on_npu, version):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_npu_available", lambda: on_npu)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)

    config = OpsImplementationConfig(attn_implementation=f"flash_attention_{version}")

    assert config.attn_implementation == f"veomni_flash_attention_{version}_with_sp"


@pytest.mark.parametrize(
    "backend",
    [
        veomni_attention.flash_attention_forward,
        veomni_attention.flex_attention_forward,
        veomni_attention.magi_attention_forward,
    ],
)
def test_fused_attention_forward_matches_backend_public_signatures(backend):
    fused_signature = inspect.signature(veomni_attention.fused_attention_forward)
    backend_signature = inspect.signature(backend)

    assert list(fused_signature.parameters) == list(backend_signature.parameters)
    for name, backend_parameter in backend_signature.parameters.items():
        fused_parameter = fused_signature.parameters[name]
        assert fused_parameter.kind is backend_parameter.kind
        assert fused_parameter.default == backend_parameter.default


def test_apply_veomni_attention_patch_registers_custom_facade_names(monkeypatch):
    patch_calls = []
    monkeypatch.setattr(
        veomni_attention,
        "patch_transformers_hub_kernel_loader_for_veomni",
        lambda: patch_calls.append("hub_kernel_loader"),
    )
    monkeypatch.setattr(
        veomni_attention,
        "register_veomni_flex_attention_mask_builder",
        lambda: patch_calls.append("flex_mask_builder"),
    )
    monkeypatch.setattr(
        veomni_attention,
        "register_veomni_magi_attention_mask_builder",
        lambda: patch_calls.append("magi_mask_builder"),
    )

    veomni_attention.apply_veomni_attention_patch()

    assert patch_calls == ["hub_kernel_loader", "flex_mask_builder", "magi_mask_builder"]
    for implementation in (*_FLASH_IMPLEMENTATIONS, _FLEX_IMPLEMENTATION, _MAGI_IMPLEMENTATION):
        assert ALL_ATTENTION_FUNCTIONS[implementation] is veomni_attention.fused_attention_forward
    assert ALL_ATTENTION_FUNCTIONS["flex_attention"] is hf_flex_attention_forward


def test_register_veomni_flex_attention_mask_builder_uses_sp_aware_wrapper(monkeypatch):
    mask_mapping = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    monkeypatch.setitem(mask_mapping, _FLEX_IMPLEMENTATION, object())

    veomni_attention.register_veomni_flex_attention_mask_builder()
    veomni_attention.register_veomni_flex_attention_mask_builder()

    assert ALL_MASK_ATTENTION_FUNCTIONS[_FLEX_IMPLEMENTATION] is flex_backend.flex_attention_mask_builder

    config = PreTrainedConfig()
    config._attn_implementation = _FLEX_IMPLEMENTATION
    config.sliding_window = 4
    inputs_embeds = torch.randn(1, 8, 16)
    position_ids = torch.arange(8).unsqueeze(0)

    causal_mask = create_causal_mask(config, inputs_embeds, None, None, position_ids)
    sliding_window_mask = create_sliding_window_causal_mask(config, inputs_embeds, None, None, position_ids)

    assert isinstance(causal_mask, BlockMask)
    assert isinstance(sliding_window_mask, BlockMask)
    assert causal_mask.shape == sliding_window_mask.shape == (1, 1, 8, 8)


def test_register_veomni_magi_attention_mask_builder_uses_range_builder(monkeypatch):
    mask_mapping = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    monkeypatch.setitem(mask_mapping, _MAGI_IMPLEMENTATION, object())
    monkeypatch.setattr(
        magi_mask_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=False, ulysses_size=1),
    )

    veomni_attention.register_veomni_magi_attention_mask_builder()
    veomni_attention.register_veomni_magi_attention_mask_builder()

    assert ALL_MASK_ATTENTION_FUNCTIONS[_MAGI_IMPLEMENTATION] is magi_backend.create_magi_mask

    config = PreTrainedConfig()
    config._attn_implementation = _MAGI_IMPLEMENTATION
    inputs_embeds = torch.randn(1, 8, 16)
    position_ids = torch.arange(8).unsqueeze(0)

    attention_mask = create_causal_mask(config, inputs_embeds, None, None, position_ids)

    assert isinstance(attention_mask, magi_backend.MagiAttentionMask)


def test_registered_magi_mask_builder_rejects_implicit_packed_visibility(monkeypatch):
    mask_mapping = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    monkeypatch.setitem(mask_mapping, _MAGI_IMPLEMENTATION, object())
    monkeypatch.setattr(
        magi_mask_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=False, ulysses_size=1),
    )
    veomni_attention.register_veomni_magi_attention_mask_builder()
    config = PreTrainedConfig()
    config._attn_implementation = _MAGI_IMPLEMENTATION
    inputs_embeds = torch.randn(1, 8, 16)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    packed_position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])

    with pytest.raises(ValueError, match="cannot recover packed boundaries"):
        create_causal_mask(
            config,
            inputs_embeds,
            attention_mask,
            None,
            packed_position_ids,
        )


def test_veomni_flex_attention_mask_builder_uses_ulysses_global_sequence_length(monkeypatch):
    monkeypatch.setattr(
        flex_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=True, ulysses_size=2),
    )
    veomni_attention.register_veomni_flex_attention_mask_builder()

    config = PreTrainedConfig()
    config._attn_implementation = _FLEX_IMPLEMENTATION
    config.sliding_window = 4
    local_inputs_embeds = torch.randn(1, 4, 16)
    full_attention_mask = torch.ones(1, 8, dtype=torch.long)
    local_position_ids = torch.arange(4).unsqueeze(0)

    causal_mask = create_causal_mask(
        config,
        local_inputs_embeds,
        full_attention_mask,
        None,
        local_position_ids,
    )
    sliding_window_mask = create_sliding_window_causal_mask(
        config,
        local_inputs_embeds,
        full_attention_mask,
        None,
        local_position_ids,
    )

    assert causal_mask.shape == sliding_window_mask.shape == (1, 1, 8, 8)
    zero = torch.tensor(0)
    last_query = torch.tensor(7)
    first_key = torch.tensor(0)
    nearby_key = torch.tensor(6)
    assert causal_mask.mask_mod(zero, zero, last_query, first_key)
    assert sliding_window_mask.mask_mod(zero, zero, last_query, nearby_key)
    assert not sliding_window_mask.mask_mod(zero, zero, last_query, first_key)


@pytest.mark.parametrize(
    ("attention_mask", "expected_message"),
    [
        (None, "requires a full-sequence 2D attention mask"),
        (torch.ones(1, 4, dtype=torch.long), "local q_length \\* ulysses_size"),
    ],
)
def test_veomni_flex_attention_mask_builder_rejects_incomplete_ulysses_metadata(
    monkeypatch,
    attention_mask,
    expected_message,
):
    monkeypatch.setattr(
        flex_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=True, ulysses_size=2),
    )
    veomni_attention.register_veomni_flex_attention_mask_builder()
    config = PreTrainedConfig()
    config._attn_implementation = _FLEX_IMPLEMENTATION
    local_inputs_embeds = torch.randn(1, 4, 16)

    with pytest.raises(ValueError, match=expected_message):
        create_causal_mask(config, local_inputs_embeds, attention_mask, None)


@pytest.mark.parametrize("implementation", [*_FLASH_IMPLEMENTATIONS, _FLEX_IMPLEMENTATION, _MAGI_IMPLEMENTATION])
def test_fused_attention_forward_dispatches_to_selected_adapter(monkeypatch, implementation):
    captured = {}

    def replacement_adapter(module, query, key, value, attention_mask, **kwargs):
        captured.update(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )
        return query.transpose(1, 2) + 1, "attention-metadata"

    monkeypatch.setitem(veomni_attention._ATTENTION_FORWARD_DISPATCH, implementation, replacement_adapter)
    module = _FakeAttentionModule(implementation)
    query = torch.randn(2, 4, 3, 4, dtype=torch.float16)
    key = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    value = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    attention_mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    marker = object()

    output, attention_metadata = veomni_attention.fused_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.25,
        scaling=0.5,
        sliding_window=16,
        softcap=30.0,
        skip_ulysses=True,
        contract_marker=marker,
    )

    assert captured["module"] is module
    assert captured["query"] is query
    assert captured["key"] is key
    assert captured["value"] is value
    assert captured["attention_mask"] is attention_mask
    assert captured["kwargs"] == {
        "dropout": 0.25,
        "scaling": 0.5,
        "sliding_window": 16,
        "softcap": 30.0,
        "skip_ulysses": True,
        "contract_marker": marker,
    }
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert attention_metadata == "attention-metadata"


def test_fused_attention_forward_rejects_unregistered_implementation():
    module = _FakeAttentionModule("unregistered_attention")
    query = torch.randn(1, 4, 3, 4, dtype=torch.float16)

    with pytest.raises(
        ValueError, match="Unsupported VeOmni fused attention implementation: 'unregistered_attention'"
    ):
        veomni_attention.fused_attention_forward(module, query, query, query, attention_mask=None)


def test_veomni_ops_config_rewrites_flex_to_sp_aware_registration(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    config = OpsImplementationConfig(attn_implementation="flex_attention")

    assert config.attn_implementation == _FLEX_IMPLEMENTATION


def test_veomni_ops_config_rewrites_magi_to_sp_aware_registration(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    config = OpsImplementationConfig(attn_implementation="magi_attention")

    assert config.attn_implementation == _MAGI_IMPLEMENTATION
