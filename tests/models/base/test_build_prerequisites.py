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

"""Build-time refusals that ``build_foundation_model`` must raise before any weight is read."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig

import veomni.models.auto as auto
from tests.models.compare import eager_ops_config, ops_config_scope
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.ops.config import get_ops_config


_TOY_CONFIG_DIR = Path(__file__).resolve().parents[2] / "toy_config"
_HUB_FLASH_IMPLEMENTATIONS = (
    "flash_attention_2_hub",
    "flash_attention_3_hub",
    "veomni_flash_attention_2_hub",
    "veomni_flash_attention_3_hub",
)


def _normalized_hub_name(implementation: str) -> str:
    return implementation if implementation.startswith("veomni_") else f"veomni_{implementation}"


def _forbid_config_load(monkeypatch: pytest.MonkeyPatch) -> Mock:
    build_config = Mock(side_effect=AssertionError("model config loading must not start"))
    monkeypatch.setattr(auto, "build_config", build_config)
    return build_config


def _stub_loader(monkeypatch: pytest.MonkeyPatch) -> Mock:
    load_model = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(auto, "get_loader", lambda config: SimpleNamespace(load_model=load_model))
    return load_model


def _host(monkeypatch: pytest.MonkeyPatch, *, on_npu: bool, modeling_backend: str) -> None:
    monkeypatch.setenv("MODELING_BACKEND", modeling_backend)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_npu_available", lambda: on_npu)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr(auto, "is_torch_npu_available", lambda: on_npu)


# --- MagiAttention context parallelism ---------------------------------------------


@pytest.mark.parametrize("entry", ("ops", "installed", "override"))
@pytest.mark.parametrize("attn_implementation", ("magi_attention", "veomni_magi_attention"))
def test_magi_context_parallelism_is_refused_before_config_load(
    monkeypatch: pytest.MonkeyPatch, entry: str, attn_implementation: str
):
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_size=2))
    build_config = _forbid_config_load(monkeypatch)
    installed = eager_ops_config()
    if entry == "installed":
        installed.attn_implementation = attn_implementation

    with ops_config_scope(installed):
        with pytest.raises(ValueError, match=r"MagiAttention.*cp_size == 1.*cp_size=2"):
            if entry == "ops":
                auto.build_foundation_model(
                    config_path="unused",
                    ops_implementation=SimpleNamespace(attn_implementation=attn_implementation),
                )
            elif entry == "override":
                auto.build_foundation_model(config_path="unused", attn_implementation=attn_implementation)
            else:
                auto.build_foundation_model(config_path="unused")
        assert get_ops_config() is installed, "a refused build must not install its ops config"

    build_config.assert_not_called()


# --- Hub FlashAttention ------------------------------------------------------------


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
def test_hub_flash_config_maps_to_veomni_backend(monkeypatch: pytest.MonkeyPatch, implementation: str):
    _host(monkeypatch, on_npu=False, modeling_backend="veomni")

    config = OpsImplementationConfig(attn_implementation=implementation)

    assert config.attn_implementation == _normalized_hub_name(implementation)


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
def test_hub_flash_config_rejects_huggingface_modeling_backend(monkeypatch: pytest.MonkeyPatch, implementation: str):
    _host(monkeypatch, on_npu=False, modeling_backend="hf")

    with pytest.raises(ValueError, match="requires MODELING_BACKEND=veomni"):
        OpsImplementationConfig(attn_implementation=implementation)


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
def test_hub_flash_config_rejects_npu(monkeypatch: pytest.MonkeyPatch, implementation: str):
    _host(monkeypatch, on_npu=True, modeling_backend="veomni")

    with pytest.raises(ValueError, match="not supported on Ascend NPU"):
        OpsImplementationConfig(attn_implementation=implementation)


@pytest.mark.parametrize("on_npu", (False, True))
@pytest.mark.parametrize("version", (2, 3))
def test_local_flash_config_is_not_treated_as_hub(monkeypatch: pytest.MonkeyPatch, on_npu: bool, version: int):
    _host(monkeypatch, on_npu=on_npu, modeling_backend="veomni")

    config = OpsImplementationConfig(attn_implementation=f"flash_attention_{version}")

    assert config.attn_implementation == f"veomni_flash_attention_{version}"


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("on_npu", "modeling_backend", "error"),
    ((True, "veomni", "Ascend NPU"), (False, "hf", "MODELING_BACKEND")),
)
def test_hub_flash_override_is_refused_before_config_load(
    monkeypatch: pytest.MonkeyPatch, implementation: str, on_npu: bool, modeling_backend: str, error: str
):
    _host(monkeypatch, on_npu=on_npu, modeling_backend=modeling_backend)
    build_config = _forbid_config_load(monkeypatch)
    preload = Mock(side_effect=AssertionError("HF flash-attention preloading must not start"))
    hub_load = Mock(side_effect=AssertionError("Hub kernel loading must not start"))
    monkeypatch.setattr("transformers.modeling_utils.lazy_import_flash_attention", preload)
    monkeypatch.setattr("transformers.integrations.hub_kernels.load_and_register_attn_kernel", hub_load)
    installed = eager_ops_config()

    with ops_config_scope(installed):
        with pytest.raises(ValueError, match=error):
            auto.build_foundation_model("unused", attn_implementation=implementation)
        assert get_ops_config() is installed

    build_config.assert_not_called()
    preload.assert_not_called()
    hub_load.assert_not_called()


@pytest.mark.parametrize("implementation", _HUB_FLASH_IMPLEMENTATIONS)
@pytest.mark.parametrize("entry", ("ops", "override", "installed"))
def test_hub_flash_build_passes_normalized_name_to_loader(
    monkeypatch: pytest.MonkeyPatch, implementation: str, entry: str
):
    _host(monkeypatch, on_npu=False, modeling_backend="veomni")
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=False, global_rank=0))
    load_model = _stub_loader(monkeypatch)
    installed = eager_ops_config()
    kwargs = {}
    if entry == "ops":
        kwargs["ops_implementation"] = OpsImplementationConfig(attn_implementation=implementation)
    elif entry == "override":
        kwargs["attn_implementation"] = implementation
    else:
        installed = OpsImplementationConfig(attn_implementation=implementation)

    with ops_config_scope(installed):
        auto.build_foundation_model(PretrainedConfig(), init_device="cpu", **kwargs)

    assert load_model.call_args.kwargs["init_kwargs"]["attn_implementation"] == _normalized_hub_name(implementation)


# --- Model-declared prerequisites ----------------------------------------------------


class _StubLoader:
    """Counts constructions so the prerequisite gate is the only thing under test."""

    def __init__(self) -> None:
        self.calls = 0

    def load_model(self, **kwargs) -> torch.nn.Module:
        self.calls += 1
        return torch.nn.Identity()


def _build_with_indexer_objective(
    monkeypatch: pytest.MonkeyPatch,
    toy: str,
    loader: _StubLoader,
    *,
    indexer_loss: bool = True,
    coef: float = 1.0,
    implementation: str = "tilelang",
) -> int:
    """Run ``build_foundation_model`` with the KL objective on the config and its kernels on the ops config."""
    monkeypatch.setattr(
        auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=False, sp_enabled=False, global_rank=0)
    )
    monkeypatch.setattr(auto, "get_loader", lambda config: loader)
    ops = eager_ops_config()
    ops.dsa_indexer_implementation = implementation
    ops.dsa_attention_implementation = implementation
    config = auto.build_config(str(_TOY_CONFIG_DIR / toy), dsa_indexer_loss=indexer_loss, dsa_indexer_loss_coef=coef)

    with ops_config_scope(ops):
        auto.build_foundation_model(config_path=config, init_device="meta")
    return loader.calls


def test_unsupported_indexer_objective_is_refused_before_the_model_is_built(monkeypatch: pytest.MonkeyPatch):
    loader = _StubLoader()
    with pytest.raises(ValueError, match="dsa_indexer_loss requires"):
        _build_with_indexer_objective(monkeypatch, "deepseek_v4_toy", loader, implementation="eager")
    assert loader.calls == 0, "the refusal must come before the model is constructed"


def test_supported_indexer_objective_builds(monkeypatch: pytest.MonkeyPatch):
    assert _build_with_indexer_objective(monkeypatch, "deepseek_v4_toy", _StubLoader()) == 1


def test_disabled_indexer_objective_admits_eager_kernels(monkeypatch: pytest.MonkeyPatch):
    calls = _build_with_indexer_objective(
        monkeypatch, "deepseek_v4_toy", _StubLoader(), indexer_loss=False, implementation="eager"
    )
    assert calls == 1


def test_zero_coefficient_admits_eager_kernels(monkeypatch: pytest.MonkeyPatch):
    calls = _build_with_indexer_objective(
        monkeypatch, "deepseek_v4_toy", _StubLoader(), coef=0.0, implementation="eager"
    )
    assert calls == 1


def test_model_without_the_hook_is_not_refused(monkeypatch: pytest.MonkeyPatch):
    calls = _build_with_indexer_objective(
        monkeypatch, "glm_moe_dsa_toy", _StubLoader(), indexer_loss=False, implementation="eager"
    )
    assert calls == 1
