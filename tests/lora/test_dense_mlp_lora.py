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

"""Dense LoRA on the patched ``gate_proj`` / ``up_proj`` / ``down_proj`` MLPs."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from veomni.arguments import OpsImplementationConfig
from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
from veomni.lora.layers import LoraLinear
from veomni.ops import OP_REGISTRY
from veomni.ops.config import get_ops_config, set_ops_config


def _qwen3_mlp() -> nn.Module:
    from transformers import Qwen3Config

    from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3MLP

    return Qwen3MLP(Qwen3Config(hidden_size=16, intermediate_size=32, hidden_act="silu"))


def _llama_mlp() -> nn.Module:
    from transformers import LlamaConfig

    from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaMLP

    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4, hidden_act="silu", mlp_bias=True)
    return LlamaMLP(config)


def _gemma3_mlp() -> nn.Module:
    from transformers import Gemma3TextConfig

    from veomni.models.transformers.gemma3.generated.patched_modeling_gemma3_gpu import Gemma3MLP

    return Gemma3MLP(Gemma3TextConfig(hidden_size=16, intermediate_size=32, hidden_activation="gelu_pytorch_tanh"))


def _deepseek_v4_mlp() -> nn.Module:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import DeepseekV4MLP

    return DeepseekV4MLP(DeepseekV4Config(hidden_size=16, intermediate_size=32, hidden_act="silu", swiglu_limit=0.5))


_FAMILIES: dict[str, tuple[Callable[[], nn.Module], str]] = {
    "qwen3": (_qwen3_mlp, "standard"),
    "llama": (_llama_mlp, "standard"),
    "gemma3": (_gemma3_mlp, "geglu"),
    "deepseek_v4": (_deepseek_v4_mlp, "standard"),
}
_PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]


def _build_mlp(family: str, impl: str, dtype: torch.dtype, device: str) -> nn.Module:
    build, variant = _FAMILIES[family]
    if impl not in OP_REGISTRY.list_available("swiglu_mlp", variant):
        pytest.skip(f"swiglu_mlp/{variant}/{impl} is unavailable on this device")
    saved_cfg = get_ops_config()
    set_ops_config(OpsImplementationConfig(swiglu_mlp_implementation=impl))
    try:
        torch.manual_seed(0)
        mlp = build()
    finally:
        set_ops_config(saved_cfg)
    assert mlp.veomni_swiglu_mlp.impl == impl
    with torch.no_grad():
        for name in _PROJECTIONS:
            getattr(mlp, name).weight.normal_(0.0, 0.5)
    return mlp.to(device=device, dtype=dtype)


def _wrap(mlp: nn.Module) -> VeOmniLoraModel:
    return VeOmniLoraModel(mlp, VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=_PROJECTIONS))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("impl", ["eager", "liger_kernel"])
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_lora_projections_match_merged_weights(family: str, impl: str, monkeypatch: pytest.MonkeyPatch):
    """Adapters on every projection must change the output exactly as their merged weights do."""
    # TF32 rounding of the separate adapter matmuls exceeds the fp32 tolerance below.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    device = _device()
    mlp = _build_mlp(family, impl, torch.float32, device)
    x = torch.randn(2, 5, 16, device=device)
    base_out = mlp(x)

    model = _wrap(mlp)
    assert all(isinstance(getattr(mlp, name), LoraLinear) for name in _PROJECTIONS)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLinear):
                for lora_b in module.lora_B.values():
                    lora_b.weight.normal_(0.0, 0.1)

    lora_out = model(x)
    lora_out.sum().backward()
    for name in _PROJECTIONS:
        assert getattr(mlp, name).lora_B["default"].weight.grad is not None
    assert not torch.allclose(lora_out, base_out)

    merged = model.merge_and_unload()
    assert all(type(getattr(merged, name)) is nn.Linear for name in _PROJECTIONS)
    with torch.no_grad():
        torch.testing.assert_close(merged(x), lora_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_lora_projections_at_init_match_eager_row_bitwise(family: str):
    """The module path that LoRA takes must reproduce the eager ``swiglu_mlp`` row in bf16."""
    device = _device()
    mlp = _build_mlp(family, "eager", torch.bfloat16, device)
    x = torch.randn(2, 5, 16, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        base_out = mlp(x)
        lora_out = _wrap(mlp)(x)
    assert torch.equal(lora_out, base_out)
