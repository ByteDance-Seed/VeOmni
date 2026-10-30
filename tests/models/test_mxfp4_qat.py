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

"""CPU train-to-export numerical check for Dense Qwen3 MXFP4 QAT."""

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.tiny_configs import tiny_qwen3_config
from tests.tools.training_utils import make_eager_ops_config
from veomni.distributed.parallel_state import ParallelState, use_parallel_state
from veomni.models import build_config, build_foundation_model
from veomni.models.qat import export_mxfp4, load_mxfp4


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
pytest.importorskip("torchao", minversion="0.18.0")
MXTensor = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor").MXTensor
MXFakeQuantizedLinear = pytest.importorskip("torchao.prototype.qat").MXFakeQuantizedLinear
_TARGETS = ["model.layers.*.self_attn.*_proj", "model.layers.*.mlp.*_proj"]


@pytest.fixture(autouse=True)
def _cpu_model(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    with use_parallel_state(ParallelState(device_type="cpu")):
        yield


def _bf16_forward(model, tokens, **kwargs):
    # Cast parameters while retaining differentiable FP32 masters and FP32 RoPE buffers.
    params = {name: param.to(torch.bfloat16) for name, param in model.named_parameters()}
    return torch.func.functional_call(model, params, (), {"input_ids": tokens, "use_cache": False, **kwargs})


def test_qwen3_mxfp4_training_and_export_roundtrip(tmp_path):
    torch.manual_seed(31)
    model = build_foundation_model(
        tiny_qwen3_config(num_hidden_layers=1),
        init_device="cpu",
        torch_dtype="float32",
        ops_implementation=make_eager_ops_config(qat_implementation="torchao_mxfp4", qat_target_modules=_TARGETS),
    )
    prepared = {name: module for name, module in model.named_modules() if isinstance(module, MXFakeQuantizedLinear)}
    assert len(prepared) == 7 and type(model.lm_head) is torch.nn.Linear
    assert ".generated." in type(model).__module__
    before = {name: module.weight.detach().clone() for name, module in prepared.items()}
    keys = set(model.state_dict())
    called = set()
    handles = [
        module.register_forward_hook(lambda _module, _inputs, _output, name=name: called.add(name))
        for name, module in prepared.items()
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.tensor([[3, 7, 12, 5, 11, 8, 4], [6, 9, 15, 14, 10, 3, 13]])
    for microbatch in tokens.split(1):
        loss = _bf16_forward(model, microbatch, labels=microbatch).loss.sum() / len(tokens)
        assert torch.isfinite(loss)
        loss.backward()
    for handle in handles:
        handle.remove()
    assert called == set(prepared)
    for module in prepared.values():
        grad = module.weight.grad
        assert grad is not None and grad.dtype == torch.float32
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    optimizer.step()
    assert set(model.state_dict()) == keys
    assert all(param.dtype == torch.float32 for param in model.parameters())
    assert all(not torch.equal(module.weight, before[name]) for name, module in prepared.items())

    model.eval()
    with torch.no_grad():
        expected = _bf16_forward(model, tokens).logits
    hf_path = tmp_path / "hf"
    model.save_pretrained(hf_path)
    reloaded = build_foundation_model(
        build_config(str(hf_path)), weights_path=str(hf_path), init_device="cpu", torch_dtype="float32"
    ).eval()
    assert reloaded.config.ops_config.qat_implementation == "torchao_mxfp4"
    assert reloaded.config.ops_config.qat_target_modules == _TARGETS
    with torch.no_grad():
        torch.testing.assert_close(_bf16_forward(reloaded, tokens).logits, expected, rtol=0, atol=0)

    export_mxfp4(reloaded, tmp_path / "packed")
    torch.testing.assert_close(reloaded.state_dict(), model.state_dict(), rtol=0, atol=0)
    assert all(param.dtype == torch.float32 for param in reloaded.parameters())
    packed = load_mxfp4(tmp_path / "packed")
    assert sum(isinstance(param, MXTensor) for param in packed.parameters()) == len(prepared)
    with torch.no_grad():
        torch.testing.assert_close(packed(input_ids=tokens, use_cache=False).logits, expected, rtol=0, atol=0)
