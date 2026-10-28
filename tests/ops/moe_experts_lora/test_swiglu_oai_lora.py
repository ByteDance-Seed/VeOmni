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

"""CPU oracle and CPU-emulated NPU composition; these do not validate NPU kernels."""

import ast
import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _cpu_launch_plan(_context):
    from scripts.ci.accelerator.launch_plan import LaunchSpec

    return LaunchSpec(nproc=1, device="cpu")


pytestmark = pytest.mark.launch_plan(provider=_cpu_launch_plan)


_ROOT = Path(__file__).resolve().parents[3]
_PREFIX = "_swiglu_oai_lora_contract"


@pytest.fixture
def kernels(monkeypatch):
    """Load the exact kernel sources without optional model or device initialization."""
    parents = [
        "",
        ".ops",
        ".ops.kernels",
        ".ops.kernels.moe_experts_lora",
        ".ops.kernels.moe_experts_lora.independent",
        ".ops.kernels.moe_experts_lora.shared",
        ".ops.kernels.moe_experts",
        ".ops.kernels.moe_experts.standard",
        ".distributed",
        ".utils",
        ".ops.kernels.moe_experts.shared",
    ]
    for suffix in parents:
        package = types.ModuleType(_PREFIX + suffix)
        package.__path__ = [str(_ROOT / "veomni" / suffix.lstrip(".").replace(".", "/"))]
        monkeypatch.setitem(sys.modules, package.__name__, package)
    sys.modules[_PREFIX + ".utils"].logging = SimpleNamespace(get_logger=logging.getLogger)
    state = types.ModuleType(_PREFIX + ".distributed.parallel_state")
    state.get_parallel_state = lambda: SimpleNamespace(ep_enabled=False)
    monkeypatch.setitem(sys.modules, state.__name__, state)
    module = importlib.import_module(_PREFIX + ".ops.kernels.moe_experts_lora.swiglu_oai")
    yield module
    for name in list(sys.modules):
        if name.startswith(_PREFIX):
            sys.modules.pop(name, None)


@pytest.fixture
def npu_cpu_primitives(monkeypatch):
    """Emulate routing/GEMM only; autograd and activation math stay real Torch."""
    npu = types.ModuleType("torch_npu")

    def permute(hidden, selected):
        order = torch.argsort(selected.reshape(-1), stable=True)
        token_ids = torch.arange(hidden.shape[0]).repeat_interleave(selected.shape[1])[order]
        return hidden[token_ids], order

    def unpermute(output, order, probs):
        token_ids = torch.div(order, probs.shape[1], rounding_mode="floor")
        result = output.new_zeros((probs.shape[0], output.shape[-1]))
        return result.index_add(0, token_ids, output * probs.reshape(-1)[order, None])

    def group_gemm(hidden, weight, counts):
        chunks = hidden.split(counts.tolist())
        return torch.cat([part @ weight[i] for i, part in enumerate(chunks)])

    npu.npu_moe_token_permute = permute
    npu.npu_moe_token_unpermute = unpermute
    npu.npu_swiglu = lambda gate_up, dim: F.silu(gate_up.chunk(2, dim)[0]) * gate_up.chunk(2, dim)[1]
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    standard = types.ModuleType(_PREFIX + ".ops.kernels.moe_experts.standard.npu")
    standard.npu_group_gemm = group_gemm
    monkeypatch.setitem(sys.modules, standard.__name__, standard)


def tensors(layout, dtype, tokens=5):
    generator = torch.Generator().manual_seed(2307)

    def rand(shape):
        return torch.randn(shape, generator=generator, dtype=dtype).requires_grad_()

    experts, hidden, intermediate, rank = 3, 4, 3, 2
    prefix = (experts,) if layout == "independent" else ()
    return [
        rand((tokens, hidden)),
        rand((tokens, 2)),
        torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.int64)[:tokens],
        rand((experts, 2 * intermediate, hidden)),
        rand((experts, hidden, intermediate)),
        rand((*prefix, rank, hidden)),
        rand((*prefix, intermediate, rank)),
        rand((*prefix, rank, hidden)),
        rand((*prefix, intermediate, rank)),
        rand((*prefix, rank, intermediate)),
        rand((*prefix, hidden, rank)),
    ]


def oracle(args, layout, alpha, limit):
    hidden, routing, selected, fc1, fc2, ag, bg, au, bu, ad, bd = args

    def expert(weight, index):
        return weight[index] if layout == "independent" else weight

    rows = []
    for token in range(hidden.shape[0]):
        total = torch.zeros_like(hidden[token])
        for top in range(selected.shape[1]):
            index = int(selected[token, top])
            gate, up = F.linear(hidden[token], fc1[index]).chunk(2)
            gate = gate + 0.3 * F.linear(F.linear(hidden[token], expert(ag, index)), expert(bg, index))
            up = up + 0.7 * F.linear(F.linear(hidden[token], expert(au, index)), expert(bu, index))
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            mid = (gate * torch.sigmoid(alpha * gate)) * (up + 1.0)
            output = F.linear(mid, fc2[index])
            output = output + 0.2 * F.linear(F.linear(mid, expert(ad, index)), expert(bd, index))
            total = total + routing[token, top] * output
        rows.append(total)
    return torch.stack(rows)


@pytest.mark.parametrize("layout", ["shared", "independent"])
@pytest.mark.parametrize("backend", ["eager", "npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alpha,limit", [(1.702, 7.0), (0.9, 1.0)])
def test_forward_and_all_tensor_gradients_match_independent_oracle(
    kernels, npu_cpu_primitives, layout, backend, dtype, alpha, limit
):
    args = tensors(layout, dtype)
    wrapper = getattr(kernels, layout + "_" + backend + "_wrapper")
    output = wrapper(
        *args,
        num_experts=3,
        lora_scale_gate=0.3,
        lora_scale_up=0.7,
        lora_scale_down=0.2,
        swiglu_alpha=alpha,
        swiglu_limit=limit,
    )
    expected = oracle(args, layout, alpha, limit)
    tolerance = 3e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
    differentiable = [value for value in args if value.requires_grad]
    gradient = torch.autograd.grad(output.square().sum(), differentiable, retain_graph=True)
    reference_gradient = torch.autograd.grad(expected.square().sum(), differentiable)
    for actual, reference in zip(gradient, reference_gradient, strict=True):
        torch.testing.assert_close(actual, reference, rtol=tolerance * 10, atol=tolerance * 10)


@pytest.mark.parametrize("alpha,limit", [(0.0, 7.0), (float("nan"), 7.0), (1.702, -1.0), (1.702, float("inf"))])
def test_invalid_activation_attributes_fail_explicitly(kernels, alpha, limit):
    with pytest.raises(ValueError, match="positive finite"):
        kernels._activation(alpha, limit)


@pytest.fixture
def runtime(kernels, monkeypatch):
    """Load real registry and LoRA classes while isolating optional package startup."""
    import transformers.utils
    import transformers.utils.import_utils

    # A package-install probe must not initialize optional CANN libraries for
    # this explicitly CPU-only launch plan.
    monkeypatch.setattr(transformers.utils, "is_torch_npu_available", lambda: False)
    monkeypatch.setattr(transformers.utils.import_utils, "is_torch_npu_available", lambda: False)
    for suffix in (".lora", ".utils", ".ops.kernels.moe_experts.swiglu_oai"):
        package = types.ModuleType(_PREFIX + suffix)
        package.__path__ = [str(_ROOT / "veomni" / suffix.lstrip(".").replace(".", "/"))]
        monkeypatch.setitem(sys.modules, package.__name__, package)
    sys.modules[_PREFIX + ".utils"].logging = SimpleNamespace(get_logger=logging.getLogger)
    registry = importlib.import_module(_PREFIX + ".ops.registry")
    ops = sys.modules[_PREFIX + ".ops"]
    ops.OP_REGISTRY, ops.VeomniOp = registry.OP_REGISTRY, registry.VeomniOp
    monkeypatch.setitem(sys.modules, "veomni.ops", ops)
    name = _PREFIX + ".ops.kernels.moe_experts_lora"
    spec = importlib.util.spec_from_file_location(name, _ROOT / "veomni/ops/kernels/moe_experts_lora/__init__.py")
    family = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, family)
    spec.loader.exec_module(family)
    base_eager = importlib.import_module(_PREFIX + ".ops.kernels.moe_experts.swiglu_oai.eager")
    registry.register_op(
        "moe_experts", "swiglu_oai", "eager", description="MiniMax dependency eager", wrapper=base_eager.wrapper
    )
    standard_eager = importlib.import_module(_PREFIX + ".ops.kernels.moe_experts.standard.eager")
    registry.register_op(
        "moe_experts", "standard", "eager", description="Standard dependency eager", wrapper=standard_eager.wrapper
    )
    layers = importlib.import_module(_PREFIX + ".lora.moe_layers")

    return SimpleNamespace(registry=registry, layers=layers)


def make_experts(runtime, dtype=torch.float32):
    """Minimal base module with real operator dispatch and expert parameter layout."""

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts, self.hidden_dim, self.intermediate_dim = 3, 4, 3
            self.swiglu_alpha, self.swiglu_limit = 1.702, 1.0
            self.gate_up_proj = torch.nn.Parameter(torch.randn(3, 6, 4, dtype=dtype))
            self.down_proj = torch.nn.Parameter(torch.randn(3, 4, 3, dtype=dtype))
            self.veomni_moe = runtime.registry.VeomniOp("moe_experts", "swiglu_oai", "eager")

    return Experts()


def test_native_wrapper_accepts_explicit_oai_without_an_act_fn(runtime):
    expert = make_experts(runtime)
    runtime.layers.LoraIndependentExperts(expert, r=2, lora_alpha=4)
    expert = make_experts(runtime)
    expert.swiglu_limit = float("nan")
    with pytest.raises(ValueError, match="swiglu_limit"):
        runtime.layers.LoraIndependentExperts(expert, r=2, lora_alpha=4)


@pytest.mark.parametrize("layout", ["shared", "independent"])
def test_expert_reference_bypass_never_reads_adapter_weights(runtime, monkeypatch, layout):
    base = make_experts(runtime)
    cls = runtime.layers.LoraSharedExperts if layout == "shared" else runtime.layers.LoraIndependentExperts
    hidden, routing, selected = tensors(layout, torch.float32)[:3]
    unused = hidden.new_empty(0)
    expected = base.veomni_moe(
        hidden,
        routing,
        selected,
        unused,
        unused,
        base.down_proj,
        base.gate_up_proj,
        num_experts=3,
        swiglu_alpha=1.702,
        swiglu_limit=1.0,
    )
    layer = cls(base, r=2, lora_alpha=4)

    def forbidden(*args, **kwargs):
        pytest.fail("Adapter weights must not be read in reference inference")

    monkeypatch.setattr(layer, "get_lora_A_weight", forbidden)
    monkeypatch.setattr(layer, "get_lora_B_weight", forbidden)
    layer.disable_adapters = True
    torch.testing.assert_close(layer(hidden, selected, routing), expected, rtol=0, atol=0)


def test_expert_disable_context_restores_nested_and_exception_states(runtime):
    model_module = importlib.import_module(_PREFIX + ".lora.model")
    layers = importlib.import_module(_PREFIX + ".lora.layers")
    config = importlib.import_module(_PREFIX + ".lora.config")
    dense = layers.LoraLinear(torch.nn.Linear(4, 4), adapter_name="default", r=2, lora_alpha=4)
    dense.disable_adapters = True
    expert = runtime.layers.LoraIndependentExperts(make_experts(runtime), r=2, lora_alpha=4)
    base = torch.nn.ModuleDict({"dense": dense, "expert": expert})
    model = model_module.VeOmniLoraModel(base, config.VeOmniLoraConfig(target_modules=["dense"]), inject=False)
    with pytest.raises(RuntimeError, match="inside"):
        with model.disable_adapter():
            assert dense.disable_adapters and expert.disable_adapters
            with model.disable_adapter():
                assert expert.disable_adapters
            assert expert.disable_adapters
            raise RuntimeError("inside")
    assert dense.disable_adapters and not expert.disable_adapters
    expert._base_moe = None
    with pytest.raises(NotImplementedError, match="base VeomniOp"):
        with model.disable_adapter():
            pytest.fail("Unsupported experts must fail before the context starts")
    assert dense.disable_adapters and not expert.disable_adapters


def test_clamped_standard_experts_are_rejected_before_wrapping(runtime):
    expert = make_experts(runtime)
    expert.veomni_moe = runtime.registry.VeomniOp("moe_experts", "standard", "eager")
    expert.act_fn, expert.limit = F.silu, 1.0
    with pytest.raises(ValueError, match="Clamped standard"):
        runtime.layers.LoraIndependentExperts(expert, r=2, lora_alpha=4)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_opt_in_upcast_preserves_frozen_base_and_buffers(device):
    path = _ROOT / "veomni/distributed/torch_parallelize.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_upcast_trainable_parameters"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    module = torch.nn.Module()
    module.base = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16, device=device), requires_grad=False)
    module.router = torch.nn.Parameter(torch.ones(4, device=device), requires_grad=False)
    module.adapter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16, device=device))
    module.register_buffer("score_bias", torch.ones(4, device=device))
    original = list(module.parameters())
    scope["_upcast_trainable_parameters"](module)
    assert all(a is b for a, b in zip(original, module.parameters(), strict=True))
    assert module.base.dtype == torch.bfloat16
    assert module.router.dtype == module.adapter.dtype == module.score_bias.dtype == torch.float32


@pytest.mark.parametrize(
    "enabled,peft,fsdp,mode,error",
    [
        (True, True, True, "fsdp2", None),
        (False, False, False, "ddp", None),
        (True, True, False, "ddp", ValueError),
        (True, True, False, "fsdp2", ValueError),
        (True, False, True, "fsdp2", ValueError),
        (1, True, True, "fsdp2", TypeError),
    ],
)
def test_preserve_frozen_dtype_rejects_execution_without_parameter_casting(enabled, peft, fsdp, mode, error):
    path = _ROOT / "veomni/distributed/torch_parallelize.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate_preserve_frozen_dtype"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    state = SimpleNamespace(fsdp_enabled=fsdp, dp_mode=mode)
    if error is not None:
        with pytest.raises(error, match="preserve_frozen_dtype"):
            scope["_validate_preserve_frozen_dtype"](
                enabled, peft, state, SimpleNamespace(enable=True, param_dtype="bfloat16")
            )
    else:
        scope["_validate_preserve_frozen_dtype"](
            enabled, peft, state, SimpleNamespace(enable=True, param_dtype="bfloat16")
        )


@pytest.mark.parametrize("enable,param_dtype", [(False, "bfloat16"), (True, None)])
def test_preserve_frozen_dtype_requires_active_compute_dtype(enable, param_dtype):
    path = _ROOT / "veomni/distributed/torch_parallelize.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate_preserve_frozen_dtype"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    with pytest.raises(ValueError, match="non-None param_dtype"):
        scope["_validate_preserve_frozen_dtype"](
            True,
            True,
            SimpleNamespace(fsdp_enabled=True, dp_mode="fsdp2"),
            SimpleNamespace(enable=enable, param_dtype=param_dtype),
        )


@pytest.mark.parametrize("layout", ["shared", "independent"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_registry_and_expert_wrapper_reference_parity(runtime, layout, dtype):
    """Real registry, family registration, constructors and kernels; CPU, EP disabled.

    Only package startup/logging and distributed state are isolated. The base
    eager operator is the unchanged dependency from MiniMax model PR #1315.
    This does not import the complete generated multimodal model or run NPU.
    """
    torch.manual_seed(1702)
    base = make_experts(runtime, dtype)
    hidden = torch.randn(5, 4, dtype=dtype)
    selected = torch.tensor([[0, 1], [1, 0], [2, 1], [0, 2], [1, 2]])
    routing = torch.softmax(torch.randn(5, 2, dtype=dtype), dim=-1)
    unused = hidden.new_empty(0)
    expected = base.veomni_moe(
        hidden,
        routing,
        selected,
        unused,
        unused,
        base.down_proj,
        base.gate_up_proj,
        num_experts=3,
        swiglu_alpha=1.702,
        swiglu_limit=1.0,
    )
    cls = runtime.layers.LoraSharedExperts if layout == "shared" else runtime.layers.LoraIndependentExperts
    wrapped = cls(base, r=2, lora_alpha=4)
    assert wrapped.veomni_moe_lora.variant == layout + "_swiglu_oai"
    torch.testing.assert_close(wrapped(hidden, selected, routing), expected, rtol=0, atol=0)
    for logical in ("gate_proj", "up_proj", "down_proj"):
        torch.nn.init.normal_(wrapped.get_lora_B_weight(logical), std=0.1)
    adapted = wrapped(hidden, selected, routing)
    assert not torch.allclose(adapted, expected, rtol=0.01, atol=0.01)
    wrapped.disable_adapters = True
    torch.testing.assert_close(wrapped(hidden, selected, routing), expected, rtol=0, atol=0)
