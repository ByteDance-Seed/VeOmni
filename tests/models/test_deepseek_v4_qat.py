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

"""QAT wiring for DeepSeek V4's patched modeling.

The recipe -- which tensors are rounded the way FP8 inference rounds them, and
which stay in the model dtype -- is expressed as *which call sites go through*
the ``veomni_qat_*`` helpers. That makes it invisible to a forward-output test:
quantizing one tensor too few still produces a plausible loss curve, it just
trains for a kernel nobody deploys. So the coverage tests read the generated
source instead of running it, and fail on any site they have not been told about.

Numerics live in ``tests/ops/test_qat_fp8_blockwise.py``; what is checked here is
the wiring, plus the two properties that distinguish the recipes from each other
(the KV split at the RoPE boundary, and the indexer covering the whole head).
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling_gpu
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_npu as modeling_npu
from veomni.ops.dispatch import OpsConfigSlot
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


DEVICE = get_device_type()

# The projections DeepSeek V4 serves as a true FP8 GEMM, and the ones it does
# not. Both halves are pinned: a new projection that lands in neither set fails
# the test rather than silently defaulting to unquantized.
#
# `weights_proj` produces one score per head, so its [index_n_heads,
# hidden_size] weight is too short to tile at 128 even if we wanted it. The
# compressors run their projections in FP32 upstream. `DeepseekV4MLP` is the
# shared expert and nothing else -- every V4 layer is an MoE block, so there are
# no dense MLPs sharing the class. The routed experts live in
# `DeepseekV4Experts` and are a separate recipe (FP4 weights on V4-Flash).
_EXPECTED_QAT = {
    "DeepseekV4Attention.forward": {"q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj"},
    "DeepseekV4Indexer.forward": {"q_b_proj"},
    "DeepseekV4HCACompressor.forward": set(),
    "DeepseekV4CSACompressor.forward": set(),
    "DeepseekV4MLP.forward": {"gate_proj", "up_proj", "down_proj"},
}
_EXPECTED_PLAIN = {
    "DeepseekV4Attention.forward": set(),
    "DeepseekV4Indexer.forward": {"weights_proj", "kv_proj", "gate_proj"},
    "DeepseekV4HCACompressor.forward": {"kv_proj", "gate_proj"},
    "DeepseekV4CSACompressor.forward": {"kv_proj", "gate_proj"},
    "DeepseekV4MLP.forward": set(),
}

# The activation-only recipe: tensors quantized because inference *stores* them
# that way, listed as (helper, argument) pairs.
#
# The counts carry weight. Each compressor finalizes its output in two separate
# branches -- one for packed sequences, one for the windowed/cached path -- and
# quantizing only one of them would under-quantize a real training
# configuration while leaving every shape and loss curve plausible. The main
# attention's Q appears nowhere on purpose: it is never stored, so inference
# keeps it in BF16 all the way into attention.
_EXPECTED_ACT_QUANT = {
    "DeepseekV4Attention.forward": [("veomni_qat_fake_quant_kv", "kv")],
    "DeepseekV4Indexer.forward": [
        ("veomni_qat_fake_quant_act", "compressed"),
        ("veomni_qat_fake_quant_act", "q"),
    ],
    "DeepseekV4HCACompressor.forward": [
        ("veomni_qat_fake_quant_kv", "compressed"),
        ("veomni_qat_fake_quant_kv", "compressed"),
    ],
    "DeepseekV4CSACompressor.forward": [
        ("veomni_qat_fake_quant_kv", "compressed"),
        ("veomni_qat_fake_quant_kv", "compressed"),
    ],
    "DeepseekV4MLP.forward": [],
    # The routed experts, wired on the fused path only. Two tensors are absent
    # on purpose: the intermediate feeding the second expert GEMM never leaves
    # the fused MoE autograd function, so it cannot be reached from the model,
    # and the kernel's output stays in the model dtype because inference
    # combines the expert results in BF16.
    "DeepseekV4Experts.forward": [
        ("veomni_qat_fake_quant_expert_weight", "self.down_proj"),
        ("veomni_qat_fake_quant_expert_weight", "self.gate_up_proj"),
        ("veomni_qat_fake_quant_act", "hidden_states"),
    ],
}


def _require_tilelang_cuda():
    pytest.importorskip("tilelang")
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        pytest.skip("DeepSeek V4 TileLang kernels require an NVIDIA CUDA GPU")
    if get_gpu_compute_capability() < 90:
        pytest.skip("DeepSeek V4 TileLang kernels require SM90 or later")


def _method_ast(module, qualname):
    cls_name, method_name = qualname.split(".")
    source = inspect.getsource(getattr(getattr(module, cls_name), method_name))
    return ast.parse(textwrap.dedent(source))


def _act_quant_calls(module, qualname):
    """List the ``(helper, first argument)`` pairs of activation fake-quant calls."""
    calls = []
    for node in ast.walk(_method_ast(module, qualname)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id.startswith("veomni_qat_fake_quant"):
            calls.append((node.func.id, ast.unparse(node.args[0])))
    return sorted(calls)


def _projection_calls(module, qualname):
    """Split the projections *qualname* calls into quantized and plain sets."""
    tree = _method_ast(module, qualname)
    quantized, plain = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "veomni_qat_linear":
            operand = node.args[0]
            assert isinstance(operand, ast.Attribute) and operand.value.id == "self", (
                f"{qualname}: veomni_qat_linear must be handed a submodule of self, got {ast.dump(operand)}"
            )
            quantized.add(operand.attr)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            if func.attr.endswith("_proj"):
                plain.add(func.attr)
    return quantized, plain


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_qat_covers_exactly_the_projections_served_as_fp8_gemms(modeling):
    for qualname in _EXPECTED_QAT:
        quantized, plain = _projection_calls(modeling, qualname)
        assert quantized == _EXPECTED_QAT[qualname], (
            f"{qualname}: fake-quantized projections drifted from the recipe. "
            f"Update _EXPECTED_QAT only if inference changed too."
        )
        assert plain == _EXPECTED_PLAIN[qualname], (
            f"{qualname}: unquantized projections drifted. A projection added here is trained in "
            f"the model dtype; confirm inference does the same before listing it."
        )


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_qat_covers_exactly_the_activations_inference_stores_quantized(modeling):
    for qualname, expected in _EXPECTED_ACT_QUANT.items():
        assert _act_quant_calls(modeling, qualname) == sorted(expected), (
            f"{qualname}: activation fake-quant sites drifted from the recipe. Check both the "
            f"packed and the windowed branch before updating _EXPECTED_ACT_QUANT."
        )


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_main_attention_query_is_never_fake_quantized(modeling):
    """The one activation the recipe calls out as staying BF16.

    Quantizing it would be an easy mistake to make by symmetry with the indexer,
    where both sides of the product *are* rounded.
    """
    quantized_args = {arg for _, arg in _act_quant_calls(modeling, "DeepseekV4Attention.forward")}
    assert "q" not in quantized_args and "q_residual" not in quantized_args


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_qat_is_off_until_asked_for(modeling):
    """The slot has to default to off: NPU and pre-SM90 GPUs have no kernel."""
    assert isinstance(modeling.veomni_qat_implementation, OpsConfigSlot)
    assert modeling.veomni_qat_implementation.field_name == "qat_implementation"
    assert modeling.veomni_qat_implementation.value == "none"


def test_qat_slot_binds_from_the_ops_config():
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops

    slot = OpsConfigSlot("qat_implementation", default="none")
    module = type("FakeModule", (), {"veomni_qat_implementation": slot})

    _bind_veomni_ops(module, OpsImplementationConfig(qat_implementation="fp8_blockwise"))

    assert slot.value == "fp8_blockwise"


def test_qat_linear_helper_is_a_passthrough_while_disabled(monkeypatch):
    """Disabled QAT must be bit-identical, not merely close.

    Every call site routes through the helper whether or not QAT is on, so an
    `enabled=False` path that perturbed the result would change the numerics of
    every run that never asked for QAT.
    """
    monkeypatch.setattr(modeling_gpu, "veomni_qat_implementation", OpsConfigSlot("qat_implementation", default="none"))

    torch.manual_seed(0)
    linear = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    x = torch.randn(4, 256, dtype=torch.bfloat16)

    assert torch.equal(modeling_gpu.veomni_qat_linear(linear, x), linear(x))


def test_qat_linear_helper_fake_quantizes_when_enabled(monkeypatch):
    _require_tilelang_cuda()
    from veomni.ops.qat import qat_linear

    monkeypatch.setattr(
        modeling_gpu,
        "veomni_qat_implementation",
        OpsConfigSlot("qat_implementation", default="fp8_blockwise"),
    )

    torch.manual_seed(0)
    linear = nn.Linear(256, 128, bias=False, device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(8, 256, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    actual = modeling_gpu.veomni_qat_linear(linear, x)

    assert torch.equal(actual, qat_linear(linear, x, enabled=True))
    assert not torch.equal(actual, linear(x)), "the enabled path returned the unquantized product"

    # The straight-through estimator has to reach the real parameter, otherwise
    # QAT would train nothing.
    actual.sum().backward()
    assert linear.weight.grad is not None and linear.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "helper, args",
    [("veomni_qat_fake_quant_kv", (64,)), ("veomni_qat_fake_quant_act", ())],
)
def test_activation_helpers_pass_through_while_disabled(monkeypatch, helper, args):
    monkeypatch.setattr(modeling_gpu, "veomni_qat_implementation", OpsConfigSlot("qat_implementation", default="none"))

    x = torch.randn(2, 3, 512, dtype=torch.bfloat16)

    assert getattr(modeling_gpu, helper)(x, *args) is x


@pytest.mark.parametrize(
    "helper, args, shape",
    [("veomni_qat_fake_quant_kv", (64,), (2, 0, 512)), ("veomni_qat_fake_quant_act", (), (2, 0, 128))],
)
def test_activation_helpers_pass_empty_entries_through(monkeypatch, helper, args, shape):
    """A compressor produces a zero-length KV until its first window closes."""
    monkeypatch.setattr(
        modeling_gpu, "veomni_qat_implementation", OpsConfigSlot("qat_implementation", default="fp8_blockwise")
    )

    x = torch.zeros(shape, dtype=torch.bfloat16)

    assert getattr(modeling_gpu, helper)(x, *args) is x


def test_kv_helper_quantizes_the_nope_channels_and_spares_the_rope_tail(monkeypatch):
    """The split is the whole point of this recipe: RoPE channels stay BF16."""
    _require_tilelang_cuda()
    from veomni.ops.qat import fp8_fake_quant_act_prefix

    monkeypatch.setattr(
        modeling_gpu, "veomni_qat_implementation", OpsConfigSlot("qat_implementation", default="fp8_blockwise")
    )

    torch.manual_seed(0)
    head_dim, rope_features = 512, 64
    nope = head_dim - rope_features
    x = torch.randn(2, 3, head_dim, device=DEVICE, dtype=torch.bfloat16)

    actual = modeling_gpu.veomni_qat_fake_quant_kv(x, rope_features)

    assert torch.equal(actual[..., nope:], x[..., nope:]), "the RoPE tail was quantized"
    assert not torch.equal(actual[..., :nope], x[..., :nope]), "the NoPE channels were left alone"
    # 64-wide blocks, not 128: a 448-channel NoPE half is not a multiple of 128.
    assert torch.equal(actual, fp8_fake_quant_act_prefix(x, nope, block_size=64))


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_expert_weight_recipe_follows_the_checkpoint_dtype(monkeypatch, expert_dtype):
    """V4-Flash ships FP4 experts; the geometry differs, not just the width."""
    _require_tilelang_cuda()
    from veomni.ops.qat import fp4_fake_quant_weight, fp8_fake_quant_stacked_weight

    monkeypatch.setattr(
        modeling_gpu, "veomni_qat_implementation", OpsConfigSlot("qat_implementation", default="fp8_blockwise")
    )

    torch.manual_seed(0)
    weight = torch.randn(2, 256, 384, device=DEVICE, dtype=torch.bfloat16)

    actual = modeling_gpu.veomni_qat_fake_quant_expert_weight(weight, expert_dtype)

    expected = fp4_fake_quant_weight(weight) if expert_dtype == "fp4" else fp8_fake_quant_stacked_weight(weight)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, weight)


def test_expert_weight_recipe_is_a_passthrough_while_disabled(monkeypatch):
    monkeypatch.setattr(modeling_gpu, "veomni_qat_implementation", OpsConfigSlot("qat_implementation", default="none"))

    weight = torch.randn(2, 256, 384, dtype=torch.bfloat16)

    assert modeling_gpu.veomni_qat_fake_quant_expert_weight(weight, "fp4") is weight


def test_expert_weights_reach_the_fused_kernel_quantized(monkeypatch):
    """Pin what the fused kernel is handed, since QAT is applied outside it."""
    _require_tilelang_cuda()
    from veomni.ops.qat import fp4_fake_quant_weight, fp8_fake_quant_act

    monkeypatch.setattr(
        modeling_gpu, "veomni_qat_implementation", OpsConfigSlot("qat_implementation", default="fp8_blockwise")
    )

    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=256,
        intermediate_size=128,
        hidden_act="silu",
        swiglu_limit=7.0,
        expert_dtype="fp4",
    )
    torch.manual_seed(0)
    experts = modeling_gpu.DeepseekV4Experts(config).to(device=DEVICE, dtype=torch.bfloat16)
    # `__init__` leaves the expert weights uninitialized, and a NaN would make
    # every `torch.equal` below false regardless of the quantization.
    with torch.no_grad():
        experts.gate_up_proj.normal_(std=0.05)
        experts.down_proj.normal_(std=0.05)
    hidden_states = torch.randn(4, config.hidden_size, device=DEVICE, dtype=torch.bfloat16)

    captured = {}
    kernel_output = torch.randn(hidden_states.shape, device=DEVICE, dtype=torch.bfloat16)

    def fake_fused_moe_forward(**kwargs):
        captured.update(kwargs)
        return kernel_output

    class _FusedSlot:
        use_non_eager_impl = True

    monkeypatch.setattr(modeling_gpu, "veomni_moe_experts_forward", _FusedSlot())
    monkeypatch.setattr(modeling_gpu, "fused_moe_forward", fake_fused_moe_forward)

    top_k_index = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], device=DEVICE)
    top_k_weights = torch.full((4, 2), 0.5, device=DEVICE, dtype=torch.bfloat16)
    output = experts(hidden_states, top_k_index, top_k_weights)

    assert torch.equal(captured["fc1_1_2_weight"], fp4_fake_quant_weight(experts.gate_up_proj))
    assert torch.equal(captured["fc2_weight"], fp4_fake_quant_weight(experts.down_proj))
    assert torch.equal(captured["hidden_states"], fp8_fake_quant_act(hidden_states, block_size=128))
    # The kernel's own output is returned untouched: inference combines the
    # expert results in BF16, so nothing rounds it on the way out.
    assert torch.equal(output, kernel_output)


def test_expert_weight_gradients_survive_the_straight_through_estimator():
    """The reason quantizing outside the fused kernel is safe at all.

    The fused MoE autograd function saves whatever weight tensor it was handed
    and writes gradients shaped like it, so the gradient has to travel back
    through the fake-quant node to land on the parameter.
    """
    _require_tilelang_cuda()
    from veomni.ops.kernels.moe import apply_veomni_fused_moe_patch

    apply_veomni_fused_moe_patch("triton")

    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=256,
        intermediate_size=128,
        hidden_act="silu",
        swiglu_limit=7.0,
        expert_dtype="fp4",
    )
    torch.manual_seed(0)
    experts = modeling_gpu.DeepseekV4Experts(config).to(device=DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        experts.gate_up_proj.normal_(std=0.05)
        experts.down_proj.normal_(std=0.05)
    hidden_states = torch.randn(8, config.hidden_size, device=DEVICE, dtype=torch.bfloat16)
    top_k_index = torch.tensor([[0, 1]] * 8, device=DEVICE)
    top_k_weights = torch.full((8, 2), 0.5, device=DEVICE, dtype=torch.bfloat16)

    class _FusedSlot:
        use_non_eager_impl = True

    def run(qat):
        for param in experts.parameters():
            param.grad = None
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(modeling_gpu, "veomni_qat_implementation", OpsConfigSlot("qat_implementation", default=qat))
            # QAT is wired on the fused path only.
            patch.setattr(modeling_gpu, "veomni_moe_experts_forward", _FusedSlot())
            out = experts(hidden_states, top_k_index, top_k_weights)
        out.float().square().mean().backward()
        return out.detach().clone(), {n: p.grad.detach().clone() for n, p in experts.named_parameters()}

    plain, plain_grads = run("none")
    quantized, quant_grads = run("fp8_blockwise")

    assert not torch.equal(plain, quantized), "fp8_blockwise did not change the expert forward"
    for name in ("gate_up_proj", "down_proj"):
        grad = quant_grads[name]
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, (
            f"no usable gradient reached {name} through the fake-quant node"
        )
        assert grad.shape == getattr(experts, name).shape
        assert not torch.equal(grad, plain_grads[name])


def test_act_helper_quantizes_the_whole_last_dimension(monkeypatch):
    """Shared by the indexer entries and the routed-expert activations."""
    _require_tilelang_cuda()
    from veomni.ops.qat import fp8_fake_quant_act

    monkeypatch.setattr(
        modeling_gpu, "veomni_qat_implementation", OpsConfigSlot("qat_implementation", default="fp8_blockwise")
    )

    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 128, device=DEVICE, dtype=torch.bfloat16)

    actual = modeling_gpu.veomni_qat_fake_quant_act(x)

    assert torch.equal(actual, fp8_fake_quant_act(x, block_size=128))
    # Unlike the KV recipe, the trailing RoPE channels are rounded too.
    assert not torch.equal(actual[..., -64:], x[..., -64:])
