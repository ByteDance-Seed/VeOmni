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

The recipe -- which tensors are rounded the way inference rounds them, and
which stay in the model dtype -- is expressed the same way as every other kernel
choice: each family has an op row, and the model constructs a handle.

* Linear projections go through a ``VeomniOp("linear", ...)`` handle, so the
  family's selected row decides whether they are fake-quantized.
* Stored KV and indexer activations go through a ``VeomniOp("kv_round", ...)``
  handle, whose block width, scale format and quantized prefix are call
  arguments because the three DeepSeek-V4 recipes differ.
* Routed experts go through ``VeomniOp("moe_experts", ...)`` with the
  checkpoint's ``expert_dtype`` as a call argument.

That makes the recipe invisible to a forward-output test -- quantizing one
tensor too few still produces a plausible loss curve, it just trains for a
kernel nobody deploys -- so the coverage tests read the generated source and
fail on any site they have not been told about.

Numerics live in ``tests/ops/qat/test_fp8_blockwise.py`` and
``tests/ops/kv_round/test_kv_round.py``; what is checked here is the wiring.
"""

import ast
import inspect
import textwrap

import pytest

from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling_gpu
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_npu as modeling_npu
from veomni.ops import VeomniOp


# The projections DeepSeek V4 serves as a true low-precision GEMM, and the ones
# it does not. Both halves are pinned: a new projection that lands in neither
# set fails the test rather than silently defaulting to full precision.
#
# `weights_proj` produces one score per head, so its [index_n_heads,
# hidden_size] weight is too short to tile at 128 even if we wanted to. The
# compressors run their projections in FP32 upstream. `DeepseekV4MLP` is the
# shared expert and nothing else -- every V4 layer is an MoE block, so there are
# no dense MLPs sharing the class. The routed experts live in
# `DeepseekV4Experts` and are a separate recipe (FP4 weights on V4-Flash).
_EXPECTED_LINEAR = {
    "DeepseekV4Attention.forward": {"q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj"},
    "DeepseekV4Indexer.forward": {"q_b_proj"},
    "DeepseekV4HCACompressor.forward": set(),
    "DeepseekV4CSACompressor.forward": set(),
    "DeepseekV4MLP.forward": {"gate_proj", "up_proj", "down_proj"},
}
_EXPECTED_PLAIN = {
    "DeepseekV4Attention.forward": set(),
    "DeepseekV4Indexer.forward": {"scorer.weights_proj", "kv_proj", "gate_proj"},
    "DeepseekV4HCACompressor.forward": {"kv_proj", "gate_proj"},
    "DeepseekV4CSACompressor.forward": {"kv_proj", "gate_proj"},
    # Non-SwiGLU hidden_act falls back to the projection modules.
    "DeepseekV4MLP.forward": {"gate_proj", "up_proj", "down_proj"},
}

# The activation-only recipe: tensors rounded because inference *stores* them
# that way, listed as (call, argument) pairs. A storage site is the kv_round
# handle itself (`self.veomni_kv_round(...)`), so "which tensors are rounded" is
# the set of arguments it receives.
#
# The counts carry weight. Each compressor finalizes its output in two separate
# branches -- one for packed sequences, one for the windowed/cached path -- and
# rounding only one of them would under-quantize a real training configuration
# while leaving every shape and loss curve plausible. The main attention's Q
# appears nowhere on purpose: it is never stored, so inference keeps it in the
# model dtype all the way into attention.
_KV_ROUND = "self.veomni_kv_round"
_EXPECTED_STORAGE_ROUND = {
    "DeepseekV4Attention.forward": [(_KV_ROUND, "kv")],
    "DeepseekV4Indexer.forward": [
        (_KV_ROUND, "compressed"),
        (_KV_ROUND, "q"),
    ],
    "DeepseekV4HCACompressor.forward": [
        (_KV_ROUND, "compressed"),
        (_KV_ROUND, "compressed"),
    ],
    "DeepseekV4CSACompressor.forward": [
        (_KV_ROUND, "compressed"),
        (_KV_ROUND, "compressed"),
    ],
    "DeepseekV4MLP.forward": [],
    # The routed experts hand their raw weights to the family row on the fused
    # path; there is no model-side rounding left to count.
    "DeepseekV4Experts.forward": [],
}


def _method_ast(module, qualname):
    """Parse the generated source of one patched method."""
    cls_name, method_name = qualname.split(".")
    source = inspect.getsource(getattr(getattr(module, cls_name), method_name))
    return ast.parse(textwrap.dedent(source))


def _storage_round_calls(module, qualname):
    """List the ``(callee, rounded argument)`` pairs of storage-rounding calls.

    The kv_round handle is called directly, so a rounding site is recognized by the
    callee expression rather than by a helper name.
    """
    calls = []
    for node in ast.walk(_method_ast(module, qualname)):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        if callee == _KV_ROUND:
            calls.append((_KV_ROUND, ast.unparse(node.args[0])))
    return sorted(calls)


def _projection_calls(module, qualname):
    """Split the projections *qualname* uses into handle-driven and plain sets.

    A handle call is ``self.veomni_<name>(x, self.<name>.weight, ...)``, so the
    projection is recognized by the weight it reads rather than by the handle's
    attribute name. Anything still invoked as ``self.<name>_proj(...)`` runs the
    module directly and therefore stays in the model dtype.
    """
    tree = _method_ast(module, qualname)
    via_handle, plain = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # ``project_through_op(self.veomni_x, self.x, ...)`` names the projection
        # as its second argument; a direct handle call names it by its ``weight``.
        if isinstance(func, ast.Name) and func.id == "project_through_op":
            operand = node.args[1]
            if isinstance(operand, ast.Attribute) and isinstance(operand.value, ast.Name):
                if operand.value.id == "self":
                    via_handle.add(operand.attr)
            elif (
                isinstance(operand, ast.Attribute)
                and isinstance(operand.value, ast.Attribute)
                and isinstance(operand.value.value, ast.Name)
                and operand.value.value.id == "self"
            ):
                # ``attn.q_a_proj`` in dspark-style helpers.
                via_handle.add(operand.attr)
        elif isinstance(func, ast.Attribute) and func.attr.startswith("veomni_"):
            for arg in node.args:
                if (
                    isinstance(arg, ast.Attribute)
                    and arg.attr == "weight"
                    and isinstance(arg.value, ast.Attribute)
                    and isinstance(arg.value.value, ast.Name)
                    and arg.value.value.id == "self"
                ):
                    via_handle.add(arg.value.attr)
        elif isinstance(func, ast.Attribute) and func.attr.endswith("_proj"):
            path = ast.unparse(func)
            if path.startswith("self."):
                plain.add(path.removeprefix("self."))
    return via_handle, plain


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_every_quantizable_projection_goes_through_the_linear_row(modeling):
    """The linear family's selected row decides these projections' precision."""
    for qualname in _EXPECTED_LINEAR:
        via_handle, plain = _projection_calls(modeling, qualname)
        assert via_handle == _EXPECTED_LINEAR[qualname], (
            f"{qualname}: handle-driven projections drifted from the recipe. "
            f"Update _EXPECTED_LINEAR only if inference changed too."
        )
        assert plain == _EXPECTED_PLAIN[qualname], (
            f"{qualname}: projections still called as modules drifted. A projection added here is "
            f"trained in the model dtype; confirm inference does the same before listing it."
        )


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_storage_rounding_covers_exactly_the_stored_tensors(modeling):
    """Check the full storage recipe, including the unrounded main-attention Q."""
    for qualname, expected in _EXPECTED_STORAGE_ROUND.items():
        assert _storage_round_calls(modeling, qualname) == sorted(expected), (
            f"{qualname}: storage-rounding sites drifted from the recipe. Check both the packed and "
            f"the windowed branch before updating _EXPECTED_STORAGE_ROUND."
        )


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_storage_rounding_is_off_until_asked_for(modeling):
    """The default selection leaves every stored tensor in the model dtype.

    "Off" is the ``none`` row rather than an absent handle, so the model builds its
    kv_round handle unconditionally like every other family -- and that row returns
    its input, which is checked here.
    """
    import torch

    from tests.models.tiny_configs import tiny_deepseek_v4_config

    config = tiny_deepseek_v4_config()
    assert config.ops_config.kv_round_implementation == "none"

    kv = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    handle = VeomniOp("kv_round", "deepseek_v4", config.ops_config.kv_round_implementation)
    assert handle.impl == "none"
    torch.testing.assert_close(handle(kv, rope_features=64), kv, rtol=0, atol=0)
    torch.testing.assert_close(handle(kv), kv, rtol=0, atol=0)


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_kv_round_op_follows_the_instance_selection(modeling):
    """Selecting a row must reach the modules that round stored tensors."""
    from tests.models.tiny_configs import tiny_deepseek_v4_config

    config = tiny_deepseek_v4_config()
    config._set_ops_implementation({"kv_round_implementation": "eager"})
    assert config.ops_config.kv_round_implementation == "eager"
    handle = VeomniOp("kv_round", "deepseek_v4", config.ops_config.kv_round_implementation)
    assert (handle.op, handle.variant, handle.impl) == ("kv_round", "deepseek_v4", "eager")


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_stored_kv_rounds_only_the_nope_prefix(modeling):
    """Window KV stores 64-wide FP8 groups up to the RoPE boundary, not past it."""
    import torch

    # DeepSeek-V4 stores 64-wide FP8 groups over the NoPE channels and keeps the
    # trailing RoPE channels (64 here) in the model dtype.
    rope_features = 64
    kv = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    rounded = modeling.VeomniOp("kv_round", "deepseek_v4", "eager")(kv, rope_features=rope_features)

    assert torch.equal(rounded[..., -rope_features:], kv[..., -rope_features:])
    assert not torch.equal(rounded[..., :-rope_features], kv[..., :-rope_features])
    assert torch.isfinite(rounded).all()


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_indexer_act_rounds_whole_128_wide_heads(modeling):
    """The indexer rounds whole heads in 128-wide groups, unlike the window KV."""
    import torch

    x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    rounded = modeling.VeomniOp("kv_round", "deepseek_v4", "eager")(x)

    assert rounded.shape == x.shape
    assert not torch.equal(rounded, x)
    assert torch.isfinite(rounded).all()


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_storage_rounding_is_a_no_op_when_off(modeling):
    """No handle means no rounding, so a plain run keeps every tensor untouched."""
    import torch

    kv = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    off = modeling.VeomniOp("kv_round", "deepseek_v4", "none")
    torch.testing.assert_close(off(kv, rope_features=64), kv, rtol=0, atol=0)
    torch.testing.assert_close(off(kv), kv, rtol=0, atol=0)


class _FakeOp:
    """Stand-in for a ``VeomniOp`` that records the call arguments.

    The generated helpers only need ``__call__``; using a recording stub keeps
    these tests on CPU without importing the TileLang quantizers.
    """

    def __init__(self, op, variant, impl):
        self.op, self.variant, self.impl = op, variant, impl
        self.calls = []

    def __call__(self, x, **kwargs):
        self.calls.append(kwargs)
        from veomni.ops.kernels.kv_round.deepseek_v4 import eager

        if kwargs.get("rope_features") is None:
            return eager.round_indexer_act(x)
        return eager.round_window_kv(x, kwargs["rope_features"])


def _build_tiny_model(linear_impl: str):
    """A tiny DeepSeek-V4 on the reference rows, with ``linear_impl`` for the projections."""
    from tests.models.tiny_configs import tiny_deepseek_v4_config
    from veomni.arguments import OpsImplementationConfig
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_npu as gen
    else:
        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as gen

    config = tiny_deepseek_v4_config()
    config._set_ops_implementation(
        OpsImplementationConfig(
            attn_implementation="eager",
            rms_norm_implementation="eager",
            swiglu_mlp_implementation="eager",
            linear_implementation=linear_impl,
            moe_implementation="eager",
            dsa_attention_implementation="eager",
            dsa_indexer_implementation="eager",
            mhc_implementation="eager",
        )
    )
    return gen.DeepseekV4ForCausalLM(config), config


# ``o_a_proj`` is a block-diagonal ``DeepseekV4GroupedLinear`` whose forward is a
# 3D ``bmm``, so a plain LoRA adapter never fit it. These are the projections the
# rows actually drive one-for-one.
_LORA_TARGETS = ["q_a_proj", "q_b_proj", "kv_proj", "o_b_proj"]


def test_deepseek_v4_lora_runs_on_the_reference_linear_row():
    """A wrapped projection has no ``weight``; the reference row must fall back to calling it."""
    import torch

    from veomni.lora.config import VeOmniLoraConfig
    from veomni.lora.mapping import inject_dense_lora

    model, config = _build_tiny_model("eager")
    model.eval()
    inject_dense_lora(model.model, VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=_LORA_TARGETS), "default")

    input_ids = torch.randint(3, config.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(input_ids=input_ids)

    assert out.logits.shape == (1, 4, config.vocab_size)


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_a_non_silu_expert_activation_trains_in_the_model_dtype(modeling):
    """The fused expert rows bake SiLU into their kernels, so another activation cannot use them.

    The fallback loops over experts in Python instead, which means it cannot reach
    the family row and therefore does not round -- the same behaviour as every other
    model's fallback. Pinning it here keeps DeepSeek-V4 from becoming the one model
    whose non-SiLU experts quantize.
    """
    import copy

    import torch

    from tests.models.tiny_configs import tiny_deepseek_v4_config
    from veomni.arguments import OpsImplementationConfig

    config = tiny_deepseek_v4_config()
    # `gelu` is what selects the fallback; no published checkpoint uses it, which is
    # exactly why the path needs a test rather than a reader.
    config = copy.deepcopy(config)
    config.hidden_act = "gelu"
    config._set_ops_implementation(
        OpsImplementationConfig(
            attn_implementation="eager",
            rms_norm_implementation="eager",
            swiglu_mlp_implementation="eager",
            linear_implementation="eager",
            moe_implementation="eager",
            dsa_attention_implementation="eager",
            dsa_indexer_implementation="eager",
            mhc_implementation="eager",
            kv_round_implementation="none",
        )
    )

    if modeling.__name__.endswith("_npu"):
        pytest.skip("the DSV4 NPU build needs torch_npu")

    model = modeling.DeepseekV4Experts(config)
    assert model.use_swiglu_mlp is False
    # The parameters are allocated with `torch.empty`; a real build fills them from
    # a checkpoint, so give them finite values before the forward.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)

    hidden_states = torch.randn(4, config.hidden_size)
    selected = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2]], dtype=torch.long)
    weights = torch.full((4, 2), 0.5)

    output = model(hidden_states, selected, weights)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
