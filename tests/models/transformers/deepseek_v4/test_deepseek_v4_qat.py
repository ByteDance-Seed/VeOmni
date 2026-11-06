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
which stay in the model dtype -- used to be a global ``qat_implementation``
recipe that every call site passed around. It is now expressed the same way as
every other kernel choice: each family has an op row, and the model constructs
a handle.

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
# that way, listed as (helper, argument) pairs.
#
# The counts carry weight. Each compressor finalizes its output in two separate
# branches -- one for packed sequences, one for the windowed/cached path -- and
# rounding only one of them would under-quantize a real training configuration
# while leaving every shape and loss curve plausible. The main attention's Q
# appears nowhere on purpose: it is never stored, so inference keeps it in the
# model dtype all the way into attention.
_EXPECTED_STORAGE_ROUND = {
    "DeepseekV4Attention.forward": [("round_stored_kv", "kv")],
    "DeepseekV4Indexer.forward": [
        ("round_indexer_act", "compressed"),
        ("round_indexer_act", "q"),
    ],
    "DeepseekV4HCACompressor.forward": [
        ("round_stored_kv", "compressed"),
        ("round_stored_kv", "compressed"),
    ],
    "DeepseekV4CSACompressor.forward": [
        ("round_stored_kv", "compressed"),
        ("round_stored_kv", "compressed"),
    ],
    "DeepseekV4MLP.forward": [],
    # The routed experts hand their raw weights to the family row, which does the
    # rounding; there is no model-side call site left to count.
    "DeepseekV4Experts.forward": [],
}


def _method_ast(module, qualname):
    """Parse the generated source of one patched method."""
    cls_name, method_name = qualname.split(".")
    source = inspect.getsource(getattr(getattr(module, cls_name), method_name))
    return ast.parse(textwrap.dedent(source))


def _storage_round_calls(module, qualname):
    """List the ``(helper, first argument)`` pairs of storage-rounding calls."""
    calls = []
    for node in ast.walk(_method_ast(module, qualname)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id in ("round_stored_kv", "round_indexer_act"):
            calls.append((node.func.id, ast.unparse(node.args[0])))
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
        if isinstance(func, ast.Attribute) and func.attr.startswith("veomni_"):
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
    """An unset ``kv_round_implementation`` must leave every tensor in the model dtype."""
    from tests.models.tiny_configs import tiny_deepseek_v4_config

    config = tiny_deepseek_v4_config()
    assert config.ops_config.kv_round_implementation is None
    assert modeling._storage_round_op(config, "fp8") is None


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_storage_round_op_follows_the_instance_selection(modeling):
    """Selecting a row must reach the modules that round stored tensors."""
    from tests.models.tiny_configs import tiny_deepseek_v4_config
    from veomni.utils.device import IS_NPU_AVAILABLE

    config = tiny_deepseek_v4_config()
    config._set_ops_implementation({"kv_round_implementation": "eager"})
    handle = modeling._storage_round_op(config, "fp8")
    assert handle is not None
    assert (handle.op, handle.variant, handle.impl) == ("kv_round", "fp8", "eager")
    if IS_NPU_AVAILABLE:
        # No CUDA row here; the NPU variant still builds the same handle shape.
        assert handle.op == "kv_round"


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_stored_kv_rounds_only_the_nope_prefix(modeling):
    """Window KV stores 64-wide FP8 groups up to the RoPE boundary, not past it."""
    import torch

    # DeepSeek-V4 stores 64-wide FP8 groups over the NoPE channels and keeps the
    # trailing RoPE channels (64 here) in the model dtype.
    rope_features = 64
    kv = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    rounded = modeling.round_stored_kv(kv, rope_features, _FakeOp("kv_round", "fp8", "eager"))

    assert torch.equal(rounded[..., -rope_features:], kv[..., -rope_features:])
    assert not torch.equal(rounded[..., :-rope_features], kv[..., :-rope_features])
    assert torch.isfinite(rounded).all()


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_indexer_act_rounds_whole_128_wide_heads(modeling):
    """The indexer rounds whole heads in 128-wide groups, unlike the window KV."""
    import torch

    x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    rounded = modeling.round_indexer_act(x, _FakeOp("kv_round", "fp8", "eager"))

    assert rounded.shape == x.shape
    assert not torch.equal(rounded, x)
    assert torch.isfinite(rounded).all()


@pytest.mark.parametrize("modeling", [modeling_gpu, modeling_npu], ids=["gpu", "npu"])
def test_storage_rounding_is_a_no_op_when_off(modeling):
    """No handle means no rounding, so a plain run keeps every tensor untouched."""
    import torch

    kv = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    assert modeling.round_stored_kv(kv, 64, None) is kv
    assert modeling.round_indexer_act(kv, None) is kv


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
        from veomni.ops.kernels.kv_round.fp8.eager import fp8_roundtrip

        return fp8_roundtrip(x, **kwargs)
