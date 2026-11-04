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

"""Which expert activation each fused MoE backend can actually compute.

Every MoE model in the repo was SwiGLU, so the backends hardcoded ``aten.silu``
and nothing noticed. DiffusionGemma is the first with ``gelu_pytorch_tanh``
experts: it trained on SiLU while being served through GELU, across all 30 MoE
layers. ``fused_moe_forward`` now carries an ``activation`` argument, and the
Triton path honours it -- both non-EP (``ops/kernels/moe/group_gemm.py``) and in
the EP autograd functions it dispatches to (``distributed/moe/moe_layer.py``).
Quack, NPU and MLU take the argument only to reject anything but SiLU.

Two things are pinned here: the build-time gate admits exactly what the chosen
backend implements and refuses the rest, and the numerical tests below show the
Triton kernel computes the activation it was asked for rather than SiLU.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import check_model_build_prerequisites
from veomni.ops.kernels.moe_experts.standard.activation_gate import (
    _KERNEL_ACTIVATIONS,
    moe_expert_activation,
    refuse_fused_moe_on_unsupported_activation,
)
from veomni.ops.kernels.moe_experts.standard.triton import group_gemm_fused_moe_forward
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type
from veomni.utils.import_utils import is_fused_moe_available


_GATE_MODULE = "veomni.ops.kernels.moe_experts.standard.activation_gate"


def _all_eager_ops_config(**overrides) -> OpsImplementationConfig:
    """All-eager baseline so a test exercises only the slot it names."""
    fields = {
        "attn_implementation": "eager",
        "moe_implementation": "eager",
        "cross_entropy_loss_implementation": "eager",
        "rms_norm_implementation": "eager",
        "swiglu_mlp_implementation": "eager",
        "rotary_pos_emb_implementation": "eager",
        "load_balancing_loss_implementation": "eager",
        "rms_norm_gated_implementation": "eager",
        "causal_conv1d_implementation": "eager",
        "chunk_gated_delta_rule_implementation": "eager",
    }
    fields.update(overrides)
    return OpsImplementationConfig(**fields)


def test_the_text_tower_names_the_expert_activation():
    """Multimodal MoEs keep their experts in the text tower, so it wins."""
    config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_activation="gelu_pytorch_tanh"),
        vision_config=SimpleNamespace(hidden_act="gelu"),
    )
    assert moe_expert_activation(config) == ("gelu_pytorch_tanh", "text_config.hidden_activation")


def test_an_omni_thinker_text_tower_names_the_expert_activation():
    """Qwen3-Omni nests the MoE text tower under ``thinker_config.text_config``."""
    config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_act="silu"),
            vision_config=SimpleNamespace(hidden_act="gelu"),
            audio_config=SimpleNamespace(hidden_act="gelu"),
        ),
        talker_config=SimpleNamespace(text_config=SimpleNamespace(hidden_act="silu")),
    )
    assert moe_expert_activation(config) == ("silu", "thinker_config.text_config.hidden_act")


def test_the_shipped_omni_toy_config_resolves_to_silu():
    """Regression for the nesting the first draft of this guard missed."""
    import json
    from pathlib import Path

    raw = json.loads(Path("tests/toy_config/qwen3omni_toy/config.json").read_text())

    def to_ns(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in value.items()})
        return value

    activation, field = moe_expert_activation(to_ns(raw))
    assert activation == "silu"
    assert field == "thinker_config.text_config.hidden_act"


def test_a_flat_config_is_read_at_the_top_level():
    assert moe_expert_activation(SimpleNamespace(hidden_act="silu")) == ("silu", "hidden_act")


def test_a_vision_tower_alone_never_names_the_expert_activation():
    """An image tower has its own activation and no experts."""
    config = SimpleNamespace(vision_config=SimpleNamespace(hidden_act="gelu"))
    assert moe_expert_activation(config) is None


def test_the_real_diffusion_gemma_config_resolves_to_gelu():
    """The case this guard exists for, read off the shipped config class."""
    DiffusionGemmaConfig = pytest.importorskip(
        "veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma"
    ).DiffusionGemmaConfig

    activation, field = moe_expert_activation(DiffusionGemmaConfig())
    assert activation == "gelu_pytorch_tanh"
    assert field.endswith("hidden_activation")


@pytest.mark.parametrize("kernel", ["triton", "quack", "npu"])
@pytest.mark.parametrize("activation", ["silu", "swish"])
def test_swiglu_experts_are_what_every_backend_computes(kernel, activation):
    refuse_fused_moe_on_unsupported_activation(SimpleNamespace(hidden_act=activation), kernel)


@pytest.mark.parametrize("activation", ["gelu_pytorch_tanh", "gelu_tanh", "gelu", "gelu_new"])
def test_triton_serves_the_gelu_family(activation):
    """The Triton path resolves the activation at runtime, so GELU is in scope."""
    config = SimpleNamespace(text_config=SimpleNamespace(hidden_activation=activation))
    refuse_fused_moe_on_unsupported_activation(config, "triton")


@pytest.mark.parametrize("activation", ["gelu_pytorch_tanh", "gelu", "gelu_new"])
@pytest.mark.parametrize("kernel", ["quack", "npu"])
def test_the_silu_only_backends_refuse_the_gelu_family(activation, kernel):
    """Quack and NPU take the ``activation`` argument only to reject it."""
    config = SimpleNamespace(text_config=SimpleNamespace(hidden_activation=activation))
    with pytest.raises(ValueError, match="cannot compute the expert activation"):
        refuse_fused_moe_on_unsupported_activation(config, kernel)


@pytest.mark.parametrize("kernel", ["triton", "quack", "npu"])
def test_no_backend_admits_an_activation_none_of_them_implement(kernel):
    config = SimpleNamespace(text_config=SimpleNamespace(hidden_activation="quick_gelu"))
    with pytest.raises(ValueError, match="cannot compute the expert activation"):
        refuse_fused_moe_on_unsupported_activation(config, kernel)


def test_an_unrecognised_backend_is_held_to_swiglu():
    """A third-party kernel that delegates here has unknown activation support.

    ``apply_veomni_fused_moe_patch`` is typed ``str`` so out-of-tree backends can
    intercept their own name. One that instead falls through to this gate must
    not be waved past on the Triton entry's behalf.
    """
    refuse_fused_moe_on_unsupported_activation(SimpleNamespace(hidden_act="silu"), "seed-kernels")
    config = SimpleNamespace(hidden_act="gelu_pytorch_tanh")
    with pytest.raises(ValueError, match="cannot compute the expert activation"):
        refuse_fused_moe_on_unsupported_activation(config, "seed-kernels")


def test_the_triton_gate_matches_the_kernel_alias_table():
    """The gate duplicates the kernel's alias table; pin them together.

    ``activation_gate`` cannot import the table -- it lives behind a
    module-level ``import triton`` and the gate runs on CPU-only config paths --
    so the two drift silently. A name added to the kernel but not the gate is
    refused at build time even though it works; a name added to the gate but not
    the kernel turns a build-time refusal into a first-forward crash.
    """
    _skip_without_triton_moe()
    from veomni.ops.kernels.moe_experts.shared.activation import _ACTIVATION_ALIASES

    assert _KERNEL_ACTIVATIONS["triton"] == frozenset(_ACTIVATION_ALIASES)


def test_the_refusal_names_the_activation_and_a_way_out():
    """A guard nobody can act on gets worked around instead of fixed."""
    config = SimpleNamespace(text_config=SimpleNamespace(hidden_activation="gelu_pytorch_tanh"))
    with pytest.raises(ValueError) as excinfo:
        refuse_fused_moe_on_unsupported_activation(config, "quack")
    message = str(excinfo.value)
    assert "gelu_pytorch_tanh" in message
    assert "text_config.hidden_activation" in message
    # Name what the backend does implement, so the reader can see 'fused_triton'
    # is the cheaper move than dropping to eager.
    assert "'silu'" in message
    assert "moe_implementation=eager" in message
    # The escape hatch is not available at the scale this matters, and saying so
    # here is the point -- the eager expert loop cannot resolve global expert ids
    # under EP, so a 26B run has to fix the kernel rather than switch backends.
    assert "ep_size=1" in message


def test_an_unreadable_config_warns_rather_than_passing_silently():
    """A config naming neither field cannot be ruled on, so it must not raise.

    Refusing here would break every model whose activation lives somewhere this
    does not look. Warning is the compromise, and it has to be audible.
    """
    with patch(f"{_GATE_MODULE}.logger") as mock_logger:
        refuse_fused_moe_on_unsupported_activation(SimpleNamespace(), "triton")
    mock_logger.warning_rank0.assert_called_once()
    assert "could not be checked" in mock_logger.warning_rank0.call_args.args[0]


def test_a_missing_config_warns():
    """Without a config there is nothing to rule on; the SiLU-only backends still refuse at forward."""
    with patch(f"{_GATE_MODULE}.logger") as mock_logger:
        refuse_fused_moe_on_unsupported_activation(None, "triton")
    mock_logger.warning_rank0.assert_called_once()


def test_build_time_prerequisite_refuses_unsupported_activation():
    """The guard fires in ``check_model_build_prerequisites``, before ``load_model``."""
    DiffusionGemmaConfig = pytest.importorskip(
        "veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma"
    ).DiffusionGemmaConfig

    config = DiffusionGemmaConfig(
        text_config={"hidden_activation": "quick_gelu"},
        ops_implementation=_all_eager_ops_config(moe_implementation="fused_triton"),
    )
    with pytest.raises(ValueError, match="cannot compute the expert activation"):
        check_model_build_prerequisites(config)


def test_eager_moe_keeps_serving_a_gelu_activation():
    """The eager expert loop reads ``config.hidden_activation`` and is correct."""
    DiffusionGemmaConfig = pytest.importorskip(
        "veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma"
    ).DiffusionGemmaConfig

    config = DiffusionGemmaConfig(ops_implementation=_all_eager_ops_config(moe_implementation="eager"))
    check_model_build_prerequisites(config)


def test_diffusion_gemma_builds_on_fused_triton():
    """The 26B recipe, which is ``fused_triton`` + ``gelu_pytorch_tanh``.

    An earlier revision of this gate refused every non-SwiGLU activation on
    every backend, which was right while the kernel hardcoded SiLU and wrong the
    moment it learned the GELU family -- it blocked the only configuration the
    fix was for. Pin that the production pairing builds.
    """
    DiffusionGemmaConfig = pytest.importorskip(
        "veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma"
    ).DiffusionGemmaConfig

    config = DiffusionGemmaConfig(ops_implementation=_all_eager_ops_config(moe_implementation="fused_triton"))
    check_model_build_prerequisites(config)


def _eager_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    activation,
) -> torch.Tensor:
    """Reference expert loop, in the shape ``DiffusionGemmaExperts`` uses."""
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        gate = F.linear(x, fc1_1_weight[idx])
        up = F.linear(x, fc1_2_weight[idx])
        y = F.linear(activation(gate) * up, fc2_weight[idx])
        y = y * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, y.to(output.dtype))

    return output


def _moe_inputs(num_tokens, num_experts, hidden_dim, ffn_dim, topk, seed):
    torch.manual_seed(seed)
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    hidden_states = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), topk, dim=-1)
    return (
        hidden_states,
        routing_weights.to(dtype),
        selected_experts,
        0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype),
        0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype),
        0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype),
    )


def _skip_without_triton_moe():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for the fused MoE kernel.")
    if not is_fused_moe_available():
        pytest.skip("Triton fused MoE is not available in this environment.")


# Relative L2, not ``atol``. An absolute tolerance cannot separate these two
# activations at any input scale: measured on the shapes below, a kernel call
# sits 0.45% from a matching reference (bf16 with an MoE topk sum, constraint
# 33) and 10.5% from a mismatched one, but in absolute terms that is 4.9e-4
# against 9.5e-3 -- both under the 3.2e-2 an earlier draft of this file
# inherited from `test_fused_moe_split_vs_merged`, so both arms passed and the
# file reported the kernel as correct whichever activation it ran. 2% is above
# the noise and five times under a wrong activation.
_ACTIVATION_REL_L2 = 2e-2

# The `0.1 *` on the inputs is load-bearing, not a scale nobody thought about.
# GELU and SiLU both approach the identity away from the origin, so at unit-scale
# weights the pre-activations reach |gate| ~ 22 and the two agree to 0.55% --
# indistinguishable from bf16 noise, and the test would pass whichever the kernel
# computed. `test_the_two_activations_are_separable_here` fails closed on that.
_MOE_SHAPE = dict(num_tokens=256, num_experts=32, hidden_dim=512, ffn_dim=256, topk=4)


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm()).item()


def _fused_and_eager(activation: str):
    """Run the kernel and the reference loop on one activation *name*.

    The kernel is asked for *activation* explicitly. Letting it default was what
    made the GELU arm of this file meaningless: it compared a SiLU kernel call
    against a GELU reference and recorded the gap as an expected failure.
    """
    hidden_states, routing_weights, selected_experts, fc1_1, fc1_2, fc2 = _moe_inputs(seed=0, **_MOE_SHAPE)
    num_experts = _MOE_SHAPE["num_experts"]
    fused = group_gemm_fused_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_weight=fc1_1,
        fc1_2_weight=fc1_2,
        fc2_weight=fc2,
        activation=activation,
    )
    eager = _eager_moe_forward(
        num_experts, routing_weights, selected_experts, hidden_states, fc1_1, fc1_2, fc2, ACT2FN[activation]
    )
    return fused, eager


def test_the_two_activations_are_separable_here():
    """Guard the discriminating power of the two tests below.

    If the inputs ever move into the regime where GELU and SiLU coincide, both
    arms pass regardless of what the kernel computes, and this file would go on
    reporting a fixed kernel. Fail here instead.
    """
    _skip_without_triton_moe()

    _, eager_silu = _fused_and_eager("silu")
    _, eager_gelu = _fused_and_eager("gelu_pytorch_tanh")

    separation = _rel_l2(eager_silu, eager_gelu)
    assert separation > 5 * _ACTIVATION_REL_L2, (
        f"SiLU and gelu_pytorch_tanh differ by only {separation:.2%} on these inputs, which is "
        f"too close to the {_ACTIVATION_REL_L2:.0%} threshold for the tests below to mean "
        "anything. Lower the input scale in `_moe_inputs` back towards the origin."
    )


def test_fused_moe_computes_swiglu():
    """The case the kernel was written for, and the control for the next test."""
    _skip_without_triton_moe()

    fused, eager = _fused_and_eager("silu")

    assert _rel_l2(fused, eager) < _ACTIVATION_REL_L2


@pytest.mark.parametrize("activation", ["gelu_pytorch_tanh", "gelu"])
def test_fused_moe_computes_the_configured_gelu_activation(activation):
    """DiffusionGemma's experts, through the kernel its recipe selects.

    This was ``xfail(strict=True)`` while the kernel hardcoded SiLU; it now
    holds, and the SiLU control above plus
    ``test_the_two_activations_are_separable_here`` are what keep it honest.
    """
    _skip_without_triton_moe()

    fused, eager = _fused_and_eager(activation)

    assert _rel_l2(fused, eager) < _ACTIVATION_REL_L2


def test_fused_moe_does_not_fall_back_to_silu_on_a_gelu_request():
    """The kernel must be *wrong* against a SiLU reference when asked for GELU.

    The assertion above passes either if the kernel honours ``gelu_pytorch_tanh``
    or if GELU and SiLU happen to coincide on these inputs. This arm separates
    the two: a silent SiLU fallback would land close to the SiLU reference.
    """
    _skip_without_triton_moe()

    hidden_states, routing_weights, selected_experts, fc1_1, fc1_2, fc2 = _moe_inputs(seed=0, **_MOE_SHAPE)
    num_experts = _MOE_SHAPE["num_experts"]
    fused_gelu = group_gemm_fused_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_weight=fc1_1,
        fc1_2_weight=fc1_2,
        fc2_weight=fc2,
        activation="gelu_pytorch_tanh",
    )
    eager_silu = _eager_moe_forward(
        num_experts, routing_weights, selected_experts, hidden_states, fc1_1, fc1_2, fc2, F.silu
    )

    assert _rel_l2(fused_gelu, eager_silu) > _ACTIVATION_REL_L2
