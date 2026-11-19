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

"""Refuse fused MoE when the backend cannot compute the expert activation.

``fused_moe_forward`` carries an ``activation`` argument, but the backends
honour it unevenly. The Triton group-gemm resolves it at runtime through
``_kernels/kernel/triton_utils/activation.py``, so it serves the GELU family as
well as SiLU, both on the non-EP path (``ops/kernels/moe/group_gemm.py``) and in
the EP autograd functions it dispatches to (``distributed/moe/moe_layer.py``).
Quack (``ops/kernels/moe/quack_gemm.py``), NPU
(``ops/kernels/moe/npu_group_gemm.py``) and MLU accept the argument only to
reject anything but SiLU, and they do so on the first forward.

This gate moves that refusal to build time -- ``_bind_veomni_ops`` runs it
before the fused MoE pointer or any OpSlot is bound, so a wrong recipe never
reaches a forward. ``apply_veomni_fused_moe_patch`` keeps its signature because
third-party backends intercept it by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .....utils import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.get_logger(__name__)

_SWIGLU_ACTIVATIONS = frozenset({"silu", "swish"})

# What each backend actually implements. Keep the Triton entry in step with
# ``_ACTIVATION_ALIASES`` in ``..shared.activation``:
# that table is the authority, but importing it here would pull Triton into
# every config path, including CPU-only ones.
# ``test_the_triton_gate_matches_the_kernel_alias_table`` fails closed on drift.
#
# An unrecognised kernel name gets the SwiGLU set, because an unknown kernel's
# activation support cannot be assumed.
_KERNEL_ACTIVATIONS: dict[str, frozenset[str]] = {
    "triton": frozenset({"silu", "swish", "gelu", "gelu_new", "gelu_pytorch_tanh", "gelu_tanh"}),
    "quack": _SWIGLU_ACTIVATIONS,
    "npu": _SWIGLU_ACTIVATIONS,
}


def moe_expert_activation(config: "PretrainedConfig") -> Optional[tuple[str, str]]:
    """Name the activation the MoE experts will use, and the field it came from.

    Experts always live in a text tower, never in a vision/audio sub-config, so
    the search order is: top-level ``text_config`` (VL / DiffusionGemma /
    Qwen3.5-MoE), then ``thinker_config.text_config`` (Omni), then the thinker
    itself, then the top-level config (dense MoE like Qwen3-MoE / DeepSeek-V3).
    Returns ``None`` when none of those name an activation field.
    """
    sources: list[tuple[str, object]] = []
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        sources.append(("text_config.", text_config))
    thinker_config = getattr(config, "thinker_config", None)
    if thinker_config is not None:
        thinker_text = getattr(thinker_config, "text_config", None)
        if thinker_text is not None:
            sources.append(("thinker_config.text_config.", thinker_text))
        sources.append(("thinker_config.", thinker_config))
    sources.append(("", config))
    for prefix, source in sources:
        for field in ("hidden_activation", "hidden_act"):
            value = getattr(source, field, None)
            if isinstance(value, str) and value:
                return value, f"{prefix}{field}"
    return None


def refuse_fused_moe_on_unsupported_activation(config: Optional["PretrainedConfig"], kernel: str) -> None:
    """Fail closed when *kernel* cannot compute the activation *config* asks for.

    Without a config there is nothing to check against; the SiLU-only backends
    still refuse other activations on their first forward.
    """
    kernel = kernel.removeprefix("fused_")
    supported = _KERNEL_ACTIVATIONS.get(kernel, _SWIGLU_ACTIVATIONS)
    resolved = moe_expert_activation(config) if config is not None else None
    if resolved is None:
        logger.warning_rank0(
            f"fused MoE kernel {kernel!r} implements {sorted(supported)} as the expert "
            "activation, and this config names neither `hidden_activation` nor `hidden_act`, so "
            "that could not be checked. Confirm the experts use one of those."
        )
        return
    activation, field = resolved
    if activation in supported:
        return
    raise ValueError(
        f"moe_implementation='fused_{kernel}' cannot compute the expert activation "
        f"{field}={activation!r}; it implements {sorted(supported)}. The experts would run a "
        f"different function from the rest of the model -- and from inference. Either set "
        f"`model.ops_implementation.moe_implementation=eager`, which runs the config's activation "
        f"but only works at ep_size=1 (the eager expert loop cannot resolve global expert ids "
        f"under expert parallelism), or teach the kernel this activation."
    )
