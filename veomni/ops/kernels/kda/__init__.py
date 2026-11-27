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

"""chunk_kda kernel family (Kimi K3 KDA linear attention).

``eager`` is the pure-PyTorch reference vendored from MindSpeed-MM's
``chunk_kda_naive`` (see ``.kimi_k3.eager``): portable, autograd-friendly,
and the numerical baseline for the fused paths. ``fla`` binds
``fla.ops.kda.chunk_kda`` on CUDA after asserting the installed
flash-linear-attention exposes the KDA gating contract (beta-sigmoid,
in-kernel gate, safe-gate bounds). ``ascendc`` is the fused AscendC
forward+backward path from flash-linear-attention-npu (MindSpeed-MM
kimi_k3 wrapper) and is the production path on NPU.

The KDA-adjacent eager primitives (``ShortConvolution`` /
``FusedRMSNormGated``, vendored in ``.kimi_k3.eager``) are deliberately
not registered as ops: the modeling constructs them directly and no
fused-row selection applies to them (unlike the GDN family's registered
``causal_conv1d`` / ``rms_norm_gated`` rows).
"""

import inspect

from ...platform import GpuKernelRequirement, NpuKernelRequirement
from ...registry import register_op


# KDA call knobs the modeling source always passes; a flash-linear-attention
# build missing any of them cannot serve the Kimi K3 training contract.
# The gating trio must be real keyword parameters: a signature that merely
# takes ``**kwargs`` accepts them silently and drops them, training different
# math (raw beta, unbounded gate) -- exactly the silent degradation the
# review forbids. ``A_log``/``dt_bias`` are consumed through ``**kwargs`` by
# design in supported builds, so they only gate static signatures.
_FLA_STRICT_PARAMS = (
    "use_beta_sigmoid_in_kernel",
    "safe_gate",
    "lower_bound",
)
_FLA_LENIENT_PARAMS = (
    "use_qk_l2norm_in_kernel",
    "use_gate_in_kernel",
    "A_log",
    "dt_bias",
    "cu_seqlens",
    "disable_recompute",
)


def _eager(q, k, v, g, beta, **kwargs):
    from .kimi_k3.eager import chunk_kda_naive

    # The reference uses fla's historical name for the transposed state
    # layout; accept the renamed form too so callers can use either.
    if "state_v_first" in kwargs:
        if kwargs.get("transpose_state_layout"):
            raise ValueError("Pass only one of `state_v_first` and `transpose_state_layout`.")
        kwargs["transpose_state_layout"] = kwargs.pop("state_v_first")
    return chunk_kda_naive(q=q, k=k, v=v, g=g, beta=beta, **kwargs)


def _fla(q, k, v, g, beta, **kwargs):
    import importlib.metadata

    from fla.ops.kda import chunk_kda as fn

    params = inspect.signature(fn).parameters
    has_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    # The K3 gating contract must be spelled as real keyword arguments: a
    # ``**kwargs`` catch-all accepts them and silently drops them (raw beta,
    # unbounded gate), so var-keyword support does not count for these.
    missing = [name for name in _FLA_STRICT_PARAMS if name not in params]
    if "transpose_state_layout" not in params and "state_v_first" not in params:
        missing.append("transpose_state_layout/state_v_first")
    if not has_var_keyword:
        missing.extend(name for name in _FLA_LENIENT_PARAMS if name not in params)
    if missing:
        try:
            version = importlib.metadata.version("fla")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        raise NotImplementedError(
            f"The installed flash-linear-attention ({version}) chunk_kda cannot serve the "
            f"Kimi K3 gating contract (missing {sorted(missing)}); the 'fla' row refuses "
            "rather than silently degrading. Use the 'eager' (naive reference) "
            "implementation instead."
        )
    # NPU-only D2H hoist hint consumed by the ascendc path; fla derives host
    # metadata itself.
    kwargs.pop("cu_seqlens_cpu", None)
    # fla renamed transpose_state_layout to state_v_first; map the modeling
    # side's historical name onto whichever the installed build accepts.
    if "transpose_state_layout" not in params and "state_v_first" in params:
        kwargs["state_v_first"] = kwargs.pop("transpose_state_layout", False)
    return fn(q=q, k=k, v=v, g=g, beta=beta, **kwargs)


def _ascendc(q, k, v, g, beta, **kwargs):
    from .kimi_k3.ascendc import chunk_kda_ascendc

    return chunk_kda_ascendc(q, k, v, g, beta, **kwargs)


register_op(
    "chunk_kda",
    "kimi_k3",
    "eager",
    wrapper=_eager,
    description="Kimi K3 KDA chunked linear attention, pure-PyTorch reference (MindSpeed-MM chunk_kda_naive)",
)

register_op(
    "chunk_kda",
    "kimi_k3",
    "fla",
    wrapper=_fla,
    description="flash-linear-attention chunked KDA with beta-sigmoid gating (version-asserted)",
    requirement=GpuKernelRequirement(),
    requires=("fla",),
)

register_op(
    "chunk_kda",
    "kimi_k3",
    "ascendc",
    wrapper=_ascendc,
    description="Kimi K3 KDA chunked linear attention fused AscendC kernel (NPU)",
    requirement=NpuKernelRequirement(),
    requires=("fla_npu", "triton_ascend_kernels"),
    default=True,
)
