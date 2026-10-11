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
"""FFPA for DiffusionGemma's wide global-attention heads.

DiffusionGemma runs 25 sliding layers at ``head_dim=256`` and 5 full-attention layers at
``global_head_dim=512``. No fused sdpa kernel takes a head that wide -- flash caps at 256 and
memory-efficient attention has no kernel either -- so on those five layers sdpa falls back to
the unfused math backend and materializes the whole ``[B, H, Nq, Nkv]`` score matrix. FFPA's
Split-D tiling is built for exactly this case (it covers head dims 320..1024), and on an A800
at the real shapes it runs the layer about 2x faster on a fraction of the peak memory -- a
quarter at prompt 2048, an eighth at 4096.

This is opt-in through ``config.attn_ffpa`` and deliberately does *not* introduce a new
``attn_implementation`` name. That name drives mask construction elsewhere in the model --
``_create_decoder_causal_mask`` only bakes the sliding window into the mask for
``sdpa``/``eager``/``flex_attention`` -- so an unrecognized name would quietly give every
sliding layer full attention over the encoder cache. Keeping ``_attn_implementation="sdpa"``
and swapping only the kernel leaves all of that untouched.

The flag is a request, not a preference: one that cannot be honoured raises. A run that
asked for FFPA and silently got sdpa reports another kernel's step time, MFU and peak
memory, and the five global layers are exactly where those two differ most -- a production
trial was read as FFPA's throughput when the package had in fact not imported. That is a
recurring accident rather than a one-off, because the package is installed out of band and
is not in the lockfile, so any ``uv sync`` removes it under a live config. To run on sdpa
deliberately, say so with ``attn_ffpa: false``.

Which refusals happen where is worth being precise about, because not every fallback is a
falsified record:

* ``ffpa_preflight`` raises at model construction for the two things a config decides: the
  package not importing, and ``attn_implementation`` not being sdpa. Both are static, so
  they should not cost a weight load and a first step to discover.
* dtype and device cannot be checked there -- they are settled by a later ``.to()`` -- so
  ``ffpa_attention_forward`` falls back on them, but warns, since like the two above they
  hold for the whole run rather than for one call.
* The per-layer width check and the per-shape checks below are silent. They are the normal
  case, not a thwarted request: only five layers are wide enough, and ``ffpa_attn_func``
  delegates short sequences to sdpa internally too.

Four properties of the kernel shape what the wrapper does:

* FFPA delegates to sdpa for ``8 <= Nq < 512`` and for ``Nkv < 512``
  (``FFPAAttnMeta.fallback``). The decoder's canvas is 256 long, so in practice only the
  encoder ever runs FFPA. We short-circuit on a slightly wider condition (any ``Nq`` under
  512) to skip the mask conversion below, which would otherwise be pure waste.
* With a *boolean* mask the Triton kernel returns NaN whenever the leading 64-wide key tile is
  fully masked: the running max starts at -inf and ``exp(-inf - -inf)`` is NaN. Prompts are
  left-padded (``DiffusionGemmaCollator``), which puts the dead keys exactly there, so every
  padded batch would hit it. Passing the same mask additively with ``finfo.min`` keeps the
  running max finite, and agrees with sdpa to bf16 rounding on every query row that has at
  least one visible key.
* ``is_causal`` means different things on the two sides. sdpa aligns the triangle to the top
  left, FFPA to the bottom right (``k <= r + (Nkv - Nq)``); they only coincide when
  ``Nq == Nkv``, and diverge grossly otherwise (0.75 absolute at 512x1024). Unequal lengths
  therefore fall back rather than being silently mistranslated. The flag itself is not
  decorative: the checkpoint sets ``use_bidirectional_attention = "vision"``, so text
  attention is causal, and for an unpadded batch transformers hands the layer
  ``attention_mask=None`` and expects the backend to apply causality itself.
* Dropout is unsupported here and falls back, even though FFPA implements it. This model runs
  ``attention_dropout=0.0``, so the branch only matters if someone turns dropout on.

Turning the flag on also costs ``torch.compile(fullgraph=True)``: FFPA wraps its autograd
entry in ``@torch._dynamo.disable`` on purpose, to keep Dynamo from inlining the backward, so
every global layer becomes a graph break.

``scripts/bench_diffusion_gemma_ffpa.py`` measures all of the above.
"""

from typing import TYPE_CHECKING, Callable, Optional

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ....utils import logging


if TYPE_CHECKING:
    from .configuration_diffusion_gemma import DiffusionGemmaTextConfig


logger = logging.get_logger(__name__)

try:
    from ffpa_attn import TritonBackend, ffpa_attn_func

    FFPA_AVAILABLE = True
    FFPA_IMPORT_ERROR = None
except Exception as error:  # noqa: BLE001
    # Not just ModuleNotFoundError. The package is installed out of band with `--no-deps`
    # against a torch it declares itself too old for, and it registers `torch.library` ops at
    # import time, so it can just as well raise AttributeError or RuntimeError. This module is
    # imported unconditionally by `modeling_diffusion_gemma`, which the model registry imports
    # at startup, so letting anything escape here would take down every model in the process.
    FFPA_AVAILABLE = False
    # A string, not the exception: keeping the object alive would pin its traceback, and with
    # it the half-initialized module and every torch object those frames reference.
    FFPA_IMPORT_ERROR = f"{type(error).__name__}: {error}"


# Head dims FFPA implements; at or below 256 it forwards to sdpa internally anyway.
FFPA_MIN_HEAD_DIM = 320
FFPA_MAX_HEAD_DIM = 1024

# Mirrors the `8 <= Nq < 512` / `Nkv < 512` fallback in `ffpa_attn.functional.FFPAAttnMeta`.
# Only an optimization -- drifting out of sync with the library costs a wasted mask conversion
# or a missed speedup, never a wrong result.
FFPA_MIN_SEQ_LEN = 512

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _require_ffpa(config: "DiffusionGemmaTextConfig") -> None:
    """Refuse an `attn_ffpa` this process cannot honour, whatever the layer."""
    if not FFPA_AVAILABLE:
        raise ImportError(
            f"`attn_ffpa` is set but ffpa-attn did not import ({FFPA_IMPORT_ERROR}). "
            "Install it with `uv pip install --no-deps ffpa-attn` -- note that a later `uv sync` "
            "drops it again, since it is not in the lockfile -- or set `attn_ffpa: false` to "
            "run the global layers on sdpa."
        )
    if config._attn_implementation != "sdpa":
        raise ValueError(
            f"`attn_ffpa` replaces sdpa, but `attn_implementation` is `{config._attn_implementation}`. "
            "Set `attn_ffpa: false` to keep that backend, or `attn_implementation: sdpa` to run "
            "FFPA on the global layers."
        )


def ffpa_preflight(config: "DiffusionGemmaTextConfig") -> None:
    """Fail an unhonourable `attn_ffpa` at construction, before a weight load and a step."""
    if config.attn_ffpa:
        _require_ffpa(config)


def ffpa_is_usable(config: "DiffusionGemmaTextConfig", head_dim: int) -> bool:
    """Whether FFPA serves a layer of this width. Shape-dependent checks come later."""
    if not config.attn_ffpa:
        return False
    # `ffpa_preflight` has normally raised this at build time already. Repeating it covers a
    # model assembled without going through `DiffusionGemmaModel.__init__`, where returning
    # True would reach a `ffpa_attn_func` the failed import never bound.
    _require_ffpa(config)
    # Silent: this one is per-layer and expected, since only the global layers are wide enough.
    return FFPA_MIN_HEAD_DIM <= head_dim <= FFPA_MAX_HEAD_DIM


def _as_additive_mask(attention_mask: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
    if attention_mask is None or attention_mask.dtype != torch.bool:
        return attention_mask
    # The model's decoder mask broadcasts over the query axis (`to_4d` uses `expand`, not a
    # copy), and FFPA accepts `size(-2) in (1, Nq)`. Collapsing first keeps the additive copy
    # as small as the boolean one -- at the encoder's shapes that is KiB instead of MiB, per
    # layer, held for the whole backward pass.
    if attention_mask.dim() >= 2 and attention_mask.stride(-2) == 0:
        attention_mask = attention_mask[..., :1, :]
    return torch.zeros(attention_mask.shape, dtype=dtype, device=attention_mask.device).masked_fill_(
        ~attention_mask, torch.finfo(dtype).min
    )


def ffpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Drop-in for transformers' ``sdpa_attention_forward`` on wide heads."""
    # Same rule as `sdpa_attention_forward`: the flag only applies when no mask was built, and
    # a single query row is a decode step that wants to see everything.
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    # dtype and device are the two `ffpa_preflight` cannot see, since `.to()` runs after
    # construction. They hold for every call once set, so say so rather than leaving the run
    # to report sdpa's cost as FFPA's -- the shape checks beside them really are per-call.
    if query.dtype not in _SUPPORTED_DTYPES:
        logger.warning_once(
            f"`attn_ffpa` is set but the model runs in {query.dtype}; the kernel implements only "
            "float16 and bfloat16, so the global layers stay on sdpa."
        )
    elif query.device.type != "cuda":
        logger.warning_once(
            f"`attn_ffpa` is set but this call is on {query.device.type}; the kernel is CUDA-only, "
            "so the global layers stay on sdpa."
        )

    if (
        query.dtype not in _SUPPORTED_DTYPES
        or query.device.type != "cuda"
        or dropout > 0.0
        or query.shape[2] < FFPA_MIN_SEQ_LEN
        or key.shape[2] < FFPA_MIN_SEQ_LEN
        # sdpa's causal triangle is top-left aligned, FFPA's is bottom-right aligned.
        or (is_causal and query.shape[2] != key.shape[2])
        # A 3D mask means `[B, 1, Nq, Nkv]` to FFPA but `[H, Nq, Nkv]` to sdpa, which
        # broadcasts right-aligned. Both mask builders in this model emit 4D, so this only
        # keeps the two from drifting apart.
        or (attention_mask is not None and attention_mask.dim() != 4)
    ):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
        )

    attn_output = ffpa_attn_func(
        query,
        key,
        value,
        attn_mask=_as_additive_mask(attention_mask, query.dtype),
        # Forwarding this matters: `scaling` is 1.0 here, because Gemma folds the attention
        # scale into `q_norm`, and FFPA would otherwise apply its own 1/sqrt(head_dim).
        scale=scaling,
        is_causal=is_causal,
        # The global layers are grouped-query: 16 heads over `num_global_key_value_heads`.
        enable_gqa=key.shape[1] != query.shape[1],
        # CuTeDSL rejects masks outright, and its A100 path is slower than Triton's anyway.
        forward_backend=TritonBackend(),
        backward_backend=TritonBackend(),
    )
    # Callers reshape from [B, Nq, H, D], matching `eager_attention_forward`.
    return attn_output.transpose(1, 2).contiguous(), None


def resolve_attention_interface(config: "DiffusionGemmaTextConfig", head_dim: int, default: Callable) -> Callable:
    """The attention callable for a layer of this width."""
    if ffpa_is_usable(config, head_dim):
        return ffpa_attention_forward
    return ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, default)
