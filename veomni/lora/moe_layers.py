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

"""VeOmni-owned LoRA wrappers for MoE expert modules.

Two flavours, both replacing the experts module in-place at the same
parent attribute (so downstream lookups by FQN continue to resolve):

* :class:`LoraIndependentExperts` — **Mode 1, default**. One LoRA pair
  *per expert* (3D ``[E, r, H]`` and ``[E, O, r]`` tensors). Equivalent in
  semantics to PEFT 0.19's ``target_parameters`` 3D-LoRA path, but VeOmni
  owns it end-to-end so we can (a) dispatch into a fused MoE-LoRA triton
  kernel and (b) keep eager / fused on identical math + key conventions.
* :class:`LoraSharedExperts` — **Mode 2**. A single LoRA pair (2D
  ``[r, H]`` and ``[O, r]`` tensors) shared across all experts of the
  layer. PEFT does not natively support this — a 2D parameter target
  isn't expressible via ``target_parameters`` (which assumes a leading
  expert dim).

Logical LoRA targets (seed-style two-LoRA on fused gate_up)
-----------------------------------------------------------
Even though the experts module stores ``gate_up_proj`` as one fused 3-D
``nn.Parameter`` of shape ``[E, 2I, H]``, the wrapper allocates **two
independent LoRA pairs** for it — one for the gate half and one for the
up half — keyed ``"gate_proj"`` and ``"up_proj"`` internally. The
``down_proj`` parameter gets a single LoRA pair keyed ``"down_proj"``.
So every wrapped experts module owns exactly three logical LoRA pairs:

==================  =====================  =====================
logical key          shared shapes (2-D)    independent shapes (3-D)
==================  =====================  =====================
``gate_proj``       ``A:[r,H]`` ``B:[I,r]`` ``A:[E,r,H]`` ``B:[E,I,r]``
``up_proj``         ``A:[r,H]`` ``B:[I,r]`` ``A:[E,r,H]`` ``B:[E,I,r]``
``down_proj``       ``A:[r,I]`` ``B:[H,r]`` ``A:[E,r,I]`` ``B:[E,H,r]``
==================  =====================  =====================

Why two LoRAs instead of one ``[2I, r]`` pair on the merged ``gate_up_proj``?
Sharing one ``A:[r, H]`` projection between gate and up forces both
deltas through the *same* low-rank subspace of ``x`` — strictly less
expressive (rank ``r`` in the joint (gate, up) output space) than two
independent rank-``r`` adapters (rank up to ``2r``). The seed_kernels
reference (``seed_fused_lora_moe`` — ``lora_fc1_1_*`` for gate,
``lora_fc1_2_*`` for up) takes the same view and so does v4 PEFT when
``gate_proj`` / ``up_proj`` are unmerged. Slightly more LoRA params on
gate_up (``2r(H + I)`` vs ``r(H + 2I)``) in exchange for the extra
expressivity and v4↔v5 semantic consistency.

Forward strategy
----------------
For one expert ``e`` with the merged base weight ``W_gu_e`` of shape
``[2I, H]`` (gate stacked above up) and per-half LoRA pairs
``(A_g, B_g)`` / ``(A_u, B_u)``::

    gate_aug_e(x) = W_gate_e @ x + B_g_e @ (A_g_e @ x) * s
    up_aug_e(x)   = W_up_e   @ x + B_u_e @ (A_u_e @ x) * s
    mid_e         = SiLU(gate_aug_e(x)) * up_aug_e(x)
    out_e         = W_dn_e @ mid_e + B_dn_e @ (A_dn_e @ mid_e) * s

The LoRA delta is added to the **pre-activation** linear (i.e.
``gate_aug`` / ``up_aug``) before SiLU — never to the post-activation
``mid`` — because SiLU is non-linear. This is enforced in both eager
forwards and the fused kernel by chunking the merged ``gate_up_proj``
output *after* adding the per-half LoRA deltas.

The right-hand factorisation ``B @ (A @ x) * s`` avoids materialising an
``[E, O, H]`` delta tensor (which is what PEFT's ``ParamWrapper`` does for
Mode 1 and what the docs warn about as runtime overhead). For Mode 2 the
gate/up LoRA contributions depend only on the *input* token and can be
computed once per token outside the per-expert dispatch loop; for Mode 1
every expert's LoRA is independent so it must be computed inside the
loop. The down LoRA always lives inside the loop because its input is
the per-expert intermediate activation ``silu(gate_aug) * up_aug``.

Expected experts module layout
------------------------------
The base experts module must own two 3-D ``nn.Parameter`` s: a fused
``gate_up_proj`` of shape ``[E, 2I, H]`` and a ``down_proj`` of shape
``[E, H, I]`` — the layout used by every Qwen3-MoE family on
``transformers >= 5.0.0`` (Qwen3-MoE / Qwen3.5-MoE / Qwen3-VL-MoE /
Qwen3-Omni-MoE generated or patched modeling).
:func:`_validate_fused_layout` raises with a clear message if the
experts module exposes anything else (older split ``gate_proj`` /
``up_proj`` / ``down_proj`` layouts are no longer supported — the
project is v5-only).

Both wrappers reorganise the wrapped experts module's parameters into
**per-spec sub-modules** that mirror PEFT's standard ``LoraLayer``
layout. Each base parameter ends up under
``<spec>.base_layer.weight`` and each LoRA pair under
``<spec>.lora_A.<adapter>.weight`` / ``<spec>.lora_B.<adapter>.weight``,
where ``<spec>`` is one of the three logical keys ``gate_proj`` /
``up_proj`` / ``down_proj``. The post-wrap FQN of e.g. the fused
gate_up base parameter therefore moves from
``model.layers.0.mlp.experts.gate_up_proj`` (bare ``nn.Parameter``) to
``model.layers.0.mlp.experts.gate_up_proj.base_layer.weight`` (PEFT-style
sub-module path).

The bare-model parallel plan (``model.layers.*.mlp.experts.gate_up_proj``
etc.) is rewritten to the wrapped FQN by
``_rewrite_plan_for_moe_lora_wrappers`` (in
``veomni/distributed/parallel_plan.py``) so :meth:`ParallelPlan.apply`
still finds and EP-shards the base experts after wrapping, and so the
EP-aware rank-0 broadcast / per-rank load paths slice the disk-side
``[E, ...]`` tensors down to ``[E_local, ...]`` correctly.

A fused-Triton path is bound for both modes in
``veomni/lora/ops/moe_group_gemm.py`` (non-EP and EP).

PEFT-format save/load compatibility (PEFT-aligned FQN layout)
-------------------------------------------------------------
Because each spec sub-module holds the canonical ``lora_A`` / ``lora_B``
``nn.ModuleDict`` containers, the saved adapter keys come out as
``base_model.model....experts.<spec>.lora_A.weight`` -- byte-identical to PEFT's
standard ``LoraLayer`` storage, so the file is loadable by any third-party tool
that consumes PEFT ``adapter_model`` files.

These wrappers are owned end-to-end by the native :mod:`veomni.lora` stack (no
``peft`` runtime dependency). :class:`veomni.lora.model.VeOmniLoraModel`
installs them (via :func:`veomni.lora.moe_layers.inject_moe_lora`), saves them
through :func:`veomni.lora.state_dict.get_lora_state_dict`, and re-installs them
on resume from the ``veomni_lora`` block of ``adapter_config.json`` — the MoE
mode (``shared`` / ``independent``) lives there, so there is **no separate
sidecar file**. When loading a stock-PEFT adapter (no ``veomni_lora`` block),
the mode is inferred from the on-disk LoRA tensor rank (3-D per-expert →
independent, 2-D → shared).
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..utils import logging


if TYPE_CHECKING:
    from .config import VeOmniLoraConfig


logger = logging.get_logger(__name__)


# Module FQNs of PEFT-wrapped models gain a ``base_model.model.`` prefix.
# Patterns supplied by the user in ``lora_config['target_parameters']`` are
# written against the *base* model FQN (e.g. ``model.layers.0.mlp.experts.gate_up_proj``)
# and stripped before matching.
_PEFT_PREFIX = "base_model.model."


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a PEFT-style glob (``*``) to a fully-anchored regex."""
    parts = [re.escape(piece) for piece in pattern.split("*")]
    return re.compile(".*".join(parts) + r"\Z")


def _strip_peft_prefix(fqn: str) -> str:
    return fqn[len(_PEFT_PREFIX) :] if fqn.startswith(_PEFT_PREFIX) else fqn


def _find_target_parameter_modules(
    model: nn.Module,
    patterns: list[str],
) -> list[tuple[nn.Module, str, str, nn.Module]]:
    """Find experts modules whose 3D parameters match any of ``patterns``.

    Returns a list of ``(parent_module, parent_fqn, attr_name, base_module)``
    tuples — one per *unique* experts module that owns at least one matching
    parameter. ``base_module`` is the experts module to be wrapped;
    ``parent.<attr_name> is base_module``.

    Parameters with ``ndim != 3`` are ignored: shared LoRA is defined as a
    broadcast over the expert dimension, so the target must be 3D.
    """
    if not patterns:
        return []
    compiled = [_glob_to_regex(p) for p in patterns]

    seen_modules: set[int] = set()
    matches: list[tuple[nn.Module, str, str, nn.Module]] = []
    for fqn, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if param.ndim != 3:
                continue
            full = f"{fqn}.{pname}" if fqn else pname
            stripped = _strip_peft_prefix(full)
            if not any(rx.fullmatch(stripped) for rx in compiled):
                continue
            mod_id = id(module)
            if mod_id in seen_modules:
                continue
            seen_modules.add(mod_id)
            parent_fqn, _, attr_name = fqn.rpartition(".")
            parent = model.get_submodule(parent_fqn) if parent_fqn else model
            matches.append((parent, parent_fqn, attr_name, module))
            break  # one match per module is enough — we wrap the whole module

    return matches


# Fused experts layout: base experts module owns these two 3D Parameters.
_FUSED_PARAMS = ("gate_up_proj", "down_proj")

# Logical LoRA spec keys per experts-module parameter. ``gate_up_proj`` (the
# fused 2I-output linear) decomposes into two independent rank-r adapters —
# one for the gate half, one for the up half (seed_kernels-style); see
# ``seed_fused_lora_moe`` ``lora_fc1_1_*`` / ``lora_fc1_2_*`` for the
# reference layout. ``down_proj`` keeps a single LoRA pair.
_LORA_SPEC_KEYS = ("gate_proj", "up_proj", "down_proj")


def _is_lora_param_name(name: str) -> bool:
    """True iff ``name`` (a wrapper-local FQN) belongs to a LoRA tensor.

    Under the PEFT-aligned layout every LoRA parameter lives at
    ``<spec>.lora_A.<adapter>.weight`` or ``<spec>.lora_B.<adapter>.weight``.
    Used by both wrappers to (a) flip ``requires_grad`` on after the
    blanket-freeze, (b) decide whether the wrapper is fully meta-init (so
    ``reset_lora_parameters`` should be deferred) and (c) classify
    parameters in the cross-EP grad-sync hook installer. We split on
    ``.`` and look for the canonical ``lora_A`` / ``lora_B`` segment so
    the check is robust to extra prefixes added by FSDP wrapping.
    """
    parts = name.split(".")
    return "lora_A" in parts or "lora_B" in parts


class _BaseParamHolder(nn.Module):
    """Tiny holder exposing an ``nn.Parameter`` as ``.weight``.

    Matches PEFT's convention of storing the original weight under
    ``base_layer.weight`` even when the wrapped tensor is a 3-D fused MoE
    parameter (no ``nn.Linear`` involved). The ``_is_bare_param_holder``
    marker tells :func:`veomni.lora.weight_loading.build_lora_key_overrides`
    to map the *bare* base-model key (``...experts.gate_up_proj``, no
    ``.weight`` because it was an ``nn.Parameter`` directly on the
    experts module) to the wrapped FQN
    (``...experts.gate_up_proj.base_layer.weight``).
    """

    _is_bare_param_holder: bool = True

    def __init__(self, param: nn.Parameter) -> None:
        super().__init__()
        # Re-register the same Parameter object — no copy, so EP slicing
        # done by ParallelPlan.apply on the original param continues to
        # apply unchanged.
        self.weight = param


class _LoraSpec(nn.Module):
    """PEFT-aligned per-spec sub-module of a MoE-LoRA wrapper.

    Hosts (optionally) the lifted base parameter under
    ``base_layer.weight`` and the LoRA pair under
    ``lora_A.<adapter>.weight`` / ``lora_B.<adapter>.weight``. The exact
    FQN structure mirrors PEFT's standard ``LoraLayer``, so the native
    ``veomni.lora.state_dict`` helpers and
    ``veomni.lora.weight_loading.build_lora_key_overrides`` work without any
    per-wrapper specials.

    Three flavours used by both wrappers:

    * ``base_layer`` only (no LoRA) -- e.g. the ``gate_up_proj`` spec
      under seed-style two-LoRA, where the fused base is covered by two
      separate LoRA-only specs (``gate_proj``, ``up_proj``).
    * ``base_layer`` + LoRA -- the canonical PEFT layout, used for the
      ``down_proj`` spec.
    * LoRA only -- the ``gate_proj`` and ``up_proj`` specs under
      seed-style two-LoRA on the fused gate_up base.
    """

    def __init__(self, *, base_param: nn.Parameter | None = None) -> None:
        super().__init__()
        if base_param is not None:
            self.base_layer = _BaseParamHolder(base_param)
        # Always allocate the (initially empty) ModuleDict containers so
        # callers can do ``spec.lora_A[name] = ...`` uniformly without a
        # presence check; FSDP / state_dict skip empty ModuleDicts.
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

    @property
    def has_base_layer(self) -> bool:
        return hasattr(self, "base_layer")

    @property
    def base_weight(self) -> torch.Tensor:
        """Direct accessor to the wrapped base parameter for forward-pass code."""
        return self.base_layer.weight


def _validate_fused_layout(base_layer: nn.Module) -> None:
    """Raise if ``base_layer`` is not the v5-style fused MoE experts layout.

    VeOmni MoE-LoRA is v5-only: the base experts module must own a fused
    ``gate_up_proj`` (3-D ``nn.Parameter`` of shape ``[E, 2I, H]``) and a
    ``down_proj`` (``[E, H, I]``). Older split ``gate_proj`` / ``up_proj``
    layouts are not supported — the project no longer ships v4 patches.
    """

    def has(name: str) -> bool:
        return hasattr(base_layer, name) and isinstance(getattr(base_layer, name), nn.Parameter)

    if has("gate_up_proj") and has("down_proj"):
        return
    raise ValueError(
        f"VeOmni MoE-LoRA cannot wrap {type(base_layer).__name__!s}: "
        "expected fused experts layout (gate_up_proj + down_proj as 3-D "
        "nn.Parameters). Got attrs: "
        f"{[n for n, _ in base_layer.named_parameters(recurse=False)]}"
    )


class LoraSharedExperts(nn.Module):
    """Wrap a MoE experts module to add a single LoRA pair shared across experts.

    Args:
        base_layer: The original experts module (e.g. ``Qwen3MoeExperts``).
            The wrapper takes ownership: ``base_layer``'s Parameters and
            Buffers are *lifted* (re-registered) onto ``self`` at their
            original attribute names (``gate_up_proj``, ``down_proj``,
            ...), then frozen. ``base_layer`` is left drained — the
            wrapper never calls ``base_layer.forward()`` or holds a
            reference to it.
        r: LoRA rank.
        lora_alpha: LoRA alpha. Scaling is ``alpha / r`` (or ``alpha / sqrt(r)``
            when ``use_rslora=True``), matching PEFT.
        use_rslora: Use rank-stabilised LoRA scaling.
        adapter_name: PEFT-style adapter name. Currently a single adapter is
            supported; the name is stored for save/load helpers.

    LoRA parameters live inside per-spec :class:`_LoraSpec` sub-modules,
    one per *logical* LoRA target — three sub-modules total per wrapped
    experts module. Each spec exposes PEFT's standard ``lora_A`` /
    ``lora_B`` ``nn.ModuleDict`` containers keyed by adapter name and
    holding an ``nn.Linear(in_features, r)`` (A) or
    ``nn.Linear(r, out_features)`` (B). The full FQN map under
    ``experts``:

    * ``gate_up_proj.base_layer.weight`` — fused base ``[E, 2I, H]``
    * ``down_proj.base_layer.weight``    — base ``[E, H, I]``
    * ``gate_proj.lora_A.<adapter>.weight`` / ``gate_proj.lora_B.<adapter>.weight``
    * ``up_proj.lora_A.<adapter>.weight`` / ``up_proj.lora_B.<adapter>.weight``
    * ``down_proj.lora_A.<adapter>.weight`` / ``down_proj.lora_B.<adapter>.weight``

    The fused ``gate_up_proj`` parameter on the base experts module is
    therefore covered by **two independent rank-r adapters** (one per
    half) — see the file docstring "Logical LoRA targets (seed-style
    two-LoRA on fused gate_up)" for the rationale. ``gate_proj`` and
    ``up_proj`` specs hold *only* LoRA (no ``base_layer``) since they
    share the fused base; ``down_proj`` carries both base and LoRA in
    the canonical PEFT layout.

    The resulting FQNs (``...experts.<spec>.lora_A.<adapter>.weight``)
    are byte-identical to PEFT's standard ``LoraLayer`` storage, so the
    standard ``model.save_pretrained`` / ``PeftModel.from_pretrained``
    path works once the wrappers are installed, and adding a second
    adapter is a one-line ``spec.lora_A[other_adapter] = nn.Linear(...)``.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int,
        lora_alpha: int,
        use_rslora: bool = False,
        adapter_name: str = "default",
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {r}")

        # Validate before stealing — base_layer's params/buffers are lifted
        # into self below so a downstream caller seeing the drained
        # base_layer would get a confusing error.
        _validate_fused_layout(base_layer)

        self.r = r
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora
        self.adapter_name = adapter_name

        # Geometry sourced from the base module — these are the standard
        # attribute names used across all v5-patched Qwen3 MoE families.
        self.num_experts = base_layer.num_experts
        self.hidden_dim = base_layer.hidden_dim
        self.intermediate_dim = base_layer.intermediate_dim
        self.act_fn = base_layer.act_fn

        # Steal the base experts' Parameters / Buffers, then re-register
        # the Parameters under per-spec sub-modules at the PEFT-aligned
        # path ``<spec>.base_layer.weight``. Same Parameter object — no
        # copy — so any EP slicing applied to the original tensor by
        # ``ParallelPlan.apply`` continues to apply transparently.
        # Buffers (e.g. residual masks) keep living at the wrapper's
        # top level since PEFT has no analogous convention for them.
        base_params: dict[str, nn.Parameter] = {}
        for name in list(base_layer._parameters):
            param = base_layer._parameters.pop(name)
            if param is not None:
                base_params[name] = param
        for name in list(base_layer._buffers):
            buf = base_layer._buffers.pop(name)
            if buf is not None:
                self.register_buffer(name, buf)

        # Three logical LoRA pairs per experts module — same key set as
        # the fused gate_up base ⇒ ``gate_proj`` + ``up_proj`` (seed-style
        # two-LoRA), plus ``down_proj`` for the down base. Shapes follow
        # F.linear semantics: F.linear(x, W) computes x @ W.T, so for
        # output dim ``O`` and input dim ``H`` the pair (A: [r, H],
        # B: [O, r]) gives delta_W = B @ A of shape [O, H].
        self._lora_specs = {
            "gate_proj": (self.hidden_dim, self.intermediate_dim),
            "up_proj": (self.hidden_dim, self.intermediate_dim),
            "down_proj": (self.intermediate_dim, self.hidden_dim),
        }

        # Per-spec sub-modules:
        #   ``gate_up_proj`` -- base only (no LoRA; covered by gate_proj +
        #                       up_proj LoRA-only specs).
        #   ``down_proj``    -- base + LoRA (canonical PEFT layout).
        #   ``gate_proj``    -- LoRA only.
        #   ``up_proj``      -- LoRA only.
        # Order matters only for state_dict iteration; we keep base-first
        # so PEFT's get_peft_model_state_dict sees them in a stable order.
        self.add_module("gate_up_proj", _LoraSpec(base_param=base_params["gate_up_proj"]))
        self.add_module("down_proj", _LoraSpec(base_param=base_params["down_proj"]))
        self.add_module("gate_proj", _LoraSpec())
        self.add_module("up_proj", _LoraSpec())

        # Inherit dtype/device from the lifted fused base (always present
        # after ``_validate_fused_layout``) so new Linears land on the
        # same device (typically meta or cuda).
        ref = self.gate_up_proj.base_layer.weight
        factory_kwargs = {"dtype": ref.dtype, "device": ref.device}

        for spec_name, (in_feat, out_feat) in self._lora_specs.items():
            spec: _LoraSpec = getattr(self, spec_name)
            # ModuleDict[adapter -> Linear(bias=False)] — PEFT's exact
            # convention so ``get_peft_model_state_dict`` /
            # ``set_peft_model_state_dict`` round-trip with no remap, and
            # adding a second adapter is just a fresh assignment.
            spec.lora_A[adapter_name] = nn.Linear(in_feat, r, bias=False, **factory_kwargs)
            spec.lora_B[adapter_name] = nn.Linear(r, out_feat, bias=False, **factory_kwargs)

        scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)
        self.register_buffer("lora_scaling", torch.tensor(scaling, dtype=torch.float32))
        # Python-float copy of the scaling factor — used by the fused MoE-LoRA
        # forward kernel (``veomni.lora.ops.moe_group_gemm``) which takes
        # the scale as a plain float to avoid a host/device sync inside the
        # autograd.Function. Kept in lock-step with ``lora_scaling``.
        self._lora_scale_value: float = float(scaling)

        # Freeze base, then unfreeze lora_*. ``_is_lora_param_name``
        # detects the canonical ``lora_A`` / ``lora_B`` segments in the
        # FQN, so it works regardless of FSDP's optional renaming.
        for p in self.parameters():
            p.requires_grad = False
        for n, p in self.named_parameters():
            if _is_lora_param_name(n):
                p.requires_grad = True

        # Init: kaiming-uniform A, zero B → no-op vs base at init (PEFT default
        # for ``init_lora_weights=True``). Skip when params are on meta device
        # (post-meta init weight loading will call ``reset_lora_parameters``).
        if not any(p.is_meta for n, p in self.named_parameters() if _is_lora_param_name(n)):
            self.reset_lora_parameters()

    # ------------------------------------------------------------------
    # PEFT-compatible accessors.
    # ------------------------------------------------------------------

    def _get_lora_module(self, role: str, param_name: str, adapter_name: str | None = None) -> nn.Module:
        adapter = adapter_name or self.adapter_name
        spec: _LoraSpec = getattr(self, param_name)
        return getattr(spec, f"lora_{role}")[adapter]

    def get_lora_A_weight(self, param_name: str, adapter_name: str | None = None) -> torch.Tensor:
        """Active LoRA A weight for ``param_name``, shape ``[r, in_features]``."""
        return self._get_lora_module("A", param_name, adapter_name).weight

    def get_lora_B_weight(self, param_name: str, adapter_name: str | None = None) -> torch.Tensor:
        """Active LoRA B weight for ``param_name``, shape ``[out_features, r]``."""
        return self._get_lora_module("B", param_name, adapter_name).weight

    @torch.no_grad()
    def reset_lora_parameters(self, adapter_name: str | None = None, init_lora_weights: bool = True) -> None:
        """Initialise LoRA A (kaiming-uniform) and B (zeros). Idempotent.

        Signature matches PEFT's ``LoraLayer.reset_lora_parameters`` so
        ``veomni.lora.weight_loading.init_lora_parameter`` can call it without
        special-casing — passing ``adapter_name=None`` resets every adapter,
        mirroring PEFT's behaviour when no adapter is selected.
        """
        if not init_lora_weights:
            return
        for pname in self._lora_specs:
            spec: _LoraSpec = getattr(self, pname)
            for ad, mod in spec.lora_A.items():
                if adapter_name is not None and ad != adapter_name:
                    continue
                nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
            for ad, mod in spec.lora_B.items():
                if adapter_name is not None and ad != adapter_name:
                    continue
                nn.init.zeros_(mod.weight)

    # PEFT-style ``lora_A`` / ``lora_B`` accessors expected by some helpers
    # (e.g. VeOmni's ``_init_lora_parameter`` introspection of adapter names).
    # We expose the LoRA-only ``gate_proj`` spec as the canonical
    # representative — every spec carries the same adapter set so callers
    # iterating ``lora_A.keys()`` always see the right adapter list.
    @property
    def lora_A(self) -> dict[str, nn.Module]:
        first_pname = next(iter(self._lora_specs))
        spec: _LoraSpec = getattr(self, first_pname)
        return dict(spec.lora_A)

    @property
    def lora_B(self) -> dict[str, nn.Module]:
        first_pname = next(iter(self._lora_specs))
        spec: _LoraSpec = getattr(self, first_pname)
        return dict(spec.lora_B)

    # ------------------------------------------------------------------
    # Cross-EP gradient sync for shared LoRA params
    # ------------------------------------------------------------------

    def _ensure_ep_grad_sync_hooks(self) -> None:
        """Install a SUM all-reduce on each shared LoRA param's grad across the EP group.

        Why this is needed
        ------------------
        Under EP>1, ``build_parallelize_model`` wraps the entire wrapper module
        with an inner ``fully_shard(..., mesh=ep_fsdp, shard_placement_fn=Shard(1))``
        and calls ``set_gradient_divide_factor(world_size)`` on it. The
        ``ep_fsdp`` mesh has size 1 here (all ranks belong to a different EP
        group), so reduce-scatter is a no-op for data — but the divide-by-N
        still applies to *every* parameter in the wrapped module.

        For the EP-sharded base ``gate_up_proj`` / ``down_proj`` this is fine:
        each rank holds a different slice of experts, its local grad is for its
        own params, and no cross-EP sync is meaningful.

        For the *shared* LoRA params (logically replicated across EP — every
        rank holds the same ``[r, H]`` / ``[O, r]`` tensor), the local grad on
        each rank is only the partial contribution from that rank's local
        experts. With no cross-EP all-reduce the optimizer steps each rank
        independently → ranks diverge after the first step, breaking the
        "shared" semantics, and any DCP round-trip is no longer bit-exact.

        Fix: a SUM all-reduce across the EP group on the post-FSDP grad. After
        FSDP gives each rank ``local_grad / world_size``, the SUM all-reduce
        produces ``(sum_over_ep_ranks(local_grad)) / world_size`` =
        ``total_grad / world_size`` on every rank — matching the EP=1+DP=W
        view where FSDP all-reduces the full grad across the DP group and
        divides by W.

        Idempotent (guarded by ``_ep_grad_hooks_done``); no-op when EP is
        disabled or distributed is not initialised.
        """
        if getattr(self, "_ep_grad_hooks_done", False):
            return

        from ..distributed.parallel_state import get_parallel_state

        if not (dist.is_available() and dist.is_initialized()):
            self._ep_grad_hooks_done = True
            return
        ps = get_parallel_state()
        if not ps.ep_enabled:
            self._ep_grad_hooks_done = True
            return

        ep_group = ps.ep_group

        # Local import so this module stays importable on torch builds where
        # ``torch.distributed.tensor`` is absent (already true everywhere we
        # support, but mirrors the pattern used in ``reset_lora_parameters``
        # below to keep the hook self-contained).
        from torch.distributed.tensor import DTensor

        def _make_hook():
            def _hook(p: torch.Tensor) -> None:
                if p.grad is None:
                    return
                grad = p.grad
                # Reach the underlying storage that the optimizer reads.
                # ``DTensor._local_tensor`` is the actual ``torch.Tensor``
                # backing the local shard -- always writable, always aliases
                # grad's storage. ``to_local()`` only guarantees "the local
                # component"; for ``Replicate`` / ``Partial`` placements after
                # a redistribute it may legally return a fresh copy whose
                # ``copy_`` would not propagate back. Today's shared-LoRA
                # path uses ``Shard(dim=1)`` on the ``ep_fsdp`` mesh so
                # ``to_local()`` happens to be a view, but pinning to
                # ``_local_tensor`` makes the writability invariant explicit
                # and future-proofs against placement changes.
                local_storage = grad._local_tensor if isinstance(grad, DTensor) else grad
                # NCCL all_reduce requires a contiguous buffer. When the local
                # storage is non-contiguous (e.g. ``Shard(d>0)`` with d>0
                # strided against a row-major global), allocate a temporary
                # contiguous buffer, run the collective on it, and copy the
                # SUM-reduced result back into the (possibly strided)
                # underlying storage.
                if local_storage.is_contiguous():
                    dist.all_reduce(local_storage, op=dist.ReduceOp.SUM, group=ep_group)
                else:
                    buf = local_storage.contiguous()
                    dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=ep_group)
                    local_storage.copy_(buf)

            return _hook

        for n, p in self.named_parameters(recurse=True):
            if _is_lora_param_name(n) and p.requires_grad:
                p.register_post_accumulate_grad_hook(_make_hook())
        self._ep_grad_hooks_done = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Lazy-install cross-EP gradient sync on shared LoRA params (idempotent).
        # Must run after FSDP wrapping has converted params to DTensors, hence
        # the lazy install at first forward rather than in ``__init__``.
        self._ensure_ep_grad_sync_hooks()
        # Fused-kernel path: available when the user opted into a non-eager
        # ``moe_implementation`` whose patch function bound a LoRA-aware
        # kernel (currently 'fused_triton'; Quack/NPU leave
        # ``_fused_lora_moe_forward = None`` so we transparently fall back to
        # eager). The bound kernel handles the EP branch internally via
        # ``preprocess`` / ``token_pre_all2all`` / ``EPMergedFc1SharedLoRAGroupGemm``
        # / ``tokens_post_all2all`` — no EP gating needed here.
        from ..distributed.parallel_state import get_parallel_state
        from . import ops as _lora_ops

        if _lora_ops._fused_lora_moe_forward is not None:
            return self._fused_forward(_lora_ops._fused_lora_moe_forward, hidden_states, top_k_index, top_k_weights)
        # Eager fallback. Note: the eager forward indexes ``base.gate_up_proj``
        # / ``base.down_proj`` by global ``top_k_index``, which only works when
        # the experts module owns the *full* expert set. Under EP the experts
        # module is local-sliced and ``top_k_index`` carries global ids, so
        # eager would index out of range. Surface that as a clear error rather
        # than letting the indexing fail downstream.
        if get_parallel_state().ep_enabled:
            raise RuntimeError(
                "LoraSharedExperts: eager forward does not support expert parallelism (EP). "
                "Set ops_implementation.moe_implementation='fused_triton' to use the EP-aware "
                "fused LoRA path, or disable EP."
            )
        return self._eager_forward(hidden_states, top_k_index, top_k_weights)

    def _fused_forward(
        self,
        fused_kernel,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch into the bound fused MoE-LoRA kernel.

        ``fused_kernel`` is the kernel pointer captured by ``forward`` so we
        don't re-read the module attribute twice (cheap optimisation, also
        keeps this method side-effect free for testing). Passes both halves
        of the seed-style gate_up LoRA pair as separate ``(A, B, scale)``
        triples — the kernel keeps them split end-to-end to avoid the
        rank-collapse a merged ``[2I, H]`` LoRA would impose.
        """
        return fused_kernel(
            num_experts=self.num_experts,
            routing_weights=top_k_weights.to(hidden_states.dtype),
            selected_experts=top_k_index,
            hidden_states=hidden_states,
            fc1_1_2_weight=self.gate_up_proj.base_layer.weight,
            fc2_weight=self.down_proj.base_layer.weight,
            lora_a_gate=self.get_lora_A_weight("gate_proj"),
            lora_b_gate=self.get_lora_B_weight("gate_proj"),
            lora_a_up=self.get_lora_A_weight("up_proj"),
            lora_b_up=self.get_lora_B_weight("up_proj"),
            lora_a_down=self.get_lora_A_weight("down_proj"),
            lora_b_down=self.get_lora_B_weight("down_proj"),
            lora_scale_gate=self._lora_scale_value,
            lora_scale_up=self._lora_scale_value,
            lora_scale_down=self._lora_scale_value,
        )

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.lora_scaling.to(hidden_states.dtype)
        a_gate = self.get_lora_A_weight("gate_proj")
        b_gate = self.get_lora_B_weight("gate_proj")
        a_up = self.get_lora_A_weight("up_proj")
        b_up = self.get_lora_B_weight("up_proj")
        a_dn = self.get_lora_A_weight("down_proj")
        b_dn = self.get_lora_B_weight("down_proj")

        # Pull base weight tensors once (expensive only via FSDP unshard);
        # downstream loop just indexes per-expert slices on the local tensor.
        gate_up_w = self.gate_up_proj.base_layer.weight
        down_w = self.down_proj.base_layer.weight

        # Two independent shared LoRA deltas — gate and up each get their
        # own rank-r adapter. Both depend only on x ⇒ compute once and slice
        # per expert below. Cat into a single ``[N, 2I]`` block so the
        # per-expert add lines up with the merged ``gate_up_proj`` output
        # before chunk + SiLU (LoRA must enter pre-activation).
        gate_delta = F.linear(F.linear(hidden_states, a_gate), b_gate) * scale  # [N, I]
        up_delta = F.linear(F.linear(hidden_states, a_up), b_up) * scale  # [N, I]
        lora_x_gate_up = torch.cat([gate_delta, up_delta], dim=-1)  # [N, 2I]

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_up = F.linear(current_state, gate_up_w[expert_idx]) + lora_x_gate_up[token_idx]
            gate, up = gate_up.chunk(2, dim=-1)
            mid = self.act_fn(gate) * up

            # down LoRA depends on the per-expert intermediate, so compute inside the loop.
            lora_x_down = F.linear(F.linear(mid, a_dn), b_dn) * scale
            current_hidden_states = F.linear(mid, down_w[expert_idx]) + lora_x_down
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.lora_alpha}, num_experts={self.num_experts}"


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1 — independent per-expert LoRA.
#
# Same wrapping protocol and key conventions as ``LoraSharedExperts`` so the
# shared save/load + resume machinery (PEFT round-trip + VeOmni sidecar) Just
# Works. The differences are entirely in (a) parameter shape — 3-D, leading
# expert dim — and (b) forward — every expert reads its own LoRA slice rather
# than a single broadcast pair.
# ──────────────────────────────────────────────────────────────────────────────


class _LoraParam3D(nn.Module):
    """Container exposing one 3-D LoRA tensor as ``.weight``.

    Mirrors ``nn.Linear``'s ``.weight`` attribute so the state-dict key
    ``<spec>.lora_A.<adapter>.weight`` round-trips through PEFT's
    ``get_peft_model_state_dict`` / ``set_peft_model_state_dict`` exactly the
    same way it does for the 2-D :class:`LoraSharedExperts` case.

    Why a wrapper rather than a bare ``nn.ParameterDict``?
        ``ParameterDict[adapter -> Parameter]`` would produce keys like
        ``<spec>.lora_A.<adapter>`` (no trailing ``.weight``), which works in
        principle but breaks symmetry with both PEFT's standard ``LoraLayer``
        storage and our shared wrapper. Keeping ``.weight`` everywhere lets
        every consumer (``veomni.lora.state_dict`` key remap, FSDP load
        helpers, the round-trip test) treat both modes interchangeably.
    """

    def __init__(self, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype, device=device))


class LoraIndependentExperts(nn.Module):
    """Wrap a MoE experts module to add an independent LoRA pair per expert.

    Args:
        base_layer: The original experts module (e.g. ``Qwen3MoeExperts``).
            The wrapper takes ownership: ``base_layer``'s Parameters and
            Buffers are *lifted* (re-registered) onto ``self`` at their
            original attribute names, then frozen. See
            :class:`LoraSharedExperts` for the lifting rationale (FQN
            preservation under the bare-model parallel plan).
        r: LoRA rank.
        lora_alpha: LoRA alpha. Scaling is ``alpha / r`` (or ``alpha / sqrt(r)``
            when ``use_rslora=True``), matching PEFT.
        use_rslora: Use rank-stabilised LoRA scaling.
        adapter_name: PEFT-style adapter name. Currently a single adapter is
            supported; the name is stored for save/load helpers.

    LoRA tensors are 3-D with the leading dim equal to ``num_experts``;
    the fused ``gate_up_proj`` base parameter is covered by **two
    independent rank-r adapters** (seed-style two-LoRA, see file
    docstring), keyed ``gate_proj`` / ``up_proj``::

        gate_proj.lora_A.<adapter>.weight   # [E, r, H]
        gate_proj.lora_B.<adapter>.weight   # [E, I, r]
        up_proj.lora_A.<adapter>.weight     # [E, r, H]
        up_proj.lora_B.<adapter>.weight     # [E, I, r]
        down_proj.lora_A.<adapter>.weight   # [E, r, I]
        down_proj.lora_B.<adapter>.weight   # [E, H, r]

    Base parameters live at ``gate_up_proj.base_layer.weight`` and
    ``down_proj.base_layer.weight`` (per-spec sub-modules, PEFT-aligned;
    see :class:`LoraSharedExperts` for the layout rationale).

    Forward semantics for token ``t`` routed to expert ``e`` (with the
    merged base ``W_gu_e`` chunked into gate / up halves on the fly)::

        gate_aug_e(x_t) = W_gate_e @ x_t + B_g_e @ (A_g_e @ x_t) * scale
        up_aug_e(x_t)   = W_up_e   @ x_t + B_u_e @ (A_u_e @ x_t) * scale
        mid_t           = SiLU(gate_aug_e(x_t)) * up_aug_e(x_t)
        out_t           = W_dn_e @ mid_t + B_dn_e @ (A_dn_e @ mid_t) * scale

    The wrapper's ``forward`` dispatches into a fused MoE-LoRA triton
    kernel (non-EP and EP) when bound, falling back to the eager loop
    above otherwise.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int,
        lora_alpha: int,
        use_rslora: bool = False,
        adapter_name: str = "default",
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {r}")

        # Validate before stealing — see LoraSharedExperts.__init__ for the
        # rationale on this ordering.
        _validate_fused_layout(base_layer)

        self.r = r
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora
        self.adapter_name = adapter_name

        self.num_experts = base_layer.num_experts
        self.hidden_dim = base_layer.hidden_dim
        self.intermediate_dim = base_layer.intermediate_dim
        self.act_fn = base_layer.act_fn

        # Steal base experts' Parameters / Buffers — see
        # LoraSharedExperts.__init__ for the per-spec sub-module layout
        # rationale (PEFT-aligned ``<spec>.base_layer.weight``). The
        # ExtraParallel slicing in ``parallel_plan.apply`` keys on the
        # rewritten plan emitted by ``_rewrite_plan_for_moe_lora_wrappers``.
        base_params: dict[str, nn.Parameter] = {}
        for name in list(base_layer._parameters):
            param = base_layer._parameters.pop(name)
            if param is not None:
                base_params[name] = param
        for name in list(base_layer._buffers):
            buf = base_layer._buffers.pop(name)
            if buf is not None:
                self.register_buffer(name, buf)

        # Three logical LoRA pairs per experts module — same key set as
        # LoraSharedExperts. The 3-D parameter shapes derived below add a
        # leading expert dim. ``gate_proj`` and ``up_proj`` together cover
        # the fused ``gate_up_proj`` base param (seed-style two-LoRA);
        # ``down_proj`` covers the down base param.
        self._lora_specs = {
            "gate_proj": (self.hidden_dim, self.intermediate_dim),
            "up_proj": (self.hidden_dim, self.intermediate_dim),
            "down_proj": (self.intermediate_dim, self.hidden_dim),
        }

        # Per-spec sub-modules — same layout as LoraSharedExperts.
        self.add_module("gate_up_proj", _LoraSpec(base_param=base_params["gate_up_proj"]))
        self.add_module("down_proj", _LoraSpec(base_param=base_params["down_proj"]))
        self.add_module("gate_proj", _LoraSpec())
        self.add_module("up_proj", _LoraSpec())

        ref = self.gate_up_proj.base_layer.weight
        factory_kwargs = {"dtype": ref.dtype, "device": ref.device}

        # Per-spec ModuleDict[adapter -> _LoraParam3D]. Leading expert dim
        # makes every LoRA slice independent; downstream forward indexes by
        # expert_idx into ``spec.lora_A[adapter].weight[expert_idx]``.
        for spec_name, (in_feat, out_feat) in self._lora_specs.items():
            spec: _LoraSpec = getattr(self, spec_name)
            spec.lora_A[adapter_name] = _LoraParam3D((self.num_experts, r, in_feat), **factory_kwargs)
            spec.lora_B[adapter_name] = _LoraParam3D((self.num_experts, out_feat, r), **factory_kwargs)

        scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)
        self.register_buffer("lora_scaling", torch.tensor(scaling, dtype=torch.float32))
        # Python-float copy used by the (Round 2) fused MoE-LoRA kernel which
        # takes the scale as a plain float to avoid a host/device sync inside
        # the autograd.Function. Kept in lock-step with ``lora_scaling``.
        self._lora_scale_value: float = float(scaling)

        # Freeze base, then unfreeze lora_*. ``_is_lora_param_name`` looks
        # at canonical ``lora_A`` / ``lora_B`` segments so it works under
        # both pre-FSDP and FSDP-wrapped naming.
        for p in self.parameters():
            p.requires_grad = False
        for n, p in self.named_parameters():
            if _is_lora_param_name(n):
                p.requires_grad = True

        # Init: per-expert kaiming-uniform A, zero B → no-op vs base at init,
        # matching PEFT's ``init_lora_weights=True`` default. Skip on meta.
        if not any(p.is_meta for n, p in self.named_parameters() if _is_lora_param_name(n)):
            self.reset_lora_parameters()

    # ── PEFT-compatible accessors (same surface as LoraSharedExperts) ──────

    def _get_lora_container(self, role: str, param_name: str, adapter_name: str | None = None) -> _LoraParam3D:
        adapter = adapter_name or self.adapter_name
        spec: _LoraSpec = getattr(self, param_name)
        return getattr(spec, f"lora_{role}")[adapter]

    def get_lora_A_weight(self, param_name: str, adapter_name: str | None = None) -> torch.Tensor:
        """Active LoRA A weight for ``param_name``, shape ``[E, r, in_features]``."""
        return self._get_lora_container("A", param_name, adapter_name).weight

    def get_lora_B_weight(self, param_name: str, adapter_name: str | None = None) -> torch.Tensor:
        """Active LoRA B weight for ``param_name``, shape ``[E, out_features, r]``."""
        return self._get_lora_container("B", param_name, adapter_name).weight

    @torch.no_grad()
    def reset_lora_parameters(self, adapter_name: str | None = None, init_lora_weights: bool = True) -> None:
        """Per-expert kaiming-uniform A, zero B — idempotent.

        ``kaiming_uniform_`` is applied on each per-expert 2-D slice
        ``A[e]`` of shape ``[r, in_feat]`` so each expert sees the same
        textbook variance an ``nn.Linear`` would. (PEFT's standard 2-D
        path applies it once; we mirror that semantics per-expert.)

        DTensor handling (EP path): under EP the LoRA-A weight is a
        ``DTensor`` sharded along dim-0 (expert dim). DTensor indexing
        ``w[e]`` returns a *DTensor view* whose in-place ``kaiming_uniform_``
        does not write back to the underlying local storage (we observed
        the local shard staying all-zeros after the loop). We therefore
        operate on ``DTensor._local_tensor`` directly: each rank
        initialises its own ``[E_local, r, in_feat]`` slice in-place,
        which is correct because Mode 1 LoRA is EP-sharded along the
        expert dim (one local LoRA per local expert). ``zeros_`` on B is
        DTensor-safe regardless of layout.
        """
        if not init_lora_weights:
            return
        from torch.distributed.tensor import DTensor

        for pname in self._lora_specs:
            spec: _LoraSpec = getattr(self, pname)
            for ad, container in spec.lora_A.items():
                if adapter_name is not None and ad != adapter_name:
                    continue
                w = container.weight  # [E, r, in_feat]; DTensor under EP
                local_w = w._local_tensor if isinstance(w, DTensor) else w
                for e in range(local_w.shape[0]):
                    nn.init.kaiming_uniform_(local_w[e], a=math.sqrt(5))
            for ad, container in spec.lora_B.items():
                if adapter_name is not None and ad != adapter_name:
                    continue
                nn.init.zeros_(container.weight)

    # PEFT-style ``lora_A`` / ``lora_B`` accessors expected by some helpers
    # (see ``veomni.lora.weight_loading.init_lora_parameter``). Read-only views
    # of the per-target ModuleDicts; we expose the union of adapter names.
    @property
    def lora_A(self) -> dict[str, _LoraParam3D]:
        first_pname = next(iter(self._lora_specs))
        spec: _LoraSpec = getattr(self, first_pname)
        return dict(spec.lora_A)

    @property
    def lora_B(self) -> dict[str, _LoraParam3D]:
        first_pname = next(iter(self._lora_specs))
        spec: _LoraSpec = getattr(self, first_pname)
        return dict(spec.lora_B)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Mirrors LoraSharedExperts.forward: prefer the fused branch whenever
        # the active ``moe_implementation`` bound a Mode 1 LoRA-aware kernel
        # (currently only 'fused_triton'; Quack/NPU leave the pointer as
        # ``None`` so we transparently fall back). The bound kernel handles
        # the EP branch internally via ``preprocess`` / ``token_pre_all2all``
        # / ``EPMergedFc1IndependentLoRAGroupGemm`` / ``tokens_post_all2all``.
        from ..distributed.parallel_state import get_parallel_state
        from . import ops as _lora_ops

        if _lora_ops._fused_independent_lora_moe_forward is not None:
            return self._fused_forward(
                _lora_ops._fused_independent_lora_moe_forward, hidden_states, top_k_index, top_k_weights
            )
        if get_parallel_state().ep_enabled:
            raise RuntimeError(
                "LoraIndependentExperts: eager forward does not support expert parallelism (EP). "
                "Set ops_implementation.moe_implementation='fused_triton' to use the EP-aware "
                "fused LoRA path, or disable EP."
            )
        return self._eager_forward(hidden_states, top_k_index, top_k_weights)

    def _fused_forward(
        self,
        fused_kernel,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch into the bound fused independent-LoRA MoE kernel.

        ``fused_kernel`` is the kernel pointer captured by ``forward`` so we
        don't re-read the module attribute twice (cheap optimisation, also
        keeps this method side-effect free for testing). Passes both halves
        of the seed-style gate_up LoRA pair as separate ``(A, B, scale)``
        triples — the kernel keeps them split end-to-end.
        """
        return fused_kernel(
            num_experts=self.num_experts,
            routing_weights=top_k_weights.to(hidden_states.dtype),
            selected_experts=top_k_index,
            hidden_states=hidden_states,
            fc1_1_2_weight=self.gate_up_proj.base_layer.weight,
            fc2_weight=self.down_proj.base_layer.weight,
            lora_a_gate=self.get_lora_A_weight("gate_proj"),
            lora_b_gate=self.get_lora_B_weight("gate_proj"),
            lora_a_up=self.get_lora_A_weight("up_proj"),
            lora_b_up=self.get_lora_B_weight("up_proj"),
            lora_a_down=self.get_lora_A_weight("down_proj"),
            lora_b_down=self.get_lora_B_weight("down_proj"),
            lora_scale_gate=self._lora_scale_value,
            lora_scale_up=self._lora_scale_value,
            lora_scale_down=self._lora_scale_value,
        )

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.lora_scaling.to(hidden_states.dtype)
        a_gate = self.get_lora_A_weight("gate_proj")  # [E, r, H]
        b_gate = self.get_lora_B_weight("gate_proj")  # [E, I, r]
        a_up = self.get_lora_A_weight("up_proj")  # [E, r, H]
        b_up = self.get_lora_B_weight("up_proj")  # [E, I, r]
        a_dn = self.get_lora_A_weight("down_proj")  # [E, r, I]
        b_dn = self.get_lora_B_weight("down_proj")  # [E, H, r]

        gate_up_w = self.gate_up_proj.base_layer.weight
        down_w = self.down_proj.base_layer.weight

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # Per-expert LoRA on gate and up halves — independent rank-r
            # adapters, both computed inside the per-expert loop. Cat the
            # per-half deltas into [n_e, 2I] so the add lines up with the
            # merged ``gate_up_proj`` output before chunk + SiLU.
            gate_delta = F.linear(F.linear(current_state, a_gate[expert_idx]), b_gate[expert_idx]) * scale  # [n_e, I]
            up_delta = F.linear(F.linear(current_state, a_up[expert_idx]), b_up[expert_idx]) * scale  # [n_e, I]
            gate_up = F.linear(current_state, gate_up_w[expert_idx]) + torch.cat([gate_delta, up_delta], dim=-1)
            gate, up = gate_up.chunk(2, dim=-1)
            mid = self.act_fn(gate) * up

            lora_x_down = F.linear(F.linear(mid, a_dn[expert_idx]), b_dn[expert_idx]) * scale
            current_hidden_states = F.linear(mid, down_w[expert_idx]) + lora_x_down
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.lora_alpha}, num_experts={self.num_experts}, mode=independent"


def _apply_moe_lora_with(
    wrapper_cls,
    model: nn.Module,
    target_parameter_patterns: list[str],
    r: int,
    lora_alpha: int,
    use_rslora: bool,
    adapter_name: str,
    fail_on_no_match: bool,
    freeze_base_model: bool,
) -> list[str]:
    """Shared search-and-replace machinery for both ``apply_*_moe_lora`` flavours.

    Walks ``model.named_modules()``, finds each module owning at least one 3-D
    ``nn.Parameter`` matching ``target_parameter_patterns`` (PEFT-style globs;
    leading ``base_model.model.`` prefix stripped before matching), and replaces
    it in its parent with a fresh ``wrapper_cls`` instance.
    """
    matches = _find_target_parameter_modules(model, target_parameter_patterns)
    if not matches:
        if fail_on_no_match:
            raise ValueError(
                f"No 3D parameters in the model matched target_parameter_patterns="
                f"{target_parameter_patterns!r}. Verify the patterns include the "
                f"leading 'model.' prefix as in 'model.layers.*.mlp.experts.gate_up_proj'."
            )
        return []

    if freeze_base_model:
        for p in model.parameters():
            p.requires_grad = False

    wrapped_fqns: list[str] = []
    for parent, parent_fqn, attr_name, base_module in matches:
        wrapper = wrapper_cls(
            base_layer=base_module,
            r=r,
            lora_alpha=lora_alpha,
            use_rslora=use_rslora,
            adapter_name=adapter_name,
        )
        setattr(parent, attr_name, wrapper)
        wrapped_fqns.append(f"{parent_fqn}.{attr_name}" if parent_fqn else attr_name)

    # Stash the patterns on the model so the save helper can serialise them
    # without re-discovering. Use a non-Parameter attribute prefixed with
    # ``_veomni_`` so it doesn't show up in state_dict / FSDP traversal.
    model._veomni_moe_lora_patterns = list(target_parameter_patterns)
    return sorted(wrapped_fqns)


def apply_shared_moe_lora(
    model: nn.Module,
    target_parameter_patterns: list[str],
    r: int,
    lora_alpha: int,
    use_rslora: bool = False,
    adapter_name: str = "default",
    fail_on_no_match: bool = True,
    freeze_base_model: bool = True,
) -> list[str]:
    """In-place wrap experts modules in ``model`` with :class:`LoraSharedExperts` (Mode 2).

    Args:
        target_parameter_patterns: e.g. ``["model.layers.*.mlp.experts.gate_up_proj",
            "model.layers.*.mlp.experts.down_proj"]``. Multiple patterns that
            point at the same experts module are deduplicated — each module is
            wrapped at most once. The pattern list is stashed on the model
            (as ``model._veomni_moe_lora_patterns``) so the save helper can
            reproduce it in the sidecar without re-scanning.
        fail_on_no_match: raise if zero modules matched (default). Set ``False``
            for "best-effort" wiring.
        freeze_base_model: when True (default), set ``requires_grad=False`` on
            every parameter in ``model`` *before* wrapping; the wrapper then
            unfreezes only its own ``<spec>.lora_A`` / ``<spec>.lora_B``. This mirrors
            PEFT's ``get_peft_model`` semantics so the function is safe to call
            standalone. Pass ``False`` if PEFT (or the trainer) has already
            arranged ``requires_grad`` and you want to preserve other trainable
            adapters.

    Returns:
        Sorted list of wrapped module FQNs (post-wrap, in the original model's
        namespace; PEFT prefix preserved if present).
    """
    return _apply_moe_lora_with(
        LoraSharedExperts,
        model,
        target_parameter_patterns,
        r,
        lora_alpha,
        use_rslora,
        adapter_name,
        fail_on_no_match,
        freeze_base_model,
    )


def apply_independent_moe_lora(
    model: nn.Module,
    target_parameter_patterns: list[str],
    r: int,
    lora_alpha: int,
    use_rslora: bool = False,
    adapter_name: str = "default",
    fail_on_no_match: bool = True,
    freeze_base_model: bool = True,
) -> list[str]:
    """In-place wrap experts modules in ``model`` with :class:`LoraIndependentExperts` (Mode 1, default).

    Same surface as :func:`apply_shared_moe_lora` modulo the wrapper class — see
    that function's docstring for argument semantics. The replaced experts
    modules now own per-expert 3-D LoRA tensors instead of a single shared 2-D
    pair; behaviour is otherwise identical from the trainer / save / load
    perspective (the MoE mode is recorded in the ``adapter_config.json``
    ``veomni_lora`` block, the native key remap is shared, and the PEFT-format
    round-trip works on the same key shape).
    """
    return _apply_moe_lora_with(
        LoraIndependentExperts,
        model,
        target_parameter_patterns,
        r,
        lora_alpha,
        use_rslora,
        adapter_name,
        fail_on_no_match,
        freeze_base_model,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Module-type predicates
# ──────────────────────────────────────────────────────────────────────────────


def is_lora_shared_experts(module: nn.Module) -> bool:
    """Type-stable check for the Mode 2 (shared) wrapper."""
    return isinstance(module, LoraSharedExperts)


def is_lora_independent_experts(module: nn.Module) -> bool:
    """Type-stable check for the Mode 1 (independent per-expert) wrapper."""
    return isinstance(module, LoraIndependentExperts)


def is_lora_moe_experts(module: nn.Module) -> bool:
    """True for either MoE-LoRA wrapper flavour (Mode 1 or Mode 2)."""
    return isinstance(module, (LoraSharedExperts, LoraIndependentExperts))


def has_lora_shared_experts(model: nn.Module) -> bool:
    """True iff ``model`` contains at least one :class:`LoraSharedExperts`."""
    return any(is_lora_shared_experts(m) for _, m in model.named_modules())


def has_lora_independent_experts(model: nn.Module) -> bool:
    """True iff ``model`` contains at least one :class:`LoraIndependentExperts`."""
    return any(is_lora_independent_experts(m) for _, m in model.named_modules())


def has_lora_moe_experts(model: nn.Module) -> bool:
    """True iff ``model`` contains any MoE-LoRA wrapper (Mode 1 or Mode 2)."""
    return any(is_lora_moe_experts(m) for _, m in model.named_modules())


def iter_moe_lora_parameters(model: nn.Module):
    """Yield ``(fqn, parameter)`` pairs for every MoE-LoRA tunable parameter.

    Walks every :class:`LoraSharedExperts` / :class:`LoraIndependentExperts`
    and yields each ``<spec>.lora_A.<adapter>.weight`` /
    ``<spec>.lora_B.<adapter>.weight`` weight (2-D for shared, 3-D for
    independent), with full FQN suitable for ``state_dict`` lookup.
    """
    for fqn, module in model.named_modules():
        if not is_lora_moe_experts(module):
            continue
        prefix = f"{fqn}." if fqn else ""
        for n, p in module.named_parameters(recurse=True):
            if _is_lora_param_name(n):
                yield prefix + n, p


# ──────────────────────────────────────────────────────────────────────────────
# Native integration surface (VeOmniLoraConfig-driven)
#
# The wrappers above are config-agnostic (they take plain r / lora_alpha / mode
# args). These helpers bridge them to the native :class:`VeOmniLoraConfig` so
# :class:`veomni.lora.model.VeOmniLoraModel` installs and introspects MoE-LoRA
# without a ``peft.LoraConfig`` or the retired ``veomni_moe_lora.json`` sidecar.
# ──────────────────────────────────────────────────────────────────────────────


def inject_moe_lora(
    model: nn.Module,
    config: VeOmniLoraConfig,
    adapter_name: str,
) -> list[str]:
    """Replace matched MoE experts modules in ``model`` with LoRA wrappers.

    ``model`` is the inner base model (``VeOmniLoraModel.base_model.model``).
    Dispatches on ``config.moe_mode`` (``"shared"`` -> :class:`LoraSharedExperts`,
    else :class:`LoraIndependentExperts`). ``freeze_base_model=False`` because
    :func:`veomni.lora.mapping.mark_lora_trainable` runs the freeze/unfreeze pass
    after all injection (dense + MoE) completes.

    Returns the list of wrapped experts-module FQNs (relative to ``model``).
    """
    if not config.target_parameters:
        return []

    mode = config.moe_mode or "independent"
    apply_fn = apply_shared_moe_lora if mode == "shared" else apply_independent_moe_lora
    wrapped = apply_fn(
        model,
        target_parameter_patterns=list(config.target_parameters),
        r=config.r,
        lora_alpha=config.lora_alpha,
        use_rslora=config.use_rslora,
        adapter_name=adapter_name,
        fail_on_no_match=True,
        freeze_base_model=False,
    )
    logger.info_rank0(
        f"Injected {mode} MoE-LoRA into {len(wrapped)} experts module(s) (showing first 3): {wrapped[:3]}"
    )
    return wrapped


def get_moe_wrapper_metadata(model: nn.Module) -> dict | None:
    """Introspect installed MoE-LoRA wrappers into a metadata dict, or ``None``.

    Unlike the legacy ``veomni_moe_lora.json`` sidecar, VeOmni-native adapters
    persist MoE settings inside ``adapter_config.json`` (``target_parameters`` +
    the ``veomni_lora.moe_mode`` block), so this is only used for introspection
    / assertions, not for the save/resume contract. Returns ``None`` when no
    wrapper is present; raises if shared and independent wrappers are mixed
    (unsupported in a single model).
    """
    wrappers = [(fqn, m) for fqn, m in model.named_modules() if is_lora_moe_experts(m)]
    if not wrappers:
        return None
    shared = sum(1 for _, m in wrappers if is_lora_shared_experts(m))
    indep = sum(1 for _, m in wrappers if is_lora_independent_experts(m))
    if shared and indep:
        raise ValueError(
            f"Model mixes shared ({shared}) and independent ({indep}) MoE-LoRA wrappers; "
            "a single VeOmniLoraModel supports only one MoE mode."
        )
    sample = wrappers[0][1]
    return {
        "moe_mode": "shared" if shared else "independent",
        "r": sample.r,
        "lora_alpha": sample.lora_alpha,
        "use_rslora": sample.use_rslora,
        "wrapped_fqns": [fqn for fqn, _ in wrappers],
    }
