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
  kernel in Round 2 and (b) keep eager / fused on identical math + key
  conventions.
* :class:`LoraSharedExperts` — **Mode 2**. A single LoRA pair (2D
  ``[r, H]`` and ``[O, r]`` tensors) shared across all experts of the
  layer. PEFT does not natively support this — a 2D parameter target
  isn't expressible via ``target_parameters`` (which assumes a leading
  expert dim).

Forward strategy
----------------
For one expert ``e`` with base weight ``W_e`` of shape ``[O, H]``::

    aug_out_e(x) = (W_e + B_e @ A_e * s) @ x
                 = W_e @ x + B_e @ (A_e @ x) * s

The right-hand form avoids materialising an ``[E, O, H]`` delta tensor
(which is what PEFT's ``ParamWrapper`` does for Mode 1 and what the docs
warn about as runtime overhead). For Mode 2 the LoRA pair is shared, so
the gate/up LoRA contribution depends only on the *input* token and can
be computed once per token outside the per-expert dispatch loop; for
Mode 1 every expert's LoRA is independent so it must be computed inside
the loop. The down LoRA always lives inside the loop because its input
is the per-expert intermediate activation ``silu(gate) * up``.

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

Both wrappers preserve the original module under ``base_layer`` and the
original 3D parameters move to ``mlp.experts.base_layer.<param>``; the EP
parallel plan must be aware of this on resume (handled in a later phase
when the trainer wires through to ``build_parallelize_model``).

A fused-Triton path is bound for Mode 2 in
``veomni/ops/kernels/moe/lora_group_gemm.py``; the Mode 1 fused path lands
in Round 2 and falls back to eager until then.

PEFT save/load compatibility
----------------------------
Both wrappers store LoRA weights as ``self.lora_A_<param>`` /
``self.lora_B_<param>``, ``nn.ModuleDict`` keyed by adapter name. Mode 2
uses ``nn.Linear(in, r, bias=False)`` (2D ``.weight``); Mode 1 uses a tiny
:class:`_LoraParam3D` container exposing a 3D ``.weight``. Either way the
state-dict key shape is ``lora_A_<param>.<adapter>.weight`` and PEFT's
``get_peft_model_state_dict`` / ``set_peft_model_state_dict`` round-trip
unchanged.

VeOmni's ``_remap_adapter_key`` (used by FSDP1 / rank-0 broadcast loaders)
recognises the ``lora_A_<param>`` / ``lora_B_<param>`` attribute names via
a ``startswith`` extension in ``veomni/utils/lora_utils.py``.

A sidecar file ``veomni_moe_lora.json`` is written next to PEFT's
``adapter_config.json`` to record (a) which experts modules were wrapped,
(b) the wrapper kind (``"shared"`` or ``"independent"``) and (c) rank /
alpha / use_rslora / adapter_name. The resume path
(``PeftModel.from_pretrained``) reads the sidecar and re-installs the
matching wrappers *before* PEFT loads the state dict; without the
wrappers in place, the saved ``lora_A_<param>.<adapter>.weight`` keys
would have no destination and PEFT silently drops them.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


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
            The wrapper takes ownership: its parameters are frozen and forward
            never calls ``base_layer.forward()``.
        r: LoRA rank.
        lora_alpha: LoRA alpha. Scaling is ``alpha / r`` (or ``alpha / sqrt(r)``
            when ``use_rslora=True``), matching PEFT.
        use_rslora: Use rank-stabilised LoRA scaling.
        adapter_name: PEFT-style adapter name. Currently a single adapter is
            supported; the name is stored for save/load helpers.

    LoRA parameters live in PEFT-style ``ModuleDict`` containers, one per
    target parameter, each keyed by adapter name and holding an
    ``nn.Linear(in_features, r)`` (for A) or ``nn.Linear(r, out_features)``
    (for B): ``lora_A_gate_up_proj``, ``lora_B_gate_up_proj``,
    ``lora_A_down_proj``, ``lora_B_down_proj``.

    The resulting FQNs (``...experts.lora_A_<param>.<adapter>.weight``) pass
    PEFT's ``get_peft_model_state_dict`` filter and round-trip through
    ``set_peft_model_state_dict`` unchanged, so the standard
    ``model.save_pretrained`` / ``PeftModel.from_pretrained`` path works
    once the wrappers are installed.
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

        self.base_layer = base_layer
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

        # Validate the experts module is a fused gate_up_proj + down_proj
        # layout (the only layout VeOmni MoE-LoRA supports — see file
        # docstring "Expected experts module layout").
        _validate_fused_layout(base_layer)
        # Shapes for each LoRA pair: A is [r, in_features], B is [out_features, r].
        # Naming follows F.linear semantics: F.linear(x, W) computes x @ W.T,
        # so for base weight W_e of shape [O, H], the pair (A: [r, H], B: [O, r])
        # gives delta_W = B @ A of shape [O, H].
        self._lora_specs = {
            "gate_up_proj": (self.hidden_dim, 2 * self.intermediate_dim),
            "down_proj": (self.intermediate_dim, self.hidden_dim),
        }

        # Inherit dtype/device from the base module's first parameter so the
        # new Linears land on the same device (typically meta or cuda).
        ref = next(base_layer.parameters())
        factory_kwargs = {"dtype": ref.dtype, "device": ref.device}

        for pname, (in_feat, out_feat) in self._lora_specs.items():
            # Use ModuleDict[adapter -> Linear(bias=False)] to match PEFT's
            # save/load conventions exactly. The Linear is never invoked via
            # __call__ in our forward — we read .weight directly — but the
            # nn.Linear container is what makes get_peft_model_state_dict /
            # set_peft_model_state_dict round-trip our keys.
            a_dict = nn.ModuleDict({adapter_name: nn.Linear(in_feat, r, bias=False, **factory_kwargs)})
            b_dict = nn.ModuleDict({adapter_name: nn.Linear(r, out_feat, bias=False, **factory_kwargs)})
            self.add_module(f"lora_A_{pname}", a_dict)
            self.add_module(f"lora_B_{pname}", b_dict)

        scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)
        self.register_buffer("lora_scaling", torch.tensor(scaling, dtype=torch.float32))
        # Python-float copy of the scaling factor — used by the fused MoE-LoRA
        # forward kernel (``veomni.ops.kernels.moe.lora_group_gemm``) which takes
        # the scale as a plain float to avoid a host/device sync inside the
        # autograd.Function. Kept in lock-step with ``lora_scaling``.
        self._lora_scale_value: float = float(scaling)

        # Freeze base; only our lora_* are trainable.
        for p in self.base_layer.parameters():
            p.requires_grad = False
        for n, p in self.named_parameters():
            if n.startswith("base_layer."):
                continue
            p.requires_grad = True

        # Init: kaiming-uniform A, zero B → no-op vs base at init (PEFT default
        # for ``init_lora_weights=True``). Skip when params are on meta device
        # (post-meta init weight loading will call ``reset_lora_parameters``).
        if not any(p.is_meta for n, p in self.named_parameters() if not n.startswith("base_layer.")):
            self.reset_lora_parameters()

    # ------------------------------------------------------------------
    # PEFT-compatible accessors.
    # ------------------------------------------------------------------

    def _get_lora_linear(self, role: str, param_name: str, adapter_name: str | None = None) -> nn.Linear:
        adapter = adapter_name or self.adapter_name
        return getattr(self, f"lora_{role}_{param_name}")[adapter]

    def get_lora_A_weight(self, param_name: str, adapter_name: str | None = None) -> torch.Tensor:
        """Active LoRA A weight for ``param_name``, shape ``[r, in_features]``."""
        return self._get_lora_linear("A", param_name, adapter_name).weight

    def get_lora_B_weight(self, param_name: str, adapter_name: str | None = None) -> torch.Tensor:
        """Active LoRA B weight for ``param_name``, shape ``[out_features, r]``."""
        return self._get_lora_linear("B", param_name, adapter_name).weight

    @torch.no_grad()
    def reset_lora_parameters(self, adapter_name: str | None = None, init_lora_weights: bool = True) -> None:
        """Initialise LoRA A (kaiming-uniform) and B (zeros). Idempotent.

        Signature matches PEFT's ``LoraLayer.reset_lora_parameters`` so VeOmni's
        ``_init_lora_parameter`` (in ``utils/lora_utils.py``) can call it
        without special-casing — passing ``adapter_name=None`` resets every
        adapter, mirroring PEFT's behaviour when no adapter is selected.
        """
        if not init_lora_weights:
            return
        for pname in self._lora_specs:
            a_dict = getattr(self, f"lora_A_{pname}")
            b_dict = getattr(self, f"lora_B_{pname}")
            for ad, lin in a_dict.items():
                if adapter_name is not None and ad != adapter_name:
                    continue
                nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            for ad, lin in b_dict.items():
                if adapter_name is not None and ad != adapter_name:
                    continue
                nn.init.zeros_(lin.weight)

    # PEFT-style ``lora_A`` / ``lora_B`` accessors expected by some helpers
    # (e.g. VeOmni's ``_init_lora_parameter`` introspection of adapter names).
    # These are read-only views into our per-target ModuleDicts; we expose the
    # union of adapter names across all target parameters.
    @property
    def lora_A(self) -> dict[str, nn.Linear]:
        # Return one Linear per adapter (taken from the first target param) so
        # callers iterating ``lora_A.keys()`` see the right adapter list.
        first_pname = next(iter(self._lora_specs))
        return dict(getattr(self, f"lora_A_{first_pname}"))

    @property
    def lora_B(self) -> dict[str, nn.Linear]:
        first_pname = next(iter(self._lora_specs))
        return dict(getattr(self, f"lora_B_{first_pname}"))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Fused-kernel path: available when (a) the user opted into a
        # non-eager ``moe_implementation`` whose patch function bound a
        # LoRA-aware kernel (currently 'fused_triton'; Quack/NPU leave
        # ``_fused_lora_moe_forward = None`` so we transparently fall back
        # to eager), and (b) we are not under expert parallelism (EP
        # support comes in Phase 5; the kernel itself raises
        # NotImplementedError on EP, but we short-circuit here so the
        # eager path runs cleanly). Mirrors the
        # ``veomni_moe_experts_forward.use_non_eager_impl`` guard inside
        # the generated MoE experts forward.
        from ..distributed.parallel_state import get_parallel_state
        from ..ops.kernels import moe as _moe_ops

        use_fused = _moe_ops._fused_lora_moe_forward is not None and not get_parallel_state().ep_enabled
        if use_fused:
            return self._fused_forward(_moe_ops._fused_lora_moe_forward, hidden_states, top_k_index, top_k_weights)
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
        keeps this method side-effect free for testing).
        """
        base = self.base_layer
        return fused_kernel(
            num_experts=base.num_experts,
            routing_weights=top_k_weights.to(hidden_states.dtype),
            selected_experts=top_k_index,
            hidden_states=hidden_states,
            fc1_1_2_weight=base.gate_up_proj,
            fc2_weight=base.down_proj,
            lora_a_gate_up=self.get_lora_A_weight("gate_up_proj"),
            lora_b_gate_up=self.get_lora_B_weight("gate_up_proj"),
            lora_a_down=self.get_lora_A_weight("down_proj"),
            lora_b_down=self.get_lora_B_weight("down_proj"),
            lora_scale_gate_up=self._lora_scale_value,
            lora_scale_down=self._lora_scale_value,
        )

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_layer
        scale = self.lora_scaling.to(hidden_states.dtype)
        a_gu = self.get_lora_A_weight("gate_up_proj")
        b_gu = self.get_lora_B_weight("gate_up_proj")
        a_dn = self.get_lora_A_weight("down_proj")
        b_dn = self.get_lora_B_weight("down_proj")

        # Shared LoRA delta on gate_up depends only on x → compute once.
        # F.linear(x, A): [N, H] @ [H, r] -> [N, r]
        # F.linear(., B): [N, r] @ [r, 2I] -> [N, 2I]
        lora_x_gate_up = F.linear(F.linear(hidden_states, a_gu), b_gu) * scale

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=base.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == base.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_up = F.linear(current_state, base.gate_up_proj[expert_idx]) + lora_x_gate_up[token_idx]
            gate, up = gate_up.chunk(2, dim=-1)
            mid = self.act_fn(gate) * up

            # down LoRA depends on the per-expert intermediate, so compute inside the loop.
            lora_x_down = F.linear(F.linear(mid, a_dn), b_dn) * scale
            current_hidden_states = F.linear(mid, base.down_proj[expert_idx]) + lora_x_down
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
    ``lora_A_<param>.<adapter>.weight`` round-trips through PEFT's
    ``get_peft_model_state_dict`` / ``set_peft_model_state_dict`` exactly the
    same way it does for the 2-D :class:`LoraSharedExperts` case.

    Why a wrapper rather than a bare ``nn.ParameterDict``?
        ``ParameterDict[adapter -> Parameter]`` would produce keys like
        ``lora_A_<param>.<adapter>`` (no trailing ``.weight``), which works in
        principle but breaks symmetry with both PEFT's standard ``LoraLayer``
        storage and our shared wrapper. Keeping ``.weight`` everywhere lets
        every consumer (``_remap_adapter_key``, FSDP load helpers, the
        round-trip test) treat both modes interchangeably.
    """

    def __init__(self, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype, device=device))


class LoraIndependentExperts(nn.Module):
    """Wrap a MoE experts module to add an independent LoRA pair per expert.

    Args:
        base_layer: The original experts module (e.g. ``Qwen3MoeExperts``).
            The wrapper takes ownership: its parameters are frozen and forward
            never calls ``base_layer.forward()``.
        r: LoRA rank.
        lora_alpha: LoRA alpha. Scaling is ``alpha / r`` (or ``alpha / sqrt(r)``
            when ``use_rslora=True``), matching PEFT.
        use_rslora: Use rank-stabilised LoRA scaling.
        adapter_name: PEFT-style adapter name. Currently a single adapter is
            supported; the name is stored for save/load helpers.

    LoRA tensors are 3-D with the leading dim equal to ``num_experts``::

        lora_A_gate_up_proj.<adapter>.weight   # [E, r, H]
        lora_B_gate_up_proj.<adapter>.weight   # [E, 2I, r]
        lora_A_down_proj.<adapter>.weight      # [E, r, I]
        lora_B_down_proj.<adapter>.weight      # [E, H, r]

    Forward semantics for token ``t`` routed to expert ``e``::

        delta_W_e = B_e @ A_e * (alpha / r)        # [O, H], never materialised
        out_t = base_e @ x_t + B_e @ (A_e @ x_t) * scale

    Equivalent in math to PEFT 0.19's ``target_parameters`` 3-D path; we own
    it so the wrapper's ``forward`` can dispatch into a fused MoE-LoRA triton
    kernel (Round 2). Until that lands, only the eager forward is exercised.
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

        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora
        self.adapter_name = adapter_name

        self.num_experts = base_layer.num_experts
        self.hidden_dim = base_layer.hidden_dim
        self.intermediate_dim = base_layer.intermediate_dim
        self.act_fn = base_layer.act_fn

        # Validate the experts module is a fused gate_up_proj + down_proj
        # layout (the only layout VeOmni MoE-LoRA supports — see file
        # docstring "Expected experts module layout").
        _validate_fused_layout(base_layer)
        # Per-target (in_features, out_features) — same as LoraSharedExperts;
        # 3-D parameter shapes derived below add a leading expert dim.
        self._lora_specs = {
            "gate_up_proj": (self.hidden_dim, 2 * self.intermediate_dim),
            "down_proj": (self.intermediate_dim, self.hidden_dim),
        }

        ref = next(base_layer.parameters())
        factory_kwargs = {"dtype": ref.dtype, "device": ref.device}

        # Per-target ModuleDict[adapter -> _LoraParam3D]. Leading expert dim
        # makes every LoRA slice independent; downstream forward indexes by
        # expert_idx the same way it indexes ``base.gate_up_proj[expert_idx]``.
        for pname, (in_feat, out_feat) in self._lora_specs.items():
            a_dict = nn.ModuleDict({adapter_name: _LoraParam3D((self.num_experts, r, in_feat), **factory_kwargs)})
            b_dict = nn.ModuleDict({adapter_name: _LoraParam3D((self.num_experts, out_feat, r), **factory_kwargs)})
            self.add_module(f"lora_A_{pname}", a_dict)
            self.add_module(f"lora_B_{pname}", b_dict)

        scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)
        self.register_buffer("lora_scaling", torch.tensor(scaling, dtype=torch.float32))
        # Python-float copy used by the (Round 2) fused MoE-LoRA kernel which
        # takes the scale as a plain float to avoid a host/device sync inside
        # the autograd.Function. Kept in lock-step with ``lora_scaling``.
        self._lora_scale_value: float = float(scaling)

        # Freeze base; only our lora_* are trainable.
        for p in self.base_layer.parameters():
            p.requires_grad = False
        for n, p in self.named_parameters():
            if n.startswith("base_layer."):
                continue
            p.requires_grad = True

        # Init: per-expert kaiming-uniform A, zero B → no-op vs base at init,
        # matching PEFT's ``init_lora_weights=True`` default. Skip on meta.
        if not any(p.is_meta for n, p in self.named_parameters() if not n.startswith("base_layer.")):
            self.reset_lora_parameters()

    # ── PEFT-compatible accessors (same surface as LoraSharedExperts) ──────

    def _get_lora_container(self, role: str, param_name: str, adapter_name: str | None = None) -> _LoraParam3D:
        adapter = adapter_name or self.adapter_name
        return getattr(self, f"lora_{role}_{param_name}")[adapter]

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
        """
        if not init_lora_weights:
            return
        for pname in self._lora_specs:
            a_dict = getattr(self, f"lora_A_{pname}")
            b_dict = getattr(self, f"lora_B_{pname}")
            for ad, container in a_dict.items():
                if adapter_name is not None and ad != adapter_name:
                    continue
                w = container.weight  # [E, r, in_feat]
                for e in range(w.shape[0]):
                    nn.init.kaiming_uniform_(w[e], a=math.sqrt(5))
            for ad, container in b_dict.items():
                if adapter_name is not None and ad != adapter_name:
                    continue
                nn.init.zeros_(container.weight)

    # PEFT-style ``lora_A`` / ``lora_B`` accessors expected by some helpers
    # (see ``veomni.utils.lora_utils._init_lora_parameter``). Read-only views
    # of the per-target ModuleDicts; we expose the union of adapter names.
    @property
    def lora_A(self) -> dict[str, _LoraParam3D]:
        first_pname = next(iter(self._lora_specs))
        return dict(getattr(self, f"lora_A_{first_pname}"))

    @property
    def lora_B(self) -> dict[str, _LoraParam3D]:
        first_pname = next(iter(self._lora_specs))
        return dict(getattr(self, f"lora_B_{first_pname}"))

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Mirrors LoraSharedExperts.forward: enable the fused path when (a)
        # the active ``moe_implementation`` bound a Mode 1 LoRA-aware kernel
        # (currently only 'fused_triton'; Quack/NPU leave the pointer as
        # ``None`` so we transparently fall back), and (b) we are not under
        # expert parallelism (EP support comes in Phase 5).
        from ..distributed.parallel_state import get_parallel_state
        from ..ops.kernels import moe as _moe_ops

        use_fused = _moe_ops._fused_independent_lora_moe_forward is not None and not get_parallel_state().ep_enabled
        if use_fused:
            return self._fused_forward(
                _moe_ops._fused_independent_lora_moe_forward, hidden_states, top_k_index, top_k_weights
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
        keeps this method side-effect free for testing).
        """
        base = self.base_layer
        return fused_kernel(
            num_experts=base.num_experts,
            routing_weights=top_k_weights.to(hidden_states.dtype),
            selected_experts=top_k_index,
            hidden_states=hidden_states,
            fc1_1_2_weight=base.gate_up_proj,
            fc2_weight=base.down_proj,
            lora_a_gate_up=self.get_lora_A_weight("gate_up_proj"),
            lora_b_gate_up=self.get_lora_B_weight("gate_up_proj"),
            lora_a_down=self.get_lora_A_weight("down_proj"),
            lora_b_down=self.get_lora_B_weight("down_proj"),
            lora_scale_gate_up=self._lora_scale_value,
            lora_scale_down=self._lora_scale_value,
        )

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_layer
        scale = self.lora_scaling.to(hidden_states.dtype)
        a_gu = self.get_lora_A_weight("gate_up_proj")  # [E, r, H]
        b_gu = self.get_lora_B_weight("gate_up_proj")  # [E, 2I, r]
        a_dn = self.get_lora_A_weight("down_proj")  # [E, r, I]
        b_dn = self.get_lora_B_weight("down_proj")  # [E, H, r]

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=base.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == base.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # Per-expert LoRA on gate_up — independent, computed inside the loop.
            gate_up_lora = F.linear(F.linear(current_state, a_gu[expert_idx]), b_gu[expert_idx]) * scale
            gate_up = F.linear(current_state, base.gate_up_proj[expert_idx]) + gate_up_lora
            gate, up = gate_up.chunk(2, dim=-1)
            mid = self.act_fn(gate) * up

            lora_x_down = F.linear(F.linear(mid, a_dn[expert_idx]), b_dn[expert_idx]) * scale
            current_hidden_states = F.linear(mid, base.down_proj[expert_idx]) + lora_x_down
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
            unfreezes only its own ``lora_A_*`` / ``lora_B_*``. This mirrors
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
    perspective (sidecar carries the kind, ``_remap_adapter_key`` is shared,
    PEFT round-trip works on the same key shape).
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
    and yields each ``lora_A_*`` / ``lora_B_*`` weight (2-D for shared, 3-D
    for independent), with full FQN suitable for ``state_dict`` lookup.
    """
    for fqn, module in model.named_modules():
        if not is_lora_moe_experts(module):
            continue
        prefix = f"{fqn}." if fqn else ""
        for n, p in module.named_parameters(recurse=True):
            if n.startswith("base_layer."):
                continue
            if n.startswith("lora_A_") or n.startswith("lora_B_"):
                yield prefix + n, p


# ──────────────────────────────────────────────────────────────────────────────
# Sidecar metadata + I/O — kept in this module so save/load callers don't need
# to know the JSON schema.
#
# Schema v2 (current)::
#
#     {
#       "version": 2,
#       "mode": "shared" | "independent",   # which wrapper class to rebuild
#       "r": <int>, "lora_alpha": <int>, "use_rslora": <bool>, "adapter_name": <str>,
#       "target_parameter_patterns": [<glob>, ...],
#       "wrapped_fqns": [<fqn>, ...]
#     }
# ──────────────────────────────────────────────────────────────────────────────

MOE_LORA_SIDECAR_NAME = "veomni_moe_lora.json"
_MOE_LORA_SIDECAR_VERSION = 2


def get_moe_lora_metadata(model: nn.Module) -> dict[str, Any] | None:
    """Return a sidecar dict describing the installed MoE-LoRA wrappers, or ``None``.

    Detects the wrapper *kind* from the live module instances (single-mode
    only — mixing shared and independent wrappers in one model is not supported
    in Round 1 and raises ``ValueError`` here so the user catches it before save).
    Recovers the original ``target_parameter_patterns`` from the stash placed by
    :func:`apply_shared_moe_lora` / :func:`apply_independent_moe_lora`, falling
    back to per-FQN expansion when the stash is missing (e.g. checkpoint that
    didn't round-trip through ``apply_*_moe_lora``).
    """
    wrappers = [(fqn, m) for fqn, m in model.named_modules() if is_lora_moe_experts(m)]
    if not wrappers:
        return None

    shared_count = sum(1 for _, m in wrappers if is_lora_shared_experts(m))
    indep_count = sum(1 for _, m in wrappers if is_lora_independent_experts(m))
    if shared_count and indep_count:
        raise ValueError(
            f"Model has a mix of shared ({shared_count}) and independent ({indep_count}) "
            "MoE-LoRA wrappers. The sidecar schema only describes one kind per save; "
            "mixed installs are not supported."
        )
    mode = "shared" if shared_count else "independent"

    sample = wrappers[0][1]
    target_parameter_patterns = getattr(model, "_veomni_moe_lora_patterns", None)
    if target_parameter_patterns is None:
        # Fallback: synthesise patterns from the wrapped FQNs and the layout's
        # target list. Less compact than user-supplied globs but still unique.
        target_parameter_patterns = []
        for fqn, m in wrappers:
            for pname in m._lora_specs:
                target_parameter_patterns.append(f"{fqn}.{pname}")
    return {
        "version": _MOE_LORA_SIDECAR_VERSION,
        "mode": mode,
        "r": sample.r,
        "lora_alpha": sample.lora_alpha,
        "use_rslora": sample.use_rslora,
        "adapter_name": sample.adapter_name,
        "target_parameter_patterns": list(target_parameter_patterns),
        "wrapped_fqns": [fqn for fqn, _ in wrappers],
    }


def write_moe_lora_sidecar(model: nn.Module, save_path: str) -> str | None:
    """If ``model`` has any MoE-LoRA wrappers, write the sidecar to disk.

    Returns the written path, or ``None`` if no wrappers were found.
    """
    metadata = get_moe_lora_metadata(model)
    if metadata is None:
        return None
    os.makedirs(save_path, exist_ok=True)
    sidecar_path = os.path.join(save_path, MOE_LORA_SIDECAR_NAME)
    with open(sidecar_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return sidecar_path


def read_moe_lora_sidecar(adapter_path: str) -> dict[str, Any] | None:
    """Load the sidecar dict if present, else return ``None``.

    Validates ``mode`` is one of ``"shared"`` / ``"independent"``; returns the
    raw dict on success so callers can inspect / dispatch on it.
    """
    sidecar_path = os.path.join(adapter_path, MOE_LORA_SIDECAR_NAME)
    if not os.path.isfile(sidecar_path):
        return None
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    mode = sidecar.get("mode")
    if mode not in ("shared", "independent"):
        raise ValueError(f"{sidecar_path}: invalid 'mode' field {mode!r}; expected 'shared' or 'independent'.")
    return sidecar


def apply_moe_lora_from_sidecar(
    model: nn.Module,
    sidecar: dict[str, Any],
    *,
    freeze_base_model: bool = True,
) -> list[str]:
    """Convenience wrapper: rebuild the right MoE-LoRA wrappers from a sidecar dict.

    Dispatches on ``sidecar["mode"]``. Used by the resume path so the model has
    the right wrappers in place *before* PEFT's ``set_peft_model_state_dict``
    matches saved keys against the in-memory FQNs.
    """
    mode = sidecar["mode"]
    apply_fn = apply_shared_moe_lora if mode == "shared" else apply_independent_moe_lora
    return apply_fn(
        model,
        target_parameter_patterns=sidecar["target_parameter_patterns"],
        r=sidecar["r"],
        lora_alpha=sidecar["lora_alpha"],
        use_rslora=sidecar.get("use_rslora", False),
        adapter_name=sidecar.get("adapter_name", "default"),
        freeze_base_model=freeze_base_model,
    )
