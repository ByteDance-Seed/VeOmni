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

"""Shared-LoRA wrapper for MoE expert modules ("Mode 2").

PEFT 0.19's ``target_parameters`` covers the *independent* per-expert LoRA
case ("Mode 1"): a 3D LoRA tensor whose dim-0 matches the expert dim.
There is no PEFT support for a *shared* LoRA — a single 2D LoRA pair
broadcast across all experts. This module fills that gap.

Forward strategy
----------------
For one expert ``e`` with base weight ``W_e`` of shape ``[O, H]``::

    aug_out_e(x) = (W_e + B @ A * s) @ x
                 = W_e @ x + B @ (A @ x) * s

The right-hand form avoids materialising an ``[E, O, H]`` delta tensor
(which is what PEFT's ``ParamWrapper`` does for Mode 1 and what the docs
warn about as runtime overhead). For ``gate_up_proj`` the LoRA contribution
``B @ (A @ x) * s`` only depends on the *input* token, so we compute it
once per token outside the per-expert dispatch loop. For ``down_proj`` the
input is the per-expert intermediate activation ``silu(gate) * up`` and so
the LoRA term is computed inside the loop.

Layouts supported (auto-detected from the base experts module):

- **v5 fused**: base owns ``gate_up_proj`` ``[E, 2I, H]`` and ``down_proj``
  ``[E, H, I]`` (Qwen3-MoE / Qwen3.5-MoE / Qwen3-VL-MoE / Qwen3-Omni-MoE
  generated/patched modeling).
- **v4 split**: base owns ``gate_proj`` ``[E, I, H]``, ``up_proj``
  ``[E, I, H]`` and ``down_proj`` ``[E, H, I]`` (Qwen3-MoE
  ``apply_veomni_qwen3_moe_patch`` v4 path).

The wrapper preserves the original module under ``base_layer`` (mirroring
PEFT's wrapping convention) and replaces the experts module *in its
parent at the same attribute name*, so downstream lookups by FQN
(``mlp.experts``) continue to resolve. The original 3D parameters move to
``mlp.experts.base_layer.<param>``; the EP parallel plan must be aware of
this when ``share_expert_lora=True`` (handled in a later phase when the
trainer wires through to ``build_parallelize_model``).

Phase 4 will add a fused-Triton path; the wrapper has a single dispatch
point in ``forward`` for that.

PEFT save/load compatibility
----------------------------
LoRA weights are stored as ``self.lora_A_<param>`` / ``self.lora_B_<param>``,
each an ``nn.ModuleDict`` of ``nn.Linear`` keyed by adapter name (mirroring
the structure used by PEFT's ``LoraLayer`` so the same machinery works):

- in-memory FQN: ``mlp.experts.lora_A_gate_up_proj.<adapter>.weight``
- ``get_peft_model_state_dict`` filter passes (key contains ``lora_`` *and*
  the adapter name) and ``remove_adapter_name`` strips ``<adapter>`` exactly
  as for PEFT-managed Linear LoRA.
- on load, ``set_peft_model_state_dict`` re-inserts the adapter name via the
  generic ``rpartition('lora_')`` path, restoring the full FQN.

VeOmni's ``_remap_adapter_key`` (used by FSDP1 / rank-0 broadcast loaders)
needs to recognise the ``lora_A_<param>`` / ``lora_B_<param>`` attribute
names too — handled by a one-line ``startswith`` extension in
``veomni/utils/lora_utils.py``.

A sidecar file ``veomni_share_lora.json`` is written next to PEFT's
``adapter_config.json`` so that the resume path (``PeftModel.from_pretrained``)
knows to re-install the wrappers *before* PEFT loads the state dict; without
the wrappers in place, the saved ``lora_A_<param>.<adapter>.weight`` keys
would have no destination.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# Module FQNs of PEFT-wrapped models gain a ``base_model.model.`` prefix.
# Patterns supplied by the user in ``lora_config['target_parameters']`` are
# written against the *base* model FQN (e.g. ``model.layers.0.mlp.experts.gate_up_proj``)
# and stripped before matching.
_PEFT_PREFIX = "base_model.model."


def _glob_to_regex(pattern: str) -> "re.Pattern[str]":
    """Convert a PEFT-style glob (``*``) to a fully-anchored regex."""
    parts = [re.escape(piece) for piece in pattern.split("*")]
    return re.compile(".*".join(parts) + r"\Z")


def _strip_peft_prefix(fqn: str) -> str:
    return fqn[len(_PEFT_PREFIX) :] if fqn.startswith(_PEFT_PREFIX) else fqn


def _find_target_parameter_modules(
    model: nn.Module,
    patterns: List[str],
) -> List[Tuple[nn.Module, str, str, nn.Module]]:
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
    matches: List[Tuple[nn.Module, str, str, nn.Module]] = []
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


# v5 fused layout: base experts module owns these two 3D Parameters.
_FUSED_V5_PARAMS = ("gate_up_proj", "down_proj")
# v4 split layout: base experts module owns these three 3D Parameters.
_SPLIT_V4_PARAMS = ("gate_proj", "up_proj", "down_proj")


def _detect_layout(base_layer: nn.Module) -> str:
    """Return ``"fused_v5"`` or ``"split_v4"``; raise if neither matches."""

    def has(name: str) -> bool:
        return hasattr(base_layer, name) and isinstance(getattr(base_layer, name), nn.Parameter)

    if has("gate_up_proj") and has("down_proj"):
        return "fused_v5"
    if has("gate_proj") and has("up_proj") and has("down_proj"):
        return "split_v4"
    raise ValueError(
        f"LoraSharedExperts cannot detect MoE layout of {type(base_layer).__name__!s}: "
        f"expected v5 fused (gate_up_proj/down_proj) or v4 split "
        f"(gate_proj/up_proj/down_proj). Got attrs: "
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
    (for B):

    - v5 fused → ``lora_A_gate_up_proj``, ``lora_B_gate_up_proj``,
      ``lora_A_down_proj``, ``lora_B_down_proj``.
    - v4 split → same pattern with three target params.

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

        # Shapes for each LoRA pair: A is [r, in_features], B is [out_features, r].
        # Naming follows F.linear semantics: F.linear(x, W) computes x @ W.T,
        # so for base weight W_e of shape [O, H], the pair (A: [r, H], B: [O, r])
        # gives delta_W = B @ A of shape [O, H].
        self._layout = _detect_layout(base_layer)
        if self._layout == "fused_v5":
            self._lora_specs = {
                "gate_up_proj": (self.hidden_dim, 2 * self.intermediate_dim),
                "down_proj": (self.intermediate_dim, self.hidden_dim),
            }
        else:  # split_v4
            self._lora_specs = {
                "gate_proj": (self.hidden_dim, self.intermediate_dim),
                "up_proj": (self.hidden_dim, self.intermediate_dim),
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

    def _get_lora_linear(self, role: str, param_name: str, adapter_name: Optional[str] = None) -> nn.Linear:
        adapter = adapter_name or self.adapter_name
        return getattr(self, f"lora_{role}_{param_name}")[adapter]

    def get_lora_A_weight(self, param_name: str, adapter_name: Optional[str] = None) -> torch.Tensor:
        """Active LoRA A weight for ``param_name``, shape ``[r, in_features]``."""
        return self._get_lora_linear("A", param_name, adapter_name).weight

    def get_lora_B_weight(self, param_name: str, adapter_name: Optional[str] = None) -> torch.Tensor:
        """Active LoRA B weight for ``param_name``, shape ``[out_features, r]``."""
        return self._get_lora_linear("B", param_name, adapter_name).weight

    @torch.no_grad()
    def reset_lora_parameters(self, adapter_name: Optional[str] = None, init_lora_weights: bool = True) -> None:
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
    def lora_A(self) -> Dict[str, nn.Linear]:
        # Return one Linear per adapter (taken from the first target param) so
        # callers iterating ``lora_A.keys()`` see the right adapter list.
        first_pname = next(iter(self._lora_specs))
        return dict(getattr(self, f"lora_A_{first_pname}"))

    @property
    def lora_B(self) -> Dict[str, nn.Linear]:
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
        # Phase 4 will add a fused-kernel branch here, mirroring the
        # ``veomni_moe_experts_forward.use_non_eager_impl`` guard inside
        # Qwen3MoeExperts.forward. For now we are eager-only.
        if self._layout == "fused_v5":
            return self._eager_forward_fused_v5(hidden_states, top_k_index, top_k_weights)
        return self._eager_forward_split_v4(hidden_states, top_k_index, top_k_weights)

    def _eager_forward_fused_v5(
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

    def _eager_forward_split_v4(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_layer
        scale = self.lora_scaling.to(hidden_states.dtype)
        a_g = self.get_lora_A_weight("gate_proj")
        b_g = self.get_lora_B_weight("gate_proj")
        a_u = self.get_lora_A_weight("up_proj")
        b_u = self.get_lora_B_weight("up_proj")
        a_dn = self.get_lora_A_weight("down_proj")
        b_dn = self.get_lora_B_weight("down_proj")

        # Shared LoRA deltas on gate and up depend only on x.
        lora_x_gate = F.linear(F.linear(hidden_states, a_g), b_g) * scale
        lora_x_up = F.linear(F.linear(hidden_states, a_u), b_u) * scale

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

            gate = F.linear(current_state, base.gate_proj[expert_idx]) + lora_x_gate[token_idx]
            up = F.linear(current_state, base.up_proj[expert_idx]) + lora_x_up[token_idx]
            mid = self.act_fn(gate) * up

            lora_x_down = F.linear(F.linear(mid, a_dn), b_dn) * scale
            current_hidden_states = F.linear(mid, base.down_proj[expert_idx]) + lora_x_down
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def extra_repr(self) -> str:
        return f"layout={self._layout}, r={self.r}, alpha={self.lora_alpha}, num_experts={self.num_experts}"


def apply_shared_moe_lora(
    model: nn.Module,
    target_parameter_patterns: List[str],
    r: int,
    lora_alpha: int,
    use_rslora: bool = False,
    adapter_name: str = "default",
    fail_on_no_match: bool = True,
    freeze_base_model: bool = True,
) -> List[str]:
    """In-place wrap experts modules in ``model`` with :class:`LoraSharedExperts`.

    Walks ``model.named_modules()``, finds each module that owns at least one
    3D ``nn.Parameter`` matching ``target_parameter_patterns`` (PEFT-style
    globs; the leading ``base_model.model.`` prefix is stripped before
    matching), and replaces it in its parent with a wrapper.

    Args:
        target_parameter_patterns: e.g. ``["model.layers.*.mlp.experts.gate_up_proj",
            "model.layers.*.mlp.experts.down_proj"]``. Multiple patterns that
            point at the same experts module are deduplicated — each module is
            wrapped at most once. The pattern list is also stashed on the model
            (as ``model._veomni_share_lora_patterns``) so the save helper can
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

    wrapped_fqns: List[str] = []
    for parent, parent_fqn, attr_name, base_module in matches:
        wrapper = LoraSharedExperts(
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
    # `_veomni_` so it doesn't show up in state_dict / FSDP traversal.
    model._veomni_share_lora_patterns = list(target_parameter_patterns)
    return sorted(wrapped_fqns)


def is_lora_shared_experts(module: nn.Module) -> bool:
    """Type-stable check used by save/load helpers."""
    return isinstance(module, LoraSharedExperts)


def has_lora_shared_experts(model: nn.Module) -> bool:
    """True iff ``model`` contains at least one :class:`LoraSharedExperts`."""
    for _, m in model.named_modules():
        if isinstance(m, LoraSharedExperts):
            return True
    return False


def iter_shared_lora_parameters(model: nn.Module):
    """Yield ``(fqn, parameter)`` pairs for every shared-LoRA tunable parameter.

    Walks every :class:`LoraSharedExperts` and yields each ``lora_A_*`` /
    ``lora_B_*`` ``nn.ModuleDict[adapter -> Linear]`` weight, with full FQN
    suitable for state_dict lookup.
    """
    for fqn, module in model.named_modules():
        if not is_lora_shared_experts(module):
            continue
        prefix = f"{fqn}." if fqn else ""
        for n, p in module.named_parameters(recurse=True):
            if n.startswith("base_layer."):
                continue
            if n.startswith("lora_A_") or n.startswith("lora_B_"):
                yield prefix + n, p


def get_shared_lora_metadata(model: nn.Module) -> Optional[Dict[str, Any]]:
    """Return a sidecar dict describing the shared-LoRA wrappers, or None.

    The dict is JSON-serialisable and contains everything needed to rebuild the
    wrappers on resume: rank/alpha/rslora/adapter_name plus the original
    ``target_parameter_patterns`` (recovered from the stash placed on
    ``model`` by :func:`apply_shared_moe_lora`, falling back to the wrapped
    FQNs if the stash is missing — e.g. when the model was loaded from a
    checkpoint without going through ``apply_shared_moe_lora``).
    """
    wrappers = [(fqn, m) for fqn, m in model.named_modules() if is_lora_shared_experts(m)]
    if not wrappers:
        return None
    sample = wrappers[0][1]
    target_parameter_patterns = getattr(model, "_veomni_share_lora_patterns", None)
    if target_parameter_patterns is None:
        # Fallback: synthesise patterns from the wrapped FQNs and the layout's
        # target list. Less compact than user-supplied globs but still unique.
        target_parameter_patterns = []
        for fqn, m in wrappers:
            for pname in m._lora_specs:
                target_parameter_patterns.append(f"{fqn}.{pname}")
    return {
        "kind": "lora_shared_experts",
        "version": 1,
        "r": sample.r,
        "lora_alpha": sample.lora_alpha,
        "use_rslora": sample.use_rslora,
        "adapter_name": sample.adapter_name,
        "target_parameter_patterns": list(target_parameter_patterns),
        "wrapped_fqns": [fqn for fqn, _ in wrappers],
    }


# ----------------------------------------------------------------------
# Sidecar I/O — kept in this module so save/load callers don't need to know
# the JSON schema.
# ----------------------------------------------------------------------

SHARED_LORA_SIDECAR_NAME = "veomni_share_lora.json"


def write_shared_lora_sidecar(model: nn.Module, save_path: str) -> Optional[str]:
    """If ``model`` has shared-MoE-LoRA wrappers, write the sidecar to disk.

    Returns the written path, or ``None`` if no wrappers were found.
    """
    metadata = get_shared_lora_metadata(model)
    if metadata is None:
        return None
    os.makedirs(save_path, exist_ok=True)
    sidecar_path = os.path.join(save_path, SHARED_LORA_SIDECAR_NAME)
    with open(sidecar_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return sidecar_path


def read_shared_lora_sidecar(adapter_path: str) -> Optional[Dict[str, Any]]:
    """Load the sidecar dict if present, else return None."""
    sidecar_path = os.path.join(adapter_path, SHARED_LORA_SIDECAR_NAME)
    if not os.path.isfile(sidecar_path):
        return None
    with open(sidecar_path) as f:
        return json.load(f)


def apply_shared_moe_lora_from_sidecar(
    model: nn.Module,
    sidecar: Dict[str, Any],
    *,
    freeze_base_model: bool = True,
) -> List[str]:
    """Convenience wrapper: rebuild :class:`LoraSharedExperts` from a sidecar dict.

    Used by the resume path so the model has the right wrappers in place
    *before* PEFT's ``set_peft_model_state_dict`` matches saved keys against
    the in-memory FQNs.
    """
    return apply_shared_moe_lora(
        model,
        target_parameter_patterns=sidecar["target_parameter_patterns"],
        r=sidecar["r"],
        lora_alpha=sidecar["lora_alpha"],
        use_rslora=sidecar.get("use_rslora", False),
        adapter_name=sidecar.get("adapter_name", "default"),
        freeze_base_model=freeze_base_model,
    )
