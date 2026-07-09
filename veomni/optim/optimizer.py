# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

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

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .muon import DistributedMuon, split_muon_adamw_params


def _collect_ep_replicated_lora_param_ids(model: "nn.Module") -> set[int]:
    """Return ``id(p)`` of every LoRA Parameter that is *replicated* across the EP mesh.

    ``LoraSharedExperts`` (Mode 2) wraps a MoE experts module so that its
    extra LoRA tensors live alongside the EP-sharded base experts inside the
    same module. ``build_parallelize_model`` therefore pulls the LoRA
    parameters into the inner ``fully_shard(..., mesh=ep_fsdp, Shard(1))``
    wrap together with the base ``gate_up_proj`` / ``down_proj``. From the
    optimizer's point of view they look like "ep params" (their DTensor
    mesh has ``ep_fsdp`` as a dim) and the ExtraParallel-aware grad clipper
    would all-reduce their squared norms across the ``ep`` group — which
    is correct for the EP-sliced base experts but **double-counts** the
    shared LoRA whose values are identical on every EP rank (they are
    replicated by construction; ``LoraSharedExperts._ensure_ep_grad_sync_hooks``
    keeps their gradients in lock-step across the EP group).

    Excluding these param ids from the ``"ep"`` optimizer/clip-grad-norm
    bucket and putting them in the ``"non_extra_parallel"`` bucket gives
    the right semantics: gradients still get cross-EP-summed (by the
    LoraSharedExperts hook), the optimizer steps consistently on every
    rank, and ``clip_grad_norm`` only reduces them across the FSDP mesh
    (not the EP mesh) so the norm is computed once.

    Detection by class name (``LoraSharedExperts``) so this stays free of
    a circular import (``veomni.lora.moe_layers`` already imports from
    ``veomni.distributed.parallel_state``).

    Call-order requirement
    ----------------------
    Must run after FSDP wrapping has converted params to DTensors, hence
    after :func:`build_parallelize_model`. Otherwise the snapshotted
    ``id(p)`` values point at the pre-wrap ``nn.Parameter`` objects, which
    get replaced by new DTensor-backed parameters when ``fully_shard``
    rewraps the wrapper module -- the snapshot then goes stale (Python may
    reuse those addresses for unrelated objects, causing false positives
    in :func:`extra_parallel_fsdp2_clip_grad_norm`).

    ``id(p)`` stability after this point is guaranteed because
    ``build_optimizer`` populates ``model._extra_parallel_param_groups``
    from the same post-wrap Parameter objects in the same call, and those
    lists hold long-lived references that prevent garbage collection.
    """
    from torch.distributed._tensor import DTensor

    out: set[int] = set()
    for mod in model.modules():
        # Walk the MRO so we still match after FSDP2's ``fully_shard`` rebases the
        # class to ``FSDPLoraSharedExperts(LoraSharedExperts, FSDPModule)``
        # (see torch's ``_fully_shard``); a bare ``__class__.__name__`` check
        # would silently skip every wrapper post-FSDP.
        if any(b.__name__ == "LoraSharedExperts" for b in type(mod).__mro__):
            for n, p in mod.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                # Catches wrong call-order misuse: if any matched LoRA param
                # is still a plain ``nn.Parameter`` (not a DTensor) at this
                # point, FSDP2 hasn't wrapped yet -- ``build_optimizer`` was
                # invoked before ``build_parallelize_model``. The snapshot
                # we are about to take would be stale immediately.
                assert isinstance(p, DTensor), (
                    f"_collect_ep_replicated_lora_param_ids: LoRA param {n!r} on a "
                    f"LoraSharedExperts wrapper is not a DTensor -- call "
                    f"build_parallelize_model before build_optimizer so FSDP2 has "
                    f"converted these params to DTensors before we snapshot id(p)."
                )
                out.add(id(p))
    return out


logger = logging.get_logger(__name__)


def filter_empty_param_groups(
    param_groups: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop param groups with no trainable parameters.

    VLM trainers pass separate vit/backbone groups; when vit is frozen the vit
    group is empty. PyTorch DCP only persists optimizer state for non-empty
    groups, so keeping empty groups breaks resume (KeyError: 'betas' on
    optimizer.step() after set_optimizer_state_dict).

    ``params`` may be a generator/iterator. We materialize it into a list so
    that (a) an empty iterator is detected correctly (``bool(generator)`` is
    always True regardless of contents) and (b) it is not exhausted before the
    optimizer consumes it. Frozen params (``requires_grad=False``) are dropped
    too: they never receive optimizer state, so a group containing only frozen
    params is effectively empty and would hit the same resume ``KeyError``.
    Non-empty groups are returned as shallow copies with the filtered
    ``params`` list.
    """
    total = 0
    non_empty: List[Dict[str, Any]] = []
    for group in param_groups:
        total += 1
        params = group.get("params")
        if params is None:
            continue
        params_list = [p for p in params if getattr(p, "requires_grad", True)]
        if not params_list:
            continue
        new_group = dict(group)
        new_group["params"] = params_list
        non_empty.append(new_group)
    if len(non_empty) < total:
        logger.info_rank0(
            "Dropped %d empty optimizer param group(s) before optimizer construction.",
            total - len(non_empty),
        )
    return non_empty


def restore_optimizer_param_group_defaults(optimizer: Any) -> None:
    """Re-apply optimizer defaults to param groups after DCP load.

    Recurses into containers of optimizers so the safety net reaches every real
    optimizer regardless of how it was passed:

    * a plain ``dict`` of optimizers (the form PyTorch DCP APIs natively accept),
    * a ``MultiOptimizer`` (ExtraParallel+FSDP2 / Muon), which has no top-level
      ``defaults``/``param_groups`` and wraps sub-optimizers in ``optimizers_dict``.
    """
    if isinstance(optimizer, dict):
        for sub_optimizer in optimizer.values():
            restore_optimizer_param_group_defaults(sub_optimizer)
        return

    sub_optimizers = getattr(optimizer, "optimizers_dict", None)
    if sub_optimizers is not None:
        for sub_optimizer in sub_optimizers.values():
            restore_optimizer_param_group_defaults(sub_optimizer)
        return

    defaults = getattr(optimizer, "defaults", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if not defaults or not param_groups:
        return
    for group in param_groups:
        for key, value in defaults.items():
            group.setdefault(key, value)


# https://github.com/meta-llama/llama-recipes/blob/v0.0.4/src/llama_recipes/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=True,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # update uncentered variance

                bias_correction1 = 1 - beta1**step  # adjust using bias1
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:  # lr update to compensation
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:  # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)


class MultiOptimizer(Optimizer, Stateful):
    """
    A container that handles multiple optimizers (for extra_parallel and non-extra_parallel parameters when extra_parallel+fsdp2 is enabled)

    Mapping of name -> torch.optim.Optimizer with convenience methods.
    Compatible with torch.distributed.checkpoint optimizer APIs that accept a Mapping.

    This class is needed for ExtraParallel+FSDP2 case because ExtraParallel and non-ExtraParallel param have different FSDP sharding dimension (dim-0 vs. dim-1)
    """

    def __init__(
        self,
        root_model: nn.Module,
        optimizers: dict,  # {"para_1": opt_1, "para_2": opt_2, ..., "parak": opt_k, "non_extra_parallel": opt_{k+1}}
        key_names: list[str],
    ):
        self.model = root_model
        self.optimizers_dict = optimizers
        self._is_multi_optimizer: bool = True
        self.key_names = key_names

    def step(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.step()

    def zero_grad(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.zero_grad()

    def state_dict(
        self,
    ) -> Dict[str, Any]:
        # get the flatten state dict for multi-optimizer
        merged: Dict[str, Any] = {}
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            sd = get_optimizer_state_dict(self.model, opt, options=StateDictOptions(flatten_optimizer_state_dict=True))
            # check for key clashes before merging
            overlap = set(merged.keys()) & set(sd.keys())
            if overlap:
                raise KeyError(
                    f"Key clash detected while merging state dict for optimizer '{name}': {', '.join(sorted(overlap))}"
                )
            else:
                logger.info_rank0("No clashes when merging MultiOptimizer state dicts")
            merged.update(sd)

        return merged

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Feed the same merged flattened dict to each sub-optimizer; PyTorch will
        # pick out only the entries for parameters that belong to that optimizer.
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            set_optimizer_state_dict(
                self.model,
                opt,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )

    def register_step_pre_hook(self, hook):
        return [opt.register_step_pre_hook(hook) for opt in self.optimizers_dict.values()]

    def __len__(self) -> int:
        return len(self.optimizers_dict)


def _should_build_extra_parallel_aware(model: "nn.Module") -> bool:
    ps = get_parallel_state()
    if ps.dp_mode == "fsdp2" and ps.any_extra_parallel_enabled:
        return True

    return False


def _make_param_groups_for_subset(
    model: "nn.Module",
    params: Iterable[torch.nn.Parameter],
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    decay_param_names = set(get_parameter_names(model, no_decay_modules, no_decay_params))
    name_by_param = {p: n for n, p in model.named_parameters()}
    params = [p for p in params if p.requires_grad]
    decayed = [p for p in params if name_by_param.get(p) in decay_param_names]
    undecayed = [p for p in params if name_by_param.get(p) not in decay_param_names]
    groups: List[Dict[str, Any]] = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups


# adapted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer_pt_utils.py#L1123
def get_parameter_names(model, forbidden_layer_types, forbidden_param_names):
    forbidden_layer_types = [] if forbidden_layer_types is None else forbidden_layer_types
    forbidden_param_names = [] if forbidden_param_names is None else forbidden_param_names
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_param_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if child.__class__.__name__ not in forbidden_layer_types
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_param_names)
        ]

    result += [
        k for k in model._parameters.keys() if not any(forbidden in k.lower() for forbidden in forbidden_param_names)
    ]
    return result


def build_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    muon_kwargs: Optional[Dict[str, Any]] = None,
) -> "torch.optim.Optimizer":
    if optimizer_type == "muon":
        if param_groups is not None:
            logger.warning_rank0(
                "build_optimizer(optimizer_type='muon') ignores the provided "
                "param_groups argument; all parameters are re-split by "
                "split_muon_adamw_params. To control vit vs. backbone learning "
                "rates with Muon, file a follow-up or use AdamW for now."
            )
        return _build_muon_with_adamw(
            model,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
            no_decay_modules=no_decay_modules,
            no_decay_params=no_decay_params,
            muon_kwargs=muon_kwargs,
        )

    if param_groups is not None:
        param_groups = filter_empty_param_groups(param_groups)
        if not param_groups:
            raise ValueError("All optimizer param groups are empty; no trainable parameters to optimize.")

    if _should_build_extra_parallel_aware(model):
        return build_extra_parallel_fsdp2_optimizer(
            model, lr, betas, eps, weight_decay, fused, optimizer_type, param_groups, no_decay_modules, no_decay_params
        )
    if param_groups is None:
        decay_param_names = get_parameter_names(model, no_decay_modules, no_decay_params)
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_param_names and p.requires_grad],
                "weight_decay": weight_decay,
            },
        ]
        no_decay_parameters, no_decay_parameter_names = [], []
        for n, p in model.named_parameters():
            if n not in decay_param_names and p.requires_grad:
                no_decay_parameter_names.append(n)
                no_decay_parameters.append(p)

        if len(no_decay_parameters) > 0:
            logger.info_rank0(f"Parameters without weight decay: {no_decay_parameter_names}")
            param_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

    # Filter again to cover the auto-built groups above: if the model has no
    # decayed params requiring grad, the decay group is empty and would carry
    # the same DCP-resume KeyError as an empty VLM vit group.
    param_groups = filter_empty_param_groups(param_groups)
    if not param_groups:
        raise ValueError("All optimizer param groups are empty; no trainable parameters to optimize.")

    if optimizer_type == "adamw":
        foreach = not fused
        fused = fused
        optim = AdamW(param_groups, lr, betas, eps, weight_decay, fused=fused, foreach=foreach)
    elif optimizer_type == "anyprecision_adamw":
        optim = AnyPrecisionAdamW(param_groups, lr, betas, eps, weight_decay)
    else:
        raise ValueError(
            "Only adamw, anyprecision_adamw and muon are supported as optimizers; "
            f"got optimizer_type={optimizer_type!r}."
        )

    return optim


def _is_extra_parallel_param(p: torch.nn.Parameter, extra_parallel_names: Sequence[str]) -> Optional[str]:
    """Return the ExtraParallel name for a DTensor param, or ``None``."""
    if DTensor is None or not isinstance(p, DTensor):
        return None
    mesh = getattr(p, "device_mesh", None)
    names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
    for para in extra_parallel_names:
        if f"{para}_fsdp" in names:
            return para
    return None


def _build_muon_with_adamw(
    model: "nn.Module",
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    fused: bool,
    no_decay_modules: Optional[List[str]],
    no_decay_params: Optional[List[str]],
    muon_kwargs: Optional[Dict[str, Any]],
) -> "MultiOptimizer":
    """Build a Muon (2D/3D hidden weights) + AdamW (everything else) MultiOptimizer.

    ExtraParallel params are split into separate sub-optimizers so DCP state
    dict keys and grad-clipping metadata stay keyed by mesh.
    """
    muon_kwargs = dict(muon_kwargs or {})
    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(
        model,
        no_decay_modules=no_decay_modules,
        no_decay_params=no_decay_params,
    )
    if not muon_params:
        raise ValueError(
            "Muon optimizer was selected but the model has no eligible 2D/3D parameters. "
            "Falling back to AdamW would be silent so we raise instead."
        )

    extra_parallel_aware = _should_build_extra_parallel_aware(model)
    parallel_state = get_parallel_state() if extra_parallel_aware else None
    extra_parallel_names = list(parallel_state.extra_parallel_names) if extra_parallel_aware else []

    def _split_by_ep(
        params: List[torch.nn.Parameter],
    ) -> Tuple[Dict[str, List[torch.nn.Parameter]], List[torch.nn.Parameter]]:
        per_para: Dict[str, List[torch.nn.Parameter]] = {p: [] for p in extra_parallel_names}
        non_para: List[torch.nn.Parameter] = []
        if not extra_parallel_aware:
            return per_para, list(params)
        for p in params:
            para = _is_extra_parallel_param(p, extra_parallel_names)
            if para is not None:
                per_para[para].append(p)
            else:
                non_para.append(p)
        return per_para, non_para

    muon_per_para, muon_non_para = _split_by_ep(muon_params)
    adamw_per_para, adamw_non_para = _split_by_ep(adamw_params)

    logger.info_rank0(
        f"Muon optimizer: {len(muon_params)} param(s) on Muon, {len(adamw_params)} on AdamW. "
        f"First few Muon params: {muon_names[:5]}; first few AdamW params: {adamw_names[:5]}."
    )
    if extra_parallel_aware:
        for para in extra_parallel_names:
            logger.info_rank0(
                f"Muon split for {para}: muon_{para}={len(muon_per_para[para])}, "
                f"adamw_{para}={len(adamw_per_para[para])}, "
                f"muon_non_extra_parallel={len(muon_non_para)}, adamw_non_extra_parallel={len(adamw_non_para)}"
            )

    def _make_muon(params: List[torch.nn.Parameter]) -> DistributedMuon:
        return DistributedMuon(
            params,
            lr=muon_kwargs.get("lr", 2e-2),
            weight_decay=muon_kwargs.get("weight_decay", 0.0),
            momentum=muon_kwargs.get("momentum", 0.95),
            nesterov=muon_kwargs.get("nesterov", True),
            ns_coefficients=tuple(muon_kwargs.get("ns_coefficients", (3.4445, -4.7750, 2.0315))),
            eps=muon_kwargs.get("eps", 1e-7),
            ns_steps=muon_kwargs.get("ns_steps", 5),
            adjust_lr_fn=muon_kwargs.get("adjust_lr_fn", "match_rms_adamw"),
        )

    def _make_adamw(params: List[torch.nn.Parameter]) -> AdamW:
        groups = _make_param_groups_for_subset(model, params, weight_decay, no_decay_modules, no_decay_params)
        foreach = not fused
        return AdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=fused, foreach=foreach)

    optimizer_dict: Dict[str, Optimizer] = {}
    if extra_parallel_aware:
        for para in extra_parallel_names:
            if muon_per_para[para]:
                optimizer_dict[f"muon_{para}"] = _make_muon(muon_per_para[para])
        if muon_non_para:
            optimizer_dict["muon_non_extra_parallel"] = _make_muon(muon_non_para)
    else:
        optimizer_dict["muon"] = _make_muon(muon_non_para)

    if extra_parallel_aware:
        for para in extra_parallel_names:
            if adamw_per_para[para]:
                optimizer_dict[para] = _make_adamw(adamw_per_para[para])
        if adamw_non_para:
            optimizer_dict["non_extra_parallel"] = _make_adamw(adamw_non_para)
    elif adamw_params:
        optimizer_dict["adamw"] = _make_adamw(adamw_non_para)

    # Grad clipping groups by mesh, not by optimizer type.
    if extra_parallel_aware:
        model._extra_parallel_param_groups = {
            para: list(muon_per_para[para]) + list(adamw_per_para[para]) for para in extra_parallel_names
        }
        model._extra_parallel_param_groups["non_extra_parallel"] = list(muon_non_para) + list(adamw_non_para)

    return MultiOptimizer(model, optimizer_dict, key_names=list(optimizer_dict.keys()))


def build_extra_parallel_fsdp2_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[List[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
):
    """
    Build a MultiOptimizer instance when model is parallelized with ExtraParallel+FSDP2

    If param_groups provided, it can be a list of dicts with arbitrary parameter groups:
    - Example: [{"params": params1, "lr": lr1},
                {"params": params2, "lr": lr2},
                {"params": params3, "lr": lr3}]
    - Each group's params are automatically split into ExtraParallel1, ExtraParallel2, ... and non-ExtraParallel based on DTensor mesh
    - Custom learning rates and other optimizer settings are preserved per group
    """
    parallel_state = get_parallel_state()

    # Collect all ExtraParallel and non-ExtraParallel parameters across all groups
    extra_parallel_groups = {
        para: []  # List[Dict[str, Any]]
        for para in parallel_state.extra_parallel_names
    }
    non_extra_parallel_groups: List[Dict[str, Any]] = []

    # Process custom param_groups if provided
    if param_groups is not None:
        # Validate param_groups structure
        assert isinstance(param_groups, list), "param_groups must be a list"

        # Process each parameter group
        for group_config in param_groups:
            assert "params" in group_config, (
                f"Each group in param_groups must contain 'params' key, got: {group_config}"
            )

            # Extract group-specific settings
            group_lr = group_config.get("lr", lr)
            group_params = group_config["params"]

            # Split this group's params into ExtraParallel and non-ExtraParallel
            group_extra_parallel_params = {
                para: []  # List[torch.nn.Parameter]
                for para in parallel_state.extra_parallel_names
            }
            group_non_extra_parallel_params: List[torch.nn.Parameter] = []

            for p in group_params:
                if not p.requires_grad:
                    continue

                # Check if this parameter is part of ExtraParallel
                is_extra_parallel_params = False
                if DTensor is not None and isinstance(p, DTensor):
                    mesh = getattr(p, "device_mesh", None)
                    names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                    for para in parallel_state.extra_parallel_names:
                        if f"{para}_fsdp" in names:
                            group_extra_parallel_params[para].append(p)
                            is_extra_parallel_params = True
                            break

                if not is_extra_parallel_params:
                    group_non_extra_parallel_params.append(p)

            # Create subgroups with weight decay handling
            for para in parallel_state.extra_parallel_names:
                if group_extra_parallel_params[para]:
                    group_para_subgroups = _make_param_groups_for_subset(
                        model, group_extra_parallel_params[para], weight_decay, no_decay_modules, no_decay_params
                    )
                    for subgroup in group_para_subgroups:
                        subgroup["lr"] = group_lr
                        # Preserve other custom settings from original group
                        for key, value in group_config.items():
                            if key not in ["params", "lr", "weight_decay"]:
                                subgroup[key] = value
                    extra_parallel_groups[para].extend(group_para_subgroups)

            if group_non_extra_parallel_params:
                group_non_extra_parallel_subgroups = _make_param_groups_for_subset(
                    model, group_non_extra_parallel_params, weight_decay, no_decay_modules, no_decay_params
                )
                for subgroup in group_non_extra_parallel_subgroups:
                    subgroup["lr"] = group_lr
                    # Preserve other custom settings from original group
                    for key, value in group_config.items():
                        if key not in ["params", "lr", "weight_decay"]:
                            subgroup[key] = value
                non_extra_parallel_groups.extend(group_non_extra_parallel_subgroups)
    else:
        # Default case (param_groups is None): all model parameters with uniform settings(lr)
        extra_parallel_params = {
            para: []  # List[torch.nn.Parameter]
            for para in parallel_state.extra_parallel_names
        }
        non_extra_parallel_params: List[torch.nn.Parameter] = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # Check if this parameter is part of ExtraParallel
            is_extra_parallel_params = False
            if DTensor is not None and isinstance(p, DTensor):
                mesh = getattr(p, "device_mesh", None)
                names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                logger.debug_rank0(f"param {name} has device_mesh {mesh} with mesh dim names {names}")

                for para in parallel_state.extra_parallel_names:
                    if f"{para}_fsdp" in names:
                        logger.debug_rank0(f"Adding {name} to {para}_params in extra_parallel+fsdp2 optimizer")
                        extra_parallel_params[para].append(p)
                        is_extra_parallel_params = True
                        break

            if not is_extra_parallel_params:
                logger.debug_rank0(f"Adding {name} to non_extra_parallel_params in extra_parallel+fsdp2 optimizer")
                non_extra_parallel_params.append(p)

        # Build param groups with weight decay handling
        extra_parallel_groups = {
            para: _make_param_groups_for_subset(
                model, extra_parallel_params[para], weight_decay, no_decay_modules, no_decay_params
            )
            for para in parallel_state.extra_parallel_names
        }
        non_extra_parallel_groups = _make_param_groups_for_subset(
            model, non_extra_parallel_params, weight_decay, no_decay_modules, no_decay_params
        )

    def _build(groups: Sequence[Dict[str, Any]]) -> Optimizer:
        if optimizer_type == "adamw":
            nonlocal fused
            foreach = not fused
            _fused = fused
            return AdamW(groups, lr, betas, eps, weight_decay, fused=_fused, foreach=foreach)
        elif optimizer_type == "anyprecision_adamw":
            return AnyPrecisionAdamW(groups, lr, betas, eps, weight_decay)
        else:
            raise ValueError("Only adamw and anyprecision_adamw are supported as optimizers.")

    optimizer_dict: Dict[str, Optimizer] = {}
    for para in parallel_state.extra_parallel_names:
        if extra_parallel_groups[para]:
            optimizer_dict[para] = _build(extra_parallel_groups[para])
    if non_extra_parallel_groups:
        optimizer_dict["non_extra_parallel"] = _build(non_extra_parallel_groups)

    # cache for ExtraParallel-aware grad clipping helpers
    model._extra_parallel_param_groups = {
        para: [p for g in extra_parallel_groups[para] for p in g.get("params", [])]
        if extra_parallel_groups[para]
        else []
        for para in parallel_state.extra_parallel_names
    }
    model._extra_parallel_param_groups["non_extra_parallel"] = (
        [p for g in non_extra_parallel_groups for p in g.get("params", [])] if non_extra_parallel_groups else []
    )
    # Sidecar set for the grad clipper: ids of ``LoraSharedExperts`` LoRA params
    # that landed in some ``extra_parallel_params[para]`` bucket above. They live
    # on the inner ``ep_fsdp`` FSDP mesh (so they cannot be moved to the
    # ``non_extra_parallel`` optimizer — its DTensors are on a different mesh and
    # ``foreach_*`` ops reject mixed-mesh tensor lists), but unlike the EP-sliced
    # base experts and Mode-1 per-expert LoRA they are *replicated* across the
    # EP group (kept in lock-step by ``LoraSharedExperts._ensure_ep_grad_sync_hooks``).
    # ``extra_parallel_fsdp2_clip_grad_norm`` uses this set to skip the EP all-reduce
    # for these params; otherwise their squared norms are summed once per EP rank
    # and the global norm comes out larger than the EP=1 baseline, breaking
    # cross-EP grad-norm parity (and therefore any optimizer step that depends
    # on the clipped grad).
    model._ep_replicated_lora_param_ids = _collect_ep_replicated_lora_param_ids(model)

    key_names = list(optimizer_dict.keys())

    # Build MultiOptimizer and attach a pre-step hook to sanitize DTensor states
    multi_opt = MultiOptimizer(model, optimizer_dict, key_names=key_names)

    return multi_opt
