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

"""DTensor-aware Muon optimizer for FSDP2 and MoE expert weights.

``DistributedMuon`` keeps upstream ``torch.optim.Muon`` numerics for 2D
weights and adds batched Newton-Schulz for 3D MoE expert stacks. Sharded
params are gathered only when the NS iteration needs the full trailing
``[M, K]`` matrix.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.optim.optimizer import Optimizer

from ..utils import logging
from ..utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


logger = logging.get_logger(__name__)


try:
    # Reuse upstream's private NS constants and the LR-adjust helper so we
    # match its math without copying it.
    from torch.optim._muon import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_C,
        DEFAULT_NS_STEPS,
        EPS,
        _adjust_lr,
    )

    _MUON_AVAILABLE = True
except ImportError:  # pragma: no cover - torch < 2.9 fallback
    _MUON_AVAILABLE = False
    DEFAULT_A = 3.4445
    DEFAULT_B = -4.7750
    DEFAULT_C = 2.0315
    DEFAULT_NS_STEPS = 5
    EPS = 1e-7
    _adjust_lr = None  # type: ignore[assignment]


__all__ = [
    "DEFAULT_NS_COEFFICIENTS",
    "DEFAULT_NS_STEPS",
    "DistributedMuon",
    "batched_gram_newton_schulz",
    "NS_IMPLEMENTATIONS",
    "batched_newton_schulz",
    "run_newton_schulz",
    "split_muon_adamw_params",
]


DEFAULT_NS_COEFFICIENTS: Tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C)

# Lookup-table-like 2D weights are kept on AdamW.
_DEFAULT_ADAMW_NAME_PATTERNS: Tuple[str, ...] = (
    "embed_tokens",
    "embedding",
    "lm_head",
    "output_layer",
)


def _as_coeff_schedule(
    ns_coefficients: Sequence[Any],
    ns_steps: int,
) -> Tuple[Tuple[float, float, float], ...]:
    """Expand a single (a,b,c) or validate a per-step coefficient schedule."""
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if len(ns_coefficients) == 0:
        raise ValueError("ns_coefficients must not be empty")

    # Single triple: (a, b, c) reused for every step (torch.optim.Muon style).
    first = ns_coefficients[0]
    if len(ns_coefficients) == 3 and not isinstance(first, (list, tuple)):
        a, b, c = (float(ns_coefficients[0]), float(ns_coefficients[1]), float(ns_coefficients[2]))
        return tuple((a, b, c) for _ in range(ns_steps))

    # Flat sequences that are not a single (a,b,c) are invalid.
    if all(not isinstance(x, (list, tuple)) for x in ns_coefficients):
        raise ValueError(
            "Coefficients must be a tuple of exactly 3 values (a, b, c), or a "
            f"sequence of (a, b, c) triples; got length-{len(ns_coefficients)} flat sequence"
        )

    schedule: List[Tuple[float, float, float]] = []
    for row in ns_coefficients:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"Per-step ns_coefficients must be a sequence of (a, b, c) triples; got {row!r}")
        schedule.append((float(row[0]), float(row[1]), float(row[2])))
    if not schedule:
        raise ValueError("ns_coefficients schedule is empty")
    return tuple(schedule)


def _flatten_matrix_batch(grad: Tensor) -> Tuple[Tensor, torch.Size]:
    """Reshape ``[..., M, K]`` into ``[B, M, K]`` while remembering the original shape."""
    if grad.ndim < 2:
        raise ValueError(f"Input must have ndim >= 2, got shape {tuple(grad.shape)}")
    original_shape = grad.shape
    if grad.ndim == 2:
        return grad.unsqueeze(0), original_shape
    if grad.ndim == 3:
        return grad, original_shape
    return grad.reshape(-1, grad.shape[-2], grad.shape[-1]), original_shape


@torch.no_grad()
def batched_newton_schulz(
    grad: Tensor,
    ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run quintic Newton-Schulz on each trailing ``[M, K]`` matrix."""
    schedule = _as_coeff_schedule(ns_coefficients, ns_steps)
    if grad.ndim < 2:
        raise ValueError(f"Input must have ndim >= 2, got shape {tuple(grad.shape)}")

    original_dtype = grad.dtype
    ortho = grad.to(compute_dtype)

    # Keep the Gram matrix on the smaller trailing dimension.
    transposed = ortho.size(-2) > ortho.size(-1)
    if transposed:
        ortho = ortho.mT

    norm = ortho.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    ortho = ortho / norm

    # Keep the 2D path byte-compatible with upstream; use the batched form for 3D.
    for a, b, c in schedule:
        A = ortho @ ortho.mT
        if A.ndim == 2:
            gram_update = torch.addmm(A, A, A, beta=b, alpha=c)
            ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
        else:
            # ``baddbmm`` is strictly 3D; flatten any leading batch dims.
            *batch, M_, K_ = ortho.shape
            B_ = 1
            for d in batch:
                B_ *= d
            A_3d = A.reshape(B_, M_, M_)
            ortho_3d = ortho.reshape(B_, M_, K_)
            gram_update = torch.baddbmm(A_3d, A_3d, A_3d, beta=b, alpha=c)
            ortho = torch.baddbmm(ortho_3d, gram_update, ortho_3d, beta=a).reshape(*batch, M_, K_)
        del A

    if transposed:
        ortho = ortho.mT

    return ortho.to(original_dtype)


@torch.no_grad()
def batched_gram_newton_schulz(
    grad: Tensor,
    ns_coefficients: Sequence[Any] = DEFAULT_NS_COEFFICIENTS,
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
    compute_dtype: torch.dtype = torch.float16,
    reset_iterations: Sequence[int] = (2,),
) -> Tensor:
    """Pure-PyTorch Gram Newton-Schulz (Dao-AILab algorithm, no custom kernels).

    Algebraically rearranges quintic NS so the iterative work stays on the small
    square Gram matrix, with optional intermediate restarts for stability.
    Square matrices fall back to standard Newton-Schulz.
    """
    schedule = _as_coeff_schedule(ns_coefficients, ns_steps)
    reset_set = {int(i) for i in reset_iterations}
    original_dtype = grad.dtype
    X, original_shape = _flatten_matrix_batch(grad)

    # Strictly rectangular only — same policy as Dao-AILab/gram-newton-schulz.
    if X.size(-2) == X.size(-1):
        out = batched_newton_schulz(
            X,
            ns_coefficients=schedule,
            ns_steps=len(schedule),
            eps=eps,
            compute_dtype=torch.bfloat16 if compute_dtype == torch.float16 else compute_dtype,
        )
        return out.to(original_dtype).reshape(original_shape)

    tall_skinny = X.size(-2) > X.size(-1)
    # Match upstream package numerics: normalize in fp32, iterate in half.
    X = X.to(torch.float32)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    X = X.to(compute_dtype)

    if tall_skinny:
        R = torch.bmm(X.transpose(1, 2), X)
    else:
        R = torch.bmm(X, X.transpose(1, 2))

    batch = R.size(0)
    eye = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch, -1, -1).contiguous()
    Q: Optional[Tensor] = None

    for i, (a, b, c) in enumerate(schedule):
        if i in reset_set and i != 0:
            if Q is None:
                raise RuntimeError("Gram-NS restart requested before Q was initialized")
            if tall_skinny:
                X = torch.bmm(X, Q)
                R = torch.bmm(X.transpose(1, 2), X)
            else:
                X = torch.bmm(Q, X)
                R = torch.bmm(X, X.transpose(1, 2))
            Q = None

        # Z = b R + c R @ R
        Z = torch.baddbmm(R, R, R, beta=b, alpha=c)
        if i == 0 or i in reset_set:
            Q = Z + a * eye
        else:
            assert Q is not None
            # Q = a Q + Q @ Z
            Q = torch.baddbmm(Q, Q, Z, beta=a, alpha=1.0)

        # Skip Gram update before a restart or after the final step.
        if i < len(schedule) - 1 and (i + 1) not in reset_set:
            assert Q is not None
            # RZ = a R + R @ Z ; R = a RZ + Z @ RZ
            RZ = torch.baddbmm(R, R, Z, beta=a, alpha=1.0)
            R = torch.baddbmm(RZ, Z, RZ, beta=a, alpha=1.0)

    assert Q is not None
    if tall_skinny:
        X = torch.bmm(X, Q)
    else:
        X = torch.bmm(Q, X)
    return X.to(original_dtype).reshape(original_shape)


_PACKAGE_GRAM_NS_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _package_gram_newton_schulz(
    grad: Tensor,
    schedule: Tuple[Tuple[float, float, float], ...],
    eps: float,
    reset_iterations: Sequence[int],
    use_kernels: bool,
) -> Tensor:
    """Call Dao-AILab ``GramNewtonSchulz`` (optional quack/CuTeDSL kernels)."""
    try:
        from gram_newton_schulz import GramNewtonSchulz
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "muon_ns_implementation='gram_quack' requires the gram-newton-schulz "
            "package. Install with: "
            "`pip install gram-newton-schulz --no-build-isolation` "
            "(needs Hopper/Blackwell + quack-kernels)."
        ) from exc

    key = (schedule, float(eps), tuple(int(i) for i in reset_iterations), bool(use_kernels))
    ortho = _PACKAGE_GRAM_NS_CACHE.get(key)
    if ortho is None:
        ortho = GramNewtonSchulz(
            ns_epsilon=float(eps),
            ns_use_kernels=bool(use_kernels),
            ns_coefficients=[list(t) for t in schedule],
            gram_newton_schulz_reset_iterations=[int(i) for i in reset_iterations],
            # torch.compile is brittle across torch/CUDA combos; keep eager.
            compile_kwargs=None,
        )
        _PACKAGE_GRAM_NS_CACHE[key] = ortho
    return ortho(grad)


# Newton-Schulz backend selectors for DistributedMuon / run_newton_schulz.
NS_IMPLEMENTATIONS = ("std", "gram", "gram_quack")
_GRAM_QUACK_FALLBACK_WARNED = False


def _gram_quack_unavailable_reason(grad: Tensor) -> Optional[str]:
    if not IS_CUDA_AVAILABLE or grad.device.type != get_device_type():
        return f"{grad.device.type} tensors are unsupported; quack kernels require CUDA SM90 or newer"
    compute_capability = get_gpu_compute_capability(grad.device)
    if compute_capability < 90:
        major, minor = divmod(compute_capability, 10)
        return f"CUDA compute capability {major}.{minor} is unsupported; quack kernels require 9.0 or newer"
    return None


@torch.no_grad()
def run_newton_schulz(
    grad: Tensor,
    ns_coefficients: Sequence[Any] = DEFAULT_NS_COEFFICIENTS,
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
    ns_implementation: str = "gram_quack",
    gram_ns_reset_iterations: Sequence[int] = (2,),
    compute_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Dispatch Newton-Schulz / Gram Newton-Schulz for a ``[..., M, K]`` update.

    ``ns_implementation``:
      - ``std``: torch Muon-compatible Newton-Schulz
      - ``gram``: pure-PyTorch Gram Newton-Schulz
      - ``gram_quack`` (default): quack CuTeDSL GEMM kernels (falls back to ``gram`` if unavailable)
    """
    if ns_implementation not in NS_IMPLEMENTATIONS:
        raise ValueError(f"Unknown ns_implementation={ns_implementation!r}; expected one of {NS_IMPLEMENTATIONS}")

    if ns_implementation == "std":
        schedule = _as_coeff_schedule(ns_coefficients, ns_steps)
        return batched_newton_schulz(
            grad,
            ns_coefficients=schedule,
            ns_steps=len(schedule),
            eps=eps,
            compute_dtype=torch.bfloat16 if compute_dtype is None else compute_dtype,
        )

    schedule = _as_coeff_schedule(ns_coefficients, ns_steps)
    if ns_implementation == "gram_quack":
        fallback_reason = _gram_quack_unavailable_reason(grad)
        if fallback_reason is None:
            try:
                return _package_gram_newton_schulz(
                    grad,
                    schedule=schedule,
                    eps=eps,
                    reset_iterations=gram_ns_reset_iterations,
                    use_kernels=True,
                )
            except ImportError as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"
        global _GRAM_QUACK_FALLBACK_WARNED
        if not _GRAM_QUACK_FALLBACK_WARNED:
            _GRAM_QUACK_FALLBACK_WARNED = True
            logger.warning_rank0(
                "[Muon] ns_implementation=gram_quack requested but gram-newton-schulz/quack "
                f"is unavailable ({fallback_reason}). Falling back to pure-PyTorch gram path."
            )
    return batched_gram_newton_schulz(
        grad,
        ns_coefficients=schedule,
        ns_steps=len(schedule),
        eps=eps,
        compute_dtype=torch.float16 if compute_dtype is None else compute_dtype,
        reset_iterations=gram_ns_reset_iterations,
    )


def _is_adamw_by_name(name: str, extra_patterns: Sequence[str]) -> bool:
    lname = name.lower()
    for pat in _DEFAULT_ADAMW_NAME_PATTERNS:
        if pat in lname:
            return True
    for pat in extra_patterns:
        if pat and pat.lower() in lname:
            return True
    return False


def _is_muon_eligible_ndim(param: Tensor) -> bool:
    """Return True for dense linears and 3D MoE expert stacks."""
    return param.ndim in (2, 3)


def split_muon_adamw_params(
    model: "nn.Module",
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    extra_adamw_name_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[Tensor], List[Tensor], List[str], List[str]]:
    """Split model parameters into Muon-eligible weights and AdamW fallback weights."""
    no_decay_modules = no_decay_modules or []
    no_decay_params = no_decay_params or []
    extra_patterns = list(extra_adamw_name_patterns or ())

    forced_adamw_fqns: set = set()
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        is_embedding = isinstance(module, nn.Embedding)
        is_no_decay = cls_name in no_decay_modules
        if is_embedding or is_no_decay:
            for pname, _p in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{pname}" if module_name else pname
                forced_adamw_fqns.add(fqn)

    muon_params: List[Tensor] = []
    adamw_params: List[Tensor] = []
    muon_names: List[str] = []
    adamw_names: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        muon_ok = _is_muon_eligible_ndim(param)
        forced_adamw = (
            (not muon_ok)
            or name in forced_adamw_fqns
            or _is_adamw_by_name(name, extra_patterns)
            or any(p and p.lower() in name.lower() for p in no_decay_params)
        )
        if forced_adamw:
            adamw_params.append(param)
            adamw_names.append(name)
        else:
            muon_params.append(param)
            muon_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names


_KIND_LOCAL = "local"
_KIND_FSDP_GATHER_2D = "fsdp_gather_2d"
_KIND_MOE_LOCAL_3D = "moe_local_3d"
_KIND_MOE_GATHER_3D = "moe_gather_3d"


def _shard_dims(p: DTensor) -> List[int]:
    """Return the list of tensor dims along which ``p`` is sharded."""
    return [pl.dim for pl in p.placements if isinstance(pl, Shard)]


def _classify_param(p: Tensor) -> str:
    """Return one of ``_KIND_*`` describing how Muon should treat ``p``."""
    if not isinstance(p, DTensor):
        return _KIND_LOCAL

    shard_dims = _shard_dims(p)
    if not shard_dims:
        return _KIND_LOCAL

    if p.ndim == 2:
        return _KIND_FSDP_GATHER_2D

    if p.ndim == 3:
        if all(d == 0 for d in shard_dims):
            return _KIND_MOE_LOCAL_3D
        return _KIND_MOE_GATHER_3D

    raise ValueError(
        f"DistributedMuon got an unexpected param rank {p.ndim} "
        f"(shape={tuple(p.shape)}). Only 2D and 3D params are supported."
    )


def _full_grad(grad: Tensor) -> Tensor:
    """Return a replicated tensor, all-gathering DTensor gradients if needed."""
    if isinstance(grad, DTensor):
        return grad.full_tensor()
    return grad


def _wrap_full_as_dtensor_like(full: Tensor, ref: Tensor) -> Tensor:
    """Wrap ``full`` as a DTensor with ``ref``'s placements."""
    if not isinstance(ref, DTensor):
        return full

    mesh = ref.device_mesh
    replicated = DTensor.from_local(
        full,
        device_mesh=mesh,
        placements=[Replicate()] * mesh.ndim,
        run_check=False,
    )
    return replicated.redistribute(device_mesh=mesh, placements=ref.placements)


class DistributedMuon(Optimizer):
    """Muon optimizer that handles dense and 3D-MoE FSDP2-sharded DTensors.

    Matches :class:`torch.optim.Muon` for 2D params and extends to batched
    Newton-Schulz for 3D MoE expert stacks. Communication is added only where
    the NS iteration needs to see the full ``[..., M, K]`` matrix.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: Optional[str] = None,
        ns_implementation: str = "gram_quack",
        gram_ns_reset_iterations: Sequence[int] = (2,),
    ) -> None:
        if not _MUON_AVAILABLE:
            raise RuntimeError(
                f"DistributedMuon requires torch>=2.9 (torch.optim.Muon). Found torch=={torch.__version__}."
            )
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= float(lr):
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= float(momentum):
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= float(weight_decay):
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")
        if ns_implementation not in NS_IMPLEMENTATIONS:
            raise ValueError(f"ns_implementation must be one of {NS_IMPLEMENTATIONS}, got {ns_implementation!r}")

        defaults: Dict[str, Any] = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "ns_implementation": ns_implementation,
            "gram_ns_reset_iterations": tuple(int(i) for i in gram_ns_reset_iterations),
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if not _is_muon_eligible_ndim(p):
                    raise ValueError(
                        "DistributedMuon supports only 2D and 3D parameters; "
                        f"got param with shape {tuple(p.size())}. Route 1D/4D+ "
                        "params (biases, norms, conv weights) to AdamW via "
                        "split_muon_adamw_params."
                    )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_coefficients = tuple(group["ns_coefficients"])
            ns_steps = int(group["ns_steps"])
            eps = float(group["eps"])
            adjust_lr_fn = group["adjust_lr_fn"]
            ns_implementation = str(group.get("ns_implementation", "gram_quack"))
            gram_ns_reset_iterations = tuple(group.get("gram_ns_reset_iterations", (2,)))

            for p in group["params"]:
                if p.grad is None:
                    continue
                if torch.is_complex(p):
                    raise RuntimeError("DistributedMuon does not support complex parameters")
                if p.grad.is_sparse:
                    raise RuntimeError("DistributedMuon does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                buf: Tensor = state["momentum_buffer"]

                grad = p.grad
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf

                kind = _classify_param(p)
                ortho = self._compute_ortho(
                    update,
                    kind,
                    ns_coefficients,
                    ns_steps,
                    eps,
                    ns_implementation=ns_implementation,
                    gram_ns_reset_iterations=gram_ns_reset_iterations,
                )

                lr_shape = p.shape[-2:] if p.ndim >= 2 else p.shape
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, lr_shape)

                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                if isinstance(p, DTensor):
                    target_dtype = p.to_local().dtype
                    if isinstance(ortho, DTensor):
                        update_dt = ortho.to(dtype=target_dtype)
                    else:
                        update_dt = _wrap_full_as_dtensor_like(ortho.to(dtype=target_dtype), p)
                else:
                    update_dt = ortho.to(dtype=p.dtype)

                p.add_(update_dt, alpha=-adjusted_lr)

        return loss

    @staticmethod
    def _compute_ortho(
        update: Tensor,
        kind: str,
        ns_coefficients: Tuple[float, float, float],
        ns_steps: int,
        eps: float,
        ns_implementation: str = "gram_quack",
        gram_ns_reset_iterations: Sequence[int] = (2,),
    ) -> Tensor:
        """Run Newton-Schulz on ``update`` according to its layout kind."""

        def _ns(x: Tensor) -> Tensor:
            return run_newton_schulz(
                x,
                ns_coefficients=ns_coefficients,
                ns_steps=ns_steps,
                eps=eps,
                ns_implementation=ns_implementation,
                gram_ns_reset_iterations=gram_ns_reset_iterations,
            )

        if kind == _KIND_LOCAL:
            return _ns(update)

        if kind == _KIND_FSDP_GATHER_2D:
            return _ns(_full_grad(update))

        if kind == _KIND_MOE_LOCAL_3D:
            assert isinstance(update, DTensor)
            local = update._local_tensor
            local_ortho = _ns(local)
            return DTensor.from_local(
                local_ortho,
                device_mesh=update.device_mesh,
                placements=update.placements,
                run_check=False,
            )

        if kind == _KIND_MOE_GATHER_3D:
            return _ns(_full_grad(update))

        raise ValueError(f"Unknown DistributedMuon kind: {kind!r}")
