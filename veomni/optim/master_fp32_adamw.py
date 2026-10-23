"""Opt-in AdamW with FP32 master weights and BF16 moments.

Only local shards are touched; state tensors retain the parameter DTensor layout.
BF16 parameters are compute copies. The FP32 master is updated directly,
including decoupled weight decay; only the compute copy is rounded.
"""

import math

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _align_state_to_parameter(tensor, param):
    """Restore an ExtraParallel-dropped DTensor to the parameter's runtime mesh.

    DCP stores ExtraParallel state on the temporary full ``(fsdp, extra)`` mesh,
    then drops the extra dimension before handing it back to the optimizer. The
    resulting local shard is correct, but its mesh and global metadata are not
    equal to the parameter's runtime layout. Rebuild that DTensor around the
    same local storage so a custom optimizer loader can consume a real cold
    resume without silently changing the state layout.
    """
    if not isinstance(param, DTensor):
        if isinstance(tensor, DTensor):
            raise RuntimeError("Optimizer checkpoint DTensor layout does not match parameter")
        return tensor
    if not isinstance(tensor, DTensor):
        raise RuntimeError("Optimizer checkpoint DTensor layout does not match parameter")
    if tensor.device_mesh == param.device_mesh and tensor.placements == param.placements:
        return tensor

    # A single ExtraParallel dimension is removed on load. Keep the FSDP/HSDP
    # prefix placements and reject every other mismatch rather than guessing.
    if len(tensor.placements) + 1 != len(param.placements):
        raise RuntimeError("Optimizer checkpoint DTensor mesh/placements do not match parameter")
    if tuple(tensor.placements) != tuple(param.placements[: len(tensor.placements)]):
        raise RuntimeError("Optimizer checkpoint DTensor mesh/placements do not match parameter")
    local = tensor.to_local()
    if local.shape != param.to_local().shape:
        raise RuntimeError("Optimizer checkpoint DTensor local shape does not match parameter")
    if tensor.device_mesh.device_type != param.device_mesh.device_type:
        raise RuntimeError("Optimizer checkpoint DTensor device type does not match parameter")

    if local.device.type == "cpu" and param.device_mesh.device_type != "cpu":
        metadata = DTensor.from_local(
            local.to("meta"),
            param.device_mesh,
            param.placements,
            run_check=False,
            shape=param.shape,
            stride=param.stride(),
        )
        return DTensor(local, metadata._spec, requires_grad=False)
    return DTensor.from_local(
        local,
        param.device_mesh,
        param.placements,
        run_check=False,
        shape=param.shape,
        stride=param.stride(),
    )


class MasterFP32AdamW(Optimizer):
    requires_exact_state_dtypes = True
    moment_dtype = torch.bfloat16

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, chunk_size=4194304):
        if lr < 0 or eps < 0 or weight_decay < 0 or chunk_size <= 0:
            raise ValueError("Invalid AdamW hyperparameters")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("Invalid AdamW betas")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, chunk_size=chunk_size))

    def load_state_dict(self, state_dict):
        preserved = []
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for key, param in zip(saved_group["params"], group["params"], strict=True):
                saved = state_dict["state"].get(key, {})
                if saved and set(saved) != {"step", "master_weight", "exp_avg", "exp_avg_sq"}:
                    raise RuntimeError("Legacy AdamW checkpoint requires explicit migration")
                for name in ("master_weight", "exp_avg", "exp_avg_sq"):
                    if name in saved:
                        expected_dtype = torch.float32 if name == "master_weight" else self.moment_dtype
                        saved[name] = _align_state_to_parameter(saved[name], param)
                        if saved[name].dtype != expected_dtype or saved[name].shape != param.shape:
                            raise RuntimeError(f"Optimizer state {name} requires explicit precision/shape migration")
                        preserved.append((param, name, saved[name]))
        super().load_state_dict(state_dict)
        # Restore original tensors, not the BF16-rounded copies made by PyTorch.
        for param, name, tensor in preserved:
            dtype = torch.float32 if name == "master_weight" else self.moment_dtype
            self.state[param][name] = tensor.to(device=param.device, dtype=dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("MasterFP32AdamW does not support sparse gradients")
                if p.dtype not in (torch.bfloat16, torch.float32):
                    raise RuntimeError("MasterFP32AdamW supports BF16/FP32 parameters only")
                state = self.state[p]
                if not state:
                    state["step"] = torch.tensor(0.0)
                    for key in ("exp_avg", "exp_avg_sq"):
                        state[key] = torch.zeros_like(p, dtype=self.moment_dtype)
                    state["master_weight"] = p.detach().to(dtype=torch.float32, copy=True)
                if "master_weight" not in state:
                    raise RuntimeError("Missing master_weight state: legacy checkpoint migration must be explicit")
                local = [_local(t) for t in (p, p.grad, state["exp_avg"], state["exp_avg_sq"], state["master_weight"])]
                if not all(t.is_contiguous() for t in local):
                    raise RuntimeError("MasterFP32AdamW requires contiguous local shards")
                w, g, m, v, c = [t.view(-1) for t in local]
                state["step"] += 1
                step = int(state["step"].item())
                lr = group["lr"]
                correction1 = 1 - beta1**step
                correction2 = math.sqrt(1 - beta2**step)
                for start in range(0, w.numel(), group["chunk_size"]):
                    section = slice(start, start + group["chunk_size"])
                    grad = g[section].float()
                    first = m[section].float().clone()
                    second = v[section].float().clone()
                    first.lerp_(grad, 1 - beta1)
                    second.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m[section].copy_(first)
                    v[section].copy_(second)
                    denom = second.sqrt().div_(correction2).add_(group["eps"])
                    effective = c[section]
                    effective.mul_(1 - lr * group["weight_decay"])
                    effective.addcdiv_(first, denom, value=-lr / correction1)
                    w[section].copy_(effective)
        return loss


class FullFP32AdamW(MasterFP32AdamW):
    """FP32 master and FP32 moments with BF16/FP32 compute parameters."""

    moment_dtype = torch.float32
