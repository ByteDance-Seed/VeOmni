"""Selective optimizer-only CPU offload with FP32 masters and explicit moment dtypes.

The caller must finish gradient synchronization and global clipping before
``step``. Model weights and gradients stay on their original devices; only
selected groups' master weights and moments live on pageable CPU memory.
Transfers use one bounded BF16/FP32 staging buffer, never a full master copy.
"""

import json
import math
import re
import time

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from .master_fp32_adamw import FullFP32AdamW, _local


def select_cpu_offload_groups(model, groups, patterns):
    """Split groups by full parameter-name regex, preserving optimizer settings."""
    if not patterns:
        raise ValueError("cpu_offload_adamw requires explicit cpu_offload_param_patterns")
    expressions = [re.compile(pattern) for pattern in patterns]
    names = {id(param): name for name, param in model.named_parameters()}
    hits = [0] * len(expressions)
    result = []
    for group in groups:
        selected, resident = [], []
        for param in group["params"]:
            name = names.get(id(param))
            if name is None:
                raise ValueError("Optimizer parameter is absent from model.named_parameters()")
            matches = [bool(expression.search(name)) for expression in expressions]
            for index, matched in enumerate(matches):
                hits[index] += int(matched)
            (selected if any(matches) else resident).append(param)
        for params, offload in ((selected, True), (resident, False)):
            if params:
                result.append({**group, "params": params, "cpu_offload": offload})
    if not all(hits):
        missing = [pattern for pattern, count in zip(patterns, hits) if count == 0]
        raise ValueError(f"CPU offload patterns match no trainable parameters: {missing}")
    return result


def resolve_moment_dtype(dtype):
    """Accept only the explicitly supported storage precisions."""
    if dtype in ("float32", torch.float32):
        return torch.float32
    if dtype in ("bfloat16", torch.bfloat16):
        return torch.bfloat16
    raise ValueError(f"Adam moment dtype must be float32 or bfloat16, got {dtype!r}")


class CPUOffloadAdamW(FullFP32AdamW):
    """FP32-master AdamW with per-group ``cpu_offload`` (default False).

    Resident and CPU-offloaded moment storage dtypes are independent and default
    to FP32. Arithmetic uses FP32 even with BF16 moment storage. The rounded
    moments feed the next step; this is not equivalent to FP32 AdamW training.
    The current run owns placement and precision policy. Resume requires exact
    saved-state dtype agreement; converting BF16 to FP32 cannot recover lost
    bits, while FP32 to BF16 discards them. Neither conversion is implicit.
    """

    requires_exact_state_dtypes = True

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        chunk_size=4194304,
        resident_moment_dtype=torch.float32,
        cpu_moment_dtype=torch.float32,
    ):
        self.resident_moment_dtype = resolve_moment_dtype(resident_moment_dtype)
        self.cpu_moment_dtype = resolve_moment_dtype(cpu_moment_dtype)
        params = list(params)
        if params and isinstance(params[0], dict):
            # Consuming our runtime-only flag must not erase the caller's policy
            # when the same group definitions are used to construct a resume.
            params = [dict(group) for group in params]
        super().__init__(params, lr, betas, eps, weight_decay, chunk_size)
        # Keep storage policy out of serialized param groups. PyTorch's FQN
        # loader expects every current group key in older checkpoints; adding
        # cpu_offload there would prevent loading FullFP32AdamW checkpoints.
        self._offloaded = set()
        for group in self.param_groups:
            policy = group.pop("cpu_offload", False)
            if not isinstance(policy, bool):
                raise ValueError("cpu_offload must be a bool")
            if policy:
                self._offloaded.update(id(param) for param in group["params"])
        self._staging = {}
        self._state_receipt_emitted = False

    def _state_dtype(self, param, name):
        if name == "master_weight":
            return torch.float32
        return self.cpu_moment_dtype if id(param) in self._offloaded else self.resident_moment_dtype

    def load_state_dict(self, state_dict):
        restored = []
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for key, param in zip(saved_group["params"], group["params"], strict=True):
                saved = state_dict["state"].get(key, {})
                if saved:
                    if set(saved) != {"step", "master_weight", "exp_avg", "exp_avg_sq"}:
                        raise RuntimeError("CPU offload requires complete master_weight/exp_avg/exp_avg_sq/step state")
                    for name in ("master_weight", "exp_avg", "exp_avg_sq"):
                        expected_dtype = self._state_dtype(param, name)
                        if saved[name].dtype != expected_dtype:
                            raise RuntimeError(
                                f"Optimizer checkpoint dtype mismatch for {name}: saved={saved[name].dtype}, "
                                f"configured={expected_dtype}; explicit precision migration is required"
                            )
                        if tuple(saved[name].shape) != tuple(param.shape):
                            raise RuntimeError(f"Shape mismatch for optimizer state {name}")
                restored.append((param, saved, id(param) in self._offloaded))
        # PyTorch's normal loader casts all states to the compute parameter's
        # dtype/device. Even restoring afterwards transiently allocates the full
        # optimizer on the accelerator. Load group metadata without any tensors.
        Optimizer.load_state_dict(self, {**state_dict, "state": {}})
        for param, saved, offload in restored:
            if not saved:
                continue
            target = torch.device("cpu") if offload else param.device
            state = {"step": torch.as_tensor(saved["step"]).detach().cpu().clone()}
            for name in ("master_weight", "exp_avg", "exp_avg_sq"):
                tensor = saved[name]
                if isinstance(param, DTensor) != isinstance(tensor, DTensor):
                    raise RuntimeError("Optimizer checkpoint DTensor layout does not match parameter")
                if isinstance(param, DTensor):
                    if tensor.placements != param.placements or tensor.device_mesh != param.device_mesh:
                        raise RuntimeError("Optimizer checkpoint DTensor mesh/placements do not match parameter")
                # Reuse DCP's already correctly placed buffers. Cloning every
                # resident state here would transiently double accelerator
                # optimizer memory during resume.
                state[name] = tensor.detach().to(device=target, dtype=self._state_dtype(param, name))
            self.state[param] = state

    def _emit_state_receipt(self):
        if self._state_receipt_emitted:
            return
        entries = {}
        for state in self.state.values():
            if not state:
                continue
            for name in ("master_weight", "exp_avg", "exp_avg_sq"):
                tensor = _local(state[name])
                identity = (name, str(tensor.dtype), str(tensor.device))
                entry = entries.setdefault(
                    identity, dict(state=name, dtype=str(tensor.dtype), device=str(tensor.device), numel=0, bytes=0)
                )
                entry["numel"] += tensor.numel()
                entry["bytes"] += tensor.numel() * tensor.element_size()
        if not entries:
            return
        payload = dict(
            optimizer="CPUOffloadAdamW",
            scope="this_optimizer_actual_initialized_states_only",
            initialized_parameter_tensors=sum(bool(state) for state in self.state.values()),
            configured_parameter_tensors=sum(len(group["params"]) for group in self.param_groups),
            configured_offloaded_parameter_tensors=len(self._offloaded),
            learning_rates=[group["lr"] for group in self.param_groups],
            optimizer_step_seconds=self.last_step_seconds,
            timing_scope="this_suboptimizer_current_rank_device_synchronized",
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            cpu_state_bytes=sum(item["bytes"] for item in entries.values() if item["device"].startswith("cpu")),
            accelerator_state_bytes=sum(
                item["bytes"] for item in entries.values() if not item["device"].startswith("cpu")
            ),
            local_state=list(entries.values()),
        )
        print("AI4SE_OPTIMIZER_STATE_RECEIPT " + json.dumps(payload, sort_keys=True), flush=True)
        self._state_receipt_emitted = True

    def _buffer(self, tensor, size):
        key = (tensor.device, tensor.dtype)
        buffer = self._staging.get(key)
        if buffer is None or buffer.numel() < size:
            # Explicit accelerator pinning is needed on torch_npu; CPU-only
            # tests deliberately do not request pinned memory.
            buffer = torch.empty(size, dtype=tensor.dtype, device="cpu")
            if tensor.device.type != "cpu":
                buffer = buffer.pin_memory(device=tensor.device.type)
            self._staging[key] = buffer
        return buffer[:size]

    @staticmethod
    def _synchronize(device):
        if device.type != "cpu":
            getattr(torch, device.type).synchronize(device)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        devices = {
            param.device
            for group in self.param_groups
            for param in group["params"]
            if param.grad is not None and param.device.type != "cpu"
        }
        for device in sorted(devices, key=str):
            self._synchronize(device)
        started = time.perf_counter()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                offload = id(param) in self._offloaded
                if param.grad.is_sparse or param.dtype not in (torch.bfloat16, torch.float32):
                    raise RuntimeError("CPUOffloadAdamW requires dense BF16/FP32 parameters and gradients")
                weight, gradient = _local(param), _local(param.grad)
                if not weight.is_contiguous() or not gradient.is_contiguous():
                    raise RuntimeError("CPUOffloadAdamW requires contiguous local shards")
                state = self.state[param]
                if not state:
                    target = torch.device("cpu") if offload else param.device
                    # Transfer BF16 first: do not create a full FP32 NPU master.
                    master = param.detach().to(device=target, copy=True).float()
                    state.update(
                        step=torch.tensor(0.0),
                        master_weight=master,
                        exp_avg=torch.zeros_like(master, dtype=self._state_dtype(param, "exp_avg")),
                        exp_avg_sq=torch.zeros_like(master, dtype=self._state_dtype(param, "exp_avg_sq")),
                    )
                tensors = [_local(state[key]) for key in ("master_weight", "exp_avg", "exp_avg_sq")]
                target = torch.device("cpu") if offload else weight.device
                state_names = ("master_weight", "exp_avg", "exp_avg_sq")
                if any(
                    t.dtype != self._state_dtype(param, name) or t.device != target or not t.is_contiguous()
                    for name, t in zip(state_names, tensors, strict=True)
                ):
                    raise RuntimeError("Invalid CPUOffloadAdamW state dtype/device/layout")
                w, g, master, m, v = [tensor.view(-1) for tensor in (weight, gradient, *tensors)]
                state["step"] += 1
                step = int(state["step"].item())
                correction1 = 1 - beta1**step
                correction2 = math.sqrt(1 - beta2**step)
                for start in range(0, w.numel(), group["chunk_size"]):
                    section = slice(start, start + group["chunk_size"])
                    staging = None
                    if offload and g.device.type != "cpu":
                        staging = self._buffer(g, g[section].numel())
                        staging.copy_(g[section], non_blocking=True)
                        self._synchronize(g.device)
                        grad = staging.float()
                    else:
                        grad = g[section].float()
                    first = m[section].float().clone()
                    second = v[section].float().clone()
                    first.lerp_(grad, 1 - beta1)
                    second.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m[section].copy_(first)
                    v[section].copy_(second)
                    denom = second.sqrt().div_(correction2).add_(group["eps"])
                    effective = master[section]
                    effective.mul_(1 - group["lr"] * group["weight_decay"])
                    effective.addcdiv_(first, denom, value=-group["lr"] / correction1)
                    if staging is not None:
                        staging.copy_(effective)
                        w[section].copy_(staging, non_blocking=True)
                        self._synchronize(w.device)
                    else:
                        w[section].copy_(effective)
        for device in sorted(devices, key=str):
            self._synchronize(device)
        self.last_step_seconds = time.perf_counter() - started
        self._emit_state_receipt()
        if self._offloaded and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            print(
                f"AI4SE_CPU_OPTIMIZER_STEP global_rank0_suboptimizer_seconds={self.last_step_seconds:.6f}", flush=True
            )
        return loss
