"""Opt-in Qwen3.8 logical-matrix Muon with FP32 masters and CPU state offload.

This correctness-first path reconstructs only the current parameter's matrix
shards. Expert-axis shards stay local. It does not alter sequence parallelism,
model parameter layout, or gradient reduction/clipping. Pure Muon parameters
retain an unused zero second-moment tensor for the existing DCP state schema;
this first implementation prioritizes safe resume over CPU memory savings.
"""

import json
import math
import time
from contextlib import contextmanager

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from .cpu_offload_adamw import CPUOffloadAdamW
from .master_fp32_adamw import _local


# Generated with the Polar Express author's optimal_composition(l=1e-3,
# num_iters=8, degree=5, safety_factor_eps=1e-2, cushion=.02).
# https://github.com/NoahAmsel/PolarExpress/blob/main/polar_express.py
# The Qwen report specifies eight steps, but does not publish its coefficient
# table. This pins an explicit reproducible Polar Express schedule, not a claim
# of bitwise equivalence to unpublished Qwen training code.
POLAR_EXPRESS_8 = (
    (8.237312490495555, -23.1577474145582, 16.680568411445915),
    (4.082441999064829, -2.8930477353325843, 0.5252849256975644),
    (3.9263479922546485, -2.854746803476524, 0.5318022422894979),
    (3.2982187133085197, -2.4245419810267106, 0.4863200835884415),
    (2.2970369434552587, -1.6366255812590325, 0.4002628455953631),
    (1.8763805351440381, -1.23478965777222, 0.3589188750166826),
    (1.856442348565278, -1.2132449881003775, 0.3568003487859341),
    (1.8749978971827306, -1.2499957943500184, 0.3749978971672879),
)
RECIPE = "qwen38-polar8-fp32-v1"


@contextmanager
def _strict_matmul(device_type):
    """Disallow autocast and hardware down-conversion during NS only."""
    backend = None
    option = None
    if device_type == "npu":
        backend, option = torch.npu.matmul, "allow_hf32"
    elif device_type == "cuda":
        backend, option = torch.backends.cuda.matmul, "allow_tf32"
    previous = getattr(backend, option) if backend is not None else None
    try:
        if backend is not None:
            setattr(backend, option, False)
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        if backend is not None:
            setattr(backend, option, previous)


def polar_express_fp32(matrix):
    """Eight FP32 polynomial iterations on a single complete logical matrix."""
    if matrix.ndim != 2 or matrix.dtype != torch.float32:
        raise ValueError("Polar Express requires one FP32 matrix")
    with _strict_matmul(matrix.device.type):
        return _polar_express(matrix)


def _polar_express(matrix):
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.mT if transposed else matrix
    x = x / (x.norm() * 1.01 + 1e-14)
    for a, b, c in POLAR_EXPRESS_8:
        gram = x @ x.mT
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x
    return x.mT if transposed else x


def logical_plan(model, name, param):
    """Return exhaustive output-row intervals; reject unknown trainable maps.

    Each interval is (algorithm, start, end); 3D stacks apply the same plan
    independently to each expert. Output projections are whole maps: the Qwen
    report's per-head rule applies to Q/K/V and GDN *input* projections.
    """
    modules = dict(model.named_modules())
    owner_path, _, leaf = name.rpartition(".")
    owner = modules.get(owner_path, model)
    parent_path, _, projection = owner_path.rpartition(".")
    parent = modules.get(parent_path, model)
    rows = param.shape[-2] if param.ndim >= 2 else 1
    adam = [("adamw", 0, rows)]
    if not param.requires_grad:
        raise ValueError("Frozen parameters must not enter the optimizer")
    if "lora_" in name:
        raise ValueError("The Qwen3.8 Muon recipe does not support LoRA")
    if "indexer" in name or ".visual." in name:
        raise ValueError("This recipe requires the QSA indexer and vision tower to remain frozen")
    if isinstance(owner, torch.nn.Embedding):
        if "ngram_embedding" in name:
            raise ValueError("This recipe requires the CPU PLE lookup tables to remain frozen")
        return adam
    if param.ndim < 2 or leaf == "bias" or isinstance(owner, (torch.nn.Conv1d, torch.nn.Conv2d)):
        return adam
    if projection in (
        "lm_head",
        "gate",
        "router",
        "shared_expert_gate",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
    ):
        return adam
    if projection in ("q_proj", "k_proj", "v_proj") and hasattr(parent, "head_dim"):
        width = int(parent.head_dim)
        if width <= 0 or rows % width:
            raise ValueError(f"Invalid attention head layout: {name}")
        if projection == "q_proj":
            if rows % (2 * width):
                raise ValueError(f"Expected interleaved Q/gate rows: {name}")
            return [("muon" if i % 2 == 0 else "adamw", i * width, (i + 1) * width) for i in range(rows // width)]
        return [("muon", i, i + width) for i in range(0, rows, width)]
    if projection == "in_proj_qkv" and hasattr(parent, "head_k_dim"):
        k = int(parent.head_k_dim) * int(parent.num_k_heads)
        v = int(parent.head_v_dim) * int(parent.num_v_heads)
        if rows != 2 * k + v:
            raise ValueError(f"Invalid GDN Q/K/V layout: {name}")
        return [("muon", i, i + int(parent.head_k_dim)) for i in range(0, 2 * k, int(parent.head_k_dim))] + [
            ("muon", i, i + int(parent.head_v_dim)) for i in range(2 * k, rows, int(parent.head_v_dim))
        ]
    if leaf == "gate_up_proj" and param.ndim == 3:
        if rows % 2:
            raise ValueError(f"Odd gate/up projection width: {name}")
        return [("muon", 0, rows // 2), ("muon", rows // 2, rows)]
    if leaf == "down_proj" and param.ndim == 3:
        return [("muon", 0, rows)]
    if isinstance(owner, torch.nn.Linear) and projection in (
        "o_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "key_proj",
        "value_proj",
    ):
        return [("muon", 0, rows)]
    raise ValueError(f"Unclassified trainable parameter: {name}, shape={tuple(param.shape)}")


def add_qwen38_plans(model, groups):
    names = {id(p): n for n, p in model.named_parameters()}
    planned = []
    for group in groups:
        for p in group["params"]:
            plan = logical_plan(model, names[id(p)], p)
            planned.append({**group, "params": [p], "qwen38_plan": json.dumps(plan), "qwen38_recipe": RECIPE})
    return planned


def _matrix_placements(param):
    """Keep whole-expert ownership; reconstruct both matrix axes only."""
    if not isinstance(param, DTensor):
        return None
    placements = []
    for placement in param.placements:
        if isinstance(placement, Shard):
            placements.append(placement if param.ndim == 3 and placement.dim == 0 else Replicate())
        elif isinstance(placement, Replicate):
            placements.append(placement)
        else:
            raise ValueError("Muon requires reduced gradients and Shard/Replicate parameter placements")
    return tuple(placements)


class Qwen38MuonAdamW(CPUOffloadAdamW):
    """Mixed logical-slice updates; never orthogonalize an FSDP matrix fragment.

    Gradients must already be globally synchronized and clipped. FP32 master,
    momentum and Adam second moments retain each parameter's DTensor layout.
    Reconstruction is sequential and bounded by ``max_matrix_parameter_numel``;
    a separate limit bounds each logical matrix's polynomial workspace.
    """

    requires_exact_recipe = True

    def __init__(
        self, *args, momentum=0.95, max_matrix_parameter_numel=134217728, max_logical_matrix_numel=33554432, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.resident_moment_dtype != torch.float32 or self.cpu_moment_dtype != torch.float32:
            raise ValueError("Qwen3.8 Muon requires FP32 master and all optimizer states")
        if not 0 <= momentum < 1:
            raise ValueError("Invalid Muon momentum")
        self.max_matrix_parameter_numel = max_matrix_parameter_numel
        self.max_logical_matrix_numel = max_logical_matrix_numel
        for group in self.param_groups:
            if group.get("qwen38_recipe") != RECIPE or len(group["params"]) != 1:
                raise ValueError("Use add_qwen38_plans to construct exact logical-slice groups")
            group.setdefault("muon_momentum", momentum)
            p = group["params"][0]
            plan = json.loads(group["qwen38_plan"])
            rows = p.shape[-2] if p.ndim >= 2 else 1
            end = 0
            for algorithm, start, stop in plan:
                if algorithm not in ("muon", "adamw") or start != end or stop <= start:
                    raise ValueError("Logical row plan must cover each row exactly once")
                end = stop
            if end != rows:
                raise ValueError("Incomplete logical row plan")
            if any(algorithm == "muon" for algorithm, _, _ in plan):
                placements = _matrix_placements(p)
                shape = (
                    compute_local_shape_and_global_offset(p.shape, p.device_mesh, placements)[0]
                    if placements is not None
                    else p.shape
                )
                if math.prod(shape) > max_matrix_parameter_numel:
                    raise ValueError(f"Muon reconstruction exceeds the explicit workspace limit: {tuple(shape)}")
                if any((stop - start) * p.shape[-1] > max_logical_matrix_numel for _, start, stop in plan):
                    raise ValueError("Muon logical matrix exceeds the explicit workspace limit")

    def _emit_state_receipt(self):
        # CPUOffloadAdamW.step runs only the pure Adam groups. Emit after *all*
        # mixed groups have initialized and updated instead.
        return

    def _emit_step_timing(self):
        # Parent timing covers only fallback groups, not this full optimizer.
        return

    def _emit_complete_receipt(self):
        if self._state_receipt_emitted:
            return
        entries = {}
        for state in self.state.values():
            for key in ("master_weight", "exp_avg", "exp_avg_sq"):
                if key not in state:
                    continue
                t = _local(state[key])
                identity = f"{key}:{t.dtype}:{t.device}"
                entries[identity] = entries.get(identity, 0) + t.numel() * t.element_size()
        print(
            "QWEN38_MUON_STATE "
            + json.dumps(
                dict(
                    recipe=RECIPE,
                    initialized=len(self.state),
                    configured=len(self.param_groups),
                    state_bytes=entries,
                    muon_second_moment="unused_zero_DCP_compatibility",
                    seconds=self.last_step_seconds,
                ),
                sort_keys=True,
            ),
            flush=True,
        )
        self._state_receipt_emitted = True

    def load_state_dict(self, state_dict):
        for saved, current in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for key in ("qwen38_recipe", "qwen38_plan", "muon_momentum"):
                if saved.get(key) != current[key]:
                    raise RuntimeError(f"Muon resume recipe mismatch: {key}; AdamW migration is not implicit")
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        devices = {p.device for g in self.param_groups for p in g["params"] if p.device.type != "cpu"}
        for device in sorted(devices, key=str):
            self._synchronize(device)
        started = time.perf_counter()
        # Reuse the already validated chunked AdamW path for pure fallback maps.
        all_groups = self.param_groups
        adam_groups, muon_groups = [], []
        for group in all_groups:
            plan = json.loads(group["qwen38_plan"])
            (muon_groups if any(a == "muon" for a, _, _ in plan) else adam_groups).append(group)
        try:
            self.param_groups = adam_groups
            super().step()
        finally:
            self.param_groups = all_groups
        for group in muon_groups:
            p = group["params"][0]
            if p.grad is None:
                continue
            if p.grad.is_sparse or p.dtype not in (torch.bfloat16, torch.float32):
                raise RuntimeError("Muon requires dense BF16/FP32 compute parameters")
            if isinstance(p, DTensor) and (not isinstance(p.grad, DTensor) or p.grad.placements != p.placements):
                raise RuntimeError("Muon requires gradients reduced to the parameter shard layout")
            target = torch.device("cpu") if id(p) in self._offloaded else p.device
            state = self.state[p]
            if not state:
                master = p.detach().to(device=target, copy=True).float()
                state.update(
                    step=torch.tensor(0.0),
                    master_weight=master,
                    exp_avg=torch.zeros_like(master),
                    exp_avg_sq=torch.zeros_like(master),
                )
            master, first, second = [_local(state[k]) for k in ("master_weight", "exp_avg", "exp_avg_sq")]
            if any(t.dtype != torch.float32 or t.device != target for t in (master, first, second)):
                raise RuntimeError("Invalid Muon state precision or placement")
            grad = _local(p.grad).to(device=target, dtype=torch.float32)
            state["step"] += 1
            step = int(state["step"].item())
            plan = json.loads(group["qwen38_plan"])
            offset = (
                compute_local_shape_and_global_offset(p.shape, p.device_mesh, p.placements)[1][-2]
                if isinstance(p, DTensor)
                else 0
            )
            update = torch.empty_like(grad)
            beta1, beta2 = group["betas"]
            mu = group["muon_momentum"]
            # Intersect logical rows with the *local* matrix shard. This handles
            # Q and gate sharing a Parameter without duplicate optimizer ownership.
            for algorithm, start, stop in plan:
                lo, hi = max(0, start - offset), min(grad.shape[-2], stop - offset)
                if lo >= hi:
                    continue
                section = (..., slice(lo, hi), slice(None))
                g, m, v = grad[section], first[section], second[section]
                if algorithm == "muon":
                    m.lerp_(g, 1 - mu)
                    update[section] = g.lerp(m, mu)
                else:
                    m.lerp_(g, 1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    update[section] = m / (v.sqrt() / math.sqrt(1 - beta2**step) + group["eps"]) / (1 - beta1**step)
            del grad
            update = update.to(device=p.device)
            placements = _matrix_placements(p)
            if isinstance(p, DTensor):
                distributed_update = DTensor.from_local(
                    update, p.device_mesh, p.placements, shape=p.shape, stride=p.stride(), run_check=False
                )
                complete = distributed_update.redistribute(placements=placements)
                update = complete.to_local()
            # Each polynomial uses one logical matrix. Expert batches are not
            # flattened into a shared matrix and do not require EP all-gathers.
            matrices = update.unbind(0) if p.ndim == 3 else (update,)
            for matrix in matrices:
                for algorithm, start, stop in plan:
                    if algorithm == "muon":
                        block = matrix[start:stop]
                        orthogonal = polar_express_fp32(block)
                        block.copy_(orthogonal.mul_(0.2 * math.sqrt(max(block.shape))))
            if isinstance(p, DTensor):
                update = complete.redistribute(placements=p.placements).to_local()
            update = update.to(device=target)
            master.mul_(1 - group["lr"] * group["weight_decay"]).add_(update, alpha=-group["lr"])
            # FP32 master remains authoritative; only the compute copy is cast.
            _local(p).copy_(master.to(device=p.device, dtype=p.dtype))
        for device in sorted(devices, key=str):
            self._synchronize(device)
        self.last_step_seconds = time.perf_counter() - started
        self._emit_complete_receipt()
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"QWEN38_MUON_STEP seconds={self.last_step_seconds:.6f}", flush=True)
        return loss
