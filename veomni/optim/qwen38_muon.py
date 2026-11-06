# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Opt-in Qwen3.8 logical-matrix Muon with FP32 masters and CPU state offload.

This correctness-first path reconstructs only the current parameter's matrix
shards. Expert-axis shards stay local. It does not alter sequence parallelism,
model parameter layout, or gradient reduction/clipping. Pure Muon parameters
retain an unused zero second-moment tensor for the existing DCP state schema;
this first implementation prioritizes safe resume over CPU memory savings.
"""

import json
import math
import os
from contextlib import contextmanager
from functools import wraps

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from ..utils.dist_utils import raise_if_any_rank_failed
from .checkpoint_adapter import register_rank_local_adamw
from .cpu_offload_adamw import CPUOffloadAdamW
from .file_backed_state import FileBackedFP32State
from .master_fp32_adamw import FullFP32AdamW, MasterFP32AdamW, _local, _wrap_local_like_parameter


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


def _check_cuda_fp32_policy():
    """Fail closed unless CUDA matmul is already configured for IEEE FP32.

    PyTorch's legacy and hierarchical precision APIs cannot always be mixed,
    and their getters do not reveal whether a value was inherited. Do not
    mutate a process-wide precision policy that cannot be restored faithfully.
    """
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("Qwen3.8 Muon requires IEEE FP32; forced cuBLAS TF32 is unsupported")
    try:
        backend = torch.backends.cuda.matmul
        precision = getattr(backend, "fp32_precision", None)
        allow_tf32 = backend.allow_tf32
        mode = torch.get_float32_matmul_precision()
    except RuntimeError as exc:
        raise RuntimeError(
            "Qwen3.8 Muon cannot safely inspect mixed CUDA matmul precision APIs; "
            "configure IEEE FP32 in a fresh process"
        ) from exc
    if precision not in (None, "none", "ieee") or allow_tf32 or mode != "highest":
        raise RuntimeError("Qwen3.8 Muon requires CUDA matmul IEEE FP32 (highest), without TF32")


def _cuda_step_preflight(optimizers, group, *, closure=False):
    """Validate policy on every participant before any child optimizer step."""
    error = None
    try:
        _check_cuda_fp32_policy()
        import torch.optim.optimizer as optimizer_module

        if (
            closure
            or getattr(optimizer_module, "_global_optimizer_pre_hooks", None)
            or getattr(optimizer_module, "_global_optimizer_post_hooks", None)
        ):
            raise RuntimeError("Qwen3.8 Muon CUDA steps do not support closures or global optimizer step hooks")
        if any(
            getattr(opt, "_optimizer_step_pre_hooks", None) or getattr(opt, "_optimizer_step_post_hooks", None)
            for opt in optimizers
        ):
            raise RuntimeError("Qwen3.8 Muon CUDA steps do not support optimizer step hooks")
        supported_children = (Qwen38MuonAdamW, torch.optim.AdamW, CPUOffloadAdamW, FullFP32AdamW, MasterFP32AdamW)
        if any(type(opt) not in supported_children for opt in optimizers):
            raise RuntimeError("Qwen3.8 Muon CUDA steps do not support custom optimizer children")
    except Exception as exc:
        error = exc

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        if group is None:
            raise RuntimeError("Distributed Qwen3.8 Muon CUDA steps require an explicit precision process group")
        from veomni.utils.dist_utils import raise_if_any_rank_failed

        raise_if_any_rank_failed(error, "checking Qwen3.8 Muon CUDA precision", group=group)
    elif error is not None:
        raise error


def _optimizer_uses_cuda(optimizer):
    return any(param.device.type == "cuda" for group in optimizer.param_groups for param in group["params"])


@contextmanager
def _strict_matmul(device_type):
    """Disallow autocast and hardware down-conversion during NS only."""
    backend = None
    option = None
    if device_type == "npu":
        backend, option = torch.npu.matmul, "allow_hf32"
    elif device_type == "cuda":
        _check_cuda_fp32_policy()
    previous = getattr(backend, option) if backend is not None else None
    try:
        if backend is not None:
            setattr(backend, option, "ieee" if option == "fp32_precision" else False)
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


def _is_native_lora_weight(modules, name, param):
    """Recognize native dense or MoE adapter A/B weights, not arbitrary names."""
    parts = name.split(".")
    if len(parts) < 3 or parts[-3] not in ("lora_A", "lora_B") or parts[-1] != "weight":
        return False

    from ..lora.layers import LoraLinear
    from ..lora.moe_layers import LoraIndependentExperts, LoraSharedExperts

    wrapper_path = ".".join(parts[:-3])
    wrapper = modules.get(wrapper_path)
    if isinstance(wrapper, LoraLinear):
        expected_ndim = 2
    else:
        experts_path, _, projection = wrapper_path.rpartition(".")
        experts = modules.get(experts_path)
        if projection not in ("gate_proj", "up_proj", "down_proj"):
            return False
        if isinstance(experts, LoraSharedExperts):
            expected_ndim = 2
        elif isinstance(experts, LoraIndependentExperts):
            expected_ndim = 3
        else:
            return False
    factors = getattr(wrapper, parts[-3], None)
    return (
        isinstance(factors, torch.nn.ModuleDict)
        and parts[-2] in factors
        and getattr(factors[parts[-2]], "weight", None) is param
        and param.ndim == expected_ndim
    )


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
    if "indexer" in name or ".visual." in name:
        raise ValueError("This recipe requires the QSA indexer and vision tower to remain frozen")
    if "lora_" in name:
        if _is_native_lora_weight(modules, name, param):
            return adam
        raise ValueError(f"Unsupported LoRA parameter for the Qwen3.8 Muon recipe: {name}")
    if isinstance(owner, torch.nn.Embedding):
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
            metadata = {}
            if "ngram_embedding" in names[id(p)]:
                if group.get("weight_decay") != 0.0:
                    raise ValueError(
                        "Trainable PLE lookup tables require explicit zero weight decay; set no_decay_params accordingly"
                    )
                metadata["qwen38_ple_lookup"] = True
            planned.append(
                {**group, **metadata, "params": [p], "qwen38_plan": json.dumps(plan), "qwen38_recipe": RECIPE}
            )
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


def _validate_file_muon_gradient_traffic(groups, state=None):
    """Check Muon communication participants without entering untrusted subgroups."""
    if not torch.distributed.is_initialized():
        return
    plans, preparation_error = [], None
    try:
        for index, group in enumerate(groups):
            if not any(algorithm == "muon" for algorithm, _, _ in json.loads(group["qwen38_plan"])):
                continue
            param = group["params"][0]
            if not isinstance(param, DTensor):
                continue
            target = _matrix_placements(param)
            local_cpu_matrix = param.device.type == "cpu" and hasattr(param, "_veomni_optimizer_compute_device")
            grad = param.grad
            signature = (
                index,
                grad is not None,
                str(param.device.type),
                param.device.type,
                local_cpu_matrix,
                tuple(map(str, param.placements)),
                tuple(map(str, target)),
                tuple(map(str, grad.placements))
                if isinstance(grad, DTensor)
                else ("none" if grad is None else "plain"),
                float(_local(state[param]["step"]).item()) if state is not None and grad is not None else None,
            )
            for axis, (placement, wanted) in enumerate(zip(param.placements, target, strict=True)):
                if placement == wanted:
                    continue
                process_group = param.device_mesh.get_group(axis)
                ranks = tuple(torch.distributed.get_process_group_ranks(process_group))
                if len(ranks) <= 1:
                    continue
                # Rank membership alone does not identify a process group:
                # two groups may have the same members but different channels.
                # group_name is assigned collectively when the group is created.
                group_name = process_group.group_name
                if not isinstance(group_name, str) or not group_name:
                    raise RuntimeError("Muon communication group has no stable identity")
                plans.append((index, axis, ranks, group_name, signature))
    except BaseException as exc:
        preparation_error = exc
    raise_if_any_rank_failed(preparation_error, "preparing Muon communication checks")
    world_plans = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(world_plans, tuple(plans))
    group_error = None
    try:
        by_rank = [{(row[0], row[1]): row for row in entries} for entries in world_plans]
        for rank, entries in enumerate(world_plans):
            if len(by_rank[rank]) != len(entries):
                raise RuntimeError("Duplicate Muon communication plan")
            for row in entries:
                index, axis, ranks, _, _ = row
                if rank not in ranks or len(set(ranks)) != len(ranks):
                    raise RuntimeError("Invalid Muon communication group membership")
                if any(peer < 0 or peer >= len(by_rank) or by_rank[peer].get((index, axis)) != row for peer in ranks):
                    raise RuntimeError("Muon gradient participation or communication group differs across ranks")
    except BaseException as exc:
        group_error = exc
    raise_if_any_rank_failed(group_error, "checking Muon gradient participation")


class Qwen38MuonAdamW(CPUOffloadAdamW):
    """Mixed logical-slice updates; never orthogonalize an FSDP matrix fragment.

    Gradients must already be globally synchronized and clipped. FP32 master,
    momentum and Adam second moments retain each parameter's DTensor layout.
    Reconstruction is sequential and bounded by ``max_matrix_parameter_numel``;
    a separate limit bounds each logical matrix's polynomial workspace.
    """

    requires_exact_recipe = True

    def _patch_step_function(self):
        """Place CUDA preflight outside PyTorch's optimizer step-hook wrapper."""
        super()._patch_step_function()
        cls = type(self)
        wrapped_step = cls.step
        if getattr(wrapped_step, "_qwen38_outer_preflight", None) is wrapped_step:
            return

        @wraps(wrapped_step)
        def guarded_step(optimizer, *args, **kwargs):
            closure = args[0] if args else kwargs.get("closure")
            if _optimizer_uses_cuda(optimizer) and not getattr(optimizer, "_cuda_parent_preflight", False):
                _cuda_step_preflight(
                    [optimizer], getattr(optimizer, "_cuda_precision_group", None), closure=closure is not None
                )
            direct_file_profile = optimizer._ple_world_profile and not optimizer._ple_parent_preflight
            if direct_file_profile:
                optimizer._prepare_file_step(closure)
                optimizer._ple_parent_preflight = True
            try:
                return wrapped_step(optimizer, *args, **kwargs)
            finally:
                if direct_file_profile:
                    optimizer._ple_parent_preflight = False

        guarded_step._qwen38_outer_preflight = guarded_step
        cls.step = guarded_step

    def __init__(
        self,
        *args,
        momentum=0.95,
        max_matrix_parameter_numel=134217728,
        max_logical_matrix_numel=33554432,
        cuda_precision_group=None,
        ple_moments_scratch_dir=None,
        ple_moments_world_profile=False,
        ple_world_receipt=None,
        ple_factory_permit=None,
        **kwargs,
    ):
        from .ple_world_profile import validate_constructor

        validate_constructor(ple_moments_scratch_dir, ple_moments_world_profile, ple_world_receipt, ple_factory_permit)
        if ple_moments_scratch_dir is not None and not ple_moments_world_profile:
            raise ValueError("File-backed PLE moments require an explicitly enabled full-world optimizer profile")
        super().__init__(*args, **kwargs)
        self._cuda_precision_group = cuda_precision_group
        self._cuda_parent_preflight = False
        if self.resident_moment_dtype != torch.float32 or self.cpu_moment_dtype != torch.float32:
            raise ValueError("Qwen3.8 Muon requires FP32 master and all optimizer states")
        if not 0 <= momentum < 1:
            raise ValueError("Invalid Muon momentum")
        self.max_matrix_parameter_numel = max_matrix_parameter_numel
        self.max_logical_matrix_numel = max_logical_matrix_numel
        self._ple_lookup_ids = {
            id(group["params"][0])
            for group in self.param_groups
            if group.get("qwen38_ple_lookup") and len(group["params"]) == 1
        }
        self._ple_file_mode_enabled = ple_moments_scratch_dir is not None
        self._ple_world_profile = ple_moments_world_profile
        self._ple_scratch_root = ple_moments_scratch_dir
        self._ple_parent_preflight = False
        self._ple_moment_storage = (
            FileBackedFP32State(ple_moments_scratch_dir)
            if self._ple_file_mode_enabled and self._ple_lookup_ids
            else None
        )
        self._ple_state_poisoned = False
        for group in self.param_groups:
            if group.get("qwen38_recipe") != RECIPE or len(group["params"]) != 1:
                raise ValueError("Use add_qwen38_plans to construct exact logical-slice groups")
            if group.get("qwen38_ple_lookup") and group["weight_decay"] != 0.0:
                raise ValueError("Trainable PLE lookup tables require zero weight decay in the final optimizer groups")
            if group.get("qwen38_ple_lookup"):
                p = group["params"][0]
                if self._ple_file_mode_enabled and id(p) not in self._offloaded and p.device.type != "cpu":
                    raise ValueError("File-backed PLE moments require CPU optimizer offload")
                # Owner-local lazy counters cannot be checkpointed as replicated scalars.
                mesh = p.device_mesh.mesh.tolist() if isinstance(p, DTensor) else None
                group["qwen38_ple_step_layout"] = json.dumps(["owner-grid-v1", mesh])
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

    def _allocate_moment(self, param, master, name):
        if self._ple_moment_storage is None or id(param) not in self._ple_lookup_ids:
            return super()._allocate_moment(param, master, name)
        local = _local(master)
        if local.device.type != "cpu" or local.dtype != torch.float32:
            raise RuntimeError("File-backed PLE moments require an FP32 CPU master")
        return _wrap_local_like_parameter(self._ple_moment_storage.zeros(local.shape), param)

    def _allocate_checkpoint_state(self, param, field, dtype):
        if (
            self._ple_moment_storage is None
            or id(param) not in self._ple_lookup_ids
            or field not in ("exp_avg", "exp_avg_sq")
        ):
            return None
        if dtype != torch.float32:
            raise RuntimeError("File-backed PLE moments require FP32 checkpoint state")
        local = _local(param)
        return _wrap_local_like_parameter(self._ple_moment_storage.zeros(local.shape), param)

    def _finish_state_chunk(self, param, state, start, count):
        if self._ple_moment_storage is None or id(param) not in self._ple_lookup_ids:
            return
        try:
            for field in ("exp_avg", "exp_avg_sq"):
                if not self._ple_moment_storage.evict_range(_local(state[field]), start, count):
                    raise RuntimeError("PLE moment lost its file-backed storage: " + field)
        except BaseException:
            self._ple_state_poisoned = True
            raise

    def checkpoint_commit_tensor(self, tensor):
        if self._ple_moment_storage is None:
            return
        try:
            self._ple_moment_storage.evict(_local(tensor))
        except BaseException:
            self._ple_state_poisoned = True
            raise

    def checkpoint_evict_tensor_chunk(self, tensor, offset_bytes, size_bytes):
        if self._ple_moment_storage is None or not self._ple_moment_storage.owns(tensor):
            return
        if not tensor.is_contiguous() or offset_bytes % tensor.element_size() or size_bytes % tensor.element_size():
            raise RuntimeError("Cannot safely evict an irregular file-backed optimizer state")
        try:
            self._ple_moment_storage.evict_range(
                tensor, offset_bytes // tensor.element_size(), size_bytes // tensor.element_size()
            )
        except BaseException:
            self._ple_state_poisoned = True
            raise

    def checkpoint_evict_file_backed_state(self):
        if self._ple_moment_storage is None:
            return
        try:
            self._ple_moment_storage.evict_all()
        except BaseException:
            self._ple_state_poisoned = True
            raise

    def checkpoint_abort_file_backed_state(self):
        if self._ple_file_mode_enabled:
            self._ple_state_poisoned = True

    @property
    def checkpoint_save_requires_state(self):
        return any(group.get("qwen38_ple_lookup") for group in self.param_groups)

    @contextmanager
    def checkpoint_state(self, *, load=False):
        """Fill save coverage without an optimizer step or changing live lazy state.

        Load targets need all buffers; save only fills missing PLE owners. These
        temporary zero-step masters are discarded on resume, after model restore.
        """
        added = []
        try:
            with torch.no_grad():
                for group in self.param_groups:
                    if not load and not group.get("qwen38_ple_lookup"):
                        continue
                    for p in group["params"]:
                        if self.state.get(p):
                            continue
                        target = torch.device("cpu") if id(p) in self._offloaded else p.device
                        master = p.detach().to(device=target, copy=True).float()
                        self.state[p] = dict(
                            step=torch.tensor(0.0, dtype=torch.float32, device="cpu"),
                            master_weight=master,
                            exp_avg=self._allocate_moment(p, master, "exp_avg"),
                            exp_avg_sq=self._allocate_moment(p, master, "exp_avg_sq"),
                        )
                        added.append(p)
            yield
        finally:
            if not load:
                for p in added:
                    self.state.pop(p, None)

    @staticmethod
    def _ple_step_tensor(p, step):
        if not isinstance(p, DTensor):
            return torch.as_tensor(step).detach().cpu().clone()
        shape = torch.Size(p.device_mesh.mesh.shape)
        local = torch.as_tensor(step).detach().cpu().reshape((1,) * len(shape)).clone()
        placements = tuple(Shard(axis) for axis in range(len(shape)))
        metadata = DTensor.from_local(
            local.to("meta"),
            p.device_mesh,
            placements,
            run_check=False,
            shape=shape,
            stride=torch.empty(shape, device="meta").stride(),
        )
        # Keep counters on CPU even when the owning parameter mesh is NPU/CUDA.
        return DTensor(local, metadata._spec, requires_grad=False)

    def state_dict(self):
        if getattr(self, "_veomni_rank_local_checkpoint", False):
            return super().state_dict()
        with self.checkpoint_state():
            result = super().state_dict()
            for saved, group in zip(result["param_groups"], self.param_groups, strict=True):
                if group.get("qwen38_ple_lookup"):
                    for key, p in zip(saved["params"], group["params"], strict=True):
                        state = dict(result["state"][key])
                        state["step"] = self._ple_step_tensor(p, state["step"])
                        result["state"][key] = state
            return result

    def _prepare_load_state_dict(self, state_dict):
        state_dict = {**state_dict, "state": dict(state_dict["state"])}
        if getattr(self, "_veomni_rank_local_checkpoint", False):
            for saved, current in zip(state_dict["param_groups"], self.param_groups, strict=True):
                for key in ("qwen38_recipe", "qwen38_plan", "muon_momentum", "qwen38_ple_lookup"):
                    if saved.get(key) != current.get(key):
                        raise RuntimeError(f"Muon resume recipe mismatch: {key}; AdamW migration is not implicit")
                if current.get("qwen38_ple_lookup"):
                    for key in saved["params"]:
                        entry = state_dict["state"].get(key)
                        if entry is not None:
                            step = entry["step"]
                            if isinstance(step, DTensor) or not torch.is_tensor(step) or step.ndim != 0:
                                raise RuntimeError("PLE rank-local optimizer step must be scalar")
                            if not torch.isfinite(step) or step < 0 or step != step.floor():
                                raise RuntimeError("Invalid PLE optimizer step")
            return state_dict
        user_pre_hook = any(
            not getattr(hook, "_veomni_internal_dcp_filter", False)
            for hook in self._optimizer_load_state_dict_pre_hooks.values()
        )
        for saved, current in zip(state_dict["param_groups"], self.param_groups, strict=True):
            if current.get("qwen38_ple_lookup"):
                if (
                    saved.get("qwen38_ple_lookup") is not True
                    or saved.get("weight_decay") != 0.0
                    or saved.get("qwen38_ple_step_layout") != current["qwen38_ple_step_layout"]
                ):
                    raise RuntimeError(
                        "Muon resume requires the PLE lookup marker and zero weight decay with owner-step layout; explicit migration required"
                    )
                for key, p in zip(saved["params"], current["params"], strict=True):
                    entry = state_dict["state"].get(key)
                    if not entry:
                        continue
                    step = entry["step"]
                    if isinstance(p, DTensor):
                        expected = self._ple_step_tensor(p, 0.0)
                        if (
                            not isinstance(step, DTensor)
                            or step.shape != expected.shape
                            or step.device_mesh != expected.device_mesh
                            or step.placements != expected.placements
                        ):
                            raise RuntimeError("PLE optimizer step requires the exact owner-grid checkpoint layout")
                    elif isinstance(step, DTensor) or step.ndim != 0:
                        raise RuntimeError("PLE optimizer step layout does not match parameter")
                    scalar = _local(step).detach().cpu().reshape(()).clone()
                    if not torch.isfinite(scalar) or scalar < 0 or scalar != scalar.floor():
                        raise RuntimeError("Invalid PLE optimizer step")
                    # A user pre-hook may change state values or aliasing;
                    # retain step-zero state so none of its effects are lost.
                    # Internal DCP cleanup hooks alone preserve lazy allocation.
                    if scalar.item() == 0 and not user_pre_hook:
                        state_dict["state"].pop(key)
                    else:
                        state_dict["state"][key] = {**entry, "step": scalar}
            for key in ("qwen38_recipe", "qwen38_plan", "muon_momentum"):
                if saved.get(key) != current[key]:
                    raise RuntimeError(f"Muon resume recipe mismatch: {key}; AdamW migration is not implicit")
        return state_dict

    @torch.no_grad()
    def _prepare_file_step(self, closure=None):
        """Validate and allocate before any parent optimizer mutates state."""
        from torch.optim.optimizer import _global_optimizer_post_hooks, _global_optimizer_pre_hooks

        error = None
        try:
            if self._ple_state_poisoned:
                raise RuntimeError("File-backed optimizer state failed; resume from a complete checkpoint")
            if (
                closure is not None
                or _global_optimizer_pre_hooks
                or _global_optimizer_post_hooks
                or self._optimizer_step_pre_hooks
                or self._optimizer_step_post_hooks
            ):
                raise ValueError("File-backed PLE optimizer does not support closures or step hooks")
            for group in self.param_groups:
                if not any(algorithm == "muon" for algorithm, _, _ in json.loads(group["qwen38_plan"])):
                    continue
                p = group["params"][0]
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.dtype not in (torch.bfloat16, torch.float32):
                    raise RuntimeError("Muon requires dense BF16/FP32 compute parameters")
                if isinstance(p, DTensor) and (not isinstance(p.grad, DTensor) or p.grad.placements != p.placements):
                    raise RuntimeError("Muon requires gradients reduced to the parameter shard layout")
                if tuple(_local(p.grad).shape) != tuple(_local(p).shape):
                    raise RuntimeError("Muon gradient shard shape differs from its parameter")
                target = torch.device("cpu") if id(p) in self._offloaded else p.device
                state = self.state[p]
                if not state:
                    master = p.detach().to(device=target, copy=True).float()
                    state.update(
                        step=torch.tensor(0.0, dtype=torch.float32, device="cpu"),
                        master_weight=master,
                        exp_avg=torch.zeros_like(master),
                        exp_avg_sq=torch.zeros_like(master),
                    )
                for field in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
                    tensor = _local(state.get(field))
                    if not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
                        raise RuntimeError("Invalid Muon optimizer state: " + field)
                    if field == "step":
                        value = float(tensor.item()) if tensor.ndim == 0 else float("nan")
                        if not math.isfinite(value) or value < 0 or not value.is_integer():
                            raise RuntimeError("Invalid Muon optimizer step")
                    elif tensor.device != target or tuple(tensor.shape) != tuple(_local(p).shape):
                        raise RuntimeError("Invalid Muon state placement or shard shape: " + field)
                    elif isinstance(p, DTensor) and (
                        not isinstance(state[field], DTensor) or state[field].placements != p.placements
                    ):
                        raise RuntimeError("Invalid Muon state shard layout: " + field)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "preparing file-backed Muon state")
        _validate_file_muon_gradient_traffic(self.param_groups, self.state)

    @torch.no_grad()
    def step(self, closure=None):
        try:
            if self._ple_world_profile and not self._ple_parent_preflight:
                self._prepare_file_step(closure)
            elif self._ple_state_poisoned:
                raise RuntimeError("File-backed optimizer state failed; resume from a complete checkpoint")
            return self._step_impl(closure)
        except BaseException:
            if self._ple_file_mode_enabled:
                self._ple_state_poisoned = True
            raise

    def _step_impl(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        devices = {p.device for g in self.param_groups for p in g["params"] if p.device.type != "cpu"}
        for device in sorted(devices, key=str):
            self._synchronize(device)
        # Reuse the already validated chunked AdamW path for pure fallback maps.
        all_groups = self.param_groups
        adam_groups, muon_groups = [], []
        for group in all_groups:
            plan = json.loads(group["qwen38_plan"])
            (muon_groups if any(a == "muon" for a, _, _ in plan) else adam_groups).append(group)
        adam_error = None
        try:
            self.param_groups = adam_groups
            try:
                super().step()
            except BaseException as exc:
                adam_error = exc
        finally:
            self.param_groups = all_groups
        if self._ple_world_profile:
            raise_if_any_rank_failed(adam_error, "updating file-backed PLE optimizer state")
        elif adam_error is not None:
            raise adam_error
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
                    step=torch.tensor(0.0, dtype=torch.float32, device="cpu"),
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
        return loss


@contextmanager
def _rank_local_checkpoint_context(optimizer):
    if getattr(optimizer, "_veomni_rank_local_checkpoint", False):
        raise RuntimeError("Reentrant rank-local optimizer checkpoint context")
    optimizer._veomni_rank_local_checkpoint = True
    try:
        yield
    finally:
        del optimizer._veomni_rank_local_checkpoint


register_rank_local_adamw(
    Qwen38MuonAdamW,
    _rank_local_checkpoint_context,
    lambda optimizer: (
        (
            "qwen38-polar8-v1",
            optimizer.max_matrix_parameter_numel,
            optimizer.max_logical_matrix_numel,
            "file-backed-ple-v1",
        )
        if optimizer._ple_file_mode_enabled
        else ("qwen38-polar8-v1", optimizer.max_matrix_parameter_numel, optimizer.max_logical_matrix_numel)
    ),
    lambda optimizer, param, field: torch.float32 if field == "step" else optimizer._state_dtype(param, field),
    lambda optimizer, param: id(param) in optimizer._offloaded,
    lambda optimizer, param, field, dtype: optimizer._allocate_checkpoint_state(param, field, dtype),
)
