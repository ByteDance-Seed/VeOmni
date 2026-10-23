import argparse
import gc
import importlib
import importlib.metadata
import importlib.util
import math
import os
import subprocess
from dataclasses import dataclass, field
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from packaging.version import Version
from torch.distributed._tensor import DTensor, Shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import ModelArguments, TrainingArguments
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import (
    _init_parallel_state,
    clear_parallel_state,
    get_parallel_state,
    use_parallel_state,
)
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_optimizer
from veomni.utils import helper
from veomni.utils.device import (
    get_device_id,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


_CI_LAUNCH_4 = LaunchSpec(nproc=4)
_CI_LAUNCH_8 = LaunchSpec(nproc=8)

pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


# from veomni.optim.optimizer import build_optimizer

logger = helper.create_logger(__name__)


def _torch_npu_version() -> str:
    """Return the version of torch_npu if installed, otherwise return "0.0.0"."""
    if importlib.util.find_spec("torch_npu") is None:
        return None
    return str(importlib.metadata.version("torch_npu"))


@dataclass
class Argument:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


class ToyMoeAndEmbedModel(torch.nn.Module):
    """
    This toy model with MoE+Embedding module has all param value set to 1
    and all its submodules' forward only returns the sum of all its param
    so whatever the input is, the grad of each param is always 1 after its local backward
    As a result, the MoE forward in this model does not have all2all,
    so EP param grad accumulation across ranks is not real,
    where it only accumulates the ep_fsdp ranks, missing accumulation between ep ranks
    """

    _no_split_modules = ["ToyMoeAndEmbedDecoderLayer", "ToyEmbed"]

    def __init__(self):
        super().__init__()
        self.embed_tokens = ToyEmbed()
        self.bias = torch.nn.Parameter(torch.ones(16), requires_grad=True)
        self.decoder = ToyMoeAndEmbedDecoderLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = (x + self.bias).sum()
        loss = loss + self.embed_tokens() + self.decoder()
        return loss

    def init_weights(self):
        self.embed_tokens.weight.data.fill_(1.0)
        self.decoder.embed_tokens.weight.data.fill_(1.0)
        self.bias.data.fill_(1.0)
        self.decoder.regular_mlp.data.fill_(1.0)
        self.decoder.moe.experts.data.fill_(1.0)

    def get_parallel_plan(self):
        ep_plan = {"decoder.moe.experts": Shard(0)}
        emb_plan = {"embed_tokens.weight": Shard(0), "decoder.embed_tokens.weight": Shard(0)}
        parallel_plan = ParallelPlan(
            extra_parallel_plan={
                "ep": ep_plan,
                "emb": emb_plan,
            }
        )
        parallel_plan.extra_parallel_fsdp_no_shard_module = {
            "ep": {"decoder.moe"},
            "emb": {"embed_tokens", "decoder.embed_tokens"},
        }

        return parallel_plan


class ToyEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(64, 16), requires_grad=True)

    def forward(self) -> torch.Tensor:
        return self.weight.sum()


class ToyMoeAndEmbedDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = ToyEmbed()
        self.regular_mlp = torch.nn.Parameter(torch.ones(64, 16), requires_grad=True)
        self.moe = ToyMoeExperts()

    def forward(self) -> torch.Tensor:
        return self.embed_tokens() + self.regular_mlp.sum() + self.moe()


class ToyMoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.Parameter(torch.ones(64, 16, 32), requires_grad=True)

    def forward(self) -> torch.Tensor:
        return self.experts.sum()


# (label, ep_size, emb_size, dp_replicate_size, cpu_offload)
_FSDP2_4GPU_SCENARIOS: tuple[tuple[str, int, int, int | None, bool], ...] = (
    ("fsdp2_no_extra_no_offload", 1, 1, None, False),
    ("fsdp2_no_extra_cpu_offload", 1, 1, None, True),
    ("fsdp2_ep2_no_offload", 2, 1, None, False),
    ("fsdp2_ep2_cpu_offload", 2, 1, None, True),
    ("fsdp2_ep4_no_offload", 4, 1, None, False),
    ("fsdp2_ep4_cpu_offload", 4, 1, None, True),
    ("fsdp2_emb4_no_offload", 1, 4, None, False),
    ("fsdp2_emb4_cpu_offload", 1, 4, None, True),
    ("fsdp2_ep2_emb2_no_offload", 2, 2, None, False),
    ("fsdp2_ep2_emb2_cpu_offload", 2, 2, None, True),
    ("fsdp2_ep2_emb4_no_offload", 2, 4, None, False),
    ("fsdp2_ep2_emb4_cpu_offload", 2, 4, None, True),
)

_HSDP_8GPU_SCENARIOS: tuple[tuple[str, int, int, int | None, bool], ...] = (
    ("hsdp_no_extra_no_offload", 1, 1, 2, False),
    ("hsdp_no_extra_cpu_offload", 1, 1, 2, True),
    ("hsdp_ep4_no_offload", 4, 1, 2, False),
    ("hsdp_ep4_cpu_offload", 4, 1, 2, True),
    ("hsdp_emb4_no_offload", 1, 4, 2, False),
    ("hsdp_emb4_cpu_offload", 1, 4, 2, True),
    ("hsdp_ep2_emb4_no_offload", 2, 4, 2, False),
    ("hsdp_ep2_emb4_cpu_offload", 2, 4, 2, True),
)


def _hsdp_extra_parallel_buggy_on_npu() -> bool:
    npu_ver = _torch_npu_version()
    return bool(npu_ver and Version(npu_ver) < Version("2.9.0"))


def _run_clip_scenario(
    *,
    label: str,
    world_size: int,
    ep_size: int,
    emb_size: int,
    dp_replicate_size: int | None,
    cpu_offload: bool,
) -> None:
    dp_replicate = dp_replicate_size or 1
    assert world_size % dp_replicate == 0
    dp_shard = world_size // dp_replicate

    state_name = f"clip_w{world_size}_rep{dp_replicate}_ep{ep_size}_emb{emb_size}"
    ps = _init_parallel_state(
        dp_size=world_size,
        dp_replicate_size=dp_replicate,
        dp_shard_size=dp_shard,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        ulysses_size=1,
        extra_parallel_sizes=(ep_size, emb_size),
        extra_parallel_placement_innermost=(False, False),
        extra_parallel_names=("ep", "emb"),
        dp_mode="fsdp2",
        name=state_name,
    )

    with use_parallel_state(ps):
        assert get_parallel_state() is ps
        model_args = ModelArguments(config_path="test")
        torch.manual_seed(42)
        model = ToyMoeAndEmbedModel()
        model = build_parallelize_model(
            model,
            init_device="meta",
            weights_path=None,
            mixed_precision=model_args.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=False,
            enable_fsdp_offload=cpu_offload,
            basic_modules=[],
            enable_reentrant=False,
            enable_forward_prefetch=model_args.accelerator.fsdp_config.forward_prefetch,
            broadcast_model_weights_from_rank0=False,
            max_load_broadcast_size=model_args.accelerator.fsdp_config.max_load_broadcast_size,
        )

        fsdp_group = ps.fsdp_group
        ep_group = ps.extra_parallel_group("ep") if ps.extra_parallel_enabled("ep") else None
        emb_group = ps.extra_parallel_group("emb") if ps.extra_parallel_enabled("emb") else None

        ep_fsdp_group = None
        if ps.extra_parallel_group("ep") and ps.extra_parallel_fsdp_device_mesh["ep"] is not None:
            ep_fsdp_group = ps.extra_parallel_fsdp_device_mesh["ep"]["ep_fsdp"].get_group()
            assert dist.get_world_size(group=ep_fsdp_group) == dp_shard // ep_size, (
                f"[{label}] expected ep_fsdp group size {dp_shard // ep_size}, got {dist.get_world_size(group=ep_fsdp_group)}"
            )

        emb_fsdp_group = None
        if ps.extra_parallel_group("emb") and ps.extra_parallel_fsdp_device_mesh["emb"] is not None:
            emb_fsdp_group = ps.extra_parallel_fsdp_device_mesh["emb"]["emb_fsdp"].get_group()
            assert dist.get_world_size(group=emb_fsdp_group) == dp_shard // emb_size, (
                f"[{label}] expected emb_fsdp group size {dp_shard // emb_size}, got {dist.get_world_size(group=emb_fsdp_group)}"
            )
            if emb_size == 2 and dp_shard == 4:
                assert dist.get_world_size(group=emb_fsdp_group) == 2

        # build optimizer to register ep param groups when ep is enabled
        optimizer = build_optimizer(
            model,
            lr=model_args.optimizer.lr,
            weight_decay=model_args.optimizer.weight_decay,
            fused=True,
            optimizer_type=model_args.optimizer.type,
            no_decay_modules=model_args.optimizer.no_decay_modules,
            no_decay_params=model_args.optimizer.no_decay_params,
        )
        logger.info_rank0(
            "[%s] group sizes - fsdp: %s, ep: %s, ep_fsdp: %s, emb: %s, emb_fsdp: %s",
            label,
            dist.get_world_size(group=fsdp_group) if fsdp_group is not None else None,
            dist.get_world_size(group=ep_group) if ep_group is not None else None,
            dist.get_world_size(group=ep_fsdp_group) if ep_fsdp_group is not None else None,
            dist.get_world_size(group=emb_group) if emb_group is not None else None,
            dist.get_world_size(group=emb_fsdp_group) if emb_fsdp_group is not None else None,
        )
        device_type = get_device_type()
        tensor_device = torch.device(f"{device_type}:{get_device_id()}")
        max_grad_norm = model_args.optimizer.max_grad_norm

        def check_model_param_grad_one_by_one(target_model, expected_grad, ep_expected_grad, emb_expected_grad, msg):
            for name, param in target_model.named_parameters():
                grad = param.grad
                if grad is None:
                    continue
                grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
                if name == "decoder.moe.experts":
                    torch.testing.assert_close(
                        grad_local,
                        torch.full_like(grad_local, ep_expected_grad),
                        atol=1e-6,
                        rtol=1e-6,
                        msg=f"[{label}] ({msg}) Gradient mismatch for {name}, shape {grad_local.shape}, expected {ep_expected_grad}",
                    )
                elif "embed_tokens" in name:
                    torch.testing.assert_close(
                        grad_local,
                        torch.full_like(grad_local, emb_expected_grad),
                        atol=1e-6,
                        rtol=1e-6,
                        msg=f"[{label}] ({msg}) Gradient mismatch for {name}, shape {grad_local.shape}, expected {emb_expected_grad}",
                    )
                else:
                    torch.testing.assert_close(
                        grad_local,
                        torch.full_like(grad_local, expected_grad),
                        atol=1e-6,
                        rtol=1e-6,
                        msg=f"[{label}] ({msg}) Gradient mismatch for {name}, shape {grad_local.shape}, expected {expected_grad}",
                    )

        for step in range(2):
            inputs = torch.ones(1, 16, device=tensor_device)
            loss = model(inputs)
            loss.backward()

            expected = 1.0
            ep_expected = 1.0 / ps.extra_parallel_sizes["ep"]
            emb_expected = 1.0 / ps.extra_parallel_sizes["emb"]
            check_model_param_grad_one_by_one(
                model,
                expected_grad=expected,
                ep_expected_grad=ep_expected,
                emb_expected_grad=emb_expected,
                msg=f"Before clipping step={step}",
            )

            expected_total_grad_norm = math.sqrt(
                (64 * 16 + 64 * 16) * (1 / ps.extra_parallel_sizes["emb"] ** 2)
                + 16
                + 64 * 16
                + (64 * 16 * 32) * (1 / ps.extra_parallel_sizes["ep"] ** 2)
            )
            total_grad_norm_pre_clip = veomni_clip_grad_norm(model, max_grad_norm)

            torch.testing.assert_close(
                total_grad_norm_pre_clip.item(),
                expected=expected_total_grad_norm,
                atol=1e-6,
                rtol=1e-6,
                msg=f"[{label}] step={step} total grad norm mismatch",
            )

            clip_coeff = min(max_grad_norm / expected_total_grad_norm, 1.0)
            ep_clip_coeff = 1.0 / ps.extra_parallel_sizes["ep"] * clip_coeff
            emb_clip_coeff = 1.0 / ps.extra_parallel_sizes["emb"] * clip_coeff
            check_model_param_grad_one_by_one(
                model,
                expected_grad=clip_coeff,
                ep_expected_grad=ep_clip_coeff,
                emb_expected_grad=emb_clip_coeff,
                msg=f"After clipping step={step}",
            )
            model.zero_grad()

        del optimizer, model
        gc.collect()
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["fsdp2_4gpu", "hsdp_8gpu"], required=True)
    parser.add_argument("--reverse-scenarios", action="store_true")
    cli_args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    get_torch_device().set_device(f"{get_device_type()}:{local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())
    world_size = dist.get_world_size()

    if cli_args.suite == "fsdp2_4gpu":
        scenarios = list(_FSDP2_4GPU_SCENARIOS)
    else:
        scenarios = [
            s for s in _HSDP_8GPU_SCENARIOS if not (_hsdp_extra_parallel_buggy_on_npu() and (s[1] > 1 or s[2] > 1))
        ]

    reverse = cli_args.reverse_scenarios or os.environ.get("VEOMNI_REVERSE_SCENARIOS") == "1"
    if reverse:
        scenarios = list(reversed(scenarios))

    try:
        for label, ep_size, emb_size, dp_replicate_size, cpu_offload in scenarios:
            _run_clip_scenario(
                label=label,
                world_size=world_size,
                ep_size=ep_size,
                emb_size=emb_size,
                dp_replicate_size=dp_replicate_size,
                cpu_offload=cpu_offload,
            )
        dist.barrier()
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


def _run_clip_grad_norm_suite(suite: str, nproc: int, reverse_scenarios: bool = False) -> None:
    command = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        "--standalone",
        "tests/utils/test_extra_parallel_clip_grad_norm.py",
        f"--suite={suite}",
    ]
    if reverse_scenarios:
        command.append("--reverse-scenarios")
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


@pytest.mark.launch_plan(provider=_CI_LAUNCH_4)
def test_clip_grad_norm_fsdp2_4gpu():
    """Fused 4-rank FSDP2 worker covering no-extra, EP2, EP4, emb4, and EP2+emb4 across no_offload/cpu_offload."""
    _run_clip_grad_norm_suite("fsdp2_4gpu", nproc=_CI_LAUNCH_4.nproc)


# ---------------------------------------------------------------------------
# HSDP regression: dp = (dp_replicate=2, dp_shard=4) on 8 GPUs.
#
# The grad-norm reduction must skip the replicate dim for BOTH non-ExtraParallel
# params (reduce over `dp_shard_sp`, not `dp_sp`) and ExtraParallel params
# (reduce over `ep_fsdp`/`emb_fsdp`, not the full `*_replicate` mesh). If the
# replicate dim leaks into the sum, the norm is inflated by sqrt(dp_replicate),
# so these reuse the SAME expected total norm as the pure-FSDP cases above.
# ---------------------------------------------------------------------------


@pytest.mark.launch_plan(provider=_CI_LAUNCH_8)
def test_clip_grad_norm_hsdp_8gpu():
    """Fused 8-rank HSDP worker covering (2, 4) no-extra, EP4, emb4, and EP2+emb4 across no_offload/cpu_offload."""
    _run_clip_grad_norm_suite("hsdp_8gpu", nproc=_CI_LAUNCH_8.nproc)


# ===========================================================================
# Single-process inf-norm reduction unit tests (no distributed init).
#
# These complement the 8-GPU torchrun integration tests above by pinning the
# ``norm_type=inf`` reduction maths of ``extra_parallel_fsdp2_clip_grad_norm``
# with multiple ``extra_parallel_names`` buckets. The historical bug was a
# variadic ``torch.maximum`` call that only worked when the combined dict size
# was 1 and crashed with ``TypeError`` on the Mode-2 shared-LoRA case (``ep``
# populated in both ``extra_parallel_total`` and
# ``extra_parallel_replicated_total``). ``get_parallel_state`` is monkey-patched
# so every group / mesh is None and the reduction runs single-process, so these
# run on CPU with no GPUs.
# ===========================================================================


# Direct ``importlib`` lookup: ``veomni.distributed.fsdp2.__init__`` shadows
# the submodule name with ``from .clip_grad_norm import clip_grad_norm`` (the
# function), so ``import veomni.distributed.fsdp2.clip_grad_norm`` would
# resolve to the function, not the module we need to monkey-patch.
cgn_mod = importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")
extra_parallel_fsdp2_clip_grad_norm = cgn_mod.extra_parallel_fsdp2_clip_grad_norm


@dataclass
class _FakeParallelState:
    """Minimal stub for ``get_parallel_state()`` consumed by clip_grad_norm.

    All groups and meshes are ``None`` so ``_fsdp2_reduce_group`` skips
    every ``dist.all_reduce`` -- the reduction runs single-process and
    only the in-process maths is exercised.
    """

    extra_parallel_names: list[str] = field(default_factory=lambda: ["ep"])
    fsdp_group: object | None = None
    extra_parallel_fsdp_device_mesh: dict[str, object | None] = field(default_factory=dict)
    # ``!= "fsdp2"`` makes ``_fsdp_grad_norm_reduce_groups`` return no groups, so
    # the non-ExtraParallel reduction stays single-process (no ``dist`` needed).
    dp_mode: str = "none"

    def extra_parallel_enabled(self, name: str) -> bool:
        return False

    def extra_parallel_group(self, name: str) -> object | None:
        return None

    def extra_parallel_flat_group(self, name: str) -> object | None:
        return None


def _make_model(
    param_groups: dict[str, list[nn.Parameter]],
    replicated_ids: set,
    persistent_ids: set | None = None,
) -> nn.Module:
    """Wrap a ``param_groups`` dict in the duck-typed object the clipper expects."""
    m = nn.Module()
    m._extra_parallel_param_groups = param_groups
    m._ep_replicated_lora_param_ids = replicated_ids
    m._persistent_extra_parallel_param_ids = persistent_ids or set()
    return m


def _param_with_grad(grad: torch.Tensor) -> nn.Parameter:
    p = nn.Parameter(torch.zeros_like(grad), requires_grad=True)
    p.grad = grad.clone()
    return p


@pytest.mark.parametrize("extra_names", [["ep"], ["ep", "emb"]])
def test_inf_norm_with_replicated_bucket_does_not_crash(extra_names):
    """``norm_type=inf`` with both EP and EP-replicated buckets populated.

    Pre-fix: ``torch.maximum(non_ep, *ep.values(), *ep_rep.values())`` is a
    3+ arg call on a 2-arg op -- raises ``TypeError`` at the first crash
    point. Post-fix: iterative ``torch.maximum`` reduces over any number
    of bucket values. The Mode-2 shared-LoRA EP case is exactly the
    smallest failing instance.
    """
    p_non = _param_with_grad(torch.tensor([1.0, -2.0, 0.5]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=list(extra_names),
        extra_parallel_fsdp_device_mesh=dict.fromkeys(extra_names),
    )
    grads_seen = [p_non.grad.abs().max()]

    param_groups: dict[str, list[nn.Parameter]] = {"non_extra_parallel": [p_non]}
    replicated_ids: set = set()
    for i, name in enumerate(extra_names):
        # One EP-bucket param + one EP-replicated-bucket param per axis --
        # for ``extra_names=["ep"]`` this is the Mode-2 shared-LoRA case
        # (1 sharded + 1 replicated == 2 entries that get unpacked alongside
        # ``non_extra_parallel_total``, total 3 args to torch.maximum).
        p_ep = _param_with_grad(torch.tensor([3.0 + i, -1.0, 0.0]))
        p_ep_rep = _param_with_grad(torch.tensor([-(7.0 + i), 0.1, 4.0]))
        param_groups[name] = [p_ep, p_ep_rep]
        replicated_ids.add(id(p_ep_rep))
        grads_seen.append(p_ep.grad.abs().max())
        grads_seen.append(p_ep_rep.grad.abs().max())

    model = _make_model(param_groups, replicated_ids)
    expected = torch.stack(grads_seen).max().to(torch.float32)
    with (
        mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps),
        mock.patch.object(cgn_mod, "get_device_type", return_value="cpu"),
    ):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=math.inf, error_if_nonfinite=False, foreach=False
        )

    # The expected inf-norm is the elementwise max across every grad.
    # Single-process + max_norm=1e9 means no clipping is actually applied,
    # so this just validates the reduction maths.
    assert total.item() == pytest.approx(expected.item(), abs=1e-6), (
        f"inf-norm mismatch with {extra_names=}: got {total.item()}, expected {expected.item()}"
    )


def test_inf_norm_single_axis_no_replicated_still_works():
    """Sanity: the legacy K1+K2==1 happy path still passes after the fix.

    Pre-fix this was the only working combination. Keep it asserted so a
    future refactor that breaks the 1-arg reduction case fails fast.
    """
    p_non = _param_with_grad(torch.tensor([0.25]))
    p_ep = _param_with_grad(torch.tensor([5.0, -6.0]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=["ep"],
        extra_parallel_fsdp_device_mesh={"ep": None},
    )
    model = _make_model({"non_extra_parallel": [p_non], "ep": [p_ep]}, replicated_ids=set())

    with (
        mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps),
        mock.patch.object(cgn_mod, "get_device_type", return_value="cpu"),
    ):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=math.inf, error_if_nonfinite=False, foreach=False
        )

    assert total.item() == pytest.approx(6.0, abs=1e-6)


def test_persistent_extra_parallel_norm_reduces_once_over_flat_mesh():
    """A persistent 2D shard owns unique values and needs one flat-mesh reduction."""
    p_non = _param_with_grad(torch.tensor([12.0]))
    p_persistent = _param_with_grad(torch.tensor([3.0, 4.0]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=["ple"],
        extra_parallel_fsdp_device_mesh={"ple": None},
    )
    fake_ps.extra_parallel_enabled = lambda _name: True
    fake_ps.extra_parallel_group = lambda name: f"{name}_group"
    fake_ps.extra_parallel_flat_group = lambda name: f"{name}_flat_group"
    model = _make_model(
        {"non_extra_parallel": [p_non], "ple": [p_persistent]},
        replicated_ids=set(),
        persistent_ids={id(p_persistent)},
    )
    reduce_calls = []

    def record_reduce(params, norm_type, reduce_groups):
        reduce_calls.append((list(params), list(reduce_groups)))
        return cgn_mod._local_pth_sum(params, float(norm_type))

    with (
        mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps),
        mock.patch.object(cgn_mod, "get_device_type", return_value="cpu"),
        mock.patch.object(cgn_mod, "_fsdp2_reduce_group", side_effect=record_reduce),
    ):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=2.0, error_if_nonfinite=False, foreach=False
        )

    assert total.item() == pytest.approx(13.0, abs=1e-6)
    persistent_calls = [call for call in reduce_calls if len(call[0]) == 1 and call[0][0] is p_persistent]
    assert len(persistent_calls) == 1
    assert persistent_calls[0][1] == [("ple_flat", "ple_flat_group")]


def test_persistent_ple_and_ep_norm_use_independent_reduction_groups():
    """PLE uses one flat reduction while ordinary EP uses EP-FSDP then EP."""

    class _FakeMesh:
        def __init__(self, group):
            self.group = group

        def get_group(self):
            return self.group

    p_non = _param_with_grad(torch.tensor([12.0]))
    p_ple = _param_with_grad(torch.tensor([3.0, 4.0]))
    p_ep = _param_with_grad(torch.tensor([5.0, 12.0]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=["ple", "ep"],
        extra_parallel_fsdp_device_mesh={
            "ple": {"ple_fsdp": _FakeMesh("ple_fsdp_group")},
            "ep": {"ep_fsdp": _FakeMesh("ep_fsdp_group")},
        },
    )
    fake_ps.extra_parallel_enabled = lambda _name: True
    fake_ps.extra_parallel_group = lambda name: f"{name}_group"
    fake_ps.extra_parallel_flat_group = lambda name: f"{name}_flat_group"
    model = _make_model(
        {
            "non_extra_parallel": [p_non],
            "ple": [p_ple],
            "ep": [p_ep],
        },
        replicated_ids=set(),
        persistent_ids={id(p_ple)},
    )
    reduce_calls = []

    def record_reduce(params, norm_type, reduce_groups):
        reduce_calls.append((list(params), list(reduce_groups)))
        return cgn_mod._local_pth_sum(params, float(norm_type))

    with (
        mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps),
        mock.patch.object(cgn_mod, "get_device_type", return_value="cpu"),
        mock.patch.object(cgn_mod, "_fsdp2_reduce_group", side_effect=record_reduce),
    ):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=2.0, error_if_nonfinite=False, foreach=False
        )

    assert total.item() == pytest.approx(math.sqrt(338.0), abs=1e-6)
    ple_calls = [call for call in reduce_calls if len(call[0]) == 1 and call[0][0] is p_ple]
    ep_calls = [call for call in reduce_calls if len(call[0]) == 1 and call[0][0] is p_ep]
    assert len(ple_calls) == 1
    assert ple_calls[0][1] == [("ple_flat", "ple_flat_group")]
    assert len(ep_calls) == 1
    assert ep_calls[0][1] == [("ep_fsdp", "ep_fsdp_group"), ("ep", "ep_group")]


if __name__ == "__main__":
    main()
