import importlib
import importlib.metadata
import importlib.util
import math
import subprocess
from dataclasses import dataclass, field
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from packaging.version import Version
from torch.distributed._tensor import DTensor, Shard

from veomni.arguments import TrainingArguments, parse_args
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_optimizer
from veomni.utils import helper
from veomni.utils.device import (
    get_device_id,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


# from veomni.optim.optimizer import build_optimizer

logger = helper.create_logger(__name__)


def _torch_npu_version() -> str:
    """Return the version of torch_npu if installed, otherwise return "0.0.0"."""
    if importlib.util.find_spec("torch_npu") is None:
        return None
    return str(importlib.metadata.version("torch_npu"))


@dataclass
class Argument:
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


def main():
    dist.init_process_group(backend=get_dist_comm_backend())
    args = parse_args(Argument)

    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    init_parallel_state(
        dp_size=args.train.accelerator.dp_size,
        dp_replicate_size=args.train.accelerator.dp_replicate_size,
        dp_shard_size=args.train.accelerator.dp_shard_size,
        tp_size=args.train.accelerator.tp_size,
        pp_size=args.train.accelerator.pp_size,
        cp_size=args.train.accelerator.cp_size,
        ulysses_size=args.train.accelerator.ulysses_size,
        extra_parallel_sizes=args.train.accelerator.extra_parallel_sizes,
        extra_parallel_placement_innermost=args.train.accelerator.extra_parallel_placement_innermost,
        extra_parallel_names=args.train.accelerator.extra_parallel_names,
        dp_mode=args.train.accelerator.fsdp_config.fsdp_mode,
    )

    model = ToyMoeAndEmbedModel()
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=None,
        mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
        enable_gradient_checkpointing=args.train.gradient_checkpointing.enable,
        enable_fsdp_offload=args.train.accelerator.fsdp_config.offload,
        basic_modules=[],
        enable_reentrant=args.train.gradient_checkpointing.enable_reentrant,
        enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
        max_load_broadcast_size=args.train.accelerator.fsdp_config.max_load_broadcast_size,
    )

    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    fsdp_group = ps.fsdp_group
    ep_group = ps.extra_parallel_group("ep") if ps.extra_parallel_enabled("ep") else None
    emb_group = ps.extra_parallel_group("emb") if ps.extra_parallel_enabled("emb") else None

    ep_fsdp_group = None
    if ps.extra_parallel_group("ep") and ps.extra_parallel_fsdp_device_mesh["ep"] is not None:
        ep_fsdp_group = ps.extra_parallel_fsdp_device_mesh["ep"]["ep_fsdp"].get_group()

    emb_fsdp_group = None
    if ps.extra_parallel_group("emb") and ps.extra_parallel_fsdp_device_mesh["emb"] is not None:
        emb_fsdp_group = ps.extra_parallel_fsdp_device_mesh["emb"]["emb_fsdp"].get_group()

    # build optimizer to register ep param groups when ep is enabled
    _ = build_optimizer(
        model,
        lr=args.train.optimizer.lr,
        weight_decay=args.train.optimizer.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer.type,
        no_decay_modules=args.train.optimizer.no_decay_modules,
        no_decay_params=args.train.optimizer.no_decay_params,
    )
    logger.info_rank0(
        "group sizes - fsdp: %s, ep: %s, ep_fsdp: %s, emb: %s, emb_fsdp: %s",
        dist.get_world_size(group=fsdp_group) if fsdp_group is not None else None,
        dist.get_world_size(group=ep_group) if ep_group is not None else None,
        dist.get_world_size(group=ep_fsdp_group) if ep_fsdp_group is not None else None,
        dist.get_world_size(group=emb_group) if emb_group is not None else None,
        dist.get_world_size(group=emb_fsdp_group) if emb_fsdp_group is not None else None,
    )
    device_type = get_device_type()
    tensor_device = torch.device(f"{device_type}:{get_device_id()}")
    max_grad_norm = args.train.optimizer.max_grad_norm

    def check_model_param_grad_one_by_one(expected_grad, ep_expected_grad, emb_expected_grad, msg):
        # check them one-by-one
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
            logger.info_rank0(f"{msg}: the local grad for {name}: {grad_local}")
            if name == "decoder.moe.experts":
                torch.testing.assert_close(
                    grad_local,
                    torch.full_like(grad_local, ep_expected_grad),
                    atol=1e-6,
                    rtol=1e-6,
                    msg=f"Gradient mismatch for {name}, which has local shape {grad_local.shape}, value {grad_local}, expected value {ep_expected_grad} ",
                )
            elif "embed_tokens" in name:
                torch.testing.assert_close(
                    grad_local,
                    torch.full_like(grad_local, emb_expected_grad),
                    atol=1e-6,
                    rtol=1e-6,
                    msg=f"Gradient mismatch for {name}, which has local shape {grad_local.shape}, value {grad_local}, expected value {emb_expected_grad} ",
                )
            else:
                torch.testing.assert_close(
                    grad_local,
                    torch.full_like(grad_local, expected_grad),
                    atol=1e-6,
                    rtol=1e-6,
                    msg=f"Gradient mismatch for {name}, which has local shape {grad_local.shape}, value {grad_local}, expected value {expected_grad} ",
                )

    total_grad_norm_pre_clip = None
    for step in range(3):
        inputs = torch.ones(1, 16, device=tensor_device)
        loss = model(inputs)
        loss.backward()

        logger.info_rank0("manually checking the initial param grads before any clipping")
        # On GPU, the local gard of each param after local backward is 1.0
        # At loss.backward(), reduce scatter is triggered to **average** grad for the same param against different data input on each fsdp rank
        # By default, this is achieved by dividing sum of param grad on each rank by fsdp size
        # * For example, for pure FSDP2 on 8 GPUs,
        #   * the local grad of each param after backward is  1.0 x 8 (every rank every param local grad is 1.0) / 8 (fsdp size)
        #   * this is trasparent to dtensor, by inferring dtensor's fsdp size from its device mesh
        # * When ep+fsdp2 is enabled, the divide factor for ep params should still be world size (num of data inputs)
        #   * in implementation, since ep param in VeOmni can only see ep_fsdp dim, so we need to override its divide factor
        #   * by applying set_gradient_divide_factor(world_size) for EP modules in torch_parallelize
        # In general, the divide factor for each param should be its num of different input data, which is overall dp size

        # In this test specifically, model forward is unrelated to inputs and the local grad is always 1
        # Since the test toy MoE forward does not have all2all like real ones,
        # the data of ep params would see have only ep_fsdp num
        # * If there is no grad divide factor set, the default grad divide factor is ep_fsdp_size, the local grad after backward is still 1
        # * Since we set grad divide factor to world_size (= fsdp_size = ep size * ep_fsdp_size), we expect grad here to be 1/ep_size
        expected = 1.0
        ep_expected = 1.0 / ps.extra_parallel_sizes["ep"]
        emb_expected = 1.0 / ps.extra_parallel_sizes["emb"]
        check_model_param_grad_one_by_one(
            expected_grad=expected, ep_expected_grad=ep_expected, emb_expected_grad=emb_expected, msg="Before clipping"
        )

        # Every local param grad is 1.0 / ps.ep_size, model total norm should be sqrt(1 * non_ep_param_num + 1/ep_size^2 * ep_param_num)
        expected_total_grad_norm = math.sqrt(
            (64 * 16 + 64 * 16) * (1 / ps.extra_parallel_sizes["emb"] ** 2)
            + 16
            + 64 * 16
            + (64 * 16 * 32) * (1 / ps.extra_parallel_sizes["ep"] ** 2)
        )
        total_grad_norm_pre_clip = veomni_clip_grad_norm(model, max_grad_norm)

        # check whether total grad norm meets our expectation
        torch.testing.assert_close(total_grad_norm_pre_clip, expected=expected_total_grad_norm, atol=1e-6, rtol=1e-6)

        # go through each param grad one-by-one after clipping to check whether their value meets our expectation
        clip_coeff = min(max_grad_norm / expected_total_grad_norm, 1.0)
        ep_clip_coeff = 1.0 / ps.extra_parallel_sizes["ep"] * min(max_grad_norm / expected_total_grad_norm, 1.0)
        emb_clip_coeff = 1.0 / ps.extra_parallel_sizes["emb"] * min(max_grad_norm / expected_total_grad_norm, 1.0)
        logger.info_rank0("Checking model param grad one-by-one after clipping")
        check_model_param_grad_one_by_one(
            expected_grad=clip_coeff,
            ep_expected_grad=ep_clip_coeff,
            emb_expected_grad=emb_clip_coeff,
            msg="After clipping",
        )

        logger.info_rank0(f"step: {step}, loss: {loss.item()}, grad_norm_pre_clip: {total_grad_norm_pre_clip}, ")
        model.zero_grad()

    dist.barrier()
    dist.destroy_process_group()


def _run_clip_grad_norm_fsdp2_test(
    ep_size: int, emb_size: int, cpu_offload: bool, dp_replicate_size: int | None = None
) -> None:
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--master_port=4321",
        "tests/utils/test_extra_parallel_clip_grad_norm.py",
        f"--train.accelerator.ep_size={ep_size}",
        "--train.accelerator.ep_outside=False",
        f"--train.accelerator.extra_parallel_sizes={emb_size}",
        "--train.accelerator.extra_parallel_placement_innermost=False",
        "--train.accelerator.extra_parallel_names=emb",
        "--train.accelerator.fsdp_config.fsdp_mode=fsdp2",
        "--train.init_device=meta",
        "--train.checkpoint.output_dir='debug'",
    ]
    # HSDP: split the dp dim into (dp_replicate, dp_shard). dp_shard is inferred
    # as dp_size // dp_replicate_size by the argument resolver. The expected
    # total grad norm is identical to the pure-FSDP case: the clip reduces over
    # the shard group only, so replicating over dp_replicate must NOT inflate it.
    if dp_replicate_size is not None:
        command.append(f"--train.accelerator.dp_replicate_size={dp_replicate_size}")
    if cpu_offload:
        command.append("--train.accelerator.fsdp_config.offload=True")
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


@pytest.mark.parametrize("cpu_offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_clip_grad_norm_fsdp2_no_extra_parallel(cpu_offload: bool):
    _run_clip_grad_norm_fsdp2_test(ep_size=1, emb_size=1, cpu_offload=cpu_offload)


@pytest.mark.parametrize("cpu_offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_clip_grad_norm_fsdp2_ep4(cpu_offload: bool):
    _run_clip_grad_norm_fsdp2_test(ep_size=4, emb_size=1, cpu_offload=cpu_offload)


@pytest.mark.parametrize("cpu_offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_clip_grad_norm_fsdp2_ep8(cpu_offload: bool):
    _run_clip_grad_norm_fsdp2_test(ep_size=8, emb_size=1, cpu_offload=cpu_offload)


@pytest.mark.parametrize("cpu_offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_clip_grad_norm_fsdp2_emb8(cpu_offload: bool):
    _run_clip_grad_norm_fsdp2_test(ep_size=1, emb_size=8, cpu_offload=cpu_offload)


@pytest.mark.parametrize("cpu_offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_clip_grad_norm_fsdp2_ep2_emb4(cpu_offload: bool):
    _run_clip_grad_norm_fsdp2_test(ep_size=2, emb_size=4, cpu_offload=cpu_offload)


# ---------------------------------------------------------------------------
# HSDP regression: dp = (dp_replicate=2, dp_shard=4) on 8 GPUs.
#
# The grad-norm reduction must skip the replicate dim for BOTH non-ExtraParallel
# params (reduce over `dp_shard_sp`, not `dp_sp`) and ExtraParallel params
# (reduce over `ep_fsdp`/`emb_fsdp`, not the full `*_replicate` mesh). If the
# replicate dim leaks into the sum, the norm is inflated by sqrt(dp_replicate),
# so these reuse the SAME expected total norm as the pure-FSDP cases above.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cpu_offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_clip_grad_norm_hsdp_no_extra_parallel(cpu_offload: bool):
    _run_clip_grad_norm_fsdp2_test(ep_size=1, emb_size=1, cpu_offload=cpu_offload, dp_replicate_size=2)


@pytest.mark.skipif(
    _torch_npu_version() and Version(_torch_npu_version()) < Version("2.9.0"),
    reason="torch_npu < 2.9.0 grad_norm with HSDP is buggy",
)
@pytest.mark.parametrize("cpu_offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_clip_grad_norm_hsdp_ep4(cpu_offload: bool):
    _run_clip_grad_norm_fsdp2_test(ep_size=4, emb_size=1, cpu_offload=cpu_offload, dp_replicate_size=2)


@pytest.mark.skipif(
    _torch_npu_version() and Version(_torch_npu_version()) < Version("2.9.0"),
    reason="torch_npu < 2.9.0 grad_norm with HSDP is buggy",
)
@pytest.mark.parametrize("cpu_offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_clip_grad_norm_hsdp_emb4(cpu_offload: bool):
    _run_clip_grad_norm_fsdp2_test(ep_size=1, emb_size=4, cpu_offload=cpu_offload, dp_replicate_size=2)


@pytest.mark.skipif(
    _torch_npu_version() and Version(_torch_npu_version()) < Version("2.9.0"),
    reason="torch_npu < 2.9.0 grad_norm with HSDP is buggy",
)
@pytest.mark.parametrize("cpu_offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_clip_grad_norm_hsdp_ep2_emb4(cpu_offload: bool):
    _run_clip_grad_norm_fsdp2_test(ep_size=2, emb_size=4, cpu_offload=cpu_offload, dp_replicate_size=2)


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


def _make_model(param_groups: dict[str, list[nn.Parameter]], replicated_ids: set) -> nn.Module:
    """Wrap a ``param_groups`` dict in the duck-typed object the clipper expects."""
    m = nn.Module()
    m._extra_parallel_param_groups = param_groups
    m._ep_replicated_lora_param_ids = replicated_ids
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
    with mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps):
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

    with mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=math.inf, error_if_nonfinite=False, foreach=False
        )

    assert total.item() == pytest.approx(6.0, abs=1e-6)


if __name__ == "__main__":
    main()
