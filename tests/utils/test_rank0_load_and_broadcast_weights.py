import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import distributed as dist
from torch import nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import find_free_port
from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import (
    _init_parallel_state,
    clear_parallel_state,
    get_parallel_state,
    use_parallel_state,
)
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.checkpoint.weights import load_model_weights, rank0_load_and_broadcast_weights
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


_CI_LAUNCH = LaunchSpec(nproc=4)


try:
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
except ImportError:  # pragma: no cover - torch < 2.2 fallback
    DTensor = None  # type: ignore[assignment]
    Replicate = None  # type: ignore[assignment]
    distribute_tensor = None  # type: ignore[assignment]


logger = helper.create_logger(__name__)


@dataclass
class BroadcastTestArguments:
    weights_path: str = ""
    parallel_weights_path: str = ""
    standard_weights_path: str = ""
    device_type: str = get_device_type()
    backend: str = get_dist_comm_backend()
    mode: str = "broadcast"  # "broadcast" | "load_weights" | "all"


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)
    test: "BroadcastTestArguments" = field(default_factory=BroadcastTestArguments)


class ToyEmbed(torch.nn.Module):
    def __init__(self, num_embeddings: int, embed_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embed_dim), requires_grad=True)
        if self.weight.device.type != "meta":
            self.reset_parameters()

    def forward(self) -> torch.Tensor:
        return self.weight.sum()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)


class ToyMoeAndEmbedDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extra_embed_tokens = ToyEmbed(64, 16)
        self.regular_mlp = torch.nn.Parameter(torch.ones(64, 16), requires_grad=True)
        self.moe = ToyMoeExperts()

    def forward(self) -> torch.Tensor:
        return self.extra_embed_tokens() + self.regular_mlp.sum() + self.moe()


class ToyMoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.Parameter(torch.ones(64, 16, 32), requires_grad=True)

    def forward(self) -> torch.Tensor:
        return self.experts.sum()


class ToyMoeAndEmbedModel(torch.nn.Module):
    _no_split_modules = ["ToyMoeAndEmbedDecoderLayer", "ToyEmbed"]

    def __init__(self):
        super().__init__()
        self.input_embed_tokens = ToyEmbed(7, 4)
        self.output_embed_tokens = ToyEmbed(7, 4)
        if self.output_embed_tokens.weight.device.type != "meta":
            self.output_embed_tokens.weight.data.copy_(self.input_embed_tokens.weight.data)

        self.linear1 = nn.Linear(4, 3, bias=True)
        self.linear2 = nn.Linear(3, 2, bias=False)
        self.linear3 = nn.Linear(2, 1, bias=True)

        self.register_buffer("buffer", torch.arange(2, dtype=torch.float32))

        self.extra_embed_tokens = ToyEmbed(64, 16)
        self.bias = torch.nn.Parameter(torch.ones(16), requires_grad=True)
        self.decoder = ToyMoeAndEmbedDecoderLayer()

        self.config = SimpleNamespace(tie_word_embeddings=True)

    def get_input_embeddings(self) -> nn.Module:
        return self.input_embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.output_embed_tokens

    def init_weights(self):
        self.input_embed_tokens.reset_parameters()
        self.output_embed_tokens.weight.data.copy_(self.input_embed_tokens.weight.data)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.fill_(1.0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.fill_(2.0)
        nn.init.xavier_uniform_(self.extra_embed_tokens.weight)
        nn.init.xavier_uniform_(self.decoder.extra_embed_tokens.weight)
        self.bias.data.fill_(3.0)
        nn.init.xavier_uniform_(self.decoder.regular_mlp)
        nn.init.xavier_uniform_(self.decoder.moe.experts)

    def get_parallel_plan(self):
        ep_plan = {"decoder.moe.experts": Shard(0)}
        emb_plan = {"extra_embed_tokens.weight": Shard(0), "decoder.extra_embed_tokens.weight": Shard(0)}
        parallel_plan = ParallelPlan(
            extra_parallel_plan={
                "ep": ep_plan,
                "emb": emb_plan,
            }
        )
        parallel_plan.extra_parallel_fsdp_no_shard_module = {
            "ep": {"decoder.moe"},
            "emb": {"extra_embed_tokens", "decoder.extra_embed_tokens"},
        }
        parallel_plan.cpu_load_param_name = ["linear3.bias", "decoder.extra_embed_tokens.weight"]

        return parallel_plan


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embed_tokens = ToyEmbed(7, 4)
        self.output_embed_tokens = ToyEmbed(7, 4)
        if self.output_embed_tokens.weight.device.type != "meta":
            self.output_embed_tokens.weight.data.copy_(self.input_embed_tokens.weight.data)

        self.linear1 = nn.Linear(4, 3, bias=True)
        self.linear2 = nn.Linear(3, 2, bias=False)
        self.linear3 = nn.Linear(2, 1, bias=True)

        self.register_buffer("buffer", torch.arange(2, dtype=torch.float32))

        self.config = SimpleNamespace(tie_word_embeddings=True)

    def get_input_embeddings(self) -> nn.Module:
        return self.input_embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.output_embed_tokens

    def init_weights(self):
        self.input_embed_tokens.reset_parameters()
        self.output_embed_tokens.weight.data.copy_(self.input_embed_tokens.weight.data)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.fill_(1.0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.fill_(2.0)


def _write_checkpoint(checkpoint_dir: Path, is_parallel: bool) -> str:
    torch.manual_seed(0)
    if is_parallel:
        model = ToyMoeAndEmbedModel().cpu()
    else:
        model = ToyModel()
    model.init_weights()
    model.output_embed_tokens.weight = model.input_embed_tokens.weight  # tie before saving
    model.buffer.uniform_(-0.5, 0.5)
    state_dict = {name: tensor.detach().clone().cpu() for name, tensor in model.state_dict().items()}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoint to {checkpoint_dir}")
    save_file(state_dict, str(checkpoint_dir / "model.safetensors"))
    return str(checkpoint_dir)


def _clone_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if DTensor is not None and isinstance(tensor, DTensor):
        assert Replicate is not None
        local_tensor = tensor.to_local()
        if local_tensor.device.type == "cpu":
            mesh_device_type = tensor.device_mesh.device_type
            if mesh_device_type == "cpu":
                raise RuntimeError("CPU DTensor gather is not supported in this test helper.")
            tensor = tensor.to(mesh_device_type)
        placements = [Replicate()] * tensor.device_mesh.ndim
        tensor = tensor.redistribute(tensor.device_mesh, placements).to_local()
    return tensor.detach().cpu().clone()


def _assert_model_parameters_on_device(model: nn.Module, expected_device_type: str) -> None:
    for name, param in model.named_parameters():
        local_param = param.to_local() if DTensor is not None and isinstance(param, DTensor) else param
        assert local_param.device.type == expected_device_type, (
            f"Expected {name} to be on {expected_device_type}, got {local_param.device.type}."
        )


def _check_rank0_broadcast(args: Arguments, weights_path: Path, *, cpu_offload: bool) -> None:
    if not weights_path.exists():
        raise ValueError("`weights_path` must point to an existing directory.")

    dtensor_factory = distribute_tensor if distribute_tensor is not None else None
    if dtensor_factory is None:
        raise RuntimeError("torch.distributed.tensor.distribute_tensor is required for fsdp2 weight loading test")
    load_device = "cpu" if cpu_offload else get_device_type()

    rank0_load_model = ToyMoeAndEmbedModel()
    all_rank_load_model = ToyMoeAndEmbedModel()

    # 1.1 rank0_load_model init with no weights_path
    rank0_load_model = build_parallelize_model(
        rank0_load_model,
        weights_path=None,
        init_device=args.model.accelerator.init_device,
        mixed_precision=args.model.accelerator.fsdp_config.mixed_precision,
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=cpu_offload,
        basic_modules=[],
        broadcast_model_weights_from_rank0=True,
    )

    cpu_load_param_name = None
    if hasattr(rank0_load_model, "get_parallel_plan"):
        cpu_load_param_name = getattr(rank0_load_model.get_parallel_plan(), "cpu_load_param_name", None)

    # 1.2 rank0_load_model load from weights_path
    rank0_load_and_broadcast_weights(
        rank0_load_model,
        str(weights_path),
        init_device=load_device,
        dtensor_factory=dtensor_factory,
        cpu_load_param_name=cpu_load_param_name,
        max_load_broadcast_size=0.0,
    )

    # 2.1 all_rank_load_model init with no weights_path
    all_rank_load_model = build_parallelize_model(
        all_rank_load_model,
        weights_path=None,
        init_device=args.model.accelerator.init_device,
        mixed_precision=args.model.accelerator.fsdp_config.mixed_precision,
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=cpu_offload,
        basic_modules=[],
        broadcast_model_weights_from_rank0=False,
    )

    # 2.2 all_rank_load_model load from weights_path
    load_model_weights(
        all_rank_load_model,
        str(weights_path),
        init_device=load_device,
        dtensor_factory=dtensor_factory,
    )
    if cpu_offload:
        _assert_model_parameters_on_device(rank0_load_model, "cpu")
        _assert_model_parameters_on_device(all_rank_load_model, "cpu")
    torch.testing.assert_close(
        rank0_load_model.get_input_embeddings().weight,
        all_rank_load_model.get_input_embeddings().weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.get_output_embeddings().weight,
        all_rank_load_model.get_output_embeddings().weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.linear1.weight,
        all_rank_load_model.linear1.weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.linear1.bias,
        all_rank_load_model.linear1.bias,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.linear2.weight,
        all_rank_load_model.linear2.weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.buffer,
        all_rank_load_model.buffer,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.linear3.weight,
        all_rank_load_model.linear3.weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.linear3.bias,
        all_rank_load_model.linear3.bias,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.extra_embed_tokens.weight,
        all_rank_load_model.extra_embed_tokens.weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.decoder.moe.experts,
        all_rank_load_model.decoder.moe.experts,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rank0_load_model.decoder.extra_embed_tokens.weight,
        all_rank_load_model.decoder.extra_embed_tokens.weight,
        atol=0.0,
        rtol=0.0,
    )

    assert (
        all_rank_load_model.get_input_embeddings().weight.data_ptr()
        == all_rank_load_model.get_output_embeddings().weight.data_ptr()
    )

    del rank0_load_model, all_rank_load_model
    dist.barrier()


def _check_load_weights_no_scatter(args: Arguments, weights_path: Path, *, cpu_offload: bool) -> None:
    if not weights_path.exists():
        raise ValueError("`weights_path` must point to an existing directory.")

    fsdp_model = build_parallelize_model(
        ToyModel(),
        weights_path=str(weights_path),
        init_device=args.model.accelerator.init_device,
        mixed_precision=args.model.accelerator.fsdp_config.mixed_precision,
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=cpu_offload,
        basic_modules=[],
    )

    reference_model = ToyModel().to(get_device_type())
    load_model_weights(reference_model, str(weights_path), init_device=get_device_type())
    reference_model = reference_model.cpu()

    if cpu_offload:
        _assert_model_parameters_on_device(fsdp_model, "cpu")

    torch.testing.assert_close(
        _clone_to_cpu(fsdp_model.get_input_embeddings().weight),
        reference_model.get_input_embeddings().weight.detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _clone_to_cpu(fsdp_model.get_output_embeddings().weight),
        reference_model.get_output_embeddings().weight.detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _clone_to_cpu(fsdp_model.linear1.weight),
        reference_model.linear1.weight.detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _clone_to_cpu(fsdp_model.linear1.bias),
        reference_model.linear1.bias.detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _clone_to_cpu(fsdp_model.linear2.weight),
        reference_model.linear2.weight.detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _clone_to_cpu(fsdp_model.buffer),
        reference_model.buffer.detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _clone_to_cpu(fsdp_model.linear3.weight),
        reference_model.linear3.weight.detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _clone_to_cpu(fsdp_model.linear3.bias),
        reference_model.linear3.bias.detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )

    assert fsdp_model.get_input_embeddings().weight is fsdp_model.get_output_embeddings().weight

    del fsdp_model, reference_model
    dist.barrier()


def run_rank0_broadcast_test(args: Arguments) -> None:
    weights_path = Path(args.test.weights_path)
    get_torch_device().set_device(args.train.local_rank)
    dist.init_process_group(backend=args.test.backend)
    ps = _init_parallel_state(
        dp_size=args.model.accelerator.dp_size,
        dp_replicate_size=args.model.accelerator.dp_replicate_size,
        dp_shard_size=args.model.accelerator.dp_shard_size,
        tp_size=args.model.accelerator.tp_size,
        pp_size=args.model.accelerator.pp_size,
        cp_size=args.model.accelerator.cp_size,
        ulysses_size=args.model.accelerator.ulysses_size,
        extra_parallel_sizes=args.model.accelerator.extra_parallel_sizes,
        extra_parallel_placement_innermost=args.model.accelerator.extra_parallel_placement_innermost,
        extra_parallel_names=args.model.accelerator.extra_parallel_names,
        dp_mode=args.model.accelerator.fsdp_config.fsdp_mode,
        name="broadcast",
    )
    try:
        with use_parallel_state(ps):
            _check_rank0_broadcast(args, weights_path, cpu_offload=args.model.accelerator.fsdp_config.offload)
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


def run_load_weights_test(args: Arguments) -> None:
    weights_path = Path(args.test.weights_path)
    get_torch_device().set_device(args.train.local_rank)
    dist.init_process_group(backend=args.test.backend)
    ps = _init_parallel_state(
        dp_size=args.model.accelerator.dp_size,
        dp_replicate_size=args.model.accelerator.dp_replicate_size,
        dp_shard_size=args.model.accelerator.dp_shard_size,
        tp_size=args.model.accelerator.tp_size,
        pp_size=args.model.accelerator.pp_size,
        cp_size=args.model.accelerator.cp_size,
        ulysses_size=args.model.accelerator.ulysses_size,
        extra_parallel_sizes=args.model.accelerator.extra_parallel_sizes,
        extra_parallel_placement_innermost=args.model.accelerator.extra_parallel_placement_innermost,
        extra_parallel_names=args.model.accelerator.extra_parallel_names,
        dp_mode=args.model.accelerator.fsdp_config.fsdp_mode,
        name="load_weights",
    )
    try:
        with use_parallel_state(ps):
            _check_load_weights_no_scatter(args, weights_path, cpu_offload=args.model.accelerator.fsdp_config.offload)
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


def run_all_tests(args: Arguments) -> None:
    parallel_weights_path = Path(args.test.parallel_weights_path)
    standard_weights_path = Path(args.test.standard_weights_path)

    get_torch_device().set_device(args.train.local_rank)
    dist.init_process_group(backend=args.test.backend)
    world_size = dist.get_world_size()

    ps_broadcast = _init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        extra_parallel_sizes=(2, 2),
        extra_parallel_placement_innermost=(False, False),
        extra_parallel_names=("ep", "emb"),
        dp_mode="fsdp2",
        name="broadcast_ep2_emb2",
    )
    ps_load_weights = _init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        extra_parallel_sizes=(1,),
        extra_parallel_placement_innermost=(False,),
        extra_parallel_names=("ep",),
        dp_mode="fsdp2",
        name="load_weights_no_extra",
    )

    scenarios = [
        (
            "broadcast_no_offload",
            ps_broadcast,
            lambda: _check_rank0_broadcast(args, parallel_weights_path, cpu_offload=False),
        ),
        (
            "broadcast_cpu_offload",
            ps_broadcast,
            lambda: _check_rank0_broadcast(args, parallel_weights_path, cpu_offload=True),
        ),
        (
            "load_weights_no_offload",
            ps_load_weights,
            lambda: _check_load_weights_no_scatter(args, standard_weights_path, cpu_offload=False),
        ),
        (
            "load_weights_cpu_offload",
            ps_load_weights,
            lambda: _check_load_weights_no_scatter(args, standard_weights_path, cpu_offload=True),
        ),
    ]
    if os.environ.get("VEOMNI_REVERSE_SCENARIOS") == "1":
        scenarios = list(reversed(scenarios))

    try:
        for label, ps, fn in scenarios:
            with use_parallel_state(ps):
                assert get_parallel_state() is ps, f"[{label}] unexpected parallel state"
                fn()
        dist.barrier()
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


@pytest.mark.launch_plan(provider=_CI_LAUNCH)
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed required")
def test_load_dist_model_weights_matches_standard(tmp_path: Path) -> None:
    """Fused 4-rank worker covering broadcast (EP2+emb2) and load_weights (no scatter) across no_offload/cpu_offload."""
    parallel_weights_path = _write_checkpoint(tmp_path / "ckpt_parallel", is_parallel=True)
    standard_weights_path = _write_checkpoint(tmp_path / "ckpt_standard", is_parallel=False)

    world_size = _CI_LAUNCH.nproc
    port = find_free_port()
    command = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        f"--master_port={port}",
        "tests/utils/test_rank0_load_and_broadcast_weights.py",
        "--model.config_path=test",
        "--data.train_path=tests",
        "--train.checkpoint.output_dir=.tests/cache",
        "--model.accelerator.fsdp_config.fsdp_mode=fsdp2",
        "--model.accelerator.init_device=meta",
        "--model.accelerator.fsdp_config.mixed_precision.enable=False",
        "--model.accelerator.gradient_checkpointing.enable=False",
        f"--test.parallel_weights_path={parallel_weights_path}",
        f"--test.standard_weights_path={standard_weights_path}",
        f"--test.device_type={get_device_type()}",
        f"--test.backend={get_dist_comm_backend()}",
        "--test.mode=all",
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    args = parse_args(Arguments)
    mode = getattr(args.test, "mode", "broadcast")
    if mode == "all":
        run_all_tests(args)
    elif mode == "load_weights":
        run_load_weights_test(args)
    else:
        run_rank0_broadcast_test(args)
