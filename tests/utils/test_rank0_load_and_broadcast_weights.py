import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import distributed as dist
from torch import nn

from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.module_utils import load_model_weights, rank0_load_and_broadcast_weights
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


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
    device_type: str = get_device_type()
    backend: str = get_dist_comm_backend()
    mode: str = "broadcast"  # "broadcast" | "load_weights"


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
        placements = [Replicate()] * tensor.device_mesh.ndim
        tensor = tensor.redistribute(tensor.device_mesh, placements).to_local()
    return tensor.detach().cpu().clone()


def run_rank0_broadcast_test(args: Arguments) -> None:
    weights_path = Path(args.test.weights_path)
    if not weights_path.exists():
        raise ValueError("`--test.weights_path` must point to an existing directory.")

    get_torch_device().set_device(args.train.local_rank)
    dist.init_process_group(backend=args.test.backend)

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

    dtensor_factory = distribute_tensor if distribute_tensor is not None else None
    if dtensor_factory is None:
        raise RuntimeError("torch.distributed.tensor.distribute_tensor is required for fsdp2 weight loading test")

    try:
        rank0_load_model = ToyMoeAndEmbedModel()
        all_rank_load_model = ToyMoeAndEmbedModel()
        """
        model before parallelize:
        ToyMoeAndEmbedModel(
            (input_embed_tokens): ToyEmbed()
            (output_embed_tokens): ToyEmbed()
            (linear1): Linear(in_features=4, out_features=3, bias=True)
            (linear2): Linear(in_features=3, out_features=2, bias=False)
            (linear3): Linear(in_features=2, out_features=1, bias=True)
            (extra_embed_tokens): ToyEmbed()
            (decoder): ToyMoeAndEmbedDecoderLayer(
                (extra_embed_tokens): ToyEmbed()
                (moe): ToyMoeExperts()
            )
        )

        model after parallelize:
        FSDPToyMoeAndEmbedModel(
            (input_embed_tokens): FSDPToyEmbed()
            (output_embed_tokens): FSDPToyEmbed()
            (linear1): Linear(in_features=4, out_features=3, bias=True)
            (linear2): Linear(in_features=3, out_features=2, bias=False)
            (linear3): Linear(in_features=2, out_features=1, bias=True)
            (extra_embed_tokens): FSDPToyEmbed()
            (decoder): FSDPToyMoeAndEmbedDecoderLayer(
                (extra_embed_tokens): FSDPToyEmbed()
                (moe): FSDPToyMoeExperts()
            )
        )
        """

        # 1.1 rank0_load_model init with no weights_path
        rank0_load_model = build_parallelize_model(
            rank0_load_model,
            weights_path=None,
            init_device=args.train.init_device,
            mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=False,
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
            init_device=get_device_type(),
            dtensor_factory=dtensor_factory,
            cpu_load_param_name=cpu_load_param_name,
            max_load_broadcast_size=0.0,
        )

        # 2.1 all_rank_load_model init with no weights_path
        all_rank_load_model = build_parallelize_model(
            all_rank_load_model,
            weights_path=None,
            init_device=args.train.init_device,
            mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=False,
            basic_modules=[],
            broadcast_model_weights_from_rank0=False,
        )

        # 2.2 all_rank_load_model load from weights_path
        load_model_weights(
            all_rank_load_model,
            str(weights_path),
            init_device=get_device_type(),
            dtensor_factory=dtensor_factory,
        )
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

        dist.barrier()
    finally:
        dist.destroy_process_group()
        from veomni.distributed import parallel_state as _ps

        _ps._PARALLEL_STATE = None


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed required")
def test_load_dist_model_weights_matches_standard(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    weights_path = _write_checkpoint(checkpoint_dir, is_parallel=True)

    world_size = 4
    port = 12345 + random.randint(0, 100)
    command = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        f"--master_port={port}",
        "tests/utils/test_rank0_load_and_broadcast_weights.py",
        "--model.config_path=test",
        "--data.train_path=tests",
        "--train.checkpoint.output_dir=.tests/cache",
        "--train.accelerator.fsdp_config.fsdp_mode=fsdp2",
        "--train.init_device=meta",
        "--train.accelerator.fsdp_config.mixed_precision.enable=False",
        "--train.gradient_checkpointing.enable=False",
        "--train.broadcast_model_weights_from_rank0=True",
        "--train.accelerator.ep_size=2",
        "--train.accelerator.ep_outside=False",
        "--train.accelerator.extra_parallel_sizes=2",
        "--train.accelerator.extra_parallel_placement_innermost=False",
        "--train.accelerator.extra_parallel_names=emb",
        f"--test.weights_path={weights_path}",
        f"--test.device_type={get_device_type()}",
        f"--test.backend={get_dist_comm_backend()}",
        "--test.mode=broadcast",
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


def run_load_weights_test(args: Arguments) -> None:
    """
    Worker entrypoint for test_load_weights_no_scatter.

    Verifies that load_model_weights (which now passes src_data_rank=None to
    distribute_tensor) produces bit-for-bit identical parameters to a reference
    single-rank load. This is the code path fixed in:
    https://github.com/ByteDance-Seed/VeOmni/issues/637
    """
    weights_path = Path(args.test.weights_path)
    if not weights_path.exists():
        raise ValueError("`--test.weights_path` must point to an existing directory.")

    get_torch_device().set_device(args.train.local_rank)
    dist.init_process_group(backend=args.test.backend)

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

    try:
        # build_parallelize_model with weights_path triggers load_model_weights
        # (the every-rank-reads-from-disk path, fixed in issue #637).
        fsdp_model = build_parallelize_model(
            ToyModel(),
            weights_path=str(weights_path),
            init_device=args.train.init_device,
            mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=False,
            basic_modules=[],
        )
        """
        fsdp_model:
        FSDPToyModel(
            (input_embed_tokens): ToyEmbed()
            (output_embed_tokens): ToyEmbed()
            (linear1): Linear(in_features=4, out_features=3, bias=True)
            (linear2): Linear(in_features=3, out_features=2, bias=False)
            (linear3): Linear(in_features=2, out_features=1, bias=True)
        )

        reference_model:
        ToyModel(
            (input_embed_tokens): ToyEmbed()
            (output_embed_tokens): ToyEmbed()
            (linear1): Linear(in_features=4, out_features=3, bias=True)
            (linear2): Linear(in_features=3, out_features=2, bias=False)
            (linear3): Linear(in_features=2, out_features=1, bias=True)
        )

        """

        reference_model = ToyModel().to(get_device_type())
        load_model_weights(reference_model, str(weights_path), init_device=get_device_type())
        reference_model = reference_model.cpu()

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

        dist.barrier()
    finally:
        dist.destroy_process_group()
        from veomni.distributed import parallel_state as _ps

        _ps._PARALLEL_STATE = None


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed required")
def test_load_weights_no_scatter(tmp_path: Path) -> None:
    """
    Regression test for https://github.com/ByteDance-Seed/VeOmni/issues/637.

    Ensures load_model_weights with src_data_rank=None (all-ranks-read path)
    loads bit-for-bit correct parameters into an FSDP2 model.
    The rank0_load_and_broadcast_weights path is tested separately in
    test_load_dist_model_weights_matches_standard.
    """
    checkpoint_dir = tmp_path / "ckpt"
    weights_path = _write_checkpoint(checkpoint_dir, is_parallel=False)

    world_size = 4
    port = 12345 + random.randint(0, 100)
    command = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        f"--master_port={port}",
        "tests/utils/test_rank0_load_and_broadcast_weights.py",
        "--model.config_path=test",
        "--data.train_path=tests",
        "--train.checkpoint.output_dir=.tests/cache",
        "--train.accelerator.fsdp_config.fsdp_mode=fsdp2",
        "--train.init_device=meta",
        "--train.accelerator.fsdp_config.mixed_precision.enable=False",
        "--train.gradient_checkpointing.enable=False",
        "--train.broadcast_model_weights_from_rank0=False",
        f"--test.weights_path={weights_path}",
        f"--test.device_type={get_device_type()}",
        f"--test.backend={get_dist_comm_backend()}",
        "--test.mode=load_weights",
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    args = parse_args(Arguments)
    if getattr(args.test, "mode", "broadcast") == "load_weights":
        run_load_weights_test(args)
    else:
        run_rank0_broadcast_test(args)
