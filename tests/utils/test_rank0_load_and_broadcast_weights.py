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

from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.module_utils import load_model_weights, rank0_load_and_broadcast_weights
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


try:
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
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


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)
    test: "BroadcastTestArguments" = field(default_factory=BroadcastTestArguments)


class TinyEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embed_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embed_dim))
        if self.weight.device.type != "meta":
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embeddings = TinyEmbedding(7, 4)
        self.output_embeddings = TinyEmbedding(7, 4)
        if self.output_embeddings.weight.device.type != "meta":
            self.output_embeddings.weight.data.copy_(self.input_embeddings.weight.data)
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.linear2 = nn.Linear(3, 2, bias=False)
        self.register_buffer("buffer", torch.arange(2, dtype=torch.float32))
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self._no_split_modules: list[str] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.input_embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.output_embeddings

    def init_weights(self) -> None:
        self.input_embeddings.reset_parameters()
        self.output_embeddings.weight.data.copy_(self.input_embeddings.weight.data)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)


def _write_checkpoint(checkpoint_dir: Path) -> str:
    torch.manual_seed(0)
    model = TinyModel().cpu()
    model.init_weights()
    model.output_embeddings.weight = model.input_embeddings.weight  # tie before saving
    model.buffer.uniform_(-0.5, 0.5)
    state_dict = {name: tensor.detach().clone().cpu() for name, tensor in model.state_dict().items()}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
        ep_outside=args.train.ep_outside,
    )

    try:
        base_model = TinyModel()
        fsdp_model = build_parallelize_model(
            base_model,
            weights_path=None,
            init_device=args.train.init_device,
            enable_mixed_precision=False,
            enable_gradient_checkpointing=False,
            basic_modules=[],
            broadcast_model_weights_from_rank0=True,
        )

        dtensor_factory = distribute_tensor if distribute_tensor is not None else None
        if dtensor_factory is None:
            raise RuntimeError("torch.distributed.tensor.distribute_tensor is required for fsdp2 weight loading test")

        rank0_load_and_broadcast_weights(
            fsdp_model, str(weights_path), init_device=get_device_type(), dtensor_factory=dtensor_factory
        )

        reference_model = TinyModel().to(get_device_type())
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

        assert fsdp_model.get_input_embeddings().weight is fsdp_model.get_output_embeddings().weight
        assert (
            reference_model.get_input_embeddings().weight.data_ptr()
            == reference_model.get_output_embeddings().weight.data_ptr()
        )

        dist.barrier()
    finally:
        dist.destroy_process_group()
        from veomni.distributed import parallel_state as _ps

        _ps._PARALLEL_STATE = None


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed required")
def test_load_dist_model_weights_matches_standard(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    weights_path = _write_checkpoint(checkpoint_dir)

    world_size = 2
    port = 12345 + random.randint(0, 100)
    command = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        f"--master_port={port}",
        "tests/utils/test_rank0_load_and_broadcast_weights.py",
        "--model.config_path=test",
        "--data.train_path=tests",
        "--train.output_dir=.tests/cache",
        "--train.data_parallel_mode=fsdp2",
        "--train.init_device=meta",
        "--train.enable_mixed_precision=False",
        "--train.enable_gradient_checkpointing=False",
        "--train.broadcast_model_weights_from_rank0=True",
        f"--test.weights_path={weights_path}",
        f"--test.device_type={get_device_type()}",
        f"--test.backend={get_dist_comm_backend()}",
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    args = parse_args(Arguments)
    run_rank0_broadcast_test(args)
