import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest
import torch
from safetensors.torch import save_file
from torch import distributed as dist
from torch import nn
from torch.multiprocessing import spawn

from veomni.distributed.parallel_state import init_parallel_state
from veomni.models.module_utils import _load_state_dict, load_dist_model_weights, load_model_weights


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.linear2 = nn.Linear(3, 2, bias=False)
        self.register_buffer("buffer", torch.arange(2, dtype=torch.float32))
        self.config = SimpleNamespace(tie_word_embeddings=False)


def _write_checkpoint(checkpoint_dir: Path) -> None:
    torch.manual_seed(0)
    model = TinyModel().cpu()
    state_dict = {name: tensor.detach().clone().cpu() for name, tensor in model.state_dict().items()}
    save_file(state_dict, str(checkpoint_dir / "model.safetensors"))


def _log_checkpoint(weights_path: str, rank: int) -> None:
    for iterator in _load_state_dict(weights_path):
        for name, tensor in iterator:
            flat = tensor.flatten().tolist()
            print(f"[rank{rank}] checkpoint {name}: {flat}")


def _extract_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _dist_worker(
    rank: int, world_size: int, init_method: str, weights_path: str, device_type: str, backend: str
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    if device_type == "cuda":
        torch.cuda.set_device(rank)

    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    init_parallel_state(dp_size=world_size, device_type=device_type)
    try:
        _log_checkpoint(weights_path, rank)
        dist.barrier()

        model_dist = TinyModel()
        load_dist_model_weights(model_dist, weights_path, init_device=device_type)
        dist_state = _extract_state(model_dist)

        model_ref = TinyModel()
        load_model_weights(model_ref, weights_path, init_device=device_type)
        ref_state = _extract_state(model_ref)

        for name in ref_state:
            assert name in dist_state, f"missing key {name} in dist load state"
            diff = (dist_state[name] - ref_state[name]).abs()
            max_diff = diff.max().item() if diff.numel() else 0.0
            print(f"[rank{rank}] compare {name}: max_abs_diff={max_diff}")
            torch.testing.assert_close(dist_state[name], ref_state[name], atol=0.0, rtol=0.0)

        dist.barrier()
    finally:
        dist.destroy_process_group()
        from veomni.distributed import parallel_state as _ps

        _ps._PARALLEL_STATE = None


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed required")
def test_load_dist_model_weights_matches_standard(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(checkpoint_dir)

    init_file = tmp_path / "dist_init"
    if init_file.exists():
        init_file.unlink()
    init_method = f"file://{init_file}"

    world_size = 2
    has_cuda = torch.cuda.is_available()
    device_type = "cuda" if has_cuda else "cpu"
    backend = "nccl" if has_cuda else "gloo"

    spawn(
        _dist_worker,
        args=(world_size, init_method, str(checkpoint_dir), device_type, backend),
        nprocs=world_size,
        join=True,
    )
