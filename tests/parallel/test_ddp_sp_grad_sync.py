# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DDP + Ulysses gradient-sync parity.

DDP wraps with ``process_group=dp_group``, which Ulysses shrinks to
``world / sp_size``. At ``dp_size == 1`` DDP's all-reduce is a no-op, so without
the ``fsdp_group`` reduction in ``veomni_clip_grad_norm`` each rank would step on
the gradient of its own ``1 / sp_size`` slice only.

Every case below runs the same total token set two ways in one process — sharded
across ranks under a DDP + SP parallel state, and whole on an unwrapped replica
of the same weights — and requires the post-clip gradients and the reported grad
norm to agree. The worker also asserts that the rank's own pre-reduction
gradients differ from the reference, so the parity check cannot pass vacuously.
"""

import argparse
import os
import random
import subprocess
from typing import Optional

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


TOKENS_PER_RANK = 8  # per micro-batch
INPUT_DIM = 16
HIDDEN_DIM = 32
MAX_GRAD_NORM = 0.5
DATA_SEED = 1234
MODEL_SEED = 7


class ToyModel(nn.Module):
    """Two-layer MLP plus a buffer that must stay rank-local under DDP."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, INPUT_DIM)
        # Stands in for derived per-rank state such as dynamic-rope ``inv_freq``,
        # which each rank recomputes from its own sequence length. Deliberately
        # unused by ``forward`` so it cannot perturb the parity comparison.
        self.register_buffer("rank_marker", torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(hidden_states)))

    def init_weights(self) -> None:
        """The meta-init path calls this after ``to_empty()``, which leaves
        uninitialized memory behind."""
        torch.manual_seed(MODEL_SEED)
        for linear in (self.fc1, self.fc2):
            nn.init.normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
        self.rank_marker.zero_()


def _token_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Mean over tokens, so per-rank means average back to the global mean."""
    return outputs.square().sum(dim=-1).mean()


def main() -> None:
    from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
    from veomni.distributed.parallel_state import clear_parallel_state, get_parallel_state, init_parallel_state
    from veomni.distributed.sequence_parallel.loss import reduce_sequence_parallel_loss
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.module_utils import init_empty_weights

    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses-size", type=int, required=True)
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--init-device", default=None, help="defaults to the accelerator; 'meta' exercises the load")
    cli_args = parser.parse_args()

    dist.init_process_group(backend=get_dist_comm_backend())
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"{get_device_type()}:{local_rank}")
    get_torch_device().set_device(device)

    sp_size = cli_args.ulysses_size
    assert world_size % sp_size == 0, f"world_size {world_size} must be divisible by ulysses_size {sp_size}"
    dp_size = world_size // sp_size

    init_parallel_state(dp_size=dp_size, ulysses_size=sp_size, dp_mode="ddp")
    parallel_state = get_parallel_state()
    assert parallel_state.sp_size == sp_size
    assert dist.get_world_size(parallel_state.dp_group) == dp_size
    assert dist.get_world_size(parallel_state.fsdp_group) == world_size

    init_device = cli_args.init_device or get_device_type()
    torch.manual_seed(MODEL_SEED)
    if init_device == "meta":
        # DDP materializes nothing itself, so this only reaches a working wrap if
        # build_parallelize_model does the meta -> device pass first.
        with init_empty_weights():
            model = ToyModel()
        assert next(model.parameters()).is_meta
    else:
        model = ToyModel().to(device)
    model.train()
    model = build_parallelize_model(
        model,
        weights_path=None,
        enable_gradient_checkpointing=False,
        basic_modules=[],
        init_device=init_device,
    )
    assert not next(model.parameters()).is_meta

    # DDP broadcasts rank0's parameters once in its constructor, so snapshotting
    # the reference here — not before the wrap — guarantees identical weights.
    reference = ToyModel().to(device)
    reference.load_state_dict(model.module.state_dict())
    reference.train()

    # Same tokens on every rank; each rank then takes the slice Ulysses would
    # have handed it, and the reference consumes the whole micro-batch. The
    # clip happens once per step, after every micro-batch has accumulated.
    generator = torch.Generator().manual_seed(DATA_SEED)
    global_tokens = torch.randn(
        cli_args.micro_batches, world_size * TOKENS_PER_RANK, INPUT_DIM, generator=generator
    ).to(device)
    local_tokens = global_tokens[:, rank * TOKENS_PER_RANK : (rank + 1) * TOKENS_PER_RANK]

    for micro_batch in local_tokens:
        # ``ReduceLoss`` scales the incoming gradient by ``sp_size * local /
        # global``, which is 1 for equal token counts: each rank is left with the
        # gradient of its own slice mean, exactly as in a real SP step.
        loss = _token_loss(model(micro_batch))
        loss = reduce_sequence_parallel_loss(
            loss,
            torch.tensor(float(TOKENS_PER_RANK), device=device),
            group=parallel_state.sp_group,
        )
        loss.backward()

    # A buffer written after the wrap must survive the forward: DDP re-broadcasts
    # rank0's buffers before every forward when ``broadcast_buffers`` is on.
    assert model.broadcast_buffers is False, "DDP must be wrapped with broadcast_buffers=False"
    model.module.rank_marker.fill_(float(rank))
    with torch.no_grad():
        model(local_tokens[0])
    assert model.module.rank_marker.item() == float(rank), (
        f"[rank {rank}] rank-local buffer was overwritten: {model.module.rank_marker.item()}"
    )

    pre_sync_grads = [param.grad.detach().clone() for param in model.module.parameters()]

    for micro_batch in global_tokens:
        _token_loss(reference(micro_batch)).backward()
    reference_grads = [param.grad.detach().clone() for param in reference.parameters()]
    reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), MAX_GRAD_NORM)

    grad_norm = veomni_clip_grad_norm(model, MAX_GRAD_NORM)

    torch.testing.assert_close(
        grad_norm,
        reference_norm.item(),
        rtol=1e-4,
        atol=1e-6,
        msg=lambda message: f"[rank {rank}] grad norm differs from the sp=1 reference: {message}",
    )
    for (name, param), reference_param, pre_sync_grad, reference_grad in zip(
        model.module.named_parameters(), reference.parameters(), pre_sync_grads, reference_grads
    ):
        torch.testing.assert_close(
            param.grad,
            reference_param.grad,
            rtol=1e-4,
            atol=1e-6,
            msg=lambda message, name=name: f"[rank {rank}] {name} differs from the sp=1 reference: {message}",
        )
        # Sanity, against the reference gradient as it stood before its own clip:
        # without the fsdp_group reduction the rank keeps ``pre_sync_grad``, so
        # the assertion above is not satisfied by accident.
        assert not torch.allclose(pre_sync_grad, reference_grad, rtol=1e-4, atol=1e-6), (
            f"[rank {rank}] {name} already matched the reference before the sync; the parity check is vacuous"
        )

    dist.barrier()
    dist.destroy_process_group()
    clear_parallel_state()

    if rank == 0:
        print(
            f"DDP + SP gradient parity passed at world_size={world_size}, ulysses_size={sp_size}, "
            f"micro_batches={cli_args.micro_batches}, init_device={init_device}"
        )


def _run_ddp_sp_grad_sync_test(
    world_size: int, ulysses_size: int, micro_batches: int = 1, init_device: Optional[str] = None
) -> None:
    command = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={world_size}",
        # Fresh port per run: back-to-back torchrun launches otherwise collide
        # with the previous one's TIME_WAIT socket.
        f"--master_port={29500 + random.randint(0, 200)}",
        "tests/parallel/test_ddp_sp_grad_sync.py",
        f"--ulysses-size={ulysses_size}",
        f"--micro-batches={micro_batches}",
    ]
    if init_device is not None:
        command.append(f"--init-device={init_device}")
    assert subprocess.run(command, check=True).returncode == 0


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
def test_ddp_sp2_grad_sync():
    """sp=2 over 2 ranks: ``dp_size == 1``, so DDP's own all-reduce does nothing."""
    _run_ddp_sp_grad_sync_test(world_size=2, ulysses_size=2)


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_ddp_sp4_grad_sync():
    _run_ddp_sp_grad_sync_test(world_size=4, ulysses_size=4)


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_ddp_dp2_sp2_grad_sync():
    """dp=2 x sp=2: the DDP all-reduce covers dp only; the sp half must still sync."""
    _run_ddp_sp_grad_sync_test(world_size=4, ulysses_size=2)


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_ddp_sp4_grad_sync_with_gradient_accumulation():
    """The trainer clips once per step, after several micro-batches have accumulated."""
    _run_ddp_sp_grad_sync_test(world_size=4, ulysses_size=4, micro_batches=2)


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
def test_ddp_sp2_grad_sync_from_meta_init():
    """``init_device`` defaults to meta, and only DDP's own materialize pass gets
    the module to a state its constructor accepts."""
    _run_ddp_sp_grad_sync_test(world_size=2, ulysses_size=2, init_device="meta")


if __name__ == "__main__":
    main()
