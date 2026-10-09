"""DCP round-trip for a parameter that is registered under more than one name.

``ParallelPlan.apply`` walks the deduplicated ``model.named_parameters()``, so a model
that registers one parameter under several names -- a shared encoder/decoder trunk --
gets a ``fqn2spec_info`` keyed by whichever name came first. State dicts are *not*
deduplicated, and ``_apply_extra_parallel_dim`` skips any name it cannot find there, so
the alias used to be written with the ExtraParallel dimension still collapsed: an
``[E/ep, ...]`` local shard recorded as if it were the whole ``[E, ...]`` tensor.

That is not a cosmetic duplicate. Both names reach ``set_model_state_dict`` for one
parameter object, so on resume the truncated alias can be assigned over the correctly
assembled tensor and leave every rank holding one rank's experts.

None of it is observable single-process, or at ``extra_parallel_sizes=(1,)``.
"""

import tempfile
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._tensor import Shard
from torch.distributed.checkpoint import FileSystemReader

from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.utils.device import get_torch_device


NUM_EXPERTS = 8


class ToyExpertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.Parameter(torch.ones(NUM_EXPERTS, 16, 32), requires_grad=True)
        self.mlp = nn.Parameter(torch.ones(16, 32), requires_grad=True)

    def forward(self) -> torch.Tensor:
        return self.experts.sum() + self.mlp.sum()


class ToyAliasedExpertsModel(nn.Module):
    """One expert layer reachable under two names, as a shared encoder/decoder trunk is.

    ``self.encoder`` *is* ``self.decoder``, so ``state_dict()`` reports
    ``encoder.experts`` and ``decoder.experts`` for a single parameter while
    ``named_parameters()`` reports only the first. The parallel plan names the decoder,
    matching the convention that the canonical name is the one the checkpoint uses.
    """

    _no_split_modules = ["ToyExpertLayer"]

    def __init__(self):
        super().__init__()
        self.decoder = ToyExpertLayer()
        self.encoder = self.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.sum() + self.decoder()) * 0.0

    def init_weights(self):
        self.decoder.experts.data.fill_(1.0)
        self.decoder.mlp.data.fill_(1.0)

    def get_parallel_plan(self):
        return ParallelPlan(extra_parallel_plan={"ep": {"decoder.experts": Shard(0)}})


def _build_model():
    from veomni.distributed.torch_parallelize import build_parallelize_model

    # FSDP2 requires meta init; the weights come from `ToyAliasedExpertsModel.init_weights`.
    return build_parallelize_model(
        ToyAliasedExpertsModel(),
        init_device="meta",
        weights_path=None,
        mixed_precision=MixedPrecisionConfig(enable=False),
        enable_gradient_checkpointing=False,
        basic_modules=[],
    )


def _fill_experts_distinctly(model, value: float) -> None:
    """Give each rank its own expert values, so a lost shard is visible."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".experts"):
                local = param.to_local() if hasattr(param, "to_local") else param
                local.fill_(value)


def _aliased_extra_parallel_worker(ep_size: int = 2):
    from veomni.checkpoint.dcp_checkpointer import ModelState
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.utils.device import get_device_type

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    get_torch_device().set_device(f"{get_device_type()}:{rank}")
    _init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        extra_parallel_sizes=(ep_size,),
        extra_parallel_names=("ep",),
    )

    model = _build_model()

    # Every state-dict name for a sharded parameter needs a spec, not just the canonical
    # one, or the ExtraParallel dimension is silently left collapsed for the rest.
    # The plan only runs when an ExtraParallel dimension is actually enabled, so there is
    # no `_fqn2spec_info` to inspect at ep_size 1.
    if ep_size > 1:
        fqn2spec_info = model._fqn2spec_info
        assert "decoder.experts" in fqn2spec_info, sorted(fqn2spec_info)
        assert "encoder.experts" in fqn2spec_info, (
            f"[Rank {rank}] the aliased name has no ExtraParallel spec, so it will be saved "
            f"as a local shard: {sorted(fqn2spec_info)}"
        )

    _fill_experts_distinctly(model, value=1.0 + rank)

    payload = [tempfile.mkdtemp(prefix="veomni-dcp-alias-") if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    checkpoint_dir = payload[0]

    before = {name: param.to_local().clone() for name, param in model.named_parameters() if name.endswith(".experts")}
    assert before, "the toy model exposes no expert tensors, so this test would pass vacuously"

    dcp.save(state_dict={"model": ModelState(model)}, checkpoint_id=checkpoint_dir)
    dist.barrier()

    # Save side: the expert dimension must be whole on disk. Read on every rank, since
    # asserting on rank 0 alone would strand the others on the next collective.
    metadata = FileSystemReader(checkpoint_dir).read_metadata()
    truncated = {
        key: tuple(item.size)
        for key, item in metadata.state_dict_metadata.items()
        if key.endswith(".experts") and getattr(item, "size", None) is not None and item.size[0] != NUM_EXPERTS
    }
    assert not truncated, (
        f"[Rank {rank}] {len(truncated)} expert tensor(s) written with the ExtraParallel "
        f"dimension collapsed (expected a leading {NUM_EXPERTS}): {truncated}"
    )

    # Load side: a truncated alias landing on the shared parameter shows up here as one
    # rank holding another's values.
    fresh = _build_model()
    _fill_experts_distinctly(fresh, value=-1.0)
    dcp.load(state_dict={"model": ModelState(fresh)}, checkpoint_id=checkpoint_dir)

    after = {name: param.to_local().clone() for name, param in fresh.named_parameters() if name.endswith(".experts")}
    assert set(after) == set(before), f"[Rank {rank}] expert names moved across the round trip"
    for name, expected in before.items():
        torch.testing.assert_close(
            after[name],
            expected,
            rtol=0,
            atol=0,
            msg=lambda formatted, name=name: (f"[Rank {rank}] {name} did not survive the DCP round trip\n{formatted}"),
        )

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("ep_size", [1, 2], ids=["no_ep", "ep2"])
def test_aliased_extra_parallel_parameters_survive_a_dcp_round_trip(ep_size: int):
    """The ``ep2`` arm is the regression.

    At ``ep_size == 1`` there is no expert dimension to collapse, so the alias is written
    correctly either way and the arm only guards the plumbing around it.
    """
    from ..tools.launch_utils import torchrun

    torchrun(partial(_aliased_extra_parallel_worker, ep_size), world_size=2)
