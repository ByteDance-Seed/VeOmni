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
"""DCP round-trip for DiffusionGemma's shared encoder/decoder parameters under EP.

DiffusionGemma is the only VeOmni model whose encoder and decoder are the same
parameter objects, so its state dict carries every trunk tensor twice: once under
the canonical ``model.decoder.*`` name and once under the aliased
``model.encoder.language_model.*`` name. ``_apply_extra_parallel_dim`` keys off
``_fqn2spec_info``, which only lists the canonical names from the parallel plan, so
the alias used to be written with the EP dimension still collapsed -- a
``[E/ep, ...]`` local shard recorded as if it were the whole ``[E, ...]`` tensor.

That is not a cosmetic duplicate. On the save side it makes the HF export
unloadable, and on the load side both names reach ``set_model_state_dict`` for one
parameter object, so the truncated alias can overwrite the correctly assembled
expert weights and leave every EP rank holding the same ``E/ep`` experts.

None of it is observable single-process, or at ``ep_size == 1``.
"""

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from veomni.arguments import MixedPrecisionConfig


TOY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "toy_config", "diffusion_gemma_toy")

EXPERT_LEAVES = (".experts.gate_up_proj", ".experts.down_proj")


def _build_model(ep_size: int):
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model

    from ..tools.training_utils import make_eager_ops_config

    ops_config = make_eager_ops_config()
    ops_config.attn_implementation = "sdpa"
    if ep_size > 1:
        ops_config.moe_implementation = "fused_triton"

    model = build_foundation_model(
        config_path=TOY_CONFIG_PATH,
        weights_path=None,
        torch_dtype="bfloat16" if ep_size > 1 else "float32",
        init_device="meta",
        ops_implementation=ops_config,
    )
    return build_parallelize_model(
        model,
        weights_path=None,
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(enable=True),
        enable_gradient_checkpointing=False,
        basic_modules=[],
    )


def _expert_params(model) -> dict[str, torch.Tensor]:
    return {name: param for name, param in model.named_parameters() if name.endswith(EXPERT_LEAVES)}


def _fill_experts_distinctly(model, rank: int) -> None:
    """Give every EP rank's expert shard its own value, so a lost shard is visible.

    ``init_device="meta"`` leaves the weights unmaterialised and every rank would
    otherwise agree, which is exactly the state that hides a shard overwrite. The
    per-tensor offsets are whole numbers because these weights are bf16 under EP, where
    fractional steps around 2.0 collapse onto the same representable value.
    """
    with torch.no_grad():
        for index, (_, param) in enumerate(sorted(_expert_params(model).items())):
            local = param.to_local() if hasattr(param, "to_local") else param
            local.fill_(1000 * (rank + 1) + index)


def _step_once(model, optimizer, device) -> None:
    """Populate the optimizer state, so saving it exercises more than empty dicts."""
    from ..models.test_diffusion_gemma import build_toy_batch

    batch = {k: v.to(device) for k, v in build_toy_batch(model.config, batch_size=2, prompt_len=24).items()}
    model(**batch).loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def _dcp_round_trip_worker(ep_size: int = 2):
    import tempfile

    from torch.distributed.checkpoint import FileSystemReader

    from veomni.checkpoint import layout
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.optim import build_optimizer
    from veomni.utils.device import get_device_type

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")
    torch.cuda.set_device(device)
    _init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        extra_parallel_sizes=(ep_size,),
        extra_parallel_names=("ep",),
    )

    model = _build_model(ep_size)
    _fill_experts_distinctly(model, rank)

    # Every rank must agree on the directory, and it has to outlive the save.
    if rank == 0:
        checkpoint_root = tempfile.mkdtemp(prefix="dg-dcp-ep-")
        payload = [checkpoint_root]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    checkpoint_root = payload[0]

    # Save the optimizer too. Its state dict reaches the same ExtraParallel preprocessing
    # through `key_match="substring"`, which asserts that a state key matches exactly one
    # registered FQN -- and registering the alias FQNs is what could have made that
    # ambiguous. Without a step the state is empty and that path never runs.
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.0, fused=False)
    _step_once(model, optimizer, device)
    _fill_experts_distinctly(model, rank)

    before = {name: param.to_local().clone() for name, param in _expert_params(model).items()}
    assert before, "the toy model exposes no expert tensors, so this test would pass vacuously"

    # `global_steps` must be truthy for the real `global_step_N` layout; 0 saves to the root.
    DistributedCheckpointer.save(checkpoint_root, {"model": model, "optimizer": optimizer}, global_steps=1)
    dist.barrier()

    checkpoint_dir = layout.step_dir(checkpoint_root, 1)

    # Save side: no expert tensor may be recorded with a truncated global expert dim.
    # `num_experts` is the authority; a `[E/ep, ...]` entry is an un-restored local shard.
    # Read on every rank -- the metadata read is local, and asserting on rank 0 alone
    # would strand the others on the next collective instead of failing the test.
    num_experts = model.config.text_config.num_experts
    metadata = FileSystemReader(layout.weights_dir(checkpoint_dir)).read_metadata()
    truncated = {
        key: tuple(item.size)
        for key, item in metadata.state_dict_metadata.items()
        if key.endswith(EXPERT_LEAVES) and getattr(item, "size", None) is not None and item.size[0] != num_experts
    }
    assert not truncated, (
        f"[Rank {rank}] {len(truncated)} expert tensors were written with the EP dimension "
        f"collapsed (expected a leading {num_experts}): {dict(list(truncated.items())[:3])}"
    )

    # Load side: the experts must come back bit-for-bit on every rank. A truncated alias
    # landing on the shared parameter shows up here as rank 1 holding rank 0's values.
    fresh = _build_model(ep_size)
    fresh_optimizer = build_optimizer(fresh, lr=1e-4, weight_decay=0.0, fused=False)
    _step_once(fresh, fresh_optimizer, device)
    DistributedCheckpointer.load(checkpoint_dir, {"model": fresh, "optimizer": fresh_optimizer})

    after = {name: param.to_local().clone() for name, param in _expert_params(fresh).items()}
    assert set(after) == set(before), f"[Rank {rank}] expert names moved across the round trip"
    for name, expected in before.items():
        torch.testing.assert_close(
            after[name],
            expected,
            rtol=0,
            atol=0,
            msg=lambda formatted, name=name: f"[Rank {rank}] {name} did not survive the DCP round trip\n{formatted}",
        )

    dist.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("ep_size", [1, 2], ids=["no_ep", "ep2"])
def test_experts_survive_a_dcp_round_trip(ep_size: int):
    """The aliased encoder name must not smuggle a local expert shard into the checkpoint.

    The ``ep2`` arm is the regression. At ``ep_size == 1`` there is no expert dimension
    to collapse, so the alias is written correctly and the arm only guards the plumbing.
    """
    from ..tools.launch_utils import torchrun

    torchrun(partial(_dcp_round_trip_worker, ep_size), world_size=2)
