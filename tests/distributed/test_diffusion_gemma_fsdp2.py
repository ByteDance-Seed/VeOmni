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
"""FSDP2 coverage for DiffusionGemma's shared encoder/decoder parameters.

DiffusionGemma is the only VeOmni model whose encoder and decoder *are* the same
parameter objects, which puts it on a collision course with FSDP2's per-unit parameter
management. None of that is observable single-process, so this is the only place the
combination is exercised at all.

Concretely, listing both layer types in ``_no_split_modules`` -- as upstream does -- makes
``fully_shard`` build two units over one set of parameters. Sharding and the tie both look
fine afterwards, so only the backward pass reveals it, dying in ``setStorage`` once one
unit frees the all-gather buffer the other still needs.
"""

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import MixedPrecisionConfig


TOY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "toy_config", "diffusion_gemma_toy")


def _local_numel(param: torch.Tensor) -> int:
    """Local shard size. ``DTensor.numel()`` reports the global one, so it can't show sharding."""
    return param.to_local().numel() if hasattr(param, "to_local") else param.numel()


def _shared_parameter_report(model) -> tuple[int, int, list[str]]:
    """Count how many encoder parameters are still the very same object as the decoder's."""
    base = model.model
    decoder_params = dict(base.decoder.named_parameters())
    total, shared, broken = 0, 0, []
    for name, param in base.encoder.language_model.named_parameters():
        decoder_param = decoder_params.get(name)
        if decoder_param is None:
            continue
        total += 1
        if decoder_param is param:
            shared += 1
        else:
            broken.append(name)
    return shared, total, broken


def _shared_weights_worker(ep_size: int = 1):
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    from ..models.test_diffusion_gemma import build_toy_batch
    from ..tools.training_utils import make_eager_ops_config

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")
    _init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        extra_parallel_sizes=(ep_size,),
        extra_parallel_names=("ep",),
    )

    ops_config = make_eager_ops_config()
    # The decoder's 4D masks rule out flash attention, and the model rejects it outright.
    ops_config.attn_implementation = "sdpa"
    if ep_size > 1:
        # Under EP the router still emits global expert ids while each rank holds only
        # `E/ep` experts, so the eager per-expert loop cannot work -- the fused kernel owns
        # the all-to-all. It requires bf16, which is also what real training runs.
        ops_config.moe_implementation = "fused_triton"
    torch_dtype = "bfloat16" if ep_size > 1 else "float32"

    model = build_foundation_model(
        config_path=TOY_CONFIG_PATH,
        weights_path=None,
        torch_dtype=torch_dtype,
        init_device="meta",
        ops_implementation=ops_config,
    )
    model.train()
    config = model.config

    model = build_parallelize_model(
        model,
        weights_path=None,
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(enable=True),
        enable_gradient_checkpointing=False,
        basic_modules=[],
    )

    shared, total, broken = _shared_parameter_report(model)
    assert total > 0, "the report found no shared parameters at all, so it is not testing anything"
    assert shared == total, f"[Rank {rank}] fully_shard broke {len(broken)} shared parameters: {broken[:5]}"

    # The tie must not come at the cost of sharding: the shared parameters have to be
    # owned by the decoder's unit, not silently replicated on every rank.
    named = list(model.named_parameters())
    replicated = [
        n
        for n, p in named
        if _local_numel(p) == p.numel()
        and p.numel() > 1
        # Expert tensors are split across ranks by EP itself. With ep_size == world_size
        # the FSDP subgroup inside the EP group has one rank, so FSDP rightly leaves them
        # alone; they are still only 1/ep of the full expert set.
        and not (ep_size > 1 and ".experts." in n)
    ]
    assert not replicated, f"[Rank {rank}] parameters left unsharded: {replicated[:5]}"

    batch = {k: v.to(device) for k, v in build_toy_batch(config, batch_size=2, prompt_len=24).items()}
    loss = model(**batch).loss
    assert torch.isfinite(loss), f"[Rank {rank}] loss is not finite: {loss.item()}"

    # The actual regression: this is where the two-unit layout dies.
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"[Rank {rank}] non-finite gradient in {name}"

    shared, total, broken = _shared_parameter_report(model)
    assert shared == total, f"[Rank {rank}] backward broke {len(broken)} shared parameters: {broken[:5]}"

    dist.barrier()


def _recompute_worker(ep_size: int = 1):
    """Same model, same batch, gradient checkpointing off then on."""
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    from ..models.test_diffusion_gemma import build_toy_batch
    from ..tools.training_utils import make_eager_ops_config

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"{get_device_type()}:{rank}")
    _init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        extra_parallel_sizes=(ep_size,),
        extra_parallel_names=("ep",),
    )

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
    model.train()
    config = model.config
    model = build_parallelize_model(
        model,
        weights_path=None,
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(enable=True),
        enable_gradient_checkpointing=False,
        basic_modules=[],
    )

    batch = {k: v.to(device) for k, v in build_toy_batch(config, batch_size=2, prompt_len=24).items()}

    def run():
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        return loss.detach().clone(), {
            n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
        }

    plain_loss, plain_grads = run()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoder_layers = model.model.encoder.language_model.layers
    assert all(layer.gradient_checkpointing for layer in encoder_layers), (
        "the encoder is not being recomputed, so this test would pass without testing anything"
    )
    recomputed_loss, recomputed_grads = run()

    # The replay all-gathers the units it is about to read, and every one of those has to
    # be put back. A unit left unsharded registers the *unsharded* parameter, so the
    # gradient comparison below would not even see the real one -- it reads
    # `named_parameters()`, which is also what the optimizer steps.
    from torch.distributed.fsdp import FSDPModule

    left_unsharded = [
        name
        for name, module in model.named_modules()
        if isinstance(module, FSDPModule)
        and (group := module._get_fsdp_state()._fsdp_param_group) is not None
        and group.is_unsharded
        and module is not model
    ]
    assert not left_unsharded, f"[Rank {rank}] FSDP units left unsharded after backward: {left_unsharded[:5]}"

    torch.testing.assert_close(plain_loss, recomputed_loss, msg=f"[Rank {rank}] loss diverged under recompute")
    assert plain_grads, f"[Rank {rank}] no gradients to compare"
    for name, grad in plain_grads.items():
        torch.testing.assert_close(grad, recomputed_grads[name], msg=lambda m, n=name: f"[Rank {rank}] {n}: {m}")

    dist.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("ep_size", [1, 2], ids=["no_ep", "ep2"])
@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
def test_recompute_matches_the_plain_backward(ep_size: int):
    """Recomputing the encoder must not change a single gradient.

    The EP variant is the one with teeth. Under expert parallel the encoder and the
    decoder share one expert FSDP unit, and FSDP2 unshards a unit from a hook on its
    forward outputs -- which fires long after the replay that has to produce those
    outputs. The decoder's backward reshards the unit first, so without the explicit
    unshard in `DiffusionGemmaCacheAwareCheckpointLayer` the encoder's replay reads fp32
    sharded weights and the fused MoE kernel refuses them outright.
    """
    from ..tools.launch_utils import torchrun

    torchrun(partial(_recompute_worker, ep_size), world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("ep_size", [1, 2], ids=["no_ep", "ep2"])
@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
def test_shared_weights_survive_fsdp2(ep_size: int):
    """Encoder/decoder sharing must survive `fully_shard` *and* a backward pass.

    The EP variant is the one that pinned down two separate bugs: the expert modules have
    to be collapsed into one (otherwise two FSDP units claim the same tensor on different
    meshes), and the experts forward has to reach VeOmni's fused kernel rather than the
    per-expert loop over global expert ids.
    """
    from ..tools.launch_utils import torchrun

    torchrun(partial(_shared_weights_worker, ep_size), world_size=2)
