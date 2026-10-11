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
    from veomni import _apply_patches
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    from ..models.test_diffusion_gemma import build_toy_batch
    from ..tools.training_utils import make_eager_ops_config

    _apply_patches()

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
    from veomni import _apply_patches
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    from ..models.test_diffusion_gemma import build_toy_batch
    from ..tools.training_utils import make_eager_ops_config

    _apply_patches()

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


def _component_grad_norm_worker(ep_size: int = 2, return_logits: bool = True):
    """``print_ar_df_norm``'s split backward, against a single backward of the sum."""
    from types import SimpleNamespace

    from veomni import _apply_patches
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_config, build_foundation_model
    from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer
    from veomni.utils.device import get_device_type

    from ..models.test_diffusion_gemma import build_toy_batch, toy_runtime
    from ..tools.training_utils import make_eager_ops_config

    _apply_patches()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.getenv("LOCAL_RANK", rank))
    device = torch.device(f"{get_device_type()}:{local_rank}")
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

    model_config = build_config(TOY_CONFIG_PATH)
    if model_config.text_config.num_experts < ep_size:
        model_config.text_config.num_experts = ep_size
    assert model_config.text_config.num_experts % ep_size == 0
    model = build_foundation_model(
        config_path=model_config,
        weights_path=None,
        torch_dtype="bfloat16" if ep_size > 1 else "float32",
        init_device="meta",
        ops_implementation=ops_config,
    )
    model.train()
    model.config.print_ar_df_norm = True
    config = model.config
    model = build_parallelize_model(
        model,
        weights_path=None,
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(enable=True),
        enable_gradient_checkpointing=False,
        basic_modules=[],
    )

    trainer = DiffusionGemmaTrainer.__new__(DiffusionGemmaTrainer)
    trainer.base = SimpleNamespace(model=toy_runtime(model))

    batch = {k: v.to(device) for k, v in build_toy_batch(config, batch_size=2, prompt_len=24).items()}

    def grads():
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    model.zero_grad(set_to_none=True)
    model(**batch, use_cache=False, return_logits=return_logits).loss.backward()
    single = grads()
    assert single, f"[Rank {rank}] the single backward produced no gradients"

    model.zero_grad(set_to_none=True)
    outputs = model(**batch, use_cache=False, return_logits=return_logits)
    loss_dict = {"encoder_ar_loss": outputs.encoder_ar_loss, "diffusion_loss": outputs.diffusion_loss}
    trainer._backward_and_print_component_grad_norms(outputs.loss, loss_dict)
    split = grads()

    # Two backwards have to leave the optimizer exactly what one backward of the sum
    # would, per parameter and not just in aggregate. It comes out bit-identical, so the
    # tolerance is `assert_close`'s own for the dtype -- enough to survive a future
    # kernel reassociating a sum, and nothing like enough to hide a missing term. Note
    # that a tolerance derived from the tensor (`expected.abs().max()`) would not do:
    # reducing a sharded DTensor is a collective, and one rank failing the assert while
    # the others enter it is a hang instead of a message.
    assert set(split) == set(single), f"[Rank {rank}] the split backward changed which parameters have gradients"
    for name, expected in single.items():
        torch.testing.assert_close(
            split[name],
            expected,
            msg=lambda m, n=name: f"[Rank {rank}] {n} is not grad(AR + diffusion): {m}",
        )

    # Only the diffusion term is asserted to have a magnitude. These weights come from
    # `init_device="meta"` with no checkpoint, which leaves both losses at log(vocab)
    # and the AR term with a gradient of exactly zero -- 127 gradient tensors that all
    # sum to 0. Whether each term reports its own magnitude belongs to
    # `TestComponentGradNorms`, which runs on trained-shaped weights.
    assert float(loss_dict["diffusion_grad_norm"]) > 0, f"[Rank {rank}] the diffusion term reported no gradient"
    assert "encoder_ar_grad_norm" in loss_dict, f"[Rank {rank}] the AR term was never reported"
    if not return_logits:
        assert outputs.logits is None, f"[Rank {rank}] chunked training must not materialize logits"

    dist.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("ep_size", [1, 2], ids=["no_ep", "ep2"])
@pytest.mark.parametrize("return_logits", [True, False], ids=["materialized_logits", "chunked_diffusion_loss"])
def test_component_grad_norms_survive_the_shared_parameter_aliasing(ep_size: int, return_logits: bool):
    """``print_ar_df_norm``'s split backward, where FSDP2 can break it.

    A gradient stash used to be a list indexed by parameter order, and DiffusionGemma's
    encoder parameters *are* the decoder's, so `named_parameters` deduplicates them --
    until FSDP2 unshards and each side gets its own copy, taking the toy model from 131
    names to 244. Stashing in one state and returning in the other is what killed the
    26B recipe, in `zip(strict=True)`, before anything reached the optimizer.

    Nothing single-process sees it: the equivalent `TestComponentGradNorms` cases run on
    a bare model, where the count never moves. What they do cover, and this cannot, is
    the size of each term's norm -- these weights come from `init_device="meta"` with no
    checkpoint, so the AR term here has no gradient to measure.

    `return_logits=False` is the trainer's production path: the diffusion term reaches
    `lm_head.weight` through `ChunkLoss` rather than a materialized matmul. Without this
    arm the custom autograd Function never runs under FSDP2/EP.
    """
    from ..tools.launch_utils import torchrun

    torchrun(partial(_component_grad_norm_worker, ep_size, return_logits), world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("ep_size", [1, 2], ids=["no_ep", "ep2"])
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
def test_shared_weights_survive_fsdp2(ep_size: int):
    """Encoder/decoder sharing must survive `fully_shard` *and* a backward pass.

    The EP variant is the one that pinned down two separate bugs: the expert modules have
    to be collapsed into one (otherwise two FSDP units claim the same tensor on different
    meshes), and the experts forward has to reach VeOmni's fused kernel rather than the
    per-expert loop over global expert ids.
    """
    from ..tools.launch_utils import torchrun

    torchrun(partial(_shared_weights_worker, ep_size), world_size=2)
