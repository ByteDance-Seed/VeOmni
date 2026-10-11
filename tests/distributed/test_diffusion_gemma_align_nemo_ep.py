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
"""Folding `per_expert_scale` must pair each expert with its own scale, under sharding.

`align_nemo.fold_per_expert_scale` rewrites every expert as ``W' = sW``. Which scale
reaches which expert is the whole question, and it is one that cannot be asked in a
single process: by the time the fold runs in training, expert parallel has already
cut ``down_proj`` to a contiguous block of experts per rank while ``per_expert_scale``
stayed whole, and FSDP has cut what remains again on a different mesh.

Nor can the fold be asked to check itself. A fold that pairs expert ``e`` with expert
``e'``'s scale leaves a perfectly self-consistent model -- every later forward,
backward and checkpoint agrees with itself -- and the obvious post-condition, that
each expert's norm grew by its own scale, recomputes the expectation from the vector
it just multiplied by, so it holds for any scale, right or wrong.

So the expectation is stated in whole-tensor terms, with ``full_tensor()`` doing the
reassembly, and never reconstructs which experts are local -- that reconstruction is
the step under test. The defect this caught was precisely there: the fold read
``down_proj``'s FSDP placement as if it cut the expert dimension, when VeOmni cuts
the intermediate one, which on four ranks left each holding a differently shaped
scale and deadlocked the fold.

Comparing the pre- and post-fold *forward* would also be valid in principle -- the
fold is a reparameterization -- but not reachable: running a forward first is what
makes FSDP2 rebind the shared encoder/decoder parameters onto separate buffers, a
state the trainer never sees because it folds before the first step.
"""

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from veomni.arguments import MixedPrecisionConfig


TOY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "toy_config", "diffusion_gemma_toy")


def _build_model(ep_size: int):
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.auto import build_foundation_model

    from ..tools.training_utils import make_eager_ops_config

    ops_config = make_eager_ops_config()
    ops_config.attn_implementation = "sdpa"
    if ep_size > 1:
        # Eager expert dispatch indexes the stacked weights by global expert id, which
        # is out of range once EP has cut them; `fused_moe_forward` owns that mapping.
        ops_config.moe_implementation = "fused_triton"

    model = build_foundation_model(
        config_path=TOY_CONFIG_PATH,
        weights_path=None,
        torch_dtype="bfloat16",
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


def _set_from_global(param: torch.Tensor, whole: torch.Tensor) -> None:
    """Write a value every rank agrees on into whatever shard this rank owns."""
    from torch.distributed.tensor import distribute_tensor

    whole = whole.to(device=param.device, dtype=param.dtype)
    if isinstance(param, DTensor):
        param.to_local().copy_(distribute_tensor(whole, param.device_mesh, param.placements).to_local())
    else:
        param.copy_(whole)


@torch.no_grad()
def _give_the_model_values(model, device) -> torch.Tensor:
    """Materialize the meta shell with values, and hand back the expert scales used.

    The scales span [0.25, 1.75] rather than the checkpoint's [0.98, 1.02]. The point
    is not realism: a mispairing has to be loud enough that no tolerance loose enough
    for bf16 could absorb it.
    """
    generator = torch.Generator(device="cpu").manual_seed(0)
    for _, param in model.named_parameters():
        local = param.to_local() if hasattr(param, "to_local") else param
        if local.numel():
            local.copy_(torch.empty(local.shape).uniform_(-0.05, 0.05, generator=generator).to(local.dtype))

    num_experts = model.config.text_config.num_experts
    scales = torch.linspace(0.25, 1.75, num_experts)
    for layer in model.model.decoder.layers:
        _set_from_global(layer.router.per_expert_scale, scales)
    return scales


def _drift(before: torch.Tensor, after: torch.Tensor) -> float:
    return ((after - before).norm() / before.norm()).item()


def _expected_products(model, scales, ep_size) -> list[torch.Tensor]:
    """What each layer's `down_proj` must become, in whole-tensor terms.

    Deliberately not expressed per shard. Reconstructing which experts are local is
    the step under test, so an expectation that reconstructs them too would agree
    with the fold by construction -- and would have missed the real defect here,
    which was the fold reading `down_proj`'s FSDP placement as if it cut the expert
    dimension when it cuts the intermediate one instead.

    `full_tensor()` sidesteps the question by letting DTensor reassemble the tensor.
    What is left to assume is only expert parallel's block, which is a contiguous
    slice keyed on the ep rank and is covered on its own in
    `tests/models/test_diffusion_gemma.py::TestAlignNemo`.
    """
    from veomni.distributed.parallel_state import get_parallel_state

    ep_rank = get_parallel_state().extra_parallel_rank("ep") if ep_size > 1 else 0
    expected = []
    for layer in model.model.decoder.layers:
        down_proj = layer.experts.down_proj
        whole = down_proj.full_tensor() if isinstance(down_proj, DTensor) else down_proj
        block = scales[ep_rank * whole.shape[0] : (ep_rank + 1) * whole.shape[0]].to(whole.device)
        expected.append(whole.float() * block[:, None, None].float())
    return expected


def _fold_preserves_the_forward_worker(ep_size: int):
    from veomni import _apply_patches
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.models.transformers.diffusion_gemma import align_nemo
    from veomni.utils.device import get_device_type

    _apply_patches()

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
    scales = _give_the_model_values(model, device)
    expected = _expected_products(model, scales, ep_size)

    os.environ["VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD"] = "1"
    try:
        folded = align_nemo.fold_per_expert_scale(model)
    finally:
        del os.environ["VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD"]

    expected_layers = model.config.text_config.num_hidden_layers
    assert folded == expected_layers, (
        f"[Rank {rank}] folded {folded} expert tensors, expected {expected_layers} -- the encoder and decoder "
        "share one stack, so a different count means the dedup mis-fired"
    )
    for layer in model.model.decoder.layers:
        scale = layer.router.per_expert_scale
        local = scale.to_local() if hasattr(scale, "to_local") else scale
        assert torch.all(local == 1.0), f"[Rank {rank}] `per_expert_scale` was not reset to one"
        assert not scale.requires_grad, f"[Rank {rank}] `per_expert_scale` is still trainable after the fold"

    # bf16 (which EP's expert kernel requires) holds ~3 decimal digits, and the fold
    # rounds once on the way back into the parameter, so `rtol` is that rounding and
    # nothing else. A mispairing is not a near miss at this tolerance: neighbouring
    # scales here differ by 21%, so pairing an expert with the wrong one lands four
    # orders of magnitude outside it.
    for index, (layer, want) in enumerate(zip(model.model.decoder.layers, expected)):
        down_proj = layer.experts.down_proj
        got = down_proj.full_tensor() if isinstance(down_proj, DTensor) else down_proj
        torch.testing.assert_close(
            got.float(),
            want,
            rtol=8e-3,
            atol=0,
            msg=lambda formatted, index=index: (
                f"[Rank {rank}] layer {index}'s experts were not each scaled by their own entry of "
                f"`per_expert_scale`, which spans [{scales.min():.2f}, {scales.max():.2f}]\n{formatted}"
            ),
        )

    dist.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 devices")
@pytest.mark.parametrize("ep_size", [1, 2], ids=["no_ep", "ep2"])
def test_folding_the_expert_scale_leaves_the_forward_alone(ep_size: int):
    """`ep2` is the arm that matters: it is the one where the two operands are cut
    differently, EP taking a block of experts from `down_proj` and leaving
    `per_expert_scale` whole. At `ep_size == 1` only FSDP's split is in play, which
    both tensors get identically, so that arm guards the plumbing rather than the
    pairing."""
    from ..tools.launch_utils import torchrun

    torchrun(partial(_fold_preserves_the_forward_worker, ep_size), world_size=4)
