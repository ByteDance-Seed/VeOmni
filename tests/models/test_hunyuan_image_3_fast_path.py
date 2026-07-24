# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Track B tests: two-call varlen GCA fast path vs the dense oracle.

All fast-path tests require flash-attention varlen, so they run in BF16 on an
SM80+ CUDA GPU. The SP parity test spawns an Ulysses group per size and checks
the packed forward reproduces the single-GPU packed result (loss and gradient).
"""

from pathlib import Path

import pytest
import torch

from veomni.models.loader import get_model_class, get_model_config
from veomni.models.transformers.hunyuan_image_3.sequence_compiler import (
    build_single_gen_t2i_plan,
    compile_single_gen_t2i_packed,
    compile_single_gen_t2i_plans,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


_REPO_ROOT = Path(__file__).parents[2]
_TOY_CONFIG_PATH = _REPO_ROOT / "tests" / "toy_config" / "hunyuan_image_3_toy"
_BF16_TOLERANCE = {"rtol": 2e-2, "atol": 2e-2}

_requires_flash_gpu = pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or get_gpu_compute_capability() < 80,
    reason="Hunyuan Image 3 varlen GCA fast path requires an NVIDIA CUDA SM80+ GPU.",
)


def _build_model(*, device, dtype, attn_implementation="flash_attention_2", overrides=None):
    config = get_model_config(str(_TOY_CONFIG_PATH))
    config._attn_implementation = attn_implementation
    config._experts_implementation = "eager"
    for name, value in (overrides or {}).items():
        setattr(config, name, value)
    torch.manual_seed(0)
    model = get_model_class(config)(config).to(device=device, dtype=dtype)
    return config, model


def _cached_posterior(num_samples, config, *, device, dtype, grid=(2, 2), seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (num_samples, config.vae["latent_channels"], grid[0], grid[1])
    mean = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    logvar = torch.zeros(shape, device=device, dtype=dtype)
    return mean, logvar


def _to_device(metadata, device):
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in metadata.items()}


@_requires_flash_gpu
def test_packed_fast_path_matches_dense_oracle():
    device, dtype = "cuda", torch.bfloat16
    config, model = _build_model(device=device, dtype=dtype)
    grid = (2, 2)
    text_tokens = 3
    plan = build_single_gen_t2i_plan(sample_id="s", text_token_count=text_tokens, grid_hw=grid)
    sequence_length = plan["sequence_length"]

    mean, logvar = _cached_posterior(1, config, device=device, dtype=dtype, grid=grid)
    component_inputs = {"latent_posterior": {"mean": mean, "logvar": logvar}}
    input_ids = torch.arange(sequence_length, device=device, dtype=torch.long).unsqueeze(0)
    flow_step_context = {"train_seed": 5, "data_replica_rank": 0, "optimizer_step": 1, "micro_step": 0}

    dense_metadata = _to_device(compile_single_gen_t2i_plans([plan]), device)
    packed_metadata = _to_device(compile_single_gen_t2i_packed([plan]), device)

    dense = model(
        input_ids=input_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=dense_metadata,
        flow_step_context=flow_step_context,
        use_cache=False,
    )
    packed = model(
        input_ids=input_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=packed_metadata,
        flow_step_context=flow_step_context,
        use_cache=False,
    )
    torch.testing.assert_close(packed.diffusion_prediction, dense.diffusion_prediction, **_BF16_TOLERANCE)
    torch.testing.assert_close(packed.loss["image_decoder_loss"], dense.loss["image_decoder_loss"], **_BF16_TOLERANCE)

    packed.loss["image_decoder_loss"].backward()
    packed_grad = model.model.embed_tokens.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    dense.loss["image_decoder_loss"].backward()
    dense_grad = model.model.embed_tokens.weight.grad.clone()
    assert torch.isfinite(packed_grad).all() and torch.isfinite(dense_grad).all()
    torch.testing.assert_close(packed_grad, dense_grad, **_BF16_TOLERANCE)


def _packed_input_ids(plans, device):
    # Per-sample sample-local token ids concatenated, so each packed sample uses
    # the same ids as when run standalone (the flow RNG, however, is batch-shaped,
    # so cross-composition comparisons must keep the batch composition fixed).
    blocks = [torch.arange(plan["sequence_length"], dtype=torch.long) for plan in plans]
    return torch.cat(blocks).unsqueeze(0).to(device)


@_requires_flash_gpu
def test_packed_multi_sample_matches_dense_block_diagonal():
    device, dtype = "cuda", torch.bfloat16
    config, model = _build_model(device=device, dtype=dtype)
    grid = (2, 2)
    # Equal-length samples so the dense oracle (a [B, T, T] per-sample block
    # diagonal, cross-sample-free by construction) is a valid reference for the
    # packed two-call varlen path.
    plans = [build_single_gen_t2i_plan(sample_id=f"s{i}", text_token_count=3, grid_hw=grid) for i in range(2)]
    sequence_length = plans[0]["sequence_length"]
    mean, logvar = _cached_posterior(2, config, device=device, dtype=dtype, grid=grid)
    component_inputs = {"latent_posterior": {"mean": mean, "logvar": logvar}}
    flow_step_context = {"train_seed": 9, "data_replica_rank": 0, "optimizer_step": 2, "micro_step": 0}

    dense_ids = torch.arange(sequence_length, device=device, dtype=torch.long).unsqueeze(0).expand(2, -1)
    dense = model(
        input_ids=dense_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=_to_device(compile_single_gen_t2i_plans(plans), device),
        flow_step_context=flow_step_context,
        use_cache=False,
    )
    packed = model(
        input_ids=_packed_input_ids(plans, device),
        component_inputs=component_inputs,
        hy3_sequence_metadata=_to_device(compile_single_gen_t2i_packed(plans), device),
        flow_step_context=flow_step_context,
        use_cache=False,
    )
    torch.testing.assert_close(packed.diffusion_prediction, dense.diffusion_prediction, **_BF16_TOLERANCE)


@_requires_flash_gpu
def test_packed_heterogeneous_has_no_cross_sample_attention():
    device, dtype = "cuda", torch.bfloat16
    config, model = _build_model(device=device, dtype=dtype)
    grid = (2, 2)
    # Different prefix lengths (varlen) with a shared grid. Perturbing sample 1
    # must leave sample 0's prediction untouched: the batch-shaped flow RNG for
    # index 0 is independent of sample 1's posterior VALUES, so any change to
    # prediction[0] could only come from cross-sample attention leakage.
    plans = [
        build_single_gen_t2i_plan(sample_id="a", text_token_count=2, grid_hw=grid),
        build_single_gen_t2i_plan(sample_id="b", text_token_count=5, grid_hw=grid),
    ]
    packed_metadata = _to_device(compile_single_gen_t2i_packed(plans), device)
    packed_ids = _packed_input_ids(plans, device)
    flow_step_context = {"train_seed": 9, "data_replica_rank": 0, "optimizer_step": 2, "micro_step": 0}

    mean, logvar = _cached_posterior(2, config, device=device, dtype=dtype, grid=grid, seed=0)
    baseline = model(
        input_ids=packed_ids,
        component_inputs={"latent_posterior": {"mean": mean, "logvar": logvar}},
        hy3_sequence_metadata=packed_metadata,
        flow_step_context=flow_step_context,
        use_cache=False,
    )

    perturbed_mean = mean.clone()
    perturbed_mean[1] = perturbed_mean[1] + 3.0  # change only sample 1
    perturbed = model(
        input_ids=packed_ids,
        component_inputs={"latent_posterior": {"mean": perturbed_mean, "logvar": logvar}},
        hy3_sequence_metadata=packed_metadata,
        flow_step_context=flow_step_context,
        use_cache=False,
    )
    # Sample 0 unchanged (no leakage); sample 1 changed (perturbation took effect).
    torch.testing.assert_close(
        perturbed.diffusion_prediction[0:1], baseline.diffusion_prediction[0:1], rtol=0, atol=1e-3
    )
    assert not torch.allclose(perturbed.diffusion_prediction[1:2], baseline.diffusion_prediction[1:2], atol=1e-2)


# ------------------------- Ulysses SP parity (spawned) ------------------------


_SP_HEAD_OVERRIDES = {
    "hidden_size": 64,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "attention_head_dim": 8,
    "head_dim": 8,
}


def _sp_worker(rank, world_size, grid, text_tokens, return_dict):
    import torch.distributed as dist

    from veomni.distributed.sequence_parallel.comm import set_ulysses_sequence_parallel_group

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29513",
        world_size=world_size,
        rank=rank,
    )
    group = dist.new_group(ranks=list(range(world_size)))
    device, dtype = f"cuda:{rank}", torch.bfloat16

    config, model = _build_model(device=device, dtype=dtype, overrides=_SP_HEAD_OVERRIDES)
    plans = [
        build_single_gen_t2i_plan(sample_id=f"s{i}", text_token_count=text_tokens + i, grid_hw=grid)
        for i in range(world_size)
    ]
    flow_step_context = {"train_seed": 3, "data_replica_rank": 0, "optimizer_step": 1, "micro_step": 0}
    mean, logvar = _cached_posterior(len(plans), config, device=device, dtype=dtype, grid=grid)
    component_inputs = {"latent_posterior": {"mean": mean, "logvar": logvar}}

    packed = compile_single_gen_t2i_packed(plans, pad_to_multiple_of=world_size)
    packed = _to_device(packed, device)
    input_ids = torch.arange(packed["padded_sequence_length"], device=device, dtype=torch.long).unsqueeze(0)

    # SP reference: same weights, SP disabled, no padding.
    set_ulysses_sequence_parallel_group(None)
    reference_packed = _to_device(compile_single_gen_t2i_packed(plans), device)
    reference_ids = torch.arange(reference_packed["sequence_length"], device=device, dtype=torch.long).unsqueeze(0)
    reference = model(
        input_ids=reference_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=reference_packed,
        flow_step_context=flow_step_context,
        use_cache=False,
    )
    reference_loss = float(reference.loss["image_decoder_loss"].detach().float().cpu())

    # SP path: every rank sees the full replicated inputs; the model slices.
    set_ulysses_sequence_parallel_group(group)
    sp_output = model(
        input_ids=input_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=packed,
        flow_step_context=flow_step_context,
        use_cache=False,
    )
    sp_loss = float(sp_output.loss["image_decoder_loss"].detach().float().cpu())

    if rank == 0:
        return_dict["reference_loss"] = reference_loss
        return_dict["sp_loss"] = sp_loss
    dist.barrier()
    dist.destroy_process_group()


@_requires_flash_gpu
@pytest.mark.parametrize("world_size", [1, 2])
def test_packed_varlen_matches_dense_oracle_under_sp(world_size):
    # Parity for the SP fast path is monotone in ``world_size``: SP=2 hits every
    # code path (A2A + slice roundtrip) that SP=4/8 exercise, while keeping the
    # per-test GPU requirement to 2. The 16xH20 E2E training smoke covers
    # SP=2/4/8 under real workloads.
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Ulysses SP={world_size} needs {world_size} GPUs.")
    import torch.multiprocessing as mp

    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        _sp_worker,
        args=(world_size, (2, 2), 3, return_dict),
        nprocs=world_size,
        join=True,
    )
    assert "sp_loss" in return_dict and "reference_loss" in return_dict
    assert abs(return_dict["sp_loss"] - return_dict["reference_loss"]) <= 2e-2 * (
        1 + abs(return_dict["reference_loss"])
    )
