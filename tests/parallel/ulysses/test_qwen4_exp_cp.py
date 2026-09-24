# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU/Gloo CP correctness: python -m torch.distributed.run --nproc_per_node=4 this_file.

Uses the generated production model and real U/CP collectives. Only GDN device
kernels are replaced by deterministic PyTorch references. No NPU claim.
"""

import copy
import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import AutoConfig

from veomni.distributed.parallel_state import _init_parallel_state as init_parallel_state
from veomni.distributed.parallel_state import clear_parallel_state


# Import real MoE communication without the unrelated CUDA GEMM package init.
# This CPU-only test never calls a MoE kernel.
moe_package = types.ModuleType("veomni.distributed.moe")
moe_package.__path__ = [str(Path(__file__).resolve().parents[3] / "veomni/distributed/moe")]
sys.modules[moe_package.__name__] = moe_package
model_code = importlib.import_module(
    "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_"
    + os.environ.get("QWEN4_CP_TEST_MODULE", "gpu")
)


MODULE = model_code.__name__
NO_SP = SimpleNamespace(
    sp_enabled=False,
    sp_size=1,
    sp_rank=0,
    sp_group=None,
    ulysses_enabled=False,
    ulysses_size=1,
    ulysses_rank=0,
    ulysses_group=None,
    cp_enabled=False,
    cp_size=1,
    cp_rank=0,
    cp_group=None,
    extra_parallel_sizes={"ple": 1},
)


def conv_ref(x, weight, bias=None, activation=None, **kwargs):
    boundaries = kwargs.get("cu_seqlens", kwargs.get("cu_seq_lens_q"))
    bounds = [0, x.shape[1]] if boundaries is None else boundaries.tolist()
    outputs = []
    for a, b in zip(bounds, bounds[1:]):
        y = torch.nn.functional.conv1d(
            x[:, a:b].transpose(1, 2), weight[:, None], bias, padding=weight.shape[-1] - 1, groups=weight.shape[0]
        )[..., : b - a]
        outputs.append(torch.nn.functional.silu(y).transpose(1, 2))
    return torch.cat(outputs, 1), None


def conv_baseline(x, weight, bias=None, **kwargs):
    return conv_ref(x.transpose(1, 2), weight, bias=bias, **kwargs)[0].transpose(1, 2)


def recurrent_ref(q, k, v, g, beta, **kwargs):
    boundaries = kwargs.pop("cu_seqlens", None)
    bounds = [0, q.shape[1]] if boundaries is None else boundaries.tolist()
    kwargs.pop("head_first", None)
    initial = kwargs.pop("initial_state", None)
    want_final = kwargs.pop("output_final_state", False)
    outputs, states = [], []
    for idx, (a, b) in enumerate(zip(bounds, bounds[1:])):
        y, state = model_code.torch_recurrent_gated_delta_rule(
            q[:, a:b],
            k[:, a:b],
            v[:, a:b],
            g[:, a:b],
            beta[:, a:b],
            initial_state=initial[idx : idx + 1] if initial is not None else None,
            output_final_state=want_final,
            **kwargs,
        )
        outputs.append(y)
        if want_final:
            states.append(state)
    return torch.cat(outputs, 1), torch.cat(states) if want_final else None


def compare(module, full_input, local_input, baseline_kwargs, local_kwargs, label):
    baseline = copy.deepcopy(module)
    with (
        patch(f"{MODULE}.get_parallel_state", return_value=NO_SP),
        patch(f"{MODULE}.veomni_qsa_attention_implementation", SimpleNamespace(value="eager")),
        patch(f"{MODULE}.causal_conv1d_fn", conv_baseline),
        patch(f"{MODULE}.torch_chunk_gated_delta_rule", recurrent_ref),
    ):
        expected = baseline(full_input, **baseline_kwargs)
        expected = expected[0] if isinstance(expected, tuple) else getattr(expected, "last_hidden_state", expected)
        # Nonuniform upstream detects token permutation, not only sum invariance.
        upstream = torch.arange(expected.numel(), dtype=expected.dtype).reshape_as(expected).sin()
        (expected * upstream).sum().backward()
    actual = module(local_input, **local_kwargs)
    actual = actual[0] if isinstance(actual, tuple) else getattr(actual, "last_hidden_state", actual)
    rank, world = dist.get_rank(), dist.get_world_size()
    width = expected.shape[1] // world
    sl = slice(rank * width, (rank + 1) * width)
    torch.testing.assert_close(actual, expected[:, sl], atol=5e-5, rtol=5e-5)
    (actual * upstream[:, sl]).sum().backward()
    if local_input.is_floating_point():
        torch.testing.assert_close(local_input.grad, full_input.grad[:, sl], atol=8e-5, rtol=8e-5)
    for (name, param), (_, ref) in zip(module.named_parameters(), baseline.named_parameters()):
        if ref.grad is None:
            assert param.grad is None, name
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        dist.all_reduce(grad)
        torch.testing.assert_close(
            grad, ref.grad, atol=2e-4, rtol=2e-4, msg=lambda s, name=name: f"{label}/{name}: {s}"
        )
    if rank == 0:
        print(f"PASS {label}: outputs, input gradients, all parameter gradients", flush=True)


def check_independent_segments(module, hidden, ids, boundaries, label):
    """Compare packed mixing to independent examples, without inserting EOS."""
    packed = copy.deepcopy(module)
    separate = copy.deepcopy(module)
    x = hidden.detach().clone().requires_grad_()
    y = hidden.detach().clone().requires_grad_()
    is_ple = label == "PLE"
    with patch(f"{MODULE}.get_parallel_state", return_value=NO_SP):
        if is_ple:
            actual = packed(x, input_ids=ids, past_key_values=None, cu_seq_lens_q=boundaries)
        else:
            actual = packed(x, linear_attn_cu_seq_lens_q=boundaries)
        pieces = []
        for a, b in zip(boundaries.tolist(), boundaries.tolist()[1:]):
            local_bounds = boundaries.new_tensor([0, b - a])
            if is_ple:
                out = separate(y[:, a:b], input_ids=ids[:, a:b], past_key_values=None, cu_seq_lens_q=local_bounds)
            else:
                out = separate(y[:, a:b], linear_attn_cu_seq_lens_q=local_bounds)
            pieces.append(out)
        expected = torch.cat(pieces, 1)
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
        weight = torch.arange(actual.numel()).reshape_as(actual).sin()
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=8e-5, rtol=8e-5)
        for (name, param), (_, ref) in zip(packed.named_parameters(), separate.named_parameters()):
            if ref.grad is not None:
                torch.testing.assert_close(param.grad, ref.grad, atol=2e-4, rtol=2e-4, msg=name)
    if dist.get_rank() == 0:
        print(f"PASS {label}: packed versus independently evaluated examples", flush=True)


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    world, rank = dist.get_world_size(), dist.get_rank()
    cp_size = int(os.environ.get("QWEN4_CP_TEST_CP_SIZE", "2"))
    assert world >= 2 and world % cp_size == 0
    init_parallel_state(
        dp_size=1, ulysses_size=world // cp_size, cp_size=cp_size, allow_hybrid_cp=True, device_type="cpu", name=None
    )
    config = AutoConfig.from_pretrained("tests/toy_config/qwen4_exp_toy").text_config
    config.num_attention_heads = world // cp_size * 3  # U divides heads; full SP does not.
    config.num_key_value_heads = int(os.environ.get("QWEN4_CP_TEST_KV_HEADS", "1"))
    config.linear_num_key_heads = int(os.environ.get("QWEN4_CP_TEST_GDN_KEY_HEADS", str(world)))
    config.linear_num_value_heads = config.linear_num_key_heads * 3
    gdn_head_dim = int(os.environ.get("QWEN4_CP_TEST_GDN_HEAD_DIM", "64"))
    config.linear_key_head_dim = gdn_head_dim
    config.linear_value_head_dim = gdn_head_dim
    config._attn_implementation = "eager"
    length = world * 8
    sl = slice(rank * 8, (rank + 1) * 8)
    bounds_list = [0, 7, length]
    if os.environ.get("QWEN4_CP_TEST_PACKED_EDGES"):
        width = length // cp_size
        bounds_list = sorted({0, 1, width - 1, width, width + 1, width + 3, length - 2, length})
    bounds = torch.tensor(bounds_list, dtype=torch.int32)
    torch.manual_seed(123)
    hidden = torch.randn(1, length, config.hidden_size)
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    angles = torch.arange(length * rotary_dim).reshape(1, length, rotary_dim).float() / 13
    rope = (angles.cos(), angles.sin())
    qsa = model_code.Qwen4ExpTextAttention(config, layer_idx=1).float()
    if cp_size > 1:
        try:
            qsa(hidden[:, sl], position_embeddings=rope, attention_mask=torch.ones(1, 1, length, length))
        except NotImplementedError as exc:
            assert "explicit attention masks" in str(exc)
        else:
            raise AssertionError("CP2 must reject explicit masks before collectives")
    compare(
        qsa,
        hidden.clone().requires_grad_(),
        hidden[:, sl].clone().requires_grad_(),
        dict(position_embeddings=rope, attention_mask=None, cu_seq_lens_q=bounds),
        dict(position_embeddings=rope, attention_mask=None, cu_seq_lens_q=bounds),
        "QSA packed/GQA",
    )
    ple = model_code.Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0).float()
    ids = (torch.arange(length).unsqueeze(0) + 3) % config.vocab_size
    ids[ids == config.eos_token_id] = 3
    check_independent_segments(ple, hidden.repeat(1, 1, config.hc_count), ids, bounds, "PLE")
    compare(
        ple,
        hidden.repeat(1, 1, config.hc_count).requires_grad_(),
        hidden[:, sl].repeat(1, 1, config.hc_count).requires_grad_(),
        dict(input_ids=ids, past_key_values=None, cu_seq_lens_q=bounds),
        dict(input_ids=ids[:, sl], past_key_values=None, cu_seq_lens_q=bounds),
        "PLE token/conv halos",
    )
    gdn = model_code.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).float()
    gdn.veomni_gdn_cp_implementation = os.environ.get("QWEN4_CP_TEST_GDN_CP", "headwise")
    gdn.veomni_causal_conv1d_fn = conv_ref
    gdn.veomni_chunk_gated_delta_rule = recurrent_ref
    check_independent_segments(gdn, hidden, ids, bounds, "GDN")
    # Baseline packed segments must reset recurrent state too.
    compare(
        gdn,
        hidden.clone().requires_grad_(),
        hidden[:, sl].clone().requires_grad_(),
        dict(cu_seq_lens_q=bounds),
        dict(linear_attn_cu_seq_lens_q=bounds),
        "GDN packed recurrence",
    )
    model = model_code.Qwen4ExpTextModel(config).float()
    for layer in model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.veomni_gdn_cp_implementation = os.environ.get("QWEN4_CP_TEST_GDN_CP", "headwise")
            layer.linear_attn.veomni_causal_conv1d_fn = conv_ref
            layer.linear_attn.veomni_chunk_gated_delta_rule = recurrent_ref
    positions = torch.cat([torch.arange(b - a) for a, b in zip(bounds_list, bounds_list[1:])]).unsqueeze(0)
    mask = torch.ones(1, length, dtype=torch.long)
    compare(
        model,
        ids.clone(),
        ids[:, sl].clone(),
        dict(position_ids=positions, attention_mask=mask, cu_seq_lens_q=bounds, linear_attn_cu_seq_lens_q=bounds),
        dict(
            position_ids=positions[:, sl], attention_mask=mask, cu_seq_lens_q=bounds, linear_attn_cu_seq_lens_q=bounds
        ),
        "packed text model with PLE/GDN/QSA/MoE/GR2",
    )
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    compare(
        model,
        ids.clone(),
        ids[:, sl].clone(),
        dict(position_ids=positions, attention_mask=mask, cu_seq_lens_q=bounds, linear_attn_cu_seq_lens_q=bounds),
        dict(
            position_ids=positions[:, sl], attention_mask=mask, cu_seq_lens_q=bounds, linear_attn_cu_seq_lens_q=bounds
        ),
        "packed text model with non-reentrant gradient checkpointing",
    )
    clear_parallel_state()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
