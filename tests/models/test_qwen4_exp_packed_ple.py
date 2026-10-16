# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Independent-sample output/gradient references, not packed-versus-packed parity."""

import copy
import importlib.util
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "packed_ple_under_test", ROOT / "veomni/models/transformers/qwen4_exp/packed_ple.py"
)
packed_ple = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packed_ple)


def reference(conv, inputs, bounds):
    reach = (conv.kernel_size[0] - 1) * conv.dilation[0]
    return torch.cat([conv(F.pad(inputs[..., a:b], (reach, 0))) for a, b in zip(bounds, bounds[1:])], -1)


@pytest.mark.parametrize("kernel,dilation", [(1, 1), (2, 1), (4, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_independent_sample_output_and_gradients(kernel, dilation, dtype):
    torch.manual_seed(8173)
    conv = torch.nn.Conv1d(5, 5, kernel, groups=5, dilation=dilation, bias=True).to(dtype)
    expected_conv = copy.deepcopy(conv)
    bounds = [0, 1, 4, 5, 13, 32]
    x = torch.randn(1, 5, 32, dtype=dtype, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    reach = (kernel - 1) * dilation
    padded = F.pad(x, (reach, 0))
    positions = packed_ple.local_packed_positions(torch.tensor(bounds), 1, 32)
    border = packed_ple.local_packed_boundary_indices(torch.tensor(bounds), 1, 32, reach)
    actual = packed_ple.correct_packed_ple_convolution(padded, conv(padded), conv, positions, border)
    expected = reference(expected_conv, expected_x, bounds)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad)
    for actual_p, expected_p in zip(conv.parameters(), expected_conv.parameters()):
        torch.testing.assert_close(actual_p.grad, expected_p.grad)


def test_changing_a_cannot_change_b_or_send_gradient_into_a():
    conv = torch.nn.Conv1d(3, 3, 4, dilation=3, groups=3, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1)
    a = torch.randn(1, 3, 13)
    b = torch.randn(1, 3, 19)
    positions = packed_ple.local_packed_positions(torch.tensor([0, 13, 32]), 1, 32)
    border = packed_ple.local_packed_boundary_indices(torch.tensor([0, 13, 32]), 1, 32, 9)

    def run(a):
        x = torch.cat([a, b], -1).requires_grad_()
        padded = F.pad(x, (9, 0))
        y = packed_ple.correct_packed_ple_convolution(padded, conv(padded), conv, positions, border)
        F.silu(y[..., 13:]).square().sum().backward()
        assert torch.count_nonzero(x.grad[..., :13]) == 0
        return y[..., 13:]

    torch.testing.assert_close(run(a), run(a * 3), rtol=0, atol=0)


def test_multiple_batch_rows_and_sp_offsets():
    bounds = torch.tensor([0, 3, 16, 20, 32])
    full = packed_ple.local_packed_positions(bounds, 2, 16)
    shards = [packed_ple.local_packed_positions(bounds, 2, 4, rank, 4) for rank in range(4)]
    assert torch.equal(torch.cat(shards, -1), full)
    assert full[1].tolist() == list(range(4)) + list(range(12))


def test_correction_with_real_differentiable_left_halo():
    torch.manual_seed(17)
    conv = torch.nn.Conv1d(3, 3, 4, dilation=3, groups=3, bias=False)
    ref_conv = copy.deepcopy(conv)
    bounds = [0, 13, 32, 35, 64]
    x = torch.randn(1, 3, 64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    outputs = []
    for rank in range(4):
        a, b = rank * 16, (rank + 1) * 16
        halo_input = F.pad(x, (9, 0))[..., a : b + 9]
        positions = packed_ple.local_packed_positions(torch.tensor(bounds), 1, 16, rank, 4)
        border = packed_ple.local_packed_boundary_indices(torch.tensor(bounds), 1, 16, 9, rank, 4)
        outputs.append(
            packed_ple.correct_packed_ple_convolution(halo_input, conv(halo_input), conv, positions, border)
        )
    y = torch.cat(outputs, -1)
    expected = reference(ref_conv, ref_x, bounds)
    torch.testing.assert_close(y, expected)
    upstream = torch.randn_like(y)
    (y * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(conv.weight.grad, ref_conv.weight.grad)


@pytest.mark.parametrize("bounds", [[1, 32], [0, 31], [0, 3, 3, 32], [0, 17, 32]])
def test_invalid_global_boundaries_are_rejected(bounds):
    with pytest.raises((ValueError, RuntimeError)):
        packed_ple.local_packed_positions(torch.tensor(bounds), 2, 16)


def production_method(name, class_name=None):
    import ast
    import textwrap
    from types import SimpleNamespace

    path = ROOT / "veomni/models/transformers/qwen4_exp/qwen4_exp_gpu_patch_gen_config.py"
    raw = path.read_text()
    nodes = ast.parse(raw).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    # Compile the real patchgen method without importing device-specific model kernels.
    source = textwrap.dedent("\n".join(raw.splitlines()[node.lineno - 1 : node.end_lineno]))
    namespace = {"torch": torch, "F": F, "get_parallel_state": lambda: SimpleNamespace(sp_enabled=False)}
    exec("from __future__ import annotations\n" + source, namespace)
    return namespace[name]


def test_production_ngram_history_matches_independent_samples():
    from types import MethodType, SimpleNamespace

    embedding = SimpleNamespace(
        eos_token_id=0,
        context_len=2,
        ngram_size=3,
        heads_per_ngram=2,
        layer_multipliers=torch.tensor([31, 53, 71]),
        ngram_heads_vocab_sizes=torch.tensor([101, 103, 107, 109]),
        ngram_heads_offsets=torch.tensor([0, 101, 204, 311]),
        rows_per_checkpoint_shard=1000,
        _distributed_lookup=lambda shards, rows, output_dtype=None: rows[:, None].float(),
    )
    embedding._shift_right_ignore_eos = MethodType(
        production_method("_shift_right_ignore_eos", "PatchedQwen4ExpTextNGramEmbedding"), embedding
    )
    forward = production_method("forward", "PatchedQwen4ExpTextNGramEmbedding")
    ids = torch.tensor([[2, 4, 5, 0, 7, 8, 9, 3, 1, 2, 3, 4, 5]])
    bounds = [0, 1, 5, 6, 13]
    positions = packed_ple.local_packed_positions(torch.tensor(bounds), 1, ids.shape[-1])
    actual = forward(embedding, ids, None, packed_positions=positions)
    expected = torch.cat([forward(embedding, ids[:, a:b], None) for a, b in zip(bounds, bounds[1:])], 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ple", "conv", "ngram", "text_model"])
def test_packed_cache_rejection_precedes_mutation(target):
    from types import SimpleNamespace

    class Cache:
        touched = False

        def update_conv_state(self, *args, **kwargs):
            self.touched = True
            raise AssertionError("Packed rejection must precede cache updates")

    cache = Cache()
    positions = torch.arange(4).view(1, -1)
    if target == "ple":
        method = production_method("qwen4_exp_text_ple_layer_forward_patched")
        call = partial(method, None, None, None, cache, packed_positions=positions, packed_boundary_indices=positions)
    elif target == "conv":
        method = production_method("qwen4_exp_text_ple_layer_short_conv_patched")
        call = partial(method, None, None, cache, packed_positions=positions, packed_boundary_indices=positions)
    elif target == "ngram":
        method = production_method("forward", "PatchedQwen4ExpTextNGramEmbedding")
        call = partial(method, None, positions, cache, packed_positions=positions)
    else:
        method = production_method("qwen4_exp_text_model_forward_patched")
        model = SimpleNamespace(config=SimpleNamespace(ple_layer_ids=[2]))
        call = partial(
            method,
            model,
            inputs_embeds=torch.zeros(1, 4, 8),
            past_key_values=cache,
            cu_seq_lens_q=torch.tensor([0, 4]),
        )
    with pytest.raises(NotImplementedError, match="cached"):
        call()
    assert not cache.touched
    assert not hasattr(cache, "position_ids")
