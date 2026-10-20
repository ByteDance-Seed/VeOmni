# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Independent-sample output/gradient references, not packed-versus-packed parity."""

import copy
import importlib.util
from functools import partial
from itertools import accumulate
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


@pytest.mark.parametrize(
    "lengths,sp_size",
    [
        ([1] * 32, 4),
        ([3, 5, 2, 7, 1, 4, 6, 4], 4),
        ([13, 19], 4),
    ],
)
def test_host_boundary_indices_match_local_tensor_metadata(lengths, sp_size):
    local_length = 8
    assert sum(lengths) == local_length * sp_size
    bounds = torch.tensor([0, *accumulate(lengths)])
    for rank in range(sp_size):
        host = packed_ple.local_packed_boundary_indices(
            bounds,
            1,
            local_length,
            9,
            rank,
            sp_size,
            packed_seq_lens=lengths,
        )
        tensor = packed_ple.local_packed_boundary_indices(
            bounds,
            1,
            local_length,
            9,
            rank,
            sp_size,
        )
        valid_tensor = tensor[tensor < local_length]
        assert torch.equal(host, valid_tensor)


def test_fragmented_pack_boundary_count_is_local():
    local_length = 8
    sp_size = 4
    lengths = [1] * (local_length * sp_size)
    bounds = torch.tensor([0, *accumulate(lengths)])
    for rank in range(sp_size):
        border = packed_ple.local_packed_boundary_indices(
            bounds,
            1,
            local_length,
            9,
            rank,
            sp_size,
            packed_seq_lens=lengths,
        )
        assert border.numel() == local_length
        assert border.tolist() == list(range(local_length))
        assert border.numel() < bounds.numel() * 8


def test_host_boundary_indices_match_multirow_local_tensor_metadata():
    batch_size = 2
    local_length = 8
    sp_size = 2
    lengths = [3, 13, 4, 12]
    bounds = torch.tensor([0, *accumulate(lengths)])
    for rank in range(sp_size):
        host = packed_ple.local_packed_boundary_indices(
            bounds,
            batch_size,
            local_length,
            9,
            rank,
            sp_size,
            packed_seq_lens=lengths,
        )
        tensor = packed_ple.local_packed_boundary_indices(
            bounds,
            batch_size,
            local_length,
            9,
            rank,
            sp_size,
        )
        valid_tensor = tensor[tensor < batch_size * local_length]
        assert torch.equal(host, valid_tensor)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chunked_boundary_correction_matches_reference(dtype, monkeypatch):
    torch.manual_seed(991)
    kernel, dilation, channels = 4, 3, 5
    conv = torch.nn.Conv1d(channels, channels, kernel, groups=channels, dilation=dilation, bias=True).to(dtype)
    expected_conv = copy.deepcopy(conv)
    bounds = [0, 1, 4, 5, 13, 32]
    x = torch.randn(1, channels, 32, dtype=dtype, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    reach = (kernel - 1) * dilation
    padded = F.pad(x, (reach, 0))
    positions = packed_ple.local_packed_positions(torch.tensor(bounds), 1, 32)
    border = packed_ple.local_packed_boundary_indices(torch.tensor(bounds), 1, 32, reach)

    monkeypatch.setattr(packed_ple, "_BOUNDARY_CORRECTION_BUDGET_BYTES", 1)
    actual = packed_ple.correct_packed_ple_convolution(padded, conv(padded), conv, positions, border)
    expected = reference(expected_conv, expected_x, bounds)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad)
    for actual_p, expected_p in zip(conv.parameters(), expected_conv.parameters()):
        torch.testing.assert_close(actual_p.grad, expected_p.grad)


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


def production_method(name, class_name=None, extra_namespace=None):
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

    def sequence_parallel_state(parallel_state):
        if getattr(parallel_state, "cp_enabled", False):
            return True, parallel_state.cp_group, parallel_state.cp_rank, parallel_state.cp_size
        if getattr(parallel_state, "ulysses_enabled", False):
            return True, parallel_state.ulysses_group, parallel_state.ulysses_rank, parallel_state.ulysses_size
        return False, None, 0, 1

    def validate_packed(lengths, batch_size, sequence_length):
        lengths = tuple(int(length) for length in lengths)
        if batch_size != 1 or not lengths or any(length <= 0 for length in lengths):
            raise ValueError("invalid packed lengths")
        if sum(lengths) != sequence_length:
            raise ValueError("packed lengths do not cover the sequence")
        return lengths

    class Qwen4ExpModelOutputWithPast:
        def __init__(self, last_hidden_state, past_key_values):
            self.last_hidden_state = last_hidden_state
            self.past_key_values = past_key_values

    namespace = {
        "torch": torch,
        "F": F,
        "SimpleNamespace": SimpleNamespace,
        "get_parallel_state": lambda: SimpleNamespace(sp_enabled=False),
        "_qwen4_exp_sequence_parallel_state": sequence_parallel_state,
        "_qwen4_exp_validate_packed_seq_lens": validate_packed,
        "Qwen4ExpModelOutputWithPast": Qwen4ExpModelOutputWithPast,
        "local_packed_positions": packed_ple.local_packed_positions,
        "local_packed_boundary_indices": packed_ple.local_packed_boundary_indices,
        "correct_packed_ple_convolution": packed_ple.correct_packed_ple_convolution,
    }
    namespace.update(extra_namespace or {})
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


def test_production_packed_ngram_uses_previous_rank_halo_under_sp():
    from types import MethodType, SimpleNamespace

    ids = torch.tensor([[2, 4, 5, 0, 7, 8, 9, 3, 1, 2, 3, 4]])
    bounds = torch.tensor([0, 5, ids.shape[-1]])
    local_length = ids.shape[-1] // 2
    context_len = 2
    local_tails = [
        ids[:, start + local_length - context_len : start + local_length]
        for start in range(0, ids.shape[-1], local_length)
    ]

    def all_gather(outputs, _tensor, group=None):
        assert len(outputs) == len(local_tails)
        for output, tail in zip(outputs, local_tails):
            output.copy_(tail)

    def make_embedding():
        embedding = SimpleNamespace(
            eos_token_id=0,
            context_len=context_len,
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
        return embedding

    embedding = make_embedding()
    forward = production_method("forward", "PatchedQwen4ExpTextNGramEmbedding")
    expected = torch.cat(
        [forward(embedding, ids[:, a:b], None).view(1, b - a, -1) for a, b in zip(bounds, bounds[1:])],
        1,
    )

    outputs = []
    for rank in range(2):
        parallel_state = SimpleNamespace(
            cp_enabled=True,
            cp_group=object(),
            cp_rank=rank,
            cp_size=2,
            ulysses_enabled=False,
        )
        sp_forward = production_method(
            "forward",
            "PatchedQwen4ExpTextNGramEmbedding",
            extra_namespace={
                "dist": SimpleNamespace(all_gather=all_gather),
                "get_parallel_state": lambda state=parallel_state: state,
            },
        )
        local_ids = ids[:, rank * local_length : (rank + 1) * local_length]
        positions = packed_ple.local_packed_positions(bounds, 1, local_length, rank, 2)
        outputs.append(sp_forward(embedding, local_ids, None, packed_positions=positions).view(1, local_length, -1))

    torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=0, atol=0)


def test_production_text_model_builds_packed_ple_metadata_once():
    from types import SimpleNamespace

    captured = []

    class Layer:
        def __call__(self, hidden_states, **kwargs):
            captured.append(kwargs)
            return hidden_states

    parallel_state = SimpleNamespace(
        sp_enabled=True,
        cp_enabled=True,
        cp_group=object(),
        cp_rank=1,
        cp_size=2,
        ulysses_enabled=False,
    )
    forward = production_method(
        "qwen4_exp_text_model_forward_patched",
        extra_namespace={"get_parallel_state": lambda: parallel_state},
    )
    inputs_embeds = torch.zeros(1, 4, 8, requires_grad=True)
    model = SimpleNamespace(
        config=SimpleNamespace(
            ple_layer_ids=[2],
            ple_conv_kernel_size=4,
            ngram_size=2,
            hc_count=1,
            num_hidden_layers=1,
        ),
        layers=[Layer()],
        rotary_emb=lambda hidden_states, position_ids: (torch.zeros(1), torch.zeros(1)),
        hyper_connection_mixer=lambda hidden_states: hidden_states,
    )
    output = forward(
        model,
        inputs_embeds=inputs_embeds,
        position_ids=torch.zeros(4, 1, 4, dtype=torch.long),
        attention_mask={"linear_attention": None},
        use_cache=False,
        ple_input_ids=torch.zeros(1, 4, dtype=torch.long),
        cu_seq_lens_q=torch.tensor([0, 3, 8]),
        packed_seq_lens=(3, 5),
    )
    assert torch.equal(output.last_hidden_state, inputs_embeds)
    assert len(captured) == 1
    positions = captured[0]["ple_packed_positions"]
    border = captured[0]["ple_packed_boundary_indices"]
    assert positions.tolist() == [[1, 2, 3, 4]]
    assert border.tolist() == [0, 1, 2, 3]


def test_production_cached_single_segment_skips_packed_ple_metadata():
    from types import SimpleNamespace

    captured = []

    class Layer:
        def __call__(self, hidden_states, **kwargs):
            captured.append(kwargs)
            return hidden_states

    forward = production_method("qwen4_exp_text_model_forward_patched")
    inputs_embeds = torch.zeros(1, 4, 8)
    cache = SimpleNamespace()
    model = SimpleNamespace(
        config=SimpleNamespace(
            ple_layer_ids=[2],
            ple_conv_kernel_size=4,
            ngram_size=2,
            hc_count=1,
            num_hidden_layers=1,
        ),
        layers=[Layer()],
        rotary_emb=lambda hidden_states, position_ids: (torch.zeros(1), torch.zeros(1)),
        hyper_connection_mixer=lambda hidden_states: hidden_states,
    )
    cu_seq_lens_q = torch.tensor([0, 4], dtype=torch.int32)
    output = forward(
        model,
        inputs_embeds=inputs_embeds,
        position_ids=torch.zeros(4, 1, 4, dtype=torch.long),
        attention_mask={"linear_attention": None},
        past_key_values=cache,
        use_cache=True,
        ple_input_ids=torch.zeros(1, 4, dtype=torch.long),
        cu_seq_lens_q=cu_seq_lens_q,
    )
    assert torch.equal(output.last_hidden_state, inputs_embeds)
    assert captured[0]["packed_seq_lens"] == (4,)
    assert captured[0]["ple_packed_positions"] is None
    assert captured[0]["ple_packed_boundary_indices"] is None
    assert torch.equal(captured[0]["cu_seq_lens_q"], cu_seq_lens_q)


def test_production_cached_single_segment_decoder_keeps_attention_culen():
    from types import SimpleNamespace

    captured = {}

    class Ple:
        def __call__(self, hidden_states, *args, **kwargs):
            captured["ple_cu_seq_lens_q"] = kwargs.get("cu_seq_lens_q")
            return torch.zeros_like(hidden_states)

    class HyperConnection:
        def __call__(self, hidden_states):
            weights = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 1, dtype=hidden_states.dtype)
            return hidden_states, hidden_states, weights

    class Attention:
        def __call__(self, hidden_states, position_embeddings, **kwargs):
            captured["attention_cu_seq_lens_q"] = kwargs.get("cu_seq_lens_q")
            return hidden_states, None

    method = production_method("qwen4_exp_text_decoder_layer_forward_patched")
    cu_seq_lens_q = torch.tensor([0, 4], dtype=torch.int32)
    hidden_states = torch.zeros(1, 4, 8)
    result = method(
        SimpleNamespace(
            ple=Ple(),
            layer_type="full_attention",
            attn_hyper_connection=HyperConnection(),
            self_attn=Attention(),
            mlp_hyper_connection=HyperConnection(),
            mlp=lambda hidden_states: hidden_states,
        ),
        hidden_states,
        position_embeddings=(torch.zeros(1), torch.zeros(1)),
        past_key_values=object(),
        ple_input_ids=torch.zeros(1, 4, dtype=torch.long),
        cu_seq_lens_q=cu_seq_lens_q,
        packed_seq_lens=(4,),
    )
    assert torch.equal(result, hidden_states)
    assert captured["ple_cu_seq_lens_q"] is None
    assert torch.equal(captured["attention_cu_seq_lens_q"], cu_seq_lens_q)


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
            cu_seq_lens_q=torch.tensor([0, 2, 4]),
        )
    with pytest.raises(NotImplementedError, match="cached"):
        call()
    assert not cache.touched
    assert not hasattr(cache, "position_ids")
