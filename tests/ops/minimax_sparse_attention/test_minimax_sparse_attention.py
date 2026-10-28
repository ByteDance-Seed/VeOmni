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

"""TND eager MiniMax sparse attention against the upstream per-sample BSND reference."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops.kernels.minimax_sparse_attention import (
    minimax_sparse_attention_eager,
    minimax_sparse_indexer_eager,
)
from veomni.ops.kernels.minimax_sparse_attention.eager import (
    _expand_selected_blocks,
    _pack,
    _packed_to_padded_indices,
    _unpack,
)
from veomni.utils.import_utils import is_torch_npu_available


# The last segment stands in for the collator's synthetic SP tail padding: it is
# an ordinary packed sequence and must not interact with the real samples.
_SEQUENCE_LENGTHS = (5, 1, 9, 3)
_NUM_QUERY_HEADS = 4
_NUM_GROUPS = 2
_HEAD_DIM = 8
_BLOCK_SIZE = 2
_TOPK_BLOCKS = 3
_LOCAL_BLOCKS = 1


def _upstream():
    return importlib.import_module("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")


def _upstream_indexer():
    upstream = _upstream()
    config = upstream.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=_NUM_QUERY_HEADS,
        num_key_value_heads=_NUM_GROUPS,
        index_n_heads=_NUM_GROUPS,
        index_head_dim=_HEAD_DIM,
        index_block_size=_BLOCK_SIZE,
        index_topk_blocks=_TOPK_BLOCKS,
        index_local_blocks=_LOCAL_BLOCKS,
        head_dim=_HEAD_DIM,
        rotary_dim=_HEAD_DIM,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["minimax_m3_sparse"],
        mlp_layer_types=["dense"],
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    indexer = upstream.MiniMaxM3VLIndexer(config, layer_idx=0).eval()
    for norm in (indexer.q_norm, indexer.k_norm):
        torch.nn.init.normal_(norm.weight, std=0.5)
    return config, indexer, upstream.MiniMaxM3VLRotaryEmbedding(config)


def _cu_seqlens(lengths):
    return F.pad(torch.tensor(lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))


def _project_index_qk(indexer, hidden_states, position_embeddings):
    """Upstream indexer projection + norm + RoPE, returned as ``[S, G, D]`` / ``[S, 1, D]``."""
    upstream = _upstream()
    seq_len = hidden_states.shape[1]
    index_q = indexer.q_norm(indexer.q_proj(hidden_states).view(1, seq_len, -1, indexer.head_dim)).transpose(1, 2)
    index_k = indexer.k_norm(indexer.k_proj(hidden_states).view(1, seq_len, 1, indexer.head_dim)).transpose(1, 2)
    cos, sin = position_embeddings
    index_q, index_k = upstream.apply_rotary_pos_emb(index_q, index_k, cos, sin)
    return index_q[0].transpose(0, 1), index_k[0].transpose(0, 1)


def _canonical_block_indices(block_indices):
    """Pad to ``topk_blocks`` slots and sort, since slot order and width are not part of the contract."""
    block_indices = F.pad(block_indices, (0, _TOPK_BLOCKS - block_indices.shape[-1]), value=-1)
    return block_indices.sort(dim=-1, descending=True).values


def _per_sample_inputs():
    config, indexer, rotary = _upstream_indexer()
    samples = []
    for seq_len in _SEQUENCE_LENGTHS:
        hidden_states = torch.randn(1, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        position_embeddings = rotary(hidden_states, position_ids)
        with torch.no_grad():
            reference_indices = indexer(hidden_states, position_embeddings, None, position_ids)
            index_q, index_k = _project_index_qk(indexer, hidden_states, position_embeddings)
        samples.append(
            SimpleNamespace(
                seq_len=seq_len,
                position_ids=position_ids,
                reference_indices=reference_indices,
                index_q=index_q,
                index_k=index_k,
                q=torch.randn(seq_len, _NUM_QUERY_HEADS, _HEAD_DIM),
                k=torch.randn(seq_len, _NUM_GROUPS, _HEAD_DIM),
                v=torch.randn(seq_len, _NUM_GROUPS, _HEAD_DIM),
            )
        )
    return indexer, samples


def _upstream_attention(indexer, sample, q, k, v, block_indices):
    """Upstream ``build_block_mask`` + ``eager_attention_forward`` on one unpadded sample."""
    upstream = _upstream()
    query, key, value = (tensor.transpose(0, 1).unsqueeze(0) for tensor in (q, k, v))
    attention_mask = indexer.build_block_mask(
        block_indices, None, sample.seq_len, query.dtype, query.device, sample.position_ids
    )
    module = SimpleNamespace(num_key_value_groups=_NUM_QUERY_HEADS // _NUM_GROUPS, training=False)
    output, _ = upstream.eager_attention_forward(module, query, key, value, attention_mask, scaling=_HEAD_DIM**-0.5)
    return output[0]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_indexer_matches_upstream_per_sample():
    _, samples = _per_sample_inputs()
    block_indices = minimax_sparse_indexer_eager(
        torch.cat([sample.index_q for sample in samples]),
        torch.cat([sample.index_k for sample in samples]),
        _cu_seqlens(_SEQUENCE_LENGTHS),
        max(_SEQUENCE_LENGTHS),
        block_size=_BLOCK_SIZE,
        topk_blocks=_TOPK_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
    )

    assert block_indices.shape == (sum(_SEQUENCE_LENGTHS), _NUM_GROUPS, _TOPK_BLOCKS)
    assert block_indices.dtype == torch.int32
    expected = torch.cat([_canonical_block_indices(sample.reference_indices[0].transpose(0, 1)) for sample in samples])
    assert torch.equal(_canonical_block_indices(block_indices).long(), expected)


# float64 guards padded query rows: a fully masked row turns into NaN once the
# float32 softmax sees the float64 dtype-min mask, and leaks into K/V gradients.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_attention_matches_upstream_per_sample_forward_and_backward(dtype):
    indexer, samples = _per_sample_inputs()
    for sample in samples:
        sample.q, sample.k, sample.v = (getattr(sample, name).to(dtype) for name in ("q", "k", "v"))
    packed_q, packed_k, packed_v = (
        torch.cat([getattr(sample, name) for sample in samples]).requires_grad_(True) for name in ("q", "k", "v")
    )
    block_indices = torch.cat(
        [_canonical_block_indices(sample.reference_indices[0].transpose(0, 1)) for sample in samples]
    ).to(torch.int32)
    output = minimax_sparse_attention_eager(
        packed_q,
        packed_k,
        packed_v,
        block_indices,
        _cu_seqlens(_SEQUENCE_LENGTHS),
        max(_SEQUENCE_LENGTHS),
        block_size=_BLOCK_SIZE,
        scale=_HEAD_DIM**-0.5,
    )
    grad_output = torch.randn_like(output)
    packed_grads = torch.autograd.grad(output, (packed_q, packed_k, packed_v), grad_output)

    expected_outputs = []
    expected_grads = [[], [], []]
    offset = 0
    for sample in samples:
        q, k, v = (getattr(sample, name).clone().requires_grad_(True) for name in ("q", "k", "v"))
        expected = _upstream_attention(indexer, sample, q, k, v, sample.reference_indices)
        grads = torch.autograd.grad(expected, (q, k, v), grad_output[offset : offset + sample.seq_len])
        expected_outputs.append(expected)
        for accumulated, grad in zip(expected_grads, grads):
            accumulated.append(grad)
        offset += sample.seq_len

    torch.testing.assert_close(output, torch.cat(expected_outputs))
    for packed_grad, grads in zip(packed_grads, expected_grads):
        assert torch.isfinite(packed_grad).all()
        torch.testing.assert_close(packed_grad, torch.cat(grads))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_indexer_output_feeds_attention_with_wider_topk_slots():
    """The indexer pads to ``topk_blocks`` slots even when the batch has fewer key blocks."""
    _, samples = _per_sample_inputs()
    short = [sample for sample in samples if sample.seq_len <= 3]
    lengths = [sample.seq_len for sample in short]
    block_indices = minimax_sparse_indexer_eager(
        torch.cat([sample.index_q for sample in short]),
        torch.cat([sample.index_k for sample in short]),
        _cu_seqlens(lengths),
        max(lengths),
        block_size=_BLOCK_SIZE,
        topk_blocks=_TOPK_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
    )
    assert block_indices.shape[-1] == _TOPK_BLOCKS
    assert (block_indices[..., -1] == -1).all()

    output = minimax_sparse_attention_eager(
        torch.cat([sample.q for sample in short]),
        torch.cat([sample.k for sample in short]),
        torch.cat([sample.v for sample in short]),
        block_indices,
        _cu_seqlens(lengths),
        max(lengths),
        block_size=_BLOCK_SIZE,
        scale=_HEAD_DIM**-0.5,
    )
    assert torch.isfinite(output).all()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_padded_layout_round_trips_sp_padding_segment():
    packed = torch.arange(14, dtype=torch.float32).view(7, 2)

    indices, padding = _packed_to_padded_indices(_cu_seqlens([2, 3, 2]), 3, 7, packed.device)
    padded = _unpack(packed, indices, padding, fill=-1)

    assert indices.tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert padding.tolist() == [[False, False, True], [False, False, False], [False, False, True]]
    assert padded.shape == (3, 3, 2)
    assert (padded[padding] == -1).all()
    assert torch.equal(_pack(padded, indices), packed)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_rejects_device_tensor_max_seqlen():
    q = torch.randn(4, _NUM_QUERY_HEADS, _HEAD_DIM)
    k = torch.randn(4, _NUM_GROUPS, _HEAD_DIM)
    block_indices = torch.zeros(4, _NUM_GROUPS, _TOPK_BLOCKS, dtype=torch.int32)
    with pytest.raises(TypeError, match="Python int"):
        minimax_sparse_attention_eager(
            q, k, k, block_indices, _cu_seqlens([4]), torch.tensor(4), block_size=_BLOCK_SIZE, scale=_HEAD_DIM**-0.5
        )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_rejects_block_indices_not_grouped_by_kv_head():
    q = torch.randn(4, _NUM_QUERY_HEADS, _HEAD_DIM)
    k = torch.randn(4, _NUM_GROUPS, _HEAD_DIM)
    block_indices = torch.zeros(4, _NUM_QUERY_HEADS, _TOPK_BLOCKS, dtype=torch.int32)
    with pytest.raises(ValueError, match="one selection per KV head"):
        minimax_sparse_attention_eager(
            q, k, k, block_indices, _cu_seqlens([4]), 4, block_size=_BLOCK_SIZE, scale=_HEAD_DIM**-0.5
        )


@pytest.mark.parametrize(("seq_len", "block_size"), [(9, 2), (8, 4), (5, 8)], ids=["ragged", "exact", "one-block"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_bfloat16_block_expansion_matches_boolean_expansion(seq_len, block_size):
    """The NPU-only bfloat16 expansion path must build exactly the boolean mask."""
    torch.manual_seed(0)
    num_key_blocks = -(-seq_len // block_size)
    selected = torch.rand(2, _NUM_GROUPS, seq_len, num_key_blocks) > 0.5
    kwargs = dict(block_size=block_size, seq_len=seq_len, heads_per_group=_NUM_QUERY_HEADS // _NUM_GROUPS)

    expected = _expand_selected_blocks(selected, via_bfloat16=False, **kwargs)
    actual = _expand_selected_blocks(selected, via_bfloat16=True, **kwargs)

    assert actual.dtype == torch.bool
    assert actual.shape == (2, _NUM_QUERY_HEADS, seq_len, seq_len)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_npu_fused_attention_matches_cpu_reference_forward_and_backward(dtype):
    torch.manual_seed(42)
    lengths = _SEQUENCE_LENGTHS
    total_length, head_dim = sum(lengths), 64
    cu_seqlens = _cu_seqlens(lengths)
    cpu_inputs = [
        torch.randn(total_length, heads, head_dim).to(dtype).float().requires_grad_()
        for heads in (_NUM_QUERY_HEADS, _NUM_GROUPS, _NUM_GROUPS)
    ]
    npu_inputs = [tensor.detach().to(device="npu", dtype=dtype).requires_grad_() for tensor in cpu_inputs]
    block_indices = minimax_sparse_indexer_eager(
        cpu_inputs[1].detach(),
        cpu_inputs[1][:, :1].detach(),
        cu_seqlens,
        max(lengths),
        block_size=_BLOCK_SIZE,
        topk_blocks=_TOPK_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
    )
    kwargs = dict(block_size=_BLOCK_SIZE, scale=head_dim**-0.5)
    expected = minimax_sparse_attention_eager(*cpu_inputs, block_indices, cu_seqlens, max(lengths), **kwargs)
    actual = minimax_sparse_attention_eager(
        *npu_inputs,
        block_indices.to("npu"),
        cu_seqlens,
        max(lengths),
        **kwargs,
    )
    grad_output = torch.randn_like(expected).to(dtype).float()
    expected_grads = torch.autograd.grad(expected, cpu_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, npu_inputs, grad_output.to(device="npu", dtype=dtype))
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    for result, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(result.float().cpu(), reference, atol=tolerance, rtol=tolerance)
