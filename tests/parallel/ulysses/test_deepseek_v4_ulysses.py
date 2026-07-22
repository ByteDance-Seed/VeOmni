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

"""Ulysses SP forward/backward equivalence for DeepSeek-V4 eager attention."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


_PATCHED_MODULE = "veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu"


def _broadcast_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def _build_causal_mask(seq_len: int, sliding_window: int | None, device, dtype) -> torch.Tensor:
    q_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
    k_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)
    causal = k_idx <= q_idx
    if sliding_window is not None:
        causal = causal & (k_idx > q_idx - sliding_window)
    full_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    return full_mask.masked_fill(~causal, torch.finfo(dtype).min)


def test_sliding_attention_sp_only_gathers_mqa_kv(monkeypatch):
    """Sliding-only layers must not all-gather unused compressor inputs."""
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    layer = dsv4.DeepseekV4Attention(config, layer_idx=0).float().eval()
    layer.compressor = None
    sp_size = 2
    sp_state = SimpleNamespace(ulysses_enabled=True, ulysses_group=object(), ulysses_size=sp_size, ulysses_rank=0)
    gather_shapes = []

    def fake_gather_outputs(x, gather_dim, **_kwargs):
        gather_shapes.append(tuple(x.shape))
        return torch.cat([x] * sp_size, dim=gather_dim)

    def fake_gather_seq_scatter_heads(x, seq_dim, head_dim, **_kwargs):
        return torch.cat([x] * sp_size, dim=seq_dim).chunk(sp_size, dim=head_dim)[0]

    def fake_gather_heads_scatter_seq(x, head_dim, seq_dim, **_kwargs):
        return torch.cat([x.chunk(sp_size, dim=seq_dim)[0]] * sp_size, dim=head_dim)

    monkeypatch.setattr(dsv4, "get_parallel_state", lambda: sp_state)
    monkeypatch.setattr(dsv4, "gather_outputs", fake_gather_outputs)
    monkeypatch.setattr(dsv4, "gather_seq_scatter_heads", fake_gather_seq_scatter_heads)
    monkeypatch.setattr(dsv4, "gather_heads_scatter_seq", fake_gather_heads_scatter_seq)

    local_seq_len = 4
    hidden_states = torch.randn(1, local_seq_len, config.hidden_size)
    position_ids = torch.arange(local_seq_len).unsqueeze(0)
    rotary = dsv4.DeepseekV4RotaryEmbedding(config)
    position_embeddings = {
        "main": rotary(hidden_states, position_ids=position_ids, layer_type="main"),
        "compress": rotary(hidden_states, position_ids=position_ids, layer_type="compress"),
    }
    output, _ = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=_build_causal_mask(local_seq_len * sp_size, config.sliding_window, "cpu", torch.float32),
    )

    assert output.shape == hidden_states.shape
    assert gather_shapes == [(1, 1, local_seq_len, config.head_dim)]


def test_model_sp_uses_lightweight_full_sequence_references(monkeypatch):
    """Mask and packed metadata helpers must not allocate full-width dummy hidden states."""
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    class TakeFirstHyperConnection(nn.Module):
        def forward(self, hidden_states):
            return hidden_states[:, :, 0]

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    model = dsv4.DeepseekV4Model(config).float().eval()
    model.layers = nn.ModuleList()
    model.hc_head = TakeFirstHyperConnection()
    sp_size = 2
    sp_state = SimpleNamespace(ulysses_enabled=True, ulysses_group=object(), ulysses_size=sp_size)
    helper_input_shapes = {}

    def fake_gather_outputs(x, gather_dim, **_kwargs):
        return torch.cat([x] * sp_size, dim=gather_dim)

    def fake_build_metadata(reference, *_args, **_kwargs):
        helper_input_shapes["metadata"] = tuple(reference.shape)
        return {}

    def fake_create_mask(*, inputs_embeds, **_kwargs):
        helper_input_shapes["mask"] = tuple(inputs_embeds.shape)
        seq_len = inputs_embeds.shape[1]
        return inputs_embeds.new_zeros((inputs_embeds.shape[0], 1, seq_len, seq_len))

    monkeypatch.setattr(dsv4, "get_parallel_state", lambda: sp_state)
    monkeypatch.setattr(dsv4, "gather_outputs", fake_gather_outputs)
    monkeypatch.setattr(dsv4, "build_packed_compression_metadata", fake_build_metadata)
    monkeypatch.setattr(dsv4, "create_sliding_window_causal_mask", fake_create_mask)

    local_seq_len = 4
    full_seq_len = local_seq_len * sp_size
    inputs_embeds = torch.randn(1, local_seq_len, config.hidden_size)
    model(
        inputs_embeds=inputs_embeds,
        position_ids=torch.arange(local_seq_len).unsqueeze(0),
        attention_mask=torch.ones(1, full_seq_len, dtype=torch.bool),
        cu_seq_lens_q=torch.tensor([0, full_seq_len], dtype=torch.int32),
        use_cache=False,
    )

    assert helper_input_shapes == {
        "metadata": (1, full_seq_len, 1),
        "mask": (1, full_seq_len, 1),
    }


def test_lightweight_references_preserve_metadata_and_mask_semantics():
    """The real helpers must ignore feature width, not merely accept width one."""
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    config._attn_implementation = "eager"
    seq_len = 8
    full_width_reference = torch.empty(1, seq_len, config.hidden_size)
    lightweight_reference = torch.empty(1, seq_len, 1)
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    sequence_slices = ((0, 4), (4, 8))
    compress_rates = tuple(config.compress_rates.values())
    hca_rate = config.compress_rates["heavily_compressed_attention"]

    full_metadata = dsv4.build_packed_compression_metadata(
        full_width_reference,
        position_ids,
        sequence_slices,
        compress_rates,
        block_bias_rates=(hca_rate,),
    )
    lightweight_metadata = dsv4.build_packed_compression_metadata(
        lightweight_reference,
        position_ids,
        sequence_slices,
        compress_rates,
        block_bias_rates=(hca_rate,),
    )
    assert full_metadata.keys() == lightweight_metadata.keys()
    for rate in full_metadata:
        assert full_metadata[rate].keys() == lightweight_metadata[rate].keys()
        for name in full_metadata[rate]:
            torch.testing.assert_close(full_metadata[rate][name], lightweight_metadata[rate][name], rtol=0, atol=0)

    attention_mask = torch.ones(1, seq_len, dtype=torch.bool)
    full_mask = dsv4.create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=full_width_reference,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    lightweight_mask = dsv4.create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=lightweight_reference,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    torch.testing.assert_close(full_mask, lightweight_mask, rtol=0, atol=0)
    assert full_width_reference.numel() == lightweight_reference.numel() * config.hidden_size


def _run_deepseek_v4_attention_sp_fw_bw(
    rank: int,
    world_size: int,
    init_file: str,
    seq_len: int,
    with_compressor: bool,
) -> None:
    """Compare SP vs baseline forward outputs and parameter grads."""
    device_type = get_device_type()
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from transformers import AutoConfig

    from veomni.distributed.parallel_state import init_parallel_state
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    init_parallel_state(dp_size=1, ulysses_size=world_size, device_type=device_type)

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    torch.manual_seed(0)
    # Layer 0 is HCA (compressor). Layer type with sliding-only is unavailable on
    # the toy config, so disable the compressor when we want pure sliding MQA.
    layer = dsv4.DeepseekV4Attention(config, layer_idx=0).to(device=device_type, dtype=torch.float32)
    if not with_compressor:
        layer.compressor = None
    _broadcast_module(layer)
    layer.train()

    bsz = 1
    hidden = config.hidden_size
    if rank == 0:
        full_hidden = torch.randn(bsz, seq_len, hidden, device=device_type, dtype=torch.float32)
        full_position_ids = torch.arange(seq_len, device=device_type).view(1, -1)
    else:
        full_hidden = torch.empty(bsz, seq_len, hidden, device=device_type, dtype=torch.float32)
        full_position_ids = torch.empty(bsz, seq_len, device=device_type, dtype=torch.long)
    dist.broadcast(full_hidden, src=0)
    dist.broadcast(full_position_ids, src=0)

    full_mask = _build_causal_mask(seq_len, config.sliding_window, device_type, torch.float32)

    rotary = dsv4.DeepseekV4RotaryEmbedding(config).to(device=device_type)
    _broadcast_module(rotary)
    position_embeddings = {
        "main": rotary(full_hidden, position_ids=full_position_ids, layer_type="main"),
        "compress": rotary(full_hidden, position_ids=full_position_ids, layer_type="compress"),
    }

    shard_len = seq_len // world_size
    local_slice = slice(rank * shard_len, (rank + 1) * shard_len)
    local_hidden = full_hidden[:, local_slice].contiguous().detach().requires_grad_(True)
    local_position_ids = full_position_ids[:, local_slice].contiguous()
    local_position_embeddings = {
        "main": rotary(local_hidden, position_ids=local_position_ids, layer_type="main"),
        "compress": rotary(local_hidden, position_ids=local_position_ids, layer_type="compress"),
    }

    baseline_out = None
    baseline_param_grads = None
    baseline_input_grad = None
    if rank == 0:
        no_sp_state = SimpleNamespace(ulysses_enabled=False)
        with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
            baseline_hidden = full_hidden.detach().clone().requires_grad_(True)
            baseline_out, _ = layer(
                baseline_hidden,
                position_embeddings=position_embeddings,
                position_ids=full_position_ids,
                attention_mask=full_mask,
            )
            # Mean over full tensor keeps SP local mean scale comparable after
            # all-gather of shards (each rank only holds 1/sp of the sequence).
            (baseline_out.mean()).backward()
            baseline_param_grads = {
                name: (param.grad.detach().clone() if param.grad is not None else None)
                for name, param in layer.named_parameters()
            }
            baseline_input_grad = baseline_hidden.grad.detach().clone()
            layer.zero_grad(set_to_none=True)
            baseline_out = baseline_out.detach()

    dist.barrier()

    sp_out_local, _ = layer(
        local_hidden,
        position_embeddings=local_position_embeddings,
        position_ids=local_position_ids,
        attention_mask=full_mask,
    )
    # Scale local loss by 1 so the full-sequence mean matches baseline.mean():
    # sum_local / (B*S*H) * sp = mean over full when grads are all-reduced.
    total_numel = float(bsz * seq_len * sp_out_local.shape[-1])
    (sp_out_local.sum() / total_numel).backward()

    for param in layer.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

    out_list = [torch.empty_like(sp_out_local) for _ in range(world_size)]
    dist.all_gather(out_list, sp_out_local.detach())
    sp_out_full = torch.cat(out_list, dim=1)

    input_grad_list = [torch.empty_like(local_hidden.grad) for _ in range(world_size)]
    dist.all_gather(input_grad_list, local_hidden.grad.detach())
    sp_input_grad_full = torch.cat(input_grad_list, dim=1)

    if rank == 0:
        torch.testing.assert_close(
            sp_out_full,
            baseline_out,
            rtol=1e-4,
            atol=1e-4,
            msg=lambda msg: f"{msg}\nForward mismatch (compressor={with_compressor})",
        )
        torch.testing.assert_close(
            sp_input_grad_full,
            baseline_input_grad,
            rtol=1e-4,
            atol=1e-4,
            msg=lambda msg: f"{msg}\nInput grad mismatch (compressor={with_compressor})",
        )
        for name, param in layer.named_parameters():
            baseline_grad = baseline_param_grads.get(name)
            if baseline_grad is None and param.grad is None:
                continue
            assert baseline_grad is not None and param.grad is not None, f"Missing grad for {name}"
            torch.testing.assert_close(
                param.grad,
                baseline_grad,
                rtol=1e-4,
                atol=1e-4,
                msg=lambda msg, n=name: f"{msg}\nParam grad mismatch for {n} (compressor={with_compressor})",
            )

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs >=2 devices")
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("with_compressor", [False, True])
def test_deepseek_v4_attention_ulysses_fw_bw_equivalence(world_size: int, seq_len: int, with_compressor: bool):
    """SP-sharded DeepSeek-V4 attention matches full-sequence eager fw/bw."""
    assert seq_len % world_size == 0
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_deepseek_v4_attention_sp_fw_bw,
            args=(world_size, init_file, seq_len, with_compressor),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="needs >=4 devices")
@pytest.mark.parametrize("world_size", [4])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("with_compressor", [True])
def test_deepseek_v4_attention_ulysses_fw_bw_4gpu(world_size: int, seq_len: int, with_compressor: bool):
    """Same fw/bw check at ulysses_size=4 (4-GPU)."""
    assert seq_len % world_size == 0
    # Toy config has 8 heads; SP=4 keeps 2 local heads per rank.
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init4")
        mp.spawn(
            _run_deepseek_v4_attention_sp_fw_bw,
            args=(world_size, init_file, seq_len, with_compressor),
            nprocs=world_size,
            join=True,
        )
