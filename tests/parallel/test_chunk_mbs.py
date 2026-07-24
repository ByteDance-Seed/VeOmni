import copy
import sys
import types
from contextlib import contextmanager
from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from veomni.arguments import ChunkMBSConfig, MixedPrecisionConfig
from veomni.distributed.chunk_mbs import (
    PackedSequenceRange,
    _VariableSplitAllToAll,
    apply_chunk_mbs,
    build_chunk_mbs_ranges,
    chunk_mbs_context,
)
from veomni.distributed.sequence_parallel.ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads


def _config(chunk_mbs=2):
    return ChunkMBSConfig(enable=True, chunk_mbs=chunk_mbs)


def test_build_chunk_mbs_ranges_from_dynamic_packed_batch():
    batch = {"cu_seq_lens_q": torch.tensor([0, 3, 5, 9, 10], dtype=torch.int32)}

    ranges = build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))

    assert ranges is not None
    assert [
        (range_.segment_start, range_.segment_end, range_.token_start, range_.token_end, range_.max_length)
        for range_ in ranges
    ] == [(0, 2, 0, 5, 3), (2, 4, 5, 10, 4)]


def test_build_chunk_mbs_ranges_from_fixed_sample_packed_batch(monkeypatch):
    import veomni.data.data_collator as data_collator

    monkeypatch.setattr(
        data_collator,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0),
    )
    collator = data_collator.MainCollator()
    batch = collator(
        [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([1, 2, 3], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([4, 5], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                "labels": torch.tensor([4, 5], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([6, 7, 8, 9], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([6, 7, 8, 9], dtype=torch.long),
            },
        ]
    )

    ranges = build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))

    assert ranges is not None
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 5, 9]
    assert [
        (range_.segment_start, range_.segment_end, range_.token_start, range_.token_end, range_.max_length)
        for range_ in ranges
    ] == [(0, 2, 0, 5, 3), (2, 3, 5, 9, 4)]


def test_build_chunk_mbs_ranges_noops_when_micro_batch_is_small():
    batch = {"cu_seq_lens_q": torch.tensor([0, 3, 5], dtype=torch.int32)}

    assert build_chunk_mbs_ranges(batch, _config(chunk_mbs=2)) is None


def test_preforward_builds_chunk_ranges_in_base_parallel_state(monkeypatch):
    import veomni.trainer.base as base_trainer

    active_state = None

    @contextmanager
    def use_parallel_state(name):
        nonlocal active_state
        previous_state = active_state
        active_state = name
        try:
            yield
        finally:
            active_state = previous_state

    def build_ranges(batch, config):
        assert active_state == "base"
        return [PackedSequenceRange(0, 2, 0, 4, 2)]

    monkeypatch.setattr(base_trainer, "use_parallel_state", use_parallel_state)
    monkeypatch.setattr(base_trainer, "build_chunk_mbs_ranges", build_ranges)
    trainer = object.__new__(base_trainer.BaseTrainer)
    trainer.args = types.SimpleNamespace(
        train=types.SimpleNamespace(chunk_mbs_config=_config(chunk_mbs=1), local_rank=0)
    )
    trainer.device = torch.device("cpu")
    trainer.LOG_SAMPLE = False

    batch = {"cu_seq_lens_q": torch.tensor([0, 2, 4], dtype=torch.int32)}
    assert trainer.preforward(batch) == batch
    assert trainer._chunk_mbs_ranges == [PackedSequenceRange(0, 2, 0, 4, 2)]
    assert active_state is None


@pytest.mark.parametrize(
    "cu_seq_lens_q",
    [
        torch.tensor([0, 3, 2, 5], dtype=torch.int32),
        torch.tensor([0, 3, 3, 5], dtype=torch.int32),
    ],
)
def test_build_chunk_mbs_ranges_rejects_non_increasing_segments(cu_seq_lens_q):
    with pytest.raises(ValueError, match="strictly increasing cu_seq_lens_q"):
        build_chunk_mbs_ranges({"cu_seq_lens_q": cu_seq_lens_q}, _config(chunk_mbs=4))


def test_build_chunk_mbs_ranges_rejects_nonzero_start():
    with pytest.raises(ValueError, match="start from 0"):
        build_chunk_mbs_ranges({"cu_seq_lens_q": torch.tensor([1, 3, 5], dtype=torch.int32)}, _config(chunk_mbs=1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bool, torch.int64])
def test_build_chunk_mbs_ranges_rejects_invalid_cu_seq_lens_dtype(dtype):
    with pytest.raises(TypeError, match="cu_seq_lens_q must use torch.int32"):
        build_chunk_mbs_ranges({"cu_seq_lens_q": torch.tensor([0, 3, 5], dtype=dtype)}, _config(chunk_mbs=1))


def test_build_chunk_mbs_ranges_rejects_asymmetric_qk_metadata():
    batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 5], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2, 5], dtype=torch.int32),
    }

    with pytest.raises(ValueError, match="identical cu_seq_lens_q and cu_seq_lens_k"):
        build_chunk_mbs_ranges(batch, _config(chunk_mbs=1))


def test_build_chunk_mbs_ranges_rejects_asymmetric_qk_max_lengths():
    batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 5], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 3, 5], dtype=torch.int32),
        "max_length_q": 3,
        "max_length_k": 4,
    }

    with pytest.raises(ValueError, match="identical max_length_q and max_length_k"):
        build_chunk_mbs_ranges(batch, _config(chunk_mbs=1))


def test_build_chunk_mbs_ranges_rejects_non_integer_linear_attn_cu_seq_lens():
    batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 5], dtype=torch.int32),
        "linear_attn_cu_seq_lens_q": torch.tensor([0.0, 3.0, 5.0]),
    }

    with pytest.raises(TypeError, match="linear_attn_cu_seq_lens_q must use torch.int32"):
        build_chunk_mbs_ranges(batch, _config(chunk_mbs=1))


@pytest.mark.parametrize(
    ("linear_attn_cu_seq_lens_q", "match"),
    [
        (torch.tensor([1, 3, 5, 10], dtype=torch.int32), "start from 0"),
        (torch.tensor([0, 5, 3, 10], dtype=torch.int32), "strictly increasing linear_attn_cu_seq_lens_q"),
        (torch.tensor([0, 3, 5, 9], dtype=torch.int32), "end at cu_seq_lens_q"),
    ],
)
def test_build_chunk_mbs_ranges_rejects_invalid_linear_attn_cu_seq_lens(linear_attn_cu_seq_lens_q, match):
    batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 5, 10], dtype=torch.int32),
        "linear_attn_cu_seq_lens_q": linear_attn_cu_seq_lens_q,
    }
    with pytest.raises(ValueError, match=match):
        build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))


def test_build_chunk_mbs_ranges_rejects_misaligned_linear_attn_boundary():
    batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 5, 9, 10], dtype=torch.int32),
        "linear_attn_cu_seq_lens_q": torch.tensor([0, 3, 5, 10], dtype=torch.int32),
    }

    with pytest.raises(ValueError, match="must align with linear_attn_cu_seq_lens_q"):
        build_chunk_mbs_ranges(batch, _config(chunk_mbs=3))


def test_build_chunk_mbs_ranges_rejects_non_packed_batch_dim():
    batch = {"hidden_states": torch.ones(4, 8, 2)}

    with pytest.raises(ValueError, match="packed-sequence FlashAttention kwargs"):
        build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))


class _ToyDecoderLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.calls = []

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        cu_seq_lens_q=None,
        cu_seq_lens_k=None,
        max_length_q=None,
        max_length_k=None,
        use_cache=None,
        past_key_value=None,
        past_key_values=None,
        layer_past=None,
        **kwargs,
    ):
        linear_attn_cu_seq_lens_q = kwargs.get("linear_attn_cu_seq_lens_q")
        self.calls.append(
            {
                "hidden_shape": tuple(hidden_states.shape),
                "position_ids": position_ids.clone(),
                "cu_seq_lens_q": cu_seq_lens_q.clone(),
                "cu_seq_lens_k": cu_seq_lens_k.clone(),
                "max_length_q": max_length_q,
                "max_length_k": max_length_k,
                "attention_shape": tuple(attention_mask.shape),
                "has_linear_attn_cu": "linear_attn_cu_seq_lens_q" in kwargs,
                "linear_attn_cu_seq_lens_q": (
                    linear_attn_cu_seq_lens_q.clone() if linear_attn_cu_seq_lens_q is not None else None
                ),
                "use_cache": use_cache,
                "past_key_value": past_key_value,
                "past_key_values": past_key_values,
                "layer_past": layer_past,
            }
        )
        cos, sin = position_embeddings
        return hidden_states + position_ids.unsqueeze(-1).to(hidden_states.dtype) + cos + sin


class _ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ToyDecoderLayer(), _ToyDecoderLayer()])


class _ToyVisionBlock(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _AuxDecoderLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states):
        return hidden_states


class _ToyMoeDecoderLayer(_ToyDecoderLayer):
    pass


class DeepseekV3MoE(nn.Module):
    pass


class DeepseekV3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.mlp = DeepseekV3MoE()

    def forward(self, hidden_states):
        return hidden_states


class GptOssExperts(nn.Module):
    pass


class GptOssDecoderLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.experts = GptOssExperts()

    def forward(self, hidden_states):
        return hidden_states


class _PlainDecoderLayer(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _ToyRoot(nn.Module):
    _no_split_modules = ["_ToyDecoderLayer", "_ToyVisionBlock"]

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel()
        self.model.visual = _ToyVisionBlock()

    def gradient_checkpointing_enable(self, checkpoint_func=None, gradient_checkpointing_kwargs=None):
        if checkpoint_func is None:
            checkpoint_func = partial(checkpoint, **(gradient_checkpointing_kwargs or {}))
        for layer in self.model.language_model.layers:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = checkpoint_func

    def gradient_checkpointing_disable(self):
        for layer in self.model.language_model.layers:
            layer.gradient_checkpointing = False


class _FSDPDecoderLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class _FSDPRoot(nn.Module):
    _no_split_modules = ["_FSDPDecoderLayer"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FSDPDecoderLayer()])

    def gradient_checkpointing_enable(self, checkpoint_func=None, gradient_checkpointing_kwargs=None):
        if checkpoint_func is None:
            checkpoint_func = partial(checkpoint, **(gradient_checkpointing_kwargs or {}))
        for layer in self.layers:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = checkpoint_func


def _local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _run_real_fsdp2_chunk_mbs(rank, world_size, init_path):
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    import veomni.distributed.chunk_mbs as chunk_mbs

    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=world_size)
    try:
        chunk_mbs.get_parallel_state = lambda: types.SimpleNamespace(
            sp_enabled=False,
            tp_enabled=False,
            pp_enabled=False,
            any_extra_parallel_enabled=False,
        )
        torch.manual_seed(0)
        baseline_root = _FSDPRoot()
        chunked_root = copy.deepcopy(baseline_root)
        baseline_layer = baseline_root.layers[0]
        chunked_layer = chunked_root.layers[0]
        checkpoint_calls = []
        hook_calls = []

        baseline_root.gradient_checkpointing_enable(partial(checkpoint, use_reentrant=False))

        def checkpoint_func(function, *args, **kwargs):
            checkpoint_calls.append(None)
            return checkpoint(function, *args, use_reentrant=False, **kwargs)

        chunked_root.gradient_checkpointing_enable(checkpoint_func)
        apply_chunk_mbs(chunked_root, _config(chunk_mbs=2))

        mesh = init_device_mesh("cpu", (world_size,))
        fully_shard(baseline_layer, mesh=mesh)
        fully_shard(chunked_layer, mesh=mesh)
        chunked_layer.register_forward_pre_hook(lambda *_: hook_calls.append("pre"))
        chunked_layer.register_forward_hook(lambda *_: hook_calls.append("post"))

        cu_seq_lens = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        ranges = build_chunk_mbs_ranges(
            {"cu_seq_lens_q": cu_seq_lens, "cu_seq_lens_k": cu_seq_lens}, _config(chunk_mbs=2)
        )
        baseline_hidden = torch.randn(1, 4, 4, requires_grad=True)
        chunked_hidden = baseline_hidden.detach().clone().requires_grad_()
        forward_kwargs = {
            "attention_mask": torch.ones(1, 1, 4, 4),
            "position_ids": torch.arange(4).view(1, 4),
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": 1,
            "max_length_k": 1,
        }

        baseline_output = baseline_layer(baseline_hidden, **forward_kwargs)
        baseline_output.float().square().mean().backward()
        with chunk_mbs_context(ranges):
            chunked_output = chunked_layer(chunked_hidden, **forward_kwargs)
            chunked_output.float().square().mean().backward()

        torch.testing.assert_close(chunked_output, baseline_output)
        torch.testing.assert_close(chunked_hidden.grad, baseline_hidden.grad)
        baseline_grads = {name: _local_tensor(param.grad) for name, param in baseline_layer.named_parameters()}
        chunked_grads = {name: _local_tensor(param.grad) for name, param in chunked_layer.named_parameters()}
        assert chunked_grads.keys() == baseline_grads.keys()
        for name, baseline_grad in baseline_grads.items():
            torch.testing.assert_close(chunked_grads[name], baseline_grad)
        assert checkpoint_calls == [None, None]
        assert hook_calls == ["pre", "post"]
    finally:
        dist.destroy_process_group()


class _Qwen3VLRoot(nn.Module):
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, layer):
        super().__init__()
        self._no_split_modules = [layer.__class__.__name__, "Qwen3VLVisionBlock"]
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([layer])


class _Qwen3_5Root(nn.Module):
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]

    def __init__(self, layer):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([layer])


class _Qwen3_5LinearAttentionStub(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.25))
        self.cu_seq_lens_calls = []

    def forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        cu_seq_lens_q=None,
    ):
        self.cu_seq_lens_calls.append(cu_seq_lens_q.detach().clone())
        projected = self.proj(hidden_states)
        outputs = []
        decay = torch.sigmoid(self.decay)
        cu_values = [int(value) for value in cu_seq_lens_q.tolist()]
        for start, end in zip(cu_values, cu_values[1:]):
            state = torch.zeros_like(projected[:, 0])
            for token_idx in range(start, end):
                state = torch.tanh(projected[:, token_idx] + decay * state)
                outputs.append(state)
        return torch.stack(outputs, dim=1)


class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.scale = nn.Parameter(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        self.calls = []

    def forward(
        self,
        hidden_states,
        position_embeddings,
        position_ids,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
        attention_mask=None,
        **kwargs,
    ):
        assert "linear_attn_cu_seq_lens_q" not in kwargs
        self.calls.append((tuple(hidden_states.shape), cu_seq_lens_q.tolist()))
        assert attention_mask is None
        assert torch.equal(cu_seq_lens_q, cu_seq_lens_k)
        assert max_length_q == max_length_k

        cos, sin = position_embeddings
        hidden_states = (
            hidden_states + cos + sin + position_ids.unsqueeze(-1).to(hidden_states.dtype) * 0.01
        ) * self.scale
        states = hidden_states.squeeze(0).view(
            hidden_states.shape[1], self.group.size(), hidden_states.shape[-1] // self.group.size()
        )
        states = gather_seq_scatter_heads(states, seq_dim=0, head_dim=1, group=self.group)
        states = torch.cat(
            [states[start:end].cumsum(dim=0) for start, end in zip(cu_seq_lens_q[:-1], cu_seq_lens_q[1:])]
        )
        states = gather_heads_scatter_seq(states, head_dim=1, seq_dim=0, group=self.group)
        return states.reshape_as(hidden_states)


class _SPQwen3VLRoot(nn.Module):
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, group):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([Qwen3VLTextDecoderLayer(group)])


class Qwen3VLMoeTextTopKRouter(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(num_experts * hidden_size, dtype=torch.float32).view(num_experts, -1) / 20
        )

    def forward(self, hidden_states):
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = nn.functional.linear(flat_states, self.weight)
        selected_experts = router_logits.argmax(dim=-1, keepdim=True)
        routing_weights = torch.ones_like(selected_experts, dtype=hidden_states.dtype)
        return router_logits, routing_weights, selected_experts


class Qwen3VLMoeTextExperts(nn.Module):
    def __init__(self, group, hidden_size):
        super().__init__()
        self.group = group
        self.weight = nn.Parameter(torch.eye(hidden_size))

    def forward(self, hidden_states, selected_experts, routing_weights):
        del routing_weights
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        selected_experts = selected_experts.reshape(-1)
        world_size = self.group.size()
        rank = self.group.rank()
        input_splits = [(selected_experts == expert).sum().item() for expert in range(world_size)]
        split_tensor = torch.tensor(input_splits, dtype=torch.int64, device=flat_states.device)
        gathered_splits = [torch.empty_like(split_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_splits, split_tensor, group=self.group)
        output_splits = [int(splits[rank].item()) for splits in gathered_splits]

        permutation = selected_experts.argsort(stable=True)
        permuted_states = flat_states[permutation]
        dispatched_states = _VariableSplitAllToAll.apply(self.group, permuted_states, output_splits, input_splits)
        expert_output = nn.functional.linear(dispatched_states, self.weight)
        returned_states = _VariableSplitAllToAll.apply(self.group, expert_output, input_splits, output_splits)
        return returned_states[permutation.argsort()].reshape_as(hidden_states)


class Qwen3VLMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, group, hidden_size):
        super().__init__()
        self.gate = Qwen3VLMoeTextTopKRouter(hidden_size, group.size())
        self.experts = Qwen3VLMoeTextExperts(group, hidden_size)

    def forward(self, hidden_states):
        router_logits, routing_weights, selected_experts = self.gate(hidden_states)
        output = self.experts(hidden_states, selected_experts, routing_weights)
        return output, router_logits


class Qwen3VLMoeTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, ep_group, use_sp, sp_group=None):
        super().__init__()
        self.group = sp_group if sp_group is not None else ep_group
        self.use_sp = use_sp
        self.scale = nn.Parameter(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        self.mlp = Qwen3VLMoeTextSparseMoeBlock(ep_group, hidden_size=4)
        self.router_logits = []

    def forward(
        self,
        hidden_states,
        position_embeddings,
        position_ids,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
        attention_mask=None,
        output_router_logits=False,
        **kwargs,
    ):
        assert attention_mask is None
        assert torch.equal(cu_seq_lens_q, cu_seq_lens_k)
        assert max_length_q == max_length_k

        cos, sin = position_embeddings
        states = (hidden_states + cos + sin + position_ids.unsqueeze(-1).to(hidden_states.dtype) * 0.01) * self.scale
        if self.use_sp:
            states = states.squeeze(0).view(hidden_states.shape[1], 2, 2)
            states = gather_seq_scatter_heads(states, seq_dim=0, head_dim=1, group=self.group)
            states = torch.cat(
                [states[start:end].cumsum(dim=0) for start, end in zip(cu_seq_lens_q[:-1], cu_seq_lens_q[1:])]
            )
            states = gather_heads_scatter_seq(states, head_dim=1, seq_dim=0, group=self.group)
            states = states.reshape_as(hidden_states)

        moe_output, router_logits = self.mlp(states)
        if output_router_logits:
            self.router_logits.append(router_logits)
        return states + moe_output


class _Qwen3VLMoeRoot(nn.Module):
    _no_split_modules = ["Qwen3VLMoeTextDecoderLayer", "Qwen3VLMoeVisionBlock"]

    def __init__(self, ep_group, use_sp, sp_group=None):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([Qwen3VLMoeTextDecoderLayer(ep_group, use_sp, sp_group)])


def _packed_cumsum_reference(hidden_states, position_embeddings, position_ids, cu_seq_lens, scale):
    cos, sin = position_embeddings
    states = (hidden_states + cos + sin + position_ids.unsqueeze(-1).to(hidden_states.dtype) * 0.01) * scale
    states = states.squeeze(0).view(hidden_states.shape[1], 2, 2)
    states = torch.cat([states[start:end].cumsum(dim=0) for start, end in zip(cu_seq_lens[:-1], cu_seq_lens[1:])])
    return states.reshape_as(hidden_states)


def _run_ulysses_chunk_mbs(rank, world_size, init_path, use_checkpoint):
    import os

    import torch.distributed as dist

    import veomni.distributed.chunk_mbs as chunk_mbs
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

    device_type = get_device_type()
    backend = "gloo" if device_type == "cpu" else get_dist_comm_backend()
    if device_type == "cpu" and sys.platform == "darwin":
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    if device_type != "cpu":
        get_torch_device().set_device(rank)
    dist.init_process_group(backend, init_method=f"file://{init_path}", rank=rank, world_size=world_size)
    try:
        group = dist.group.WORLD
        chunk_mbs.get_parallel_state = lambda: types.SimpleNamespace(
            sp_enabled=True,
            ulysses_enabled=True,
            cp_enabled=False,
            tp_enabled=False,
            pp_enabled=False,
            any_extra_parallel_enabled=False,
            ulysses_group=group,
        )

        position_ids = torch.cat([torch.arange(length) for length in (4, 3, 2, 3)]).view(1, -1)
        batch = {"position_ids": position_ids}
        from veomni.data.data_collator import add_flash_attention_kwargs_from_position_ids

        add_flash_attention_kwargs_from_position_ids(batch)
        cu_seq_lens = batch["cu_seq_lens_q"]
        ranges = build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))
        position_ids = position_ids.to(device_type)
        full_hidden = torch.arange(48, dtype=torch.float32, device=device_type).view(1, 12, 4) / 10
        full_position_embeddings = (
            torch.arange(48, dtype=torch.float32, device=device_type).view(1, 12, 4) / 100,
            torch.full((1, 12, 4), 0.25, device=device_type),
        )

        reference_hidden = full_hidden.detach().clone().requires_grad_()
        reference_scale = nn.Parameter(torch.tensor([0.5, 1.0, 1.5, 2.0], device=device_type))
        reference_output = _packed_cumsum_reference(
            reference_hidden, full_position_embeddings, position_ids, cu_seq_lens, reference_scale
        )
        reference_output.square().sum().backward()

        model = _SPQwen3VLRoot(group).to(device_type)
        layer = model.model.language_model.layers[0]
        if use_checkpoint:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        apply_chunk_mbs(model, _config(chunk_mbs=2))

        local_seq_len = full_hidden.shape[1] // world_size
        local_start = rank * local_seq_len
        local_hidden = full_hidden[:, local_start : local_start + local_seq_len].detach().clone().requires_grad_()
        local_position_embeddings = tuple(
            value[:, local_start : local_start + local_seq_len] for value in full_position_embeddings
        )
        local_position_ids = position_ids[:, local_start : local_start + local_seq_len]
        with chunk_mbs_context(ranges):
            local_output = layer(
                local_hidden,
                position_embeddings=local_position_embeddings,
                position_ids=local_position_ids,
                attention_mask=None,
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=4,
                max_length_k=4,
                linear_attn_cu_seq_lens_q=batch["linear_attn_cu_seq_lens_q"].to(device_type),
            )
            local_output.square().sum().backward()

        gathered_outputs = [torch.empty_like(local_output) for _ in range(world_size)]
        gathered_hidden_grads = [torch.empty_like(local_hidden.grad) for _ in range(world_size)]
        dist.all_gather(gathered_outputs, local_output.detach())
        dist.all_gather(gathered_hidden_grads, local_hidden.grad)
        output = torch.cat(gathered_outputs, dim=1)
        hidden_grad = torch.cat(gathered_hidden_grads, dim=1)
        dist.all_reduce(layer.scale.grad)

        torch.testing.assert_close(output, reference_output.detach())
        torch.testing.assert_close(hidden_grad, reference_hidden.grad)
        torch.testing.assert_close(layer.scale.grad, reference_scale.grad)
        first_shard_length = (7 + world_size - 1) // world_size
        second_shard_length = (5 + world_size - 1) // world_size
        assert layer.calls[:2] == [
            ((1, first_shard_length, 4), [0, 4, 7, first_shard_length * world_size]),
            ((1, second_shard_length, 4), [0, 2, 5, second_shard_length * world_size]),
        ]
        assert len(layer.calls) == (4 if use_checkpoint else 2)
    finally:
        dist.destroy_process_group()


def _run_ep_chunk_mbs(rank, world_size, init_path, use_sp, use_checkpoint, uneven_ranges=False, use_fsdp=False):
    import os

    import torch.distributed as dist

    import veomni.distributed.chunk_mbs as chunk_mbs
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

    device_type = get_device_type()
    backend = "gloo" if device_type == "cpu" else get_dist_comm_backend()
    if device_type == "cpu" and sys.platform == "darwin":
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    if device_type != "cpu":
        get_torch_device().set_device(rank)
    dist.init_process_group(backend, init_method=f"file://{init_path}", rank=rank, world_size=world_size)
    try:
        if use_fsdp:
            from torch.distributed.device_mesh import init_device_mesh

            assert world_size == 4
            ep_mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
            ep_group = ep_mesh["ep"].get_group()
            sp_group = ep_group
        else:
            ep_group = dist.group.WORLD
            sp_group = ep_group
            ep_mesh = types.SimpleNamespace(mesh_dim_names=("ep",), get_group=lambda _name: ep_group)
        if use_sp and uneven_ranges and not use_fsdp:
            assert world_size == 4
            sp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
            sp_group = sp_groups[rank // 2]
        chunk_mbs.get_parallel_state = lambda: types.SimpleNamespace(
            sp_enabled=use_sp,
            ulysses_enabled=use_sp,
            cp_enabled=False,
            tp_enabled=False,
            pp_enabled=False,
            any_extra_parallel_enabled=True,
            extra_parallel_names=("ep",),
            extra_parallel_enabled=lambda name: name == "ep",
            ep_group=ep_group,
            extra_parallel_fsdp_device_mesh={"ep": ep_mesh},
            ulysses_group=sp_group if use_sp else None,
        )

        torch.manual_seed(0)
        baseline_model = _Qwen3VLMoeRoot(ep_group, use_sp, sp_group).to(device_type)
        chunked_model = _Qwen3VLMoeRoot(ep_group, use_sp, sp_group).to(device_type)
        baseline_layer = baseline_model.model.language_model.layers[0]
        chunked_layer = chunked_model.model.language_model.layers[0]
        with torch.no_grad():
            baseline_layer.mlp.experts.weight.mul_(dist.get_rank(ep_group) + 1)
        chunked_layer.load_state_dict(baseline_layer.state_dict())

        if use_checkpoint:
            for layer in (baseline_layer, chunked_layer):
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        apply_chunk_mbs(chunked_model, _config(chunk_mbs=2))
        if use_fsdp:
            from torch.distributed._composable.fsdp import fully_shard

            fully_shard(baseline_layer.mlp.experts, mesh=ep_mesh["ep_fsdp"])
            fully_shard(chunked_layer.mlp.experts, mesh=ep_mesh["ep_fsdp"])

        if use_fsdp:
            data_group = ep_mesh.get_local_rank("ep_fsdp")
        else:
            data_group = rank // 2 if use_sp and uneven_ranges else rank
        if uneven_ranges and data_group % 2 == 1:
            cu_seq_lens = torch.tensor([0, 2, 4, 6, 8, 10, 12], dtype=torch.int32)
        else:
            cu_seq_lens = torch.tensor([0, 4, 7, 9, 12], dtype=torch.int32)
        ranges = build_chunk_mbs_ranges(
            {"cu_seq_lens_q": cu_seq_lens, "cu_seq_lens_k": cu_seq_lens}, _config(chunk_mbs=2)
        )
        assert ranges is not None and len(ranges) == 2
        position_ids = (
            torch.cat([torch.arange(end - start) for start, end in zip(cu_seq_lens[:-1], cu_seq_lens[1:])])
            .view(1, -1)
            .to(device_type)
        )
        full_hidden = torch.arange(48, dtype=torch.float32, device=device_type).view(1, 12, 4) / 10
        full_position_embeddings = (
            torch.arange(48, dtype=torch.float32, device=device_type).view(1, 12, 4) / 100,
            torch.full((1, 12, 4), 0.25, device=device_type),
        )
        if use_sp:
            local_start = dist.get_rank(sp_group) * 6
            hidden_states = full_hidden[:, local_start : local_start + 6] + data_group * 0.03
            local_position_ids = position_ids[:, local_start : local_start + 6]
            local_position_embeddings = tuple(
                value[:, local_start : local_start + 6] for value in full_position_embeddings
            )
        else:
            hidden_states = full_hidden + rank * 0.03
            local_position_ids = position_ids
            local_position_embeddings = full_position_embeddings
        forward_kwargs = {
            "position_embeddings": local_position_embeddings,
            "position_ids": local_position_ids,
            "attention_mask": None,
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": int(cu_seq_lens.diff().max().item()),
            "max_length_k": int(cu_seq_lens.diff().max().item()),
            "output_router_logits": True,
        }

        baseline_hidden = hidden_states.detach().clone().requires_grad_()
        baseline_output = baseline_layer(baseline_hidden, **forward_kwargs)
        baseline_router_logits = torch.cat(baseline_layer.router_logits)
        baseline_loss = baseline_output.float().square().mean() + baseline_router_logits.float().square().mean() * 0.03
        baseline_loss.backward()

        chunked_hidden = hidden_states.detach().clone().requires_grad_()
        with chunk_mbs_context(ranges):
            chunked_output = chunked_layer(chunked_hidden, **forward_kwargs)
            chunked_router_logits = torch.cat(chunked_layer.router_logits[:2])
            chunked_loss = (
                chunked_output.float().square().mean() + chunked_router_logits.float().square().mean() * 0.03
            )
            chunked_loss.backward()

        torch.testing.assert_close(chunked_output, baseline_output)
        torch.testing.assert_close(chunked_router_logits, baseline_router_logits)
        torch.testing.assert_close(chunked_hidden.grad, baseline_hidden.grad)
        baseline_grads = {name: param.grad for name, param in baseline_layer.named_parameters()}
        chunked_grads = {name: param.grad for name, param in chunked_layer.named_parameters()}
        assert chunked_grads.keys() == baseline_grads.keys()
        for name, baseline_grad in baseline_grads.items():
            chunked_grad = chunked_grads[name]
            if use_sp and ".experts." not in name:
                dist.all_reduce(baseline_grad, group=sp_group)
                dist.all_reduce(chunked_grad, group=sp_group)
            baseline_grad = _local_tensor(baseline_grad)
            chunked_grad = _local_tensor(chunked_grad)
            torch.testing.assert_close(
                chunked_grad, baseline_grad, msg=lambda msg, parameter_name=name: f"{parameter_name}: {msg}"
            )
    finally:
        dist.destroy_process_group()


def _run_ep_chunk_mbs_single_round_fallback(rank, world_size, init_path):
    import os

    import torch.distributed as dist

    import veomni.distributed.chunk_mbs as chunk_mbs
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

    device_type = get_device_type()
    backend = "gloo" if device_type == "cpu" else get_dist_comm_backend()
    if device_type == "cpu" and sys.platform == "darwin":
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    if device_type != "cpu":
        get_torch_device().set_device(rank)
    dist.init_process_group(backend, init_method=f"file://{init_path}", rank=rank, world_size=world_size)
    try:
        group = dist.group.WORLD
        ep_mesh = types.SimpleNamespace(mesh_dim_names=("ep",), get_group=lambda _name: group)
        chunk_mbs.get_parallel_state = lambda: types.SimpleNamespace(
            any_extra_parallel_enabled=True,
            extra_parallel_names=("ep",),
            extra_parallel_enabled=lambda name: name == "ep",
            ep_group=group,
            extra_parallel_fsdp_device_mesh={"ep": ep_mesh},
        )
        if rank == 0:
            cu_seq_lens = torch.tensor([0, 6, 12], dtype=torch.int32)
        else:
            cu_seq_lens = torch.tensor([0, 2, 4, 6, 8, 10, 12], dtype=torch.int32)

        ranges = build_chunk_mbs_ranges({"cu_seq_lens_q": cu_seq_lens}, _config(chunk_mbs=2))

        assert ranges is None
    finally:
        dist.destroy_process_group()


def _run_fsdp2_ulysses_chunk_mbs(rank, world_size, init_path, uneven_ranges=False):
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    import veomni.distributed.chunk_mbs as chunk_mbs

    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=world_size)
    try:
        if uneven_ranges:
            assert world_size == 4
            sp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
            group = sp_groups[rank // 2]
            data_group = rank // 2
        else:
            group = dist.group.WORLD
            data_group = 0
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        chunk_mbs.get_parallel_state = lambda: types.SimpleNamespace(
            sp_enabled=True,
            ulysses_enabled=True,
            cp_enabled=False,
            tp_enabled=False,
            pp_enabled=False,
            fsdp_enabled=True,
            fsdp_mesh=mesh,
            any_extra_parallel_enabled=False,
            ulysses_group=group,
        )

        baseline_model = _SPQwen3VLRoot(group)
        chunked_model = _SPQwen3VLRoot(group)
        baseline_layer = baseline_model.model.language_model.layers[0]
        chunked_layer = chunked_model.model.language_model.layers[0]
        chunked_layer.load_state_dict(baseline_layer.state_dict())
        for layer in (baseline_layer, chunked_layer):
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        apply_chunk_mbs(chunked_model, _config(chunk_mbs=2))

        fully_shard(baseline_layer, mesh=mesh)
        fully_shard(chunked_layer, mesh=mesh)

        if uneven_ranges and data_group == 1:
            cu_seq_lens = torch.tensor([0, 2, 4, 6, 8, 10, 12], dtype=torch.int32)
        else:
            cu_seq_lens = torch.tensor([0, 4, 7, 9, 12], dtype=torch.int32)
        ranges = build_chunk_mbs_ranges(
            {"cu_seq_lens_q": cu_seq_lens, "cu_seq_lens_k": cu_seq_lens}, _config(chunk_mbs=2)
        )
        assert ranges is not None and len(ranges) == 2
        position_ids = torch.cat(
            [torch.arange(end - start) for start, end in zip(cu_seq_lens[:-1], cu_seq_lens[1:])]
        ).view(1, -1)
        full_hidden = torch.arange(48, dtype=torch.float32).view(1, 12, 4) / 10
        full_position_embeddings = (
            torch.arange(48, dtype=torch.float32).view(1, 12, 4) / 100,
            torch.full((1, 12, 4), 0.25),
        )
        local_start = dist.get_rank(group) * 6
        local_position_embeddings = tuple(
            value[:, local_start : local_start + 6] for value in full_position_embeddings
        )
        local_position_ids = position_ids[:, local_start : local_start + 6]
        forward_kwargs = {
            "position_embeddings": local_position_embeddings,
            "position_ids": local_position_ids,
            "attention_mask": None,
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": int(cu_seq_lens.diff().max().item()),
            "max_length_k": int(cu_seq_lens.diff().max().item()),
        }

        local_hidden = full_hidden[:, local_start : local_start + 6] + data_group * 0.03
        baseline_hidden = local_hidden.detach().clone().requires_grad_()
        baseline_output = baseline_layer(baseline_hidden, **forward_kwargs)
        baseline_output.square().sum().backward()
        baseline_grad = _local_tensor(baseline_layer.scale.grad).detach().clone()

        chunked_hidden = local_hidden.detach().clone().requires_grad_()
        with chunk_mbs_context(ranges):
            chunked_output = chunked_layer(chunked_hidden, **forward_kwargs)
            chunked_output.square().sum().backward()
        chunked_grad = _local_tensor(chunked_layer.scale.grad).detach().clone()

        torch.testing.assert_close(chunked_output, baseline_output)
        torch.testing.assert_close(chunked_hidden.grad, baseline_hidden.grad)
        torch.testing.assert_close(chunked_grad, baseline_grad)
    finally:
        dist.destroy_process_group()


def _packed_causal_attention_mask(cu_seq_lens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    cu_values = [int(v) for v in cu_seq_lens.tolist()]
    total_seq_len = cu_values[-1]
    mask = torch.full((1, 1, total_seq_len, total_seq_len), -1e4, dtype=dtype)
    for start, end in zip(cu_values, cu_values[1:]):
        length = end - start
        mask[:, :, start:end, start:end] = torch.triu(torch.full((length, length), -1e4, dtype=dtype), diagonal=1)
    return mask


def _run_decoder_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    max_length: int,
    ranges,
    linear_attn_cu_seq_lens: torch.Tensor | None = None,
):
    layer.train()
    for param in layer.parameters():
        param.grad = None

    hidden_states = hidden_states.detach().clone().requires_grad_(True)
    forward_kwargs = {
        "position_embeddings": position_embeddings,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "max_length_q": max_length,
        "max_length_k": max_length,
        "use_cache": False,
    }
    if linear_attn_cu_seq_lens is not None:
        forward_kwargs["linear_attn_cu_seq_lens_q"] = linear_attn_cu_seq_lens

    with chunk_mbs_context(ranges):
        output = layer(hidden_states, **forward_kwargs)
        loss = output.float().square().mean()
        loss.backward()

    param_grads = {name: param.grad.detach().clone() for name, param in layer.named_parameters()}
    return output.detach(), hidden_states.grad.detach().clone(), param_grads


def test_apply_chunk_mbs_slices_packed_samples(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = _ToyRoot()
    layer = model.model.language_model.layers[0]
    apply_chunk_mbs(model, _config(chunk_mbs=2))

    assert getattr(layer, "_chunk_mbs_wrapped", False)
    assert not getattr(layer.proj, "_chunk_mbs_wrapped", False)
    assert not getattr(model.model.visual, "_chunk_mbs_wrapped", False)

    batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 5, 9, 10], dtype=torch.int32),
        "linear_attn_cu_seq_lens_q": torch.tensor([0, 3, 5, 10], dtype=torch.int32),
    }
    ranges = build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))
    hidden_states = torch.arange(20, dtype=torch.float32).view(1, 10, 2).requires_grad_()
    position_ids = torch.arange(10).view(1, 10)
    position_embeddings = (torch.ones(1, 10, 2), torch.full((1, 10, 2), 2.0))
    attention_mask = torch.ones(1, 1, 10, 10)

    with chunk_mbs_context(ranges):
        output = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seq_lens_q=batch["cu_seq_lens_q"],
            cu_seq_lens_k=batch["cu_seq_lens_q"],
            max_length_q=4,
            max_length_k=4,
            linear_attn_cu_seq_lens_q=batch["linear_attn_cu_seq_lens_q"],
        )

    expected = hidden_states + position_ids.unsqueeze(-1).to(hidden_states.dtype) + 3.0
    assert torch.equal(output, expected)
    output.sum().backward()
    assert torch.equal(hidden_states.grad, torch.ones_like(hidden_states))
    assert [call["hidden_shape"] for call in layer.calls] == [(1, 5, 2), (1, 5, 2)]
    assert [call["position_ids"].tolist() for call in layer.calls] == [[[0, 1, 2, 3, 4]], [[5, 6, 7, 8, 9]]]
    assert [call["cu_seq_lens_q"].tolist() for call in layer.calls] == [[0, 3, 5], [0, 4, 5]]
    assert [call["cu_seq_lens_k"].tolist() for call in layer.calls] == [[0, 3, 5], [0, 4, 5]]
    assert [call["max_length_q"] for call in layer.calls] == [3, 4]
    assert [call["max_length_k"] for call in layer.calls] == [3, 4]
    assert [call["attention_shape"] for call in layer.calls] == [(1, 1, 5, 5), (1, 1, 5, 5)]
    assert [call["has_linear_attn_cu"] for call in layer.calls] == [True, True]
    assert [call["linear_attn_cu_seq_lens_q"].tolist() for call in layer.calls] == [[0, 3, 5], [0, 5]]


def test_apply_chunk_mbs_preserves_gradient_checkpointing_without_ranges(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = _ToyRoot()
    layer = model.model.language_model.layers[0]
    checkpoint_calls = []

    def checkpoint_func(function, *args):
        checkpoint_calls.append(None)
        return function(*args)

    layer.gradient_checkpointing = True
    layer._gradient_checkpointing_func = checkpoint_func
    apply_chunk_mbs(model, _config(chunk_mbs=2))

    hidden_states = torch.zeros(1, 2, 2)
    position_ids = torch.arange(2).view(1, 2)
    forward_kwargs = dict(
        position_embeddings=(torch.ones_like(hidden_states), torch.ones_like(hidden_states)),
        attention_mask=torch.ones(1, 1, 2, 2),
        position_ids=position_ids,
        cu_seq_lens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 2], dtype=torch.int32),
        max_length_q=2,
        max_length_k=2,
    )
    output = layer(hidden_states, **forward_kwargs)

    assert output.shape == hidden_states.shape
    assert checkpoint_calls == [None]
    assert layer.gradient_checkpointing

    model.gradient_checkpointing_disable()
    layer(hidden_states, **forward_kwargs)
    assert checkpoint_calls == [None]

    model.gradient_checkpointing_enable(checkpoint_func)
    layer.eval()
    layer(hidden_states, **forward_kwargs)
    assert checkpoint_calls == [None]

    layer.train()
    cache = object()
    layer(
        hidden_states,
        **forward_kwargs,
        use_cache=True,
        past_key_value=cache,
        past_key_values=cache,
        layer_past=cache,
    )
    assert checkpoint_calls == [None, None]
    assert layer.calls[-1]["use_cache"] is False
    assert layer.calls[-1]["past_key_value"] is None
    assert layer.calls[-1]["past_key_values"] is None
    assert layer.calls[-1]["layer_past"] is None
    assert layer.gradient_checkpointing


def test_slice_kwargs_slices_multimodal_position_ids():
    import veomni.distributed.chunk_mbs as chunk_mbs

    position_ids = torch.arange(16).view(4, 1, 4)
    seq_range = PackedSequenceRange(segment_start=0, segment_end=1, token_start=0, token_end=2, max_length=2)

    chunk_kwargs = chunk_mbs._slice_kwargs({"position_ids": position_ids}, seq_range, full_seq_len=4)

    assert torch.equal(chunk_kwargs["position_ids"], position_ids[:, :, :2])


def test_slice_kwargs_slices_one_dimensional_cache_position():
    import veomni.distributed.chunk_mbs as chunk_mbs

    cache_position = torch.arange(4)
    seq_range = PackedSequenceRange(segment_start=1, segment_end=2, token_start=2, token_end=4, max_length=2)

    chunk_kwargs = chunk_mbs._slice_kwargs({"cache_position": cache_position}, seq_range, full_seq_len=4)

    assert torch.equal(chunk_kwargs["cache_position"], cache_position[2:4])


def test_slice_kwargs_does_not_slice_attention_head_dimension():
    import veomni.distributed.chunk_mbs as chunk_mbs

    attention_mask = torch.arange(64).view(1, 4, 4, 4)
    seq_range = PackedSequenceRange(segment_start=0, segment_end=1, token_start=1, token_end=3, max_length=2)

    chunk_kwargs = chunk_mbs._slice_kwargs({"attention_mask": attention_mask}, seq_range, full_seq_len=4)

    assert torch.equal(chunk_kwargs["attention_mask"], attention_mask[:, :, 1:3, 1:3])


def test_slice_kwargs_slices_two_dimensional_square_attention_mask():
    import veomni.distributed.chunk_mbs as chunk_mbs

    attention_mask = torch.arange(16).view(4, 4)
    seq_range = PackedSequenceRange(segment_start=0, segment_end=1, token_start=1, token_end=3, max_length=2)

    chunk_kwargs = chunk_mbs._slice_kwargs({"attention_mask": attention_mask}, seq_range, full_seq_len=4)

    assert torch.equal(chunk_kwargs["attention_mask"], attention_mask[1:3, 1:3])


@pytest.mark.parametrize("model_type", ["dense", "moe"])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_apply_chunk_mbs_matches_qwen3_vl_decoder_layer(monkeypatch, model_type, use_checkpoint):
    import veomni.distributed.chunk_mbs as chunk_mbs

    if model_type == "dense":
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
            Qwen3VLTextConfig as TextConfig,
        )
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
            Qwen3VLTextDecoderLayer as TextDecoderLayer,
        )
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
            Qwen3VLTextRotaryEmbedding as TextRotaryEmbedding,
        )
    else:
        from veomni.models.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_gpu import (
            Qwen3VLMoeTextConfig as TextConfig,
        )
        from veomni.models.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_gpu import (
            Qwen3VLMoeTextDecoderLayer as TextDecoderLayer,
        )
        from veomni.models.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_gpu import (
            Qwen3VLMoeTextRotaryEmbedding as TextRotaryEmbedding,
        )

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    torch.manual_seed(0)
    config_kwargs = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000,
    )
    if model_type == "moe":
        config_kwargs.update(
            moe_intermediate_size=24,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            mlp_only_layers=[],
        )
    config = TextConfig(**config_kwargs)
    config._attn_implementation = "eager"

    baseline_layer = TextDecoderLayer(config, layer_idx=0)
    if model_type == "moe":
        nn.init.normal_(baseline_layer.mlp.experts.gate_up_proj, mean=0.0, std=config.initializer_range)
        nn.init.normal_(baseline_layer.mlp.experts.down_proj, mean=0.0, std=config.initializer_range)
    chunked_root = _Qwen3VLRoot(copy.deepcopy(baseline_layer))
    chunked_layer = chunked_root.model.language_model.layers[0]
    checkpoint_calls = []
    baseline_hook_calls = []
    chunked_hook_calls = []

    baseline_layer.register_forward_pre_hook(lambda *_: baseline_hook_calls.append("pre"))
    baseline_layer.register_forward_hook(lambda *_: baseline_hook_calls.append("post"))
    chunked_layer.register_forward_pre_hook(lambda *_: chunked_hook_calls.append("pre"))
    chunked_layer.register_forward_hook(lambda *_: chunked_hook_calls.append("post"))

    if use_checkpoint:
        baseline_layer.gradient_checkpointing = True
        baseline_layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

        def checkpoint_func(function, *args, **kwargs):
            checkpoint_calls.append(None)
            return checkpoint(function, *args, use_reentrant=False, **kwargs)

        chunked_layer.gradient_checkpointing = True
        chunked_layer._gradient_checkpointing_func = checkpoint_func

    apply_chunk_mbs(chunked_root, _config(chunk_mbs=2))

    assert getattr(chunked_layer, "_chunk_mbs_wrapped", False)

    cu_seq_lens = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    ranges = build_chunk_mbs_ranges({"cu_seq_lens_q": cu_seq_lens}, _config(chunk_mbs=2))
    hidden_states = torch.randn(1, 9, config.hidden_size)
    text_position_ids = torch.arange(9, dtype=torch.long).view(1, 9)
    position_ids = torch.stack(
        (
            text_position_ids,
            text_position_ids,
            text_position_ids // 2,
            text_position_ids % 3,
        ),
        dim=0,
    )
    rotary_emb = TextRotaryEmbedding(config)
    position_embeddings = rotary_emb(hidden_states, position_ids[1:])
    assert position_embeddings[0].shape == (1, 9, config.head_dim)
    attention_mask = _packed_causal_attention_mask(cu_seq_lens, hidden_states.dtype)
    max_length = int((cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item())

    baseline_output, baseline_hidden_grad, baseline_param_grads = _run_decoder_layer(
        baseline_layer,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids[0],
        cu_seq_lens,
        max_length,
        ranges=None,
    )
    chunked_output, chunked_hidden_grad, chunked_param_grads = _run_decoder_layer(
        chunked_layer,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids[0],
        cu_seq_lens,
        max_length,
        ranges=ranges,
    )

    if use_checkpoint:
        assert checkpoint_calls == [None, None]
        assert chunked_layer.gradient_checkpointing
        assert baseline_hook_calls[:2] == ["pre", "post"]
        assert baseline_hook_calls.count("pre") == 2
        assert chunked_hook_calls == ["pre", "post"]
    else:
        assert chunked_hook_calls == baseline_hook_calls == ["pre", "post"]

    torch.testing.assert_close(chunked_output, baseline_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunked_hidden_grad, baseline_hidden_grad, rtol=1e-5, atol=1e-5)
    assert chunked_param_grads.keys() == baseline_param_grads.keys()
    for name, baseline_grad in baseline_param_grads.items():
        torch.testing.assert_close(chunked_param_grads[name], baseline_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_apply_chunk_mbs_matches_qwen3_5_decoder_layer(monkeypatch, layer_type, use_checkpoint):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    import veomni.distributed.chunk_mbs as chunk_mbs
    from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import (
        Qwen3_5DecoderLayer,
        Qwen3_5TextConfig,
        Qwen3_5TextRotaryEmbedding,
        eager_attention_forward,
    )

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(
            sp_enabled=False,
            tp_enabled=False,
            pp_enabled=False,
            any_extra_parallel_enabled=False,
        ),
    )

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=[layer_type],
    )
    attention_metadata_calls = {}
    attention_backend_name = "qwen3_5_chunk_mbs_test"

    def recording_attention_backend(module, query, key, value, attention_mask, **kwargs):
        cu_seq_lens_q = kwargs["cu_seq_lens_q"]
        cu_seq_lens_k = kwargs["cu_seq_lens_k"]
        attention_metadata_calls.setdefault(id(module), []).append(
            (
                tuple(cu_seq_lens_q.tolist()),
                cu_seq_lens_q.dtype,
                tuple(cu_seq_lens_k.tolist()),
                cu_seq_lens_k.dtype,
                kwargs["max_length_q"],
                kwargs["max_length_k"],
            )
        )
        return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, attention_backend_name, recording_attention_backend)
    config._attn_implementation = attention_backend_name if layer_type == "full_attention" else "eager"

    baseline_layer = Qwen3_5DecoderLayer(config, layer_idx=0)
    if layer_type == "linear_attention":
        baseline_layer.linear_attn = _Qwen3_5LinearAttentionStub(config.hidden_size)
    chunked_root = _Qwen3_5Root(copy.deepcopy(baseline_layer))
    chunked_layer = chunked_root.model.language_model.layers[0]
    baseline_attention_id = id(baseline_layer.self_attn) if layer_type == "full_attention" else None
    chunked_attention_id = id(chunked_layer.self_attn) if layer_type == "full_attention" else None
    checkpoint_calls = []

    if use_checkpoint:
        baseline_layer.gradient_checkpointing = True
        baseline_layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

        def checkpoint_func(function, *args, **kwargs):
            checkpoint_calls.append(None)
            return checkpoint(function, *args, use_reentrant=False, **kwargs)

        chunked_layer.gradient_checkpointing = True
        chunked_layer._gradient_checkpointing_func = checkpoint_func

    apply_chunk_mbs(chunked_root, _config(chunk_mbs=2))

    assert getattr(chunked_layer, "_chunk_mbs_wrapped", False)

    cu_seq_lens = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    batch = {
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "linear_attn_cu_seq_lens_q": cu_seq_lens,
    }
    ranges = build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))
    hidden_states = torch.randn(1, 9, config.hidden_size)
    position_ids = torch.arange(9, dtype=torch.long).view(1, 9)
    position_embeddings = Qwen3_5TextRotaryEmbedding(config)(hidden_states, position_ids)
    attention_mask = _packed_causal_attention_mask(cu_seq_lens, hidden_states.dtype)
    max_length = int((cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item())

    baseline_output, baseline_hidden_grad, baseline_param_grads = _run_decoder_layer(
        baseline_layer,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids,
        cu_seq_lens,
        max_length,
        ranges=None,
        linear_attn_cu_seq_lens=cu_seq_lens,
    )
    chunked_output, chunked_hidden_grad, chunked_param_grads = _run_decoder_layer(
        chunked_layer,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids,
        cu_seq_lens,
        max_length,
        ranges=ranges,
        linear_attn_cu_seq_lens=cu_seq_lens,
    )

    if use_checkpoint:
        assert checkpoint_calls == [None, None]
        assert chunked_layer.gradient_checkpointing

    torch.testing.assert_close(chunked_output, baseline_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunked_hidden_grad, baseline_hidden_grad, rtol=1e-5, atol=1e-5)
    assert chunked_param_grads.keys() == baseline_param_grads.keys()
    for name, baseline_grad in baseline_param_grads.items():
        torch.testing.assert_close(chunked_param_grads[name], baseline_grad, rtol=1e-5, atol=1e-5)

    full_metadata = ((0, 3, 5, 9), torch.int32, (0, 3, 5, 9), torch.int32, 4, 4)
    first_chunk_metadata = ((0, 3, 5), torch.int32, (0, 3, 5), torch.int32, 3, 3)
    second_chunk_metadata = ((0, 4), torch.int32, (0, 4), torch.int32, 4, 4)
    expected_baseline_calls = [full_metadata, full_metadata] if use_checkpoint else [full_metadata]
    expected_chunked_calls = [first_chunk_metadata, second_chunk_metadata]
    if use_checkpoint:
        expected_chunked_calls.extend([second_chunk_metadata, first_chunk_metadata])

    if layer_type == "linear_attention":
        baseline_calls = [tuple(call.tolist()) for call in baseline_layer.linear_attn.cu_seq_lens_calls]
        chunked_calls = [tuple(call.tolist()) for call in chunked_layer.linear_attn.cu_seq_lens_calls]
        assert baseline_calls == [call[0] for call in expected_baseline_calls]
        assert chunked_calls == [call[0] for call in expected_chunked_calls]
    else:
        assert attention_metadata_calls[baseline_attention_id] == expected_baseline_calls
        assert attention_metadata_calls[chunked_attention_id] == expected_chunked_calls


def test_replace_hidden_states_does_not_mutate_kwargs():
    import veomni.distributed.chunk_mbs as chunk_mbs

    hidden_states = torch.zeros(1, 4, 2)
    replacement = torch.ones(1, 2, 2)
    kwargs = {"hidden_states": hidden_states, "use_cache": False}

    args, chunk_kwargs = chunk_mbs._replace_hidden_states((), kwargs, replacement)

    assert args == ()
    assert chunk_kwargs["hidden_states"] is replacement
    assert kwargs["hidden_states"] is hidden_states


def test_chunk_mbs_checkpoint_tracks_keyword_hidden_states(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )
    model = _ToyRoot()
    layer = model.model.language_model.layers[0]
    model.gradient_checkpointing_enable(partial(checkpoint, use_reentrant=False))
    apply_chunk_mbs(model, _config(chunk_mbs=2))

    cu_seq_lens = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    ranges = build_chunk_mbs_ranges({"cu_seq_lens_q": cu_seq_lens, "cu_seq_lens_k": cu_seq_lens}, _config(chunk_mbs=2))
    hidden_states = torch.zeros(1, 4, 2, requires_grad=True)
    with chunk_mbs_context(ranges):
        output = layer(
            hidden_states=hidden_states,
            position_embeddings=(torch.ones_like(hidden_states), torch.ones_like(hidden_states)),
            attention_mask=torch.ones(1, 1, 4, 4),
            position_ids=torch.arange(4).view(1, 4),
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=1,
            max_length_k=1,
        )
        output.sum().backward()

    assert torch.equal(hidden_states.grad, torch.ones_like(hidden_states))


@pytest.mark.parametrize("shape", [(2, 4, 2), (4, 2)])
def test_chunked_forward_rejects_non_packed_hidden_state_shape(shape):
    import veomni.distributed.chunk_mbs as chunk_mbs

    ranges = [PackedSequenceRange(segment_start=0, segment_end=1, token_start=0, token_end=4, max_length=4)]

    with pytest.raises(ValueError, match=r"shape \[1, sequence, hidden\]"):
        chunk_mbs._chunked_forward(lambda hidden_states: hidden_states, ranges, (torch.zeros(shape),), {})


def test_apply_chunk_mbs_rejects_non_ulysses_sequence_parallel(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(
            sp_enabled=True,
            ulysses_enabled=False,
            cp_enabled=True,
            any_extra_parallel_enabled=False,
        ),
    )

    with pytest.raises(RuntimeError, match="Ulysses without context parallelism"):
        apply_chunk_mbs(_ToyRoot(), _config(chunk_mbs=2))


def test_apply_chunk_mbs_rejects_non_qwen3_vl_ulysses(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(
            sp_enabled=True,
            ulysses_enabled=True,
            cp_enabled=False,
            tp_enabled=False,
            pp_enabled=False,
            any_extra_parallel_enabled=False,
        ),
    )

    with pytest.raises(RuntimeError, match="Qwen3-VL decoder layers only"):
        apply_chunk_mbs(_ToyRoot(), _config(chunk_mbs=2))


def test_apply_chunk_mbs_rejects_asynchronous_ulysses(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(
            sp_enabled=True,
            ulysses_enabled=True,
            async_enabled=True,
            cp_enabled=False,
            tp_enabled=False,
            pp_enabled=False,
            any_extra_parallel_enabled=False,
        ),
    )

    with pytest.raises(RuntimeError, match="asynchronous Ulysses"):
        apply_chunk_mbs(_SPQwen3VLRoot(None), _config(chunk_mbs=2))


def test_apply_chunk_mbs_rejects_extra_parallel_for_non_qwen3_vl_moe(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(
            sp_enabled=False,
            tp_enabled=False,
            pp_enabled=False,
            any_extra_parallel_enabled=True,
            extra_parallel_names=("ep",),
            extra_parallel_enabled=lambda name: name == "ep",
        ),
    )

    with pytest.raises(RuntimeError, match="Qwen3-VL-MoE with expert parallelism only"):
        apply_chunk_mbs(_ToyRoot(), _config(chunk_mbs=2))


@pytest.mark.parametrize("mode", ["tp_enabled", "pp_enabled"])
def test_apply_chunk_mbs_rejects_tensor_and_pipeline_parallelism(monkeypatch, mode):
    import veomni.distributed.chunk_mbs as chunk_mbs

    parallel_state = types.SimpleNamespace(
        sp_enabled=False,
        tp_enabled=False,
        pp_enabled=False,
        any_extra_parallel_enabled=False,
    )
    setattr(parallel_state, mode, True)
    monkeypatch.setattr(chunk_mbs, "get_parallel_state", lambda: parallel_state)

    with pytest.raises(RuntimeError, match="tensor or pipeline parallelism"):
        apply_chunk_mbs(_ToyRoot(), _config(chunk_mbs=2))


def test_apply_chunk_mbs_rejects_moe_decoder_layer(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )
    model = nn.Module()
    model._no_split_modules = ["_ToyMoeDecoderLayer"]
    model.decoder = nn.ModuleList([_ToyMoeDecoderLayer()])

    with pytest.raises(RuntimeError, match="MoE decoder layers"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))


@pytest.mark.parametrize("decoder_cls", [DeepseekV3DecoderLayer, GptOssDecoderLayer])
def test_apply_chunk_mbs_rejects_moe_submodules_when_ep_is_disabled(monkeypatch, decoder_cls):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )
    model = nn.Module()
    model._no_split_modules = [decoder_cls.__name__]
    model.decoder = nn.ModuleList([decoder_cls()])

    with pytest.raises(RuntimeError, match="MoE decoder layers"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))


def test_apply_chunk_mbs_requires_decoder_layer_in_no_split_modules(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = nn.Module()
    model._no_split_modules = ["_ToyVisionBlock"]

    with pytest.raises(ValueError, match="model._no_split_modules"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))


def test_apply_chunk_mbs_rejects_ambiguous_decoder_layer_classes(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = nn.Module()
    model._no_split_modules = ["_ToyDecoderLayer", "_AuxDecoderLayer"]
    model.decoder = _ToyDecoderLayer()
    model.aux_decoder = _AuxDecoderLayer()

    with pytest.raises(ValueError, match="exactly one decoder layer class"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))
    assert not getattr(model.decoder, "_chunk_mbs_wrapped", False)
    assert not getattr(model.aux_decoder, "_chunk_mbs_wrapped", False)


def test_apply_chunk_mbs_rejects_ambiguous_decoder_stacks(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = nn.Module()
    model._no_split_modules = ["_ToyDecoderLayer"]
    model.decoder = nn.ModuleList([_ToyDecoderLayer()])
    model.aux_decoder = nn.ModuleList([_ToyDecoderLayer()])

    with pytest.raises(ValueError, match="exactly one decoder stack"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))
    assert not getattr(model.decoder[0], "_chunk_mbs_wrapped", False)
    assert not getattr(model.aux_decoder[0], "_chunk_mbs_wrapped", False)


def test_apply_chunk_mbs_rejects_incompatible_decoder_layer(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = nn.Module()
    model._no_split_modules = ["_PlainDecoderLayer"]
    model.decoder = _PlainDecoderLayer()

    with pytest.raises(TypeError, match="GradientCheckpointingLayer"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))
    assert not getattr(model.decoder, "_chunk_mbs_wrapped", False)


def test_build_parallelize_model_applies_chunk_mbs_before_fsdp2(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs
    import veomni.distributed.torch_parallelize as torch_parallelize

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )
    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: types.SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2"),
    )

    model = _ToyRoot()

    def fake_parallelize_model_fsdp2(model, **kwargs):
        layer = model.model.language_model.layers[0]
        assert getattr(layer, "_chunk_mbs_wrapped", False)
        assert layer.gradient_checkpointing
        assert getattr(layer._gradient_checkpointing_func, "_chunk_mbs_wrapped", False)
        return model

    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", fake_parallelize_model_fsdp2)

    result = torch_parallelize.build_parallelize_model(
        model,
        mixed_precision=MixedPrecisionConfig(enable=False),
        chunk_mbs_config=_config(chunk_mbs=2),
    )

    assert result is model

    cu_seq_lens = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    ranges = build_chunk_mbs_ranges({"cu_seq_lens_q": cu_seq_lens}, _config(chunk_mbs=2))
    hidden_states = torch.zeros(1, 4, 2, requires_grad=True)
    with chunk_mbs_context(ranges):
        output = result.model.language_model.layers[0](
            hidden_states,
            position_embeddings=(torch.ones_like(hidden_states), torch.ones_like(hidden_states)),
            attention_mask=torch.ones(1, 1, 4, 4),
            position_ids=torch.arange(4).view(1, 4),
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=1,
            max_length_k=1,
        )
        output.sum().backward()

    assert torch.equal(hidden_states.grad, torch.ones_like(hidden_states))


@pytest.mark.skipif(sys.platform == "darwin", reason="CPU FSDP2 process groups are not supported on macOS CI.")
def test_chunk_mbs_recompute_with_real_fsdp2(tmp_path):
    torch.multiprocessing.spawn(
        _run_real_fsdp2_chunk_mbs,
        args=(2, str(tmp_path / "pg")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_chunk_mbs_with_ulysses_matches_packed_reference(tmp_path, world_size, use_checkpoint):
    torch.multiprocessing.spawn(
        _run_ulysses_chunk_mbs,
        args=(world_size, str(tmp_path / f"ulysses-{world_size}-{use_checkpoint}"), use_checkpoint),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_chunk_mbs_with_qwen3_vl_moe_ep_matches_packed_reference(tmp_path, use_sp, use_checkpoint):
    torch.multiprocessing.spawn(
        _run_ep_chunk_mbs,
        args=(2, str(tmp_path / f"ep-sp{use_sp}-checkpoint{use_checkpoint}"), use_sp, use_checkpoint),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_chunk_mbs_ep_synchronizes_uneven_dynamic_ranges(tmp_path, use_checkpoint):
    torch.multiprocessing.spawn(
        _run_ep_chunk_mbs,
        args=(
            2,
            str(tmp_path / f"ep-uneven-checkpoint{use_checkpoint}"),
            False,
            use_checkpoint,
            True,
        ),
        nprocs=2,
        join=True,
    )


def test_chunk_mbs_ep_falls_back_to_one_round_for_uneven_dynamic_ranges(tmp_path):
    torch.multiprocessing.spawn(
        _run_ep_chunk_mbs_single_round_fallback,
        args=(2, str(tmp_path / "ep-uneven-single-round")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_chunk_mbs_sp_ep_synchronizes_uneven_dynamic_ranges(tmp_path, use_checkpoint):
    torch.multiprocessing.spawn(
        _run_ep_chunk_mbs,
        args=(
            4,
            str(tmp_path / f"sp-ep-uneven-checkpoint{use_checkpoint}"),
            True,
            use_checkpoint,
            True,
        ),
        nprocs=4,
        join=True,
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="CPU FSDP2 process groups are not supported on macOS CI.")
@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_chunk_mbs_fsdp2_ep_synchronizes_uneven_dynamic_ranges(tmp_path, use_sp, use_checkpoint):
    torch.multiprocessing.spawn(
        _run_ep_chunk_mbs,
        args=(
            4,
            str(tmp_path / f"fsdp2-sp{use_sp}-ep-uneven-checkpoint{use_checkpoint}"),
            use_sp,
            use_checkpoint,
            True,
            True,
        ),
        nprocs=4,
        join=True,
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="CPU FSDP2 process groups are not supported on macOS CI.")
def test_chunk_mbs_with_ulysses_and_real_fsdp2(tmp_path):
    torch.multiprocessing.spawn(
        _run_fsdp2_ulysses_chunk_mbs,
        args=(2, str(tmp_path / "fsdp2-ulysses")),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="CPU FSDP2 process groups are not supported on macOS CI.")
def test_chunk_mbs_fsdp2_ulysses_synchronizes_uneven_dynamic_ranges(tmp_path):
    torch.multiprocessing.spawn(
        _run_fsdp2_ulysses_chunk_mbs,
        args=(4, str(tmp_path / "fsdp2-ulysses-uneven"), True),
        nprocs=4,
        join=True,
    )


def test_dit_trainer_rejects_chunk_mbs():
    from veomni.trainer.dit_trainer import DiTTrainer

    args = types.SimpleNamespace(
        train=types.SimpleNamespace(
            chunk_mbs_config=_config(chunk_mbs=2),
            channel_loss=types.SimpleNamespace(enable=False),
        )
    )

    with pytest.raises(ValueError, match="not supported by DiTTrainer"):
        DiTTrainer(args)


@pytest.mark.parametrize(
    ("compile_config", "enable_reentrant", "match"),
    [
        (types.SimpleNamespace(enable=True), False, "not supported with torch.compile"),
        (None, True, "requires non-reentrant gradient checkpointing"),
    ],
)
def test_build_parallelize_model_rejects_unsupported_chunk_mbs_combinations(
    monkeypatch, compile_config, enable_reentrant, match
):
    import veomni.distributed.torch_parallelize as torch_parallelize

    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: object())

    with pytest.raises(ValueError, match=match):
        torch_parallelize.build_parallelize_model(
            _ToyRoot(),
            mixed_precision=MixedPrecisionConfig(enable=False),
            compile_config=compile_config,
            chunk_mbs_config=_config(chunk_mbs=2),
            enable_reentrant=enable_reentrant,
        )
