from types import SimpleNamespace

import torch
import torch.nn as nn

from veomni.models.seed_omni.foundation.qwen3_moe_foundation.modeling_qwen3_moe_foundation import (
    MultimodalQwen3MoeModel,
    Qwen3MoeDecoderLayer,
)
from veomni.models.transformers.qwen3_5 import qwen3_5_gpu_patch_gen_config as qwen35_config
from veomni.models.transformers.qwen3_moe import qwen3_moe_gpu_patch_gen_config as qwen3_moe_config


def test_qwen35_model_forward_preserves_explicit_mm_token_type_ids(monkeypatch):
    class _ParallelState:
        sp_enabled = False
        fsdp_enabled = False

    captured = {}
    monkeypatch.setattr(qwen35_config, "get_parallel_state", lambda: _ParallelState())
    monkeypatch.setattr(qwen35_config, "Qwen3_5ModelOutputWithPast", lambda **kwargs: kwargs)

    class _FakeModel:
        config = SimpleNamespace(image_token_id=10, video_token_id=20)
        rope_deltas = None
        language_model = staticmethod(lambda **kwargs: {})

        def get_input_embeddings(self):
            return lambda input_ids: torch.zeros((*input_ids.shape, 4))

        def compute_3d_position_ids(self, **kwargs):
            captured["mm_token_type_ids"] = kwargs["mm_token_type_ids"]
            return torch.zeros((3, 1, 4), dtype=torch.long)

    input_ids = torch.tensor([[10, 1, 2, 3]])
    supplied = torch.full((1, 4), 7, dtype=torch.int32)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    qwen35_config.qwen3_5_model_forward(
        _FakeModel(),
        input_ids=input_ids,
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        mm_token_type_ids=supplied,
        image_mask=mask,
        video_mask=mask,
    )

    assert captured["mm_token_type_ids"] is supplied


class _CaptureAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["hidden_states"], None


def _foundation_decoder_layer():
    layer = object.__new__(Qwen3MoeDecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = nn.Identity()
    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = nn.Identity()
    layer.self_attn = _CaptureAttention()
    return layer


def test_foundation_decoder_prefers_supplied_fa_metadata_and_keeps_legacy_fallback():
    layer = _foundation_decoder_layer()
    supplied_cu = torch.tensor([0, 4], dtype=torch.int32)
    layer(
        torch.zeros((1, 4, 2)),
        position_ids=torch.tensor([[[2, 0, 0, 0]]]),
        cu_seq_lens_q=supplied_cu,
        cu_seq_lens_k=supplied_cu,
        max_length_q=4,
        max_length_k=4,
    )
    actual = layer.self_attn.kwargs
    assert actual["cu_seq_lens_q"] is supplied_cu
    assert actual["cu_seq_lens_k"] is supplied_cu
    assert actual["max_length_q"] == 4
    assert actual["max_length_k"] == 4

    layer(torch.zeros((1, 4, 2)), position_ids=torch.tensor([[[0, 1, 0, 1]]]))
    fallback = layer.self_attn.kwargs
    torch.testing.assert_close(
        fallback["cu_seq_lens_q"],
        torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    assert int(fallback["max_length_q"]) == 2


class _CaptureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, hidden_states, *args, **kwargs):
        self.kwargs = kwargs
        return (hidden_states,)


class _FakeRotaryEmbedding(nn.Module):
    def forward(self, hidden_states, position_ids):
        return hidden_states, hidden_states


def test_foundation_gradient_checkpointing_preserves_fa_metadata():
    model = object.__new__(MultimodalQwen3MoeModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_router_logits=False,
        output_hidden_states=False,
        use_cache=False,
        use_return_dict=True,
        rope_type="1d",
    )
    model.gradient_checkpointing = True
    model.train()
    model.embed_tokens = nn.Embedding(16, 4)
    layer = _CaptureLayer()
    model.layers = nn.ModuleList([layer])
    model.norm = nn.Identity()
    model.rotary_emb = _FakeRotaryEmbedding()
    model._update_causal_mask = lambda *args: None
    model._gradient_checkpointing_func = lambda function, *args: function(*args)

    supplied_cu = torch.tensor([0, 4], dtype=torch.int32)
    model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        position_ids=torch.tensor([[0, 1, 2, 3]]),
        cu_seq_lens_q=supplied_cu,
        cu_seq_lens_k=supplied_cu,
        max_length_q=4,
        max_length_k=4,
    )

    assert layer.kwargs["cu_seq_lens_q"] is supplied_cu
    assert layer.kwargs["cu_seq_lens_k"] is supplied_cu
    assert layer.kwargs["max_length_q"] == 4
    assert layer.kwargs["max_length_k"] == 4


class _CaptureQwen3MoeModel:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            last_hidden_state=torch.ones((1, 4, 2)),
            router_logits=(torch.ones((4, 2)),),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class _CaptureLoadBalancingSlot:
    def __init__(self, *, enabled, captured):
        self.use_non_eager_impl = enabled
        self.captured = captured

    def __call__(self, _router_logits, _num_experts, _top_k, attention_mask):
        self.captured.append(attention_mask)
        return torch.tensor(0.0)


def _run_qwen3_moe_router_mask(monkeypatch, *, use_slot: bool, local_mask):
    captured = []
    slot = _CaptureLoadBalancingSlot(enabled=use_slot, captured=captured)
    monkeypatch.setattr(qwen3_moe_config, "veomni_load_balancing_loss", slot, raising=False)

    def _eager_loss(_router_logits, _num_experts, _top_k, attention_mask):
        captured.append(attention_mask)
        return torch.tensor(0.0)

    monkeypatch.setattr(qwen3_moe_config, "load_balancing_loss_func", _eager_loss)
    model = _CaptureQwen3MoeModel()
    fake_self = SimpleNamespace(
        config=SimpleNamespace(output_router_logits=True, vocab_size=8),
        model=model,
        lm_head=nn.Identity(),
        num_experts=2,
        num_experts_per_tok=1,
    )
    global_mask = torch.ones((1, 8), dtype=torch.long)
    kwargs = {} if local_mask is None else {"router_attention_mask": local_mask}
    qwen3_moe_config.qwen3_moe_forcausallm_forward_patched(
        fake_self,
        attention_mask=global_mask,
        output_router_logits=True,
        **kwargs,
    )
    assert "router_attention_mask" not in model.kwargs
    return captured, global_mask


def test_qwen3_moe_aux_loss_consumes_local_router_mask_without_forwarding(monkeypatch):
    local_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
    for use_slot in (False, True):
        captured, _ = _run_qwen3_moe_router_mask(monkeypatch, use_slot=use_slot, local_mask=local_mask)
        assert len(captured) == 1
        assert captured[0] is local_mask


def test_qwen3_moe_aux_loss_falls_back_to_attention_mask(monkeypatch):
    captured, global_mask = _run_qwen3_moe_router_mask(monkeypatch, use_slot=False, local_mask=None)
    assert len(captured) == 1
    assert captured[0] is global_mask
