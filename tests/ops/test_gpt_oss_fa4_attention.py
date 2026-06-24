import torch
from torch import nn
from transformers import GptOssConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

from veomni.ops.kernels import attention as veomni_attention


class _FakeAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"_attn_implementation": "veomni_flash_attention_4_with_sp"})()
        self.is_causal = True
        self.layer_idx = 3
        self.proj = nn.Linear(4, 4)


def test_gpt_oss_fa4_attention_forwards_sliding_window_and_sinks(monkeypatch):
    captured = {}

    def fake_flash_attention_forward(query, key, value, attention_mask, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(query)

    monkeypatch.setattr(veomni_attention, "_flash_attention_forward", fake_flash_attention_forward)

    module = _FakeAttentionModule()
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    sinks = torch.randn(2)

    output, attn_weights = veomni_attention.flash_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        scaling=0.5,
        sliding_window=8,
        s_aux=sinks,
    )

    assert output.shape == (1, 3, 2, 4)
    assert attn_weights is None
    assert captured["attn_implementation"] == "veomni_flash_attention_4_with_sp"
    assert captured["sliding_window"] == 8
    assert captured["s_aux"] is sinks
    assert captured["softmax_scale"] == 0.5
    assert captured["layer_idx"] == 3


def test_gpt_oss_attention_passes_learnable_sinks_to_attention_backend():
    captured = {}
    backend_name = "veomni_test_gpt_oss_attention"

    def fake_attention_backend(module, query, key, value, attention_mask, **kwargs):
        captured["module"] = module
        captured["s_aux"] = kwargs["s_aux"]
        captured["sliding_window"] = kwargs["sliding_window"]
        return torch.zeros_like(query.transpose(1, 2)), None

    ALL_ATTENTION_FUNCTIONS.register(backend_name, fake_attention_backend)
    try:
        config = GptOssConfig(
            hidden_size=16,
            head_dim=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            sliding_window=8,
        )
        config._attn_implementation = backend_name
        attention = GptOssAttention(config, layer_idx=0)
        hidden_states = torch.randn(1, 3, 16)
        cos = torch.ones(1, 3, 2)
        sin = torch.zeros(1, 3, 2)

        output, attn_weights = attention(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )
    finally:
        type(ALL_ATTENTION_FUNCTIONS)._global_mapping.pop(backend_name, None)

    assert output.shape == hidden_states.shape
    assert attn_weights is None
    assert captured["module"] is attention
    assert captured["s_aux"] is attention.sinks
    assert captured["sliding_window"] == 8
