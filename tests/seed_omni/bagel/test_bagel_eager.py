"""Smoke coverage for native eager Bagel Qwen2-MoT train and sequential infer."""

from __future__ import annotations

import pytest
import torch

from tests.seed_omni.bagel.helpers import (
    build_toy_conversation,
    config_cls,
    native_model_cls,
    run_eager_mot,
    tiny_bagel_qwen2_cfg,
)
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_FLOW_HIDDEN, BAGEL_FLOW_QUERY
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def _eager_config(**overrides):
    return config_cls("bagel_qwen2_mot")(**{**tiny_bagel_qwen2_cfg(), **overrides, "attn_implementation": "sdpa"})


def test_eager_packed_train_forward_runs() -> None:
    model = native_model_cls("bagel_qwen2_mot")(_eager_config()).train()
    conversation = build_toy_conversation(
        "mixed",
        hidden_size=int(model.config.hidden_size),
        vocab_size=int(model.config.vocab_size),
        patch_latent_dim=4,
        device=model.device,
        dtype=model.dtype,
    )

    hidden_states = run_eager_mot(model, conversation)
    assert hidden_states.shape[-1] == int(model.config.hidden_size)
    assert torch.isfinite(hidden_states).all()

    hidden_states.float().square().mean().backward()
    query_grad = model.model.layers[0].self_attn.q_proj.weight.grad
    gen_query_grad = model.model.layers[0].self_attn.q_proj_moe_gen.weight.grad
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert gen_query_grad is not None and torch.isfinite(gen_query_grad).all()
    assert query_grad.abs().max() > 0
    assert gen_query_grad.abs().max() > 0


def _eager_cuda_model():
    device = torch.device(get_device_type())
    return native_model_cls("bagel_qwen2_mot")(_eager_config()).to(device=device, dtype=torch.bfloat16).eval()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="eager sequential infer smoke uses CUDA bf16")
def test_eager_und_generate_runs() -> None:
    model = _eager_cuda_model()
    prompt = ConversationItem(
        type="text",
        value=torch.randn(4, int(model.config.hidden_size), device=model.device, dtype=model.dtype),
        role="user",
    )

    outputs = model.generate([prompt], generation_kwargs={"infer_type": "infer_und"})
    conversation = outputs["conversation_list"]
    tail = conversation[-1]
    assert tail.type == "output"
    assert torch.is_tensor(tail.value)
    assert tail.value.shape == (1, int(model.config.hidden_size))
    assert torch.isfinite(tail.value).all()
    assert model._generation_state.main.cache is not None


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="eager sequential infer smoke uses CUDA bf16")
def test_eager_gen_denoise_runs() -> None:
    model = _eager_cuda_model()
    hidden_size = int(model.config.hidden_size)
    prompt = ConversationItem(
        type="text",
        value=torch.randn(3, hidden_size, device=model.device, dtype=model.dtype),
        role="user",
    )
    model.generate([prompt], generation_kwargs={"infer_type": "infer_gen"})

    query = ConversationItem(
        type="output",
        value=torch.randn(5, hidden_size, device=model.device, dtype=model.dtype),
        role="assistant",
        source=BAGEL_FLOW_QUERY,
        meta={"timestep": 0.5},
    )
    outputs = model.denoise_branch(
        [query],
        generation_kwargs={"infer_type": "infer_gen", "cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
    )
    tail = outputs["conversation_list"][-1]
    assert tail.source == BAGEL_FLOW_HIDDEN
    assert torch.is_tensor(tail.value)
    assert tail.value.shape[0] == 5
    assert tail.value.shape[-1] == hidden_size
    assert torch.isfinite(tail.value).all()
