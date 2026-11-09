"""Smoke coverage for native eager Bagel Qwen2-MoT train and sequential infer."""

from __future__ import annotations

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.seed_omni.modules.bagel.helpers import (
    build_toy_conversation,
    config_cls,
    native_model_cls,
    run_eager_mot,
    tiny_bagel_qwen2_cfg,
)
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_PHASE_KEY,
    BAGEL_START_TOKEN,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _eager_config(**overrides):
    return config_cls("bagel_qwen2_mot")(**{**tiny_bagel_qwen2_cfg(), **overrides, "attn_implementation": "sdpa"})


def test_eager_packed_train_forward_runs() -> None:
    """Verify eager packed MoT forward and backward populate both und and gen expert gradients."""
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


def _eager_model(dtype: torch.dtype):
    return native_model_cls("bagel_qwen2_mot")(_eager_config()).to(device="cpu", dtype=dtype).eval()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_eager_und_generate_runs(dtype: torch.dtype) -> None:
    """Verify eager understanding generation preserves the configured dtype across FP32, FP16, and BF16."""
    model = _eager_model(dtype)
    prompt = ConversationItem(
        type="text",
        value=torch.randn(4, int(model.config.hidden_size), device=model.device, dtype=model.dtype),
        role="user",
    )
    bos = ConversationItem(
        type="text",
        value=torch.randn(1, int(model.config.hidden_size), device=model.device, dtype=model.dtype),
        role="assistant",
        meta={BAGEL_PHASE_KEY: BAGEL_START_TOKEN},
    )

    outputs = model.generate([prompt, bos])
    conversation = outputs["conversation_list"]
    tail = conversation[-1]
    assert model._generation_state.main.cache is not None
    assert tail.type == "output"
    assert tail.meta.get(BAGEL_PHASE_KEY) == BAGEL_START_TOKEN
    assert torch.is_tensor(tail.value)
    assert tail.value.dtype == dtype
    assert tail.value.shape == (1, int(model.config.hidden_size))
    assert torch.isfinite(tail.value).all()

    outputs = model.generate(conversation)
    tail = outputs["conversation_list"][-1]
    assert tail.type == "output"
    assert BAGEL_PHASE_KEY not in tail.meta

    outputs = model.generate(outputs["conversation_list"])
    tail = outputs["conversation_list"][-1]
    assert tail.type == "output"
    assert torch.is_tensor(tail.value)
    assert tail.value.dtype == dtype
    assert tail.value.shape == (1, int(model.config.hidden_size))
    assert torch.isfinite(tail.value).all()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_eager_gen_denoise_runs(dtype: torch.dtype) -> None:
    """Verify eager ``denoise_branch`` preserves the configured dtype across FP32, FP16, and BF16."""
    model = _eager_model(dtype)
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
        meta={BAGEL_PHASE_KEY: BAGEL_FLOW_QUERY, "timestep": 0.5},
    )
    outputs = model.denoise_branch(
        [query],
        generation_kwargs={"infer_type": "infer_gen", "cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
    )
    tail = outputs["conversation_list"][-1]
    assert tail.meta.get(BAGEL_PHASE_KEY) == BAGEL_FLOW_HIDDEN
    assert torch.is_tensor(tail.value)
    assert tail.value.dtype == dtype
    assert tail.value.shape[0] == 5
    assert tail.value.shape[-1] == hidden_size
    assert torch.isfinite(tail.value).all()
