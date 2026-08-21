"""Accelerated Bagel Qwen2-MoT vs eager CE / MSE alignment on toy data."""

from __future__ import annotations

import copy

import pytest
import torch

from tests.seed_omni.bagel.helpers import (
    ALIGN_ATOL,
    ALIGN_GRAD_ATOL,
    ALIGN_GRAD_RTOL,
    ALIGN_RTOL,
    ToyCase,
    build_toy_conversation,
    clone_conversation,
    config_cls,
    conversation_ce_loss,
    conversation_mse_loss,
    native_model_cls,
    run_accelerated_mot,
    run_eager_mot,
    tiny_align_qwen2_cfg,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import BagelQwen2MoTAccelerated
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


_PATCH_LATENT_DIM = 4
_CASES: tuple[ToyCase, ...] = ("ce_only", "vit_ce", "mse_only", "mixed")


def _eager_config():
    return config_cls("bagel_qwen2_mot")(**tiny_align_qwen2_cfg(), attn_implementation="sdpa")


def _flex_config():
    return config_cls("bagel_qwen2_mot")(
        **tiny_align_qwen2_cfg(),
        attn_implementation="veomni_flex_attention_with_sp",
    )


def _shared_heads(hidden_size: int, vocab_size: int, device: torch.device) -> tuple[torch.nn.Linear, torch.nn.Linear]:
    lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False).to(device=device, dtype=torch.float32)
    llm2vae = torch.nn.Linear(hidden_size, _PATCH_LATENT_DIM, bias=False).to(device=device, dtype=torch.float32)
    return lm_head, llm2vae


def _build_aligned_models(device: torch.device, dtype: torch.dtype):
    torch.manual_seed(29)
    eager = native_model_cls("bagel_qwen2_mot")(_eager_config()).to(device=device, dtype=dtype).train()
    accelerated = BagelQwen2MoTAccelerated(_flex_config()).to(device=device, dtype=dtype).train()
    accelerated.load_state_dict(copy.deepcopy(eager.state_dict()))
    return eager, accelerated


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FlexAttention alignment requires CUDA")
@pytest.mark.parametrize("case", _CASES)
def test_accelerated_ce_mse_matches_eager_on_toy_data(case: ToyCase) -> None:
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    eager, accelerated = _build_aligned_models(device, dtype)
    hidden_size = int(eager.config.hidden_size)
    vocab_size = int(eager.config.vocab_size)
    conversation = build_toy_conversation(
        case,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        patch_latent_dim=_PATCH_LATENT_DIM,
        device=device,
        dtype=dtype,
    )
    eager_conversation = clone_conversation(conversation)
    accelerated_conversation = clone_conversation(conversation)
    lm_head, llm2vae = _shared_heads(hidden_size, vocab_size, device)

    eager_hidden = run_eager_mot(eager, eager_conversation)
    accelerated_hidden = run_accelerated_mot(accelerated, accelerated_conversation)

    assert torch.isfinite(eager_hidden).all()
    assert torch.isfinite(accelerated_hidden).all()
    assert eager_hidden.shape == accelerated_hidden.shape

    eager_ce = conversation_ce_loss(eager_conversation, lm_head)
    accelerated_ce = conversation_ce_loss(accelerated_conversation, lm_head)
    eager_mse = conversation_mse_loss(eager_conversation, llm2vae)
    accelerated_mse = conversation_mse_loss(accelerated_conversation, llm2vae)

    if case in {"ce_only", "vit_ce", "mixed"}:
        assert eager_ce is not None and accelerated_ce is not None
        assert torch.isfinite(eager_ce) and torch.isfinite(accelerated_ce)
        torch.testing.assert_close(accelerated_ce, eager_ce, atol=ALIGN_ATOL, rtol=ALIGN_RTOL)
    else:
        assert eager_ce is None and accelerated_ce is None

    if case in {"mse_only", "mixed"}:
        assert eager_mse is not None and accelerated_mse is not None
        assert torch.isfinite(eager_mse) and torch.isfinite(accelerated_mse)
        torch.testing.assert_close(accelerated_mse, eager_mse, atol=ALIGN_ATOL, rtol=ALIGN_RTOL)
    else:
        assert eager_mse is None and accelerated_mse is None

    eager_loss = eager_hidden.float().square().mean()
    accelerated_loss = accelerated_hidden.float().square().mean()
    if eager_ce is not None:
        eager_loss = eager_loss + eager_ce
        accelerated_loss = accelerated_loss + accelerated_ce
    if eager_mse is not None:
        eager_loss = eager_loss + eager_mse
        accelerated_loss = accelerated_loss + accelerated_mse
    eager_loss.backward()
    accelerated_loss.backward()

    for name in (
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
    ):
        eager_grad = dict(eager.named_parameters())[name].grad
        accelerated_grad = dict(accelerated.named_parameters())[name].grad
        assert eager_grad is not None and accelerated_grad is not None
        assert torch.isfinite(eager_grad).all() and torch.isfinite(accelerated_grad).all()
        torch.testing.assert_close(
            accelerated_grad,
            eager_grad,
            atol=ALIGN_GRAD_ATOL,
            rtol=ALIGN_GRAD_RTOL,
            msg=f"gradient mismatch for {name}",
        )
