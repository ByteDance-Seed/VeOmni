"""Dense-oracle parity tests for BAGEL span-wise FlashAttention."""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from tests.seed_omni.bagel.contracts.helpers import config_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoTAttention
from veomni.utils.device import get_device_type, get_torch_device


def _dense_attention_mask(split_lens: list[int], attn_modes: list[str], device: torch.device) -> torch.Tensor:
    sample_len = sum(split_lens)
    visible = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    cursor = 0
    for length, mode in zip(split_lens, attn_modes, strict=True):
        if mode == "causal":
            visible[cursor : cursor + length, cursor : cursor + length] = torch.ones(
                (length, length), dtype=torch.bool, device=device
            ).tril()
        else:
            visible[cursor : cursor + length, cursor : cursor + length] = True
        visible[cursor : cursor + length, :cursor] = True
        cursor += length

    cursor = 0
    for length, mode in zip(split_lens, attn_modes, strict=True):
        if mode == "noise":
            visible[:, cursor : cursor + length] = False
            visible[cursor : cursor + length, cursor : cursor + length] = True
        cursor += length

    return torch.zeros_like(visible, dtype=torch.bfloat16).masked_fill_(~visible, float("-inf"))


def _dense_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sample_splits: list[list[int]],
    sample_attn_modes: list[list[str]],
) -> torch.Tensor:
    sample_lens = [sum(split_lens) for split_lens in sample_splits]
    query_samples = torch.split(query, sample_lens, dim=0)
    key_samples = torch.split(key, sample_lens, dim=0)
    value_samples = torch.split(value, sample_lens, dim=0)
    outputs: list[torch.Tensor] = []

    for query_sample, key_sample, value_sample, split_lens, attn_modes in zip(
        query_samples,
        key_samples,
        value_samples,
        sample_splits,
        sample_attn_modes,
        strict=True,
    ):
        mask = _dense_attention_mask(split_lens, attn_modes, query.device)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output = scaled_dot_product_attention(
                query_sample.transpose(0, 1).unsqueeze(0),
                key_sample.transpose(0, 1).unsqueeze(0),
                value_sample.transpose(0, 1).unsqueeze(0),
                attn_mask=mask.unsqueeze(0),
            )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    ("sample_splits", "sample_attn_modes"),
    [
        ([[2, 3, 2]], [["causal", "full", "causal"]]),
        ([[2, 2, 2, 2]], [["full", "noise", "noise", "causal"]]),
        ([[1, 4, 1, 3]], [["full", "full", "full", "noise"]]),
        ([[2, 3], [1, 2, 1]], [["causal", "noise"], ["full", "causal", "full"]]),
    ],
)
@pytest.mark.skipif(get_torch_device().device_count() < 1, reason="device_count should be >= 1")
def test_spanwise_flash_attention_matches_dense_forward_and_gradients(
    sample_splits: list[list[int]],
    sample_attn_modes: list[list[str]],
) -> None:
    device = torch.device(f"{get_device_type()}:0")
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(
        **tiny_bagel_qwen2_cfg(),
        attn_implementation="veomni_flash_attention_2_with_sp",
    )
    attention = BagelQwen2MoTAttention(config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    seq_len = sum(sum(split_lens) for split_lens in sample_splits)
    generator = torch.Generator(device=device).manual_seed(2917)
    tensors = [
        torch.randn(
            seq_len,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    flash_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
    dense_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]

    flash_output = attention._spanwise_flash_attention(
        *flash_inputs,
        sample_splits=sample_splits,
        sample_attn_modes=sample_attn_modes,
    )
    dense_output = _dense_oracle(
        *dense_inputs,
        sample_splits=sample_splits,
        sample_attn_modes=sample_attn_modes,
    )
    torch.testing.assert_close(flash_output, dense_output, rtol=2e-2, atol=2e-2)

    output_gradient = torch.randn(
        flash_output.shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    flash_gradients = torch.autograd.grad(flash_output, flash_inputs, output_gradient)
    dense_gradients = torch.autograd.grad(dense_output, dense_inputs, output_gradient)
    for flash_gradient, dense_gradient in zip(flash_gradients, dense_gradients, strict=True):
        torch.testing.assert_close(flash_gradient, dense_gradient, rtol=3e-2, atol=3e-2)
