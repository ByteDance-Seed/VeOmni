#!/usr/bin/env python3
"""Exercise compiled FlexAttention forward/backward on every A100."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.utils import is_torch_flex_attn_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=129)
    return parser.parse_args()


def causal_mask(_batch, _head, query_index, key_index):
    return query_index >= key_index


def identity_score(score, _batch, _head, _query_index, _key_index):
    # Transformers 4.51.3 passes a score_mod together with its causal BlockMask,
    # even when the model does not request soft-capping or a dense score mask.
    return score


def main() -> None:
    args = parse_args()
    assert is_torch_flex_attn_available()
    assert torch.__version__.startswith("2.7.1"), torch.__version__
    assert torch.version.cuda and torch.version.cuda.startswith("12.6"), torch.version.cuda
    assert triton.__version__ == "3.3.1", triton.__version__

    gpu_count = torch.cuda.device_count()
    assert gpu_count == args.expected_gpus, f"expected {args.expected_gpus} GPUs, found {gpu_count}"

    compiled_flex_attention = torch.compile(flex_attention)
    batch_size = 1
    query_heads = 8
    key_value_heads = 2
    head_dim = 64

    for device_index in range(gpu_count):
        device = torch.device("cuda", device_index)
        props = torch.cuda.get_device_properties(device)
        assert (props.major, props.minor) == (8, 0), (
            f"GPU {device_index} is sm{props.major}{props.minor}, expected an A100 (sm80)"
        )

        with torch.cuda.device(device):
            query = torch.randn(
                batch_size,
                query_heads,
                args.sequence_length,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            key = torch.randn(
                batch_size,
                key_value_heads,
                args.sequence_length,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            value = torch.randn_like(key, requires_grad=True)
            block_mask = create_block_mask(
                causal_mask,
                B=batch_size,
                H=None,
                Q_LEN=args.sequence_length,
                KV_LEN=args.sequence_length,
                device=str(device),
                _compile=True,
            )

            output, logsumexp = compiled_flex_attention(
                query,
                key,
                value,
                score_mod=identity_score,
                block_mask=block_mask,
                enable_gqa=True,
                return_lse=True,
            )
            with torch.no_grad():
                repeated_key = key.repeat_interleave(query_heads // key_value_heads, dim=1)
                repeated_value = value.repeat_interleave(query_heads // key_value_heads, dim=1)
                reference = F.scaled_dot_product_attention(
                    query,
                    repeated_key,
                    repeated_value,
                    is_causal=True,
                )

            torch.testing.assert_close(output, reference, rtol=5e-2, atol=5e-2)
            assert torch.isfinite(logsumexp).all()
            output.float().square().mean().backward()
            for name, tensor in (("query", query), ("key", key), ("value", value)):
                assert tensor.grad is not None, f"{name} gradient was not produced"
                assert torch.isfinite(tensor.grad).all(), f"{name} gradient is not finite"
            torch.cuda.synchronize(device)

        print(
            f"GPU {device_index}: {props.name}, FlexAttention BF16 causal GQA "
            f"forward/backward OK at sequence length {args.sequence_length}"
        )

    print("A100 FlexAttention smoke test passed.")


if __name__ == "__main__":
    main()
