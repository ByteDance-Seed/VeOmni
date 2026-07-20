#!/usr/bin/env python3
"""Validate the VeOmni CUDA stack and run FlashAttention 2 on every GPU."""

from __future__ import annotations

import argparse
import importlib.metadata

import flash_attn
import peft
import torch
import transformers
import veomni
from flash_attn import flash_attn_func
from veomni.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM  # noqa: F401
from veomni.utils.arguments import ModelArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA is not available inside the container"
    assert torch.__version__.startswith("2.7.1"), torch.__version__
    assert torch.version.cuda and torch.version.cuda.startswith("12.6"), torch.version.cuda
    assert torch._C._GLIBCXX_USE_CXX11_ABI is True
    assert transformers.__version__ == "4.51.3", transformers.__version__
    assert flash_attn.__version__ == "2.7.4.post1", flash_attn.__version__
    assert (
        ModelArguments.__dataclass_fields__["attn_implementation"].default
        == "flash_attention_2"
    )

    gpu_count = torch.cuda.device_count()
    assert gpu_count == args.expected_gpus, f"expected {args.expected_gpus} GPUs, found {gpu_count}"

    print(f"VeOmni:       {veomni.__version__}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"NCCL:         {torch.cuda.nccl.version()}")
    print(f"Transformers: {transformers.__version__}")
    print(f"FlashAttn:    {flash_attn.__version__}")
    print(f"PEFT:         {peft.__version__}")
    print(f"Liger kernel: {importlib.metadata.version('liger-kernel')}")

    for device_index in range(gpu_count):
        props = torch.cuda.get_device_properties(device_index)
        assert (props.major, props.minor) == (8, 0), (
            f"GPU {device_index} is sm{props.major}{props.minor}, expected an A100 (sm80)"
        )
        with torch.cuda.device(device_index):
            query = torch.randn(
                2,
                128,
                8,
                64,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            output = flash_attn_func(query, query, query, causal=True)
            assert output.shape == query.shape
            assert torch.isfinite(output).all()
            output.float().square().mean().backward()
            assert query.grad is not None
            assert torch.isfinite(query.grad).all()
            torch.cuda.synchronize()
        print(f"GPU {device_index}: {props.name}, FlashAttention 2 BF16 forward/backward OK")

    print("A100 CUDA/VeOmni smoke test passed.")


if __name__ == "__main__":
    main()
