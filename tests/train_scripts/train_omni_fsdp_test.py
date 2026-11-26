"""
Two-model FSDP2 / DDP pipeline smoke runner launched via torchrun.

Scenario: two independently wrapped models (encoder -> LLM).
Tests that gradients flow through both models on every rank and that
VeOmni's optimizer + LR scheduler step end-to-end.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel

from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


os.environ.setdefault("NCCL_DEBUG", "OFF")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

HIDDEN = 64
SEQ_LEN = 32
VOCAB = 512
BATCH = 2
MAX_STEPS = 3
LR = 1e-3


def _tiny_llama_config(**extra) -> LlamaConfig:
    """Return a minimal LlamaConfig for pipeline smoke testing."""
    return LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 4,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=VOCAB,
        max_position_embeddings=128,
        **extra,
    )


class CombinedModels(nn.Module):
    """Thin container so build_optimizer sees both sub-models' parameters."""

    def __init__(self, encoder: nn.Module, llm: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.llm = llm


def _local_grad_norm(model: nn.Module) -> float:
    """Compute the L2 norm of local parameter gradient shards on this rank."""
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
            total += grad.detach().float().norm().item() ** 2
    return math.sqrt(total)


def main() -> None:
    """Run the two-model FSDP2 / DDP training pipeline smoke test."""
    if not dist.is_initialized():
        dist.init_process_group(backend=get_dist_comm_backend())

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    device_type = get_device_type()
    get_torch_device().set_device(f"{device_type}:{local_rank}")
    device = torch.device(f"{device_type}:{local_rank}")

    try:
        fsdp_enabled = world_size > 1
        dp_mode = "fsdp2" if fsdp_enabled else "ddp"
        init_device = "meta" if fsdp_enabled else device_type

        _init_parallel_state(dp_size=world_size, dp_mode=dp_mode)

        enc_cfg = _tiny_llama_config()
        llm_cfg = _tiny_llama_config()

        if init_device == "meta":
            encoder: nn.Module = LlamaModel(enc_cfg).to("meta")
            llm: nn.Module = LlamaForCausalLM(llm_cfg).to("meta")
        else:
            torch.manual_seed(42)
            encoder = LlamaModel(enc_cfg).to(device)
            llm = LlamaForCausalLM(llm_cfg).to(device)

        no_mp = MixedPrecisionConfig(enable=False)
        encoder = build_parallelize_model(
            encoder,
            init_device=init_device,
            weights_path=None,
            mixed_precision=no_mp,
            enable_gradient_checkpointing=False,
            basic_modules=["LlamaDecoderLayer"],
        )
        llm = build_parallelize_model(
            llm,
            init_device=init_device,
            weights_path=None,
            mixed_precision=no_mp,
            enable_gradient_checkpointing=False,
            basic_modules=["LlamaDecoderLayer"],
        )
        encoder.train()
        llm.train()

        combined = CombinedModels(encoder, llm)
        optimizer = build_optimizer(combined, lr=LR, weight_decay=1e-2, fused=False)
        lr_scheduler = build_lr_scheduler(optimizer, train_steps=MAX_STEPS, lr=LR)

        results = {
            "loss": [],
            "grad_norm_encoder": [],
            "grad_norm_llm": [],
            "optimizer_state_ok": [],
            "lr": [],
        }

        torch.manual_seed(42 + rank)
        for step in range(MAX_STEPS):
            optimizer.zero_grad()

            input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
            labels = input_ids.clone()

            enc_out = encoder(input_ids=input_ids)
            llm_out = llm(inputs_embeds=enc_out.last_hidden_state, labels=labels)
            loss = llm_out.loss
            loss.backward()

            grad_norm = veomni_clip_grad_norm(combined, max_norm=1.0)
            gnorm_enc = _local_grad_norm(encoder)
            gnorm_llm = _local_grad_norm(llm)
            assert math.isfinite(grad_norm) and grad_norm > 0, f"rank {rank} step {step}: grad_norm={grad_norm}"
            assert math.isfinite(gnorm_enc) and gnorm_enc > 0, (
                f"rank {rank} step {step}: encoder grad_norm={gnorm_enc}"
            )
            assert math.isfinite(gnorm_llm) and gnorm_llm > 0, f"rank {rank} step {step}: llm grad_norm={gnorm_llm}"

            optimizer.step()
            lr_scheduler.step()

            if rank == 0:
                results["loss"].append(loss.item())
                results["grad_norm_encoder"].append(gnorm_enc)
                results["grad_norm_llm"].append(gnorm_llm)
                results["optimizer_state_ok"].append(len(optimizer.state) > 0)
                results["lr"].append(lr_scheduler.get_last_lr()[0])

        if rank == 0:
            output_dir = os.environ.get("OMNI_FSDP_OUTPUT_DIR", "/tmp/omni_fsdp_test")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        dist.barrier()
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


if __name__ == "__main__":
    main()
