"""
Two-model FSDP pipeline test using VeOmni infrastructure.

Scenario: two independently-FSDP2-wrapped models (encoder → LLM).
Tests that gradient flows correctly through both models and that
VeOmni's optimizer + LR scheduler work end-to-end.

VeOmni infrastructure used:
  - init_parallel_state()       — distributed setup + FSDP mesh
  - build_parallelize_model()   — FSDP2 / no-op wrapping per model
  - build_optimizer()           — AdamW over combined param set
  - build_lr_scheduler()        — constant+warmup schedule
  - veomni_clip_grad_norm()     — FSDP2-aware gradient clipping

Run with torchrun:
    torchrun --nproc_per_node=2 tests/train_scripts/train_omni_fsdp_test.py
    torchrun --nproc_per_node=1 tests/train_scripts/train_omni_fsdp_test.py
"""

import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel

from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


os.environ.setdefault("NCCL_DEBUG", "OFF")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ── Toy model dimensions ──────────────────────────────────────────────────────
HIDDEN = 64
SEQ_LEN = 32
VOCAB = 512
BATCH = 2
MAX_STEPS = 3
LR = 1e-3


def _tiny_llama_config(**extra) -> LlamaConfig:
    """Return a minimal LlamaConfig for use in tests."""
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


# ── Combined module container ─────────────────────────────────────────────────


class CombinedModels(nn.Module):
    """Thin nn.Module container so build_optimizer sees all parameters at once."""

    def __init__(self, encoder: nn.Module, llm: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.llm = llm


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Distributed init ──────────────────────────────────────────────────────
    if not dist.is_initialized():
        dist.init_process_group(backend=get_dist_comm_backend())

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    device_type = get_device_type()
    get_torch_device().set_device(f"{device_type}:{local_rank}")
    device = torch.device(f"{device_type}:{local_rank}")

    # ── VeOmni parallel state (FSDP2 mesh when multi-GPU, plain DDP otherwise) ─
    fsdp_enabled = world_size > 1
    dp_mode = "fsdp2" if fsdp_enabled else "ddp"
    init_device = "meta" if fsdp_enabled else device_type

    init_parallel_state(
        dp_size=world_size,
        dp_mode=dp_mode,
    )

    # ── Build two tiny HuggingFace models ─────────────────────────────────────
    # FSDP2 requires models on "meta"; single-GPU path materialises on device.
    enc_cfg = _tiny_llama_config()
    llm_cfg = _tiny_llama_config()

    if init_device == "meta":
        encoder: nn.Module = LlamaModel(enc_cfg).to("meta")
        llm: nn.Module = LlamaForCausalLM(llm_cfg).to("meta")
    else:
        torch.manual_seed(42)
        encoder = LlamaModel(enc_cfg).to(device)
        llm = LlamaForCausalLM(llm_cfg).to(device)

    # ── VeOmni parallelize (FSDP2 or no-op) ──────────────────────────────────
    # Mixed precision disabled to keep numerics simple for this test.
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

    # ── Combined container for optimizer ─────────────────────────────────────
    combined = CombinedModels(encoder, llm)

    # ── VeOmni optimizer + LR scheduler ──────────────────────────────────────
    optimizer = build_optimizer(combined, lr=LR, weight_decay=1e-2, fused=False)
    lr_scheduler = build_lr_scheduler(optimizer, train_steps=MAX_STEPS, lr=LR)

    # ── Training loop ─────────────────────────────────────────────────────────
    results = {
        "loss": [],
        "grad_norm_encoder": [],
        "grad_norm_llm": [],
        "lr": [],
    }

    torch.manual_seed(42 + rank)
    for step in range(MAX_STEPS):
        optimizer.zero_grad()

        # Dummy batch on device
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
        labels = input_ids.clone()

        # ── Forward: encoder → LLM ─────────────────────────────────────────
        # encoder produces hidden states for each token position (B, T, HIDDEN).
        # We pass them directly as inputs_embeds to the LLM, bypassing its own
        # embedding lookup — this is the correct pattern for FSDP2 chaining:
        # each model's sub-modules are only accessed inside its own forward().
        enc_out = encoder(input_ids=input_ids)  # LlamaBaseModelOutput
        hidden = enc_out.last_hidden_state  # (B, T, HIDDEN) — plain tensor

        llm_out = llm(inputs_embeds=hidden, labels=labels)
        loss = llm_out.loss

        # ── Backward ──────────────────────────────────────────────────────
        loss.backward()

        # ── VeOmni grad norm (FSDP2-aware) ────────────────────────────────
        grad_norm = veomni_clip_grad_norm(combined, max_norm=1.0)

        # Per-model norms for reporting
        def _model_grad_norm(model: nn.Module) -> float:
            total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                    total += g.detach().float().norm() ** 2
            return math.sqrt(total)

        gnorm_enc = _model_grad_norm(encoder)
        gnorm_llm = _model_grad_norm(llm)

        optimizer.step()
        lr_scheduler.step()

        cur_lr = lr_scheduler.get_last_lr()[0]
        loss_val = loss.item()

        if rank == 0:
            print(
                f"step {step}:  loss={loss_val:.4f}  "
                f"grad_norm={grad_norm:.4f}  "
                f"gnorm_enc={gnorm_enc:.4f}  gnorm_llm={gnorm_llm:.4f}  "
                f"lr={cur_lr:.2e}"
            )
            results["loss"].append(loss_val)
            results["grad_norm_encoder"].append(gnorm_enc)
            results["grad_norm_llm"].append(gnorm_llm)
            results["lr"].append(cur_lr)

    # ── Assertions ────────────────────────────────────────────────────────────
    if rank == 0:
        for i, (gne, gnl) in enumerate(zip(results["grad_norm_encoder"], results["grad_norm_llm"])):
            assert math.isfinite(gne) and gne > 0, f"step {i}: encoder grad_norm={gne}"
            assert math.isfinite(gnl) and gnl > 0, f"step {i}: llm grad_norm={gnl}"
        assert len(optimizer.state) > 0, "optimizer state is empty"
        print("\n[PASS] All assertions passed.")

        # Save results for the pytest wrapper
        output_dir = os.environ.get("OMNI_FSDP_OUTPUT_DIR", "/tmp/omni_fsdp_test")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
