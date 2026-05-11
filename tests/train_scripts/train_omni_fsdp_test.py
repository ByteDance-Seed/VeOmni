"""
Two-model FSDP pipeline test script.

Scenario: ModelA (encoder) and ModelB (decoder/LLM) are separately FSDP2-wrapped.
Data flows: input → A.forward() → hidden → B.forward(hidden) → loss.
Verifies:
  - Backward propagates through both FSDP models correctly.
  - Both models receive non-zero, finite gradients.
  - Optimizer states are populated for both models.
  - Grad norm is finite and positive after each step.

Run with torchrun:
    torchrun --nproc_per_node=2 tests/train_scripts/train_omni_fsdp_test.py
    torchrun --nproc_per_node=1 tests/train_scripts/train_omni_fsdp_test.py  (single GPU)
"""

import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard


os.environ.setdefault("NCCL_DEBUG", "OFF")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ──────────────────────────────────────────────────────────────────────────────
# Toy models
# ──────────────────────────────────────────────────────────────────────────────

HIDDEN = 64
SEQ_LEN = 32
VOCAB = 256
BATCH = 2


class ToyEncoder(nn.Module):
    """Simulates a vision / audio encoder producing hidden states."""

    _no_split_modules = ["ToyEncoderLayer"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyEncoderLayer() for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ToyEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN, HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        return self.norm(F.gelu(self.fc(x)))


class ToyLLM(nn.Module):
    """Simulates a causal LLM that accepts optional encoder context.

    The encoder's mean-pooled representation is *added* to every token
    embedding so that gradients flow back to the encoder on all steps.
    """

    _no_split_modules = ["ToyLLMLayer"]

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([ToyLLMLayer() for _ in range(2)])
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        x = self.embed(input_ids)  # (B, T, HIDDEN)
        if encoder_hidden is not None:
            # Add mean-pooled encoder context to every token position.
            # This ensures encoder gradients flow through the loss.
            ctx = encoder_hidden.mean(dim=1, keepdim=True)  # (B, 1, HIDDEN)
            x = x + ctx
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


class ToyLLMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN, HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        return self.norm(F.gelu(self.fc(x)) + x)


# ──────────────────────────────────────────────────────────────────────────────
# FSDP2 wrapping helpers
# ──────────────────────────────────────────────────────────────────────────────


def wrap_fsdp2(model: nn.Module, mesh=None) -> nn.Module:
    """Apply FSDP2 fully_shard to leaf layers first, then to the root."""
    kwargs = {} if mesh is None else {"mesh": mesh}
    for layer in model.modules():
        cls_name = type(layer).__name__
        no_split = getattr(model, "_no_split_modules", [])
        if cls_name in no_split and layer is not model:
            fully_shard(layer, **kwargs)
    fully_shard(model, **kwargs)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Gradient norm utility (works for both FSDP2 and plain modules)
# ──────────────────────────────────────────────────────────────────────────────


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad
            if hasattr(g, "to_local"):
                g = g.to_local()
            total += g.detach().float().norm() ** 2
    return math.sqrt(total)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # ── distributed init ──────────────────────────────────────────────
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if torch.cuda.is_available() else None

    # ── build models ──────────────────────────────────────────────────
    encoder = ToyEncoder().to(device)
    llm = ToyLLM().to(device)

    # ── FSDP2 wrapping (only when multi-GPU) ──────────────────────────
    fsdp_enabled = world_size > 1 and torch.cuda.is_available()
    if fsdp_enabled:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (world_size,))
        encoder = wrap_fsdp2(encoder, mesh)
        llm = wrap_fsdp2(llm, mesh)
        if rank == 0:
            print(f"[FSDP2] world_size={world_size}, mesh={mesh}")
    else:
        if rank == 0:
            print("[Single-GPU] FSDP2 wrapping skipped.")

    # ── optimizer (covers both models) ────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(llm.parameters()),
        lr=1e-3,
    )

    # ── training loop ─────────────────────────────────────────────────
    results = {"loss": [], "grad_norm_encoder": [], "grad_norm_llm": [], "optimizer_state_ok": []}

    torch.manual_seed(42 + rank)
    for step in range(3):
        optimizer.zero_grad()

        # Dummy batch
        pixel_values = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=device)
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
        labels = input_ids.clone()

        # ── forward through encoder (A), then LLM (B) ────────────────
        encoder_hidden = encoder(pixel_values)  # [B, SEQ_LEN, HIDDEN]
        out = llm(input_ids=input_ids, encoder_hidden=encoder_hidden, labels=labels)
        loss = out["loss"]

        # ── backward ─────────────────────────────────────────────────
        loss.backward()

        # ── grad norms ───────────────────────────────────────────────
        gnorm_enc = compute_grad_norm(encoder)
        gnorm_llm = compute_grad_norm(llm)

        optimizer.step()

        # ── optimizer state check (after first step) ─────────────────
        opt_ok = len(optimizer.state) > 0

        loss_val = loss.item()
        if rank == 0:
            print(
                f"step {step}: loss={loss_val:.4f}  "
                f"grad_norm_encoder={gnorm_enc:.4f}  "
                f"grad_norm_llm={gnorm_llm:.4f}  "
                f"optimizer_state_ok={opt_ok}"
            )
            results["loss"].append(loss_val)
            results["grad_norm_encoder"].append(gnorm_enc)
            results["grad_norm_llm"].append(gnorm_llm)
            results["optimizer_state_ok"].append(opt_ok)

    # ── assertions ────────────────────────────────────────────────────
    if rank == 0:
        for step_i, (gne, gnl, opt_ok) in enumerate(
            zip(results["grad_norm_encoder"], results["grad_norm_llm"], results["optimizer_state_ok"])
        ):
            assert math.isfinite(gne) and gne > 0, f"step {step_i}: encoder grad_norm={gne} is bad"
            assert math.isfinite(gnl) and gnl > 0, f"step {step_i}: llm grad_norm={gnl} is bad"
            assert opt_ok, f"step {step_i}: optimizer state is empty"
        print("\n[PASS] All assertions passed: both FSDP models receive valid gradients.")

        # ── save results for pytest ───────────────────────────────────
        output_dir = os.environ.get("OMNI_FSDP_OUTPUT_DIR", "/tmp/omni_fsdp_test")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
