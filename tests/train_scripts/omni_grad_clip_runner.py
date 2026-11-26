"""
SeedOmni per-module and global grad-clip parity runner launched via torchrun.

Wraps two Omni modules (``encoder`` and ``decoder``) under their own named
``ParallelState``s and verifies that ``veomni_omni_model_clip_grad_norm``
(under both ``per_module`` with distinct per-module thresholds and ``global``
with a shared threshold) numerically matches an unsharded single-model
reference on pre-clip norms and every post-clip parameter gradient across
``DDP+SP4``, ``FSDP2``, and ``FSDP2+EP2``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor, Shard

from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.clip_grad_norm import veomni_omni_model_clip_grad_norm
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import (
    _init_parallel_state,
    clear_parallel_state,
    get_parallel_state_by_name,
    set_parallel_state,
    use_parallel_state,
)
from veomni.distributed.sequence_parallel.loss import reduce_sequence_parallel_loss
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime
from veomni.optim import build_optimizer
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


os.environ.setdefault("NCCL_DEBUG", "OFF")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

HIDDEN = 64
LR = 1e-3
# Distinct thresholds well below the ~0.055 / ~0.169 / ~0.177 pre-clip norms so
# clipping always scales gradients and per-module threshold routing is exercised.
ENC_MAX_NORM = 0.01
DEC_MAX_NORM = 0.03
GLOBAL_MAX_NORM = 0.05


class _ToyBlock(nn.Module):
    """Two-linear block used as the FSDP2 wrap unit."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


class _ToyDenseModule(nn.Module):
    """Dense two-block toy OmniModule."""

    _no_split_modules = ["_ToyBlock"]

    def __init__(self, dim: int = HIDDEN) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.block1 = _ToyBlock(dim)
        self.block2 = _ToyBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


class _ToyMoeExperts(nn.Module):
    """3D expert bank sharded on dim-0 under EP."""

    def __init__(self, num_experts: int = 8, dim: int = HIDDEN) -> None:
        super().__init__()
        self.experts = nn.Parameter(torch.empty(num_experts, dim, dim))

    def forward(self, x: torch.Tensor, expert_slice: slice | None = None) -> torch.Tensor:
        weights = self.experts if expert_slice is None else self.experts[expert_slice]
        return torch.einsum("...d,edh->...eh", x, weights).mean(dim=-2)


class _ToyMoeDecoderLayer(nn.Module):
    """Decoder layer enclosing both a dense block and an EP expert bank."""

    def __init__(self, num_experts: int = 8, dim: int = HIDDEN) -> None:
        super().__init__()
        self.block = _ToyBlock(dim)
        self.moe = _ToyMoeExperts(num_experts=num_experts, dim=dim)

    def forward(self, x: torch.Tensor, expert_slice: slice | None = None) -> torch.Tensor:
        return self.moe(self.block(x), expert_slice=expert_slice)


class _ToyMoeModule(nn.Module):
    """Toy MoE OmniModule declaring an EP ParallelPlan on decoder.moe.experts."""

    _no_split_modules = ["_ToyMoeDecoderLayer"]

    def __init__(self, num_experts: int = 8, dim: int = HIDDEN) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.decoder = _ToyMoeDecoderLayer(num_experts=num_experts, dim=dim)

    def forward(self, x: torch.Tensor, expert_slice: slice | None = None) -> torch.Tensor:
        return self.decoder(x, expert_slice=expert_slice)

    def get_parallel_plan(self) -> ParallelPlan:
        plan = ParallelPlan(extra_parallel_plan={"ep": {"decoder.moe.experts": Shard(0)}})
        plan.extra_parallel_fsdp_no_shard_module = {"ep": {"decoder.moe"}}
        return plan


def _init_deterministic_module(cls: type[nn.Module], seed: int, device: torch.device) -> nn.Module:
    """Instantiate ``cls`` on ``device`` with deterministic normal weights."""
    torch.manual_seed(seed)
    mod = cls().to(device)
    for param in mod.parameters():
        nn.init.normal_(param, mean=0.0, std=0.08)
    return mod


def _bind_module_runtime_handle(name: str, model: nn.Module, max_grad_norm: float, dp_mode: str) -> ModuleRuntime:
    """Create a lightweight ModuleRuntime handle bound to ``name`` and ``model``."""
    rt = object.__new__(ModuleRuntime)
    rt.model_name = name
    rt.model = model
    rt.args = SimpleNamespace(
        optimizer=SimpleNamespace(max_grad_norm=max_grad_norm),
        accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode=dp_mode)),
    )
    return rt


def _run_forward_backward_pair(
    encoder: nn.Module,
    decoder: nn.Module,
    ref_encoder: nn.Module,
    ref_decoder: nn.Module,
    global_tokens: torch.Tensor,
    rank: int,
    world_size: int,
    sp_size: int,
    ep_size: int,
) -> None:
    """Run one zero_grad + forward + backward pass on both the distributed pair and the unsharded reference."""
    encoder.zero_grad()
    decoder.zero_grad()
    ref_encoder.zero_grad()
    ref_decoder.zero_grad()

    local_tokens = global_tokens[rank]
    ps_dec = get_parallel_state_by_name("decoder")

    with use_parallel_state("encoder"):
        hidden = encoder(local_tokens)
    with use_parallel_state("decoder"):
        out = decoder(hidden)
        loss = out.square().sum(dim=-1).mean()
        if sp_size > 1:
            loss = reduce_sequence_parallel_loss(
                loss,
                torch.tensor(float(local_tokens.size(0)), device=local_tokens.device),
                group=ps_dec.sp_group,
            )
    loss.backward()

    experts_per_ep = 8 // ep_size
    ref_losses = []
    for r in range(world_size):
        ref_h = ref_encoder(global_tokens[r])
        if ep_size > 1:
            r_ep_rank = r % ep_size
            expert_slice = slice(r_ep_rank * experts_per_ep, (r_ep_rank + 1) * experts_per_ep)
            ref_o = ref_decoder(ref_h, expert_slice=expert_slice)
        else:
            ref_o = ref_decoder(ref_h)
        ref_losses.append(ref_o.square().sum(dim=-1).mean())
    torch.stack(ref_losses).mean().backward()


def _assert_both_modules_grads_match(
    encoder: nn.Module,
    decoder: nn.Module,
    ref_encoder: nn.Module,
    ref_decoder: nn.Module,
    ep_size: int,
    dec_ep_rank: int,
    rank: int,
) -> None:
    """Assert every parameter gradient of both encoder and decoder matches the reference."""
    for label, mod, ref_mod, mod_ep_size, mod_ep_rank in (
        ("encoder", encoder, ref_encoder, 1, 0),
        ("decoder", decoder, ref_decoder, ep_size, dec_ep_rank),
    ):
        raw_mod = mod.module if hasattr(mod, "module") else mod
        for (name, param), (_, ref_param) in zip(raw_mod.named_parameters(), ref_mod.named_parameters(), strict=True):
            assert param.grad is not None and ref_param.grad is not None, f"missing grad for {label}.{name}"
            actual_g = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            expected_g = (
                ref_param.grad.chunk(mod_ep_size, dim=0)[mod_ep_rank]
                if (mod_ep_size > 1 and "moe.experts" in name)
                else ref_param.grad
            )
            torch.testing.assert_close(
                actual_g,
                expected_g,
                rtol=1e-4,
                atol=1e-6,
                msg=lambda m, label=label, name=name: f"[rank {rank}] {label}.{name} grad mismatch: {m}",
            )


def main() -> None:
    """Verify per_module and global Omni grad clipping against a single-model reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-mode", choices=("ddp", "fsdp2"), default="fsdp2")
    parser.add_argument("--ulysses-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    cli_args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend=get_dist_comm_backend())

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    device_type = get_device_type()
    get_torch_device().set_device(f"{device_type}:{local_rank}")
    device = torch.device(f"{device_type}:{local_rank}")

    try:
        dp_mode = cli_args.dp_mode
        sp_size = cli_args.ulysses_size
        ep_size = cli_args.ep_size
        dp_size = world_size // sp_size

        # Encoder is always dense (ep_size=1) while decoder uses ep_size (MoE when ep_size > 1),
        # so the two modules have distinct ParallelState objects when EP is active.
        for state_name, mod_ep in (("base", 1), ("encoder", 1), ("decoder", ep_size)):
            _init_parallel_state(
                dp_size=dp_size,
                dp_shard_size=dp_size,
                ulysses_size=sp_size,
                dp_mode=dp_mode,
                extra_parallel_sizes=(mod_ep,),
                extra_parallel_placement_innermost=(False,),
                extra_parallel_names=("ep",),
                name=state_name,
            )

        dec_cls = _ToyMoeModule if ep_size > 1 else _ToyDenseModule
        ref_encoder = _init_deterministic_module(_ToyDenseModule, seed=101, device=device)
        ref_decoder = _init_deterministic_module(dec_cls, seed=202, device=device)
        ref_encoder.train()
        ref_decoder.train()

        shared_root = Path(os.environ.get("OMNI_FSDP_OUTPUT_DIR", tempfile.gettempdir())) / "grad_clip_ckpt"
        enc_ckpt = shared_root / "encoder"
        dec_ckpt = shared_root / "decoder"
        if rank == 0:
            enc_ckpt.mkdir(parents=True, exist_ok=True)
            dec_ckpt.mkdir(parents=True, exist_ok=True)
            save_file(
                {k: v.detach().cpu() for k, v in ref_encoder.state_dict().items()},
                str(enc_ckpt / "model.safetensors"),
            )
            save_file(
                {k: v.detach().cpu() for k, v in ref_decoder.state_dict().items()},
                str(dec_ckpt / "model.safetensors"),
            )
        dist.barrier()

        no_mp = MixedPrecisionConfig(enable=False)

        built = {}
        for name, cls, basic, ckpt, mod_ep in (
            ("encoder", _ToyDenseModule, ["_ToyBlock"], enc_ckpt, 1),
            (
                "decoder",
                dec_cls,
                ["_ToyMoeDecoderLayer"] if ep_size > 1 else ["_ToyBlock"],
                dec_ckpt,
                ep_size,
            ),
        ):
            with use_parallel_state(name):
                mod = cls().to("meta")
                mod = build_parallelize_model(
                    mod,
                    weights_path=str(ckpt),
                    init_device="meta",
                    mixed_precision=no_mp,
                    enable_gradient_checkpointing=False,
                    basic_modules=basic,
                    broadcast_model_weights_from_rank0=False,
                )
                if mod_ep > 1:
                    _ = build_optimizer(mod, lr=LR, weight_decay=0.0, fused=False)
                mod.train()
                built[name] = mod

        encoder, decoder = built["encoder"], built["decoder"]

        if dp_mode == "fsdp2":
            assert all(isinstance(p, DTensor) for p in encoder.parameters())
            assert all(isinstance(p, DTensor) for p in decoder.parameters())
        if ep_size > 1:
            assert get_parallel_state_by_name("encoder") is not get_parallel_state_by_name("decoder")
            assert len(decoder._extra_parallel_param_groups["ep"]) == 1
            assert len(decoder._extra_parallel_param_groups["non_extra_parallel"]) > 0

        # Distinct per-module thresholds: ENC_MAX_NORM != DEC_MAX_NORM != GLOBAL_MAX_NORM
        enc_rt = _bind_module_runtime_handle("encoder", encoder, ENC_MAX_NORM, dp_mode)
        dec_rt = _bind_module_runtime_handle("decoder", decoder, DEC_MAX_NORM, dp_mode)
        runtimes = OrderedDict([("encoder", enc_rt), ("decoder", dec_rt)])

        gen = torch.Generator().manual_seed(9001)
        global_tokens = torch.randn(world_size, 4, HIDDEN, generator=gen).to(device)
        dec_ep_rank = get_parallel_state_by_name("decoder").ep_rank if ep_size > 1 else 0

        # ── Pass 1: per_module scope via veomni_omni_model_clip_grad_norm ──
        _run_forward_backward_pair(
            encoder, decoder, ref_encoder, ref_decoder, global_tokens, rank, world_size, sp_size, ep_size
        )

        ref_enc_pre_grads = {n: p.grad.detach().clone() for n, p in ref_encoder.named_parameters()}
        raw_encoder = encoder.module if hasattr(encoder, "module") else encoder
        if dp_mode == "ddp" and sp_size > 1:
            for n, p in raw_encoder.named_parameters():
                assert not torch.allclose(p.grad, ref_enc_pre_grads[n], rtol=1e-4, atol=1e-6), (
                    f"[rank {rank}] {n} already matched reference before DDP+SP sync"
                )
        elif dp_mode == "fsdp2":
            local_sq = sum(
                (p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad).float().pow(2).sum().item()
                for p in raw_encoder.parameters()
            )
            ref_sq = sum(g.float().pow(2).sum().item() for g in ref_enc_pre_grads.values())
            assert not math.isclose(math.sqrt(local_sq), math.sqrt(ref_sq), rel_tol=1e-3), (
                f"[rank {rank}] local unreduced shard norm unexpectedly matched full reference norm"
            )

        ref_enc_norm = float(torch.nn.utils.clip_grad_norm_(ref_encoder.parameters(), ENC_MAX_NORM).item())
        ref_dec_norm = float(torch.nn.utils.clip_grad_norm_(ref_decoder.parameters(), DEC_MAX_NORM).item())
        assert ref_enc_norm > ENC_MAX_NORM, f"expected encoder clipping to activate: {ref_enc_norm} <= {ENC_MAX_NORM}"
        assert ref_dec_norm > DEC_MAX_NORM, f"expected decoder clipping to activate: {ref_dec_norm} <= {DEC_MAX_NORM}"
        ref_combined_norm = math.sqrt(ref_enc_norm**2 + ref_dec_norm**2)

        # Clear ambient ParallelState before clipping so any missing rt._scoped() fails
        set_parallel_state(None)
        omni_per_module_norm = veomni_omni_model_clip_grad_norm(
            runtimes, max_grad_norm=GLOBAL_MAX_NORM, grad_clip_scope="per_module"
        )
        torch.testing.assert_close(omni_per_module_norm, ref_combined_norm, rtol=1e-4, atol=1e-6)
        _assert_both_modules_grads_match(encoder, decoder, ref_encoder, ref_decoder, ep_size, dec_ep_rank, rank)

        # ── Pass 2: global scope via veomni_omni_model_clip_grad_norm ──
        _run_forward_backward_pair(
            encoder, decoder, ref_encoder, ref_decoder, global_tokens, rank, world_size, sp_size, ep_size
        )

        ref_global_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(ref_encoder.parameters()) + list(ref_decoder.parameters()),
                GLOBAL_MAX_NORM,
            ).item()
        )
        assert ref_global_norm > GLOBAL_MAX_NORM, (
            f"expected global clipping to activate: {ref_global_norm} <= {GLOBAL_MAX_NORM}"
        )
        set_parallel_state(None)
        omni_global_norm = veomni_omni_model_clip_grad_norm(
            runtimes, max_grad_norm=GLOBAL_MAX_NORM, grad_clip_scope="global"
        )
        torch.testing.assert_close(omni_global_norm, ref_global_norm, rtol=1e-4, atol=1e-6)
        _assert_both_modules_grads_match(encoder, decoder, ref_encoder, ref_decoder, ep_size, dec_ep_rank, rank)

        if rank == 0:
            output_dir = os.environ.get("OMNI_FSDP_OUTPUT_DIR", "/tmp/omni_fsdp_test")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dp_mode": dp_mode,
                        "ulysses_size": sp_size,
                        "ep_size": ep_size,
                        "enc_norm": ref_enc_norm,
                        "dec_norm": ref_dec_norm,
                        "combined_norm": omni_per_module_norm,
                        "global_norm": omni_global_norm,
                    },
                    f,
                    indent=2,
                )

        dist.barrier()
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


if __name__ == "__main__":
    main()
