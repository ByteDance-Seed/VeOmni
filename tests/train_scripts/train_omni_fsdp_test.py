"""
SeedOmni multi-module runtime acceptance script launched via torchrun.

Modes:
  1. ``--mode=pipeline`` (default):
     Two-model FSDP2 / DDP forward-backward-step pipeline smoke test.
  2. ``--mode=sp_consume_tokens --ulysses-size={1,4}``:
     Runs two ``MetricMeterMixin`` modules (``vision_encoder -> llm_backbone``)
     through ``TrainNodeRunner`` and ``OmniEnvironMeter`` over the same 4-sample
     global batches at ``(sp=1, dp=4)`` vs ``(sp=4, dp=1)`` so the caller can
     assert exact per-module ``trace/<module>/consume_tokens`` equality.
  3. ``--mode=grad_clip --dp-mode={ddp,fsdp2} --ulysses-size={1,2,4} --ep-size={1,2}``:
     Wraps two Omni modules under their own named ``ParallelState``s and
     verifies that ``ModuleRuntime.clip_grad_norm`` /
     ``veomni_omni_model_clip_grad_norm`` (both ``per_module`` and ``global``
     scopes) numerically match an unsharded single-model reference on the same
     weights and global batch across DDP(+SP), FSDP2, and FSDP2+EP.
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
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel

from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.clip_grad_norm import (
    veomni_clip_grad_norm,
    veomni_omni_model_clip_grad_norm,
)
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import (
    _init_parallel_state,
    clear_parallel_state,
    get_parallel_state,
    get_parallel_state_by_name,
    use_parallel_state,
)
from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor, sp_pad
from veomni.distributed.sequence_parallel.loss import reduce_sequence_parallel_loss
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime
from veomni.models.seed_omni.accelerated.utils.executor import TrainNodeRunner
from veomni.models.seed_omni.graphs.base import NodeDef
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.mixins.metric_meter_mixin import MetricMeterMixin
from veomni.models.seed_omni.mixins.training_module_mixin import (
    TrainingModuleMixin,
    post_forward,
    pre_forward,
)
from veomni.models.seed_omni.modeling_omni import LOSS_KEY
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.omni_helper import OmniEnvironMeter


os.environ.setdefault("NCCL_DEBUG", "OFF")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

HIDDEN = 64
SEQ_LEN = 32
VOCAB = 512
BATCH = 2
MAX_STEPS = 3
LR = 1e-3
CLIP_MAX_NORM = 0.5


def _tiny_llama_config(**extra) -> LlamaConfig:
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
    def __init__(self, encoder: nn.Module, llm: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.llm = llm


def _write_results(payload: dict) -> None:
    output_dir = os.environ.get("OMNI_FSDP_OUTPUT_DIR", "/tmp/omni_fsdp_test")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_pipeline_mode(device: torch.device, world_size: int, rank: int) -> None:
    fsdp_enabled = world_size > 1
    dp_mode = "fsdp2" if fsdp_enabled else "ddp"
    init_device = "meta" if fsdp_enabled else get_device_type()

    _init_parallel_state(
        dp_size=world_size,
        dp_mode=dp_mode,
    )

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
    for _step in range(MAX_STEPS):
        optimizer.zero_grad()

        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
        labels = input_ids.clone()

        enc_out = encoder(input_ids=input_ids)
        hidden = enc_out.last_hidden_state
        llm_out = llm(inputs_embeds=hidden, labels=labels)
        loss = llm_out.loss
        loss.backward()

        grad_norm = veomni_clip_grad_norm(combined, max_norm=1.0)
        assert math.isfinite(grad_norm) and grad_norm > 0

        def _model_grad_norm(model: nn.Module) -> float:
            total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                    total += g.detach().float().norm().item() ** 2
            return math.sqrt(total)

        gnorm_enc = _model_grad_norm(encoder)
        gnorm_llm = _model_grad_norm(llm)

        optimizer.step()
        lr_scheduler.step()

        cur_lr = lr_scheduler.get_last_lr()[0]
        loss_val = loss.item()

        if rank == 0:
            results["loss"].append(loss_val)
            results["grad_norm_encoder"].append(gnorm_enc)
            results["grad_norm_llm"].append(gnorm_llm)
            results["optimizer_state_ok"].append(len(optimizer.state) > 0)
            results["lr"].append(cur_lr)

    if rank == 0:
        for i, (gne, gnl) in enumerate(zip(results["grad_norm_encoder"], results["grad_norm_llm"])):
            assert math.isfinite(gne) and gne > 0, f"step {i}: encoder grad_norm={gne}"
            assert math.isfinite(gnl) and gnl > 0, f"step {i}: llm grad_norm={gnl}"
        assert len(optimizer.state) > 0, "optimizer state is empty"
        _write_results(results)


class _MeteredSPModule(MetricMeterMixin, TrainingModuleMixin, BaseMixin, nn.Module):
    """Test-only Omni module that stashes pre-slice seqlens and slices/gathers under SP."""

    def __init__(self, hidden_size: int, flops_per_token: float, *, emit_loss: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.flops_per_token = flops_per_token
        self.emit_loss = emit_loss

    def estimate_flops(self, seqlens: list[int]) -> float:
        return float(sum(seqlens)) * self.flops_per_token / 1e12

    @pre_forward("forward")
    def forward_pre(
        self,
        hidden: torch.Tensor,
        seqlens: list[int],
        **kwargs,
    ) -> dict:
        del kwargs
        unpad_size = int(hidden.size(1))
        assert unpad_size == sum(seqlens), f"expected {sum(seqlens)} packed tokens, got {unpad_size}"

        self.metric_meter_set_seqlens("forward", seqlens)

        ps = get_parallel_state()
        if ps.sp_size > 1:
            padded = sp_pad(hidden, dim=1, pad_value=0)
            hidden = slice_input_tensor(padded, dim=1, padding=False, group=ps.sp_group)

        return {"hidden": hidden, "unpad_size": unpad_size}

    def forward(self, hidden: torch.Tensor, unpad_size: int, **kwargs) -> dict:
        del kwargs
        ps = get_parallel_state()
        expected_local = math.ceil(unpad_size / ps.sp_size)
        assert hidden.size(1) == expected_local, (
            f"expected SP-sliced length {expected_local} under sp_size={ps.sp_size}, got {hidden.size(1)}"
        )
        return {"hidden": self.proj(hidden), "unpad_size": unpad_size}

    @post_forward("forward")
    def forward_post(self, hidden: torch.Tensor, unpad_size: int, **kwargs) -> dict:
        del kwargs
        ps = get_parallel_state()
        if ps.sp_size > 1:
            hidden = gather_outputs(
                hidden,
                gather_dim=1,
                padding_dim=1,
                unpad_dim_size=unpad_size,
                scale_grad=False,
                group=ps.sp_group,
            )
        assert hidden.size(1) == unpad_size
        out = {"hidden": hidden}
        if self.emit_loss:
            out[LOSS_KEY] = hidden.float().square().mean()
        return out


def _run_sp_consume_tokens_mode(
    device: torch.device,
    world_size: int,
    rank: int,
    sp_size: int,
) -> None:
    assert world_size == 4, f"sp_consume_tokens requires world_size=4, got {world_size}"
    assert world_size % sp_size == 0
    dp_size = world_size // sp_size

    base_ps = _init_parallel_state(dp_size=dp_size, ulysses_size=sp_size, dp_mode="fsdp2", name="base")
    _init_parallel_state(dp_size=dp_size, ulysses_size=sp_size, dp_mode="fsdp2", name="vision_encoder")
    _init_parallel_state(dp_size=dp_size, ulysses_size=sp_size, dp_mode="fsdp2", name="llm_backbone")

    torch.manual_seed(17)
    vision_mod = _MeteredSPModule(HIDDEN, flops_per_token=2e6, emit_loss=False).to(device)
    llm_mod = _MeteredSPModule(HIDDEN, flops_per_token=6e6, emit_loss=True).to(device)

    global_steps = [
        [
            {"vision": [24, 40], "llm": [52, 60]},
            {"vision": [32, 64], "llm": [72, 72]},
            {"vision": [48, 80], "llm": [88, 88]},
            {"vision": [60, 100], "llm": [104, 104]},
        ],
        [
            {"vision": [20, 28], "llm": [44, 52]},
            {"vision": [36, 44], "llm": [64, 64]},
            {"vision": [52, 60], "llm": [80, 96]},
            {"vision": [68, 76], "llm": [112, 128]},
        ],
    ]

    environ_meter = OmniEnvironMeter(
        global_batch_size=4,
        enable_multisource=False,
        parallel_state=base_ps,
    )
    nodes = [
        NodeDef(name="vision_encoder", module="vision_encoder", method="forward"),
        NodeDef(name="llm_backbone", module="llm_backbone", method="forward"),
    ]
    modules = {"vision_encoder": vision_mod, "llm_backbone": llm_mod}

    step_records = []
    dp_rank = base_ps.dp_rank
    micro_batches_per_step = 4 // dp_size

    for step_idx, step_samples in enumerate(global_steps, start=1):
        for micro_idx in range(micro_batches_per_step):
            sample_idx = dp_rank * micro_batches_per_step + micro_idx
            sample = step_samples[sample_idx]
            micro_batch = {"conversation_list": [sample]}
            environ_meter.add(micro_batch)

            runner = TrainNodeRunner(scope_fn=lambda name: use_parallel_state(get_parallel_state_by_name(name)))
            vision_tokens = sum(sample["vision"])
            llm_tokens = sum(sample["llm"])

            batch_vision = {
                "hidden": torch.ones(1, vision_tokens, HIDDEN, device=device),
                "seqlens": sample["vision"],
            }
            runner(modules["vision_encoder"], nodes[0], batch_vision)

            batch_llm = {
                "hidden": torch.ones(1, llm_tokens, HIDDEN, device=device),
                "seqlens": sample["llm"],
            }
            runner(modules["llm_backbone"], nodes[1], batch_llm)
            batch_llm[LOSS_KEY].backward()

        module_metrics = {name: mod.metric_meter_collect() for name, mod in modules.items()}
        metrics = environ_meter.step(delta_time=1.0, global_step=step_idx, module_metrics=module_metrics)
        step_records.append(
            {
                "step": step_idx,
                "consumed_chunk_num": int(metrics["consumed_chunk_num"]),
                "consume_tokens": {k: int(v) for k, v in environ_meter.consume_tokens.items()},
                "trace/vision_encoder/consume_tokens(M)": float(metrics["trace/vision_encoder/consume_tokens(M)"]),
                "trace/llm_backbone/consume_tokens(M)": float(metrics["trace/llm_backbone/consume_tokens(M)"]),
                "trace/vision_encoder/avg_seq_len": float(metrics["trace/vision_encoder/avg_seq_len"]),
                "trace/llm_backbone/avg_seq_len": float(metrics["trace/llm_backbone/avg_seq_len"]),
            }
        )

    if rank == 0:
        _write_results({"ulysses_size": sp_size, "steps": step_records})


class _ToyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


class _ToyDenseModule(nn.Module):
    _no_split_modules = ["_ToyBlock"]

    def __init__(self, dim: int = HIDDEN) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.block1 = _ToyBlock(dim)
        self.block2 = _ToyBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


class _ToyMoeExperts(nn.Module):
    def __init__(self, num_experts: int = 8, dim: int = HIDDEN) -> None:
        super().__init__()
        self.experts = nn.Parameter(torch.empty(num_experts, dim, dim))

    def forward(self, x: torch.Tensor, expert_slice: slice | None = None) -> torch.Tensor:
        weights = self.experts if expert_slice is None else self.experts[expert_slice]
        return torch.einsum("...d,edh->...eh", x, weights).mean(dim=-2)


class _ToyMoeDecoderLayer(nn.Module):
    def __init__(self, num_experts: int = 8, dim: int = HIDDEN) -> None:
        super().__init__()
        self.block = _ToyBlock(dim)
        self.moe = _ToyMoeExperts(num_experts=num_experts, dim=dim)

    def forward(self, x: torch.Tensor, expert_slice: slice | None = None) -> torch.Tensor:
        return self.moe(self.block(x), expert_slice=expert_slice)


class _ToyMoeModule(nn.Module):
    _no_split_modules = ["_ToyMoeDecoderLayer"]

    def __init__(self, num_experts: int = 8, dim: int = HIDDEN) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.decoder = _ToyMoeDecoderLayer(num_experts=num_experts, dim=dim)

    def forward(self, x: torch.Tensor, expert_slice: slice | None = None) -> torch.Tensor:
        return self.decoder(x, expert_slice=expert_slice)

    def get_parallel_plan(self) -> ParallelPlan:
        plan = ParallelPlan(
            extra_parallel_plan={
                "ep": {"decoder.moe.experts": Shard(0)},
            }
        )
        plan.extra_parallel_fsdp_no_shard_module = {"ep": {"decoder.moe"}}
        return plan


def _init_deterministic_module(cls: type[nn.Module], seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    mod = cls().to(device)
    for p in mod.parameters():
        nn.init.normal_(p, mean=0.0, std=0.08)
    return mod


def _bind_module_runtime_handle(name: str, model: nn.Module, max_grad_norm: float, dp_mode: str) -> ModuleRuntime:
    rt = object.__new__(ModuleRuntime)
    rt.model_name = name
    rt.model = model
    rt.args = SimpleNamespace(
        optimizer=SimpleNamespace(max_grad_norm=max_grad_norm),
        accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode=dp_mode)),
    )
    return rt


def _full_grad_for_comparison(
    name: str,
    param: nn.Parameter,
    ref_grad: torch.Tensor,
    ep_size: int,
    ep_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad = param.grad
    assert grad is not None, f"missing grad for {name}"
    full_grad = grad.full_tensor() if isinstance(grad, DTensor) else grad
    if ep_size > 1 and "moe.experts" in name:
        ref_slice = ref_grad.chunk(ep_size, dim=0)[ep_rank]
        return full_grad, ref_slice
    return full_grad, ref_grad


def _run_grad_clip_mode(
    device: torch.device,
    world_size: int,
    rank: int,
    dp_mode: str,
    sp_size: int,
    ep_size: int,
) -> None:
    dp_size = world_size // sp_size
    extra_sizes = (ep_size,)
    extra_names = ("ep",)
    extra_innermost = (False,)

    for state_name in ("base", "encoder", "decoder"):
        _init_parallel_state(
            dp_size=dp_size,
            dp_shard_size=dp_size,
            ulysses_size=sp_size,
            dp_mode=dp_mode,
            extra_parallel_sizes=extra_sizes,
            extra_parallel_placement_innermost=extra_innermost,
            extra_parallel_names=extra_names,
            name=state_name,
        )

    module_cls = _ToyMoeModule if ep_size > 1 else _ToyDenseModule
    ref_encoder = _init_deterministic_module(module_cls, seed=101, device=device)
    ref_decoder = _init_deterministic_module(module_cls, seed=202, device=device)
    ref_encoder.train()
    ref_decoder.train()

    shared_root = Path(os.environ.get("OMNI_FSDP_OUTPUT_DIR", tempfile.gettempdir())) / "grad_clip_ckpt"
    enc_ckpt = shared_root / "encoder"
    dec_ckpt = shared_root / "decoder"
    if rank == 0:
        enc_ckpt.mkdir(parents=True, exist_ok=True)
        dec_ckpt.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu() for k, v in ref_encoder.state_dict().items()}, str(enc_ckpt / "model.safetensors")
        )
        save_file(
            {k: v.detach().cpu() for k, v in ref_decoder.state_dict().items()}, str(dec_ckpt / "model.safetensors")
        )
    dist.barrier()

    no_mp = MixedPrecisionConfig(enable=False)

    with use_parallel_state("encoder"):
        encoder = module_cls().to("meta")
        encoder = build_parallelize_model(
            encoder,
            weights_path=str(enc_ckpt),
            init_device="meta",
            mixed_precision=no_mp,
            enable_gradient_checkpointing=False,
            basic_modules=["_ToyBlock"],
            broadcast_model_weights_from_rank0=False,
        )
        if ep_size > 1:
            _ = build_optimizer(encoder, lr=LR, weight_decay=0.0, fused=False)
    with use_parallel_state("decoder"):
        decoder = module_cls().to("meta")
        decoder = build_parallelize_model(
            decoder,
            weights_path=str(dec_ckpt),
            init_device="meta",
            mixed_precision=no_mp,
            enable_gradient_checkpointing=False,
            basic_modules=["_ToyBlock"],
            broadcast_model_weights_from_rank0=False,
        )
        if ep_size > 1:
            _ = build_optimizer(decoder, lr=LR, weight_decay=0.0, fused=False)

    encoder.train()
    decoder.train()

    if dp_mode == "fsdp2":
        assert all(isinstance(p, DTensor) for p in encoder.parameters())
        assert all(isinstance(p, DTensor) for p in decoder.parameters())
    if ep_size > 1:
        assert len(encoder._extra_parallel_param_groups["ep"]) == 1
        assert len(encoder._extra_parallel_param_groups["non_extra_parallel"]) > 0

    enc_rt = _bind_module_runtime_handle("encoder", encoder, CLIP_MAX_NORM, dp_mode)
    dec_rt = _bind_module_runtime_handle("decoder", decoder, CLIP_MAX_NORM, dp_mode)
    runtimes = OrderedDict([("encoder", enc_rt), ("decoder", dec_rt)])

    gen = torch.Generator().manual_seed(9001)
    tokens_per_rank = 4
    global_tokens = torch.randn(world_size, tokens_per_rank, HIDDEN, generator=gen).to(device)
    local_tokens = global_tokens[rank]

    ps_enc = get_parallel_state_by_name("encoder")
    ps_dec = get_parallel_state_by_name("decoder")
    ep_rank = ps_enc.ep_rank if ep_size > 1 else 0

    with use_parallel_state("encoder"):
        hidden = encoder(local_tokens)
    with use_parallel_state("decoder"):
        out = decoder(hidden)
        loss = out.square().sum(dim=-1).mean()
        if sp_size > 1:
            loss = reduce_sequence_parallel_loss(
                loss,
                torch.tensor(float(tokens_per_rank), device=device),
                group=ps_dec.sp_group,
            )
    loss.backward()

    ref_losses = []
    num_experts = 8
    experts_per_ep = num_experts // ep_size
    for r in range(world_size):
        if ep_size > 1:
            r_ep_rank = r % ep_size
            expert_slice = slice(r_ep_rank * experts_per_ep, (r_ep_rank + 1) * experts_per_ep)
            ref_h = ref_encoder(global_tokens[r], expert_slice=expert_slice)
            ref_o = ref_decoder(ref_h, expert_slice=expert_slice)
        else:
            ref_h = ref_encoder(global_tokens[r])
            ref_o = ref_decoder(ref_h)
        ref_losses.append(ref_o.square().sum(dim=-1).mean())
    ref_loss = torch.stack(ref_losses).mean()
    ref_loss.backward()

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

    ref_enc_norm = float(torch.nn.utils.clip_grad_norm_(ref_encoder.parameters(), CLIP_MAX_NORM).item())
    ref_dec_norm = float(torch.nn.utils.clip_grad_norm_(ref_decoder.parameters(), CLIP_MAX_NORM).item())
    ref_combined_norm = math.sqrt(ref_enc_norm**2 + ref_dec_norm**2)

    enc_norm = enc_rt.clip_grad_norm(CLIP_MAX_NORM)
    dec_norm = dec_rt.clip_grad_norm(CLIP_MAX_NORM)
    combined_norm = math.sqrt(enc_norm**2 + dec_norm**2)

    torch.testing.assert_close(enc_norm, ref_enc_norm, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(dec_norm, ref_dec_norm, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(combined_norm, ref_combined_norm, rtol=1e-4, atol=1e-6)

    for (name, param), (_, ref_param) in zip(
        raw_encoder.named_parameters(), ref_encoder.named_parameters(), strict=True
    ):
        actual_g, expected_g = _full_grad_for_comparison(name, param, ref_param.grad, ep_size, ep_rank)
        torch.testing.assert_close(
            actual_g,
            expected_g,
            rtol=1e-4,
            atol=1e-6,
            msg=lambda m, name=name: f"[rank {rank}] encoder.{name} grad mismatch: {m}",
        )

    raw_decoder = decoder.module if hasattr(decoder, "module") else decoder
    for (name, param), (_, ref_param) in zip(
        raw_decoder.named_parameters(), ref_decoder.named_parameters(), strict=True
    ):
        actual_g, expected_g = _full_grad_for_comparison(name, param, ref_param.grad, ep_size, ep_rank)
        torch.testing.assert_close(
            actual_g,
            expected_g,
            rtol=1e-4,
            atol=1e-6,
            msg=lambda m, name=name: f"[rank {rank}] decoder.{name} grad mismatch: {m}",
        )

    encoder.zero_grad()
    decoder.zero_grad()
    ref_encoder.zero_grad()
    ref_decoder.zero_grad()

    with use_parallel_state("encoder"):
        hidden2 = encoder(local_tokens)
    with use_parallel_state("decoder"):
        out2 = decoder(hidden2)
        loss2 = out2.square().sum(dim=-1).mean()
        if sp_size > 1:
            loss2 = reduce_sequence_parallel_loss(
                loss2,
                torch.tensor(float(tokens_per_rank), device=device),
                group=ps_dec.sp_group,
            )
    loss2.backward()

    ref_losses2 = []
    for r in range(world_size):
        if ep_size > 1:
            r_ep_rank = r % ep_size
            expert_slice = slice(r_ep_rank * experts_per_ep, (r_ep_rank + 1) * experts_per_ep)
            ref_h2 = ref_encoder(global_tokens[r], expert_slice=expert_slice)
            ref_o2 = ref_decoder(ref_h2, expert_slice=expert_slice)
        else:
            ref_h2 = ref_encoder(global_tokens[r])
            ref_o2 = ref_decoder(ref_h2)
        ref_losses2.append(ref_o2.square().sum(dim=-1).mean())
    torch.stack(ref_losses2).mean().backward()

    ref_global_norm = float(
        torch.nn.utils.clip_grad_norm_(
            list(ref_encoder.parameters()) + list(ref_decoder.parameters()),
            CLIP_MAX_NORM,
        ).item()
    )
    omni_global_norm = veomni_omni_model_clip_grad_norm(
        runtimes, max_grad_norm=CLIP_MAX_NORM, grad_clip_scope="global"
    )
    torch.testing.assert_close(omni_global_norm, ref_global_norm, rtol=1e-4, atol=1e-6)

    for (name, param), (_, ref_param) in zip(
        raw_encoder.named_parameters(), ref_encoder.named_parameters(), strict=True
    ):
        actual_g, expected_g = _full_grad_for_comparison(name, param, ref_param.grad, ep_size, ep_rank)
        torch.testing.assert_close(actual_g, expected_g, rtol=1e-4, atol=1e-6)

    if rank == 0:
        _write_results(
            {
                "dp_mode": dp_mode,
                "ulysses_size": sp_size,
                "ep_size": ep_size,
                "enc_norm": enc_norm,
                "dec_norm": dec_norm,
                "combined_norm": combined_norm,
                "global_norm": omni_global_norm,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("pipeline", "sp_consume_tokens", "grad_clip"),
        default="pipeline",
    )
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
        if cli_args.mode == "pipeline":
            _run_pipeline_mode(device, world_size, rank)
        elif cli_args.mode == "sp_consume_tokens":
            _run_sp_consume_tokens_mode(device, world_size, rank, cli_args.ulysses_size)
        elif cli_args.mode == "grad_clip":
            _run_grad_clip_mode(
                device,
                world_size,
                rank,
                dp_mode=cli_args.dp_mode,
                sp_size=cli_args.ulysses_size,
                ep_size=cli_args.ep_size,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


if __name__ == "__main__":
    main()
