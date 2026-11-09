"""
Four-rank SeedOmni SP metric-meter runner launched via torchrun.

Runs two ``MetricMeterMixin`` modules (``vision_encoder -> llm_backbone``)
through ``TrainNodeRunner`` and ``OmniEnvironMeter`` over a fixed 4-sample-per-step
schedule at a given ``--ulysses-size`` so ``sp=1`` (``dp=4``) and ``sp=4`` (``dp=1``)
can be compared for exact per-module ``trace/<module>/consume_tokens`` equality.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from veomni.distributed.parallel_state import (
    _init_parallel_state,
    clear_parallel_state,
    get_parallel_state,
    get_parallel_state_by_name,
    use_parallel_state,
)
from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor, sp_pad
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
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.omni_helper import OmniEnvironMeter


os.environ.setdefault("NCCL_DEBUG", "OFF")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

HIDDEN = 64


class _MeteredSPModule(MetricMeterMixin, TrainingModuleMixin, BaseMixin, nn.Module):
    """Test-only OmniModule that stashes pre-slice seqlens and slices/gathers under SP."""

    def __init__(self, hidden_size: int, flops_per_token: float, *, emit_loss: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.flops_per_token = flops_per_token
        self.emit_loss = emit_loss

    def estimate_flops(self, seqlens: list[int]) -> float:
        """Return theoretical TFLOPs proportional to total token count."""
        return float(sum(seqlens)) * self.flops_per_token / 1e12

    @pre_forward("forward")
    def forward_pre(self, hidden: torch.Tensor, seqlens: list[int], **kwargs) -> dict:
        """Stash full-sample seqlens before slicing the sequence across the SP group."""
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
        """Project the local sequence shard and verify its SP-sliced length."""
        del kwargs
        ps = get_parallel_state()
        expected_local = math.ceil(unpad_size / ps.sp_size)
        assert hidden.size(1) == expected_local, (
            f"expected SP-sliced length {expected_local} under sp_size={ps.sp_size}, got {hidden.size(1)}"
        )
        return {"hidden": self.proj(hidden), "unpad_size": unpad_size}

    @post_forward("forward")
    def forward_post(self, hidden: torch.Tensor, unpad_size: int, **kwargs) -> dict:
        """All-gather the sequence back to full length and optionally emit a scalar loss."""
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


def main() -> None:
    """Execute two global steps of the metered graph and write step metrics on rank 0."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses-size", type=int, required=True)
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
        sp_size = cli_args.ulysses_size
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
                sample = step_samples[dp_rank * micro_batches_per_step + micro_idx]
                environ_meter.add({"conversation_list": [sample]})

                runner = TrainNodeRunner(scope_fn=lambda name: use_parallel_state(get_parallel_state_by_name(name)))
                batch_vision = {
                    "hidden": torch.ones(1, sum(sample["vision"]), HIDDEN, device=device),
                    "seqlens": sample["vision"],
                }
                runner(modules["vision_encoder"], nodes[0], batch_vision)

                batch_llm = {
                    "hidden": torch.ones(1, sum(sample["llm"]), HIDDEN, device=device),
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
            output_dir = os.environ.get("OMNI_FSDP_OUTPUT_DIR", "/tmp/omni_fsdp_test")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump({"ulysses_size": sp_size, "steps": step_records}, f, indent=2)

        dist.barrier()
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


if __name__ == "__main__":
    main()
