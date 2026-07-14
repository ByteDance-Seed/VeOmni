# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Non-PEFT MoE weight-loading matrix on a self-contained fake EP model.

Verifies that a MoE model (EP ``Shard(0)`` experts) loads **bit-identically**
through all three weight loaders

    * ``load_model_weights``            (every-rank-reads)
    * ``rank0_load_and_broadcast_weights``
    * ``load_model_weights_ep_sharded`` (per-rank ExtraParallel-slice stream)

for both checkpoint expert layouts

    * **merged**     -- experts already fused ``[E, out, in]`` (one key, no converter fires)
    * **non-merged** -- one ``[out, in]`` tensor per expert key, needs a
      fusion ``CheckpointTensorConverter`` (models the HF per-expert ->
      fused-veomni case, e.g. Qwen3-MoE).

Assertions:
    * merged   x {plain, rank0, ep_sharded} -> gathered experts == reference.
    * nonmerge x {plain, rank0}             -> converter fuses -> == reference.
    * nonmerge x ep_sharded                 -> raises ``NotImplementedError``
      up front (a fusion converter can't be streamed per-rank; the loader
      must bail so the caller falls back to the whole-tensor loader). This
      is the silent-corruption guard for Qwen3-MoE.

Run directly (2 GPUs):
    torchrun --nproc_per_node=2 --master_port=4331 \
        tests/utils/test_moe_ep_sharded_load_matrix.py
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Shard

from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.checkpoint_tensor_loading import ConvertedCheckpointTensor
from veomni.utils import helper
from veomni.utils.device import get_dist_comm_backend, get_torch_device


logger = helper.create_logger(__name__)

# ── model dims (tiny; E divisible by ep=2) ────────────────────────────
E, OUT, IN = 8, 6, 4
DENSE_OUT, DENSE_IN = 3, 4
VOCAB, EMB = 10, 4


# ──────────────────────────────────────────────────────────────────────
# Per-expert -> fused converter (models Qwen3MoeCheckpointTensorConverter)
# ──────────────────────────────────────────────────────────────────────
class _PerExpertFuseConverter:
    """Accumulate per-expert ``moe.experts.<e>`` rows, emit fused ``moe.experts``.

    ``is_dim0_zero_pad`` is intentionally NOT a pure dim-0 zero-pad (it stacks
    E separate tensors into one), so the streaming loader must bail on it.
    """

    _PREFIX = "moe.experts."

    def __init__(self) -> None:
        self._buf: dict[int, torch.Tensor] = {}

    def can_handle(self, name: str) -> bool:
        if not name.startswith(self._PREFIX):
            return False
        return name[len(self._PREFIX) :].isdigit()

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        idx = int(name[len(self._PREFIX) :])
        self._buf[idx] = tensor.clone()
        if len(self._buf) < E:
            return None
        return self._flush()

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        return [self._flush()] if self._buf else []

    def _flush(self) -> ConvertedCheckpointTensor:
        fused = torch.stack([self._buf[i] for i in sorted(self._buf)], dim=0)
        self._buf.clear()
        return ConvertedCheckpointTensor(name="moe.experts", tensor=fused)

    def is_dim0_zero_pad(self, name: str) -> bool:
        return False  # fusion, not a pure zero-pad -> not streamable


# ──────────────────────────────────────────────────────────────────────
# Fake MoE model with an EP Shard(0) experts param
# ──────────────────────────────────────────────────────────────────────
class _FakeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.Parameter(torch.empty(E, OUT, IN))


class FakeMoeModel(nn.Module):
    _no_split_modules = ["_FakeExperts"]

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Parameter(torch.empty(VOCAB, EMB))
        self.linear = nn.Linear(DENSE_IN, DENSE_OUT, bias=True)
        self.moe = _FakeExperts()
        # tie_word_embeddings False -> skip the post-load embedding tie path.
        self.config = SimpleNamespace(tie_word_embeddings=False)

    # every-rank fresh init (build_parallelize_model calls this when no weights)
    def init_weights(self) -> None:
        with torch.no_grad():
            self.embed.normal_()
            nn.init.xavier_uniform_(self.linear.weight)
            self.linear.bias.fill_(0.0)
            nn.init.xavier_uniform_(self.moe.experts)

    def get_parallel_plan(self) -> ParallelPlan:
        plan = ParallelPlan(extra_parallel_plan={"ep": {"moe.experts": Shard(0)}})
        plan.extra_parallel_fsdp_no_shard_module = {"ep": {"moe"}}
        return plan

    # A fusion converter is always registered; it only *fires* on the
    # per-expert checkpoint (fused keys don't match ``can_handle``).
    @staticmethod
    def _create_checkpoint_tensor_converter(model):
        return _PerExpertFuseConverter()


# ──────────────────────────────────────────────────────────────────────
# Reference weights + checkpoint writers
# ──────────────────────────────────────────────────────────────────────
def _reference_state() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(1234)
    return {
        "embed": torch.randn(VOCAB, EMB, generator=g),
        "linear.weight": torch.randn(DENSE_OUT, DENSE_IN, generator=g),
        "linear.bias": torch.randn(DENSE_OUT, generator=g),
        "moe.experts": torch.randn(E, OUT, IN, generator=g),
    }


def _write_merged(ckpt_dir: Path, ref: dict[str, torch.Tensor]) -> str:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.clone() for k, v in ref.items()}, str(ckpt_dir / "model.safetensors"))
    return str(ckpt_dir)


def _write_per_expert(ckpt_dir: Path, ref: dict[str, torch.Tensor]) -> str:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.clone() for k, v in ref.items() if k != "moe.experts"}
    for e in range(E):
        state[f"moe.experts.{e}"] = ref["moe.experts"][e].clone()
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return str(ckpt_dir)


# ──────────────────────────────────────────────────────────────────────
# Gather helper (DTensor -> full replicated CPU tensor)
# ──────────────────────────────────────────────────────────────────────
def _full_cpu(t: torch.Tensor) -> torch.Tensor:
    """Gather an FSDP DTensor (dense param) to its full replicated tensor."""
    if isinstance(t, DTensor):
        t = t.full_tensor()  # gather every mesh dim -> full replicated tensor
    return t.detach().float().cpu()


def _local_expert_shard(t: torch.Tensor) -> torch.Tensor:
    """This rank's local ``[E/ep, ...]`` expert slice (EP params are plain local
    tensors when ``ep_fsdp==1``, else DTensors sharded on the ep mesh)."""
    if isinstance(t, DTensor):
        t = t.to_local()
    return t.detach().float().cpu()


# ──────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────
def _build_fresh(weights_path, *, loader: str):
    """Build + parallelize a fresh FakeMoeModel, loading via the chosen loader."""
    kwargs = dict(
        weights_path=weights_path,
        init_device="meta",
        mixed_precision=SimpleNamespace(enable=False),
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=False,
        basic_modules=[],
    )
    if loader == "rank0":
        kwargs["broadcast_model_weights_from_rank0"] = True
    elif loader == "ep_sharded":
        kwargs["ep_sharded_stream_load"] = True
    # loader == "plain": defaults (every-rank-reads)
    return build_parallelize_model(FakeMoeModel(), **kwargs)


def run_worker() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    get_torch_device().set_device(rank)
    import torch.distributed as dist

    dist.init_process_group(backend=get_dist_comm_backend())
    # world == ep: experts sharded across all ranks, dense FSDP-sharded.
    init_parallel_state(dp_size=world, extra_parallel_sizes=(world,), extra_parallel_names=("ep",), dp_mode="fsdp2")

    tmp = Path(os.environ["MOE_MATRIX_TMP"])
    ref = _reference_state()
    merged_dir = str(tmp / "merged")
    per_expert_dir = str(tmp / "per_expert")
    if rank == 0:
        _write_merged(tmp / "merged", ref)
        _write_per_expert(tmp / "per_expert", ref)
    dist.barrier()

    ps = get_parallel_state()
    ep_rank = ps.extra_parallel_rank("ep")
    ep_size = ps.extra_parallel_sizes["ep"]
    per = E // ep_size

    results = []

    def check(model, tag):
        params = dict(model.named_parameters())
        # EP-sharded experts: compare this rank's local [E/ep, ...] slice.
        got_experts = _local_expert_shard(params["moe.experts"])
        want_experts = ref["moe.experts"][ep_rank * per : (ep_rank + 1) * per]
        torch.testing.assert_close(got_experts, want_experts, atol=0.0, rtol=0.0)
        # Dense/FSDP params: gather full and compare.
        for name in ("linear.weight", "linear.bias", "embed"):
            got = _full_cpu(params[name])
            torch.testing.assert_close(got, ref[name], atol=0.0, rtol=0.0)
        results.append(f"  [OK] {tag}: experts slice + dense params bit-identical to reference")

    for fmt, path in [("merged", merged_dir), ("nonmerged", per_expert_dir)]:
        for loader in ["plain", "rank0", "ep_sharded"]:
            tag = f"{fmt} x {loader}"
            expect_bail = fmt == "nonmerged" and loader == "ep_sharded"
            try:
                model = _build_fresh(path, loader=loader)
            except NotImplementedError as e:
                if expect_bail:
                    results.append(f"  [OK] {tag}: bailed as expected ({str(e)[:60]}...)")
                else:
                    results.append(f"  [FAIL] {tag}: unexpected NotImplementedError: {e}")
                dist.barrier()
                continue
            if expect_bail:
                results.append(f"  [FAIL] {tag}: expected NotImplementedError bail, but load succeeded")
                dist.barrier()
                continue
            check(model, tag)
            del model
            dist.barrier()

    if rank == 0:
        print(f"\n=== NON-PEFT MoE load matrix (world={world}, ep={world}) ===")
        print("\n".join(results))
    failed = [r for r in results if "[FAIL]" in r]
    dist.barrier()
    dist.destroy_process_group()
    if failed:
        raise SystemExit(f"{len(failed)} case(s) FAILED:\n" + "\n".join(failed))


# ──────────────────────────────────────────────────────────────────────
# pytest wrapper
# ──────────────────────────────────────────────────────────────────────
# Device-agnostic gate: the worker itself only uses ``get_torch_device`` /
# ``get_dist_comm_backend`` (no CUDA-only ops, no Triton), so it runs on any
# >=2-device accelerator (CUDA / Ascend NPU) via the matching comm backend.
_ACCEL = get_torch_device()


@pytest.mark.skipif(
    not _ACCEL.is_available() or _ACCEL.device_count() < 2,
    reason="requires >= 2 accelerator devices (CUDA or NPU)",
)
def test_moe_ep_sharded_load_matrix(tmp_path):
    port = 12345 + random.randint(0, 500)
    env = dict(os.environ, MOE_MATRIX_TMP=str(tmp_path))
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        f"--master_port={port}",
        os.path.abspath(__file__),
    ]
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    # Deterministic *shared* fixture dir so every rank resolves the same
    # checkpoint path (rank0 writes it; a per-rank mkdtemp would leave the
    # other ranks with an empty dir -> HF repo-id fallback).
    os.environ.setdefault("MOE_MATRIX_TMP", "/tmp/moe_ep_matrix_fixture")
    run_worker()
    print("ALL CASES PASSED", file=sys.stderr)
