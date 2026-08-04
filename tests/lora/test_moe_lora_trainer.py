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
r"""End-to-end trainer-driven save/load/resume test for MoE-LoRA on Qwen3-MoE.

Drives ``BaseTrainer`` with the **real** :class:`~veomni.trainer.callbacks.checkpoint_callback.CheckpointerCallback`
+ :class:`~veomni.trainer.callbacks.checkpoint_callback.HFLoraCkptCallback`
so the writer subprocess produces both checkpoint formats production runs
emit, then validates both resume paths bit-exact (modulo bf16 storage):

    1. Writer subprocess
       - DCP shards under ``<output_dir>/checkpoints/global_step_<S>/``
         (model + optimizer + extra_state -- the format ``BaseTrainer``
         resumes via ``train.checkpoint.load_path``).
       - HF-format LoRA adapter under ``<output_dir>/global_step_<S>/``
         (``adapter_model.safetensors`` + ``adapter_config.json``; the MoE mode +
         rank/alpha VeOmni's wrappers need to re-install themselves on resume
         live in the ``veomni_lora`` block of ``adapter_config.json``) -- the
         format ``BaseTrainer`` resumes via ``model.lora_config.lora_adapter``.
       - Snapshots ``lora_snapshot_pre.pt`` / ``lora_snapshot_post.pt``
         (full-tensor LoRA dumps gathered on rank 0).

    2. DCP resume subprocess: ``--train.checkpoint.load_path=<DCP>`` ->
       :meth:`CheckpointerCallback._load_checkpoint` (model + optimizer
       + RNG + dataloader state) -> continue to the same ``max_steps``
       -> end-state snapshot **bit-exact** vs the writer's end-state.

    3. LoRA-adapter resume subprocess:
       ``model.lora_config.lora_adapter=<adapter>`` ->
       :meth:`BaseTrainer._setup_lora` resume branch
       (:meth:`veomni.lora.VeOmniLoraModel.from_pretrained`, which rebuilds the
       MoE wrappers from the ``veomni_lora`` block in adapter_config.json) ->
       the FSDP2 adapter-load path inside :func:`build_parallelize_model` ->
       *post-load* snapshot **bit-exact (in bf16)** vs the writer's *post-train*
       snapshot. Cast to bf16 because the adapter is written at ``model.dtype``
       (bf16 for the toy yaml); fp32 equality would require the on-disk format
       to carry fp32 mantissa bits, which it doesn't.

Mirroring production
--------------------
The ``toy_base_dir`` fixture pre-saves the toy Qwen3-MoE base weights to
disk; every trainer subprocess (writer + both resumers) is launched with
``--model.model_path=<toy_base_dir>`` so :func:`build_parallelize_model`
goes through the production load branch
(``load_model_weights(..., is_peft_model, adapter_path=...)``) instead
of the meta-init shortcut. Without ``model_path`` the gate at
``torch_parallelize.py:561`` (``if weights_path is None``) skips the
``adapter_path`` branch entirely and the LoRA-adapter resume case
collapses to a path-coverage smoke test.

Run (4 GPUs, Mode 1):
    pytest -v -s tests/lora/test_moe_lora_trainer.py::test_save_load_resume_round_trip[independent]

Or as the trainer subprocess directly:
    torchrun --nproc_per_node=4 tests/lora/test_moe_lora_trainer.py \
        tests/lora/qwen3_moe_toy_lora_independent.yaml \
        --train.checkpoint.output_dir=/tmp/test_moe_lora_run
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import pytest
import torch
import torch.distributed as dist
import yaml

from veomni.arguments import VeOmniArguments, parse_args
from veomni.data import build_dummy_dataset
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks.base import Callback, TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointerCallback, HFLoraCkptCallback
from veomni.utils import helper


_LORA_KEYS = ("lora_A", "lora_B")

# Filenames of LoRA tensor snapshots dumped by ``_SnapshotCallback``. Kept
# constant across writer and resumer subprocesses so the cross-process
# comparison can locate them by name in each ``output_dir``.
SNAPSHOT_PRE = "lora_snapshot_pre.pt"
SNAPSHOT_POST = "lora_snapshot_post.pt"

# Per-step ``loss`` / ``grad_norm`` trace dumped by ``_LogDictSaveCallback``.
# Same shape as ``tests/train_scripts/train_text_test.py``'s log_dict, so
# the e2e-style comparator (``check_metric``) can be reused if needed --
# but the trainer-level EP test in ``test_moe_lora_ep2.py`` parses this
# directly via ``json.load``.
LOG_DICT = "log_dict.json"

_TOY_CONFIG_PATH = "tests/toy_config/qwen3_moe_toy/config.json"


os.environ["NCCL_DEBUG"] = "OFF"
# DCP under FSDP2 opens a lot of file handles per shard; this avoids the
# "too many open files" failure mode on default-ulimit boxes.
torch.multiprocessing.set_sharing_strategy("file_system")

logger = helper.create_logger(__name__)


# ---------------------------------------------------------------------------
# Trainer subprocess -- the writer / DCP-resumer / adapter-resumer all run
# this same module via torchrun. The ``__main__`` entry point at the bottom
# parses args and runs ``MoeLoraTrainer.train()``.
# ---------------------------------------------------------------------------


def _to_full_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize an FSDP-sharded tensor on rank 0 and move to CPU.

    ``full_tensor`` triggers an all-gather across the FSDP group, so all
    ranks must call this; we just discard the result on non-rank-0.
    """
    full = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor.detach()
    return full.cpu()


class _FakeEnvironMeter:
    """Stub for the real environ meter -- the resume tests measure file I/O,
    not throughput, so we don't need the metric collection."""

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        pass


class _EnvironMeterCallbackTest(Callback):
    def __init__(self, trainer: MoeLoraTrainer):
        super().__init__(trainer)
        self.trainer.environ_meter = _FakeEnvironMeter()


class _SnapshotCallback(Callback):
    """Dump full LoRA tensors to ``<output_dir>/lora_snapshot_{pre,post}.pt``.

    Must be registered *after* the checkpoint callback so ``on_train_begin``
    sees the post-load model state when ``train.checkpoint.load_path`` is
    set, and *after* parallelization so adapter weights loaded via
    ``model.lora_config.lora_adapter`` are visible.
    """

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self._dump("pre")

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        self._dump("post")

    def _dump(self, when: str) -> None:
        snapshot: dict[str, torch.Tensor] = {}
        for n, p in self.trainer.model.named_parameters():
            if not any(k in n for k in _LORA_KEYS):
                continue
            snapshot[n] = _to_full_cpu(p)

        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = self.trainer.args.train.checkpoint.output_dir
            os.makedirs(output_dir, exist_ok=True)
            fname = SNAPSHOT_PRE if when == "pre" else SNAPSHOT_POST
            torch.save(snapshot, os.path.join(output_dir, fname))
            logger.info_rank0(f"[snapshot:{when}] {len(snapshot)} LoRA tensors -> {fname}")
        if dist.is_initialized():
            dist.barrier()


class _LogDictSaveCallback(Callback):
    """Append ``loss`` / ``grad_norm`` per step; flush to ``log_dict.json`` on train end.

    Mirrors ``tests/train_scripts/train_text_test.py``'s
    ``LogDictSaveCallback`` so the EP=2 save+resume parallel-alignment
    test in ``test_moe_lora_ep2.py`` can compare trainer runs the same
    way ``tests/e2e/test_e2e_parallel.py`` compares parallel modes.

    Loss is **dp-averaged** before logging so the recorded value is the
    same global-mean loss regardless of the DP layout that produced it
    -- without this, EP=1 (dp=2, rank 0 sees half the batch) and EP=2
    (dp=1, rank 0 sees the full batch) report different per-rank
    quantities (rank-0's local scaled-loss) and the alignment
    comparison degenerates into "are these two unrelated rank-0
    estimates close enough", which the e2e-style 0.1 tolerance can
    swallow even when the EP autograd path is broken. ``mean_global_loss``
    already scales each micro by ``cur_token / global_token *
    fsdp_size`` such that ``sum_across_dp(per_rank_loss) / dp_size`` is
    the global mean -- which is exactly what ``op=AVG`` over
    ``dp_group`` computes. ``grad_norm`` is already global (FSDP
    all-reduces gradients before the norm is taken) so it doesn't need
    a second reduction.

    Only ``loss`` and ``grad_norm`` are guaranteed to be appended every
    step; per-key ``loss_dict`` entries follow the trainer's emission
    pattern (a key absent on some steps yields a list shorter than
    ``loss``), so callers comparing ``log_dict`` keys other than
    ``loss`` / ``grad_norm`` must align on step indices first.

    Unconditional registration: the existing save/load/resume tests
    don't read ``log_dict.json``, so the extra file is harmless; the EP
    alignment test depends on it.
    """

    def __init__(self, trainer: MoeLoraTrainer) -> None:
        super().__init__(trainer)
        self.log_dict: dict[str, list] = defaultdict(list)

    def _dp_avg(self, value: float) -> float:
        """Return the cross-DP average of ``value``, or ``value`` itself when DP is degenerate.

        Using the trainer's parallel state instead of a hand-rolled
        ``dp_group`` lookup so this stays correct under FSDP1/FSDP2 +
        SP combinations (``dp_group`` already excludes SP/EP/PP per
        ``init_parallel_state``).
        """
        if not (dist.is_available() and dist.is_initialized()):
            return value
        from veomni.distributed.parallel_state import get_parallel_state

        ps = get_parallel_state()
        dp_group = ps.dp_group
        if dp_group is None or ps.dp_size <= 1:
            return value
        # All ranks in the dp_group must call this collectively; the
        # callback runs on every rank inside ``train_step``'s
        # ``on_step_end`` so that invariant holds.
        device = self.trainer.device if hasattr(self.trainer, "device") else torch.device("cpu")
        # float32: HCCL allreduce does not support float64 (at::kDouble).
        t = torch.tensor([float(value)], dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG, group=dp_group)
        return float(t.item())

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        # ``loss`` is already a Python float (``total_loss += loss.item()``).
        # ``grad_norm`` comes from ``veomni_clip_grad_norm`` and is a
        # ``Tensor`` -- coerce so the json dump succeeds.
        self.log_dict["loss"].append(self._dp_avg(loss))
        self.log_dict["grad_norm"].append(float(grad_norm) if grad_norm is not None else 0.0)
        for key, value in (loss_dict or {}).items():
            self.log_dict[key].append(self._dp_avg(float(value)))

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        output_dir = self.trainer.args.train.checkpoint.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, LOG_DICT), "w") as f:
            json.dump(self.log_dict, f, indent=4)


class MoeLoraTrainer(BaseTrainer):
    """Minimal trainer subclass for the round-trip test.

    Real :class:`CheckpointerCallback` + :class:`HFLoraCkptCallback` drive
    DCP + HF LoRA writes per production logic. ``_SnapshotCallback`` runs
    after both so its ``pre`` snapshot captures any state the checkpoint
    callbacks loaded.
    """

    def _build_model_assets(self) -> None:
        self.model_assets = [self.model_config]

    def _build_data_transform(self) -> None:
        pass

    def _build_dataset(self) -> None:
        args: VeOmniArguments = self.args
        self.train_dataset = build_dummy_dataset(task_type="text", size=64, max_seq_len=args.data.max_seq_len)
        args.compute_train_steps()
        self.train_steps = args.train_steps

    def _init_callbacks(self) -> None:
        self.environ_meter_callback = _EnvironMeterCallbackTest(self)
        # CheckpointerCallback drives the DCP save+load path; the
        # ``train.checkpoint.load_path`` resume case in the resume test
        # depends on its ``on_train_begin`` reload hook.
        self.checkpointer_callback = CheckpointerCallback(self)
        # HFLoraCkptCallback emits the HF-format LoRA adapter (adapter_model.safetensors
        # + adapter_config.json with the veomni_lora MoE block) at every
        # save_step. It also
        # extends DCP saves but no-ops if the DCP dir already exists, so
        # pairing it with CheckpointerCallback yields exactly one DCP
        # write + one LoRA HF write per save_step.
        self.hf_ckpt_callback = HFLoraCkptCallback(self)
        self.snapshot_callback = _SnapshotCallback(self)
        self.log_dict_callback = _LogDictSaveCallback(self)
        self.state = TrainerState()

    def on_train_begin(self) -> None:
        self.environ_meter_callback.on_train_begin(self.state)
        self.checkpointer_callback.on_train_begin(self.state)
        self.hf_ckpt_callback.on_train_begin(self.state)
        # Snapshot last so it sees state the checkpoint/HF callbacks loaded.
        self.snapshot_callback.on_train_begin(self.state)

    def on_train_end(self) -> None:
        self.environ_meter_callback.on_train_end(self.state)
        self.checkpointer_callback.on_train_end(self.state)
        self.hf_ckpt_callback.on_train_end(self.state)
        self.snapshot_callback.on_train_end(self.state)
        self.log_dict_callback.on_train_end(self.state)

    def on_epoch_begin(self) -> None:
        self.environ_meter_callback.on_epoch_begin(self.state)
        self.checkpointer_callback.on_epoch_begin(self.state)
        self.hf_ckpt_callback.on_epoch_begin(self.state)

    def on_epoch_end(self) -> None:
        self.environ_meter_callback.on_epoch_end(self.state)
        self.checkpointer_callback.on_epoch_end(self.state)
        self.hf_ckpt_callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches: list[dict[str, Any]] | None = None, **kwargs) -> None:
        self.environ_meter_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.checkpointer_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.hf_ckpt_callback.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: dict[str, float], grad_norm: float, **kwargs) -> None:
        self.environ_meter_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpointer_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.hf_ckpt_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.log_dict_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)


def main() -> None:
    """Subprocess entry point. Invoked when this file runs under ``torchrun``."""
    args: VeOmniArguments = parse_args(VeOmniArguments)
    trainer = MoeLoraTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()


# ===========================================================================
# Pytest side -- launches torchrun three times (writer, DCP resumer,
# adapter resumer) per ``share_expert_lora`` mode and asserts every
# checkpoint format round-trips cleanly.
# ===========================================================================


# ---------------------------------------------------------------------------
# Toy base-model fixture: build the Qwen3-MoE toy on CPU once per pytest
# session and save HF safetensors to disk. Both writer and resumers point
# ``--model.model_path`` at this directory so they traverse the same
# load-from-disk branch the production trainer uses.
# ---------------------------------------------------------------------------


def _build_and_save_toy_base(dest_dir: str) -> None:
    """Single-process: build the toy Qwen3-MoE on CPU and save HF weights to ``dest_dir``.

    Uses the same eager-ops config as the trainer yamls so the saved
    ``config.json`` matches what the subprocess trainer expects to load.
    The save is bf16 single-file safetensors (no index) -- VeOmni's
    :func:`_load_state_dict` resolves this layout via ``SAFE_WEIGHTS_NAME``
    so the trainer load path Just Works.

    Cost: ~40s on CPU and ~2 GB on disk for the toy config (1 B params,
    16 experts, 4 layers). Module-scoped means once per pytest invocation.
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models import build_foundation_model
    from veomni.utils import helper as _helper

    _helper.set_seed(42)
    ops = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
    )
    model = build_foundation_model(
        config_path=_TOY_CONFIG_PATH,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device="cpu",
        ops_implementation=ops,
    )
    model.save_pretrained(dest_dir, safe_serialization=True)


@pytest.fixture(scope="module")
def toy_base_dir(tmp_path_factory):
    """Pre-saved Qwen3-MoE toy base weights for trainer subprocesses.

    Lives one level above the per-test ``tmp_path`` so writer/resumer
    subprocesses across all parametrisations share the same base. Spawned
    in a *child process* (not in-process) so the fixture leaves the
    pytest-collection process free of foundation-model imports/state --
    which matters because the subprocess trainers also import veomni and
    we don't want any singleton (ops config, parallel state, etc.) leaking
    across.
    """
    import multiprocessing as mp

    base_dir = tmp_path_factory.mktemp("toy_base") / "qwen3_moe_toy"
    base_dir.mkdir(parents=True, exist_ok=False)
    base_dir_str = str(base_dir)

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_build_and_save_toy_base, args=(base_dir_str,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"toy base-model build subprocess exited with code {proc.exitcode}")

    return base_dir_str


# ---------------------------------------------------------------------------
# torchrun launcher
# ---------------------------------------------------------------------------


def torchrun_trainer(
    yaml_path: str,
    output_dir: str,
    *,
    extra_overrides: list[str] | None = None,
    nproc: int = 4,
) -> None:
    """Spawn this module under torchrun; raise on non-zero exit."""
    import subprocess

    from ..tools.launch_utils import find_free_port

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={find_free_port()}",
        __file__,
        yaml_path,
        f"--train.checkpoint.output_dir={output_dir}",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)

    result = subprocess.run(cmd, check=True)
    assert result.returncode == 0, f"torchrun exited with {result.returncode}"


# ---------------------------------------------------------------------------
# Snapshot comparators + small CLI / yaml helpers shared by the round-trip
# assertions.
# ---------------------------------------------------------------------------


def _l2_rel(actual: torch.Tensor, ref: torch.Tensor) -> float:
    """``||actual - ref||_F / ||ref||_F`` in fp32. 0.0 when ``ref`` is exactly zero."""
    a = actual.float()
    r = ref.float()
    denom = r.norm().item()
    if denom == 0.0:
        return (a - r).norm().item()
    return ((a - r).norm() / denom).item()


def _load_snapshot(path: str) -> dict[str, torch.Tensor]:
    assert os.path.isfile(path), f"Missing snapshot: {path}"
    return torch.load(path, map_location="cpu", weights_only=False)


def _compare_snapshots_bit_exact(actual_path: str, ref_path: str, *, label: str) -> None:
    """Bit-exact comparison of two LoRA snapshots dumped by ``_SnapshotCallback``.

    Both sides are cast to bf16 before ``torch.equal`` because that's the
    coarsest precision either resume path round-trips through:

    * The HF LoRA-adapter path writes ``adapter_model.safetensors`` at the
      model's storage dtype (bf16 for the toy yaml -- PEFT's
      ``save_pretrained`` casts to ``model.dtype``), so the on-disk
      bytes are bf16.
    * The DCP path stores raw fp32 bytes, but bf16 equality is implied
      by fp32 equality, so applying the cast costs nothing here and
      keeps a single comparator across both resume modes.
    """
    actual = _load_snapshot(actual_path)
    ref = _load_snapshot(ref_path)

    if set(actual) != set(ref):
        raise AssertionError(f"{label}: snapshot key sets differ: {sorted(set(actual) ^ set(ref))!r}")

    failures: list[str] = []
    for name, ref_val in ref.items():
        act_val = actual[name]
        if act_val.shape != ref_val.shape:
            failures.append(f"{name}: shape {tuple(act_val.shape)} != ref {tuple(ref_val.shape)}")
            continue
        cmp_act = act_val.to(torch.bfloat16)
        cmp_ref = ref_val.to(torch.bfloat16)
        if not torch.equal(cmp_act, cmp_ref):
            failures.append(f"{name}: not bit-exact (L2 rel {_l2_rel(cmp_act, cmp_ref):.4e})")

    if failures:
        raise AssertionError(
            f"{label}: {len(failures)}/{len(ref)} LoRA tensors mismatched after resume:\n  "
            + "\n  ".join(failures[:10])
        )


def _make_lora_adapter_resume_yaml(base_yaml: str, lora_adapter_path: str, dest: str) -> str:
    """Clone ``base_yaml`` with ``lora_config.lora_adapter`` set to the writer's adapter dir.

    The ``lora_config`` block is parsed as an opaque ``Dict``, so its
    nested keys can't be overridden via the CLI. We instead materialize a
    new yaml in the test's tmp_path and pass that to torchrun.

    The cloned yaml also trims down the resumer's training: ``max_steps=1``
    is enough to enter the train loop and trigger the ``_SnapshotCallback``
    (we only care about the post-load ``pre`` snapshot), and we disable
    further saves so the resumer doesn't redundantly write its own DCP/HF
    artifacts.
    """
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["lora_config"]["lora_adapter"] = lora_adapter_path
    cfg["train"]["max_steps"] = 1
    cfg["train"]["checkpoint"]["save_steps"] = 0
    cfg["train"]["checkpoint"]["save_hf_weights"] = False
    cfg["train"]["checkpoint"]["hf_save_steps"] = 0
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f)
    return dest


def _model_path_overrides(toy_base_dir: str) -> list[str]:
    """CLI overrides that point ``model.model_path`` (and ``config_path``) at the toy base.

    Pinning ``config_path`` alongside ``model_path`` keeps the loader from
    re-reading the in-tree toy config; both writer and resumer end up
    consuming the exact same on-disk artifacts.
    """
    return [
        f"--model.model_path={toy_base_dir}",
        f"--model.config_path={toy_base_dir}",
    ]


def _gpu_count_or_skip(min_count: int = 1, max_count: int = 4) -> int:
    """Return the nproc to use for torchrun, or skip the test if no devices."""
    from veomni.utils.device import get_torch_device

    device = get_torch_device()
    n = device.device_count() if device.is_available() else 0
    if n < min_count:
        pytest.skip(f"Requires at least {min_count} accelerator device(s); got {n}")
    return min(n, max_count)


def _yaml_for_mode(mode: str) -> str:
    return f"tests/lora/qwen3_moe_toy_lora_{mode}.yaml"


def _writer_dcp_path(writer_dir: str, save_step: int = 2) -> str:
    """Path to the writer's intermediate DCP shard (the resume target for the DCP test).

    ``save_step=2`` matches the yaml's ``save_steps=2`` / ``max_steps=4``;
    using the *intermediate* save (not the final one) means the resumer
    actually has work to do after reload, which is the property we're
    really testing.
    """
    return os.path.join(writer_dir, "checkpoints", f"global_step_{save_step}")


def _writer_adapter_path(writer_dir: str, save_step: int = 4) -> str:
    """Path to the writer's HF-format LoRA adapter (the resume target for the adapter test).

    Uses the *final* step's adapter so the resumer's pre-snapshot can be
    compared directly against the writer's post-snapshot.
    """
    return os.path.join(writer_dir, f"global_step_{save_step}")


def _assert_writer_artifacts_exist(writer_dir: str, mode: str, *, final_step: int = 4) -> None:
    """Save-side gate: writer must produce both DCP shards and the HF LoRA adapter.

    Folded in here (used to be its own ``test_moe_lora_trainer_saveload_*``
    test) so the round-trip test fails fast on a broken save path -- no
    point in attempting either resume if the writer didn't actually
    write the artifacts the resumes consume.
    """
    dcp_dir = _writer_dcp_path(writer_dir, save_step=final_step)
    hf_dir = _writer_adapter_path(writer_dir, save_step=final_step)

    assert os.path.isdir(dcp_dir), f"[{mode}] missing DCP checkpoint dir at {dcp_dir}"
    dcp_files = os.listdir(dcp_dir)
    # DCP writes one .distcp file per rank plus a .metadata file -- enough
    # to verify the save actually ran rather than just creating the dir.
    assert any(f.endswith(".metadata") for f in dcp_files), f"[{mode}] DCP metadata missing in {dcp_dir}: {dcp_files}"

    assert os.path.isdir(hf_dir), f"[{mode}] missing HF LoRA adapter dir at {hf_dir}"
    # VeOmniLoraModel embeds MoE metadata inside adapter_config.json (under the
    # ``veomni_lora`` block) instead of the legacy ``veomni_moe_lora.json`` sidecar.
    # The consolidated adapter is written as safetensors (sliceable per-rank by
    # the ep_sharded streaming loader), not the legacy ``adapter_model.bin``.
    for fname in ("adapter_model.safetensors", "adapter_config.json"):
        path = os.path.join(hf_dir, fname)
        assert os.path.isfile(path), f"[{mode}] missing {fname} in {hf_dir}: {os.listdir(hf_dir)}"
    with open(os.path.join(hf_dir, "adapter_config.json")) as f:
        adapter_cfg = json.load(f)
    assert adapter_cfg.get("veomni_lora", {}).get("moe_mode") == mode, (
        f"[{mode}] adapter_config.json veomni_lora.moe_mode != {mode!r}: {adapter_cfg.get('veomni_lora')}"
    )

    for snap in (SNAPSHOT_PRE, SNAPSHOT_POST):
        path = os.path.join(writer_dir, snap)
        assert os.path.isfile(path), f"[{mode}] missing {snap} in {writer_dir}"


# ---------------------------------------------------------------------------
# The single round-trip test: writer + DCP resume + adapter resume,
# parametrised on ``share_expert_lora`` mode.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_save_load_resume_round_trip(tmp_path, toy_base_dir, mode):
    """End-to-end save -> resume (DCP) -> resume (LoRA adapter) for one mode."""
    nproc = _gpu_count_or_skip()
    yaml_path = _yaml_for_mode(mode)
    base_overrides = _model_path_overrides(toy_base_dir)

    # ── 1. Writer ──────────────────────────────────────────────────────
    # Full N=4 steps, DCP saved at step 2 and step 4, HF LoRA at the same
    # cadence. Snapshots dumped at on_train_begin (= post-load, here
    # nothing to load so == post-init) and on_train_end (= post-train).
    writer_dir = str(tmp_path / "writer")
    torchrun_trainer(yaml_path, writer_dir, extra_overrides=base_overrides, nproc=nproc)
    _assert_writer_artifacts_exist(writer_dir, mode)

    # ── 2. DCP resume ──────────────────────────────────────────────────
    # Load step 2 DCP shard; CheckpointerCallback bumps global_step to 2
    # inside _load_checkpoint and the trainer continues for the remaining
    # 2 steps. Different output_dir keeps the resumer's saves from
    # clobbering the writer.
    #
    # DCP ships model + optimizer + RNG + dataloader state, so two
    # trainers seeing the same ordering must converge to identical
    # weights at the same step -- bit-exact under bf16 (and even under
    # fp32, since DCP saves raw bytes).
    dcp_resumer_dir = str(tmp_path / "resumer_dcp")
    dcp_overrides = base_overrides + [f"--train.checkpoint.load_path={_writer_dcp_path(writer_dir)}"]
    torchrun_trainer(yaml_path, dcp_resumer_dir, extra_overrides=dcp_overrides, nproc=nproc)
    _compare_snapshots_bit_exact(
        actual_path=os.path.join(dcp_resumer_dir, SNAPSHOT_POST),
        ref_path=os.path.join(writer_dir, SNAPSHOT_POST),
        label=f"{mode}/dcp",
    )

    # ── 3. LoRA-adapter resume ─────────────────────────────────────────
    # Load the writer's final-step HF LoRA adapter via
    # ``model.lora_config.lora_adapter`` -> ``_setup_lora`` resume branch
    # -> ``VeOmniLoraModel.from_pretrained`` (rebuilds MoE wrappers from the
    # ``veomni_lora`` block in adapter_config.json) -> FSDP2 adapter-load path
    # inside ``build_parallelize_model``. The resumer's pre-train snapshot is
    # exactly the writer's saved adapter (= the writer's post-train
    # snapshot, modulo the bf16 cast PEFT applies on save).
    adapter_resumer_dir = str(tmp_path / "resumer_adapter")
    resume_yaml = str(tmp_path / "resume_adapter.yaml")
    _make_lora_adapter_resume_yaml(yaml_path, _writer_adapter_path(writer_dir), resume_yaml)
    torchrun_trainer(resume_yaml, adapter_resumer_dir, extra_overrides=base_overrides, nproc=nproc)
    _compare_snapshots_bit_exact(
        actual_path=os.path.join(adapter_resumer_dir, SNAPSHOT_PRE),
        ref_path=os.path.join(writer_dir, SNAPSHOT_POST),
        label=f"{mode}/lora_adapter",
    )
