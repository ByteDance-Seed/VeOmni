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
r"""Trainer-driven EP=2 tests for fused MoE-LoRA on Qwen3-MoE.

Two complementary tests, both driving the merged ``MoeLoraTrainer``
subprocess (the same one ``test_moe_lora_trainer.py`` uses for
save/load/resume) end-to-end with ``--train.accelerator.ep_size=2``:

1. ``test_ep2_trainer_integration[mode]`` -- asserts that the EP=2 path
   engages the right plumbing (plan-bridges fire, slicing log emits the
   right FQNs at the right ratio, DCP consolidates EP shards before HF
   save). These are the integration claims that the kernel-level test
   (``test_moe_lora_fused.py``) cannot make about the trainer.

2. ``test_moe_lora_ep_save_load_parallel_align[mode]`` -- asserts that
   the **EP=2 save formats** are loadable across configs and that
   training resumes correctly from each. One EP=2 seeder produces both
   the consolidated HF adapter (full ``[E, r, H]``) and the sharded
   DCP checkpoint, then three resumer subprocesses validate two
   independent properties:

     * **Cross-EP adapter parity** -- EP=1 and EP=2 adapter resumers
       both load the same HF adapter; their per-step
       ``loss`` / ``grad_norm`` must match within ``rtol=atol=0.1``.
       Proves the EP-slice on adapter-read is the inverse of the
       EP-gather on adapter-save: the same on-disk adapter drives the
       same training trajectory regardless of how the model parallel
       layout decomposes it at load time.
     * **EP=2 DCP round-trip** -- an EP=2 DCP resumer continues from
       the seeder's intermediate DCP shard; its per-step trajectory
       must match the seeder's tail-end log within ``rtol=atol=0.1``
       AND its end-state ``lora_snapshot_post.pt`` must be bit-equal
       (in bf16) to the seeder's. Proves that DCP at EP=2 covers
       model + optimizer + RNG + dataloader state correctly. The
       bit-exact half is the EP=2 analogue of the EP=1 DCP
       round-trip in ``test_moe_lora_trainer.py``.

   The shared 0.1 envelope is the same one
   ``tests/e2e/test_e2e_parallel.py`` uses for its sp/ep matrix at
   the toy-config scale.

Two integration bridges that make this work
-------------------------------------------
1. **MoE-LoRA wrapper rewrite.** The MoE-LoRA wrappers
   (:class:`LoraSharedExperts` / :class:`LoraIndependentExperts`) move
   the experts module's ``gate_up_proj`` and ``down_proj`` parameters
   into PEFT-aligned per-spec sub-modules
   (``...experts.gate_up_proj.base_layer.weight`` etc.). The bare-model
   plan's pattern (``model.layers.*.mlp.experts.gate_up_proj``) is
   rewritten to the wrapped FQN by
   :func:`_rewrite_plan_for_moe_lora_wrappers` so EP-sharding still
   slices the base experts after wrapping.
2. **PEFT prefix + per-expert LoRA bridges in get_runtime_parallel_plan.**
   PEFT-wrapping prepends ``base_model.model.`` to every ``named_*``
   FQN; the runtime helper detects that and calls
   :meth:`ParallelPlan.update_prefix`. It also extends the EP plan with
   every ``LoraIndependentExperts`` wrapper's per-expert
   ``<spec>.lora_A.<adapter>.weight`` /
   ``<spec>.lora_B.<adapter>.weight`` so EP slices the LoRA tensors
   alongside the base experts (kernel asserts otherwise on
   ``len(cumsum_M) == b.shape[0]``). ``LoraSharedExperts`` LoRA stays
   replicated.

Run:
    pytest -v tests/lora/test_moe_lora_ep2.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from .test_moe_lora_trainer import (
    LOG_DICT,
    SNAPSHOT_POST,
    _compare_snapshots_bit_exact,
    _gpu_count_or_skip,
    _model_path_overrides,
    _writer_adapter_path,
    _yaml_for_mode,
    toy_base_dir,  # re-exported pytest fixture; the import binds it into this module's namespace
)


# Silence "imported but unused" for the fixture -- pytest discovers
# fixtures by symbol presence in the test module, not by call sites.
__all__ = [
    "toy_base_dir",
    "test_ep2_trainer_integration",
    "test_moe_lora_ep_save_load_parallel_align",
    "test_independent_reset_lora_parameters_under_ep_shard",
]


# Override yaml's eager MoE impl: EP requires the fused triton kernel
# (LoraSharedExperts / LoraIndependentExperts raise under ep_enabled if
# the eager path is selected -- only the fused triton kernel implements
# the EP dispatch via ``dispatch_to_ep_class``).
_FUSED_OPS_OVERRIDE = "--model.ops_implementation.moe_implementation=fused_triton"

# Two different log lines fire during EP plan application; we grep for
# both so the assertion catches every Shard(0) target regardless of
# which path it took:
#
#   * ``ParallelPlan.apply`` -- "{para} sharding: slicing param {fqn} along ..."
#     fires for *every* param matched by the EP plan, when
#     ``build_parallelize_model`` runs the plan against the model on
#     meta. This is the only signal for params that are not loaded
#     from disk (e.g. freshly-initialised MoE-LoRA tensors).
#   * ``ParallelPlan._slice_shard_tensor`` -- "ep parameter {name}: sliced ... -> ... for ep rank R/N"
#     fires from the rank-0 broadcast load path when an on-disk param
#     gets EP-sliced en route to its destination. Only base experts
#     hit this -- LoRA tensors aren't on disk for our toy fixture.
_EP_PLAN_APPLY_RE = re.compile(r"ep sharding: slicing param (\S+) along ")
_EP_LOAD_SLICE_RE = re.compile(
    r"ep parameter (\S+): sliced torch\.Size\(\[(\d+),.*?\]\) -> torch\.Size\(\[(\d+),.*?\]\) for ep rank \d+/(\d+)"
)


# ──────────────────────────────────────────────────────────────────────
# torchrun helpers (local copy: we need to capture stdout to grep the
# slicing logs, which the shared ``torchrun_trainer`` discards)
# ──────────────────────────────────────────────────────────────────────


def _torchrun_capture(yaml_path: str, output_dir: str, extra_overrides: list[str], nproc: int) -> str:
    """Same shape as ``test_moe_lora_trainer.torchrun_trainer`` but captures stdout/stderr.

    Returns the merged stdout+stderr text so the caller can grep for
    EP slicing log lines (rank 0 only -- they're emitted by
    ``logger.info_rank0`` so duplicate-rank noise is already filtered).
    """
    from ..tools.launch_utils import find_free_port

    # ``find_free_port`` per-call (not a PID-derived constant) avoids
    # "address in use" between back-to-back ``torchrun`` launches.
    # ``test_moe_lora_ep_save_load_parallel_align`` alone fires four subprocesses
    # per parametrise variant, and Linux's ``TIME_WAIT`` keeps the c10d
    # store port held for ~60s after each one exits -- a stable PID port
    # collides on the second launch.
    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={find_free_port()}",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "test_moe_lora_trainer.py")),
        yaml_path,
        f"--train.checkpoint.output_dir={output_dir}",
        *extra_overrides,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Surface the worker stdout/stderr in the pytest report so the
        # caller doesn't have to re-run with -s to see the actual error.
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
    return res.stdout + res.stderr


def _parse_ep_slices(stdout: str) -> tuple[set, dict[str, tuple[int, int, int]]]:
    """Grep both EP slicing log lines.

    Returns ``(plan_apply_fqns, load_slice_info)`` where:
      * ``plan_apply_fqns`` is the set of FQNs that ``ParallelPlan.apply``
        sliced (the union covers every EP-sharded tensor regardless of
        whether it had a disk source);
      * ``load_slice_info`` maps each base-experts FQN that the rank-0
        load path EP-sliced to ``(full_dim0, local_dim0, ep_size)`` so
        we can assert the shard arithmetic.
    """
    plan_apply_fqns: set = set()
    load_slice_info: dict[str, tuple[int, int, int]] = {}
    for line in stdout.splitlines():
        m = _EP_PLAN_APPLY_RE.search(line)
        if m:
            plan_apply_fqns.add(m.group(1))
            continue
        m = _EP_LOAD_SLICE_RE.search(line)
        if m:
            load_slice_info[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return plan_apply_fqns, load_slice_info


def _load_adapter(adapter_dir: str) -> dict[str, torch.Tensor]:
    """Load the consolidated ``adapter_model.safetensors`` written by HFLoraCkptCallback.

    ``save_lora_adapter_with_dcp`` consolidates on rank 0 with
    ``safe_serialization=True``, so the adapter is always safetensors
    (sliceable per-rank by the ep_sharded streaming loader).
    """
    from safetensors.torch import load_file

    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    assert os.path.isfile(path), f"Missing adapter_model.safetensors at {path}"
    return load_file(path, device="cpu")


def _make_seeder_yaml(base_yaml: str, dest: str, *, max_steps: int, dcp_save_steps: int) -> str:
    """Clone ``base_yaml`` for the EP=2 seeder run that produces both save formats.

    The seeder runs a real EP=2 trainer for ``max_steps`` steps and
    emits both:

      * **HF adapter** at step ``max_steps`` (``hf_save_steps=max_steps``)
        -- the consolidated full-tensor format both adapter resumers
        will load. Validates EP=2 -> full-shape gather path
        (``save_lora_adapter_with_dcp`` -> DCP consolidation).
      * **DCP shards** at step ``dcp_save_steps`` (and
        ``max_steps`` -- ``CheckpointerCallback`` always saves the
        final step). The intermediate one is the resume target for
        the DCP round-trip subprocess.

    Together with the seeder's per-step ``log_dict.json``, this lets
    the test assert:

      * cross-EP parity: adapter-EP=1 vs adapter-EP=2 trajectories
        match (proves the EP-aware ``parallel_plan.shard_tensor`` on
        adapter read is correctly the inverse of the EP-gather on
        adapter save);
      * EP=2 DCP round-trip: seeder log[dcp_save_steps:max_steps] vs
        DCP-resumer log[0:max_steps-dcp_save_steps] match (proves
        DCP at EP=2 preserves model + optimizer + RNG +
        dataloader-cursor state).

    ``load_path`` is stripped defensively so a future per-mode yaml
    edit that adds a load_path won't silently turn the seeder into a
    resume run.
    """
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["max_steps"] = max_steps
    cfg["train"]["checkpoint"]["save_steps"] = dcp_save_steps
    cfg["train"]["checkpoint"]["save_hf_weights"] = True
    cfg["train"]["checkpoint"]["hf_save_steps"] = max_steps
    cfg["train"]["checkpoint"].pop("load_path", None)
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f)
    return dest


def _make_adapter_resume_yaml(base_yaml: str, adapter_path: str, dest: str, *, max_steps: int) -> str:
    """Clone ``base_yaml`` with ``lora_config.lora_adapter`` set + saves disabled.

    The seeded adapter (built by the EP=2 seeder run, full-shape
    ``[E, r, H]`` after DCP gathered the EP shards) becomes the
    *initial* LoRA state both EP=1 and EP=2 adapter-resume runs load
    via :func:`veomni_peft_model_from_pretrained` ->
    ``PeftModel.from_pretrained`` -> the FSDP2 adapter-load path
    inside ``build_parallelize_model``.

    Without this both runs fall through to ``_init_lora_parameter`` ->
    ``nn.init.kaiming_uniform_`` which draws different RNG bytes for
    EP=1's ``[E, r, H]`` vs EP=2's ``[E_local, r, H]`` and the loss
    trajectories diverge before step 0.

    ``lora_config`` is parsed as an opaque ``Dict``, so its nested keys
    can't be set via ``--model.lora_config.lora_adapter=...``; we
    materialize a new yaml in tmp_path and pass it to torchrun.

    Saves are disabled because the resume runs only consume per-step
    ``log_dict.json`` -- DCP/HF writes would just slow them down and
    clutter tmp_path. ``load_path`` is stripped defensively for the
    same reason as :func:`_make_seeder_yaml`.
    """
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["lora_config"]["lora_adapter"] = adapter_path
    cfg["train"]["max_steps"] = max_steps
    cfg["train"]["checkpoint"]["save_steps"] = 0
    cfg["train"]["checkpoint"]["save_hf_weights"] = False
    cfg["train"]["checkpoint"]["hf_save_steps"] = 0
    cfg["train"]["checkpoint"].pop("load_path", None)
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f)
    return dest


def _make_dcp_resume_yaml(base_yaml: str, dcp_path: str, dest: str, *, max_steps: int) -> str:
    """Clone ``base_yaml`` with ``train.checkpoint.load_path`` set + saves disabled.

    The DCP resumer continues from the seeder's intermediate DCP
    checkpoint (``<seeder>/checkpoints/global_step_<dcp_save_steps>``)
    and runs to ``max_steps`` -- ``CheckpointerCallback._load_checkpoint``
    bumps ``global_step`` to the resumed value and the trainer
    continues the remaining ``max_steps - dcp_save_steps`` steps. We
    then compare the resumer's per-step trajectory against the
    seeder's tail-end log to verify DCP saved+loaded the full
    training state (model + optimizer + RNG + dataloader cursor) on
    EP=2.

    ``lora_adapter`` is *not* set: DCP resume already carries the
    LoRA weights inside its model state, and PEFT's
    ``from_pretrained`` would clobber them. Saves are disabled for
    the same reason as the adapter-resume yaml.
    """
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["max_steps"] = max_steps
    cfg["train"]["checkpoint"]["load_path"] = dcp_path
    cfg["train"]["checkpoint"]["save_steps"] = 0
    cfg["train"]["checkpoint"]["save_hf_weights"] = False
    cfg["train"]["checkpoint"]["hf_save_steps"] = 0
    cfg["model"]["lora_config"].pop("lora_adapter", None)
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f)
    return dest


def _load_log_dict(run_dir: str) -> dict[str, list]:
    """Read the per-step ``loss`` / ``grad_norm`` trace dumped by ``_LogDictSaveCallback``."""
    path = os.path.join(run_dir, LOG_DICT)
    assert os.path.isfile(path), f"Missing {LOG_DICT} at {path} -- did the trainer crash before on_train_end?"
    with open(path) as f:
        return json.load(f)


def _assert_metric_close(
    name: str, lhs: list, rhs: list, *, rtol: float, atol: float, lhs_label: str, rhs_label: str
) -> None:
    """``np.isclose`` per step with a structured failure message.

    Mirrors the comparator pattern in ``tests/e2e/utils.py::check_metric``;
    inlined here to avoid the ``tests.tools`` import (transitively pulls
    pyav, which the transformers-v5 dev install does not ship).

    Empty-list guard: ``np.all(np.isclose([], []))`` is ``True``, so a
    "trainer ran but never reached ``on_step_end``" failure mode would
    silently pass without it.
    """
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    if len(a) != len(b):
        raise AssertionError(f"[{name}] step-count mismatch: {lhs_label}({len(a)}) vs {rhs_label}({len(b)})")
    if len(a) == 0:
        raise AssertionError(
            f"[{name}] no steps recorded -- both runs reached on_train_end with an empty log_dict; "
            f"likely the trainer crashed before on_step_end (check the subprocess output)."
        )

    is_close = np.isclose(a, b, rtol=rtol, atol=atol)
    if np.all(is_close):
        return
    first_bad = int(np.where(~is_close)[0][0])
    abs_err = np.abs(a - b)
    rel_err = abs_err / np.maximum(np.abs(b), 1e-12)
    raise AssertionError(
        f"\n[{name}] alignment failed (rtol={rtol}, atol={atol}):\n"
        f"  step {first_bad}: {lhs_label}={a[first_bad]:.6f} vs {rhs_label}={b[first_bad]:.6f} "
        f"(abs={abs_err[first_bad]:.4e}, rel={rel_err[first_bad]:.4e})\n"
        f"  max abs err: {abs_err.max():.4e}, max rel err: {rel_err.max():.4e}\n"
        f"  full {lhs_label}: {a.tolist()}\n"
        f"  full {rhs_label}: {b.tolist()}"
    )


# ──────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_ep2_trainer_integration(tmp_path, toy_base_dir, mode):
    """End-to-end EP=2 trainer run: assert plan-bridge + slice + DCP-consolidate.

    Runs the merged trainer twice on the same toy yaml + base + seed --
    once at ``ep_size=1`` to provide reference adapter shapes, once at
    ``ep_size=2`` to exercise the production EP path -- and asserts:

    * ep2 trainer completes the same 4 steps without error;
    * ep2 ``parallel_plan.shard_tensor`` log shows every layer's
      ``mlp.experts.gate_up_proj`` and ``mlp.experts.down_proj`` got
      sliced 16 -> 8 (= ``ep_size=2``);
    * for ``mode="independent"``, every per-expert LoRA tensor
      (``gate_proj.lora_A`` / ``gate_proj.lora_B`` /
      ``up_proj.lora_A`` / ``up_proj.lora_B`` /
      ``down_proj.lora_A`` / ``down_proj.lora_B``) at every layer also
      got sliced 16 -> 8 (proves
      :func:`_rewrite_plan_for_moe_lora_wrappers` worked);
    * for ``mode="shared"``, NO LoRA tensor shows up in the EP slice
      log (proves the per-expert extension is correctly gated to
      Independent mode and shared LoRA stays replicated);
    * the saved ``adapter_model.safetensors`` has the same key set and the
      same per-tensor shapes as the EP=1 reference (proves DCP
      gathered the EP shards before the PEFT save helper consumed
      them; per-expert tensors come back as ``[16, ...]`` not
      ``[8, ...]``).

    Value alignment is intentionally not asserted -- see module
    docstring for why per-rank meta-init makes that ill-defined.
    """
    nproc = _gpu_count_or_skip(min_count=2, max_count=2)
    yaml_path = _yaml_for_mode(mode)
    base_overrides = _model_path_overrides(toy_base_dir) + [_FUSED_OPS_OVERRIDE]

    # ── EP=1 reference (no slicing happens; just produces shapes/keys) ──
    ep1_dir = str(tmp_path / "ep1")
    _torchrun_capture(
        yaml_path,
        ep1_dir,
        extra_overrides=base_overrides + ["--train.accelerator.ep_size=1"],
        nproc=nproc,
    )
    ep1_adapter = _load_adapter(_writer_adapter_path(ep1_dir, save_step=4))

    # ── EP=2 path under test ───────────────────────────────────────────
    ep2_dir = str(tmp_path / "ep2")
    ep2_stdout = _torchrun_capture(
        yaml_path,
        ep2_dir,
        extra_overrides=base_overrides + ["--train.accelerator.ep_size=2"],
        nproc=nproc,
    )

    # ── Assertion 1: every base experts param got sliced 16 -> 8 ──────
    # Pull from the load-slice log: rank0_load_and_broadcast_weights
    # invokes ``parallel_plan.shard_tensor`` and emits a record with the
    # full and local dim-0 sizes -- the only place we get to verify the
    # EP shard arithmetic from log alone.
    plan_apply_fqns, load_slices = _parse_ep_slices(ep2_stdout)
    # PEFT-aligned wrapper layout: base params live under
    # ``<spec>.base_layer.weight``, and ``_rewrite_plan_for_moe_lora_wrappers``
    # rewrites the bare-Param plan patterns to the wrapped FQN before
    # the load-slice path emits its records.
    base_targets = {
        f"base_model.model.model.layers.{layer}.mlp.experts.{name}.base_layer.weight"
        for layer in range(4)  # toy config has 4 MoE layers
        for name in ("gate_up_proj", "down_proj")
    }
    missing_base = base_targets - set(load_slices)
    assert not missing_base, (
        f"{mode}: EP load-slice log missed {len(missing_base)} base experts:\n"
        f"  expected: {sorted(missing_base)[:4]!r}\n"
        f"  got:      {sorted(set(load_slices) & base_targets)[:4]!r}"
    )
    for name in base_targets:
        full, local, ep = load_slices[name]
        assert full == 16 and local == 8 and ep == 2, f"{mode}/{name}: bad slice {full=} {local=} {ep=}"

    # ── Assertion 2: per-expert LoRA tensors slice iff Independent ─────
    # LoRA tensors are missing from the toy base safetensors (they're
    # freshly initialised by ``_init_lora_parameter``), so they don't
    # hit the load-slice path; ``ParallelPlan.apply`` is the only EP
    # entry point that touches them. Use ``plan_apply_fqns`` here.
    # PEFT-aligned layout: per-expert LoRA lives at
    # ``...experts.<spec>.lora_A.<adapter>.weight``, so a contains check
    # on ``.lora_A.`` / ``.lora_B.`` (with surrounding dots) catches them
    # without false-positive matches against attribute names that happen
    # to contain ``lora_A`` as a substring.
    lora_per_expert_apply = {n for n in plan_apply_fqns if ".lora_A." in n or ".lora_B." in n}
    if mode == "independent":
        # 4 layers x 6 specs (lora_{A,B} x {gate_proj, up_proj, down_proj}) = 24
        assert len(lora_per_expert_apply) == 24, (
            f"independent: expected 24 per-expert LoRA tensors EP-sliced by plan.apply, got {len(lora_per_expert_apply)}"
        )
    else:  # shared
        assert not lora_per_expert_apply, (
            f"shared: shared LoRA must stay replicated, but plan.apply sliced these: {sorted(lora_per_expert_apply)!r}"
        )

    # ── Assertion 3: DCP-consolidated adapter matches EP=1 shapes/keys ─
    ep2_adapter = _load_adapter(_writer_adapter_path(ep2_dir, save_step=4))
    assert set(ep1_adapter) == set(ep2_adapter), (
        f"{mode}: ep1 vs ep2 adapter key sets differ:\n"
        f"  only in ep1: {sorted(set(ep1_adapter) - set(ep2_adapter))[:5]!r}\n"
        f"  only in ep2: {sorted(set(ep2_adapter) - set(ep1_adapter))[:5]!r}"
    )
    shape_mismatches = [
        f"{n}: ep1 {tuple(ep1_adapter[n].shape)} != ep2 {tuple(ep2_adapter[n].shape)}"
        for n in ep1_adapter
        if ep1_adapter[n].shape != ep2_adapter[n].shape
    ]
    assert not shape_mismatches, (
        f"{mode}: DCP did not consolidate EP shards before HF save -- {len(shape_mismatches)} shape mismatches:\n  "
        + "\n  ".join(shape_mismatches[:5])
    )


# ──────────────────────────────────────────────────────────────────────
# Test 2: EP=2 save + (cross-EP adapter resume parity) + (EP=2 DCP
# round-trip parity). All four subprocesses driven by the same
# trainer entry point used in test_moe_lora_trainer.py; only the
# yamls + ``--train.accelerator.ep_size`` differ.
# ──────────────────────────────────────────────────────────────────────


# Reuse the e2e tolerance: SP/EP across DP layouts only matches at
# ~1e-1 at toy-config scale even after we DP-average the per-step loss
# in ``_LogDictSaveCallback._dp_avg`` (so EP=1 (dp=2) and EP=2 (dp=1)
# both record the same global-mean loss -- otherwise it'd be rank-0's
# local scaled-loss on different sample subsets, an apples-to-oranges
# comparison the 0.1 tolerance could swallow even with a broken EP
# autograd path). The remaining ~1e-1 envelope absorbs reduction-order
# noise (4 sequential micros vs 2 + 2 all-reduced) plus bf16 mixed-
# precision drift; tighter rtol/atol flakes here for the same reason
# ``tests/e2e/test_e2e_parallel.py`` lands at the same number.
# ``grad_norm`` is already global (``veomni_clip_grad_norm`` operates
# on the gathered DTensor grads so FSDP's reduce-scatter has already
# happened) but we keep the same envelope so a single regression mode
# covers both metrics.
#
# The DCP round-trip comparison (assertion B below) could use a much
# tighter envelope -- DCP saves model + optimizer + RNG + dataloader
# bytes, so a correct round-trip would be bit-exact modulo bf16
# storage. We deliberately use the same 1e-1 envelope as assertion A
# to keep one tolerance knob for the whole test; tightening B alone
# would just turn flakes into mode-specific debugging without
# catching a different class of regression.
_ALIGN_RTOL = 1e-1
_ALIGN_ATOL = 1e-1

# Steps the seeder + resumers cover. ``_DCP_SAVE_STEP`` < ``_ALIGN_MAX_STEPS``
# so the DCP resumer has at least one optimizer step to take after
# load (otherwise the round-trip degenerates to a no-op load + final
# barrier and DCP correctness can't be observed).
_ALIGN_MAX_STEPS = 4
_DCP_SAVE_STEP = 2


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_moe_lora_ep_save_load_parallel_align(tmp_path, toy_base_dir, mode):
    """EP=2 save + cross-EP adapter resume parity + EP=2 DCP round-trip parity.

    Validates the **two save formats** an EP=2 MoE-LoRA trainer
    emits -- the consolidated HF adapter (full ``[E, r, H]``) and the
    sharded DCP checkpoint -- with a single seeder and three resumer
    subprocesses, all driven by the same trainer entry point used in
    ``test_moe_lora_trainer.py``.

    Subprocesses (per mode):

    1. **Seeder** (EP=2, ``max_steps=_ALIGN_MAX_STEPS``,
       ``save_steps=_DCP_SAVE_STEP``, ``hf_save_steps=_ALIGN_MAX_STEPS``).
       Real EP=2 trainer -- LoRA tensors are EP-sliced
       ``[E_local, r, H]`` per ``_extend_plan_for_moe_lora_independent``
       (or replicated for Shared mode), forward/backward go through
       the fused EP-aware kernel, and the two save callbacks fire:

         * ``HFLoraCkptCallback`` at step ``_ALIGN_MAX_STEPS`` calls
           ``save_lora_adapter_with_dcp`` which uses DCP to gather
           the EP shards back into the full ``[E, r, H]`` tensor on
           rank 0 before writing ``adapter_model.safetensors``.
         * ``CheckpointerCallback`` at steps
           ``_DCP_SAVE_STEP`` (intermediate) and
           ``_ALIGN_MAX_STEPS`` writes sharded DCP shards
           (model + optimizer + RNG + dataloader cursor).

       Per-step ``log_dict.json`` records DP-averaged loss + global
       grad_norm for the full ``_ALIGN_MAX_STEPS`` trajectory.

    2. **Adapter resumer (EP=1)**. Loads the seeder's HF adapter
       (full ``[E, r, H]``) via ``model.lora_config.lora_adapter``
       -> ``veomni_peft_model_from_pretrained`` ->
       ``PeftModel.from_pretrained`` -> the FSDP2 adapter-load path
       inside ``build_parallelize_model``. At ep=1 the
       ``parallel_plan.shard_tensor`` step is a no-op, so the LoRA
       weights land in the model exactly as written. Runs
       ``_ALIGN_MAX_STEPS`` steps from a freshly-zeroed optimizer.

    3. **Adapter resumer (EP=2)**. Same as (2) but with
       ``--train.accelerator.ep_size=2``. The adapter-load path
       EP-slices the LoRA on read (Independent) or leaves it
       replicated (Shared). Runs ``_ALIGN_MAX_STEPS`` steps from a
       freshly-zeroed optimizer.

    4. **DCP resumer (EP=2)**. Loads the seeder's intermediate DCP
       shard at step ``_DCP_SAVE_STEP`` via
       ``train.checkpoint.load_path`` ->
       ``CheckpointerCallback._load_checkpoint`` (which restores
       model + optimizer + RNG + dataloader cursor and bumps
       ``global_step``). Continues to ``_ALIGN_MAX_STEPS`` -- so it
       takes ``_ALIGN_MAX_STEPS - _DCP_SAVE_STEP`` more optimizer
       steps that should retrace the seeder's tail-end trajectory
       exactly (modulo bf16 + reduction-order noise).

    Assertions:

    A. **Cross-EP adapter parity** -- adapter resumer EP=1 vs EP=2
       per-step loss + grad_norm match within ``rtol=atol=0.1``.
       Proves that the EP-slice on adapter-read is the inverse of
       the EP-gather on adapter-save, so the same on-disk adapter
       drives the same training trajectory regardless of how the
       model parallel layout decomposes it at load time. A
       regression in either:

         * the EP-gather path inside
           :func:`save_lora_adapter_with_dcp` (e.g. forgetting to
           use DCP for the LoRA-only state), or
         * ``_dispatch_parameter`` + ``parallel_plan.shard_tensor``
           on adapter read (e.g. mismatched dim-0 size between the
           on-disk full ``[E, r, H]`` and the model's local
           ``[E_local, r, H]``)

       breaks this without touching the integration test's plumbing
       claims.

    B. **EP=2 DCP round-trip** -- two complementary checks of the
       same property:

         B1. DCP resumer EP=2 per-step loss + grad_norm match the
             seeder's tail ``log[_DCP_SAVE_STEP:]`` within
             ``rtol=atol=0.1`` (fast-readable failure signal).
         B2. ``ep2_dcp/lora_snapshot_post.pt`` is **bit-equal** (in
             bf16, via ``_compare_snapshots_bit_exact``) to
             ``seeder/lora_snapshot_post.pt`` -- both are rank-0
             gathered LoRA-weight dumps at end-of-step-``_ALIGN_MAX_STEPS``,
             so a correct DCP round-trip MUST land on the same
             bytes. This is the EP=2 analogue of the EP=1 bit-exact
             DCP round-trip already covered by
             ``test_moe_lora_trainer.py``; assertion A's same check
             is impossible because the two adapter resumers run with
             different DP layouts (EP=1 -> dp=2, EP=2 -> dp=1), so
             reduction order differs and per-step loss/grad_norm
             close is the strongest possible parity.

       Together (B1 + B2) catch a regression that drops e.g.
       optimizer-momentum bytes from the DCP shard: B1 might tolerate
       the resulting 2-step drift if it's small, but B2 always
       breaks.

    Why this is not redundant with the integration test
    ---------------------------------------------------
    The integration test asserts the EP plumbing *engages* (slicing
    happens at the right FQNs, DCP consolidates the right shapes).
    This test asserts the math is *right* after that plumbing -- a
    regression in the kernel's EP autograd classes (e.g. wrong
    reduction order in
    ``EPMergedFc1IndependentLoRAGroupGemm.backward``), a regression
    in the adapter EP-slice on read, or a regression in DCP state
    coverage would fail this while passing the integration test, and
    vice versa.
    """
    nproc = _gpu_count_or_skip(min_count=2, max_count=2)
    yaml_path = _yaml_for_mode(mode)
    base_overrides = _model_path_overrides(toy_base_dir) + [_FUSED_OPS_OVERRIDE]

    # ── 1. Seeder (EP=2, saves both adapter + DCP) ──────────────────────
    seeder_dir = str(tmp_path / "seeder")
    seeder_yaml = _make_seeder_yaml(
        yaml_path,
        str(tmp_path / "seeder.yaml"),
        max_steps=_ALIGN_MAX_STEPS,
        dcp_save_steps=_DCP_SAVE_STEP,
    )
    _torchrun_capture(
        seeder_yaml,
        seeder_dir,
        extra_overrides=base_overrides + ["--train.accelerator.ep_size=2"],
        nproc=nproc,
    )
    seeder_adapter = _writer_adapter_path(seeder_dir, save_step=_ALIGN_MAX_STEPS)
    seeder_dcp = os.path.join(seeder_dir, "checkpoints", f"global_step_{_DCP_SAVE_STEP}")
    assert os.path.isfile(os.path.join(seeder_adapter, "adapter_model.safetensors")), (
        f"{mode}: seeder failed to write adapter at {seeder_adapter}"
    )
    assert os.path.isdir(seeder_dcp), f"{mode}: seeder failed to write DCP at {seeder_dcp}"

    seeder_log = _load_log_dict(seeder_dir)
    assert seeder_log.get("loss") and len(seeder_log["loss"]) == _ALIGN_MAX_STEPS, (
        f"{mode}: seeder log expected {_ALIGN_MAX_STEPS} loss entries, got {len(seeder_log.get('loss', []))}"
    )

    # Build the two resume yamls (adapter-load reused by both EP=1 and
    # EP=2 adapter resumers; DCP-load only used by the EP=2 DCP resumer).
    adapter_yaml = _make_adapter_resume_yaml(
        yaml_path,
        seeder_adapter,
        str(tmp_path / "adapter_resume.yaml"),
        max_steps=_ALIGN_MAX_STEPS,
    )
    dcp_yaml = _make_dcp_resume_yaml(
        yaml_path,
        seeder_dcp,
        str(tmp_path / "dcp_resume.yaml"),
        max_steps=_ALIGN_MAX_STEPS,
    )

    # ── 2. Adapter resumer (EP=1) ──────────────────────────────────────
    ep1_adapter_dir = str(tmp_path / "ep1_adapter")
    _torchrun_capture(
        adapter_yaml,
        ep1_adapter_dir,
        extra_overrides=base_overrides + ["--train.accelerator.ep_size=1"],
        nproc=nproc,
    )

    # ── 3. Adapter resumer (EP=2) ──────────────────────────────────────
    ep2_adapter_dir = str(tmp_path / "ep2_adapter")
    _torchrun_capture(
        adapter_yaml,
        ep2_adapter_dir,
        extra_overrides=base_overrides + ["--train.accelerator.ep_size=2"],
        nproc=nproc,
    )

    # ── 4. DCP resumer (EP=2) ──────────────────────────────────────────
    ep2_dcp_dir = str(tmp_path / "ep2_dcp")
    _torchrun_capture(
        dcp_yaml,
        ep2_dcp_dir,
        extra_overrides=base_overrides + ["--train.accelerator.ep_size=2"],
        nproc=nproc,
    )

    # ── 5. Assertion A: cross-EP adapter parity ────────────────────────
    ep1_adapter_log = _load_log_dict(ep1_adapter_dir)
    ep2_adapter_log = _load_log_dict(ep2_adapter_dir)
    # Truthy-on-non-empty guard: ``np.isclose([], [])`` is vacuously
    # True, so an empty log_dict (trainer crashed before on_step_end)
    # would silently pass the comparator.
    assert ep1_adapter_log.get("loss") and ep2_adapter_log.get("loss"), (
        f"{mode}: empty adapter-resume log_dict: ep1={list(ep1_adapter_log)} ep2={list(ep2_adapter_log)}"
    )
    assert ep1_adapter_log.get("grad_norm") and ep2_adapter_log.get("grad_norm")
    _assert_metric_close(
        f"{mode}/A.adapter/loss",
        ep1_adapter_log["loss"],
        ep2_adapter_log["loss"],
        rtol=_ALIGN_RTOL,
        atol=_ALIGN_ATOL,
        lhs_label="ep1_adapter",
        rhs_label="ep2_adapter",
    )
    _assert_metric_close(
        f"{mode}/A.adapter/grad_norm",
        ep1_adapter_log["grad_norm"],
        ep2_adapter_log["grad_norm"],
        rtol=_ALIGN_RTOL,
        atol=_ALIGN_ATOL,
        lhs_label="ep1_adapter",
        rhs_label="ep2_adapter",
    )

    # ── 6. Assertion B: EP=2 DCP round-trip ────────────────────────────
    # Two complementary checks of the same property -- a regression
    # that drops e.g. optimizer-momentum bytes from the DCP shard
    # would not necessarily move the loss outside the 1e-1 envelope
    # in 2 optimizer steps, but would always break the bit-exact
    # snapshot equality. The metric-close check stays as a fast
    # readable failure signal; the snapshot check is the strong
    # guarantee.
    ep2_dcp_log = _load_log_dict(ep2_dcp_dir)
    assert ep2_dcp_log.get("loss"), f"{mode}: empty DCP-resume log_dict: {list(ep2_dcp_log)}"
    seeder_tail_loss = seeder_log["loss"][_DCP_SAVE_STEP:]
    seeder_tail_gn = seeder_log["grad_norm"][_DCP_SAVE_STEP:]
    expected_tail_len = _ALIGN_MAX_STEPS - _DCP_SAVE_STEP
    assert len(seeder_tail_loss) == expected_tail_len, (
        f"{mode}: seeder tail length {len(seeder_tail_loss)} != expected {expected_tail_len}; "
        f"seeder log: {seeder_log['loss']}"
    )
    _assert_metric_close(
        f"{mode}/B1.dcp/loss",
        seeder_tail_loss,
        ep2_dcp_log["loss"],
        rtol=_ALIGN_RTOL,
        atol=_ALIGN_ATOL,
        lhs_label="seeder_tail",
        rhs_label="ep2_dcp",
    )
    _assert_metric_close(
        f"{mode}/B1.dcp/grad_norm",
        seeder_tail_gn,
        ep2_dcp_log["grad_norm"],
        rtol=_ALIGN_RTOL,
        atol=_ALIGN_ATOL,
        lhs_label="seeder_tail",
        rhs_label="ep2_dcp",
    )
    # Bit-exact (in bf16) end-state weight equality. ``_SnapshotCallback``
    # in the trainer subprocess gathers the full LoRA tensors on rank 0
    # at ``on_train_end``, so seeder/lora_snapshot_post.pt and
    # ep2_dcp/lora_snapshot_post.pt are both rank-0 dumps of the same
    # logical end-of-step-N model -- they MUST be bit-equal modulo bf16
    # storage. This is the EP=2 analogue of the EP=1 DCP round-trip
    # assertion in ``test_moe_lora_trainer.py``; assertion A's same-
    # check is impossible by construction (different DP layouts).
    _compare_snapshots_bit_exact(
        actual_path=os.path.join(ep2_dcp_dir, SNAPSHOT_POST),
        ref_path=os.path.join(seeder_dir, SNAPSHOT_POST),
        label=f"{mode}/B2.dcp.snapshot_post",
    )

    # Cleanup the four sizable run dirs once compared (seeder DCP and
    # adapter dirs kept implicit via tmp_path -- pytest's tmp_path
    # caches the last 3 invocations, useful for inspecting the on-disk
    # state when a parametrise variant fails halfway).
    for d in (ep1_adapter_dir, ep2_adapter_dir, ep2_dcp_dir):
        shutil.rmtree(d, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────
# Focused regression: LoraIndependentExperts.reset_lora_parameters under
# an EP-sharded DTensor. Catches the silent-zero init bug where indexing
# ``w[e]`` on a Shard(0) DTensor returns a view whose in-place
# ``kaiming_uniform_`` does not write back to the local storage,
# leaving the entire local LoRA-A shard at zeros (and therefore the
# adapter dead -- forward delta = 0, both grads = 0, never trains).
# Bug surfaces only at world_size>=2 (EP=1 has E_local==E_global so the
# bug is masked); the trainer integration tests above don't catch it
# because (a) `test_ep2_trainer_integration` only checks slice shapes,
# and (b) `test_moe_lora_ep_save_load_parallel_align` always runs in
# adapter-resume mode which bypasses the fresh-init path.
# ──────────────────────────────────────────────────────────────────────


def _reset_lora_under_ep_worker():
    """mp.spawn worker: build wrapper, fake-EP-shard one LoRA-A tensor, reset, assert."""
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.tensor import DTensor, Shard, init_device_mesh

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from utils import build_toy, full_eager_ops

    from veomni.lora.moe_layers import apply_independent_moe_lora
    from veomni.utils.device import get_device_type

    rank = dist.get_rank()
    world = dist.get_world_size()
    device_type = get_device_type()

    torch.manual_seed(0)
    model = build_toy("qwen3_moe_toy", ops=full_eager_ops())
    fqns = apply_independent_moe_lora(
        model,
        target_parameter_patterns=[
            "model.layers.*.mlp.experts.gate_up_proj",
            "model.layers.*.mlp.experts.down_proj",
        ],
        r=4,
        lora_alpha=8,
    )
    assert fqns, "no MoE-LoRA wrappers installed"

    mesh = init_device_mesh(device_type, (world,), mesh_dim_names=("ep",))
    wrapper = model.get_submodule(fqns[0])
    spec = wrapper.gate_proj
    A = spec.lora_A["default"]  # _LoraParam3D, weight shape [E, r, in_feat]

    E, r_, H = A.weight.shape
    assert E % world == 0, f"toy E={E} not divisible by world={world}"
    e_local = E // world

    # Pre-fill LoRA-A with zeros, then ask reset_lora_parameters to
    # re-init it. The previous bug was *silent*: the function returned
    # cleanly (no exception) but ``kaiming_uniform_(w[e])`` wrote into
    # a DTensor view that never propagated to the underlying
    # ``_local_tensor``, so local storage stayed all zeros and the
    # adapter was effectively dead (LoRA delta = 0, both grads = 0,
    # never trains). Starting from a known-zero tensor lets the
    # post-reset assertions below detect that silent failure mode by
    # checking that the local shard actually got populated.
    local_zeros = torch.zeros(e_local, r_, H, dtype=A.weight.dtype, device=device_type)
    A.weight = nn.Parameter(DTensor.from_local(local_zeros, mesh, [Shard(0)], run_check=False))

    wrapper.reset_lora_parameters(init_lora_weights=True)

    local_after = A.weight._local_tensor
    assert local_after.shape == (e_local, r_, H), f"rank {rank}: local shape changed: {local_after.shape}"
    nonzero = (local_after != 0).any().item()
    assert nonzero, (
        f"rank {rank}: LoRA-A local shard is ALL-ZERO after reset_lora_parameters under "
        f"EP=Shard(0). This is the regression: indexing w[e] on a sharded DTensor returns "
        f"a view whose in-place init does not propagate to local storage."
    )
    # Each per-expert slice should have non-trivial variance (kaiming).
    per_expert_std = local_after.float().reshape(e_local, -1).std(dim=-1)
    assert (per_expert_std > 1e-3).all(), (
        f"rank {rank}: at least one per-expert LoRA-A slice has near-zero std after init: {per_expert_std.tolist()}"
    )


def test_independent_reset_lora_parameters_under_ep_shard():
    """Regression: ``LoraIndependentExperts.reset_lora_parameters`` must
    populate local storage when the LoRA-A weight is an EP-sharded
    ``DTensor[Shard(0)]``. See module docstring for context."""
    from ..tools.launch_utils import torchrun

    torchrun(_reset_lora_under_ep_worker, world_size=2)
