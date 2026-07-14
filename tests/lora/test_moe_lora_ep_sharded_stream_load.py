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
r"""Trainer-driven EP=2 test for the **PEFT** ``ep_sharded_stream_load`` path.

Companion to ``tests/utils/test_moe_ep_sharded_load_matrix.py`` (which
covers the *non-PEFT* MoE load matrix with a fake model). This one
exercises the real veomni LoRA stack on the Qwen3-MoE toy and asserts
that streaming a LoRA-wrapped MoE checkpoint reconstructs **bit-identical**
weights to the reference broadcast loader.

What ``ep_sharded_stream_load`` does for a PEFT model
-----------------------------------------------------
``load_model_weights_ep_sharded`` under ``is_peft_model=True`` (wired in
``veomni/models/module_utils.py``):

  * **Base experts** — the fused ``...experts.gate_up_proj`` /
    ``down_proj`` tensors are streamed from the base checkpoint straight
    into the wrapped ``...experts.<spec>.base_layer.weight`` destinations;
    each rank reads only its ``[E/ep, ...]`` dim-0 slice.
  * **Adapter** — ``_stream_lora_adapter_ep_sharded`` reads
    ``adapter_model.safetensors``: *independent* MoE-LoRA tensors
    (EP-sharded in the runtime plan) are read per-rank as their
    ``[E/ep, r, ...]`` slice; *shared* / dense LoRA tensors are read whole
    by every rank (FSDP then shards them).

The reference path (``broadcast_model_weights_from_rank0=True``) reads the
full tensors on rank 0 and broadcasts / EP-slices them in-memory. Both
paths must land on the same local weights, so a post-load LoRA snapshot
(``lora_snapshot_pre.pt``, gathered full-tensors on rank 0) must be
bit-exact between them.

Both LoRA flavours are covered:
  * ``independent`` — the per-expert adapter EP-slice branch of
    ``_stream_lora_adapter_ep_sharded`` fires.
  * ``shared`` — the whole-tensor adapter branch fires; only the base
    experts are EP-streamed.

Run (2 GPUs):
    pytest -v -s tests/lora/test_moe_lora_ep_sharded_stream_load.py
"""

from __future__ import annotations

import os
import shutil

import pytest
import yaml

# Reuse the battle-tested trainer harness. The trainer subprocess entry
# point + snapshot comparator + CLI helpers are shared; the base-model
# fixture is *not* -- see ``fused_toy_base_dir`` below for why.
from .test_moe_lora_ep2 import _torchrun_capture
from .test_moe_lora_trainer import (
    _TOY_CONFIG_PATH,
    SNAPSHOT_PRE,
    _compare_snapshots_bit_exact,
    _gpu_count_or_skip,
    _writer_adapter_path,
    _yaml_for_mode,
)


__all__ = [
    "fused_toy_base_dir",
    "test_peft_ep_sharded_stream_load_matches_broadcast",
]


# EP requires a fused MoE-LoRA kernel: the wrappers only implement the EP
# dispatch on the fused path. The backend is device-specific — ``fused_triton``
# on GPU, ``fused_npu`` on NPU (selecting the wrong one raises a hard error in
# ``apply_veomni_fused_moe_patch``), so pick it from the active accelerator.
def _fused_ops_override() -> str:
    from veomni.utils.import_utils import is_torch_npu_available

    backend = "fused_npu" if is_torch_npu_available() else "fused_triton"
    return f"--model.ops_implementation.moe_implementation={backend}"


# Seeder trains this many steps, then writes the HF adapter at the final
# step. Kept tiny -- we only need a *non-trivial* adapter on disk, not a
# converged one.
_SEED_STEPS = 2


# ──────────────────────────────────────────────────────────────────────
# Fused base checkpoint fixture
# ──────────────────────────────────────────────────────────────────────
#
# ``ep_sharded_stream_load`` streams each rank's dim-0 expert slice
# straight from the checkpoint. That is only possible when the on-disk
# expert layout *is* the model's fused layout
# (``...experts.gate_up_proj`` / ``down_proj`` with shape ``[E, ...]``).
# The shared ``toy_base_dir`` fixture in ``test_moe_lora_trainer`` writes
# the standard HF **per-expert** layout (``...experts.<e>.gate_proj.weight``),
# which needs the ``Qwen3MoeCheckpointTensorConverter`` per-expert->fused
# fusion at load time -- a transform the stream loader explicitly refuses
# (``NotImplementedError``; covered by the non-PEFT bail case in
# ``tests/utils/test_moe_ep_sharded_load_matrix.py``).
#
# So this test builds its *own* base by dumping the fused
# ``model.state_dict()`` directly (no ``save_pretrained`` re-split),
# giving a converter-free, streamable checkpoint.


def _build_and_save_fused_toy_base(dest_dir: str) -> None:
    """Single-process (spawned): build the toy and save its **fused** state_dict.

    Unlike ``test_moe_lora_trainer._build_and_save_toy_base`` (which uses
    ``save_pretrained`` and therefore emits the per-expert HF layout), this
    writes ``model.state_dict()`` verbatim via safetensors so the expert
    tensors stay fused ``[E, ...]`` -- the layout the stream loader can
    slice per-rank without a converter.
    """
    import shutil as _shutil

    from safetensors.torch import save_file

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
    # ``.clone()`` breaks any tied-weight storage sharing (e.g.
    # lm_head/embed_tokens) that safetensors refuses to serialise.
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    os.makedirs(dest_dir, exist_ok=True)
    save_file(state, os.path.join(dest_dir, "model.safetensors"))
    _shutil.copy(_TOY_CONFIG_PATH, os.path.join(dest_dir, "config.json"))


@pytest.fixture(scope="module")
def fused_toy_base_dir(tmp_path_factory):
    """Module-scoped fused Qwen3-MoE toy base (streamable expert layout).

    Built in a spawned child process so the pytest collection process
    stays free of foundation-model imports / singleton state, matching
    the ``toy_base_dir`` fixture's isolation rationale.
    """
    import multiprocessing as mp

    base_dir = tmp_path_factory.mktemp("fused_toy_base") / "qwen3_moe_toy"
    base_dir_str = str(base_dir)

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_build_and_save_fused_toy_base, args=(base_dir_str,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"fused toy base-model build subprocess exited with code {proc.exitcode}")
    return base_dir_str


def _fused_model_path_overrides(fused_base: str) -> list[str]:
    """Point ``model.model_path`` / ``config_path`` at the fused base dir."""
    return [
        f"--model.model_path={fused_base}",
        f"--model.config_path={fused_base}",
    ]


def _make_seeder_yaml(base_yaml: str, dest: str, *, max_steps: int) -> str:
    """Clone a mode yaml into an EP=2 seeder that emits the HF adapter."""
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["max_steps"] = max_steps
    cfg["train"]["checkpoint"]["save_steps"] = max_steps
    cfg["train"]["checkpoint"]["save_hf_weights"] = True
    cfg["train"]["checkpoint"]["hf_save_steps"] = max_steps
    cfg["train"]["checkpoint"].pop("load_path", None)
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f)
    return dest


def _make_adapter_resume_yaml(base_yaml: str, adapter_path: str, dest: str) -> str:
    """Clone a mode yaml to resume from ``adapter_path`` for a single step.

    ``max_steps=1`` is enough to reach ``on_train_begin`` -> the
    ``_SnapshotCallback`` post-load dump (``lora_snapshot_pre.pt``); all
    saves are disabled so the resumers only produce that snapshot.
    ``lora_config`` is an opaque ``Dict`` (its nested keys can't be set
    via the CLI), so the adapter path is materialised into a yaml.
    """
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["lora_config"]["lora_adapter"] = adapter_path
    cfg["train"]["max_steps"] = 1
    cfg["train"]["checkpoint"]["save_steps"] = 0
    cfg["train"]["checkpoint"]["save_hf_weights"] = False
    cfg["train"]["checkpoint"]["hf_save_steps"] = 0
    cfg["train"]["checkpoint"].pop("load_path", None)
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f)
    return dest


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_peft_ep_sharded_stream_load_matches_broadcast(tmp_path, fused_toy_base_dir, mode):
    """Streaming a LoRA-wrapped MoE checkpoint == broadcast loader, bit-exact.

    Three EP=2 subprocesses on the Qwen3-MoE toy:

    1. **Seeder** (default broadcast loader) trains ``_SEED_STEPS`` steps
       and writes the HF adapter (``adapter_model.safetensors``) at the
       final step.
    2. **Reference resumer** loads base + adapter via the broadcast loader
       (``broadcast_model_weights_from_rank0=True``) and dumps its
       post-load LoRA snapshot.
    3. **Stream resumer** loads base + adapter via
       ``ep_sharded_stream_load=True``
       (``broadcast_model_weights_from_rank0=False``) and dumps its
       post-load LoRA snapshot.

    The two snapshots must be **bit-exact** (bf16): both start from the
    same base checkpoint and the same on-disk adapter, so the only
    difference is *how* each rank obtained its local slice. For
    ``independent`` the adapter's per-expert LoRA is EP-streamed; for
    ``shared`` the adapter LoRA is read whole while the base experts are
    still EP-streamed.
    """
    nproc = _gpu_count_or_skip(min_count=2, max_count=2)
    yaml_path = _yaml_for_mode(mode)
    base_overrides = _fused_model_path_overrides(fused_toy_base_dir) + [
        _fused_ops_override(),
        "--train.accelerator.ep_size=2",
    ]

    # ── 1. Seeder: produce the HF adapter (adapter_model.safetensors) ───
    seeder_dir = str(tmp_path / "seeder")
    seeder_yaml = _make_seeder_yaml(yaml_path, str(tmp_path / "seeder.yaml"), max_steps=_SEED_STEPS)
    _torchrun_capture(seeder_yaml, seeder_dir, extra_overrides=base_overrides, nproc=nproc)

    adapter_dir = _writer_adapter_path(seeder_dir, save_step=_SEED_STEPS)
    adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
    assert os.path.isfile(adapter_file), (
        f"{mode}: seeder did not write a safetensors adapter at {adapter_file} "
        f"(dir contents: {sorted(os.listdir(adapter_dir)) if os.path.isdir(adapter_dir) else 'MISSING'}). "
        "ep_sharded adapter streaming requires the safetensors format."
    )

    resume_yaml = _make_adapter_resume_yaml(yaml_path, adapter_dir, str(tmp_path / "resume.yaml"))

    # ── 2. Reference resumer (broadcast loader) ─────────────────────────
    ref_dir = str(tmp_path / "ref")
    _torchrun_capture(resume_yaml, ref_dir, extra_overrides=base_overrides, nproc=nproc)

    # ── 3. Stream resumer (ep_sharded_stream_load) ──────────────────────
    stream_dir = str(tmp_path / "stream")
    _torchrun_capture(
        resume_yaml,
        stream_dir,
        extra_overrides=base_overrides
        + [
            "--train.broadcast_model_weights_from_rank0=False",
            "--train.ep_sharded_stream_load=True",
        ],
        nproc=nproc,
    )

    # ── 4. Post-load LoRA weights must be bit-identical ─────────────────
    _compare_snapshots_bit_exact(
        actual_path=os.path.join(stream_dir, SNAPSHOT_PRE),
        ref_path=os.path.join(ref_dir, SNAPSHOT_PRE),
        label=f"{mode}/ep_sharded_stream_vs_broadcast",
    )

    for d in (ref_dir, stream_dir):
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
