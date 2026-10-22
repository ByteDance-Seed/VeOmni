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
``veomni/models/checkpoint/weights.py``):

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

import argparse
import gc
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import yaml


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.lora.test_moe_lora_ep2 import (
    _expected_qwen3_moe_toy_adapter_shapes,
    _load_adapter,
)
from tests.lora.test_moe_lora_trainer import (
    _TOY_CONFIG_PATH,
    SNAPSHOT_PRE,
    MoeLoraTrainer,
    _compare_snapshots_bit_exact,
    _gpu_count_or_skip,
    _yaml_for_mode,
)
from veomni.arguments import VeOmniArguments, parse_args
from veomni.distributed.parallel_state import clear_parallel_state


_CI_EP_LAUNCH = LaunchSpec(nproc=2)


__all__ = [
    "fused_toy_base_dir",
    "seeded_lora_adapters_dir",
    "test_peft_ep_sharded_stream_load_matches_broadcast",
]


# EP requires a fused MoE-LoRA kernel: the wrappers only implement the EP
# dispatch on the fused path. The backend is device-specific — ``triton``
# on GPU, ``npu`` on NPU — so pick it from the active accelerator.
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
# straight from the checkpoint when the on-disk expert layout *is* the
# model's fused layout (``...experts.gate_up_proj`` / ``down_proj`` with
# shape ``[E, ...]``). The shared ``toy_base_dir`` fixture in
# ``test_moe_lora_trainer`` writes the standard HF **per-expert** layout
# (``...experts.<e>.gate_proj.weight``), which instead goes through the
# ``Qwen3MoeCheckpointTensorConverter`` per-rank expert stacking (covered
# by ``tests/utils/test_moe_ep_sharded_load_matrix.py`` and
# ``tests/utils/test_ep_sharded_expert_stream_load.py``).
#
# This test builds its *own* base by dumping the fused
# ``model.state_dict()`` directly (no ``save_pretrained`` re-split),
# so it exercises the converter-free dim-0 slice path under PEFT. The
# ``per_expert`` case loads ``toy_base_dir`` instead, so the converter's
# per-rank expert stacking runs with experts under ``base_layer``.


def _build_and_save_stream_load_assets(root_dir: str) -> None:
    """Single-process (spawned): build fused base, per-expert base, and non-zero LoRA adapters in one pass."""
    import copy
    import shutil as _shutil

    from safetensors.torch import save_file

    from tests.lora.utils import build_lora_test_model
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
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
    model = build_lora_test_model(
        config_path=_TOY_CONFIG_PATH,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device="cpu",
        ops_implementation=ops,
    )

    # 1. Fused base layout
    fused_dir = os.path.join(root_dir, "fused_base")
    os.makedirs(fused_dir, exist_ok=True)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(fused_dir, "model.safetensors"))
    _shutil.copy(_TOY_CONFIG_PATH, os.path.join(fused_dir, "config.json"))

    # 2. Per-expert HF base layout
    per_expert_dir = os.path.join(root_dir, "per_expert_base")
    os.makedirs(per_expert_dir, exist_ok=True)
    model.save_pretrained(per_expert_dir, safe_serialization=True)

    # 3. Seeded non-zero adapters for independent and shared modes
    adapters_dir = os.path.join(root_dir, "adapters")
    for mode in ("independent", "shared"):
        with open(_yaml_for_mode(mode)) as f:
            ycfg = yaml.safe_load(f)
        lcfg = VeOmniLoraConfig.from_yaml(ycfg["model"]["lora_config"])
        peft_model = VeOmniLoraModel(copy.deepcopy(model), lcfg)
        with torch.no_grad():
            for idx, (name, p) in enumerate(peft_model.named_parameters()):
                if "lora_A" not in name and "lora_B" not in name:
                    continue
                numel = p.numel()
                vals = (
                    (torch.arange(1, numel + 1, dtype=torch.float32).remainder(997) + 1.0) / 1024.0
                    + (idx + 1) * 0.03125
                ).reshape(p.shape)
                if p.ndim == 3:
                    expert_offset = torch.arange(1, p.shape[0] + 1, dtype=torch.float32).view(-1, 1, 1) * 0.125
                    vals = vals + expert_offset
                p.copy_(vals.to(p.dtype))
                assert bool((p != 0).all().item()), f"Expected all non-zero adapter values for {name}"
        mode_dir = os.path.join(adapters_dir, mode)
        peft_model.save_pretrained(mode_dir, safe_serialization=True)


@pytest.fixture(scope="module")
def _stream_load_assets_root(tmp_path_factory):
    import json
    import multiprocessing as mp

    root_dir = str(tmp_path_factory.mktemp("stream_load_assets"))
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_build_and_save_stream_load_assets, args=(root_dir,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"stream load assets build subprocess exited with code {proc.exitcode}")

    adapters_dir = os.path.join(root_dir, "adapters")
    for mode in ("independent", "shared"):
        mode_dir = os.path.join(adapters_dir, mode)
        with open(os.path.join(mode_dir, "adapter_config.json")) as f:
            acfg = json.load(f)
        assert acfg.get("veomni_lora", {}).get("moe_mode") == mode
        sd = _load_adapter(mode_dir)
        expected_shapes = _expected_qwen3_moe_toy_adapter_shapes(mode)
        assert set(sd) == set(expected_shapes)
        for k, exp_shape in expected_shapes.items():
            assert tuple(sd[k].shape) == exp_shape
            assert bool((sd[k] != 0).all().item()), f"{mode}/{k} has zero elements in fixture"

    return root_dir


@pytest.fixture(scope="module")
def fused_toy_base_dir(_stream_load_assets_root):
    return os.path.join(_stream_load_assets_root, "fused_base")


@pytest.fixture(scope="module")
def seeded_lora_adapters_dir(_stream_load_assets_root):
    return os.path.join(_stream_load_assets_root, "adapters")


def _fused_model_path_overrides(fused_base: str) -> list[str]:
    """Point ``model.model_path`` / ``config_path`` at the fused base dir."""
    return [
        f"--model.model_path={fused_base}",
        f"--model.config_path={fused_base}",
    ]


def _make_adapter_resume_yaml(base_yaml: str, adapter_path: str, dest: str) -> str:
    """Clone a mode yaml to resume from ``adapter_path`` for a single step."""
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


_STREAM_LOAD_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("independent", "fused"),
    ("shared", "fused"),
    ("independent", "per_expert"),
)


def _scenario_argv(yaml_path: str, output_dir: str, base_dir: str, *, stream_load: bool) -> list[str]:
    argv = [
        "tests/lora/test_moe_lora_ep_sharded_stream_load.py",
        yaml_path,
        f"--train.checkpoint.output_dir={output_dir}",
        f"--model.model_path={base_dir}",
        f"--model.config_path={base_dir}",
        _fused_ops_override(),
        "--model.accelerator.ep_size=2",
    ]
    if stream_load:
        argv.extend(
            [
                "--model.broadcast_model_weights_from_rank0=False",
                "--model.ep_sharded_stream_load=True",
            ]
        )
    else:
        argv.extend(
            [
                "--model.broadcast_model_weights_from_rank0=True",
                "--model.ep_sharded_stream_load=False",
            ]
        )
    return argv


def _load_and_snapshot(
    yaml_path: str, output_dir: str, base_dir: str, *, stream_load: bool
) -> dict[str, torch.Tensor]:
    sys.argv = _scenario_argv(yaml_path, output_dir, base_dir, stream_load=stream_load)
    args = parse_args(VeOmniArguments)
    trainer = MoeLoraTrainer(args)
    trainer.snapshot_callback._dump("pre")
    base_local = {
        n: (p.to_local() if hasattr(p, "to_local") else p).detach().cpu().clone()
        for n, p in trainer.model.named_parameters()
        if ".base_layer.weight" in n
    }
    del trainer
    gc.collect()
    if dist.is_initialized():
        dist.barrier()
    return base_local


def _run_stream_load_worker() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--fused-base-dir", required=True)
    parser.add_argument("--per-expert-base-dir", required=True)
    parser.add_argument("--adapters-dir", required=True)
    parser.add_argument("--reverse-scenarios", action="store_true")
    cli_args = parser.parse_args()

    scenarios = list(_STREAM_LOAD_SCENARIOS)
    if cli_args.reverse_scenarios or os.environ.get("VEOMNI_REVERSE_SCENARIOS") == "1":
        scenarios = list(reversed(scenarios))

    local_rank = os.environ.get("LOCAL_RANK", "0")
    try:
        for mode, base_layout in scenarios:
            label = f"{mode}_{base_layout}"
            base_dir = cli_args.fused_base_dir if base_layout == "fused" else cli_args.per_expert_base_dir
            adapter_dir = os.path.join(cli_args.adapters_dir, mode)
            resume_yaml = _make_adapter_resume_yaml(
                _yaml_for_mode(mode),
                adapter_dir,
                os.path.join(cli_args.work_dir, f"resume_{label}_rank{local_rank}.yaml"),
            )
            ref_dir = os.path.join(cli_args.work_dir, f"ref_{label}")
            stream_dir = os.path.join(cli_args.work_dir, f"stream_{label}")

            ref_base_local = _load_and_snapshot(resume_yaml, ref_dir, base_dir, stream_load=False)
            stream_base_local = _load_and_snapshot(resume_yaml, stream_dir, base_dir, stream_load=True)

            assert set(ref_base_local) == set(stream_base_local)
            for name, ref_t in ref_base_local.items():
                assert torch.equal(stream_base_local[name], ref_t), (
                    f"[{label}] base expert local shard mismatch for {name}"
                )

            if not dist.is_initialized() or dist.get_rank() == 0:
                _compare_snapshots_bit_exact(
                    actual_path=os.path.join(stream_dir, SNAPSHOT_PRE),
                    ref_path=os.path.join(ref_dir, SNAPSHOT_PRE),
                    label=f"{mode}/{base_layout}/ep_sharded_stream_vs_broadcast",
                )
            if dist.is_initialized():
                dist.barrier()
        if dist.is_initialized():
            dist.barrier()
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


@pytest.mark.launch_plan(provider=_CI_EP_LAUNCH)
def test_peft_ep_sharded_stream_load_matches_broadcast(
    tmp_path, _stream_load_assets_root, fused_toy_base_dir, seeded_lora_adapters_dir
):
    """Fused 2-rank worker comparing streaming vs broadcast PEFT MoE-LoRA load across all 3 layouts."""
    from tests.tools.launch_utils import find_free_port

    nproc = _gpu_count_or_skip(min_count=_CI_EP_LAUNCH.nproc, max_count=_CI_EP_LAUNCH.nproc)
    work_dir = str(tmp_path / "stream_load_work")
    os.makedirs(work_dir, exist_ok=True)
    per_expert_base_dir = os.path.join(_stream_load_assets_root, "per_expert_base")

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={find_free_port()}",
        os.path.abspath(__file__),
        f"--work-dir={work_dir}",
        f"--fused-base-dir={fused_toy_base_dir}",
        f"--per-expert-base-dir={per_expert_base_dir}",
        f"--adapters-dir={seeded_lora_adapters_dir}",
    ]
    res = subprocess.run(cmd, check=True)
    assert res.returncode == 0


if __name__ == "__main__":
    _run_stream_load_worker()
