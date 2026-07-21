# Copyright 2025 Bytedance Ltd. and/or its affiliates
"""Compare resume GPU memory with / without skip-HF-weight-load using built-in tools.

Uses existing VeOmni surfaces only:
  - helper.print_device_mem_info
  - EnvironMeter max_memory_* (via step logs when training proceeds)

Note: BaseTrainer prints "VRAM usage after building model" *before* parallelize /
weight load, so this harness re-prints the same helper after parallelize and
after DCP resume so the comparison reflects the real load path.

Phase 0 materializes larger toy HF safetensors so force-HF is a real weight load
with measurable VRAM.
Phase 1 trains a short DCP.
Phase 2 resumes twice:
  A) skip HF load (default when load_path is set)
  B) force HF load on resume (test harness only)

Usage:
  VEOMNI_CHECKPOINT_TEST_NPROC=4 \\
    python -m pytest tests/checkpoints/test_resume_memory_profile_compare.py -s
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch

try:
    from .utils import MODEL_CONFIGS, get_checkpoint_test_nproc, get_output_dir
except Exception:
    from utils import MODEL_CONFIGS, get_checkpoint_test_nproc, get_output_dir  # type: ignore

from tools import hf_local_or_remote, resolve_ops_overrides
from tools.launch_utils import find_free_port
from veomni.arguments import parse_args
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.data import build_dummy_dataset
from veomni.models import build_foundation_model
from veomni.models.module_utils import save_model_weights
from veomni.trainer.base import BaseTrainer, VeOmniArguments
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from veomni.trainer.callbacks.trace_callback import EnvironMeterCallback, TqdmCallback
from veomni.utils import helper

os.environ["NCCL_DEBUG"] = "OFF"
torch.multiprocessing.set_sharing_strategy("file_system")
logger = helper.create_logger(__name__)


class SaveAtEndCallback(CheckpointerCallback):
    def on_step_end(self, state: TrainerState, **kwargs):
        pass

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if state.epoch == 0:
            self._save_checkpoint(state)

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.args.train.checkpoint.load_path is None:
            return
        super().on_train_begin(state, **kwargs)
        # Existing memory helper after DCP restore (and its empty_cache).
        helper.print_device_mem_info("VRAM usage after building model")


class ResumeProfileTrainer(BaseTrainer):
    """Minimal trainer using built-in mem meter callbacks (no torch.profiler)."""

    def _build_model_assets(self):
        self.model_assets = [self.model_config]

    def _build_data_transform(self):
        pass

    def _build_dataset(self):
        args: VeOmniArguments = self.args
        self.train_dataset = build_dummy_dataset(task_type="text", size=8192, max_seq_len=args.data.max_seq_len)
        args.compute_train_steps()
        self.train_steps = args.train_steps

    def _build_parallelized_model(self):
        super()._build_parallelized_model()
        # BaseTrainer prints this before parallelize; re-print after real materialize.
        helper.print_device_mem_info("VRAM usage after building model")

    def _build_optimizer(self):
        super()._build_optimizer()
        helper.print_device_mem_info("VRAM usage after building model")

    def _init_callbacks(self):
        self.environ_meter_callback = EnvironMeterCallback(self)
        self.tqdm_callback = TqdmCallback(self)
        self.checkpointer_callback = SaveAtEndCallback(self)
        self._callbacks = [
            self.environ_meter_callback,
            self.tqdm_callback,
            self.checkpointer_callback,
        ]
        self.state = TrainerState()


def main():
    # Test-harness only: force HF materialize even when load_path is set.
    if os.environ.get("VEOMNI_FORCE_HF_LOAD_ON_RESUME", "0") == "1":
        from veomni.trainer import base as base_mod

        _orig = base_mod.BaseTrainer._build_parallelized_model

        def _force_hf(self, *args, **kwargs):
            ckpt = self.args.train.checkpoint
            saved = ckpt.load_path
            ckpt.load_path = None  # skip_weights_load becomes False
            try:
                logger.info_rank0(
                    "FORCE_HF_LOAD_ON_RESUME: temporarily clear load_path for HF materialize "
                    f"(DCP still restores from {saved})"
                )
                return _orig(self, *args, **kwargs)
            finally:
                ckpt.load_path = saved

        base_mod.BaseTrainer._build_parallelized_model = _force_hf

    args: VeOmniArguments = parse_args(VeOmniArguments)
    trainer = ResumeProfileTrainer(args)
    trainer.train()


def _latest_ckpt(output_dir: str) -> Path:
    ckpt_root = Path(output_dir) / "checkpoints"
    steps = sorted(ckpt_root.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    assert steps, f"no checkpoints under {ckpt_root}"
    return steps[-1]


def _run(cmd: str, env: Optional[Dict[str, str]], log_path: Path) -> str:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    logger.info("Running:\n%s", cmd)
    proc = subprocess.run(cmd, shell=True, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout or "")
    if proc.returncode != 0:
        raise AssertionError(f"rc={proc.returncode}\n{(proc.stdout or '')[-8000:]}")
    return proc.stdout or ""


def _scaled_config_path(model_name: str, work_dir: Path) -> Path:
    """Make a larger toy config so VRAM deltas show up in print_device_mem_info (.2f GB)."""
    src = Path(MODEL_CONFIGS[model_name]["config_path"])
    cfg = json.loads(src.read_text())
    # ~order of hundreds of MB total params, still trains quickly on 4 GPUs.
    cfg.update(
        {
            "vocab_size": 8192,
            "hidden_size": 1024,
            "moe_intermediate_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "q_lora_rank": 256,
            "o_lora_rank": 256,
            "n_routed_experts": 16,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "index_n_heads": 16,
            "index_head_dim": 128,
            "index_topk": 32,
            "sliding_window": 64,
        }
    )
    out = work_dir / "scaled_config.json"
    out.write_text(json.dumps(cfg, indent=2))
    return out


def _prepare_hf_weights(config_path: Path, hf_dir: Path) -> Path:
    """Materialize HF safetensors so force-HF exercises a real weight load."""
    if (hf_dir / "model.safetensors").exists() or (hf_dir / "model.safetensors.index.json").exists():
        return hf_dir

    shutil.rmtree(hf_dir, ignore_errors=True)
    hf_dir.mkdir(parents=True, exist_ok=True)

    ops = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="fused_triton",
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        swiglu_mlp_implementation="eager",
    )
    model = build_foundation_model(
        config_path=str(config_path),
        weights_path=None,
        torch_dtype="bfloat16",
        init_device="cpu",
        attn_implementation="eager",
        ops_implementation=ops,
    )
    p0 = next(model.parameters())
    if p0.is_meta:
        model.to_empty(device="cpu")
    if hasattr(model, "init_weights"):
        model.init_weights()
    save_model_weights(str(hf_dir), model.state_dict())
    shutil.copy2(config_path, hf_dir / "config.json")
    total = sum(p.stat().st_size for p in hf_dir.rglob("*") if p.is_file())
    logger.info("Prepared HF toy weights at %s (%.2f MB)", hf_dir, total / (1024**2))
    return hf_dir


def _phase_cmd(
    config_path: str,
    tokenizer_path: str,
    model_name: str,
    ep_size: int,
    output_dir: str,
    max_steps: int,
    load_path: Optional[str],
    model_path: Optional[str],
) -> str:
    nproc = get_checkpoint_test_nproc()
    port = find_free_port()
    py = sys.executable
    parts = [
        f"{py} -m torch.distributed.run --nnodes=1 --nproc_per_node={nproc} --master-port={port}",
        "tests/checkpoints/test_resume_memory_profile_compare.py",
        f"--model.config_path {config_path}",
        f"--model.tokenizer_path {tokenizer_path}",
        *resolve_ops_overrides(model_name),
        "--data.train_path dummy",
        "--data.max_seq_len 128",
        f"--train.checkpoint.output_dir {output_dir}",
        "--train.accelerator.fsdp_config.fsdp_mode fsdp2",
        "--train.init_device meta",
        f"--train.accelerator.ep_size {ep_size}",
        f"--train.global_batch_size {nproc}",
        "--train.micro_batch_size 1",
        "--train.optimizer.lr 1e-7",
        "--train.optimizer.lr_warmup_ratio 0.0",
        "--train.optimizer.lr_decay_style constant",
        "--train.optimizer.lr_decay_ratio 1.0",
        "--train.optimizer.weight_decay 0.0",
        "--train.optimizer.max_grad_norm 1.0",
        f"--train.max_steps {max_steps}",
        "--train.checkpoint.manager dcp",
        "--train.checkpoint.save_async False",
        "--train.checkpoint.save_hf_weights False",
        "--train.checkpoint.save_steps 1000000",
        "--train.broadcast_model_weights_from_rank0 true",
        "--train.profile.enable false",
        "--train.gradient_checkpointing.enable false",
    ]
    if model_path is not None:
        parts.append(f"--model.model_path {model_path}")
    if load_path is not None:
        parts.append(f"--train.checkpoint.load_path {load_path}")
    return " \\\n".join(parts)


_VRAM_RE = re.compile(
    r"VRAM usage after (?P<when>building model|epoch \d+): cur (?P<cur>[\d.]+)GB, max (?P<max>[\d.]+)GB"
)
_SKIP_RE = re.compile(r"Skipping pretrained weight load for DCP resume")
_HF_RE = re.compile(
    r"starting to load model weights from|Loading model weights from disk on rank0|FORCE_HF_LOAD_ON_RESUME"
)


def _parse_vram_series(log: str) -> Dict[str, list]:
    """Keep all VRAM samples; later ones after parallelize/DCP are meaningful."""
    out: Dict[str, list] = {}
    for m in _VRAM_RE.finditer(log):
        out.setdefault(m.group("when"), []).append({"cur": float(m.group("cur")), "max": float(m.group("max"))})
    return out


def _stage_peaks(build_samples: list) -> Dict[str, Dict[str, float]]:
    """Map harness print points to stages.

    samples[0]: BaseTrainer freeze (pre-parallelize, usually 0)
    samples[1]: after parallelize / weight materialize
    samples[2]: after optimizer
    samples[3]: after DCP resume (resume runs only)
    """
    def pick(i: int) -> Dict[str, float]:
        return build_samples[i] if len(build_samples) > i else {}

    return {
        "pre_parallelize": pick(0),
        "after_materialize": pick(1),
        "after_optimizer": pick(2),
        "after_dcp_or_last": pick(3) if len(build_samples) > 3 else pick(-1 if build_samples else 0) if build_samples else {},
    }


def _summarize(name: str, log: str) -> Dict[str, Any]:
    series = _parse_vram_series(log)
    builds = series.get("building model") or []
    stages = _stage_peaks(builds)
    return {
        "name": name,
        "skipped_hf": bool(_SKIP_RE.search(log)),
        "loaded_hf": bool(_HF_RE.search(log)),
        "vram_series": series,
        "stages": stages,
        "vram_last_build": builds[-1] if builds else {},
        "vram_epoch": (series.get("epoch 1") or [{}])[-1],
        "dcp_ok": "Load distributed checkpoint from" in log and "successfully" in log,
    }


@pytest.mark.parametrize("model_name", ["deepseek_v4"])
@pytest.mark.parametrize("ep_size", [1])
def test_resume_memory_profile_compare(model_name: str, ep_size: int):
    output_dir = get_output_dir(f"resume_memprof_{model_name}", ep_size)
    shutil.rmtree(output_dir, ignore_errors=True)
    root = Path(output_dir)
    log_dir = root / "logs"
    hf_dir = root / "hf_toy_weights"
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = _scaled_config_path(model_name, root)
    tokenizer_path = hf_local_or_remote(MODEL_CONFIGS[model_name]["tokenizer_path"])
    _prepare_hf_weights(config_path, hf_dir)

    p1 = _phase_cmd(str(config_path), tokenizer_path, model_name, ep_size, output_dir, 2, None, str(hf_dir))
    out1 = _run(p1, env={}, log_path=log_dir / "phase1.log")
    assert "Start training" in out1
    ckpt = _latest_ckpt(output_dir)

    p2a = _phase_cmd(str(config_path), tokenizer_path, model_name, ep_size, output_dir, 4, str(ckpt), str(hf_dir))
    out2a = _run(p2a, env={"VEOMNI_FORCE_HF_LOAD_ON_RESUME": "0"}, log_path=log_dir / "resume_skip_hf.log")
    sum_a = _summarize("skip_hf", out2a)

    p2b = _phase_cmd(str(config_path), tokenizer_path, model_name, ep_size, output_dir, 4, str(ckpt), str(hf_dir))
    out2b = _run(p2b, env={"VEOMNI_FORCE_HF_LOAD_ON_RESUME": "1"}, log_path=log_dir / "resume_force_hf.log")
    sum_b = _summarize("force_hf", out2b)

    print("\n===== Resume memory comparison (built-in print_device_mem_info) =====")
    for s in (sum_a, sum_b):
        print(f"\n[{s['name']}] skipped_hf={s['skipped_hf']} loaded_hf={s['loaded_hf']} dcp_ok={s['dcp_ok']}")
        print(f"  stages: {s['stages']}")
        print(f"  all build samples: {s['vram_series'].get('building model')}")
        print(f"  vram after epoch1: {s['vram_epoch']}")

    assert sum_a["dcp_ok"] and sum_b["dcp_ok"]
    assert sum_a["skipped_hf"] is True
    assert sum_a["loaded_hf"] is False
    assert sum_b["loaded_hf"] is True
    assert "starting to load model weights from" in out2b or "Loading model weights from disk on rank0" in out2b

    a_mat = sum_a["stages"].get("after_materialize") or {}
    b_mat = sum_b["stages"].get("after_materialize") or {}
    assert a_mat and b_mat, f"missing materialize VRAM: {sum_a['stages']} vs {sum_b['stages']}"
    print(
        f"\nDelta after materialize (force_hf - skip_hf): "
        f"cur={b_mat['cur'] - a_mat['cur']:+.3f}GB, max={b_mat['max'] - a_mat['max']:+.3f}GB"
    )
    a_dcp = sum_a["stages"].get("after_dcp_or_last") or {}
    b_dcp = sum_b["stages"].get("after_dcp_or_last") or {}
    if a_dcp and b_dcp:
        print(
            f"Delta after DCP (force_hf - skip_hf): "
            f"cur={b_dcp['cur'] - a_dcp['cur']:+.3f}GB, max={b_dcp['max'] - a_dcp['max']:+.3f}GB"
        )
    if sum_a["vram_epoch"] and sum_b["vram_epoch"]:
        print(
            f"Delta after epoch1 (force_hf - skip_hf): "
            f"cur={sum_b['vram_epoch']['cur'] - sum_a['vram_epoch']['cur']:+.3f}GB, "
            f"max={sum_b['vram_epoch']['max'] - sum_a['vram_epoch']['max']:+.3f}GB"
        )
    # Force-HF process peak after materialize should not be lower than skip-HF.
    assert a_mat["max"] <= b_mat["max"] + 1e-3


if __name__ == "__main__":
    main()
