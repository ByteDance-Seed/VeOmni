"""Matched Qwen3.5-MoE EP load-balance functional/precision E2E.

This test requires four CUDA or Ascend NPU devices and the native hybrid
Qwen3.5-MoE fixture. Its
two synthetic-hotspot steps are functional/precision evidence only, never
performance evidence.
"""

import gc
import importlib.util
import json
import math
import os
import subprocess
import textwrap
from pathlib import Path

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, get_torch_device


_REQUIRED_DEVICES = 4
_CONFIG_PATH = Path("tests/toy_config/qwen3_5_moe_toy/config.json")
_STRICT_E2E_ENV = "VEOMNI_REQUIRE_EP_BALANCE_E2E"
_RTOL = 0.1
_ATOL = 0.1
_TELEMETRY_KEYS = {
    "before": "moe/ep_rank_imbalance_before/avg",
    "after": "moe/ep_rank_imbalance_after/avg",
    "replicas": "moe/ep_active_replicas/sum",
    "moved_tokens": "moe/ep_moved_tokens/sum",
    "moved_fraction": "moe/ep_moved_token_fraction/avg",
    "total_tokens": "moe/ep_total_routed_tokens/sum",
}

_WRAPPER_SOURCE = r"""
import json
import math
import numbers
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.environ["VEOMNI_REPO_ROOT"])

from tests.train_scripts.train_vlm_test import TestVLMTrainer
from veomni.arguments import parse_args
from veomni.trainer.callbacks import Callback
from veomni.trainer.vlm_trainer import VeOmniVLMArguments
from veomni.utils.moe_router_replay import get_active_replay, set_active_replay


class HotspotReplay:
    def __init__(self, ep_size):
        # EP2 has one hot owner; EP4 has two tied hot owners.
        self.experts = [0, 1] if ep_size == 2 else [0, 4]

    def on_router_forward(self, module, routing_scores, top_indices):
        if module.__class__.__name__ != "Qwen3_5MoeTopKRouter":
            raise RuntimeError(f"unexpected replay router: {module.__class__.__name__}")
        if top_indices.ndim != 2 or top_indices.shape[-1] != 2:
            raise RuntimeError(f"Qwen3.5 toy replay requires top-k=2, got shape={tuple(top_indices.shape)}")
        if routing_scores.device != top_indices.device:
            raise RuntimeError("routing scores and native top-k indices must share a device")
        hotspot = torch.tensor(self.experts, dtype=top_indices.dtype, device=top_indices.device)
        return hotspot.expand_as(top_indices).clone()


def finite_scalar(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().item()
    if not isinstance(value, numbers.Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class EnvelopeLogSink(Callback):
    def __init__(self, trainer, run_label):
        super().__init__(trainer)
        self.run_label = run_label
        self.metrics = defaultdict(list)

    def on_step_end(self, state, loss, loss_dict, grad_norm, **kwargs):
        for name, value in (("loss", loss), ("grad_norm", grad_norm)):
            scalar = finite_scalar(value)
            if scalar is None:
                raise RuntimeError(f"{name} must be a finite scalar")
            self.metrics[name].append(scalar)
        for name, value in self.trainer.step_env_metrics.items():
            scalar = finite_scalar(value)
            if scalar is not None:
                self.metrics[name].append(scalar)

    def on_train_end(self, state, **kwargs):
        args = self.trainer.args
        if args.train.global_rank != 0:
            return
        output_dir = args.train.checkpoint.output_dir
        os.makedirs(output_dir, exist_ok=True)
        envelope = {
            "schema_version": 1,
            "metadata": {
                "model": "qwen3_5_moe",
                "model_config": str(args.model.config_path),
                "checkpoint": str(args.model.model_path),
                "dataset": str(args.data.train_path),
                "seed": args.train.seed,
                "dtype": args.model.accelerator.fsdp_config.mixed_precision.param_dtype,
                "world_size": args.train.world_size,
                "fsdp_mode": args.model.accelerator.fsdp_config.fsdp_mode,
                "sp_size": args.model.accelerator.ulysses_size,
                "ep_size": args.model.accelerator.ep_size,
                "global_batch_size": args.train.global_batch_size,
                "micro_batch_size": args.train.micro_batch_size,
                "max_seq_len": args.data.max_seq_len,
                "gradient_checkpointing": args.model.accelerator.gradient_checkpointing.enable,
                "run_label": self.run_label,
                "feature_enabled": args.train.moe_ep_load_balance.enabled,
                "optimize_memory": args.train.moe_ep_load_balance.optimize_memory,
            },
            "metrics": dict(self.metrics),
        }
        with open(os.path.join(output_dir, "moe_ep_balance_metrics.json"), "w", encoding="utf-8") as output:
            json.dump(envelope, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")


def main():
    args = parse_args(VeOmniVLMArguments)
    trainer = TestVLMTrainer(args)
    trainer.base.logdictsave_callback = EnvelopeLogSink(
        trainer.base,
        os.environ["VEOMNI_EP_BALANCE_RUN_LABEL"],
    )
    previous_replay = get_active_replay()
    try:
        set_active_replay(HotspotReplay(args.model.accelerator.ep_size))
        trainer.train()
    finally:
        set_active_replay(previous_replay)


if __name__ == "__main__":
    main()
"""


def _prerequisite_unavailable(reason: str) -> None:
    if os.environ.get(_STRICT_E2E_ENV) == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def _require_accelerators() -> None:
    if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE):
        _prerequisite_unavailable(
            "Qwen3.5-MoE EP balance E2E requires at least 4 CUDA or NPU devices; found 0 supported devices"
        )
    if IS_CUDA_AVAILABLE and getattr(torch.version, "hip", None) is not None:
        _prerequisite_unavailable("Qwen3.5-MoE EP balance E2E requires NVIDIA CUDA; ROCm is unsupported")

    torch_device = get_torch_device()
    count = torch_device.device_count() if torch_device.is_available() else 0
    if count < _REQUIRED_DEVICES:
        _prerequisite_unavailable(
            f"Qwen3.5-MoE EP balance E2E requires at least 4 {get_device_type().upper()} devices; found {count}"
        )

    if IS_CUDA_AVAILABLE:
        unsupported = []
        for index in range(_REQUIRED_DEVICES):
            major, minor = torch_device.get_device_capability(index)
            if (major, minor) < (7, 0):
                unsupported.append(f"device {index} has {major}.{minor}")
        if unsupported:
            _prerequisite_unavailable(
                "Qwen3.5-MoE EP balance E2E requires compute capability >= 7.0 "
                f"on all 4 CUDA devices; {', '.join(unsupported)}"
            )
        requirements = tuple((name, name) for name in ("triton", "fla", "flash_attn", "liger_kernel"))
        backend = "NVIDIA CUDA fused Triton/FLA/Liger"
    else:
        requirements = (
            ("torch_npu", "torch_npu"),
            ("triton", "triton (provided by triton-ascend)"),
        )
        backend = "Ascend fused NPU/Triton"

    missing = [label for import_name, label in requirements if importlib.util.find_spec(import_name) is None]
    if missing:
        _prerequisite_unavailable(
            f"Qwen3.5-MoE EP balance E2E requires {backend}; missing package(s): {', '.join(missing)}"
        )


def _write_model_config(output_path: Path) -> None:
    output_path.write_bytes(_CONFIG_PATH.read_bytes())


def _materialize_checkpoint(config_path: Path, output_path: Path) -> None:
    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import empty_cache

    torch.manual_seed(0)
    model = build_foundation_model(
        config_path=str(config_path),
        weights_path=None,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=make_eager_ops_config(),
    )
    model.save_pretrained(output_path, save_original_format=False)
    del model
    gc.collect()
    empty_cache()


def _backend_args() -> list[str]:
    if IS_NPU_AVAILABLE:
        return [
            "--train.dyn_bsz=False",
            "--model.ops_implementation.rms_norm_gated_implementation=npu",
            "--model.ops_implementation.causal_conv1d_implementation=npu",
            "--model.ops_implementation.chunk_gated_delta_rule_implementation=npu",
        ]
    return [
        "--train.dyn_bsz=True",
        "--model.ops_implementation.moe_implementation=fused_triton",
        "--model.ops_implementation.rms_norm_gated_implementation=fla",
        "--model.ops_implementation.causal_conv1d_implementation=fla",
        "--model.ops_implementation.chunk_gated_delta_rule_implementation=fla",
    ]


def _replace_exact_arg(command: list[str], old: str, new: str) -> list[str]:
    if command.count(old) != 1:
        raise AssertionError(f"expected exactly one command argument {old!r}, got {command.count(old)}")
    return [new if item == old else item for item in command]


def _load_envelope(output_dir: Path) -> dict:
    with (output_dir / "moe_ep_balance_metrics.json").open(encoding="utf-8") as metrics_file:
        envelope = json.load(metrics_file)
    assert envelope["schema_version"] == 1
    assert isinstance(envelope["metadata"], dict)
    assert isinstance(envelope["metrics"], dict)
    return envelope


def _finite_curve(envelope: dict, name: str) -> list[float]:
    curve = envelope["metrics"].get(name)
    assert isinstance(curve, list) and curve, f"{name} must be a non-empty curve"
    assert all(not isinstance(value, bool) and isinstance(value, (int, float)) for value in curve)
    assert all(math.isfinite(value) for value in curve)
    return curve


@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("optimize_memory,checkpointing", [(False, False), (True, False), (True, True)])
def test_qwen3_5_moe_ep_load_balance_matched_precision_and_telemetry(
    tmp_path, ep_size, optimize_memory, checkpointing
):
    _require_accelerators()

    from tests.tools import DummyDataset, ParallelConfig, build_torchrun_cmd

    toy_config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    assert toy_config["text_config"]["num_experts"] == 16
    assert toy_config["text_config"]["num_experts_per_tok"] == 2

    wrapper = tmp_path / "train_qwen3_5_moe_ep_balance.py"
    wrapper.write_text(textwrap.dedent(_WRAPPER_SOURCE), encoding="utf-8")
    config_path = tmp_path / "qwen3_5_moe_ep_balance_config.json"
    _write_model_config(config_path)
    checkpoint_dir = tmp_path / "checkpoint"
    _materialize_checkpoint(config_path, checkpoint_dir)
    dataset = DummyDataset(
        num_samples=16,
        seq_len=2048,
        dataset_type="qwen3vl",
        cache_name=f"qwen3_5_moe_ep_balance_{os.getpid()}",
    )

    baseline_output = tmp_path / "baseline"
    candidate_output = tmp_path / "candidate"
    common_args = [
        "--train.seed=0",
        "--data.max_seq_len=2048",
        "--train.profile.enable=False",
        "--train.wandb.enable=False",
        f"--model.accelerator.gradient_checkpointing.enable={checkpointing}",
        "--train.moe_load_balance_monitor_interval=1",
        "--train.moe_ep_load_balance.max_replicas_per_rank=1",
        "--model.accelerator.fsdp_config.mixed_precision.enable=True",
        "--model.accelerator.fsdp_config.mixed_precision.param_dtype=bfloat16",
        "--model.accelerator.fsdp_config.mixed_precision.reduce_dtype=float32",
        "--model.accelerator.fsdp_config.mixed_precision.cast_forward_inputs=True",
        *_backend_args(),
        "--train.moe_ep_load_balance.enabled=False",
        "--train.moe_ep_load_balance.optimize_memory=False",
    ]
    parallel = ParallelConfig(sp_size=1, ep_size=ep_size, fsdp_mode="fsdp2")
    baseline_command = build_torchrun_cmd(
        script=str(wrapper),
        config_path=str(config_path),
        model_path=str(checkpoint_dir),
        train_path=dataset.save_path,
        output_dir=str(baseline_output),
        parallel_config=parallel,
        extra_args=common_args,
        nproc=_REQUIRED_DEVICES,
        init_device="meta",
        model_name="qwen3_5_moe",
    )
    candidate_command = _replace_exact_arg(
        baseline_command,
        f"--train.checkpoint.output_dir={baseline_output}",
        f"--train.checkpoint.output_dir={candidate_output}",
    )
    candidate_command = _replace_exact_arg(
        candidate_command,
        "--train.moe_ep_load_balance.enabled=False",
        "--train.moe_ep_load_balance.enabled=True",
    )
    candidate_command = _replace_exact_arg(
        candidate_command,
        "--train.moe_ep_load_balance.optimize_memory=False",
        f"--train.moe_ep_load_balance.optimize_memory={optimize_memory}",
    )

    base_env = os.environ.copy()
    base_env["CUDA_LAUNCH_BLOCKING"] = "1"
    base_env["VEOMNI_REPO_ROOT"] = str(Path.cwd())
    try:
        for run_label, command in (("baseline", baseline_command), ("candidate", candidate_command)):
            run_env = dict(base_env)
            run_env["VEOMNI_EP_BALANCE_RUN_LABEL"] = run_label
            subprocess.run(command, check=True, env=run_env)
    finally:
        del dataset

    baseline = _load_envelope(baseline_output)
    candidate = _load_envelope(candidate_output)
    for name in ("loss", "grad_norm"):
        baseline_curve = _finite_curve(baseline, name)
        candidate_curve = _finite_curve(candidate, name)
        assert len(baseline_curve) == len(candidate_curve) == 2
        torch.testing.assert_close(candidate_curve, baseline_curve, rtol=_RTOL, atol=_ATOL)

    assert baseline["metadata"]["run_label"] == "baseline"
    assert baseline["metadata"]["feature_enabled"] is False
    assert candidate["metadata"]["run_label"] == "candidate"
    assert candidate["metadata"]["feature_enabled"] is True
    assert baseline["metadata"]["optimize_memory"] is False
    assert candidate["metadata"]["optimize_memory"] is optimize_memory
    assert baseline["metadata"]["gradient_checkpointing"] is checkpointing
    assert candidate["metadata"]["gradient_checkpointing"] is checkpointing

    before = _finite_curve(candidate, _TELEMETRY_KEYS["before"])
    after = _finite_curve(candidate, _TELEMETRY_KEYS["after"])
    replicas = _finite_curve(candidate, _TELEMETRY_KEYS["replicas"])
    moved_tokens = _finite_curve(candidate, _TELEMETRY_KEYS["moved_tokens"])
    moved_fraction = _finite_curve(candidate, _TELEMETRY_KEYS["moved_fraction"])
    total_tokens = _finite_curve(candidate, _TELEMETRY_KEYS["total_tokens"])
    assert len(before) == len(after) == len(replicas) == len(moved_tokens) == len(moved_fraction) == 2
    assert all(value > 0 for value in replicas)
    assert all(value > 0 for value in moved_tokens)
    assert all(value > 0 for value in moved_fraction)
    assert all(after_value <= before_value for before_value, after_value in zip(before, after, strict=True))
    assert any(after_value < before_value for before_value, after_value in zip(before, after, strict=True))
    # Count physical execution, including checkpoint recomputation, once per
    # EP group. Both fixtures move exactly half the assignments.
    forward_calls = 2 if checkpointing else 1
    expected_total = toy_config["text_config"]["num_hidden_layers"] * 8 * 2048 * 2 * forward_calls
    assert total_tokens == [expected_total] * 2
    assert moved_tokens == [expected_total // 2] * 2
    # Two accumulated microbatches, four layers: EP2 has two groups with
    # one replica each; EP4 has one group with two replicas.
    assert replicas == [16 * forward_calls] * 2


@pytest.mark.parametrize("strict", [False, True])
def test_required_ci_cannot_silently_skip_missing_hardware(monkeypatch, strict):
    monkeypatch.setenv(_STRICT_E2E_ENV, "1" if strict else "0")
    expected = pytest.fail.Exception if strict else pytest.skip.Exception
    with pytest.raises(expected):
        _prerequisite_unavailable("Missing accelerators")
