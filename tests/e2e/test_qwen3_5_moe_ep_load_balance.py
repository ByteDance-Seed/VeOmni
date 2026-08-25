"""Matched Qwen3.5-MoE EP load-balance functional/precision E2E.

This test requires four CUDA or Ascend NPU devices. It uses a full-attention
Qwen3.5-MoE fixture so unrelated GDN packages do not gate the EP contract. Its
two synthetic-hotspot steps are functional/precision evidence only, never
performance evidence.
"""

import gc
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import yaml

from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, get_torch_device


_REQUIRED_DEVICES = 4
_CONFIG_PATH = Path("tests/toy_config/qwen3_5_moe_toy/config.json")
_NPU_GENERATED_MODEL_PATH = Path(
    "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py"
)
_GPU_WORKFLOW_PATH = Path(".github/workflows/gpu_e2e_test.yml")
_NPU_WORKFLOW_PATH = Path(".github/workflows/npu_e2e_test.yml")
_STRICT_E2E_ENV = "VEOMNI_REQUIRE_EP_BALANCE_E2E"
_HARDWARE_E2E_FILE = "tests/e2e/test_qwen3_5_moe_ep_load_balance.py"
_HARDWARE_E2E_NODEID = f"{_HARDWARE_E2E_FILE}::test_qwen3_5_moe_ep_load_balance_matched_precision_and_telemetry"
_RTOL = 0.1
_ATOL = 0.1
_TELEMETRY_KEYS = {
    "before": "moe/ep_rank_imbalance_before/avg",
    "after": "moe/ep_rank_imbalance_after/avg",
    "replicas": "moe/ep_active_replicas/sum",
    "moved_tokens": "moe/ep_moved_tokens/sum",
    "moved_fraction": "moe/ep_moved_token_fraction/avg",
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
    def on_router_forward(self, module, routing_scores, top_indices):
        if module.__class__.__name__ != "Qwen3_5MoeTopKRouter":
            raise RuntimeError(f"unexpected replay router: {module.__class__.__name__}")
        if top_indices.ndim != 2 or top_indices.shape[-1] != 2:
            raise RuntimeError(f"Qwen3.5 toy replay requires top-k=2, got shape={tuple(top_indices.shape)}")
        if routing_scores.device != top_indices.device:
            raise RuntimeError("routing scores and native top-k indices must share a device")
        hotspot = torch.tensor([0, 1], dtype=top_indices.dtype, device=top_indices.device)
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
                "dtype": args.train.accelerator.fsdp_config.mixed_precision.param_dtype,
                "world_size": args.train.world_size,
                "fsdp_mode": args.train.accelerator.fsdp_config.fsdp_mode,
                "sp_size": args.train.accelerator.ulysses_size,
                "ep_size": args.train.accelerator.ep_size,
                "global_batch_size": args.train.global_batch_size,
                "micro_batch_size": args.train.micro_batch_size,
                "max_seq_len": args.data.max_seq_len,
                "gradient_checkpointing": args.train.gradient_checkpointing.enable,
                "run_label": self.run_label,
                "feature_enabled": args.train.moe_ep_load_balance.enabled,
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
        set_active_replay(HotspotReplay())
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


def _write_moe_only_config(output_path: Path) -> None:
    """Keep the Qwen3.5-MoE contract while excluding unrelated GDN dependencies."""
    config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    text_config = config["text_config"]
    text_config["layer_types"] = ["full_attention"] * text_config["num_hidden_layers"]
    text_config["full_attention_interval"] = 1
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


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


class _FakeAccelerator:
    def __init__(self, capabilities=((8, 0),) * _REQUIRED_DEVICES):
        self.capabilities = capabilities

    def is_available(self):
        return True

    def device_count(self):
        return len(self.capabilities)

    def get_device_capability(self, index):
        return self.capabilities[index]


def _configure_gate(monkeypatch, *, cuda, npu, capabilities=((8, 0),) * _REQUIRED_DEVICES, packages=()):
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "IS_CUDA_AVAILABLE", cuda)
    monkeypatch.setattr(module, "IS_NPU_AVAILABLE", npu)
    monkeypatch.setattr(module, "get_torch_device", lambda: _FakeAccelerator(capabilities))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name in packages else None)


def _shell_commands(run_script):
    commands = []
    pending = ""
    for raw_line in run_script.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if pending:
            line = f"{pending} {line}"
        if line.endswith("\\"):
            pending = line[:-1].rstrip()
            continue
        commands.append(shlex.split(line))
        pending = ""
    if pending:
        raise ValueError("unterminated shell line continuation")
    return commands


def _pytest_targets(command):
    if "pytest" not in command:
        return ()
    pytest_index = command.index("pytest")
    return tuple(token for token in command[pytest_index + 1 :] if token.startswith("tests/"))


def _npu_workflow_commands_by_step():
    workflow = yaml.safe_load(_NPU_WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["npu_e2e_tests"]["steps"]
    return [_shell_commands(step.get("run", "")) for step in steps]


def test_npu_workflow_installs_triton_ascend_from_ascend_index_before_no_sync_e2e():
    commands_by_step = _npu_workflow_commands_by_step()
    package = "triton-ascend==3.2.1"
    ascend_index = "https://triton-ascend.osinfra.cn/pypi/simple"

    install_occurrences = [
        (index, command)
        for index, step_commands in enumerate(commands_by_step)
        for command in step_commands
        if command[:3] == ["uv", "pip", "install"] and package in command
    ]
    target_occurrences = [
        (index, command)
        for index, step_commands in enumerate(commands_by_step)
        for command in step_commands
        if _HARDWARE_E2E_NODEID in command
    ]

    assert len(install_occurrences) == 1, "the NPU workflow must install pinned triton-ascend once"
    assert len(target_occurrences) == 1, "the NPU workflow must execute the Qwen3.5 E2E exactly once"
    install_index, install_command = install_occurrences[0]
    assert "--python" in install_command
    assert install_command[install_command.index("--python") + 1] == ".venv/bin/python"
    extra_index_values = []
    for index, token in enumerate(install_command):
        if token == "--extra-index-url":
            extra_index_values.append(install_command[index + 1] if index + 1 < len(install_command) else None)
        elif token.startswith("--extra-index-url="):
            extra_index_values.append(token.split("=", 1)[1])
    assert extra_index_values == [ascend_index], "the pinned package must use the documented Ascend index"

    e2e_index, _ = target_occurrences[0]
    assert install_index < e2e_index
    assert all(
        command[:2] != ["uv", "sync"]
        for step_commands in commands_by_step[install_index + 1 : e2e_index + 1]
        for command in step_commands
    ), "uv sync must not remove triton-ascend between installation and the E2E"


def test_npu_workflow_exposes_project_venv_to_torchrun_children_without_resync():
    commands_by_step = _npu_workflow_commands_by_step()
    target_commands = [
        command for step_commands in commands_by_step for command in step_commands if _HARDWARE_E2E_NODEID in command
    ]

    assert target_commands == [
        [
            "PATH=$PWD/.venv/bin:$PATH",
            ".venv/bin/python",
            "-m",
            "pytest",
            "-v",
            "-s",
            "-x",
            _HARDWARE_E2E_NODEID,
        ]
    ]


@pytest.mark.parametrize(
    ("workflow_path", "job_name"),
    [
        (_GPU_WORKFLOW_PATH, "gpu_e2e_tests_v5"),
        (_NPU_WORKFLOW_PATH, "npu_e2e_tests"),
    ],
)
def test_dedicated_accelerator_workflow_runs_only_exact_hardware_nodeid_in_strict_mode(workflow_path, job_name):
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    module_invocations = [
        (step, command, targets)
        for step in workflow["jobs"][job_name]["steps"]
        for command in _shell_commands(step.get("run", ""))
        if (targets := _pytest_targets(command))
        if any(target == _HARDWARE_E2E_FILE or target.startswith(f"{_HARDWARE_E2E_FILE}::") for target in targets)
    ]

    assert len(module_invocations) == 1
    target_step, target_command, targets = module_invocations[0]
    assert target_step.get("env", {}).get(_STRICT_E2E_ENV) == "1"
    assert targets == (_HARDWARE_E2E_NODEID,)
    assert target_command.count(_HARDWARE_E2E_NODEID) == 1


def test_local_prerequisite_failure_remains_an_honest_skip(monkeypatch):
    monkeypatch.delenv(_STRICT_E2E_ENV, raising=False)
    _configure_gate(monkeypatch, cuda=False, npu=False)

    with pytest.raises(
        pytest.skip.Exception,
        match="requires at least 4 CUDA or NPU devices; found 0 supported devices",
    ):
        _require_accelerators()


@pytest.mark.parametrize(
    "scenario",
    [
        "no_accelerator",
        "insufficient_count",
        "rocm",
        "insufficient_sm",
        "missing_cuda_package",
        "missing_npu_package",
    ],
)
def test_strict_e2e_turns_every_prerequisite_unavailability_into_failure(monkeypatch, scenario):
    cuda_packages = {"triton", "fla", "flash_attn", "liger_kernel"}
    if scenario == "no_accelerator":
        _configure_gate(monkeypatch, cuda=False, npu=False)
    elif scenario == "insufficient_count":
        _configure_gate(monkeypatch, cuda=True, npu=False, capabilities=((8, 0),) * 3, packages=cuda_packages)
    elif scenario == "rocm":
        _configure_gate(monkeypatch, cuda=True, npu=False, packages=cuda_packages)
        monkeypatch.setattr(torch.version, "hip", "6.3", raising=False)
    elif scenario == "insufficient_sm":
        _configure_gate(
            monkeypatch,
            cuda=True,
            npu=False,
            capabilities=((8, 0), (7, 5), (6, 1), (8, 0)),
            packages=cuda_packages,
        )
    elif scenario == "missing_cuda_package":
        _configure_gate(monkeypatch, cuda=True, npu=False, packages=cuda_packages - {"liger_kernel"})
    else:
        _configure_gate(monkeypatch, cuda=False, npu=True, packages={"torch_npu"})
    monkeypatch.setenv(_STRICT_E2E_ENV, "1")

    with pytest.raises((pytest.skip.Exception, pytest.fail.Exception)) as outcome:
        _require_accelerators()

    assert isinstance(outcome.value, pytest.fail.Exception), f"strict scenario {scenario} must fail, not skip"


def test_npu_backend_uses_executable_minimum_dependency_gdn_kernels(monkeypatch):
    _configure_gate(monkeypatch, cuda=False, npu=True)

    assert _backend_args() == [
        "--train.dyn_bsz=False",
        "--model.ops_implementation.rms_norm_gated_implementation=npu",
        "--model.ops_implementation.causal_conv1d_implementation=npu",
        "--model.ops_implementation.chunk_gated_delta_rule_implementation=npu",
    ]


def test_npu_gate_names_missing_triton_ascend_import(monkeypatch):
    _configure_gate(monkeypatch, cuda=False, npu=True, packages={"torch_npu"})

    with pytest.raises(
        pytest.skip.Exception,
        match=r"requires Ascend fused NPU/Triton; missing package\(s\): triton \(provided by triton-ascend\)",
    ):
        _require_accelerators()


def test_cuda_gate_rejects_rocm_before_package_checks(monkeypatch):
    _configure_gate(monkeypatch, cuda=True, npu=False)
    monkeypatch.setattr(torch.version, "hip", "6.3", raising=False)

    with pytest.raises(
        pytest.skip.Exception,
        match="requires NVIDIA CUDA; ROCm is unsupported",
    ):
        _require_accelerators()


def test_cuda_gate_requires_sm70_on_each_selected_device(monkeypatch):
    _configure_gate(
        monkeypatch,
        cuda=True,
        npu=False,
        capabilities=((8, 0), (7, 5), (6, 1), (8, 0)),
        packages={"triton", "fla", "flash_attn", "liger_kernel"},
    )

    with pytest.raises(
        pytest.skip.Exception,
        match=r"requires compute capability >= 7\.0 on all 4 CUDA devices; device 2 has 6\.1",
    ):
        _require_accelerators()


def test_cuda_gate_requires_liger_kernel_default_dependency(monkeypatch):
    _configure_gate(
        monkeypatch,
        cuda=True,
        npu=False,
        packages={"triton", "fla", "flash_attn"},
    )

    with pytest.raises(
        pytest.skip.Exception,
        match=r"requires NVIDIA CUDA fused Triton/FLA/Liger; missing package\(s\): liger_kernel",
    ):
        _require_accelerators()


def test_moe_only_config_preserves_expert_contract_without_gdn(tmp_path):
    output_path = tmp_path / "config.json"

    _write_moe_only_config(output_path)

    source = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["text_config"]["num_experts"] == source["text_config"]["num_experts"] == 16
    assert result["text_config"]["num_experts_per_tok"] == source["text_config"]["num_experts_per_tok"] == 2
    assert result["text_config"]["layer_types"] == ["full_attention"] * result["text_config"]["num_hidden_layers"]
    assert result["text_config"]["full_attention_interval"] == 1


def test_npu_generated_model_only_precomputes_gdn_metadata_when_needed():
    source = _NPU_GENERATED_MODEL_PATH.read_text(encoding="utf-8")

    assert source.count('"linear_attention" in self.config.layer_types') == 1


def test_qwen3_5_moe_ep_load_balance_matched_precision_and_telemetry(tmp_path):
    _require_accelerators()

    from tests.tools import DummyDataset, ParallelConfig, build_torchrun_cmd

    toy_config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    assert toy_config["text_config"]["num_experts"] == 16
    assert toy_config["text_config"]["num_experts_per_tok"] == 2

    wrapper = tmp_path / "train_qwen3_5_moe_ep_balance.py"
    wrapper.write_text(textwrap.dedent(_WRAPPER_SOURCE), encoding="utf-8")
    config_path = tmp_path / "qwen3_5_moe_ep_balance_config.json"
    _write_moe_only_config(config_path)
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
        "--train.profile.enable=False",
        "--train.wandb.enable=False",
        "--train.gradient_checkpointing.enable=False",
        "--train.moe_load_balance_monitor_interval=1",
        "--train.moe_ep_load_balance.max_replicas_per_rank=1",
        "--train.accelerator.fsdp_config.mixed_precision.enable=True",
        "--train.accelerator.fsdp_config.mixed_precision.param_dtype=bfloat16",
        "--train.accelerator.fsdp_config.mixed_precision.reduce_dtype=float32",
        "--train.accelerator.fsdp_config.mixed_precision.cast_forward_inputs=True",
        *_backend_args(),
        "--train.moe_ep_load_balance.enabled=False",
    ]
    parallel = ParallelConfig(sp_size=1, ep_size=2, fsdp_mode="fsdp2")
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

    before = _finite_curve(candidate, _TELEMETRY_KEYS["before"])
    after = _finite_curve(candidate, _TELEMETRY_KEYS["after"])
    replicas = _finite_curve(candidate, _TELEMETRY_KEYS["replicas"])
    moved_tokens = _finite_curve(candidate, _TELEMETRY_KEYS["moved_tokens"])
    moved_fraction = _finite_curve(candidate, _TELEMETRY_KEYS["moved_fraction"])
    assert len(before) == len(after) == len(replicas) == len(moved_tokens) == len(moved_fraction) == 2
    assert all(value > 0 for value in replicas)
    assert all(value > 0 for value in moved_tokens)
    assert all(value > 0 for value in moved_fraction)
    assert all(after_value <= before_value for before_value, after_value in zip(before, after, strict=True))
    assert any(after_value < before_value for before_value, after_value in zip(before, after, strict=True))
