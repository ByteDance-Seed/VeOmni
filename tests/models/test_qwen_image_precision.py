"""Validate portable Qwen RoPE and the cross-device precision artifact contract."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import CheckpointError, _CheckpointFrame

from scripts.ci.launch_plan import CasePlan, LaunchSpec, runner_device_count
from tests.models.test_qwen_image_training import cpu_diffusers_normalization, snapshot, transformer
from tests.precision import qwen_image as precision_module
from tests.precision.qwen_image import capture, compare, encode, prepare, sha256, tensor_difference, verify_fixture
from veomni.arguments.arguments_types import MixedPrecisionConfig
from veomni.distributed.parallel_state import ParallelState, use_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.diffusers.qwen_image.qwen_image_transformer import modeling_qwen_image_transformer as modeling
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type, get_torch_device


_SINGLE_LAUNCH = LaunchSpec(nproc=1)
_SHARDED_LAUNCH = LaunchSpec(nproc=2)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("additive", [False, True])
@pytest.mark.parametrize("four_dimensional", [False, True])
def test_npu_attention_mask_layout_preserves_outputs_and_gradients(additive, four_dimensional):
    generator = torch.Generator().manual_seed(19)
    mask = torch.tensor([[True] * 7 + [False] * 2, [True] * 5 + [False] * 4])
    if additive:
        mask = torch.zeros_like(mask, dtype=torch.float64).masked_fill(~mask, -torch.inf)
    reference_mask = mask[:, None, None, :]
    source_mask = reference_mask if four_dimensional else mask
    actual_mask = modeling._prepare_qwen_attention_mask(source_mask, query_length=7, device_type="npu", batch_size=2)
    assert actual_mask.shape == (2, 1, 7, 9)
    assert actual_mask.is_contiguous()
    assert modeling._prepare_qwen_attention_mask(source_mask, 7, "cpu", 2) is source_mask
    assert modeling._prepare_qwen_attention_mask(source_mask, 7, "cuda", 2) is source_mask
    assert modeling._prepare_qwen_attention_mask(None, 7, "npu", 2) is None
    assert modeling._prepare_qwen_attention_mask(actual_mask, 7, "npu", 2) is actual_mask
    query_mask = torch.ones(7, 9, dtype=torch.bool).tril()
    assert modeling._prepare_qwen_attention_mask(query_mask, 7, "npu", 2) is query_mask
    torch.testing.assert_close(actual_mask, reference_mask.expand(2, 1, 7, 9), rtol=0, atol=0)

    inputs = [torch.randn(2, 3, length, 8, generator=generator, dtype=torch.float64) for length in (7, 9, 9)]
    expected_inputs = [tensor.clone().requires_grad_() for tensor in inputs]
    actual_inputs = [tensor.clone().requires_grad_() for tensor in inputs]
    expected = torch.nn.functional.scaled_dot_product_attention(*expected_inputs, attn_mask=reference_mask)
    actual = torch.nn.functional.scaled_dot_product_attention(*actual_inputs, attn_mask=actual_mask)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for candidate, reference in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-10, atol=1e-12)


def test_npu_mask_prepared_once_for_all_qwen_blocks(monkeypatch, cpu_diffusers_normalization):
    model = transformer()
    batch = dict(
        hidden_states=torch.randn(1, 4, 16),
        encoder_hidden_states=torch.randn(1, 3, 32),
        encoder_hidden_states_mask=torch.tensor([[True, True, False]]),
        timestep=torch.tensor([0.5]),
        img_shapes=[[(1, 2, 2)]],
        return_dict=False,
    )
    with use_parallel_state(ParallelState(device_type="cpu")):
        expected = modeling.QwenImageTransformer2DModel_forward(model, **batch)[0]
        expected.square().mean().backward()
        reference_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        processor = model.transformer_blocks[0].attn.processor
        original_prepare = modeling._prepare_qwen_attention_mask
        original_dispatch = processor.veomni_attn_masked
        prepared_masks = []
        dispatched_masks = []

        def prepare_on_npu(mask, query_length, device_type, batch_size):
            prepared_masks.append(original_prepare(mask, query_length, "npu", batch_size))
            return prepared_masks[-1]

        def dispatch_with_npu_contract(module, query, key, value, attn_mask=None, **kwargs):
            assert attn_mask.shape == (query.shape[0], 1, query.shape[2], key.shape[2])
            dispatched_masks.append(attn_mask)
            return original_dispatch(module, query, key, value, attn_mask, **kwargs)

        monkeypatch.setattr(modeling, "_prepare_qwen_attention_mask", prepare_on_npu)
        monkeypatch.setattr(processor, "veomni_attn_masked", dispatch_with_npu_contract)
        actual = modeling.QwenImageTransformer2DModel_forward(model, **batch)[0]
        assert len(prepared_masks) == 1
        assert len(dispatched_masks) == model.config.num_layers
        assert all(mask is prepared_masks[0] for mask in dispatched_masks)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        actual.square().mean().backward()
        for name, parameter in model.named_parameters():
            if name in reference_grads:
                torch.testing.assert_close(parameter.grad, reference_grads[name], rtol=1e-5, atol=2e-6, msg=name)
        batch["encoder_hidden_states"][:, -1] = 1000
        perturbed = modeling.QwenImageTransformer2DModel_forward(model, **batch)[0]
        torch.testing.assert_close(perturbed, actual, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_real_rope_forward_and_gradients(dtype):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2, 7, 3, 16, generator=generator).to(dtype).requires_grad_(True)
    y = x.detach().clone().requires_grad_(True)
    angles = torch.randn(7, 8, generator=generator)
    freqs = torch.polar(torch.ones_like(angles), angles)
    expected = modeling.apply_qwen_rotary_emb(x, freqs)
    actual = modeling.apply_qwen_rotary_emb(y, torch.view_as_real(freqs))
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6 if dtype == torch.float32 else 4e-3,
        rtol=1e-5 if dtype == torch.float32 else 4e-3,
    )
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(
        y.grad,
        x.grad,
        atol=1e-6 if dtype == torch.float32 else 1e-4,
        rtol=1e-5 if dtype == torch.float32 else 1e-2,
    )


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU rotary parity requires torch_npu")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_rope_parity_with_cpu(dtype):
    generator = torch.Generator().manual_seed(23)
    x_cpu = torch.randn(2, 7, 3, 16, generator=generator).to(dtype).requires_grad_(True)
    angles = torch.randn(7, 8, generator=generator)
    freqs = torch.polar(torch.ones_like(angles), angles)
    freqs_real = torch.view_as_real(freqs)

    expected = modeling.apply_qwen_rotary_emb(x_cpu, freqs_real)

    x_npu = x_cpu.detach().clone().to("npu").requires_grad_(True)
    freqs_npu = freqs_real.to("npu")
    actual = modeling.apply_qwen_rotary_emb(x_npu, freqs_npu)

    torch.testing.assert_close(
        actual.to("cpu"),
        expected,
        atol=1e-5 if dtype == torch.float32 else 5e-3,
        rtol=1e-5 if dtype == torch.float32 else 5e-3,
    )

    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(
        x_npu.grad.to("cpu"),
        x_cpu.grad,
        atol=1e-5 if dtype == torch.float32 else 1e-3,
        rtol=1e-5 if dtype == torch.float32 else 1e-2,
    )


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend for real frequency transfer")
def test_npu_rotary_frequency_builder():
    pos_embed = transformer().pos_embed
    shapes = [[(1, 2, 2)]]
    expected = pos_embed(shapes, max_txt_seq_len=3, device=torch.device("cpu"))
    actual = modeling.qwen_image_rotary_frequencies(pos_embed, shapes, 3, torch.device("npu"))
    for reference, frequencies in zip(expected, actual):
        assert frequencies.device.type == "npu"
        assert not frequencies.is_complex()
        torch.testing.assert_close(frequencies.cpu(), torch.view_as_real(reference), rtol=0, atol=0)


def test_parallel_builder_has_no_removed_chunk_dependency():
    model = torch.nn.Linear(4, 4)
    with use_parallel_state(ParallelState(device_type="cpu")):
        # This path returns an already-built model unchanged; it allocates no accelerator tensors.
        result = build_parallelize_model(
            model,
            init_device="cuda",
            mixed_precision=MixedPrecisionConfig(enable=False),
            enable_gradient_checkpointing=False,
        )
    assert result is model


def test_parallel_builder_preserves_checkpoint_validation(monkeypatch):
    # Restore global state even when checking an older builder that disables validation.
    monkeypatch.setattr(
        _CheckpointFrame, "check_recomputed_tensors_match", _CheckpointFrame.check_recomputed_tensors_match
    )
    model = transformer()
    with use_parallel_state(ParallelState(device_type="cpu")):
        build_parallelize_model(
            model,
            init_device="cuda",
            mixed_precision=MixedPrecisionConfig(enable=False),
            enable_gradient_checkpointing=True,
        )
    recompute = False

    def inconsistent_forward(value):
        return (value.double() if recompute else value).sin()

    value = torch.randn(4, requires_grad=True)
    loss = model._gradient_checkpointing_func(inconsistent_forward, value).sum()
    recompute = True
    with pytest.raises(CheckpointError, match="different metadata"):
        loss.backward()


@pytest.fixture(scope="module")
def runs(tmp_path_factory, cpu_diffusers_normalization):
    root = tmp_path_factory.mktemp("qwen_precision")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        prepare(ROOT / "tests/toy_config/qwen_image_toy/config.json", root / "fixture", grid=2, text_tokens=5)
        capture(root / "fixture", root / "reference", device="cpu", backend="diffusers")
        capture(root / "fixture", root / "candidate", device="cpu", backend="veomni")
        yield root
    finally:
        torch.set_num_threads(previous_threads)


def test_fixed_step_toolchain(runs):
    report = compare(
        runs / "reference",
        runs / "candidate",
        runs / "comparison",
        atol=2e-6,
        rtol=1e-4,
        rationale="CPU FP32 regression bound",
    )
    assert report["conclusion"] == "PASS"
    assert report["pairs"][0]["baseline"] == "cpu"
    assert report["pairs"][0]["first_divergence"] is None
    assert not (runs / "comparison/accuracy_report.json").exists()
    assert (runs / "comparison/comparison_report.json").exists()
    for kind in ("activation", "prediction", "loss", "gradient", "initial", "update"):
        assert any(f"/{kind}" in name for name in report["metrics"])
    manifest = json.loads((runs / "candidate/run_manifest.json").read_text())
    updates = [record for name, record in manifest["tensors"].items() if "/update/" in name]
    assert any(load_file(runs / "candidate" / record["path"])["value"].abs().sum() > 0 for record in updates)


@pytest.mark.skipif(
    get_device_type() not in ("cuda", "npu") or runner_device_count() < 2,
    reason="requires two GPU or NPU devices for FSDP2 capture",
)
@pytest.mark.launch_plan(provider=CasePlan.sequential(_SINGLE_LAUNCH, _SHARDED_LAUNCH))
def test_accelerator_capture_matches_two_rank_fsdp2(runs, tmp_path):
    # Fresh processes use the real accelerator kernels without CPU-test fixtures.
    environment = os.environ.copy()
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    for ranks, name in ((_SINGLE_LAUNCH.nproc, "single"), (_SHARDED_LAUNCH.nproc, "sharded")):
        command = [sys.executable]
        if ranks > 1:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={_SHARDED_LAUNCH.nproc}",
            ]
        command += [
            "-m",
            "tests.precision.qwen_image",
            "run",
            "--fixture",
            str(runs / "fixture"),
            "--output",
            str(tmp_path / name),
            "--device",
            get_device_type(),
            "--steps",
            "2",
        ]
        completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=300)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    report = compare(
        tmp_path / "single",
        tmp_path / "sharded",
        tmp_path / "comparison",
        atol=2e-5,
        rtol=2e-4,
        rationale="FP32 single-device vs FSDP2 replicated-input regression bound",
    )
    assert report["conclusion"] == "PASS"
    assert report["parallel_impact"]["fsdp2"] == "PASS"


def test_real_rope_full_training_step(runs, monkeypatch):
    def real_frequencies(pos_embed, img_shapes, text_seq_len, device):
        return tuple(
            torch.view_as_real(freq).contiguous()
            for freq in pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=device)
        )

    monkeypatch.setattr(modeling, "qwen_image_rotary_frequencies", real_frequencies)
    capture(runs / "fixture", runs / "real-rope", device="cpu")
    report = compare(
        runs / "reference",
        runs / "real-rope",
        runs / "real-comparison",
        atol=2e-6,
        rtol=1e-4,
        rationale="CPU FP32 full-step verification of real RoPE algebra",
    )
    assert report["conclusion"] == "PASS"


def test_frozen_condition_alignment(snapshot, tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (24, 16), color="red").save(image)
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"prompt": "red", "image_path": image.name}) + "\n")
    fixture = tmp_path / "fixture"
    prepare(ROOT / "tests/toy_config/qwen_image_toy/config.json", fixture, grid=2, text_tokens=5, records=records)
    training_config = tmp_path / "training.yaml"
    training_config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "condition_model_cfg": {
                        "height": 16,
                        "width": 16,
                        "max_sequence_length": 256,
                        "training_recipe": "diffsynth",
                        "image_resize_mode": "center_crop",
                    }
                }
            }
        )
    )
    for name in ("condition-a", "condition-b"):
        encode(fixture, snapshot, tmp_path / name, training_config=training_config, device="cpu")
    report = compare(
        tmp_path / "condition-a",
        tmp_path / "condition-b",
        tmp_path / "comparison",
        atol=0,
        rtol=0,
        rationale="Identical CPU frozen encoder repeatability",
    )
    assert report["conclusion"] == "PASS"
    for component in ("pixels", "encoder_hidden_states", "latents", "normalized_latents"):
        assert f"condition/{component}/0" in report["metrics"]
    path = tmp_path / "condition-b/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["condition_settings"]["height"] = 32
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="condition_settings"):
        compare(
            tmp_path / "condition-a",
            tmp_path / "condition-b",
            tmp_path / "bad",
            atol=1,
            rtol=1,
            rationale="reject different crop",
        )


@pytest.mark.parametrize("operation", ["capture", "encode"])
@pytest.mark.parametrize("fail", [False, True])
def test_diagnostics_restore_global_numerical_state(runs, snapshot, tmp_path, monkeypatch, operation, fail):
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 16), color="blue").save(image)
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"prompt": "blue", "image_path": image.name}) + "\n")
    fixture = tmp_path / "fixture"
    prepare(ROOT / "tests/toy_config/qwen_image_toy/config.json", fixture, grid=2, text_tokens=5, records=records)
    training_config = tmp_path / "training.yaml"
    training_config.write_text(yaml.safe_dump({"model": {"condition_model_cfg": {"height": 16, "width": 16}}}))

    def state():
        numpy_state = np.random.get_state()
        accelerator = precision_module.get_torch_device()
        device_type = precision_module.get_device_type()
        backend = getattr(torch.backends, device_type) if device_type in ("cuda", "mlu") else accelerator
        matmul = getattr(backend, "matmul", None)
        return {
            "env": {
                name: os.environ.get(name)
                for name in (
                    "PYTHONHASHSEED",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "NCCL_DETERMINISTIC",
                    "FLASH_ATTENTION_DETERMINISTIC",
                    "CLOSE_MATMUL_K_SHIFT",
                )
            },
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
            "torch": torch.get_rng_state().tolist(),
            "accelerator": [rng.tolist() for rng in accelerator.get_rng_state_all()]
            if device_type != "cpu" and accelerator.is_available()
            else [],
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn": (
                torch.backends.cudnn.enabled,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
            ),
            "matmul": (
                torch.get_float32_matmul_precision(),
                getattr(matmul, "allow_tf32", None),
                getattr(matmul, "allow_bf16_reduced_precision_reduction", None),
            ),
        }

    # The outer context only cleans up this test's deliberately non-default state.
    with precision_module.preserve_numerical_state():
        monkeypatch.setenv("PYTHONHASHSEED", "314")
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("medium")
        if fail:
            original = precision_module.enable_high_precision_for_bf16

            def raise_after_setting_precision():
                original()
                raise RuntimeError("diagnostic interrupted")

            monkeypatch.setattr(precision_module, "enable_high_precision_for_bf16", raise_after_setting_precision)

        before = state()

        def run():
            if operation == "capture":
                capture(runs / "fixture", tmp_path / "capture", device="cpu")
            else:
                encode(fixture, snapshot, tmp_path / "encode", training_config=training_config, device="cpu")

        if fail:
            with pytest.raises(RuntimeError, match="diagnostic interrupted"):
                run()
        else:
            run()
        assert state() == before


def test_trace_corruption_is_not_comparable(runs):
    manifest_path = runs / "candidate/run_manifest.json"
    original = manifest_path.read_text()
    data = json.loads(original)
    data["fixture_id"] = "different-inputs"
    manifest_path.write_text(json.dumps(data))
    try:
        with pytest.raises(ValueError, match="fixture_id"):
            compare(
                runs / "reference",
                runs / "candidate",
                runs / "bad",
                atol=1,
                rtol=1,
                rationale="must reject mismatched fixtures",
            )
    finally:
        manifest_path.write_text(original)
    tensor_path = runs / "candidate" / next(iter(data["tensors"].values()))["path"]
    content = tensor_path.read_bytes()
    tensor_path.write_bytes(content + b"corrupt")
    try:
        with pytest.raises(ValueError, match="checksum mismatch"):
            compare(
                runs / "reference",
                runs / "candidate",
                runs / "bad",
                atol=1,
                rtol=1,
                rationale="must reject corruption",
            )
    finally:
        tensor_path.write_bytes(content)


def test_first_divergence_and_nan_are_failures(runs):
    manifest_path = runs / "candidate/run_manifest.json"
    original = manifest_path.read_text()
    data = json.loads(original)
    name = next(name for name in data["tensors"] if "/activation/transformer_blocks.0/" in name)
    path = runs / "candidate" / data["tensors"][name]["path"]
    content = path.read_bytes()
    tensor = load_file(path)["value"]
    try:
        for label, value in [("large-delta", 1000.0), ("nan", float("nan"))]:
            modified = tensor.clone()
            modified.flatten()[0] = value
            save_file({"value": modified}, path)
            data["tensors"][name]["sha256"] = sha256(path)
            manifest_path.write_text(json.dumps(data))
            report = compare(
                runs / "reference", runs / "candidate", runs / label, atol=2e-6, rtol=1e-4, rationale="negative test"
            )
            assert report["conclusion"] == "FAIL"
            assert report["pairs"][0]["first_divergence"]["node"] == name
            assert report["pairs"][0]["nan_inf"] == (label == "nan")
    finally:
        path.write_bytes(content)
        manifest_path.write_text(original)


def test_fixture_changes_and_incomplete_runs_are_rejected(runs):
    path = runs / "fixture/inputs.pt"
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    try:
        with pytest.raises(ValueError, match="checksum mismatch"):
            verify_fixture(runs / "fixture")
    finally:
        path.write_bytes(original)
    path = runs / "candidate/run_manifest.json"
    original = path.read_text()
    data = json.loads(original)
    data["complete"] = False
    path.write_text(json.dumps(data))
    try:
        with pytest.raises(ValueError, match="Incomplete"):
            compare(
                runs / "reference", runs / "candidate", runs / "incomplete", atol=1, rtol=1, rationale="negative test"
            )
    finally:
        path.write_text(original)


def test_difference_metrics():
    assert tensor_difference(torch.zeros(4), torch.zeros(4), 0, 0)["cosine"] == 1
    assert not tensor_difference(torch.zeros(4), torch.ones(4), 0, 0)["passed"]
    assert not tensor_difference(torch.zeros(4), torch.zeros(3), 1, 1)["passed"]


def test_checkpoint_fixture_copies_exact_weights(runs, tmp_path):
    prepare(
        runs / "fixture/model/config.json",
        tmp_path / "fixture",
        weights=runs / "fixture/model",
        grid=2,
        text_tokens=5,
    )
    for original in (runs / "fixture/model").glob("*.safetensors"):
        assert sha256(original) == sha256(tmp_path / "fixture/model" / original.name)
    verify_fixture(tmp_path / "fixture")


def test_identical_partial_traces_cannot_pass(runs, tmp_path):
    for name in ("a", "b"):
        data = json.loads((runs / "candidate/run_manifest.json").read_text())
        del data["tensors"]["step0/loss"]
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete activation/loss/prediction"):
        compare(
            tmp_path / "a", tmp_path / "b", tmp_path / "result", atol=1, rtol=1, rationale="coverage negative test"
        )


@pytest.mark.parametrize("omission", ["parameter", "second-stream"])
def test_identical_parameter_and_stream_omissions_are_rejected(runs, tmp_path, omission):
    data = json.loads((runs / "candidate/run_manifest.json").read_text())
    if omission == "parameter":
        name = next(iter(data["fixture_manifest"]["parameter_shapes"]))
        for stage in ("initial", "gradient", "update"):
            del data["tensors"][f"step0/{stage}/{name}"]
    else:
        del data["tensors"]["step0/activation/transformer_blocks.0/sample0/1"]
    path = tmp_path / "run"
    path.mkdir()
    (path / "run_manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete"):
        compare(path, path, tmp_path / "result", atol=0, rtol=0, rationale="identical omissions must fail")
