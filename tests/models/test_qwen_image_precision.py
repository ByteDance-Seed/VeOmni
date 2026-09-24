"""Validate portable Qwen RoPE and the cross-device precision artifact contract."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import CheckpointError, _CheckpointFrame

from scripts.precision.qwen_image import capture, compare, encode, prepare, sha256, tensor_difference, verify_fixture
from tests.models.test_qwen_image_training import snapshot as snapshot
from tests.models.test_qwen_image_training import transformer
from veomni.arguments.arguments_types import MixedPrecisionConfig
from veomni.distributed.parallel_state import ParallelState, use_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.diffusers.qwen_image.qwen_image_transformer import modeling_qwen_image_transformer as modeling
from veomni.utils.device import IS_NPU_AVAILABLE


ROOT = Path(__file__).resolve().parents[2]


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
        actual, expected, atol=1e-6 if dtype == torch.float32 else 4e-3, rtol=1e-5 if dtype == torch.float32 else 4e-3
    )
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(
        y.grad, x.grad, atol=1e-6 if dtype == torch.float32 else 1e-4, rtol=1e-5 if dtype == torch.float32 else 1e-2
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
        actual.to("cpu"), expected, atol=1e-5 if dtype == torch.float32 else 5e-3,
        rtol=1e-5 if dtype == torch.float32 else 5e-3,
    )

    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(
        x_npu.grad.to("cpu"), x_cpu.grad, atol=1e-5 if dtype == torch.float32 else 1e-3,
        rtol=1e-5 if dtype == torch.float32 else 1e-2,
    )


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
def runs(tmp_path_factory):
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
