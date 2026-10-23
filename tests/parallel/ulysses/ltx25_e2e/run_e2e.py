"""Bounded real-weight LTX-2.5 preprocessing/training/resume validation."""

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(os.environ["LTX25_E2E_ROOT"])
BUNDLE = Path(__file__).resolve().parent
PYTHON = sys.executable
PREPROCESS = "veomni.models.diffusers.ltx2_3.ltx_condition.preprocess_dataset"


def emit(stage, **fields):
    """Emit a machine-readable stage record to the smoke-test log."""
    print("LTX25_E2E_STAGE " + json.dumps(dict(stage=stage, **fields)), flush=True)


def run(stage, argv, timeout=3600, env=None):
    """Run a bounded subprocess and record its successful completion time."""
    emit(stage, status="start")
    started = time.time()
    subprocess.run([str(v) for v in argv], check=True, timeout=timeout, env=env)
    emit(stage, status="pass", seconds=round(time.time() - started, 2))


def sha256(path):
    """Compute a file SHA256 digest without loading the whole file into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_checkpoint(path, step):
    """Validate two-rank checkpoint artifacts and return the exported adapter directory."""
    import torch

    from veomni.checkpoint import layout

    assert (path / layout.MANIFEST_FILENAME).is_file(), path
    assert Path(layout.weights_dir(str(path)), ".metadata").is_file()
    assert Path(layout.optimizer_dir(str(path)), ".metadata").is_file()
    for rank in (0, 1):
        extra_path = Path(layout.model_dir(str(path)), "extra_state", f"extra_state_rank_{rank}.pt")
        extra = torch.load(extra_path, map_location="cpu", weights_only=False)
        assert extra["lr_scheduler"]["last_epoch"] == step
        assert Path(layout.extra_state_path(str(path), rank)).is_file()
    assert Path(layout.loader_path(str(path), 0)).is_file()
    adapter = path / "lora_ckpt"
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()
    emit("checkpoint_verified", step=step, adapter_sha256=sha256(adapter / "adapter_model.safetensors"))
    return adapter


def training_config(tag, conditioning, load=None, adapter=None, max_steps=4):
    """Write a bounded SP2 training recipe with optional checkpoint or adapter loading."""
    import yaml

    config = yaml.safe_load(Path("configs/dit/ltx25_av_lora.yaml").read_text())
    m, d, t = config["model"], config["data"], config["train"]
    m.update(
        model_path=str(ROOT / "prepared-model"),
        condition_model_path=str(ROOT / "prepared-model"),
        config_path=str(ROOT / "prepared-model/config.json"),
    )
    m["condition_model_cfg"]["first_frame_conditioning_p"] = conditioning
    m["lora_config"].update(rank=4, alpha=4)
    if adapter:
        m["lora_config"]["lora_adapter"] = str(adapter)
    m["accelerator"]["ulysses_size"] = 2
    m["accelerator"]["fsdp_config"]["offload"] = True
    d.update(
        train_path=str(ROOT / "parquet"),
        max_seq_len=8192,
        log_sample=False,
        dataset_repeat=True,
        train_sample=4,
    )
    d["dataloader"].update(num_workers=0, drop_last=True, shuffle=False)
    t.update(
        global_batch_size=1,
        micro_batch_size=1,
        max_steps=max_steps,
        num_train_epochs=1,
        enable_full_determinism=True,
        bsz_warmup_ratio=0,
    )
    t["checkpoint"].update(
        output_dir=str(ROOT / "runs" / tag),
        save_steps=2,
        save_epochs=0,
        save_hf_weights=True,
        hf_save_steps=2,
        hf_save_epochs=0,
    )
    if load:
        t["checkpoint"]["load_path"] = str(load)
    path = ROOT / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def train(tag, conditioning, **kwargs):
    """Launch two observed training workers and return their checkpoint directory."""
    env = os.environ.copy()
    env["LTX25_AUDIT_DIR"] = str(ROOT / "audit" / tag)
    config = training_config(tag, conditioning, **kwargs)
    run(
        "train:" + tag,
        [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            BUNDLE / "train_observed.py",
            config,
        ],
        env=env,
    )
    return ROOT / "runs" / tag / "checkpoints"


def main():
    """Validate real-weight preprocessing, training, checkpoint resume and adapter reload."""
    ROOT.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(ROOT).free > 160 * 1024**3, "Need 160 GiB local disk for weights/cache/checkpoints"
    import torch

    assert torch.cuda.device_count() == 2
    emit("environment", torch=torch.__version__, devices=[torch.cuda.get_device_name(i) for i in range(2)])
    manifest = json.loads((BUNDLE / "weights-manifest.json").read_text())
    from range_download import download_all

    paths = download_all(manifest, ROOT / "weights", emit)
    transformer, gemma, video_vae, audio_vae = paths
    from safetensors import safe_open

    from veomni.models.diffusers.ltx2_3.ltx_core.model.video_vae.model_configurator import (
        VAE_ENCODER_COMFY_KEYS_FILTER,
        VideoEncoderConfigurator,
    )

    with safe_open(str(video_vae), framework="pt") as weights:
        with torch.device("meta"):
            encoder = VideoEncoderConfigurator.from_config(json.loads(weights.metadata()["config"]))
        expected_shapes = {name: tuple(value.shape) for name, value in encoder.state_dict().items()}
        actual_shapes = {
            mapped: tuple(weights.get_slice(name).get_shape())
            for name in weights.keys()
            if (mapped := VAE_ENCODER_COMFY_KEYS_FILTER.apply_to_key(name)) is not None
        }
        assert actual_shapes == expected_shapes, "Video encoder checkpoint coverage mismatch"
        emit("video_encoder_coverage", matched_tensors=len(actual_shapes))
    del encoder
    run(
        "prepare_model",
        [
            PYTHON,
            "-m",
            "veomni.models.diffusers.ltx2_3.ltx_condition.ltx25",
            "--transformer_path",
            transformer,
            "--output_dir",
            ROOT / "prepared-model",
        ],
    )
    media = ROOT / "media"
    media.mkdir()
    run(
        "download_media",
        [
            "curl",
            "-fsSL",
            "--retry",
            "2",
            "--max-time",
            "180",
            "https://media.w3.org/2010/05/sintel/trailer.mp4",
            "-o",
            media / "source.mp4",
        ],
        timeout=200,
    )
    expected = json.loads((BUNDLE / "media-manifest.json").read_text())
    assert sha256(media / "source.mp4") == expected["sha256"]
    for i, start in enumerate((10, 25)):
        run(
            "clip:" + str(i),
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(start),
                "-i",
                media / "source.mp4",
                "-t",
                "3",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-vf",
                "scale=256:256:force_original_aspect_ratio=increase,crop=256:256,fps=24",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                media / f"clip-{i}.mp4",
            ],
            timeout=120,
        )
    (media / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "media_path": f"clip-{i}.mp4",
                    "caption": "A short scene from the animated film Sintel, with its original soundtrack.",
                }
                for i in range(2)
            ],
            indent=2,
        )
    )
    run(
        "preprocess",
        [
            PYTHON,
            "-m",
            PREPROCESS,
            "preprocess",
            "--dataset_file",
            media / "dataset.json",
            "--checkpoint_path",
            transformer,
            "--gemma_model_path",
            gemma,
            "--video_vae_path",
            video_vae,
            "--audio_vae_path",
            audio_vae,
            "--precomputed_dir",
            ROOT / "cache",
            "--resolution_buckets",
            "256x256x49",
            "--max_sequence_length",
            "128",
            "--with_audio",
            "--device",
            "cuda:0",
        ],
    )
    run(
        "parquet",
        [
            PYTHON,
            "-m",
            PREPROCESS,
            "save-parquet",
            "--precomputed_dir",
            ROOT / "cache",
            "--output_dir",
            ROOT / "parquet",
            "--pad_to_multiple_of",
            "1",
            "--with_audio",
        ],
    )
    import pyarrow.parquet as pq

    rows = []
    for path in (ROOT / "parquet").glob("*.parquet"):
        rows.extend(pq.read_table(path).to_pylist())
    assert len(rows) == 2, len(rows)
    required = {"latents", "video_prompt_embeds", "prompt_attention_mask", "audio_prompt_embeds", "audio_latents"}
    for row in rows:
        assert required <= row.keys(), row.keys()
        shapes = {}
        for k in required:
            value = pickle.loads(row[k])
            assert isinstance(value, torch.Tensor) and torch.isfinite(value).all(), k
            shapes[k] = list(value.shape)
            if k in {"latents", "audio_latents", "video_prompt_embeds", "audio_prompt_embeds"}:
                assert value.float().std() > 0, ("Degenerate cached tensor", k)
        emit("cache_verified", shapes=shapes)
    from safetensors.torch import load_file

    results = []
    for tag, conditioning in [("t2av", 0.0), ("i2av", 1.0)]:
        full = train(tag, conditioning)
        check_checkpoint(full / "global_step_2", 2)
        full_adapter = check_checkpoint(full / "global_step_4", 4)
        resumed = train(tag + "-resumed", conditioning, load=full / "global_step_2")
        resumed_adapter = check_checkpoint(resumed / "global_step_4", 4)
        for rank in (0, 1):
            a = json.loads((ROOT / "audit" / tag / f"rank-{rank}.json").read_text())
            b = json.loads((ROOT / "audit" / (tag + "-resumed") / f"rank-{rank}.json").read_text())
            assert [x["step"] for x in a if x["event"] == "step"] == [1, 2, 3, 4]
            assert [x["step"] for x in b if x["event"] == "step"] == [3, 4]
            assert b[0]["step"] == 2 and b[0]["parameter_sha256"] == a[2]["parameter_sha256"]
            for actual in b[1:]:
                reference = next(x for x in a if x["event"] == "step" and x["step"] == actual["step"])
                assert abs(actual["loss"] - reference["loss"]) <= 2e-3 * abs(reference["loss"]) + 2e-5
        a = load_file(str(full_adapter / "adapter_model.safetensors"))
        b = load_file(str(resumed_adapter / "adapter_model.safetensors"))
        assert a.keys() == b.keys()
        error = reference = 0.0
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=2e-3, atol=2e-5)
            error += float((a[name].float() - b[name].float()).square().sum())
            reference += float(a[name].float().square().sum())
        relative = (error / max(reference, 1e-30)) ** 0.5
        assert relative < 2e-3
        emit("resume_verified", case=tag, adapter_relative_error=relative, tensors=len(a))
        results.append(dict(case=tag, adapter_relative_error=relative))
        reload_tag = tag + "-adapter-reload"
        train(reload_tag, conditioning, adapter=full_adapter, max_steps=1)
        for rank in (0, 1):
            source = json.loads((ROOT / "audit" / tag / f"rank-{rank}.json").read_text())
            loaded = json.loads((ROOT / "audit" / reload_tag / f"rank-{rank}.json").read_text())
            assert loaded[0]["parameter_sha256"] == source[-1]["exported_parameter_sha256"], (
                "Exported adapter differs after reload"
            )
        emit("adapter_reload_verified", case=tag)
    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    emit("complete", results=results)
    print("LTX25_REAL_E2E_PASS", flush=True)


if __name__ == "__main__":
    main()
