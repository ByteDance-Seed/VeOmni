"""Prepare, capture and compare fixed Qwen-Image training steps across devices."""

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import MethodType

import diffusers
import torch
import transformers
import yaml
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from veomni.arguments.arguments_types import AcceleratorConfig, MixedPrecisionConfig, OpsImplementationConfig
from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.models.auto import build_config
from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
    QWEN_IMAGE_ORIGINAL_FORWARD,
    QwenImageTransformer2DModel,
)
from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY
from veomni.models.module_utils import init_empty_weights, load_model_weights
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device, set_device
from veomni.utils.helper import enable_full_determinism, enable_high_precision_for_bf16


ROOT = Path(__file__).resolve().parents[2]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
MODEL = "Qwen/Qwen-Image"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    paths = list((ROOT / "veomni/models/diffusers/qwen_image").rglob("*.py"))
    paths += [Path(__file__), ROOT / "veomni/distributed/torch_parallelize.py"]
    return {"revision": revision, "files": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}}


def eager_ops():
    return OpsImplementationConfig(
        attn_implementation="eager",
        rotary_pos_emb_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        cross_entropy_loss_implementation="eager",
        load_balancing_loss_implementation="eager",
        moe_implementation="eager",
    )


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_to(item, device) for item in value)
    return value


def validate_batch(batch, config):
    required = {"hidden_states", "timestep", "encoder_hidden_states", "training_target", "img_shapes"}
    optional = {"encoder_hidden_states_mask", "loss_weights", "latents"}
    if not isinstance(batch, dict) or not required.issubset(batch) or batch.keys() - required - optional:
        raise ValueError(f"Fixed batch needs {sorted(required)} and optional {sorted(optional)}.")
    count = len(batch["hidden_states"])
    if not count or any(not isinstance(value, list) or len(value) != count for value in batch.values()):
        raise ValueError("Fixed batch fields must be equally sized nonempty per-sample lists.")
    for index in range(count):
        hidden, target, context = (
            batch[key][index] for key in ("hidden_states", "training_target", "encoder_hidden_states")
        )
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != config.in_channels:
            raise ValueError("Each fixed sample must be [1, image_tokens, config.in_channels].")
        if (
            target.shape != hidden.shape
            or context.ndim != 3
            or context.shape[0] != 1
            or context.shape[-1] != config.joint_attention_dim
        ):
            raise ValueError("Target or text context does not match the Qwen-Image config.")
        grid = batch["img_shapes"][index]
        if len(grid) != 1 or grid[0][0] != 1 or math.prod(grid[0]) != hidden.shape[1]:
            raise ValueError("Each base Qwen-Image sample requires one matching (1, H, W) packed grid.")
        if batch["timestep"][index].shape != (1,):
            raise ValueError("Each timestep must have shape [1].")

    def check(value):
        if isinstance(value, torch.Tensor) and (value.is_complex() or not torch.isfinite(value).all()):
            raise ValueError("Fixed inputs must be real and finite.")
        if isinstance(value, (tuple, list)):
            for item in value:
                check(item)

    for value in batch.values():
        check(value)


def prepare(config_path, output, *, weights=None, batch_file=None, records=None, seed=42, grid=4, text_tokens=16):
    """Freeze one shared input/weight bundle; random dimensions come from config."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = build_config(str(config_path))
    if (
        config.model_type != "QwenImageTransformer2DModel"
        or config.guidance_embeds
        or config.zero_cond_t
        or config.use_additional_t_cond
        or config.use_layer3d_rope
    ):
        raise ValueError("This alignment fixture targets the base Qwen-Image text-to-image transformer.")
    if config.in_channels != config.patch_size**2 * config.out_channels:
        raise ValueError("Base flow matching requires matching packed input/output widths.")
    if grid < 1 or text_tokens < 2:
        raise ValueError("grid must be positive and text_tokens must be at least 2.")
    if weights:
        weights = Path(weights)
        weight_config = build_config(str(weights))
        if json.dumps(config.to_diffuser_dict(), sort_keys=True) != json.dumps(
            weight_config.to_diffuser_dict(), sort_keys=True
        ):
            raise ValueError("Checkpoint architecture differs from --config.")
        # Copy original shards without materializing a full FP32 model on the host.
        paths = sorted(weights.glob("*.safetensors")) + sorted(weights.glob("*.safetensors.index.json"))
        if not any(path.suffix == ".safetensors" for path in paths):
            raise ValueError("--weights requires a local safetensors transformer checkpoint.")
        model = build_foundation_model(
            config, init_device="meta", torch_dtype="float32", ops_implementation=eager_ops()
        )
        expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        observed = {}
        for path in paths:
            if path.suffix == ".safetensors":
                with safe_open(path, framework="pt", device="cpu") as shard:
                    for key in shard.keys():
                        if key in observed:
                            raise ValueError(f"Duplicate checkpoint tensor: {key}")
                        observed[key] = tuple(shard.get_slice(key).get_shape())
        if observed != expected:
            raise ValueError("Checkpoint keys or tensor shapes differ from the complete model config.")
        config.save_pretrained(output / "model")
        for path in paths:
            shutil.copyfile(path, output / "model" / path.name)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = build_foundation_model(
                config, init_device="cpu", torch_dtype="float32", ops_implementation=eager_ops()
            )
            model.save_pretrained(output / "model", max_shard_size="2GB")
    parameter_shapes = {name: list(parameter.shape) for name, parameter in model.named_parameters()}
    del model
    if batch_file:
        batch = tree_to(torch.load(batch_file, map_location="cpu", weights_only=True), "cpu")
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape = (1, grid * grid, config.in_channels)
        latent, noise = [torch.randn(shape, generator=generator) for _ in range(2)]
        mask = torch.ones(1, text_tokens, dtype=torch.bool)
        mask[:, -2:] = False
        batch = {
            "hidden_states": [(latent + noise) / 2],
            "training_target": [noise - latent],
            "timestep": [torch.tensor([0.5])],
            "loss_weights": [torch.ones(1)],
            "encoder_hidden_states": [torch.randn(1, text_tokens, config.joint_attention_dim, generator=generator)],
            "encoder_hidden_states_mask": [mask],
            "img_shapes": [[(1, grid, grid)]],
        }
    validate_batch(batch, config)
    torch.save(batch, output / "inputs.pt")
    if records:
        from veomni.data.multimodal.dit.preprocess import qwen_image_preprocess

        raw = []
        for line in Path(records).read_text().splitlines():
            if line.strip():
                prompt, _, images, _ = qwen_image_preprocess(json.loads(line), data_dir=str(Path(records).parent))
                raw.append({"prompt": prompt, "image": Path(images[0]).read_bytes()})
        if not raw:
            raise ValueError("The raw image/text records file is empty.")
        torch.save(raw, output / "raw_inputs.pt")
    files = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {
        "model": MODEL,
        "source": source_identity(),
        "files": files,
        "seed": seed,
        "input_kind": "model_ready_batch" if batch_file else "synthetic_flow_fixture",
        "scope": "DiT training numerics with frozen conditioning, noise and timesteps; encoders are tested separately.",
        "sample_count": len(batch["hidden_states"]),
        "parameter_shapes": parameter_shapes,
        "block_count": config.num_layers,
    }
    manifest["fixture_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(output / "fixture_manifest.json", manifest)
    return manifest


def verify_fixture(folder):
    folder = Path(folder)
    manifest = read_json(folder / "fixture_manifest.json")
    content = {key: value for key, value in manifest.items() if key != "fixture_id"}
    if hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != manifest["fixture_id"]:
        raise ValueError("Fixture manifest identity changed.")
    actual_files = {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}
    if actual_files != set(manifest["files"]) | {"fixture_manifest.json"}:
        raise ValueError("Fixture contains unlisted or missing files.")
    for name, expected in manifest["files"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Fixture checksum mismatch: {name}")
    return manifest


class TensorRecorder:
    def __init__(self, root, rank):
        self.root, self.rank, self.records, self.nonfinite = Path(root), rank, {}, False

    def full_cpu(self, tensor):
        value = tensor.detach()
        # Every FSDP rank must participate, even though only rank zero writes.
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        return value.to(device="cpu", dtype=torch.float32).contiguous()

    def add(self, name, tensor):
        value = self.full_cpu(tensor)
        if self.rank != 0:
            return
        if name in self.records:
            raise ValueError(f"Duplicate trace tensor: {name}")
        filename = f"tensors/{len(self.records):06d}.safetensors"
        save_file({"value": value}, self.root / filename)
        finite = bool(torch.isfinite(value).all())
        self.nonfinite |= not finite
        self.records[name] = {
            "path": filename,
            "shape": list(value.shape),
            "sha256": sha256(self.root / filename),
            "finite": finite,
        }


def capture(fixture, output, *, device="cpu", dtype="fp32", backend="veomni", steps=1, learning_rate=1e-5):
    """Run identical full inputs per DP rank; torchrun enables native FSDP2."""
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive.")
    if device != "cpu" and get_device_type() != device:
        raise RuntimeError(f"Requested {device}, detected {get_device_type()}; device fallback is forbidden.")
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid WORLD_SIZE/RANK combination.")
    if world > 1 and (device == "cpu" or backend != "veomni"):
        raise ValueError("Multi-rank alignment uses the VeOmni GPU/NPU FSDP2 path.")
    if backend == "diffusers" and device == "npu":
        raise ValueError("Use CPU/GPU for the upstream complex-RoPE baseline and VeOmni for NPU.")
    fixture, output = Path(fixture), Path(output)
    manifest = verify_fixture(fixture)
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory to avoid stale traces: {output}")
    enable_full_determinism(manifest["seed"])
    enable_high_precision_for_bf16()
    torch.use_deterministic_algorithms(True)
    if device != "cpu":
        set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_group = False
    hooks = []
    try:
        if world > 1:
            accelerator = AcceleratorConfig(dp_replicate_size=1, dp_shard_size=world)
            torch.distributed.init_process_group(get_dist_comm_backend())
            owns_group = True
            init_parallel_state_from_config(accelerator, name=None)
        if rank == 0:
            (output / "tensors").mkdir(parents=True)
        if world > 1:
            torch.distributed.barrier()
        recorder = TensorRecorder(output, rank)
        config = build_config(str(fixture / "model"))
        if backend == "veomni":
            model = build_foundation_model(
                config,
                init_device="meta",
                torch_dtype=str(DTYPES[dtype]).removeprefix("torch."),
                ops_implementation=eager_ops(),
            )
            if world > 1:
                model = build_parallelize_model(
                    model,
                    weights_path=str(fixture / "model"),
                    init_device="meta",
                    mixed_precision=MixedPrecisionConfig(enable=False),
                    enable_gradient_checkpointing=False,
                    enable_reshard_after_forward=True,
                )
            else:
                load_model_weights(model, str(fixture / "model"), init_device="cpu")
                model.to(device)
        else:
            with init_empty_weights():
                model = diffusers.QwenImageTransformer2DModel(**config.to_diffuser_dict()).to(DTYPES[dtype])
            load_model_weights(model, str(fixture / "model"), init_device="cpu")
            model.to(device)
            model.forward = MethodType(QWEN_IMAGE_ORIGINAL_FORWARD, model)
        # Disable dropout, retaining autograd. This is a numerical diagnostic.
        model.eval()
        batch = tree_to(torch.load(fixture / "inputs.pt", map_location="cpu", weights_only=True), device)
        validate_batch(batch, config)
        optimizer_settings = {"lr": learning_rate, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_settings, foreach=False, fused=False)
        current = {"step": 0, "calls": {}}

        def make_hook(name):
            def hook(module, args, result):
                occurrence = current["calls"].get(name, 0)
                current["calls"][name] = occurrence + 1
                outputs = result if isinstance(result, tuple) else (result,)
                for index, value in enumerate(outputs):
                    if isinstance(value, torch.Tensor):
                        recorder.add(f"step{current['step']}/activation/{name}/sample{occurrence}/{index}", value)

            return hook

        for name, module in model.named_modules():
            if (
                name in ("img_in", "txt_in", "time_text_embed", "norm_out", "proj_out")
                or name.startswith("transformer_blocks.")
                and name.count(".") == 1
            ):
                hooks.append(module.register_forward_hook(make_hook(name)))
        absent_gradients = []
        for step in range(steps):
            current.update(step=step, calls={})
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in model.named_parameters():
                recorder.add(f"step{step}/initial/{name}", parameter)
            if backend == "veomni":
                result = model(**batch)
                predictions, loss = result.predictions, result.loss["mse_loss"]
            else:
                predictions, losses = [], []
                for index in range(len(batch["hidden_states"])):
                    sample = {key: batch[key][index] for key in ("hidden_states", "timestep", "encoder_hidden_states")}
                    sample["hidden_states"] = sample["hidden_states"].to(DTYPES[dtype])
                    sample["encoder_hidden_states"] = sample["encoder_hidden_states"].to(DTYPES[dtype])
                    sample["encoder_hidden_states_mask"] = batch.get(
                        "encoder_hidden_states_mask", [None] * len(batch["hidden_states"])
                    )[index]
                    sample["img_shapes"] = QwenImageTransformer2DModel._normalize_img_shapes(
                        batch["img_shapes"][index]
                    )
                    prediction = model(**sample, return_dict=False)[0]
                    predictions.append(prediction)
                    weight = batch.get("loss_weights", [None] * len(batch["hidden_states"]))[index]
                    sample_loss = (
                        (prediction.float() - batch["training_target"][index].float()).square().flatten(1).mean(1)
                    )
                    losses.append(sample_loss if weight is None else sample_loss * weight.float())
                loss = torch.cat(losses).mean()
            for index, prediction in enumerate(predictions):
                recorder.add(f"step{step}/prediction/{index}", prediction)
            recorder.add(f"step{step}/loss", loss)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    absent_gradients.append(f"step{step}/{name}")
                else:
                    recorder.add(f"step{step}/gradient/{name}", parameter.grad)
            optimizer.step()
            for name, parameter in model.named_parameters():
                after = recorder.full_cpu(parameter)
                if rank == 0:
                    before = load_file(output / recorder.records[f"step{step}/initial/{name}"]["path"])["value"]
                    recorder.add(f"step{step}/update/{name}", after - before)
        if device != "cpu":
            get_torch_device().synchronize()
        if rank == 0:
            platform = (
                "cpu" if device == "cpu" else "gpu" if device == "cuda" else "npu_multi" if world > 1 else "npu_single"
            )
            run = {
                "model": MODEL,
                "kind": "transformer",
                "complete": True,
                "fixture_id": manifest["fixture_id"],
                "fixture_manifest": manifest,
                "platform": platform,
                "device": device,
                "world_size": world,
                "fsdp2": world > 1,
                "sp": False,
                "backend": backend,
                "dtype": dtype,
                "steps": steps,
                "sample_count": manifest["sample_count"],
                "block_count": config.num_layers,
                "optimizer": optimizer_settings,
                "source": source_identity(),
                "versions": {
                    "torch": torch.__version__,
                    "diffusers": diffusers.__version__,
                    "transformers": transformers.__version__,
                },
                "attention": "PyTorch SDPA native dispatch",
                "deterministic": True,
                "tensors": recorder.records,
                "absent_gradients": absent_gradients,
                "nan_inf": recorder.nonfinite,
            }
            write_json(output / "run_manifest.json", run)
            if platform != "cpu":
                write_json(
                    output / "golden_manifest.json",
                    {
                        "model": MODEL,
                        "code_revision": run["source"]["revision"],
                        "weights": manifest["fixture_id"],
                        "inputs": {"path": str(fixture / "inputs.pt"), "batch_size": manifest["sample_count"]},
                        "seeds": dict.fromkeys(("torch", "python", "numpy"), manifest["seed"]),
                        "dtype": dtype,
                        "generation": {
                            "platform": platform,
                            "artifacts": ["run_manifest.json", "tensors/"],
                            "command": " ".join(sys.argv),
                        },
                        "weight_and_input_hashes": manifest["files"],
                        "source_hashes": run["source"]["files"],
                    },
                )
            return run
    finally:
        for hook in hooks:
            hook.remove()
        if owns_group:
            torch.distributed.destroy_process_group()
            clear_parallel_state()


def encode(fixture, snapshot, output, *, training_config, device="cpu"):
    """Compare frozen encoders separately, with identical raw bytes and snapshot hashes."""
    if device != "cpu" and get_device_type() != device:
        raise RuntimeError(f"Requested {device}, detected {get_device_type()}; device fallback is forbidden.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Frozen condition encoding is a single-device diagnostic.")
    fixture, snapshot, output = Path(fixture), Path(snapshot), Path(output)
    manifest = verify_fixture(fixture)
    raw = torch.load(fixture / "raw_inputs.pt", map_location="cpu", weights_only=True)
    settings = yaml.safe_load(Path(training_config).read_text())["model"]["condition_model_cfg"]
    settings = {**settings, "seed": manifest["seed"]}
    config = MODEL_CONFIG_REGISTRY["QwenImageConditionModel"]().from_pretrained(str(snapshot), **settings)
    components = [
        config.tokenizer_subfolder,
        config.text_encoder_subfolder,
        config.vae_subfolder,
        config.scheduler_subfolder,
    ]
    weights = {
        str(path.relative_to(snapshot)): sha256(path)
        for component in components
        for path in sorted((snapshot / component).rglob("*"))
        if path.is_file()
    }
    if not weights:
        raise ValueError("No condition component files found in the local snapshot.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tensors").mkdir()
    enable_full_determinism(manifest["seed"])
    enable_high_precision_for_bf16()
    torch.use_deterministic_algorithms(True)
    if device != "cpu":
        set_device(0)
    condition = (
        MODELING_REGISTRY["QwenImageConditionModel"]()._from_config(config).requires_grad_(False).eval().to(device)
    )
    recorder = TensorRecorder(output, 0)
    prompts = [item["prompt"] for item in raw]
    images = [[Image.open(io.BytesIO(item["image"])).convert("RGB")] for item in raw]
    with torch.no_grad():
        encoded = condition.get_condition(inputs=prompts, images=images)
        for index, sample in enumerate(images):
            recorder.add(f"condition/pixels/{index}", condition._image_to_tensor(sample[0]))
        for key in ("encoder_hidden_states", "encoder_hidden_states_mask", "latents"):
            for index, tensor in enumerate(encoded[key]):
                if tensor is not None:
                    recorder.add(f"condition/{key}/{index}", tensor)
        for index, parameters in enumerate(encoded["latents"]):
            latent = parameters.chunk(2, dim=1)[0]
            recorder.add(
                f"condition/normalized_latents/{index}", condition._pack_latents(condition._normalize_latents(latent))
            )
    if device != "cpu":
        get_torch_device().synchronize()
    platform = {"cpu": "cpu", "cuda": "gpu", "npu": "npu_single"}[device]
    run = {
        "model": MODEL,
        "kind": "condition",
        "complete": True,
        "fixture_id": manifest["fixture_id"],
        "fixture_manifest": manifest,
        "platform": platform,
        "device": device,
        "world_size": 1,
        "fsdp2": False,
        "sp": False,
        "backend": "veomni",
        "dtype": "bf16",
        "component_dtypes": {"text_encoder": "bf16", "vae": "fp32"},
        "steps": 1,
        "sample_count": len(raw),
        "optimizer": None,
        "source": source_identity(),
        "condition_settings": settings,
        "condition_weights": weights,
        "versions": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
        },
        "attention": "PyTorch SDPA native dispatch",
        "deterministic": True,
        "tensors": recorder.records,
        "absent_gradients": [],
        "nan_inf": recorder.nonfinite,
    }
    write_json(output / "run_manifest.json", run)
    if platform != "cpu":
        write_json(
            output / "golden_manifest.json",
            {
                "model": MODEL,
                "code_revision": run["source"]["revision"],
                "weights": str(snapshot),
                "inputs": {"path": str(fixture / "raw_inputs.pt"), "batch_size": len(raw)},
                "seeds": dict.fromkeys(("torch", "python", "numpy"), manifest["seed"]),
                "dtype": "bf16",
                "generation": {"platform": platform, "artifacts": ["run_manifest.json", "tensors/"]},
                "component_dtypes": run["component_dtypes"],
                "weight_hashes": weights,
            },
        )
    return run


def tensor_difference(reference, candidate, atol, rtol):
    if reference.shape != candidate.shape:
        return {"passed": False, "reason": "shape mismatch", "nan_inf": False}
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        return {"passed": False, "reason": "NaN/Inf", "nan_inf": True}
    ref, target = reference.reshape(-1), candidate.reshape(-1)
    square_diff = square_ref = square_target = dot = maximum = 0.0
    passed = True
    for start in range(0, ref.numel(), 1_048_576):
        a, b = ref[start : start + 1_048_576].double(), target[start : start + 1_048_576].double()
        delta = (a - b).abs()
        maximum = max(maximum, float(delta.max()))
        square_diff += float(delta.square().sum())
        square_ref += float(a.square().sum())
        square_target += float(b.square().sum())
        dot += float((a * b).sum())
        passed &= bool((delta <= atol + rtol * a.abs()).all())
    denominator = math.sqrt(square_ref * square_target)
    return {
        "passed": passed,
        "nan_inf": False,
        "max_abs": maximum,
        "rel": math.sqrt(square_diff) / max(math.sqrt(square_ref), 1e-30),
        "cosine": max(-1.0, min(1.0, dot / denominator)) if denominator else float(square_ref == square_target),
    }


def compare(reference, candidate, output, *, atol, rtol, rationale):
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)) or not rationale.strip():
        raise ValueError("Finite nonnegative tolerances and a threshold rationale are required.")
    reference, candidate, output = Path(reference), Path(candidate), Path(output)
    baseline, target = [read_json(folder / "run_manifest.json") for folder in (reference, candidate)]
    for run in (baseline, target):
        if not run.get("complete") or not run.get("tensors"):
            raise ValueError("Incomplete run; comparison cannot pass.")
        fixture = run["fixture_manifest"]
        content = {key: value for key, value in fixture.items() if key != "fixture_id"}
        if (
            hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != run["fixture_id"]
            or fixture["fixture_id"] != run["fixture_id"]
        ):
            raise ValueError("Run fixture_id does not match its bound input and parameter inventory.")
        names = set(run["tensors"])
        if run["kind"] == "condition":
            required = {
                f"condition/{stage}/{index}"
                for stage in ("pixels", "encoder_hidden_states", "latents", "normalized_latents")
                for index in range(run["sample_count"])
            }
            if not required.issubset(names):
                raise ValueError("Incomplete condition trace coverage.")
        elif run["kind"] == "transformer":
            expected = set(fixture["parameter_shapes"])
            if (
                not expected
                or run["sample_count"] != fixture["sample_count"]
                or run["block_count"] != fixture["block_count"]
            ):
                raise ValueError("Trace sample/block inventory differs from the fixed fixture.")
            for step in range(run["steps"]):
                prefix = f"step{step}/"
                params = {
                    name.removeprefix(prefix + "initial/") for name in names if name.startswith(prefix + "initial/")
                }
                gradients = {
                    name.removeprefix(prefix + "gradient/") for name in names if name.startswith(prefix + "gradient/")
                }
                absent = {name.removeprefix(prefix) for name in run["absent_gradients"] if name.startswith(prefix)}
                updates = {
                    name.removeprefix(prefix + "update/") for name in names if name.startswith(prefix + "update/")
                }
                if (
                    not gradients
                    or params != expected
                    or params != updates
                    or params != gradients | absent
                    or gradients & absent
                ):
                    raise ValueError("Incomplete parameter/gradient/update trace coverage.")
                required = {prefix + "loss"} | {f"{prefix}prediction/{index}" for index in range(run["sample_count"])}
                modules = ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out"]
                modules += [f"transformer_blocks.{index}" for index in range(run["block_count"])]
                required |= {
                    f"{prefix}activation/{module}/sample{index}/0"
                    for module in modules
                    for index in range(run["sample_count"])
                }
                required |= {
                    f"{prefix}activation/transformer_blocks.{block}/sample{index}/1"
                    for block in range(run["block_count"])
                    for index in range(run["sample_count"])
                }
                if not required.issubset(names):
                    raise ValueError("Incomplete activation/loss/prediction trace coverage.")
        else:
            raise ValueError("Unknown trace kind.")
    for key in (
        "model",
        "kind",
        "fixture_id",
        "dtype",
        "steps",
        "sample_count",
        "optimizer",
        "source",
        "attention",
        "deterministic",
    ):
        if baseline[key] != target[key]:
            raise ValueError(f"Incomparable runs: {key} differs.")
    for key in ("diffusers", "transformers"):
        if baseline["versions"][key] != target["versions"][key]:
            raise ValueError(f"Incomparable runs: {key} version differs.")
    if (
        baseline["tensors"].keys() != target["tensors"].keys()
        or baseline["absent_gradients"] != target["absent_gradients"]
    ):
        raise ValueError("Trace coverage or missing gradient sets differ; partial comparisons cannot pass.")
    if baseline["kind"] == "condition":
        for key in ("condition_settings", "condition_weights", "component_dtypes"):
            if baseline[key] != target[key]:
                raise ValueError(f"Incomparable runs: {key} differs.")
    elif not any("/gradient/" in key for key in baseline["tensors"]):
        raise ValueError("No gradients captured; training comparison cannot pass.")
    results, first = {}, None
    for name, record in baseline["tensors"].items():
        tensors = []
        for folder, metadata in ((reference, record), (candidate, target["tensors"][name])):
            path = folder / metadata["path"]
            if sha256(path) != metadata["sha256"]:
                raise ValueError(f"Trace checksum mismatch: {name}")
            tensors.append(load_file(path)["value"])
        diff = tensor_difference(*tensors, atol=atol, rtol=rtol)
        results[name] = diff
        if not diff["passed"] and first is None:
            first = {"node": name, "reason": diff.get("reason", "tolerance exceeded")}
            if "max_abs" in diff:
                first["diff"] = {key: diff[key] for key in ("max_abs", "rel", "cosine")}
    passed = all(result["passed"] for result in results.values())
    pair = {
        "baseline": baseline["platform"],
        "target": target["platform"],
        "threshold": {
            "metric": "elementwise abs(a-b) <= atol + rtol*abs(a)",
            "value": atol,
            "rtol": rtol,
            "rationale": rationale,
        },
        "first_divergence": first,
        "nan_inf": any(item["nan_inf"] for item in results.values()),
        "conclusion": "PASS" if passed else "FAIL",
    }
    accelerated = "cpu" not in (baseline["platform"], target["platform"])
    report = {
        "model": MODEL,
        "golden_manifest": str(reference / ("golden_manifest.json" if accelerated else "run_manifest.json")),
        "pairs": [pair],
        "conclusion": pair["conclusion"],
        "tensor_count": len(results),
        "metrics": results,
        "parallel_impact": {
            "fsdp2": pair["conclusion"] if baseline["fsdp2"] != target["fsdp2"] else "N/A",
            "sp": "N/A",
            "ep": "N/A",
        },
        "scope": f"{baseline['kind']} numerical comparison: {baseline['platform']} -> {target['platform']}"
        if accelerated
        else "CPU toolchain validation; not GPU/NPU precision acceptance",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / ("accuracy_report.json" if accelerated else "comparison_report.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--config", required=True)
    prep.add_argument("--weights")
    prep.add_argument("--batch-file")
    prep.add_argument("--records", help="Optional JSONL with local image paths, resolved relative to the JSONL.")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--grid", type=int, default=4)
    prep.add_argument("--text-tokens", type=int, default=16)
    prep.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--fixture", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", choices=["cpu", "cuda", "npu"], required=True)
    run.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    run.add_argument("--backend", choices=["veomni", "diffusers"], default="veomni")
    run.add_argument("--steps", type=int, default=1)
    run.add_argument("--learning-rate", type=float, default=1e-5)
    conditioning = commands.add_parser("encode")
    conditioning.add_argument("--fixture", required=True)
    conditioning.add_argument("--snapshot", required=True)
    conditioning.add_argument("--training-config", required=True)
    conditioning.add_argument("--output", required=True)
    conditioning.add_argument("--device", choices=["cpu", "cuda", "npu"], required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reference", required=True)
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--atol", type=float, required=True)
    comparison.add_argument("--rtol", type=float, required=True)
    comparison.add_argument("--rationale", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["config_path"] = args.pop("config")
        result = prepare(**args)
    elif command == "run":
        result = capture(**args)
    elif command == "encode":
        result = encode(**args)
    else:
        result = compare(**args)
    if result is not None:
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("fixture_id", "complete", "conclusion", "tensor_count", "scope")
                    if key in result
                }
            )
        )
    return int(result is not None and (result.get("conclusion") == "FAIL" or result.get("nan_inf", False)))


if __name__ == "__main__":
    raise SystemExit(main())
