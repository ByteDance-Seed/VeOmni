"""Compare V2 BAGEL text graph-level outputs against an official fixture."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import adapt_text_only_fixture, assert_text_fixture_schema  # noqa: E402

from veomni.models.seed_omni.configuration_omni import OmniConfig  # noqa: E402
from veomni.models.seed_omni.modeling_omni import OmniModel  # noqa: E402
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT, NaiveCache  # noqa: E402
from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder  # noqa: E402


class _NoopGenerateModule(nn.Module):
    def generate(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}


class _UnusedModule(nn.Module):
    def encode(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}

    def embed_latent(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_match": False, "passes": False}

    a_float = a.detach().float()
    b_float = b.detach().float()
    diff = (a_float - b_float).abs()
    cosine = 1.0
    if a.numel() > 0:
        cosine = float(F.cosine_similarity(a_float.reshape(1, -1), b_float.reshape(1, -1), dim=-1).item())
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "shape_match": True,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": cosine,
    }


def _passes(metrics: dict[str, Any], tolerance: dict[str, float]) -> bool:
    return bool(
        metrics.get("shape_match")
        and metrics["max_abs_diff"] <= tolerance["max_abs_diff"]
        and metrics["mean_abs_diff"] <= tolerance["mean_abs_diff"]
        and metrics["cosine_similarity"] >= tolerance["cosine_similarity_min"]
    )


def _cache_to_cpu(cache: NaiveCache) -> dict[str, Any]:
    return {
        "num_layers": cache.num_layers,
        "key": [
            None if cache.key_cache[idx] is None else cache.key_cache[idx].detach().cpu()
            for idx in range(cache.num_layers)
        ],
        "value": [
            None if cache.value_cache[idx] is None else cache.value_cache[idx].detach().cpu()
            for idx in range(cache.num_layers)
        ],
    }


def _compare_cache(
    v2_cache: dict[str, Any], official_cache: dict[str, Any], tolerance: dict[str, float]
) -> dict[str, Any]:
    if v2_cache["num_layers"] != official_cache["num_layers"]:
        return {"num_layers_match": False, "passes": False}

    aggregate = {
        "max_abs_diff": 0.0,
        "mean_abs_diff_max": 0.0,
        "cosine_similarity_min": 1.0,
        "all_shapes_match": True,
    }
    layers: dict[str, Any] = {}
    for layer_idx in range(v2_cache["num_layers"]):
        layer_result: dict[str, Any] = {}
        for kind in ("key", "value"):
            v2_tensor = v2_cache[kind][layer_idx]
            official_tensor = official_cache[kind][layer_idx]
            name = f"{kind}_{layer_idx}"
            if v2_tensor is None or official_tensor is None:
                both_none = v2_tensor is None and official_tensor is None
                layer_result[name] = {"both_none": both_none, "passes": both_none}
                aggregate["all_shapes_match"] = aggregate["all_shapes_match"] and both_none
                continue
            metrics = _tensor_metrics(v2_tensor, official_tensor)
            metrics["passes"] = _passes(metrics, tolerance)
            layer_result[name] = metrics
            aggregate["all_shapes_match"] = aggregate["all_shapes_match"] and bool(metrics.get("shape_match"))
            if metrics.get("shape_match"):
                aggregate["max_abs_diff"] = max(aggregate["max_abs_diff"], metrics["max_abs_diff"])
                aggregate["mean_abs_diff_max"] = max(aggregate["mean_abs_diff_max"], metrics["mean_abs_diff"])
                aggregate["cosine_similarity_min"] = min(
                    aggregate["cosine_similarity_min"], metrics["cosine_similarity"]
                )
        layers[str(layer_idx)] = layer_result

    passes = (
        aggregate["all_shapes_match"]
        and aggregate["max_abs_diff"] <= tolerance["max_abs_diff"]
        and aggregate["mean_abs_diff_max"] <= tolerance["mean_abs_diff"]
        and aggregate["cosine_similarity_min"] >= tolerance["cosine_similarity_min"]
    )
    return {"num_layers_match": True, "aggregate": aggregate, "layers": layers, "passes": passes}


def _v2_tolerance(fixture: dict[str, Any]) -> dict[str, float]:
    dtype_tolerances = fixture["tolerances"][fixture["metadata"]["dtype"]]
    if "v2_parity" in dtype_tolerances:
        return dtype_tolerances["v2_parity"]
    return dtype_tolerances


def _load_modules(
    model_root: Path, *, device: torch.device, dtype: torch.dtype
) -> tuple[BagelTextEncoder, BagelQwen2MoT]:
    text_encoder = BagelTextEncoder.from_pretrained(
        model_root / "bagel_text_encoder",
        torch_dtype=dtype,
    )
    qwen2_mot = BagelQwen2MoT.from_pretrained(
        model_root / "bagel_qwen2_mot",
        torch_dtype=dtype,
    )
    text_encoder.to(device=device, dtype=dtype).eval()
    # Official Bagel loads parameters as bf16 but leaves RoPE frequency buffers
    # in fp32. Move the module without a dtype cast to preserve that behavior.
    qwen2_mot.to(device=device).eval()
    return text_encoder, qwen2_mot


def _load_graph_config(config_dir: Path, *, infer_yaml_name: str = "infer_und.yaml") -> OmniConfig:
    modules_path = config_dir / "modules_train.yaml"
    if modules_path.exists():
        train_config = {
            "modules": yaml.safe_load(modules_path.read_text(encoding="utf-8")),
            "training_graph": yaml.safe_load((config_dir / "graph_train.yaml").read_text(encoding="utf-8"))[
                "training_graph"
            ],
        }
        infer_path = config_dir / infer_yaml_name
        if not infer_path.exists() and infer_yaml_name.startswith("infer_"):
            infer_path = config_dir / f"graph_{infer_yaml_name}"
        infer_config = yaml.safe_load(infer_path.read_text(encoding="utf-8"))
    else:
        train_config = yaml.safe_load((config_dir / "train.yaml").read_text(encoding="utf-8"))
        infer_config = yaml.safe_load((config_dir / infer_yaml_name).read_text(encoding="utf-8"))
    merged = {
        "modules": train_config["modules"],
        "training_graph": train_config["training_graph"],
        "generation_graph": infer_config["generation_graph"],
        "generation_kwargs": infer_config.get("generation_kwargs", {}),
    }
    return OmniConfig.from_dict(merged)


@torch.no_grad()
def compare_text_graph(
    fixture_path: Path,
    model_root: Path,
    *,
    config_dir: Path,
    infer_yaml_name: str = "infer_und.yaml",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
) -> dict[str, Any]:
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    assert_text_fixture_schema(fixture)
    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    tolerance = _v2_tolerance(fixture)

    text_encoder, qwen2_mot = _load_modules(model_root, device=torch_device, dtype=torch_dtype)
    config = _load_graph_config(config_dir, infer_yaml_name=infer_yaml_name)
    model = OmniModel(
        config,
        {
            "bagel_text_encoder": text_encoder,
            "bagel_siglip_navit": _NoopGenerateModule(),
            "bagel_vae": _UnusedModule(),
            "bagel_flow_connector": _UnusedModule(),
            "bagel_qwen2_mot": qwen2_mot,
        },
    ).eval()

    conversation = _to_device(adapt_text_only_fixture(fixture), torch_device)
    trace: list[str] = []
    generation_kwargs = dict(config.generation_kwargs or {})
    generation_kwargs.update({"max_new_tokens": 1, "do_sample": False, "temperature": 1.0, "top_p": 1.0})
    ctx = model.generate({"conversation_list": conversation}, trace=trace, generation_kwargs=generation_kwargs)

    hidden_state = ctx["bagel_last_hidden_state"].detach().cpu()
    logits = ctx["bagel_last_logits"].detach().cpu()
    greedy_token = ctx["bagel_last_greedy_token"].detach().cpu()

    hidden_metrics = _tensor_metrics(hidden_state, fixture["one_step"]["hidden_state"])
    logits_metrics = _tensor_metrics(logits, fixture["one_step"]["logits"])
    hidden_metrics["passes"] = _passes(hidden_metrics, tolerance)
    logits_metrics["passes"] = _passes(logits_metrics, tolerance)
    greedy_match = torch.equal(greedy_token, fixture["one_step"]["greedy_token"])
    cache_after_step = _compare_cache(
        _cache_to_cpu(ctx["past_key_values"]),
        fixture["one_step"]["cache_after_step"],
        tolerance,
    )

    all_pass = hidden_metrics["passes"] and logits_metrics["passes"] and greedy_match and cache_after_step["passes"]
    return {
        "case_id": fixture["metadata"]["case_id"],
        "dtype": dtype,
        "tolerance": tolerance,
        "trace": trace,
        "hidden_state": hidden_metrics,
        "logits": logits_metrics,
        "greedy_token_match": greedy_match,
        "cache_after_step": cache_after_step,
        "all_pass": all_pass,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/seed_omni/Bagel/bagel_7b_mot"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_text_graph(
        args.fixture,
        args.model_root,
        config_dir=args.config_dir,
        device=args.device,
        dtype=args.dtype,
    )
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
