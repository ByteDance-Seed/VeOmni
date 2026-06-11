"""Compare V2 BAGEL graph-level backward outputs against official gradient fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.seed_omni.fixtures.bagel.compare_gradient_module import (
    _collect_gradients,
    _configure_determinism,
    _gradient_targets,
    _loss_passes,
    _passes,
    _resolve_dtype,
    _tensor_metrics,
    _to_device,
)
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.bagel.flow_connector.modeling import BagelFlowConnector
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT
from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit
from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder
from veomni.models.seed_omni.modules.bagel.vae.modeling import BagelVAE


def _load_graph_config(config_dir: Path) -> OmniConfig:
    return OmniConfig.from_dict(
        {
            "modules": yaml.safe_load((config_dir / "modules_train.yaml").read_text(encoding="utf-8")),
            "training_graph": yaml.safe_load((config_dir / "graph_train.yaml").read_text(encoding="utf-8"))[
                "training_graph"
            ],
        }
    )


def _load_graph_modules(model_root: Path, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {
        "bagel_text_encoder": BagelTextEncoder.from_pretrained(model_root / "bagel_text_encoder", torch_dtype=dtype),
        "bagel_siglip_navit": BagelSiglipNavit.from_pretrained(model_root / "bagel_siglip_navit", torch_dtype=dtype),
        "bagel_vae": BagelVAE.from_pretrained(model_root / "bagel_vae", torch_dtype=dtype),
        "bagel_flow_connector": BagelFlowConnector.from_pretrained(
            model_root / "bagel_flow_connector", torch_dtype=dtype
        ),
        "bagel_qwen2_mot": BagelQwen2MoT.from_pretrained(model_root / "bagel_qwen2_mot", torch_dtype=dtype),
    }
    for name, module in modules.items():
        if name == "bagel_qwen2_mot":
            # Keep RoPE buffers in fp32, matching official BAGEL.
            module.to(device=device).train()
        else:
            module.to(device=device, dtype=dtype).train()
        module.zero_grad(set_to_none=True)
    return modules


def _gradient_modules(graph_modules: dict[str, torch.nn.Module]) -> dict[str, torch.nn.Module]:
    return {
        "text_encoder": graph_modules["bagel_text_encoder"],
        "siglip_navit": graph_modules["bagel_siglip_navit"],
        "flow_connector": graph_modules["bagel_flow_connector"],
        "qwen2_mot": graph_modules["bagel_qwen2_mot"],
    }


def compare_gradient_graph(
    fixture_path: Path,
    model_root: Path,
    *,
    config_dir: Path = Path("configs/seed_omni/Bagel/bagel_7b_mot"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
) -> dict[str, Any]:
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    case_id = fixture.get("metadata", {}).get("case_id")
    if not isinstance(case_id, str) or not case_id.startswith("gradient_"):
        raise ValueError(f"Unsupported BAGEL gradient fixture case: {case_id!r}")

    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    _configure_determinism(int(fixture["metadata"].get("seed", 1234)))
    tolerance = fixture["tolerances"][fixture["metadata"]["dtype"]]
    batch = _to_device(fixture["prepared"], torch_device)
    graph_modules = _load_graph_modules(model_root, device=torch_device, dtype=torch_dtype)
    model = OmniModel(_load_graph_config(config_dir), graph_modules).train()

    autocast_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=torch_dtype)
        if torch_device.type == "cuda" and torch_dtype != torch.float32
        else nullcontext()
    )
    with autocast_context:
        outputs = model(bagel_packed_batch=batch)
        loss = outputs["loss"]
    if loss is None:
        raise RuntimeError("BAGEL graph-level gradient fixture produced no V2 loss.")
    loss.backward()

    loss_metrics = _tensor_metrics(loss.detach().cpu().reshape(1), fixture["losses"]["total"].reshape(1))
    loss_metrics["passes"] = _loss_passes(loss_metrics, tolerance)

    actual_gradients = _collect_gradients(_gradient_modules(graph_modules), fixture["gradients"])
    expected_gradients = _gradient_targets(fixture["gradients"])
    gradient_metrics: dict[str, Any] = {}
    for name, actual in actual_gradients.items():
        metrics = _tensor_metrics(actual, expected_gradients[name])
        metrics["passes"] = _passes(metrics, tolerance)
        gradient_metrics[name] = metrics

    all_pass = bool(loss_metrics["passes"] and all(item["passes"] for item in gradient_metrics.values()))
    return {
        "case_id": case_id,
        "dtype": dtype,
        "tolerance": tolerance,
        "loss": loss_metrics,
        "gradients": gradient_metrics,
        "all_pass": all_pass,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/seed_omni/Bagel/bagel_7b_mot"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_gradient_graph(
        args.fixture,
        args.model_root,
        config_dir=args.config_dir,
        device=args.device,
        dtype=args.dtype,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
