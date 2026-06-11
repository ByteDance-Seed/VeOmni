"""Compare V2 BAGEL graph-level optimizer trajectory against official fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.seed_omni.fixtures.bagel.compare_gradient_graph import (
    _gradient_modules,
    _load_graph_config,
    _load_graph_modules,
)
from tests.seed_omni.fixtures.bagel.compare_gradient_module import (
    _collect_gradients,
    _configure_determinism,
    _loss_passes,
    _param,
    _passes,
    _resolve_dtype,
    _tensor_metrics,
    _to_device,
)
from veomni.models.seed_omni.modeling_omni import OmniModel


def _sample_param(param: torch.nn.Parameter, rows: torch.Tensor | None = None) -> torch.Tensor:
    data = param.detach().cpu()
    if rows is not None:
        return data[rows.detach().cpu().to(dtype=torch.long)]
    if data.dim() >= 2:
        return data[:4, :4]
    return data[:16]


def _collect_parameters(
    modules: dict[str, torch.nn.Module],
    expected_parameters: dict[str, Any],
) -> dict[str, torch.Tensor]:
    actual: dict[str, torch.Tensor] = {}
    for name, expected in expected_parameters.items():
        module = modules[expected["v2_module"]]
        actual[name] = _sample_param(_param(module, expected["v2_param"]), expected.get("rows"))
    return actual


def _parameter_targets(expected_parameters: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {name: data["value"] for name, data in expected_parameters.items()}


def compare_optimizer_trajectory_graph(
    fixture_path: Path,
    model_root: Path,
    *,
    config_dir: Path = Path("configs/seed_omni/Bagel/bagel_7b_mot"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
) -> dict[str, Any]:
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    _configure_determinism(int(fixture["metadata"].get("seed", 1234)))
    tolerance = fixture["tolerances"][fixture["metadata"]["dtype"]]

    graph_modules = _load_graph_modules(model_root, device=torch_device, dtype=torch_dtype)
    gradient_modules = _gradient_modules(graph_modules)
    model = OmniModel(_load_graph_config(config_dir), graph_modules).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(fixture["metadata"]["lr"]))

    autocast_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=torch_dtype)
        if torch_device.type == "cuda" and torch_dtype != torch.float32
        else nullcontext()
    )

    step_reports = []
    for expected_step in fixture["trajectory"]:
        optimizer.zero_grad(set_to_none=True)
        batch = _to_device(fixture["prepared"], torch_device)
        with autocast_context:
            outputs = model(bagel_packed_batch=batch)
            loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("BAGEL graph optimizer trajectory produced no V2 loss.")
        loss.backward()
        loss_metrics = _tensor_metrics(loss.detach().cpu().reshape(1), expected_step["loss"].reshape(1))
        loss_metrics["passes"] = _loss_passes(loss_metrics, tolerance)

        actual_gradients = _collect_gradients(gradient_modules, expected_step["gradients"])
        gradient_metrics: dict[str, Any] = {}
        for name, actual in actual_gradients.items():
            metrics = _tensor_metrics(actual, expected_step["gradients"][name]["grad"])
            metrics["passes"] = _passes(metrics, tolerance)
            gradient_metrics[name] = metrics

        optimizer.step()

        actual_parameters = _collect_parameters(gradient_modules, expected_step["parameters_after_step"])
        expected_parameters = _parameter_targets(expected_step["parameters_after_step"])
        parameter_metrics: dict[str, Any] = {}
        for name, actual in actual_parameters.items():
            metrics = _tensor_metrics(actual, expected_parameters[name])
            metrics["passes"] = _passes(metrics, tolerance)
            parameter_metrics[name] = metrics

        step_reports.append(
            {
                "step": int(expected_step["step"]),
                "loss": loss_metrics,
                "gradients": gradient_metrics,
                "parameters_after_step": parameter_metrics,
                "passes": bool(
                    loss_metrics["passes"]
                    and all(item["passes"] for item in gradient_metrics.values())
                    and all(item["passes"] for item in parameter_metrics.values())
                ),
            }
        )

    return {
        "case_id": fixture["metadata"]["case_id"],
        "dtype": dtype,
        "optimizer": fixture["metadata"]["optimizer"],
        "lr": fixture["metadata"]["lr"],
        "steps": step_reports,
        "all_pass": all(step["passes"] for step in step_reports),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/seed_omni/Bagel/bagel_7b_mot"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_optimizer_trajectory_graph(
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
