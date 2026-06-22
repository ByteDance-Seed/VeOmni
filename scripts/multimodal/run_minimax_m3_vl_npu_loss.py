#!/usr/bin/env python3
"""Run a MiniMax M3 VL generated-model loss smoke on Ascend NPU.

This script is intentionally dependency-light: it writes an SVG loss curve
without matplotlib so the root/NPU environment only needs the project runtime
stack, torch, torch_npu, and transformers>=5.12.0.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="./tests/toy_config/minimax_m3_vl_toy/config.json")
    parser.add_argument("--dataset-path", default="./tests/fixtures/minimax_m3_vl_sft/tiny_sft.jsonl")
    parser.add_argument(
        "--output-dir",
        default="./docs/usage/support_new_models/artifacts/minimax_m3_vl_npu_loss_smoke",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError as exc:
        return {"command": command, "returncode": None, "output": str(exc)}
    return {"command": command, "returncode": completed.returncode, "output": completed.stdout}


def npu_smi_command() -> list[str]:
    for path in ("/usr/local/sbin/npu-smi", "/usr/local/bin/npu-smi", "npu-smi"):
        if path == "npu-smi" or Path(path).exists():
            return [path, "info"]
    return ["/usr/local/sbin/npu-smi", "info"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o755)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o644)


def write_svg_loss_curve(losses: list[float], output_path: Path) -> None:
    width = 960
    height = 540
    left = 72
    right = 32
    top = 42
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    min_loss = min(losses)
    max_loss = max(losses)
    span = max(max_loss - min_loss, 1e-9)

    def point(index: int, loss: float) -> tuple[float, float]:
        x = left + (plot_width * index / max(len(losses) - 1, 1))
        y = top + plot_height * (1.0 - ((loss - min_loss) / span))
        return x, y

    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(i, loss) for i, loss in enumerate(losses)))
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#1d4ed8" />'
        for x, y in (point(i, loss) for i, loss in enumerate(losses))
    )
    x_axis = top + plot_height
    first_label = f"{losses[0]:.4f}"
    last_label = f"{losses[-1]:.4f}"
    min_label = f"{min_loss:.4f}"
    max_label = f"{max_loss:.4f}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2:.0f}" y="26" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#111827">MiniMax M3 VL NPU generated-model loss</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{x_axis}" stroke="#374151" stroke-width="1.5"/>
  <line x1="{left}" y1="{x_axis}" x2="{width - right}" y2="{x_axis}" stroke="#374151" stroke-width="1.5"/>
  <text x="20" y="{top + 6}" font-family="sans-serif" font-size="13" fill="#374151">{max_label}</text>
  <text x="20" y="{x_axis}" font-family="sans-serif" font-size="13" fill="#374151">{min_label}</text>
  <text x="{left}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#374151">1</text>
  <text x="{width - right}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#374151">{len(losses)}</text>
  <polyline points="{points}" fill="none" stroke="#1d4ed8" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  {circles}
  <text x="{left + 12}" y="{top + 28}" font-family="sans-serif" font-size="14" fill="#111827">first: {first_label}</text>
  <text x="{left + 12}" y="{top + 50}" font-family="sans-serif" font-size="14" fill="#111827">last: {last_label}</text>
  <text x="{width / 2:.0f}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#374151">training step</text>
</svg>
"""
    output_path.write_text(svg)
    output_path.chmod(0o644)


def load_jsonl_batcher(dataset_path: Path, batch_size: int, seed: int):
    rows = []
    with dataset_path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty dataset: {dataset_path}")
    rng = random.Random(seed)

    def batch_for_step(step: int, torch_module: Any, device: Any, vocab_size: int) -> dict[str, Any]:
        if step % max(len(rows), 1) == 0:
            rng.shuffle(rows)
        selected = [rows[(step * batch_size + offset) % len(rows)] for offset in range(batch_size)]
        input_ids = torch_module.tensor([row["input_ids"] for row in selected], dtype=torch_module.long, device=device)
        labels = torch_module.tensor([row["labels"] for row in selected], dtype=torch_module.long, device=device)
        input_ids = input_ids.remainder(vocab_size)
        active = labels != -100
        labels = labels.clone()
        labels[active] = labels[active].remainder(vocab_size)
        attention_mask = torch_module.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return rows, batch_for_step


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o755)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runtime_probe: dict[str, Any] = {
        "started_at": started_at,
        "effective_uid": os.geteuid(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "env": {
            key: os.environ.get(key, "")
            for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "LD_LIBRARY_PATH", "PYTHONPATH", "MODELING_BACKEND")
        },
        "npu_smi_info": run_command(npu_smi_command()),
    }

    try:
        import torch
        import torch_npu  # noqa: F401
        import transformers

        from veomni.models.loader import get_model_class, get_model_config

        if not hasattr(torch, "npu"):
            raise RuntimeError("torch.npu namespace is unavailable after importing torch_npu")
        if not torch.npu.is_available():
            raise RuntimeError("torch.npu.is_available() is false")
        if torch.npu.device_count() < 1:
            raise RuntimeError("torch.npu.device_count() returned zero")

        device = torch.device(args.device)
        torch.npu.set_device(device)
        tensor_smoke = (torch.ones(4, device=device) + 1).cpu().tolist()
        torch.npu.synchronize()

        runtime_probe.update(
            {
                "torch_version": torch.__version__,
                "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
                "transformers_version": transformers.__version__,
                "torch_npu_available": torch.npu.is_available(),
                "torch_npu_device_count": torch.npu.device_count(),
                "torch_npu_current_device": torch.npu.current_device(),
                "torch_npu_device_name": torch.npu.get_device_name(0),
                "tensor_smoke": tensor_smoke,
            }
        )

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        config = get_model_config(args.config_path)
        model_cls = get_model_class(config)
        model = model_cls(config).to(device)
        model.train()

        dataset_rows, batch_for_step = load_jsonl_batcher(
            Path(args.dataset_path), args.batch_size, args.seed
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        losses: list[float] = []

        for step in range(args.steps):
            batch = batch_for_step(step, torch, device, config.text_config.vocab_size)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.npu.synchronize()
            losses.append(float(loss.detach().cpu()))

        curve_path = output_dir / "loss_curve.svg"
        write_svg_loss_curve(losses, curve_path)
        result = {
            "passed": losses[-1] < losses[0],
            "date": started_at,
            "device": args.device,
            "model_class": f"{model_cls.__module__}.{model_cls.__name__}",
            "config_path": args.config_path,
            "dataset_path": args.dataset_path,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "actual_weights_loaded": False,
            "num_examples": len(dataset_rows),
            "losses": losses,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "min_loss": min(losses),
            "loss_curve": str(curve_path),
            "runtime_probe": runtime_probe,
        }
        write_json(output_dir / "npu_generated_model_loss_log.json", result)
        write_json(output_dir / "npu_runtime_probe.json", runtime_probe)
        print(json.dumps({k: result[k] for k in ("passed", "first_loss", "last_loss", "min_loss", "loss_curve")}, indent=2))
        if not result["passed"]:
            raise RuntimeError(f"expected final loss to be lower than initial loss, got {losses[0]} -> {losses[-1]}")
    except Exception as exc:
        runtime_probe["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output_dir / "npu_runtime_failure.json", runtime_probe)
        raise


if __name__ == "__main__":
    main()
