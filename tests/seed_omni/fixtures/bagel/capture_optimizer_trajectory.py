"""Capture official BAGEL multi-step optimizer trajectory fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.seed_omni.fixtures.bagel.capture_gradient import (
    DEFAULT_TEXT,
    _build_model,
    _case_settings,
    _configure_determinism,
    _cpu_batch,
    _maybe_grad_entry,
    _patched_randn_like,
    _prepare_case,
    _resolve_dtype,
)


DEFAULT_OUTPUT = Path("outputs/bagel_v2/parity/optimizer_trajectory_bf16.pt")


def _sample_param(param: torch.nn.Parameter, rows: torch.Tensor | None = None) -> torch.Tensor:
    data = param.detach().cpu()
    if rows is not None:
        return data[rows.detach().cpu().to(dtype=torch.long)]
    if data.dim() >= 2:
        return data[:4, :4]
    return data[:16]


def _maybe_param_entry(
    model: torch.nn.Module,
    *,
    name: str,
    official_param: str,
    v2_module: str,
    v2_param: str,
    rows: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    param = dict(model.named_parameters()).get(official_param)
    if param is None:
        return None
    entry: dict[str, Any] = {
        "official_param": official_param,
        "v2_module": v2_module,
        "v2_param": v2_param,
        "value": _sample_param(param, rows),
    }
    if rows is not None:
        entry["rows"] = rows.detach().cpu().to(dtype=torch.long)
    return {name: entry}


def _capture_parameters(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any] | None] = []
    text_ids = batch.get("packed_text_ids")
    label_ids = batch.get("packed_label_ids")
    last_layer = model.config.llm_config.num_hidden_layers - 1
    entries.extend(
        [
            _maybe_param_entry(
                model,
                name="early_q_proj",
                official_param="language_model.model.layers.0.self_attn.q_proj.weight",
                v2_module="qwen2_mot",
                v2_param="model.layers.0.self_attn.q_proj.weight",
            ),
            _maybe_param_entry(
                model,
                name="late_mlp_down",
                official_param=f"language_model.model.layers.{last_layer}.mlp.down_proj.weight",
                v2_module="qwen2_mot",
                v2_param=f"model.layers.{last_layer}.mlp.down_proj.weight",
            ),
            _maybe_param_entry(
                model,
                name="gen_q_proj",
                official_param="language_model.model.layers.0.self_attn.q_proj_moe_gen.weight",
                v2_module="qwen2_mot",
                v2_param="model.layers.0.self_attn.q_proj_moe_gen.weight",
            ),
            _maybe_param_entry(
                model,
                name="gen_mlp_down",
                official_param=f"language_model.model.layers.{last_layer}.mlp_moe_gen.down_proj.weight",
                v2_module="qwen2_mot",
                v2_param=f"model.layers.{last_layer}.mlp_moe_gen.down_proj.weight",
            ),
        ]
    )
    if torch.is_tensor(text_ids) and torch.is_tensor(label_ids):
        entries.append(
            _maybe_param_entry(
                model,
                name="text_embed_rows",
                official_param="language_model.model.embed_tokens.weight",
                v2_module="text_encoder",
                v2_param="embed_tokens.weight",
                rows=torch.unique(text_ids.detach().cpu()).to(dtype=torch.long),
            )
        )
    if torch.is_tensor(label_ids):
        entries.append(
            _maybe_param_entry(
                model,
                name="lm_head_rows",
                official_param="language_model.lm_head.weight",
                v2_module="text_encoder",
                v2_param="lm_head.weight",
                rows=torch.unique(label_ids.detach().cpu()).to(dtype=torch.long),
            )
        )
    entries.extend(
        [
            _maybe_param_entry(
                model,
                name="siglip_patch_embed",
                official_param="vit_model.vision_model.embeddings.patch_embedding.weight",
                v2_module="siglip_navit",
                v2_param="vision_model.embeddings.patch_embedding.weight",
            ),
            _maybe_param_entry(
                model,
                name="visual_connector_fc1",
                official_param="connector.fc1.weight",
                v2_module="siglip_navit",
                v2_param="connector.fc1.weight",
            ),
            _maybe_param_entry(
                model,
                name="vae2llm",
                official_param="vae2llm.weight",
                v2_module="flow_connector",
                v2_param="vae2llm.weight",
            ),
            _maybe_param_entry(
                model,
                name="llm2vae",
                official_param="llm2vae.weight",
                v2_module="flow_connector",
                v2_param="llm2vae.weight",
            ),
        ]
    )
    params: dict[str, Any] = {}
    for entry in entries:
        if entry is not None:
            params.update(entry)
    return params


def _run_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    fixed_noise: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    autocast_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    with autocast_context, _patched_randn_like(fixed_noise):
        output = model(**batch)
    loss = None
    ce = output.get("ce")
    mse = output.get("mse")
    if ce is not None:
        loss = ce.mean()
    if mse is not None:
        mse_loss = mse.mean()
        loss = mse_loss if loss is None else loss + mse_loss
    if loss is None:
        raise RuntimeError("Official optimizer trajectory step produced no loss.")
    return loss


def capture(args: argparse.Namespace) -> dict[str, Any]:
    _configure_determinism(args.seed)
    visual_und, visual_gen = _case_settings(args.case)
    model, tokenizer, device = _build_model(args, visual_und=visual_und, visual_gen=visual_gen)
    batch, fixed_noise = _prepare_case(args, tokenizer, device)
    dtype = _resolve_dtype(args.dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    steps = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _run_loss(model, batch, fixed_noise=fixed_noise, dtype=dtype, device=device)
        loss.backward()
        gradients = {
            name: entry
            for item in (
                _maybe_grad_entry(
                    model,
                    name=name,
                    official_param=param["official_param"],
                    v2_module=param["v2_module"],
                    v2_param=param["v2_param"],
                    rows=param.get("rows"),
                )
                for name, param in _capture_parameters(model, batch).items()
            )
            if item is not None
            for name, entry in item.items()
        }
        optimizer.step()
        steps.append(
            {
                "step": step,
                "loss": loss.detach().cpu(),
                "gradients": gradients,
                "parameters_after_step": _capture_parameters(model, batch),
            }
        )

    prepared = _cpu_batch(batch)
    if fixed_noise is not None:
        prepared["fixed_noise"] = fixed_noise.detach().cpu()
        shifted_timesteps = torch.sigmoid(batch["packed_timesteps"])
        shifted_timesteps = (
            args.timestep_shift * shifted_timesteps / (1 + (args.timestep_shift - 1) * shifted_timesteps)
        )
        prepared["shifted_timesteps"] = shifted_timesteps.detach().cpu()

    return {
        "metadata": {
            "schema_version": 1,
            "case_id": args.case,
            "boundary": "official.Bagel.forward.optimizer_trajectory",
            "dtype": args.dtype,
            "seed": args.seed,
            "steps": args.steps,
            "optimizer": "sgd",
            "lr": args.lr,
            "official_repo": str(args.official_repo),
            "official_checkpoint": str(args.model_root),
            "device": str(device),
        },
        "prepared": prepared,
        "trajectory": steps,
        "tolerances": {
            "bf16": {
                "max_abs_diff": 0.0,
                "mean_abs_diff": 0.0,
                "cosine_similarity_min": 0.999,
                "relative_l2_max": 0.0,
                "loss_max_abs_diff": 0.0,
                "near_zero_norm": 0.0,
            },
            "fp32": {
                "max_abs_diff": 1e-5,
                "mean_abs_diff": 1e-6,
                "cosine_similarity_min": 0.99999,
                "relative_l2_max": 1e-4,
                "loss_max_abs_diff": 1e-5,
                "near_zero_norm": 1e-8,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case",
        choices=("gradient_ce_only", "gradient_text_image_ce", "gradient_mse_only", "gradient_ce_mse"),
        default="gradient_ce_mse",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vit-tokens", type=int, default=2)
    parser.add_argument("--vit-patch-dim", type=int, default=3 * 14 * 14)
    parser.add_argument("--vit-max-num-patch-per-side", type=int, default=70)
    parser.add_argument("--latent-grid", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--latent-patch-size", type=int, default=2)
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = capture(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.output)
    print(json.dumps({"output": str(args.output), "case_id": args.case, "steps": args.steps}, indent=2))


if __name__ == "__main__":
    main()
