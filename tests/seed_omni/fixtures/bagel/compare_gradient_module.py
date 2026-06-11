"""Compare V2 BAGEL module-level backward outputs against official gradient fixtures."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from veomni.models.seed_omni.modules.bagel.flow_connector.modeling import BagelFlowConnector
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT
from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit
from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")


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
    actual_norm = float(a_float.norm().item())
    expected_norm = float(b_float.norm().item())
    cosine = 1.0
    if a.numel() > 0 and actual_norm > 0 and expected_norm > 0:
        cosine = float(F.cosine_similarity(a_float.reshape(1, -1), b_float.reshape(1, -1), dim=-1).item())
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "shape_match": True,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "actual_norm": actual_norm,
        "expected_norm": expected_norm,
        "relative_l2": float((a_float - b_float).norm().item() / max(expected_norm, 1e-12)),
        "cosine_similarity": cosine,
    }


def _passes(metrics: dict[str, Any], tolerance: dict[str, float]) -> bool:
    if not metrics.get("shape_match"):
        return False
    near_zero_norm = tolerance.get("near_zero_norm", 0.0)
    if metrics["actual_norm"] <= near_zero_norm and metrics["expected_norm"] <= near_zero_norm:
        return (
            metrics["max_abs_diff"] <= tolerance["max_abs_diff"]
            and metrics["mean_abs_diff"] <= tolerance["mean_abs_diff"]
        )
    return bool(
        (
            metrics["max_abs_diff"] <= tolerance["max_abs_diff"]
            and metrics["mean_abs_diff"] <= tolerance["mean_abs_diff"]
            and metrics["cosine_similarity"] >= tolerance["cosine_similarity_min"]
        )
        or (
            metrics["relative_l2"] <= tolerance["relative_l2_max"]
            and metrics["cosine_similarity"] >= tolerance["cosine_similarity_min"]
        )
    )


def _loss_passes(metrics: dict[str, Any], tolerance: dict[str, float]) -> bool:
    return bool(
        metrics.get("shape_match")
        and (
            metrics["max_abs_diff"] <= tolerance.get("loss_max_abs_diff", tolerance["max_abs_diff"])
            or metrics["relative_l2"] <= tolerance["relative_l2_max"]
        )
    )


def _load_modules(
    model_root: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    need_siglip: bool,
    need_flow: bool,
) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {
        "text_encoder": BagelTextEncoder.from_pretrained(model_root / "bagel_text_encoder", torch_dtype=dtype),
        "qwen2_mot": BagelQwen2MoT.from_pretrained(model_root / "bagel_qwen2_mot", torch_dtype=dtype),
    }
    if need_siglip:
        modules["siglip_navit"] = BagelSiglipNavit.from_pretrained(
            model_root / "bagel_siglip_navit", torch_dtype=dtype
        )
    if need_flow:
        modules["flow_connector"] = BagelFlowConnector.from_pretrained(
            model_root / "bagel_flow_connector", torch_dtype=dtype
        )
    for module in modules.values():
        module.to(device=device, dtype=dtype).train()
        module.zero_grad(set_to_none=True)
    return modules


def _param(module: torch.nn.Module, name: str) -> torch.nn.Parameter:
    params = dict(module.named_parameters())
    try:
        return params[name]
    except KeyError as exc:
        raise KeyError(f"Missing parameter {name!r}") from exc


def _sample_grad(param: torch.nn.Parameter, rows: torch.Tensor | None = None) -> torch.Tensor:
    grad = param.grad
    if grad is None:
        raise RuntimeError(f"Expected gradient for {param.shape}, got None.")
    if rows is not None:
        return grad.detach().cpu()[rows.detach().cpu().to(dtype=torch.long)]
    if grad.dim() >= 2:
        return grad.detach().cpu()[:4, :4]
    return grad.detach().cpu()[:16]


def _patched_latents(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    if "fixed_noise" not in batch:
        return None
    h, w = batch["patchified_vae_latent_shapes"][0]
    patch_h = batch["padded_latent"].shape[2] // h
    patch_w = batch["padded_latent"].shape[3] // w
    clean = batch["padded_latent"].reshape(
        batch["padded_latent"].shape[0],
        batch["padded_latent"].shape[1],
        h,
        patch_h,
        w,
        patch_w,
    )
    clean = clean.permute(0, 2, 4, 3, 5, 1).flatten(0, 2).flatten(1, 3)
    timesteps = batch["shifted_timesteps"].to(device=clean.device, dtype=clean.dtype).reshape(-1, 1)
    noise = batch["fixed_noise"].to(device=clean.device, dtype=clean.dtype)
    noised = (1.0 - timesteps) * clean + timesteps * noise
    target = noise - clean
    return noised, target


def _run_forward_backward(
    modules: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    text_encoder = modules["text_encoder"]
    qwen2_mot = modules["qwen2_mot"]

    device = batch["packed_text_ids"].device
    autocast_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    with autocast_context:
        text_embeds = text_encoder.encode(batch["packed_text_ids"])["inputs_embeds"]
        packed_sequence = text_embeds.new_zeros((int(batch["sequence_length"]), text_embeds.shape[-1]))
        packed_sequence[batch["packed_text_indexes"]] = text_embeds

        und_indexes = [batch["packed_text_indexes"]]
        if "packed_vit_tokens" in batch:
            siglip_navit = modules["siglip_navit"]
            vit_token_lens = batch["vit_token_seqlens"].to(device=text_embeds.device, dtype=torch.int32).reshape(-1)
            image_embeds = siglip_navit(
                packed_pixel_values=batch["packed_vit_tokens"],
                packed_flattened_position_ids=batch["packed_vit_position_ids"],
                cu_seqlens=torch.nn.functional.pad(torch.cumsum(vit_token_lens, dim=0), (1, 0)).to(torch.int32),
                max_seqlen=int(vit_token_lens.max().item()),
            )["image_embeds"]
            packed_sequence[batch["packed_vit_token_indexes"]] = image_embeds
            und_indexes.append(batch["packed_vit_token_indexes"])

        gen_indexes = None
        mse_target = None
        patched_latents = _patched_latents(batch)
        if patched_latents is not None:
            flow_connector = modules["flow_connector"]
            noised_latents, mse_target = patched_latents
            latent_embeds = flow_connector.embed_latent(
                latents=noised_latents,
                position_ids=batch["packed_latent_position_ids"],
                timesteps=batch["shifted_timesteps"],
            )["latent_embeds"]
            packed_sequence[batch["packed_vae_token_indexes"]] = latent_embeds
            gen_indexes = batch["packed_vae_token_indexes"]

        qwen_output = qwen2_mot(
            packed_sequence=packed_sequence,
            sample_lens=batch["sample_lens"],
            attention_mask=batch["nested_attention_masks"],
            packed_position_ids=batch["packed_position_ids"],
            packed_und_token_indexes=torch.cat(und_indexes),
            packed_gen_token_indexes=gen_indexes,
        )
        hidden_states = qwen_output["hidden_states"]
        losses: dict[str, torch.Tensor] = {}
        loss = None
        if "ce_loss_indexes" in batch:
            logits = text_encoder.decode(hidden_states=hidden_states[batch["ce_loss_indexes"]])["logits"]
            ce = F.cross_entropy(logits, batch["packed_label_ids"], reduction="none")
            losses["ce_vector"] = ce.detach().cpu()
            loss = ce.mean()
        if mse_target is not None:
            velocity = modules["flow_connector"].decode_velocity(
                hidden_states=hidden_states[batch["mse_loss_indexes"]]
            )["velocity"]
            mse = (velocity.float() - mse_target.to(device=velocity.device).float()).square()
            losses["mse_tensor"] = mse.detach().cpu()
            mse_loss = mse.mean()
            loss = mse_loss if loss is None else loss + mse_loss
    if loss is None:
        raise RuntimeError("Gradient fixture produced no V2 loss.")
    loss.backward()
    losses["total"] = loss.detach().cpu()
    return losses


def _collect_gradients(
    modules: dict[str, torch.nn.Module],
    official_gradients: dict[str, Any],
) -> dict[str, torch.Tensor]:
    actual: dict[str, torch.Tensor] = {}
    for name, expected in official_gradients.items():
        module = modules[expected["v2_module"]]
        rows = expected.get("rows")
        actual[name] = _sample_grad(_param(module, expected["v2_param"]), rows)
    return actual


def _gradient_targets(official_gradients: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {name: data["grad"] for name, data in official_gradients.items()}


def compare_gradient_module(
    fixture_path: Path,
    model_root: Path,
    *,
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
    gradients = fixture["gradients"]
    modules = _load_modules(
        model_root,
        device=torch_device,
        dtype=torch_dtype,
        need_siglip=any(data["v2_module"] == "siglip_navit" for data in gradients.values()),
        need_flow=any(data["v2_module"] == "flow_connector" for data in gradients.values()),
    )

    losses = _run_forward_backward(modules, batch, dtype=torch_dtype)
    loss_metrics = _tensor_metrics(losses["total"].reshape(1), fixture["losses"]["total"].reshape(1))
    loss_metrics["passes"] = _loss_passes(loss_metrics, tolerance)

    actual_gradients = _collect_gradients(modules, gradients)
    expected_gradients = _gradient_targets(gradients)
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


def compare_gradient_ce_only(
    fixture_path: Path,
    model_root: Path,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
) -> dict[str, Any]:
    return compare_gradient_module(fixture_path, model_root, device=device, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_gradient_module(args.fixture, args.model_root, device=args.device, dtype=args.dtype)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
