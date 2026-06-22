#!/usr/bin/env python3
"""Compare upstream HF MiniMax M3 VL against VeOmni's generated model.

The verifier uses one deterministic toy/reduced multimodal batch and the same
randomly initialized state_dict for both models. It checks forward outputs,
MoE routing hooks, key gradients, and one AdamW update delta.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_GRAD_NAMES = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.self_attn.k_proj.weight",
    "model.language_model.layers.0.self_attn.v_proj.weight",
    "model.language_model.layers.0.self_attn.o_proj.weight",
    "model.language_model.layers.0.mlp.gate.weight",
    "model.language_model.layers.0.mlp.experts.gate_up_proj",
    "model.language_model.layers.0.mlp.experts.down_proj",
    "model.multi_modal_projector.linear_1.weight",
    "model.multi_modal_projector.linear_2.weight",
    "model.multi_modal_projector.merge_linear_1.weight",
    "model.multi_modal_projector.merge_linear_2.weight",
    "lm_head.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="./tests/toy_config/minimax_m3_vl_toy/config.json")
    parser.add_argument(
        "--output-json",
        default="./docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity.json",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "npu"))
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--image-token-id", type=int, default=250)
    parser.add_argument("--video-token-id", type=int, default=251)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--grad-atol", type=float, default=2e-5)
    parser.add_argument("--grad-rtol", type=float, default=2e-5)
    parser.add_argument("--param-atol", type=float, default=2e-5)
    parser.add_argument("--param-rtol", type=float, default=2e-5)
    parser.add_argument("--grad-name", action="append", dest="grad_names", default=[])
    parser.add_argument("--allow-missing-grad", action="store_true")
    return parser.parse_args()


def set_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if hasattr(torch_module, "cuda"):
        torch_module.cuda.manual_seed_all(seed)


def patch_config(config: Any, image_token_id: int, video_token_id: int) -> Any:
    for name, value in (
        ("image_token_id", image_token_id),
        ("video_token_id", video_token_id),
        ("image_token_index", image_token_id),
        ("video_token_index", video_token_id),
        ("bos_token_id", None),
        ("eos_token_id", None),
    ):
        if hasattr(config, name):
            setattr(config, name, value)
    if hasattr(config, "text_config"):
        if hasattr(config.text_config, "bos_token_id"):
            config.text_config.bos_token_id = None
        if hasattr(config.text_config, "eos_token_id"):
            config.text_config.eos_token_id = None
        if hasattr(config.text_config, "_attn_implementation"):
            config.text_config._attn_implementation = "eager"
        if hasattr(config.text_config, "use_cache"):
            config.text_config.use_cache = False
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config, "use_cache"):
        config.use_cache = False
    return config


def tensor_stats(lhs: Any, rhs: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    lhs_cpu = lhs.detach().float().cpu()
    rhs_cpu = rhs.detach().float().cpu()
    diff = (lhs_cpu - rhs_cpu).abs()
    denom = rhs_cpu.abs().clamp_min(1e-12)
    rel = diff / denom
    return {
        "shape": list(lhs_cpu.shape),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "allclose": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def add_tensor_check(checks: list[dict[str, Any]], name: str, lhs: Any, rhs: Any, *, atol: float, rtol: float) -> None:
    item = {"name": name, "kind": "tensor"}
    item.update(tensor_stats(lhs, rhs, atol=atol, rtol=rtol))
    checks.append(item)


def add_exact_check(checks: list[dict[str, Any]], name: str, lhs: Any, rhs: Any) -> None:
    import torch

    lhs_cpu = lhs.detach().cpu() if hasattr(lhs, "detach") else lhs
    rhs_cpu = rhs.detach().cpu() if hasattr(rhs, "detach") else rhs
    equal = bool(torch.equal(lhs_cpu, rhs_cpu)) if hasattr(lhs_cpu, "shape") else lhs_cpu == rhs_cpu
    checks.append({"name": name, "kind": "exact", "equal": equal, "shape": list(lhs_cpu.shape) if hasattr(lhs_cpu, "shape") else None})


def named_param(model: Any, name: str) -> Any:
    params = dict(model.named_parameters())
    if name not in params:
        raise KeyError(f"parameter not found: {name}")
    return params[name]


def clone_named_params(model: Any, names: Iterable[str]) -> dict[str, Any]:
    return {name: named_param(model, name).detach().clone() for name in names}


def install_router_hooks(model: Any) -> tuple[list[dict[str, Any]], list[Any]]:
    records: list[dict[str, Any]] = []
    handles = []

    def hook(module: Any, _inputs: Any, output: Any) -> None:
        router_logits, top_k_weights, top_k_index = output
        records.append(
            {
                "module": module.__class__.__name__,
                "router_logits": router_logits.detach().clone(),
                "top_k_weights": top_k_weights.detach().clone(),
                "selected_experts": top_k_index.detach().clone(),
            }
        )

    for module in model.modules():
        if module.__class__.__name__ == "MiniMaxM3VLTopKRouter":
            handles.append(module.register_forward_hook(hook))
    return records, handles


def remove_hooks(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.remove()


def build_batch(config: Any, torch_module: Any, device: Any, args: argparse.Namespace) -> dict[str, Any]:
    text = config.text_config
    vocab_size = text.vocab_size
    if max(args.image_token_id, args.video_token_id) >= vocab_size:
        raise ValueError(f"toy token ids must be below vocab_size={vocab_size}")
    if args.seq_len < 6:
        raise ValueError("--seq-len must be at least 6 for the mixed image/video parity batch")

    token_values = [11, args.image_token_id, 23, args.video_token_id, 37, 41, 43, 47, 53, 59]
    while len(token_values) < args.seq_len:
        token_values.append((token_values[-1] + 7) % vocab_size)
    input_ids = torch_module.tensor([token_values[: args.seq_len]], dtype=torch_module.long, device=device)
    attention_mask = torch_module.ones_like(input_ids)
    position_ids = torch_module.arange(args.seq_len, dtype=torch_module.long, device=device).unsqueeze(0)
    labels = input_ids.clone()
    labels[(input_ids == args.image_token_id) | (input_ids == args.video_token_id)] = -100

    vision = config.vision_config
    merge = vision.spatial_merge_size
    num_patches = merge * merge
    pixel_row_size = vision.num_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(args.seed + 17)
    pixel_values = torch_module.randn((num_patches, pixel_row_size), generator=generator, dtype=torch_module.float32).to(device)
    pixel_values_videos = torch_module.randn((num_patches, pixel_row_size), generator=generator, dtype=torch_module.float32).to(device)
    image_grid_thw = torch_module.tensor([[1, merge, merge]], dtype=torch_module.long, device=device)
    video_grid_thw = torch_module.tensor([[1, merge, merge]], dtype=torch_module.long, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "pixel_values_videos": pixel_values_videos,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
    }


def jsonable_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return checks


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MODELING_BACKEND", "veomni")

    import torch
    import transformers
    from transformers import AutoConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3SparseForConditionalGeneration as HFMiniMaxM3SparseForConditionalGeneration,
    )

    if args.device == "npu":
        import torch_npu  # noqa: F401

        device = torch.device("npu:0")
        torch.npu.set_device(device)
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    from veomni.models.loader import get_model_class, get_model_config

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    set_seed(args.seed, torch)

    hf_config = patch_config(AutoConfig.from_pretrained(args.config_path), args.image_token_id, args.video_token_id)
    veomni_config = patch_config(get_model_config(args.config_path), args.image_token_id, args.video_token_id)
    veomni_cls = get_model_class(veomni_config)

    set_seed(args.seed, torch)
    hf_model = HFMiniMaxM3SparseForConditionalGeneration(hf_config)
    source_state = {name: value.detach().clone() for name, value in hf_model.state_dict().items()}
    set_seed(args.seed + 1, torch)
    veomni_model = veomni_cls(veomni_config)
    load_result = veomni_model.load_state_dict(source_state, strict=True)

    hf_model.to(device)
    veomni_model.to(device)
    hf_model.train()
    veomni_model.train()

    batch = build_batch(hf_config, torch, device, args)
    checks: list[dict[str, Any]] = []
    add_exact_check(checks, "input.attention_mask", batch["attention_mask"], batch["attention_mask"].clone())
    add_exact_check(checks, "input.position_ids", batch["position_ids"], batch["position_ids"].clone())
    metadata_expected = {
        "image_grid_thw_list": batch["image_grid_thw"].detach().cpu().tolist(),
        "video_grid_thw_list": batch["video_grid_thw"].detach().cpu().tolist(),
    }
    metadata_actual = {"image_grid_thw": 0, "video_grid_thw": 0}
    if hasattr(veomni_model, "get_metadata_collate_func"):
        metadata_batch = {
            "image_grid_thw": batch["image_grid_thw"].detach().cpu().clone(),
            "video_grid_thw": batch["video_grid_thw"].detach().cpu().clone(),
        }
        veomni_model.get_metadata_collate_func()(metadata_batch, metadata_actual)
        metadata_actual = metadata_batch["multimodal_metadata"]
    checks.append(
        {
            "name": "input.multimodal_metadata_contract",
            "kind": "metadata",
            "expected": metadata_expected,
            "actual": metadata_actual,
            "equal": metadata_actual == metadata_expected,
        }
    )

    hf_router_records, hf_handles = install_router_hooks(hf_model)
    veomni_router_records, veomni_handles = install_router_hooks(veomni_model)
    try:
        hf_outputs = hf_model(**batch)
        veomni_outputs = veomni_model(**batch)
    finally:
        remove_hooks(hf_handles)
        remove_hooks(veomni_handles)

    add_tensor_check(checks, "forward.loss", hf_outputs.loss, veomni_outputs.loss, atol=args.atol, rtol=args.rtol)
    add_tensor_check(checks, "forward.logits", hf_outputs.logits, veomni_outputs.logits, atol=args.atol, rtol=args.rtol)
    add_tensor_check(
        checks,
        "forward.image_hidden_states",
        hf_outputs.image_hidden_states,
        veomni_outputs.image_hidden_states,
        atol=args.atol,
        rtol=args.rtol,
    )
    add_tensor_check(
        checks,
        "forward.video_hidden_states",
        hf_outputs.video_hidden_states,
        veomni_outputs.video_hidden_states,
        atol=args.atol,
        rtol=args.rtol,
    )

    checks.append(
        {
            "name": "router.record_count",
            "kind": "exact",
            "equal": len(hf_router_records) == len(veomni_router_records),
            "hf": len(hf_router_records),
            "veomni": len(veomni_router_records),
        }
    )
    for index, (hf_record, veomni_record) in enumerate(zip(hf_router_records, veomni_router_records, strict=False)):
        add_tensor_check(
            checks,
            f"router.{index}.logits",
            hf_record["router_logits"],
            veomni_record["router_logits"],
            atol=args.atol,
            rtol=args.rtol,
        )
        add_tensor_check(
            checks,
            f"router.{index}.weights",
            hf_record["top_k_weights"],
            veomni_record["top_k_weights"],
            atol=args.atol,
            rtol=args.rtol,
        )
        add_exact_check(checks, f"router.{index}.selected_experts", hf_record["selected_experts"], veomni_record["selected_experts"])

    hf_model.zero_grad(set_to_none=True)
    veomni_model.zero_grad(set_to_none=True)
    hf_outputs.loss.backward()
    veomni_outputs.loss.backward()

    grad_names = tuple(args.grad_names) if args.grad_names else DEFAULT_GRAD_NAMES
    for name in grad_names:
        hf_param = named_param(hf_model, name)
        veomni_param = named_param(veomni_model, name)
        if hf_param.grad is None or veomni_param.grad is None:
            checks.append(
                {
                    "name": f"grad.{name}",
                    "kind": "grad",
                    "allclose": bool(args.allow_missing_grad and hf_param.grad is None and veomni_param.grad is None),
                    "missing": {"hf": hf_param.grad is None, "veomni": veomni_param.grad is None},
                }
            )
            continue
        add_tensor_check(checks, f"grad.{name}", hf_param.grad, veomni_param.grad, atol=args.grad_atol, rtol=args.grad_rtol)

    before_hf = clone_named_params(hf_model, grad_names)
    before_veomni = clone_named_params(veomni_model, grad_names)
    hf_optimizer = torch.optim.AdamW(hf_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)
    veomni_optimizer = torch.optim.AdamW(
        veomni_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False
    )
    hf_optimizer.step()
    veomni_optimizer.step()
    for name in grad_names:
        hf_delta = named_param(hf_model, name).detach() - before_hf[name]
        veomni_delta = named_param(veomni_model, name).detach() - before_veomni[name]
        add_tensor_check(checks, f"optimizer_delta.{name}", hf_delta, veomni_delta, atol=args.param_atol, rtol=args.param_rtol)

    passed = all(item.get("allclose", item.get("equal", False)) for item in checks)
    report = {
        "passed": passed,
        "date": started_at,
        "config_path": args.config_path,
        "device": str(device),
        "seed": args.seed,
        "image_token_id": args.image_token_id,
        "video_token_id": args.video_token_id,
        "hf_model_class": f"{HFMiniMaxM3SparseForConditionalGeneration.__module__}.MiniMaxM3SparseForConditionalGeneration",
        "veomni_model_class": f"{veomni_cls.__module__}.{veomni_cls.__name__}",
        "state_dict_load": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "strict": True,
        },
        "tolerances": {
            "forward": {"atol": args.atol, "rtol": args.rtol},
            "grad": {"atol": args.grad_atol, "rtol": args.grad_rtol},
            "param": {"atol": args.param_atol, "rtol": args.param_rtol},
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "torch_npu_version": getattr(sys.modules.get("torch_npu"), "__version__", None),
            "torch_npu_available": bool(torch.npu.is_available()) if hasattr(torch, "npu") else None,
            "torch_npu_device_count": int(torch.npu.device_count()) if hasattr(torch, "npu") else None,
        },
        "batch_contract": {
            "input_ids": batch["input_ids"].detach().cpu().tolist(),
            "labels": batch["labels"].detach().cpu().tolist(),
            "attention_mask": batch["attention_mask"].detach().cpu().tolist(),
            "position_ids": batch["position_ids"].detach().cpu().tolist(),
            "image_grid_thw": batch["image_grid_thw"].detach().cpu().tolist(),
            "video_grid_thw": batch["video_grid_thw"].detach().cpu().tolist(),
        },
        "checks": jsonable_checks(checks),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "output_json": str(output_path),
                "num_checks": len(checks),
                "failed": [item["name"] for item in checks if not item.get("allclose", item.get("equal", False))],
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
