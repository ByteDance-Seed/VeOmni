#!/usr/bin/env python3
"""Verify MiniMax M3 VL real-checkpoint payload parity gates.

This script complements the toy HF-vs-VeOmni parity gate. It has two modes:

* payload: read local public safetensors payloads or remote tensor byte ranges,
  run the VeOmni MiniMax checkpoint tensor converter on real tensors, and
  compare converted tensor names/shapes/dtypes against the generated model
  state metadata.
* forward: after payload conversion, load the full public checkpoint into both
  the upstream transformers MiniMax model and the VeOmni generated model, then
  compare fixed-prompt logits, top-k ids, and greedy decode ids.

The forward mode is intentionally guarded by --confirm-full-load because the
public MiniMax M3 checkpoint payload is large.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import struct
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


DEFAULT_OUTPUT_JSON = (
    "docs/usage/support_new_models/artifacts/"
    "minimax_m3_vl_precision_parity/real_checkpoint_payload_parity.json"
)
DEFAULT_PROMPT_IDS = "1,1209,318,257,1332"
DEFAULT_INDEX_JSON = "https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/model.safetensors.index.json"
DEFAULT_SHARD_BASE_URL = "https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/main/"
SAFETENSORS_TO_TORCH_DTYPE = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "BOOL": "bool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Local MiniMaxAI/MiniMax-M3 snapshot directory. Required for --mode forward.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Config path for model construction. Defaults to --checkpoint-dir or MiniMaxAI/MiniMax-M3.",
    )
    parser.add_argument(
        "--index-json",
        default=None,
        help=(
            "Path or URL to model.safetensors.index.json. Defaults to <checkpoint-dir>/model.safetensors.index.json "
            "or the Hugging Face MiniMaxAI/MiniMax-M3 index URL when --checkpoint-dir is omitted."
        ),
    )
    parser.add_argument(
        "--shard-base-url",
        default=None,
        help="HTTP(S) base URL or local directory for shard files. Enables remote range payload reads.",
    )
    parser.add_argument("--mode", choices=("payload", "forward"), default="payload")
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        help="Shard filename to read. May be repeated. Defaults to all shards referenced by selected keys.",
    )
    parser.add_argument(
        "--include-key-regex",
        action="append",
        default=[],
        help="Public checkpoint key regex to include. May be repeated. Defaults to every index key.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "npu"))
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--allow-incomplete-groups", action="store_true")
    parser.add_argument("--fail-on-dtype-mismatch", action="store_true")
    parser.add_argument("--no-hash", action="store_true", help="Skip SHA256 tensor fingerprints.")
    parser.add_argument("--max-report-tensors", type=int, default=80)
    parser.add_argument("--confirm-full-load", action="store_true", help="Required for --mode forward.")
    parser.add_argument("--prompt-ids", default=DEFAULT_PROMPT_IDS, help="Comma-separated token ids for forward mode.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--range-timeout", type=float, default=30.0)
    parser.add_argument("--range-retries", type=int, default=5)
    parser.add_argument("--metadata-cache-dir", default=None)
    return parser.parse_args()


def torch_dtype_from_name(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def safetensors_dtype_from_torch(tensor: Any) -> str:
    dtype = str(tensor.dtype).removeprefix("torch.")
    return {
        "bfloat16": "BF16",
        "float16": "F16",
        "float32": "F32",
        "float64": "F64",
        "int8": "I8",
        "int16": "I16",
        "int32": "I32",
        "int64": "I64",
        "uint8": "U8",
        "bool": "BOOL",
    }.get(dtype, dtype)


def load_weight_map(index_json: str) -> dict[str, str]:
    if index_json.startswith(("http://", "https://")):
        request = urllib.request.Request(index_json, headers={"User-Agent": "VeOmni-MiniMaxM3VL-payload-parity"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode())
    else:
        with Path(index_json).open() as f:
            payload = json.load(f)
    return payload["weight_map"]


def read_http_range(url: str, start: int, end: int, *, timeout: float, retries: int) -> bytes:
    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "VeOmni-MiniMaxM3VL-payload-parity",
    }
    for attempt in range(retries):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(end - start + 1)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to read HTTP range from {url}")


def resolve_shard_source(filename: str, *, checkpoint_dir: Path | None, shard_base_url: str | None, index_json: str) -> str:
    if shard_base_url:
        if shard_base_url.startswith(("http://", "https://")):
            return urljoin(shard_base_url.rstrip("/") + "/", filename)
        return str(Path(shard_base_url) / filename)
    if checkpoint_dir:
        return str(checkpoint_dir / filename)
    if index_json.startswith(("http://", "https://")):
        return urljoin(index_json.rsplit("/", 1)[0] + "/", filename)
    return str(Path(index_json).parent / filename)


def read_safetensors_header(
    source: str,
    *,
    timeout: float,
    retries: int,
    cache_dir: Path | None,
) -> tuple[dict[str, Any], int]:
    cache_path = cache_dir / f"{Path(source).name}.header.json" if cache_dir else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return cached["tensors"], int(cached["header_bytes"])

    if source.startswith(("http://", "https://")):
        header_length = struct.unpack("<Q", read_http_range(source, 0, 7, timeout=timeout, retries=retries))[0]
        header = read_http_range(source, 8, 8 + header_length - 1, timeout=timeout, retries=retries)
    else:
        with Path(source).open("rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = handle.read(header_length)
    tensors = json.loads(header.decode())
    tensors.pop("__metadata__", None)
    header_bytes = header_length + 8
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"header_bytes": header_bytes, "tensors": tensors}, sort_keys=True) + "\n")
    return tensors, header_bytes


def torch_dtype_from_safetensors(dtype: str):
    import torch

    return getattr(torch, SAFETENSORS_TO_TORCH_DTYPE[dtype])


def tensor_from_safetensors_payload(payload: bytes, *, dtype: str, shape: list[int]) -> Any:
    import torch

    tensor = torch.frombuffer(bytearray(payload), dtype=torch_dtype_from_safetensors(dtype)).clone()
    return tensor.reshape(tuple(shape))


def read_tensor_from_source(
    source: str,
    public_name: str,
    *,
    metadata: dict[str, Any],
    header_bytes: int,
    timeout: float,
    retries: int,
) -> Any:
    tensor_metadata = metadata[public_name]
    start, end = tensor_metadata["data_offsets"]
    absolute_start = header_bytes + start
    absolute_end = header_bytes + end - 1
    if source.startswith(("http://", "https://")):
        payload = read_http_range(source, absolute_start, absolute_end, timeout=timeout, retries=retries)
    else:
        with Path(source).open("rb") as handle:
            handle.seek(absolute_start)
            payload = handle.read(end - start)
    return tensor_from_safetensors_payload(payload, dtype=tensor_metadata["dtype"], shape=tensor_metadata["shape"])


def select_weight_map(args: argparse.Namespace, weight_map: dict[str, str]) -> dict[str, str]:
    selected = dict(weight_map)
    if args.include_key_regex:
        patterns = [re.compile(pattern) for pattern in args.include_key_regex]
        selected = {name: shard for name, shard in selected.items() if any(pattern.search(name) for pattern in patterns)}
    if args.shard:
        shards = {Path(name).name for name in args.shard}
        selected = {name: shard for name, shard in selected.items() if Path(shard).name in shards}
    return selected


def tensor_fingerprint(tensor: Any, *, hash_values: bool) -> dict[str, Any]:
    import torch

    tensor_cpu = tensor.detach().cpu().contiguous()
    tensor_float = tensor_cpu.float()
    result = {
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype).removeprefix("torch."),
        "numel": int(tensor_cpu.numel()),
        "mean": float(tensor_float.mean().item()) if tensor_cpu.numel() else 0.0,
        "std": float(tensor_float.std(unbiased=False).item()) if tensor_cpu.numel() else 0.0,
        "max_abs": float(tensor_float.abs().max().item()) if tensor_cpu.numel() else 0.0,
    }
    if not hash_values:
        return result

    try:
        if tensor_cpu.dtype == torch.bfloat16:
            payload = tensor_cpu.view(torch.int16).numpy().tobytes()
        else:
            payload = tensor_cpu.numpy().tobytes()
    except (RuntimeError, TypeError):
        payload = tensor_float.numpy().tobytes()
    result["sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def get_device(name: str):
    import torch

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda:0")
    if name == "npu":
        import torch_npu  # noqa: F401

        device = torch.device("npu:0")
        torch.npu.set_device(device)
        return device
    return torch.device("cpu")


def build_model_metadata(config_path: str, torch_dtype: str) -> tuple[dict[str, dict[str, Any]], int]:
    from verify_minimax_m3_vl_checkpoint_index import build_minimax_state_metadata

    _, _, model_metadata, num_experts = build_minimax_state_metadata(config_path, torch_dtype)
    return model_metadata, num_experts


def load_converted_payload(
    *,
    checkpoint_dir: Path | None,
    index_json: str,
    shard_base_url: str | None,
    selected_weight_map: dict[str, str],
    num_experts: int,
    hash_values: bool,
    range_timeout: float,
    range_retries: int,
    metadata_cache_dir: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    from veomni.models.transformers.minimax_m3_vl.checkpoint_tensor_converter import (
        MiniMaxM3VLCheckpointTensorConverter,
    )

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, filename in selected_weight_map.items():
        by_shard[filename].append(name)

    converter = MiniMaxM3VLCheckpointTensorConverter(num_experts=num_experts)
    converted_tensors: dict[str, Any] = {}
    tensor_reports = []
    duplicate_converted_keys = []
    public_keys_read = 0
    payload_bytes_read = 0
    shard_sources: dict[str, str] = {}
    metadata_cache_path = Path(metadata_cache_dir) if metadata_cache_dir else None

    for filename in sorted(by_shard):
        source = resolve_shard_source(filename, checkpoint_dir=checkpoint_dir, shard_base_url=shard_base_url, index_json=index_json)
        shard_sources[filename] = source
        if source.startswith(("http://", "https://")):
            metadata, header_bytes = read_safetensors_header(
                source,
                timeout=range_timeout,
                retries=range_retries,
                cache_dir=metadata_cache_path,
            )
            available = set(metadata)
            for public_name in sorted(by_shard[filename]):
                if public_name not in available:
                    raise KeyError(f"{public_name} not found in {source}")
                public_keys_read += 1
                tensor = read_tensor_from_source(
                    source,
                    public_name,
                    metadata=metadata,
                    header_bytes=header_bytes,
                    timeout=range_timeout,
                    retries=range_retries,
                )
                start, end = metadata[public_name]["data_offsets"]
                payload_bytes_read += end - start
                result = converter.convert(public_name, tensor)
                if result is None:
                    continue
                if result.name in converted_tensors:
                    duplicate_converted_keys.append(result.name)
                converted_tensors[result.name] = result.tensor
                tensor_reports.append(
                    {
                        "public_name": public_name,
                        "converted_name": result.name,
                        "source_shard": filename,
                        "source": source,
                        "fingerprint": tensor_fingerprint(result.tensor, hash_values=hash_values),
                    }
                )
        else:
            shard_path = Path(source)
            if not shard_path.exists():
                raise FileNotFoundError(f"missing safetensors shard: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for public_name in sorted(by_shard[filename]):
                    if public_name not in available:
                        raise KeyError(f"{public_name} not found in {shard_path}")
                    public_keys_read += 1
                    tensor = handle.get_tensor(public_name)
                    payload_bytes_read += tensor.numel() * tensor.element_size()
                    result = converter.convert(public_name, tensor)
                    if result is None:
                        continue
                    if result.name in converted_tensors:
                        duplicate_converted_keys.append(result.name)
                    converted_tensors[result.name] = result.tensor
                    tensor_reports.append(
                        {
                            "public_name": public_name,
                            "converted_name": result.name,
                            "source_shard": filename,
                            "source": str(shard_path),
                            "fingerprint": tensor_fingerprint(result.tensor, hash_values=hash_values),
                        }
                    )

    finalize_error = None
    try:
        for result in converter.finalize():
            if result.name in converted_tensors:
                duplicate_converted_keys.append(result.name)
            converted_tensors[result.name] = result.tensor
            tensor_reports.append(
                {
                    "public_name": "<converter.finalize>",
                    "converted_name": result.name,
                    "source_shard": None,
                    "fingerprint": tensor_fingerprint(result.tensor, hash_values=hash_values),
                }
            )
    except RuntimeError as exc:
        finalize_error = str(exc)

    summary = {
        "selected_public_keys": len(selected_weight_map),
        "public_keys_read": public_keys_read,
        "payload_bytes_read": payload_bytes_read,
        "converted_tensor_keys": len(converted_tensors),
        "shard_sources": shard_sources,
        "duplicate_converted_keys": sorted(set(duplicate_converted_keys)),
        "converter_finalize_error": finalize_error,
        "tensor_reports": tensor_reports,
    }
    return converted_tensors, summary


def compare_payload_to_model_metadata(
    converted_tensors: dict[str, Any],
    model_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing_model_keys = []
    shape_mismatches = []
    dtype_mismatches = []
    for name, tensor in sorted(converted_tensors.items()):
        expected = model_metadata.get(name)
        actual = {
            "shape": list(tensor.shape),
            "dtype": safetensors_dtype_from_torch(tensor),
        }
        if expected is None:
            missing_model_keys.append(name)
            continue
        if actual["shape"] != expected["shape"]:
            shape_mismatches.append({"name": name, "checkpoint": actual["shape"], "model": expected["shape"]})
        if actual["dtype"] != expected["dtype"]:
            dtype_mismatches.append({"name": name, "checkpoint": actual["dtype"], "model": expected["dtype"]})

    dtype_mismatch_groups: dict[str, int] = defaultdict(int)
    for item in dtype_mismatches:
        dtype_mismatch_groups[f"{item['checkpoint']}->{item['model']}"] += 1

    return {
        "missing_model_key_count": len(missing_model_keys),
        "shape_mismatch_count": len(shape_mismatches),
        "dtype_mismatch_count": len(dtype_mismatches),
        "dtype_mismatch_groups": dict(sorted(dtype_mismatch_groups.items())),
        "missing_model_keys_sample": missing_model_keys[:20],
        "shape_mismatches_sample": shape_mismatches[:20],
        "dtype_mismatches_sample": dtype_mismatches[:20],
    }


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


def parse_prompt_ids(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--prompt-ids must contain at least one token id")
    return values


def build_text_batch(prompt_ids: list[int], device: Any) -> dict[str, Any]:
    import torch

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def greedy_decode_ids(model: Any, prompt_ids: list[int], *, device: Any, max_new_tokens: int) -> list[int]:
    import torch

    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        batch = build_text_batch(generated[0].detach().cpu().tolist(), device)
        with torch.no_grad():
            outputs = model(**batch)
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=-1)
    return generated[0, len(prompt_ids) :].detach().cpu().tolist()


def release_accelerator_memory(device: Any) -> None:
    import torch

    gc.collect()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    elif getattr(device, "type", None) == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def patch_eager_config(config: Any) -> Any:
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(config, "text_config"):
        if hasattr(config.text_config, "_attn_implementation"):
            config.text_config._attn_implementation = "eager"
        if hasattr(config.text_config, "use_cache"):
            config.text_config.use_cache = False
    return config


def run_forward_parity(
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    config_path: str,
    index_json: str,
    shard_base_url: str | None,
    selected_weight_map: dict[str, str],
    model_metadata: dict[str, dict[str, Any]],
    num_experts: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3SparseForConditionalGeneration as HFMiniMaxM3SparseForConditionalGeneration,
    )

    from veomni.models.loader import get_model_class, get_model_config

    if not args.confirm_full_load:
        raise RuntimeError("--mode forward requires --confirm-full-load because the public payload is large")

    dtype = torch_dtype_from_name(args.torch_dtype)
    device = get_device(args.device)
    prompt_ids = parse_prompt_ids(args.prompt_ids)

    hf_config = patch_eager_config(AutoConfig.from_pretrained(config_path))
    hf_model = HFMiniMaxM3SparseForConditionalGeneration.from_pretrained(
        str(checkpoint_dir),
        config=hf_config,
        torch_dtype=dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    batch = build_text_batch(prompt_ids, device)
    with torch.no_grad():
        hf_outputs = hf_model(**batch)
    hf_logits = hf_outputs.logits.detach().cpu()
    hf_topk = torch.topk(hf_outputs.logits[:, -1, :], k=args.top_k, dim=-1).indices.detach().cpu()
    hf_greedy = greedy_decode_ids(hf_model, prompt_ids, device=device, max_new_tokens=args.max_new_tokens)
    del hf_outputs
    del hf_model
    release_accelerator_memory(device)

    converted_tensors, payload_summary = load_converted_payload(
        checkpoint_dir=checkpoint_dir,
        index_json=index_json,
        shard_base_url=shard_base_url,
        selected_weight_map=selected_weight_map,
        num_experts=num_experts,
        hash_values=not args.no_hash,
        range_timeout=args.range_timeout,
        range_retries=args.range_retries,
        metadata_cache_dir=args.metadata_cache_dir,
    )
    metadata_comparison = compare_payload_to_model_metadata(converted_tensors, model_metadata)

    veomni_config = patch_eager_config(get_model_config(config_path))
    veomni_cls = get_model_class(veomni_config)
    veomni_model = veomni_cls(veomni_config)
    load_result = veomni_model.load_state_dict(converted_tensors, strict=True)
    del converted_tensors
    release_accelerator_memory(device)

    veomni_model.to(device)
    veomni_model.eval()
    batch = build_text_batch(prompt_ids, device)
    with torch.no_grad():
        veomni_outputs = veomni_model(**batch)

    checks: list[dict[str, Any]] = []
    logits_check = {"name": "forward.logits", "kind": "tensor"}
    logits_check.update(tensor_stats(hf_logits, veomni_outputs.logits, atol=args.atol, rtol=args.rtol))
    checks.append(logits_check)

    veomni_topk = torch.topk(veomni_outputs.logits[:, -1, :], k=args.top_k, dim=-1).indices.detach().cpu()
    checks.append(
        {
            "name": "forward.last_token_topk_ids",
            "kind": "exact",
            "hf": hf_topk.tolist(),
            "veomni": veomni_topk.tolist(),
            "equal": bool(torch.equal(hf_topk, veomni_topk)),
        }
    )

    veomni_greedy = greedy_decode_ids(veomni_model, prompt_ids, device=device, max_new_tokens=args.max_new_tokens)
    checks.append(
        {
            "name": "generate.greedy_ids",
            "kind": "exact",
            "hf": hf_greedy,
            "veomni": veomni_greedy,
            "equal": hf_greedy == veomni_greedy,
        }
    )

    forward_report = {
        "device": str(device),
        "prompt_ids": prompt_ids,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "hf_model_class": f"{HFMiniMaxM3SparseForConditionalGeneration.__module__}.MiniMaxM3SparseForConditionalGeneration",
        "veomni_model_class": f"{veomni_cls.__module__}.{veomni_cls.__name__}",
        "state_dict_load": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "strict": True,
        },
        "checks": checks,
        "passed": all(item.get("allclose", item.get("equal", False)) for item in checks),
    }
    del veomni_outputs
    del veomni_model
    release_accelerator_memory(device)
    return forward_report, payload_summary, metadata_comparison


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MODELING_BACKEND", "veomni")

    import torch
    import transformers

    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if not is_transformers_version_greater_or_equal_to("5.12.0"):
        raise RuntimeError("MiniMax M3 VL payload parity requires transformers>=5.12.0.")

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if args.mode == "forward" and checkpoint_dir is None:
        raise RuntimeError("--mode forward requires --checkpoint-dir with a complete local checkpoint snapshot")
    config_path = args.config_path or (str(checkpoint_dir) if checkpoint_dir else "MiniMaxAI/MiniMax-M3")
    if args.index_json:
        index_json = args.index_json
    elif checkpoint_dir:
        index_json = str(checkpoint_dir / "model.safetensors.index.json")
    else:
        index_json = DEFAULT_INDEX_JSON
    shard_base_url = args.shard_base_url
    if checkpoint_dir is None and shard_base_url is None:
        shard_base_url = DEFAULT_SHARD_BASE_URL
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    weight_map = load_weight_map(index_json)
    selected_weight_map = select_weight_map(args, weight_map)
    if not selected_weight_map:
        raise RuntimeError("no checkpoint tensors selected; check --shard and --include-key-regex")

    model_metadata, num_experts = build_model_metadata(config_path, args.torch_dtype)
    if args.mode == "forward":
        if args.include_key_regex or args.shard:
            raise RuntimeError("--mode forward requires the complete local checkpoint; do not use --include-key-regex or --shard")
        forward_report, payload_summary, metadata_comparison = run_forward_parity(
            args=args,
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            index_json=index_json,
            shard_base_url=shard_base_url,
            selected_weight_map=selected_weight_map,
            model_metadata=model_metadata,
            num_experts=num_experts,
        )
    else:
        converted_tensors, payload_summary = load_converted_payload(
            checkpoint_dir=checkpoint_dir,
            index_json=index_json,
            shard_base_url=shard_base_url,
            selected_weight_map=selected_weight_map,
            num_experts=num_experts,
            hash_values=not args.no_hash,
            range_timeout=args.range_timeout,
            range_retries=args.range_retries,
            metadata_cache_dir=args.metadata_cache_dir,
        )
        metadata_comparison = compare_payload_to_model_metadata(converted_tensors, model_metadata)
        forward_report = None

    payload_passed = (
        payload_summary["converted_tensor_keys"] > 0
        and not payload_summary["duplicate_converted_keys"]
        and not metadata_comparison["missing_model_key_count"]
        and not metadata_comparison["shape_mismatch_count"]
        and (args.allow_incomplete_groups or payload_summary["converter_finalize_error"] is None)
        and (not args.fail_on_dtype_mismatch or not metadata_comparison["dtype_mismatch_count"])
    )

    report = {
        "passed": bool(payload_passed and (forward_report is None or forward_report["passed"])),
        "date": started_at,
        "mode": args.mode,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "config_path": config_path,
        "index_json": index_json,
        "shard_base_url": shard_base_url,
        "torch_dtype": args.torch_dtype,
        "selected_shards": sorted(set(selected_weight_map.values())),
        "selected_shard_count": len(set(selected_weight_map.values())),
        "include_key_regex": args.include_key_regex,
        "allow_incomplete_groups": args.allow_incomplete_groups,
        "fail_on_dtype_mismatch": args.fail_on_dtype_mismatch,
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "payload": {
            key: value
            for key, value in payload_summary.items()
            if key != "tensor_reports"
        },
        "metadata_comparison": metadata_comparison,
        "tensor_reports": payload_summary["tensor_reports"][: args.max_report_tensors],
        "tensor_report_truncated": len(payload_summary["tensor_reports"]) > args.max_report_tensors,
        "forward": forward_report,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "output_json": str(output_path),
                "mode": args.mode,
                "selected_public_keys": len(selected_weight_map),
                "converted_tensor_keys": payload_summary["converted_tensor_keys"],
                "selected_shard_count": len(set(selected_weight_map.values())),
                "converter_finalize_error": payload_summary["converter_finalize_error"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
