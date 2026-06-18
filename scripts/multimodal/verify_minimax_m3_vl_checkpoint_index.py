# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify MiniMax M3 VL public checkpoint coverage.

The default verifier is intentionally index-only: it reads
`model.safetensors.index.json` and builds the VeOmni MiniMax model on `meta` to
compare checkpoint state names after the runtime checkpoint converter is
applied. Non-persistent runtime buffers, such as rotary caches, are
intentionally excluded.

With `--verify-shard-metadata`, it additionally reads only the safetensors
headers for each shard via HTTP Range requests or local files. That proves
converted key, shape, and dtype coverage without downloading tensor payloads.
"""

import argparse
import json
import os
import struct
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urljoin


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
TORCH_TO_SAFETENSORS_DTYPE = {value: key for key, value in SAFETENSORS_TO_TORCH_DTYPE.items()}


def make_eager_ops_config():
    from veomni.arguments.arguments_types import OpsImplementationConfig

    return OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
    )


def load_weight_map(index_json: str) -> Dict[str, str]:
    if index_json.startswith(("http://", "https://")):
        request = urllib.request.Request(index_json, headers={"User-Agent": "VeOmni-MiniMaxM3VL-index-verifier"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode())
                break
            except (OSError, TimeoutError, json.JSONDecodeError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    else:
        with open(index_json) as f:
            payload = json.load(f)
    return payload["weight_map"]


def read_http_range(url: str, start: int, end: int, *, timeout: float, retries: int) -> bytes:
    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "VeOmni-MiniMaxM3VL-safetensors-header-verifier",
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


def read_safetensors_header(source: str, *, timeout: float, retries: int) -> tuple[Dict[str, Any], int]:
    if source.startswith(("http://", "https://")):
        header_length = struct.unpack("<Q", read_http_range(source, 0, 7, timeout=timeout, retries=retries))[0]
        header = read_http_range(source, 8, 8 + header_length - 1, timeout=timeout, retries=retries)
    else:
        with open(source, "rb") as f:
            header_length = struct.unpack("<Q", f.read(8))[0]
            header = f.read(header_length)
    payload = json.loads(header.decode())
    payload.pop("__metadata__", None)
    return payload, header_length + 8


def resolve_shard_source(filename: str, shard_base_url: str | None, index_json: str) -> str:
    if shard_base_url:
        if shard_base_url.startswith(("http://", "https://")):
            return urljoin(shard_base_url.rstrip("/") + "/", filename)
        return str(Path(shard_base_url) / filename)
    if index_json.startswith(("http://", "https://")):
        return urljoin(index_json.rsplit("/", 1)[0] + "/", filename)
    return str(Path(index_json).parent / filename)


def load_safetensors_metadata(
    weight_map: Dict[str, str],
    *,
    index_json: str,
    shard_base_url: str | None,
    timeout: float,
    retries: int,
    progress: bool,
    cache_dir: str | None,
) -> tuple[Dict[str, Dict[str, Any]], int, int]:
    tensors: Dict[str, Dict[str, Any]] = {}
    header_bytes = 0
    shard_files = sorted(set(weight_map.values()))
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
    for idx, filename in enumerate(shard_files, start=1):
        if progress:
            print(f"[safetensors-header] {idx}/{len(shard_files)} {filename}", file=sys.stderr, flush=True)
        shard_cache = cache_path / f"{filename}.header.json" if cache_path else None
        if shard_cache and shard_cache.exists():
            cached = json.loads(shard_cache.read_text())
            shard_tensors = cached["tensors"]
            shard_header_bytes = cached["header_bytes"]
        else:
            source = resolve_shard_source(filename, shard_base_url, index_json)
            shard_tensors, shard_header_bytes = read_safetensors_header(source, timeout=timeout, retries=retries)
            if shard_cache:
                shard_cache.write_text(
                    json.dumps({"header_bytes": shard_header_bytes, "tensors": shard_tensors}, sort_keys=True) + "\n"
                )
        header_bytes += shard_header_bytes
        for name, metadata in shard_tensors.items():
            metadata = dict(metadata)
            metadata["filename"] = filename
            tensors[name] = metadata
    return tensors, len(shard_files), header_bytes


def shard_index(filename: str) -> int:
    from veomni.models.checkpoint_tensor_loading import shard_index_from_filename

    try:
        return shard_index_from_filename(Path(filename).name)
    except (IndexError, ValueError):
        return 0


def public_weight_map_to_index_mapping(weight_map: Dict[str, str]) -> Dict[str, int]:
    return {fqn: shard_index(filename) for fqn, filename in weight_map.items()}


def build_minimax_state_metadata(
    config_path: str, torch_dtype: str
) -> tuple[set[str], set[str], Dict[str, Dict[str, Any]], int]:
    from veomni.models import build_foundation_model

    os.environ["MODELING_BACKEND"] = "veomni"
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype=torch_dtype,
        init_device="meta",
        ops_implementation=make_eager_ops_config(),
    )
    parameter_names = {name for name, _ in model.named_parameters()}
    state_dict = model.state_dict()
    state_names = set(state_dict)
    persistent_buffer_names = state_names - parameter_names
    state_metadata = {
        name: {
            "shape": list(tensor.shape),
            "dtype": TORCH_TO_SAFETENSORS_DTYPE.get(str(tensor.dtype).removeprefix("torch."), str(tensor.dtype)),
        }
        for name, tensor in state_dict.items()
    }
    text_config = getattr(model.config, "text_config", model.config)
    return parameter_names, persistent_buffer_names, state_metadata, text_config.num_local_experts


def torch_dtype_from_safetensors(dtype: str):
    import torch

    torch_name = SAFETENSORS_TO_TORCH_DTYPE[dtype]
    return getattr(torch, torch_name)


def convert_public_safetensors_metadata(
    public_metadata: Dict[str, Dict[str, Any]], *, num_experts: int
) -> Dict[str, Dict[str, Any]]:
    import torch

    from veomni.models.transformers.minimax_m3_vl.checkpoint_tensor_converter import (
        MiniMaxM3VLCheckpointTensorConverter,
    )

    converter = MiniMaxM3VLCheckpointTensorConverter(num_experts=num_experts)
    converted: Dict[str, Dict[str, Any]] = {}
    for name in sorted(public_metadata):
        metadata = public_metadata[name]
        tensor = torch.empty(
            tuple(metadata["shape"]),
            dtype=torch_dtype_from_safetensors(metadata["dtype"]),
            device="meta",
        )
        result = converter.convert(name, tensor)
        if result is None:
            continue
        converted[result.name] = {
            "shape": list(result.tensor.shape),
            "dtype": TORCH_TO_SAFETENSORS_DTYPE.get(
                str(result.tensor.dtype).removeprefix("torch."), str(result.tensor.dtype)
            ),
        }
    for result in converter.finalize():
        converted[result.name] = {
            "shape": list(result.tensor.shape),
            "dtype": TORCH_TO_SAFETENSORS_DTYPE.get(
                str(result.tensor.dtype).removeprefix("torch."), str(result.tensor.dtype)
            ),
        }
    return converted


def compare_metadata(
    converted_metadata: Dict[str, Dict[str, Any]],
    model_metadata: Dict[str, Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    shape_mismatches = []
    dtype_mismatches = []
    for name in sorted(set(converted_metadata) & set(model_metadata)):
        converted = converted_metadata[name]
        model = model_metadata[name]
        if converted["shape"] != model["shape"]:
            shape_mismatches.append({"name": name, "checkpoint": converted["shape"], "model": model["shape"]})
        if converted["dtype"] != model["dtype"]:
            dtype_mismatches.append({"name": name, "checkpoint": converted["dtype"], "model": model["dtype"]})
    return shape_mismatches, dtype_mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="MiniMaxAI/MiniMax-M3")
    parser.add_argument("--index-json", default=DEFAULT_INDEX_JSON)
    parser.add_argument("--shard-base-url", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--fail-on-unexpected", action="store_true")
    parser.add_argument("--verify-shard-metadata", action="store_true")
    parser.add_argument("--fail-on-dtype-mismatch", action="store_true")
    parser.add_argument("--range-timeout", type=float, default=20.0)
    parser.add_argument("--range-retries", type=int, default=5)
    parser.add_argument("--metadata-cache-dir", default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    from veomni.models.transformers.minimax_m3_vl.checkpoint_tensor_converter import (
        convert_minimax_m3_vl_fqn_to_index_mapping,
    )
    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if not is_transformers_version_greater_or_equal_to("5.12.0"):
        raise RuntimeError("MiniMax M3 VL index verification requires transformers>=5.12.0.")

    weight_map = load_weight_map(args.index_json)
    public_mapping = public_weight_map_to_index_mapping(weight_map)
    converted_mapping = convert_minimax_m3_vl_fqn_to_index_mapping(public_mapping)
    converted_keys = set(converted_mapping)
    parameter_names, persistent_buffer_names, model_metadata, num_experts = build_minimax_state_metadata(
        args.config_path, args.torch_dtype
    )
    state_names = set(model_metadata)

    missing_state_keys = sorted(state_names - converted_keys)
    unexpected_index_keys = sorted(converted_keys - state_names)
    projector_keys = sorted(key for key in state_names if "multi_modal_projector" in key)
    missing_projector_keys = sorted(key for key in projector_keys if key not in converted_keys)

    result = {
        "config_path": args.config_path,
        "index_json": args.index_json,
        "public_weight_map_keys": len(weight_map),
        "converted_index_keys": len(converted_keys),
        "model_parameter_keys": len(parameter_names),
        "model_persistent_buffer_keys": len(persistent_buffer_names),
        "model_state_keys": len(state_names),
        "missing_state_key_count": len(missing_state_keys),
        "unexpected_index_key_count": len(unexpected_index_keys),
        "missing_projector_keys": missing_projector_keys,
        "missing_state_keys_sample": missing_state_keys[:20],
        "unexpected_index_keys_sample": unexpected_index_keys[:20],
        "shard_metadata_verified": False,
        "full_checkpoint_load_executed": False,
    }

    if args.verify_shard_metadata:
        public_metadata, shard_count, header_bytes = load_safetensors_metadata(
            weight_map,
            index_json=args.index_json,
            shard_base_url=args.shard_base_url or DEFAULT_SHARD_BASE_URL,
            timeout=args.range_timeout,
            retries=args.range_retries,
            progress=args.progress,
            cache_dir=args.metadata_cache_dir,
        )
        converted_metadata = convert_public_safetensors_metadata(public_metadata, num_experts=num_experts)
        shape_mismatches, dtype_mismatches = compare_metadata(converted_metadata, model_metadata)
        missing_metadata_keys = sorted(set(model_metadata) - set(converted_metadata))
        unexpected_metadata_keys = sorted(set(converted_metadata) - set(model_metadata))
        dtype_mismatch_groups: Dict[str, int] = defaultdict(int)
        for item in dtype_mismatches:
            dtype_mismatch_groups[f"{item['checkpoint']}->{item['model']}"] += 1
        result.update(
            {
                "shard_metadata_verified": True,
                "safetensors_shards_read": shard_count,
                "safetensors_header_bytes_read": header_bytes,
                "public_safetensors_metadata_keys": len(public_metadata),
                "converted_metadata_keys": len(converted_metadata),
                "missing_metadata_key_count": len(missing_metadata_keys),
                "unexpected_metadata_key_count": len(unexpected_metadata_keys),
                "shape_mismatch_count": len(shape_mismatches),
                "dtype_mismatch_count": len(dtype_mismatches),
                "dtype_mismatch_groups": dict(sorted(dtype_mismatch_groups.items())),
                "dtype_mismatch_failure_enabled": args.fail_on_dtype_mismatch,
                "dtype_mismatch_note": (
                    "Dtype metadata differences are reported but do not fail by default; "
                    "VeOmni checkpoint dispatch casts tensors to the target parameter or buffer dtype during load."
                ),
                "missing_metadata_keys_sample": missing_metadata_keys[:20],
                "unexpected_metadata_keys_sample": unexpected_metadata_keys[:20],
                "shape_mismatches_sample": shape_mismatches[:20],
                "dtype_mismatches_sample": dtype_mismatches[:20],
                "checkpoint_values_downloaded": False,
            }
        )

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        Path(args.json_output).write_text(text + "\n")

    if missing_state_keys:
        return 1
    if args.fail_on_unexpected and unexpected_index_keys:
        return 1
    if args.verify_shard_metadata:
        if result["missing_metadata_key_count"] or result["unexpected_metadata_key_count"]:
            return 1
        if result["shape_mismatch_count"]:
            return 1
        if args.fail_on_dtype_mismatch and result["dtype_mismatch_count"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
