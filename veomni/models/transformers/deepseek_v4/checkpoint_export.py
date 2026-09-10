# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Export a VeOmni DCP to the official V4-Flash (FP4) or V4-Flash-Base (FP8) layout.

Release schemas are bundled; no target weights are needed. Tokenizer assets are
resolved from pinned Hugging Face releases (or the local HF cache). MTP is omitted
because VeOmni does not train it. See docs/usage/deepseek_v4_merge.md.
"""

import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from jinja2 import TemplateSyntaxError
from transformers.utils.chat_template_utils import _compile_jinja_template

from veomni.checkpoint.conversion import CachedFileSystemReader, export_shards, write_index
from veomni.checkpoint.dcp_checkpointer import _normalize_key
from veomni.models.checkpoint_tensor_loading import _process_moe_params
from veomni.models.transformers.deepseek_v4.checkpoint_tensor_converter import (
    DeepseekV4CheckpointTensorConverter,
)
from veomni.utils import helper
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device


logger = helper.create_logger(__name__)

SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

INDEX_NAME = "model.safetensors.index.json"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
CHAT_TEMPLATE_NAME = "chat_template.jinja"

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")


class OutputFormat:
    """Expand a bundled, header-derived release schema without reading model weights."""

    def __init__(self, name: str):
        path = os.path.join(os.path.dirname(__file__), "formats", name + ".json")
        with open(path, encoding="utf-8") as fh:
            preset = json.load(fh)
        self.name = name
        self.repo_id = preset["repo_id"]
        self.revision = preset["revision"]
        self.config = preset["config"]
        self.schema = {}
        self.weight_map = {}
        self.keys_by_shard = defaultdict(list)
        self.shard_metadata = {}
        for shard in preset["shards"]:
            filename = shard["filename"]
            self.shard_metadata[filename] = shard["metadata"]
            for suffix, (dtype, shape) in preset["templates"][shard["template"]].items():
                ids = range(self.num_experts()) if "{expert}" in suffix else [0]
                for expert in ids:
                    key = shard["prefix"] + suffix.format(expert=expert)
                    self.schema[key] = (SAFETENSORS_DTYPES[dtype], tuple(shape))
                    self.weight_map[key] = filename
                    self.keys_by_shard[filename].append(key)

    def shard_of(self, key: str) -> str:
        return self.weight_map[key]

    def tensor_nbytes(self, key: str) -> int:
        dtype, shape = self.schema[key]
        return math.prod(shape) * dtype.itemsize

    def expert_dtype(self) -> str:
        return self.config["expert_dtype"]

    def num_experts(self) -> int:
        return self.config["n_routed_experts"]


def representative_native_key(converter: DeepseekV4CheckpointTensorConverter, hf_key: str) -> str:
    """Native key standing in for a source tensor, resolving fused experts to expert 0.

    Fused expert tensors have no single native counterpart, so grouping and shard lookup
    use the first expert's key; every expert of a layer lands in the same shard.
    """
    if hf_key.endswith("mlp.experts.gate_up_proj"):
        hf_key = f"{hf_key[: -len('gate_up_proj')]}0.gate_proj.weight"
    elif hf_key.endswith("mlp.experts.down_proj"):
        hf_key = f"{hf_key[: -len('down_proj')]}0.down_proj.weight"
    return converter.export_name(hf_key)


def expert_dim_shard_count(metadata, dcp_key: str) -> int:
    """Number of equal pieces the expert dim (dim 0) is split into in the checkpoint.

    Rejects tensors that are sharded along any other dim, which is what the default FSDP
    ``Shard(1)`` expert layout looks like. Such a checkpoint is not block-transposed, so
    un-permuting it would corrupt the output.
    """
    tensor_meta = metadata.state_dict_metadata[dcp_key]
    size = tuple(tensor_meta.size)
    piece_sizes = set()
    for chunk in tensor_meta.chunks:
        chunk_size = tuple(chunk.sizes)
        if chunk_size[1:] != size[1:]:
            raise ValueError(
                f"{dcp_key}: DCP chunk {chunk_size} of {size} is sharded beyond the expert "
                "dim, so this checkpoint does not use the whole-expert Shard(0) layout that "
                "--ep-size corrects"
            )
        piece_sizes.add(chunk_size[0])
    if len(piece_sizes) != 1:
        raise ValueError(f"{dcp_key}: expert dim is split into uneven pieces {sorted(piece_sizes)}")
    per_shard = piece_sizes.pop()
    if per_shard * len(tensor_meta.chunks) != size[0]:
        raise ValueError(
            f"{dcp_key}: {len(tensor_meta.chunks)} chunks of {per_shard} experts do not tile "
            f"the {size[0]} experts exactly"
        )
    return len(tensor_meta.chunks)


def resolve_expert_fsdp_size(metadata, per_layer: Dict[int, Dict[str, str]], ep_size: int) -> int:
    """Derive the ``ep_fsdp`` mesh size from how DCP sharded the expert dim.

    The FSDP factor is read off the checkpoint rather than taken on the command line so
    that a checkpoint saved with the default ``Shard(1)`` expert layout is rejected by
    ``expert_dim_shard_count`` instead of being silently un-permuted.
    """
    counts = {
        expert_dim_shard_count(metadata, dcp_key)
        for keys in per_layer.values()
        for hf_key, dcp_key in keys.items()
        if "mlp.experts." in hf_key
    }
    if not counts:
        raise ValueError("--ep-size was passed but the checkpoint holds no fused expert tensors")
    if len(counts) != 1:
        raise ValueError(f"expert-dim shard count differs across layers: {sorted(counts)}")
    total = counts.pop()
    if total % ep_size != 0:
        raise ValueError(f"expert dim is split into {total} chunks, not a multiple of --ep-size {ep_size}")
    return total // ep_size


def unpermute_expert_dim(tensor: torch.Tensor, ep_size: int, ep_fsdp_size: int) -> torch.Tensor:
    """Reorder a fused expert tensor from DCP's mesh order into expert-id order.

    Under ``muon_expert_zero_comm`` the EP and FSDP mesh dims both shard dim 0, and the
    ``(ep_fsdp, ep)`` mesh order puts ``ep_fsdp`` outermost, so the reconstructed global
    tensor runs ``ep_fsdp`` before ``ep`` while expert ids run ``ep`` before ``ep_fsdp``.
    Swapping the two factors restores expert order.
    """
    num_experts, *rest = tensor.shape
    per_shard = num_experts // (ep_size * ep_fsdp_size)
    return tensor.reshape(ep_fsdp_size, ep_size, per_shard, *rest).transpose(0, 1).reshape(num_experts, *rest)


def split_source_tensor(
    hf_key: str, tensor: torch.Tensor, ep_size: int = 1, ep_fsdp_size: int = 1
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Yield the per-expert tensors of a fused MoE parameter, or the tensor unchanged."""
    if "mlp.experts." in hf_key:
        if ep_size > 1 and ep_fsdp_size > 1:
            tensor = unpermute_expert_dim(tensor, ep_size, ep_fsdp_size)
        yield from _process_moe_params(hf_key, tensor, ep_rank=0)
    else:
        yield hf_key, tensor


def group_source_keys(
    metadata, converter: DeepseekV4CheckpointTensorConverter
) -> Tuple[Dict[int, Dict[str, str]], Dict[str, str]]:
    """Split DCP model keys into per-layer groups plus the non-layer remainder.

    Returns ``({layer_index: {hf_key: dcp_key}}, {hf_key: dcp_key})``.
    """
    per_layer: Dict[int, Dict[str, str]] = defaultdict(dict)
    globals_: Dict[str, str] = {}
    for dcp_key in metadata.state_dict_metadata:
        hf_key = _normalize_key(dcp_key)
        if hf_key is None or hf_key.removeprefix("model.").startswith("mtp."):
            continue
        native = representative_native_key(converter, hf_key)
        match = _LAYER_RE.match(native)
        if match:
            per_layer[int(match.group(1))][hf_key] = dcp_key
        else:
            globals_[hf_key] = dcp_key
    return per_layer, globals_


def validate_source_group(metadata, keys, converter, target, shard):
    """Reject missing or incompatible source tensors using metadata only."""
    produced = set()
    for hf_key, dcp_key in keys.items():
        meta = metadata.state_dict_metadata[dcp_key]
        tensor = torch.empty(meta.size, dtype=meta.properties.dtype, device="meta")
        if "mlp.experts." in hf_key:
            if tensor.ndim != 3 or tensor.shape[0] != target.num_experts():
                raise ValueError(
                    f"{dcp_key}: expected {target.num_experts()} fused experts, got {tuple(tensor.shape)}"
                )
            if hf_key.endswith("gate_up_proj") and tensor.shape[1] % 2:
                raise ValueError(f"{dcp_key}: gate/up dimension must be even")
        for name, part in split_source_tensor(hf_key, tensor):
            native = converter.export_name(name)
            dtype, shape = target.schema[native]
            source_shape = list(shape)
            if dtype == torch.int8 and ".ffn.experts." in native:
                source_shape[-1] *= 2
            if tuple(part.shape) != tuple(source_shape):
                raise ValueError(f"{dcp_key}: source shape {tuple(part.shape)}, expected {tuple(source_shape)}")
            if dtype in (torch.int64, torch.int32, torch.bool) and part.dtype != dtype:
                raise ValueError(f"{dcp_key}: expected integer/bool dtype {dtype}, got {part.dtype}")
            produced.add(native)
            if native.endswith(".weight"):
                scale = native.removesuffix(".weight") + ".scale"
                if scale in target.schema:
                    produced.add(scale)
    expected = set(target.keys_by_shard[shard])
    if produced != expected:
        raise ValueError(
            f"{shard}: incomplete source (missing {sorted(expected - produced)[:5]}, "
            f"extra {sorted(produced - expected)[:5]})"
        )


def convert_group(
    keys: Dict[str, str],
    source: Dict[str, torch.Tensor],
    converter: DeepseekV4CheckpointTensorConverter,
    target: OutputFormat,
    expert_dtype: str,
    device: torch.device,
    ep_size: int = 1,
    ep_fsdp_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """Rename and quantize one group of source tensors into their target representation.

    ``export_tensor`` decides naming and quantization exactly as the live-model export
    does; the only extra step here is storing unquantized tensors in the dtype the
    target uses, since DCP master weights are fp32. Consumes ``source`` as it goes so
    the fp32 originals are freed before the whole converted group is held in memory.
    """
    out: Dict[str, torch.Tensor] = {}
    for hf_key, dcp_key in keys.items():
        for split_key, split_tensor in split_source_tensor(hf_key, source.pop(dcp_key), ep_size, ep_fsdp_size):
            exported = converter.export_tensor(
                split_key,
                split_tensor.to(device, non_blocking=True),
                target.weight_map,
                expert_dtype,
            )
            for native, tensor in exported:
                if native not in target.schema:
                    raise KeyError(f"converted key not present in target: {native} (from {split_key})")
                out[native] = cast_to_target_dtype(native, tensor, target).cpu().contiguous().clone()
    return out


def cast_to_target_dtype(name: str, tensor: torch.Tensor, target: OutputFormat) -> torch.Tensor:
    """Store an unquantized tensor in the target's dtype, without touching integer data.

    DCP keeps fp32 master weights while the target stores most of them as bfloat16.
    Casting is restricted to floating-point tensors: ``.to()`` converts values rather than
    reinterpreting them, so applying it to an integer buffer such as the hash-router
    ``tid2eid`` table would silently corrupt it.
    """
    want_dtype = target.schema[name][0]
    if tensor.dtype == want_dtype:
        return tensor
    if not (tensor.is_floating_point() and want_dtype.is_floating_point):
        raise ValueError(f"{name}: produced {tensor.dtype} but target stores {want_dtype}")
    return tensor.to(want_dtype)


def validate_against_target(tensors: Dict[str, torch.Tensor], target: OutputFormat) -> None:
    for name, tensor in tensors.items():
        want_dtype, want_shape = target.schema[name]
        if tensor.dtype != want_dtype or tuple(tensor.shape) != want_shape:
            raise ValueError(
                f"{name}: produced {tensor.dtype} {tuple(tensor.shape)}, target expects {want_dtype} {want_shape}"
            )


def shard_is_complete(out_dir: str, shard: str, target: OutputFormat) -> bool:
    """True when ``shard`` holds exactly the target's keys and its data is fully written.

    The header alone is not enough: safetensors writes it first, so a run killed midway
    leaves a truncated file whose header still lists every key.
    """
    path = os.path.join(out_dir, shard)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        header.pop("__metadata__", None)
        if set(header) != set(target.keys_by_shard[shard]):
            return False
        for key, meta in header.items():
            dtype, shape = target.schema[key]
            if SAFETENSORS_DTYPES[meta["dtype"]] != dtype or tuple(meta["shape"]) != shape:
                return False
        data_end = max(meta["data_offsets"][1] for meta in header.values())
        return os.path.getsize(path) == 8 + header_len + data_end
    except (OSError, ValueError, KeyError, struct.error, json.JSONDecodeError):
        return False


def write_chat_template(out_dir: str, chat_template: str) -> None:
    """Store the Jinja source as a standalone ``chat_template.jinja`` in the output.

    A base release ships no chat template, so every consumer of a fine-tuned checkpoint would
    otherwise have to rebuild the training-time prompt by hand. Transformers and vLLM both load
    this file automatically, so ``apply_chat_template`` and vLLM's server mode use the template
    the model was trained with. It stays out of ``tokenizer_config.json`` because this is the
    layout ``save_pretrained`` itself writes, and Jinja is far more readable as its own file.
    """
    path = os.path.join(out_dir, CHAT_TEMPLATE_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(chat_template if chat_template.endswith("\n") else chat_template + "\n")
    logger.info(f"wrote {CHAT_TEMPLATE_NAME} ({len(chat_template)} chars)")


def has_jinja_delimiter(source: str) -> bool:
    return any(delimiter in source for delimiter in ("{{", "{%", "{#"))


def resolve_custom_template(value: str) -> str:
    """Return the Jinja source behind ``--custom-template``, which takes a path or the text itself.

    A chat template always contains Jinja delimiters and a path never does, so the two need no
    separate flags. Everything that is not a usable template has to be rejected right here: a
    mistyped path, a file that holds something else, or a template that will not compile all
    render into the checkpoint as a prompt that only looks wrong at inference time, hours after
    the merge reported success. ``utf-8-sig`` keeps a BOM from becoming a literal prompt prefix.
    """
    if not has_jinja_delimiter(value):
        try:
            with open(value, encoding="utf-8-sig") as fh:
                value = fh.read()
        except (OSError, UnicodeDecodeError) as err:
            raise ValueError(f"holds no Jinja delimiters, so it was read as a path: {err}") from err
        if not has_jinja_delimiter(value):
            raise ValueError("points at a file that holds no Jinja delimiters, so it is not a chat template")
    try:
        _compile_jinja_template(value)
    except TemplateSyntaxError as err:
        raise ValueError(f"is not valid Jinja: {err}") from err
    return value


def copy_tokenizer_config_without_template(src: str, dst: str) -> None:
    """Copy ``tokenizer_config.json``, dropping an inlined ``chat_template``.

    Transformers prefers a standalone ``chat_template.jinja`` over this key, and its own
    ``save_pretrained`` deletes the key when it writes that file. Leaving a target release's
    template behind would hand anything that reads the JSON directly a stale second answer.
    """
    with open(src, encoding="utf-8") as fh:
        config = json.load(fh)
    if "chat_template" not in config:
        shutil.copyfile(src, dst)
        return
    del config["chat_template"]
    logger.warning(f"dropped the target chat_template from {TOKENIZER_CONFIG_NAME} for --custom-template")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def prepare_assets(target: OutputFormat) -> Dict[str, str]:
    """Download only tokenizer metadata, using the standard Hugging Face cache."""
    names = ["tokenizer.json", TOKENIZER_CONFIG_NAME]
    if target.name == "v4-flash":
        names.append("generation_config.json")
    return {name: hf_hub_download(target.repo_id, name, revision=target.revision) for name in names}


def write_assets(target: OutputFormat, out_dir: str, assets: Dict[str, str], chat_template: Optional[str]) -> None:
    for name, src in assets.items():
        dst = os.path.join(out_dir, name)
        if name == TOKENIZER_CONFIG_NAME and chat_template is not None:
            copy_tokenizer_config_without_template(src, dst)
        else:
            shutil.copyfile(src, dst)
    if "generation_config.json" not in assets:
        stale_generation = os.path.join(out_dir, "generation_config.json")
        if os.path.exists(stale_generation):
            os.remove(stale_generation)
    stale_template = os.path.join(out_dir, CHAT_TEMPLATE_NAME)
    if os.path.exists(stale_template):
        os.remove(stale_template)
    if chat_template is not None:
        write_chat_template(out_dir, chat_template)
    config = dict(target.config, num_nextn_predict_layers=0)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")


def worker_device(base: torch.device, rank: int) -> str:
    """Spread workers over the visible CUDA devices, starting from ``base``.

    One layer's quantization needs tens of GB of device memory, so co-locating every worker on
    ``--device`` would only trade the read bottleneck for an OOM.
    """
    if not IS_CUDA_AVAILABLE or base.type != get_device_type() or get_torch_device().device_count() <= 1:
        return str(base)
    return f"cuda:{((base.index or 0) + rank) % get_torch_device().device_count()}"


def run_conversion_workers(workers: int, device: torch.device, entrypoint: str, argv: list[str]) -> None:
    """Re-invoke this script in ``workers`` subprocesses, each taking every ``workers``-th shard.

    Shards are independent — each reads its own DCP slices and writes its own output file — and
    ``shard_is_complete`` already makes a re-run idempotent, so the work only needs splitting.
    Re-invoking instead of forking keeps each worker's DCP reader and CUDA context clean; the
    duplicated metadata read costs a few seconds against a run measured in tens of minutes.
    """
    procs = []
    for rank in range(workers):
        assigned = worker_device(device, rank)
        command = [
            sys.executable,
            os.path.abspath(entrypoint),
            *argv,
            # Trailing flags win in argparse, so these override whatever the parent was given.
            "--worker-index",
            str(rank),
            "--worker-count",
            str(workers),
            "--device",
            assigned,
        ]
        logger.info(f"worker {rank}: starting on {assigned}")
        procs.append(subprocess.Popen(command))

    codes = [proc.wait() for proc in procs]
    failed = [rank for rank, code in enumerate(codes) if code != 0]
    if failed:
        raise RuntimeError(f"conversion workers {failed} failed with exit codes {codes}")


def merge_checkpoint(args, parser, *, entrypoint: str, argv: list[str]) -> None:
    """Export native V4 weights using the shared DCP shard pipeline."""
    # The template is written alongside the copied assets, which both of these turn off.
    if args.custom_template is not None and (args.skip_assets or args.layers):
        parser.error(f"--custom-template writes {CHAT_TEMPLATE_NAME}, so it conflicts with --skip-assets/--layers")
    custom_template = None
    if args.custom_template is not None:
        try:
            custom_template = resolve_custom_template(args.custom_template)
        except ValueError as err:
            parser.error(f"--custom-template {err}")
    if args.ep_size < 1:
        parser.error("--ep-size must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    device = torch.device(args.device)
    if os.path.realpath(args.save_dir) == os.path.realpath(args.load_dir):
        parser.error("--save-dir must differ from --load-dir")
    os.makedirs(args.save_dir, exist_ok=True)

    target = OutputFormat(args.format)
    expert_dtype = target.expert_dtype()
    converter = DeepseekV4CheckpointTensorConverter(num_experts=target.num_experts())
    logger.info(
        f"target: {len(target.weight_map)} keys, {len(set(target.weight_map.values()))} shards, "
        f"{target.num_experts()} routed experts, expert_dtype={expert_dtype}"
    )

    reader = CachedFileSystemReader(args.load_dir)
    metadata = reader.read_metadata()
    per_layer, globals_ = group_source_keys(metadata, converter)
    logger.info(f"source: {len(per_layer)} layers, {len(globals_)} non-layer tensors")

    ep_fsdp_size = 1
    if args.ep_size > 1:
        ep_fsdp_size = resolve_expert_fsdp_size(metadata, per_layer, args.ep_size)
        logger.info(f"expert dim: un-permuting DCP mesh order (ep_size={args.ep_size}, ep_fsdp_size={ep_fsdp_size})")

    wanted: Optional[set] = None
    if args.layers:
        wanted = set()
        for part in args.layers.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                wanted.update(range(int(lo), int(hi) + 1))
            else:
                wanted.add(int(part))
        if not wanted or wanted - per_layer.keys():
            parser.error(f"--layers contains unavailable layers: {sorted(wanted - per_layer.keys())}")
        logger.info(f"restricting to layers {sorted(wanted)}")

    groups: List[Tuple[str, Dict[str, str]]] = []
    for layer in sorted(per_layer):
        if wanted is not None and layer not in wanted:
            continue
        keys = per_layer[layer]
        shards = {target.shard_of(representative_native_key(converter, k)) for k in keys}
        if len(shards) != 1:
            raise ValueError(f"layer {layer} spans multiple target shards: {sorted(shards)}")
        groups.append((shards.pop(), keys))

    if wanted is None:
        by_shard: Dict[str, Dict[str, str]] = defaultdict(dict)
        for hf_key, dcp_key in globals_.items():
            by_shard[target.shard_of(representative_native_key(converter, hf_key))][hf_key] = dcp_key
        groups.extend(sorted(by_shard.items()))

        # The index and assets copied at the end describe the target in full, so a
        # checkpoint that cannot fill every shard would leave the output pointing at
        # files that were never written.
        covered = {shard for shard, _ in groups}
        uncovered = sorted(set(target.weight_map.values()) - covered)
        if uncovered:
            raise ValueError(
                f"checkpoint does not cover {len(uncovered)} target shard(s), output would be "
                f"incomplete: {uncovered[:5]}"
            )

    for shard, keys in groups:
        validate_source_group(metadata, keys, converter, target, shard)

    assets = {}
    if args.worker_index is None and not args.skip_assets and wanted is None:
        assets = prepare_assets(target)

    total_bytes = 0
    # Every shard this run is responsible for, whether it was rewritten or found complete.
    converted_shards = {shard for shard, _ in groups}
    started = time.time()

    is_worker = args.worker_index is not None
    if is_worker:
        assigned = [group for index, group in enumerate(groups) if index % args.worker_count == args.worker_index]
    else:
        assigned = groups

    if args.workers > 1 and not is_worker:
        # The parent converts nothing itself; it only fans out and then finalizes the output.
        run_conversion_workers(args.workers, device, entrypoint, argv)
        assigned = []
        total_bytes = sum(
            os.path.getsize(os.path.join(args.save_dir, shard))
            for shard in converted_shards
            if os.path.isfile(os.path.join(args.save_dir, shard))
        )

    def transform(keys, source):
        tensors = convert_group(keys, source, converter, target, expert_dtype, device, args.ep_size, ep_fsdp_size)
        validate_against_target(tensors, target)
        shard = target.shard_of(representative_native_key(converter, next(iter(keys))))
        expected = set(target.keys_by_shard[shard])
        if set(tensors) != expected:
            raise ValueError(f"{shard}: key mismatch vs target")
        return tensors

    export_shards(
        assigned,
        reader,
        args.save_dir,
        transform,
        shard_metadata=target.shard_metadata,
        skip_shard=None if args.overwrite else lambda shard: shard_is_complete(args.save_dir, shard, target),
    )
    total_bytes += sum(os.path.getsize(os.path.join(args.save_dir, shard)) for shard, _ in assigned)

    if is_worker:
        logger.info(f"worker {args.worker_index}: done, {total_bytes / 1e9:.1f} GB")
        return

    if wanted is None:
        if not args.skip_assets:
            write_assets(target, args.save_dir, assets, custom_template)
        write_index(args.save_dir, target.weight_map, sum(target.tensor_nbytes(k) for k in target.weight_map))

    logger.info(f"done: {total_bytes / 1e9:.1f} GB in {(time.time() - started) / 60:.1f} min -> {args.save_dir}")
