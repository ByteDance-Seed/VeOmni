import argparse
import gc
import glob
import json
import os
import shutil
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan, _process_shard
from veomni.utils import helper


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = helper.create_logger(__name__)


# PEFT LoRA adapter key markers. The training DCP keeps PEFT's wrapped FQNs intact
# (e.g. ``base_model.model.<...>.lora_A.default.weight``), so a substring check on
# the HF-normalized keys is enough to spot a LoRA checkpoint.
_LORA_KEY_MARKERS = (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B.")


def _is_lora_key(hf_key: str) -> bool:
    return any(marker in hf_key for marker in _LORA_KEY_MARKERS)


def _detect_lora(all_hf_keys: Sequence[str]) -> bool:
    """Return True if any of the HF-normalized keys looks like a PEFT LoRA adapter."""
    return any(_is_lora_key(k) for k in all_hf_keys)


@torch.no_grad()
def save_lora_adapter_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    adapter_config_path: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Convert a DCP checkpoint that contains a PEFT LoRA adapter to ``adapter_model.safetensors``.

    Only ``*.lora_A.*`` / ``*.lora_B.*`` keys are exported, mirroring what
    ``veomni.utils.save_safetensor_utils.save_lora_adapter_with_dcp`` writes during
    training. The base model weights present in the DCP (frozen during LoRA fine-tuning)
    are intentionally dropped: at inference time they must come from the original
    base model path the LoRA was trained against.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving LoRA adapter to {output_dir}")

    # ``shard_size=None`` forces a single shard so we get ``adapter_model.safetensors`` directly.
    all_keys, _total_size, _all_dcp_keys = _get_sharding_plan(checkpoint_path, shard_size=None, save_dtype=save_dtype)
    lora_keys = {hf_k: dcp_k for hf_k, dcp_k in all_keys.items() if _is_lora_key(hf_k)}
    if not lora_keys:
        raise RuntimeError(
            f"LoRA conversion requested but no LoRA keys (.lora_A./.lora_B./...) found under {checkpoint_path}"
        )

    logger.info(f"Found {len(lora_keys)} LoRA tensors; loading and re-saving as adapter_model.safetensors")
    processed_dict = _process_shard(lora_keys, checkpoint_path, save_dtype)

    save_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(processed_dict, save_path, metadata={"format": "pt"})
    del processed_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if adapter_config_path is not None:
        adapter_config_path = str(adapter_config_path)
        if not os.path.isfile(adapter_config_path):
            raise FileNotFoundError(f"--adapter-config-path does not exist: {adapter_config_path}")
        shutil.copyfile(adapter_config_path, os.path.join(output_dir, "adapter_config.json"))
        logger.info(f"Copied adapter_config.json from {adapter_config_path}")
    else:
        logger.warning(
            "No --adapter-config-path provided. ``adapter_model.safetensors`` was written, but you must drop "
            "``adapter_config.json`` (from the matching training run's ``output_dir/global_step_*/``) next to it "
            "before the adapter can be loaded by peft / diffusers."
        )

    logger.info("LoRA adapter conversion complete.")


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    shard_size: int = 2_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """Convert DCP checkpoint to HuggingFace format with shard-by-shard processing (memory-efficient)."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model weights to {output_dir}")
    logger.info(
        f"Format: {'safetensors' if safe_serialization else 'pytorch'}, dtype={save_dtype}, shard_size={shard_size}"
    )

    # Plan shards from metadata
    logger.info("Analyzing DCP metadata and planning shards...")
    shards, total_size, all_dcp_keys = _get_sharding_plan(checkpoint_path, shard_size, save_dtype)

    logger.info(f"Found {len(all_dcp_keys)} model tensors, total size: ~{total_size / 1e9:.2f}GB")
    logger.info(f"Split into {len(shards)} shards")

    if len(shards) == 0:
        logger.warning("No model weights found! Check if checkpoint path is correct and contains 'model.' keys.")
        return

    # Process each shard
    weight_map = OrderedDict()
    num_shards = len(shards)

    for shard_idx, shard_keys in enumerate(shards):
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        if num_shards == 1:
            filename = weights_name
        else:
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"

        save_path = os.path.join(output_dir, filename)
        logger.info(f"Processing shard {shard_idx + 1}/{num_shards}: {filename} ({len(shard_keys)} tensors)")

        processed_dict = _process_shard(shard_keys, checkpoint_path, save_dtype)

        # Save shard
        if safe_serialization:
            save_file(processed_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(processed_dict, save_path)

        del processed_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for hf_key in shard_keys.keys():
            weight_map[hf_key] = filename

    # Save index file for multi-shard checkpoints
    if num_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger.info(f"Saved index file to {index_file}")

    logger.info("Weight conversion complete.")

    # Save model assets (config, tokenizer, processor)
    if model_assets is not None:
        for model_asset in model_assets:
            if hasattr(model_asset, "save_pretrained"):
                model_asset.save_pretrained(output_dir)
                logger.info(f"Saved model asset: {type(model_asset).__name__}")
            else:
                logger.warning(f"Model asset {model_asset} does not implement `save_pretrained`")


# ---------------------------------------------------------------------------
# HunyuanImage 3 DCP -> official HF export
# ---------------------------------------------------------------------------
#
# The T2I runtime model stores a *subset* of the official checkpoint in a fused
# layout (renamed embed/norm, fused MoE experts). Exporting back to a directory
# the official / HF path can reload requires three things:
#
#   1. Run every trained runtime tensor through ``HunyuanImage3CheckpointExporter``
#      (the exact inverse of the import converter: expert split + [gate,up]->[up,gate]
#      swap + the two renames; identity for everything else).
#   2. Restore the components the runtime never held (``lm_head`` / ``vae.decoder`` /
#      vision, per component policy) byte-for-byte from the pinned official Base.
#   3. Emit sharded safetensors + ``model.safetensors.index.json`` whose official key
#      set matches the Base's, plus the Base's config / tokenizer / auxiliary files.
#
# This path is gated on ``config.model_type == "hunyuan_image_3_moe"`` in ``main``;
# every other model keeps the generic ``save_model_weights`` path unchanged.

_HUNYUAN_IMAGE_3_MODEL_TYPE = "hunyuan_image_3_moe"

# Base files copied verbatim next to the exported weights so the output directory is a
# drop-in official checkpoint. ``*.safetensors`` / index are written by us and excluded.
_HUNYUAN_ASSET_SUFFIXES = (".json", ".py", ".md", ".txt", ".model")


def _resolve_dtype(save_dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    """Map a dtype spec to a ``torch.dtype``; ``None`` / 'base' / 'auto' means keep source."""
    if save_dtype is None:
        return None
    if isinstance(save_dtype, torch.dtype):
        return save_dtype
    if save_dtype in ("base", "auto", "source", ""):
        return None
    return getattr(torch, save_dtype)


class _StreamingShardWriter:
    """Accumulate ``(name, tensor)`` pairs and flush size-bounded safetensors shards.

    Shard filenames are only finalized in :meth:`finalize` because the total shard
    count (needed for the ``-NNNNN-of-MMMMM`` suffix) is unknown until every tensor
    has been seen. Shards are written to temporary ``.part-*`` files and renamed at
    the end; the index is always emitted so the output mirrors the official
    multi-shard layout regardless of how many shards were produced.
    """

    def __init__(self, output_dir: str, shard_size: int, save_dtype: Optional[torch.dtype]) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.save_dtype = save_dtype
        self._buffer: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._buffer_size = 0
        self._parts: list[tuple[str, list[str]]] = []
        self.total_size = 0
        self.weight_map: "OrderedDict[str, str]" = OrderedDict()

    def add(self, name: str, tensor: torch.Tensor, cast: bool = True) -> None:
        if name in self.weight_map or name in self._buffer:
            raise ValueError(f"Duplicate output tensor key: {name}")
        if cast and self.save_dtype is not None:
            tensor = tensor.to(dtype=self.save_dtype)
        tensor = tensor.detach().cpu().contiguous().clone()
        size = tensor.numel() * tensor.element_size()
        if self._buffer and self.shard_size and self._buffer_size + size > self.shard_size:
            self._flush()
        self._buffer[name] = tensor
        self._buffer_size += size
        self.total_size += size

    def _flush(self) -> None:
        part_name = f".part-{len(self._parts) + 1:05d}.safetensors"
        save_file(dict(self._buffer), os.path.join(self.output_dir, part_name), metadata={"format": "pt"})
        self._parts.append((part_name, list(self._buffer.keys())))
        self._buffer = OrderedDict()
        self._buffer_size = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def finalize(self) -> tuple["OrderedDict[str, str]", int]:
        if self._buffer:
            self._flush()
        num_shards = len(self._parts)
        for shard_idx, (part_name, keys) in enumerate(self._parts):
            if num_shards == 1:
                filename = SAFE_WEIGHTS_NAME
            else:
                prefix, extension = SAFE_WEIGHTS_NAME.rsplit(".", maxsplit=1)
                filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
            os.replace(os.path.join(self.output_dir, part_name), os.path.join(self.output_dir, filename))
            for key in keys:
                self.weight_map[key] = filename

        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": dict(self.weight_map),
        }
        with open(os.path.join(self.output_dir, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")
        return self.weight_map, self.total_size


def _base_index_weight_map(base_dir: str) -> "OrderedDict[str, str]":
    """Return the official ``{key: shard_file}`` map, falling back to a single shard."""
    index_path = os.path.join(base_dir, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            return OrderedDict(json.load(f)["weight_map"])
    single = os.path.join(base_dir, SAFE_WEIGHTS_NAME)
    if os.path.isfile(single):
        with safe_open(single, framework="pt") as handle:
            return OrderedDict((key, SAFE_WEIGHTS_NAME) for key in handle.keys())
    raise FileNotFoundError(f"No safetensors index or {SAFE_WEIGHTS_NAME} found under base_dir: {base_dir}")


def _iter_base_absent_tensors(
    base_dir: str, weight_map: "OrderedDict[str, str]", prefixes: Sequence[str]
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` for every Base key under one of ``prefixes`` (byte-for-byte).

    Reads are grouped by shard file so each Base shard is opened once.
    """
    if not prefixes:
        return
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_file in weight_map.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            keys_by_shard[shard_file].append(key)
    for shard_file, keys in keys_by_shard.items():
        with safe_open(os.path.join(base_dir, shard_file), framework="pt") as handle:
            for key in keys:
                yield key, handle.get_tensor(key)


def _copy_hunyuan_assets(base_dir: str, output_dir: str) -> None:
    """Copy config / tokenizer / auxiliary Base files (never weights or index)."""
    for path in sorted(glob.glob(os.path.join(base_dir, "*"))):
        name = os.path.basename(path)
        if not os.path.isfile(path):
            continue
        if name.endswith(".safetensors") or name in (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME):
            continue
        if not name.endswith(_HUNYUAN_ASSET_SUFFIXES):
            continue
        shutil.copyfile(path, os.path.join(output_dir, name))


def _verify_hunyuan_export(output_dir: str, base_key_set: set) -> None:
    """Reload the written index and assert its key set matches the Base's exactly."""
    logger.info("Verifying exported checkpoint against the Base official key set...")
    index_path = os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path, encoding="utf-8") as f:
        written_map = json.load(f)["weight_map"]
    written_keys = set(written_map)
    missing = sorted(base_key_set - written_keys)
    extra = sorted(written_keys - base_key_set)
    if missing or extra:
        raise RuntimeError(
            "Exported official key set does not match the Base: "
            f"missing={missing[:10]} (+{max(0, len(missing) - 10)} more), "
            f"extra={extra[:10]} (+{max(0, len(extra) - 10)} more)."
        )
    # Reload every shard's metadata so a corrupt / unreadable shard fails loudly.
    for shard_file in sorted(set(written_map.values())):
        with safe_open(os.path.join(output_dir, shard_file), framework="pt") as handle:
            shard_keys = set(handle.keys())
        for key, mapped in written_map.items():
            if mapped == shard_file and key not in shard_keys:
                raise RuntimeError(f"Index maps {key} to {shard_file} but the shard does not contain it.")
    logger.info(f"Verify OK: {len(written_keys)} official keys match the Base; all shards readable.")


@torch.no_grad()
def save_hunyuan_image_3_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    base_dir: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    shard_size: int = 2_000_000_000,
    verify: bool = False,
) -> None:
    """Export a trained HunyuanImage 3 DCP checkpoint to the official HF layout.

    Trained runtime tensors are streamed shard-by-shard from the DCP, expanded to the
    official split-expert / renamed layout by :class:`HunyuanImage3CheckpointExporter`,
    and written to size-bounded safetensors shards. Components absent from the runtime
    (per component policy) are then restored byte-for-byte from the pinned ``base_dir``.
    Config / tokenizer / auxiliary Base files are copied so ``output_dir`` reloads via
    the official / HF path.
    """
    from veomni.models.transformers.hunyuan_image_3.checkpoint_tensor_export import (
        HunyuanImage3CheckpointExporter,
        absent_official_prefixes,
    )
    from veomni.models.transformers.hunyuan_image_3.component_policy import HunyuanImage3ComponentPolicy

    output_dir = str(output_dir)
    base_dir = str(base_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(base_dir, "config.json"), encoding="utf-8") as f:
        base_config = json.load(f)

    num_experts = int(base_config["num_experts"])
    hidden_size = int(base_config["hidden_size"])
    intermediate = base_config.get("moe_intermediate_size", base_config.get("intermediate_size"))
    intermediate_size = int(intermediate[0] if isinstance(intermediate, list) else intermediate)
    # component_policy is a training recipe, absent from the official Base
    # config.json. Source it from the trained checkpoint's manifest (extra_hashes);
    # fall back to base_config only for legacy runtime-overlay base dirs.
    from veomni.checkpoint.checkpoint_manifest import read_checkpoint_manifest

    manifest = read_checkpoint_manifest(str(checkpoint_path))
    component_policy_values = None
    if manifest is not None:
        component_policy_values = (manifest.get("extra_hashes") or {}).get("component_policy")
    if component_policy_values is None:
        component_policy_values = base_config.get("component_policy")
    if component_policy_values is None:
        raise KeyError(
            "component_policy not found in the checkpoint manifest "
            f"({os.path.join(str(checkpoint_path), 'checkpoint_manifest.json')}) or in "
            f"{os.path.join(base_dir, 'config.json')}. Re-run training with the manifest-writing "
            "checkpoint callback, or point --base-dir at a runtime config carrying component_policy."
        )
    policy = HunyuanImage3ComponentPolicy.from_dict(component_policy_values)
    absent_prefixes = absent_official_prefixes(policy)

    exporter = HunyuanImage3CheckpointExporter(
        num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    save_dtype = _resolve_dtype(save_dtype)

    logger.info(f"HunyuanImage 3 export: DCP={checkpoint_path} base={base_dir} -> {output_dir}")
    logger.info(
        f"num_experts={num_experts}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
        f"dtype={save_dtype}, shard_size={shard_size}, absent_prefixes={absent_prefixes}"
    )

    base_weight_map = _base_index_weight_map(base_dir)
    base_key_set = set(base_weight_map)

    writer = _StreamingShardWriter(output_dir, shard_size, save_dtype)

    # 1) Stream trained runtime tensors -> official layout. ``shards`` is the generic
    #    merge plan keyed by runtime (HF-normalized) name; we re-expand each tensor.
    shards, _total, all_dcp_keys = _get_sharding_plan(checkpoint_path, shard_size, save_dtype=None)
    logger.info(f"Loaded DCP plan: {len(all_dcp_keys)} runtime tensors across {len(shards)} read shards")
    exported_count = 0
    for shard_idx, shard_keys in enumerate(shards):
        logger.info(f"Exporting runtime shard {shard_idx + 1}/{len(shards)} ({len(shard_keys)} tensors)")
        runtime_dict = _process_shard(shard_keys, checkpoint_path, save_dtype=None)
        for runtime_name in shard_keys:  # deterministic (planner sorts by hf key)
            tensor = runtime_dict.pop(runtime_name)
            for official_name, official_tensor in exporter.export_tensor(runtime_name, tensor):
                writer.add(official_name, official_tensor)
                exported_count += 1
            del tensor
        del runtime_dict
        gc.collect()

    # 2) Restore absent components byte-for-byte from the Base (no dtype cast).
    restored_count = 0
    for official_name, tensor in _iter_base_absent_tensors(base_dir, base_weight_map, absent_prefixes):
        writer.add(official_name, tensor, cast=False)
        restored_count += 1

    weight_map, total_size = writer.finalize()
    logger.info(
        f"Wrote {len(weight_map)} official tensors "
        f"({exported_count} exported from DCP + {restored_count} restored from Base), "
        f"~{total_size / 1e9:.2f}GB across {len(set(weight_map.values()))} shards"
    )

    # 3) Copy config / tokenizer / auxiliary files so the output reloads as official.
    _copy_hunyuan_assets(base_dir, output_dir)

    # 4) Validate the official key set matches the Base's.
    written_keys = set(weight_map)
    missing = sorted(base_key_set - written_keys)
    extra = sorted(written_keys - base_key_set)
    if missing or extra:
        raise RuntimeError(
            "Exported official key set does not match the Base index: "
            f"missing={missing[:10]} (+{max(0, len(missing) - 10)} more), "
            f"extra={extra[:10]} (+{max(0, len(extra) - 10)} more). "
            "This usually means the DCP was trained with a different component_policy than the Base config."
        )
    logger.info(f"Official key set matches Base ({len(written_keys)} keys).")

    if verify:
        _verify_hunyuan_export(output_dir, base_key_set)

    logger.info("HunyuanImage 3 export complete.")


def merge_to_hf_pt(
    load_dir: str, save_path: str, model_assets_dir: Optional[str] = None, shard_size: int = 2_000_000_000
) -> None:
    """Main conversion function: load DCP from load_dir and save HF format to save_path."""
    model_assets = None
    if model_assets_dir is not None:
        logger.info(f"Loading model assets from {model_assets_dir}")
        model_assets = []
        try:
            config = AutoConfig.from_pretrained(model_assets_dir)
            model_assets.append(config)
        except Exception as e:
            logger.warning(f"Failed to load AutoConfig: {e}")

        try:
            processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=True)
            model_assets.append(processor)
        except Exception as e:
            logger.warning(f"Failed to load AutoProcessor: {e}")

        if not model_assets:
            model_assets = None

    save_model_weights(save_path, load_dir, shard_size=shard_size, model_assets=model_assets)


def main():
    parser = argparse.ArgumentParser(
        description="Merge DCP checkpoint to HuggingFace format (streaming optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--load-dir", type=str, required=True, help="Directory containing DCP checkpoint")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for HuggingFace format checkpoint (default: <load-dir>/hf_ckpt)",
    )
    parser.add_argument(
        "--model-assets-dir",
        type=str,
        default=None,
        help="Directory containing model config and processor (optional)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2_000_000_000,
        help="Maximum shard size in bytes (default: 2GB)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "full", "lora"),
        default="auto",
        help=(
            "Conversion mode. 'auto' (default) inspects DCP keys: writes adapter_model.safetensors when the "
            "checkpoint contains PEFT LoRA keys, otherwise writes a full sharded HF safetensors dump. "
            "'full' / 'lora' force the corresponding mode."
        ),
    )
    parser.add_argument(
        "--adapter-config-path",
        type=str,
        default=None,
        help=(
            "Path to the matching adapter_config.json produced during LoRA training "
            "(usually under <output_dir>/global_step_*/adapter_config.json). Only used in 'lora' mode; "
            "copied next to adapter_model.safetensors so the adapter is loadable as-is."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help=(
            "Path to the pinned official Base checkpoint (config.json + safetensors index + shards). "
            "Required for HunyuanImage 3 export: components absent from the runtime (lm_head / vae.decoder / "
            "vision) are restored byte-for-byte from here, and config/tokenizer files are copied. When this "
            "directory's config.model_type is 'hunyuan_image_3_moe' the HunyuanImage 3 export path is used; "
            "otherwise this flag is ignored."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help=(
            "Target dtype for exported (trained) tensors, e.g. 'bfloat16' / 'float16' / 'float32'. Use 'base' "
            "to keep the source dtype. HunyuanImage 3 export only; absent components are copied byte-for-byte "
            "from the Base regardless."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After export, reload the written index/shards and assert the official key set matches the Base.",
    )
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir
    shard_size = args.shard_size

    # HunyuanImage 3 export path: gated on the Base config's model_type so every other
    # model keeps the generic DCP->HF behavior untouched.
    if args.base_dir is not None:
        base_config_path = os.path.join(args.base_dir, "config.json")
        base_model_type = None
        if os.path.isfile(base_config_path):
            with open(base_config_path, encoding="utf-8") as f:
                base_model_type = json.load(f).get("model_type")
        if base_model_type == _HUNYUAN_IMAGE_3_MODEL_TYPE:
            save_hunyuan_image_3_weights(
                output_dir=save_dir,
                checkpoint_path=load_dir,
                base_dir=args.base_dir,
                save_dtype=args.dtype,
                shard_size=shard_size,
                verify=args.verify,
            )
            return
        logger.warning(
            f"--base-dir was provided but its config.model_type is {base_model_type!r} "
            f"(not {_HUNYUAN_IMAGE_3_MODEL_TYPE!r}); ignoring --base-dir and using the generic merge path."
        )

    mode = args.mode
    if mode == "auto":
        _shards_for_detection, _size, _dcp_keys = _get_sharding_plan(load_dir, shard_size=None, save_dtype="bfloat16")
        # _shards_for_detection is a single {hf_key: dcp_key} dict when shard_size is None
        detected_lora = _detect_lora(_shards_for_detection.keys())
        mode = "lora" if detected_lora else "full"
        logger.info(
            f"Auto-detected mode: {mode} "
            f"({'LoRA keys present' if detected_lora else 'no LoRA keys; treating as full checkpoint'})"
        )

    if mode == "lora":
        save_lora_adapter_weights(
            output_dir=save_dir,
            checkpoint_path=load_dir,
            adapter_config_path=args.adapter_config_path,
        )
    else:
        if args.adapter_config_path is not None:
            logger.warning("--adapter-config-path is only used in 'lora' mode; ignoring.")
        merge_to_hf_pt(load_dir, save_dir, model_assets_dir, shard_size=shard_size)


if __name__ == "__main__":
    main()
