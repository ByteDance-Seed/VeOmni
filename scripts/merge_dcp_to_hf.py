import argparse
import gc
import json
import multiprocessing
import os
import shutil
import tempfile
import uuid
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GenerationConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan, _process_shard
from veomni.checkpoint.layout import weights_dir
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


def _aliased_hf_keys(model_assets_dir: str) -> Optional[set[str]]:
    """The state-dict names that are only a second name for a tensor exported elsewhere.

    A tensor shared under several names is reported under all of them by ``state_dict()``,
    so the training DCP carries all of them too. Every model with ``tie_word_embeddings``
    has ``lm_head.weight`` aliasing the embedding that way; DiffusionGemma is far worse,
    because its encoder *is* its decoder and 628 of its 1675 names are aliases --
    exporting them duplicates real bytes, and before the ExtraParallel spec covered alias
    FQNs the expert aliases were also the wrong shape, which made the export unloadable.

    ``named_parameters`` deduplicates and keeps the first registered name, which is the
    canonical one by construction: DiffusionGemma registers ``decoder`` before ``encoder``
    precisely so the published checkpoint is keyed by it. Everything in ``state_dict()``
    beyond that set is an alias, and ``from_pretrained`` re-establishes those ties itself.

    Returned as the names to *drop* rather than the names to keep, so a key-space
    mismatch -- the wrong assets directory, a different wrapper depth -- cannot silently
    prune real weights. Returns ``None`` when the model class cannot be resolved, which
    leaves the export as it was rather than guessing at which names are aliases.
    """
    try:
        from veomni.models.loader import get_model_class, get_model_config
        from veomni.models.module_utils import init_empty_weights

        config = get_model_config(model_assets_dir)
        model_class = get_model_class(config)
        # `MODELING_BACKEND=hf` resolves to an Auto class, which exposes `from_config`
        # instead. Weights stay on meta either way, so this costs a config parse at 26B.
        build = getattr(model_class, "_from_config", None) or model_class.from_config
        with init_empty_weights():
            model = build(config)
    except Exception as e:
        logger.warning(
            f"Could not resolve the model class from {model_assets_dir}, so tied weights cannot be "
            f"identified and every DCP tensor will be exported: {e}"
        )
        return None

    canonical = {name for name, _ in model.named_parameters()}
    canonical |= {name for name, _ in model.named_buffers()}
    # Against `state_dict()`, not against `canonical`: non-persistent buffers are named
    # but never serialized, so subtracting the other way around reports a bogus count.
    aliased = set(model.state_dict()) - canonical
    logger.info(
        f"Resolved {type(model).__name__}: {len(canonical)} distinct tensors, "
        f"{len(aliased)} aliased name(s) in its state dict"
    )
    return aliased


def _save_state_dict(
    state_dict: dict[str, torch.Tensor],
    save_path: Union[str, os.PathLike],
    *,
    safe_serialization: bool,
    local_staging_dir: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Write one shard, optionally staging locally for filesystems without mmap support."""
    save_path = os.fspath(save_path)
    if local_staging_dir is None:
        if safe_serialization:
            save_file(state_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(state_dict, save_path)
        return

    staging_dir = os.fspath(local_staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    descriptor, staged_path = tempfile.mkstemp(
        prefix="veomni-dcp-to-hf-",
        suffix=os.path.splitext(save_path)[1],
        dir=staging_dir,
    )
    os.close(descriptor)
    os.unlink(staged_path)
    destination_temporary = f"{save_path}.incomplete-{uuid.uuid4().hex}"
    try:
        if safe_serialization:
            save_file(state_dict, staged_path, metadata={"format": "pt"})
        else:
            torch.save(state_dict, staged_path)
        staged_size = os.path.getsize(staged_path)
        with open(staged_path, "rb") as source, open(destination_temporary, "xb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        copied_size = os.path.getsize(destination_temporary)
        if copied_size != staged_size:
            raise OSError(
                f"staged shard copy size mismatch for {save_path}: expected {staged_size}, got {copied_size}"
            )
        os.replace(destination_temporary, save_path)
    finally:
        if os.path.exists(staged_path):
            os.unlink(staged_path)
        if os.path.exists(destination_temporary):
            os.unlink(destination_temporary)


def _process_and_save_shard(
    shard_idx: int,
    num_shards: int,
    shard_keys: dict[str, str],
    checkpoint_path: str,
    output_dir: str,
    save_dtype: Optional[Union[str, torch.dtype]],
    safe_serialization: bool,
    local_staging_dir: Optional[str],
) -> tuple[int, str]:
    """Load and publish one independent HF shard inside a worker process."""
    torch.set_grad_enabled(False)
    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    if num_shards == 1:
        filename = weights_name
    else:
        prefix, extension = weights_name.rsplit(".", maxsplit=1)
        filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
    save_path = os.path.join(output_dir, filename)
    processed_dict = _process_shard(shard_keys, checkpoint_path, save_dtype)
    try:
        _save_state_dict(
            processed_dict,
            save_path,
            safe_serialization=safe_serialization,
            local_staging_dir=local_staging_dir,
        )
    finally:
        del processed_dict
        gc.collect()
    return shard_idx, filename


@torch.no_grad()
def save_lora_adapter_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    adapter_config_path: Optional[Union[str, os.PathLike]] = None,
    local_staging_dir: Optional[Union[str, os.PathLike]] = None,
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
    _save_state_dict(
        processed_dict,
        save_path,
        safe_serialization=True,
        local_staging_dir=local_staging_dir,
    )
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
            "``adapter_config.json`` (from the matching training run's "
            "``output_dir/checkpoints/global_step_*/lora_ckpt/``) next to it "
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
    local_staging_dir: Optional[Union[str, os.PathLike]] = None,
    num_workers: int = 1,
    aliased_keys: Optional[set[str]] = None,
) -> None:
    """Convert DCP checkpoint to HuggingFace format with shard-by-shard processing (memory-efficient)."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model weights to {output_dir}")
    logger.info(
        f"Format: {'safetensors' if safe_serialization else 'pytorch'}, dtype={save_dtype}, "
        f"shard_size={shard_size}, workers={num_workers}"
    )
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")

    # Plan shards from metadata
    logger.info("Analyzing DCP metadata and planning shards...")
    shards, total_size, all_dcp_keys = _get_sharding_plan(
        checkpoint_path, shard_size, save_dtype, drop_hf_keys=aliased_keys
    )

    logger.info(f"Found {len(all_dcp_keys)} model tensors, total size: ~{total_size / 1e9:.2f}GB")
    logger.info(f"Split into {len(shards)} shards")

    if len(shards) == 0:
        logger.warning("No model weights found! Check if checkpoint path is correct and contains 'model.' keys.")
        return

    num_shards = len(shards)
    worker_count = min(num_workers, num_shards)
    filenames: dict[int, str] = {}
    if worker_count == 1:
        for shard_idx, shard_keys in enumerate(shards):
            logger.info(f"Processing shard {shard_idx + 1}/{num_shards} ({len(shard_keys)} tensors)")
            completed_idx, filename = _process_and_save_shard(
                shard_idx,
                num_shards,
                shard_keys,
                os.fspath(checkpoint_path),
                os.fspath(output_dir),
                save_dtype,
                safe_serialization,
                os.fspath(local_staging_dir) if local_staging_dir is not None else None,
            )
            filenames[completed_idx] = filename
    else:
        # Torch may have initialized an OpenMP thread pool while inspecting the
        # checkpoint. Forking after that point inherits libgomp state without its
        # worker threads and can deadlock on the first tensor copy in the child.
        # Spawn gives every conversion worker a clean runtime instead.
        process_context = multiprocessing.get_context("spawn")
        logger.info(
            f"Processing {num_shards} shards with {worker_count} worker processes "
            f"(start_method={process_context.get_start_method()})"
        )
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=process_context) as executor:
            futures = {
                executor.submit(
                    _process_and_save_shard,
                    shard_idx,
                    num_shards,
                    shard_keys,
                    os.fspath(checkpoint_path),
                    os.fspath(output_dir),
                    save_dtype,
                    safe_serialization,
                    os.fspath(local_staging_dir) if local_staging_dir is not None else None,
                ): shard_idx
                for shard_idx, shard_keys in enumerate(shards)
            }
            for future in as_completed(futures):
                completed_idx, filename = future.result()
                filenames[completed_idx] = filename
                logger.info(f"Completed shard {completed_idx + 1}/{num_shards}: {filename}")

    # Construct the index deterministically regardless of worker completion order.
    weight_map = OrderedDict()
    for shard_idx, shard_keys in enumerate(shards):
        filename = filenames[shard_idx]
        for hf_key in shard_keys:
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


def merge_to_hf_pt(
    load_dir: str,
    save_path: str,
    model_assets_dir: Optional[str] = None,
    shard_size: int = 2_000_000_000,
    local_staging_dir: Optional[str] = None,
    num_workers: int = 1,
) -> None:
    """Main conversion function: load DCP from load_dir and save HF format to save_path."""
    model_assets = None
    aliased_keys = None
    if model_assets_dir is not None:
        aliased_keys = _aliased_hf_keys(model_assets_dir)
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
            processor_assets = ("preprocessor_config.json", "processor_config.json", "image_processor_config.json")
            if any(os.path.isfile(os.path.join(model_assets_dir, name)) for name in processor_assets):
                raise RuntimeError(
                    f"Found processor assets in {model_assets_dir}, but failed to load the processor: {e}"
                ) from e
            logger.warning(f"Failed to load AutoProcessor: {e}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_assets_dir, trust_remote_code=True)
                model_assets.append(tokenizer)
            except Exception as tokenizer_error:
                tokenizer_assets = ("tokenizer.json", "tokenizer_config.json")
                if any(os.path.isfile(os.path.join(model_assets_dir, name)) for name in tokenizer_assets):
                    raise RuntimeError(
                        f"Found tokenizer assets in {model_assets_dir}, but failed to load the tokenizer: {tokenizer_error}"
                    ) from tokenizer_error
                logger.warning(f"Failed to load AutoTokenizer: {tokenizer_error}")

        generation_config_path = os.path.join(model_assets_dir, "generation_config.json")
        if os.path.isfile(generation_config_path):
            try:
                generation_config = GenerationConfig.from_pretrained(model_assets_dir)
                model_assets.append(generation_config)
            except Exception as e:
                raise RuntimeError(
                    f"Found {generation_config_path}, but failed to load the generation config: {e}"
                ) from e

        if not model_assets:
            model_assets = None

    save_model_weights(
        save_path,
        load_dir,
        shard_size=shard_size,
        model_assets=model_assets,
        local_staging_dir=local_staging_dir,
        num_workers=num_workers,
        aliased_keys=aliased_keys,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Merge DCP checkpoint to HuggingFace format (streaming optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        required=True,
        help=(
            "A checkpoint step directory (checkpoints/global_step_N) or a DCP directory. "
            "Given a step directory, the weights are read from its model/ckpt subdirectory; "
            "the optimizer state sitting beside them is never read."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for HuggingFace format checkpoint (default: <load-dir>/hf_ckpt)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes used to load and write independent HF shards (default: 1)",
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
        "--local-staging-dir",
        type=str,
        default=None,
        help=(
            "Write each shard to this local directory before streaming it to --save-dir. "
            "Use this when the destination filesystem does not support safetensors mmap writes."
        ),
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
            "(usually under <output_dir>/checkpoints/global_step_*/lora_ckpt/adapter_config.json). "
            "Only used in 'lora' mode; "
            "copied next to adapter_model.safetensors so the adapter is loadable as-is."
        ),
    )
    args = parser.parse_args()

    # A step directory is not itself a DCP directory: it holds model/ckpt and
    # model/optimizer. Resolve to the weights, but keep the export next to the
    # rest of the step rather than burying it under model/.
    save_root = args.load_dir
    load_dir = weights_dir(args.load_dir)
    if not os.path.exists(os.path.join(load_dir, ".metadata")):
        load_dir = args.load_dir
    save_dir = os.path.join(save_root, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir
    shard_size = args.shard_size

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
            local_staging_dir=args.local_staging_dir,
        )
    else:
        if args.adapter_config_path is not None:
            logger.warning("--adapter-config-path is only used in 'lora' mode; ignoring.")
        merge_to_hf_pt(
            load_dir,
            save_dir,
            model_assets_dir,
            shard_size=shard_size,
            local_staging_dir=args.local_staging_dir,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
