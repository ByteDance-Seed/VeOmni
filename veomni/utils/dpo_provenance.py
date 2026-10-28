# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Private, CPU-resident evidence for opt-in DPO packed-segment accounting."""

import hashlib
import inspect
import json
import marshal
import types
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import torch


DPO_FEATURE_KEY = "_veomni_dpo_feature_segments"
DPO_PROOF_KEY = "_veomni_dpo_packed_proof"
DPO_APPROVED_KEY = "_veomni_dpo_approved_proof"
DPO_POSTCHECK_KEY = "_veomni_dpo_post_callback_proof"
DPO_CALLBACK_METADATA_KEYS = frozenset(
    {"ds_idx", "source_name", "cur_token_num", "_veomni_channel_loss_source_provenance"}
)

_DPO_PRODUCER_METHODS = (
    "encode",
    "__call__",
    "apply_chat_template",
    "encode_messages",
    "encode_plus",
    "_encode_plus",
    "_batch_encode_plus",
    "_call_one",
    "_get_padding_truncation_strategies",
    "set_truncation_and_padding",
    "_convert_encoding",
    "_switch_to_input_mode",
    "_update_prefix_labels",
    "_tokenize_and_remap",
    "_tokenize",
    "prepare_for_tokenization",
)


@dataclass(frozen=True)
class DPOFeatureSegments:
    """Lengths from a producer that declares an unpadded-output contract."""

    chosen: int
    rejected: int
    producer_contract_version: int = 0
    feature_digest: str = ""


def _stable_tokenizer_backend(serialized: str) -> str:
    """Normalize the small runtime fields without decoding a large vocabulary."""
    decoder = json.JSONDecoder()
    try:
        prefix = '{"version":'
        if not serialized.startswith(prefix):
            raise ValueError("unexpected tokenizer serialization")
        _, end_version = decoder.raw_decode(serialized, len(prefix))
        truncation_key = ',"truncation":'
        if not serialized.startswith(truncation_key, end_version):
            raise ValueError("unexpected tokenizer truncation field")
        truncation_start = end_version + len(truncation_key)
        _, truncation_end = decoder.raw_decode(serialized, truncation_start)
        padding_key = ',"padding":'
        if not serialized.startswith(padding_key, truncation_end):
            raise ValueError("unexpected tokenizer padding field")
        padding_start = truncation_end + len(padding_key)
        _, padding_end = decoder.raw_decode(serialized, padding_start)
        return (
            serialized[:truncation_start]
            + "null"
            + serialized[truncation_end:padding_start]
            + "null"
            + serialized[padding_end:]
        )
    except (ValueError, TypeError):
        # Keep compatibility with a future tokenizer serialization order.
        backend_config = json.loads(serialized)
        backend_config.pop("padding", None)
        backend_config.pop("truncation", None)
        return json.dumps(backend_config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _live_producer_methods(producer: Any) -> dict[str, str]:
    """Identify the live entry points, including replacements made on a class."""

    def stable_code(code: types.CodeType) -> types.CodeType:
        constants = tuple(stable_code(item) if isinstance(item, types.CodeType) else item for item in code.co_consts)
        return code.replace(co_filename="", co_firstlineno=0, co_consts=constants)

    methods = {}
    for name in _DPO_PRODUCER_METHODS:
        implementation = inspect.getattr_static(type(producer), name, None)
        if implementation is None:
            continue
        implementation = getattr(implementation, "__func__", implementation)
        if not callable(implementation):
            raise ValueError(f"DPO producer has a non-callable {name} descriptor")
        code = getattr(implementation, "__code__", None)
        data = f"{getattr(implementation, '__module__', None)}.{getattr(implementation, '__qualname__', None)}".encode(
            "utf-8"
        )
        if code is not None:
            data += marshal.dumps(stable_code(code))
            data += repr((implementation.__defaults__, implementation.__kwdefaults__)).encode("utf-8")
        methods[name] = hashlib.sha256(data).hexdigest()
    return methods


def dpo_tokenizer_fingerprint(tokenizer: Any) -> str:
    """Bind an audited fast-tokenizer contract to its backend and chat settings.

    This digest identifies a configuration; it does not prove that the
    configuration actually emits unpadded sequences. The operator must audit
    that behavior before supplying the digest as an explicit attestation.
    """
    for name in _DPO_PRODUCER_METHODS:
        if name in getattr(tokenizer, "__dict__", {}):
            raise ValueError(f"DPO cannot attest an instance-overridden tokenizer {name}")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        raise ValueError("DPO tokenizer fingerprint requires a serializable fast-tokenizer backend")
    # Transformers resets these transient runtime settings on each encode.
    # The DPO producer requests single-sequence encoding with no padding.
    backend_config = _stable_tokenizer_backend(backend.to_str())
    identity = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "live_methods": _live_producer_methods(tokenizer),
        "transformers_version": version("transformers"),
        "tokenizers_version": version("tokenizers"),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "init_kwargs": getattr(tokenizer, "init_kwargs", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "split_special_tokens": getattr(tokenizer, "split_special_tokens", None),
        "add_bos_token": getattr(tokenizer, "add_bos_token", None),
        "add_eos_token": getattr(tokenizer, "add_eos_token", None),
        "add_prefix_space": getattr(tokenizer, "add_prefix_space", None),
        "legacy": getattr(tokenizer, "legacy", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "model_input_names": getattr(tokenizer, "model_input_names", None),
    }
    data = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    digest = hashlib.sha256(data)
    digest.update(b"\0")
    digest.update(backend_config.encode("utf-8"))
    return digest.hexdigest()


def dpo_declared_producer_identity(producer: Any, kind: str) -> str:
    """Bind an explicit unpadded-output contract to its producer and configuration."""
    cls = type(producer)
    if cls.__dict__.get("dpo_unpadded_output_contract") is not True:
        raise ValueError(f"DPO {kind} must declare dpo_unpadded_output_contract=True")
    for method in _DPO_PRODUCER_METHODS:
        if method in getattr(producer, "__dict__", {}):
            raise ValueError(f"DPO cannot attest an instance-overridden {kind} {method}")
    contract_id = getattr(producer, "dpo_unpadded_output_contract_id", None)
    if callable(contract_id):
        contract_id = contract_id()
    if type(contract_id) is not str or not contract_id:
        raise ValueError(f"DPO {kind} must declare a stable dpo_unpadded_output_contract_id")
    identity = {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "contract_id": contract_id,
        "live_methods": _live_producer_methods(producer),
    }
    if cls.__module__ == "veomni.data.chat_template":
        # Built-in template semantics include constructor state and inherited methods.
        identity["state"] = {k: v for k, v in vars(producer).items() if k != "tokenizer"}
        identity["code"] = [
            inspect.getsource(base) for base in cls.__mro__ if base.__module__ == "veomni.data.chat_template"
        ]
    data = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class DPOPackedProof:
    """A checked packed layout; feature counts use the loader's token convention."""

    segment_lengths: tuple[int, ...]
    feature_token_counts: tuple[int, ...]
    collator_tail: int
    source_ids: tuple[str | int | None, ...]
    input_digest: str
    packed_digest: str
    forward_digest: str
    global_digest: str
    source_indices: tuple[int, ...] = ()
    source_keys: tuple[str, ...] = ()
    callback_metadata_digests: tuple[tuple[str, str], ...] = ()
    layout_sp_size: int = 1
    layout_sp_rank: int = 0
    producer_contract_version: int = 0


@dataclass(frozen=True)
class DPOInvalidProof:
    """Inconsistent evidence; the trainer must coordinate rejection across ranks."""

    reason: str


def packed_batch_digest(
    batch: dict[str, Any],
    source_keys: tuple[str, ...] = (),
    *,
    after_callbacks: bool = False,
    global_only: bool = False,
    feature: bool = False,
    include_keys: tuple[str, ...] | None = None,
) -> str:
    """Fingerprint the local CPU batch so later mutation cannot reuse its evidence."""
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("DPO packed provenance tensors must be on CPU")
            digest.update(f"tensor:{value.dtype}:{tuple(value.shape)}:".encode())
            digest.update(memoryview(value.contiguous().reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise ValueError("DPO packed provenance has non-string keys")
            digest.update(f"dict:{type(value).__module__}.{type(value).__qualname__}:".encode())
            for key in sorted(value):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(f"{type(value).__name__}:{len(value)}:".encode())
            for item in value:
                update(item)
        elif value is None or type(value) in (bool, int, float, str):
            digest.update(f"{type(value).__name__}:{value!r};".encode())
        else:
            raise ValueError("DPO packed provenance has an unsupported value")

    if global_only:
        for key in (
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "linear_attn_cu_seq_lens_q",
            "packed_sequence_slices",
            "max_length_q",
            "max_length_k",
            "attention_mask_is_all_ones",
            "tail_padding_length",
        ):
            update(key)
            update(batch.get(key))
        return digest.hexdigest()

    required = ("input_ids", "labels", "position_ids", "attention_mask")
    if not feature:
        required += ("cu_seq_lens_q",)
    for key in required:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"DPO packed {key} must be a CPU tensor")
    if any(type(key) is not str for key in batch):
        raise ValueError("DPO packed batch has non-string keys")
    update(source_keys)
    ignored = {DPO_PROOF_KEY, DPO_APPROVED_KEY, DPO_POSTCHECK_KEY}
    if after_callbacks:
        ignored.update(("ds_idx", "source_name", "cur_token_num", "_veomni_channel_loss_source_provenance"))
        ignored.update(source_keys)
    for key in sorted(batch):
        if key in ignored or (include_keys is not None and key not in include_keys):
            continue
        update(key)
        update(batch[key])
    return digest.hexdigest()


def callback_metadata_digests(batch: dict[str, Any], source_keys: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Fingerprint metadata that standard callbacks may remove, but not add or change."""
    return tuple(
        (key, packed_batch_digest(batch, source_keys, include_keys=(key,)))
        for key in sorted(DPO_CALLBACK_METADATA_KEYS | set(source_keys))
        if key in batch
    )
