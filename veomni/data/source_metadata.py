# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from collections.abc import Sequence
from typing import Any, TypeAlias, cast


SOURCE_METADATA_VERSION = 1
SOURCE_METADATA_KEY = "_veomni_source_metadata"
PACKED_SOURCE_METADATA_KEY = "_veomni_packed_source_metadata"
PACKED_SOURCE_COORDINATE_SPACE = "packed_pre_sp"
SOURCE_DIAGNOSTIC_FIELDS = ("row_id", "sample_id", "part_index")
LEGACY_SOURCE_NAME_KEYS = ("channel_name", "source_name", "dataset_name", "data_name")
INTERNAL_SOURCE_METADATA_KEYS = frozenset(
    {
        SOURCE_METADATA_KEY,
        PACKED_SOURCE_METADATA_KEY,
        "channel_id",
        "ds_idx",
        "dataset_id",
        "source_id",
        *LEGACY_SOURCE_NAME_KEYS,
        "cur_token_num",
        *SOURCE_DIAGNOSTIC_FIELDS,
    }
)
SOURCE_METADATA_ENVELOPE_KEYS = frozenset(
    {
        SOURCE_METADATA_KEY,
        PACKED_SOURCE_METADATA_KEY,
    }
)
LEGACY_SOURCE_METADATA_KEYS = INTERNAL_SOURCE_METADATA_KEYS - SOURCE_METADATA_ENVELOPE_KEYS

SourceId: TypeAlias = int | str


def validate_source_id(source_id: object) -> SourceId:
    """Return a source ID after enforcing its stable primitive representation."""
    if type(source_id) not in (int, str):
        raise ValueError("source_ids must contain only int or str values; bool is not supported")
    return cast(SourceId, source_id)


def validate_source_ids(source_ids: Sequence[object], *, expected_count: int) -> list[SourceId]:
    """Validate source IDs as unique typed primitives aligned with the sources."""
    if len(source_ids) != expected_count:
        raise ValueError("source_ids length must match datasets length")
    validated = [validate_source_id(source_id) for source_id in source_ids]
    if len(set(validated)) != expected_count:
        raise ValueError("source_ids must be unique")
    return validated


def _validate_diagnostic_field(name: str, value: object) -> int | str:
    if name == "part_index":
        if type(value) is not int:
            raise ValueError("part_index must be an int")
    elif type(value) not in (int, str):
        raise ValueError(f"{name} must be an int or str")
    return cast(int | str, value)


def make_source_metadata(
    source_id: SourceId,
    source_name: str | None,
    *,
    row_id: int | str | None = None,
    sample_id: int | str | None = None,
    part_index: int | None = None,
) -> dict[str, Any]:
    """Build the versioned metadata envelope attached to a transformed sample."""
    source_id = validate_source_id(source_id)
    if source_name is not None and not isinstance(source_name, str):
        raise ValueError("source_name must be a str or None")
    metadata = {
        "schema_version": SOURCE_METADATA_VERSION,
        "source_id": source_id,
    }
    if source_name is not None:
        metadata["source_name"] = source_name
    diagnostics = {
        "row_id": row_id,
        "sample_id": sample_id,
        "part_index": part_index,
    }
    for name, value in diagnostics.items():
        if value is not None:
            metadata[name] = _validate_diagnostic_field(name, value)
    return metadata


def normalize_source_metadata(metadata: object) -> dict[str, Any]:
    """Validate a raw metadata envelope and copy only its public fields."""
    if not isinstance(metadata, dict):
        raise ValueError(f"{SOURCE_METADATA_KEY} must be a dict")
    if metadata.get("schema_version") != SOURCE_METADATA_VERSION:
        raise ValueError(f"{SOURCE_METADATA_KEY}.schema_version must be {SOURCE_METADATA_VERSION}")
    if "source_id" not in metadata:
        raise ValueError(f"{SOURCE_METADATA_KEY}.source_id is required")
    source_name = metadata.get("source_name")
    if source_name is not None and not isinstance(source_name, str):
        raise ValueError(f"{SOURCE_METADATA_KEY}.source_name must be a str")
    normalized = make_source_metadata(metadata["source_id"], source_name)
    for name in SOURCE_DIAGNOSTIC_FIELDS:
        if name in metadata:
            normalized[name] = _validate_diagnostic_field(name, metadata[name])
    return normalized


def _normalize_packed_segment(
    segment: object,
    *,
    expected_segment_index: int,
    expected_token_start: int,
) -> dict[str, Any]:
    if not isinstance(segment, dict):
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.segments entries must be dicts")
    source_id = validate_source_id(segment.get("source_id"))
    source_name = segment.get("source_name")
    if source_name is not None and not isinstance(source_name, str):
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.segments source_name must be a str")
    coordinate_names = (
        "segment_index",
        "sample_index",
        "subsegment_index",
        "token_start",
        "token_length",
    )
    raw_integer_fields = {name: segment.get(name) for name in coordinate_names}
    if any(type(value) is not int for value in raw_integer_fields.values()):
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.segments coordinate fields must be ints")
    integer_fields = cast(dict[str, int], raw_integer_fields)
    if integer_fields["segment_index"] != expected_segment_index:
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.segments segment_index must be contiguous")
    if integer_fields["sample_index"] < 0 or integer_fields["subsegment_index"] < 0:
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.segments sample indices must be non-negative")
    if integer_fields["token_start"] != expected_token_start or integer_fields["token_length"] <= 0:
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.segments token coordinates must be contiguous and non-empty")

    normalized: dict[str, Any] = {
        "source_id": source_id,
        **({"source_name": source_name} if source_name is not None else {}),
    }
    for name in SOURCE_DIAGNOSTIC_FIELDS:
        if name in segment:
            normalized[name] = _validate_diagnostic_field(name, segment[name])
    normalized.update(integer_fields)
    return normalized


def make_packed_source_metadata(segments: list[dict], *, valid_token_count: int) -> dict[str, Any]:
    """Build a packed, pre-sequence-parallel metadata envelope."""
    if type(valid_token_count) is not int or valid_token_count < 0:
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.valid_token_count must be a non-negative int")
    normalized_segments = []
    expected_token_start = 0
    current_sample_index = 0
    next_subsegment_index = 0
    for segment_index, segment in enumerate(segments):
        normalized = _normalize_packed_segment(
            segment,
            expected_segment_index=segment_index,
            expected_token_start=expected_token_start,
        )
        sample_index = normalized["sample_index"]
        subsegment_index = normalized["subsegment_index"]
        if segment_index == 0:
            valid_sample_order = sample_index == 0 and subsegment_index == 0
        elif sample_index == current_sample_index:
            valid_sample_order = subsegment_index == next_subsegment_index
        else:
            valid_sample_order = sample_index == current_sample_index + 1 and subsegment_index == 0
        if not valid_sample_order:
            raise ValueError(
                f"{PACKED_SOURCE_METADATA_KEY}.segments sample_index/subsegment_index must start at 0 "
                "and advance contiguously"
            )
        if sample_index != current_sample_index:
            current_sample_index = sample_index
            next_subsegment_index = 0
        next_subsegment_index += 1
        normalized_segments.append(normalized)
        expected_token_start += normalized["token_length"]
    if expected_token_start != valid_token_count:
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.valid_token_count must equal the covered segment token count")
    source_name_presence = {"source_name" in segment for segment in normalized_segments}
    if len(source_name_presence) > 1:
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.segments source_name must be present on all segments or none")
    source_names_by_typed_id: dict[tuple[type, SourceId], str] = {}
    for segment in normalized_segments:
        if "source_name" not in segment:
            continue
        typed_source_id = (type(segment["source_id"]), segment["source_id"])
        source_name = segment["source_name"]
        previous_name = source_names_by_typed_id.setdefault(typed_source_id, source_name)
        if previous_name != source_name:
            raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.segments with the same source_id must use one source_name")
    return {
        "schema_version": SOURCE_METADATA_VERSION,
        "coordinate_space": PACKED_SOURCE_COORDINATE_SPACE,
        "valid_token_count": valid_token_count,
        "segments": normalized_segments,
    }


def normalize_packed_source_metadata(metadata: object) -> dict[str, Any]:
    """Validate the packed envelope without accepting a legacy fallback."""
    if not isinstance(metadata, dict):
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY} must be a dict")
    if metadata.get("schema_version") != SOURCE_METADATA_VERSION:
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.schema_version must be {SOURCE_METADATA_VERSION}")
    if metadata.get("coordinate_space") != PACKED_SOURCE_COORDINATE_SPACE:
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.coordinate_space must be {PACKED_SOURCE_COORDINATE_SPACE!r}")
    segments = metadata.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.segments must be a list")
    valid_token_count = metadata.get("valid_token_count")
    if type(valid_token_count) is not int:
        raise ValueError(f"{PACKED_SOURCE_METADATA_KEY}.valid_token_count must be a non-negative int")
    return make_packed_source_metadata(
        segments,
        valid_token_count=valid_token_count,
    )


def attach_source_metadata(
    sample: Any,
    *,
    source_id: SourceId,
    source_name: str | None,
    ds_idx: int | None = None,
) -> Any:
    """Attach canonical and compatibility source fields after data transforms."""
    source_id = validate_source_id(source_id)
    if isinstance(sample, list):
        for part_index, item in enumerate(sample):
            if isinstance(item, dict):
                existing_metadata = item.get(SOURCE_METADATA_KEY)
                has_part_index = "part_index" in item or (
                    isinstance(existing_metadata, dict) and "part_index" in existing_metadata
                )
                if not has_part_index:
                    item["part_index"] = part_index
            attach_source_metadata(
                item,
                source_id=source_id,
                source_name=source_name,
                ds_idx=ds_idx,
            )
        return sample
    if not isinstance(sample, dict):
        return sample

    existing_metadata = sample.get(SOURCE_METADATA_KEY)
    diagnostics = {}
    if isinstance(existing_metadata, dict):
        diagnostics.update(
            {name: existing_metadata[name] for name in SOURCE_DIAGNOSTIC_FIELDS if name in existing_metadata}
        )
    diagnostics.update({name: sample[name] for name in SOURCE_DIAGNOSTIC_FIELDS if name in sample})

    if ds_idx is not None:
        sample["ds_idx"] = ds_idx
    sample["source_id"] = source_id
    for key in LEGACY_SOURCE_NAME_KEYS:
        sample.pop(key, None)
    if source_name is not None:
        sample["source_name"] = source_name
    sample[SOURCE_METADATA_KEY] = make_source_metadata(source_id, source_name, **diagnostics)
    return sample


def strip_source_metadata(model_inputs: dict[str, Any], *, strip_legacy: bool = False) -> None:
    """Remove source-observation fields before invoking a model.

    The namespaced canonical envelopes always belong to VeOmni. Generic keys
    such as ``source_id`` and ``sample_id`` are removed only when an envelope
    establishes ownership, or when the caller explicitly owns the legacy
    multisource contract. This preserves unrelated custom-model inputs.
    """
    owns_legacy_fields = strip_legacy or any(key in model_inputs for key in SOURCE_METADATA_ENVELOPE_KEYS)
    for key in SOURCE_METADATA_ENVELOPE_KEYS:
        model_inputs.pop(key, None)
    if owns_legacy_fields:
        for key in LEGACY_SOURCE_METADATA_KEYS:
            model_inputs.pop(key, None)
