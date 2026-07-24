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

"""Model-local segment plan and compiler for the initial T2I reference path."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch


SINGLE_GEN_T2I_V1 = "single_gen_t2i_v1"
_STATIC_CONTROL_TOKEN_COUNT = 3


class UnsupportedSegmentPattern(ValueError):
    """Raised when a segment plan is outside the frozen reference capability."""


class SegmentRegionSpec(TypedDict):
    region_role: Literal["content", "control", "payload", "terminator"]
    token_span: tuple[int, int]
    embedding_role: Literal["token_embedding", "timestep_embedding", "image_projection"]
    attention_role: Literal["causal_prefix", "full_image_suffix"]
    position_role: Literal["sequence_1d", "image_2d"]
    loss_role: Literal["none", "flow"]


class SegmentSpec(TypedDict):
    segment_id: str
    kind: Literal["text", "gen_image"]
    token_span: tuple[int, int]
    image_id: str | None
    grid_hw: tuple[int, int] | None
    regions: list[SegmentRegionSpec]


class MultimodalSegmentPlan(TypedDict):
    capability: Literal["single_gen_t2i_v1"]
    sample_id: str
    sequence_length: int
    segments: list[SegmentSpec]


@dataclass(frozen=True)
class _ValidatedPlan:
    sequence_length: int
    timestep_position: int
    payload_start: int
    payload_stop: int
    grid_height: int
    grid_width: int


def build_single_gen_t2i_plan(
    *,
    sample_id: str,
    text_token_count: int,
    grid_hw: tuple[int, int],
    image_id: str = "generated_image",
) -> MultimodalSegmentPlan:
    """Build the frozen physical layout without assigning tokenizer-specific IDs."""
    text_token_count = _require_positive_int(text_token_count, "text_token_count")
    grid_height, grid_width = _normalize_grid_hw(grid_hw)

    text_stop = text_token_count
    static_control_stop = text_stop + _STATIC_CONTROL_TOKEN_COUNT
    timestep_stop = static_control_stop + 1
    payload_stop = timestep_stop + grid_height * grid_width
    sequence_length = payload_stop + 1

    return {
        "capability": SINGLE_GEN_T2I_V1,
        "sample_id": sample_id,
        "sequence_length": sequence_length,
        "segments": [
            {
                "segment_id": "text",
                "kind": "text",
                "token_span": (0, text_stop),
                "image_id": None,
                "grid_hw": None,
                "regions": [
                    {
                        "region_role": "content",
                        "token_span": (0, text_stop),
                        "embedding_role": "token_embedding",
                        "attention_role": "causal_prefix",
                        "position_role": "sequence_1d",
                        "loss_role": "none",
                    }
                ],
            },
            {
                "segment_id": "gen_image",
                "kind": "gen_image",
                "token_span": (text_stop, sequence_length),
                "image_id": image_id,
                "grid_hw": (grid_height, grid_width),
                "regions": [
                    {
                        "region_role": "control",
                        "token_span": (text_stop, static_control_stop),
                        "embedding_role": "token_embedding",
                        "attention_role": "causal_prefix",
                        "position_role": "sequence_1d",
                        "loss_role": "none",
                    },
                    {
                        "region_role": "control",
                        "token_span": (static_control_stop, timestep_stop),
                        "embedding_role": "timestep_embedding",
                        "attention_role": "causal_prefix",
                        "position_role": "sequence_1d",
                        "loss_role": "none",
                    },
                    {
                        "region_role": "payload",
                        "token_span": (timestep_stop, payload_stop),
                        "embedding_role": "image_projection",
                        "attention_role": "full_image_suffix",
                        "position_role": "image_2d",
                        "loss_role": "flow",
                    },
                    {
                        "region_role": "terminator",
                        "token_span": (payload_stop, sequence_length),
                        "embedding_role": "token_embedding",
                        "attention_role": "full_image_suffix",
                        "position_role": "sequence_1d",
                        "loss_role": "none",
                    },
                ],
            },
        ],
    }


def validate_single_gen_t2i_plan(plan: Mapping[str, object]) -> None:
    """Validate that a plan exactly matches ``single_gen_t2i_v1``."""
    _validate_plan(plan)


def compile_single_gen_t2i_plans(
    plans: Sequence[Mapping[str, object]],
    *,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Compile unpacked plans into dense reference tensors.

    All samples must have the same sequence length because this milestone does
    not implement packing or sequence-parallel padding.
    """
    if isinstance(plans, Mapping) or not isinstance(plans, Sequence) or not plans:
        raise TypeError("plans must be a non-empty sequence of segment plans.")

    validated = [_validate_plan(plan) for plan in plans]
    sequence_length = validated[0].sequence_length
    if any(item.sequence_length != sequence_length for item in validated[1:]):
        raise UnsupportedSegmentPattern("The unpacked reference batch requires one shared sequence length.")
    shared_grid = (validated[0].grid_height, validated[0].grid_width)
    if any((item.grid_height, item.grid_width) != shared_grid for item in validated[1:]):
        raise UnsupportedSegmentPattern("The unpacked reference batch requires one shared image grid.")

    batch_size = len(validated)
    image_token_count = shared_grid[0] * shared_grid[1]
    position_ids = torch.empty((batch_size, 2, sequence_length), dtype=torch.long, device=device)
    dense_attention_mask = torch.empty(
        (batch_size, sequence_length, sequence_length),
        dtype=torch.bool,
        device=device,
    )
    timestep_sample_index = torch.full(
        (batch_size, sequence_length),
        -1,
        dtype=torch.long,
        device=device,
    )
    image_output_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
        device=device,
    )
    timestep_positions = torch.empty((batch_size,), dtype=torch.long, device=device)
    image_payload_indices = torch.empty(
        (batch_size, image_token_count),
        dtype=torch.long,
        device=device,
    )

    for sample_index, item in enumerate(validated):
        diagonal_positions = torch.arange(sequence_length, dtype=torch.long, device=device)
        position_ids[sample_index, 0] = diagonal_positions
        position_ids[sample_index, 1] = diagonal_positions

        payload_length = item.grid_height * item.grid_width
        beta_y = item.payload_start + (payload_length - item.grid_height) / 2
        beta_x = item.payload_start + (payload_length - item.grid_width) / 2
        y_coordinates = (torch.arange(item.grid_height, dtype=torch.float32, device=device) + beta_y).to(torch.long)
        x_coordinates = (torch.arange(item.grid_width, dtype=torch.float32, device=device) + beta_x).to(torch.long)
        position_ids[sample_index, 0, item.payload_start : item.payload_stop] = y_coordinates.repeat_interleave(
            item.grid_width
        )
        position_ids[sample_index, 1, item.payload_start : item.payload_stop] = x_coordinates.repeat(item.grid_height)

        allowed = torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device).tril()
        allowed[item.payload_start :, item.payload_start :] = True
        dense_attention_mask[sample_index] = allowed
        timestep_sample_index[sample_index, item.timestep_position] = sample_index
        image_output_mask[sample_index, item.payload_start : item.payload_stop] = True
        timestep_positions[sample_index] = item.timestep_position
        image_payload_indices[sample_index] = torch.arange(
            item.payload_start,
            item.payload_stop,
            dtype=torch.long,
            device=device,
        )

    return {
        "capability": SINGLE_GEN_T2I_V1,
        "sequence_length": sequence_length,
        "position_ids": position_ids,
        "dense_attention_mask": dense_attention_mask,
        "timestep_sample_index": timestep_sample_index,
        "timestep_positions": timestep_positions,
        "image_output_mask": image_output_mask,
        "image_payload_indices": image_payload_indices,
        "causal_prefix_lengths": tuple(item.payload_start for item in validated),
        "image_suffix_lengths": tuple(item.sequence_length - item.payload_start for item in validated),
        "grid_hw": tuple((item.grid_height, item.grid_width) for item in validated),
    }


def compile_single_gen_t2i_packed(
    plans: Sequence[Mapping[str, object]],
    *,
    device: torch.device | str | None = None,
    pad_to_multiple_of: int = 1,
) -> dict[str, object]:
    """Compile heterogeneous plans into a packed, padding-free varlen layout.

    Unlike :func:`compile_single_gen_t2i_plans` (the dense oracle path, which
    requires one shared sequence length and grid), this produces the flattened
    ``[1, T_total]`` metadata that drives the two-call varlen GCA fast path:

    * a causal-prefix call ``FA(Q[:P], K[:P], V[:P], causal=True)`` per sample,
    * an image-suffix call ``FA(Q[P:P+I], K[:P+I], V[:P+I], causal=False)``.

    Samples are laid out contiguously, so sample ``j`` occupies packed-global
    positions ``[sample_start_j, sample_start_j + sequence_length_j)``. All
    ``cu_seqlens`` and gather/scatter indices are computed on the logical
    (padding-free) sequence; ``pad_to_multiple_of`` only extends the trailing
    tensors so a downstream sequence-parallel slice divides evenly. The padded
    tail is never referenced by any index and is excluded from every length.
    """
    if isinstance(plans, Mapping) or not isinstance(plans, Sequence) or not plans:
        raise TypeError("plans must be a non-empty sequence of segment plans.")
    if isinstance(pad_to_multiple_of, bool) or not isinstance(pad_to_multiple_of, int) or pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be a positive integer.")

    validated = [_validate_plan(plan) for plan in plans]
    num_samples = len(validated)

    sample_lengths = [item.sequence_length for item in validated]
    prefix_lengths = [item.payload_start for item in validated]
    image_suffix_lengths = [item.sequence_length - item.payload_start for item in validated]
    grids = [(item.grid_height, item.grid_width) for item in validated]
    image_token_counts = [height * width for height, width in grids]

    sample_starts = [0]
    for length in sample_lengths:
        sample_starts.append(sample_starts[-1] + length)
    logical_length = sample_starts[-1]
    padded_length = ((logical_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    position_ids = torch.zeros((1, 2, padded_length), dtype=torch.long, device=device)
    timestep_sample_index = torch.full((1, padded_length), -1, dtype=torch.long, device=device)
    image_output_mask = torch.zeros((1, padded_length), dtype=torch.bool, device=device)
    timestep_positions = torch.empty((num_samples,), dtype=torch.long, device=device)

    prefix_index_blocks: list[torch.Tensor] = []
    image_suffix_index_blocks: list[torch.Tensor] = []
    image_payload_index_blocks: list[torch.Tensor] = []

    for sample_index, item in enumerate(validated):
        sample_start = sample_starts[sample_index]
        sample_stop = sample_starts[sample_index + 1]

        # Sample-local coordinates: the packed layout never adds sample_start to
        # the positions, so every valid sample restarts its 2D RoPE grid at zero.
        diagonal_positions = torch.arange(item.sequence_length, dtype=torch.long, device=device)
        position_ids[0, 0, sample_start:sample_stop] = diagonal_positions
        position_ids[0, 1, sample_start:sample_stop] = diagonal_positions

        payload_length = item.grid_height * item.grid_width
        beta_y = item.payload_start + (payload_length - item.grid_height) / 2
        beta_x = item.payload_start + (payload_length - item.grid_width) / 2
        y_coordinates = (torch.arange(item.grid_height, dtype=torch.float32, device=device) + beta_y).to(torch.long)
        x_coordinates = (torch.arange(item.grid_width, dtype=torch.float32, device=device) + beta_x).to(torch.long)
        payload_global_start = sample_start + item.payload_start
        payload_global_stop = sample_start + item.payload_stop
        position_ids[0, 0, payload_global_start:payload_global_stop] = y_coordinates.repeat_interleave(item.grid_width)
        position_ids[0, 1, payload_global_start:payload_global_stop] = x_coordinates.repeat(item.grid_height)

        timestep_global_position = sample_start + item.timestep_position
        timestep_sample_index[0, timestep_global_position] = sample_index
        timestep_positions[sample_index] = timestep_global_position
        image_output_mask[0, payload_global_start:payload_global_stop] = True

        prefix_index_blocks.append(
            torch.arange(sample_start, sample_start + item.payload_start, dtype=torch.long, device=device)
        )
        image_suffix_index_blocks.append(
            torch.arange(payload_global_start, sample_stop, dtype=torch.long, device=device)
        )
        image_payload_index_blocks.append(
            torch.arange(payload_global_start, payload_global_stop, dtype=torch.long, device=device)
        )

    prefix_gather_index = torch.cat(prefix_index_blocks)
    image_suffix_gather_index = torch.cat(image_suffix_index_blocks)
    image_payload_indices = torch.cat(image_payload_index_blocks).unsqueeze(0)

    return {
        "capability": SINGLE_GEN_T2I_V1,
        "layout": "packed_varlen",
        "num_samples": num_samples,
        "sequence_length": logical_length,
        "padded_sequence_length": padded_length,
        "position_ids": position_ids,
        "timestep_sample_index": timestep_sample_index,
        "timestep_positions": timestep_positions,
        "image_output_mask": image_output_mask,
        "image_payload_indices": image_payload_indices,
        # gather Q/K/V for the causal-prefix call; the same index scatters its
        # output back to the packed prefix positions.
        "prefix_gather_index": prefix_gather_index,
        # gather Q for the image-suffix call; K/V reuse the full packed sequence.
        "image_suffix_gather_index": image_suffix_gather_index,
        "cu_seqlens_q_prefix": _cu_seqlens(prefix_lengths, device=device),
        "cu_seqlens_k_prefix": _cu_seqlens(prefix_lengths, device=device),
        "cu_seqlens_q_image_suffix": _cu_seqlens(image_suffix_lengths, device=device),
        "cu_seqlens_k_full": _cu_seqlens(sample_lengths, device=device),
        "max_prefix_length": max(prefix_lengths),
        "max_image_suffix_length": max(image_suffix_lengths),
        "max_full_length": max(sample_lengths),
        "causal_prefix_lengths": tuple(prefix_lengths),
        "image_suffix_lengths": tuple(image_suffix_lengths),
        "sample_lengths": tuple(sample_lengths),
        "image_token_counts": tuple(image_token_counts),
        "grid_hw": tuple(grids),
    }


def _cu_seqlens(lengths: Sequence[int], *, device: torch.device | str | None) -> torch.Tensor:
    """Build a padding-free ``[0, l_1, l_1 + l_2, ...]`` cumulative-length tensor."""
    cu = torch.zeros((len(lengths) + 1,), dtype=torch.int32, device=device)
    if lengths:
        cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return cu


def _validate_plan(plan: Mapping[str, object]) -> _ValidatedPlan:
    if not isinstance(plan, Mapping):
        raise TypeError("Each segment plan must be a mapping.")
    if plan.get("capability") != SINGLE_GEN_T2I_V1:
        raise UnsupportedSegmentPattern(f"Unsupported Hunyuan Image 3 capability: {plan.get('capability')!r}.")
    sample_id = plan.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise UnsupportedSegmentPattern("sample_id must be a non-empty string.")
    sequence_length = _require_positive_int(plan.get("sequence_length"), "sequence_length")

    segments = plan.get("segments")
    if not isinstance(segments, list) or len(segments) != 2:
        raise UnsupportedSegmentPattern("single_gen_t2i_v1 requires exactly one text and one gen_image segment.")
    text_segment, image_segment = segments
    if not isinstance(text_segment, Mapping) or not isinstance(image_segment, Mapping):
        raise UnsupportedSegmentPattern("segments must be mappings.")
    if text_segment.get("kind") != "text" or image_segment.get("kind") != "gen_image":
        raise UnsupportedSegmentPattern("single_gen_t2i_v1 requires text followed by gen_image.")
    if text_segment.get("segment_id") == image_segment.get("segment_id"):
        raise UnsupportedSegmentPattern("segment_id values must be unique within a sample.")

    text_span = _normalize_span(text_segment.get("token_span"), "text.token_span")
    image_span = _normalize_span(image_segment.get("token_span"), "gen_image.token_span")
    if text_span[0] != 0 or text_span[1] != image_span[0] or image_span[1] != sequence_length:
        raise UnsupportedSegmentPattern("Top-level segments must partition the full sequence without gaps.")
    if text_segment.get("image_id") is not None or text_segment.get("grid_hw") is not None:
        raise UnsupportedSegmentPattern("The text segment cannot carry image metadata.")
    image_id = image_segment.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        raise UnsupportedSegmentPattern("The gen_image segment requires a non-empty image_id.")
    grid_height, grid_width = _normalize_grid_hw(image_segment.get("grid_hw"))

    text_regions = text_segment.get("regions")
    if not isinstance(text_regions, list) or len(text_regions) != 1:
        raise UnsupportedSegmentPattern("The text segment requires one content region.")
    _validate_region(
        text_regions[0],
        expected_role="content",
        expected_span=text_span,
        embedding_role="token_embedding",
        attention_role="causal_prefix",
        position_role="sequence_1d",
        loss_role="none",
    )

    image_regions = image_segment.get("regions")
    if not isinstance(image_regions, list) or len(image_regions) != 4:
        raise UnsupportedSegmentPattern(
            "The gen_image segment requires static, timestep, payload, and terminator regions."
        )
    static_region, timestep_region, payload_region, terminator_region = image_regions
    static_span = _normalize_span(_mapping_value(static_region, "token_span"), "static_control.token_span")
    timestep_span = _normalize_span(_mapping_value(timestep_region, "token_span"), "timestep.token_span")
    payload_span = _normalize_span(_mapping_value(payload_region, "token_span"), "payload.token_span")
    terminator_span = _normalize_span(_mapping_value(terminator_region, "token_span"), "terminator.token_span")

    if (
        static_span[0] != image_span[0]
        or static_span[1] != timestep_span[0]
        or timestep_span[1] != payload_span[0]
        or payload_span[1] != terminator_span[0]
        or terminator_span[1] != image_span[1]
    ):
        raise UnsupportedSegmentPattern("gen_image regions must partition their parent segment without gaps.")
    if static_span[1] - static_span[0] != _STATIC_CONTROL_TOKEN_COUNT:
        raise UnsupportedSegmentPattern("single_gen_t2i_v1 requires exactly three static control tokens.")
    if timestep_span[1] - timestep_span[0] != 1 or terminator_span[1] - terminator_span[0] != 1:
        raise UnsupportedSegmentPattern("The timestep and terminator regions must each contain one token.")
    if payload_span[1] - payload_span[0] != grid_height * grid_width:
        raise UnsupportedSegmentPattern("The image payload length must equal grid_height * grid_width.")

    _validate_region(
        static_region,
        expected_role="control",
        expected_span=static_span,
        embedding_role="token_embedding",
        attention_role="causal_prefix",
        position_role="sequence_1d",
        loss_role="none",
    )
    _validate_region(
        timestep_region,
        expected_role="control",
        expected_span=timestep_span,
        embedding_role="timestep_embedding",
        attention_role="causal_prefix",
        position_role="sequence_1d",
        loss_role="none",
    )
    _validate_region(
        payload_region,
        expected_role="payload",
        expected_span=payload_span,
        embedding_role="image_projection",
        attention_role="full_image_suffix",
        position_role="image_2d",
        loss_role="flow",
    )
    _validate_region(
        terminator_region,
        expected_role="terminator",
        expected_span=terminator_span,
        embedding_role="token_embedding",
        attention_role="full_image_suffix",
        position_role="sequence_1d",
        loss_role="none",
    )
    return _ValidatedPlan(
        sequence_length=sequence_length,
        timestep_position=timestep_span[0],
        payload_start=payload_span[0],
        payload_stop=payload_span[1],
        grid_height=grid_height,
        grid_width=grid_width,
    )


def _validate_region(
    region: object,
    *,
    expected_role: str,
    expected_span: tuple[int, int],
    embedding_role: str,
    attention_role: str,
    position_role: str,
    loss_role: str,
) -> None:
    if not isinstance(region, Mapping):
        raise UnsupportedSegmentPattern("Segment regions must be mappings.")
    expected = {
        "region_role": expected_role,
        "token_span": expected_span,
        "embedding_role": embedding_role,
        "attention_role": attention_role,
        "position_role": position_role,
        "loss_role": loss_role,
    }
    actual = dict(region)
    actual["token_span"] = _normalize_span(actual.get("token_span"), f"{expected_role}.token_span")
    if any(actual.get(key) != value for key, value in expected.items()):
        raise UnsupportedSegmentPattern(f"Invalid {expected_role} region contract: {actual!r}.")


def _mapping_value(value: object, key: str) -> object:
    if not isinstance(value, Mapping):
        raise UnsupportedSegmentPattern("Segment regions must be mappings.")
    return value.get(key)


def _normalize_span(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise UnsupportedSegmentPattern(f"{name} must be a two-element span.")
    start, stop = value
    if isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int) or not isinstance(stop, int):
        raise UnsupportedSegmentPattern(f"{name} values must be integers.")
    if start < 0 or stop <= start:
        raise UnsupportedSegmentPattern(f"{name} must be a non-empty left-closed, right-open span.")
    return start, stop


def _normalize_grid_hw(value: object) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise UnsupportedSegmentPattern("grid_hw must contain height and width.")
    return _require_positive_int(value[0], "grid_height"), _require_positive_int(value[1], "grid_width")


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedSegmentPattern(f"{name} must be a positive integer.")
    return value


__all__ = [
    "MultimodalSegmentPlan",
    "SINGLE_GEN_T2I_V1",
    "SegmentRegionSpec",
    "SegmentSpec",
    "UnsupportedSegmentPattern",
    "build_single_gen_t2i_plan",
    "compile_single_gen_t2i_packed",
    "compile_single_gen_t2i_plans",
    "validate_single_gen_t2i_plan",
]
