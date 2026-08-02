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

"""Physical sequence layout for the Hunyuan Image 3 ``single_gen_t2i_v1`` capability.

Three things live here because they must never disagree:

* :class:`T2ILayout` — the layout arithmetic. Three numbers in, every offset out.
  Shared by the data transform (which materialises token ids and the loss mask)
  and by the packed compiler, so the offsets are computed exactly once.
* :func:`compile_single_gen_t2i_packed` — the batch-level packed varlen metadata
  consumed by the model forward and the two-call GCA fast path.
* :func:`collate_hunyuan_image_3_metadata` — the ``MainCollator`` hook that runs
  the compiler once per batch, in the main process, after pack + SP padding.

The compiler is batch-level by necessity: ``cu_seqlens``, the packed-global
gather indices and the sequence-parallel padded length are all cross-sample
quantities a per-sample transform cannot know. The hook is module-level so
``functools.partial(...)`` over it stays picklable across DataLoader workers.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ....utils.seqlen_pos_transform_utils import len2culen


SINGLE_GEN_T2I_V1 = "single_gen_t2i_v1"

#: ``<boi> <img_size_*> <img_ratio_*>`` — the fixed control prefix of a generated image.
STATIC_CONTROL_TOKEN_COUNT = 3


class UnsupportedSequenceLayout(ValueError):
    """Raised when a requested layout is outside the frozen reference capability."""


@dataclass(frozen=True)
class T2ILayout:
    """One ``single_gen_t2i_v1`` sample's physical layout.

    ::

        text | <boi> <img_size> <img_ratio> | <timestep> | payload | <eoi>
        <--- causal prefix ------------------------------>|<-- full-attention -->

    Tokens through ``<timestep>`` are causal with 1D positions and no loss. The
    projected latent payload is full-attention with 2D grid positions and flow
    loss. ``<eoi>`` stays in the full-attention suffix with a 1D position and no
    loss; the Base template adds no trailing ``<eos>``.

    Offsets are sample-local. :func:`compile_single_gen_t2i_packed` adds each
    sample's start to turn them into packed-global coordinates.
    """

    text_len: int
    grid_h: int
    grid_w: int

    def __post_init__(self) -> None:
        for name in ("text_len", "grid_h", "grid_w"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise UnsupportedSequenceLayout(f"{name} must be a positive integer, got {value!r}.")

    @property
    def grid_hw(self) -> tuple[int, int]:
        return self.grid_h, self.grid_w

    @property
    def payload_len(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def timestep_pos(self) -> int:
        return self.text_len + STATIC_CONTROL_TOKEN_COUNT

    @property
    def payload_start(self) -> int:
        return self.timestep_pos + 1

    @property
    def payload_stop(self) -> int:
        return self.payload_start + self.payload_len

    @property
    def seq_len(self) -> int:
        return self.payload_stop + 1  # + <eoi>

    def build_input_ids(self, text_ids: Sequence[int], *, im_start_id: int, image_token_id: int, im_end_id: int):
        """Materialise the token ids for one sample.

        Payload and ``<timestep>`` ids are overwritten by the model
        (``patch_embed`` / ``timestep_emb``), so their placeholder id is
        irrelevant; the controls and ``<eoi>`` keep real special-token
        embeddings.
        """
        if len(text_ids) != self.text_len:
            raise UnsupportedSequenceLayout(f"Expected {self.text_len} text ids, got {len(text_ids)}.")
        input_ids = (
            list(text_ids)
            + [im_start_id, image_token_id, image_token_id]  # <boi>, <img_size_*>, <img_ratio_*>
            + [image_token_id]  # <timestep> placeholder (overwritten)
            + [image_token_id] * self.payload_len  # <img> payload (overwritten)
            + [im_end_id]  # <eoi>
        )
        return torch.tensor(input_ids, dtype=torch.long)

    def build_image_output_mask(self) -> torch.Tensor:
        """Per-sample flow-loss token mask; the only ``True`` span is the payload."""
        mask = torch.zeros((self.seq_len,), dtype=torch.bool)
        mask[self.payload_start : self.payload_stop] = True
        return mask


def compile_single_gen_t2i_packed(
    layouts: Sequence[T2ILayout],
    *,
    device: torch.device | str | None = None,
    pad_to_multiple_of: int = 1,
) -> dict[str, object]:
    """Compile heterogeneous layouts into a packed, padding-free varlen batch.

    Produces the flattened ``[1, T_total]`` metadata that drives the two-call
    varlen GCA fast path:

    * a causal-prefix call ``FA(Q[:P], K[:P], V[:P], causal=True)`` per sample,
    * an image-suffix call ``FA(Q[P:P+I], K[:P+I], V[:P+I], causal=False)``.

    Samples are laid out contiguously, so sample ``j`` occupies packed-global
    positions ``[sample_start_j, sample_start_j + seq_len_j)``. All ``cu_seqlens``
    and gather indices are computed on the logical (padding-free) sequence;
    ``pad_to_multiple_of`` only extends the trailing tensors so a downstream
    sequence-parallel slice divides evenly. The padded tail is never referenced
    by any index and is excluded from every length.
    """
    if isinstance(layouts, Mapping) or not isinstance(layouts, Sequence) or not layouts:
        raise TypeError("layouts must be a non-empty sequence of T2ILayout.")
    if any(not isinstance(layout, T2ILayout) for layout in layouts):
        raise TypeError("layouts must contain T2ILayout instances.")
    if isinstance(pad_to_multiple_of, bool) or not isinstance(pad_to_multiple_of, int) or pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be a positive integer.")

    num_samples = len(layouts)
    sample_lengths = [layout.seq_len for layout in layouts]
    prefix_lengths = [layout.payload_start for layout in layouts]
    image_suffix_lengths = [layout.seq_len - layout.payload_start for layout in layouts]

    sample_starts = [0]
    for length in sample_lengths:
        sample_starts.append(sample_starts[-1] + length)
    logical_length = sample_starts[-1]
    padded_length = ((logical_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    position_ids = torch.zeros((1, 2, padded_length), dtype=torch.long, device=device)
    timestep_positions = torch.empty((num_samples,), dtype=torch.long, device=device)

    prefix_index_blocks: list[torch.Tensor] = []
    image_suffix_index_blocks: list[torch.Tensor] = []
    image_payload_index_blocks: list[torch.Tensor] = []

    for sample_index, layout in enumerate(layouts):
        sample_start = sample_starts[sample_index]
        sample_stop = sample_starts[sample_index + 1]

        # Sample-local coordinates: the packed layout never adds sample_start to
        # the positions, so every valid sample restarts its 2D RoPE grid at zero.
        diagonal_positions = torch.arange(layout.seq_len, dtype=torch.long, device=device)
        position_ids[0, 0, sample_start:sample_stop] = diagonal_positions
        position_ids[0, 1, sample_start:sample_stop] = diagonal_positions

        # Official convention: each axis is centred inside the whole image-token
        # span, so the extent is ``payload_len`` (= grid_h * grid_w) on both axes,
        # not the per-axis length. Mixed-parity grids therefore land on
        # half-integer offsets, and the float add then long cast is the
        # truncation every downstream consumer must reproduce.
        beta_y = layout.payload_start + (layout.payload_len - layout.grid_h) / 2
        beta_x = layout.payload_start + (layout.payload_len - layout.grid_w) / 2
        y_coordinates = (torch.arange(layout.grid_h, dtype=torch.float32, device=device) + beta_y).to(torch.long)
        x_coordinates = (torch.arange(layout.grid_w, dtype=torch.float32, device=device) + beta_x).to(torch.long)
        payload_global_start = sample_start + layout.payload_start
        payload_global_stop = sample_start + layout.payload_stop
        position_ids[0, 0, payload_global_start:payload_global_stop] = y_coordinates.repeat_interleave(layout.grid_w)
        position_ids[0, 1, payload_global_start:payload_global_stop] = x_coordinates.repeat(layout.grid_h)

        timestep_positions[sample_index] = sample_start + layout.timestep_pos

        prefix_index_blocks.append(torch.arange(sample_start, payload_global_start, dtype=torch.long, device=device))
        image_suffix_index_blocks.append(
            torch.arange(payload_global_start, sample_stop, dtype=torch.long, device=device)
        )
        image_payload_index_blocks.append(
            torch.arange(payload_global_start, payload_global_stop, dtype=torch.long, device=device)
        )

    return {
        "capability": SINGLE_GEN_T2I_V1,
        "layout": "packed_varlen",
        "num_samples": num_samples,
        "sequence_length": logical_length,
        "padded_sequence_length": padded_length,
        "position_ids": position_ids,
        "timestep_positions": timestep_positions,
        "image_payload_indices": torch.cat(image_payload_index_blocks).unsqueeze(0),
        # gather Q/K/V for the causal-prefix call; the same index scatters its
        # output back to the packed prefix positions.
        "prefix_gather_index": torch.cat(prefix_index_blocks),
        # gather Q for the image-suffix call; K/V reuse the full packed sequence.
        "image_suffix_gather_index": torch.cat(image_suffix_index_blocks),
        # The prefix call is self-attention, so Q and K share one cu_seqlens.
        "cu_seqlens_prefix": _cu_seqlens(prefix_lengths, device=device),
        "cu_seqlens_q_image_suffix": _cu_seqlens(image_suffix_lengths, device=device),
        "cu_seqlens_k_full": _cu_seqlens(sample_lengths, device=device),
        "max_prefix_length": max(prefix_lengths),
        "max_image_suffix_length": max(image_suffix_lengths),
        "max_full_length": max(sample_lengths),
        "grid_hw": tuple(layout.grid_hw for layout in layouts),
    }


def _cu_seqlens(lengths: Sequence[int], *, device: torch.device | str | None) -> torch.Tensor:
    """Build a padding-free ``[0, l_1, l_1 + l_2, ...]`` cumulative-length tensor."""
    return len2culen(torch.tensor(lengths, dtype=torch.int32, device=device))


# --------------------------------------------------------------------------
# MainCollator wiring
# --------------------------------------------------------------------------


def collate_hunyuan_image_3_metadata(
    batch: dict,
    sp_pad: Mapping[str, int],
    *,
    sp_size: int = 1,
) -> None:
    """Finalize ``hy3_sequence_metadata`` + ``component_inputs`` in place.

    Runs after the collator's pack + SP pad/slice stages (data_collator.py) and
    rebuilds the batch-level packed metadata from the per-sample staging scalars
    the transform emitted. ``sp_size`` is bound at hook-build time (main process)
    so the compiled ``padded_sequence_length`` matches the collator's SP-padded
    ``input_ids``.
    """
    if "hy3_text_token_count" not in batch or "hy3_grid_hw" not in batch:
        raise ValueError("Hunyuan Image 3 metadata hook requires hy3_text_token_count and hy3_grid_hw staging.")
    text_counts = batch.pop("hy3_text_token_count")  # [num_samples]
    grid_hw = batch.pop("hy3_grid_hw")  # [num_samples, 2]

    layouts = [
        T2ILayout(
            text_len=int(text_counts[sample_index]),
            grid_h=int(grid_hw[sample_index, 0]),
            grid_w=int(grid_hw[sample_index, 1]),
        )
        for sample_index in range(int(text_counts.shape[0]))
    ]
    packed = compile_single_gen_t2i_packed(layouts, pad_to_multiple_of=max(int(sp_size), 1))

    input_ids = batch.get("input_ids")
    if input_ids is not None and input_ids.size(-1) != packed["padded_sequence_length"]:
        raise ValueError(
            f"Collated input_ids length {input_ids.size(-1)} does not match the compiled "
            f"padded_sequence_length {packed['padded_sequence_length']}."
        )
    batch["hy3_sequence_metadata"] = packed

    # Reassemble the model's ``component_inputs`` nested dict so it rides
    # BaseTrainer.preforward's recursive device move (which recurses into dicts,
    # not into lists). The staging key uses ``pack_mode="list"``, so
    # PackingCollator keeps one ``[1, C, H, W]`` tensor per packed sample instead
    # of cat-ing them; the model forward stacks them.
    if "hy3_pixel_values" not in batch:
        raise ValueError("Hunyuan Image 3 metadata hook requires hy3_pixel_values staging.")
    batch["component_inputs"] = {"pixel_values": batch.pop("hy3_pixel_values")}


def get_hunyuan_image_3_extra_collate_infos() -> dict:
    """Per-key pack/pad/slice rules (tuples: pack_dim, sp_slice, sp_pad_value, sp_pad_scale, pack_mode).

    input_ids/labels/image_output_mask stay UNsliced (sp_slice=False) — the model
    performs the Ulysses slice internally on the full replicated sequence — but keep
    an sp_pad_value so the collator pads them to a multiple of sp_size, matching the
    compiler's ``pad_to_multiple_of``. The reconstruction scalars pack along dim 0
    (one row per sample) and are neither padded nor sliced.

    A consequence worth spelling out: ``image_output_mask`` reaches
    ``count_loss_token`` intact on every SP rank, so the flow-loss token count is
    replicated rather than sharded. ``mean_global_loss`` is invariant to that (its
    ``SP-sum(cur) / world-sum(global) * effective_dp_size`` scales by ``sp_size`` in
    both numerator and denominator), so nothing downstream needs a correction.

    ``hy3_pixel_values`` uses ``pack_mode="list"`` so the collator preserves
    per-sample tensors and the model decides how to batch them.
    """
    from ....utils.constants import IGNORE_INDEX

    return {
        "input_ids": (-1, False, 0, 1),
        "labels": (-1, False, IGNORE_INDEX, 1),
        "image_output_mask": (-1, False, 0, 1),
        "hy3_text_token_count": (0, False, None, None),
        "hy3_grid_hw": (0, False, None, None),
        "hy3_pixel_values": (0, False, None, None, "list"),
    }


__all__ = [
    "SINGLE_GEN_T2I_V1",
    "STATIC_CONTROL_TOKEN_COUNT",
    "T2ILayout",
    "UnsupportedSequenceLayout",
    "collate_hunyuan_image_3_metadata",
    "compile_single_gen_t2i_packed",
    "get_hunyuan_image_3_extra_collate_infos",
]
