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

"""MainCollator metadata hook for Hunyuan Image 3 ``single_gen_t2i_v1``.

Module-level so ``functools.partial(collate_hunyuan_image_3_metadata, ...)`` is
picklable across DataLoader workers. It runs after the collator's pack + SP
pad/slice stages (data_collator.py) and finalizes the packed varlen metadata the
model forward consumes, from the per-sample staging tensors the transform emitted.
"""

from typing import Mapping

from .sequence_compiler import build_single_gen_t2i_plan, compile_single_gen_t2i_packed


def collate_hunyuan_image_3_metadata(
    batch: dict,
    sp_pad: Mapping[str, int],
    *,
    sp_size: int = 1,
) -> None:
    """Finalize ``hy3_sequence_metadata`` + ``component_inputs`` in place.

    ``sp_size`` is bound at hook-build time (main process) so the compiled
    ``padded_sequence_length`` matches the collator's SP-padded ``input_ids``.
    """
    if "hy3_text_token_count" not in batch or "hy3_grid_hw" not in batch:
        raise ValueError("Hunyuan Image 3 metadata hook requires hy3_text_token_count and hy3_grid_hw staging.")
    text_counts = batch.pop("hy3_text_token_count")  # [num_samples]
    grid_hw = batch.pop("hy3_grid_hw")  # [num_samples, 2]
    num_samples = int(text_counts.shape[0])

    plans = [
        build_single_gen_t2i_plan(
            sample_id=f"s{sample_index}",
            text_token_count=int(text_counts[sample_index]),
            grid_hw=(int(grid_hw[sample_index, 0]), int(grid_hw[sample_index, 1])),
        )
        for sample_index in range(num_samples)
    ]
    packed = compile_single_gen_t2i_packed(plans, pad_to_multiple_of=max(int(sp_size), 1))

    input_ids = batch.get("input_ids")
    if input_ids is not None and input_ids.size(-1) != packed["padded_sequence_length"]:
        raise ValueError(
            f"Collated input_ids length {input_ids.size(-1)} does not match the compiled "
            f"padded_sequence_length {packed['padded_sequence_length']}."
        )
    batch["hy3_sequence_metadata"] = packed

    # Reassemble the model's ``component_inputs`` nested dict from the packed
    # latent staging so it rides BaseTrainer.preforward's recursive device move.
    # Under P1a the staging keys use ``pack_mode="list"`` (data_collator.py) so
    # PackingCollator preserves per-sample tensors instead of ``torch.cat(dim=0)``-
    # ing them into a shape-uniform batch tensor. Under ``mbs=1`` these lists have
    # length 1 (byte-identical to the pre-P1a stacked tensor shape once the model
    # forward stacks them). Under ``mbs>1`` + ``same_bucket_batching=True`` all
    # per-sample tensors share ``(C, H, W)`` by construction, so the model's
    # smart ``_encode_pixel_values_to_posterior`` / ``_get_latent_posterior`` will
    # ``torch.cat`` them into the batched fast path. The ``same_bucket_batching=
    # False`` heterogeneous case is rejected by the model with a clear error
    # (deferred until P1b lands the per-sample forward path — see
    # plan_bucketing.md).
    if "hy3_pixel_values" in batch:
        batch["component_inputs"] = {"pixel_values": batch.pop("hy3_pixel_values")}
    elif "hy3_latent_mean" in batch and "hy3_latent_logvar" in batch:
        batch["component_inputs"] = {
            "latent_posterior": {
                "mean": batch.pop("hy3_latent_mean"),
                "logvar": batch.pop("hy3_latent_logvar"),
            }
        }
    else:
        raise ValueError("Hunyuan Image 3 metadata hook requires pixel_values or latent posterior staging.")


def get_hunyuan_image_3_extra_collate_infos() -> dict:
    """Per-key pack/pad/slice rules (tuples: pack_dim, sp_slice, sp_pad_value, sp_pad_scale, pack_mode).

    input_ids/labels/image_output_mask stay UNsliced (sp_slice=False) — the model
    performs the Ulysses slice internally on the full replicated sequence — but keep
    an sp_pad_value so the collator pads them to a multiple of sp_size, matching the
    compiler's ``pad_to_multiple_of``. The reconstruction scalars pack along dim 0
    (one row per sample) and are neither padded nor sliced.

    ``sp_slice=False`` on ``image_output_mask`` is also the **single source of
    truth** for the SP layout of the flow-loss token normalizer: since the mask
    reaches ``count_loss_token`` intact on every SP rank, ``image_decoder_tokens``
    is replicated across the SP group. ``mean_global_loss``'s formula
    ``SP-sum(cur) / world-sum(global) * effective_dp_size`` is invariant to
    sharded-vs-replicated (both scale by ``sp_size`` in numerator and denominator),
    so no additional signal is needed downstream.

    Latent staging (``hy3_pixel_values`` / ``hy3_latent_mean`` / ``hy3_latent_logvar``)
    uses ``pack_mode="list"`` (P1a): the collator preserves per-sample tensors so
    heterogeneous ``(C, H, W)`` shapes (mbs>1 with per-sample bucket selection) don't
    crash on ``torch.cat(dim=0)``. The model side (``_encode_pixel_values_to_posterior``
    / ``_get_latent_posterior``) detects uniform shape and batches the VAE encode.
    """
    from ....utils.constants import IGNORE_INDEX

    return {
        "input_ids": (-1, False, 0, 1),
        "labels": (-1, False, IGNORE_INDEX, 1),
        "image_output_mask": (-1, False, 0, 1),
        "hy3_text_token_count": (0, False, None, None),
        "hy3_grid_hw": (0, False, None, None),
        "hy3_pixel_values": (0, False, None, None, "list"),
        "hy3_latent_mean": (0, False, None, None, "list"),
        "hy3_latent_logvar": (0, False, None, None, "list"),
    }


__all__ = [
    "collate_hunyuan_image_3_metadata",
    "get_hunyuan_image_3_extra_collate_infos",
]
