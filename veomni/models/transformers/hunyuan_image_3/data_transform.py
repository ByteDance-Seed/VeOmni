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

"""``single_gen_t2i_v1`` data transform for the Hunyuan Image 3 T2I task.

Maps one raw ``{prompt, image, width, height}`` sample to the per-sample staging
tensors the ``MainCollator`` packs and the model metadata hook
(:mod:`.metadata_collate`) finalizes into ``hy3_sequence_metadata``. The transform
is per-sample (runs in DataLoader workers); cross-sample packing and the packed
GCA metadata are produced later by the metadata hook.
"""

from typing import Any, Optional

import torch

from ....data.data_transform import DATA_TRANSFORM_REGISTRY
from ....utils.constants import IGNORE_INDEX
from .resolution_policy import HunyuanImage3ResolutionPolicy
from .sequence_compiler import build_single_gen_t2i_plan


def _resolve_grid_and_latent(
    sample,
    *,
    latent_source,
    resolution_policy,
    image_processor,
    target_image_key,
    width_key,
    height_key,
    latent_channels,
    default_base_size,
):
    """Return ``(grid_hw, staging)`` where staging carries per-sample latent tensors.

    Per-sample bucket selection uses the processor's ``argmin_i |ratios[i] - h/w|``
    against the single anchor at ``default_base_size`` (P3 single-anchor invariant).
    The trainer's ``HunyuanImage3BucketBatchSampler`` handles cross-sample /
    cross-rank same-bucket coordination out-of-band (before this transform runs), so
    this function stays purely per-sample and picklable across DataLoader workers —
    unlike the deprecated ``ScheduledResolutionTransform`` which threaded a
    scheduler counter through worker processes.
    """
    if latent_source == "posterior_cache":
        mean = sample.get("latent_mean")
        logvar = sample.get("latent_logvar")
        if mean is None or logvar is None:
            raise ValueError("posterior_cache samples must provide 'latent_mean' and 'latent_logvar'.")
        mean = torch.as_tensor(mean, dtype=torch.float32)
        logvar = torch.as_tensor(logvar, dtype=torch.float32)
        if mean.ndim != 3 or mean.shape != logvar.shape:
            raise ValueError("Cached latent mean/logvar must be [C, H, W] with identical shapes.")
        if mean.shape[0] != latent_channels:
            raise ValueError("Cached latent channel count does not match vae.latent_channels.")
        grid_hw = (mean.shape[1], mean.shape[2])
        staging = {
            "hy3_latent_mean": mean.unsqueeze(0),
            "hy3_latent_logvar": logvar.unsqueeze(0),
        }
        return grid_hw, staging

    if latent_source != "online_vae":
        raise ValueError(f"Unsupported latent_source: {latent_source!r}.")
    if image_processor is None:
        raise ValueError("online_vae requires an image processor.")
    image = sample[target_image_key]
    width = sample.get(width_key) if width_key else None
    height = sample.get(height_key) if height_key else None
    processed = image_processor.preprocess_target_image(image, width=width, height=height, base_size=default_base_size)
    grid_hw = processed.grid_hw
    staging = {"hy3_pixel_values": processed.pixel_values.unsqueeze(0)}
    return grid_hw, staging


@DATA_TRANSFORM_REGISTRY.register("hunyuan_image_3_moe")
def process_sample_hunyuan_image_3(
    sample: dict,
    *,
    tokenizer: Optional[Any] = None,
    resolution_policy: HunyuanImage3ResolutionPolicy,
    image_processor: Optional[Any] = None,
    latent_source: str = "online_vae",
    target_image_key: str = "image",
    width_key: Optional[str] = "width",
    height_key: Optional[str] = "height",
    prompt_dropout_prob: float = 0.0,
    text_key: str = "prompt",
    max_seq_len: int = 8192,
    latent_channels: int = 32,
    default_base_size: int = 1024,
    im_start_id: int = 128000,
    im_end_id: int = 128001,
    image_token_id: int = 128006,
    **kwargs,
) -> list[dict]:
    """Transform one raw T2I sample into single-sample staging tensors.

    P3: cross-sample / cross-rank bucket coordination is handled by
    :class:`.bucket_batch_sampler.HunyuanImage3BucketBatchSampler` (main-process,
    ``num_workers>0`` safe) instead of the removed ``ScheduledResolutionTransform``.
    Each sample independently picks its bucket via ``argmin`` in
    ``preprocess_target_image``; when ``same_bucket_batching=True`` the sampler
    guarantees a whole micro-batch shares a bucket by construction, so the
    ``argmin`` results align across the mbs samples in that micro-batch.
    """
    prompt = sample.get(text_key, "")
    if prompt_dropout_prob and torch.rand(()).item() < prompt_dropout_prob:
        prompt = ""

    if tokenizer is not None:
        text_ids = list(tokenizer.encode(prompt))
    else:
        # Toy fallback: one id per whitespace token (at least one).
        text_ids = [(i % max(image_token_id, 1)) for i in range(max(len(prompt.split()), 1))]
    if not text_ids:
        text_ids = [im_start_id]

    grid_hw, staging = _resolve_grid_and_latent(
        sample,
        latent_source=latent_source,
        resolution_policy=resolution_policy,
        image_processor=image_processor,
        target_image_key=target_image_key,
        width_key=width_key,
        height_key=height_key,
        latent_channels=latent_channels,
        default_base_size=default_base_size,
    )
    grid_height, grid_width = grid_hw
    text_token_count = len(text_ids)

    plan = build_single_gen_t2i_plan(
        sample_id=str(sample.get("id", "sample")),
        text_token_count=text_token_count,
        grid_hw=(grid_height, grid_width),
    )
    sequence_length = plan["sequence_length"]
    if sequence_length > max_seq_len:
        raise ValueError(f"Compiled sequence length {sequence_length} exceeds max_seq_len {max_seq_len}.")

    # Physical layout (RFC §6.3): text | <boi> <img_size> <img_ratio> | <timestep>
    # | <img> x N | <eoi>. Payload/timestep ids are overwritten by the model
    # (patch_embed / timestep_emb), so their placeholder ids are irrelevant; the
    # controls and <eoi> keep real special-token embeddings.
    payload_count = grid_height * grid_width
    input_ids = (
        list(text_ids)
        + [im_start_id, image_token_id, image_token_id]  # <boi>, <img_size_*>, <img_ratio_*> (see note)
        + [image_token_id]  # <timestep> placeholder (overwritten)
        + [image_token_id] * payload_count  # <img> payload (overwritten)
        + [im_end_id]  # <eoi>
    )
    assert len(input_ids) == sequence_length, (len(input_ids), sequence_length)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.full((sequence_length,), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.ones((sequence_length,), dtype=torch.long)

    image_output_mask = torch.zeros((sequence_length,), dtype=torch.bool)
    payload_start = text_token_count + 3 + 1
    image_output_mask[payload_start : payload_start + payload_count] = True

    feature = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "image_output_mask": image_output_mask,
        # Reconstruction scalars for the metadata hook (packed over samples).
        "hy3_text_token_count": torch.tensor([text_token_count], dtype=torch.long),
        "hy3_grid_hw": torch.tensor([[grid_height, grid_width]], dtype=torch.long),
        **staging,
    }
    return [feature]


__all__ = ["process_sample_hunyuan_image_3"]
