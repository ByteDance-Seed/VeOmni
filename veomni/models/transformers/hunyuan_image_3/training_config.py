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

"""Typed training/data/model configuration for the Hunyuan Image 3 T2I task.

These schemas are consumed by the VLM argument subclasses (``data.image_generation``,
``train.flow``) and validated before the model/dataset are built.
They reuse the model-local :class:`ResolutionPolicyConfig` (Track C) as the single
source of truth for the resolution policy; nothing here duplicates that schema.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

from .resolution_policy import ResolutionPolicyConfig


# ------------------------------- data.image_generation ------------------------------


@dataclass
class ImageGenerationDataConfig:
    """``data.image_generation`` — image-generation (T2I / IT2I) task data schema (RFC §5.1).

    Covers text-to-image today and image+text-to-image (editing / conditioning) as an
    additive extension. Every column describes the generation *target* image, leaving
    the ``condition_*`` namespace free for future IT2I input/reference images.
    """

    task_type: Literal["t2i"] = "t2i"
    latent_source: Literal["online_vae", "posterior_cache"] = "online_vae"
    target_image_key: str = "image"
    target_width_key: Optional[str] = "width"
    target_height_key: Optional[str] = "height"
    resolution_policy: ResolutionPolicyConfig = field(default_factory=ResolutionPolicyConfig)
    prompt_dropout_prob: float = 0.0
    # P3: single-anchor default_base_size drives per-sample argmin bucket selection
    # (was hardcoded 1024 in HunyuanImage3ImageProcessorConfig).
    default_base_size: int = 1024
    # P3: when True, ``HunyuanImage3BucketBatchSampler`` groups mbs samples per
    # micro-batch into one bucket and every DP rank at the same
    # ``(global_step, micro_step)`` picks the same bucket — enables mbs>1 with
    # heterogeneous datasets, load-balanced VAE encode, DCP-resumable cursors.
    # When False, standard ``StatefulDistributedSampler`` gives per-sample
    # random buckets; only mbs=1 is currently supported (heterogeneous mbs>1
    # forward path is deferred to P1b in plan_bucketing.md).
    same_bucket_batching: bool = True

    def __post_init__(self):
        # The shared parser recurses into nested dataclasses, so ``resolution_policy``
        # is normally already a ``ResolutionPolicyConfig``; normalize a raw dict for
        # manual construction.
        if isinstance(self.resolution_policy, dict):
            self.resolution_policy = ResolutionPolicyConfig(**self.resolution_policy)
        if not isinstance(self.target_image_key, str) or not self.target_image_key:
            raise ValueError("data.image_generation.target_image_key must be a non-empty string.")
        if not 0.0 <= float(self.prompt_dropout_prob) <= 1.0:
            raise ValueError("data.image_generation.prompt_dropout_prob must be in [0, 1].")
        if isinstance(self.default_base_size, bool) or not isinstance(self.default_base_size, int):
            raise TypeError("data.image_generation.default_base_size must be an int.")
        if self.default_base_size <= 0:
            raise ValueError("data.image_generation.default_base_size must be positive.")
        if not isinstance(self.same_bucket_batching, bool):
            raise TypeError("data.image_generation.same_bucket_batching must be a bool.")


# --------------------------------- validation ---------------------------------

# RFC §5.2: only these (latent_source, vae_encoder, vae_decoder) combinations are
# supported for the initial single_gen_t2i_v1 capability. The VAE's online dtype
# is fixed (float32) and derived from latent_source, not independently configured.
_ALLOWED_LATENT_SOURCE_COMBOS = {
    "online_vae": {"vae_encoder": "frozen", "vae_decoder": "absent"},
    "posterior_cache": {"vae_encoder": "absent", "vae_decoder": "absent"},
}


def validate_hunyuan_image_3_training_args(
    *,
    generation: Optional[ImageGenerationDataConfig],
    component_policy: Mapping[str, str],
    flow: Optional[Mapping[str, Any]],
) -> None:
    """Validate the generation/flow/component_policy combination before build.

    ``flow`` carries the flow-matching training objective — the noise schedule
    and prediction target used by ``prepare_reference_flow_batch`` /
    ``flow_matching_loss`` in the model forward. Since commit ``440b3364``
    decoupled ``vlm_trainer`` from HI3, it is passed through as an **opaque
    ``dict[str, Any]``**: this validator only asserts presence, and the actual
    key normalization + defaults live in
    ``veomni.schedulers.flow_matching._normalize_flow_config`` (the single
    source of truth for accepted keys such as ``num_train_timesteps`` /
    ``timestep_sampling`` / ``training_shift`` / ``prediction_type`` /
    ``loss_weighting``).
    """
    if generation is None:
        raise ValueError("Hunyuan Image 3 requires data.image_generation for multimodal_generation.")
    if flow is None:
        raise ValueError("Hunyuan Image 3 requires train.flow.")

    latent_source = generation.latent_source
    if latent_source not in _ALLOWED_LATENT_SOURCE_COMBOS:
        raise ValueError(f"Unsupported latent_source: {latent_source!r}.")
    expected = _ALLOWED_LATENT_SOURCE_COMBOS[latent_source]

    vae_encoder = component_policy.get("vae_encoder")
    vae_decoder = component_policy.get("vae_decoder")
    if vae_encoder != expected["vae_encoder"] or vae_decoder != expected["vae_decoder"]:
        raise ValueError(
            f"latent_source={latent_source!r} requires component_policy vae_encoder="
            f"{expected['vae_encoder']!r}, vae_decoder={expected['vae_decoder']!r}; got "
            f"vae_encoder={vae_encoder!r}, vae_decoder={vae_decoder!r}."
        )

    # P3: same_bucket_batching=True requires a single-anchor policy so the
    # BucketBatchSampler + BucketIndexer operate on a well-defined argmin table.
    # The check runs on the resolved (dataclass-normalized) anchor list:
    # an empty list means "every preset anchor" — three anchors, would break the
    # single-anchor invariant — so we require at least one anchor entry, all of
    # them sharing base_size == default_base_size.
    if generation.same_bucket_batching:
        anchors = list(generation.resolution_policy.anchors)
        if not anchors:
            raise ValueError(
                "same_bucket_batching=True requires an explicit single-anchor "
                "``data.image_generation.resolution_policy.anchors`` list. Empty means "
                "'every preset anchor' (three of them), which breaks the single-"
                "anchor invariant. Add:  anchors: [{base_size: "
                f"{generation.default_base_size}, weight: 1.0}}]"
            )
        distinct_base_sizes = {anchor.base_size for anchor in anchors}
        if distinct_base_sizes != {generation.default_base_size}:
            raise ValueError(
                "same_bucket_batching=True requires all "
                "``data.image_generation.resolution_policy.anchors[*].base_size`` to equal "
                f"``data.image_generation.default_base_size`` ({generation.default_base_size}). "
                f"Got anchor base_sizes: {sorted(distinct_base_sizes)}. Multi-anchor "
                "policies are unsupported under the P3 single-anchor invariant; use "
                "same_bucket_batching=False + mbs=1 for multi-anchor exploration."
            )


__all__ = [
    "ImageGenerationDataConfig",
    "validate_hunyuan_image_3_training_args",
]
