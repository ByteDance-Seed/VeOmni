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

"""VLMTrainer integration hooks for the Hunyuan Image 3 T2I generation task.

Only genuinely HI3-specific orchestration lives here: the model-owned data
transform (typed ``ImageGenerationDataConfig`` + HI3 resolution policy +
bucket scheduler + optional online-VAE image processor) and the bucket
``batch_sampler`` dataloader wiring. Two other hooks that used to live here
have moved into ``VLMTrainer`` itself because they were generic:

* Model-asset build for generation tasks (no processor / no chat template,
  best-effort tokenizer) is now a ``data_type == 'multimodal_generation'``
  branch in ``VLMTrainer._build_model_assets``.
* Per-micro-step flow config + RNG identity injection is now an
  ``if args.train.flow:`` branch in ``VLMTrainer.train_step``.

An instance of this class is attached to the model class as
``trainer_hooks`` in this package's ``__init__`` (mirroring
``_create_checkpoint_tensor_converter``); ``VLMTrainer`` resolves it with
``getattr(model, "trainer_hooks", None)`` and dispatches through it — the
optional-capability pattern also used by ``apply_component_policy`` and
``get_extra_collate_infos``. Heavy imports are kept lazy (inside methods) to
avoid import cycles at model-package import time.
"""

from ....utils import helper


logger = helper.create_logger(__name__)


class HunyuanImage3TrainerHooks:
    """Generation-task trainer hooks for ``model_type == 'hunyuan_image_3_moe'``.

    Stateless: every method receives the ``VLMTrainer`` and reads/writes state on
    its ``base`` (``BaseTrainer``) exactly as the former inline branches did.
    """

    def build_data_transform(self, trainer) -> None:
        """Build the T2I sample transform + resolution policy + bucket scheduler."""
        from ....data import build_data_transform
        from .processing_hunyuan_image_3 import (
            HunyuanImage3ImageProcessor,
            HunyuanImage3ImageProcessorConfig,
        )
        from .resolution_policy import (
            build_hunyuan_image_3_bucket_scheduler,
            build_resolution_policy,
        )
        from .training_config import (
            ImageGenerationDataConfig,
            validate_hunyuan_image_3_training_args,
        )

        base = trainer.base
        args = base.args
        config = base.model_config
        # ``data.image_generation`` arrives as an opaque dict from the shared arg parser;
        # build the typed schema here (its __post_init__ validates) and stash it so
        # build_dataloader reuses the same instance (constructed once).
        generation = ImageGenerationDataConfig(**args.data.image_generation) if args.data.image_generation else None
        validate_hunyuan_image_3_training_args(
            generation=generation,
            component_policy=args.train.component_policy,
            flow=args.train.flow,
        )
        base.image_generation = generation

        vae_spatial_factor = int(config.vae_downsample_factor[0])
        resolution_policy = build_resolution_policy(
            generation.resolution_policy,
            vae_spatial_factor=vae_spatial_factor,
            patch_size=config.patch_size,
        )
        # Persisted for build_dataloader (BucketIndexer + BucketBatchSampler need
        # it too — no duplicated build).
        base.resolution_policy = resolution_policy

        # Deterministic per-micro-step resolution scheduler (RFC "Segment plan").
        # Selection is a pure function of (scheduler_seed, global_step, micro_step),
        # so every rank agrees without a collective and DCP resume reproduces the
        # sequence from the restored global_step.
        base.bucket_scheduler = None
        if resolution_policy.config.selection_mode == "synchronized_weighted":
            base.bucket_scheduler = build_hunyuan_image_3_bucket_scheduler(resolution_policy)
            logger.info_rank0(
                f"Hunyuan Image 3 bucket scheduler active: "
                f"{base.bucket_scheduler.num_buckets} buckets, "
                f"policy_hash={base.bucket_scheduler.policy_hash()}."
            )

        image_processor = None
        if generation.latent_source == "online_vae":
            image_processor = HunyuanImage3ImageProcessor(
                HunyuanImage3ImageProcessorConfig(
                    resolution_policy=generation.resolution_policy,
                    vae_spatial_factor=vae_spatial_factor,
                    patch_size=config.patch_size,
                    default_base_size=generation.default_base_size,
                )
            )

        base.data_transform = build_data_transform(
            "hunyuan_image_3_moe",
            tokenizer=getattr(base, "tokenizer", None),
            resolution_policy=resolution_policy,
            image_processor=image_processor,
            latent_source=generation.latent_source,
            target_image_key=generation.target_image_key,
            width_key=generation.target_width_key,
            height_key=generation.target_height_key,
            prompt_dropout_prob=generation.prompt_dropout_prob,
            text_key=args.data.text_keys,
            max_seq_len=args.data.max_seq_len,
            latent_channels=config.vae["latent_channels"],
            default_base_size=generation.default_base_size,
            im_start_id=config.im_start_id,
            im_end_id=config.im_end_id,
            image_token_id=config.image_token_id,
        )

    def build_dataloader(self, trainer) -> bool:
        """Build a ``batch_sampler``-driven dataloader for same-bucket batching.

        Returns ``True`` when it built the dataloader; ``False`` to let the
        trainer fall back to the generic ``BaseTrainer._build_dataloader`` (e.g.
        ``same_bucket_batching=False``).

        Runs the generic ``BucketIndexer`` (wired to this policy's
        ``select_bucket``) once to produce per-sample ``bucket_ids``, then
        constructs a ``BucketBatchSampler`` shared across all DP ranks (each
        rank owns its own bucket-local shard; cross-rank agreement comes from
        the pure-function ``BucketScheduler.select_bucket_id`` — no collective).
        The sampler + its indexer fingerprint are stashed on ``base`` for the
        DCP manifest + resume.
        """
        base = trainer.base
        args = base.args
        generation = getattr(base, "image_generation", None)
        if generation is None or not generation.same_bucket_batching:
            return False

        from ....distributed.parallel_state import get_parallel_state
        from .resolution_policy import (
            build_hunyuan_image_3_bucket_batch_sampler,
            build_hunyuan_image_3_bucket_indexer,
        )

        parallel_state = get_parallel_state()
        scheduler = getattr(base, "bucket_scheduler", None)
        resolution_policy = getattr(base, "resolution_policy", None)
        if scheduler is None or resolution_policy is None:
            raise RuntimeError(
                "same_bucket_batching=True requires a resolution policy with "
                "selection_mode='synchronized_weighted' so ``BucketScheduler`` is built."
            )

        # ``MappingDataset.__getitem__(int)`` runs the transform and returns the
        # HI3 staging tensors, hiding the raw ``(width, height)`` columns from the
        # indexer. Unwrap to the underlying HF dataset so BucketIndexer sees them.
        raw_dataset = getattr(base.train_dataset, "_data", base.train_dataset)
        indexer = build_hunyuan_image_3_bucket_indexer(
            resolution_policy,
            default_base_size=generation.default_base_size,
            width_key=generation.target_width_key or "width",
            height_key=generation.target_height_key or "height",
            image_key=generation.target_image_key,
        )
        bucket_ids = indexer.index(raw_dataset)
        fingerprint = indexer.fingerprint(raw_dataset)
        base.bucket_indexer_fingerprint = fingerprint
        logger.info_rank0(
            f"Hunyuan Image 3 BucketIndexer: N={len(bucket_ids)} samples over "
            f"{scheduler.num_buckets} buckets, fingerprint={fingerprint}."
        )

        dp_size = int(parallel_state.dp_size)
        # ``dataloader_batch_size`` is per-rank; num_micro_batch = grad accum.
        num_micro_batch = args.train.global_batch_size // (args.train.micro_batch_size * dp_size)
        if num_micro_batch <= 0:
            raise ValueError(
                f"num_micro_batch resolved to {num_micro_batch}; check "
                f"global_batch_size={args.train.global_batch_size} vs "
                f"micro_batch_size={args.train.micro_batch_size} * dp_size={dp_size}."
            )
        batch_sampler = build_hunyuan_image_3_bucket_batch_sampler(
            resolution_policy,
            bucket_ids=bucket_ids,
            dp_rank=int(parallel_state.dp_rank),
            dp_size=dp_size,
            micro_batch_size=int(args.train.micro_batch_size),
            num_micro_batch=int(num_micro_batch),
            seed=int(args.train.seed),
            schedule_fn=scheduler.select_bucket_id,
        )
        # The sampler drops buckets that cannot serve mbs on any DP rank (see
        # ``drop_insufficient_bucket`` policy — per-bucket WARNINGs and a
        # summary are emitted from the sampler itself). The scheduler must be
        # restricted to match or it would still weighted-pick a dropped
        # bucket_id and ``_resolve_bucket`` would raise mid-run. This mutation
        # shifts the scheduler's ``policy_hash`` — the DCP manifest gate then
        # rejects any resume where the drop set drifted.
        if batch_sampler.dropped_bucket_ids:
            scheduler.restrict_to(batch_sampler.active_bucket_ids)
        base.bucket_batch_sampler = batch_sampler
        logger.info_rank0(
            f"Hunyuan Image 3 BucketBatchSampler ready: mbs={args.train.micro_batch_size}, "
            f"num_micro_batch={num_micro_batch}, dp_size={dp_size}."
        )

        # Reuse the shared builder (dyn_bsz/batch_sampler mutual-exclusion is
        # enforced there) instead of duplicating its body.
        base._build_dataloader(batch_sampler=batch_sampler)
        return True


__all__ = ["HunyuanImage3TrainerHooks"]
