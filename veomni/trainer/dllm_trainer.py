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

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoImageProcessor
from transformers.modeling_outputs import ModelOutput

from ..arguments import VeOmniArguments
from ..data import DiffusionGemmaCorruptionEpoch, build_chat_template, build_data_transform
from ..data.chat_template import MultimodalChatTemplate
from ..data.data_collator import DataCollateInfo, MainCollator, PackingCollator
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.parallel_state import get_parallel_state, use_parallel_state
from ..distributed.sequence_parallel.loss import sum_over_ulysses_shards
from ..models.model_runtime import VeOmniModelRuntime
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..utils import helper
from ..utils.device import synchronize
from ..utils.loss_utils import dllm_mean_global_loss
from .base import BaseTrainer, VeOmniIter


__all__ = ["DllmCollator", "DllmModelRuntime", "DllmTrainer", "VeOmniArguments"]

logger = helper.create_logger(__name__)

#: Models whose loss and router statistics `DllmTrainer` knows how to reduce over a sequence shard.
SEQUENCE_PARALLEL_MODEL_TYPES = ("llada", "llada_moe", "llada_v", "llada2_uni")
ULYSSES_ATTENTION_TYPES = (
    "veomni_flash_attention_2_with_sp",
    "veomni_flash_attention_3_with_sp",
    "veomni_flash_attention_4_with_sp",
)


def count_supervised_samples(micro_batch: Dict[str, torch.Tensor]) -> int:
    """Count the supervised samples in one micro-batch.

    `MainCollator` packs several samples into one row and marks where each starts
    with ``position_ids == 0``, so under packing a row is not a sample. Segments are
    counted when ``position_ids`` lines up with the supervision marker; otherwise
    every row is one sample.

    The marker is ``answer_mask`` when the transform emits it, else ``loss_weight``.
    A LLaDA draw masks nothing in about 1/(N+1) of samples with an N-token answer;
    its weights are all zero but the sample is still part of the mean, and leaving
    it out would bias the loss up by that fraction.
    """
    marker = micro_batch.get("answer_mask", micro_batch["loss_weight"])
    position_ids = micro_batch.get("position_ids")
    if position_ids is not None and position_ids.dim() == 3:
        position_ids = position_ids[:, 0, :]
    if position_ids is None or position_ids.shape != marker.shape:
        return int((marker.sum(dim=-1) > 0).sum())
    count = 0
    for row_positions, row_weight in zip(position_ids, marker):
        starts = row_positions == 0
        starts[0] = True
        segment_ids = starts.cumsum(dim=0) - 1
        per_segment = torch.zeros(int(segment_ids[-1]) + 1, dtype=torch.float64, device=row_weight.device)
        per_segment.index_add_(0, segment_ids, row_weight.to(torch.float64))
        count += int((per_segment > 0).sum())
    return count


def _attach_supervised_sample_count(micro_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    micro_batch["num_supervised_samples"] = torch.tensor(count_supervised_samples(micro_batch))
    micro_batch.pop("answer_mask", None)
    return micro_batch


@dataclass
class DllmCollator(MainCollator):
    """`MainCollator` that counts samples between packing and the SP slice.

    The count needs whole segments, and after the slice a rank holds a stretch of
    the row that can start or end mid-sample. It runs at sp=1 too, so both paths
    divide by the same number.
    """

    def __post_init__(self):
        super().__post_init__()
        packing = next(i for i, stage in enumerate(self.preforward_pipeline) if isinstance(stage, PackingCollator))
        self.preforward_pipeline.insert(packing + 1, _attach_supervised_sample_count)


class DllmModelRuntime(VeOmniModelRuntime):
    """`VeOmniModelRuntime` with a hook between weight loading and the optimizer.

    Also keeps text-only chat templates on the tokenizer when the checkpoint ships a
    processor: the runtime builds every template from ``processor or tokenizer``, and a
    text template stores whatever it is handed as its tokenizer, which a processor
    cannot stand in for (it has no ``encode``).
    """

    def _build_model_assets(self) -> None:
        super()._build_model_assets()
        if (
            self.chat_template is not None
            and self.processor is not None
            and not isinstance(self.chat_template, MultimodalChatTemplate)
        ):
            self.chat_template = build_chat_template(self.args.chat_template, self.tokenizer)

    def after_weights_land(self) -> None:
        """Last chance to change the parameter list, between loading and the optimizer.

        `_build_parallelized_model` ends by writing the checkpoint into the shards, and
        `_build_optimizer` reads `requires_grad` and takes a moment per trainable
        tensor. A subclass that reparameterizes the model has to do it in this gap:
        earlier and there are no values to fold, later and the optimizer is already
        holding state for a parameter that no longer moves.

        No-op by default.
        """

    def _build_optimizer(self, param_groups: List[Dict[str, Any]] | None = None) -> None:
        self.after_weights_land()
        super()._build_optimizer(param_groups=param_groups)


class DllmTrainer:
    """Generic discrete-diffusion LM trainer.

    Composes `BaseTrainer`.
    Replaces token-count loss reduction with a per-sample mean (`dllm_mean_global_loss`).

    Model-specific geometry (canvas packing, encoder-decoder split, self-conditioning,
    multimodal collators) belongs on a subclass such as `DiffusionGemmaTrainer`.
    """

    base: BaseTrainer

    #: Set per step by `on_micro_batches_ready`; only 1 for a forward outside `train_step`.
    micro_batches_num_samples: int = 1
    #: Set per step by `on_micro_batches_ready` alongside the sample count.
    num_micro_batches: int = 1
    #: The image preprocessing for `data_type: llada_v` / `llada2_uni`; ``None`` for text-only data.
    image_processor: Any = None

    def __init__(self, args: VeOmniArguments):
        # BaseTrainer.__init__ is NOT called here; we call its private
        # helpers one-by-one so the sequence is explicit.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base.device = self.base._setup(args)  # registers ParallelState("base") before seed
        self.base.model = self._build_model_runtime()

        with use_parallel_state(self.base.model.parallel_state):
            self._build_image_processor()
            self._build_data_transform()
            self.base._build_dataset()
            self._build_collate_fn()
            self.base._build_dataloader()
        self.base._build_lr_scheduler()
        self.base._build_training_context(self.base.model)
        self.base._init_callbacks()

    def _build_model_runtime(self) -> DllmModelRuntime:
        """Build (and own) this job's model. Override to swap in another runtime."""
        return DllmModelRuntime(self.base.args.model, "base", train=self.base.args.train)

    def _build_image_processor(self) -> None:
        """Load the image preprocessing that `data_type: llada_v` / `llada2_uni` need.

        Checked against the model's vision tower here, since a mismatch would only show
        up as image slot counts that disagree with the embeddings.
        """
        args: VeOmniArguments = self.base.args
        runtime = self.base.model
        model_config = runtime.model_config
        if args.data.data_type == "llada_v":
            if model_config.model_type != "llada_v":
                raise ValueError(f"`data_type: llada_v` needs a LLaDA-V model, got `{model_config.model_type}`.")
            self.image_processor = AutoImageProcessor.from_pretrained(args.model.tokenizer_path)
            size = self.image_processor.size
            image_size = model_config.vision_config.image_size
            if (size["height"], size["width"]) != (image_size, image_size):
                raise ValueError(
                    f"The image processor resizes to {size['height']}x{size['width']}, but the vision tower "
                    f"expects {image_size}x{image_size}; each image's slot count would not match its embeddings."
                )
        elif args.data.data_type == "llada2_uni":
            if model_config.model_type != "llada2_uni":
                raise ValueError(
                    f"`data_type: llada2_uni` needs a LLaDA2.0-Uni model, got `{model_config.model_type}`."
                )
            self.image_processor = AutoImageProcessor.from_pretrained(args.model.tokenizer_path)
            processor = self.image_processor
            patch_size = model_config.vision_config.patch_size
            if (processor.patch_size, processor.merge_size, processor.temporal_patch_size) != (patch_size, 1, 1):
                raise ValueError(
                    f"The image processor patchifies at patch_size={processor.patch_size}, "
                    f"merge_size={processor.merge_size}, temporal_patch_size={processor.temporal_patch_size}; the "
                    f"vision tower reads single {patch_size}px patches, one VQ token each."
                )
        else:
            return
        if runtime.processor is None:
            runtime.model_assets.append(self.image_processor)

    def _build_data_transform(self):
        args: VeOmniArguments = self.base.args
        # Every rank calls `set_seed(train.seed)` with no rank offset, so a transform that
        # draws its mask from the process RNG hands every rank the same one; `llada`
        # keys a per-sample generator on this seed and the epoch instead.
        extra_kwargs = {"mask_seed": args.train.seed, "mask_epoch": self._corruption_epoch}
        mask_token_id = getattr(self.base.model.model_config, "mask_token_id", None)
        if mask_token_id is not None:
            extra_kwargs["mask_token_id"] = mask_token_id
        if self.image_processor is not None:
            extra_kwargs["image_processor"] = self.image_processor
        if args.data.data_type == "llada_v":
            extra_kwargs["image_seq_len"] = self.base.model.model_config.image_seq_len
        self._warn_if_the_corruption_epoch_can_lag()
        self.base.data_transform = build_data_transform(
            args.data.data_type,
            tokenizer=self.base.model.tokenizer,
            chat_template=self.base.model.chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            **extra_kwargs,
        )

    def _build_collate_fn(self):
        if get_parallel_state().sp_enabled:
            self._check_sequence_parallel()
        args: VeOmniArguments = self.base.args
        # `seq_classification` keeps each packed sample's first label: that boundary rule
        # protects next-token targets, and here it would drop a masked token's label
        # while leaving its `loss_weight`. It also stops the SP stage shifting labels.
        # Pixels stay whole under SP, unlike the patch-sharded ViTs the default assumes:
        # every rank encodes every image and keeps the slots in its shard (see
        # `ulysses_local_image_slots`), which needs the default full-length `image_mask`.
        self.base.collate_fn = DllmCollator(
            data_collate_info={
                "loss_weight": DataCollateInfo(-1, True, 0, 1),
                "answer_mask": DataCollateInfo(-1, True, 0, 1),
                "pixel_values": DataCollateInfo(0, False, None, None),
            },
            pad_to_length=args.train.pad_to_length,
            seq_classification=True,
        )

    def _check_sequence_parallel(self) -> None:
        """Refuse the SP configurations whose numbers would come out wrong rather than crash.

        Ulysses rides on `flash_attention_forward`'s all-to-all, which only the
        `veomni_flash_*_with_sp` backends run: sdpa or eager would attend within the
        shard and train normally on a truncated context.
        """
        state = get_parallel_state()
        if state.cp_size > 1:
            raise ValueError(
                f"DllmTrainer supports Ulysses sequence parallelism only, got `cp_size: {state.cp_size}`. "
                "Use `model.accelerator.ulysses_size` instead."
            )
        model_config = self.base.model.model_config
        if model_config.model_type not in SEQUENCE_PARALLEL_MODEL_TYPES:
            raise ValueError(
                f"DllmTrainer has no sequence-parallel path for `{model_config.model_type}` yet (supported: "
                f"{', '.join(SEQUENCE_PARALLEL_MODEL_TYPES)}): its loss, image slots or MoE statistics have not "
                "been checked against a sequence shard. Set `model.accelerator.ulysses_size: 1`."
            )
        attn_implementation = model_config._attn_implementation
        if attn_implementation not in ULYSSES_ATTENTION_TYPES:
            raise ValueError(
                f"Sequence parallelism needs a `veomni_flash_attention_*_with_sp` backend for its all-to-all, "
                f"got `{attn_implementation}`. Set `model.ops_implementation.attn_implementation: flash_attention_2`."
            )
        vision_config = getattr(model_config, "vision_config", None)
        vision_attn = getattr(vision_config, "_attn_implementation", None)
        if vision_attn in ULYSSES_ATTENTION_TYPES:
            raise ValueError(
                f"The vision tower runs `{vision_attn}`, whose all-to-all would split a patch sequence every "
                "rank holds whole. The image models pin their tower to sdpa; something reset it."
            )
        heads, kv_heads = model_config.num_attention_heads, model_config.num_key_value_heads
        ulysses_size = state.ulysses_size
        kv_split = ulysses_size % kv_heads if ulysses_size > kv_heads else kv_heads % ulysses_size
        if heads % ulysses_size or kv_split:
            raise ValueError(
                f"`ulysses_size: {ulysses_size}` must divide num_attention_heads ({heads}), and divide "
                f"num_key_value_heads ({kv_heads}) or be a multiple of it, so that each rank holds whole heads."
            )

    @property
    def _corruption_epoch(self) -> DiffusionGemmaCorruptionEpoch:
        """Epoch the data transform keys its corruption RNG on.

        Lazy so the unit tests that construct the trainer with `__new__` still work.
        """
        holder = getattr(self, "_corruption_epoch_holder", None)
        if holder is None:
            holder = self._corruption_epoch_holder = DiffusionGemmaCorruptionEpoch()
        return holder

    def _warn_if_the_corruption_epoch_can_lag(self) -> None:
        """Flag the one configuration where the epoch push misses its fork.

        `DynamicBatchSizeDataLoader.load_state_dict` forks its workers from
        `on_train_begin`, before the epoch loop, then skips the rebuild on the resumed
        epoch -- so a run resumed into epoch N draws epoch 0's noise for that one
        epoch. Needs `dyn_bsz` *and* multiple epochs.
        """
        args: VeOmniArguments = self.base.args
        if args.train.dyn_bsz and args.train.num_train_epochs > 1:
            logger.warning_rank0(
                "`dyn_bsz` with num_train_epochs > 1: a run resumed mid-training will draw the "
                "first epoch's corruption noise for the epoch it resumes into, because dynamic "
                "batching forks its dataloader workers before the epoch hook runs. Set "
                "`dyn_bsz: false` for a bit-exact resume."
            )

    def on_micro_batches_ready(self, micro_batches: List[Dict[str, torch.Tensor]]) -> None:
        """Count the samples across the whole step, mirroring `micro_batches_token_len`.

        `dllm_mean_global_loss` divides by the step's total sample count, so normalizing
        by the current micro-batch alone would scale the accumulated gradient up by the
        number of micro-batches.

        Zero-weight samples are excluded, or they would silently shrink the gradient by
        the fraction of the batch they occupy. Counting rows instead would make the
        effective learning rate track a data-dependent, per-step ratio that nothing logs.

        Not floored at 1: `dllm_mean_global_loss` all-reduces this across data-parallel
        ranks and already handles a global zero, so a rank whose samples all degraded
        should report 0 and let the other ranks' samples carry the step.

        ``answer_mask`` and ``num_supervised_samples`` exist only for this count, so they
        are dropped from the batch before the model sees it.
        """
        self.micro_batches_num_samples = sum(self._micro_batch_sample_count(mb) for mb in micro_batches)
        self.num_micro_batches = len(micro_batches)
        for micro_batch in micro_batches:
            micro_batch.pop("answer_mask", None)
            micro_batch.pop("num_supervised_samples", None)

    def _micro_batch_sample_count(self, micro_batch: Dict[str, torch.Tensor]) -> int:
        """The count `DllmCollator` took before the SP slice, else `count_supervised_samples`."""
        count = micro_batch.get("num_supervised_samples")
        if count is not None:
            return int(count)
        return count_supervised_samples(micro_batch)

    def _supervised_sample_count(self, loss_weight: torch.Tensor) -> int:
        """Count rows that contribute to the loss."""
        return int((loss_weight.sum(dim=-1) > 0).sum())

    def preforward(self, micro_batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.base.preforward(micro_batch)

    #: Whether this trainer's model returns a loss that is already whole across the
    #: sequence-parallel group. The LLaDA family does not -- it sums over its own
    #: shard -- so `postforward` sums over the group itself; a subclass whose model
    #: reduces first sets it, see `DiffusionGemmaTrainer`.
    loss_spans_sp_group: bool = False

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Normalize by sample count rather than token count.

        The model returns the local weighted sum ``sum_i w_i * CE_i``. Token-count
        reduction would rescale it by the local/global ratio of loss-token counts; for a
        diffusion objective that ratio moves with each batch's corruption draw instead of
        holding the loss on a fixed per-sample footing.

        Under Ulysses the model's sum covers this rank's shard, and the shards of a
        group add up to the unsharded sum. A MoE ``aux_loss`` already arrives whole
        (see `ulysses_load_balancing_loss`).
        """
        local_weighted_sum = outputs.loss
        spans_sp_group = self.loss_spans_sp_group
        if not spans_sp_group:
            local_weighted_sum = sum_over_ulysses_shards(local_weighted_sum)
            # A CP shard is not summed here; `_check_sequence_parallel` refuses CP for this path.
            spans_sp_group = get_parallel_state().cp_size == 1
        loss = dllm_mean_global_loss(
            local_weighted_sum, self.micro_batches_num_samples, sum_spans_sp_group=spans_sp_group
        )
        loss_dict = {"foundation_loss": loss}
        # A MoE router's load-balancing term is already a per-token mean, so it joins
        # after the per-sample normalization rather than being divided by the sample count.
        # Dividing by the micro-batch count, with FSDP averaging over ranks, applies
        # `router_aux_loss_coef` once per step whatever the gradient accumulation.
        aux_loss = getattr(outputs, "aux_loss", None)
        if isinstance(aux_loss, torch.Tensor):
            aux_loss = (
                self.base.model.model_config.router_aux_loss_coef * aux_loss.to(loss.device) / self.num_micro_batches
            )
            loss_dict["aux_loss"] = aux_loss
            loss = loss + aux_loss
        return loss, loss_dict

    def forward_backward_step(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with use_parallel_state(self.base.model.parallel_state):
            micro_batch = self.preforward(micro_batch)

            with (
                self.base.model_fwd_context,
                set_batch_invariant_mode(self.base.args.train.enable_batch_invariant_mode),
            ):
                outputs: ModelOutput = self.base.model(**micro_batch, use_cache=False)

            loss, loss_dict = self.postforward(outputs, micro_batch)

            with (
                self.base.model_bwd_context,
                set_batch_invariant_mode(self.base.args.train.enable_batch_invariant_mode),
            ):
                loss.backward()

        del micro_batch
        return loss, loss_dict

    def on_train_begin(self):
        self.base.on_train_begin()

    def on_train_end(self):
        self.base.on_train_end()

    def on_epoch_begin(self):
        """Also tell the transform which epoch it is drawing corruption for.

        Without this a sample redraws the same mask in every epoch. It has to run before
        the epoch's iterator forks its workers, which is what carries the value into
        them; `_warn_if_the_corruption_epoch_can_lag` covers the one path where it does not.
        """
        self.base.on_epoch_begin()
        self._corruption_epoch.epoch = int(self.base.state.epoch)

    def on_epoch_end(self):
        self.base.on_epoch_end()

    def on_step_begin(self, micro_batches=None):
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None, aux_metrics=None):
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm, aux_metrics=aux_metrics)

    def train_step(
        self,
        data_iterator: Any,
    ) -> Dict[str, float]:
        args: VeOmniArguments = self.base.args
        runtime = self.base.model
        self.base.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.base._reset_async_activation_offload_if_enabled(runtime)
        self.on_step_begin(micro_batches=micro_batches)

        self.base.sync_before_train_step()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        with use_parallel_state(runtime.parallel_state):
            self.on_micro_batches_ready(micro_batches)
        num_micro_steps = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches):
            self.base.model_reshard(micro_step, num_micro_steps)
            self.base._configure_hsdp_allreduce(micro_step, num_micro_steps)
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            loss, loss_dict = self.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        with use_parallel_state(runtime.parallel_state):
            grad_norm = veomni_clip_grad_norm(
                runtime.model,
                args.model.optimizer.max_grad_norm,
                error_if_nonfinite=True,
            )

        runtime.optimizer.step()
        runtime.lr_scheduler.step()
        runtime.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def train(self):
        args: VeOmniArguments = self.base.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {self.base.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.base.start_epoch, args.train.num_train_epochs):
            if hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.base.state.epoch = epoch

            self.on_epoch_begin()

            self.base.data_iterator = VeOmniIter(
                self.base.train_dataloader, use_background_prefetcher=args.data.dataloader.use_background_prefetcher
            )

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(self.base.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
            if args.data.dataloader.use_background_prefetcher:
                self.base.data_iterator.stop()

        self.on_train_end()

        if args.data.dataloader.use_background_prefetcher:
            self.base.data_iterator.stop()

        synchronize()

        self.base.destroy_distributed()
