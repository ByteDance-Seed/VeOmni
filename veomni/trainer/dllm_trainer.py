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

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers import AutoImageProcessor, GenerationConfig
from transformers.modeling_outputs import ModelOutput

from ..arguments import VeOmniArguments
from ..data import DiffusionGemmaCollator, DiffusionGemmaCorruptionEpoch, build_chat_template, build_data_transform
from ..data.chat_template import MultimodalChatTemplate
from ..data.data_collator import DataCollateInfo, MainCollator, PackingCollator
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.parallel_state import get_parallel_state, use_parallel_state
from ..distributed.sequence_parallel.loss import sum_over_ulysses_shards
from ..distributed.torch_compile import mark_compile_step_begin
from ..models.model_runtime import VeOmniModelRuntime
from ..models.transformers.diffusion_gemma import align_nemo
from ..ops.batch_invariant import set_batch_invariant_mode
from ..utils import helper
from ..utils.constants import IGNORE_INDEX
from ..utils.device import synchronize
from ..utils.dist_utils import all_reduce
from ..utils.loss_utils import dllm_mean_global_loss
from .base import BaseTrainer, VeOmniIter


__all__ = [
    "DllmCollator",
    "DllmModelRuntime",
    "DllmTrainer",
    "VeOmniArguments",
    "diffusion_gemma_sample_cost",
    "pad_decoder_canvases",
]

logger = helper.create_logger(__name__)

#: ``data.data_type`` that turns on DiffusionGemma's text+image path.
MULTIMODAL_DATA_TYPE = "diffusion_gemma_mm"

#: Models whose loss and router statistics `DllmTrainer` knows how to reduce over a sequence shard.
SEQUENCE_PARALLEL_MODEL_TYPES = ("llada", "llada_moe", "llada_v", "llada2_uni")
ULYSSES_ATTENTION_TYPES = (
    "veomni_flash_attention_2",
    "veomni_flash_attention_3",
    "veomni_flash_attention_4",
)


def diffusion_gemma_sample_cost(sample: Dict[str, torch.Tensor]) -> int:
    """What one sample costs a ``dyn_bsz`` token budget: `L + K*C`.

    The default measure is the sample's ``attention_mask`` sum, which for
    DiffusionGemma is the encoder row `L = P + K*C` -- the prompt plus every canvas.
    Nothing is missing from it on the encoder's side. What is missing is the
    *decoder*, which runs a second pass over the `K*C` canvas tokens against that
    cache, so a canvas token is paid for twice and a prompt token once. Two samples
    with the same `L` and different `K` therefore cost differently while the default
    measure calls them equal.

    Measured at production attention geometry with `scripts/bench_dg_step_cost.py`:
    holding `L` fixed and raising `K` from 1 to 7 moves the step time by 1.3-1.6x,
    and charging the canvas tokens twice is what flattens it. Least squares puts a
    canvas token higher still, at 2.6-2.7x a prompt token rather than 2x, because the
    decoder's attention over the cache grows faster than its token count -- but 2x is
    where the spread of time-per-budget-unit is smallest at the production expert
    count, and it needs no fitted constant to explain. Do not raise it to the fitted
    value without re-measuring: past 2x the measure over-charges the cases that
    dominate. Self-conditioning is a third decoder pass and so the term one would
    expect to matter here; sweeping it moved the fit by under 1%, so there is none.

    Memory is a separate question and the default measure already answers it: at
    fixed `L`, the transient allocation within a step moved by 4% across the same `K`
    sweep. So a budget that undercharges canvases misses a step time, not the memory
    the step holds. Transient allocation and not peak -- see constraint 32 on why the
    two are not interchangeable; no total was collected.

    The encoder half is counted off ``input_ids`` rather than the mask, because the
    packed collator concatenates the whole of ``input_ids`` -- so this is the row the
    model is handed, with no assumption that the mask happens to be all ones.
    """
    for key in ("input_ids", "decoder_input_ids"):
        if key not in sample:
            raise KeyError(
                f"Expected '{key}' in sample: a DiffusionGemma sample costs its encoder row plus "
                "its canvas tokens, and neither half can be defaulted -- a missing canvas count "
                "would silently cost the sample as if the decoder were free."
            )
    return int(sample["input_ids"].numel()) + int(sample["decoder_input_ids"].numel())


def _local_canvas_count(decoder_input_ids: torch.Tensor) -> int:
    return int(decoder_input_ids.shape[1]) if decoder_input_ids.ndim == 3 else 1


def pad_decoder_canvases(micro_batch: Dict[str, torch.Tensor], target_k: int) -> Dict[str, torch.Tensor]:
    """Pad a 2-D or short 3-D canvas batch to ``target_k`` decoder slots.

    Dummy slots are ``IGNORE_INDEX`` / zero-weight so they add no diffusion
    gradient. They are still decoded, which is the point: under
    ``canvas_batching: serial`` FSDP and expert-parallel collectives fire once per
    canvas, so every rank must loop the same number of times. ``flat_batch`` decodes in one
    pass and so cannot desynchronize that way; the padding is kept for it anyway --
    see ``DllmTrainer.preforward``.

    Never reached under ``flat_seq_pack``, where a dummy slot has no sample to belong
    to and the all-visible mask below would read across the packed row; see
    ``DllmTrainer._align_all_canvas_batch``.
    """
    if target_k <= 1:
        return micro_batch
    decoder_input_ids = micro_batch["decoder_input_ids"]
    if decoder_input_ids.ndim == 2:
        micro_batch["decoder_input_ids"] = decoder_input_ids.unsqueeze(1)
        if "labels" in micro_batch:
            micro_batch["labels"] = micro_batch["labels"].unsqueeze(1)
        if "loss_weight" in micro_batch:
            micro_batch["loss_weight"] = micro_batch["loss_weight"].unsqueeze(1)
        mask = micro_batch.get("decoder_attention_mask")
        if mask is not None and mask.ndim == 2:
            micro_batch["decoder_attention_mask"] = mask.unsqueeze(1)
        decoder_input_ids = micro_batch["decoder_input_ids"]
    current_k = int(decoder_input_ids.shape[1])
    if current_k >= target_k:
        return micro_batch
    extra = target_k - current_k
    micro_batch["decoder_input_ids"] = F.pad(decoder_input_ids, (0, 0, 0, extra), value=0)
    if "labels" in micro_batch:
        micro_batch["labels"] = F.pad(micro_batch["labels"], (0, 0, 0, extra), value=IGNORE_INDEX)
    if "loss_weight" in micro_batch:
        micro_batch["loss_weight"] = F.pad(micro_batch["loss_weight"], (0, 0, 0, extra), value=0)
    mask = micro_batch.get("decoder_attention_mask")
    if mask is not None and mask.ndim == 3:
        micro_batch["decoder_attention_mask"] = F.pad(mask, (0, 0, 0, extra), value=True)
    return micro_batch


def _first_eos_token_id(eos_token_id: int | list[int] | None) -> int:
    """Official CanvasChunker uses a single EOS id (Gemma4 special_tokens.EOS = 1)."""
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else 1
    return int(eos_token_id) if eos_token_id is not None else 1


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
    cannot stand in for (it has no ``encode``). For canvas models (`diffusion_gemma`),
    also exports ``generation_config.json`` beside the weights and folds
    ``per_expert_scale`` into the experts when requested.
    """

    def _build_model_assets(self) -> None:
        super()._build_model_assets()
        if (
            getattr(self, "chat_template", None) is not None
            and getattr(self, "processor", None) is not None
            and not isinstance(self.chat_template, MultimodalChatTemplate)
        ):
            self.chat_template = build_chat_template(self.args.chat_template, self.tokenizer)

        model_config = getattr(self, "model_config", None)
        is_canvas = (
            getattr(model_config, "model_type", None) == "diffusion_gemma"
            or getattr(model_config, "canvas_length", None) is not None
        )
        if is_canvas:
            model_path = self.args.model_path
            if model_path is None:
                generation_config = GenerationConfig.from_model_config(self.model_config)
            else:
                try:
                    generation_config = GenerationConfig.from_pretrained(model_path)
                except OSError:
                    logger.warning_rank0(
                        "The source model has no generation_config.json; exporting defaults derived from model config."
                    )
                    generation_config = GenerationConfig.from_model_config(self.model_config)
            self.model_assets.append(generation_config)

    def after_weights_land(self) -> None:
        """Last chance to change the parameter list, between loading and the optimizer.

        `_build_parallelized_model` ends by writing the checkpoint into the shards, and
        `_build_optimizer` reads `requires_grad` and takes a moment per trainable
        tensor. A runtime that reparameterizes the model has to do it in this gap:
        earlier and there are no values to fold, later and the optimizer is already
        holding state for a parameter that no longer moves.

        Folds `per_expert_scale` into the experts when the attribution switch asks for
        it (off by default; see `align_nemo`).
        """
        if hasattr(self, "model") and self.model is not None:
            align_nemo.fold_per_expert_scale(self.model)

    def _build_optimizer(self, param_groups: List[Dict[str, Any]] | None = None) -> None:
        self.after_weights_land()
        super()._build_optimizer(param_groups=param_groups)


class DllmTrainer:
    """Unified discrete-diffusion LM trainer (LLaDA family and DiffusionGemma).

    Composes `BaseTrainer`.
    Replaces token-count loss reduction with a per-sample mean (`dllm_mean_global_loss`).
    Automatically dispatches between single-stream masked diffusion (LLaDA / LLaDA-MoE /
    LLaDA-V / LLaDA2.0-Uni) and encoder-decoder canvas diffusion (`DiffusionGemma`).
    """

    base: BaseTrainer

    #: Set per step by `on_micro_batches_ready`; only 1 for a forward outside `train_step`.
    micro_batches_num_samples: int = 1
    #: Set per step by `on_micro_batches_ready` alongside the sample count.
    num_micro_batches: int = 1
    #: The image preprocessing for `data_type: llada_v` / `llada2_uni`; ``None`` for text-only data.
    image_processor: Any = None
    #: Position within the step that the self-conditioning draw is keyed on; reset by
    #: `on_micro_batches_ready` and advanced by each draw.
    _self_conditioning_micro_step: int = 0

    def __init__(self, args: VeOmniArguments):
        if args.train.channel_loss.enable:
            raise ValueError(
                "train.channel_loss is only supported by causal-LM trainers; DllmTrainer uses diffusion objectives."
            )
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

    @property
    def _is_canvas_model(self) -> bool:
        model = getattr(getattr(self, "base", None), "model", None)
        model_config = getattr(model, "model_config", None)
        return (
            getattr(model_config, "model_type", None) == "diffusion_gemma"
            or getattr(model_config, "canvas_length", None) is not None
            or getattr(model_config, "canvas_batching", None) is not None
        )

    @property
    def _is_multimodal(self) -> bool:
        args = getattr(getattr(self, "base", None), "args", None)
        data = getattr(args, "data", None)
        return getattr(data, "data_type", None) == MULTIMODAL_DATA_TYPE

    def _build_model_runtime(self) -> DllmModelRuntime:
        """Build (and own) this job's model. Override to swap in another runtime."""
        return DllmModelRuntime(self.base.args.model, "base", train=self.base.args.train)

    def _build_image_processor(self) -> None:
        """Load the image preprocessing that `data_type: llada_v` / `llada2_uni` / `diffusion_gemma_mm` need.

        Checked against the model's vision tower here, since a mismatch would only show
        up as image slot counts that disagree with the embeddings.
        """
        args: VeOmniArguments = self.base.args
        runtime = self.base.model
        model_config = runtime.model_config
        if self._is_multimodal:
            if runtime.processor is None:
                raise ValueError(
                    f"`data_type: {MULTIMODAL_DATA_TYPE}` needs the checkpoint's processor, and none loaded from "
                    f"`model.tokenizer_path: {args.model.tokenizer_path}`."
                )
            if runtime.chat_template is not None:
                raise ValueError(
                    f"`data_type: {MULTIMODAL_DATA_TYPE}` lays prompts out through the processor's own chat "
                    "template; leave `model.chat_template` unset."
                )
            return
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
                    "vision tower reads single {patch_size}px patches, one VQ token each."
                )
        else:
            return
        if runtime.chat_template is None:
            raise ValueError(
                f"`data_type: {args.data.data_type}` needs `model.chat_template`, and it needs the tokenizer at "
                f"{args.model.tokenizer_path} to load. A `processor_class` in its preprocessor_config.json makes "
                "AutoProcessor build that processor instead; reconvert the checkpoint so that "
                "preprocessor_config.json has no `processor_class` (scripts/convert_llada2uni_to_veomni.py does)."
            )
        if runtime.processor is None:
            runtime.model_assets.append(self.image_processor)

    def _build_data_transform(self):
        args: VeOmniArguments = self.base.args
        if self._is_canvas_model:
            model_config = self.base.model.model_config
            text_config = model_config.get_text_config()
            shared = {
                "max_seq_len": args.data.max_seq_len,
                "text_keys": args.data.text_keys,
                "canvas_length": model_config.canvas_length,
                "max_canvases": model_config.max_canvases,
                "canvas_start": model_config.canvas_start,
                "compute_all_canvas_df_loss": bool(getattr(model_config, "compute_all_canvas_df_loss", False)),
                "noise_eps": float(getattr(model_config, "training_noise_eps", 1e-3)),
                "vocab_size": text_config.vocab_size,
                "pad_token_id": text_config.pad_token_id or 0,
                "eos_token_id": _first_eos_token_id(text_config.eos_token_id),
                # Every rank calls `set_seed(train.seed)` with no rank offset, so the
                # transform cannot draw corruption from the process RNG without handing
                # every rank the same noise. It derives a per-sample generator from this
                # seed and the epoch instead; see `diffusion_gemma_noise_generator`.
                "corruption_seed": args.train.seed,
                "corruption_epoch": self._corruption_epoch,
            }
            self._warn_if_the_encoder_budget_is_tight(shared)
            self._warn_if_the_corruption_epoch_can_lag()
            if self._is_multimodal:
                extra = {"processor": self.base.model.processor}
            else:
                extra = {"tokenizer": self.base.model.tokenizer, "chat_template": self.base.model.chat_template}
            self.base.data_transform = build_data_transform(args.data.data_type, **shared, **extra)
            return

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
        args: VeOmniArguments = self.base.args
        if self._is_canvas_model:
            model_config = self.base.model.model_config
            text_config = model_config.get_text_config()
            pack_samples = model_config.canvas_batching == "flat_seq_pack"
            if pack_samples and self._is_multimodal:
                raise ValueError(
                    "`canvas_batching: flat_seq_pack` cannot pack the multimodal data types: the packed "
                    "encoder runs varlen attention kernels that carry no image block boundaries, so an "
                    "image span would be attended as text. Use `flat_seq` with images, or pack text only."
                )
            if args.train.dyn_bsz and not pack_samples:
                raise ValueError(
                    f"`dyn_bsz: true` needs `canvas_batching: flat_seq_pack`, got "
                    f"{model_config.canvas_batching!r}. Without packing the collator pads every prompt to "
                    "the longest in the micro-batch, so a batch selected to hold N tokens is handed a "
                    "row of up to `micro_batch_size * max(L)` instead -- the budget stops describing "
                    "what the step costs, in the direction that OOMs. Set `flat_seq_pack`, or "
                    "`dyn_bsz: false`."
                )
            # Read here rather than inside the collator: the collator is pickled into the
            # dataloader workers, and the global parallel state does not travel with it.
            self.base.collate_fn = DiffusionGemmaCollator(
                pad_token_id=text_config.pad_token_id or 0,
                pack_samples=pack_samples,
                sp_size=get_parallel_state().sp_size,
            )
            # Set beside the collator because it belongs to the same decision: what a batch
            # holds and what a sample costs have to agree, or the budget describes a batch
            # the collator does not build.
            self.base.dyn_bsz_length_fn = diffusion_gemma_sample_cost
            self._warn_if_the_budget_unit_does_not_match_train_size()
            self._check_the_budget_can_hold_a_sample()
            return

        if get_parallel_state().sp_enabled:
            self._check_sequence_parallel()
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

    def _warn_if_the_budget_unit_does_not_match_train_size(self) -> None:
        """Flag that `train_size` is now counted in a unit it may not be declared in.

        `compute_train_steps` estimates the number of optimizer steps as
        `train_size / (global_batch_size * max_seq_len)` -- that is, it assumes the run
        consumes `train_size` units of the same budget the buffer spends. The budget now
        charges a canvas token twice, so a dataset declared in plain tokens is spent
        between 1x and 2x faster than that estimate, the run ends early, and the LR
        schedule -- sized from `train_steps` -- never reaches its end. Nothing else
        notices: the loss curve looks like a run that simply stopped.

        A warning rather than a correction because the ratio depends on the data. It is
        `1 + K*C/L` per sample, which only the dataset knows.
        """
        args: VeOmniArguments = self.base.args
        if not args.train.dyn_bsz:
            return
        logger.warning_rank0(
            "`dyn_bsz` with DiffusionGemma costs a sample as `L + K*C` -- the encoder row plus the "
            "canvas tokens the decoder passes over a second time -- so `data.train_size` has to be "
            "declared in that same unit, not in plain tokens. Declared low, the run spends its budget "
            "faster than `compute_train_steps` predicts, stops before `train_steps`, and leaves the "
            "learning-rate schedule unfinished with nothing in the logs to say so. Multiply the "
            "dataset's token count by its mean `1 + K*C/L`, which is between 1 and 2."
        )

    def _check_the_budget_can_hold_a_sample(self) -> None:
        """A budget too small for one sample, which the two runtimes handle oppositely.

        The budget per micro-batch is `micro_batch_size * max_seq_len`, a unit that
        assumes a sample costs its row. A DiffusionGemma sample costs up to
        `max_seq_len + max_canvases * canvas_length`, so at `micro_batch_size: 1` --
        what the recipe ships -- the budget cannot hold even one worst-case sample.

        What happens then depends on `dyn_bsz_runtime`, in opposite directions, so this
        raises for one and warns for the other. `DynBszBuffer.get_samples` takes the
        first sample regardless (`force=True`): the dial reads as set, packs one sample
        per row exactly as `micro_batch_size` already would, and the only thing it
        changes is to admit rows over the budget it was meant to bound. That is wasted
        configuration, so it warns. `DynamicBatchingSizeDataset` instead **discards**
        every oversized sample, and `force_generate_long_sequence` cannot be turned on
        -- so a budget under the sample cost silently drops the training data, which is
        a refusal.

        Raising `micro_batch_size` is the fix in both cases, because that is the factor
        the budget is expressed in; under packing it scales the budget rather than the
        number of rows.
        """
        args: VeOmniArguments = self.base.args
        if not args.train.dyn_bsz:
            return
        model_config = self.base.model.model_config
        canvas_length = model_config.canvas_length or 0
        worst_case = args.data.max_seq_len + model_config.max_canvases * canvas_length
        budget = args.train.micro_batch_size * args.data.max_seq_len
        shape = (
            f"`dyn_bsz` has {budget} tokens per micro-batch (`micro_batch_size` "
            f"{args.train.micro_batch_size} x `max_seq_len` {args.data.max_seq_len}), and one "
            f"DiffusionGemma sample costs up to {worst_case} of them -- `max_seq_len` plus the "
            f"{model_config.max_canvases} x {canvas_length} canvas tokens the decoder passes over "
            "a second time. Raise `micro_batch_size` to at least "
            f"{-(-2 * worst_case // args.data.max_seq_len)}; under packing it scales the budget "
            "rather than the number of rows."
        )
        if budget >= 2 * worst_case:
            return
        if budget < worst_case and args.train.dyn_bsz_runtime != "main":
            raise ValueError(
                f"{shape} `dyn_bsz_runtime: {args.train.dyn_bsz_runtime}` discards a sample that "
                "does not fit rather than passing it through, and `force_generate_long_sequence` "
                "is not implemented -- so at this budget the wide samples are dropped and an epoch "
                "can produce nothing at all. `dyn_bsz_runtime: main` passes an oversized sample "
                "instead of dropping it."
            )
        oversized = (
            " And a sample that does not fit at all is passed through rather than dropped, so the"
            " budget stops bounding the row it was set to bound."
            if budget < worst_case
            else ""
        )
        logger.warning_rank0(
            f"{shape} Below twice the sample cost the budget packs one sample per row on the wide "
            f"samples, which is what `micro_batch_size` alone already does.{oversized}"
        )

    @staticmethod
    def _warn_if_the_encoder_budget_is_tight(transform_options: Dict[str, Any]) -> None:
        """Flag a `max_seq_len` that cannot hold every canvas plus a prompt.

        Eq. 13 puts all ``K`` canvases in the encoder, so an answer using every one of
        ``max_canvases`` occupies ``max_canvases * canvas_length`` tokens before the
        prompt gets a look in. Let that fill the budget and the two paths degrade
        differently and silently: the text-only one keeps a single token of prompt,
        training the model to continue text with no instruction, while the multimodal
        one drops later blocks rather than truncating image spans. Neither shows up
        in the loss.
        """
        canvas_length = transform_options["canvas_length"]
        max_seq_len = transform_options["max_seq_len"]
        deepest_context = transform_options["max_canvases"] * canvas_length
        if deepest_context + canvas_length > max_seq_len:
            logger.warning(
                f"data.max_seq_len ({max_seq_len}) leaves {max_seq_len - deepest_context} tokens for the prompt "
                f"once all {transform_options['max_canvases']} canvases occupy {deepest_context} tokens. Raise "
                f"max_seq_len to at least {deepest_context + canvas_length}, or lower model_config.max_canvases "
                f"({transform_options['max_canvases']})."
            )

    def _check_sequence_parallel(self) -> None:
        """Refuse the SP configurations whose numbers would come out wrong rather than crash.

        Ulysses rides on `flash_attention_forward`'s all-to-all, which only the
        `veomni_flash_*` backends run: sdpa or eager would attend within the
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
                f"Sequence parallelism needs a `veomni_flash_attention_*` backend for its all-to-all, "
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
        # Restart the within-step counter the self-conditioning draw is keyed on, so a
        # step's masks depend on the step alone and not on how many forwards preceded
        # it in this process. That is what makes them survive a resume.
        self._self_conditioning_micro_step = 0
        if self._print_ar_df_norm_enabled():
            if len(micro_batches) > 1:
                raise ValueError(
                    "`print_ar_df_norm` needs one micro-batch per step, and this step has "
                    f"{len(micro_batches)}. Its per-term gradient norms would silently cover more than "
                    "the micro-batch they are attributed to. Size the global batch so that each rank "
                    "gets one micro-batch, or turn `print_ar_df_norm` off."
                )
            self._component_grad_sq = {"encoder_ar_grad_norm": 0.0, "diffusion_grad_norm": 0.0}
            self._remaining_micro_steps = len(micro_batches)

    def _micro_batch_sample_count(self, micro_batch: Dict[str, torch.Tensor]) -> int:
        """The count `DllmCollator` / `DiffusionGemmaCollator` took, else `count_supervised_samples`."""
        loss_weight = micro_batch.get("loss_weight")
        if (
            "num_supervised_examples" in micro_batch
            or "decoder_input_ids" in micro_batch
            or self._is_canvas_model
            or (isinstance(loss_weight, torch.Tensor) and loss_weight.ndim == 3)
        ):
            return self._example_sample_count(micro_batch)
        count = micro_batch.get("num_supervised_samples")
        if count is not None:
            return int(count)
        return count_supervised_samples(micro_batch)

    def _example_sample_count(self, micro_batch: Dict[str, torch.Tensor]) -> int:
        """Count original examples, not the K decoder rows of an all-canvas sample.

        ``dllm_mean_global_loss`` divides by this. Counting expanded canvas rows
        would make a K-canvas answer weigh K times a one-canvas one, which is the
        opposite of averaging those canvases inside one example.
        """
        count = micro_batch.get("num_supervised_examples")
        if count is not None:
            return int(count.reshape(-1)[0].item()) if torch.is_tensor(count) else int(count)
        return self._supervised_sample_count(micro_batch["loss_weight"])

    def _supervised_sample_count(self, loss_weight: torch.Tensor) -> int:
        """Count rows (or canvases in packed/stacked rows) that contribute to the loss."""
        if loss_weight.ndim == 3:
            return int((loss_weight.flatten(1).sum(dim=-1) > 0).sum())
        model = getattr(getattr(self, "base", None), "model", None)
        canvas_length = getattr(getattr(model, "model_config", None), "canvas_length", None)
        batch, seq = loss_weight.shape[0], loss_weight.shape[-1]
        if canvas_length and seq != canvas_length and seq % int(canvas_length) == 0:
            chunks = loss_weight.reshape(batch * (seq // int(canvas_length)), int(canvas_length))
            return int((chunks.sum(dim=-1) > 0).sum())
        return int((loss_weight.sum(dim=-1) > 0).sum())

    def _align_all_canvas_batch(self, micro_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pad the canvas axis to the step's global max K, except where K cannot be padded.

        `flat_seq_pack` is that exception, and not as a preference: a padded canvas
        belongs to no sample, and neither mask it could be given is legal. All-visible
        reads the whole packed row -- every other sample's prompt, and on the last
        canvases their clean answers -- while all-hidden leaves a run with no owner to
        take decoder positions from. `canvas_owner_segments` refuses both, so under
        packing the padding cannot be expressed at all rather than merely being waste.

        Skipping it costs nothing here for the reason `flat_batch` never needed it:
        the decoder runs one pass whatever K is, so no collective count depends on K.
        The branch is on the config, equal on every rank, so the ranks also agree
        about not reaching the all-reduce below.
        """
        if getattr(self.base.model.model_config, "canvas_batching", None) == "flat_seq_pack":
            return micro_batch
        local_k = _local_canvas_count(micro_batch["decoder_input_ids"])
        target_k = local_k
        if dist.is_available() and dist.is_initialized():
            target_k = int(all_reduce(local_k, op="max"))
        return pad_decoder_canvases(micro_batch, target_k)

    def _noisy_token_ids(self, micro_batch: Dict[str, Any]) -> torch.Tensor:
        return micro_batch["decoder_input_ids"]

    def preforward(self, micro_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the micro-batch for forward, attaching self-conditioning on canvas models.

        Every denoising step hands the model the previous step's prediction, so a model
        finetuned without that signal would meet it cold at inference. Here the signal is
        the model's own prediction for the *same* noisy sequence rather than for the
        previous, differently-noised one -- the standard self-conditioning approximation,
        which costs one extra forward instead of a second denoising step.

        Set `self_conditioning_prob` to 0 (or omit it) to skip the pass entirely.
        """
        micro_batch = self.base.preforward(micro_batch)
        if "decoder_input_ids" not in micro_batch:
            return micro_batch

        micro_batch = self._align_all_canvas_batch(micro_batch)
        probability = float(getattr(self.base.model.model_config, "self_conditioning_prob", 0.0) or 0.0)
        if not self.base.model.training or probability <= 0.0:
            return micro_batch

        condition = (
            torch.rand(
                self._self_conditioning_sample_count(micro_batch),
                generator=self._self_conditioning_generator(),
            )
            < probability
        )
        conditioning_batch = {k: v for k, v in micro_batch.items() if k not in ("labels", "loss_weight")}
        with torch.no_grad(), set_batch_invariant_mode(self.base.args.train.enable_batch_invariant_mode):
            embeddings = self.base.model(
                **conditioning_batch,
                use_cache=False,
                return_logits=False,
                return_self_conditioning_embeddings=True,
            ).self_conditioning_embeddings

        micro_batch["self_conditioning_embeddings"] = embeddings
        micro_batch["self_conditioning_mask"] = condition.to(device=embeddings.device)
        return micro_batch

    def _self_conditioning_generator(self) -> torch.Generator:
        """CPU generator for this micro-batch's Bernoulli, derived rather than carried.

        Keyed on ``(train.seed, global_step, micro-step)``. Seeding from `train.seed`
        alone and letting one long-lived generator advance is what the checkpoint
        cannot express: `CheckpointerCallback._extra_state` persists only the default
        CPU RNG, so a resumed run rewinds this stream to step 0 and replays the first
        steps' masks at step N. Deriving from the step counter instead makes the mask
        a pure function of where training is, which resume already restores.

        The default generators are not an option either: Triton compile / autotune
        consumes both the CUDA generator and the default CPU one, so a later
        `torch.rand` -- even on CPU -- disagrees across DP ranks that share a seed, and
        between attention backends on the same seed.
        """
        micro_step = self._self_conditioning_micro_step
        self._self_conditioning_micro_step = micro_step + 1
        digest = hashlib.blake2b(digest_size=8)
        for value in (int(self.base.args.train.seed), int(self.base.state.global_step), micro_step):
            digest.update((value & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest.digest(), "little"))
        return generator

    def _self_conditioning_sample_count(self, micro_batch: Dict[str, Any]) -> int:
        """One Bernoulli per canvas, including canvases packed along the sequence."""
        return self._decoder_sample_count(self._noisy_token_ids(micro_batch))

    def _decoder_sample_count(self, decoder_input_ids: torch.Tensor) -> int:
        """How many training samples sit in this decoder tensor."""
        if decoder_input_ids.ndim == 3:
            return int(decoder_input_ids.shape[0] * decoder_input_ids.shape[1])
        canvas_length = int(self.base.model.model_config.canvas_length)
        batch, seq = decoder_input_ids.shape[:2]
        if seq == canvas_length:
            return batch
        if canvas_length < 1 or seq % canvas_length != 0:
            raise ValueError(
                f"Packed decoder sequence length {seq} is not a multiple of canvas_length {canvas_length}."
            )
        return batch * (seq // canvas_length)

    #: Whether this trainer's model returns a loss that is already whole across the
    #: sequence-parallel group. The LLaDA family does not -- it sums over its own
    #: shard -- so `postforward` sums over the group itself; DiffusionGemma reduces
    #: inside the model (or replicates canvases across SP), which sets this automatically.
    loss_spans_sp_group: bool = False

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Normalize by sample count rather than token count.

        The model returns the local weighted sum ``sum_i w_i * CE_i``. Token-count
        reduction would rescale it by the local/global ratio of loss-token counts; for a
        diffusion objective that ratio moves with each batch's corruption draw instead of
        reflecting sample length.

        Under Ulysses sequence parallelism a LLaDA shard's sum covers only its slice of
        the row, while every shard reports the whole row's sample count -- so the sum is
        completed across the Ulysses group first, and `dllm_mean_global_loss` is told that
        its input already spans the group. DiffusionGemma's Eq. 13 terms arrive whole
        across the SP group (`diffusion_loss` runs on replicated canvases and
        `encoder_ar_loss` goes through `sum_over_shards` inside the model).
        """
        local_weighted_sum = outputs.loss
        spans_sp_group = (
            self.loss_spans_sp_group or getattr(outputs, "diffusion_loss", None) is not None or self._is_canvas_model
        )
        if not spans_sp_group and get_parallel_state().sp_enabled:
            local_weighted_sum = sum_over_ulysses_shards(local_weighted_sum)
            # A CP shard is not summed here; `_check_sequence_parallel` refuses CP for this path.
            spans_sp_group = get_parallel_state().cp_size == 1
        loss = dllm_mean_global_loss(
            local_weighted_sum, self.micro_batches_num_samples, sum_spans_sp_group=spans_sp_group
        )
        loss_dict: dict[str, torch.Tensor] = {"foundation_loss": loss}
        if getattr(outputs, "encoder_ar_loss", None) is not None:
            loss_dict["diffusion_loss"] = dllm_mean_global_loss(
                outputs.diffusion_loss, self.micro_batches_num_samples, sum_spans_sp_group=spans_sp_group
            )
            loss_dict["encoder_ar_loss"] = dllm_mean_global_loss(
                outputs.encoder_ar_loss, self.micro_batches_num_samples, sum_spans_sp_group=spans_sp_group
            )
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
            forward_kwargs: Dict[str, Any] = {"use_cache": False}
            if "decoder_input_ids" in micro_batch or self._is_canvas_model:
                forward_kwargs["return_logits"] = False

            with (
                self.base.model_fwd_context,
                set_batch_invariant_mode(self.base.args.train.enable_batch_invariant_mode),
            ):
                outputs: ModelOutput = self.base.model(**micro_batch, **forward_kwargs)

            loss, loss_dict = self.postforward(outputs, micro_batch)

            with (
                self.base.model_bwd_context,
                set_batch_invariant_mode(self.base.args.train.enable_batch_invariant_mode),
            ):
                self._backward(loss, loss_dict)

        del micro_batch
        return loss, loss_dict

    def _print_ar_df_norm_enabled(self) -> bool:
        model = getattr(getattr(self, "base", None), "model", None)
        model_config = getattr(model, "model_config", None)
        return bool(getattr(model_config, "print_ar_df_norm", False))

    def _backward(self, loss: torch.Tensor, loss_dict: dict[str, torch.Tensor]) -> None:
        """One backward of the loss, unless ``print_ar_df_norm`` splits it."""
        if self._print_ar_df_norm_enabled():
            self._backward_and_print_component_grad_norms(loss, loss_dict)
        else:
            loss.backward()

    def _clone_grads(self) -> dict[str, torch.Tensor]:
        """Stash this model's gradients by parameter *name*, never by position."""
        return {
            name: param.grad.detach().clone()
            for name, param in self.base.model.named_parameters()
            if param.grad is not None
        }

    def _add_grads(self, stashed: dict[str, torch.Tensor]) -> None:
        """Return a stash to the parameters it came from, or refuse to."""
        pending, rejected = [], []
        for name, param in self.base.model.named_parameters():
            grad = stashed.get(name)
            if grad is None:
                continue
            if grad.shape != param.shape or isinstance(grad, DTensor) is not isinstance(param, DTensor):
                rejected.append(name)
            elif isinstance(grad, DTensor) and (
                grad.placements != param.placements or grad.device_mesh != param.device_mesh
            ):
                rejected.append(name)
            else:
                pending.append((param, grad))

        dropped = sorted(set(stashed) - {name for name, _ in self.base.model.named_parameters()})
        if self._every_rank(bool(rejected) or bool(dropped)):
            raise RuntimeError(
                f"the split backward cannot return its gradient stash: {len(rejected)} parameters moved "
                f"since it was taken {sorted(rejected)[:5]}, {len(dropped)} no longer exist {dropped[:5]}. "
                "This step would optimize on part of Eq. 13."
            )

        for param, grad in pending:
            if param.grad is None:
                param.grad = grad
            else:
                param.grad.add_(grad)

    def _every_rank(self, flag: bool) -> bool:
        """Turn a rank-local condition into one every rank agrees on."""
        if not dist.is_available() or not dist.is_initialized():
            return flag
        return bool(all_reduce(int(flag), op="max", group=get_parallel_state().dp_group))

    def _backward_and_print_component_grad_norms(self, loss: torch.Tensor, loss_dict: dict[str, torch.Tensor]) -> None:
        """Two backwards when both terms require grad; otherwise one."""
        ar_loss = loss_dict.get("encoder_ar_loss")
        diffusion_loss = loss_dict.get("diffusion_loss")
        is_ar_loss_required = isinstance(ar_loss, torch.Tensor) and ar_loss.requires_grad
        is_diff_loss_required = isinstance(diffusion_loss, torch.Tensor) and diffusion_loss.requires_grad

        if is_ar_loss_required and is_diff_loss_required:
            ar_loss.backward(retain_graph=True)
            ar_norm = self.base.model.compute_grad_norm()
            ar_grads = self._clone_grads()
            self.base.model.zero_grad(set_to_none=True)
            diffusion_loss.backward()
            diff_norm = self.base.model.compute_grad_norm()
            self._add_grads(ar_grads)
        else:
            loss.backward()
            ar_norm = self.base.model.compute_grad_norm() if is_ar_loss_required else 0.0
            diff_norm = self.base.model.compute_grad_norm() if is_diff_loss_required else 0.0

        sq = getattr(self, "_component_grad_sq", None)
        if sq is None:
            sq = self._component_grad_sq = {"encoder_ar_grad_norm": 0.0, "diffusion_grad_norm": 0.0}
        sq["encoder_ar_grad_norm"] += ar_norm**2
        sq["diffusion_grad_norm"] += diff_norm**2
        remaining = getattr(self, "_remaining_micro_steps", 1) - 1
        self._remaining_micro_steps = remaining
        if remaining > 0:
            return
        device = loss.device
        for key, value in sq.items():
            loss_dict[key] = torch.tensor(value**0.5, device=device, dtype=torch.float32)
        if self._print_ar_df_norm_enabled():
            logger.info(
                f"encoder_ar_grad_norm={float(loss_dict['encoder_ar_grad_norm']):.6g} "
                f"diffusion_grad_norm={float(loss_dict['diffusion_grad_norm']):.6g}"
            )

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
            mark_compile_step_begin(getattr(runtime, "_veomni_compile_uses_cuda_graphs", False))
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
