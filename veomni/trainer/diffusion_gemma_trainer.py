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
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers import GenerationConfig
from transformers.modeling_outputs import ModelOutput

from ..arguments import VeOmniArguments
from ..data import DiffusionGemmaCollator, build_data_transform
from ..distributed.parallel_state import get_parallel_state, use_parallel_state
from ..models.transformers.diffusion_gemma import align_nemo
from ..ops.batch_invariant import set_batch_invariant_mode
from ..utils import helper
from ..utils.constants import IGNORE_INDEX
from ..utils.dist_utils import all_reduce
from ..utils.loss_utils import dllm_mean_global_loss
from .dllm_trainer import DllmModelRuntime, DllmTrainer


__all__ = ["DiffusionGemmaModelRuntime", "DiffusionGemmaTrainer", "VeOmniArguments"]

logger = helper.create_logger(__name__)

#: ``data.data_type`` that turns on the text+image path.
MULTIMODAL_DATA_TYPE = "diffusion_gemma_mm"


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
    see ``DiffusionGemmaTrainer.preforward``.

    Never reached under ``flat_seq_pack``, where a dummy slot has no sample to belong
    to and the all-visible mask below would read across the packed row; see
    ``DiffusionGemmaTrainer._align_all_canvas_batch``.
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


class DiffusionGemmaModelRuntime(DllmModelRuntime):
    """`DllmModelRuntime` that folds the expert scale on request and exports a generation config."""

    def after_weights_land(self) -> None:
        """Fold `per_expert_scale` into the experts, when the switch asks for it.

        Off by default and an attribution knob, not a default: it puts VeOmni on NeMo
        AutoModel's parameterization so the two can be compared as a framework change
        rather than a different optimization problem. See `align_nemo`.
        """
        align_nemo.fold_per_expert_scale(self.model)

    def _build_model_assets(self) -> None:
        """Also export the sampler's ``generation_config.json`` beside the weights.

        Falls back to defaults derived from the model config when the source checkpoint
        has none, so an exported checkpoint can always be sampled from.
        """
        super()._build_model_assets()
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


class DiffusionGemmaTrainer(DllmTrainer):
    """`DllmTrainer` specialized for DiffusionGemma's encoder-decoder canvas.

    The data transform needs the canvas geometry and vocabulary size so it can lay out
    the canvas and sample uniform corruption noise. Images condition the canvas only
    through the encoder, so the text-only and text+image objectives -- selected by
    ``data.data_type`` -- differ solely in how the prompt is built, not in the collator
    or the loss. The text+image path also cannot reuse `VLMTrainer`: that one batches
    through the packing `MainCollator`, and DiffusionGemma's prompt and canvas are two
    separate sequence dimensions that packing cannot express.
    """

    #: Position within the step that the self-conditioning draw is keyed on; reset by
    #: `on_micro_batches_ready` and advanced by each draw.
    _self_conditioning_micro_step: int = 0

    #: Both terms of Eq. 13 arrive whole across the SP group, so the sample-count
    #: reduction needs no `sp_size` term -- the diffusion term runs on canvases SP
    #: replicates, and the AR term goes through `sum_over_shards` inside the model.
    #: See `dllm_mean_global_loss` for the cancellation this licenses.
    loss_spans_sp_group: bool = True

    @property
    def _is_multimodal(self) -> bool:
        return self.base.args.data.data_type == MULTIMODAL_DATA_TYPE

    def _build_collate_fn(self):
        args: VeOmniArguments = self.base.args
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

    def _build_model_runtime(self) -> DiffusionGemmaModelRuntime:
        return DiffusionGemmaModelRuntime(self.base.args.model, "base", train=self.base.args.train)

    def _build_image_processor(self) -> None:
        """Images reach DiffusionGemma through the checkpoint's own processor, not an image processor.

        The processor owns the image handling *and* the chat template on the text+image
        path (the transform drives its Jinja directly), so a VeOmni `ChatTemplate` named
        there would be built and silently ignored.
        """
        if not self._is_multimodal:
            return
        runtime = self.base.model
        if runtime.processor is None:
            raise ValueError(
                f"`data_type: {MULTIMODAL_DATA_TYPE}` needs the checkpoint's processor, and none loaded from "
                f"`model.tokenizer_path: {self.base.args.model.tokenizer_path}`."
            )
        if runtime.chat_template is not None:
            raise ValueError(
                f"`data_type: {MULTIMODAL_DATA_TYPE}` lays prompts out through the processor's own chat "
                "template; leave `model.chat_template` unset."
            )

    def _build_data_transform(self):
        args: VeOmniArguments = self.base.args
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

    def on_micro_batches_ready(self, micro_batches: List[Dict[str, torch.Tensor]]) -> None:
        super().on_micro_batches_ready(micro_batches)
        # Restart the within-step counter the self-conditioning draw is keyed on, so a
        # step's masks depend on the step alone and not on how many forwards preceded
        # it in this process. That is what makes them survive a resume.
        self._self_conditioning_micro_step = 0
        if self._print_ar_df_norm_enabled():
            if len(micro_batches) > 1:
                # The split backward has to start each micro-batch from no gradient at
                # all, and under FSDP2 it cannot tell whether it did. An accumulation
                # behind a parameter left unsharded -- the root group always, and every
                # parameter when `reshard_after_backward` is false -- is reachable
                # through neither the stash nor `zero_grad`, so the AR term would add
                # onto it and report two micro-batches as one. Refuse here, before the
                # first forward, rather than mid-step on a number nobody can check.
                raise ValueError(
                    "`print_ar_df_norm` needs one micro-batch per step, and this step has "
                    f"{len(micro_batches)}. Its per-term gradient norms would silently cover more than "
                    "the micro-batch they are attributed to. Size the global batch so that each rank "
                    "gets one micro-batch, or turn `print_ar_df_norm` off."
                )
            # One micro-batch by the check above, so the RSS below is over a single term
            # and the two norms add up to `grad_norm`'s split exactly.
            self._component_grad_sq = {"encoder_ar_grad_norm": 0.0, "diffusion_grad_norm": 0.0}
            self._remaining_micro_steps = len(micro_batches)

    def _micro_batch_sample_count(self, micro_batch: Dict[str, torch.Tensor]) -> int:
        # The encoder `position_ids` do not index the canvas axis `loss_weight` lives on.
        return self._example_sample_count(micro_batch)

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
        if self.base.model.model_config.canvas_batching == "flat_seq_pack":
            return micro_batch
        local_k = _local_canvas_count(micro_batch["decoder_input_ids"])
        target_k = local_k
        if dist.is_available() and dist.is_initialized():
            target_k = int(all_reduce(local_k, op="max"))
        return pad_decoder_canvases(micro_batch, target_k)

    def _noisy_token_ids(self, micro_batch: Dict[str, Any]) -> torch.Tensor:
        return micro_batch["decoder_input_ids"]

    def preforward(self, micro_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Attach the self-conditioning signal that the sampler feeds at inference.

        Every denoising step hands the model the previous step's prediction, so a model
        finetuned without that signal would meet it cold at inference. Here the signal is
        the model's own prediction for the *same* noisy sequence rather than for the
        previous, differently-noised one -- the standard self-conditioning approximation,
        which costs one extra forward instead of a second denoising step.

        The attached keys (`self_conditioning_embeddings` / `self_conditioning_mask`) are
        DiffusionGemma decoder kwargs, not a generic dLLM interface.

        Set `self_conditioning_prob` to 0 (or omit it) to skip the pass entirely.
        """
        micro_batch = super().preforward(micro_batch)
        # Under `canvas_batching: serial` the decoder runs once per stacked canvas, and
        # each pass issues FSDP all-gathers plus an EP all-to-all per MoE layer. Ranks
        # that drew different K (the first-step dump was 1/2/5/6/8) would then issue
        # different numbers of collectives and hang. Align to the step's max K *before*
        # the self-conditioning Bernoulli so every rank also draws the same number of
        # bits. `flat_batch` runs one pass whatever K is and so cannot hang that way, but the
        # alignment stays on for it too: nothing has run without it at scale, and it is
        # also what keeps the Bernoulli count equal across ranks.
        #
        # `flat_seq_pack` is the one mode this is skipped for, and has to be --
        # `_align_all_canvas_batch` says why. Its ranks then draw different numbers of
        # bits, which is neither a hang (the draw is a local CPU generator, not a
        # collective) nor a divergence worth keeping: the ranks are holding different
        # samples, so a shared bit was never applied to a shared canvas.
        micro_batch = self._align_all_canvas_batch(micro_batch)
        probability = float(getattr(self.base.model.model_config, "self_conditioning_prob", 0.0) or 0.0)
        if not self.base.model.training or probability <= 0.0:
            return micro_batch

        # Labels are dropped so the conditioning pass skips the loss; nothing else about
        # the batch changes, including the corruption draw.
        #
        # The pass runs whether or not anything in this micro-batch ends up conditioned.
        # Skipping it when the Bernoulli selects nothing looks like free savings and is
        # not: the pass carries an FSDP2 all-gather and, with `ep_size > 1`, an all-to-all
        # per MoE layer, so making it conditional on a per-rank random draw lets ranks
        # disagree about how many collectives to issue. That deadlocks.
        #
        # Drawn *before* the no_grad pass, on a dedicated generator rather than a
        # default one; `_self_conditioning_generator` explains both.
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
        between attention backends on the same seed. Symptom is a step-to-step
        grad-norm spike while the loss barely moves.

        Every input is identical on every rank, which keeps the existing property that
        DP ranks and attention backends walk the same sequence.
        """
        micro_step = self._self_conditioning_micro_step
        self._self_conditioning_micro_step = micro_step + 1
        # blake2b rather than arithmetic on the three counters: consecutive Mersenne
        # Twister seeds correlate in their first outputs, and a step draws only a
        # handful of bits. The CPU generator keeps only the seed's low 32 bits, so two
        # steps of a long run can land on the same mask, which costs nothing here.
        digest = hashlib.blake2b(digest_size=8)
        for value in (int(self.base.args.train.seed), int(self.base.state.global_step), micro_step):
            digest.update((value & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest.digest(), "little"))
        return generator

    def _self_conditioning_sample_count(self, micro_batch: Dict[str, Any]) -> int:
        """One Bernoulli per canvas, including canvases packed along the sequence.

        Packing concatenates canvases along the sequence with batch=1;
        `decoder_input_ids.shape[0]` would then turn `self_conditioning_prob` into an
        all-or-nothing per pack. At unpacked `micro_batch_size: 1` this is still one
        draw per step.
        """
        return self._decoder_sample_count(self._noisy_token_ids(micro_batch))

    def _decoder_sample_count(self, decoder_input_ids: torch.Tensor) -> int:
        """How many training samples sit in this decoder tensor.

        ``process_diffusion_gemma_example`` draws one canvas per example unless
        ``compute_all_canvas_df_loss`` stacks them as ``(batch, K, canvas_length)``.
        Unpacked batches are ``(batch, canvas_length)``. Packed batches concatenate
        whole canvases along the sequence, so ``(1, n_samples * canvas_length)`` is
        ``n_samples`` draws, not one. The stacked all-canvas layout is ``K`` draws
        per example, matching one Bernoulli per denoised canvas.
        """
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

    def _supervised_sample_count(self, loss_weight: torch.Tensor) -> int:
        """Count canvases that contribute to the loss, including packed rows.

        Tests construct this trainer with `__new__` and no `model_config`; without a
        canvas length the packed split is unavailable and a row is one sample.
        """
        if loss_weight.ndim == 3:
            return int((loss_weight.flatten(1).sum(dim=-1) > 0).sum())
        model = getattr(getattr(self, "base", None), "model", None)
        canvas_length = getattr(getattr(model, "model_config", None), "canvas_length", None)
        batch, seq = loss_weight.shape[0], loss_weight.shape[-1]
        if canvas_length and seq != canvas_length and seq % int(canvas_length) == 0:
            chunks = loss_weight.reshape(batch * (seq // int(canvas_length)), int(canvas_length))
            return int((chunks.sum(dim=-1) > 0).sum())
        return super()._supervised_sample_count(loss_weight)

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Same per-sample reduction, with the encoder-AR and canvas terms reported apart.

        Per-sample here means one *answer*, and one answer contributes one canvas'
        mean cross-entropy whatever its length -- the ``1/C`` of the paper's Eq. 13,
        which normalizes the drawn canvas alone.

        The encoder's next-token term is normalized the same way. Reporting them apart
        matters because they move for different reasons: the canvas term follows the
        corruption draw and the encoder term the prompt mix, and a run where one of
        them quietly stops improving looks no different in their sum.
        """
        loss, metrics = super().postforward(outputs, micro_batch)
        if outputs.encoder_ar_loss is not None:
            metrics["diffusion_loss"] = dllm_mean_global_loss(
                outputs.diffusion_loss, self.micro_batches_num_samples, sum_spans_sp_group=self.loss_spans_sp_group
            )
            metrics["encoder_ar_loss"] = dllm_mean_global_loss(
                outputs.encoder_ar_loss, self.micro_batches_num_samples, sum_spans_sp_group=self.loss_spans_sp_group
            )
        return loss, metrics

    def forward_backward_step(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with use_parallel_state(self.base.model.parallel_state):
            micro_batch = self.preforward(micro_batch)

            with (
                self.base.model_fwd_context,
                set_batch_invariant_mode(self.base.args.train.enable_batch_invariant_mode),
            ):
                outputs: ModelOutput = self.base.model(**micro_batch, use_cache=False, return_logits=False)

            loss, loss_dict = self.postforward(outputs, micro_batch)

            with (
                self.base.model_bwd_context,
                set_batch_invariant_mode(self.base.args.train.enable_batch_invariant_mode),
            ):
                self._backward(loss, loss_dict)

        del micro_batch
        return loss, loss_dict

    def _print_ar_df_norm_enabled(self) -> bool:
        return bool(getattr(self.base.model.model_config, "print_ar_df_norm", False))

    def _backward(self, loss: torch.Tensor, loss_dict: dict[str, torch.Tensor]) -> None:
        """One backward of the Eq. 13 sum, unless ``print_ar_df_norm`` splits it.

        Default is ``loss.backward()`` once. With the flag, the AR term is backwarded
        with ``retain_graph``, its sharded grads are cloned, then the diffusion term
        frees the graph; the two buffers are added so the optimizer still steps on
        ``grad(AR + diffusion)``. That is two backwards -- extra all-gathers with
        FSDP2/EP, same collectives on every rank -- and prints each term's param-grad
        L2 before clipping.
        """
        if self._print_ar_df_norm_enabled():
            self._backward_and_print_component_grad_norms(loss, loss_dict)
        else:
            loss.backward()

    def _clone_grads(self) -> dict[str, torch.Tensor]:
        """Stash this model's gradients by parameter *name*, never by position.

        The encoder's parameters are the decoder's objects, so `named_parameters`
        deduplicates them away -- until FSDP2 unshards, which gives each side its own
        copy and makes both names appear. On the toy model that is 244 names in the
        unsharded state against 131 in the sharded one, so a positional pairing holds
        only as long as the stash and the return happen in the same state. A name
        survives the difference; an index does not.
        """
        return {
            name: param.grad.detach().clone()
            for name, param in self.base.model.named_parameters()
            if param.grad is not None
        }

    def _add_grads(self, stashed: dict[str, torch.Tensor]) -> None:
        """Return a stash to the parameters it came from, or refuse to.

        A name is only the same gradient if the parameter is still laid out the way it
        was when the stash was taken. Matching the shape alone would let a differently
        placed DTensor through, and ``add_`` would answer that with an implicit
        redistribute -- a collective, from a line that reads like a local add.

        Everything is checked before anything is written. A half-applied stash is not a
        state the caller can retry from, and the check has to agree across ranks anyway.
        """
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
        """Turn a rank-local condition into one every rank agrees on.

        Raising on one rank alone strands the others in the next collective -- the
        grad-norm all-reduce is right there -- until the NCCL watchdog fires, which
        replaces the message with a timeout half an hour later. Reduced over the data
        parallel group, the same one the gradients themselves are reduced over, so this
        cannot outlive the ranks that run this backward.
        """
        if not dist.is_available() or not dist.is_initialized():
            return flag
        return bool(all_reduce(int(flag), op="max", group=get_parallel_state().dp_group))

    def _backward_and_print_component_grad_norms(self, loss: torch.Tensor, loss_dict: dict[str, torch.Tensor]) -> None:
        """Two backwards when both terms require grad; otherwise one.

        Autograd and FSDP/EP reductions are linear, so adding the AR stash onto the
        diffusion grads is ``grad(AR + diffusion)``. It relies on ``loss`` being that
        sum -- it backwards the two terms and never ``loss`` itself -- and on the step
        holding a single micro-batch, which ``on_micro_batches_ready`` is what enforces.
        """
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
