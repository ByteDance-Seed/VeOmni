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

"""LLaDA as a single SeedOmni module.

LLaDA is a mask predictor, not a decoder: it is supervised in place, over the
masked positions of the very sequence it is given. That makes it a
one-node graph -- there is no second module to hand anything to, and the CPU
preprocessor has already drawn the corruption before the batch reaches the GPU.

The forward returns ``_loss`` as the *local weighted sum* ``sum_i w_i * CE_i``
with ``w_i = 1/(t_s * N_s)`` baked in by the collator, exactly as
``LLaDAForCausalLM`` produces it. Dividing by the step's global sample count is
the trainer's job and stays where it is for the non-Omni path
(``dllm_mean_global_loss``); the Omni wrapper does the same division in
``LladaOmniTrainingMixin.post_forward`` using the count
``on_micro_batches_ready`` recorded for this step, and declares
``loss_is_step_mean`` so the trainer does not divide the backward again.

Inference is the same predictor run the other way: ``generate`` starts from a
fully masked answer span and commits the most confident masked positions each
FSM pass, so the graph's denoising state loops over one module until
``text_done``. Nothing is autoregressive -- there is no left-to-right cursor and
no KV cache, because every masked position is predicted from the same
bidirectional pass over the whole row.
"""

from __future__ import annotations

from typing import Any

import torch

from ....transformers.llada.modeling_llada import LLaDAForCausalLM
from ...graphs.generation_graph import FSM_SIGNAL_KEY
from ..module_modeling_base import PretrainedOmniModule
from .configuration import LladaOmniConfig
from .processing import LladaOmniPreprocessor


#: Emitted when the masked span is fully decoded (or truncated by EOS). The FSM
#: watches this to leave the denoising state.
SIGNAL_TEXT_DONE = "text_done"


class LladaOmni(LLaDAForCausalLM, PretrainedOmniModule):
    """The vendored LLaDA model, registered as an OmniModule.

    ``LLaDAForCausalLM`` comes first so its ``forward`` (which knows how to build
    the bidirectional mask and call ``dllm_lm_head_loss``) is the one the graph
    invokes; ``PretrainedOmniModule`` contributes the asset binding and the
    no-op inference defaults.

    ``preprocessor_class`` is what ``OmniProcessor.from_config`` reads to build
    the CPU worker; without it the sample would reach the graph unprocessed and
    ``batch_pre`` would find no tensors to stack.
    """

    config_class = LladaOmniConfig
    preprocessor_class = LladaOmniPreprocessor
    base_model_prefix = "llada_omni"
    _no_split_modules = ["LLaDADecoderLayer"]
    SIGNAL_TEXT_DONE = SIGNAL_TEXT_DONE

    def __init__(self, config: LladaOmniConfig):
        super().__init__(config)
        self._reset_diffusion_state()

    @property
    def tokenizer(self) -> Any:
        return getattr(self, "_tokenizer", None)

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    @property
    def _eos_token_id(self) -> int | None:
        """The id that closes the answer turn, read off the bound tokenizer.

        A property rather than a value cached by a setter: ``bind_module_assets``
        assigns ``_tokenizer`` directly (that is the name in
        ``MODULE_ASSET_ATTRS``), so a setter on a differently-named attribute
        never runs on the inference path and the loop would find no EOS at all.
        Reading it per pass keeps one source of truth.
        """
        tokenizer = getattr(self, "_tokenizer", None)
        eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        return int(eos) if eos is not None else None

    def _reset_diffusion_state(self) -> None:
        """Clear the per-request denoising buffers.

        A generation request is a sequence of FSM body passes over the same
        module, so the working tensor lives here rather than in ``generate``'s
        frame: ``_diff_x`` is ``None`` on the first pass and carries the
        partially-decoded row on every later one.
        """
        self._diff_x: torch.Tensor | None = None
        self._diff_prompt_len: int = 0
        self._diff_gen_len: int = 0
        self._diff_steps: int = 0
        self._diff_step: int = 0
        self._diff_num_transfer: list[int] = []
        self._diff_done: bool = False

    def reset_local_inference_state(self) -> None:
        super().reset_local_inference_state()
        self._reset_diffusion_state()

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # LLaDA owns attention and really does validate its backend: the
        # bidirectional mask has to be representable by the chosen kernel.
        return super()._check_and_adjust_attn_implementation(attn_implementation, **kwargs)

    @staticmethod
    def _make_num_transfer_schedule(gen_len: int, steps: int) -> list[int]:
        """How many masked positions each denoising step commits.

        Split as evenly as possible, the remainder going to the earliest steps.
        LLaDA's reference sampler uses exactly this shape: earlier steps commit
        one extra token so the later ones land on fully-decoded context.
        """
        base, rem = divmod(gen_len, steps)
        return [base + 1 if index < rem else base for index in range(steps)]

    def _prompt_ids(self, conversation_list: Any) -> torch.Tensor:
        """The clean prompt tokens the FSM extends with a masked span.

        The CPU preprocessor already templated and tokenized the request under
        ``inference=True`` (no corruption), so this reads its ``input_ids`` back
        instead of re-running the template on the GPU side: a second renderer
        would drift from the one the training path uses.

        The carrier is flat here -- ``[ConversationItem, ...]`` -- because
        ``OmniProcessor.preprocess`` unwraps the per-sample batch dimension for a
        single request, while the training path hands over one list per sample.
        Both shapes reach this method (a caller may pass either), so a bare item
        is wrapped before iterating rather than assumed away.
        """
        items = conversation_list or []
        if items and not isinstance(items[0], (list, tuple)):
            items = [items]
        for sample in items:
            for item in sample:
                value = getattr(item, "value", None)
                if isinstance(value, dict) and "input_ids" in value:
                    return value["input_ids"].to(device=self.device, dtype=torch.long).reshape(-1)
        raise ValueError(
            "LladaOmni.generate: no tokenized prompt in the carrier. The module's preprocessor must run "
            "before the FSM; check that `data_type: seedomni` and `processing.py` are both wired."
        )

    def _diffusion_init(
        self,
        conversation_list: Any,
        generation_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the initial row: the prompt, then a fully masked answer span.

        LLaDA is a mask predictor, so generation starts from the *whole* answer
        present as ``mask_token_id`` and denoises it in place -- there is no
        left-to-right cursor and no KV cache. ``gen_length`` therefore has to be
        stated up front; it is a budget, and a shorter answer ends on EOS.
        """
        gk = generation_kwargs or {}
        gen_length = int(gk.get("gen_length", 256))
        steps = max(1, int(gk.get("diffusion_steps", gk.get("steps", gen_length))))
        prompt_ids = self._prompt_ids(conversation_list)
        prompt_len = int(prompt_ids.numel())
        mask_id = int(self.config.mask_token_id)
        x = torch.full((prompt_len + gen_length,), mask_id, dtype=torch.long, device=self.device)
        x[:prompt_len] = prompt_ids

        self._diff_x = x
        self._diff_prompt_len = prompt_len
        self._diff_gen_len = gen_length
        self._diff_steps = steps
        self._diff_step = 0
        self._diff_num_transfer = self._make_num_transfer_schedule(gen_length, steps)
        self._diff_done = False
        return self._decode_forward()

    def _decode_forward(self) -> dict[str, Any]:
        """Run the mask predictor over the current row and hand back its logits.

        Returns the full row's logits so the next FSM pass can commit the
        highest-confidence positions. A mask predictor has no causal structure,
        so every masked position is predicted from the same bidirectional pass;
        there is nothing to cache between steps and the row is re-encoded whole.
        """
        x = self._diff_x
        if x is None:
            raise RuntimeError("LladaOmni: decode forward without an initialised row.")
        outputs = self.model(
            input_ids=x.unsqueeze(0),
            attention_mask=torch.ones_like(x, dtype=torch.long).unsqueeze(0),
            use_cache=False,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return {"llada_logits": logits}

    def _diffusion_step(
        self,
        llada_logits: torch.Tensor | None,
        generation_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Commit the most confident masked positions, then decide whether to stop.

        Selection is by confidence over the *answer span only*: the prompt tokens
        are given, and letting them compete would spend the per-step budget on
        positions that are already correct.

        A step commits a fixed count from the schedule rather than every position
        above a threshold. The threshold form (also used by LLaDA's reference
        sampler) can commit zero positions on a hard step, which stalls the FSM
        until ``max_new_tokens`` cuts it — a budget that must cover ``steps``
        passes, not tokens.
        """
        if llada_logits is None:
            raise RuntimeError("LladaOmni.generate expects `llada_logits` from the backbone after init.")
        logits = llada_logits.squeeze(0) if llada_logits.dim() == 3 else llada_logits
        start = self._diff_prompt_len
        span = logits[start:].float()
        xb = self._diff_x[start:]
        mask_id = int(self.config.mask_token_id)
        active = xb == mask_id

        x0 = span.argmax(dim=-1)
        probs = torch.softmax(span, dim=-1)
        confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        # Prompt and already-committed positions must never be re-masked, so they
        # are excluded from the ranking rather than merely outscored.
        neg_inf = torch.finfo(confidence.dtype).min
        confidence = torch.where(active, confidence, torch.full_like(confidence, neg_inf))

        num_to_transfer = (
            self._diff_num_transfer[self._diff_step]
            if self._diff_step < len(self._diff_num_transfer)
            else int(active.sum().item())
        )
        kk = min(num_to_transfer, int(active.sum().item()))
        if kk > 0:
            chosen = torch.topk(confidence, kk).indices
            xb = xb.clone()
            xb[chosen] = x0[chosen].to(xb.dtype)
            self._diff_x[start:] = xb

        self._diff_step += 1
        finished = self._diff_step >= self._diff_steps or not bool((self._diff_x[start:] == mask_id).any())
        if not finished:
            eos_id = self._eos_token_id
            if eos_id is not None:
                gen = self._diff_x[start:]
                eos_hits = (gen == eos_id).nonzero(as_tuple=True)[0]
                # An EOS is only a stop once everything before it is decoded;
                # a mask after it would still be part of the answer.
                if len(eos_hits) > 0 and bool((gen[: eos_hits[0]] != mask_id).all()):
                    finished = True
        if finished:
            return self._finish_diffusion()

        return self._decode_forward()

    def _finish_diffusion(self) -> dict[str, Any]:
        """Decode the answered span and emit it as the FSM's terminal signal."""
        gen_ids = self._diff_x[self._diff_prompt_len :].tolist()
        mask_id = int(self.config.mask_token_id)
        eos_id = self._eos_token_id
        # A row cut by the step budget still holds masks; dropping them keeps the
        # decode from rendering a run of mask tokens as text.
        if mask_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(mask_id)]
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        tokenizer = self.tokenizer
        text = tokenizer.decode(gen_ids, skip_special_tokens=True) if tokenizer is not None else ""
        self._diff_done = True
        return {
            FSM_SIGNAL_KEY: SIGNAL_TEXT_DONE,
            "generated": {"type": "text", "value": text, "meta": {"token_ids": gen_ids}},
        }

    def generate(
        self,
        conversation_list: Any = None,
        llada_logits: torch.Tensor | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """One FSM pass: initialise the masked row, or advance it one step.

        The FSM calls this repeatedly with the shared ``ctx``, so the first pass
        (``_diff_x is None``) lays out the prompt plus the masked answer span and
        every later pass commits more of it. The graph's ``default`` transition
        loops back here until ``text_done`` leaves the state.
        """
        del kwargs
        if self._diff_x is None:
            return self._diffusion_init(conversation_list, generation_kwargs)
        return self._diffusion_step(llada_logits, generation_kwargs)

    def finalize(self, *, ctx: dict[str, Any]) -> dict[str, Any]:
        """Emit whatever was decoded when the FSM ran out of steps.

        ``max_new_tokens`` bounds FSM passes, not tokens, so a budget smaller than
        the denoising schedule cuts the answer mid-way. Emitting the partial span
        is better than returning nothing -- but it is truncated, and the caller
        can tell because the mask was dropped rather than decoded away.
        """
        if self._diff_x is None:
            return {}
        self._diff_done = True
        return self._finish_diffusion()

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Run the mask predictor over the carrier's tensors and surface ``_loss``.

        The graph passes the collated micro-batch as keyword arguments, so this
        forwards ``input_ids`` / ``attention_mask`` / ``labels`` / ``loss_weight``
        straight to the vendored implementation and moves its scalar loss to the
        key the executor reads back.
        """
        # `logits_to_keep` and friends are not part of the training contract; a
        # training step supervises the masked positions and never materialises
        # full logits (the LM head runs on the selected rows inside the loss).
        kwargs.pop("conversation_list", None)
        outputs = super().forward(use_cache=False, **kwargs)
        result: dict[str, Any] = {}
        if outputs.loss is not None:
            result["_loss"] = outputs.loss
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            result["aux_loss"] = aux_loss
        return result


__all__ = ["LladaOmni"]
