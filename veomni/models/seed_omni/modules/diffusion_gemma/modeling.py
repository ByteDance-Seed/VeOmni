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

"""DiffusionGemma as a single SeedOmni module.

One node, like the LLaDA families, but the node is two passes internally: the
encoder prefills the prompt and exports a KV cache, and the decoder denoises the
drawn canvas against it. Nothing crosses a module boundary -- the cache stays
inside this call -- so there is still nothing to hand to a sibling.

Only ``_loss`` leaves this module. The vendored forward also returns
``diffusion_loss`` and ``encoder_ar_loss`` as the two ``Eq. 13`` components, but
``OmniModel.forward`` collects nothing but the ``_loss`` key, so a component cannot
reach the trainer's log through this graph. Reporting them would need a
component-loss channel in the graph runtime, which does not exist yet. The summed
``_loss`` is unaffected -- the vendored forward already added the two terms
together.
"""

from __future__ import annotations

from typing import Any

from ....transformers.diffusion_gemma.generation import (
    EntropyBoundedSamplerConfig,
    block_autoregressive_generate,
)
from ....transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion
from ...graphs.generation_graph import FSM_SIGNAL_KEY
from ..module_modeling_base import PretrainedOmniModule
from .configuration import DiffusionGemmaOmniConfig
from .processing import DiffusionGemmaOmniPreprocessor


#: Emitted once the sampler returns. Block diffusion draws a whole canvas per step,
#: so there is nothing to loop over inside the module the way a mask predictor does.
SIGNAL_TEXT_DONE = "text_done"

#: Spellings of the token that closes a turn. ``eot_token_id`` is authoritative when
#: the checkpoint declares it; these literals cover a tokenizer that only stores it
#: in the chat template.
_END_OF_TURN_TOKENS = ("<turn|>", "<end_of_turn>")


class _VendoredForwardView:
    """Expose :class:`DiffusionGemmaOmni` to the sampler with the vendored ``forward``.

    ``block_autoregressive_generate`` calls ``model(...)`` and reads ``.logits``, plus
    ``model.model.encoder`` and ``model.config``. The module's own ``forward`` returns a
    loss dict for the training graph, so handing the sampler ``self`` would make it read
    ``.logits`` off a dict. Everything except ``forward`` is delegated unchanged, which
    keeps the weights, the config and the submodules the same objects.
    """

    def __init__(self, module: "DiffusionGemmaOmni") -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return DiffusionGemmaForBlockDiffusion.forward(self._module, *args, **kwargs)


class DiffusionGemmaOmni(DiffusionGemmaForBlockDiffusion, PretrainedOmniModule):
    """The vendored block-diffusion model, registered as an OmniModule.

    ``DiffusionGemmaForBlockDiffusion`` comes first so its ``forward`` (encoder
    prefill, decoder denoising, ``Eq. 13`` loss assembly) is the one the graph
    invokes; ``PretrainedOmniModule`` contributes the asset binding and the no-op
    inference defaults.

    ``preprocessor_class`` is what ``OmniProcessor.from_config`` reads to build the
    CPU worker; without it the sample reaches the graph unprocessed and ``batch_pre``
    finds no tensors to stack.
    """

    config_class = DiffusionGemmaOmniConfig
    preprocessor_class = DiffusionGemmaOmniPreprocessor
    base_model_prefix = "diffusion_gemma_omni"
    SIGNAL_TEXT_DONE = SIGNAL_TEXT_DONE

    def __init__(self, config: DiffusionGemmaOmniConfig):
        super().__init__(config)
        self._infer_done: bool = False

    def reset_local_inference_state(self) -> None:
        super().reset_local_inference_state()
        self._infer_done = False

    @property
    def tokenizer(self) -> Any:
        return getattr(self, "_tokenizer", None)

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def _end_of_turn_ids(self) -> list[int]:
        """Token ids that close the model's turn, so a canvas can stop early.

        ``tokenizer.eot_token_id`` is authoritative when the checkpoint declares it;
        the literal spellings cover a tokenizer that keeps it only in the chat
        template. ``eos_token`` is a different id on these checkpoints and does not
        end a turn -- treating it as one would truncate after the first sentence.
        """
        tokenizer = self.tokenizer
        if tokenizer is None:
            return []
        candidates: set[int] = set()
        eot = getattr(tokenizer, "eot_token_id", None)
        if eot is not None:
            candidates.add(int(eot))
        for token in _END_OF_TURN_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                candidates.add(int(token_id))
        return sorted(candidates)

    def _sampler_model(self) -> Any:
        """A view of this module whose ``forward`` returns the vendored output object.

        ``block_autoregressive_generate`` reads ``outputs.logits``, while this class's
        ``forward`` returns a plain dict of ``_loss`` for the training graph. Handing
        the sampler ``self`` would make it index the dict and fail; this proxy forwards
        to the vendored implementation so both callers see the shape each expects.
        """
        proxy = _VendoredForwardView(self)
        return proxy

    def generate(
        self,
        conversation_list: Any = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Denoise whole canvases against a prompt KV cache, in one FSM pass.

        The loop lives in ``block_autoregressive_generate`` -- prompt prefill, one
        entropy-bounded denoise per canvas, and Equation (4)'s re-encode of each
        committed canvas back into the cache. It is called rather than re-expressed
        as FSM states because a canvas is not a token: splitting it across passes
        would put the sampler's cache between the graph and the module, and the
        graph has nowhere to hold it.
        """
        del kwargs
        if self._infer_done:
            return {}
        gk = generation_kwargs or {}
        items = conversation_list or []
        if items and not isinstance(items[0], (list, tuple)):
            items = [items]
        prompt_inputs = None
        for sample in items:
            for item in sample:
                value = getattr(item, "value", None)
                if isinstance(value, dict) and "input_ids" in value:
                    prompt_inputs = {k: v.to(self.device) for k, v in value.items()}
                    break
            if prompt_inputs is not None:
                break
        if prompt_inputs is None:
            raise ValueError(
                "DiffusionGemmaOmni.generate: no encoded prompt in the carrier. The module's preprocessor "
                "must run before the FSM; check that `data_type: seedomni` and `processing.py` are wired."
            )

        sampler = EntropyBoundedSamplerConfig(
            num_steps=int(gk.get("num_steps", 48)),
            entropy_budget=float(gk.get("entropy_budget", 0.1)),
            stop_threshold=float(gk.get("stop_threshold", 0.005)),
            temperature_max=float(gk.get("temperature_max", 0.8)),
            temperature_min=float(gk.get("temperature_min", 0.4)),
        )
        generated = block_autoregressive_generate(
            self._sampler_model(),
            prompt_inputs,
            eos_token_ids=self._end_of_turn_ids(),
            config=sampler,
            max_blocks=int(gk.get("max_blocks", 4)),
        )
        self._infer_done = True
        token_ids = generated[0].tolist()
        tokenizer = self.tokenizer
        text = tokenizer.decode(token_ids, skip_special_tokens=True) if tokenizer is not None else ""
        return {
            FSM_SIGNAL_KEY: SIGNAL_TEXT_DONE,
            "generated": {"type": "text", "value": text, "meta": {"token_ids": token_ids}},
        }

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Run the encoder/decoder pair and surface ``_loss`` plus its two components.

        The graph passes the collated micro-batch as keyword arguments, so this
        forwards every canvas input straight to the vendored implementation. The
        component losses are passed through for logging only -- ``loss`` already
        contains them, so reporting them must not add them a second time.

        ``return_logits=False`` is forced here: this is the training entry, the batch
        always carries ``labels``, and the alternative is materialising a
        ``(B, K*C, vocab)`` fp32 logit tensor for a 262k vocabulary. With logits off the
        vendored forward scores the canvas through ``chunked_weighted_diffusion_loss``.
        """
        kwargs.pop("conversation_list", None)
        # `num_supervised_examples` is the collator's own bookkeeping for the
        # trainer's per-sample divisor; the module counts samples itself from the
        # carrier, so forwarding it would let two counters disagree.
        kwargs.pop("num_supervised_examples", None)
        kwargs.setdefault("return_logits", False)
        outputs = super().forward(**kwargs)
        result: dict[str, Any] = {}
        if outputs.loss is not None:
            result["_loss"] = outputs.loss
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            result["aux_loss"] = aux_loss
        # The self-conditioning pass asks the module for embeddings rather than a loss,
        # so that one call has no `_loss` to report. Surfacing the tensor here is what
        # lets `attach_self_conditioning` read it off the graph's dict return.
        embeddings = getattr(outputs, "self_conditioning_embeddings", None)
        if embeddings is not None:
            result["self_conditioning_embeddings"] = embeddings
        return result


__all__ = ["DiffusionGemmaOmni"]
