from typing import Any, Dict, List, Optional

import torch

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....utils.conversation import ConversationItem, maybe_merge_outputs
from ...text_encoder.modeling import TextEncoder
from .configuration import JanusTextEncoderConfig
from .processing import JanusTextEncoderPreprocessor


# Janus signal keys
SIGNAL_START_IMAGE_GEN = "start_image_gen"
SIGNAL_TEXT_DONE = "text_done"


class JanusTextEncoder(TextEncoder):
    """Janus ``TextEncoder`` — image-aware ChatML with ``<boi>`` / ``<eoi>`` emitters.

    Only the chat template and the T2I-aware ``generate`` FSM (BOS injection,
    ``<boi>`` / ``<eoi>`` signals, classifier-free guidance arming) are
    genuinely Janus-specific; accelerated-only behavior lives in
    ``accelerated.py``.
    """

    config_class = JanusTextEncoderConfig
    preprocessor_class = JanusTextEncoderPreprocessor

    def __init__(self, config: JanusTextEncoderConfig):
        super().__init__(config)

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tail = conversation_list[-1]
        if not self._prompt_encoded:
            # First step: the request was already templated + tokenized by the
            # inference CPU preprocessor (run before the FSM), so just embed + scatter.
            return self._encode_prompt(conversation_list)

        if tail.type == "output":
            outputs: Dict[str, Any] = {"conversation_list": conversation_list}
            hidden_states: torch.Tensor = tail.value
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            sampling = self._extract_sampling_kwargs(generation_kwargs, 1.0, 1.0, kwargs)
            output_token_id = self._sample_token(hidden_states, **sampling)
            self._text_token_cache.append(output_token_id)
            input_ids = self._token_id_tensor(output_token_id)
            inputs_embeds = self.encode(input_ids)["inputs_embeds"]

            tail.value = inputs_embeds
            # ``maybe_merge_outputs`` folds this row into the previous one and
            # drops it, so every write below has to go through the live tail.
            maybe_merge_outputs(conversation_list)
            current = conversation_list[-1]

            if output_token_id == self._chat_template.boi_token_id:
                # Same arming as ``emit_image_start``: the unconditional embeds go on
                # the item's meta so the VQVAE span runs with guidance.
                current.meta["cfg_uncond_inputs_embeds"] = self._maybe_arm_cfg_for_image_gen(
                    conversation_list, generation_kwargs
                )
                outputs[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
                # Flush the preceding text *without* sealing this ``<boi>`` row:
                # ``image_vq`` starts with ``janus_llama.generate``, which asserts the
                # tail is an ``output`` item and consumes ``value[-1:]`` as its input
                # embedding. Sealing would turn it into text and raise. ``<boi>``
                # itself belongs to the image span, not to the reported text.
                self._text_token_cache.pop()
                generated = self._flush_text_generated(conversation_list, seal=False)
                if generated:
                    outputs["generated"] = generated
            elif output_token_id == self._chat_template.eos_token_id:
                outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
                generated = self._flush_text_generated(conversation_list)
                if generated:
                    outputs["generated"] = generated
            return outputs

        raise ValueError(f"Invalid type: {tail.type}")

    def emit_image_start(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert conversation_list[-1].type == "output" and conversation_list[-1].value.shape[0] == 1
        conversation_list.pop()
        output_token_id = self._chat_template.boi_token_id
        input_ids = self._token_id_tensor(output_token_id)
        inputs_embeds = self.encode(input_ids)["inputs_embeds"]
        cfg_uncond_inputs_embeds = self._maybe_arm_cfg_for_image_gen(
            conversation_list,
            generation_kwargs,
        )
        conversation_list.append(
            ConversationItem(
                type="output",
                value=inputs_embeds,
                role="assistant",
                meta={
                    "cfg_uncond_inputs_embeds": cfg_uncond_inputs_embeds,
                },
            )
        )
        return {"conversation_list": conversation_list}

    def emit_image_end(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        assert conversation_list[-1].type == "output" and conversation_list[-1].value.shape[-2] == 1
        conversation_list.pop()
        output_token_id = self._chat_template.eoi_token_id
        input_ids = self._token_id_tensor(output_token_id)
        inputs_embeds = self.encode(input_ids)["inputs_embeds"]
        conversation_list.append(
            ConversationItem(
                type="output",
                value=inputs_embeds,
                role="assistant",
                meta={
                    "collapse_cfg": True,
                },
            )
        )
        return {"conversation_list": conversation_list}

    def _maybe_arm_cfg_for_image_gen(
        self,
        conversation_list: Optional[List[ConversationItem]],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """Unconditional embeds to carry into the image span, or ``None``.

        Classifier-free guidance only applies to image generation, so this is
        called at the two points that open a span (``emit_image_start`` and the
        sampled-``<boi>`` path in ``generate``). The caller stores the result on
        the output item's ``meta``.
        """
        cfg_w = (generation_kwargs or {}).get("guidance_scale")
        if cfg_w is None or float(cfg_w) <= 1.0:
            return None
        return self._build_cfg_uncond_embeds(conversation_list)

    def _build_cfg_uncond_embeds(
        self,
        conversation_list: List[ConversationItem],
    ) -> Optional[torch.Tensor]:
        """Unconditional prompt embeds, padded to the conditional prefix length.

        The conditional cache already holds every position generated so far; the
        unconditional one starts fresh. The two CFG branches must therefore enter
        the image span at the *same* sequence position, so the uncond side is
        padded out to the conditional prompt length with the pad token — matching
        what the upstream Janus implementation does for interleaved generation.
        """
        bos_id = self._chat_template.bos_token_id
        pad_id = self._chat_template.pad_token_id

        input_ids: List[int] = [bos_id]
        assert conversation_list[0].type == "text"
        input_ids.extend([pad_id] * (len(conversation_list[0].value) - 1))

        for part in conversation_list[1:]:
            if part.type == "output":
                break
            input_ids.extend([pad_id] * len(part.value))

        # Positions the conditional side has generated (each output row is one
        # token, already merged into a single row by ``maybe_merge_outputs``).
        generated = sum(
            int(part.value.size(-2)) if part.value is not None and hasattr(part.value, "size") else 0
            for part in conversation_list
            if part.type == "output"
        )
        if generated:
            input_ids.extend([pad_id] * generated)

        uncond_inputs_embeds = self.embed_tokens(torch.tensor(input_ids, dtype=torch.long, device=self.device))
        return uncond_inputs_embeds


__all__ = ["JanusTextEncoder", "SIGNAL_START_IMAGE_GEN", "SIGNAL_TEXT_DONE"]
