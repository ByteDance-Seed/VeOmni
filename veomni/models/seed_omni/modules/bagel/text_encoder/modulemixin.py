"""SeedOmni V2 graph hooks for BAGEL text token embeddings and CE loss."""

from typing import Any, Dict, Optional

import torch

from veomni.utils.tensor_utils import unflatten

from ....conversation import ConversationItem
from ....generation_graph import FSM_SIGNAL_KEY
from ....module import post_forward, pre_forward
from ...base.text_encoder.modulemixin import TextEncoderModuleMixin
from .processing import (
    apply_image_embed_markers,
    as_batched_inference_conversation,
    build_generated_text,
    image_embed_marker_items,
    output_hidden_tail,
    prepare_text_decode_inputs,
    prepare_text_encode_inputs,
    resolve_token_id,
    sampled_token_id,
    scatter_text_embeds,
    update_tail_with_generated_token,
)


SIGNAL_START_IMAGE_GEN = "start_image_gen"
SIGNAL_TEXT_DONE = "text_done"


class BagelTextEncoderModuleMixin(TextEncoderModuleMixin):
    """Training hooks for BAGEL text embeddings and CE loss."""

    tokenizer_class = object

    def init_omni_state(self) -> None:
        super().init_omni_state()
        self._eos_token_id: Optional[int] = None
        self._start_token_id: Optional[int] = None
        self._image_start_token_id: Optional[int] = None
        self._image_end_token_id: Optional[int] = None
        self._decode_has_valid_labels: bool = False
        self._encode_text_segment_count: int = 0

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self._eos_token_id = resolve_token_id(
            tokenizer,
            "<|im_end|>",
            fallback=int(eos_token_id) if eos_token_id is not None else None,
        )
        self._start_token_id = resolve_token_id(tokenizer, "<|im_start|>", fallback=self._eos_token_id)
        self._image_start_token_id = resolve_token_id(tokenizer, "<|vision_start|>", fallback=None)
        self._image_end_token_id = resolve_token_id(tokenizer, "<|vision_end|>", fallback=None)

    def prompt_encode(
        self,
        conversation_list: Optional[list[ConversationItem] | list[list[ConversationItem]]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del generation_kwargs, kwargs
        batched = as_batched_inference_conversation(conversation_list)
        dummy = self.dummy_inputs(kind="encode")
        input_ids, batch_shape, self._encode_text_segment_count = prepare_text_encode_inputs(
            batched,
            tokenizer=self._tokenizer,
            start_token_id=self._resolve_start_token_id(),
            eos_token_id=self._resolve_eos_token_id(),
            device=self.device,
            dummy_input_ids=dummy["input_ids"],
        )
        if self._encode_text_segment_count == 0:
            self._encode_batch_shape = None
            self._wrap_image_embeds_with_markers(batched)
            return {"conversation_list": conversation_list}

        self._encode_batch_shape = batch_shape
        inputs_embeds = self.encode(input_ids)["inputs_embeds"]
        self._scatter_text_embeds(batched, unflatten(inputs_embeds, batch_shape))
        self._wrap_image_embeds_with_markers(batched)
        self._encode_batch_shape = None
        return {"conversation_list": conversation_list}

    def encode_image_query_markers(
        self,
        conversation_list: Optional[list[ConversationItem] | list[list[ConversationItem]]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del generation_kwargs, kwargs
        batched = as_batched_inference_conversation(conversation_list)
        self._wrap_image_embeds_with_markers(batched, item_types={"image", "output"})
        return {"conversation_list": conversation_list}

    def token_generate(
        self,
        conversation_list: Optional[list[ConversationItem] | list[list[ConversationItem]]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        batched = as_batched_inference_conversation(conversation_list)
        if len(batched) != 1:
            raise ValueError("BAGEL text token_generate currently expects one inference conversation.")

        sample = batched[0]
        outputs: Dict[str, Any] = {"conversation_list": conversation_list}
        if self._should_start_image_generation(sample, generation_kwargs):
            outputs[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
            return outputs

        tail, hidden_states = output_hidden_tail(sample)
        sampling = self._extract_sampling_kwargs(generation_kwargs, 1.0, 1.0, kwargs)
        output_token_id = self._sample_text_token_id(hidden_states, sampling)

        if output_token_id == self._resolve_eos_token_id():
            outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
            return outputs
        if output_token_id == self._resolve_image_start_token_id():
            outputs[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
            return outputs

        self._text_token_cache.append(output_token_id)
        input_ids = self._token_id_tensor(output_token_id)
        inputs_embeds = self.encode(input_ids)["inputs_embeds"].to(device=self.device, dtype=self.dtype)
        update_tail_with_generated_token(
            sample,
            tail,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            device=self.device,
            dtype=self.dtype,
        )
        return outputs

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        dummy = self.dummy_inputs(kind="encode")
        input_ids, self._encode_batch_shape, self._encode_text_segment_count = prepare_text_encode_inputs(
            conversation_list,
            tokenizer=self._tokenizer,
            start_token_id=self._resolve_start_token_id(),
            eos_token_id=self._resolve_eos_token_id(),
            device=self.device,
            dummy_input_ids=dummy["input_ids"],
        )
        return {"input_ids": input_ids}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        dummy = self.dummy_inputs(kind="decode")
        hidden_states, labels = prepare_text_decode_inputs(
            conversation_list,
            device=self.device,
            dtype=self.dtype,
            dummy_hidden_states=dummy["hidden_states"],
            dummy_labels=dummy["labels"],
        )
        self._decode_has_valid_labels = bool(torch.any(labels != -100).item())
        return {"hidden_states": hidden_states, "labels": labels}

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        if self._decode_has_valid_labels:
            self._decode_has_valid_labels = False
            return super().decode_post(**outputs)

        self._decode_has_valid_labels = False
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("BagelTextEncoder.decode_post requires logits.")
        outputs = {"conversation_list": conversation, "_loss": logits.sum() * 0.0}
        return outputs

    def _scatter_text_embeds(
        self,
        conversation_list: list[list[ConversationItem]],
        segment_embeds: list[torch.Tensor],
    ) -> None:
        expected = self._encode_text_segment_count
        self._encode_text_segment_count = 0
        scatter_text_embeds(
            conversation_list,
            segment_embeds,
            expected=expected,
            device=self.device,
            dtype=self.dtype,
        )

    def _sample_text_token_id(self, hidden_states: torch.Tensor, sampling: Dict[str, Any]) -> int:
        return sampled_token_id(self._sample_token(hidden_states, **sampling))

    def _should_start_image_generation(
        self,
        sample: list[ConversationItem],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> bool:
        if str((generation_kwargs or {}).get("infer_mode", "")) != "gen":
            return False
        if not sample:
            return True
        tail = sample[-1]
        return tail.type != "output" or not torch.is_tensor(tail.value)

    def _wrap_image_embeds_with_markers(
        self,
        conversation_list: list[list[ConversationItem]],
        *,
        item_types: set[str] | None = None,
    ) -> None:
        if item_types is None:
            item_types = {"image"}
        image_items = image_embed_marker_items(conversation_list, item_types=item_types)
        if not image_items:
            return

        start_token_id = self._resolve_image_start_token_id()
        end_token_id = self._resolve_image_end_token_id()
        marker_ids = torch.tensor(
            [[start_token_id, end_token_id] for _ in image_items],
            dtype=torch.long,
            device=self.device,
        )
        marker_embeds = self.encode(marker_ids)["inputs_embeds"].to(device=self.device, dtype=self.dtype)
        apply_image_embed_markers(image_items, marker_embeds, device=self.device, dtype=self.dtype)

    def finalize(self, *, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not self._text_token_cache:
            return {}
        flushed = self._flush_text_generated(ctx["conversation_list"])
        if not flushed:
            return {}
        return {"generated": flushed}

    def _flush_text_generated(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]],
    ) -> Dict[str, Any]:
        token_ids = list(self._text_token_cache)
        self._text_token_cache.clear()
        return build_generated_text(conversation_list, tokenizer=self._tokenizer, token_ids=token_ids)

    def dummy_inputs(self, kind: str = "encode") -> Dict[str, torch.Tensor]:
        if kind == "decode":
            return {
                "hidden_states": torch.zeros(1, int(self.config.hidden_size), device=self.device, dtype=self.dtype),
                "labels": torch.full((1,), -100, device=self.device, dtype=torch.long),
            }
        return {"input_ids": torch.zeros(1, device=self.device, dtype=torch.long)}

    def _resolve_start_token_id(self) -> int:
        if self._start_token_id is not None:
            return int(self._start_token_id)
        resolved = resolve_token_id(self._tokenizer, "<|im_start|>", fallback=self._eos_token_id)
        if resolved is None:
            raise ValueError("Unable to resolve BAGEL start token id.")
        self._start_token_id = int(resolved)
        return int(resolved)

    def _resolve_eos_token_id(self) -> int:
        if self._eos_token_id is not None:
            return int(self._eos_token_id)
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        resolved = resolve_token_id(
            self._tokenizer,
            "<|im_end|>",
            fallback=int(eos_token_id) if eos_token_id is not None else None,
        )
        if resolved is None:
            raise ValueError("Unable to resolve BAGEL EOS token id.")
        self._eos_token_id = int(resolved)
        return int(self._eos_token_id)

    def _resolve_image_start_token_id(self) -> int:
        if self._image_start_token_id is not None:
            return int(self._image_start_token_id)
        resolved = resolve_token_id(self._tokenizer, "<|vision_start|>", fallback=None)
        if resolved is None:
            raise ValueError("Unable to resolve BAGEL vision start token id.")
        self._image_start_token_id = int(resolved)
        return int(resolved)

    def _resolve_image_end_token_id(self) -> int:
        if self._image_end_token_id is not None:
            return int(self._image_end_token_id)
        resolved = resolve_token_id(self._tokenizer, "<|vision_end|>", fallback=None)
        if resolved is None:
            raise ValueError("Unable to resolve BAGEL vision end token id.")
        self._image_end_token_id = int(resolved)
        return int(resolved)


__all__ = ["BagelTextEncoderModuleMixin", "SIGNAL_START_IMAGE_GEN", "SIGNAL_TEXT_DONE"]
