"""SeedOmni V2 graph hooks for BAGEL text token embeddings and CE loss."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from veomni.utils.tensor_utils import naflatten, unflatten

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....mixins.training_module_mixin import post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items, maybe_merge_outputs
from ...base.text_encoder.modulemixin import (
    InferenceMixin as BaseInferenceMixin,
)
from ...base.text_encoder.modulemixin import (
    TrainingMixin as BaseTrainingMixin,
)
from ...base.text_encoder.modulemixin import (
    VeOmniMixin as BaseVeOmniMixin,
)
from ..sources import BAGEL_FLOW_QUERY
from .chat_template import BagelChatTemplate
from .configuration import BagelTextEncoderConfig
from .processing import BagelTextEncoderPreprocessor, apply_image_marker


SIGNAL_START_IMAGE_GEN = "start_image_gen"
SIGNAL_TEXT_DONE = "text_done"

# Sentinel written by BagelTextEncoderPreprocessor onto every text item so
# encode_pre can skip tokenizer work already completed by a DataLoader worker.
_OMNI_TOKENIZED = "_omni_tokenized"


def prepare_bagel_encode_input_ids(
    conversation_list: Optional[List[List[ConversationItem]]],
    *,
    device: torch.device,
    fallback_input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.LongTensor | None]:
    """Pack CPU-preprocessed BAGEL text token ids (training + inference)."""
    if conversation_list is None:
        raise ValueError("BagelTextEncoder.prepare_bagel_encode_input_ids requires conversation_list.")

    input_ids: List[torch.Tensor] = []
    for item in iter_desired_items(conversation_list, types=["text"]):
        if is_dummy(item):
            continue
        if not item.meta.get(_OMNI_TOKENIZED):
            raise ValueError("BAGEL text encoder expects CPU-preprocessed text items.")
        token_ids = item.meta.get("input_ids")
        input_ids.append(token_ids.reshape(-1))

    if not input_ids:
        return fallback_input_ids, None

    flat_ids, batch_shape = naflatten(input_ids)
    return flat_ids.to(device, non_blocking=True), batch_shape


def scatter_bagel_text_embeds(
    conversation_list: List[List[ConversationItem]],
    segment_embeds: List[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Write BAGEL text embeds back onto conversation text items."""
    segment_embeds_iterator = iter(segment_embeds)
    for item in iter_desired_items(conversation_list, types=["text"]):
        if is_dummy(item):
            continue
        item.value = next(segment_embeds_iterator).to(device=device, dtype=dtype)
    if next(segment_embeds_iterator, None) is not None:
        raise RuntimeError("BAGEL text segment count mismatch during embed scatter.")


class TrainingMixin(BaseTrainingMixin):
    config: BagelTextEncoderConfig
    device: torch.device
    dtype: torch.dtype
    _chat_template: BagelChatTemplate | None
    _encode_batch_shape: torch.LongTensor | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._chat_template: Optional[BagelChatTemplate] = None

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[List[List[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().encode_pre(conversation_list, **kwargs)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().encode_post(**outputs)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[List[List[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().decode_pre(conversation_list, **kwargs)

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().decode_post(**outputs)

    def dummy_inputs(self, kind: str = "encode") -> Dict[str, torch.Tensor]:
        if kind == "encode":
            return {"input_ids": torch.zeros(1, device=self.device, dtype=torch.long)}
        return {
            "hidden_states": torch.zeros(1, int(self.config.hidden_size), device=self.device, dtype=self.dtype),
            "labels": torch.full((1,), -100, device=self.device, dtype=torch.long),
        }

    def _anchor_dummy_decode_inputs(
        self,
        conversation_list: Optional[List[List[ConversationItem]]],
        dummy: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Tie dummy CE loss to MoT hidden states without changing its value."""
        if conversation_list is None:
            return dummy

        anchor = None
        for item in iter_desired_items(
            conversation_list, types=["text", "image", "output"], roles=["user", "assistant"]
        ):
            value = item.value
            if not torch.is_tensor(value):
                continue
            if value.dim() == 3 and value.shape[0] == 1:
                value = value.squeeze(0)
            if value.dim() == 2 and int(value.shape[-1]) == int(self.config.hidden_size):
                anchor = value.to(device=self.device, dtype=self.dtype).sum() * 0.0
                break
        if anchor is None:
            return dummy

        return {
            "hidden_states": dummy["hidden_states"] + anchor,
            "labels": dummy["labels"],
        }

    def _prepare_encode_inputs(
        self,
        conversation_list: Optional[List[List[ConversationItem]]],
    ) -> torch.Tensor:
        if conversation_list is None:
            raise ValueError("BagelTextEncoder._prepare_encode_inputs requires conversation_list.")

        input_ids, self._encode_batch_shape = prepare_bagel_encode_input_ids(
            conversation_list,
            device=self.device,
            fallback_input_ids=self.dummy_inputs(kind="encode")["input_ids"],
        )
        return input_ids

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[List[List[ConversationItem]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if conversation_list is None:
            raise ValueError("BagelTextEncoder._prepare_decode_inputs requires conversation_list.")

        hidden_parts: List[torch.Tensor] = []
        shift_label_parts: List[torch.Tensor] = []
        for item in iter_desired_items(conversation_list, types=["text"]):
            if is_dummy(item):
                continue

            hidden_states = item.value
            labels = item.meta["labels"]
            if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
                hidden_states = hidden_states.squeeze(0)
            labels = labels.reshape(-1)
            if hidden_states.shape[0] != labels.shape[0]:
                raise ValueError(
                    "BAGEL text decode requires hidden-state and label lengths to match: "
                    f"got {hidden_states.shape[0]} and {labels.shape[0]}."
                )

            shift_labels = torch.full_like(labels, -100, dtype=torch.long)
            shift_labels[:-1] = labels[1:]
            hidden_parts.append(hidden_states.to(device=self.device, dtype=self.dtype))
            shift_label_parts.append(shift_labels)

        if hidden_parts:
            hidden_states = torch.cat(hidden_parts, dim=0)
            shift_labels = torch.cat(shift_label_parts, dim=0).to(device=hidden_states.device, non_blocking=True)
            if torch.any(shift_labels != -100):
                return hidden_states, shift_labels

        dummy = self._anchor_dummy_decode_inputs(conversation_list, self.dummy_inputs(kind="decode"))
        return dummy["hidden_states"], dummy["labels"]


class InferenceMixin(BaseInferenceMixin):
    config: BagelTextEncoderConfig
    device: torch.device
    dtype: torch.dtype
    _chat_template: BagelChatTemplate
    _encode_batch_shape: torch.LongTensor | None
    _text_token_cache: list[int]

    def encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """IDE stub — implemented on :class:`BagelTextEncoder` in ``modeling.py``."""
        ...

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tail = conversation_list[-1]
        batched = [conversation_list]

        if tail.type == "text" and tail.role == "user":
            input_ids, batch_shape = prepare_bagel_encode_input_ids(
                batched,
                device=self.device,
                fallback_input_ids=self.dummy_inputs(kind="encode")["input_ids"],
            )
            inputs_embeds = self.encode(input_ids)["inputs_embeds"]
            scatter_bagel_text_embeds(
                batched,
                unflatten(inputs_embeds, batch_shape),
                device=self.device,
                dtype=self.dtype,
            )
            return {"conversation_list": batched[0]}

        if tail.type == "output":
            outputs: Dict[str, Any] = {"conversation_list": batched[0]}
            hidden_states = tail.value
            if not torch.is_tensor(hidden_states):
                raise TypeError("BAGEL text generate expects the tail output value to be a hidden-state tensor.")
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            sampling = self._extract_sampling_kwargs(generation_kwargs, 1.0, 1.0, kwargs)

            output_token_id = self._sample_token(hidden_states, **sampling)
            self._text_token_cache.append(output_token_id)

            input_ids = self._token_id_tensor(output_token_id)
            inputs_embeds = self.encode(input_ids)["inputs_embeds"]
            tail.value = inputs_embeds.to(device=self.device, dtype=self.dtype)
            tail.meta["input_ids"] = input_ids.reshape(-1).detach()
            maybe_merge_outputs(batched[0])
            if output_token_id == self._chat_template.eos_token_id:
                outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
                outputs["generated"] = self._flush_text_generated(batched[0])
            return outputs

        outputs: Dict[str, Any] = {"conversation_list": batched[0]}
        infer_type = str((generation_kwargs or {}).get("infer_type", ""))
        if infer_type in {"infer_gen", "infer_edit"}:
            outputs[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
            return outputs

        raise ValueError(f"Invalid type: {tail.type}")

    def encode_image_markers(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del generation_kwargs, kwargs
        marker_embeds: Optional[torch.Tensor] = None
        for item in iter_desired_items(
            [conversation_list],
            types=["output"],
            sources=[BAGEL_FLOW_QUERY],
        ):
            if is_dummy(item):
                continue

            if marker_embeds is None:
                marker_ids = torch.tensor(
                    [[self._chat_template.vision_start_token_id, self._chat_template.vision_end_token_id]],
                    dtype=torch.long,
                    device=self.device,
                )
                marker_embeds = self.encode(marker_ids)["inputs_embeds"].to(device=self.device, dtype=self.dtype)
                marker_embeds = marker_embeds.squeeze(0)
            apply_image_marker(item, marker_embeds, device=self.device, dtype=self.dtype)

        return {"conversation_list": conversation_list}


class VeOmniMixin(TrainingMixin, InferenceMixin, BaseVeOmniMixin):
    """Training hooks for BAGEL text embeddings and CE loss."""

    preprocessor_class = BagelTextEncoderPreprocessor

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer
        self._chat_template = BagelChatTemplate(tokenizer)


__all__ = [
    "SIGNAL_START_IMAGE_GEN",
    "SIGNAL_TEXT_DONE",
    "VeOmniMixin",
]
