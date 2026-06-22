from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from veomni.utils.tensor_utils import naflatten, unflatten

from ....conversation import ConversationItem, is_dummy, maybe_merge_outputs
from ....generation_graph import FSM_SIGNAL_KEY
from ....module import CPUPreprocessor, pre_forward
from ...base.text_encoder.modulemixin import TextEncoderModuleMixin
from .chat_template import (
    JanusChatMarkers,
    _template_item,
    apply_janus_chat_template,
    merge_consecutive_text_parts,
    pack_text_input_ids,
    tokenize_template_parts,
)


SIGNAL_START_IMAGE_GEN = "start_image_gen"
SIGNAL_TEXT_DONE = "text_done"

# Sentinel written by JanusTextEncoderCPUPreprocessor onto every text part's meta
# so the thin ``encode_pre`` knows chat-template + tokenize already ran in the
# DataLoader worker (and otherwise falls back to the in-module path).
_OMNI_TOKENIZED = "_omni_tokenized"


class JanusTextEncoderCPUPreprocessor(CPUPreprocessor):
    """Worker-side chat-template + tokenize for the Janus text encoder.

    Holds only the (picklable) tokenizer + chat markers — never the model. Runs
    the same ``apply_janus_chat_template`` → tokenize → merge as the in-module
    path, but builds **CPU** tensors so it can execute in DataLoader workers and
    overlap with GPU compute. Mutates each sample in place and tags ``meta`` so
    ``encode_pre`` skips the redundant work.
    """

    def __init__(self, tokenizer: Any, chat_markers: JanusChatMarkers) -> None:
        self._tokenizer = tokenizer
        self._chat_markers = chat_markers

    def __call__(self, conversation_list: list[list[ConversationItem]]) -> None:
        for sample in conversation_list or []:
            if sample and sample[0].meta.get(_OMNI_TOKENIZED):
                continue  # idempotent: already processed
            parts = apply_janus_chat_template(sample, self._chat_markers)
            tokenize_template_parts(parts, self._tokenizer, device=None)
            parts = merge_consecutive_text_parts(parts)
            for part in parts:
                if part.type == "text":
                    part.meta[_OMNI_TOKENIZED] = True
            sample.clear()
            sample.extend(parts)


_JANUS_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
    "\n\n"
)
_JANUS_USER_PREFIX = "<|User|>: "
_JANUS_ASSISTANT_PREFIX = "\n\n<|Assistant|>:"


class JanusTextEncoderModuleMixin(TextEncoderModuleMixin):
    def init_omni_state(self) -> None:
        super().init_omni_state()
        self._chat_markers: Optional[Any] = None
        self._bos_token_id: Optional[int] = None
        self._boi_token_id: Optional[int] = None
        self._eoi_token_id: Optional[int] = None
        self._eos_token_id: Optional[int] = None
        self._pad_token_id: Optional[int] = None

    # training hooks
    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        input_ids = self._prepare_encode_inputs(self._conversation_carrier)
        return {"input_ids": input_ids}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        hidden_states, shift_labels = self._prepare_decode_inputs(self._conversation_carrier)
        return {"hidden_states": hidden_states, "shift_labels": shift_labels}

    def build_cpu_preprocessor(self) -> Optional[CPUPreprocessor]:
        """Worker-side chat-template + tokenize (see :class:`JanusTextEncoderCPUPreprocessor`).

        Returns ``None`` until the tokenizer (and thus chat markers) is loaded.
        """
        if self._chat_markers is None or getattr(self, "_tokenizer", None) is None:
            return None
        return JanusTextEncoderCPUPreprocessor(self._tokenizer, self._chat_markers)

    def _prepare_encode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> torch.Tensor:
        input_ids: list[torch.Tensor] = []
        self._encode_batch_shape = None
        for sample in conversation_list or []:
            # Fast path: the worker-side CPU preprocessor already ran chat-template
            # + tokenize (tagged on meta). Otherwise fall back to the in-module
            # device path (eager inference / no worker collator).
            if not (sample and sample[0].meta.get(_OMNI_TOKENIZED)):
                self._prepare_sample_training(sample)
            input_ids.extend(pack_text_input_ids(sample))
        # ``naflatten`` keeps the shape on CPU (avoids the post-forward D2H sync);
        # the flat ids may be CPU (worker path) or device (fallback) — move once.
        input_ids, self._encode_batch_shape = naflatten(input_ids)
        input_ids = input_ids.to(self.device, non_blocking=True)
        return input_ids

    def _prepare_sample_training(self, sample: list[ConversationItem]) -> None:
        # Build CPU tensors (mirrors the worker preprocessor); ``_prepare_encode_inputs``
        # moves the flat input_ids to device, and labels/attention_mask are moved at
        # their consumers. Keeping them on CPU here avoids per-segment device mixing.
        parts = apply_janus_chat_template(sample, self._chat_markers)
        tokenize_template_parts(parts, self._tokenizer, device=None)
        parts = merge_consecutive_text_parts(parts)
        sample.clear()
        sample.extend(parts)

    def _scatter_text_embeds(
        self,
        conversation_list: list[list[ConversationItem]],
        segment_embeds: list[torch.Tensor],
    ) -> None:
        dtype = self.dtype
        segment_embeds_iterator = iter(segment_embeds)
        for sample in conversation_list:
            for part in sample:
                if part.type != "text":
                    continue
                part.value = next(segment_embeds_iterator).to(device=self.device, dtype=dtype)
        if next(segment_embeds_iterator, None) is not None:
            raise RuntimeError(
                f"text segment count mismatch: scattered {len(segment_embeds)}, expected {len(conversation_list)}"
            )

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []

        for sample in conversation_list:
            for part in sample:
                if is_dummy(part):
                    continue
                hidden_states = part.value
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states.squeeze(0)
                if part.type == "text":
                    # ``labels`` rides in meta from tokenize (CPU). Keep all label
                    # chunks on CPU and move the concatenated result to device once
                    # below — avoids per-segment blocking H2D syncs.
                    labels = part.meta["labels"]
                    assert labels.shape[0] == hidden_states.shape[0]
                    hidden_states_chunks.append(hidden_states)
                    label_chunks.append(labels)
                elif part.type == "image":
                    hidden_states_chunks.append(hidden_states[-1:])
                    label_chunks.append(torch.full((1,), -100, dtype=torch.long))

        hidden_states = torch.cat(hidden_states_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)  # CPU

        labels = labels[..., 1:].contiguous()
        shift_labels = F.pad(labels, (0, 1), "constant", -100)
        # Single H2D move (labels are tiny host-side bookkeeping); the loss forward
        # orders after the backbone kernels on the GPU stream without a CPU stall.
        shift_labels = shift_labels.to(device=hidden_states.device, non_blocking=True)
        return hidden_states, shift_labels

    # inference hooks

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._bos_token_id = int(tokenizer.bos_token_id)
        self._eos_token_id = int(tokenizer.eos_token_id)
        self._pad_token_id = int(tokenizer.pad_token_id)
        self._boi_token_id = int(tokenizer.boi_token_id)
        self._eoi_token_id = int(tokenizer.eoi_token_id)
        self._chat_markers = JanusChatMarkers(
            bos_token=str(tokenizer.bos_token),
            eos_token=str(tokenizer.eos_token),
            boi_token=str(tokenizer.boi_token),
            eoi_token=str(tokenizer.eoi_token),
            system_prompt=_JANUS_SYSTEM_PROMPT,
            user_prefix=_JANUS_USER_PREFIX,
            assistant_prefix=_JANUS_ASSISTANT_PREFIX,
        )

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Dict[str, Any] = dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tail = conversation_list[-1]
        if tail.role == "user":
            if not self._bos_injected:
                conversation_list = apply_janus_chat_template(conversation_list, self._chat_markers)
                self._bos_injected = True

            conversation_list.append(_template_item("text", self._chat_markers.assistant_prefix, "assistant"))
            tokenize_template_parts(conversation_list, self._tokenizer, device=self.device)
            conversation_list = merge_consecutive_text_parts(conversation_list)
            for part in conversation_list:
                part.meta.pop("labels", None)
            input_ids = pack_text_input_ids(conversation_list)
            input_ids, _encode_batch_shape = naflatten(input_ids)
            inputs_embeds = self.encode(input_ids)["inputs_embeds"]
            self._scatter_text_embeds([conversation_list], unflatten(inputs_embeds, _encode_batch_shape))
            return {"conversation_list": conversation_list}

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
            maybe_merge_outputs(conversation_list)

            if output_token_id == self._boi_token_id:
                self._maybe_arm_cfg_for_image_gen(conversation_list, generation_kwargs, outputs)
                outputs[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
            elif output_token_id == self._eos_token_id:
                outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE

            if FSM_SIGNAL_KEY in outputs and outputs[FSM_SIGNAL_KEY] in (SIGNAL_TEXT_DONE, SIGNAL_START_IMAGE_GEN):
                outputs["generated"] = self._flush_text_generated(conversation_list)
            return outputs

        raise ValueError(f"Invalid type: {tail.type}")

    def emit_image_start(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert conversation_list[-1].type == "output" and conversation_list[-1].value.shape[0] == 1
        conversation_list.pop()
        output_token_id = self._boi_token_id
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
        output_token_id = self._eoi_token_id
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
    ) -> None:
        cfg_w = generation_kwargs.get("guidance_scale")
        if cfg_w is None or float(cfg_w) <= 1.0:
            return
        uncond = self._build_cfg_uncond_embeds(conversation_list)
        return uncond

    def _build_cfg_uncond_embeds(
        self,
        conversation_list: List[ConversationItem],
    ) -> Optional[torch.Tensor]:
        bos_id = self._bos_token_id
        pad_id = self._pad_token_id
        device = self.device
        dtype = self.dtype

        input_ids: List[int] = [bos_id]
        assert conversation_list[0].type == "text"
        input_ids.extend([pad_id] * (len(conversation_list[0].value) - 1))

        for part in conversation_list[1:]:
            if part.type == "output":
                break
            input_ids.extend([pad_id] * len(part.value))

        uncond_inputs_embeds = self._embed_tokens(torch.tensor(input_ids, dtype=torch.long, device=device)).to(
            dtype=dtype, device=device
        )
        return uncond_inputs_embeds


__all__ = ["JanusTextEncoderModuleMixin"]
