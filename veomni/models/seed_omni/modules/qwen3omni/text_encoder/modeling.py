"""Qwen3-Omni ``TextEncoder`` — ChatML templating + tokenize + wte / lm_head."""

from typing import Any, Dict, List, Optional

import torch

from ......utils.constants import IGNORE_INDEX
from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....utils.conversation import ConversationItem, maybe_merge_outputs
from ...base.llm_packing import item_token_len
from ...text_encoder.modeling import TextEncoder
from ..carrier import PromptSegment as PromptSegment
from ..carrier import is_thinker_input
from .configuration import Qwen3OmniTextEncoderConfig
from .processing import Qwen3OmniTextEncoderPreprocessor


SIGNAL_TEXT_DONE = "text_done"


# Where the training path leaves the talker's prompt view. On the carrier, not in
# the batch dict: ``OmniModel.forward`` merges every node's outputs into one dict
# that is splatted into every *later* node's pre-hook, and those hooks take named
# parameters -- an extra key reaches ``text_encoder.decode_pre`` as an unexpected
# keyword, and before that rides into the thinker's attention kwargs. Sample-level
# rather than per-item state, so it is written on the sample's first packed row
# and found by scanning (the same way the codec's code targets are).
TALKER_PROMPT = "qwen3omni_talker_prompt"


def talker_prompt_segments(conversation: List[ConversationItem]) -> List[PromptSegment]:
    """Per-item (role, type, length) for one sample.

    Must run while text rows still hold token ids: both they and the media rows
    answer ``size(0)`` with their row count, but once the embeds are scattered
    over them the *ids* are gone, and it is the ids the talker rebuilds its 3D
    RoPE from. Skips exactly the rows the backbone's packer skips, so segment
    ``i`` is the same row as trace slice ``i``.
    """
    return [
        PromptSegment(item.role, item.type, item_token_len(item)) for item in conversation if is_thinker_input(item)
    ]


def talker_supervised_mask(conversation: List[ConversationItem]) -> torch.Tensor:
    """Per-token: is this row's position one the thinker is scored on?

    This is what tells the talker where the prompt stops and the response the
    talker must speak begins. At inference nothing is supervised — the response
    has not been generated yet — and the talker falls back to "the prompt is
    everything I was given", which is right there. In training the response is
    already on the carrier, teacher-forced, so without this the speech-span
    prefix would swallow the very text the talker is supposed to say.

    Token granularity, not item granularity, and read from ``meta["labels"]``
    rather than ``meta["loss_mask"]``: ``tokenize`` *pops* ``loss_mask`` once it
    has expanded it into per-token labels, and ``merge_text_embeds`` then merges
    the masked ``<|im_start|>assistant\\n`` prefix into the supervised response
    as a single item. ``labels`` survives both (it is concatenated by the merge),
    and it is the only view that still resolves the boundary *inside* that
    merged item.

    Media rows are never supervised, so they contribute ``False``.
    """
    spans: List[torch.Tensor] = []
    for item in conversation:
        if not is_thinker_input(item):
            continue
        rows = item_token_len(item)
        labels = item.meta.get("labels")
        if labels is None:
            spans.append(torch.zeros(rows, dtype=torch.bool))
        else:
            spans.append((labels.reshape(-1) != IGNORE_INDEX).cpu())
    if not spans:
        return torch.zeros(0, dtype=torch.bool)
    return torch.cat(spans)


class Qwen3OmniTextEncoder(TextEncoder):
    """Qwen3-Omni thinker text encoder — owns the tokenizer, wte and lm_head.

    Image / video / audio items (already carrying embeds from their own encoder
    module) pass through ``encode`` untouched: they keep their ``(N, D)`` value,
    get wrapped by marker text rows, and the backbone splices them in by segment
    order. Only ``text`` rows are tokenized and embedded here.

    ``generate`` is the thinker's ChatML autoregression, keyed on eos /
    ``<|im_end|>``. When it fires ``text_done`` the generation graph moves on to
    the talker, so this signal is what separates the thinking phase from the
    speaking phase.

    Qwen3-Omni expects a default system turn ("You are Qwen, a virtual human
    developed by the Qwen Team…") that conditions its speech output. As in the
    V1 path (``veomni/data/data_transform.py``), the data layer owns that turn —
    the chat template here templates whatever roles the conversation carries and
    does not inject one.
    """

    config_class = Qwen3OmniTextEncoderConfig
    preprocessor_class = Qwen3OmniTextEncoderPreprocessor

    def __init__(self, config: Qwen3OmniTextEncoderConfig):
        super().__init__(config)
        self._prompt_input_ids: Optional[torch.Tensor] = None
        self._prompt_segments: List[PromptSegment] = []
        self._prompt_supervised: Optional[torch.Tensor] = None

    def reset_global_inference_state(self) -> None:
        # Global, not local: the local reset fires between AR phases within one
        # request, and the talker runs after the text phase ends — clearing there
        # would wipe the prompt view before the talker ever sees it.
        super().reset_global_inference_state()
        self._prompt_input_ids = None
        self._prompt_segments = []
        self._prompt_supervised = None

    def _encode_prompt(self, conversation_list: List[ConversationItem]) -> Dict[str, Any]:
        # Record the token-level view before the base overwrites text rows with
        # embeddings — after that the ids are gone and cannot be recovered.
        self._prompt_input_ids = self._chat_template.pack_talker_input_ids(conversation_list)
        self._prompt_segments = talker_prompt_segments(conversation_list)
        self._prompt_supervised = talker_supervised_mask(conversation_list)
        return super()._encode_prompt(conversation_list)

    def flush_talker_prompt(self, **kwargs: Any) -> Dict[str, Any]:
        """Hand the talker everything only the tokenizer / wte owner can supply.

        A generation-graph node, like the backbone's ``flush_thinker_states``
        (the FSM splats ``ctx`` into node methods, hence ``**kwargs``). Three
        things travel together because they all come from this module:

        * ``input_ids`` / ``segments`` — the token-level prompt view. The
          talker's ``forward`` computes 3D RoPE by scanning a token sequence,
          which the segment carrier does not otherwise hold.
        * ``tts_embeds`` — thinker embeddings of the three speech-span framing
          tokens, in ``(bos, eos, pad)`` order. They are thinker-vocabulary
          tokens, so only this module's ``embed_tokens`` can produce them.
        """
        del kwargs
        chat_template = self._chat_template
        if chat_template is None:
            raise RuntimeError(
                "qwen3omni_text_encoder: flush_talker_prompt needs the chat template (for the tts token ids), but it "
                "has not been attached yet — the tokenizer must be set on this module before the graph runs."
            )
        if self._prompt_input_ids is None:
            raise RuntimeError(
                "qwen3omni_text_encoder: flush_talker_prompt ran before the prompt was encoded, so there is no token "
                "view to hand the talker. The generation graph must reach this node after the thinking phase."
            )
        outputs = {
            "talker_prompt": self.talker_prompt_payload(
                self._prompt_input_ids, self._prompt_segments, self._prompt_supervised
            )
        }
        self._prompt_input_ids = None
        self._prompt_segments = []
        self._prompt_supervised = None
        return outputs

    def freeze_model(self) -> None:
        """Freeze the wte and the text head when ``config.freeze``.

        Whole-module, unlike ``qwen3_text_encoder``'s row-masked variant: that
        one keeps newly-added vision special-token rows trainable, while here
        the vocabulary is the shipped one and the talker's speech-span framing
        tokens are read through ``tts_embeds`` under ``no_grad``.
        """
        if self.config.freeze:
            self.requires_grad_(False)

    def talker_prompt_payload(
        self,
        input_ids: torch.Tensor,
        segments: List[PromptSegment],
        supervised: torch.Tensor,
        tts_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """The talker's prompt view, in one place for both graphs.

        Training reaches this from the ``encode`` training hook rather than from
        an FSM node, and the talker reads the result by parameter name either
        way, so the two paths must agree on the keys down to spelling — hence
        one definition instead of a copy per path.

        ``tts_embeds`` may be passed in when a caller builds several payloads at
        once: it is the same three-token lookup for every sample in a batch, and
        repeating it would issue one embedding collective per sample under the
        ``emb`` parallel group — a per-rank collective count that varies with the
        batch is a hang, not an error.
        """
        chat_template = self._chat_template
        if chat_template is None:
            raise RuntimeError(
                "qwen3omni_text_encoder: the talker's prompt view needs the chat template (for the tts token ids), "
                "but the tokenizer has not been attached to this module yet."
            )
        return {
            "input_ids": input_ids,
            "segments": segments,
            "supervised": supervised,
            "tts_embeds": self.tts_embeds() if tts_embeds is None else tts_embeds,
            "tts_pad_token_id": chat_template.tts_pad_token_id,
            "im_start_token_id": chat_template.im_start_token_id,
            "im_end_token_id": chat_template.im_end_token_id,
            "role_token_ids": chat_template.role_token_ids,
        }

    def tts_embeds(self) -> torch.Tensor:
        """Thinker embeddings of the speech-span framing tokens, in (bos, eos, pad) order.

        Detached: the talker's loss must not flow back into the thinker's
        embedding table through them. The understanding stack is frozen while
        the talker trains, and the thinker traces are detached for the same
        reason — this is the third route out of the thinker and it would
        otherwise be the one that leaks.
        """
        chat_template = self._chat_template
        tts_ids = torch.tensor(
            [[chat_template.tts_bos_token_id, chat_template.tts_eos_token_id, chat_template.tts_pad_token_id]],
            dtype=torch.long,
            device=self.device,
        )
        # ``self.embed_tokens``, not a raw table read: it is a
        # :class:`~veomni.distributed.emb_parallel.sharded_embedding.ShardedEmbedding`,
        # whose ``forward`` takes *global* ids and handles the ``emb`` extra-parallel
        # all-to-all itself (falling through to ``nn.Embedding`` when the table is
        # not split). There is no ``_embed_tokens`` on this class or its bases.
        with torch.no_grad():
            return self.embed_tokens(tts_ids)

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        tail = conversation_list[-1]
        if not self._prompt_encoded:
            # First step: the request was already templated + tokenized by the
            # inference CPU preprocessor (run before the FSM), so just embed + scatter.
            return self._encode_prompt(conversation_list)

        if tail.type == "output":
            outputs: Dict[str, Any] = {"conversation_list": conversation_list}
            hidden_states = tail.value
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            sampling = self._extract_sampling_kwargs(generation_kwargs, 1.0, 1.0, {})
            output_token_id = self._sample_token(hidden_states, **sampling)
            self._text_token_cache.append(output_token_id)
            input_ids = self._token_id_tensor(output_token_id)
            inputs_embeds = self.encode(input_ids)["inputs_embeds"]

            tail.value = inputs_embeds
            maybe_merge_outputs(conversation_list)

            if output_token_id in (self._chat_template.eos_token_id, self._chat_template.im_end_token_id):
                outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
                outputs["generated"] = self._flush_text_generated(conversation_list)
            return outputs

        raise ValueError(f"Invalid conversation tail type: {tail.type}")


__all__ = [
    "PromptSegment",
    "Qwen3OmniTextEncoder",
    "SIGNAL_TEXT_DONE",
    "talker_supervised_mask",
    "talker_prompt_segments",
]
