"""Accelerated wrapper for the Qwen3-Omni talker — training graph hooks.

``generate`` already lives natively on :class:`~..modeling.Qwen3OmniTalker`;
this file owns the training side, whose whole job is assembly. Everything the
talker needs is produced by three other modules and reaches it by two different
routes, so the hooks below are where those routes are read back:

* the thinker's traces (``thinker_embed`` / ``thinker_hidden``) ride on the
  conversation carrier, scattered per item by ``qwen3omni_llm``;
* the target RVQ codes ride on the carrier too, stashed on the assistant audio
  item by ``qwen3omni_codec.encode``;
* the token-level prompt view (``TALKER_PROMPT``) rides on the carrier too,
  left there by ``qwen3omni_text_encoder.encode``. It describes the sample rather
  than any one row, so it sits on the sample's first packed row and is found by
  scanning — the batch dict is not an option, since every key in it is splatted
  into every later node's pre-hook as a named parameter.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .......distributed.parallel_state import get_parallel_state
from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ...carrier import is_thinker_input
from ...codec.modeling import TALKER_CODE_LABELS
from ...llm.modeling import THINKER_EMBED, THINKER_HIDDEN
from ...text_encoder.modeling import TALKER_PROMPT
from ..configuration import Qwen3OmniTalkerConfig
from ..modeling import Qwen3OmniTalker, collect_media_metadata


class TrainingMixin(TrainingModuleMixin):
    config: Qwen3OmniTalkerConfig

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[List[List[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if get_parallel_state().sp_size > 1:
            # The talker's own attention is not Ulysses-aware, and its config
            # propagates ``attn_implementation`` down to the decoder layers, so an
            # SP variant would all-to-all a sequence nobody sliced. The thinker's
            # traces have already been SP-gathered by the time they reach the
            # carrier, so slicing here would also have to be undone.
            raise NotImplementedError(
                "qwen3omni_talker: training under sequence parallel is not implemented (sp_size="
                f"{get_parallel_state().sp_size}). Run the talker graph with sp_size=1."
            )
        samples = conversation_list or []
        # Assembled behind a catch, and the outcome agreed on before it is
        # raised. Every check below is data-dependent, so one rank can find a
        # bad sample while its peers find none; raising there would leave the
        # peers blocking in the talker's all-gather until the NCCL timeout —
        # the very hang these checks exist to prevent.
        failure: Optional[str] = None
        thinker_embeds: List[torch.Tensor] = []
        thinker_hiddens: List[torch.Tensor] = []
        talker_prompts: List[Dict[str, Any]] = []
        target_codes: List[torch.Tensor] = []
        metadata: List[Dict[str, Any]] = []
        try:
            for sample in samples:
                # Every sample is scored, none skipped: ``forward`` all-gathers
                # the sharded talker layers once per utterance, so a
                # rank-dependent count would hang rather than fail.
                codes = _target_codes(sample)
                if codes is None:
                    if any(item.role == "assistant" and item.type == "audio" for item in sample):
                        raise ValueError(
                            "a sample carries assistant audio but no codes for it, so the audio was never "
                            "tokenised — qwen3omni_codec.encode must be a node in this graph, upstream of the "
                            "talker."
                        )
                    raise ValueError(
                        "a sample in this micro-batch has no assistant audio turn. This graph trains speech "
                        "output, so every sample needs one; a partly-speech data mix would give the ranks "
                        "different utterance counts, which hangs rather than fails."
                    )
                embed, hidden = _thinker_traces(sample)
                thinker_embeds.append(embed)
                thinker_hiddens.append(hidden)
                talker_prompts.append(_talker_prompt(sample))
                target_codes.append(codes)
                metadata.append(collect_media_metadata(sample))
        except ValueError as error:
            failure = str(error)

        _agree_on_the_batch(0 if failure else len(samples), failure, self.device)

        # Raw tensors only. ``build_prompt`` runs the talker's own projections, so
        # it belongs inside the FSDP-wrapped forward, not in this hook.
        return {
            "thinker_embeds": thinker_embeds,
            "thinker_hiddens": thinker_hiddens,
            "talker_prompts": talker_prompts,
            "target_codes": target_codes,
            "media_metadata": metadata,
        }

    @post_forward("forward")
    def forward_post(self, loss: torch.Tensor, **outputs: Any) -> Dict[str, Any]:
        del outputs
        # V2 single-loss protocol: the graph collects the node's loss under ``_loss``.
        return {"_loss": loss}


def _agree_on_the_batch(local_count: int, failure: Optional[str], device: torch.device) -> None:
    """Make every rank raise, or none, and on the same step.

    ``forward`` runs the talker once per utterance, and each run all-gathers the
    sharded talker layers, so a rank-dependent count deadlocks on the collective
    rather than raising. Two ways it can differ, and one collective covers both:
    a rank that rejected its micro-batch reports zero, and a rank whose batch is
    simply a different size reports that size. Either way the counts disagree
    and everyone raises — a rank-local failure would otherwise hang its peers
    for an NCCL timeout before saying anything.
    """
    parallel_state = get_parallel_state()
    group = parallel_state.fsdp_group if parallel_state.is_initialized else None
    if group is None:
        # Nobody to agree with, so there is nothing to defer: raise the local
        # failure directly. Without this a single-process run would swallow it
        # and go on to score a batch its caller had already rejected.
        if failure:
            raise ValueError(f"qwen3omni_talker: {failure}")
        return
    counts = torch.tensor([local_count, -local_count], dtype=torch.long, device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.MAX, group=group)
    highest, lowest = int(counts[0]), -int(counts[1])
    if highest != lowest:
        raise ValueError(
            f"qwen3omni_talker: ranks disagree on how many utterances to score (this rank {local_count}, "
            f"range {lowest}..{highest}). Each one all-gathers the talker's layers, so an uneven count hangs. "
            + (
                f"This rank rejected its own micro-batch: {failure}"
                if failure
                else "Another rank rejected its micro-batch, or the batch sizes differ — use a fixed micro "
                "batch size (dyn_bsz: false) and drop_last: true for this graph."
            )
        )
    if failure:
        raise ValueError(f"qwen3omni_talker: {failure}")


def _talker_prompt(sample: Sequence[ConversationItem]) -> Dict[str, Any]:
    """The token-level prompt view the text encoder left on this sample.

    Only that module can produce it: the talker rebuilds its 3D RoPE from the
    token ids, and by the time it runs the text rows hold hidden states — the ids
    are gone. ``tts_embeds`` likewise is a lookup in the thinker's vocabulary.
    """
    for item in sample:
        payload = item.meta.get(TALKER_PROMPT)
        if payload is not None:
            return payload
    raise ValueError(
        "qwen3omni_talker: this sample carries no talker prompt view. It comes from the text encoder, which only "
        "builds it when asked — set qwen3omni_text_encoder.publish_talker_prompt: true in this graph's modules "
        "yaml, and make sure its encode node runs before the talker."
    )


def _target_codes(sample: Sequence[ConversationItem]) -> Optional[torch.Tensor]:
    """The utterance the talker is taught to say, or ``None`` if there is none.

    Written by ``qwen3omni_codec.encode`` on the assistant audio item. More than
    one assistant audio turn in a sample would mean more than one utterance to
    supervise against a single prompt, which the talker's prefill cannot express.
    """
    found = [
        item.meta[TALKER_CODE_LABELS]
        for item in sample
        if item.role == "assistant" and item.type == "audio" and TALKER_CODE_LABELS in item.meta
    ]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            f"qwen3omni_talker: a sample carries {len(found)} assistant audio turns. The talker speaks one "
            f"utterance per prompt, so the data layer must split a multi-turn sample into one sample per turn."
        )
    return found[0]


def _thinker_traces(sample: Sequence[ConversationItem]) -> Tuple[torch.Tensor, torch.Tensor]:
    """The thinker's own view of the prompt, concatenated back into one sequence.

    ``qwen3omni_llm`` scatters a ``(rows, D)`` slice per item; the talker's
    ``build_prompt`` walks the whole sequence, so they are re-joined in carrier
    order here — over exactly the rows the backbone packed, which is also
    exactly what ``talker_prompt_segments`` measured.
    """
    embeds: List[torch.Tensor] = []
    hiddens: List[torch.Tensor] = []
    for item in sample:
        if not is_thinker_input(item):
            continue
        if THINKER_EMBED not in item.meta or THINKER_HIDDEN not in item.meta:
            raise ValueError(
                "qwen3omni_talker: the conversation carries no thinker traces, so there is nothing to build the "
                "talker's prefill from — set qwen3omni_llm.collect_thinker_states: true in this graph's modules "
                "yaml, and make sure the llm node runs before the talker."
            )
        embeds.append(item.meta[THINKER_EMBED])
        hiddens.append(item.meta[THINKER_HIDDEN])
    return torch.cat(embeds, dim=0).unsqueeze(0), torch.cat(hiddens, dim=0).unsqueeze(0)


class VeOmniMixin(BaseMixin, TrainingMixin):
    """``generate()`` already lives on the native :class:`Qwen3OmniTalker`."""


class Qwen3OmniTalkerAccelerated(VeOmniMixin, Qwen3OmniTalker):
    pass


__all__ = ["Qwen3OmniTalkerAccelerated", "TrainingMixin", "VeOmniMixin"]
