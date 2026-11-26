"""Qwen3-Omni talker as a SeedOmni module — thinker states to Mimi RVQ codes.

Autoregression is delegated to upstream: the talker's ``generate`` is plain
:class:`~transformers.GenerationMixin`, and the 16-group RVQ machinery lives in
its ``forward`` / ``prepare_inputs_for_generation`` (which runs a nested code
predictor for groups 2..16 each step). Reimplementing that loop inside the FSM
would buy nothing and risk a lot, so this module's real work is *assembling the
talker prompt* from the conversation carrier.

What upstream recovers by scanning token ids, V2 already knows per item, so the
assembly here is segment-driven:

* a ChatML turn is a run of same-role rows, so no ``<|im_start|>`` scan;
* multimodal positions are rows whose type is not ``text``, so no comparison
  against placeholder token ids;
* the token-id sequence is still needed, but only because the talker's 3D RoPE
  is computed from one — the text encoder rebuilds it (placeholders included)
  and passes it through ``ctx``.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# The patchgen-generated modeling deliberately excludes the talker (the thinker
# training path forwards through the thinker only, and instantiating the talker
# would drag its layers into the FSDP _no_split_modules aggregation), so take it
# from upstream. Talker training drives this class directly rather than through
# patchgen: what it needs is a teacher-forced pass, and upstream ships only the
# decode loop, so there is no upstream forward to patch.
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerForConditionalGeneration

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import reduce_sequence_parallel_loss
from ......utils.constants import IGNORE_INDEX
from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....utils.conversation import ConversationItem
from ...module_modeling_base import PretrainedOmniModule
from ..audio.processing import _OMNI_MEL_LEN as _MEL_LEN_KEY
from ..carrier import TALKER_CODES_KEY, is_thinker_input
from .configuration import Qwen3OmniTalkerConfig


SIGNAL_CODES_DONE = "codes_done"

# Codec-side special tokens that open the speech span, in wire order. Upstream
# writes this literal tuple inline; naming it keeps the text-side alignment
# below checkable: 6 codec tokens must line up with 4 tts_pad + tts_bos + the
# first generated text row.
_NUM_CODEC_PREFIX_TOKENS = 6
_NUM_TTS_PAD_ROWS = _NUM_CODEC_PREFIX_TOKENS - 2


def _chatml_turns(
    input_ids: torch.Tensor,
    total_length: int,
    im_start_token_id: int,
    role_token_ids: Dict[str, int],
) -> List[Tuple[str, int, int]]:
    """Split the sequence into ``(role, start, end)`` turns on ``<|im_start|>``.

    Deliberately *not* keyed on ``item.role``, even though the carrier has it:
    consecutive same-role turns would merge, and for the trailing assistant
    turn that is corrupting rather than harmless. A history assistant turn
    followed by the generation prompt would look like one long turn, so the
    speech-span prefix length would absorb the history and every projected row
    after it would shift. Upstream splits on ``<|im_start|>``; matching that
    keeps the two implementations from diverging on multi-turn prompts.

    The final turn deliberately extends to ``total_length`` (past the prompt,
    over the generated text) — that is the span the talker speaks.
    """
    starts = torch.nonzero(input_ids == im_start_token_id).reshape(-1).tolist()
    if not starts:
        raise ValueError("qwen3omni_talker: the prompt contains no <|im_start|>, so ChatML turns cannot be found.")
    id_to_role = {token_id: role for role, token_id in role_token_ids.items()}
    boundaries = starts + [total_length]
    turns: List[Tuple[str, int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        role = id_to_role.get(int(input_ids[start + 1]))
        if role is None:
            raise ValueError(
                f"qwen3omni_talker: expected a role token (one of {sorted(role_token_ids)}) after <|im_start|> at "
                f"position {start}, found token id {int(input_ids[start + 1])}."
            )
        turns.append((role, start, end))
    return turns


def collect_media_metadata(conversation_list: Sequence[ConversationItem]) -> Dict[str, Any]:
    """Collect the media shape metadata the talker's 3D RoPE needs.

    ``talker_input_ids`` carries placeholder *runs*, and upstream's
    ``get_rope_index`` expands each run into TMRoPE coordinates from these
    shapes — a run alone does not say how many temporal / height / width steps
    it spans. Upstream receives them as ``generate`` kwargs; here they are read
    back off the carrier items the sibling encoders produced.
    """
    audio_lengths: List[int] = []
    image_grids: List[torch.Tensor] = []
    video_grids: List[torch.Tensor] = []
    video_seconds: List[float] = []
    # Assistant audio is skipped as well as dummies: the shapes here expand the
    # placeholder runs in ``talker_input_ids``, and the speech target has no
    # placeholder run to expand -- it never entered the sequence. It also carries
    # no mel length, since the audio tower only ever looked at the user's side.
    for item in conversation_list:
        if not is_thinker_input(item):
            continue
        if item.type == "audio":
            audio_lengths.append(int(item.meta[_MEL_LEN_KEY]))
        elif item.type == "image":
            image_grids.append(torch.as_tensor(item.meta["grid_thw"]).reshape(1, 3))
        elif item.type == "video":
            video_grids.append(torch.as_tensor(item.meta["grid_thw"]).reshape(1, 3))
            video_seconds.append(float(item.meta["second_per_grid"]))
    return {
        "audio_feature_lengths": torch.tensor(audio_lengths, dtype=torch.long) if audio_lengths else None,
        "image_grid_thw": torch.cat(image_grids, dim=0) if image_grids else None,
        "video_grid_thw": torch.cat(video_grids, dim=0) if video_grids else None,
        "video_second_per_grid": torch.tensor(video_seconds, dtype=torch.float) if video_seconds else None,
        # Audio-in-video is rejected upstream of here (see the chat template and
        # the backbone's packer), so the interleaved layout never occurs.
        "use_audio_in_video": False,
    }


def _multimodal_mask(segments: Sequence[Any], total: int) -> torch.Tensor:
    mask = torch.zeros(total, dtype=torch.bool)
    cursor = 0
    for segment in segments:
        end = cursor + segment.length
        if segment.type != "text":
            mask[cursor:end] = True
        cursor = end
    return mask


class InferenceMixin:
    """FSM ``generate`` — one shot over the finished thinker trace.

    Listed *before* :class:`~...module_modeling_base.PretrainedOmniModule` in
    :class:`Qwen3OmniTalker`'s bases so MRO resolves this concrete ``generate``.
    Unlike the thinker modules this runs once, not per token: the talker's own
    autoregression happens inside upstream's ``generate``.
    """

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        thinker_embed: Optional[torch.Tensor] = None,
        thinker_hidden: Optional[torch.Tensor] = None,
        talker_prompt: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        _require_talker_inputs(thinker_embed, thinker_hidden, talker_prompt)
        speaker = (generation_kwargs or {}).get("speaker", self.config.default_speaker)

        prompt = self.build_prompt(
            thinker_embed=thinker_embed,
            thinker_hidden=thinker_hidden,
            talker_prompt=talker_prompt,
            speaker=speaker,
        )
        codes = self.generate_codes(
            prompt,
            generation_kwargs or {},
            media_metadata=collect_media_metadata(conversation_list),
        )
        conversation_list.append(
            ConversationItem(
                type="output",
                value=codes,
                role="assistant",
                meta={TALKER_CODES_KEY: True},
            )
        )
        return {
            "conversation_list": conversation_list,
            FSM_SIGNAL_KEY: SIGNAL_CODES_DONE,
        }


def _require_talker_inputs(
    thinker_embed: Optional[torch.Tensor],
    thinker_hidden: Optional[torch.Tensor],
    talker_prompt: Optional[Dict[str, Any]],
) -> None:
    """Fail loudly on a mis-wired graph rather than on a shape error deep inside.

    Every one of these comes from another node's return dict, so a missing key
    means the graph ran the talker too early — a YAML problem whose symptom
    would otherwise surface as an opaque ``NoneType`` error.
    """
    missing = [
        name
        for name, value in (
            ("thinker_embed", thinker_embed),
            ("thinker_hidden", thinker_hidden),
            ("talker_prompt", talker_prompt),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"qwen3omni_talker: missing {', '.join(missing)} in the FSM context. The generation graph must run "
            f"qwen3omni_llm.flush_thinker_states and qwen3omni_text_encoder.flush_talker_prompt before the talker."
        )


class Qwen3OmniTalker(InferenceMixin, PretrainedOmniModule):
    """Qwen3-Omni talker — thinker hidden states to Mimi-compatible RVQ codes.

    The emitted codes are Mimi codes (16 quantizers over 2048-entry codebooks at
    12.5 Hz); ``scripts/seed_omni/qwen3omni_audio_check.py`` verifies that
    against ``kyutai/mimi`` on a real checkpoint.
    """

    config_class = Qwen3OmniTalkerConfig
    base_model_prefix = "qwen3omni_talker"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["Qwen3OmniMoeTalkerDecoderLayer", "Qwen3OmniMoeTalkerCodePredictorDecoderLayer"]
    # The wrapped ``Qwen3OmniMoeTalkerForConditionalGeneration`` and its nested
    # code predictor both support it, and this is the one trained module in the
    # speech round, so it is the one that needs the recompute. Declared here
    # because the flag is not inherited from a submodule: ``PreTrainedModel``
    # defaults it False and ``gradient_checkpointing_enable`` raises on the
    # module it was called on before it ever recurses.
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3OmniTalkerConfig):
        super().__init__(config)
        self.config = config
        self.talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(self.config.talker_config)
        self.post_init()
        # ``post_init`` unions the wrapped talker's own ``_no_split_modules``
        # onto this instance, which adds the code predictor *as a whole* — and
        # that one extra FSDP unit breaks this module. A unit's parameters
        # unshard when the unit is called, and ``embed_code_frames`` reads the
        # predictor's per-group embedding tables directly without calling it
        # (upstream reaches the same tables through the predictor's hidden
        # states, so upstream never needs them unsharded). Pinned back to the
        # declared list, the tables land in this module's root unit, which
        # FSDP2 deliberately keeps unsharded across forward and backward.
        #
        # Off the class rather than off this one, so a subclass that declares
        # another wrap unit keeps it. Note this list is also what accelerate
        # reads for `device_map`, which eager single-process inference uses:
        # dropping the predictor lets accelerate split it across devices, which
        # is why `embed_code_frames` moves each table's output onto the frame's
        # device rather than assuming one.
        self._no_split_modules = list(type(self)._no_split_modules)

    @property
    def compute_dtype(self) -> torch.dtype:
        """The dtype to build a tensor in when it will meet the talker's weights.

        The rule is "match the weight you are about to multiply or concatenate
        with", and the weight is read directly rather than inferred, because
        the module-level answer is wrong here. ``PreTrainedModel.dtype`` reports
        the first parameter it finds, which under FSDP2 mixed precision is a
        decoder layer's fp32 *shard* while the all-gather a forward body sees is
        bf16 — casting the thinker traces to it made every projection fail on
        mixed Float and BFloat16.

        ``text_projection`` speaks for the group: it, ``hidden_projection`` and
        the input embeddings are all outside ``_no_split_modules``, so they
        share this module's root FSDP unit and one precision policy. That also
        makes them unsharded whenever a forward body runs, so this reads a real
        dtype rather than a shard's. Should any of them ever be moved into a
        different precision group, they stop agreeing and this property has to
        become per-submodule.

        FSDP2 alone would not need it — ``cast_forward_inputs`` already casts
        the endpoint's floating kwargs at the root boundary, and the old
        ``.to(talker.dtype)`` was undoing that. DDP and eager get no such cast,
        which is why the dtype is stated rather than left alone.
        """
        return next(self.talker.text_projection.parameters()).dtype

    def resolve_speaker_id(self, speaker: str) -> int:
        speaker_ids = self.config.talker_config.speaker_id
        speaker_id = speaker_ids.get(str(speaker).lower())
        if speaker_id is None:
            raise ValueError(f"qwen3omni_talker: unknown speaker {speaker!r}; available: {sorted(speaker_ids)}.")
        return int(speaker_id)

    def build_prompt(
        self,
        *,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        talker_prompt: Dict[str, Any],
        speaker: str,
    ) -> Dict[str, Any]:
        """Assemble the talker's prefill from the thinker traces.

        Ports ``_get_talker_user_parts`` / ``_get_talker_assistant_parts`` to
        the conversation carrier. Two projections carry the thinker into talker
        space: ``hidden_projection`` for multimodal rows (from the accept layer)
        and ``text_projection`` for text rows (from the input embedding). System
        turns are dropped, as upstream does — the talker is not conditioned on
        the thinker's system prompt — and so are history assistant turns.

        Turn boundaries come from the token ids (see :func:`_chatml_turns`), but
        the multimodal mask comes from the carrier: the sibling encoder's row
        count is authoritative, so no placeholder-id comparison is needed.
        """
        segments = talker_prompt["segments"]
        prompt_input_ids = talker_prompt["input_ids"]
        tts_pad_token_id = talker_prompt["tts_pad_token_id"]
        device = self.device
        talker = self.talker

        dtype = self.compute_dtype
        tts_embeds = talker_prompt["tts_embeds"].to(device=device, dtype=dtype)
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(tts_embeds).chunk(3, dim=1)

        thinker_embed = thinker_embed.to(device=device, dtype=dtype)
        thinker_hidden = thinker_hidden.to(device=device, dtype=dtype)

        sequence_length = sum(segment.length for segment in segments)
        supervised = talker_prompt["supervised"]
        _check_prompt_alignment(sequence_length, supervised, prompt_input_ids, thinker_embed)
        multimodal_mask = _multimodal_mask(segments, sequence_length).to(device)

        embeds: List[torch.Tensor] = []
        input_ids: List[torch.Tensor] = []
        trailing_text_hidden: Optional[torch.Tensor] = None

        turns = _chatml_turns(
            prompt_input_ids,
            total_length=thinker_embed.size(1),  # trace rows, i.e. the whole sequence
            im_start_token_id=talker_prompt["im_start_token_id"],
            role_token_ids=talker_prompt["role_token_ids"],
        )
        for index, (role, start, end) in enumerate(turns):
            if role == "system":
                continue
            if role == "user":
                embeds.append(self._project_user_span(start, end, multimodal_mask, thinker_hidden, thinker_embed))
                input_ids.append(prompt_input_ids[start:end].to(device))
            elif role == "assistant":
                # Only the final assistant turn opens the speech span; earlier
                # ones are conversation history the talker does not speak.
                if index != len(turns) - 1:
                    continue
                span_embeds, span_ids, trailing_text_hidden = self._build_speech_span(
                    thinker_embed=thinker_embed,
                    prefix_start=start,
                    prefix_length=_speech_prefix_length(supervised, start, sequence_length),
                    speech_end=_speech_span_end(
                        prompt_input_ids,
                        start,
                        sequence_length,
                        trace_length=thinker_embed.size(1),
                        im_end_token_id=talker_prompt["im_end_token_id"],
                    ),
                    speaker_id=self.resolve_speaker_id(speaker),
                    tts_pad_embed=tts_pad_embed,
                    tts_bos_embed=tts_bos_embed,
                    tts_eos_embed=tts_eos_embed,
                    tts_pad_token_id=tts_pad_token_id,
                )
                embeds.append(span_embeds)
                input_ids.append(span_ids)
            else:
                raise ValueError(f"qwen3omni_talker: unexpected conversation role {role!r} in the prompt.")

        if trailing_text_hidden is None:
            raise ValueError(
                "qwen3omni_talker: the prompt has no trailing assistant turn, so there is no speech span to open. "
                "The generation prompt (<|im_start|>assistant) must be the last thing the thinker saw."
            )

        return {
            "inputs_embeds": torch.cat(embeds, dim=1),
            "talker_input_ids": torch.cat(input_ids, dim=-1).unsqueeze(0),
            "trailing_text_hidden": trailing_text_hidden,
            "tts_pad_embed": tts_pad_embed,
        }

    def _project_user_span(
        self,
        start: int,
        end: int,
        multimodal_mask: torch.Tensor,
        thinker_hidden: torch.Tensor,
        thinker_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Multimodal rows come from the accept layer, text rows from the wte."""
        talker = self.talker
        span_mask = multimodal_mask[start:end].unsqueeze(0)
        span = torch.empty(
            (1, end - start, self.config.talker_config.text_config.hidden_size),
            device=thinker_hidden.device,
            dtype=self.compute_dtype,
        )
        if span_mask.any():
            span[span_mask] = talker.hidden_projection(thinker_hidden[:, start:end][span_mask])
        span[~span_mask] = talker.text_projection(thinker_embed[:, start:end][~span_mask])
        return span

    def _build_speech_span(
        self,
        *,
        thinker_embed: torch.Tensor,
        prefix_start: int,
        speech_end: int,
        prefix_length: int,
        speaker_id: int,
        tts_pad_embed: torch.Tensor,
        tts_bos_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        tts_pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the prefill rows that open speech, plus the text to trail behind it.

        The span is a *sum* of a text track and a codec track. Its shape is
        fixed by the codec track: after ``prefix_length`` rows of zeros come the
        six codec special tokens, so the text track must supply six matching
        rows — four ``tts_pad``, ``tts_bos``, then the thinker's first generated
        token. Everything the thinker generated after that first token is not
        prefilled at all; it is fed one row per decode step as
        ``trailing_text_hidden``, which is how the talker stays aligned with the
        text it is speaking.
        """
        talker = self.talker
        config = self.config.talker_config
        if prefix_length < 0:
            raise ValueError(
                f"qwen3omni_talker: the speech-span prefix came out negative ({prefix_length}); the prompt boundary "
                f"landed before the final assistant turn began."
            )
        assistant_hidden = talker.text_projection(thinker_embed[:, prefix_start:speech_end])
        if assistant_hidden.size(1) <= prefix_length:
            raise ValueError(
                "qwen3omni_talker: the thinker generated no text to speak "
                f"(assistant span is {assistant_hidden.size(1)} rows, prefix alone is {prefix_length})."
            )

        text_track = torch.cat(
            (
                assistant_hidden[:, :prefix_length],
                tts_pad_embed.expand(-1, _NUM_TTS_PAD_ROWS, -1),
                tts_bos_embed,
                assistant_hidden[:, prefix_length : prefix_length + 1],
            ),
            dim=1,
        )
        codec_special_tokens = torch.tensor(
            [
                [
                    config.codec_nothink_id,
                    config.codec_think_bos_id,
                    config.codec_think_eos_id,
                    speaker_id,
                    config.codec_pad_id,
                    config.codec_bos_id,
                ]
            ],
            device=text_track.device,
            dtype=torch.long,
        )
        codec_track = torch.cat(
            (
                torch.zeros(
                    (1, prefix_length, config.text_config.hidden_size),
                    device=text_track.device,
                    dtype=self.compute_dtype,
                ),
                talker.get_input_embeddings()(codec_special_tokens).to(text_track.device),
            ),
            dim=1,
        )
        if text_track.size(1) != codec_track.size(1):
            raise AssertionError(
                f"qwen3omni_talker: speech-span tracks disagree ({text_track.size(1)} text rows vs "
                f"{codec_track.size(1)} codec rows)."
            )

        trailing_text_hidden = torch.cat((assistant_hidden[:, prefix_length + 1 :], tts_eos_embed), dim=1)
        span_ids = torch.full(
            (text_track.size(1),),
            fill_value=tts_pad_token_id,
            dtype=torch.long,
            device=text_track.device,
        )
        return text_track + codec_track, span_ids, trailing_text_hidden

    def generate_codes(
        self,
        prompt: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        media_metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Run the talker's own AR loop and return ``(num_code_groups, T)`` codes.

        Upstream carries the residual code groups out through the second slot of
        the ``hidden_states`` tuple (its own comment calls this a hack), one
        entry per decode step; ``None`` marks the prefill step, which predicts
        no codes.

        ``talker_do_sample`` only governs the semantic head. The residual code
        predictor that fills groups 1..N is hardcoded to ``do_sample=True`` in
        transformers, and its draws re-enter the AR context, so even the
        semantic group drifts between runs and the frame count varies. Codes are
        therefore reproducible only under a pinned RNG — with one seed and one
        prompt this module is bit-identical to upstream (688/688 codes on
        Qwen3-Omni-30B-A3B). Compare audio, not codes, when no seed is pinned.
        """
        config = self.config.talker_config
        vocab_size = config.text_config.vocab_size
        # The top 1024 ids are additional special tokens that must never be
        # sampled — except the codec eos, which is how generation stops.
        suppress_tokens = [i for i in range(vocab_size - 1024, vocab_size) if i != config.codec_eos_token_id]
        media_metadata = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in (media_metadata or {}).items()
        }

        result = self.talker.generate(
            inputs_embeds=prompt["inputs_embeds"],
            talker_input_ids=prompt["talker_input_ids"],
            trailing_text_hidden=prompt["trailing_text_hidden"],
            tts_pad_embed=prompt["tts_pad_embed"],
            # Explicit rather than left to generate's default: the talker's
            # forward reads it directly (``delta0 = (1 - attention_mask).sum()``)
            # to offset the RoPE deltas, so a silently-inferred mask would make
            # position ids depend on generate internals.
            attention_mask=torch.ones(
                prompt["inputs_embeds"].shape[:2], dtype=torch.long, device=prompt["inputs_embeds"].device
            ),
            **media_metadata,
            max_new_tokens=generation_kwargs.get("talker_max_new_tokens", self.config.max_new_tokens),
            do_sample=generation_kwargs.get("talker_do_sample", True),
            top_k=generation_kwargs.get("talker_top_k", 50),
            top_p=generation_kwargs.get("talker_top_p", 1.0),
            temperature=generation_kwargs.get("talker_temperature", 0.9),
            repetition_penalty=generation_kwargs.get("talker_repetition_penalty", 1.05),
            eos_token_id=config.codec_eos_token_id,
            suppress_tokens=suppress_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        residual_codes = [step[-1] for step in result.hidden_states if step[-1] is not None]
        if not residual_codes:
            raise RuntimeError(
                "qwen3omni_talker: generation produced no codec frames (the talker stopped at the prefill step)."
            )
        return torch.stack(residual_codes, dim=1).transpose(1, 2).squeeze(0)

    def embed_code_frames(self, codes: torch.Tensor) -> torch.Tensor:
        """``(G, T)`` codes to ``(1, T, D)`` — one row per codec frame.

        A frame is the *sum* of its group embeddings, which is how the decode
        loop composes its next input: the semantic group goes through the
        talker's own embedding, and groups 1..G-1 through the code predictor's
        per-group tables. Upstream reaches those tables indirectly (it reads
        them back off the predictor's ``hidden_states``); indexing them
        directly is the same arithmetic without running the predictor.
        """
        tables = self.talker.code_predictor.get_input_embeddings()
        frame = self.talker.get_input_embeddings()(codes[0])
        for group in range(1, self.config.talker_config.num_code_groups):
            # ``.to`` because the backbone's embedding and the predictor's
            # tables are not guaranteed to share a device: eager inference
            # loads with ``device_map="auto"``, which may place the code
            # predictor elsewhere when the talker alone does not fit one card.
            # A no-op every other way this module is built.
            frame = frame + tables[group - 1](codes[group]).to(frame.device)
        return frame.unsqueeze(0)

    def _text_rows(self, prompt: Dict[str, Any], num_frames: int) -> torch.Tensor:
        """The text row that rides along with each codec frame.

        The talker speaks the thinker's text one row per frame; once the text
        runs out the rows become ``tts_pad``. Mirrors the ``generation_step <
        trailing_text_hidden.shape[1]`` branch of the decode loop.
        """
        trailing = prompt["trailing_text_hidden"]
        pad = prompt["tts_pad_embed"][:, 0]
        available = trailing.shape[1]
        rows = [trailing[:, step] if step < available else pad for step in range(num_frames)]
        return torch.stack(rows, dim=1)

    def _teacher_forced_positions(
        self,
        prompt: Dict[str, Any],
        num_frames: int,
        media_metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """3D RoPE positions for prompt + frames, matching the decode loop.

        The prompt half has to go through ``get_rope_index`` because media
        placeholder runs expand into TMRoPE coordinates. The frame half simply
        continues past it: at decode step ``s`` the loop uses
        ``arange(1) + past_length + rope_deltas``, and ``past_length`` is
        ``prompt_length + s``.
        """
        prompt_length = prompt["inputs_embeds"].shape[1]
        attention_mask = torch.ones((1, prompt_length), dtype=torch.long, device=self.device)
        media = dict(media_metadata or {})
        get_rope_index = self._rope_index_fn()
        prompt_positions, rope_deltas = get_rope_index(
            prompt["talker_input_ids"],
            media.get("image_grid_thw"),
            media.get("video_grid_thw"),
            attention_mask,
            media.get("use_audio_in_video", False),
            media.get("audio_feature_lengths"),
            media.get("video_second_per_grid"),
        )
        rope_deltas = rope_deltas - (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        offsets = prompt_length + torch.arange(num_frames, device=self.device) + rope_deltas
        frame_positions = offsets.view(1, 1, num_frames).expand(3, 1, num_frames)
        return torch.cat((prompt_positions, frame_positions), dim=2)

    def _rope_index_fn(self) -> Any:
        """``get_rope_index`` bound so the thinker's audio window is visible.

        ``Qwen3OmniMoeTalkerForConditionalGeneration.get_rope_index`` is a thin
        wrapper that calls the thinker's base implementation **with the talker as
        ``self``**. That implementation reads ``self.config.audio_config.n_window``
        to size an audio placeholder run, and the talker config has no
        ``audio_config`` — so a prompt carrying a vision span *and* standalone
        audio raised ``AttributeError`` before the talker could decode.

        The window belongs to the thinker's tower, which this module does not
        own, so it is carried on this module's config (``audio_n_window``) and
        exposed through a view that fills in only the attributes the callee
        reads. Nothing is written back onto the HF config: this is a call-site
        adapter, so a stock config stays stock and the value travels with this
        module rather than mutating shared state mid-forward.
        """
        talker = self.talker
        n_window = self.config.audio_n_window
        # Imported here rather than at module scope so this file stays importable
        # without the transformers modeling module being loaded.
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoePreTrainedModelForConditionalGeneration,
        )

        class _AudioWindow:
            def __init__(self, value: int) -> None:
                self.n_window = value

        class _ConfigWithWindow:
            """The talker config plus the one field the callee expects."""

            def __init__(self, inner: Any, value: int) -> None:
                self._inner = inner
                self.audio_config = _AudioWindow(value)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        class _TalkerWithWindow:
            """The talker itself, with ``config`` swapped for the view above."""

            def __init__(self, inner: Any, value: int) -> None:
                self._inner = inner
                self.config = _ConfigWithWindow(inner.config, value)
                # Read straight off the model by the callee, not from the config.
                self.spatial_merge_size = getattr(inner, "spatial_merge_size", None)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        view = _TalkerWithWindow(talker, n_window)
        return lambda *args, **kwargs: Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index(
            view, *args, **kwargs
        )

    def forward(  # type: ignore[override]
        self,
        *,
        thinker_embeds: Sequence[torch.Tensor],
        thinker_hiddens: Sequence[torch.Tensor],
        talker_prompts: Sequence[Dict[str, Any]],
        target_codes: Sequence[torch.Tensor],
        media_metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        speaker: Optional[str] = None,
        semantic_loss_weight: Optional[float] = None,
        residual_loss_weight: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced loss over a micro-batch of utterances.

        The prefill is assembled *here* rather than in the ``pre_forward`` hook:
        ``build_prompt`` runs ``text_projection`` / ``hidden_projection`` and the
        code embedding tables, and the pre-hook executes outside the FSDP-wrapped
        call, where those parameters are still sharded ``DTensor``s.

        One utterance at a time. The talker's backbone is called without an
        attention mask or cu_seqlens, so utterances cannot be padded into a batch
        without the padding attending into the real frames — and a talker taught
        against that would learn to speak over its own silence. Every rank runs
        exactly ``len(target_codes)`` iterations, which is the uniform micro
        batch size, because the caller refuses a sample it cannot score rather
        than skipping it: each iteration all-gathers the sharded talker layers,
        so a rank-dependent count would hang instead of failing.

        Both heads are accumulated as (sum, count) and divided once, then
        token-weighted across the whole ``dp_sp`` mesh — the two heads score
        different counts (``T+1`` semantic rows including the stop token,
        ``T*(G-1)`` residual codes), and a per-rank mean of means would weight a rank
        holding one short clip like a rank holding four long ones.
        """
        lengths = {
            "thinker_embeds": len(thinker_embeds),
            "thinker_hiddens": len(thinker_hiddens),
            "talker_prompts": len(talker_prompts),
            "target_codes": len(target_codes),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"qwen3omni_talker: these are paired by position but have different lengths: {lengths}.")
        if not target_codes:
            raise ValueError(
                "qwen3omni_talker: nothing to score — this micro-batch carried no utterance. This graph trains "
                "speech output, so its data mix must supply an assistant audio turn per sample; an empty step "
                "would leave the talker without a gradient and desynchronise FSDP."
            )
        metadata = list(media_metadata) if media_metadata is not None else [None] * len(target_codes)

        semantic_sum = residual_sum = 0.0
        semantic_count = residual_count = 0
        for embed, hidden, prompt_view, codes, media in zip(
            thinker_embeds, thinker_hiddens, talker_prompts, target_codes, metadata, strict=True
        ):
            prompt = self.build_prompt(
                thinker_embed=embed,
                thinker_hidden=hidden,
                talker_prompt=prompt_view,
                speaker=self.config.default_speaker if speaker is None else speaker,
            )
            parts = self._utterance_losses(prompt, codes, media)
            semantic_sum = semantic_sum + parts.semantic_sum
            residual_sum = residual_sum + parts.residual_sum
            semantic_count += parts.semantic_count
            residual_count += parts.residual_count

        semantic_loss = _reduce_across_ranks(semantic_sum / max(semantic_count, 1), semantic_count)
        residual_loss = _reduce_across_ranks(residual_sum / max(residual_count, 1), residual_count)
        # Defaulted from the config rather than fixed at 1.0: the graph calls
        # this endpoint with a dict the pre-hook builds, so a bare parameter
        # default is the only value it could ever take.
        semantic_weight = self.config.semantic_loss_weight if semantic_loss_weight is None else semantic_loss_weight
        residual_weight = self.config.residual_loss_weight if residual_loss_weight is None else residual_loss_weight
        return {"loss": semantic_weight * semantic_loss + residual_weight * residual_loss}

    def score_utterance(
        self,
        *,
        prompt: Dict[str, Any],
        target_codes: torch.Tensor,
        media_metadata: Optional[Dict[str, Any]] = None,
        semantic_loss_weight: float = 1.0,
        residual_loss_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """One utterance's teacher-forced loss, with its two parts reported.

        The parts come back detached: they are for metrics, and carrying grad
        twice would let a logger's ``.backward()`` or a stray sum double-count
        them. This is the single-utterance form with no cross-rank reduction —
        the graph node is :meth:`forward`.
        """
        parts = self._utterance_losses(prompt, target_codes, media_metadata)
        semantic_loss = parts.semantic_sum / max(parts.semantic_count, 1)
        residual_loss = parts.residual_sum / max(parts.residual_count, 1)
        return {
            "loss": semantic_loss_weight * semantic_loss + residual_loss_weight * residual_loss,
            "semantic_loss": semantic_loss.detach(),
            "residual_loss": residual_loss.detach(),
        }

    def _utterance_losses(
        self,
        prompt: Dict[str, Any],
        target_codes: torch.Tensor,
        media_metadata: Optional[Dict[str, Any]] = None,
    ) -> "_UtteranceLosses":
        """Teacher-forced (semantic, residual) losses over one utterance's target codes.

        Upstream ships no talker training code, so the recipe here is derived
        from its decode loop and then held to it: replaying the codes a seeded
        ``generate`` produced through this forward reproduces the logits that
        run saw (see ``tests/seed_omni/test_qwen3omni_talker_training.py``).

        Two heads are trained. The backbone's ``codec_head`` predicts the
        semantic group one frame ahead, and the nested code predictor fills
        groups 1..G-1 within a frame. Each comes back as an unreduced (sum,
        count) pair so a micro-batch and a DP mesh can both divide once, by the
        true element count, instead of averaging averages.
        """
        talker = self.talker
        config = self.config.talker_config
        num_groups = config.num_code_groups
        if target_codes.dim() != 2 or target_codes.shape[0] != num_groups:
            raise ValueError(
                f"qwen3omni_talker: target_codes must be (num_code_groups={num_groups}, T), got "
                f"{tuple(target_codes.shape)}."
            )
        if target_codes.shape[1] == 0:
            raise ValueError(
                "qwen3omni_talker: an utterance with no codec frames cannot be scored (Mimi runs at 12.5 Hz, so "
                "a clip under ~80 ms quantises to nothing). Filter such clips in the data layer."
            )
        target_codes = target_codes.to(self.device)
        num_frames = target_codes.shape[1]
        prompt_length = prompt["inputs_embeds"].shape[1]

        frames = self.embed_code_frames(target_codes) + self._text_rows(prompt, num_frames)
        sequence = torch.cat((prompt["inputs_embeds"], frames), dim=1)
        positions = self._teacher_forced_positions(prompt, num_frames, media_metadata)

        hidden = talker.model(
            input_ids=None,
            attention_mask=None,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=sequence,
            use_cache=False,
        ).last_hidden_state

        semantic_sum, semantic_count = self._semantic_loss(hidden, target_codes, prompt_length)
        residual_sum, residual_count = self._residual_loss(hidden, target_codes, prompt_length)
        return _UtteranceLosses(semantic_sum, semantic_count, residual_sum, residual_count)

    def _semantic_loss(
        self,
        hidden: torch.Tensor,
        target_codes: torch.Tensor,
        prompt_length: int,
    ) -> Tuple[torch.Tensor, int]:
        """Next-frame cross entropy on the semantic group, plus the stop token.

        The row before the first frame predicts frame 0, so the predicting rows
        run ``prompt_length - 1 .. prompt_length + T - 1`` — one more than there
        are frames, because the last one has to learn ``codec_eos``. Without
        that final row the talker would never stop on its own.
        """
        num_frames = target_codes.shape[1]
        logits = self.talker.codec_head(hidden[:, prompt_length - 1 : prompt_length + num_frames])
        labels = torch.cat(
            (
                target_codes[0],
                torch.tensor([self.config.talker_config.codec_eos_token_id], device=target_codes.device),
            )
        )
        return _cross_entropy_sum(logits, labels)

    def _residual_loss(
        self,
        hidden: torch.Tensor,
        target_codes: torch.Tensor,
        prompt_length: int,
    ) -> Tuple[torch.Tensor, int]:
        """Cross entropy over groups 1..G-1, all frames scored in one batch.

        Each frame becomes its own length-``G`` predictor sequence — the
        backbone row that predicted the frame, then the frame's groups in order
        — so row ``j`` predicts group ``j`` through ``lm_head[j - 1]``. The
        decode loop walks these rows one at a time behind a KV cache; a single
        causal pass over the whole frame is the same computation.
        """
        predictor = self.talker.code_predictor
        num_groups = self.config.talker_config.num_code_groups
        num_frames = target_codes.shape[1]
        tables = predictor.get_input_embeddings()

        rows = [
            hidden[0, prompt_length - 1 : prompt_length - 1 + num_frames].unsqueeze(1),
            self.talker.get_input_embeddings()(target_codes[0]).unsqueeze(1),
        ]
        # The last group is never fed back — nothing follows it inside a frame.
        rows.extend(tables[group - 1](target_codes[group]).unsqueeze(1) for group in range(1, num_groups - 1))
        predictor_hidden = predictor.model(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=torch.cat(rows, dim=1),
            use_cache=False,
        ).last_hidden_state

        logits = torch.stack(
            [predictor.lm_head[group - 1](predictor_hidden[:, group]) for group in range(1, num_groups)],
            dim=1,
        )
        labels = target_codes[1:].transpose(0, 1)
        return _cross_entropy_sum(logits, labels)


class _UtteranceLosses(NamedTuple):
    """One utterance's two head losses, unreduced.

    Sums rather than means, because the caller has to divide by the *total*
    count over a micro-batch and then over the DP mesh; a mean would already
    have thrown the count away.
    """

    semantic_sum: torch.Tensor
    semantic_count: int
    residual_sum: torch.Tensor
    residual_count: int


def _cross_entropy_sum(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Summed CE and the number of supervised elements it covers.

    The count is measured, not assumed from the frame count: it is the
    denominator, so deriving it from the labels keeps it right if a caller ever
    masks a frame with ``-100``.
    """
    flat_labels = labels.reshape(-1)
    ce_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        flat_labels,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return ce_sum, int((flat_labels != IGNORE_INDEX).sum())


def _reduce_across_ranks(loss: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Token-weight a local mean across the whole ``dp_sp`` mesh.

    Required by constraint 7b: FSDP2 averages the per-rank losses, so without
    this a rank holding one short utterance would count as much as a rank
    holding four long ones. Same helper the text head's CE uses.
    """
    parallel_state = get_parallel_state()
    group = parallel_state.fsdp_group if parallel_state.is_initialized else None
    if group is None:
        # Guarded on the group and not merely on dist being up: with a meshless
        # ambient state ``fsdp_group`` is None, and passing that through would
        # reduce over the *world* group instead.
        return loss
    return reduce_sequence_parallel_loss(
        loss,
        torch.tensor(num_elements, dtype=torch.float, device=loss.device),
        group=group,
    )


def _speech_span_end(
    prompt_input_ids: torch.Tensor,
    start: int,
    sequence_length: int,
    trace_length: int,
    im_end_token_id: int,
) -> int:
    """Where the text the talker speaks stops — the turn's closing marker.

    The mirror of :func:`_speech_prefix_length`, and needed for the same reason:
    in training the response is already on the carrier, and so is the
    ``<|im_end|>\n`` that closes its turn. That marker is supervised — the
    thinker is scored on emitting it — so the supervised mask cannot tell it
    apart from the response, but the talker must, because those rows become
    ``trailing_text_hidden``. Two rows of slack there push ``tts_eos`` — the "the
    text is finished" cue the talker times its ``codec_eos`` against — two codec
    frames later than inference will, and the loss stays healthy while the model
    learns to end late.

    Inference needs no trimming and gets none: the sequence the thinker was given
    ends at ``<|im_start|>assistant\n``, there is no closing marker at or after
    ``start``, and the span runs to the end of the trace — which is where the
    generated response lives. Upstream arrives at the same rows by a different
    route: its trace is one row shorter than its sequence, because the final
    ``<|im_end|>`` stops generation and is never fed back.
    """
    tail = prompt_input_ids[start:sequence_length]
    hits = (tail == im_end_token_id).nonzero()
    if hits.numel() == 0:
        return trace_length
    return start + int(hits[0])


def _speech_prefix_length(supervised: torch.Tensor, start: int, sequence_length: int) -> int:
    """Rows of ``<|im_start|>assistant\n`` that open the speech span.

    The prefix is whatever the thinker was *given* inside the final assistant
    turn, so it ends at that turn's first supervised token. Searching from
    ``start`` rather than from row 0 is what makes multi-turn data work: every
    assistant turn in ordinary SFT data is supervised, so the sample's first
    supervised token sits in an *earlier* turn and a global search would put the
    boundary behind ``start`` — a negative prefix.

    With nothing supervised at or after ``start`` the turn is a bare generation
    prompt and the whole given sequence is the prompt: the inference shape, where
    the rows past ``sequence_length`` are text the thinker generated.
    """
    tail = supervised[start:sequence_length]
    hits = tail.nonzero()
    if hits.numel() == 0:
        return sequence_length - start
    return int(hits[0])


def _check_prompt_alignment(
    sequence_length: int,
    supervised: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    thinker_embed: torch.Tensor,
) -> None:
    """The three views of the sequence must agree on its length.

    ``segments`` (per-row lengths), ``input_ids`` (token-level, placeholders
    rebuilt) and ``thinker_embed`` (the backbone's own trace) are produced by
    three different modules. A disagreement means one of them mis-counted a
    media placeholder run, which would silently shift every projection after it.
    """
    if prompt_input_ids.numel() != sequence_length:
        raise ValueError(
            f"qwen3omni_talker: segments sum to {sequence_length} rows but the rebuilt token sequence has "
            f"{prompt_input_ids.numel()}. A media item's embedding row count must equal its placeholder count."
        )
    if thinker_embed.size(1) < sequence_length:
        raise ValueError(
            f"qwen3omni_talker: the thinker trace is shorter ({thinker_embed.size(1)}) than the sequence "
            f"({sequence_length}). flush_thinker_states must run after the thinker finished, not before."
        )
    if supervised.numel() != sequence_length:
        raise ValueError(
            f"qwen3omni_talker: the supervised mask covers {supervised.numel()} rows but the segments sum to "
            f"{sequence_length}. It is read positionally against the same sequence, so the two must agree."
        )


__all__ = ["Qwen3OmniTalker", "SIGNAL_CODES_DONE", "collect_media_metadata"]
