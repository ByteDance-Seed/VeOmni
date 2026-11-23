"""Qwen3-Omni talker: prompt assembly must match upstream bit-for-bit.

The talker's own autoregression is upstream's, so what is worth testing here is
the part VeOmni reimplemented: turning the conversation carrier into the talker
prefill. These tests build a tiny talker with random weights, run *upstream's*
token-scanning assembly and this module's carrier-driven assembly over the same
inputs, and require identical tensors.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from veomni.models.seed_omni.modules.qwen3omni.carrier import TALKER_CODES_KEY
from veomni.models.seed_omni.modules.qwen3omni.talker.configuration import Qwen3OmniTalkerConfig
from veomni.models.seed_omni.modules.qwen3omni.talker.modeling import Qwen3OmniTalker
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import PromptSegment


CHECKPOINT = "/opt/tiger/models/Qwen3-Omni-30B-A3B-Instruct"
requires_checkpoint = pytest.mark.skipif(
    not os.path.isdir(CHECKPOINT), reason=f"needs the Qwen3-Omni checkpoint at {CHECKPOINT}"
)

THINKER_HIDDEN = 24
IM_START, SYSTEM, USER, ASSISTANT = 1000, 1001, 1002, 1003
IM_END, AUDIO_START, AUDIO_END, AUDIO_PAD = 1004, 1005, 1006, 1007
TTS_BOS, TTS_EOS, TTS_PAD = 1008, 1009, 1010
ROLE_TOKEN_IDS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


def _shrink(raw, layers=2):
    small = dict(raw)
    small.update(
        num_hidden_layers=layers,
        hidden_size=32,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        dtype="float32",
    )
    if small.get("layer_types"):
        small["layer_types"] = small["layer_types"][:layers]
    return small


def _tiny_talker():
    """A structurally faithful talker: real config, tiny dimensions, random weights.

    Derived from the real ``config.json`` rather than hand-written so that the
    codec special-token ids, the speaker table and the projection wiring are the
    ones the checkpoint actually uses.
    """
    with open(os.path.join(CHECKPOINT, "config.json")) as handle:
        raw = json.load(handle)["talker_config"]
    tiny = dict(raw)
    tiny["text_config"] = {
        **_shrink(raw["text_config"]),
        "moe_intermediate_size": 16,
        "num_experts": 2,
        "num_experts_per_tok": 1,
    }
    tiny["code_predictor_config"] = _shrink(raw["code_predictor_config"])
    tiny["thinker_hidden_size"] = THINKER_HIDDEN
    torch.manual_seed(0)
    module = Qwen3OmniTalker(Qwen3OmniTalkerConfig(talker_config=tiny)).eval()
    _initialize_moe_experts(module)
    return module


def _initialize_moe_experts(module: Qwen3OmniTalker) -> None:
    """Fill the MoE expert tensors, which HF's ``_init_weights`` does not reach.

    They are 3D ``nn.Parameter`` stacks rather than ``nn.Linear`` weights, so a
    freshly constructed talker keeps whatever ``torch.empty`` returned — values
    around 1e37 that turn the first forward into NaN. Assembly-only tests never
    noticed; anything that runs the backbone does.
    """
    torch.manual_seed(0)
    for name, parameter in module.named_parameters():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            with torch.no_grad():
                parameter.normal_(mean=0.0, std=0.02)


def _upstream_shim(module):
    """Bind upstream's assembly helpers to a namespace exposing what they read.

    ``_get_talker_user_parts`` / ``_get_talker_assistant_parts`` only touch
    ``self.talker`` and ``self.config``, so a namespace is enough to run the
    genuine upstream code as the reference.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration as Ref

    shim = SimpleNamespace(
        talker=module.talker,
        config=SimpleNamespace(talker_config=module.config.talker_config, tts_pad_token_id=TTS_PAD),
    )
    shim._get_talker_user_parts = Ref._get_talker_user_parts.__get__(shim)
    shim._get_talker_assistant_parts = Ref._get_talker_assistant_parts.__get__(shim)
    return shim


def _upstream_assemble(module, prompt_ids, thinker_embed, thinker_hidden, speaker_id, total_length):
    """Upstream's ``generate`` prompt-assembly loop, verbatim in structure."""
    shim = _upstream_shim(module)
    talker = module.talker
    tts_ids = torch.tensor([[TTS_BOS, TTS_EOS, TTS_PAD]], dtype=torch.long)
    # Upstream projects thinker wte outputs; the test's thinker "wte" is the
    # embedding table below, shared with the module path via tts_embeds.
    tts_bos, tts_eos, tts_pad = talker.text_projection(_thinker_wte()(tts_ids)).chunk(3, dim=1)

    multimodal_mask = (prompt_ids == AUDIO_PAD).unsqueeze(0)
    starts = torch.nonzero(prompt_ids == IM_START).reshape(-1).tolist()
    boundaries = starts + [total_length]

    embeds, ids, trailing = [], [], None
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        role_token = int(prompt_ids[start + 1])
        if role_token == SYSTEM:
            continue
        elif role_token == USER:
            embeds.append(shim._get_talker_user_parts(start, end, multimodal_mask, thinker_hidden, thinker_embed))
            ids.append(prompt_ids[start:end].unsqueeze(0))
        elif role_token == ASSISTANT and index == len(boundaries) - 2:
            span_embeds, span_ids, trailing = shim._get_talker_assistant_parts(
                start, end, speaker_id, thinker_embed, tts_pad, tts_bos, tts_eos
            )
            embeds.append(span_embeds)
            ids.append(span_ids)
        elif role_token == ASSISTANT:
            continue
        else:
            raise AssertionError("unexpected role token")
    return {
        "inputs_embeds": torch.cat(embeds, dim=1),
        "talker_input_ids": torch.cat(ids, dim=1),
        "trailing_text_hidden": trailing,
    }


_WTE = None


def _thinker_wte():
    """Stand-in for the thinker's embedding table, shared by both paths."""
    global _WTE
    if _WTE is None:
        torch.manual_seed(1)
        _WTE = torch.nn.Embedding(1200, THINKER_HIDDEN)
    return _WTE


def _build_case(history: bool):
    """A prompt with an audio turn, optionally preceded by an assistant history turn.

    The history variant is the one that separates correct turn splitting from
    splitting on ``item.role``: the history assistant turn and the trailing
    generation prompt are both role ``assistant`` and would merge.
    """
    rows = [
        ("text", "system", [IM_START, SYSTEM, 20, 21, IM_END]),
        ("text", "user", [IM_START, USER, 30, AUDIO_START]),
        ("audio", "user", [AUDIO_PAD] * 7),
        ("text", "user", [AUDIO_END, 31, 32, IM_END]),
    ]
    surviving_user_rows = 4 + 7 + 4
    if history:
        rows.append(("text", "assistant", [IM_START, ASSISTANT, 40, 41, IM_END]))
        rows.append(("text", "user", [IM_START, USER, 50, 51, IM_END]))
        surviving_user_rows += 5
    rows.append(("text", "assistant", [IM_START, ASSISTANT, 60]))

    prompt_ids = torch.tensor([token for _, _, tokens in rows for token in tokens], dtype=torch.long)
    segments = [PromptSegment(role, item_type, len(tokens)) for item_type, role, tokens in rows]
    prompt_length = prompt_ids.numel()

    num_generated = 5
    torch.manual_seed(2)
    thinker_embed = torch.randn(1, prompt_length + num_generated, THINKER_HIDDEN)
    thinker_hidden = torch.randn(1, prompt_length + num_generated, THINKER_HIDDEN)
    return SimpleNamespace(
        prompt_ids=prompt_ids,
        segments=segments,
        thinker_embed=thinker_embed,
        thinker_hidden=thinker_hidden,
        prompt_length=prompt_length,
        num_generated=num_generated,
        surviving_user_rows=surviving_user_rows,
    )


def _talker_prompt_payload(prompt_ids, segments, supervised=None):
    tts_ids = torch.tensor([[TTS_BOS, TTS_EOS, TTS_PAD]], dtype=torch.long)
    return {
        "input_ids": prompt_ids,
        "segments": segments,
        # Inference supervises nothing — the response is not generated yet — so
        # the whole given sequence is the prompt.
        "supervised": torch.zeros(sum(seg.length for seg in segments), dtype=torch.bool)
        if supervised is None
        else supervised,
        "tts_embeds": _thinker_wte()(tts_ids),
        "tts_pad_token_id": TTS_PAD,
        "im_start_token_id": IM_START,
        "im_end_token_id": IM_END,
        "role_token_ids": ROLE_TOKEN_IDS,
    }


def _module_prompt(module, case, speaker="ethan", segments=None, thinker_embed=None, thinker_hidden=None):
    return module.build_prompt(
        thinker_embed=case.thinker_embed if thinker_embed is None else thinker_embed,
        thinker_hidden=case.thinker_hidden if thinker_hidden is None else thinker_hidden,
        talker_prompt=_talker_prompt_payload(case.prompt_ids, case.segments if segments is None else segments),
        speaker=speaker,
    )


@requires_checkpoint
@pytest.mark.parametrize("history", [False, True], ids=["single_turn", "multi_turn_with_history"])
def test_prompt_assembly_matches_upstream(history):
    """The carrier-driven assembly must equal upstream's token-scanning one.

    Covers both projections (audio rows via ``hidden_projection``, text rows via
    ``text_projection``), the dropped system turn, the dropped history assistant
    turn, the speech-span construction and the trailing text.
    """
    module = _tiny_talker()
    case = _build_case(history)
    speaker_id = module.resolve_speaker_id("ethan")

    with torch.no_grad():
        expected = _upstream_assemble(
            module,
            case.prompt_ids,
            case.thinker_embed,
            case.thinker_hidden,
            speaker_id,
            total_length=case.thinker_embed.size(1),
        )
        actual = _module_prompt(module, case)

    torch.testing.assert_close(actual["inputs_embeds"], expected["inputs_embeds"], rtol=0, atol=0)
    torch.testing.assert_close(actual["trailing_text_hidden"], expected["trailing_text_hidden"], rtol=0, atol=0)
    assert torch.equal(actual["talker_input_ids"], expected["talker_input_ids"])
    # The thinker's system turn must not reach the talker at all.
    assert (actual["talker_input_ids"] != SYSTEM).all()


@requires_checkpoint
def test_speech_span_prefix_is_not_merged_with_assistant_history():
    """Splitting turns by ``item.role`` would silently corrupt this case.

    A history assistant turn sits directly before the generation prompt. Both
    carry role ``assistant``, so a role-run split would treat them as one turn
    and inflate the speech-span prefix. Pinning the span's width catches that:
    it is the generation prompt's 3 rows plus the 6 codec-track rows.
    """
    module = _tiny_talker()
    case = _build_case(history=True)

    with torch.no_grad():
        actual = _module_prompt(module, case)

    # Both user turns survive; system and history assistant are dropped; the
    # speech span contributes exactly the 3 prefix rows + 6 codec rows.
    assert actual["inputs_embeds"].size(1) == case.surviving_user_rows + 9
    # Every id in the speech span is tts_pad, and it is the tail of the sequence.
    assert torch.equal(actual["talker_input_ids"][0, -9:], torch.full((9,), TTS_PAD, dtype=torch.long))
    assert (actual["talker_input_ids"][0, :-9] != TTS_PAD).all()


@requires_checkpoint
def test_trailing_text_carries_generated_tail_plus_eos():
    """``trailing_text_hidden`` is fed one row per decode step, so its length is load-bearing.

    The thinker's first generated row is prefilled inside the speech span; the
    rest trail behind it, followed by ``tts_eos``.
    """
    module = _tiny_talker()
    case = _build_case(history=False)

    with torch.no_grad():
        actual = _module_prompt(module, case)

    # Generated rows minus the one prefilled, plus tts_eos.
    assert actual["trailing_text_hidden"].size(1) == case.num_generated - 1 + 1


@requires_checkpoint
def test_speaker_and_wiring_errors_are_explicit():
    module = _tiny_talker()
    case = _build_case(history=False)

    with pytest.raises(ValueError, match="unknown speaker"):
        _module_prompt(module, case, speaker="nobody")

    # A media item whose embedding rows disagree with its placeholder run.
    bad_segments = list(case.segments)
    bad_segments[2] = PromptSegment("user", "audio", 6)
    with pytest.raises(ValueError, match="segments sum to"):
        _module_prompt(module, case, segments=bad_segments)

    # flush_thinker_states run before the thinker finished.
    with pytest.raises(ValueError, match="thinker trace is shorter"):
        _module_prompt(
            module, case, thinker_embed=case.thinker_embed[:, :5], thinker_hidden=case.thinker_hidden[:, :5]
        )


@requires_checkpoint
def test_generate_appends_a_codes_item_and_signals_done(monkeypatch):
    """Carrier plumbing: codes land as an ``output`` row the codec finds by source."""
    from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
    from veomni.models.seed_omni.modules.qwen3omni.talker.modeling import SIGNAL_CODES_DONE
    from veomni.models.seed_omni.utils.conversation import ConversationItem

    module = _tiny_talker()
    case = _build_case(history=False)
    num_groups = module.config.talker_config.num_code_groups
    fake_codes = torch.randint(0, 100, (num_groups, 12))
    seen = {}

    def fake_generate_codes(prompt, kwargs, media_metadata=None):
        seen["media_metadata"] = media_metadata
        return fake_codes

    monkeypatch.setattr(module, "generate_codes", fake_generate_codes)

    # A carrier with real media metadata: the talker must forward the shapes its
    # 3D RoPE needs to expand each placeholder run.
    conversation_list = [
        ConversationItem(type="text", value=torch.zeros(1), role="user"),
        ConversationItem(type="audio", value=torch.zeros(7, 4), role="user", meta={"audio_mel_len": 28}),
        ConversationItem(type="image", value=torch.zeros(4, 4), role="user", meta={"grid_thw": [1, 4, 4]}),
        ConversationItem(
            type="video", value=torch.zeros(8, 4), role="user", meta={"grid_thw": [2, 4, 4], "second_per_grid": 0.5}
        ),
    ]
    outputs = module.generate(
        conversation_list=conversation_list,
        thinker_embed=case.thinker_embed,
        thinker_hidden=case.thinker_hidden,
        talker_prompt=_talker_prompt_payload(case.prompt_ids, case.segments),
    )

    assert outputs[FSM_SIGNAL_KEY] == SIGNAL_CODES_DONE
    codes_item = outputs["conversation_list"][-1]
    assert codes_item.type == "output"
    # The codec finds these by a metadata stamp, not a `source` tag —
    # `ConversationItem` has no such field.
    assert codes_item.meta[TALKER_CODES_KEY] is True
    # The codec accepts (num_code_groups, T); keep the talker's output on that contract.
    assert codes_item.value.shape == (num_groups, 12)

    metadata = seen["media_metadata"]
    assert metadata["audio_feature_lengths"].tolist() == [28]
    assert metadata["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert metadata["video_grid_thw"].tolist() == [[2, 4, 4]]
    assert metadata["video_second_per_grid"].tolist() == [0.5]
    assert metadata["use_audio_in_video"] is False


@requires_checkpoint
def test_media_metadata_is_absent_for_a_text_only_prompt():
    """A text-only turn must pass no media shapes, not empty tensors.

    ``get_rope_index`` branches on ``is None``, so a zero-length tensor would
    send it down the multimodal path with nothing to expand.
    """
    from veomni.models.seed_omni.modules.qwen3omni.talker.modeling import collect_media_metadata
    from veomni.models.seed_omni.utils.conversation import ConversationItem

    metadata = collect_media_metadata([ConversationItem(type="text", value=torch.zeros(3), role="user")])
    assert metadata["audio_feature_lengths"] is None
    assert metadata["image_grid_thw"] is None
    assert metadata["video_grid_thw"] is None
    assert metadata["video_second_per_grid"] is None


@requires_checkpoint
def test_missing_graph_inputs_name_the_offending_node():
    module = _tiny_talker()
    with pytest.raises(ValueError, match="flush_thinker_states"):
        module.generate(conversation_list=[], thinker_embed=None, thinker_hidden=None, talker_prompt=None)


@requires_checkpoint
def test_the_code_predictor_is_not_an_fsdp_unit_of_its_own():
    """``post_init`` unions the wrapped talker's wrap list; this module must undo it.

    An FSDP unit unshards when it is *called*, and ``embed_code_frames`` reads
    the code predictor's per-group embedding tables without calling the
    predictor. Left in the list, the predictor becomes its own unit and the
    first training step dies inside an embedding lookup on a sharded weight,
    naming neither this module nor the reason.
    """
    module = _tiny_talker()

    assert set(module._no_split_modules) == {
        "Qwen3OmniMoeTalkerDecoderLayer",
        "Qwen3OmniMoeTalkerCodePredictorDecoderLayer",
    }
    # The thing the union would have added, spelled out: the tables this module
    # indexes live under a class that must stay inside the root unit.
    assert type(module.talker.code_predictor).__name__ not in module._no_split_modules


@requires_checkpoint
def test_the_traces_are_cast_to_what_the_projection_computes_in():
    """FSDP2 mixed precision makes ``talker.dtype`` disagree with the matmul.

    Sharded storage stays fp32 while the all-gather a forward sees is bf16, and
    ``PreTrainedModel.dtype`` reports the former. Casting the thinker traces to
    it fed Float into a BFloat16 projection.

    The simulation covers the three submodules ``build_prompt`` touches, which
    is all it needs; under a real wrap the rest of the root unit (``codec_head``
    and the code predictor's tables and heads) is in the compute dtype too, so
    this says nothing about ``_utterance_losses``.
    """
    module = _tiny_talker()
    # What FSDP2 presents to a forward body: the root unit (projections and
    # embeddings) all-gathered into the compute dtype, the decoder layers still
    # fp32 shards. ``PreTrainedModel.dtype`` reports the first parameter, which
    # is a decoder layer's — hence the disagreement.
    module.talker.text_projection.to(torch.bfloat16)
    module.talker.hidden_projection.to(torch.bfloat16)
    module.talker.get_input_embeddings().to(torch.bfloat16)

    assert module.talker.dtype == torch.float32, "the fixture must reproduce the skew to be worth anything"
    assert module.compute_dtype == torch.bfloat16

    prompt = _module_prompt(module, _build_case(history=False))
    assert prompt["inputs_embeds"].dtype == torch.bfloat16


@requires_checkpoint
def test_a_training_shaped_prompt_leaves_the_response_to_be_spoken():
    """The one property the boundary rule exists for, held against the real pipeline.

    `loss_mask` does not survive `tokenize` and the masked `<|im_start|>assistant`
    prefix is merged into the supervised response, so a boundary read at item
    granularity silently becomes "the whole sample" — and the speech-span prefix
    then swallows the very text the talker is supposed to say. Only the per-token
    labels still resolve it, so this drives the actual template + tokenizer.
    """
    import numpy as np
    from transformers import AutoTokenizer

    from veomni.models.seed_omni.modules.qwen3omni.talker.modeling import (
        _chatml_turns,
        _speech_prefix_length,
        _speech_span_end,
    )
    from veomni.models.seed_omni.modules.qwen3omni.text_encoder.chat_template import Qwen3OmniChatTemplate
    from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import (
        talker_prompt_segments,
        talker_supervised_mask,
    )
    from veomni.models.seed_omni.utils.conversation import ConversationItem

    template = Qwen3OmniChatTemplate(AutoTokenizer.from_pretrained(CHECKPOINT))
    conversation = template.tokenize_conversation(
        [
            ConversationItem(type="text", value="Please say hello out loud.", role="user"),
            ConversationItem(type="text", value="Hello there, nice to meet you!", role="assistant"),
            ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant"),
        ]
    )

    segments = talker_prompt_segments(conversation)
    supervised = talker_supervised_mask(conversation)
    input_ids = template.pack_talker_input_ids(conversation)
    sequence_length = sum(segment.length for segment in segments)

    # The three views the talker reads positionally must agree, and the speech
    # target must be in none of them.
    assert input_ids.numel() == sequence_length == supervised.numel()
    assert all(segment.type != "audio" for segment in segments)

    _, start, _ = _chatml_turns(
        input_ids,
        total_length=sequence_length,
        im_start_token_id=template.im_start_token_id,
        role_token_ids=template.role_token_ids,
    )[-1]
    prefix_length = _speech_prefix_length(supervised, start, sequence_length)

    assert template.tokenizer.decode(input_ids[start : start + prefix_length]) == "<|im_start|>assistant\n"
    assert "Hello there" in template.tokenizer.decode(input_ids[start + prefix_length : sequence_length])

    # The other end. `<|im_end|>\n` is supervised too — the thinker is scored on
    # closing the turn — so the mask cannot exclude it and the span end is read
    # off the marker itself. Left in, those rows would land in
    # `trailing_text_hidden` and push `tts_eos` two codec frames late.
    speech_end = _speech_span_end(
        input_ids,
        start,
        sequence_length,
        trace_length=sequence_length,
        im_end_token_id=template.im_end_token_id,
    )

    assert supervised[speech_end], "the closing marker is supervised, so only the token id can find it"
    assert template.tokenizer.decode(input_ids[start + prefix_length : speech_end]) == "Hello there, nice to meet you!"
