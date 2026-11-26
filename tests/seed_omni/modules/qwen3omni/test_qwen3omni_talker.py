"""Qwen3-Omni talker: prompt assembly must match upstream bit-for-bit.

The talker's own autoregression is upstream's, so what is worth testing here is
the part VeOmni reimplemented: turning the conversation carrier into the talker
prefill. These tests build a tiny talker with random weights, run *upstream's*
token-scanning assembly and this module's carrier-driven assembly over the same
inputs, and require identical tensors.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.tools.hf_paths import hf_local_or_remote
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata
from veomni.models.seed_omni.modules.qwen3omni.carrier import TALKER_CODES_KEY
from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import TALKER_CODE_LABELS
from veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated import Qwen3OmniLlmAccelerated
from veomni.models.seed_omni.modules.qwen3omni.llm.configuration import Qwen3OmniLlmConfig
from veomni.models.seed_omni.modules.qwen3omni.llm.modeling import (
    THINKER_EMBED,
    THINKER_HIDDEN,
    Qwen3OmniLlm,
    pack_qwen3omni_conversations_for_forward,
    qwen3omni_vision_position_ids,
)
from veomni.models.seed_omni.modules.qwen3omni.talker.accelerated.accelerated import (
    _talker_prompt,
    _target_codes,
    _thinker_traces,
)
from veomni.models.seed_omni.modules.qwen3omni.talker.configuration import Qwen3OmniTalkerConfig
from veomni.models.seed_omni.modules.qwen3omni.talker.modeling import Qwen3OmniTalker, _speech_prefix_length
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.accelerated.accelerated import _stash_talker_prompts
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import TALKER_PROMPT, PromptSegment
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as _Upstream,
)


# These cases read the *identity* of the shipped model — token ids, the speaker
# table, codec special ids, patch/merge sizes — then shrink everything to a size
# that runs on CPU. The toy config carries the same values, so no weights are
# needed and the whole file runs in CI.
CONFIG_DIR = Path(__file__).parents[4] / "tests" / "toy_config" / "qwen3omni_toy"
# One case decodes the prompt back to text to read the speech-span boundaries
# instead of only token ids, so it needs the real tokenizer. Resolves through
# CI_HF_MODELS_DIR when the checkpoint is staged, and skips otherwise.
TOKENIZER = hf_local_or_remote("Qwen/Qwen3-Omni-30B-A3B-Instruct")
requires_tokenizer = pytest.mark.skipif(
    not os.path.isdir(TOKENIZER), reason=f"needs a Qwen3-Omni tokenizer at {TOKENIZER}"
)

THINKER_HIDDEN_SIZE = 24
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
    """A structurally faithful talker: shipped config, tiny dimensions, random weights.

    Derived from the shipped ``config.json`` rather than hand-written so that the
    codec special-token ids, the speaker table and the projection wiring are the
    ones the checkpoint actually uses.
    """
    with open(os.path.join(CONFIG_DIR, "config.json")) as handle:
        raw = json.load(handle)["talker_config"]
    tiny = dict(raw)
    tiny["text_config"] = {
        **_shrink(raw["text_config"]),
        "moe_intermediate_size": 16,
        "num_experts": 2,
        "num_experts_per_tok": 1,
    }
    tiny["code_predictor_config"] = _shrink(raw["code_predictor_config"])
    tiny["thinker_hidden_size"] = THINKER_HIDDEN_SIZE
    torch.manual_seed(0)
    module = Qwen3OmniTalker(
        Qwen3OmniTalkerConfig(talker_config=tiny, ops_implementation=make_eager_ops_config())
    ).eval()
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
        _WTE = torch.nn.Embedding(1200, THINKER_HIDDEN_SIZE)
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
    thinker_embed = torch.randn(1, prompt_length + num_generated, THINKER_HIDDEN_SIZE)
    thinker_hidden = torch.randn(1, prompt_length + num_generated, THINKER_HIDDEN_SIZE)
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


def test_missing_graph_inputs_name_the_offending_node():
    module = _tiny_talker()
    with pytest.raises(ValueError, match="flush_thinker_states"):
        module.generate(conversation_list=[], thinker_embed=None, thinker_hidden=None, talker_prompt=None)


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


@requires_tokenizer
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

    template = Qwen3OmniChatTemplate(AutoTokenizer.from_pretrained(TOKENIZER))
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


# ── teacher-forced training ──────────────────────────────────────────────────


def _prompt(module: Qwen3OmniTalker, length: int = 6, trailing: int = 3) -> dict:
    """A prompt shaped like `build_prompt`'s output, without the thinker."""
    torch.manual_seed(1)
    hidden = module.config.talker_config.text_config.hidden_size
    vocab = module.config.talker_config.text_config.vocab_size
    return {
        "inputs_embeds": torch.randn(1, length, hidden),
        # Real ids matter only in that get_rope_index reads them; with no media
        # placeholders any in-range ids give a plain 0..L-1 ramp.
        "talker_input_ids": torch.randint(0, vocab - 1024, (1, length)),
        "trailing_text_hidden": torch.randn(1, trailing, hidden),
        "tts_pad_embed": torch.randn(1, 1, hidden),
    }


def _decode_reference(module: Qwen3OmniTalker, prompt: dict, max_new_tokens: int) -> tuple:
    """Run the real decode loop, taping the codes and both heads' logits."""
    talker = module.talker
    config = module.config.talker_config
    predictor = talker.code_predictor
    residual_logits: list[torch.Tensor] = []
    original = predictor.generate

    def taped(*args, **kwargs):
        kwargs["output_logits"] = True
        result = original(*args, **kwargs)
        residual_logits.append(torch.stack(result.logits, dim=0).float())
        return result

    predictor.generate = taped
    try:
        torch.manual_seed(7)
        with torch.no_grad():
            output = talker.generate(
                inputs_embeds=prompt["inputs_embeds"],
                talker_input_ids=prompt["talker_input_ids"],
                trailing_text_hidden=prompt["trailing_text_hidden"],
                tts_pad_embed=prompt["tts_pad_embed"],
                attention_mask=torch.ones(prompt["inputs_embeds"].shape[:2], dtype=torch.long),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=config.codec_eos_token_id,
                output_hidden_states=True,
                output_logits=True,
                return_dict_in_generate=True,
            )
    finally:
        predictor.generate = original

    codes = torch.stack([step[-1] for step in output.hidden_states if step[-1] is not None], dim=1)
    codes = codes.transpose(1, 2).squeeze(0)
    semantic = torch.stack(output.logits, dim=0).float()
    return codes, semantic, residual_logits


def _teacher_forced_hidden(module: Qwen3OmniTalker, prompt: dict, codes: torch.Tensor) -> torch.Tensor:
    frames = module.embed_code_frames(codes) + module._text_rows(prompt, codes.shape[1])
    sequence = torch.cat((prompt["inputs_embeds"], frames), dim=1)
    positions = module._teacher_forced_positions(prompt, codes.shape[1])
    with torch.no_grad():
        return module.talker.model(
            input_ids=None,
            attention_mask=None,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=sequence,
            use_cache=False,
        ).last_hidden_state


def test_teacher_forcing_reproduces_the_decode_loops_semantic_logits():
    """The whole training path rests on this: same codes in, same logits out."""
    module = _tiny_talker()
    prompt = _prompt(module)
    codes, reference, _ = _decode_reference(module, prompt, max_new_tokens=8)
    prompt_length = prompt["inputs_embeds"].shape[1]

    hidden = _teacher_forced_hidden(module, prompt, codes)
    steps = reference.shape[0]
    logits = module.talker.codec_head(hidden[:, prompt_length - 1 : prompt_length - 1 + steps]).float()

    torch.testing.assert_close(logits[0], reference[:, 0], rtol=1e-4, atol=1e-4)


def test_teacher_forcing_reproduces_the_code_predictors_logits():
    """The residual head is indexed per group; an off-by-one would still train."""
    module = _tiny_talker()
    prompt = _prompt(module)
    codes, _, reference = _decode_reference(module, prompt, max_new_tokens=8)
    num_groups = module.config.talker_config.num_code_groups
    prompt_length = prompt["inputs_embeds"].shape[1]
    predictor = module.talker.code_predictor
    tables = predictor.get_input_embeddings()

    hidden = _teacher_forced_hidden(module, prompt, codes)
    frames = codes.shape[1]
    rows = [
        hidden[0, prompt_length - 1 : prompt_length - 1 + frames].unsqueeze(1),
        module.talker.get_input_embeddings()(codes[0]).unsqueeze(1),
    ]
    rows.extend(tables[g - 1](codes[g]).unsqueeze(1) for g in range(1, num_groups - 1))
    with torch.no_grad():
        predictor_hidden = predictor.model(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=torch.cat(rows, dim=1),
            use_cache=False,
        ).last_hidden_state
    logits = torch.stack(
        [predictor.lm_head[g - 1](predictor_hidden[:, g]) for g in range(1, num_groups)], dim=1
    ).float()

    expected = torch.cat(reference, dim=1).permute(1, 0, 2)[:frames]
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


def test_scoring_one_utterance_returns_the_summed_loss_and_both_parts():
    module = _tiny_talker()
    prompt = _prompt(module)
    codes, _, _ = _decode_reference(module, prompt, max_new_tokens=6)

    out = module.score_utterance(prompt=prompt, target_codes=codes)

    assert out["loss"].ndim == 0
    assert out["loss"].requires_grad
    torch.testing.assert_close(out["loss"], out["semantic_loss"] + out["residual_loss"])
    # The parts are reported for metrics only — carrying grad twice would let a
    # logger's .backward() or a stray sum double-count them.
    assert not out["semantic_loss"].requires_grad
    assert not out["residual_loss"].requires_grad


def test_loss_weights_scale_the_two_heads_independently():
    module = _tiny_talker()
    prompt = _prompt(module)
    codes, _, _ = _decode_reference(module, prompt, max_new_tokens=6)

    base = module.score_utterance(prompt=prompt, target_codes=codes)
    semantic_only = module.score_utterance(prompt=prompt, target_codes=codes, residual_loss_weight=0.0)

    torch.testing.assert_close(semantic_only["loss"], base["semantic_loss"])


def test_the_stop_token_is_scored_after_the_last_frame():
    """Without an eos row the talker would train to speak forever."""
    module = _tiny_talker()
    prompt = _prompt(module)
    codes, _, _ = _decode_reference(module, prompt, max_new_tokens=6)
    frames = codes.shape[1]
    prompt_length = prompt["inputs_embeds"].shape[1]

    hidden = _teacher_forced_hidden(module, prompt, codes)
    scored = hidden[:, prompt_length - 1 : prompt_length + frames]

    # One more scored row than there are frames: the extra one carries codec_eos.
    assert scored.shape[1] == frames + 1


def test_gradients_reach_both_heads_and_the_backbone():
    module = _tiny_talker()
    prompt = _prompt(module)
    codes, _, _ = _decode_reference(module, prompt, max_new_tokens=6)

    module.score_utterance(prompt=prompt, target_codes=codes)["loss"].backward()

    def grad_of(param):
        return param.grad is not None and param.grad.abs().sum() > 0

    assert grad_of(module.talker.codec_head.weight), "semantic head got no gradient"
    assert grad_of(module.talker.code_predictor.lm_head[0].weight), "residual head got no gradient"
    assert grad_of(module.talker.model.layers[0].self_attn.q_proj.weight), "backbone got no gradient"
    # The residual loss reads a backbone row, so the predictor's tables must also
    # pull gradient through the frame embedding, not only through their own head.
    assert grad_of(module.talker.code_predictor.get_input_embeddings()[0].weight)


def test_target_codes_must_have_one_row_per_code_group():
    module = _tiny_talker()
    prompt = _prompt(module)
    wrong = torch.zeros(module.config.talker_config.num_code_groups - 1, 4, dtype=torch.long)

    with pytest.raises(ValueError, match="target_codes must be"):
        module.score_utterance(prompt=prompt, target_codes=wrong)


def test_frames_past_the_text_fall_back_to_tts_pad():
    """Speech usually outlasts the text; those frames ride on tts_pad."""
    module = _tiny_talker()
    prompt = _prompt(module, trailing=2)

    rows = module._text_rows(prompt, num_frames=5)

    torch.testing.assert_close(rows[:, :2], prompt["trailing_text_hidden"])
    for step in range(2, 5):
        torch.testing.assert_close(rows[:, step], prompt["tts_pad_embed"][:, 0])


def _batched_loss(module: Qwen3OmniTalker, pairs: list[tuple[dict, torch.Tensor]]) -> torch.Tensor:
    """Drive the graph node with prefills that are already built.

    `forward` calls `build_prompt` itself (it runs the talker's own projections,
    so it has to be inside the FSDP-wrapped call). These tests supply synthetic
    prefills rather than a whole thinker trace, so `build_prompt` hands each one
    back in turn and the loop, the counts and the reduction are what is measured.
    """
    handed = iter(prompt for prompt, _ in pairs)
    original = module.build_prompt
    module.build_prompt = lambda **kwargs: next(handed)
    try:
        return module.forward(
            thinker_embeds=[None] * len(pairs),
            thinker_hiddens=[None] * len(pairs),
            talker_prompts=[{}] * len(pairs),
            target_codes=[codes for _, codes in pairs],
        )["loss"]
    finally:
        module.build_prompt = original


def test_the_batch_loss_is_summed_ce_over_summed_count_not_a_mean_of_means():
    """Independently recomputed from raw CE sums, so a wrong weight cannot hide."""
    module = _tiny_talker()
    a_prompt = _prompt(module)
    b_prompt = _prompt(module, length=8, trailing=2)
    a, _, _ = _decode_reference(module, a_prompt, max_new_tokens=3)
    b, _, _ = _decode_reference(module, b_prompt, max_new_tokens=7)

    semantic_sum = residual_sum = 0.0
    semantic_n = residual_n = 0
    for prompt, codes in ((a_prompt, a), (b_prompt, b)):
        parts = module._utterance_losses(prompt, codes)
        semantic_sum += float(parts.semantic_sum.detach())
        residual_sum += float(parts.residual_sum.detach())
        semantic_n += parts.semantic_count
        residual_n += parts.residual_count
    expected = semantic_sum / semantic_n + residual_sum / residual_n

    batched = _batched_loss(module, [(a_prompt, a), (b_prompt, b)])

    torch.testing.assert_close(batched, torch.tensor(expected, dtype=batched.dtype), rtol=1e-5, atol=1e-5)


def test_the_semantic_count_includes_the_stop_row_and_the_residual_count_does_not():
    module = _tiny_talker()
    prompt = _prompt(module)
    codes, _, _ = _decode_reference(module, prompt, max_new_tokens=5)
    frames = codes.shape[1]
    groups = module.config.talker_config.num_code_groups

    parts = module._utterance_losses(prompt, codes)

    assert parts.semantic_count == frames + 1
    assert parts.residual_count == frames * (groups - 1)


def test_a_batched_loss_still_reaches_the_heads():
    module = _tiny_talker()
    prompt = _prompt(module)
    codes, _, _ = _decode_reference(module, prompt, max_new_tokens=4)

    _batched_loss(module, [(prompt, codes), (prompt, codes)]).backward()

    assert module.talker.codec_head.weight.grad.abs().sum() > 0
    assert module.talker.code_predictor.lm_head[0].weight.grad.abs().sum() > 0


def test_a_micro_batch_with_no_utterance_says_so_rather_than_producing_no_gradient():
    module = _tiny_talker()

    with pytest.raises(ValueError, match="nothing to score"):
        module.forward(thinker_embeds=[], thinker_hiddens=[], talker_prompts=[], target_codes=[])


def test_a_zero_frame_clip_is_refused_rather_than_averaged_into_a_nan():
    module = _tiny_talker()
    prompt = _prompt(module)
    empty = torch.zeros(module.config.talker_config.num_code_groups, 0, dtype=torch.long)

    with pytest.raises(ValueError, match="no codec frames"):
        module.score_utterance(prompt=prompt, target_codes=empty)


def _traced_text(rows: int, hidden: int, role: str = "user") -> ConversationItem:
    return ConversationItem(
        type="text",
        value=torch.zeros(rows, dtype=torch.long),
        role=role,
        meta={THINKER_EMBED: torch.randn(rows, hidden), THINKER_HIDDEN: torch.randn(rows, hidden)},
    )


def _coded_audio(groups: int, frames: int) -> ConversationItem:
    return ConversationItem(
        type="audio",
        value=torch.zeros(1),
        role="assistant",
        meta={TALKER_CODE_LABELS: torch.zeros(groups, frames, dtype=torch.long)},
    )


def test_the_thinker_traces_are_rejoined_in_carrier_order():
    sample = [_traced_text(3, 8), _traced_text(2, 8, role="assistant")]

    embed, hidden = _thinker_traces(sample)

    assert embed.shape == (1, 5, 8)
    assert hidden.shape == (1, 5, 8)
    torch.testing.assert_close(embed[0, :3], sample[0].meta[THINKER_EMBED])
    torch.testing.assert_close(embed[0, 3:], sample[1].meta[THINKER_EMBED])


def test_dummy_rows_hold_no_trace_and_are_not_looked_for():
    sample = [
        _traced_text(3, 8),
        ConversationItem(type="image", value=torch.zeros(4, 8), role="user", is_dummy=True),
    ]

    embed, _ = _thinker_traces(sample)

    assert embed.shape == (1, 3, 8)


def test_a_conversation_without_traces_names_the_flag_that_produces_them():
    with pytest.raises(ValueError, match="collect_thinker_states"):
        _thinker_traces([ConversationItem(type="text", value=torch.zeros(3, dtype=torch.long), role="user")])


def test_the_code_targets_come_off_the_assistant_audio_turn():
    codes = _coded_audio(4, 9)
    sample = [_traced_text(3, 8), codes]

    assert _target_codes(sample) is codes.meta[TALKER_CODE_LABELS]


def test_a_sample_with_no_assistant_audio_has_no_target():
    assert _target_codes([_traced_text(3, 8)]) is None


def test_user_side_audio_is_not_mistaken_for_a_target():
    heard = ConversationItem(
        type="audio", value=torch.zeros(1), role="user", meta={TALKER_CODE_LABELS: torch.zeros(4, 2)}
    )

    assert _target_codes([heard]) is None


def test_two_assistant_utterances_in_one_sample_say_to_split_it():
    sample = [_coded_audio(4, 3), _coded_audio(4, 5)]

    with pytest.raises(ValueError, match="one utterance per prompt"):
        _target_codes(sample)


def test_the_talker_names_the_text_encoder_flag_when_the_prompt_view_is_missing():
    with pytest.raises(ValueError, match="publish_talker_prompt"):
        _talker_prompt([_traced_text(3, 8)])


def test_the_prompt_view_is_found_wherever_on_the_sample_it_was_left():
    """The talker scans, so which row hosts it stays the writer's business."""
    payload = {"segments": [], "supervised": torch.zeros(0, dtype=torch.bool)}
    sample = [_traced_text(3, 8), _traced_text(2, 8, role="assistant")]
    sample[1].meta[TALKER_PROMPT] = payload

    assert _talker_prompt(sample) is payload


def test_the_text_encoder_leaves_one_view_per_sample_on_the_carrier():
    """Round-trip: what the encode hook stashes is what the talker reads back."""
    samples = [[_traced_text(3, 8)], [_traced_text(2, 8), _traced_text(4, 8, role="assistant")]]
    payloads = [{"which": 0}, {"which": 1}]

    _stash_talker_prompts(samples, payloads)

    assert [_talker_prompt(sample) for sample in samples] == payloads


def test_a_sample_with_nothing_in_the_thinker_sequence_has_nowhere_to_leave_it():
    with pytest.raises(ValueError, match="nowhere to leave"):
        _stash_talker_prompts([[_coded_audio(4, 3)]], [{"which": 0}])


def test_the_prefix_is_the_generation_marker_when_the_response_is_teacher_forced():
    """Training shape: rows 9..11 are `<|im_start|>assistant\\n`, 12.. the response."""
    supervised = torch.zeros(20, dtype=torch.bool)
    supervised[12:] = True

    assert _speech_prefix_length(supervised, start=9, sequence_length=20) == 3


def test_a_supervised_earlier_turn_does_not_push_the_prefix_negative():
    """Ordinary multi-turn SFT supervises every assistant turn, including history."""
    supervised = torch.zeros(30, dtype=torch.bool)
    supervised[4:9] = True  # a history assistant turn
    supervised[25:] = True  # the turn being spoken

    # Searched from the final turn's start, so the history turn is behind us.
    assert _speech_prefix_length(supervised, start=22, sequence_length=30) == 3


def test_nothing_supervised_makes_the_whole_given_sequence_the_prompt():
    """The inference shape: rows past `sequence_length` are text just generated."""
    supervised = torch.zeros(12, dtype=torch.bool)

    assert _speech_prefix_length(supervised, start=9, sequence_length=12) == 3


# ── thinker backbone (packing, TMRoPE, traces) ───────────────────────────────


TINY_TEXT_CONFIG = dict(
    num_hidden_layers=6,
    hidden_size=64,
    intermediate_size=64,
    moe_intermediate_size=32,
    num_experts=4,
    num_experts_per_tok=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=64,
    dtype="float32",
)


HIDDEN = 8
MERGE = 2
POS_PER_SEC = 13

VISION_START, VISION_END = 151652, 151653
IMAGE_PAD, VIDEO_PAD = 151655, 151656
AUDIO_START, AUDIO_END, AUDIO_PAD = 151669, 151670, 151675


def _upstream_self():
    """Bind upstream ``get_rope_index`` to a config-carrying namespace.

    Mirrors how the v1 data pipeline calls it (see ``get_position_id`` in the
    patchgen-generated modeling) — no weights needed.
    """
    ns = SimpleNamespace(
        spatial_merge_size=MERGE,
        config=SimpleNamespace(
            image_token_id=IMAGE_PAD,
            video_token_id=VIDEO_PAD,
            audio_token_id=AUDIO_PAD,
            vision_start_token_id=VISION_START,
            audio_start_token_id=AUDIO_START,
            position_id_per_seconds=POS_PER_SEC,
        ),
    )
    ns.get_llm_pos_ids_for_vision = _Upstream.get_llm_pos_ids_for_vision.__get__(ns)
    ns.get_rope_index = _Upstream.get_rope_index.__get__(ns)
    return ns


def _text(n: int) -> ConversationItem:
    return ConversationItem(type="text", value=torch.zeros(n, HIDDEN), role="user")


def _media(kind: str, n: int, grid_thw, num_deepstack: int = 2, **meta) -> ConversationItem:
    return ConversationItem(
        type=kind,
        value=torch.zeros(n, HIDDEN),
        role="user",
        meta={
            "grid_thw": torch.tensor(grid_thw),
            "deepstack": [torch.zeros(n, HIDDEN) for _ in range(num_deepstack)],
            **meta,
        },
    )


def _pack(sample):
    return pack_qwen3omni_conversations_for_forward([sample], torch.device("cpu"), MERGE, POS_PER_SEC)


def test_pack_concatenates_embeds_and_marks_visual_positions():
    grid = (1, 4, 6)
    image_len = 4 * 6 // (MERGE**2)
    sample = [_text(3), _media("image", image_len, grid), _text(2)]
    packed = _pack(sample)

    total = 3 + image_len + 2
    assert packed["inputs_embeds"].shape == (1, total, HIDDEN)
    assert packed["position_ids"].shape == (3, 1, total)
    assert packed["visual_pos_masks"].tolist() == [[False] * 3 + [True] * image_len + [False] * 2]
    # one entry per deepstack layer, each covering the image rows
    assert len(packed["deepstack_visual_embeds"]) == 2
    assert packed["deepstack_visual_embeds"][0].shape == (image_len, HIDDEN)
    assert packed["cu_seq_lens"].tolist() == [0, total]
    assert packed["max_length"] == total


def test_text_and_audio_positions_are_plain_sequential():
    """TMRoPE degenerates to 1-D for text + standalone audio.

    Upstream takes its non-vision branch here (``attention_mask.cumsum``), so
    all three rows must be identical ``arange`` — the same thing the packer
    emits for any non-visual segment.
    """
    audio_len = 7
    sample = [
        _text(4),
        ConversationItem(type="audio", value=torch.zeros(audio_len, HIDDEN), role="user"),
        _text(2),
    ]
    packed = _pack(sample)
    total = 4 + audio_len + 2

    expected = torch.arange(total, dtype=torch.float).view(1, 1, -1).expand(3, 1, -1)
    assert torch.equal(packed["position_ids"], expected)
    # a text/audio-only sample has no DeepStack rows to inject
    assert packed["visual_pos_masks"] is None
    assert packed["deepstack_visual_embeds"] is None


def test_audio_and_video_share_one_wall_clock_position_axis():
    """The temporal row is ``position_id_per_seconds * elapsed_seconds`` for both modalities.

    This is the whole point of TMRoPE and it spans two modules: the audio tower
    fixes the token rate (13 tokens/second, pinned in ``test_qwen3omni_audio``)
    while this packer fixes the position rate -- 1 position per audio token, but
    ``second_per_grid * 13`` per video grid step. If either side drifts, audio
    and video silently describe different timelines and nothing raises. So
    decode the position ids back into seconds and check both modalities land
    where their content actually sits in time.
    """
    audio_seconds = 2.0
    audio_len = int(audio_seconds * POS_PER_SEC)
    # temporal_patch_size=2 sampled at 25fps -> a grid step spans 0.08s.
    second_per_grid = 2 / 25
    grid_t = 5
    grid = (grid_t, 4, 4)
    video_len = grid_t * 4 * 4 // (MERGE**2)

    sample = [
        ConversationItem(type="audio", value=torch.zeros(audio_len, HIDDEN), role="user"),
        _media("video", video_len, grid, second_per_grid=second_per_grid),
    ]
    temporal = _pack(sample)["position_ids"][0, 0]
    seconds = temporal / POS_PER_SEC

    audio_seconds_axis = seconds[:audio_len]
    expected_audio = torch.arange(audio_len, dtype=torch.float) / POS_PER_SEC
    assert torch.allclose(audio_seconds_axis, expected_audio)

    # The video starts one position unit after the audio's last, so its own
    # timeline is offset by the audio's duration -- and from there each grid
    # step must advance the clock by exactly second_per_grid seconds.
    video_seconds_axis = torch.unique(seconds[audio_len:], sorted=True)
    frame_offset = audio_seconds_axis[-1] + 1.0 / POS_PER_SEC
    expected_video = frame_offset + torch.arange(grid_t, dtype=torch.float) * second_per_grid
    assert torch.allclose(video_seconds_axis, expected_video)


@pytest.mark.parametrize(
    ("grid", "second_per_grid"),
    [
        ((1, 4, 6), 1.0),  # still image
        ((1, 2, 2), 1.0),  # single merged token
        ((3, 4, 4), 1.0),  # multi-frame, image convention (t * 1 * pos_per_sec)
        ((3, 4, 4), 0.5),  # video at 2 fps — block max stays integral
        ((2, 4, 4), 0.5),  # video at 2 fps, fractional block max (6.5)
        ((3, 4, 4), 2 / 25),  # video at 12.5 fps, temporal_patch_size=2
    ],
)
def test_vision_positions_match_upstream_get_rope_index(grid, second_per_grid):
    """Segment-level packing must equal upstream's token-level TMRoPE."""
    grid_t, grid_h, grid_w = grid
    media_len = grid_t * grid_h * grid_w // (MERGE**2)
    is_video = second_per_grid != 1.0
    pad_token = VIDEO_PAD if is_video else IMAGE_PAD
    head, tail = 3, 2

    input_ids = torch.tensor(
        [[1] * head + [VISION_START] + [pad_token] * media_len + [VISION_END] + [2] * tail],
        dtype=torch.long,
    )
    grid_thw = torch.tensor([grid], dtype=torch.long)
    ups = _upstream_self()
    upstream_pos, _ = ups.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=None if is_video else grid_thw,
        video_grid_thw=grid_thw if is_video else None,
        # VeOmni's per-video convention: a zero-length audio entry means
        # "this video has no audio track" (see Patch.1 on get_rope_index).
        audio_seqlens=torch.tensor([0], dtype=torch.long) if is_video else None,
        second_per_grids=torch.tensor([second_per_grid]) if is_video else None,
    )

    # Carrier view: <|vision_start|> / <|vision_end|> are ordinary text rows, and
    # the media item stands in for the whole <|*_pad|> run.
    sample = [
        _text(head),
        _text(1),
        _media(
            "video" if is_video else "image",
            media_len,
            grid,
            second_per_grid=second_per_grid,
        ),
        _text(1 + tail),
    ]
    packed = _pack(sample)

    # No dtype coercion on either side: upstream keeps TMRoPE positions in float
    # because ``second_per_grid * position_id_per_seconds`` is usually fractional.
    # Rounding here would shift every position after the clip.
    assert packed["position_ids"].dtype == torch.float
    assert torch.equal(packed["position_ids"], upstream_pos)


def test_vision_position_ids_advance_cursor_by_max_plus_one():
    """The next segment starts at ``previous.max() + 1`` (upstream's rule).

    Getting this wrong is silent: the model still runs, but every token after an
    image sits at the wrong RoPE offset.
    """
    grid = (1, 4, 6)
    image_len = 4 * 6 // (MERGE**2)
    start = 5
    vision_pos = qwen3omni_vision_position_ids(start, torch.tensor(grid), MERGE, position_id_per_seconds=POS_PER_SEC)
    assert vision_pos.shape == (3, image_len)

    sample = [_text(start), _media("image", image_len, grid), _text(3)]
    packed = _pack(sample)
    pos = packed["position_ids"][:, 0, :]
    assert torch.equal(pos[:, start : start + image_len], vision_pos)
    resume = float(vision_pos.max()) + 1.0
    assert pos[:, start + image_len :].tolist() == [[resume, resume + 1, resume + 2]] * 3


def test_multiple_samples_restart_positions_and_extend_cu_seqlens():
    """Varlen packing: each sample keeps its own position origin."""
    packed = pack_qwen3omni_conversations_for_forward(
        [[_text(3)], [_text(5)]], torch.device("cpu"), MERGE, POS_PER_SEC
    )
    assert packed["cu_seq_lens"].tolist() == [0, 3, 8]
    assert packed["max_length"] == 5
    assert packed["position_ids"][0, 0, :].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]


def test_video_without_second_per_grid_is_rejected():
    """Defaulting a video's timeline to 1.0 would be silently wrong, so refuse it.

    ``second_per_grid`` is ``temporal_patch_size / fps`` and only the vision
    preprocessor knows it. A still image is genuinely one instant, so it keeps
    the 1.0 default.
    """
    video = _media("video", 4, (1, 4, 4))
    with pytest.raises(ValueError, match="second_per_grid"):
        _pack([_text(2), video])

    # same grid as an image: allowed, no metadata needed
    assert _pack([_text(2), _media("image", 4, (1, 4, 4))])["position_ids"].shape == (3, 1, 6)


def test_a_clip_with_sound_packs_like_its_silent_twin():
    """Without the time interleave the packer places a clip by its frames alone.

    Built with the data layer's own key and type, so a sound track is stated the
    way a decoded clip states it: the packer reads the frame timeline and skips
    the track, leaving the positions exactly those of the silent clip.
    """
    silent = _media("video", 4, (1, 4, 4), second_per_grid=0.5)
    sounding = _media(
        "video",
        4,
        (1, 4, 4),
        second_per_grid=0.5,
        **{AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=16_000, num_samples=8)},
    )

    assert torch.equal(_pack([_text(2), sounding])["position_ids"], _pack([_text(2), silent])["position_ids"])


def test_a_silent_video_reaches_the_packer(tmp_path):
    """The other half of the guard: it must not refuse a clip with no sound."""
    del tmp_path
    video = _media("video", 4, (1, 4, 4), second_per_grid=0.5)

    assert _pack([_text(2), video])["position_ids"].shape == (3, 1, 6)


def test_zero_length_segments_are_skipped():
    """An empty text row (e.g. an empty assistant turn) must not break packing.

    It carries no position — so it contributes no rows to the packed sequence —
    but it keeps its *slot* in the shape metadata: the hidden-state scatter pairs
    rows to items by position, so an item the packer dropped while the scatter
    still visited it would take the next item's slice and raise ``StopIteration``
    on the one after, leaving the carrier half-overwritten.
    """
    packed = _pack([_text(2), _text(0), _text(3)])
    assert packed["inputs_embeds"].shape == (1, 5, HIDDEN)
    assert packed["position_ids"][0, 0, :].tolist() == [0, 1, 2, 3, 4]
    # Three segments out, one of them empty — not two.
    assert packed["inputs_embeds_shape"].tolist() == [[2], [0], [3]]


def test_an_empty_segment_survives_the_round_trip_to_the_carrier():
    """Packing and the scatter must agree on how many segments there are."""
    from veomni.models.seed_omni.modules.base.llm_packing import scatter_llm_hidden_states
    from veomni.utils.tensor_utils import unflatten

    sample = [_text(2), _text(0), _text(3)]
    packed = _pack(sample)
    segments = unflatten(packed["inputs_embeds"][0], packed["inputs_embeds_shape"])

    scatter_llm_hidden_states([sample], segments)

    assert [tuple(item.value.shape) for item in sample] == [(2, HIDDEN), (0, HIDDEN), (3, HIDDEN)]


def test_thinker_state_collection_matches_a_one_shot_run():
    """The talker reads layer ``accept_hidden_layer``, not the final hidden state.

    Upstream builds its two thinker traces by concatenating ``hidden_states[0]``
    and ``hidden_states[accept_hidden_layer]`` over every generation step. The
    FSM only ever sees one step at a time, so the backbone accumulates them —
    and the result has to equal a single teacher-forced pass over the same
    sequence, otherwise the talker is conditioned on the wrong states.
    """
    accept = 3
    config = Qwen3OmniLlmConfig(
        ops_implementation=make_eager_ops_config(),
        text_config=TINY_TEXT_CONFIG,
        accept_hidden_layer=accept,
        collect_thinker_states=True,
    )
    module = Qwen3OmniLlm(config).eval()
    torch.manual_seed(0)

    prompt_len, decode_steps = 5, 3
    prompt = torch.randn(prompt_len, config.text_config.hidden_size)
    fed_back = [torch.randn(1, config.text_config.hidden_size) for _ in range(decode_steps)]

    module.reset_global_inference_state()
    conversation = [ConversationItem(type="text", value=prompt, role="user")]
    with torch.no_grad():
        module.generate(conversation_list=conversation)
        for embed in fed_back:
            conversation[-1].value = embed  # what the text encoder writes back
            module.generate(conversation_list=conversation)
    states = module.flush_thinker_states()

    full = torch.cat([prompt, *fed_back], dim=0).unsqueeze(0)
    with torch.no_grad():
        reference = module.language_model(inputs_embeds=full, use_cache=False, output_hidden_states=True)

    covered = prompt_len + decode_steps
    assert states["thinker_embed"].shape == (1, covered, config.text_config.hidden_size)
    assert torch.equal(states["thinker_embed"], full)
    assert torch.allclose(states["thinker_hidden"], reference.hidden_states[accept], atol=1e-5)
    # guard the whole point of the exercise: the accept layer is not the last one
    assert not torch.allclose(reference.hidden_states[accept], reference.last_hidden_state)
    # flush clears, so a second conversation cannot inherit the first one's trace
    assert module._accept_hidden_cache == []


def test_thinker_states_are_not_collected_by_default():
    """All-layer hidden states are a memory cost text-only inference should not pay."""
    config = Qwen3OmniLlmConfig(ops_implementation=make_eager_ops_config(), text_config=TINY_TEXT_CONFIG)
    assert config.collect_thinker_states is False

    module = Qwen3OmniLlm(config).eval()
    module.reset_global_inference_state()
    conversation = [ConversationItem(type="text", value=torch.randn(4, config.text_config.hidden_size), role="user")]
    with torch.no_grad():
        module.generate(conversation_list=conversation)
    assert module._accept_hidden_cache == []


def test_a_single_sample_batch_gets_an_explicit_causal_mask():
    """A one-segment ``cu_seq_lens`` is declined by the packed-mask router.

    ``mask_utils._should_use_packed_mask`` requires more segments than batch rows
    ("one segment per row is not packing"), so a single-sample batch falls through
    to Transformers' ``find_packed_sequence_indices``. That helper treats a
    position *gap* as a boundary, and TMRoPE's temporal row both repeats across a
    vision span and jumps after it — so the row reads as several sequences, and
    the resulting mask indexes past the row count. The wrapper supplies an
    explicit mask for exactly that shape.

    Asserted on behaviour, not on the source: the point is that the forward runs
    and its result does not depend on the mask the backend would have derived.
    """
    from veomni.models.utils.mask_utils import _should_use_packed_mask

    module = _accelerated_llm()
    sample = _text_sample(module, [3, 4])
    inputs = module.pre_forward(method="forward", conversation_list=[sample])

    # The shape really is the declined one — otherwise this test proves nothing.
    assert inputs["cu_seq_lens_q"].numel() == 2
    assert not _should_use_packed_mask(inputs["cu_seq_lens_q"], "eager", 1)

    # The backend's own heuristic misreads a TMRoPE row containing a vision span
    # (it repeats there, then jumps) — that is the case the explicit mask guards.
    from transformers.masking_utils import find_packed_sequence_indices

    tmrope_with_vision = torch.tensor([[0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0]])
    assert find_packed_sequence_indices(tmrope_with_vision) is not None

    # ...and the wrapper answers it with a mask of the packed row's own width, so
    # the backend never runs that heuristic. A mask of the wrong length is exactly
    # what made this raise, so its shape is the observable.
    seen: dict[str, object] = {}
    real_forward = module.language_model.forward

    def spy(*args, **kwargs):
        seen["attention_mask"] = kwargs.get("attention_mask")
        return real_forward(*args, **kwargs)

    module.language_model.forward = spy
    try:
        with torch.no_grad():
            module.forward(**inputs)
    finally:
        del module.language_model.forward

    mask = seen["attention_mask"]
    assert mask is not None, "the backend would derive the mask from TMRoPE positions and misread them"
    assert mask.shape == (1, inputs["inputs_embeds"].size(1))
    assert mask.all()


def test_the_router_balancing_loss_goes_through_the_bound_op():
    """`load_balancing_loss_implementation` has to reach this objective.

    Calling the standalone function directly ignores the row, so a fused backend
    configured for the run would silently not be used for the auxiliary loss.
    """
    module = _accelerated_llm(text_config={"output_router_logits": True, "router_aux_loss_coef": 0.01})

    # The handle exists and is what `router_balancing_loss` uses.
    from veomni.ops import VeomniOp

    assert isinstance(module.veomni_load_balancing_loss, VeomniOp)

    cfg = module.config.text_config
    gate_logits = (torch.randn(6, cfg.num_experts),)
    with torch.no_grad():
        loss = module.router_balancing_loss(gate_logits)
    assert loss.numel() == 1 and torch.isfinite(loss)


def test_the_router_loss_is_computed_over_the_whole_sequence_under_sp():
    """The balancing term is a *product* of batch means, so per-rank values do not
    average to the full-batch value; the logits are gathered instead."""
    module = _accelerated_llm(text_config={"output_router_logits": True})
    # With sp_size == 1 the helper is the identity, which is the baseline the
    # gathered path has to match.
    logits = (torch.randn(4, module.config.text_config.num_experts),)
    assert module._full_sequence_router_logits(logits) is logits


def test_the_router_gather_still_has_its_length_when_the_loss_runs():
    """`_sp_router_own_len` must survive until the router loss has consumed it.

    `forward_post` clears its per-forward state after scattering, and the router
    block sits *after* that clear. Clearing the router length with the rest left
    `None` reaching `.narrow(..., None)` — a `TypeError` deep inside the gather,
    on every SP step with the router objective on. The single-device test above
    cannot see it, which is what let it through.

    Drives `forward_post` itself rather than the helper: the bug is an ordering
    one inside that method, so calling the consumer directly would pass either
    way.
    """
    from unittest import mock

    module = _accelerated_llm(text_config={"output_router_logits": True})
    sample = _text_sample(module, [4])
    module.pre_forward(method="forward", conversation_list=[sample])

    n_experts = module.config.text_config.num_experts
    hidden = module.config.text_config.hidden_size
    outputs = {
        "hidden_states": torch.randn(1, 4, hidden),
        "router_logits": (torch.randn(4, n_experts),),
        "router_attention_mask": None,
        "all_hidden_states": None,
    }
    module._pack_inputs_embeds_shape = torch.tensor([[4]])

    def fake_gather(t, **kw):
        return t

    with mock.patch(
        "veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.gather_outputs",
        side_effect=fake_gather,
    ):
        with mock.patch(
            "veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state"
        ) as ps:
            ps.return_value.sp_size = 2
            ps.return_value.sp_group = object()
            # `forward_pre` would have stored both of these; the ordering bug is
            # what clears the router one before the loss runs.
            module._sp_own_len = 4
            module._sp_router_own_len = 4
            result = module.forward_post(**outputs)

    assert "_loss" in result, "the router objective contributed nothing"
    assert module._sp_router_own_len is None, "per-forward state must still be cleared"


def test_an_empty_segment_keeps_its_slot_without_inferring_a_width():
    """An empty row is kept so packing and the hidden-state scatter agree.

    Every backbone-input producer emits a 2-D `(rows, hidden)` row, and an empty
    one keeps its hidden width (`(0, hidden)`), so the packer only has to *keep*
    the row. Checked in every position, since an empty segment may come before
    the one that would otherwise supply the width.
    """
    for label, sample in (
        ("leading", [_text(0), _text(3)]),
        ("middle", [_text(2), _text(0), _text(3)]),
        ("trailing", [_text(2), _text(0)]),
    ):
        packed = _pack(sample)
        rows = sum(int(item.value.size(0)) for item in sample)
        assert packed["inputs_embeds"].shape == (1, rows, HIDDEN), label
        # The empty row keeps its slot, so the two sides count the same items.
        assert packed["inputs_embeds_shape"].shape[0] == len(sample), label


def test_the_backbone_registers_the_expert_fusion_converter():
    """Without it an upstream-root load leaves the experts at their fused-shape
    *initialisation*: the per-expert keys match no parameter, and nothing raises.
    """
    from veomni.models.seed_omni.modules.qwen3omni.llm.modeling import Qwen3OmniLlm

    assert hasattr(Qwen3OmniLlm, "_create_checkpoint_tensor_converter")
    assert hasattr(Qwen3OmniLlm, "_convert_fqn_to_index_mapping")


def test_generate_refuses_sequence_parallelism():
    """`forward_pre` slices the sample per rank; `generate` has no such hook.

    So under Ulysses the exchange would rebuild the full sequence from a prompt
    every rank already holds (4 rows in, 8 out) while `cu_seq_lens` still
    described the original — a wrong answer rather than a crash. The shared
    `SimpleArGenerationMixin` refuses the same configuration.
    """
    from unittest import mock

    module = _accelerated_llm()
    module.reset_global_inference_state()
    conversation = [ConversationItem(type="text", value=torch.randn(4, HIDDEN), role="user")]

    fake_ps = mock.MagicMock()
    fake_ps.sp_size = 2
    with mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.modeling.get_parallel_state", return_value=fake_ps):
        with pytest.raises(ValueError, match="sp_size=1"):
            module.generate(conversation_list=conversation)


def test_dummy_items_are_excluded_but_donate_deepstack():
    """FSDP dummy rows must not enter the sequence, yet still anchor DeepStack.

    Without the anchor the vision merger that produced the dummy features gets
    no gradient on this rank and FSDP's reduce-scatter desyncs.
    """
    dummy = _media("image", 4, (1, 4, 4))
    dummy.is_dummy = True
    packed = _pack([_text(3), dummy])

    assert packed["inputs_embeds"].shape == (1, 3, HIDDEN)
    assert packed["visual_pos_masks"] is None
    assert len(packed["deepstack_visual_embeds"]) == 2


# ── training path: handing the talker its two traces via the carrier ──────────


def _accelerated_llm(**config_overrides) -> "Qwen3OmniLlmAccelerated":
    torch.manual_seed(0)
    # The MoE objective's flags live on the *text* config, which is where the
    # upstream thinker reads them; they are not fields of the wrapper config.
    text_config = {**TINY_TEXT_CONFIG, **config_overrides.pop("text_config", {})}
    config = Qwen3OmniLlmConfig(
        ops_implementation=make_eager_ops_config(), text_config=text_config, **config_overrides
    )
    return Qwen3OmniLlmAccelerated(config).eval()


def _run_forward_node(module, conversation_list):
    """Drive the node the way OmniModel._run_train_node does."""
    inputs = module.pre_forward(method="forward", conversation_list=conversation_list)
    outputs = module.forward(**inputs)
    return module.post_forward(method="forward", **outputs), inputs


def _text_sample(module, lengths):
    hidden = module.config.text_config.hidden_size
    torch.manual_seed(3)
    return [ConversationItem(type="text", value=torch.randn(n, hidden), role="user") for n in lengths]


def test_training_leaves_the_thinker_traces_on_the_carrier_items():
    """The talker reads its prompt off the carrier, so the thinker must write it there.

    Inference cannot do this -- it accumulates per-step traces and hands them
    over through the FSM's ctx -- but training is one pass over the packed
    sequence, so the same per-item split that returns hidden states to
    ``item.value`` returns these to ``item.meta``.
    """
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    sample = _text_sample(module, [4, 3])
    originals = [item.value.clone() for item in sample]

    with torch.no_grad():
        out, _ = _run_forward_node(module, [sample])

    assert out["conversation_list"][0] is sample
    for item, original in zip(sample, originals, strict=True):
        # index 0 of the layer stack IS the packed input embedding, so the
        # published trace must be exactly what this item fed in.
        assert torch.equal(item.meta[THINKER_EMBED], original)
        assert item.meta[THINKER_HIDDEN].shape == original.shape
        # item.value stays the final hidden state -- that belongs to the text head.
        assert item.value.shape == original.shape
        assert not torch.equal(item.value, original)


def test_the_published_hidden_trace_is_the_layer_the_talker_accepts():
    """Not the last layer. The lm_head reads that one; the talker reads layer 24."""
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    sample = _text_sample(module, [5])

    with torch.no_grad():
        _, inputs = _run_forward_node(module, [sample])
        reference = module.forward(**inputs)["all_hidden_states"]

    published = sample[0].meta[THINKER_HIDDEN]
    accept = reference[2].squeeze(0)
    final = reference[-1].squeeze(0)
    assert torch.equal(published, accept)
    assert not torch.equal(accept, final), "tiny model made the check vacuous"


def test_thinker_traces_are_detached_from_the_thinker():
    """This round trains the talker against a frozen stack.

    A live graph here would keep every thinker activation alive to produce
    parameter gradients that are masked off anyway.
    """
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    sample = _text_sample(module, [4])
    _run_forward_node(module, [sample])

    assert not sample[0].meta[THINKER_EMBED].requires_grad
    assert not sample[0].meta[THINKER_HIDDEN].requires_grad


def test_thinker_only_training_publishes_nothing():
    """Retaining every layer is a memory cost a graph without a talker should not pay."""
    module = _accelerated_llm()
    sample = _text_sample(module, [4])

    with torch.no_grad():
        _, inputs = _run_forward_node(module, [sample])

    assert inputs["output_hidden_states"] is False
    assert THINKER_EMBED not in sample[0].meta
    assert THINKER_HIDDEN not in sample[0].meta


def test_traces_skip_dummy_items_like_the_hidden_state_scatter_does():
    """Dummies never enter the packed sequence, so they have no slice to receive."""
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    hidden = module.config.text_config.hidden_size
    real = _text_sample(module, [4])[0]
    dummy = ConversationItem(
        type="image",
        value=torch.zeros(4, hidden),
        role="user",
        is_dummy=True,
        meta={"grid_thw": torch.tensor((1, 4, 4)), "deepstack": [torch.zeros(4, hidden) for _ in range(2)]},
    )

    with torch.no_grad():
        _run_forward_node(module, [[real, dummy]])

    assert THINKER_EMBED in real.meta
    assert THINKER_EMBED not in dummy.meta


def test_an_accept_layer_past_the_backbone_depth_says_so():
    """Bare indexing raises IndexError from inside a post hook, naming nothing."""
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=99)
    sample = _text_sample(module, [4])
    with pytest.raises(ValueError, match="accept_hidden_layer=99"):
        _run_forward_node(module, [sample])
