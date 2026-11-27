"""Tests for Qwen3-Omni talker (prompt assembly, teacher forcing, loss) and thinker LLM (packing, TMRoPE, traces, SP)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from transformers.masking_utils import find_packed_sequence_indices
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration as _Ref

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.hf_paths import hf_local_or_remote
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata
from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.modules.base.llm_packing import scatter_llm_hidden_states
from veomni.models.seed_omni.modules.base.sp_shard import sp_local_len, sp_valid_in_window
from veomni.models.seed_omni.modules.qwen3omni.carrier import TALKER_CODES_KEY
from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import TALKER_CODE_LABELS
from veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated import (
    Qwen3OmniLlmAccelerated,
    _fold_fsdp_dummy_anchors,
    _qwen3omni_modality_groups,
)
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
from veomni.models.seed_omni.modules.qwen3omni.talker.modeling import (
    SIGNAL_CODES_DONE,
    Qwen3OmniTalker,
    _chatml_turns,
    _speech_prefix_length,
    _speech_span_end,
    collect_media_metadata,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.accelerated.accelerated import _stash_talker_prompts
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.chat_template import Qwen3OmniChatTemplate
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import (
    PromptSegment,
    talker_prompt_segments,
    talker_supervised_mask,
)
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem
from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as _Upstream,
)
from veomni.models.utils.mask_utils import _should_use_packed_mask
from veomni.ops import VeomniOp
from veomni.utils.tensor_utils import unflatten


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
CONFIG_DIR = Path(__file__).parents[4] / "tests" / "toy_config" / "qwen3omni_toy"
TOKENIZER = hf_local_or_remote("Qwen/Qwen3-Omni-30B-A3B-Instruct")
requires_tokenizer = pytest.mark.skipif(
    not os.path.isdir(TOKENIZER), reason=f"needs a Qwen3-Omni tokenizer at {TOKENIZER}"
)

THINKER_HIDDEN_SIZE = 24
IM_START, SYSTEM, USER, ASSISTANT = 1000, 1001, 1002, 1003
IM_END, AUDIO_START, AUDIO_END, AUDIO_PAD = 1004, 1005, 1006, 1007
TTS_BOS, TTS_EOS, TTS_PAD = 1008, 1009, 1010
ROLE_TOKEN_IDS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}

_WTE = None


def _thinker_wte():
    global _WTE
    if _WTE is None:
        torch.manual_seed(1)
        _WTE = torch.nn.Embedding(1200, THINKER_HIDDEN_SIZE)
    return _WTE


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


def _tiny_talker() -> Qwen3OmniTalker:
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
    for name, param in module.named_parameters():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            with torch.no_grad():
                param.normal_(mean=0.0, std=0.02)
    return module


def _upstream_assemble(module, prompt_ids, thinker_embed, thinker_hidden, speaker_id, total_length):
    shim = SimpleNamespace(
        talker=module.talker,
        config=SimpleNamespace(talker_config=module.config.talker_config, tts_pad_token_id=TTS_PAD),
    )
    shim._get_talker_user_parts = _Ref._get_talker_user_parts.__get__(shim)
    shim._get_talker_assistant_parts = _Ref._get_talker_assistant_parts.__get__(shim)

    tts_ids = torch.tensor([[TTS_BOS, TTS_EOS, TTS_PAD]], dtype=torch.long)
    tts_bos, tts_eos, tts_pad = module.talker.text_projection(_thinker_wte()(tts_ids)).chunk(3, dim=1)
    multimodal_mask = (prompt_ids == AUDIO_PAD).unsqueeze(0)
    starts = torch.nonzero(prompt_ids == IM_START).reshape(-1).tolist()
    boundaries = starts + [total_length]

    embeds, ids, trailing = [], [], None
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        role_token = int(prompt_ids[start + 1])
        if role_token == SYSTEM or (role_token == ASSISTANT and index != len(boundaries) - 2):
            continue
        elif role_token == USER:
            embeds.append(shim._get_talker_user_parts(start, end, multimodal_mask, thinker_hidden, thinker_embed))
            ids.append(prompt_ids[start:end].unsqueeze(0))
        elif role_token == ASSISTANT:
            span_embeds, span_ids, trailing = shim._get_talker_assistant_parts(
                start, end, speaker_id, thinker_embed, tts_pad, tts_bos, tts_eos
            )
            embeds.append(span_embeds)
            ids.append(span_ids)
    return {
        "inputs_embeds": torch.cat(embeds, dim=1),
        "talker_input_ids": torch.cat(ids, dim=1),
        "trailing_text_hidden": trailing,
    }


def _build_case(history: bool):
    rows = [
        ("text", "system", [IM_START, SYSTEM, 20, 21, IM_END]),
        ("text", "user", [IM_START, USER, 30, AUDIO_START]),
        ("audio", "user", [AUDIO_PAD] * 7),
        ("text", "user", [AUDIO_END, 31, 32, IM_END]),
    ]
    surviving_user_rows = 15
    if history:
        rows.append(("text", "assistant", [IM_START, ASSISTANT, 40, 41, IM_END]))
        rows.append(("text", "user", [IM_START, USER, 50, 51, IM_END]))
        surviving_user_rows += 5
    rows.append(("text", "assistant", [IM_START, ASSISTANT, 60]))

    prompt_ids = torch.tensor([tok for _, _, toks in rows for tok in toks], dtype=torch.long)
    segments = [PromptSegment(role, item_type, len(toks)) for item_type, role, toks in rows]
    num_generated = 5
    torch.manual_seed(2)
    return SimpleNamespace(
        prompt_ids=prompt_ids,
        segments=segments,
        thinker_embed=torch.randn(1, prompt_ids.numel() + num_generated, THINKER_HIDDEN_SIZE),
        thinker_hidden=torch.randn(1, prompt_ids.numel() + num_generated, THINKER_HIDDEN_SIZE),
        num_generated=num_generated,
        surviving_user_rows=surviving_user_rows,
    )


def _talker_prompt_payload(prompt_ids, segments, supervised=None):
    return {
        "input_ids": prompt_ids,
        "segments": segments,
        "supervised": torch.zeros(sum(s.length for s in segments), dtype=torch.bool)
        if supervised is None
        else supervised,
        "tts_embeds": _thinker_wte()(torch.tensor([[TTS_BOS, TTS_EOS, TTS_PAD]], dtype=torch.long)),
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


# ── talker prompt & generation ───────────────────────────────────────────────


@pytest.mark.parametrize("history", [False, True], ids=["single_turn", "multi_turn_with_history"])
def test_prompt_assembly_matches_upstream(history):
    module = _tiny_talker()
    case = _build_case(history)
    speaker_id = module.resolve_speaker_id("ethan")

    with torch.no_grad():
        expected = _upstream_assemble(
            module, case.prompt_ids, case.thinker_embed, case.thinker_hidden, speaker_id, case.thinker_embed.size(1)
        )
        actual = _module_prompt(module, case)

    torch.testing.assert_close(actual["inputs_embeds"], expected["inputs_embeds"], rtol=0, atol=0)
    torch.testing.assert_close(actual["trailing_text_hidden"], expected["trailing_text_hidden"], rtol=0, atol=0)
    assert torch.equal(actual["talker_input_ids"], expected["talker_input_ids"])
    assert (actual["talker_input_ids"] != SYSTEM).all()
    # Speech span is exactly the 3 generation-prefix rows + 6 codec rows (never merged with assistant history).
    assert actual["inputs_embeds"].size(1) == case.surviving_user_rows + 9
    assert torch.equal(actual["talker_input_ids"][0, -9:], torch.full((9,), TTS_PAD, dtype=torch.long))
    assert (actual["talker_input_ids"][0, :-9] != TTS_PAD).all()
    # Trailing text holds remaining generated rows plus tts_eos.
    assert actual["trailing_text_hidden"].size(1) == case.num_generated


def test_talker_prompt_errors_fsdp_units_and_mixed_precision():
    module = _tiny_talker()
    case = _build_case(history=False)

    with pytest.raises(ValueError, match="unknown speaker"):
        _module_prompt(module, case, speaker="nobody")
    bad_segments = list(case.segments)
    bad_segments[2] = PromptSegment("user", "audio", 6)
    with pytest.raises(ValueError, match="segments sum to"):
        _module_prompt(module, case, segments=bad_segments)
    with pytest.raises(ValueError, match="thinker trace is shorter"):
        _module_prompt(
            module, case, thinker_embed=case.thinker_embed[:, :5], thinker_hidden=case.thinker_hidden[:, :5]
        )
    with pytest.raises(ValueError, match="flush_thinker_states"):
        module.generate(conversation_list=[], thinker_embed=None, thinker_hidden=None, talker_prompt=None)

    # Code predictor stays in root FSDP unit so `embed_code_frames` can index its tables without calling it.
    assert set(module._no_split_modules) == {
        "Qwen3OmniMoeTalkerDecoderLayer",
        "Qwen3OmniMoeTalkerCodePredictorDecoderLayer",
    }
    assert type(module.talker.code_predictor).__name__ not in module._no_split_modules
    assert hasattr(Qwen3OmniTalker, "_create_checkpoint_tensor_converter")
    assert hasattr(Qwen3OmniTalker, "_convert_fqn_to_index_mapping")

    # Under FSDP2 mixed precision, projections run in bf16 while sharded decoder layers report fp32.
    module.talker.text_projection.to(torch.bfloat16)
    module.talker.hidden_projection.to(torch.bfloat16)
    module.talker.get_input_embeddings().to(torch.bfloat16)
    assert module.talker.dtype == torch.float32 and module.compute_dtype == torch.bfloat16
    assert _module_prompt(module, case)["inputs_embeds"].dtype == torch.bfloat16


def test_generate_appends_codes_and_forwards_media_metadata(monkeypatch):
    module = _tiny_talker()
    case = _build_case(history=False)
    num_groups = module.config.talker_config.num_code_groups
    seen = {}

    def fake_generate_codes(prompt, kwargs, media_metadata=None):
        seen["media_metadata"] = media_metadata
        return torch.randint(0, 100, (num_groups, 12))

    monkeypatch.setattr(module, "generate_codes", fake_generate_codes)
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
    assert codes_item.type == "output" and codes_item.meta[TALKER_CODES_KEY] is True
    assert codes_item.value.shape == (num_groups, 12)

    meta = seen["media_metadata"]
    assert meta["audio_feature_lengths"].tolist() == [28]
    assert meta["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert meta["video_grid_thw"].tolist() == [[2, 4, 4]]
    assert meta["video_second_per_grid"].tolist() == [0.5]
    assert meta["use_audio_in_video"] is False

    # Text-only prompt emits None (never empty tensors) for all media metadata keys.
    text_meta = collect_media_metadata([ConversationItem(type="text", value=torch.zeros(3), role="user")])
    assert all(
        text_meta[k] is None
        for k in ("audio_feature_lengths", "image_grid_thw", "video_grid_thw", "video_second_per_grid")
    )


@requires_tokenizer
def test_training_shaped_prompt_leaves_response_to_be_spoken():
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
    seq_len = sum(s.length for s in segments)
    assert input_ids.numel() == seq_len == supervised.numel()
    assert all(s.type != "audio" for s in segments)

    _, start, _ = _chatml_turns(
        input_ids,
        total_length=seq_len,
        im_start_token_id=template.im_start_token_id,
        role_token_ids=template.role_token_ids,
    )[-1]
    prefix_len = _speech_prefix_length(supervised, start, seq_len)
    speech_end = _speech_span_end(
        input_ids, start, seq_len, trace_length=seq_len, im_end_token_id=template.im_end_token_id
    )
    assert template.tokenizer.decode(input_ids[start : start + prefix_len]) == "<|im_start|>assistant\n"
    assert supervised[speech_end]
    assert template.tokenizer.decode(input_ids[start + prefix_len : speech_end]) == "Hello there, nice to meet you!"


# ── talker teacher-forced training ───────────────────────────────────────────


def _prompt(module: Qwen3OmniTalker, length: int = 6, trailing: int = 3) -> dict:
    torch.manual_seed(1)
    hidden = module.config.talker_config.text_config.hidden_size
    vocab = module.config.talker_config.text_config.vocab_size
    return {
        "inputs_embeds": torch.randn(1, length, hidden),
        "talker_input_ids": torch.randint(0, vocab - 1024, (1, length)),
        "trailing_text_hidden": torch.randn(1, trailing, hidden),
        "tts_pad_embed": torch.randn(1, 1, hidden),
    }


def _decode_reference(module: Qwen3OmniTalker, prompt: dict, max_new_tokens: int) -> tuple:
    talker, config = module.talker, module.config.talker_config
    predictor = talker.code_predictor
    residual_logits: list[torch.Tensor] = []
    original = predictor.generate

    def taped(*args, **kwargs):
        kwargs["output_logits"] = True
        res = original(*args, **kwargs)
        residual_logits.append(torch.stack(res.logits, dim=0).float())
        return res

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

    codes = (
        torch.stack([step[-1] for step in output.hidden_states if step[-1] is not None], dim=1)
        .transpose(1, 2)
        .squeeze(0)
    )
    semantic = torch.stack(output.logits, dim=0).float()
    return codes, semantic, residual_logits


def _teacher_forced_hidden(module: Qwen3OmniTalker, prompt: dict, codes: torch.Tensor) -> torch.Tensor:
    sequence = torch.cat(
        (prompt["inputs_embeds"], module.embed_code_frames(codes) + module._text_rows(prompt, codes.shape[1])), dim=1
    )
    with torch.no_grad():
        return module.talker.model(
            input_ids=None,
            attention_mask=None,
            position_ids=module._teacher_forced_positions(prompt, codes.shape[1]),
            past_key_values=None,
            inputs_embeds=sequence,
            use_cache=False,
        ).last_hidden_state


def test_teacher_forcing_matches_decode_logits_for_both_heads():
    module = _tiny_talker()
    prompt = _prompt(module)
    codes, ref_semantic, ref_residual = _decode_reference(module, prompt, max_new_tokens=8)
    prompt_len = prompt["inputs_embeds"].shape[1]
    frames = codes.shape[1]
    num_groups = module.config.talker_config.num_code_groups

    hidden = _teacher_forced_hidden(module, prompt, codes)
    semantic_logits = module.talker.codec_head(
        hidden[:, prompt_len - 1 : prompt_len - 1 + ref_semantic.shape[0]]
    ).float()
    torch.testing.assert_close(semantic_logits[0], ref_semantic[:, 0], rtol=1e-4, atol=1e-4)

    predictor = module.talker.code_predictor
    tables = predictor.get_input_embeddings()
    rows = [
        hidden[0, prompt_len - 1 : prompt_len - 1 + frames].unsqueeze(1),
        module.talker.get_input_embeddings()(codes[0]).unsqueeze(1),
        *(tables[g - 1](codes[g]).unsqueeze(1) for g in range(1, num_groups - 1)),
    ]
    with torch.no_grad():
        pred_hidden = predictor.model(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=torch.cat(rows, dim=1),
            use_cache=False,
        ).last_hidden_state
    residual_logits = torch.stack(
        [predictor.lm_head[g - 1](pred_hidden[:, g]) for g in range(1, num_groups)], dim=1
    ).float()
    torch.testing.assert_close(
        residual_logits, torch.cat(ref_residual, dim=1).permute(1, 0, 2)[:frames], rtol=1e-4, atol=1e-4
    )


def test_score_utterance_weights_grads_pad_fallback_and_validation():
    module = _tiny_talker()
    prompt = _prompt(module, trailing=2)
    codes, _, _ = _decode_reference(module, prompt, max_new_tokens=6)

    # Frames past trailing text fall back to tts_pad_embed.
    text_rows = module._text_rows(prompt, num_frames=5)
    torch.testing.assert_close(text_rows[:, :2], prompt["trailing_text_hidden"])
    for step in range(2, 5):
        torch.testing.assert_close(text_rows[:, step], prompt["tts_pad_embed"][:, 0])

    # Loss = semantic + residual, sub-metrics detached, weights scale independently, grads hit all parts.
    out = module.score_utterance(prompt=prompt, target_codes=codes)
    assert out["loss"].ndim == 0 and out["loss"].requires_grad
    assert not out["semantic_loss"].requires_grad and not out["residual_loss"].requires_grad
    torch.testing.assert_close(out["loss"], out["semantic_loss"] + out["residual_loss"])
    torch.testing.assert_close(
        module.score_utterance(prompt=prompt, target_codes=codes, residual_loss_weight=0.0)["loss"],
        out["semantic_loss"],
    )

    out["loss"].backward()
    for param in (
        module.talker.codec_head.weight,
        module.talker.code_predictor.lm_head[0].weight,
        module.talker.model.layers[0].self_attn.q_proj.weight,
        module.talker.code_predictor.get_input_embeddings()[0].weight,
    ):
        assert param.grad is not None and param.grad.abs().sum() > 0

    groups = module.config.talker_config.num_code_groups
    with pytest.raises(ValueError, match="target_codes must be"):
        module.score_utterance(prompt=prompt, target_codes=torch.zeros(groups - 1, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="no codec frames"):
        module.score_utterance(prompt=prompt, target_codes=torch.zeros(groups, 0, dtype=torch.long))


def test_batched_loss_reduces_by_total_counts_and_backprops():
    module = _tiny_talker()
    a_prompt, b_prompt = _prompt(module), _prompt(module, length=8, trailing=2)
    a, _, _ = _decode_reference(module, a_prompt, max_new_tokens=3)
    b, _, _ = _decode_reference(module, b_prompt, max_new_tokens=7)
    groups = module.config.talker_config.num_code_groups

    sem_sum = res_sum = 0.0
    sem_n = res_n = 0
    for prompt, codes in ((a_prompt, a), (b_prompt, b)):
        parts = module._utterance_losses(prompt, codes)
        # Semantic includes EOS row; residual does not.
        assert parts.semantic_count == codes.shape[1] + 1
        assert parts.residual_count == codes.shape[1] * (groups - 1)
        sem_sum += float(parts.semantic_sum.detach())
        res_sum += float(parts.residual_sum.detach())
        sem_n += parts.semantic_count
        res_n += parts.residual_count

    pairs = [(a_prompt, a), (b_prompt, b)]
    handed = iter(p for p, _ in pairs)
    with mock.patch.object(module, "build_prompt", side_effect=lambda **_: next(handed)):
        batched = module.forward(
            thinker_embeds=[None, None], thinker_hiddens=[None, None], talker_prompts=[{}, {}], target_codes=[a, b]
        )["loss"]
    torch.testing.assert_close(
        batched, torch.tensor(sem_sum / sem_n + res_sum / res_n, dtype=batched.dtype), rtol=1e-5, atol=1e-5
    )
    batched.backward()
    assert module.talker.codec_head.weight.grad.abs().sum() > 0
    assert module.talker.code_predictor.lm_head[0].weight.grad.abs().sum() > 0

    with pytest.raises(ValueError, match="nothing to score"):
        module.forward(thinker_embeds=[], thinker_hiddens=[], talker_prompts=[], target_codes=[])


def test_talker_carrier_helpers_and_prefix_length():
    def traced(rows, role="user"):
        return ConversationItem(
            type="text",
            value=torch.zeros(rows, dtype=torch.long),
            role=role,
            meta={THINKER_EMBED: torch.randn(rows, 8), THINKER_HIDDEN: torch.randn(rows, 8)},
        )

    def coded(frames):
        return ConversationItem(
            type="audio",
            value=torch.zeros(1),
            role="assistant",
            meta={TALKER_CODE_LABELS: torch.zeros(4, frames, dtype=torch.long)},
        )

    # _thinker_traces rejoins in carrier order, skips dummies, raises if missing.
    s0, s1 = traced(3), traced(2, "assistant")
    dummy = ConversationItem(type="image", value=torch.zeros(4, 8), role="user", is_dummy=True)
    embed, hidden = _thinker_traces([s0, dummy, s1])
    assert embed.shape == hidden.shape == (1, 5, 8)
    torch.testing.assert_close(embed[0, :3], s0.meta[THINKER_EMBED])
    torch.testing.assert_close(embed[0, 3:], s1.meta[THINKER_EMBED])
    with pytest.raises(ValueError, match="collect_thinker_states"):
        _thinker_traces([ConversationItem(type="text", value=torch.zeros(3, dtype=torch.long), role="user")])

    # _target_codes picks assistant audio only and rejects multiple utterances.
    c0 = coded(9)
    assert _target_codes([s0, c0]) is c0.meta[TALKER_CODE_LABELS]
    assert _target_codes([s0]) is None
    assert (
        _target_codes(
            [
                ConversationItem(
                    type="audio", value=torch.zeros(1), role="user", meta={TALKER_CODE_LABELS: torch.zeros(4, 2)}
                )
            ]
        )
        is None
    )
    with pytest.raises(ValueError, match="one utterance per prompt"):
        _target_codes([coded(3), coded(5)])

    # _stash_talker_prompts / _talker_prompt round-trip and validation.
    samples = [[traced(3)], [traced(2), traced(4, "assistant")]]
    payloads = [{"which": 0}, {"which": 1}]
    _stash_talker_prompts(samples, payloads)
    assert [_talker_prompt(s) for s in samples] == payloads
    with pytest.raises(ValueError, match="publish_talker_prompt"):
        _talker_prompt([traced(3)])
    with pytest.raises(ValueError, match="nowhere to leave"):
        _stash_talker_prompts([[coded(3)]], [{"which": 0}])

    # _speech_prefix_length: single-turn SFT, multi-turn SFT (prior assistant supervised), inference (none supervised).
    sup = torch.zeros(30, dtype=torch.bool)
    sup[4:9] = True
    sup[25:] = True
    assert _speech_prefix_length(sup, start=22, sequence_length=30) == 3
    assert _speech_prefix_length(torch.zeros(12, dtype=torch.bool), start=9, sequence_length=12) == 3


# ── thinker backbone (packing, TMRoPE, traces, SP handoff) ───────────────────


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
HIDDEN, MERGE, POS_PER_SEC = 8, 2, 13
VISION_START, VISION_END = 151652, 151653
IMAGE_PAD, VIDEO_PAD = 151655, 151656
UP_AUDIO_START, UP_AUDIO_PAD = 151669, 151675


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


def _accelerated_llm(**config_overrides) -> Qwen3OmniLlmAccelerated:
    torch.manual_seed(0)
    text_config = {**TINY_TEXT_CONFIG, **config_overrides.pop("text_config", {})}
    return Qwen3OmniLlmAccelerated(
        Qwen3OmniLlmConfig(ops_implementation=make_eager_ops_config(), text_config=text_config, **config_overrides)
    ).eval()


def _text_sample(module, lengths):
    hidden = module.config.text_config.hidden_size
    torch.manual_seed(3)
    return [ConversationItem(type="text", value=torch.randn(n, hidden), role="user") for n in lengths]


def _run_forward_node(module, conversation_list):
    inputs = module.pre_forward(method="forward", conversation_list=conversation_list)
    outputs = module.forward(**inputs)
    return module.post_forward(method="forward", **outputs), inputs


@pytest.mark.parametrize(
    ("grid", "second_per_grid"),
    [
        ((1, 4, 6), 1.0),
        ((3, 4, 4), 1.0),
        ((2, 4, 4), 0.5),
        ((3, 4, 4), 2 / 25),
    ],
)
def test_vision_positions_match_upstream_get_rope_index(grid, second_per_grid):
    grid_t, grid_h, grid_w = grid
    media_len = grid_t * grid_h * grid_w // (MERGE**2)
    is_video = second_per_grid != 1.0
    pad_token = VIDEO_PAD if is_video else IMAGE_PAD

    ns = SimpleNamespace(
        spatial_merge_size=MERGE,
        config=SimpleNamespace(
            image_token_id=IMAGE_PAD,
            video_token_id=VIDEO_PAD,
            audio_token_id=UP_AUDIO_PAD,
            vision_start_token_id=VISION_START,
            audio_start_token_id=UP_AUDIO_START,
            position_id_per_seconds=POS_PER_SEC,
        ),
    )
    ns.get_llm_pos_ids_for_vision = _Upstream.get_llm_pos_ids_for_vision.__get__(ns)
    ns.get_rope_index = _Upstream.get_rope_index.__get__(ns)

    input_ids = torch.tensor([[1, 1, 1, VISION_START, *([pad_token] * media_len), VISION_END, 2, 2]], dtype=torch.long)
    grid_thw = torch.tensor([grid], dtype=torch.long)
    upstream_pos, _ = ns.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=None if is_video else grid_thw,
        video_grid_thw=grid_thw if is_video else None,
        audio_seqlens=torch.tensor([0], dtype=torch.long) if is_video else None,
        second_per_grids=torch.tensor([second_per_grid]) if is_video else None,
    )

    packed = _pack(
        [
            _text(3),
            _text(1),
            _media("video" if is_video else "image", media_len, grid, second_per_grid=second_per_grid),
            _text(3),
        ]
    )
    assert packed["position_ids"].dtype == torch.float
    assert torch.equal(packed["position_ids"], upstream_pos)


def test_pack_conversations_tmrope_wall_clock_empty_segments_and_dummies():
    # Text + audio degenerate to plain 1D positions without visual masks.
    audio_packed = _pack(
        [_text(4), ConversationItem(type="audio", value=torch.zeros(7, HIDDEN), role="user"), _text(2)]
    )
    assert torch.equal(
        audio_packed["position_ids"], torch.arange(13, dtype=torch.float).view(1, 1, -1).expand(3, 1, -1)
    )
    assert audio_packed["visual_pos_masks"] is None and audio_packed["deepstack_visual_embeds"] is None

    # Audio and video share one wall-clock axis; cursor advances by max + 1 after vision.
    audio_len, second_per_grid, grid_t = int(2.0 * POS_PER_SEC), 2 / 25, 5
    video_len = grid_t * 4 * 4 // (MERGE**2)
    av = _pack(
        [
            ConversationItem(type="audio", value=torch.zeros(audio_len, HIDDEN), role="user"),
            _media("video", video_len, (grid_t, 4, 4), second_per_grid=second_per_grid),
        ]
    )
    seconds = av["position_ids"][0, 0] / POS_PER_SEC
    assert torch.allclose(seconds[:audio_len], torch.arange(audio_len, dtype=torch.float) / POS_PER_SEC)
    video_axis = torch.unique(seconds[audio_len:], sorted=True)
    assert torch.allclose(
        video_axis, seconds[audio_len - 1] + 1.0 / POS_PER_SEC + torch.arange(grid_t) * second_per_grid
    )

    vpos = qwen3omni_vision_position_ids(5, torch.tensor((1, 4, 6)), MERGE, position_id_per_seconds=POS_PER_SEC)
    img_packed = _pack([_text(5), _media("image", 6, (1, 4, 6)), _text(3)])
    assert img_packed["visual_pos_masks"].tolist() == [[False] * 5 + [True] * 6 + [False] * 3]
    resume = float(vpos.max()) + 1.0
    assert img_packed["position_ids"][:, 0, 11:].tolist() == [[resume, resume + 1, resume + 2]] * 3

    # Multi-sample varlen packing restarts positions per sample.
    multi = pack_qwen3omni_conversations_for_forward([[_text(3)], [_text(5)]], torch.device("cpu"), MERGE, POS_PER_SEC)
    assert multi["cu_seq_lens"].tolist() == [0, 3, 8] and multi["max_length"] == 5
    assert multi["position_ids"][0, 0].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]

    # Video without second_per_grid raises; video with audio metadata packs identically to silent twin.
    with pytest.raises(ValueError, match="second_per_grid"):
        _pack([_text(2), _media("video", 4, (1, 4, 4))])
    silent = _media("video", 4, (1, 4, 4), second_per_grid=0.5)
    sounding = _media(
        "video",
        4,
        (1, 4, 4),
        second_per_grid=0.5,
        **{AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=16_000, num_samples=8)},
    )
    assert torch.equal(_pack([_text(2), sounding])["position_ids"], _pack([_text(2), silent])["position_ids"])

    # Empty segments keep their slot in any position and round-trip through scatter_llm_hidden_states.
    for sample in ([_text(0), _text(3)], [_text(2), _text(0), _text(3)], [_text(2), _text(0)]):
        packed = _pack(sample)
        assert packed["inputs_embeds_shape"].shape[0] == len(sample)
        scatter_llm_hidden_states([sample], unflatten(packed["inputs_embeds"][0], packed["inputs_embeds_shape"]))
        assert [item.value.shape[1] for item in sample] == [HIDDEN] * len(sample)

    # Dummy visual items stay out of inputs_embeds but donate DeepStack anchors.
    dummy = _media("image", 4, (1, 4, 4))
    dummy.is_dummy = True
    with_dummy = _pack([_text(3), dummy])
    assert with_dummy["inputs_embeds"].shape == (1, 3, HIDDEN) and with_dummy["visual_pos_masks"] is None
    assert len(with_dummy["deepstack_visual_embeds"]) == 2


def test_thinker_state_collection_inference_and_training():
    # Inference: step-by-step collection matches one-shot hidden_states[accept_hidden_layer], off by default.
    assert (
        Qwen3OmniLlmConfig(
            ops_implementation=make_eager_ops_config(), text_config=TINY_TEXT_CONFIG
        ).collect_thinker_states
        is False
    )
    accept = 3
    infer_mod = Qwen3OmniLlm(
        Qwen3OmniLlmConfig(
            ops_implementation=make_eager_ops_config(),
            text_config=TINY_TEXT_CONFIG,
            accept_hidden_layer=accept,
            collect_thinker_states=True,
        )
    ).eval()
    torch.manual_seed(0)
    hidden = infer_mod.config.text_config.hidden_size
    prompt = torch.randn(5, hidden)
    steps = [torch.randn(1, hidden) for _ in range(3)]

    infer_mod.reset_global_inference_state()
    conv = [ConversationItem(type="text", value=prompt, role="user")]
    with torch.no_grad():
        infer_mod.generate(conversation_list=conv)
        for emb in steps:
            conv[-1].value = emb
            infer_mod.generate(conversation_list=conv)
        states = infer_mod.flush_thinker_states()
        full = torch.cat([prompt, *steps], dim=0).unsqueeze(0)
        ref = infer_mod.language_model(inputs_embeds=full, use_cache=False, output_hidden_states=True)
    assert torch.equal(states["thinker_embed"], full)
    assert torch.allclose(states["thinker_hidden"], ref.hidden_states[accept], atol=1e-5)
    assert not torch.allclose(ref.hidden_states[accept], ref.last_hidden_state)
    assert infer_mod._accept_hidden_cache == []

    # Training: publishes detached THINKER_EMBED and THINKER_HIDDEN[accept], skips dummies, off by default.
    train_mod = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    sample = _text_sample(train_mod, [4, 3])
    originals = [i.value.clone() for i in sample]
    dummy = ConversationItem(
        type="image",
        value=torch.zeros(4, hidden),
        role="user",
        is_dummy=True,
        meta={"grid_thw": torch.tensor((1, 4, 4)), "deepstack": [torch.zeros(4, hidden) for _ in range(2)]},
    )
    _, inputs = _run_forward_node(train_mod, [[*sample, dummy]])
    with torch.no_grad():
        all_h = train_mod.forward(**inputs)["all_hidden_states"]
    assert torch.equal(sample[0].meta[THINKER_EMBED], originals[0])
    assert torch.equal(sample[0].meta[THINKER_HIDDEN], all_h[2][0, :4])
    assert not sample[0].meta[THINKER_EMBED].requires_grad and not sample[0].meta[THINKER_HIDDEN].requires_grad
    assert THINKER_EMBED not in dummy.meta

    default_mod = _accelerated_llm()
    s_default = _text_sample(default_mod, [4])
    with torch.no_grad():
        _, default_inputs = _run_forward_node(default_mod, [s_default])
    assert default_inputs["output_hidden_states"] is False and THINKER_EMBED not in s_default[0].meta

    bad_layer_mod = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=99)
    with pytest.raises(ValueError, match="accept_hidden_layer=99"):
        _run_forward_node(bad_layer_mod, [_text_sample(bad_layer_mod, [4])])


def test_llm_causal_mask_router_loss_converter_and_sp_guards():
    # Expert fusion converter registered on Qwen3OmniLlm.
    assert hasattr(Qwen3OmniLlm, "_create_checkpoint_tensor_converter")
    assert hasattr(Qwen3OmniLlm, "_convert_fqn_to_index_mapping")

    # Single-sample batch gets an explicit causal mask so Transformers does not misread TMRoPE jumps.
    module = _accelerated_llm(text_config={"output_router_logits": True, "router_aux_loss_coef": 0.01})
    sample = _text_sample(module, [3, 4])
    inputs = module.pre_forward(method="forward", conversation_list=[sample])
    assert not _should_use_packed_mask(inputs["cu_seq_lens_q"], "eager", 1)
    assert find_packed_sequence_indices(torch.tensor([[0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0]])) is not None

    seen = {}
    real_fwd = module.language_model.forward

    def spy(*args, **kwargs):
        seen["attention_mask"] = kwargs.get("attention_mask")
        return real_fwd(*args, **kwargs)

    with mock.patch.object(module.language_model, "forward", side_effect=spy), torch.no_grad():
        module.forward(**inputs)
    assert seen["attention_mask"].shape == (1, inputs["inputs_embeds"].size(1)) and seen["attention_mask"].all()

    # Router balancing loss routes through bound VeomniOp; sp_size==1 is identity; SP forward_post keeps _sp_router_own_len.
    assert isinstance(module.veomni_load_balancing_loss, VeomniOp)
    logits = (torch.randn(4, module.config.text_config.num_experts),)
    with torch.no_grad():
        assert torch.isfinite(module.router_balancing_loss(logits))
    assert module._full_sequence_router_logits(logits) is logits

    hidden = module.config.text_config.hidden_size
    module.pre_forward(method="forward", conversation_list=[_text_sample(module, [4])])
    module._pack_inputs_embeds_shape = torch.tensor([[4]])
    with (
        mock.patch(
            "veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.gather_outputs",
            side_effect=lambda t, **_: t,
        ),
        mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state") as ps,
    ):
        ps.return_value.sp_size, ps.return_value.sp_group, ps.return_value.skip_sp_gather_between_modules = (
            2,
            object(),
            False,
        )
        module._sp_own_len = module._sp_router_own_len = 4
        res = module.forward_post(
            hidden_states=torch.randn(1, 4, hidden),
            router_logits=(torch.randn(4, module.config.text_config.num_experts),),
            router_attention_mask=None,
            all_hidden_states=None,
        )
    assert "_loss" in res and module._sp_router_own_len is None

    # generate refuses Ulysses SP.
    module.reset_global_inference_state()
    with mock.patch(
        "veomni.models.seed_omni.modules.qwen3omni.llm.modeling.get_parallel_state",
        return_value=mock.MagicMock(sp_size=2),
    ):
        with pytest.raises(ValueError, match="sp_size=1"):
            module.generate(
                conversation_list=[ConversationItem(type="text", value=torch.randn(4, HIDDEN), role="user")]
            )


def test_llm_sp_sharded_handoff_regressions():
    module = _accelerated_llm()
    hidden = module.config.text_config.hidden_size

    # 1. Empty dummy SP window stays finite (sum() * 0, not mean() * 0 -> NaN).
    embeds = torch.ones(1, 4, 8)
    empty_dummy = ConversationItem(
        type="image", value=torch.zeros(0, 8), role="user", is_dummy=True, meta={_OMNI_TOKEN_LEN: 0}
    )
    folded = _fold_fsdp_dummy_anchors(embeds, [[empty_dummy]])
    assert torch.isfinite(folded).all()
    torch.testing.assert_close(folded, embeds)

    # 2. Mixed-modality batch regathers producers before packing.
    mixed = [
        *_text_sample(module, [4]),
        ConversationItem(
            type="image",
            value=torch.zeros(4, hidden),
            role="user",
            meta={"grid_thw": torch.tensor((1, 4, 4)), "deepstack": []},
        ),
    ]
    gathered = {"called": False}
    with (
        mock.patch(
            "veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.gather_sp_sharded_groups",
            side_effect=lambda *_, **__: gathered.__setitem__("called", True),
        ),
        mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state") as ps,
        mock.patch.object(module, "_normalize_visual_inputs", side_effect=lambda e, m, d, **_: (e, m, d)),
    ):
        ps.return_value.sp_size, ps.return_value.sp_group = 2, object()
        ps.return_value.skip_sp_gather_between_modules, ps.return_value.fsdp_enabled = True, False
        try:
            module.forward_pre(method="forward", conversation_list=[mixed])
        except Exception:
            pass
    assert gathered["called"]

    # 3. forward_post emits a rank-local window whenever skip_sp_gather_between_modules is True.
    window_lengths = [4, 3]
    full = sum(window_lengths)
    window = sp_local_len(full, 2)
    sample = [
        ConversationItem(
            type="text",
            value=torch.randn(sp_valid_in_window(full, i, 2), hidden),
            role="user",
            meta={_OMNI_TOKEN_LEN: n, "labels": torch.zeros(n, dtype=torch.long)},
        )
        for i, n in enumerate(window_lengths)
    ]
    module._pack_inputs_embeds_shape = torch.tensor([[n] for n in window_lengths])
    for rank in range(2):
        with mock.patch(
            "veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state"
        ) as ps:
            ps.return_value.sp_size, ps.return_value.sp_rank, ps.return_value.sp_group = 2, rank, object()
            ps.return_value.skip_sp_gather_between_modules = True
            module._conversation_carrier = [sample]
            res = module.forward_post(
                hidden_states=torch.randn(1, window, hidden), router_logits=None, router_attention_mask=None
            )
        assert sum(item.value.size(0) for item in res["conversation_list"][0]) == sp_valid_in_window(full, rank, 2)

    # 4. _normalize_visual_inputs sizes mask from full_len, and forward_pre excludes assistant speech from full_len.
    with mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state") as ps:
        ps.return_value.sp_size, ps.return_value.sp_rank, ps.return_value.sp_group, ps.return_value.fsdp_enabled = (
            2,
            0,
            object(),
            False,
        )
        _, mask, deepstack = module._normalize_visual_inputs(torch.randn(1, 4, hidden), None, [], full_len=8)
    assert mask.size(1) == 8 and deepstack is None

    seen = {}
    with (
        mock.patch.object(
            module,
            "_normalize_visual_inputs",
            side_effect=lambda e, m, d, *, full_len: (
                seen.__setitem__("full_len", full_len) or e,
                torch.zeros(1, full_len, dtype=torch.bool),
                None,
            ),
        ),
        mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state") as ps,
    ):
        ps.return_value.sp_size, ps.return_value.fsdp_enabled, ps.return_value.skip_sp_gather_between_modules = (
            1,
            False,
            False,
        )
        module.forward_pre(
            method="forward",
            conversation_list=[
                [
                    ConversationItem(
                        type="text", value=torch.randn(4, hidden), role="user", meta={_OMNI_TOKEN_LEN: 4}
                    ),
                    ConversationItem(
                        type="text", value=torch.randn(3, hidden), role="assistant", meta={_OMNI_TOKEN_LEN: 3}
                    ),
                    ConversationItem(
                        type="audio", value=torch.zeros(24000, 8), role="assistant", meta={_OMNI_TOKEN_LEN: 24000}
                    ),
                ]
            ],
        )
    assert seen["full_len"] == 7

    # 5. _qwen3omni_modality_groups includes user audio so mixed text+audio batches regather audio too.
    text_item = ConversationItem(type="text", value=torch.zeros(2, hidden), role="user")
    user_audio = ConversationItem(type="audio", value=torch.zeros(3, hidden), role="user")
    assistant_audio = ConversationItem(type="audio", value=torch.zeros(5, hidden), role="assistant")
    groups = _qwen3omni_modality_groups([[text_item, user_audio, assistant_audio]])
    assert groups == [[text_item], [user_audio]]

    # 6. 1-token SP shard traces do not double-squeeze into 1D in forward_post.
    traced_mod = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=1)
    one_tok_sample = [
        ConversationItem(type="text", value=torch.randn(1, hidden), role="user", meta={_OMNI_TOKEN_LEN: 2})
    ]
    traced_mod._conversation_carrier = [one_tok_sample]
    traced_mod._pack_inputs_embeds_shape = torch.tensor([[2]])
    with mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state") as ps:
        ps.return_value.sp_size, ps.return_value.sp_rank, ps.return_value.sp_group = 2, 0, object()
        ps.return_value.skip_sp_gather_between_modules = True
        traced_mod.forward_post(
            hidden_states=torch.randn(1, 1, hidden),
            all_hidden_states=(torch.randn(1, 1, hidden), torch.randn(1, 1, hidden)),
            router_logits=None,
            router_attention_mask=None,
        )
    assert one_tok_sample[0].meta[THINKER_EMBED].shape == (1, hidden)
    assert one_tok_sample[0].meta[THINKER_HIDDEN].shape == (1, hidden)
