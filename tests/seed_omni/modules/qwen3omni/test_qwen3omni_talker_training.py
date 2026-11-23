"""Teacher-forced talker training, held against the decode loop it must match.

Upstream ships no talker training code, so `Qwen3OmniTalker.forward` is derived
from `Qwen3OmniMoeTalkerForConditionalGeneration`'s decode loop rather than
ported from a reference. That makes one property the whole safety net: replaying
the codes a `generate` call produced back through the teacher-forced pass must
reproduce the logits that call saw, at every step and for both heads.

If the frame embedding, the text-row offset, the 3D RoPE continuation or the
per-group head indexing were off by anything, those logits would diverge — and
training would silently optimise a model that decodes differently.
"""

from __future__ import annotations

import pytest
import torch

from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import TALKER_CODE_LABELS
from veomni.models.seed_omni.modules.qwen3omni.llm.modeling import THINKER_EMBED, THINKER_HIDDEN
from veomni.models.seed_omni.modules.qwen3omni.talker.accelerated.accelerated import (
    _talker_prompt,
    _target_codes,
    _thinker_traces,
)
from veomni.models.seed_omni.modules.qwen3omni.talker.modeling import (
    Qwen3OmniTalker,
    _speech_prefix_length,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.accelerated.accelerated import _stash_talker_prompts
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import TALKER_PROMPT
from veomni.models.seed_omni.utils.conversation import ConversationItem

# The sibling module derives its tiny talker from the real config.json, so the
# codec special-token ids and the num_code_groups here are the checkpoint's own.
from .test_qwen3omni_talker import _tiny_talker, requires_checkpoint


pytestmark = requires_checkpoint


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
