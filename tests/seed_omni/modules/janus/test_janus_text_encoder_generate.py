"""Janus text-encoder FSM step: the sampled ``<boi>`` transition.

A sampled ``<boi>`` ends the text phase and opens an image span. Three things
have to hold at that moment, and each used to break:

* The row carrying the ``<boi>`` embedding must stay an ``output`` item, because
  ``image_vq`` starts with ``janus_llama.generate``, which asserts the tail is
  output and reads ``value[-1:]`` as its next input.
* The classifier-free-guidance embeds must be stored on an item that is *in* the
  conversation list. ``maybe_merge_outputs`` folds the current row into the
  previous one and drops it, so writing them on the pre-merge object lost them.
* The text decoded so far is still reported, so the caller sees the words that
  preceded the image.
"""

from __future__ import annotations

import torch

from veomni.models.seed_omni.modules.janus.text_encoder.chat_template import JanusChatTemplate
from veomni.models.seed_omni.modules.janus.text_encoder.configuration import JanusTextEncoderConfig
from veomni.models.seed_omni.modules.janus.text_encoder.modeling import JanusTextEncoder
from veomni.models.seed_omni.utils.conversation import ConversationItem


class _FakeTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    boi_token = "<boi>"
    eoi_token = "<eoi>"
    bos_token_id = 1
    eos_token_id = 2
    boi_token_id = 3
    eoi_token_id = 4
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def _encoder() -> JanusTextEncoder:
    model = JanusTextEncoder(JanusTextEncoderConfig(vocab_size=16, hidden_size=8, tie_word_embeddings=True))
    model._tokenizer = _FakeTokenizer()
    model._chat_template = JanusChatTemplate(_FakeTokenizer())
    model._prompt_encoded = True
    model._text_token_cache = [10, 11]
    return model


def _conversation() -> list[ConversationItem]:
    """A prompt row plus one merged output row, as ``generate`` sees them."""
    return [
        ConversationItem(type="text", value=torch.zeros(3, dtype=torch.long), role="user"),
        ConversationItem(type="output", value=torch.zeros(1, 4, 8), role="assistant", meta={}),
    ]


def test_a_sampled_boi_keeps_an_output_tail_for_the_image_span(monkeypatch):
    model = _encoder()
    monkeypatch.setattr(model, "_sample_token", lambda *a, **k: model._chat_template.boi_token_id)

    conversation = _conversation()
    outputs = model.generate(conversation, generation_kwargs={"guidance_scale": 5.0})

    # ``image_vq`` opens with janus_llama.generate, which asserts on this.
    assert conversation[-1].type == "output"
    assert outputs.get("signal") == "start_image_gen" or "start_image_gen" in str(outputs)


def test_a_sampled_boi_arms_cfg_on_the_live_tail(monkeypatch):
    model = _encoder()
    monkeypatch.setattr(model, "_sample_token", lambda *a, **k: model._chat_template.boi_token_id)
    monkeypatch.setattr(model, "_build_cfg_uncond_embeds", lambda conversation_list: torch.zeros(1, 2, 8))

    conversation = _conversation()
    model.generate(conversation, generation_kwargs={"guidance_scale": 5.0})

    # Stored on an item still in the list — not on the row the merge dropped.
    assert "cfg_uncond_inputs_embeds" in conversation[-1].meta
    assert conversation[-1].meta["cfg_uncond_inputs_embeds"] is not None


def test_a_sampled_boi_reports_the_text_so_far(monkeypatch):
    model = _encoder()
    monkeypatch.setattr(model, "_sample_token", lambda *a, **k: model._chat_template.boi_token_id)

    conversation = _conversation()
    outputs = model.generate(conversation, generation_kwargs={})

    assert outputs["generated"]["type"] == "text"
    assert outputs["generated"]["meta"]["token_ids"] == [10, 11]
    # The text before the image is sealed, the ``<boi>`` row is not: the list ends
    # with an output row for the backbone to consume.
    assert conversation[-1].type == "output"


def test_generate_accepts_no_generation_kwargs(monkeypatch):
    """``emit_image_start`` defaults ``generation_kwargs`` to ``None``."""
    model = _encoder()
    monkeypatch.setattr(model, "_sample_token", lambda *a, **k: model._chat_template.eos_token_id)

    conversation = _conversation()
    outputs = model.generate(conversation, generation_kwargs=None)

    assert "text_done" in str(outputs)


def test_cfg_uncond_prefix_covers_the_text_already_generated():
    """Both CFG branches must enter the image span at the same position.

    The conditional cache holds every token generated so far; the unconditional one
    starts from the prompt. If the uncond prefix stopped at the prompt, the two
    branches would run the image span at different positions.
    """
    model = _encoder()
    prompt_len = 4
    generated = 3
    conversation = [
        ConversationItem(type="text", value=torch.zeros(prompt_len, dtype=torch.long), role="user"),
        ConversationItem(type="output", value=torch.zeros(1, generated, 8), role="assistant", meta={}),
    ]

    embeds = model._build_cfg_uncond_embeds(conversation)

    assert embeds.shape[-2] == prompt_len + generated
