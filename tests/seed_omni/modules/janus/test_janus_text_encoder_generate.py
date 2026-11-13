"""Janus text-encoder FSM step: the sampled ``<boi>`` transition.

A sampled ``<boi>`` ends the text phase and opens an image span. Three things
have to hold at that moment, and each used to break:

* ``<boi>`` is an ordinary text token, so the row is sealed as ``text`` like any
  other span. The backbone reads only ``value[-1:]``, never the type.
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


def test_a_sampled_boi_seals_the_tail_arms_cfg_and_reports_the_text(monkeypatch):
    model = _encoder()
    monkeypatch.setattr(model, "_sample_token", lambda *a, **k: model._chat_template.boi_token_id)
    monkeypatch.setattr(model, "_build_cfg_uncond_embeds", lambda conversation_list: torch.zeros(1, 2, 8))

    conversation = _conversation()
    outputs = model.generate(conversation, generation_kwargs={"guidance_scale": 5.0})

    # Sealed as text; ``<boi>`` itself belongs to the image span, not the reply.
    assert conversation[-1].type == "text"
    assert outputs["generated"]["meta"]["token_ids"] == [10, 11]
    # CFG armed on an item still in the list — not on the row the merge dropped.
    assert conversation[-1].meta["cfg_uncond_inputs_embeds"] is not None
    assert "start_image_gen" in str(outputs)


def test_generate_accepts_no_generation_kwargs_and_pads_the_cfg_prefix(monkeypatch):
    """``emit_image_start`` defaults ``generation_kwargs`` to ``None``.

    Both CFG branches must also enter the image span at the same position: the
    conditional cache already holds the text generated so far, so the
    unconditional prefix is padded out to that length.
    """
    model = _encoder()
    monkeypatch.setattr(model, "_sample_token", lambda *a, **k: model._chat_template.eos_token_id)
    outputs = model.generate(_conversation(), generation_kwargs=None)
    assert "text_done" in str(outputs)

    prompt_len, generated = 4, 3
    prompt = [
        ConversationItem(type="text", value=torch.zeros(prompt_len, dtype=torch.long), role="user"),
        ConversationItem(type="output", value=torch.zeros(1, generated, 8), role="assistant", meta={}),
    ]
    # The tail row's last position is the token the backbone feeds as the span's
    # first input, so it must not also be in the prefix; counting it would put the
    # unconditional branch one position ahead of the conditional one.
    assert model._build_cfg_uncond_embeds(prompt).shape[-2] == prompt_len + generated - 1
