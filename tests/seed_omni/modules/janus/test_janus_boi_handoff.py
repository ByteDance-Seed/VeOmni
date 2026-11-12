"""The text_ar -> image_vq hand-off after a sampled ``<boi>``.

The base checkpoint rarely samples ``<boi>`` on its own, so the transition is
driven directly: run one text-encoder step returning the begin-image token, then
the backbone step ``image_vq`` starts with. This pins the contract between the two
-- the tail row is sealed as ``text`` (``<boi>`` is an ordinary text token) and its
last embedding is what the backbone consumes.
"""

from __future__ import annotations

import torch

from veomni.models.seed_omni.modules.janus.llama.configuration import JanusLlamaConfig
from veomni.models.seed_omni.modules.janus.llama.modeling import JanusLlama
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


def _text_encoder() -> JanusTextEncoder:
    model = JanusTextEncoder(JanusTextEncoderConfig(vocab_size=16, hidden_size=8, tie_word_embeddings=True))
    model._tokenizer = _FakeTokenizer()
    model._chat_template = JanusChatTemplate(_FakeTokenizer())
    model._prompt_encoded = True
    model._text_token_cache = [10, 11]
    return model


def _conversation() -> list[ConversationItem]:
    return [
        ConversationItem(type="text", value=torch.zeros(3, dtype=torch.long), role="user"),
        ConversationItem(type="output", value=torch.zeros(1, 4, 8), role="assistant", meta={}),
    ]


def test_the_boi_row_hands_one_embedding_to_the_image_span(monkeypatch):
    """``image_vq`` starts with the backbone reading ``value[-1:]`` regardless of type."""
    encoder = _text_encoder()
    monkeypatch.setattr(encoder, "_sample_token", lambda *a, **k: encoder._chat_template.boi_token_id)

    conversation = _conversation()
    encoder.generate(conversation, generation_kwargs={})

    # ``<boi>`` is sealed as text, exactly as the training template emits it.
    assert conversation[-1].type == "text"

    # The backbone's one hard requirement: the tail holds embeddings it can slice.
    assert conversation[-1].value is not None
    assert conversation[-1].value.size(-2) >= 1


def test_the_backbone_accepts_a_text_tail(monkeypatch):
    """A sealed ``<boi>`` row is what ``image_vq``'s first node sees.

    ``JanusLlama.generate`` used to assert ``tail.type == "output"``, which the
    seal violated. It now only requires the row to carry embeddings.
    """
    llama = JanusLlama(
        JanusLlamaConfig(
            text_config={
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_hidden_layers": 0,
                "vocab_size": 16,
            }
        )
    )
    # Simulate the state ``image_vq`` starts from: a text tail mid-image-span.
    llama._past_key_values = object()
    conversation = [
        ConversationItem(type="text", value=torch.zeros(1, 2, 8), role="assistant"),
    ]
    llama.reset_local_inference_state = lambda: None
    forward_calls = []

    def _fake_forward(**kwargs):
        forward_calls.append(kwargs)
        hidden = torch.zeros(1, 8)
        return {"past_key_values": object(), "hidden_states": hidden}

    monkeypatch.setattr(llama, "forward", _fake_forward)
    monkeypatch.setattr(llama, "_tail_hidden_from_forward", lambda h: h.unsqueeze(1))

    llama.generate(conversation)  # must not raise on ``type == "text"``
    assert forward_calls, "the backbone ran with a text tail"
