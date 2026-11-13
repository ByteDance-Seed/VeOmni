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


def test_a_sampled_boi_seals_the_row_and_the_backbone_consumes_it(monkeypatch):
    """``<boi>`` is text, and ``image_vq``'s first node reads only ``value[-1:]``.

    ``JanusLlama.generate`` used to assert ``tail.type == "output"``, which the
    seal violated; the type was never part of the contract.
    """
    encoder = JanusTextEncoder(JanusTextEncoderConfig(vocab_size=16, hidden_size=8, tie_word_embeddings=True))
    encoder._tokenizer = _FakeTokenizer()
    encoder._chat_template = JanusChatTemplate(_FakeTokenizer())
    encoder._prompt_encoded = True
    encoder._text_token_cache = [10, 11]
    monkeypatch.setattr(encoder, "_sample_token", lambda *a, **k: encoder._chat_template.boi_token_id)

    conversation = [
        ConversationItem(type="text", value=torch.zeros(3, dtype=torch.long), role="user"),
        ConversationItem(type="output", value=torch.zeros(1, 4, 8), role="assistant", meta={}),
    ]
    encoder.generate(conversation, generation_kwargs={})

    # ``<boi>`` is sealed as text, exactly as the training template emits it.
    assert conversation[-1].type == "text"
    assert conversation[-1].value is not None

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
    llama._past_key_values = object()
    llama.reset_local_inference_state = lambda: None
    forward_calls = []
    monkeypatch.setattr(
        llama,
        "forward",
        lambda **kw: forward_calls.append(kw) or {"past_key_values": object(), "hidden_states": torch.zeros(1, 8)},
    )
    monkeypatch.setattr(llama, "_tail_hidden_from_forward", lambda h: h.unsqueeze(1))

    llama.generate(conversation)  # must not raise on ``type == "text"``
    assert forward_calls, "the backbone ran with a text tail"
