"""SeedOmni base ``TextEncoder`` on a ``ShardedEmbedding`` table."""

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.seed_omni.modules.base.text_encoder.configuration import TextEncoderConfig
from veomni.models.seed_omni.modules.base.text_encoder.modeling import TextEncoder


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


_VOCAB, _HIDDEN = 8, 4


@pytest.mark.parametrize("pad_token_id", [None, 3])
def test_base_text_encoder_passes_pad_token_id_to_sharded_embedding(pad_token_id):
    text_encoder = TextEncoder(TextEncoderConfig(vocab_size=_VOCAB, hidden_size=_HIDDEN, pad_token_id=pad_token_id))
    assert text_encoder.embed_tokens.padding_idx == pad_token_id


def test_base_text_encoder_tied_head_projects_through_sharded_embedding():
    text_encoder = TextEncoder(TextEncoderConfig(vocab_size=_VOCAB, hidden_size=_HIDDEN, tie_word_embeddings=True))
    ids = torch.tensor([[0, 2, 5]], dtype=torch.long)
    embeds = text_encoder.encode(input_ids=ids)["inputs_embeds"]
    logits = text_encoder._project(embeds)
    expected = F.linear(embeds, text_encoder.embed_tokens.weight)
    torch.testing.assert_close(logits, expected)
