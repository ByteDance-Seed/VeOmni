from veomni.data.multimodal.multimodal_chat_template import JanusChatTemplate


class _JanusTokenizer:
    """Minimal tokenizer whose `<image_placeholder>` is added on demand."""

    unk_token_id = 3

    def __init__(self):
        self._vocab = {"<begin_of_image>": 10, "<end_of_image>": 11}
        self._next_id = 20

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)

    def add_special_tokens(self, special_tokens_dict):
        added = 0
        for token in special_tokens_dict["additional_special_tokens"]:
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
                added += 1
        return added

    def encode(self, text, add_special_tokens=False):
        return [0]


def test_janus_image_token_id_resolved_after_registering_special_token():
    tokenizer = _JanusTokenizer()
    template = JanusChatTemplate(tokenizer)

    # The image placeholder id must reflect the token id assigned by
    # add_special_tokens, not the unk id captured before registration.
    assert template.image_token_id != tokenizer.unk_token_id
    assert template.image_token_id == tokenizer.convert_tokens_to_ids("<image_placeholder>")
