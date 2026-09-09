from veomni.data.data_transform import process_plaintext_example


class _PlaintextTokenizer:
    eos_token_id = 13

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [10, 11, 12]


def test_plaintext_does_not_build_mtp_labels():
    tokenizer = _PlaintextTokenizer()

    enabled = process_plaintext_example(
        {"text": "ignored"},
        tokenizer=tokenizer,
        max_seq_len=4,
        text_keys="text",
        mtp_num_hidden_layers=2,
    )[0]
    assert "mtp_labels" not in enabled
