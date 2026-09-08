import torch

from veomni.data.data_transform import process_plaintext_example
from veomni.utils.constants import IGNORE_INDEX


class _PlaintextTokenizer:
    eos_token_id = 13

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [10, 11, 12]


def test_plaintext_builds_mtp_labels_only_when_enabled():
    tokenizer = _PlaintextTokenizer()

    enabled = process_plaintext_example(
        {"text": "ignored"},
        tokenizer=tokenizer,
        max_seq_len=4,
        text_keys="text",
        mtp_num_hidden_layers=2,
    )[0]
    disabled = process_plaintext_example({"text": "ignored"}, tokenizer, 4, text_keys="text")[0]

    expected = torch.tensor(
        [
            [12, 13, IGNORE_INDEX, IGNORE_INDEX],
            [13, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
        ]
    )
    assert torch.equal(enabled["mtp_labels"], expected)
    assert "mtp_labels" not in disabled
