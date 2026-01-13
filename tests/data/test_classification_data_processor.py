import pytest
import torch

from veomni.data.constants import IGNORE_INDEX
from veomni.data.data_transform import process_classification_example


class DummyTokenizer:
    """
    Minimal tokenizer stub:
    - returns [101] + [1,2,3,...,n] + [102]
    - length depends on whitespace-separated tokens
    """

    def encode(self, text, add_special_tokens=True):
        n = 0 if text is None else len(str(text).split())
        ids = list(range(1, n + 1))
        if add_special_tokens:
            return [101] + ids + [102]
        return ids


def _assert_sample_structure(sample, expected_len: int):
    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels", "position_ids"}

    assert sample["input_ids"].dtype == torch.long
    assert sample["attention_mask"].dtype == torch.long
    assert sample["labels"].dtype == torch.long
    assert sample["position_ids"].dtype == torch.long

    assert sample["input_ids"].shape == (expected_len,)
    assert sample["attention_mask"].shape == (expected_len,)
    assert sample["labels"].shape == (expected_len,)
    assert sample["position_ids"].shape == (expected_len,)

    # attention_mask all ones
    assert torch.all(sample["attention_mask"] == 1)

    # position_ids = [0..L-1]
    assert torch.equal(sample["position_ids"], torch.arange(expected_len, dtype=torch.long))


def test_process_classification_example_basic_last_token_label():
    tok = DummyTokenizer()
    ex = {"text": "hello world", "label": 3}  # 2 words -> [101,1,2,102] len=4
    out = process_classification_example(
        example=ex,
        tokenizer=tok,
        max_seq_len=128,
    )

    assert isinstance(out, list) and len(out) == 1
    sample = out[0]
    _assert_sample_structure(sample, expected_len=4)

    # labels: all IGNORE except last token == label
    assert torch.all(sample["labels"][:-1] == IGNORE_INDEX)
    assert int(sample["labels"][-1].item()) == 3


def test_process_classification_example_label_offset_applied():
    tok = DummyTokenizer()
    ex = {"text": "a b c", "label": 4}
    out = process_classification_example(
        example=ex,
        tokenizer=tok,
        max_seq_len=128,
        label_offset=1,  # 4 -> 3
    )
    sample = out[0]
    assert int(sample["labels"][-1].item()) == 3


def test_process_classification_example_truncates_to_max_seq_len():
    tok = DummyTokenizer()
    # 10 words -> len = 12 with [101]...[102]
    ex = {"text": "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10", "label": 1}
    out = process_classification_example(
        example=ex,
        tokenizer=tok,
        max_seq_len=6,  # force truncate
    )
    sample = out[0]
    _assert_sample_structure(sample, expected_len=6)

    # last token gets label, others ignore
    assert torch.all(sample["labels"][:-1] == IGNORE_INDEX)
    assert int(sample["labels"][-1].item()) == 1


def test_process_classification_example_missing_text_key_raises():
    tok = DummyTokenizer()
    ex = {"label": 0}
    with pytest.raises(KeyError, match="'text'"):
        process_classification_example(ex, tok, max_seq_len=16)


def test_process_classification_example_missing_label_key_raises():
    tok = DummyTokenizer()
    ex = {"text": "hi"}
    with pytest.raises(ValueError, match="Missing label key"):
        process_classification_example(ex, tok, max_seq_len=16)


def test_process_classification_example_label_not_intlike_raises():
    tok = DummyTokenizer()
    ex = {"text": "hi", "label": "not_an_int"}
    with pytest.raises(ValueError, match="not an int-like value"):
        process_classification_example(ex, tok, max_seq_len=16)


def test_process_classification_example_negative_after_offset_raises():
    tok = DummyTokenizer()
    ex = {"text": "hi", "label": 0}
    with pytest.raises(ValueError, match="became negative"):
        process_classification_example(ex, tok, max_seq_len=16, label_offset=1)