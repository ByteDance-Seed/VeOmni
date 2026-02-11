import pytest
import torch
from transformers import AutoTokenizer

from veomni.data.constants import IGNORE_INDEX
from veomni.data.data_transform import process_classification_example


class DummyTokenizer:
    """
    Minimal tokenizer stub:
    - for n whitespace tokens: ids = [101] + [1..n] + [102] when add_special_tokens=True
    """

    def encode(self, text, add_special_tokens=True):
        n = 0 if text is None else len(str(text).split())
        ids = list(range(1, n + 1))
        if add_special_tokens:
            return [101] + ids + [102]
        return ids


def _assert_tensor_1d_long(t: torch.Tensor, expected: list[int]):
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.long
    assert t.ndim == 1
    assert t.shape == (len(expected),)
    assert t.tolist() == expected


def _assert_sample_exact(
    sample: dict,
    expected_input_ids: list[int],
    expected_label_val: int,
):
    # keys
    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels", "position_ids"}

    L = len(expected_input_ids)

    # input_ids exact
    _assert_tensor_1d_long(sample["input_ids"], expected_input_ids)

    # attention_mask exact: all ones
    _assert_tensor_1d_long(sample["attention_mask"], [1] * L)

    # position_ids exact: [0..L-1]
    _assert_tensor_1d_long(sample["position_ids"], list(range(L)))

    # labels exact: all IGNORE_INDEX except last token == expected_label_val
    labels = sample["labels"]
    assert labels.dtype == torch.long
    assert labels.shape == (L,)
    assert labels[:-1].tolist() == [IGNORE_INDEX] * (L - 1)
    assert int(labels[-1].item()) == expected_label_val


def test_basic_outputs_exact_values():
    tok = DummyTokenizer()
    ex = {"text": "hello world", "label": 3}  # 2 words -> [101,1,2,102]
    out = process_classification_example(example=ex, tokenizer=tok, max_seq_len=128)

    assert isinstance(out, list) and len(out) == 1
    sample = out[0]

    _assert_sample_exact(
        sample,
        expected_input_ids=[101, 1, 2, 102],
        expected_label_val=3,
    )


def test_truncates_to_max_seq_len_and_checks_exact_values():
    tok = DummyTokenizer()
    # 10 words -> tokens = [101] + [1..10] + [102] => length 12
    ex = {"text": "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10", "label": 1}

    out = process_classification_example(example=ex, tokenizer=tok, max_seq_len=6)
    sample = out[0]

    # IMPORTANT:
    # truncation happens after tokenization, so we keep the first 6 tokens of:
    # [101,1,2,3,4,5,6,7,8,9,10,102] -> [101,1,2,3,4,5]
    # last token is now 5 (not 102), and that last position gets the label.
    _assert_sample_exact(
        sample,
        expected_input_ids=[101, 1, 2, 3, 4, 5],
        expected_label_val=1,
    )


def test_text_keys_list_picks_first_existing_key():
    tok = DummyTokenizer()
    ex = {"prompt": "a b c", "label": 7}  # 3 words -> [101,1,2,3,102]
    out = process_classification_example(
        example=ex,
        tokenizer=tok,
        max_seq_len=128,
        text_keys=["text", "prompt", "content"],  # "prompt" exists
        label_key="label",
    )
    sample = out[0]

    _assert_sample_exact(
        sample,
        expected_input_ids=[101, 1, 2, 3, 102],
        expected_label_val=7,
    )


def test_text_keys_list_missing_all_raises_valueerror():
    tok = DummyTokenizer()
    ex = {"content": "hi", "label": 0}
    with pytest.raises(ValueError, match=r"None of the keys .* are found in the example"):
        process_classification_example(
            example=ex,
            tokenizer=tok,
            max_seq_len=16,
            text_keys=["text", "prompt"],  # none exist
        )


def test_missing_text_key_default_string_raises_keyerror():
    tok = DummyTokenizer()
    ex = {"label": 0}
    # default text_keys="text" uses example["text"] -> KeyError
    with pytest.raises(KeyError):
        process_classification_example(example=ex, tokenizer=tok, max_seq_len=16)


def test_missing_label_key_raises_valueerror():
    tok = DummyTokenizer()
    ex = {"text": "hi"}
    with pytest.raises(ValueError, match=r"Missing label key 'label'"):
        process_classification_example(example=ex, tokenizer=tok, max_seq_len=16)


def test_label_not_intlike_raises_valueerror():
    tok = DummyTokenizer()
    ex = {"text": "hi", "label": "not_an_int"}
    with pytest.raises(ValueError, match=r"not an int-like value"):
        process_classification_example(example=ex, tokenizer=tok, max_seq_len=16)


def test_label_zero_is_valid_and_on_last_token():
    tok = DummyTokenizer()
    ex = {"text": "hi", "label": 0}  # 1 word -> [101,1,102]
    out = process_classification_example(example=ex, tokenizer=tok, max_seq_len=128)
    sample = out[0]

    _assert_sample_exact(
        sample,
        expected_input_ids=[101, 1, 102],
        expected_label_val=0,
    )


def test_tokens_length_equals_max_seq_len_no_truncation():
    tok = DummyTokenizer()
    # 3 words -> tokens length = 3 + 2 = 5
    ex = {"text": "a b c", "label": 9}
    out = process_classification_example(example=ex, tokenizer=tok, max_seq_len=5)
    sample = out[0]

    # should NOT truncate because len(tokens)==max_seq_len (only truncates when >)
    _assert_sample_exact(
        sample,
        expected_input_ids=[101, 1, 2, 3, 102],
        expected_label_val=9,
    )


def test_process_classification_example_whitespace_with_real_qwen3_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-0.6B",
        trust_remote_code=False,
    )

    cases = [
        ("", [151643]),
        ("   ", [262, 151643]),
    ]

    for text, expected_input_ids in cases:
        ex = {"text": text, "label": 7}
        out = process_classification_example(
            example=ex,
            tokenizer=tok,
            max_seq_len=128,
        )
        sample = out[0]

        # 1) input_ids
        assert sample["input_ids"].tolist() == expected_input_ids

        L = len(expected_input_ids)

        # 2) attention_mask
        assert sample["attention_mask"].tolist() == [1] * L

        # 3) position_ids
        assert sample["position_ids"].tolist() == list(range(L))

        # 4) labels: IGNORE except last
        assert sample["labels"][:-1].tolist() == [IGNORE_INDEX] * (L - 1)
        assert int(sample["labels"][-1].item()) == 7
