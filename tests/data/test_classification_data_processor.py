import pytest
import torch
from transformers import AutoTokenizer

from veomni.data.data_transform import process_classification_example, process_dpo_example
from veomni.utils.constants import IGNORE_INDEX


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


# ===================== DPO data transform tests =====================


class DummyChatTemplate:
    """Minimal chat template stub that returns fixed tokenized outputs."""

    def __init__(self, chosen_result, rejected_result):
        self._chosen = chosen_result
        self._rejected = rejected_result
        self._call_count = 0

    def encode_messages(self, messages, max_seq_len=None):
        self._call_count += 1
        if self._call_count == 1:
            return self._chosen
        return self._rejected


def test_dpo_conversation_format_basic():
    """Conversation format should call chat_template and produce [2, L] tensors."""
    chosen_tok = {
        "input_ids": [10, 20, 30],
        "attention_mask": [1, 1, 1],
        "labels": [IGNORE_INDEX, 20, 30],
    }
    rejected_tok = {
        "input_ids": [40, 50, 60],
        "attention_mask": [1, 1, 1],
        "labels": [IGNORE_INDEX, 50, 60],
    }
    ct = DummyChatTemplate(chosen_tok, rejected_tok)
    example = {
        "chosen": [{"role": "user", "content": "hi"}],
        "rejected": [{"role": "user", "content": "hi"}],
    }

    out = process_dpo_example(example, chat_template=ct)
    assert len(out) == 1
    sample = out[0]
    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}

    for key in sample:
        assert sample[key].shape[0] == 2

    assert torch.equal(sample["input_ids"][0], torch.tensor([10, 20, 30]))
    assert torch.equal(sample["input_ids"][1], torch.tensor([40, 50, 60]))


def test_dpo_conversation_format_different_lengths():
    """When chosen and rejected have different lengths, shorter is padded."""
    chosen_tok = {
        "input_ids": [1, 2],
        "attention_mask": [1, 1],
        "labels": [IGNORE_INDEX, 2],
    }
    rejected_tok = {
        "input_ids": [3, 4, 5, 6],
        "attention_mask": [1, 1, 1, 1],
        "labels": [IGNORE_INDEX, 4, 5, 6],
    }
    ct = DummyChatTemplate(chosen_tok, rejected_tok)
    example = {"chosen": [{"role": "user", "content": "a"}], "rejected": [{"role": "user", "content": "b"}]}

    sample = process_dpo_example(example, chat_template=ct)[0]
    assert sample["input_ids"].shape == (2, 4)

    assert sample["input_ids"][0].tolist() == [1, 2, 0, 0]
    assert sample["attention_mask"][0].tolist() == [1, 1, 0, 0]
    assert sample["labels"][0].tolist() == [IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX]

    assert sample["input_ids"][1].tolist() == [3, 4, 5, 6]


def test_dpo_plaintext_format_with_prompt():
    """Plaintext format should mask prompt tokens in labels with IGNORE_INDEX."""
    tok = DummyTokenizer()
    example = {"prompt": "hello", "chosen": " world", "rejected": " bad"}

    sample = process_dpo_example(example, tokenizer=tok)[0]
    assert sample["input_ids"].shape[0] == 2

    chosen_labels = sample["labels"][0].tolist()
    prompt_len = len(tok.encode("hello", add_special_tokens=True))
    assert chosen_labels[:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert all(v != IGNORE_INDEX for v in chosen_labels[prompt_len:])


def test_dpo_plaintext_format_no_prompt():
    """Plaintext format without prompt should have no IGNORE_INDEX masking."""
    tok = DummyTokenizer()
    example = {"chosen": "good", "rejected": "bad thing"}

    sample = process_dpo_example(example, tokenizer=tok)[0]
    chosen_labels = sample["labels"][0].tolist()
    non_pad = [v for v in chosen_labels if v != IGNORE_INDEX]
    assert len(non_pad) > 0


def test_dpo_plaintext_truncation():
    """max_seq_len should truncate the tokenized output."""
    tok = DummyTokenizer()
    example = {"chosen": "a b c d e", "rejected": "x y"}

    sample = process_dpo_example(example, tokenizer=tok, max_seq_len=4)[0]
    chosen_len = sample["attention_mask"][0].sum().item()
    assert chosen_len <= 4


def test_dpo_output_structure():
    """Verify output is a list with one dict containing the right keys and shapes."""
    chosen_tok = {"input_ids": [1], "attention_mask": [1], "labels": [1]}
    rejected_tok = {"input_ids": [2], "attention_mask": [1], "labels": [2]}
    ct = DummyChatTemplate(chosen_tok, rejected_tok)
    example = {"chosen": [{"role": "user", "content": "a"}], "rejected": [{"role": "user", "content": "b"}]}

    out = process_dpo_example(example, chat_template=ct)
    assert isinstance(out, list)
    assert len(out) == 1
    sample = out[0]
    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}
    for key in sample:
        assert sample[key].ndim == 2
        assert sample[key].shape[0] == 2
