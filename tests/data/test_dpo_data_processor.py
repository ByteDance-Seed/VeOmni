import types

import pytest
import torch

from veomni.data.data_transform import process_dpo_example
from veomni.utils.constants import IGNORE_INDEX


pytestmark = pytest.mark.unit


def _fake_ps(sp_enabled: bool, sp_size: int = 1, sp_rank: int = 0):
    return types.SimpleNamespace(sp_enabled=sp_enabled, sp_size=sp_size, sp_rank=sp_rank)


class _FakeChatTemplate:
    """Returns messages directly as tokenized output, enabling deterministic tests."""

    def encode_messages(self, messages, max_seq_len=None):
        return {"input_ids": messages, "attention_mask": [1] * len(messages), "labels": [IGNORE_INDEX] + messages[1:]}


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text)))


# ---- process_dpo_example tests ----


def test_dpo_conversation_format():
    """Conversation-format DPO: flat 1-D concat with position_ids reset and correct keys."""
    sample = process_dpo_example({"chosen": [10, 20, 30], "rejected": [40, 50]}, chat_template=_FakeChatTemplate())[0]

    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels", "position_ids"}
    assert sample["input_ids"].tolist() == [10, 20, 30, 40, 50]
    assert sample["position_ids"].tolist() == [0, 1, 2, 0, 1]
    assert sample["attention_mask"].sum().item() == 5
    for v in sample.values():
        assert v.ndim == 1


def test_dpo_plaintext_prompt_masking():
    """Plaintext DPO: prompt tokens masked with IGNORE_INDEX in both chosen and rejected."""
    sample = process_dpo_example({"prompt": "ab", "chosen": "cd", "rejected": "efg"}, tokenizer=_FakeTokenizer())[0]

    chosen_len, rejected_len = 4, 5  # "abcd"=4, "abefg"=5
    assert sample["input_ids"].shape == (chosen_len + rejected_len,)
    assert sample["position_ids"].tolist() == [*range(chosen_len), *range(rejected_len)]
    assert sample["labels"][:2].tolist() == [IGNORE_INDEX, IGNORE_INDEX]
    assert sample["labels"][chosen_len : chosen_len + 2].tolist() == [IGNORE_INDEX, IGNORE_INDEX]


def test_dpo_plaintext_truncation():
    """max_seq_len truncates each sequence independently before concatenation."""
    sample = process_dpo_example({"chosen": "abcdefgh", "rejected": "xyz"}, tokenizer=_FakeTokenizer(), max_seq_len=4)[
        0
    ]

    assert sample["input_ids"].shape == (4 + 3,)
    assert sample["position_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2]


# ---- MainCollator integration tests ----


def _make_flat_dpo_sample(chosen_ids, rejected_ids):
    c = torch.tensor(chosen_ids, dtype=torch.long)
    r = torch.tensor(rejected_ids, dtype=torch.long)
    return {
        "input_ids": torch.cat([c, r]),
        "attention_mask": torch.ones(len(chosen_ids) + len(rejected_ids), dtype=torch.long),
        "labels": torch.cat([c, r]),
        "position_ids": torch.cat([torch.arange(len(chosen_ids)), torch.arange(len(rejected_ids))]),
    }


def test_dpo_main_collator_sp_disabled(monkeypatch):
    """MainCollator packs flat DPO samples; position_ids resets mark sequence boundaries."""
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))

    s1 = _make_flat_dpo_sample([1, 2, 3], [4, 5, 6])
    s2 = _make_flat_dpo_sample([7, 8], [9, 10])
    batch = m.MainCollator()([s1, s2])

    assert batch["input_ids"].view(-1).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert batch["position_ids"].view(-1).tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 0, 1]
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 6, 8, 10]


def test_dpo_main_collator_sp_enabled(monkeypatch):
    """MainCollator packs and SP-slices flat DPO samples."""
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=2, sp_rank=0))

    batch = m.MainCollator()([_make_flat_dpo_sample([1, 2, 3, 4], [5, 6, 7, 8])])
    assert batch["input_ids"].view(-1).shape[0] == 4  # 8 tokens / sp_size=2
