import types

import torch

from veomni.utils.constants import IGNORE_INDEX


def _fake_ps(sp_enabled: bool, sp_size: int = 1, sp_rank: int = 0):
    return types.SimpleNamespace(sp_enabled=sp_enabled, sp_size=sp_size, sp_rank=sp_rank)


def _make_flat_dpo_sample(chosen_ids, rejected_ids):
    """Build a flat DPO sample (chosen+rejected concatenated, position_ids reset)."""
    c = torch.tensor(chosen_ids, dtype=torch.long)
    r = torch.tensor(rejected_ids, dtype=torch.long)
    return {
        "input_ids": torch.cat([c, r]),
        "attention_mask": torch.ones(len(chosen_ids) + len(rejected_ids), dtype=torch.long),
        "labels": torch.cat([c, r]),
        "position_ids": torch.cat([torch.arange(len(chosen_ids)), torch.arange(len(rejected_ids))]),
    }


def test_dpo_flat_sample_structure():
    """Flat DPO sample has concatenated chosen+rejected with position_ids reset."""
    sample = _make_flat_dpo_sample([10, 20, 30], [40, 50])
    assert sample["input_ids"].shape == (5,)
    assert torch.equal(sample["input_ids"], torch.tensor([10, 20, 30, 40, 50]))
    assert torch.equal(sample["position_ids"], torch.tensor([0, 1, 2, 0, 1]))


def test_dpo_flat_with_main_collator_sp_disabled(monkeypatch):
    """MainCollator packs flat DPO samples; position_ids resets mark sequence boundaries."""
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))

    s1 = _make_flat_dpo_sample([1, 2, 3], [4, 5, 6])
    s2 = _make_flat_dpo_sample([7, 8], [9, 10])

    collator = m.MainCollator()
    batch = collator([s1, s2])

    # s1 chosen(3) + s1 rejected(3) + s2 chosen(2) + s2 rejected(2) = 10 tokens
    packed_ids = batch["input_ids"].view(-1).tolist()
    assert packed_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # position_ids: [0,1,2, 0,1,2, 0,1, 0,1]
    packed_pos = batch["position_ids"].view(-1).tolist()
    assert packed_pos == [0, 1, 2, 0, 1, 2, 0, 1, 0, 1]

    # cu_seq_lens should have 4 sequences (2 per DPO sample)
    cu = batch["cu_seq_lens_q"].tolist()
    assert cu == [0, 3, 6, 8, 10]


def test_dpo_flat_with_main_collator_sp_enabled(monkeypatch):
    """MainCollator packs and SP-slices flat DPO samples."""
    import veomni.data.data_collator as m

    sp_size = 2
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=sp_size, sp_rank=0))

    s1 = _make_flat_dpo_sample([1, 2, 3, 4], [5, 6, 7, 8])

    collator = m.MainCollator()
    batch = collator([s1])

    total_tokens = 8
    chunk_len = total_tokens // sp_size
    assert batch["input_ids"].view(-1).shape[0] == chunk_len


def test_process_dpo_example_conversation_format(monkeypatch):
    """process_dpo_example produces flat concatenated sample from conversation-format data."""
    from veomni.data.data_transform import process_dpo_example

    chosen_ids = [10, 20, 30]
    rejected_ids = [40, 50]

    class FakeChatTemplate:
        def encode_messages(self, messages, max_seq_len=None):
            return {
                "input_ids": messages,
                "attention_mask": [1] * len(messages),
                "labels": [IGNORE_INDEX] + messages[1:],
            }

    result = process_dpo_example(
        {"chosen": chosen_ids, "rejected": rejected_ids},
        chat_template=FakeChatTemplate(),
        max_seq_len=2048,
    )

    assert len(result) == 1
    sample = result[0]

    assert sample["input_ids"].shape == (5,)
    assert torch.equal(sample["input_ids"], torch.tensor([10, 20, 30, 40, 50]))
    assert torch.equal(sample["position_ids"], torch.tensor([0, 1, 2, 0, 1]))
    assert sample["attention_mask"].shape == (5,)
    assert sample["attention_mask"].sum().item() == 5


def test_process_dpo_example_plaintext_format():
    """process_dpo_example produces flat concatenated sample from plaintext-format data."""
    from veomni.data.data_transform import process_dpo_example

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=True):
            return list(range(len(text)))

    result = process_dpo_example(
        {"prompt": "ab", "chosen": "cd", "rejected": "efg"},
        tokenizer=FakeTokenizer(),
        max_seq_len=2048,
    )

    assert len(result) == 1
    sample = result[0]

    chosen_len = 4  # "ab" + "cd"
    rejected_len = 5  # "ab" + "efg"
    assert sample["input_ids"].shape == (chosen_len + rejected_len,)
    assert torch.equal(
        sample["position_ids"],
        torch.cat([torch.arange(chosen_len), torch.arange(rejected_len)]),
    )
    assert sample["labels"][:2].tolist() == [IGNORE_INDEX, IGNORE_INDEX]
