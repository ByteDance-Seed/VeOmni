import types

import pytest
import torch

from veomni.trainer import dllm_trainer
from veomni.trainer.dllm_trainer import DllmTrainer


def _fake_ps(ulysses_size: int = 1, sp_rank: int = 0, cp_size: int = 1):
    sp_size = ulysses_size * cp_size
    return types.SimpleNamespace(
        sp_enabled=sp_size > 1,
        sp_size=sp_size,
        sp_rank=sp_rank if sp_size > 1 else -1,
        ulysses_size=ulysses_size,
        cp_size=cp_size,
    )


def _model_config(**overrides):
    fields = {
        "model_type": "llada",
        "_attn_implementation": "veomni_flash_attention_2",
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _trainer(monkeypatch, ulysses_size: int = 1, sp_rank: int = 0, cp_size: int = 1, **config) -> DllmTrainer:
    import veomni.data.data_collator as collator_module

    state = _fake_ps(ulysses_size, sp_rank, cp_size)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr(dllm_trainer, "get_parallel_state", lambda: state)
    trainer = DllmTrainer.__new__(DllmTrainer)
    trainer.base = types.SimpleNamespace(
        args=types.SimpleNamespace(train=types.SimpleNamespace(pad_to_length=False)),
        model=types.SimpleNamespace(model_config=_model_config(**config)),
    )
    return trainer


def _packed_features():
    # three samples, seven tokens: the SP stage pads to eight, and rank 0's shard ends mid-sample
    features = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, -100], "loss_weight": [0.0, 0.5, 0.0]},
        {"input_ids": [4, 9], "labels": [-100, -100], "loss_weight": [0.0, 0.0]},
        {"input_ids": [5, 6], "labels": [5, 6], "loss_weight": [0.5, 1.0]},
    ]
    answer_masks = [[0, 1, 1], [1, 1], [1, 1]]
    out = []
    for feature, answer_mask in zip(features, answer_masks):
        out.append(
            {
                "input_ids": torch.tensor(feature["input_ids"]),
                "attention_mask": torch.ones(len(feature["input_ids"]), dtype=torch.long),
                "labels": torch.tensor(feature["labels"]),
                "loss_weight": torch.tensor(feature["loss_weight"]),
                # the middle sample's answer drew no mask, and still counts
                "answer_mask": torch.tensor(answer_mask),
            }
        )
    return out


def test_packed_row_counts_segments_not_rows():
    trainer = DllmTrainer.__new__(DllmTrainer)
    micro_batch = {
        # three packed samples; the middle one drew no masked answer token
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]]),
        "loss_weight": torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0]]),
    }
    assert trainer._micro_batch_sample_count(micro_batch) == 2


def test_answer_mask_counts_samples_whose_draw_masked_nothing():
    trainer = DllmTrainer.__new__(DllmTrainer)
    micro_batch = {
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3, 0]]),
        "loss_weight": torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0]]),
        # the middle sample has an answer the draw left unmasked; the last token is pad
        "answer_mask": torch.tensor([[0, 1, 1, 0, 1, 0, 0, 1, 1, 0]]),
    }
    assert trainer._micro_batch_sample_count(micro_batch) == 3


def test_unpacked_rows_match_row_count():
    trainer = DllmTrainer.__new__(DllmTrainer)
    loss_weight = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    micro_batch = {"position_ids": torch.arange(3).expand(3, 3).clone(), "loss_weight": loss_weight}
    assert trainer._micro_batch_sample_count(micro_batch) == trainer._supervised_sample_count(loss_weight) == 2
    assert trainer._micro_batch_sample_count({"loss_weight": loss_weight}) == 2


def test_pad_tail_is_not_a_sample():
    trainer = DllmTrainer.__new__(DllmTrainer)
    # `pad_to_length` pads position_ids with 0, so every pad token opens a zero-weight segment
    micro_batch = {
        "position_ids": torch.tensor([[0, 1, 2, 0, 0, 0]]),
        "loss_weight": torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
    }
    assert trainer._micro_batch_sample_count(micro_batch) == 1


def test_step_count_sums_micro_batches():
    trainer = DllmTrainer.__new__(DllmTrainer)
    packed = {"position_ids": torch.tensor([[0, 1, 0, 1]]), "loss_weight": torch.tensor([[1.0, 0.0, 0.0, 1.0]])}
    trainer.on_micro_batches_ready([packed, packed])
    assert trainer.micro_batches_num_samples == 4


def test_answer_mask_is_dropped_before_the_forward():
    trainer = DllmTrainer.__new__(DllmTrainer)
    packed = {
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "loss_weight": torch.zeros(1, 4),
        "answer_mask": torch.tensor([[0, 1, 0, 1]]),
    }
    trainer.on_micro_batches_ready([packed])
    assert trainer.micro_batches_num_samples == 2
    assert "answer_mask" not in packed


def test_epoch_reaches_the_transform_holder():
    trainer = DllmTrainer.__new__(DllmTrainer)
    trainer.base = types.SimpleNamespace(state=types.SimpleNamespace(epoch=3), on_epoch_begin=lambda: None)
    holder = trainer._corruption_epoch
    trainer.on_epoch_begin()
    assert holder.epoch == 3


def test_diffusion_gemma_keeps_its_example_count():
    # the encoder `position_ids` can coincide in shape with the canvas `loss_weight`
    # and must not be read as its segment boundaries
    trainer = DllmTrainer.__new__(DllmTrainer)
    micro_batch = {
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "loss_weight": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        "num_supervised_examples": torch.tensor(3),
    }
    assert trainer._micro_batch_sample_count(micro_batch) == 3


def test_collator_packs_loss_weight(monkeypatch):
    trainer = _trainer(monkeypatch)
    trainer._build_collate_fn()
    features = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "labels": torch.tensor([-100, 2, -100]),
            "loss_weight": torch.tensor([0.0, 0.5, 0.0]),
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "labels": torch.tensor([4, 5]),
            "loss_weight": torch.tensor([0.5, 1.0]),
        },
    ]
    for feature in features:
        feature["answer_mask"] = (feature["labels"] != -100).long()
    batch = trainer.base.collate_fn(features)
    assert torch.equal(batch["loss_weight"], torch.tensor([[0.0, 0.5, 0.0, 0.5, 1.0]]))
    # a masked token opening a packed sample keeps its label, as it keeps its weight
    assert torch.equal(batch["labels"], torch.tensor([[-100, 2, -100, 4, 5]]))
    assert torch.equal(batch["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
    # counted by the collator, and the marker it counted from never reaches the model
    assert "answer_mask" not in batch
    assert int(batch["num_supervised_samples"]) == 2
    assert trainer._micro_batch_sample_count(batch) == 2
    trainer.on_micro_batches_ready([batch])
    assert trainer.micro_batches_num_samples == 2
    assert "num_supervised_samples" not in batch


def test_sequence_parallel_shards_count_the_whole_row(monkeypatch):
    whole = _trainer(monkeypatch)
    whole._build_collate_fn()
    reference = whole.base.collate_fn(_packed_features())
    assert int(reference["num_supervised_samples"]) == 3

    shards = []
    for rank in range(2):
        trainer = _trainer(monkeypatch, ulysses_size=2, sp_rank=rank)
        trainer._build_collate_fn()
        shard = trainer.base.collate_fn(_packed_features())
        # each peer reports the whole row's count, though rank 0 holds two samples and a half
        assert int(shard["num_supervised_samples"]) == 3
        assert "answer_mask" not in shard
        # the full-row boundaries, computed before the slice, so attention sees whole samples
        assert torch.equal(shard["cu_seq_lens_q"], torch.tensor([0, 3, 5, 7, 8], dtype=torch.int32))
        shards.append(shard)

    for key in ("input_ids", "labels", "loss_weight", "position_ids"):
        joined = torch.cat([shard[key] for shard in shards], dim=-1)
        # labels are not shifted for next-token prediction; the tail is one token of SP padding
        assert torch.equal(joined[..., :-1], reference[key]), key
    assert int(torch.cat([shard["labels"] for shard in shards], dim=-1)[0, -1]) == -100
    assert float(torch.cat([shard["loss_weight"] for shard in shards], dim=-1)[0, -1]) == 0.0


@pytest.mark.parametrize(
    "setup, match",
    [
        ({"ulysses_size": 1, "cp_size": 2}, "Ulysses sequence parallelism only"),
        ({"ulysses_size": 2, "model_type": "qwen3"}, "no sequence-parallel path for `qwen3`"),
        ({"ulysses_size": 2, "_attn_implementation": "sdpa"}, "veomni_flash_attention"),
        (
            {
                "ulysses_size": 2,
                "model_type": "llada_v",
                "vision_config": types.SimpleNamespace(_attn_implementation="veomni_flash_attention_2"),
            },
            "vision tower",
        ),
        ({"ulysses_size": 2, "num_attention_heads": 3, "num_key_value_heads": 3}, "must divide"),
        ({"ulysses_size": 4, "num_attention_heads": 8, "num_key_value_heads": 3}, "multiple of it"),
        ({"ulysses_size": 4, "num_attention_heads": 8, "num_key_value_heads": 6}, "multiple of it"),
    ],
)
def test_sequence_parallel_refusals(monkeypatch, setup, match):
    trainer = _trainer(monkeypatch, **setup)
    with pytest.raises(ValueError, match=match):
        trainer._build_collate_fn()


@pytest.mark.parametrize("model_type", ["llada", "llada_moe", "llada_v", "llada2_uni"])
def test_sequence_parallel_builds_with_repeated_kv_heads(monkeypatch, model_type):
    trainer = _trainer(monkeypatch, ulysses_size=4, model_type=model_type, num_key_value_heads=2)
    trainer._build_collate_fn()
    assert isinstance(trainer.base.collate_fn, dllm_trainer.DllmCollator)


def test_sequence_parallel_keeps_images_whole(monkeypatch):
    # two image samples, five tokens padded to six: every rank gets all pixels and the full-length mask
    features = _packed_features()[::2]
    features[0]["image_mask"] = torch.tensor([False, True, True])
    features[0]["pixel_values"] = torch.randn(1, 3, 4, 4)
    features[1]["image_mask"] = torch.tensor([True, False])
    features[1]["pixel_values"] = torch.randn(1, 3, 4, 4)
    pixels = torch.cat([feature["pixel_values"] for feature in features])
    for rank in range(2):
        trainer = _trainer(monkeypatch, ulysses_size=2, sp_rank=rank, model_type="llada_v")
        trainer._build_collate_fn()
        shard = trainer.base.collate_fn([dict(feature) for feature in features])
        assert torch.equal(shard["pixel_values"], pixels)
        assert torch.equal(shard["image_mask"], torch.tensor([[False, True, True, True, False, False]]))
        assert shard["input_ids"].shape == (1, 3)
