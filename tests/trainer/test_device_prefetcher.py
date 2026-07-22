import copy
from types import SimpleNamespace

import torch

import veomni.trainer.base as trainer_base
from veomni.trainer.base import BaseTrainer, DevicePrefetcher, VeOmniIter, _move_batch_to_device
from veomni.utils.constants import IGNORE_INDEX


class _StatefulIterator:
    def __init__(self, batches):
        self.batches = list(batches)
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.idx]
        self.idx += 1
        return copy.deepcopy(batch)

    def state_dict(self):
        return {"idx": self.idx}


def _batch(value):
    return [
        {
            "input_ids": torch.tensor([value, value + 1]),
            "labels": torch.tensor([value + 2, IGNORE_INDEX]),
            "cu_seq_lens_q": torch.tensor([0, 2], dtype=torch.int32),
            "linear_attn_cu_seq_lens_q": torch.tensor([0, 2], dtype=torch.int32),
            "multimodal_metadata": {
                "pixel_values": torch.tensor([float(value)]),
                "vit_image_cu_seqlens": torch.tensor([0, 1], dtype=torch.int32),
            },
        }
    ]


def test_move_batch_to_device_recurses_and_preserves_cpu_fa_kwargs():
    batch = _batch(1)

    moved = _move_batch_to_device(batch, torch.device("meta"))

    assert moved[0]["input_ids"].device.type == "meta"
    assert moved[0]["labels"].device.type == "cpu"
    assert moved[0]["cu_seq_lens_q"].device.type == "cpu"
    assert moved[0]["linear_attn_cu_seq_lens_q"].device.type == "cpu"
    assert torch.equal(moved[0]["cu_seq_lens_q"], batch[0]["cu_seq_lens_q"])
    assert torch.equal(moved[0]["linear_attn_cu_seq_lens_q"], batch[0]["linear_attn_cu_seq_lens_q"])
    assert moved[0]["multimodal_metadata"]["pixel_values"].device.type == "meta"


def test_device_prefetcher_returns_consumed_batch_state():
    prefetcher = DevicePrefetcher(_StatefulIterator([_batch(1), _batch(10)]), device=torch.device("cpu"))

    first = next(prefetcher)
    assert first[0]["input_ids"].tolist() == [1, 2]
    assert prefetcher.state_dict() == {"idx": 1}

    second = next(prefetcher)
    assert second[0]["input_ids"].tolist() == [10, 11]
    assert prefetcher.state_dict() == {"idx": 2}


def test_veomni_iter_device_prefetcher_is_cuda_only(monkeypatch):
    monkeypatch.setattr(trainer_base, "IS_CUDA_AVAILABLE", False)
    iterator = VeOmniIter(_StatefulIterator([_batch(1)]), use_device_prefetcher=True, device=torch.device("cpu"))

    assert not isinstance(iterator.iterator, DevicePrefetcher)
    assert next(iterator)[0]["input_ids"].tolist() == [1, 2]


def test_build_data_iterator_passes_device_prefetch_flag(monkeypatch):
    captured = {}

    class _FakeIter:
        def __init__(self, dataloader, **kwargs):
            captured["dataloader"] = dataloader
            captured.update(kwargs)

    monkeypatch.setattr(trainer_base, "VeOmniIter", _FakeIter)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.train_dataloader = object()
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(
        data=SimpleNamespace(dataloader=SimpleNamespace(use_background_prefetcher=True, use_device_prefetcher=True))
    )

    assert isinstance(trainer.build_data_iterator(), _FakeIter)
    assert captured["dataloader"] is trainer.train_dataloader
    assert captured["use_background_prefetcher"] is True
    assert captured["use_device_prefetcher"] is True
    assert captured["device"] == torch.device("cpu")
