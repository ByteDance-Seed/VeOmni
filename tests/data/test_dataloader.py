import types
from functools import partial
from typing import Literal

import pytest
import torch
from utils import DummyDataset, process_dummy_example

from veomni.data import build_dataloader, build_dataset
from veomni.data.data_collator import NoopDataCollator
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy


def _fake_ps(sp_size: int):
    sp_enabled = sp_size > 1
    return types.SimpleNamespace(
        dp_size=1,
        dp_rank=0,
        sp_enabled=sp_enabled,
        sp_size=sp_size,
        sp_rank=0,
    )


@pytest.fixture(scope="session")
def dummy_dataset_ci():
    dummy = DummyDataset(size=40, num_shard=1, dataset_name="ci_dyn_bsz_shared")
    yield dummy
    dummy.clean_cache()


@pytest.mark.parametrize("dataset_name", ["iterable", "mapping"])
@pytest.mark.parametrize("dyn_bsz", [True, False])
@pytest.mark.parametrize("sp_size", [1, 2])
@pytest.mark.parametrize("dyn_bsz_runtime", ["main", "worker"])
def test_build_dataloader_dyn_bsz_sp_filling(
    monkeypatch,
    dummy_dataset_ci,
    dataset_name: str,
    dyn_bsz: bool,
    sp_size: int,
    dyn_bsz_runtime: Literal["main", "worker"],
):
    import veomni.data.data_collator as m_col
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=sp_size)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_col, "get_parallel_state", lambda: ps)

    global_batch_size = 8
    micro_batch_size = 2
    max_seq_len = 100

    if dyn_bsz:
        if dyn_bsz_runtime == "main":
            dataloader_batch_size = 1
        else:
            dataloader_batch_size = global_batch_size // micro_batch_size
    else:
        dataloader_batch_size = global_batch_size

    transform = partial(process_dummy_example, max_seq_len=max_seq_len)

    dataset = build_dataset(
        dataset_name=dataset_name,
        train_path=dummy_dataset_ci.save_path,
        transform=transform,
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        dataloader_batch_size=dataloader_batch_size,
        max_seq_len=max_seq_len,
        train_steps=1,
        num_workers=0,
        dyn_bsz=dyn_bsz,
        dyn_bsz_runtime=dyn_bsz_runtime,
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    micro_batches = next(iter(dl))

    if dyn_bsz:
        assert len(micro_batches) == global_batch_size // micro_batch_size
        for micro_batch in micro_batches:
            assert max_seq_len * (micro_batch_size - 1) <= sum(micro_batch["id"]) <= max_seq_len * micro_batch_size
    else:
        assert len(micro_batches) == global_batch_size // micro_batch_size
        for micro_batch in micro_batches:
            assert len(micro_batch["id"]) == micro_batch_size


@pytest.mark.parametrize("dyn_bsz_runtime", ["main", "worker"])
def test_build_dataloader_dyn_bsz_count_mode(
    monkeypatch, dummy_dataset_ci, dyn_bsz_runtime: Literal["main", "worker"]
):
    import veomni.data.data_collator as m_col
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_col, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1 if dyn_bsz_runtime == "main" else 2,
        max_seq_len=16,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime=dyn_bsz_runtime,
        dyn_bsz_count_mode="effective",
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    if dyn_bsz_runtime == "main":
        assert isinstance(dl, DynamicBatchSizeDataLoader)
        assert isinstance(dl.batching_strategy, TextBatchingStrategy)
        assert dl.batching_strategy.buffer._get_length_fn is m_ds.get_length_by_labels_fn
        assert dl.batching_strategy.physical_token_cap == 48
        assert dl.batching_strategy.buffer._get_physical_length_fn is m_ds.get_length_by_attention_mask_fn
    else:
        assert isinstance(dl.dataset, m_ds.DynamicBatchingSizeDataset)
        assert dl.dataset.get_length_fn is m_ds.get_length_by_labels_fn
        assert dl.dataset.physical_token_cap == 48
        assert dl.dataset.get_physical_length_fn is m_ds.get_length_by_attention_mask_fn


def test_build_dataloader_dyn_bsz_physical_overflow_ratio(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1,
        max_seq_len=16,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_count_mode="effective",
        dyn_bsz_physical_overflow_ratio=1.25,
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    assert dl.batching_strategy.physical_token_cap == 40


def test_build_dataloader_suppresses_persistent_workers_with_zero_workers(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1,
        max_seq_len=16,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        persistent_workers=True,
        in_order=False,
        seed=0,
    )

    assert dl._dataloader.persistent_workers is False
    assert dl._dataloader.in_order is False


def test_build_dataloader_forwards_worker_scheduling_kwargs(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    captured = {}

    class FakeDistributedDataloader:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_dl, "DistributedDataloader", FakeDistributedDataloader)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1,
        max_seq_len=16,
        train_steps=1,
        num_workers=2,
        dyn_bsz=False,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
        in_order=False,
        seed=0,
    )

    assert captured["persistent_workers"] is True
    assert captured["in_order"] is False


def test_build_dataloader_rejects_invalid_physical_overflow_ratio(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    with pytest.raises(ValueError, match="dyn_bsz_physical_overflow_ratio must be >= 1.0"):
        build_dataloader(
            "native",
            dataset=dataset,
            micro_batch_size=2,
            global_batch_size=4,
            dataloader_batch_size=1,
            max_seq_len=16,
            train_steps=1,
            num_workers=0,
            dyn_bsz=True,
            dyn_bsz_runtime="main",
            dyn_bsz_count_mode="effective",
            dyn_bsz_physical_overflow_ratio=0.5,
            dyn_bsz_buffer_size=1,
            drop_last=True,
            prefetch_factor=None,
            seed=0,
        )


class _RestartableLoader:
    """Minimal stand-in for the per-rank torch DataLoader that
    DynamicBatchSizeDataLoader wraps: a plain, restartable, sized iterable."""

    def __init__(self, items):
        """Keep the samples to replay on every iteration."""
        self._items = items

    def __iter__(self):
        """Start a fresh pass over the samples."""
        return iter(self._items)

    def __len__(self):
        """Number of samples per pass."""
        return len(self._items)


def _dyn_bsz_sample(index: int, num_tokens: int = 10):
    """Build a raw sample whose token ids all equal ``index``."""
    return {
        "id": index,
        "input_ids": torch.full((num_tokens,), index, dtype=torch.long),
        "attention_mask": torch.ones(num_tokens, dtype=torch.long),
    }


def _merging_collate_fn(samples):
    """A collate_fn that merges the raw samples into one dict, like MainCollator."""
    return {"ids": [sample["id"] for sample in samples]}


def _is_padding(micro_batch) -> bool:
    """True if a (collated or uncollated) micro batch is flagged as padding."""
    if isinstance(micro_batch, dict):
        return bool(micro_batch.get("padding_flag", False))
    return all(bool(sample.get("padding_flag", False)) for sample in micro_batch)


@pytest.mark.parametrize("collate_fn", [_merging_collate_fn, NoopDataCollator(), None])
def test_dynamic_batch_dataloader_drop_last_false_tail(collate_fn):
    """``drop_last=False`` must pad only a genuinely incomplete tail step.

    Two regressions are covered:

    1. the tail drain emitted a step even when the leftover samples already filled
       whole steps, fabricating one made entirely of duplicated, already-yielded
       samples;
    2. flagging the padding micro batch assumed it was a dict, so it raised
       ``TypeError: list indices must be integers or slices, not str`` for the
       uncollated ``list[dict]`` shape -- which is what ``collate_fn=None`` (this
       class's own default) and ``NoopDataCollator`` (installed by
       ``build_dataloader(build_collate_fn=False)``) produce.
    """
    num_samples, num_micro_batch, train_steps = 9, 2, 8
    dataloader = DynamicBatchSizeDataLoader(
        dataloader=_RestartableLoader([_dyn_bsz_sample(i) for i in range(num_samples)]),
        batching_strategy=TextBatchingStrategy(token_micro_bsz=10, buffer_size=3),
        collate_fn=collate_fn,
        num_micro_batch=num_micro_batch,
        length=train_steps,
        drop_last=False,
    )

    steps = list(dataloader)

    assert not any(all(_is_padding(micro_batch) for micro_batch in step) for step in steps), (
        "an all-padding step carries no real data and no loss tokens; it must not be emitted"
    )
    assert len(steps) <= train_steps + 1, f"at most one padded tail step may follow the {train_steps} requested steps"


@pytest.mark.parametrize("collate_fn", [_merging_collate_fn, NoopDataCollator(), None])
def test_dynamic_batch_dataloader_drop_last_false_pads_incomplete_tail(collate_fn):
    """A genuinely incomplete tail step is padded up to ``num_micro_batch``.

    Four one-sample micro batches, three per step and two requested steps leave two
    real micro batches in the drain, so exactly one padding copy must be appended,
    for the collated ``dict`` shape as well as the uncollated ``list[dict]`` one.
    """
    num_samples, num_micro_batch, train_steps = 4, 3, 2
    dataloader = DynamicBatchSizeDataLoader(
        dataloader=_RestartableLoader([_dyn_bsz_sample(i) for i in range(num_samples)]),
        batching_strategy=TextBatchingStrategy(token_micro_bsz=10, buffer_size=3),
        collate_fn=collate_fn,
        num_micro_batch=num_micro_batch,
        length=train_steps,
        drop_last=False,
    )

    steps = list(dataloader)

    assert len(steps) == train_steps + 1, "the incomplete tail must be emitted as one extra step"
    assert all(len(step) == num_micro_batch for step in steps)
    assert not any(_is_padding(micro_batch) for step in steps[:-1] for micro_batch in step), (
        "only the tail step may carry padding"
    )
    assert [_is_padding(micro_batch) for micro_batch in steps[-1]] == [False, False, True]
    assert steps[-1][-1] is not steps[-1][-2], "the padding micro batch must be a copy, not an alias"
