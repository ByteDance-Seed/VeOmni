import types
from functools import partial
from typing import Literal

import pytest
import torch
from utils import DummyDataset, process_dummy_example

from veomni.data import build_dataloader, build_dataset
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy


def _fake_ps(sp_size: int, *, cp_size: int = 1, ulysses_size: int | None = None):
    sp_enabled = sp_size > 1
    return types.SimpleNamespace(
        dp_size=1,
        dp_rank=0,
        sp_enabled=sp_enabled,
        sp_size=sp_size,
        sp_rank=0,
        cp_size=cp_size,
        cp_rank=0,
        ulysses_size=sp_size if ulysses_size is None else ulysses_size,
        ulysses_rank=0,
        gdn_context_parallel_implementation="kcp" if cp_size > 1 else "disabled",
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


@pytest.mark.parametrize("dyn_bsz_runtime", ["main", "worker"])
@pytest.mark.parametrize(("count_mode", "expected_cap"), [("total", 8), ("effective", 12)])
def test_build_dataloader_dyn_bsz_accounts_for_per_sample_cp_rounding(
    monkeypatch,
    dummy_dataset_ci,
    dyn_bsz_runtime: Literal["main", "worker"],
    count_mode: Literal["total", "effective"],
    expected_cap: int,
):
    import veomni.data.data_collator as m_col
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=4, cp_size=2, ulysses_size=2)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_col, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=8),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=1,
        global_batch_size=2,
        dataloader_batch_size=1,
        max_seq_len=8,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime=dyn_bsz_runtime,
        dyn_bsz_count_mode=count_mode,
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    if dyn_bsz_runtime == "main":
        physical_cap = dl.batching_strategy.physical_token_cap
        physical_length_fn = dl.batching_strategy.buffer._get_physical_length_fn
    else:
        physical_cap = dl.dataset.physical_token_cap
        physical_length_fn = dl.dataset.get_physical_length_fn

    assert physical_cap == expected_cap
    assert physical_length_fn({"attention_mask": torch.ones(3, dtype=torch.long)}) == 8
    assert physical_length_fn({"attention_mask": torch.ones(2, dtype=torch.long)}) == 8


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


@pytest.mark.parametrize(("cp_size", "ulysses_size"), [(1, 4), (2, 2)])
def test_debug_sample_alignment_makes_cp_and_non_cp_use_one_physical_policy(
    monkeypatch, dummy_dataset_ci, cp_size, ulysses_size
):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=4, cp_size=cp_size, ulysses_size=ulysses_size)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setenv("VEOMNI_DYN_BSZ_SAMPLE_ALIGNMENT", "16")

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=32),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=1,
        global_batch_size=2,
        dataloader_batch_size=1,
        max_seq_len=32,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_count_mode="total",
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    physical_length_fn = dl.batching_strategy.buffer._get_physical_length_fn
    assert dl.batching_strategy.physical_token_cap == 32
    assert physical_length_fn({"attention_mask": torch.ones(1, dtype=torch.long)}) == 16
    assert physical_length_fn({"attention_mask": torch.ones(17, dtype=torch.long)}) == 32


@pytest.mark.parametrize("value", ["", "0", "-1", "4.0", "true"])
def test_debug_sample_alignment_rejects_invalid_values(monkeypatch, value):
    import veomni.data.data_loader as m_dl

    monkeypatch.setenv("VEOMNI_DYN_BSZ_SAMPLE_ALIGNMENT", value)
    with pytest.raises(ValueError, match="must be a positive base-10 integer"):
        m_dl._debug_physical_length_multiple()


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
