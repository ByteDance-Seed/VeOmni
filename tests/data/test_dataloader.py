import types
from functools import partial

import pytest
from utils import DummyDataset, process_dummy_example

from veomni.data import build_dataloader, build_dataset


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
def test_build_dataloader_dyn_bsz_sp_filling(
    monkeypatch, dummy_dataset_ci, dataset_name: str, dyn_bsz: bool, sp_size: int
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
        dataloader_batch_size = 1
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
