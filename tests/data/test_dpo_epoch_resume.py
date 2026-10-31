"""Exercise DPO cursor recovery through the native shuffled map loader."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_loader import DistributedDataloader
from veomni.data.dataset import MappingDataset, get_length_fn, get_length_fn_by_count_mode
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback
from veomni.trainer.text_dpo_trainer import _dpo_training_iterator


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _unpack(items):
    return items[0]


def _make_loader(workers, drop_last, count_mode=None, dynamic=True, num_micro_batch=1, buffer_size=1):
    samples = MappingDataset(
        [
            {"id": i, "attention_mask": torch.ones(1, dtype=torch.int64), "labels": torch.ones(1, dtype=torch.int64)}
            for i in range(60)
        ],
        seed=7,
    )
    sampler = StatefulDistributedSampler(samples, num_replicas=1, rank=0, shuffle=True, seed=7)
    options = {"multiprocessing_context": "fork", "prefetch_factor": 1} if workers else {}
    source = DistributedDataloader(
        samples,
        batch_size=1,
        sampler=sampler,
        num_workers=workers,
        collate_fn=_unpack,
        drop_last=drop_last,
        snapshot_every_n_steps=1,
        **options,
    )
    if not dynamic:
        return source
    strategy = TextBatchingStrategy(
        token_micro_bsz=1,
        buffer_size=buffer_size,
        get_length_fn=get_length_fn_by_count_mode(count_mode) if count_mode else None,
        physical_token_cap=2 if count_mode == "effective" else None,
        get_physical_length_fn=get_length_fn if count_mode == "effective" else None,
    )
    return DynamicBatchSizeDataLoader(source, strategy, num_micro_batch=num_micro_batch, length=6, drop_last=drop_last)


def _close(loader):
    iterator = getattr(loader, "_data_iter", getattr(loader, "_iterator", None))
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


def _trace(iterator, dynamic=True):
    result = []
    for _ in range(3):
        sample = next(iterator)
        if dynamic:
            ids = tuple(micro[0]["id"] for micro in sample)
        else:
            ids = (sample["id"],)
        result.append((ids, float(torch.rand(()))))
    return result


class _DPOTrainer(SimpleNamespace):
    def validate_dpo_provenance_checkpoint(self, _state):
        pass


def _fresh_cursor_worker(_rank, checkpoint, result, count_mode):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    loader = _make_loader(0, True, count_mode)
    loader._dpo_disallow_implicit_restart = True
    try:
        loader.set_epoch(2)
        loader.load_state_dict(state["cursor"])
        torch.set_rng_state(state["rng"])
        torch.save(_trace(iter(loader)), result)
    finally:
        _close(loader)


@pytest.mark.parametrize("count_mode", ["total", "effective"])
def test_native_strategy_cursor_survives_fresh_process(tmp_path, count_mode):
    torch.manual_seed(12345)
    original = _make_loader(0, True, count_mode)
    original.set_epoch(2)
    iterator = iter(original)
    next(iterator)
    checkpoint = tmp_path / "cursor.pt"
    torch.save({"cursor": original.state_dict(), "rng": torch.get_rng_state()}, checkpoint)
    expected = _trace(iterator)
    result = tmp_path / "result.pt"
    mp.spawn(_fresh_cursor_worker, args=(str(checkpoint), str(result), count_mode), nprocs=1, join=True)
    assert torch.load(result, weights_only=True) == expected


@pytest.mark.parametrize("workers", [0, 1, 2])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("phase", ["initial", "within", "boundary"])
@pytest.mark.parametrize(("dynamic", "num_micro_batch", "buffer_size"), [(False, 1, 1), (True, 1, 1), (True, 2, 2)])
def test_native_dpo_cursor_restores_epoch_samples_and_cpu_rng(
    workers, drop_last, phase, dynamic, num_micro_batch, buffer_size
):
    torch.manual_seed(12345)
    step = {"initial": 0, "within": 2, "boundary": 6}[phase]
    boundary = phase == "boundary"
    original = _make_loader(
        workers, drop_last, dynamic=dynamic, num_micro_batch=num_micro_batch, buffer_size=buffer_size
    )
    original.set_epoch(2)
    try:
        iterator = iter(original)
        for _ in range(step):
            next(iterator)
        cursor = original.state_dict()
        saved_rng = torch.get_rng_state()
        if boundary:
            original.set_epoch(3)
            iterator = iter(original)
        expected = _trace(iterator, dynamic)
    finally:
        _close(original)

    restored = _make_loader(
        workers, drop_last, dynamic=dynamic, num_micro_batch=num_micro_batch, buffer_size=buffer_size
    )
    try:
        checkpoint = SimpleNamespace(
            load_path="/unused",
            save_path=None,
            save_steps=0,
            save_epochs=0,
            save_hf_weights=False,
            hf_save_steps=0,
            hf_save_epochs=0,
        )
        args = SimpleNamespace(
            train=SimpleNamespace(
                global_rank=0,
                checkpoint=checkpoint,
                num_train_epochs=5,
                seed=7,
                dataloader_batch_size=1,
                dyn_bsz=dynamic,
            ),
            data=SimpleNamespace(
                dataloader=SimpleNamespace(
                    num_workers=workers, shuffle=True, persistent_workers=False, in_order=True, drop_last=drop_last
                )
            ),
            train_steps=6,
        )
        trainer = _DPOTrainer(
            args=args,
            state=SimpleNamespace(global_step=0, epoch=0),
            train_dataloader=restored,
            channel_loss_callback=None,
            offline_cache_writer=None,
            environ_meter=None,
            policy_model=SimpleNamespace(parallel_state=SimpleNamespace(dp_size=1, dp_rank=0)),
        )
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._read_current = lambda _: {
            "global_step": 12 + step,
            "dpo_position": {"epoch": 2, "step_in_epoch": step, "epoch_complete": boundary},
            "train_dataloader": cursor,
            "torch_rng_state": saved_rng,
        }
        with patch("veomni.trainer.callbacks.global_state_callback.set_device_rng_state"):
            callback.load_global_state()
        restored.set_epoch(trainer.start_epoch)
        resumed_iterator = _dpo_training_iterator(
            restored, False, getattr(trainer, "_dpo_restore_fixed_iterator_rng", False)
        )
        assert _trace(resumed_iterator, dynamic) == expected
    finally:
        _close(restored)
