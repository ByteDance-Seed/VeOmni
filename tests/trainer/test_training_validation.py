import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, IterableDataset

import veomni.trainer.callbacks.base as callback_base_module
import veomni.trainer.text_trainer as text_trainer_module
import veomni.trainer.validation as validation_module
from veomni.data.data_collator import MainCollator
from veomni.data.data_loader import ExactDistributedBatchSampler
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.evaluate_callback import EvaluateCallback
from veomni.trainer.text_trainer import TextTrainer
from veomni.trainer.validation import TextValidationRunner


def _parallel_state(**overrides):
    values = {
        "dp_size": 1,
        "dp_rank": 0,
        "dp_group": None,
        "sp_size": 1,
        "sp_rank": 0,
        "sp_enabled": False,
        "tp_enabled": False,
        "pp_enabled": False,
        "any_extra_parallel_enabled": False,
        "async_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _ValidationModel(torch.nn.Module):
    def __init__(self, fail_on_call=None):
        super().__init__()
        self.dropout = torch.nn.Dropout()
        self.fail_on_call = fail_on_call
        self.calls = 0

    def forward(self, labels, loss_value, use_cache=False):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("validation forward failed")
        return SimpleNamespace(loss=loss_value)


class _ValidationDataloader:
    def __init__(self, batches):
        self.batches = batches
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __iter__(self):
        return iter(self.batches)


def _runner(model, batches):
    trainer = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        model_fwd_context=nullcontext(),
        args=SimpleNamespace(
            data=SimpleNamespace(data_type="conversation"),
            train=SimpleNamespace(enable_batch_invariant_mode=False, seed=0),
        ),
    )
    runner = TextValidationRunner.__new__(TextValidationRunner)
    runner.trainer = trainer
    runner.parallel_state = _parallel_state()
    runner.dataloader = _ValidationDataloader(batches)
    runner.dataloader_generator = torch.Generator().manual_seed(0)
    return runner


def test_text_validation_runner_computes_weighted_loss_and_restores_mixed_module_modes(monkeypatch):
    monkeypatch.setattr(validation_module, "use_parallel_state", lambda _name: nullcontext())
    monkeypatch.setattr(validation_module, "set_batch_invariant_mode", lambda _enabled: nullcontext())
    model = _ValidationModel()
    model.train()
    model.dropout.eval()
    batches = [
        [{"labels": torch.tensor([[-100, 1, 2]]), "loss_value": torch.tensor(2.0)}],
        [{"labels": torch.tensor([[-100, 3]]), "loss_value": torch.tensor(4.0)}],
    ]
    runner = _runner(model, batches)
    global_rng_state = torch.random.get_rng_state().clone()

    metrics = runner.run()

    assert metrics == {"loss": pytest.approx(8 / 3)}
    assert model.training is True
    assert model.dropout.training is False
    assert runner.dataloader.epochs == [0]
    assert torch.equal(torch.random.get_rng_state(), global_rng_state)
    assert runner.run() == metrics
    assert runner.dataloader.epochs == [0, 0]
    assert torch.equal(torch.random.get_rng_state(), global_rng_state)


@pytest.mark.parametrize(
    ("labels", "expected_units"),
    [
        ([[7, 8, 9]], 2),
        ([[7]], 0),
        ([[7, -100, 9]], 1),
        ([[-100, 8, 9]], 2),
    ],
)
def test_text_validation_runner_counts_only_shifted_causal_targets(labels, expected_units):
    runner = _runner(_ValidationModel(), [])

    # Position zero has no preceding logit, even when its label is valid.
    assert runner._count_loss_units(torch.tensor(labels)).item() == expected_units


def test_text_validation_runner_restores_module_modes_when_forward_fails(monkeypatch):
    monkeypatch.setattr(validation_module, "use_parallel_state", lambda _name: nullcontext())
    monkeypatch.setattr(validation_module, "set_batch_invariant_mode", lambda _enabled: nullcontext())
    model = _ValidationModel(fail_on_call=1)
    model.train()
    model.dropout.eval()
    runner = _runner(
        model,
        [[{"labels": torch.tensor([[-100, 1]]), "loss_value": torch.tensor(2.0)}]],
    )

    with pytest.raises(RuntimeError, match="validation forward failed"):
        runner.run()

    assert model.training is True
    assert model.dropout.training is False


def test_text_validation_runner_rejects_zero_loss_targets(monkeypatch):
    monkeypatch.setattr(validation_module, "use_parallel_state", lambda _name: nullcontext())
    monkeypatch.setattr(validation_module, "set_batch_invariant_mode", lambda _enabled: nullcontext())
    runner = _runner(
        _ValidationModel(),
        [[{"labels": torch.tensor([[-100, -100]]), "loss_value": torch.tensor(float("nan"))}]],
    )

    with pytest.raises(ValueError, match="no non-ignored loss targets"):
        runner.run()


def test_text_validation_runner_reduces_weighted_stats_over_captured_dp_group(monkeypatch):
    monkeypatch.setattr(validation_module, "use_parallel_state", lambda _name: nullcontext())
    monkeypatch.setattr(validation_module, "set_batch_invariant_mode", lambda _enabled: nullcontext())
    group = object()
    calls = []

    def fake_all_reduce(tensor, op, group):
        calls.append((tensor.dtype, op, group))
        if tensor.dtype == torch.float32:
            tensor += 12.0
        else:
            tensor += 3

    monkeypatch.setattr(validation_module.dist, "all_reduce", fake_all_reduce)
    runner = _runner(
        _ValidationModel(),
        [[{"labels": torch.tensor([[-100, 1, 2]]), "loss_value": torch.tensor(2.0)}]],
    )
    runner.parallel_state = _parallel_state(dp_size=2, dp_group=group)

    metrics = runner.run()

    assert metrics == {"loss": pytest.approx(16 / 5)}
    assert calls == [
        (torch.float32, torch.distributed.ReduceOp.SUM, group),
        (torch.int64, torch.distributed.ReduceOp.SUM, group),
    ]


def _two_rank_validation_worker(rank, world_size, init_file):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        validation_module.use_parallel_state = lambda _name: nullcontext()
        validation_module.set_batch_invariant_mode = lambda _enabled: nullcontext()
        sampler = ExactDistributedBatchSampler(dataset_size=5, batch_size=2, num_replicas=world_size, rank=rank)
        batches = []
        for batch_indices in sampler:
            loss_values = [index + 1 for index in batch_indices]
            batches.append(
                [
                    {
                        "labels": torch.tensor([[-100, *loss_values]]),
                        "loss_value": torch.tensor(sum(loss_values) / len(loss_values), dtype=torch.float32),
                    }
                ]
            )

        model = _ValidationModel()
        runner = _runner(model, batches)
        runner.parallel_state = _parallel_state(
            dp_size=world_size,
            dp_rank=rank,
            dp_group=dist.group.WORLD,
        )
        metrics = runner.run()

        assert metrics == {"loss": pytest.approx(3.0)}
        assert [len(micro_batches[0]["labels"][0]) - 1 for micro_batches in batches] == (
            [2, 1] if rank == 0 else [1, 1]
        )
        assert model.training is True
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for the CPU collective test.")
def test_text_validation_runner_two_rank_gloo_uneven_batches(tmp_path):
    mp.spawn(
        _two_rank_validation_worker,
        args=(2, str(tmp_path / "validation_pg")),
        nprocs=2,
        join=True,
    )


@dataclass
class _DataloaderArgs:
    type: str = "native"
    num_workers: int = 0
    worker_num_threads: int | None = None
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    in_order: bool = False
    drop_last: bool = True
    pin_memory: bool = False
    use_background_prefetcher: bool = False


@dataclass
class _DataArgs:
    train_path: str = "train.jsonl"
    eval_path: str = "eval.jsonl"
    data_type: str = "conversation"
    datasets_type: str = "mapping"
    multisource_datasets_type: str = "interleave"
    max_seq_len: int = 128
    dataloader: _DataloaderArgs = field(default_factory=_DataloaderArgs)


def _builder_trainer():
    return SimpleNamespace(
        args=SimpleNamespace(
            data=_DataArgs(),
            train=SimpleNamespace(
                eval_steps=10,
                eval_epochs=1,
                seed=7,
                micro_batch_size=2,
                moe_load_balance_monitor_interval=0,
                profile=SimpleNamespace(enable=False),
                torch_compile=SimpleNamespace(enable=False),
            ),
        ),
        model_config=SimpleNamespace(num_experts=8),
        data_transform=object(),
        collate_fn=object(),
    )


class _MapDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return index


class _PackedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return [{name: value.clone() for name, value in self.samples[index].items()}]


class _TinyPackedLossModel(torch.nn.Module):
    def __init__(self, classification):
        super().__init__()
        self.classification = classification

    def forward(self, labels, use_cache=False, **_kwargs):
        targets = labels if self.classification else labels[..., 1:]
        valid_targets = targets[targets != -100]
        return SimpleNamespace(loss=valid_targets.float().mean())


def test_text_validation_runner_builds_single_source_eval_path_with_exact_sampler(monkeypatch):
    captured = {}
    dataloader_sentinel = object()
    trainer = _builder_trainer()
    trainer.args.data.dataloader.persistent_workers = True

    def fake_build_dataset(**kwargs):
        captured["dataset"] = kwargs
        return _MapDataset()

    def fake_build_dataloader(**kwargs):
        captured["dataloader"] = kwargs
        return dataloader_sentinel

    monkeypatch.setattr(validation_module, "get_parallel_state", lambda: _parallel_state(dp_size=2, dp_rank=1))
    monkeypatch.setattr(validation_module, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(validation_module, "build_dataloader", fake_build_dataloader)

    runner = TextValidationRunner(trainer)

    assert captured["dataset"]["dataset_name"] == "mapping"
    assert captured["dataset"]["train_path"] == "eval.jsonl"
    assert captured["dataset"]["shuffle"] is False
    assert captured["dataset"]["split_by_node"] is False
    assert captured["dataloader"]["dyn_bsz"] is False
    assert captured["dataloader"]["in_order"] is True
    assert captured["dataloader"]["persistent_workers"] is False
    assert isinstance(captured["dataloader"]["batch_sampler"], ExactDistributedBatchSampler)
    assert captured["dataloader"]["generator"] is runner.dataloader_generator
    assert runner.dataloader is dataloader_sentinel


@pytest.mark.parametrize(
    ("data_type", "classification", "samples", "expected_loss"),
    [
        (
            "conversation",
            False,
            [
                {
                    "input_ids": torch.tensor([0, 1, 2]),
                    "attention_mask": torch.ones(3, dtype=torch.long),
                    "labels": torch.tensor([-100, 1, 2]),
                },
                {
                    "input_ids": torch.tensor([3, 4]),
                    "attention_mask": torch.ones(2, dtype=torch.long),
                    "labels": torch.tensor([-100, 4]),
                },
            ],
            7 / 3,
        ),
        (
            "classification",
            True,
            [
                {
                    "input_ids": torch.tensor([0, 1]),
                    "attention_mask": torch.ones(2, dtype=torch.long),
                    "labels": torch.tensor([-100, 2]),
                },
                {
                    "input_ids": torch.tensor([3, 4]),
                    "attention_mask": torch.ones(2, dtype=torch.long),
                    "labels": torch.tensor([-100, 4]),
                },
            ],
            3.0,
        ),
    ],
)
def test_text_validation_runner_real_collator_and_dataloader_integration(
    monkeypatch, data_type, classification, samples, expected_loss
):
    import veomni.data.data_collator as data_collator_module
    import veomni.data.data_loader as data_loader_module

    parallel_state = _parallel_state()
    monkeypatch.setattr(validation_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(data_loader_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(data_collator_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(validation_module, "use_parallel_state", lambda _name: nullcontext())
    monkeypatch.setattr(validation_module, "set_batch_invariant_mode", lambda _enabled: nullcontext())
    monkeypatch.setattr(validation_module, "build_dataset", lambda **_kwargs: _PackedDataset(samples))
    trainer = _builder_trainer()
    trainer.args.data.data_type = data_type
    trainer.collate_fn = MainCollator(seq_classification=classification)
    trainer.model = _TinyPackedLossModel(classification=classification)
    trainer.device = torch.device("cpu")
    trainer.model_fwd_context = nullcontext()
    trainer.args.train.enable_batch_invariant_mode = False

    runner = TextValidationRunner(trainer)

    assert runner.run() == {"loss": pytest.approx(expected_loss)}


class _IterableValidationDataset(IterableDataset):
    def __iter__(self):
        return iter(())


def test_text_validation_runner_rejects_iterable_validation_data(monkeypatch):
    monkeypatch.setattr(validation_module, "get_parallel_state", lambda: _parallel_state())
    monkeypatch.setattr(validation_module, "build_dataset", lambda **_kwargs: _IterableValidationDataset())

    with pytest.raises(ValueError, match="requires a map-style dataset"):
        TextValidationRunner(_builder_trainer())


@pytest.mark.parametrize(
    ("data_type", "eval_path", "match"),
    [
        ("plaintext", "eval.jsonl", "data_type='plaintext'"),
        ("conversation", "eval.yaml", "multisource evaluation YAML"),
    ],
)
def test_text_validation_runner_rejects_non_exact_dataset_semantics(monkeypatch, data_type, eval_path, match):
    monkeypatch.setattr(validation_module, "get_parallel_state", lambda: _parallel_state())
    trainer = _builder_trainer()
    trainer.args.data.data_type = data_type
    trainer.args.data.eval_path = eval_path

    with pytest.raises(ValueError, match=match):
        TextValidationRunner(trainer)


@pytest.mark.parametrize(
    ("parallel_overrides", "match"),
    [
        ({"sp_enabled": True}, "sequence/context parallelism"),
        ({"tp_enabled": True}, "tensor parallelism"),
        ({"pp_enabled": True}, "pipeline parallelism"),
        ({"any_extra_parallel_enabled": True}, "ExtraParallel/EP"),
        ({"async_enabled": True}, "async sequence parallelism"),
    ],
)
def test_text_validation_runner_rejects_unimplemented_parallel_semantics(monkeypatch, parallel_overrides, match):
    monkeypatch.setattr(
        validation_module,
        "get_parallel_state",
        lambda: _parallel_state(**parallel_overrides),
    )

    with pytest.raises(ValueError, match=match):
        TextValidationRunner(_builder_trainer())


@pytest.mark.parametrize(
    ("config_name", "match"),
    [
        ("torch_compile", "torch.compile"),
        ("moe_load_balance_monitor_interval", "MoE router monitoring"),
        ("profile", "torch profiler"),
    ],
)
def test_text_validation_runner_rejects_unimplemented_execution_modes(monkeypatch, config_name, match):
    monkeypatch.setattr(validation_module, "get_parallel_state", lambda: _parallel_state())
    trainer = _builder_trainer()
    config = getattr(trainer.args.train, config_name)
    if hasattr(config, "enable"):
        config.enable = True
    else:
        setattr(trainer.args.train, config_name, 1)

    with pytest.raises(ValueError, match=match):
        TextValidationRunner(trainer)


def test_base_trainer_rejects_validation_without_task_owned_runner():
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(
        data=SimpleNamespace(eval_path="eval.jsonl"),
        train=SimpleNamespace(eval_steps=10, eval_epochs=1),
    )

    with pytest.raises(ValueError, match="supported only by TextTrainer"):
        trainer._validate_validation_configuration()

    trainer.validation_runner = object()
    trainer._validate_validation_configuration()


@pytest.mark.parametrize(
    ("eval_path", "eval_steps", "eval_epochs", "expected"),
    [("eval.jsonl", 10, 0, True), ("eval.jsonl", 0, 1, True), ("eval.jsonl", 0, 0, False), (None, 10, 1, False)],
)
def test_text_trainer_installs_validation_runner_only_when_scheduled(
    monkeypatch, eval_path, eval_steps, eval_epochs, expected
):
    installed = []

    class _Runner:
        is_requested = staticmethod(TextValidationRunner.is_requested)

        def __init__(self, trainer):
            installed.append(trainer)

    monkeypatch.setattr(text_trainer_module, "TextValidationRunner", _Runner)
    trainer = TextTrainer.__new__(TextTrainer)
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(eval_path=eval_path),
            train=SimpleNamespace(eval_steps=eval_steps, eval_epochs=eval_epochs),
        )
    )

    trainer._build_validation_runner()

    assert bool(installed) is expected
    assert hasattr(trainer.base, "validation_runner") is expected


def test_evaluate_callback_deduplicates_step_and_epoch_trigger(monkeypatch):
    monkeypatch.setattr(callback_base_module, "get_parallel_state", lambda: _parallel_state())

    class _Runner:
        def __init__(self):
            self.calls = 0

        def run(self):
            self.calls += 1
            return {"loss": 1.25}

    runner = _Runner()
    trainer = SimpleNamespace(
        validation_runner=runner,
        args=SimpleNamespace(
            train=SimpleNamespace(
                eval_steps=2,
                eval_epochs=1,
                global_rank=0,
                wandb=SimpleNamespace(enable=False),
            )
        ),
    )
    callback = EvaluateCallback(trainer)
    state = TrainerState(global_step=2, epoch=0)

    callback.on_step_end(state)
    callback.on_epoch_end(state)

    assert runner.calls == 1
    assert trainer.last_validation_metrics == {"loss": 1.25}

    state.global_step = 3
    state.epoch = 1
    callback.on_epoch_end(state)
    assert runner.calls == 2
