"""Fixed DiT microbatches through the existing condition, model and saver paths."""

import copy
import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset
from torch import nn
from transformers.modeling_outputs import ModelOutput

import veomni.data.data_loader as loader_module
import veomni.models.model_runtime as runtime_module
import veomni.trainer.base as base_module
import veomni.trainer.dit_trainer as dit_module
from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.dataset import MappingDataset
from veomni.trainer.base import BaseTrainer
from veomni.trainer.dit_trainer import (
    DiTDataArguments,
    DiTDataCollator,
    DiTModelArguments,
    DiTModelRuntime,
    DiTTrainer,
    DiTTrainingArguments,
    OfflineEmbeddingSaver,
    VeOmniDiTArguments,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _trainer(monkeypatch, task, micro_batch_size, dp_size=1, dp_rank=0, shuffle=True):
    monkeypatch.setenv("WORLD_SIZE", str(dp_size))
    state = SimpleNamespace(dp_size=dp_size, dp_rank=dp_rank, sp_size=1, sp_enabled=False)
    monkeypatch.setattr("veomni.distributed.parallel_state.get_parallel_state", lambda: state)
    monkeypatch.setattr(dit_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr(loader_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr(runtime_module, "get_parallel_state_by_name", lambda name: state)
    monkeypatch.setattr(dit_module, "use_parallel_state", lambda state: nullcontext())
    args = VeOmniDiTArguments(
        model=DiTModelArguments(config_path="unused"),
        data=DiTDataArguments(train_path="unused", datasets_type="mapping"),
        train=DiTTrainingArguments(
            training_task=task,
            micro_batch_size=micro_batch_size,
            global_batch_size=4 * dp_size,
        ),
    )
    args.data.dataloader.num_workers = 0
    args.data.dataloader.prefetch_factor = None
    args.data.dataloader.pin_memory = False
    args.data.dataloader.shuffle = shuffle
    args._train_steps = 1
    trainer = DiTTrainer.__new__(DiTTrainer)
    trainer.base = SimpleNamespace(args=args, _setup=Mock(return_value=torch.device("cpu")), LOG_SAMPLE=False)
    trainer._setup()
    trainer.base._setup.assert_called_once_with(args)
    assert trainer.base.device == torch.device("cpu")
    return trainer


def _load(trainer, samples):
    trainer.base.train_dataset = [(sample,) for sample in samples]
    trainer._build_dataloader()
    return iter(trainer.base.train_dataloader)


@pytest.mark.parametrize("dataloader_type", ["native", "custom"])
def test_dit_forwards_replay_timeout_to_native_loader(monkeypatch, dataloader_type):
    trainer = _trainer(monkeypatch, "online_training", 1)
    trainer.base.args.data.dataloader.type = dataloader_type
    trainer.base.args.data.resume_replay_timeout_seconds = 37
    trainer.base.train_dataset = []
    build = Mock(return_value=object())
    monkeypatch.setattr(dit_module, "build_dataloader", build)
    trainer._build_dataloader()
    kwargs = build.call_args.kwargs
    if dataloader_type == "native":
        assert kwargs["resume_replay_timeout_seconds"] == 37
    else:
        assert "resume_replay_timeout_seconds" not in kwargs


@pytest.mark.parametrize("task", ["offline_training", "online_training"])
@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_setup_and_loader_honor_fixed_microbatches(monkeypatch, task, micro_batch_size, dp_size):
    trainer = _trainer(monkeypatch, task, micro_batch_size, dp_size)
    train = trainer.base.args.train
    assert train.micro_batch_size == micro_batch_size
    assert train.dyn_bsz is False
    assert train.dataloader_batch_size == 4
    assert train.gradient_accumulation_steps == 4 // micro_batch_size
    batches = next(_load(trainer, [{"value": torch.ones(i + 1)} for i in range(4 * dp_size)]))
    assert len(batches) == 4 // micro_batch_size
    assert all(len(batch["value"]) == micro_batch_size for batch in batches)
    assert all(isinstance(batch["value"], list) for batch in batches)


@pytest.mark.parametrize("task", ["offline_training", "online_training", "offline_embedding"])
@pytest.mark.parametrize("shuffle", [True, False])
def test_dataloader_shuffle_reaches_sampler(monkeypatch, task, shuffle):
    trainer = _trainer(monkeypatch, task, micro_batch_size=1, shuffle=shuffle)
    expected = shuffle and task != "offline_embedding"
    assert trainer.base.args.data.dataloader.shuffle is expected
    _load(trainer, [{"value": torch.ones(1)} for _ in range(4)])
    assert trainer.base.train_dataloader.sampler.shuffle is expected


@pytest.mark.parametrize("task", ["offline_training", "online_training", "offline_embedding"])
def test_offline_embedding_keeps_the_last_batch(monkeypatch, task):
    trainer = _trainer(monkeypatch, task, micro_batch_size=1)
    expected = task != "offline_embedding"
    assert trainer.base.saves_model_state is expected
    assert trainer.base.args.data.dataloader.drop_last is expected
    assert not hasattr(trainer.base.args.data, "drop_last")
    _load(trainer, [{"value": torch.ones(1)} for _ in range(4)])
    assert trainer.base.train_dataloader.drop_last is expected


def test_dit_registers_offline_writer_before_callbacks(monkeypatch, tmp_path):
    configured = _trainer(monkeypatch, "offline_embedding", micro_batch_size=1)
    args = configured.base.args
    writer = OfflineEmbeddingSaver(str(tmp_path), dataset_length=3)
    seen = []
    monkeypatch.setattr(BaseTrainer, "_setup", lambda self, args: torch.device("cpu"))
    monkeypatch.setattr(BaseTrainer, "_init_callbacks", lambda self: seen.append(self.offline_cache_writer))
    monkeypatch.setattr(DiTTrainer, "_build_model_runtime", lambda self: SimpleNamespace(parallel_state=object()))
    monkeypatch.setattr(DiTTrainer, "_build_data_transform", lambda self: None)
    monkeypatch.setattr(DiTTrainer, "_build_dataset", lambda self: setattr(self, "offline_embedding_saver", writer))
    monkeypatch.setattr(DiTTrainer, "_build_dataloader", lambda self: None)
    trainer = DiTTrainer(args)
    assert seen == [writer]
    assert trainer.base.offline_cache_writer is writer
    assert trainer.base.saves_model_state is False


def test_offline_embedding_checkpoint_restores_output_writer(monkeypatch, tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
    from veomni.checkpoint import layout
    from veomni.models.diffusers.hunyuanvideo15.hunyuanvideo15_condition.conditioning import FlowMatchingConditioner
    from veomni.trainer.callbacks.base import TrainerState
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    dit = _trainer(monkeypatch, "offline_embedding", micro_batch_size=1)
    output = tmp_path / "output"
    writer = OfflineEmbeddingSaver(str(output), dataset_length=3)
    for value in (0, 1):
        writer.save({"value": torch.tensor(value)})
    assert writer.buffer and not list(output.glob("*.parquet"))

    checkpoint_root = str(tmp_path / "ckpt")
    trainer = _make_mock_trainer(save_path=checkpoint_root)
    trainer.args.train.training_task = "offline_embedding"
    trainer.saves_model_state = dit.base.saves_model_state
    trainer.model = DiTModelRuntime.__new__(DiTModelRuntime)
    trainer.model.condition_model = FlowMatchingConditioner(seed=42, device="cpu")
    trainer.offline_embedding_runtime = trainer.model
    posterior = torch.zeros(1, 2, 2, 2)
    trainer.model.condition_model.sample_posterior(posterior)
    trainer.train_dataloader = None
    trainer.data_iterator = None
    trainer.offline_cache_writer = writer
    trainer.environ_meter.state_dict.return_value = {}
    trainer.channel_loss_callback = None
    GlobalStateCallback(trainer).save_global_state(TrainerState(global_step=2))
    expected_condition = trainer.model.condition_model.sample_posterior(posterior)

    step = layout.step_dir(checkpoint_root, 2)
    assert layout.checkpoint_is_complete(step)
    assert layout.read_manifest(step)["model_state"] is False
    assert not writer.buffer and writer.index == 1

    resumed = _make_mock_trainer(save_path=checkpoint_root)
    resumed.args.train.training_task = "offline_embedding"
    resumed.args.train.checkpoint.load_path = step
    resumed.model = DiTModelRuntime.__new__(DiTModelRuntime)
    resumed.model.condition_model = FlowMatchingConditioner(seed=42, device="cpu")
    resumed.offline_embedding_runtime = resumed.model
    resumed.train_dataloader = None
    resumed.data_iterator = None
    resumed.channel_loss_callback = None
    resumed.offline_cache_writer = OfflineEmbeddingSaver(str(output), dataset_length=3)
    GlobalStateCallback(resumed).load_global_state()
    assert resumed.state.global_step == 2
    torch.testing.assert_close(
        resumed.model.condition_model.sample_posterior(posterior), expected_condition, rtol=0, atol=0
    )
    resumed.offline_cache_writer.save({"value": torch.tensor(2)})
    resumed.offline_cache_writer.save_last()

    result = []
    for path in sorted(output.glob("*.parquet")):
        result.extend(int(pickle.loads(row["value"])) for row in Dataset.from_parquet(str(path)))
    assert result == [0, 1, 2]


def test_offline_embedding_rejects_uncheckpointable_private_rng():
    runtime = DiTModelRuntime.__new__(DiTModelRuntime)
    runtime.condition_model = SimpleNamespace(generator=torch.Generator())
    with pytest.raises(RuntimeError, match="cannot save and restore its private RNG"):
        runtime.offline_condition_state()


def test_offline_embedding_resume_rejects_unproven_or_extra_output(monkeypatch, tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
    from veomni.checkpoint import layout
    from veomni.trainer.callbacks.base import TrainerState
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    _trainer(monkeypatch, "offline_embedding", micro_batch_size=1)
    output = tmp_path / "output"
    writer = OfflineEmbeddingSaver(str(output), dataset_length=3)
    writer.save({"value": torch.tensor(0)})
    root = str(tmp_path / "ckpt")
    trainer = _make_mock_trainer(save_path=root)
    trainer.args.train.training_task = "offline_embedding"
    trainer.saves_model_state = False
    trainer.model = SimpleNamespace(optimizer=None, checkpoint=None)
    trainer.train_dataloader = None
    trainer.data_iterator = None
    trainer.offline_cache_writer = writer
    trainer.environ_meter.state_dict.return_value = {}
    trainer.channel_loss_callback = None
    GlobalStateCallback(trainer).save_global_state(TrainerState(global_step=1))
    step = layout.step_dir(root, 1)

    resumed = _make_mock_trainer(save_path=root)
    resumed.args.train.training_task = "offline_embedding"
    resumed.args.train.checkpoint.load_path = step
    resumed.model = SimpleNamespace(optimizer=None, checkpoint=None)
    resumed.train_dataloader = None
    resumed.data_iterator = None
    resumed.offline_cache_writer = OfflineEmbeddingSaver(str(output), dataset_length=3)
    (output / "rank_0_shard_1.parquet").write_bytes(b"unexpected later output")
    with pytest.raises(ValueError, match="output shards differ"):
        GlobalStateCallback(resumed).load_global_state()
    assert resumed.state.global_step == 0

    (output / "rank_0_shard_1.parquet").unlink()
    for extra in ("rank_0_shard_00.parquet", "rank_0_shard_extra.parquet"):
        (output / extra).write_bytes(b"unexpected output")
        with pytest.raises(ValueError, match="unexpected shard name"):
            GlobalStateCallback(resumed).load_global_state()
        assert resumed.state.global_step == 0
        (output / extra).unlink()
    path = layout.extra_state_path(step, 0)
    state = torch.load(path, weights_only=False)
    state.pop("offline_cache_writer")
    torch.save(state, path)
    with pytest.raises(ValueError, match="missing its output writer state"):
        GlobalStateCallback(resumed).load_global_state()
    assert resumed.state.global_step == 0


def _offline_embedding_writer_failure_worker(rank, source, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_mode="fsdp2", device_type="cpu")
        trainer = _make_mock_trainer()
        trainer.args.train.training_task = "offline_embedding"
        trainer.args.train.global_rank = rank
        trainer.args.train.world_size = 2
        trainer.args.train.checkpoint.load_path = source
        trainer.model = SimpleNamespace(optimizer=None, checkpoint=None)
        trainer.train_dataloader = None
        trainer.data_iterator = None
        trainer.channel_loss_callback = None

        def fail_restore(_state):
            raise OSError("writer output became unreadable")

        trainer.offline_cache_writer = (
            SimpleNamespace(validate_state_dict=lambda _state: None, load_state_dict=fail_restore)
            if rank == 0
            else None
        )
        callback = GlobalStateCallback(trainer)
        callback._read_current = lambda _path: {
            "global_step": 2,
            "train_dataloader": None,
            "offline_cache_writer": {"version": 1} if rank == 0 else None,
            "torch_rng_state": torch.get_rng_state(),
        }
        with pytest.raises((OSError, RuntimeError)) as failure:
            callback.load_global_state()
        assert ("writer output became unreadable" if rank == 0 else "restoring offline-cache state") in str(
            failure.value
        )
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_offline_embedding_writer_failure_is_coordinated(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous
    from veomni.checkpoint import layout

    layout.write_manifest(str(tmp_path), global_step=2, world_size=2, model_state=False)
    spawn_with_rendezvous(_offline_embedding_writer_failure_worker, args=(str(tmp_path),), nprocs=2)


@pytest.mark.parametrize("shuffle", [True, False])
def test_dataloader_shuffle_reaches_base_trainer_builders(monkeypatch, shuffle):
    trainer = _trainer(monkeypatch, "offline_training", micro_batch_size=1, shuffle=shuffle)
    built = {}

    def build_dataset(**kwargs):
        built.update(kwargs)
        return list(range(4))

    monkeypatch.setattr(base_module, "build_dataset", build_dataset)
    base = SimpleNamespace(args=trainer.base.args, data_transform=None)
    BaseTrainer._build_dataset(base)
    assert built["shuffle"] is shuffle

    base.train_dataset = [({"value": torch.ones(1)},) for _ in range(4)]
    base.collate_fn = DiTDataCollator()
    BaseTrainer._build_dataloader(base)
    assert base.train_dataloader.sampler.shuffle is shuffle


class _Condition:
    def __init__(self):
        self.encoded = []
        self.processed = []

    def get_condition(self, *, value):
        assert not torch.is_grad_enabled()
        self.encoded.append(len(value))
        return {"value": [x + 1 for x in value]}

    def process_condition(self, *, value):
        assert not torch.is_grad_enabled()
        self.processed.append(len(value))
        return {"value": value}


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, *, value):
        losses = [(x * self.weight).square().mean() for x in value]
        return ModelOutput(loss={"mse": torch.stack(losses).mean()})


@pytest.mark.parametrize("task", ["offline_training", "online_training"])
@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
def test_train_step_matches_equal_sample_adamw_update(monkeypatch, task, micro_batch_size):
    trainer = _trainer(monkeypatch, task, micro_batch_size)
    model = _Model()
    reference = copy.deepcopy(model)
    condition = _Condition()
    runtime = DiTModelRuntime.__new__(DiTModelRuntime)
    runtime.model_name = "base"
    runtime.model = model
    runtime.condition_model = condition
    runtime.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    runtime.lr_scheduler = torch.optim.lr_scheduler.StepLR(runtime.optimizer, step_size=1, gamma=0.5)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    gradients = []

    def clip():
        gradients.append(model.weight.grad.clone())
        return model.weight.grad.norm()

    runtime.clip_grad_norm = clip
    trainer.base.model = runtime
    trainer.base.state = SimpleNamespace(global_step=0)
    trainer.base.model_fwd_context = nullcontext()
    trainer.base.model_bwd_context = nullcontext()
    for name in (
        "on_step_begin",
        "on_step_end",
        "sync_before_train_step",
        "_reset_async_activation_offload_if_enabled",
        "model_reshard",
        "_configure_hsdp_allreduce",
    ):
        setattr(trainer.base, name, Mock())
    values = [torch.arange(n, dtype=torch.float32) + i for i, n in enumerate((1, 3, 2, 5))]
    root_calls = []
    handle = model.register_forward_pre_hook(lambda module, args: root_calls.append(1))
    trainer.train_step(_load(trainer, [{"value": x} for x in values]))
    handle.remove()
    expected_values = [x + 1 for x in values] if task == "online_training" else values
    expected_loss = reference(value=expected_values).loss["mse"]
    expected_loss.backward()
    torch.testing.assert_close(gradients[0], reference.weight.grad)
    reference_optimizer.step()
    torch.testing.assert_close(model.weight, reference.weight)
    for key, value in reference_optimizer.state[reference.weight].items():
        torch.testing.assert_close(runtime.optimizer.state[model.weight][key], value)
    k = 4 // micro_batch_size
    assert len(root_calls) == k
    assert condition.processed == [micro_batch_size] * k
    assert condition.encoded == ([micro_batch_size] * k if task == "online_training" else [])
    assert trainer.base.state.global_step == 1
    assert runtime.lr_scheduler.last_epoch == 1
    assert runtime.optimizer.param_groups[0]["lr"] == 0.005
    assert model.weight.grad is None
    torch.testing.assert_close(torch.tensor(trainer.base.on_step_end.call_args.kwargs["loss"]), expected_loss.detach())


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_embedding_keeps_one_microbatch_and_saves_each_sample(monkeypatch, tmp_path, micro_batch_size, dp_size):
    trainer = _trainer(monkeypatch, "offline_embedding", micro_batch_size, dp_size)
    train = trainer.base.args.train
    assert train.micro_batch_size == micro_batch_size
    assert train.global_batch_size == micro_batch_size * dp_size
    assert train.dataloader_batch_size == micro_batch_size
    assert train.gradient_accumulation_steps == 1
    condition = _Condition()
    trainer.base.model = SimpleNamespace(condition_model=condition)
    trainer.offline_embedding_saver = OfflineEmbeddingSaver(str(tmp_path), dataset_length=micro_batch_size)
    batches = next(_load(trainer, [{"value": torch.tensor([float(i)])} for i in range(micro_batch_size * dp_size)]))
    assert len(batches) == 1
    expected = [x + 1 for x in batches[0]["value"]]
    assert trainer.forward_backward_step(batches[0]) == (0.0, {})
    trainer.offline_embedding_saver.save_last()
    saved = Dataset.from_parquet(str(tmp_path / "rank_0_shard_0.parquet"))
    assert len(saved) == micro_batch_size
    for row, value in zip(saved, expected):
        torch.testing.assert_close(pickle.loads(row["value"]), value)
    assert condition.encoded == [micro_batch_size]
    assert condition.processed == []


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_embedding_tail_padding_saves_original_samples_exactly_once(monkeypatch, tmp_path, micro_batch_size, dp_size):
    saved_ids = []
    for rank in range(dp_size):
        trainer = _trainer(monkeypatch, "offline_embedding", micro_batch_size, dp_size, rank)
        dataset = MappingDataset([{"value": torch.tensor([float(i)])} for i in range(5)], transform=lambda row: [row])
        trainer.base.train_dataset = dataset
        trainer.base._build_dataset = Mock()
        trainer.offline_embedding_save_dir = str(tmp_path)
        trainer.base.model = SimpleNamespace(condition_model=_Condition())
        trainer._build_dataset()
        trainer._build_dataloader()
        assert len(trainer.base.train_dataloader) == trainer.base.args.train_steps
        for micro_batches in trainer.base.train_dataloader:
            assert len(micro_batches) == 1
            trainer.forward_backward_step(micro_batches[0])
        trainer.offline_embedding_saver.save_last()
        rows = Dataset.from_parquet(str(tmp_path / f"rank_{rank}_shard_0.parquet"))
        saved_ids.extend(int(pickle.loads(row["value"]).item()) - 1 for row in rows)
    assert sorted(saved_ids) == list(range(5))


def test_collator_aligns_missing_keys_with_none():
    single = DiTDataCollator()([{"a": 1, "b": 2}])
    assert dict(single) == {"a": [1], "b": [2]}
    mixed = DiTDataCollator()([{"a": 1, "x": "fl2va"}, {"a": 2, "y": "ref2va"}, {"a": 3}])
    assert dict(mixed) == {"a": [1, 2, 3], "x": ["fl2va", None, None], "y": [None, "ref2va", None]}


@pytest.mark.parametrize("offload", [False, True])
def test_fused_dit_branch_reloads_weights_and_optimizer_under_its_model_state(monkeypatch, offload):
    from contextlib import contextmanager

    import veomni.models.checkpoint.weights as weights_module
    from tests.train_scripts import train_dit_test

    runtime = train_dit_test.TestDiTModelRuntime.__new__(train_dit_test.TestDiTModelRuntime)
    runtime.args = SimpleNamespace(
        accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(offload=offload)), model_path="initial"
    )
    runtime.train_args = SimpleNamespace(seed=42, enable_full_determinism=False)
    runtime.model_name = "second-model"
    runtime.model = nn.Linear(2, 2)
    state = object()
    active_state = []
    monkeypatch.setattr(
        runtime_module, "get_parallel_state_by_name", lambda name: state if name == runtime.model_name else None
    )

    @contextmanager
    def model_scope(current):
        assert current is state
        active_state.append(current)
        try:
            yield
        finally:
            active_state.pop()

    loaded = []

    def load(model, path, **kwargs):
        assert active_state == [state]
        assert model is runtime.model and path == "initial"
        loaded.append(kwargs)

    def optimizer():
        assert active_state == [state]
        assert len(loaded) == 1

    monkeypatch.setattr(train_dit_test, "use_parallel_state", model_scope)
    monkeypatch.setattr(weights_module, "load_model_weights", load)
    monkeypatch.setattr(runtime, "_build_optimizer", optimizer)
    monkeypatch.setattr("veomni.utils.helper.set_seed", lambda *args: None)
    model_device = "test_accelerator"
    monkeypatch.setattr("veomni.utils.device.get_device_type", lambda: model_device)
    runtime.reset_for_dataset_branch()
    assert not active_state
    assert loaded[0]["init_device"] == ("cpu" if offload else model_device)
    assert loaded[0]["dtensor_to_cpu"] is offload


def test_fused_dit_branch_resizes_schedule_and_runs_full_training_lifecycle(monkeypatch):
    from tests.train_scripts import train_dit_test
    from veomni.trainer.callbacks import TrainerState

    trainer = train_dit_test.TestDiTTrainer.__new__(train_dit_test.TestDiTTrainer)
    train_args = SimpleNamespace(
        checkpoint=SimpleNamespace(output_dir="first"),
        num_train_epochs=1,
        local_rank=0,
    )
    data_args = SimpleNamespace(train_path="first", dataloader=SimpleNamespace(use_background_prefetcher=False))
    args = SimpleNamespace(train=train_args, data=data_args, train_steps=1)
    model = SimpleNamespace(parallel_state=object(), reset_for_dataset_branch=Mock())
    events = []
    trainer.base = SimpleNamespace(
        args=args,
        model=model,
        state=TrainerState(global_step=9),
        start_epoch=1,
        start_step=1,
        _log_callback=SimpleNamespace(log_dict={"loss": [1.0]}),
        train_dataloader=None,
        destroy_distributed=lambda: events.append("destroy"),
    )
    trainer.training_task = "offline_training"
    monkeypatch.setattr(train_dit_test, "use_parallel_state", lambda state: nullcontext())

    def dataset():
        assert args.data.train_path == "second"
        args.train_steps = 3

    schedule = []
    trainer._build_dataset = dataset
    trainer._build_dataloader = lambda: None
    trainer.base._build_lr_scheduler = lambda: schedule.append(args.train_steps)
    trainer.on_train_begin = lambda: events.append("begin")
    trainer.on_epoch_begin = lambda: None
    trainer.train_step = lambda batch: events.append("step")
    trainer.on_epoch_end = lambda: None
    trainer.on_train_end = lambda: events.append("end")
    monkeypatch.setattr(dit_module, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(dit_module.helper, "print_device_mem_info", lambda *args: None)
    trainer._run_extra_train_path("second")
    assert schedule == [3]
    assert events == ["begin", "step", "step", "step", "end", "sync", "destroy"]
    assert trainer.base.state.global_step == 0
    assert trainer.base._log_callback.log_dict == {}
    assert args.train.checkpoint.output_dir == "first_padding"
