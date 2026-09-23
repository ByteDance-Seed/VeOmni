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
import veomni.trainer.dit_trainer as dit_module
from veomni.data.dataset import MappingDataset
from veomni.trainer.dit_trainer import (
    DiTDataArguments,
    DiTModelArguments,
    DiTModelRuntime,
    DiTTrainer,
    DiTTrainingArguments,
    OfflineEmbeddingSaver,
    VeOmniDiTArguments,
)


def _trainer(monkeypatch, task, micro_batch_size, dp_size=1, dp_rank=0):
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
