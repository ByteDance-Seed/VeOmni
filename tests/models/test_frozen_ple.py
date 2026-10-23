from types import SimpleNamespace

import torch
from torch import nn

from veomni.models.frozen_ple import FrozenPLEGuard, is_ple_table, prepare_ple_storage


def _qwen4_model():
    model = nn.Module()
    model.config = type("Config", (), {"model_type": "qwen4_exp"})()
    model.ple = nn.Module()
    model.ple.ple_embedding = nn.Module()
    model.ple.ple_embedding.ngram_embedding = nn.ModuleDict({"shard_0": nn.Embedding(4, 2)})
    model.other = nn.Linear(2, 2)
    return model


def test_is_ple_table_matches_shard_weights_only():
    assert is_ple_table("model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight")
    assert not is_ple_table("model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.bias")
    assert not is_ple_table("other.weight")


def test_guard_excludes_table_from_optimizer_and_rejects_mutation():
    model = _qwen4_model()
    guard = FrozenPLEGuard(model, expected_device="cpu")
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    assert not table.requires_grad
    assert model.other.weight.requires_grad

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    guard.verify(model, optimizer)

    table.requires_grad_(True)
    try:
        guard.verify(model, optimizer)
    except RuntimeError as exc:
        assert "Frozen PLE invariant violated" in str(exc)
    else:
        raise AssertionError("verify should reject a thawed PLE table")


def test_prepare_ple_storage_changes_only_the_table():
    model = _qwen4_model()
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    other = model.other.weight
    table.data = table.data.to(torch.float32)
    other.data = other.data.to(torch.float32)
    prepare_ple_storage(model, "bfloat16")
    assert model.ple.ple_embedding.ngram_embedding["shard_0"].weight.dtype == torch.bfloat16
    assert model.other.weight.dtype == torch.float32


def test_note_restored_accepts_checkpoint_copy_and_rejects_a_later_write():
    model = _qwen4_model()
    guard = FrozenPLEGuard(model, expected_device="cpu")
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    table.copy_(torch.ones_like(table))
    try:
        guard.verify(model)
    except RuntimeError as exc:
        assert "Frozen PLE invariant violated" in str(exc)
    else:
        raise AssertionError("checkpoint copy_ should fail the pre-restore check")

    guard.note_restored(model)
    guard.verify(model)

    table.copy_(torch.zeros_like(table))
    try:
        guard.verify(model)
    except RuntimeError as exc:
        assert "Frozen PLE invariant violated" in str(exc)
    else:
        raise AssertionError("verify should reject a write after resume")


def test_guard_rejects_other_model_types():
    model = nn.Linear(2, 2)
    model.config = type("Config", (), {"model_type": "llama"})()
    try:
        FrozenPLEGuard(model)
    except ValueError as exc:
        assert "qwen4_exp" in str(exc)
    else:
        raise AssertionError("non-qwen4_exp models should be rejected")


def test_training_arguments_validate_frozen_ple_dtype():
    from veomni.arguments.arguments_types import TrainingArguments
    from veomni.trainer.vlm_trainer import VLMTrainingArguments

    enabled = TrainingArguments(freeze_ple_embeddings=True, frozen_ple_dtype="bfloat16")
    assert enabled.freeze_ple_embeddings is True
    assert enabled.frozen_ple_dtype == "bfloat16"
    assert VLMTrainingArguments(freeze_ple_embeddings=True).freeze_ple_embeddings is True

    try:
        TrainingArguments(frozen_ple_dtype="bfloat16")
    except ValueError as exc:
        assert "freeze_ple_embeddings" in str(exc)
    else:
        raise AssertionError("frozen_ple_dtype without freeze_ple_embeddings should be rejected")

    try:
        TrainingArguments(freeze_ple_embeddings=True, frozen_ple_dtype="float16")
    except ValueError as exc:
        assert "bfloat16" in str(exc)
    else:
        raise AssertionError("float16 is not a supported frozen PLE dtype")


def test_runtime_freezes_ple_before_building_the_optimizer(monkeypatch):
    monkeypatch.setattr("veomni.utils.device.get_device_type", lambda: "cpu")

    def build_sgd(model, **kwargs):
        params = [param for param in model.parameters() if param.requires_grad]
        assert params, "the non-PLE weights should stay trainable"
        return torch.optim.SGD(params, lr=0.1)

    monkeypatch.setattr("veomni.optim.build_optimizer", build_sgd)

    from veomni.models.model_runtime import VeOmniModelRuntime

    model = _qwen4_model()
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    table.data = table.data.to(torch.float32)
    runtime = VeOmniModelRuntime.__new__(VeOmniModelRuntime)
    runtime.model = model
    runtime.train_args = SimpleNamespace(freeze_ple_embeddings=True, frozen_ple_dtype="bfloat16")
    runtime.args = SimpleNamespace(
        optimizer=SimpleNamespace(
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            type="adamw",
            no_decay_modules=[],
            no_decay_params=[],
        )
    )

    runtime._apply_frozen_ple()
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    assert table.dtype == torch.bfloat16
    assert not table.requires_grad

    runtime._build_optimizer()
    optimized = {id(param) for group in runtime.optimizer.param_groups for param in group["params"]}
    assert id(table) not in optimized
    assert id(model.other.weight) in optimized
    runtime.optimizer.step()

    table.copy_(torch.ones_like(table))
    try:
        runtime._resync_frozen_ple_after_load()
        runtime.optimizer.step()
    except RuntimeError as exc:
        raise AssertionError("checkpoint restore should re-arm the frozen PLE check") from exc

    table.copy_(torch.zeros_like(table))
    try:
        runtime.optimizer.step()
    except RuntimeError as exc:
        assert "Frozen PLE invariant violated" in str(exc)
    else:
        raise AssertionError("optimizer.step should reject a write after the checkpoint restore")


def test_runtime_leaves_ple_trainable_when_freeze_is_disabled():
    from veomni.models.model_runtime import VeOmniModelRuntime

    model = _qwen4_model()
    runtime = VeOmniModelRuntime.__new__(VeOmniModelRuntime)
    runtime.model = model
    runtime.train_args = SimpleNamespace(freeze_ple_embeddings=False, frozen_ple_dtype=None)
    runtime._apply_frozen_ple()
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    assert table.requires_grad
    assert runtime._frozen_ple_guard is None
