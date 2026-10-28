"""An exact resume must reconstruct the same next learning rate."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.optim.lr_scheduler import build_lr_scheduler
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback, _validate_file_profile_checkpoint_schedule
from veomni.trainer.callbacks.global_state_callback import _resume_identity


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def _runtime_identity(monkeypatch):
    monkeypatch.setenv("VEOMNI_EXACT_RESUME_RUNTIME_ID", "sha256:" + "a" * 64)


def _trainer():
    optimizer = SimpleNamespace(
        type="qwen38_muon",
        lr=5e-6,
        lr_min=1e-7,
        lr_start=1e-8,
        lr_warmup_ratio=0.1,
        lr_decay_style="cosine",
        lr_decay_ratio=1.0,
        weight_decay=0.0,
        no_decay_modules=[],
        no_decay_params=[],
        betas=(0.9, 0.95),
        muon_momentum=0.95,
        max_grad_norm=0.5,
        grad_clip_scope="per_module",
        qwen38_ple_moments_scratch_dir="/private/fresh",
    )
    args = SimpleNamespace(
        model=SimpleNamespace(
            accelerator=SimpleNamespace(
                fsdp_config=SimpleNamespace(
                    mixed_precision=SimpleNamespace(enable=True, param_dtype="bfloat16", reduce_dtype="float32"),
                    low_precision_reduce_scatter_comm=False,
                )
            ),
            optimizer=optimizer,
            model_config={"router_aux_loss_coef": 0.0},
            ops_implementation=SimpleNamespace(cross_entropy_loss_implementation="chunk_loss"),
        ),
        train=SimpleNamespace(world_size=16, num_train_epochs=2, loss_chunk_size=512),
        train_steps=100,
        data=SimpleNamespace(dataloader=SimpleNamespace()),
    )
    contract = SimpleNamespace(
        install=lambda: None,
        validate_cursor=lambda *args, **kwargs: None,
        validate_before_checkpoint=lambda *args: None,
        checkpoint_identity=lambda: "test-provider-v1",
    )
    return SimpleNamespace(
        args=args,
        model=SimpleNamespace(model_config={"router_aux_loss_coef": 0.0}),
        train_dataloader=SimpleNamespace(_veomni_exact_checkpoint_contract=contract),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr", 1e-5),
        ("lr_min", 2e-7),
        ("lr_start", 2e-8),
        ("lr_warmup_ratio", 0.2),
        ("lr_decay_style", "constant"),
        ("lr_decay_ratio", 0.9),
        ("weight_decay", 0.1),
        ("no_decay_params", ["A_log"]),
        ("betas", (0.9, 0.999)),
        ("muon_momentum", 0.9),
        ("max_grad_norm", 1.0),
        ("grad_clip_scope", "global"),
    ],
)
def test_resume_identity_rejects_changed_update_semantics(field, value):
    trainer = _trainer()
    reference = _resume_identity(trainer)
    setattr(trainer.args.model.optimizer, field, value)
    assert _resume_identity(trainer) != reference


def test_resume_identity_binds_epoch_count_but_not_scratch_path():
    trainer = _trainer()
    reference = _resume_identity(trainer)
    trainer.args.model.optimizer.qwen38_ple_moments_scratch_dir = "/private/resume"
    assert _resume_identity(trainer) == reference
    trainer.args.train.num_train_epochs = 3
    assert _resume_identity(trainer) != reference


def test_resume_identity_binds_attested_runtime(monkeypatch):
    trainer = _trainer()
    reference = _resume_identity(trainer)
    monkeypatch.setenv("VEOMNI_EXACT_RESUME_RUNTIME_ID", "sha256:" + "b" * 64)
    assert _resume_identity(trainer) != reference
    monkeypatch.delenv("VEOMNI_EXACT_RESUME_RUNTIME_ID")
    with pytest.raises(RuntimeError, match="runtime/source identity"):
        _resume_identity(trainer)


def test_resume_identity_rejects_unvalidated_multisource_dpo():
    trainer = _trainer()
    trainer.args.data.data_type = "dpo"
    with pytest.raises(RuntimeError, match="preference/DPO"):
        _resume_identity(trainer)


@pytest.mark.parametrize("case", ["epoch_save", "final_hf", "step_at_epoch", "second_epoch_boundary"])
def test_file_profile_rejects_unsupported_checkpoint_schedule_before_training(case):
    trainer = _trainer()
    trainer.args.train.checkpoint = SimpleNamespace(save_epochs=0, save_hf_weights=False, save_steps=31)
    if case == "epoch_save":
        trainer.args.train.checkpoint.save_epochs = 1
    elif case == "final_hf":
        trainer.args.train.checkpoint.save_hf_weights = True
    elif case == "step_at_epoch":
        trainer.args.train.checkpoint.save_steps = 20
    else:
        trainer.args.train.checkpoint.save_steps = 200
    with pytest.raises(ValueError, match="File-backed"):
        _validate_file_profile_checkpoint_schedule(trainer)


def test_file_profile_accepts_only_interior_step_saves():
    trainer = _trainer()
    trainer.args.train.checkpoint = SimpleNamespace(save_epochs=0, save_hf_weights=False, save_steps=31)
    _validate_file_profile_checkpoint_schedule(trainer)


def test_file_profile_checks_epoch_before_writing_model_checkpoint(monkeypatch):
    from veomni.trainer.callbacks import global_state_callback

    trainer = _trainer()
    trainer.args.train.checkpoint = SimpleNamespace(
        save_epochs=0, save_hf_weights=False, save_steps=31, hf_save_steps=0, hf_save_epochs=0
    )
    calls = []
    trainer.save_dcp = lambda state: calls.append(state)
    monkeypatch.setattr(global_state_callback, "_has_file_profile", lambda _: True)
    monkeypatch.setattr(global_state_callback, "validate_resumed_prefetch_is_drained", lambda *args: None)
    callback = CheckpointCallback(trainer)
    with pytest.raises(RuntimeError, match="unvalidated epoch"):
        callback._save_dcp(SimpleNamespace(global_step=100, epoch=0))
    assert not calls


@pytest.mark.parametrize(
    "case",
    [
        "reduce",
        "low_precision_reduce",
        "override",
        "resolved",
        "ops",
        "loss_chunk",
        "batch_invariant",
        "determinism",
        "provider_identity",
        "fsdp_mode",
        "fsdp_scope",
        "gradient_checkpointing",
        "torch_compile",
        "ulysses_async",
        "additional_optimizer_field",
        "additional_fsdp_field",
    ],
)
def test_resume_identity_rejects_changed_model_or_precision_semantics(case):
    trainer = _trainer()
    reference = _resume_identity(trainer)
    if case == "reduce":
        trainer.args.model.accelerator.fsdp_config.mixed_precision.reduce_dtype = "bfloat16"
    elif case == "low_precision_reduce":
        trainer.args.model.accelerator.fsdp_config.low_precision_reduce_scatter_comm = True
    elif case == "override":
        trainer.args.model.model_config["router_aux_loss_coef"] = 0.1
    elif case == "resolved":
        trainer.model.model_config["router_aux_loss_coef"] = 0.1
    elif case == "ops":
        trainer.args.model.ops_implementation.cross_entropy_loss_implementation = "eager"
    elif case == "loss_chunk":
        trainer.args.train.loss_chunk_size = 1024
    elif case == "batch_invariant":
        trainer.args.train.enable_batch_invariant_mode = True
    elif case == "determinism":
        trainer.args.train.enable_full_determinism = True
    elif case == "fsdp_mode":
        trainer.args.model.accelerator.fsdp_config.fsdp_mode = "ddp"
    elif case == "fsdp_scope":
        trainer.args.model.accelerator.fsdp_config.fsdp_scope = "model"
    elif case == "gradient_checkpointing":
        trainer.args.model.accelerator.gradient_checkpointing = SimpleNamespace(enable=True, enable_reentrant=True)
    elif case == "torch_compile":
        trainer.args.model.accelerator.torch_compile = SimpleNamespace(enable=True)
    elif case == "ulysses_async":
        trainer.args.model.accelerator.enable_async = True
    elif case == "additional_optimizer_field":
        trainer.args.model.optimizer.custom_update_setting = 1
    elif case == "additional_fsdp_field":
        trainer.args.model.accelerator.fsdp_config.other_setting = True
    else:
        trainer.train_dataloader._veomni_exact_checkpoint_contract.checkpoint_identity = lambda: "test-provider-v2"
    assert _resume_identity(trainer) != reference


def test_same_scheduler_configuration_restores_next_lr():
    def build():
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=5e-6)
        scheduler = build_lr_scheduler(
            optimizer,
            train_steps=200,
            lr=5e-6,
            lr_min=1e-7,
            lr_start=1e-8,
            lr_warmup_ratio=0.1,
            lr_decay_style="cosine",
        )
        return optimizer, scheduler

    optimizer, scheduler = build()
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    saved_optimizer = deepcopy(optimizer.state_dict())
    saved_scheduler = deepcopy(scheduler.state_dict())
    optimizer.step()
    scheduler.step()
    expected = scheduler.get_last_lr()

    restored_optimizer, restored_scheduler = build()
    restored_optimizer.load_state_dict(saved_optimizer)
    restored_scheduler.load_state_dict(saved_scheduler)
    restored_optimizer.step()
    restored_scheduler.step()
    assert restored_scheduler.get_last_lr() == expected
