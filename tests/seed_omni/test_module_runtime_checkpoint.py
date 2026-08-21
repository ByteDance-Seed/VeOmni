"""``ModuleRuntime.skip_hf_weight_load`` — the frozen-module resume carve-out.

A full non-LoRA resume normally skips the initial HF weight materialization to
avoid a second memory peak. A fully-frozen OmniModule gets no checkpoint manager
(``ModuleRuntime._init_checkpoint``), so nothing would ever restore its weights —
it has to veto the skip and load the released HF ones.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from veomni.distributed.torch_compile import CompileConfig
from veomni.models.seed_omni.accelerator.module_runtime import ModuleRuntime
from veomni.trainer.base import BaseTrainer


_RESUME_PATH = "/tmp/checkpoint/global_step_10"


def _build_module_runtime(model: nn.Module, *, load_path: str | None = _RESUME_PATH) -> ModuleRuntime:
    runtime = ModuleRuntime.__new__(ModuleRuntime)
    runtime.model = model
    runtime.module_name = "test_module"
    runtime.args = SimpleNamespace(model_path="/tmp/hf-model", lora_config=None)
    runtime.train = SimpleNamespace(checkpoint=SimpleNamespace(load_path=load_path))
    runtime._has_trainable_parameters = None
    return runtime


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (nn.Linear(2, 2), True),
        (nn.Module(), True),
    ],
)
def test_module_allows_skip_hf_weight_load(model: nn.Module, expected: bool) -> None:
    assert _build_module_runtime(model).skip_hf_weight_load is expected


def test_frozen_module_with_persistent_state_must_load_hf_weights() -> None:
    model = nn.Linear(2, 2)
    model.requires_grad_(False)

    assert _build_module_runtime(model).skip_hf_weight_load is False


def test_buffer_only_module_with_persistent_state_must_load_hf_weights() -> None:
    model = nn.Module()
    model.register_buffer("running_scale", torch.ones(1))

    assert _build_module_runtime(model).skip_hf_weight_load is False


def test_without_resume_path_hf_weights_are_always_loaded() -> None:
    """No checkpoint to restore from ⇒ the HF pass is the only weight source."""
    assert _build_module_runtime(nn.Linear(2, 2), load_path=None).skip_hf_weight_load is False


def test_parallelize_forwards_module_skip_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    parallelize = MagicMock(side_effect=lambda model, **kwargs: model)
    monkeypatch.setattr(
        "veomni.models.seed_omni.accelerator.module_runtime.build_parallelize_model",
        parallelize,
    )

    model = nn.Linear(2, 2)
    model.requires_grad_(False)
    runtime = _build_module_runtime(model)
    runtime.args = SimpleNamespace(
        model_path="/tmp/hf-model",
        lora_config=None,
        fqn_to_index_mapping=None,
        basic_modules=[],
        optimizer=SimpleNamespace(type="adamw", muon_expert_zero_comm=False),
        accelerator=SimpleNamespace(
            init_device="meta",
            broadcast_model_weights_from_rank0=False,
            ep_sharded_stream_load=False,
            chunk_mbs_config=SimpleNamespace(enable=False),
            torch_compile=CompileConfig(),
            gradient_checkpointing=SimpleNamespace(enable=False, enable_reentrant=False, early_stop=True),
            fsdp_config=SimpleNamespace(
                reshard_after_forward=True,
                mixed_precision=SimpleNamespace(enable=False),
                forward_prefetch=False,
                offload=False,
                offload_pin_memory=False,
                max_load_broadcast_size=20.0,
            ),
        ),
    )

    runtime._parallelize_module_model(model)

    assert parallelize.call_args.kwargs["should_skip_hf_weight_load"] is False


@pytest.mark.parametrize(
    ("caller_allows_skip", "fallback_allows_skip", "expected"),
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_base_trainer_combines_caller_decision_with_resume_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caller_allows_skip: bool,
    fallback_allows_skip: bool,
    expected: bool,
) -> None:
    fallback = MagicMock(return_value=fallback_allows_skip)
    parallelize = MagicMock(side_effect=lambda model, **kwargs: model)
    monkeypatch.setattr("veomni.trainer.base.should_skip_hf_weight_load", fallback)
    monkeypatch.setattr("veomni.trainer.base.build_parallelize_model", parallelize)

    fsdp_config = SimpleNamespace(
        reshard_after_forward=True,
        mixed_precision=SimpleNamespace(enable=False),
        forward_prefetch=False,
        offload=False,
        offload_pin_memory=False,
        max_load_broadcast_size=20.0,
    )
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = nn.Linear(2, 2)
    trainer.args = SimpleNamespace(
        model=SimpleNamespace(
            model_path="/tmp/hf-model",
            lora_config=None,
            fqn_to_index_mapping=None,
            basic_modules=[],
            optimizer=SimpleNamespace(type="adamw", muon_expert_zero_comm=False),
            accelerator=SimpleNamespace(
                fsdp_config=fsdp_config,
                chunk_mbs_config=SimpleNamespace(enable=False),
                init_device="meta",
                gradient_checkpointing=SimpleNamespace(enable=False, enable_reentrant=False, early_stop=True),
                broadcast_model_weights_from_rank0=False,
                ep_sharded_stream_load=False,
                torch_compile=CompileConfig(),
            ),
        ),
        train=SimpleNamespace(
            checkpoint=SimpleNamespace(load_path=_RESUME_PATH),
        ),
    )

    trainer._build_parallelized_model(skip_hf_weight_load=caller_allows_skip)

    assert parallelize.call_args.kwargs["should_skip_hf_weight_load"] is expected
    assert fallback.call_count == int(caller_allows_skip)
