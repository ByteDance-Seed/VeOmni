from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from veomni.distributed.torch_compile import CompileConfig
from veomni.trainer.base import BaseTrainer
from veomni.trainer.omni.omni_module_trainer import OmniModuleTrainer


def _build_module_trainer(
    model: nn.Module,
) -> OmniModuleTrainer:
    trainer = OmniModuleTrainer.__new__(OmniModuleTrainer)
    trainer.base = SimpleNamespace(
        model=model,
        args=SimpleNamespace(
            model=SimpleNamespace(
                model_path="/tmp/hf-model",
            ),
        ),
    )
    trainer.module_name = "test_module"
    trainer._has_trainable_parameters = None
    return trainer


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (nn.Linear(2, 2), True),
        (nn.Module(), True),
    ],
)
def test_module_allows_skip_hf_weight_load(
    model: nn.Module,
    expected: bool,
) -> None:
    trainer = _build_module_trainer(model)

    assert trainer.skip_hf_weight_load is expected


def test_frozen_module_with_persistent_state_must_load_hf_weights() -> None:
    model = nn.Linear(2, 2)
    model.requires_grad_(False)
    trainer = _build_module_trainer(model)

    assert trainer.skip_hf_weight_load is False


def test_buffer_only_module_with_persistent_state_must_load_hf_weights() -> None:
    model = nn.Module()
    model.register_buffer("running_scale", torch.ones(1))
    trainer = _build_module_trainer(model)

    assert trainer.skip_hf_weight_load is False


def test_build_parallelized_model_forwards_module_skip_decision() -> None:
    model = nn.Linear(2, 2)
    model.requires_grad_(False)
    trainer = _build_module_trainer(model)
    trainer.base._build_parallelized_model = MagicMock()

    trainer._build_parallelized_model()

    trainer.base._build_parallelized_model.assert_called_once_with(skip_hf_weight_load=False)


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
        ),
        train=SimpleNamespace(
            checkpoint=SimpleNamespace(load_path="/tmp/checkpoint/global_step_10"),
            optimizer=SimpleNamespace(type="adamw", muon_expert_zero_comm=False),
            chunk_mbs_config=SimpleNamespace(enable=False),
            init_device="meta",
            accelerator=SimpleNamespace(fsdp_config=fsdp_config),
            gradient_checkpointing=SimpleNamespace(enable=False, enable_reentrant=False, early_stop=True),
            broadcast_model_weights_from_rank0=False,
            ep_sharded_stream_load=False,
            torch_compile=CompileConfig(),
        ),
    )

    trainer._build_parallelized_model(skip_hf_weight_load=caller_allows_skip)

    assert parallelize.call_args.kwargs["should_skip_hf_weight_load"] is expected
    assert fallback.call_count == int(caller_allows_skip)
