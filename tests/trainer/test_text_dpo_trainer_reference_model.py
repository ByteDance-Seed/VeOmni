from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from veomni.trainer import text_dpo_trainer


def _args():
    fsdp_config = SimpleNamespace(
        reshard_after_forward=False,
        forward_prefetch=False,
        offload=False,
        offload_pin_memory=False,
        max_load_broadcast_size=None,
    )
    accelerator = SimpleNamespace(init_device="cpu", fsdp_config=fsdp_config)
    model = SimpleNamespace(
        config_path="config",
        model_path="weights",
        accelerator=accelerator,
        ops_implementation=SimpleNamespace(),
        model_config={},
        basic_modules=[],
        broadcast_model_weights_from_rank0=False,
    )
    return SimpleNamespace(model=model, dpo_config=SimpleNamespace(refer_model_precision="float32"))


def test_build_reference_model_reads_parallel_plan_from_base_model(monkeypatch):
    trainer = text_dpo_trainer.TextDPOTrainer.__new__(text_dpo_trainer.TextDPOTrainer)
    policy = Mock()
    policy.get_parallel_plan.return_value.cpu_load_param_name = "policy_param"
    trainer.base = SimpleNamespace(args=_args(), model=policy)

    reference = Mock(_no_split_modules=[])
    monkeypatch.setattr(text_dpo_trainer, "build_foundation_model", lambda **kwargs: reference)
    parallelize = Mock(return_value=reference)
    monkeypatch.setattr(text_dpo_trainer, "build_parallelize_model", parallelize)
    monkeypatch.setattr(text_dpo_trainer.helper, "print_device_mem_info", lambda *args, **kwargs: None)

    trainer._build_reference_model()

    assert parallelize.call_args.kwargs["cpu_load_param_name"] == "policy_param"
