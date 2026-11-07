"""Muon+Adam checkpoints must survive same-step export deduplication."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime, _make_mock_trainer
from tests.optim.test_qwen38_muon_device import _build
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.models.checkpoint.manager import ModelCheckpointManager
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_muon_adam_export_epoch_reuses_dcp_and_resumes_update(tmp_path):
    def build():
        runtime = _make_mock_runtime(save_path=str(tmp_path))
        runtime.model, _, runtime.optimizer = _build(torch.device("cpu"), torch.bfloat16)
        runtime.lr_scheduler = None
        runtime.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
        return runtime

    def set_gradients(model, phase):
        for index, param in enumerate(model.parameters()):
            values = torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape) - 5 + index
            if phase == 2:
                values = values.flip(-1) * 0.37 + torch.eye(4)
            param.grad = (values / 16).to(param.dtype)

    def assert_equal(left_runtime, right_runtime):
        left_leaf = left_runtime.optimizer.optimizers_dict["mixed"]
        right_leaf = right_runtime.optimizer.optimizers_dict["mixed"]
        for left, right in zip(left_runtime.model.parameters(), right_runtime.model.parameters(), strict=True):
            assert id(left) in left_leaf._offloaded and id(right) in right_leaf._offloaded
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            assert left_leaf.state[left].keys() == right_leaf.state[right].keys()
            assert {"step", "master_weight", "exp_avg", "exp_avg_sq"} <= left_leaf.state[left].keys()
            for key in left_leaf.state[left]:
                actual, expected = right_leaf.state[right][key], left_leaf.state[left][key]
                if torch.is_tensor(expected):
                    assert actual.dtype == expected.dtype == torch.float32
                    assert actual.device.type == expected.device.type == "cpu"
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                else:
                    assert actual == expected

    source = build()
    leaf = source.optimizer.optimizers_dict["mixed"]
    groups = leaf.param_groups
    assert {item[0] for group in groups for item in json.loads(group["qwen38_plan"])} == {"muon", "adamw"}
    assert all(id(param) in leaf._offloaded for group in groups for param in group["params"])
    set_gradients(source.model, 1)
    source.optimizer.step()
    source.optimizer.zero_grad()
    trainer = _make_mock_trainer(save_path=str(tmp_path))
    trainer.args.train.checkpoint.save_steps = 0
    trainer.args.train.checkpoint.save_epochs = 1
    trainer.args.train.checkpoint.hf_save_steps = 1
    trainer.args.train.checkpoint.hf_save_epochs = 0
    trainer.train_dataloader = None
    trainer.data_iterator = None
    trainer.environ_meter.state_dict.return_value = {}
    trainer.channel_loss_callback = None
    trainer.offline_cache_writer = None
    state = TrainerState(global_step=1, epoch=0)
    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("torch.distributed.barrier"),
    ):
        source.checkpoint = ModelCheckpointManager(source)
        trainer.model = source
        trainer.save_dcp.side_effect = source.checkpoint.save_dcp
        trainer.save_hf_or_lora.side_effect = lambda state, stage="step_end": source.checkpoint.prepare_export(
            state, stage
        )
        callback = CheckpointCallback(trainer)
        global_callback = GlobalStateCallback(trainer)
        callback.on_step_end(state)
        global_callback.on_step_end(state)
        callback.on_epoch_end(state)
        global_callback.on_epoch_end(state)
        trainer.save_dcp.assert_called_once_with(state)

        target = build()
        target.train_args.checkpoint.load_path = str(tmp_path / "global_step_1")
        target.checkpoint = ModelCheckpointManager(target)
        target.checkpoint.load()
        assert_equal(source, target)
        for runtime in (source, target):
            set_gradients(runtime.model, 2)
            runtime.optimizer.step()
        assert_equal(source, target)
