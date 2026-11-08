# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-PR gate: a cyclic cursor belongs to the saved optimizer generation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.data.test_cyclic_packing import Samples, loader
from veomni.trainer.callbacks import global_state_callback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.skipif(
    not hasattr(global_state_callback, "_saved_multi_optimizer_generations"), reason="requires owner-aware DCP PR"
)
def test_cyclic_cursor_dcp_and_next_optimizer_step(tmp_path):
    from tests.checkpoints.test_checkpoint_callback import _make_mock_runtime, _make_mock_trainer
    from veomni.checkpoint import layout
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.models.checkpoint.manager import ModelCheckpointManager
    from veomni.optim.lr_scheduler import build_lr_scheduler
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.trainer.callbacks.base import TrainerState
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    class StepView:
        def __init__(self, trainer):
            self.trainer = trainer

        @property
        def value(self):
            return self.trainer.state.global_step

    def build():
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.args.train_steps = 2
        trainer.args.train.num_train_epochs = 3
        trainer.environ_meter.state_dict.return_value = {}
        trainer.channel_loss_callback = None
        trainer.offline_cache_writer = None
        runtime = _make_mock_runtime(save_path=str(tmp_path))
        runtime.model = torch.nn.Linear(2, 2)
        leaf = torch.optim.AdamW(runtime.model.parameters(), lr=0.01)
        runtime.optimizer = MultiOptimizer(runtime.model, {"model": leaf}, ["model"])
        runtime.lr_scheduler = build_lr_scheduler(runtime.optimizer, train_steps=6, lr=0.01)
        runtime.parallel_state = SimpleNamespace(global_rank=0, dp_mode="ddp", extra_parallel_names=[])
        runtime.checkpoint = ModelCheckpointManager(runtime)
        trainer.model = runtime
        trainer.train_dataloader = loader(Samples(), StepView(trainer), train_steps=2, train_epochs=3)
        trainer.data_iterator = SimpleNamespace(state_dict=trainer.train_dataloader.state_dict)
        return trainer

    def update(trainer):
        batches = next(trainer.train_dataloader)
        value = sum(int(batch["tokens"].sum()) for batch in batches)
        input_tensor = torch.tensor([[float(value), 1.0]])
        trainer.model.optimizer.zero_grad()
        loss = trainer.model.model(input_tensor).square().sum()
        loss.backward()
        grads = [param.grad.detach().clone() for param in trainer.model.model.parameters()]
        trainer.model.optimizer.step()
        trainer.model.lr_scheduler.step()
        trainer.model.optimizer.zero_grad(set_to_none=True)
        trainer.state.global_step += 1
        trainer.train_dataloader.commit_step()
        return value, loss.detach(), grads

    with (
        patch("veomni.models.checkpoint.manager.build_checkpointer", return_value=DistributedCheckpointer),
        patch("torch.distributed.barrier"),
    ):
        source = build()
        update(source)
        update(source)
        state = TrainerState(global_step=2, epoch=0)
        source.model.checkpoint.save_dcp(state)
        GlobalStateCallback(source).save_global_state(state)

        restored = build()
        path = layout.step_dir(str(tmp_path), 2)
        restored.args.train.checkpoint.load_path = path
        restored.model.train_args.checkpoint.load_path = path
        restored.model.checkpoint.load()
        GlobalStateCallback(restored).load_global_state()
        source.train_dataloader.set_epoch(1)
        restored.train_dataloader.set_epoch(1)
        actual = update(restored)
        expected = update(source)
        assert actual[0] == expected[0]
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(
            restored.model.optimizer.state_dict(), source.model.optimizer.state_dict(), rtol=0, atol=0
        )
        for actual_param, expected_param in zip(restored.model.model.parameters(), source.model.model.parameters()):
            torch.testing.assert_close(actual_param, expected_param, rtol=0, atol=0)
