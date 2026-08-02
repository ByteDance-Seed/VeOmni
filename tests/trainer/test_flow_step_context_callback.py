# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""FlowStepContextCallback: per-micro-step RNG identity for flow-matching runs.

The callback is registered for EVERY trainer, so both halves matter: it must
stamp the exact identity ``veomni.schedulers.flow_matching`` expects, and it must
stay completely inert for the text/VLM models that carry no ``flow`` recipe.
"""

from types import SimpleNamespace

import pytest
import torch

from veomni.schedulers.flow_matching import DEFAULT_REFERENCE_FLOW_CONFIG, prepare_reference_flow_batch
from veomni.trainer.callbacks import base as callback_base
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.flow_callback import FlowStepContextCallback


@pytest.fixture
def dp_rank_2(monkeypatch):
    """Pin the ParallelState the callback captures at construction."""
    monkeypatch.setattr(callback_base, "get_parallel_state", lambda: SimpleNamespace(dp_rank=2))


def _callback(model_config, seed=1234):
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(seed=seed)),
        model_config=model_config,
    )
    return FlowStepContextCallback(trainer)


def test_stamps_one_identity_per_micro_batch(dp_rank_2):
    callback = _callback(SimpleNamespace(flow=dict(DEFAULT_REFERENCE_FLOW_CONFIG)))
    assert callback.enabled

    micro_batches = [{"input_ids": 0}, {"input_ids": 1}, {"input_ids": 2}]
    callback.on_step_begin(TrainerState(global_step=17), micro_batches=micro_batches)

    assert [mb["flow_step_context"]["micro_step"] for mb in micro_batches] == [0, 1, 2]
    for micro_batch in micro_batches:
        context = micro_batch["flow_step_context"]
        # The rank is the pure data-parallel replica rank: ranks differing only
        # by SP/EP must land on one identity so they draw the same noise.
        assert context["train_seed"] == 1234
        assert context["data_replica_rank"] == 2
        assert context["optimizer_step"] == 17
        # Accepted verbatim by the sampler -- no reshaping between the two.
        prepare_reference_flow_batch(
            torch.zeros(1, 4, 2, 2),
            torch.zeros(1, 4, 2, 2),
            vae_config={"scaling_factor": 0.5, "shift_factor": None},
            flow_config=None,
            flow_step_context=context,
        )


def test_identity_tracks_the_optimizer_step(dp_rank_2):
    """A resumed run replays step N with step N's identity; two different steps
    must not share one, or the objective repeats noise across the run."""
    callback = _callback(SimpleNamespace(flow=dict(DEFAULT_REFERENCE_FLOW_CONFIG)))

    first, second = [{}], [{}]
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=first)
    callback.on_step_begin(TrainerState(global_step=2), micro_batches=second)

    assert first[0]["flow_step_context"] != second[0]["flow_step_context"]


@pytest.mark.parametrize(
    "model_config",
    [None, SimpleNamespace(), SimpleNamespace(flow=None)],
    ids=["no_model_config", "no_flow_attr", "flow_is_none"],
)
def test_inert_without_a_flow_recipe(dp_rank_2, model_config):
    """Text/VLM runs construct this callback too; it must not touch their batches."""
    callback = _callback(model_config)
    assert not callback.enabled

    micro_batches = [{"input_ids": 0}]
    callback.on_step_begin(TrainerState(global_step=3), micro_batches=micro_batches)
    assert micro_batches == [{"input_ids": 0}]


def test_tolerates_an_empty_step(dp_rank_2):
    callback = _callback(SimpleNamespace(flow=dict(DEFAULT_REFERENCE_FLOW_CONFIG)))
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=None)
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=[])
