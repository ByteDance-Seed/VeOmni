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

"""Unit tests for OmniTrainer explicit step contexts (no GPU / distributed)."""

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from veomni.distributed.offloading import build_activation_offloading_context, custom_save_on_cpu
from veomni.trainer.omni.omni_trainer import cascade_module_reshard


class _FakeModuleRuntime:
    def __init__(self) -> None:
        self.reshard_calls: list[bool] = []

    def _model_reshard(self, reshard: bool) -> None:
        self.reshard_calls.append(reshard)


def _module_runtimes(n: int = 2) -> OrderedDict[str, _FakeModuleRuntime]:
    return OrderedDict((f"m{i}", _FakeModuleRuntime()) for i in range(n))


def test_cascade_module_reshard_noop_without_accumulation():
    runtimes = _module_runtimes()
    cascade_module_reshard(runtimes, micro_step=0, num_micro_steps=1)
    for runtime in runtimes.values():
        assert runtime.reshard_calls == []


def test_cascade_module_reshard_noop_on_middle_step():
    runtimes = _module_runtimes()
    cascade_module_reshard(runtimes, micro_step=1, num_micro_steps=3)
    for runtime in runtimes.values():
        assert runtime.reshard_calls == []


@pytest.mark.parametrize(
    ("micro_step", "expected"),
    [
        (0, False),
        (2, True),
    ],
)
def test_cascade_module_reshard_first_and_last(micro_step, expected):
    runtimes = _module_runtimes(n=2)
    cascade_module_reshard(runtimes, micro_step=micro_step, num_micro_steps=3)
    for runtime in runtimes.values():
        assert runtime.reshard_calls == [expected]


def test_build_activation_offloading_disabled():
    fwd, bwd = build_activation_offloading_context(enable_activation=False, enable_gradient_checkpointing=False)
    assert isinstance(fwd, nullcontext)
    assert isinstance(bwd, nullcontext)


def test_build_activation_offloading_enabled():
    fwd, bwd = build_activation_offloading_context(
        enable_activation=True,
        enable_gradient_checkpointing=False,
        activation_gpu_limit=1.0,
    )
    assert isinstance(fwd, custom_save_on_cpu)
    assert isinstance(bwd, nullcontext)


def test_build_step_contexts_offload_flags():
    """Smoke: _build_step_contexts wires nullcontext when offload is disabled."""
    from veomni.trainer.omni.omni_trainer import OmniTrainer

    trainer = object.__new__(OmniTrainer)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            accelerator=SimpleNamespace(offload_config=None),
            gradient_checkpointing=SimpleNamespace(enable=False),
            enable_batch_invariant_mode=False,
        )
    )
    OmniTrainer._build_step_contexts(trainer)
    assert isinstance(trainer.fwd_activation_offload_ctx, nullcontext)
    assert isinstance(trainer.bwd_activation_offload_ctx, nullcontext)


def test_omni_model_runtime_collect_step_metrics_skips_empty_meters():
    from veomni.models.seed_omni.accelerator.omni_model_runtime import OmniModelRuntime

    class _MeteredRuntime(_FakeModuleRuntime):
        def collect_step_metrics(self):
            return ("flops", [1, 2, 3])

    class _EmptyRuntime(_FakeModuleRuntime):
        def collect_step_metrics(self):
            return None

    module_runtimes = OrderedDict([("a", _MeteredRuntime()), ("b", _EmptyRuntime())])
    runtime = OmniModelRuntime(SimpleNamespace(), module_runtimes=module_runtimes)

    assert runtime.collect_step_metrics() == {"a": ("flops", [1, 2, 3])}


def test_omni_model_runtime_forwards_composed_model_surface():
    """The runtime is the single model handle: unknown attrs reach the OmniModel."""
    from veomni.models.seed_omni.accelerator.omni_model_runtime import OmniModelRuntime

    model = SimpleNamespace(config="omni-config", modules_dict={"a": "module-a"})
    runtime = OmniModelRuntime(model)

    assert runtime.config == "omni-config"
    assert runtime.modules_dict == {"a": "module-a"}
    with pytest.raises(AttributeError):
        _ = runtime.not_a_model_attribute
