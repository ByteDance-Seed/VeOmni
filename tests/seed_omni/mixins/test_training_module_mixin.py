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

import pytest
from torch import nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _Native(nn.Module):
    def forward(self, **kwargs):
        return {"path": ["native"], **kwargs}


class _Mid(TrainingModuleMixin, _Native):
    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        outputs["path"] = ["mid", *outputs["path"]]
        return outputs


class _Leaf(_Mid):
    pass


class _NoNative(TrainingModuleMixin, nn.Module):
    pass


def test_default_forward_delegates_to_the_native_forward_below_the_mixin():
    class _Plain(TrainingModuleMixin, _Native):
        pass

    assert _Plain().forward(x=1) == {"path": ["native"], "x": 1}


def test_a_subclass_forward_calling_super_reaches_the_native_forward_once():
    assert _Mid().forward(x=1) == {"path": ["mid", "native"], "x": 1}
    assert _Leaf().forward(x=1) == {"path": ["mid", "native"], "x": 1}


def test_default_forward_without_a_native_forward_is_not_implemented():
    with pytest.raises(NotImplementedError, match="_NoNative.forward"):
        _NoNative().forward()


def test_all_registered_classes_are_module_mixins():
    from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_MODEL_REGISTRY, BaseMixin
    from veomni.models.seed_omni.modules.module_modeling_base import PretrainedOmniModule

    accelerated_keys = set(OMNI_ACCELERATED_MODEL_REGISTRY.valid_keys())
    for name in OMNI_MODEL_REGISTRY.valid_keys():
        native_cls = OMNI_MODEL_REGISTRY[name]()
        assert name in accelerated_keys, f"{name} must register OMNI_ACCELERATED_MODEL_REGISTRY"
        assert issubclass(native_cls, PretrainedOmniModule), f"{name} native must inherit PretrainedOmniModule"
        assert not issubclass(native_cls, BaseMixin), f"{name} native must not inherit BaseMixin"
        accelerated_cls = OMNI_ACCELERATED_MODEL_REGISTRY[name]()
        assert issubclass(accelerated_cls, BaseMixin), f"{name} accelerated must inherit BaseMixin"


def test_the_mixin_declares_no_dummy_inputs_of_its_own():
    """Modules that need one take different shapes, and a mixin stub would shadow them.

    The mixin sits ahead of the modeling class in MRO, so a `dummy_inputs` here wins
    over every module's own -- silently, and with an incompatible signature. That is
    exactly what `LladaUniImageTokenizer.dummy_inputs()` hit: the mixin's
    `(*, batch_size, device, dtype)` replaced a no-argument method whose only caller
    passes no arguments, so the image tokenizer's encode path raised `TypeError`.
    """
    assert not hasattr(TrainingModuleMixin, "dummy_inputs"), (
        "TrainingModuleMixin must not declare `dummy_inputs`: it shadows each module's own "
        "implementation through MRO. Declare it on the module that needs one."
    )


def test_every_accelerated_module_resolves_dummy_inputs_from_a_model_class():
    """A module whose modeling class defines `dummy_inputs` must not have it shadowed."""
    from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY

    shadowed = []
    for name in OMNI_ACCELERATED_MODEL_REGISTRY.valid_keys():
        accelerated_cls = OMNI_ACCELERATED_MODEL_REGISTRY[name]()
        resolved = accelerated_cls.dummy_inputs
        if getattr(resolved, "__qualname__", "").startswith("TrainingModuleMixin."):
            shadowed.append(name)
    assert not shadowed, f"these modules resolve `dummy_inputs` from the mixin instead of their model: {shadowed}"
