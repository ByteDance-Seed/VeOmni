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

from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin


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
