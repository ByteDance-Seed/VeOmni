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

"""Model-class hooks run by ``post_process_after_weight_loading``."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from veomni.models.module_utils import post_process_after_weight_loading


class _SharedStack(nn.Module):
    def __init__(self, tie_word_embeddings=False):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.decoder = nn.Linear(4, 4)
        self.encoder = nn.Linear(4, 4)
        self.missing_seen = None

    @classmethod
    def _validate_checkpoint_missing_parameters(cls, model, missing):
        model.missing_seen = set(missing)

    @classmethod
    def retie_shared_weights(cls, model):
        model.encoder.weight = model.decoder.weight
        return 1

    def _init_weights(self, module):
        pass

    def get_input_embeddings(self):
        return self.decoder

    def get_output_embeddings(self):
        return None


def test_shared_submodules_are_retied_after_loading():
    model = _SharedStack()
    assert model.encoder.weight is not model.decoder.weight

    post_process_after_weight_loading(model, buffer_dict={})

    assert model.encoder.weight is model.decoder.weight


@pytest.mark.parametrize("should_skip_hf_weight_load", [True, False], ids=["checkpoint_resume", "random_init"])
def test_shared_submodules_are_retied_when_no_hf_weights_are_loaded(should_skip_hf_weight_load):
    """``to_empty`` breaks the tie on these paths too, and nothing else restores it.

    A checkpoint resume skips the HF load, so without the hook the encoder trains as an
    independent copy of the decoder with optimizer state of its own.
    """
    from veomni.distributed.torch_parallelize import _materialize_and_load_weights

    class Initializable(_SharedStack):
        def init_weights(self):
            pass

    with torch.device("meta"):
        model = Initializable()
    model.encoder.weight = model.decoder.weight

    _materialize_and_load_weights(
        model,
        weights_path="unused" if should_skip_hf_weight_load else None,
        materialize_device="cpu",
        should_skip_hf_weight_load=should_skip_hf_weight_load,
        is_peft_model=False,
        adapter_path=None,
        broadcast_from_rank0=False,
    )

    assert not model.decoder.weight.is_meta
    assert model.encoder.weight is model.decoder.weight


def test_the_missing_parameter_validator_sees_the_missing_names_before_they_are_initialized():
    model = _SharedStack()

    post_process_after_weight_loading(model, buffer_dict={}, parameter_names_left={"encoder.bias"})

    assert model.missing_seen == {"encoder.bias"}


def test_a_validator_can_refuse_the_checkpoint():
    class Strict(_SharedStack):
        @classmethod
        def _validate_checkpoint_missing_parameters(cls, model, missing):
            if missing:
                raise RuntimeError(f"missing {sorted(missing)}")

    with pytest.raises(RuntimeError, match="encoder.bias"):
        post_process_after_weight_loading(Strict(), buffer_dict={}, parameter_names_left={"encoder.bias"})


def test_hooks_are_looked_up_on_the_class_not_forwarded_attributes():
    class Forwarding(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.config = SimpleNamespace()
            self.inner = inner

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    inner = _SharedStack()
    post_process_after_weight_loading(Forwarding(inner), buffer_dict={})

    assert inner.encoder.weight is not inner.decoder.weight


def test_tied_embeddings_without_an_output_projection_are_left_alone():
    model = _SharedStack(tie_word_embeddings=True)
    before = model.decoder.weight

    post_process_after_weight_loading(model, buffer_dict={})

    assert model.decoder.weight is before
    assert isinstance(model.decoder.weight, torch.nn.Parameter)
