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

"""Who decides that a LoRA config was a mistake.

A composed ``lora_config`` reaches every module, so a module it did not target
just stays frozen (``ModuleRuntime.on_lora_matched_nothing``). A sibling doing
full-parameter SFT still trains. Only the composer can see that LoRA was
requested and *nothing anywhere* is trainable.
"""

from types import SimpleNamespace

import pytest

from veomni.models.seed_omni.accelerator.omni_model_runtime import _reject_lora_that_matched_nothing


def _runtime(*, lora: bool, trainable: bool) -> SimpleNamespace:
    """A built ModuleRuntime as the composer sees it: a request and an outcome."""
    return SimpleNamespace(
        args=SimpleNamespace(lora_config={"r": 8} if lora else None),
        has_trainable_parameters=trainable,
    )


def test_a_job_without_lora_is_not_this_checks_business():
    """A fully-frozen job is legal on its own terms (an encode-only pass)."""
    _reject_lora_that_matched_nothing({"llm": _runtime(lora=False, trainable=False)})


def test_lora_that_adapted_its_module_passes():
    _reject_lora_that_matched_nothing(
        {
            "llm": _runtime(lora=True, trainable=True),
            "vision_encoder": _runtime(lora=True, trainable=False),
        }
    )


def test_a_sibling_doing_full_sft_does_not_fail_the_lora_miss():
    """A LoRA miss on the LLM is already logged by that module; a fully-trained
    VAE beside it means the job still has parameters to train."""
    _reject_lora_that_matched_nothing(
        {
            "llm": _runtime(lora=True, trainable=False),
            "vae": _runtime(lora=False, trainable=True),
        }
    )


def test_lora_that_matched_nowhere_is_rejected():
    with pytest.raises(ValueError, match="no trainable adapters"):
        _reject_lora_that_matched_nothing(
            {
                "llm": _runtime(lora=True, trainable=False),
                "vision_encoder": _runtime(lora=True, trainable=False),
            }
        )


def test_an_offline_cache_pass_freezes_everything_by_design():
    """``train_type: offline_cache`` is usually the training YAML with one flag
    overridden, so it carries the job's ``lora_config`` into a run that trains
    nothing on purpose — which is why ``OmniTrainer`` allows an empty optimizer."""
    _reject_lora_that_matched_nothing(
        {"llm": _runtime(lora=True, trainable=False)},
        SimpleNamespace(train_type="offline_cache"),
    )


def test_a_normal_train_job_is_still_checked():
    with pytest.raises(ValueError, match="no trainable adapters"):
        _reject_lora_that_matched_nothing(
            {"llm": _runtime(lora=True, trainable=False)},
            SimpleNamespace(train_type="train"),
        )
