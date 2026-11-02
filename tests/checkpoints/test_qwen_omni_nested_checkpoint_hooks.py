# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Known Omni load hooks must be safe when PEFT nests the model."""

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.multi_optimizer_checkpoint import _check_model_serialization_hooks


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class Qwen3OmniMoeForConditionalGeneration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

        def _drop_talker_and_code2wav_keys(
            module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            if prefix:
                return
            for key in list(state_dict):
                if key.startswith(("talker.", "code2wav.")):
                    del state_dict[key]

        self.register_load_state_dict_pre_hook(_drop_talker_and_code2wav_keys)


Qwen3OmniMoeForConditionalGeneration.__module__ = "veomni.models.transformers.qwen3_omni_moe.generated.test"


class Qwen2_5OmniForConditionalGeneration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

        def _drop_talker_and_token2wav_keys(
            module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            if prefix:
                return
            for key in list(state_dict):
                if key.startswith(("talker.", "token2wav.")):
                    del state_dict[key]

        self.register_load_state_dict_pre_hook(_drop_talker_and_token2wav_keys)


Qwen2_5OmniForConditionalGeneration.__module__ = "veomni.models.transformers.qwen2_5_omni.generated.test"


@pytest.mark.parametrize("model_class", [Qwen3OmniMoeForConditionalGeneration, Qwen2_5OmniForConditionalGeneration])
def test_nested_builtin_omni_hook_allows_peft_prefix(model_class):
    root = torch.nn.Module()
    root.base_model = torch.nn.Module()
    root.base_model.model = model_class()
    _check_model_serialization_hooks(root)


def test_builtin_omni_hook_allows_actual_peft_wrapper():
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        Qwen3OmniMoeForConditionalGeneration(), peft.LoraConfig(r=2, lora_alpha=4, target_modules=["linear"])
    )
    assert "base_model.model" in dict(model.named_modules())
    _check_model_serialization_hooks(model)
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(42)
    model.load_state_dict(saved, strict=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def test_root_builtin_omni_hook_still_rejects_filtered_model_tensors():
    root = Qwen3OmniMoeForConditionalGeneration()
    root.talker = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="load hook is a no-op"):
        _check_model_serialization_hooks(root)


def test_unrecognized_nested_hook_still_fails_closed():
    root = torch.nn.Module()
    root.base_model = torch.nn.Module()
    root.base_model.model = Qwen3OmniMoeForConditionalGeneration()
    root.base_model.model.register_load_state_dict_pre_hook(lambda *args: None)
    with pytest.raises(RuntimeError, match="Cannot prove model serialization hook"):
        _check_model_serialization_hooks(root)


def test_known_name_nested_hook_that_changes_state_still_fails_closed():
    class Qwen3OmniMoeForConditionalGeneration(torch.nn.Module):
        def __init__(self):
            super().__init__()

            def _drop_talker_and_code2wav_keys(
                module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            ):
                state_dict.clear()

            self.register_load_state_dict_pre_hook(_drop_talker_and_code2wav_keys)

    Qwen3OmniMoeForConditionalGeneration.__module__ = "veomni.models.transformers.qwen3_omni_moe.generated.test"
    root = torch.nn.Module()
    root.base_model = Qwen3OmniMoeForConditionalGeneration()
    with pytest.raises(RuntimeError, match="nested Qwen Omni load hook"):
        _check_model_serialization_hooks(root)


def test_known_name_nested_hook_that_changes_real_weight_is_rejected():
    class Qwen3OmniMoeForConditionalGeneration(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

            def _drop_talker_and_code2wav_keys(
                module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            ):
                if prefix + "linear.weight" in state_dict:
                    state_dict[prefix + "linear.weight"].zero_()

            self.register_load_state_dict_pre_hook(_drop_talker_and_code2wav_keys)

    Qwen3OmniMoeForConditionalGeneration.__module__ = "veomni.models.transformers.qwen3_omni_moe.generated.test"
    root = torch.nn.Module()
    root.base_model = Qwen3OmniMoeForConditionalGeneration()
    with pytest.raises(RuntimeError, match="nested Qwen Omni load hook"):
        _check_model_serialization_hooks(root)


def test_known_qwen25_nested_hook_that_changes_errors_is_rejected():
    class Qwen2_5OmniForConditionalGeneration(torch.nn.Module):
        def __init__(self):
            super().__init__()

            def _drop_talker_and_token2wav_keys(
                module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            ):
                if prefix:
                    error_msgs.append("unexpected mutation")
                    return

            self.register_load_state_dict_pre_hook(_drop_talker_and_token2wav_keys)

    Qwen2_5OmniForConditionalGeneration.__module__ = "veomni.models.transformers.qwen2_5_omni.generated.test"
    root = torch.nn.Module()
    root.base_model = Qwen2_5OmniForConditionalGeneration()
    with pytest.raises(RuntimeError, match="nested Qwen Omni load hook"):
        _check_model_serialization_hooks(root)
