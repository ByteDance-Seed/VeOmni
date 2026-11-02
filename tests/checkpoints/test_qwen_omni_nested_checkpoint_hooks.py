# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Known Omni load hooks must be safe when PEFT nests the model."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer


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
def test_nested_omni_hook_allows_peft_dcp_roundtrip(tmp_path, model_class):
    peft = pytest.importorskip("peft")

    def build():
        model = peft.get_peft_model(model_class(), peft.LoraConfig(r=2, lora_alpha=4, target_modules=["linear"]))
        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = MultiOptimizer(model, {"leaf": torch.optim.AdamW(params)}, ["leaf"])
        return model, optimizer

    model, optimizer = build()
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.full_like(param, 0.25)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    parallel = SimpleNamespace(dp_mode="ddp")
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(
        checkpoint, {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    target, restored = build()
    DistributedCheckpointer.load(
        checkpoint, {"model": target, "optimizer": restored, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    actual = target.state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    for source_param, target_param in zip(model.parameters(), target.parameters(), strict=True):
        if source_param.requires_grad:
            source_param.grad = torch.full_like(source_param, 0.125)
            target_param.grad = torch.full_like(target_param, 0.125)
    optimizer.step()
    restored.step()
    for source_param, target_param in zip(model.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(source_param, target_param, rtol=0, atol=0)
