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
"""Tests for multi-head loss plumbing."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.generic import ModelOutput


@dataclass
class _OutputWithLossDict(CausalLMOutputWithPast):
    loss_dict: Optional[dict] = None


def test_dict_in_loss_field_is_destroyed_by_model_output():
    """Document why a loss dictionary cannot be stored in ModelOutput.loss."""
    losses = {"foundation_loss": torch.tensor(1.0), "mtp_loss": torch.tensor(0.5)}

    out = CausalLMOutputWithPast(loss=losses)

    assert out.loss is None
    assert out.foundation_loss.item() == 1.0
    assert "loss" not in out


def test_loss_dict_field_survives_with_all_other_fields_none():
    """Verify a dedicated loss_dict field survives ModelOutput initialization."""
    foundation = torch.tensor(1.0)
    out = _OutputWithLossDict(
        loss=foundation,
        loss_dict={"foundation_loss": foundation, "mtp_loss": torch.tensor(0.5)},
    )

    assert out.loss is foundation
    assert set(out.loss_dict) == {"foundation_loss", "mtp_loss"}
    assert out.loss_dict["mtp_loss"].item() == 0.5


def test_loss_dict_tensors_stay_in_the_pytree():
    """Verify loss_dict tensors remain visible to PyTorch pytree traversal."""
    from torch.utils._pytree import tree_flatten

    mtp = torch.tensor(0.5, requires_grad=True)
    out = _OutputWithLossDict(loss=torch.tensor(1.0, requires_grad=True), loss_dict={"mtp_loss": mtp})

    leaves, _ = tree_flatten(out)
    assert any(leaf is mtp for leaf in leaves), "loss_dict tensors must be reachable from the pytree"


def test_postforward_prefers_loss_dict_over_loss():
    """Verify postforward composes all named head losses when loss_dict exists."""
    from veomni.trainer.base import BaseTrainer

    foundation = torch.tensor(2.0)
    out = _OutputWithLossDict(
        loss=foundation, loss_dict={"foundation_loss": foundation, "mtp_loss": torch.tensor(1.0)}
    )

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.micro_batch_token_len = {"foundation_tokens": torch.tensor(1), "mtp_tokens": torch.tensor(1)}
    trainer.micro_batches_token_len = {"foundation_tokens": torch.tensor(1), "mtp_tokens": torch.tensor(1)}
    trainer.global_micro_batches_token_len = {"foundation_tokens": 1.0, "mtp_tokens": 1.0}

    loss, loss_dict = trainer.postforward(out, {})

    assert set(loss_dict) == {"foundation_loss", "mtp_loss"}
    assert loss.item() == 3.0


def test_postforward_falls_back_to_plain_loss_tensor():
    """Verify postforward preserves the single-loss compatibility path."""
    from veomni.trainer.base import BaseTrainer

    out = CausalLMOutputWithPast(loss=torch.tensor(2.0), logits=torch.zeros(1))

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.micro_batch_token_len = {"foundation_tokens": torch.tensor(1)}
    trainer.micro_batches_token_len = {"foundation_tokens": torch.tensor(1)}
    trainer.global_micro_batches_token_len = {"foundation_tokens": 1.0}

    loss, loss_dict = trainer.postforward(out, {})

    assert set(loss_dict) == {"foundation_loss"}
    assert loss.item() == 2.0


def test_model_output_subclass_field_order_is_still_loss_first():
    """Verify the custom output preserves the base ModelOutput field contract."""
    from dataclasses import fields

    assert fields(_OutputWithLossDict)[0].name == "loss"
    assert issubclass(_OutputWithLossDict, ModelOutput)


def test_text_trainer_builds_model_sample_collator(monkeypatch):
    """Verify TextTrainer wires model-provided sample collation hooks."""
    import veomni.data.data_collator as data_collator
    from veomni.trainer.text_trainer import TextTrainer

    def sample_hook(feature):
        """Return a sample unchanged for collator wiring verification."""
        return feature

    model = SimpleNamespace(
        get_extra_collate_infos=lambda: {"mtp_labels": (-1, True, -100, 1)},
        get_sample_collate_func=lambda: sample_hook,
    )
    trainer = TextTrainer.__new__(TextTrainer)
    trainer.base = SimpleNamespace(
        model=model,
        args=SimpleNamespace(
            data=SimpleNamespace(data_type="conversation"),
            train=SimpleNamespace(pad_to_length=False),
        ),
    )
    monkeypatch.setattr(
        data_collator,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=False, sp_size=1),
    )

    trainer._build_collate_fn()

    assert trainer.base.collate_fn.sample_collate_func is sample_hook
    assert "mtp_labels" in trainer.base.collate_fn.collate_infos
