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

"""Unit tests for DiffusionGemma's generation entry point.

The sampler itself is covered where it lives
(``tests/models/test_diffusion_gemma_generation.py``); what these tests pin is the
FSM contract around it: one pass must emit exactly one ``generated`` row and one
``text_done``, a second pass must emit nothing, and the end-of-turn ids must come
from the tokenizer rather than from a guess.

A 26B checkpoint cannot load here, and calling the sampler for real would only
re-test the sampler, so it is stubbed -- but the stub receives the real argument
names, which is what catches a rename.
"""

from __future__ import annotations

import torch

from veomni.models.seed_omni.modules.diffusion_gemma import modeling as mod
from veomni.models.seed_omni.modules.diffusion_gemma.modeling import SIGNAL_TEXT_DONE, DiffusionGemmaOmni


class _StubTokenizer:
    """Enough of a tokenizer for the end-of-turn and decode paths."""

    eot_token_id = 42
    eos_token_id = 1
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<turn|>": 42, "<end_of_turn>": 43, "<eos>": 1}.get(token, 0)

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002 - mirrors HF
        return "|".join(str(i) for i in ids)


class _StubConversationItem:
    def __init__(self, value):
        self.type = "text"
        self.value = value
        self.meta = {}


def _bare_module() -> DiffusionGemmaOmni:
    """A module carrying only what ``generate`` reads.

    ``__new__`` plus a real ``nn.Module.__init__``: loading the 26B config would
    allocate a 26B skeleton, and the fields ``generate`` touches are five.
    """
    module = DiffusionGemmaOmni.__new__(DiffusionGemmaOmni)
    torch.nn.Module.__init__(module)
    module._stub_anchor = torch.nn.Parameter(torch.zeros(1))
    module._tokenizer = _StubTokenizer()
    module._infer_done = False
    return module


def _request(prompt_ids: list[int]) -> list[list[_StubConversationItem]]:
    return [[_StubConversationItem({"input_ids": torch.tensor(prompt_ids)})]]


def test_end_of_turn_ids_come_from_the_tokenizer_not_from_eos():
    """``eos`` does not close a turn on these checkpoints; ``eot`` does."""
    module = _bare_module()

    ids = module._end_of_turn_ids()

    assert 42 in ids
    # `eos_token_id` is 1 here and is deliberately absent: treating it as an
    # end-of-turn would truncate the answer after its first sentence.
    assert 1 not in ids
    assert ids == sorted(ids)


def test_one_pass_emits_text_done_and_the_decoded_span(monkeypatch):
    """The FSM relies on exactly one `generated` row from the single pass."""
    captured: dict = {}

    def fake_generate(model, prompt_inputs, eos_token_ids, config, max_blocks):
        captured["prompt_inputs"] = prompt_inputs
        captured["eos_token_ids"] = eos_token_ids
        captured["num_steps"] = config.num_steps
        captured["max_blocks"] = max_blocks
        return torch.tensor([[7, 8, 42]])

    monkeypatch.setattr(mod, "block_autoregressive_generate", fake_generate)
    module = _bare_module()

    out = module.generate(
        conversation_list=_request([1, 2, 3]),
        generation_kwargs={"num_steps": 12, "max_blocks": 2},
    )

    assert out[mod.FSM_SIGNAL_KEY] == SIGNAL_TEXT_DONE
    assert out["generated"]["type"] == "text"
    assert out["generated"]["meta"]["token_ids"] == [7, 8, 42]
    # The sampler's budget reaches it through the argument names, not a positional
    # guess: a rename in `EntropyBoundedSamplerConfig` has to fail here.
    assert captured["num_steps"] == 12
    assert captured["max_blocks"] == 2
    assert 42 in captured["eos_token_ids"]
    assert captured["prompt_inputs"]["input_ids"].tolist() == [1, 2, 3]


def test_second_pass_emits_nothing(monkeypatch):
    """A repeated FSM pass must not sample a second answer."""
    calls = {"n": 0}

    def fake_generate(*args, **kwargs):  # noqa: ARG001
        calls["n"] += 1
        return torch.tensor([[5]])

    monkeypatch.setattr(mod, "block_autoregressive_generate", fake_generate)
    module = _bare_module()

    module.generate(conversation_list=_request([1]), generation_kwargs={})
    out = module.generate(conversation_list=_request([1]), generation_kwargs={})

    assert calls["n"] == 1
    assert out == {}


def test_missing_prompt_is_refused_rather_than_sampled_from_nothing():
    """An unprocessed carrier must fail loudly instead of sampling a blank prompt."""
    module = _bare_module()

    try:
        module.generate(conversation_list=[[_StubConversationItem("<not encoded>")]], generation_kwargs={})
    except ValueError as error:
        assert "preprocessor" in str(error)
    else:
        raise AssertionError("generate accepted a carrier with no encoded prompt")


def test_reset_clears_the_emit_guard_between_requests():
    """Without the reset, the second request would return an empty answer."""
    module = _bare_module()
    module._infer_done = True

    module.reset_local_inference_state()

    assert module._infer_done is False
