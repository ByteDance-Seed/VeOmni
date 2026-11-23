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

"""Unit tests for LLaDA's mask-predictor generation.

The denoising loop is exercised against a stub backbone, because what these
tests pin is the *selection* rule and the termination conditions, not the
predictor's weights: a real checkpoint would answer "does this converge" but a
stub answering with a known distribution says exactly which positions get
committed and in what order. A shape-only test would pass while committing the
prompt or re-masking decoded tokens, which is the failure mode that matters.

`llada2_uni`'s FSM is block-causal and commits a whole block per step, so it is
not a reference for this: LLaDA ranks the entire masked span at once.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.modules.llada.modeling import SIGNAL_TEXT_DONE, LladaOmni
from veomni.models.seed_omni.utils.conversation import ConversationItem


VOCAB = 16
PROMPT = [1, 2, 3]
ANSWER_LEN = 4


class _StubTokenizer:
    eos_token_id = 9

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002 - signature mirrors HF
        return "|".join(str(i) for i in ids)


def _bare_module(prompt_ids: list[int] = PROMPT) -> LladaOmni:
    """A LladaOmni carrying only what the denoising loop actually reads.

    The loop needs ``self.device`` (an ``nn.Module`` attribute), the mask id, the
    tokenizer and the EOS id — nothing else from the loaded model. Building the
    real ``__init__`` would load an 8B config and a stub backbone, so the fields
    are assigned onto a module skeleton; what this makes explicit is the loop's
    dependency list, which is the point.
    """
    module = LladaOmni.__new__(LladaOmni)
    torch.nn.Module.__init__(module)
    # `self.device` is a `PreTrainedModel` property that reads the first
    # parameter's device, so the skeleton needs one; it is never used numerically.
    module._stub_anchor = torch.nn.Parameter(torch.zeros(1))
    module.config = SimpleNamespace(mask_token_id=VOCAB - 1)
    module._tokenizer = _StubTokenizer()
    module._reset_diffusion_state()
    module._prompt = prompt_ids
    return module


def _stub_decode_forward(preferred: list[int] | None = None):
    """Replace ``_decode_forward`` with a stub whose confidence decays by position.

    ``preferred[i]`` is the token the predictor favours at answer position ``i``;
    its logit is ``10 - i``, so position 0 is the most confident and the ranking
    is unambiguous. Equal logits would leave ``topk`` to break ties arbitrarily,
    and a test written on top of that would pin the tie-break order instead of
    the confidence rule.
    """

    def _forward(self) -> dict:
        total = self._diff_x.numel()
        logits = torch.zeros(1, total, VOCAB)
        for index in range(total):
            answer_pos = index - self._diff_prompt_len
            if answer_pos < 0:
                logits[0, index, 0] = 100.0
                continue
            token = preferred[answer_pos] if preferred is not None else answer_pos % VOCAB
            logits[0, index, token] = 10.0 - answer_pos
        return {"llada_logits": logits}

    return _forward


def test_num_transfer_schedule_puts_the_remainder_first():
    """The schedule must commit every position in `steps` passes, front-loaded."""
    assert LladaOmni._make_num_transfer_schedule(4, 2) == [2, 2]
    assert LladaOmni._make_num_transfer_schedule(5, 2) == [3, 2]
    # 128 positions over 64 steps is the config's default shape: two per pass.
    schedule = LladaOmni._make_num_transfer_schedule(128, 64)
    assert sum(schedule) == 128
    assert len(schedule) == 64
    assert schedule[0] >= schedule[-1]


def test_init_masks_the_answer_span_and_keeps_the_prompt():
    """The first pass must leave the prompt intact and mask the whole answer."""
    module = _bare_module()
    module._decode_forward = _stub_decode_forward().__get__(module)
    item = ConversationItem(type="text", value={"input_ids": torch.tensor(PROMPT)}, role="assistant")
    item.meta["_omni_token_len"] = len(PROMPT)

    module.generate(
        conversation_list=[[item]],
        generation_kwargs={"gen_length": ANSWER_LEN, "diffusion_steps": 2},
    )

    mask_id = module.config.mask_token_id
    assert module._diff_prompt_len == len(PROMPT)
    assert module._diff_x[: len(PROMPT)].tolist() == PROMPT
    assert module._diff_x[len(PROMPT) :].tolist() == [mask_id] * ANSWER_LEN


def test_each_step_commits_only_the_masked_span():
    """A pass must never rewrite a prompt token, even if it looks confident."""
    module = _bare_module()
    preferred = [7] * ANSWER_LEN
    module._decode_forward = _stub_decode_forward(preferred).__get__(module)
    item = ConversationItem(type="text", value={"input_ids": torch.tensor(PROMPT)}, role="assistant")
    item.meta["_omni_token_len"] = len(PROMPT)
    gk = {"gen_length": ANSWER_LEN, "diffusion_steps": 2}

    out = module.generate(conversation_list=[[item]], generation_kwargs=gk)
    out = module.generate(llada_logits=out["llada_logits"], generation_kwargs=gk)

    assert module._diff_x[: len(PROMPT)].tolist() == PROMPT
    # Two of four committed; the rest must still be mask, not guessed.
    mask_id = module.config.mask_token_id
    assert module._diff_x[len(PROMPT) :].tolist() == [7, 7, mask_id, mask_id]
    assert module._diff_step == 1
    # Two positions were committed by the first schedule entry, not all four.
    assert out.get(SIGNAL_TEXT_DONE) is None


def test_denoising_terminates_with_text_done_and_decodes_the_span():
    """The loop must run exactly `diffusion_steps` passes and emit a clean span."""
    module = _bare_module()
    module._decode_forward = _stub_decode_forward([5, 6, 7, 8]).__get__(module)
    item = ConversationItem(type="text", value={"input_ids": torch.tensor(PROMPT)}, role="assistant")
    item.meta["_omni_token_len"] = len(PROMPT)
    gk = {"gen_length": ANSWER_LEN, "diffusion_steps": 4}

    out = module.generate(conversation_list=[[item]], generation_kwargs=gk)
    for _ in range(ANSWER_LEN):
        if "generated" in out:
            break
        out = module.generate(llada_logits=out["llada_logits"], generation_kwargs=gk)

    assert out[FSM_SIGNAL_KEY] == SIGNAL_TEXT_DONE
    assert out["generated"]["value"] == "5|6|7|8"
    assert out["generated"]["meta"]["token_ids"] == [5, 6, 7, 8]


def test_all_masked_span_finishes_early_when_every_position_is_decoded():
    """Filling the span before the step budget must stop the loop, not spin."""
    module = _bare_module()
    module._decode_forward = _stub_decode_forward([1, 1, 1, 1]).__get__(module)
    item = ConversationItem(type="text", value={"input_ids": torch.tensor(PROMPT)}, role="assistant")
    item.meta["_omni_token_len"] = len(PROMPT)
    gk = {"gen_length": ANSWER_LEN, "diffusion_steps": 50}

    out = module.generate(conversation_list=[[item]], generation_kwargs=gk)
    out = module.generate(llada_logits=out["llada_logits"], generation_kwargs=gk)

    # The schedule's first entry commits two of four; the rest stay masked.
    assert out.get(SIGNAL_TEXT_DONE) is None
    assert module._diff_step == 1


def test_finalize_emits_a_truncated_span_instead_of_masking_it():
    """A budget cut must drop the undecoded masks, not decode them as text."""
    module = _bare_module()
    mask_id = module.config.mask_token_id
    module._diff_x = torch.tensor(PROMPT + [4, mask_id, mask_id])
    module._diff_prompt_len = len(PROMPT)

    out = module.finalize(ctx={})

    assert out["generated"]["value"] == "4"
    assert out["generated"]["meta"]["token_ids"] == [4]


def test_eos_stops_the_loop_once_everything_before_it_is_decoded():
    """An EOS followed by masks is not a finished answer."""
    module = _bare_module()
    mask_id = module.config.mask_token_id
    eos = _StubTokenizer.eos_token_id
    module._diff_x = torch.tensor(PROMPT + [4, eos, mask_id])
    module._diff_prompt_len = len(PROMPT)
    module._diff_gen_len = 3
    module._diff_steps = 10
    module._diff_step = 1
    module._diff_num_transfer = LladaOmni._make_num_transfer_schedule(3, 10)

    out = module._diffusion_step(torch.zeros(1, 6, VOCAB), {"gen_length": 3, "diffusion_steps": 10})

    # The mask after EOS means the answer is not complete, so no signal fires.
    assert out.get(SIGNAL_TEXT_DONE) is None


def test_reset_clears_the_denoising_row_between_requests():
    """A second request must not inherit the previous one's partial span."""
    module = _bare_module()
    module._diff_x = torch.tensor(PROMPT + [4, 5, 6, 7])
    module._diff_step = 3

    module.reset_local_inference_state()

    assert module._diff_x is None
    assert module._diff_step == 0
