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
"""DiffusionGemma corruption noise must not come from the process RNG.

Every rank runs `set_seed(train.seed)` with no rank offset, and a DataLoader worker
seeds itself from that same base seed on every rank. The corruption draws have fixed
shapes, so drawing them from the global stream hands the n-th sample of the n-th worker
a bit-identical `t`, corruption mask and noise on every rank -- the global batch then
carries a fraction of the independent noise samples it appears to.

These tests pin the replacement: a generator derived from `(seed, sample, canvas)`.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from veomni.data.data_transform import (
    _build_diffusion_gemma_decoder_targets,
    diffusion_gemma_sample_key,
    process_diffusion_gemma_example,
)


CANVAS_LENGTH = 64
VOCAB_SIZE = 512
NOISE_EPS = 1e-3
EOS_TOKEN_ID = 1
BLOCKS = 3


class FakeChatTemplate:
    """Minimal stand-in for `ChatTemplate`: the prompt is ignored, the answer supervised."""

    def __init__(self, prompt_length: int = 5):
        self.prompt_length = prompt_length

    def encode_messages(self, messages, max_seq_len: int):
        answer = list(messages)
        input_ids = [7] * self.prompt_length + answer
        labels = [-100] * self.prompt_length + answer
        return {"input_ids": input_ids[:max_seq_len], "labels": labels[:max_seq_len]}


def build_trainer():
    """A `DllmTrainer` with just enough state for `_build_data_transform`.

    Built with `__new__`, matching the trainer unit tests in
    `tests/models/test_diffusion_gemma.py`.
    """
    DllmTrainer = pytest.importorskip("veomni.trainer.dllm_trainer").DllmTrainer

    trainer = DllmTrainer.__new__(DllmTrainer)
    text_config = SimpleNamespace(vocab_size=VOCAB_SIZE, pad_token_id=0, eos_token_id=EOS_TOKEN_ID)
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(max_seq_len=1024, text_keys="messages", data_type="diffusion_gemma"),
            train=SimpleNamespace(seed=42, dyn_bsz=False, num_train_epochs=1),
        ),
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                canvas_length=CANVAS_LENGTH,
                max_canvases=4,
                canvas_start=0,
                compute_all_canvas_df_loss=True,
                get_text_config=lambda: text_config,
            ),
            tokenizer=None,
            chat_template=FakeChatTemplate(),
        ),
        state=SimpleNamespace(epoch=0),
        on_epoch_begin=lambda: None,
    )
    return trainer


def build_targets(
    answer_token: int,
    corruption_seed: int,
    corruption_epoch: int = 0,
    compute_all_canvas_df_loss: bool = True,
):
    """Run the shared canvas builder on a uniform answer.

    Every block holds the same token, so any difference between the returned canvases
    is corruption noise and nothing else.
    """
    answer_ids = torch.full((BLOCKS * CANVAS_LENGTH,), answer_token, dtype=torch.long)
    encoder_ids = torch.cat([torch.full((5,), 7, dtype=torch.long), answer_ids])
    return _build_diffusion_gemma_decoder_targets(
        answer_ids,
        answer_ids.clone(),
        encoder_ids,
        CANVAS_LENGTH,
        VOCAB_SIZE,
        NOISE_EPS,
        EOS_TOKEN_ID,
        BLOCKS,
        compute_all_canvas_df_loss,
        diffusion_gemma_sample_key(corruption_seed, corruption_epoch, encoder_ids),
    )


def test_corruption_leaves_the_process_rng_untouched():
    """The regression that matters: no draw may reach for the global stream.

    Holding this makes the corruption independent of how many draws came before, which
    is what decouples one rank's noise from another's.
    """
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    build_targets(answer_token=11, corruption_seed=42)
    assert torch.equal(torch.get_rng_state(), before)


def test_single_canvas_draw_leaves_the_process_rng_untouched():
    """`compute_all_canvas_df_loss=False` also draws *which* block to supervise."""
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    build_targets(answer_token=11, corruption_seed=42, compute_all_canvas_df_loss=False)
    assert torch.equal(torch.get_rng_state(), before)


def test_the_epoch_redraws_the_supervised_block():
    """An epoch-free key would pin every answer to one block for the whole run.

    The default objective supervises one canvas per sample, so the other blocks are
    only ever denoised if a later epoch draws them.
    """
    drawn = set()
    for epoch in range(30):
        _, labels, _, context_mask, _ = build_targets(
            answer_token=11,
            corruption_seed=42,
            corruption_epoch=epoch,
            compute_all_canvas_df_loss=False,
        )
        assert labels.shape == (CANVAS_LENGTH,)
        # The answer is uniform, so the drawn block only shows in the context mask,
        # which covers the prompt plus every block before the one being supervised.
        drawn.add(int(context_mask.sum()))
    assert len(drawn) == BLOCKS


def test_the_epoch_redraws_the_noise():
    first, _, _, _, _ = build_targets(answer_token=11, corruption_seed=42, corruption_epoch=0)
    second, _, _, _, _ = build_targets(answer_token=11, corruption_seed=42, corruption_epoch=1)
    assert not torch.equal(first, second)


def test_different_samples_get_different_noise():
    """Two samples processed at the same stream position must not share a draw.

    Before the fix these two calls were bit-identical: same seed, same call order and
    same shapes meant the same `t` and the same corruption mask.
    """
    first, _, _, _, _ = build_targets(answer_token=11, corruption_seed=42)
    second, _, _, _, _ = build_targets(answer_token=12, corruption_seed=42)
    # The clean content differs by construction, so compare where each canvas was
    # corrupted rather than the raw ids.
    first_corrupted = first != 11
    second_corrupted = second != 12
    assert not torch.equal(first_corrupted, second_corrupted)


def test_same_sample_is_reproducible():
    """Same seed and same sample re-derive the same canvas, so a resume is consistent."""
    first, _, _, _, _ = build_targets(answer_token=11, corruption_seed=42)
    second, _, _, _, _ = build_targets(answer_token=11, corruption_seed=42)
    assert torch.equal(first, second)


def test_seed_changes_the_noise():
    first, _, _, _, _ = build_targets(answer_token=11, corruption_seed=42)
    second, _, _, _, _ = build_targets(answer_token=11, corruption_seed=43)
    assert not torch.equal(first, second)


def test_each_canvas_of_one_sample_draws_independently():
    """`(K, C)` rows share a sample key but not a generator."""
    canvases, _, _, _, _ = build_targets(answer_token=11, corruption_seed=42)
    assert canvases.shape == (BLOCKS, CANVAS_LENGTH)
    for index in range(1, BLOCKS):
        assert not torch.equal(canvases[0], canvases[index])


def test_transform_end_to_end_decorrelates_and_spares_the_global_rng():
    """Same checks through the registered text-only transform."""
    transform_kwargs = {
        "chat_template": FakeChatTemplate(),
        "max_seq_len": 1024,
        "canvas_length": CANVAS_LENGTH,
        "max_canvases": 4,
        "vocab_size": VOCAB_SIZE,
        "noise_eps": NOISE_EPS,
        "eos_token_id": EOS_TOKEN_ID,
        "text_keys": "messages",
        "compute_all_canvas_df_loss": True,
        "corruption_seed": 42,
    }
    first_answer = [11] * (2 * CANVAS_LENGTH)
    second_answer = [12] * (2 * CANVAS_LENGTH)

    torch.manual_seed(1234)
    before = torch.get_rng_state()
    first = process_diffusion_gemma_example({"messages": first_answer}, **transform_kwargs)[0]
    second = process_diffusion_gemma_example({"messages": second_answer}, **transform_kwargs)[0]
    assert torch.equal(torch.get_rng_state(), before)

    first_corrupted = first["decoder_input_ids"] != 11
    second_corrupted = second["decoder_input_ids"] != 12
    assert not torch.equal(first_corrupted, second_corrupted)


def test_the_trainer_epoch_survives_the_dataset_re_wrapping_the_transform():
    """The epoch has to reach the transform object the *dataset* ends up holding.

    `build_mapping_dataset` / `build_iterable_dataset` wrap the trainer's transform in
    a second `functools.partial` to bind `source_name`, and partial flattens a nested
    partial by copying its keyword dict. Anything the trainer writes into its own
    partial's keywords is therefore invisible downstream; only mutating a shared object
    that the copied dict points at gets through.
    """
    trainer = build_trainer()
    # `_build_data_transform` has to hand over the shared object, not `holder.epoch`.
    trainer._build_data_transform()
    assert trainer.base.data_transform.keywords["corruption_epoch"] is trainer._corruption_epoch

    # Exactly what `build_mapping_dataset` / `build_iterable_dataset` do to it.
    dataset_transform = partial(trainer.base.data_transform, source_name=None)
    example = {"messages": [11] * (2 * CANVAS_LENGTH)}

    trainer.on_epoch_begin()
    first = dataset_transform(example)[0]["decoder_input_ids"]
    trainer.base.state.epoch = 1
    trainer.on_epoch_begin()
    second = dataset_transform(example)[0]["decoder_input_ids"]

    assert not torch.equal(first, second)
