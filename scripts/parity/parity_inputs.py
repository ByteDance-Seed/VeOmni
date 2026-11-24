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
"""The one input both stacks are fed, described without reference to either.

Imported by ``dump_forward.py`` inside each venv, so it must stay importable with
nothing but ``torch`` -- a stack-specific import here would make the two runs read
different definitions of "the same input", which is the whole thing being avoided.

The geometry is the SFT training layout: a clean sequence of ``prompt + K*C`` tokens
for the encoder, and a noised canvas of ``K*C`` tokens for the decoder, whose RoPE
positions continue the prompt. Nothing is padded. Padding is a real code path in both
stacks, but it is a second axis of disagreement, and a comparison that fails should
say which one it failed on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParityInputs:
    """Shapes are ``[B, S]`` for the encoder and ``[B, K*C]`` for the canvas."""

    clean_input_ids: torch.Tensor
    canvas_ids: torch.Tensor
    encoder_position_ids: torch.Tensor
    decoder_position_ids: torch.Tensor
    prefix_lengths: torch.Tensor
    num_canvases: int
    canvas_length: int

    @property
    def batch_size(self) -> int:
        return int(self.clean_input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.clean_input_ids.shape[1])

    @property
    def response_length(self) -> int:
        return self.num_canvases * self.canvas_length


def build(
    *,
    vocab_size: int,
    canvas_length: int,
    num_canvases: int = 3,
    prefix_lengths: tuple[int, ...] = (5,),
    seed: int = 20260918,
) -> ParityInputs:
    """Draw the inputs from one seeded stream, so both venvs reconstruct them exactly.

    Every example shares one key axis, so ``S`` is sized by the longest prompt; with a
    single example that is just its own. Token ids start at 3 to stay clear of the
    pad/eos/bos ids, which some paths treat specially.
    """
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(prefix_lengths)
    response_length = num_canvases * canvas_length
    sequence_length = max(prefix_lengths) + response_length

    def ids(shape):
        return torch.randint(3, vocab_size, shape, generator=generator, dtype=torch.long)

    prefix = torch.tensor(prefix_lengths, dtype=torch.long)
    return ParityInputs(
        clean_input_ids=ids((batch_size, sequence_length)),
        # The canvas is the *noised* response, so it is drawn independently rather than
        # copied from the clean ids: identical halves would let a leak go unseen.
        canvas_ids=ids((batch_size, response_length)),
        encoder_position_ids=torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1).contiguous(),
        # The canvas continues the prompt it can attend to, matching generation.
        decoder_position_ids=(prefix.unsqueeze(-1) + torch.arange(response_length)).contiguous(),
        prefix_lengths=prefix,
        num_canvases=num_canvases,
        canvas_length=canvas_length,
    )


def per_canvas_context_mask(inputs: ParityInputs) -> torch.Tensor:
    """``[B, K, S]`` -- canvas ``k`` sees the clean cache up to its own block boundary.

    VeOmni's native way of being told the visibility. NeMo is told the same thing as a
    dense ``[B, 1, K*C, S + K*C]`` mask that it builds itself from the prompt length and
    block size; ``tests/models/test_diffusion_gemma_mask_matches_nemo.py`` is what says
    those two descriptions denote the same set.
    """
    boundary = inputs.prefix_lengths.unsqueeze(-1) + torch.arange(inputs.num_canvases) * inputs.canvas_length
    return torch.arange(inputs.sequence_length) < boundary.unsqueeze(-1)
