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
"""Freeze real Sudoku rows into the one input tensor set both stacks are fed.

``parity_inputs.build`` draws token ids at random, which is enough to compare two
implementations as functions but says nothing about the geometry the recipe runs.
This writes the same ``ParityInputs`` structure from the actual SFT data, so the
comparison happens at the shape the 26B model is trained on: a 221-token prompt, a
163-token supervised answer padded into one 256-token canvas, and a 477-token
encoder row.

Tokenisation is inlined rather than imported from either stack. VeOmni reaches it
through ``DiffusionGemmaSudokuOfficialTemplate`` and NeMo through a Jinja file, and
the point of this file is to be neither -- a comparison whose input is built by one
of the two stacks cannot say the other was fed the same thing. ``--check-templates``
asserts the inlined form agrees with both.

Corruption follows ``corrupt_uniform_random``: DiffusionGemma has no mask token, so
corrupted positions are drawn uniformly over the vocabulary, matching the
checkpoint's own canvas initialisation.

    python scripts/parity/make_real_inputs.py \
        --jsonl /mnt/hdfs/user_dir/text_diff/data/sudoku/train.jsonl \
        --tokenizer /mnt/hdfs/user_dir/omni_v2/ckpt/diffusiongemma-26B-A4B-it \
        --rows 2 --out /tmp/dg_parity/real_inputs.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_inputs import ParityInputs  # noqa: E402


SYSTEM = (
    "Solve the following Sudoku puzzle. Empty cells are represented by 0. "
    "Output ONLY the solved puzzle immediately as a 9x9 grid of numbers separated by spaces. "
    "Do not include ####, explanations, or any other text."
)


def encode_row(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """``(prompt_ids, answer_ids)`` for one row, by the aligned Sudoku boundary.

    Mirrors ``veomni.data.chat_template.DiffusionGemmaSudokuOfficialTemplate``. The
    answer carries the turn close, which is supervised: the model has to learn to
    stop, and leaving it out would make the last canvas position an unsupervised
    pad in a way training never sees.
    """
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"expected system/user/assistant, got {roles}")
    if messages[0].get("content", "").strip() != SYSTEM:
        raise ValueError("the row's system instruction is not the released SFT prompt")

    prompt = f"<|turn>system\n{SYSTEM}<turn|>\n<|turn>user\n{messages[1]['content'].strip()}<turn|>\n<|turn>model\n"
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(messages[2]["content"].strip() + "<turn|>\n", add_special_tokens=False)
    return prompt_ids, answer_ids


def build_real(
    rows: list[dict],
    tokenizer,
    *,
    canvas_length: int,
    vocab_size: int,
    eps: float = 1e-3,
    seed: int = 20260920,
) -> tuple[ParityInputs, torch.Tensor]:
    """``(inputs, supervised_mask)`` -- the canvas layout the Sudoku recipe runs.

    Every row is one canvas: the answer is shorter than ``canvas_length`` and the
    remainder is pad, which is exactly what training feeds. ``supervised_mask`` is
    ``[B, canvas_length]`` and marks the answer positions, so the loss the two
    stacks are differentiated through covers the tokens the recipe supervises and
    not the pad.
    """
    encoded = [encode_row(tokenizer, row["messages"]) for row in rows]
    prefix_lengths = [len(prompt) for prompt, _ in encoded]
    if max(len(answer) for _, answer in encoded) > canvas_length:
        raise ValueError("an answer does not fit one canvas; the Sudoku recipe assumes it does")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_size = len(rows)
    sequence_length = max(prefix_lengths) + canvas_length

    clean = torch.full((batch_size, sequence_length), pad_id, dtype=torch.long)
    supervised = torch.zeros((batch_size, canvas_length), dtype=torch.bool)
    for row, (prompt, answer) in enumerate(encoded):
        clean[row, : len(prompt)] = torch.tensor(prompt, dtype=torch.long)
        clean[row, len(prompt) : len(prompt) + len(answer)] = torch.tensor(answer, dtype=torch.long)
        supervised[row, : len(answer)] = True

    # The clean response, as the canvas sees it before noise: answer then pad.
    response = torch.stack(
        [clean[row, prefix_lengths[row] : prefix_lengths[row] + canvas_length] for row in range(batch_size)]
    )

    # `corrupt_uniform_random` with `block_size == canvas_length`: one corruption
    # level per canvas, drawn from U(eps, 1), applied only to supervised positions.
    generator = torch.Generator().manual_seed(seed)
    level = eps + (1.0 - eps) * torch.rand((batch_size, 1), generator=generator)
    corrupted = (torch.rand((batch_size, canvas_length), generator=generator) < level) & supervised
    replacement = torch.randint(0, vocab_size, (batch_size, canvas_length), generator=generator, dtype=torch.long)
    canvas_ids = torch.where(corrupted, replacement, response)

    prefix = torch.tensor(prefix_lengths, dtype=torch.long)
    inputs = ParityInputs(
        clean_input_ids=clean,
        canvas_ids=canvas_ids,
        encoder_position_ids=torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1).contiguous(),
        decoder_position_ids=(prefix.unsqueeze(-1) + torch.arange(canvas_length)).contiguous(),
        prefix_lengths=prefix,
        num_canvases=1,
        canvas_length=canvas_length,
    )
    return inputs, supervised


def check_templates(rows, tokenizer, encoded, veomni_repo: Path | None, nemo_template: Path | None) -> None:
    """Assert the inlined boundary is the one both stacks actually tokenize to.

    Cheap, and it is the assumption the whole comparison rests on: if the ids differ
    the two stacks were never fed the same example, and every number downstream is
    about two different inputs rather than two implementations.
    """
    if veomni_repo is not None:
        sys.path.insert(0, str(veomni_repo))
        from veomni.data.chat_template import build_chat_template

        template = build_chat_template("diffusion_gemma_sudoku_official", tokenizer)
        for row, (prompt, answer) in zip(rows, encoded):
            got = template.encode_messages(row["messages"], max_seq_len=8192)["input_ids"]
            if got != prompt + answer:
                raise AssertionError("VeOmni's Sudoku template disagrees with the inlined boundary")
        print(f"  VeOmni template: agrees on {len(rows)} rows")

    if nemo_template is not None:
        rendered_template = nemo_template.read_text()
        for row, (prompt, answer) in zip(rows, encoded):
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, chat_template=rendered_template)
            got = tokenizer.encode(text, add_special_tokens=False)
            if got != prompt + answer:
                raise AssertionError(
                    f"NeMo's jinja template disagrees with the inlined boundary "
                    f"({len(got)} ids against {len(prompt) + len(answer)})"
                )
        print(f"  NeMo jinja template: agrees on {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2, help="examples in the batch")
    parser.add_argument("--skip", type=int, default=0, help="rows to skip before taking --rows")
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-3, help="the recipe's training_noise_eps")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--check-templates", action="store_true")
    parser.add_argument("--veomni-repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--nemo-template", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    # Straight out of the file rather than through `AutoConfig`: this module is meant
    # to be readable from either venv, and only one of the two registers the
    # `diffusion_gemma` config class with transformers.
    vocab_size = json.loads((args.tokenizer / "config.json").read_text())["text_config"]["vocab_size"]

    rows = []
    with args.jsonl.open() as handle:
        for index, line in enumerate(handle):
            if index < args.skip:
                continue
            if len(rows) == args.rows:
                break
            rows.append(json.loads(line))

    encoded = [encode_row(tokenizer, row["messages"]) for row in rows]
    print(f"read {len(rows)} rows from {args.jsonl}")
    for index, (prompt, answer) in enumerate(encoded):
        print(
            f"  row {args.skip + index}: prompt {len(prompt)}, answer {len(answer)}, total {len(prompt) + len(answer)}"
        )

    if args.check_templates:
        check_templates(rows, tokenizer, encoded, args.veomni_repo, args.nemo_template)

    inputs, supervised = build_real(
        rows, tokenizer, canvas_length=args.canvas_length, vocab_size=vocab_size, eps=args.eps, seed=args.seed
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "clean_input_ids": inputs.clean_input_ids,
            "canvas_ids": inputs.canvas_ids,
            "encoder_position_ids": inputs.encoder_position_ids,
            "decoder_position_ids": inputs.decoder_position_ids,
            "prefix_lengths": inputs.prefix_lengths,
            "num_canvases": inputs.num_canvases,
            "canvas_length": inputs.canvas_length,
            "supervised_mask": supervised,
            "source": {"jsonl": str(args.jsonl), "skip": args.skip, "rows": args.rows, "seed": args.seed},
        },
        args.out,
    )
    corrupted = int(
        (
            inputs.canvas_ids
            != torch.stack(
                [
                    inputs.clean_input_ids[
                        r, int(inputs.prefix_lengths[r]) : int(inputs.prefix_lengths[r]) + args.canvas_length
                    ]
                    for r in range(inputs.batch_size)
                ]
            )
        ).sum()
    )
    print(
        f"wrote {args.out}: encoder {tuple(inputs.clean_input_ids.shape)}, "
        f"canvas {tuple(inputs.canvas_ids.shape)}, supervised {int(supervised.sum())}, corrupted {corrupted}"
    )


if __name__ == "__main__":
    main()
