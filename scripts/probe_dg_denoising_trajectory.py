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
"""How many denoising steps a checkpoint needs, and what its grid looks like on the way.

Two DiffusionGemma checkpoints trained identically apart from `attn_dg_prefix`
score the same accuracy and hold the same fixed point, yet one of them drops a
token from a sixth to two-fifths of its Sudoku answers (constraint 40). The
difference is not in the conditional distribution -- it is that the sampler
needs roughly twice as many steps to satisfy its stopping rule, and a longer
trajectory is more likely to terminate on a grid whose rows are misaligned.

Step count is the sensitive measure. Over 36 generations per arm it separated
the two at Mann-Whitney z=5.5 where the malformed-output rate they cause managed
only p=0.011, so this is the cheap probe to reach for when asking whether a
change hurt the model -- far cheaper than scoring a format rate to the same
confidence.

`EntropyBoundSampler.accept_canvas` is wrapped to keep each step's canvas, which
is the real token sequence rather than a re-tokenization of the finished text:
a fused cell like `16` need not split the way the canvas held it, and measuring
the decoded string instead introduces that confound into the one quantity the
probe is about.

Needs a `transformers` that ships `models/diffusion_gemma` (the sampler lives
there). VeOmni 5.16.1 does; this script still prefers the HuggingFace class for
the sampler rather than the vendored training model:

    python scripts/probe_dg_denoising_trajectory.py --checkpoint <hf dir> \
        --rows 0 1 2 3 --seeds 0 1 2
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer


try:
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import EntropyBoundSampler
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion
except ImportError as error:  # pragma: no cover - environment, not logic
    raise SystemExit(
        f"this probe needs a transformers that ships models/diffusion_gemma; the installed one does not ({error})"
    ) from error

EVAL_JSONL = "/mnt/hdfs/user_dir/text_diff/data/sudoku/eval.jsonl"

#: verl's specification of the released Sudoku prompt. Rendered literally and
#: checked against the tokenizer's chat template, so a template drift fails here
#: rather than showing up as a quietly different trajectory.
OFFICIAL_SUDOKU_SYSTEM_PROMPT = (
    "Solve the following Sudoku puzzle. Empty cells are represented by 0. "
    "Output ONLY the solved puzzle immediately as a 9x9 grid of numbers separated by spaces. "
    "Do not include ####, explanations, or any other text."
)

WELL_FORMED = ",".join(["9"] * 9)


def load_rows(path: str, count: int) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if len(rows) >= count:
                break
            by_role = {message["role"]: message["content"] for message in json.loads(line)["messages"]}
            rows.append({"system": by_role["system"], "user": by_role["user"], "answer": by_role["assistant"]})
    return rows


def encode_prompt(tokenizer, row: dict) -> list[int]:
    if row["system"].strip() != OFFICIAL_SUDOKU_SYSTEM_PROMPT:
        raise ValueError("unexpected Sudoku system instruction")

    templated = tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    if not isinstance(templated, list):
        templated = templated["input_ids"]

    rendered = (
        f"<|turn>system\n{OFFICIAL_SUDOKU_SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n{row['user'].strip()}<turn|>\n"
        "<|turn>model\n"
    )
    expected = [tokenizer.bos_token_id, *tokenizer.encode(rendered, add_special_tokens=False)]
    if templated != expected:
        raise ValueError(f"prompt does not match the released Sudoku form ({len(templated)} against {len(expected)})")
    return templated


def grid_shape(text: str) -> str:
    """Cells per row, which is the structure the format metric rejects on."""
    return ",".join(str(len(line.split())) for line in text.strip("\n").split("\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-jsonl", default=EVAL_JSONL)
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-denoising-steps", type=int, default=48)
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--per-step", action="store_true", help="print each step's canvas shape")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint, dtype="bfloat16", device_map="cuda")
    model.eval()
    device = next(model.parameters()).device
    eos_id = model.generation_config.eos_token_id
    eos_id = eos_id[0] if isinstance(eos_id, (list, tuple)) else eos_id
    pad_id = model.generation_config.pad_token_id if model.generation_config.pad_token_id is not None else eos_id

    rows = load_rows(args.eval_jsonl, max(args.rows) + 1)
    trace: list[dict] = []
    original = EntropyBoundSampler.accept_canvas

    def recording_accept(self, current_canvas, denoiser_canvas, logits, cur_step):
        accepted = original(self, current_canvas, denoiser_canvas, logits, cur_step)
        trace.append(
            {
                "step": len(trace),
                "accepted_positions": int(self.accepted_token_mask.sum()),
                "mean_entropy": float(torch.distributions.Categorical(logits=logits[0]).entropy().mean()),
                "canvas": tokenizer.decode(accepted[0], skip_special_tokens=True),
            }
        )
        return accepted

    EntropyBoundSampler.accept_canvas = recording_accept
    records = []
    try:
        for index in args.rows:
            for seed in args.seeds:
                trace.clear()
                prompt = torch.tensor(encode_prompt(tokenizer, rows[index]), device=device).unsqueeze(0)
                torch.manual_seed(seed)
                out = model.generate(
                    input_ids=prompt,
                    max_new_tokens=args.canvas_length,
                    max_denoising_steps=args.max_denoising_steps,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
                final = tokenizer.decode(out.sequences[0, prompt.shape[1] :], skip_special_tokens=True)
                shape = grid_shape(final)

                verdict = "ok" if shape == WELL_FORMED else "MALFORMED"
                print(f"row {index:3d} seed {seed:3d}  steps {len(trace):3d}  {verdict:9s}  {shape}")
                if args.per_step:
                    for entry in trace:
                        print(
                            f"    step {entry['step']:3d}  accepted {entry['accepted_positions']:4d}  "
                            f"mean H {entry['mean_entropy']:7.4f}  {grid_shape(entry['canvas'])}"
                        )
                records.append({"row": index, "seed": seed, "steps": len(trace), "shape": shape, "final": final})
    finally:
        EntropyBoundSampler.accept_canvas = original

    steps = [record["steps"] for record in records]
    malformed = [record for record in records if record["shape"] != WELL_FORMED]
    print(f"\n=== {args.checkpoint}")
    print(f"generations  {len(records)}")
    print(f"malformed    {len(malformed)} ({len(malformed) / len(records):.1%})")
    print(f"steps        mean {statistics.mean(steps):.2f}  median {statistics.median(steps)}  max {max(steps)}")
    if malformed:
        print(f"             malformed runs took {sorted(record['steps'] for record in malformed)}")

    if args.output:
        # Which checkpoint produced a trajectory is the whole point of comparing
        # two of them, and reading it back off the file's timestamp is not an
        # answer. Everything that changes the numbers goes in beside the records.
        payload = {
            "checkpoint": args.checkpoint,
            "eval_jsonl": args.eval_jsonl,
            "rows": args.rows,
            "seeds": args.seeds,
            "max_denoising_steps": args.max_denoising_steps,
            "canvas_length": args.canvas_length,
            "generations": records,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"per-generation -> {args.output}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
