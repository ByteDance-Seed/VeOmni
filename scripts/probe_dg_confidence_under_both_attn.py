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
"""Is the kernel-trained checkpoint less confident, or only measured that way?

Constraint 41 records that the `attn_dg_prefix` arm needs roughly twice as many
denoising steps and holds a noisy canvas with a thinner margin. Every one of
those numbers was taken through the HuggingFace `DiffusionGemmaForBlockDiffusion`
forward, which resolves its decoder attention through `ALL_ATTENTION_FUNCTIONS`
and therefore never runs the Triton hole kernel. So the kernel arm has only ever
been measured under an attention implementation it did not train with, while the
sdpa arm has train/measure agreement. Two mechanisms fit that asymmetry:

* **mismatch** -- the weights are fine and are simply being run through a
  different forward than they were fitted to.
* **flatter minimum** -- the kernel's noisier bf16 gradients regularise training
  toward a genuinely less confident solution.

They are told apart by measuring the same weights under both forwards. Mismatch
puts the two arms level on their own diagonal (kernel weights + kernel forward
against sdpa weights + sdpa forward) and puts the whole gap off-diagonal. A
flatter minimum follows the weights: the kernel checkpoint reads less confident
under *either* forward.

The same run separates bias from rounding noise, which no previous probe did:
the per-position margin difference between the two forwards on one set of
weights is reported signed, with a standard error, so a systematic shift is
distinguishable from a wash.

Teacher-forced, no sampling: the canvas is the gold answer, optionally corrupted
at a fixed fraction to walk back along the denoising trajectory, and the margin
is `logit[gold] - max(logit[other])` at each answer position.

    for arm in true false; do
        python scripts/probe_dg_confidence_under_both_attn.py \
            --checkpoint /opt/tiger/dg_eval_seed1234-dgprefix-$arm --out /tmp/$arm.json
    done
    python scripts/probe_dg_confidence_under_both_attn.py --compare /tmp/true.json /tmp/false.json

Runs on one GPU in bfloat16 -- the dtype training used, and the only dtype in
which the two attention paths differ at all.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import torch
from transformers import AutoTokenizer


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma as modeling  # noqa: E402
from veomni.arguments.arguments_types import OpsImplementationConfig  # noqa: E402
from veomni.data.data_collator import DiffusionGemmaCollator  # noqa: E402
from veomni.models import build_foundation_model  # noqa: E402
from veomni.models.module_utils import load_model_weights  # noqa: E402
from veomni.models.transformers.diffusion_gemma.parallel_plan import retie_encoder_to_decoder  # noqa: E402


EVAL_JSONL = "/mnt/hdfs/user_dir/text_diff/data/sudoku/eval.jsonl"
OFFICIAL_SUDOKU_SYSTEM_PROMPT = (
    "Solve the following Sudoku puzzle. Empty cells are represented by 0. "
    "Output ONLY the solved puzzle immediately as a 9x9 grid of numbers separated by spaces. "
    "Do not include ####, explanations, or any other text."
)

# The Sudoku recipe's ops, held fixed across every cell of the 2x2 so that the
# only thing varying is the decoder's attention.
SUDOKU_OPS = OpsImplementationConfig(attn_implementation="sdpa", moe_implementation="fused_triton")


def load_on_one_gpu(checkpoint: str):
    """Load the 26B checkpoint onto a single device with its parameter ties intact.

    The encoder's trunk is the decoder's: 628 of the 1613 parameter names are
    aliases, which is why the checkpoint carries only 985 tensors. `to_empty`
    inside the loader rebinds every name to its own storage, so a straight
    `init_device="cuda"` load asks for 96 GiB of the 80 GiB card and leaves the
    encoder's copies unwritten besides -- the loader only fills the canonical
    names. Training does not hit this because the parallel plan re-ties after
    sharding; here the same `retie_encoder_to_decoder` does it on host memory,
    and the move to the device has to carry a memo or `nn.Module._apply` would
    split the ties straight back apart.
    """
    # `build_foundation_model` decides `empty_init` from the parallel state's rank,
    # which outside a launcher is not 0, so passing `weights_path` with a host
    # `init_device` would quietly hand back an unwritten model. The load is called
    # here instead, and the meta check below is what makes that failure loud.
    model = build_foundation_model(
        config_path=checkpoint,
        torch_dtype="bfloat16",
        init_device="meta",
        ops_implementation=SUDOKU_OPS,
    )
    load_model_weights(model, checkpoint, "cpu")
    retie_encoder_to_decoder(model)  # a no-op when the loader already restored the ties
    names = [name for name, _ in model.named_parameters(remove_duplicate=False)]
    distinct = len(list(model.named_parameters()))
    if len(names) - distinct < 600:
        raise RuntimeError(f"only {len(names) - distinct} encoder names are tied to the decoder, expected 628")
    still_meta = [name for name, param in model.named_parameters(remove_duplicate=False) if param.is_meta]
    if still_meta:
        raise RuntimeError(f"{len(still_meta)} parameters never got weights, first is {still_meta[0]}")

    memo: dict[int, torch.Tensor] = {}
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            if id(param) not in memo:
                memo[id(param)] = torch.nn.Parameter(param.data.cuda(), requires_grad=False)
            module._parameters[name] = memo[id(param)]
        for name, buffer in list(module._buffers.items()):
            if buffer is None:
                continue
            if id(buffer) not in memo:
                memo[id(buffer)] = buffer.cuda()
            module._buffers[name] = memo[id(buffer)]

    # Train mode, not eval: `SealableCache` is built with `truncate_sliding=not
    # self.training`, and `flat_seq` refuses a truncated cache because the sliding
    # layers would then hold only its tail rather than the run each canvas is
    # entitled to. This is the training forward either way, which is the one whose
    # attention is under question. Deterministic only while dropout is off.
    if model.config.text_config.attention_dropout:
        raise RuntimeError("attention dropout is on, so the two arms would not see the same forward")
    return model.train()


def load_rows(path: str, limit: int) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            by_role = {m["role"]: m["content"] for m in json.loads(line)["messages"]}
            rows.append({"system": by_role["system"], "user": by_role["user"], "answer": by_role["assistant"]})
    return rows


def encode_prompt(tokenizer, row: dict) -> list[int]:
    """Render the prompt and fail closed on a chat-template drift."""
    if row["system"].strip() != OFFICIAL_SUDOKU_SYSTEM_PROMPT:
        raise ValueError("unexpected Sudoku system instruction")
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    return ids if isinstance(ids, list) else ids["input_ids"]


def position_class(piece: str) -> str:
    if piece == "\n":
        return "newline"
    if piece.strip() == "":
        return "space"
    if piece.strip().isdigit():
        return "digit"
    return "other"


def build_feature(tokenizer, row, canvas_length, vocab_size, eos_id, corrupt, seed):
    """One sample in the layout `diffusion_gemma`'s data transform emits.

    Encoder sequence is prompt followed by the clean canvas (Eq. 13's `P + K C`);
    `decoder_context_mask` hides that canvas from the decoder so it cannot read
    the answer it is reconstructing. Corruption is applied here rather than by
    the transform because both arms have to be asked about the *same* noise.
    """
    prompt_ids = encode_prompt(tokenizer, row)
    answer_ids = tokenizer.encode(row["answer"], add_special_tokens=False)
    if len(answer_ids) > canvas_length:
        raise ValueError(f"answer needs {len(answer_ids)} positions, over the canvas's {canvas_length}")
    gold = torch.tensor(answer_ids + [eos_id] * (canvas_length - len(answer_ids)), dtype=torch.long)

    canvas = gold.clone()
    corrupted = torch.zeros(len(answer_ids), dtype=torch.bool)
    if corrupt > 0:
        generator = torch.Generator().manual_seed(seed)
        count = int(round(corrupt * len(answer_ids)))
        picked = torch.randperm(len(answer_ids), generator=generator)[:count]
        canvas[picked] = torch.randint(0, vocab_size, (count,), generator=generator)
        corrupted[picked] = True

    encoder_ids = torch.cat([torch.tensor(prompt_ids, dtype=torch.long), gold])
    context = torch.zeros(1, encoder_ids.numel(), dtype=torch.long)
    context[0, : len(prompt_ids)] = 1
    prompt_mask = torch.zeros(encoder_ids.numel(), dtype=torch.bool)
    prompt_mask[: len(prompt_ids)] = True
    feature = {
        "input_ids": encoder_ids,
        "attention_mask": torch.ones(encoder_ids.numel(), dtype=torch.long),
        "encoder_prompt_mask": prompt_mask,
        "decoder_context_mask": context,
        "decoder_input_ids": canvas.unsqueeze(0),
        "labels": gold.unsqueeze(0),
        "loss_weight": torch.full((1, canvas_length), 1.0 / canvas_length),
    }
    return feature, gold[: len(answer_ids)], corrupted


def forward_margins(model, batch, kernel: bool) -> tuple[torch.Tensor, int]:
    """Logits for every canvas in the packed row, and how often the kernel ran."""
    model.config.canvas_batching = "flat_seq_pack"
    model.config.text_config.attn_dg_prefix = kernel

    calls = []
    original = modeling.hole_decoder_attention
    modeling.hole_decoder_attention = lambda *a, **kw: (calls.append(1), original(*a, **kw))[1]
    try:
        inputs = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        modeling.hole_decoder_attention = original
    return outputs.logits.float(), len(calls)


def measure(model, tokenizer, rows, canvas_length, eos_id, corrupt, pack, kernel, seed):
    """Per-position margin and entropy over every row, under one attention arm."""
    vocab_size = model.config.text_config.vocab_size
    collator = DiffusionGemmaCollator(pad_token_id=eos_id, pack_samples=True)
    per_class = collections.defaultdict(lambda: {"count": 0, "flips": 0, "margin": 0.0, "entropy": 0.0})
    margins: list[float] = []
    kernel_calls = 0

    for start in range(0, len(rows), pack):
        chunk = rows[start : start + pack]
        built = [
            build_feature(tokenizer, row, canvas_length, vocab_size, eos_id, corrupt, seed * 1000 + start + offset)
            for offset, row in enumerate(chunk)
        ]
        batch = collator([feature for feature, _, _ in built])
        logits, calls = forward_margins(model, batch, kernel)
        kernel_calls += calls
        # The packed row carries every sample's canvas on one axis; flatten
        # whatever leading shape the mode produced back to one row per canvas.
        logits = logits.reshape(-1, canvas_length, logits.shape[-1])
        if logits.shape[0] != len(built):
            raise RuntimeError(f"got {logits.shape[0]} canvases of logits for {len(built)} samples")

        for canvas_index, (_, gold, corrupted) in enumerate(built):
            span = logits[canvas_index, : gold.numel()]
            target = gold.to(span.device)
            gold_logit = span.gather(1, target.unsqueeze(1)).squeeze(1)
            runner_up = span.scatter(1, target.unsqueeze(1), float("-inf")).max(dim=1).values
            margin = (gold_logit - runner_up).cpu()
            entropy = -(span.softmax(-1) * span.log_softmax(-1)).sum(-1).cpu()
            margins.extend(margin.tolist())

            for position, token in enumerate(target.tolist()):
                kind = position_class(tokenizer.decode([token]))
                if corrupt > 0:
                    kind = f"{kind}/{'noised' if corrupted[position] else 'held'}"
                bucket = per_class[kind]
                bucket["count"] += 1
                bucket["margin"] += margin[position].item()
                bucket["entropy"] += entropy[position].item()
                bucket["flips"] += int(margin[position].item() < 0)

    return {"per_class": {k: dict(v) for k, v in per_class.items()}, "margins": margins, "kernel_calls": kernel_calls}


def summarise(label: str, result: dict) -> None:
    print(f"\n--- {label}   (hole-kernel calls: {result['kernel_calls']})")
    print(f"{'class':18s} {'n':>8s} {'flips':>7s} {'mean margin':>13s} {'mean entropy':>13s}")
    for kind in sorted(result["per_class"]):
        bucket = result["per_class"][kind]
        n = bucket["count"]
        print(f"{kind:18s} {n:8d} {bucket['flips']:7d} {bucket['margin'] / n:13.4f} {bucket['entropy'] / n:13.4f}")
    values = torch.tensor(result["margins"])
    print(f"{'ALL':18s} {values.numel():8d} {int((values < 0).sum()):7d} {values.mean():13.4f}")


def paired_shift(label: str, a: list[float], b: list[float]) -> dict:
    """Signed per-position difference: a systematic shift against rounding noise."""
    delta = torch.tensor(a) - torch.tensor(b)
    mean = delta.mean().item()
    stderr = (delta.std() / math.sqrt(delta.numel())).item()
    print(f"{label:52s} {mean:+9.4f} +- {stderr:.4f}   (|t| = {abs(mean) / stderr:6.1f}, n = {delta.numel()})")
    return {"mean": mean, "stderr": stderr, "n": delta.numel()}


def compare(paths: list[str]) -> None:
    """Lay the 2x2 out and say whether the gap tracks the weights or the forward."""
    runs = []
    for path in paths:
        with open(path) as handle:
            runs.append(json.load(handle))
    for a, b in zip(runs, runs[1:]):
        if (a["corrupt"], a["limit"], a["pack"], a["seed"]) != (b["corrupt"], b["limit"], b["pack"], b["seed"]):
            raise ValueError("the runs were asked different questions and cannot be compared")

    print(f"\n{'checkpoint':<44} {'kernel forward':>15} {'sdpa forward':>14} {'own - other fwd':>17}")
    for run in runs:
        kernel = torch.tensor(run["cells"]["kernel_fwd"]["margins"])
        sdpa = torch.tensor(run["cells"]["sdpa_fwd"]["margins"])
        delta = kernel - sdpa
        stderr = (delta.std() / math.sqrt(delta.numel())).item()
        name = os.path.basename(run["checkpoint"])
        print(f"{name:<44} {kernel.mean():15.4f} {sdpa.mean():14.4f} {delta.mean():+11.4f} +- {stderr:.4f}")

    if len(runs) != 2:
        return
    # Same positions and same corruption in both runs, so the checkpoints can be
    # compared position by position rather than mean against mean.
    print("\ngap between the two checkpoints, paired by canvas position:")
    for arm in ("kernel_fwd", "sdpa_fwd"):
        delta = torch.tensor(runs[0]["cells"][arm]["margins"]) - torch.tensor(runs[1]["cells"][arm]["margins"])
        stderr = (delta.std() / math.sqrt(delta.numel())).item()
        print(
            f"  under {arm:11s} {delta.mean():+9.4f} +- {stderr:.4f}   "
            f"(|t| = {abs(delta.mean().item()) / stderr:6.1f}, n = {delta.numel()})"
        )
    print(
        "\nA gap that holds under both forwards is in the weights; one that appears\n"
        "only when the forward is swapped is a train/measure mismatch."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="run both forwards over this checkpoint")
    parser.add_argument("--compare", nargs="+", help="lay out the 2x2 from previously written JSON")
    parser.add_argument("--limit", type=int, default=16, help="eval rows")
    parser.add_argument("--pack", type=int, default=2, help="samples per packed row; must be >= 2")
    parser.add_argument("--corrupt", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="/tmp/dgruns/confidence.json")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if not args.checkpoint:
        raise ValueError("pass --checkpoint to measure or --compare to read measurements back")
    if args.pack < 2:
        # A single canvas per row is squeezed to two dimensions before any mode
        # dispatch, which silently routes the decoder to sdpa no matter what
        # `attn_dg_prefix` says. That trap is what made `canvas_batching` look
        # like the cause of the defect in the first place.
        raise ValueError("--pack must be at least 2 or the kernel arm silently falls back to sdpa")

    # One checkpoint per process on purpose: the ops singleton is installed at the
    # first `build_foundation_model` and a second model in the same process would
    # inherit it, which is exactly the kind of difference this probe exists to rule
    # out. Run twice and pass both JSONs to `--compare`.
    rows = load_rows(EVAL_JSONL, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    print(f"\n=== loading {args.checkpoint}", flush=True)
    model = load_on_one_gpu(args.checkpoint)
    model.config.max_canvases = 1
    model.config.canvas_start = 0
    model.config.text_config.attn_ffpa = True

    cells = {}
    for arm, kernel in (("kernel_fwd", True), ("sdpa_fwd", False)):
        result = measure(
            model,
            tokenizer,
            rows,
            model.config.canvas_length,
            tokenizer.eos_token_id,
            args.corrupt,
            args.pack,
            kernel,
            args.seed,
        )
        if kernel and not result["kernel_calls"]:
            raise RuntimeError("the hole kernel never ran, so this compares sdpa against itself")
        if not kernel and result["kernel_calls"]:
            raise RuntimeError("the sdpa arm reached the hole kernel")
        cells[arm] = result
        summarise(f"{os.path.basename(args.checkpoint)} under {arm}", result)

    paired_shift(
        "\nsame weights, both forwards: margin(kernel) - margin(sdpa)",
        cells["kernel_fwd"]["margins"],
        cells["sdpa_fwd"]["margins"],
    )

    payload = {
        "checkpoint": args.checkpoint,
        "corrupt": args.corrupt,
        "limit": args.limit,
        "pack": args.pack,
        "seed": args.seed,
        "cells": cells,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
