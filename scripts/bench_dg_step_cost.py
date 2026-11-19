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
"""What number should a DiffusionGemma token budget count?

A dynamic batch size needs one number per sample that predicts what the sample
will cost, so that a budget converts to time and memory. `DynBszBuffer` uses
`attention_mask.sum()`, which for DiffusionGemma is the encoder row `L = P + K*C`
-- the prompt plus every canvas. That is the encoder's whole input, so it cannot
be *missing* tokens. What it can be missing is the decoder, which runs a second
pass over the `K*C` canvas tokens against the encoder cache, and a third with
probability `self_conditioning_prob`.

So the question is empirical and has a shape: hold `L` fixed, move `K`, and see
whether the step time holds still. If it does, `L` is the budget and there is
nothing to fix. If it climbs with `K`, the budget has to charge the canvas
tokens more than once, and this measures how much more.

Two candidate measures are scored against the measurement:

* `L = P + K*C` -- what the buffer counts today.
* `P + (1 + d) * K*C` -- charging a canvas token once for the encoder and `d`
  times for the decoder passes it causes. `d` is fitted, not assumed.

Production attention geometry (2304 hidden, 25 sliding layers 256 wide with a
512 window to 5 full layers 512 wide) at reduced depth, with the MoE shape from
the toy config's ratios since the 26B checkpoint is not on this machine. Read the
ratios and the fitted exponent, not the absolute milliseconds.
"""

from __future__ import annotations

import argparse

import torch

from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import (
    DiffusionGemmaConfig,
    DiffusionGemmaTextConfig,
)
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion


WARMUP = 2
ITERATIONS = 5
CANVAS = 256


def build_model(layers: int, experts: int, self_conditioning: float, dtype=torch.bfloat16):
    """Production attention geometry, fewer layers, packing mode on.

    `self_conditioning_prob` is a *parameter* of the run here rather than left at
    its 0.5 default: it decides whether a third pass happens, so leaving it
    stochastic would put a coin flip inside the timing window and the fit would be
    measuring the coin. Run it at 0 and 1 and the expected cost is between them.
    """
    text_config = DiffusionGemmaTextConfig(
        num_hidden_layers=layers,
        num_experts=experts,
        top_k_experts=min(2, experts),
        moe_intermediate_size=1152,
        num_global_key_value_heads=2,
        vocab_size=8192,
    )
    config = DiffusionGemmaConfig(
        text_config=text_config,
        canvas_length=CANVAS,
        max_canvases=8,
        canvas_batching="flat_seq_pack",
        compute_all_canvas_df_loss=True,
        self_conditioning_prob=self_conditioning,
        encoder_ar_loss=True,
    )
    config._attn_implementation = "sdpa"
    config.text_config.attn_ffpa = False
    torch.manual_seed(0)
    model = DiffusionGemmaForBlockDiffusion(config).to(device="cuda", dtype=dtype)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train(), config


def packed_batch(specs, vocab_size, dtype):
    """One packed row holding `specs` as `(prompt, canvases)` pairs.

    Built here rather than through `DiffusionGemmaCollator` so the shapes are
    visible beside the numbers they explain; the collator's version is what
    `tests/models/test_diffusion_gemma_packed_collator.py` pins.
    """
    ids, positions, prompt_mask = [], [], []
    canvases, labels, weights, context = [], [], [], []
    total = sum(prompt + count * CANVAS for prompt, count in specs)
    offset = 0
    for prompt, count in specs:
        length = prompt + count * CANVAS
        ids.append(torch.randint(3, vocab_size, (length,)))
        positions.append(torch.arange(length))
        mask = torch.zeros(length, dtype=torch.bool)
        mask[:prompt] = True
        prompt_mask.append(mask)
        canvases.append(torch.randint(3, vocab_size, (count, CANVAS)))
        labels.append(torch.randint(3, vocab_size, (count, CANVAS)))
        weights.append(torch.full((count, CANVAS), 1.0 / (count * CANVAS)))
        for block in range(count):
            row = torch.zeros(total + CANVAS, dtype=torch.bool)
            row[offset : offset + prompt + block * CANVAS] = True
            row[total:] = True
            context.append(row)
        offset += length

    return {
        "input_ids": torch.cat(ids).unsqueeze(0).cuda(),
        "position_ids": torch.cat(positions).unsqueeze(0).cuda(),
        "attention_mask": torch.ones(1, total, dtype=torch.long).cuda(),
        "encoder_prompt_mask": torch.cat(prompt_mask).unsqueeze(0).cuda(),
        "decoder_input_ids": torch.cat(canvases).unsqueeze(0).cuda(),
        "decoder_attention_mask": torch.stack(context).unsqueeze(0).cuda(),
        "labels": torch.cat(labels).unsqueeze(0).cuda(),
        "loss_weight": torch.cat(weights).unsqueeze(0).to(dtype).cuda(),
    }


def measure(step):
    """Milliseconds per step and the memory allocated inside the timing window."""
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERATIONS):
        step()
    end.record()
    torch.cuda.synchronize()
    transient = (torch.cuda.max_memory_allocated() - baseline) / 2**20
    return start.elapsed_time(end) / ITERATIONS, transient


def run_case(model, config, specs, dtype):
    batch = packed_batch(specs, config.text_config.vocab_size, dtype)

    def step():
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()

    return measure(step)


def fit_decoder_charge(rows):
    """Least squares for `time = c0 + a*P + b*K*C`, and `d = b/a - 1` from it.

    The intercept has to be in the fit and out of the budget. It is the per-step
    launch and reduction overhead, which a micro-batch pays once however many
    samples it holds -- so charging it per sample would make short samples look
    arbitrarily expensive and shrink the batch for no reason. Fitting without it
    instead smears it into the token coefficients, and since the cases at small
    `L` are where it dominates, that biases exactly the ratio being measured.

    The cases hold `L` fixed within a group for the same reason of separation: a
    fit over cases whose `L` also moves cannot tell "the decoder is expensive"
    from "longer rows are expensive".
    """
    prompts = torch.tensor([float(p) for p, _, _ in rows])
    canvas_tokens = torch.tensor([float(k * CANVAS) for _, k, _ in rows])
    times = torch.tensor([t for _, _, t in rows])
    design = torch.stack([torch.ones_like(prompts), prompts, canvas_tokens], dim=1)
    solution = torch.linalg.lstsq(design, times.unsqueeze(1)).solution.flatten()
    intercept, a, b = (float(value) for value in solution)
    residual = float((design @ solution - times).abs().max())
    if a <= 0:
        return None
    return {"intercept": intercept, "prompt": a, "canvas": b, "charge": b / a - 1.0, "residual": residual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--self-conditioning", type=float, default=0.0)
    args = parser.parse_args()

    dtype = torch.bfloat16
    model, config = build_model(args.layers, args.experts, args.self_conditioning, dtype)
    print(
        f"layers={args.layers} experts={args.experts} "
        f"self_conditioning_prob={args.self_conditioning} canvas={CANVAS}\n"
    )

    # Each group holds L fixed and moves K, which is the only way to see the
    # decoder's share: if today's budget were right, a group would be flat.
    groups = [
        ("L=2048", [(1792, 1), (1536, 2), (1024, 4), (256, 7)]),
        ("L=1024", [(768, 1), (512, 2), (0 + 256, 3)]),
        ("L=3072", [(2816, 1), (2048, 4), (1024, 8)]),
    ]

    rows = []
    for name, specs in groups:
        print(f"{name}  {'P':>6} {'K':>3} {'L':>6}   {'ms':>8} {'MB':>8}   {'ms/L':>8}")
        first = None
        for prompt, count in specs:
            length = prompt + count * CANVAS
            millis, memory = run_case(model, config, [(prompt, count)], dtype)
            rows.append((prompt, count, millis))
            first = first or millis
            print(
                f"{'':>{len(name)}}  {prompt:>6} {count:>3} {length:>6}   "
                f"{millis:>8.2f} {memory:>8.0f}   {millis / length:>8.4f}"
                f"    {millis / first:>5.2f}x vs K=1"
            )
        print()

    fit = fit_decoder_charge(rows)
    if fit is None:
        print("fit failed: the prompt coefficient came out non-positive, so the model does not hold")
        return
    charge = fit["charge"]
    print(f"fit: time = {fit['intercept']:.1f}ms + {fit['prompt'] * 1000:.3f}us*P + {fit['canvas'] * 1000:.3f}us*K*C")
    print(f"  worst residual {fit['residual']:.1f}ms")
    print(f"  fitted decoder charge d = {charge:.2f}: today's budget counts P + 1*K*C, the fit says P + {1 + charge:.2f}*K*C")

    # Score both measures by how well a fixed budget predicts the *token* part of
    # the time -- the intercept removed, since a micro-batch pays it once and the
    # per-sample measure is not what should carry it. A budget is only useful if
    # this ratio holds still, which is what makes it convertible to a time cap.
    print("\n  ms per budget unit, per-step overhead removed:")
    for multiplier in (1.0, 1.5, 2.0, 2.5, 1 + charge, 3.0, 3.5):
        ratios = torch.tensor(
            [(t - fit["intercept"]) / (p + multiplier * k * CANVAS) for p, k, t in rows]
        )
        spread = float(ratios.max() / ratios.min())
        note = ""
        if abs(multiplier - 1.0) < 1e-9:
            note = "  <- today"
        elif abs(multiplier - (1 + charge)) < 1e-9:
            note = "  <- fitted"
        print(f"  P + {multiplier:.2f}*K*C: spans {spread:.2f}x{note}")


if __name__ == "__main__":
    main()
