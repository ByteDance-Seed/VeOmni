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
"""What does `flat_seq_pack` buy over `flat_seq` on the whole model?

Existing measurements leave a gap in the middle. `bench_packed_encoder.py` runs
three arms but only the encoder, so it cannot say what the decoder and the loss
do to the ratio. `bench_dg_step_cost.py` runs the whole model but only the
packed arm, since its question was what a token budget should count. Neither one
compares the two *modes* end to end, and neither goes through the collator that
training actually uses.

Three arms over the same samples, the same accumulated gradient, and the same
decoder code path -- only the shape of the micro-batch differs:

* **A** `flat_seq`, `micro_batch_size: 1`, N accumulation steps. What the recipe
  ships today, and therefore the baseline. Nothing is padded because a batch of
  one sample is its own longest.
* **B** `flat_seq_pack`, one packed row of all N. What `micro_batch_size: N`
  gives under the packing mode, with no token budget involved.
* **D** `flat_seq`, `micro_batch_size: N`, padded to the batch's longest row and
  largest canvas count. What raising `micro_batch_size` does *without* the
  packing mode, and the arm that keeps A honest: if D beats A, then most of what
  B wins is the wider micro-batch rather than the packing.

A vs B holds the accumulation boundary and the sample count fixed, so it bounds
what the mode is worth. B vs D is the "if you raise the batch size, pack or pad"
question -- but note that it is not a comparison in which *only* packing varies,
and no arrangement of these arms would be. Packing the encoder means the varlen
kernels: B runs `flash_attn_varlen_func` on the sliding layers and
`wide_varlen_attention` on the wide ones, while A and D go through sdpa against
a materialized mask. That is inseparable from the mode rather than a flaw in the
setup -- there is no way to have `flat_seq_pack` and sdpa -- so a residual
difference at equal lengths, where there is nothing to pack, is the kernels and
not the shape.

Note that "the same total work" is true of A and B and *not* of D, which runs
dummy canvases weighted zero. That is a real cost of the padded configuration
rather than a flaw in the arm, but it means D's disadvantage has two sources and
`padding_ratios` reports them separately.

Production attention geometry (2304 hidden, 25 sliding layers 256 wide with a
512 window to 5 full layers 512 wide) at reduced depth, `attn_dg_prefix` on as
the recipe has it, MoE shape from the toy config's ratios. Single GPU, so no
FSDP2 and no expert parallelism -- which cuts both ways and is worth knowing
before quoting anything: production re-all-gathers parameters every micro-step,
which A pays N times and understates B here, while production's wider MoE GEMMs
are better occupied at `micro_batch_size: 1` than this expert width, which
overstates B. Read the ratios, not the milliseconds.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager

import torch

import veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma as modeling
from veomni.data.data_collator import DiffusionGemmaCollator
from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import (
    DiffusionGemmaConfig,
    DiffusionGemmaTextConfig,
)
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion


WARMUP = 2
ITERATIONS = 5
CANVAS = 256
EOS = 1

#: `(prompt_tokens, canvases)` per sample.
#:
#: Two rules these have to obey, both learned the hard way. **Every sample needs
#: `canvases >= 2`**: the padded collator squeezes a `max_canvases == 1` batch to
#: `(B, C)`, which `_decode_canvases` routes to the plain decoder instead of
#: `_decode_canvases_flat_seq`, while the packed row is always `(1, ΣK, C)`. A
#: case with `K = 1` anywhere therefore compares two different decoder
#: implementations and calls the difference packing. `assert_one_decoder_path`
#: enforces it, but keep it true here so the assertion never has to fire.
#:
#: And **the two padding axes have to be separated**, because a padded batch pays
#: for both -- prompts to the longest row and canvases to the largest `K` -- so a
#: single "padding ratio" cannot attribute what packing saved. The `prompt
#: ragged` cases hold `K` fixed so only the encoder axis is ragged; `both ragged`
#: is what real data looks like; `equal` is the control that pads nothing.
CASES = {
    "prompt ragged 4x": (((1792, 2), (256, 2), (512, 2), (128, 2)), 4),
    "prompt ragged 8x": (((1792, 2), (256, 2), (1024, 2), (128, 2), (768, 2), (384, 2), (1536, 2), (256, 2)), 8),
    "both ragged 8x": (((768, 2), (128, 3), (1024, 2), (256, 5), (512, 2), (128, 4), (896, 2), (256, 3)), 8),
    "equal 8x": (((512, 2),) * 8, 8),
}


def build_model(layers: int, experts: int, dtype=torch.bfloat16):
    """Production attention geometry, fewer layers. The mode is set per arm.

    `attn_dg_prefix` is **on**, matching the recipe, and this is not a detail.
    Left at its `False` default the decoder falls back to a dense hole mask,
    which scores every canvas against the *whole* encoder row -- so a packed row
    holding N samples pays N times the decoder attention that the same samples
    pay apart. That is constraint 32's waste, on the arm the benchmark is trying
    to measure, in a configuration production does not run. It would show up as
    packing being slow and be reported as packing's cost.

    `attn_ffpa` is off, which unlike `attn_dg_prefix` is *not* what the recipe
    does, and the asymmetry favours the packed arm. The packed encoder returns
    before `resolve_attention_interface` (constraint 33), so B cannot reach FFPA
    however it is configured; A and D can, and constraint 28 measures it at 2.0x
    the time and a fifth of the memory on the wide layers. Turning it off for
    everyone therefore deletes an advantage A and D would have in production
    rather than levelling the arms.

    Self-conditioning is absent, and no flag here can add it. The model
    self-conditions only when handed `self_conditioning_logits`; the *trainer*
    is what draws the Bernoulli from `self_conditioning_prob` and passes them.
    Driving the model directly, as this does, means every arm runs two decoder
    passes rather than the recipe's expected 2.5, equally -- so the ratios stand
    and the absolute times are short of a real step.
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
        canvas_batching="flat_seq",
        compute_all_canvas_df_loss=True,
        encoder_ar_loss=True,
    )
    config._attn_implementation = "sdpa"
    config.text_config.attn_ffpa = False
    config.text_config.attn_dg_prefix = True
    torch.manual_seed(0)
    model = DiffusionGemmaForBlockDiffusion(config).to(device="cuda", dtype=dtype)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train(), config


def features(specs, vocab_size, dtype, seed=0):
    """The per-sample dicts the data transform hands the collator.

    Mirrors the transform's contract, and the same helper the collator tests use:
    `input_ids` is the prompt followed by every canvas, `decoder_context_mask` has
    one row per canvas holding the prompt plus the canvases before it, and
    `loss_weight` normalizes within the sample -- so a sample contributes the same
    whichever arm runs it, which is what makes the three comparable at all.
    """
    generator = torch.Generator().manual_seed(seed)
    built = []
    for index, (prompt, canvases) in enumerate(specs):
        length = prompt + canvases * CANVAS
        input_ids = torch.randint(3, vocab_size, (length,), generator=generator)
        input_ids[length - (index + 2) :] = EOS
        context = torch.zeros(canvases, length, dtype=torch.long)
        for block in range(canvases):
            context[block, : prompt + block * CANVAS] = 1
        prompt_mask = torch.zeros(length, dtype=torch.bool)
        prompt_mask[:prompt] = True
        built.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones(length, dtype=torch.long),
                "encoder_prompt_mask": prompt_mask,
                "decoder_context_mask": context,
                "decoder_input_ids": torch.randint(3, vocab_size, (canvases, CANVAS), generator=generator),
                "labels": torch.randint(3, vocab_size, (canvases, CANVAS), generator=generator),
                "loss_weight": torch.full((canvases, CANVAS), 1.0 / (canvases * CANVAS), dtype=dtype),
            }
        )
    return built


def on_cuda(batch):
    return {key: value.cuda() for key, value in batch.items() if key != "num_supervised_examples"}


@contextmanager
def mode(model, canvas_batching):
    """The arm has to say which mode it is; the model reads it, not an argument.

    A batch of one sample is not a packed batch -- the padded collator emits no
    `position_ids` and the packed mode refuses a row whose boundaries it cannot
    read -- so running arm A under `flat_seq_pack` would raise rather than
    quietly measure the wrong thing. That refusal is why this is a context
    manager and not a flag passed to `forward`.
    """
    previous = model.config.canvas_batching
    model.config.canvas_batching = canvas_batching
    try:
        yield
    finally:
        model.config.canvas_batching = previous


@contextmanager
def counting_varlen_calls():
    """Count `varlen_encoder_attention` calls, because an arm can refuse silently.

    `should_use_varlen_encoder` declines for a list of reasons (constraint 33),
    and a declining packed arm falls back to a block-diagonal mask -- which
    constraint 32 measures at 5.9-12.1x the kernel. That arm would still run, and
    the number it produced would be attributed to packing. So the packed arm has
    to be seen calling the kernel or its timing is meaningless.
    """
    calls = {"flash": 0, "wide": 0}
    # The modeling module imported the name, so that binding is the call site.
    original = modeling.varlen_encoder_attention

    def counted(query, *args, **kwargs):
        # Split by head width the way the sibling script does: a total that
        # happens to match would not notice the sliding/full division moving.
        calls["wide" if query.shape[-1] > 256 else "flash"] += 1
        return original(query, *args, **kwargs)

    modeling.varlen_encoder_attention = counted
    try:
        yield calls
    finally:
        modeling.varlen_encoder_attention = original


def assert_one_decoder_path(batches_by_arm):
    """Every arm has to reach the decoder through the same branch.

    `_decode_canvases` dispatches on rank: `(B, C)` goes to the plain decoder and
    `(B, K, C)` to the canvas walk. The padded collator squeezes to `(B, C)`
    whenever the batch's largest `K` is 1, and the packed collator never does --
    so a `K = 1` sample silently puts the arms on different implementations,
    which is not a difference between packing and padding and must not be
    reported as one. Fail closed here rather than print a caveat: the number is
    not weaker in that case, it is measuring something else.
    """
    ranks = {arm: batch["decoder_input_ids"].ndim for arm, batch in batches_by_arm.items()}
    # Every one of A's batches, not just the first: A collates each sample alone,
    # so a single `K = 1` sample anywhere makes that micro-batch rank 2 while the
    # rest of the arm, and D's batch, stay rank 3 and the check would pass.
    if len(set(ranks.values())) > 1:
        raise AssertionError(
            f"arms reached the decoder through different branches: {ranks}. Every sample in the case "
            "needs `canvases >= 2`, or the padded arms squeeze to the plain decoder while the packed "
            "arm walks the canvas axis."
        )


def measure(step):
    """Milliseconds per unit of work, plus the memory it takes to get there.

    Both memory figures, because they answer different questions and constraint
    32 was written after quoting only one of them misled. `transient` is what the
    step allocates over what is already resident; the baseline is read between
    steps, with the previous step's gradients live, so it approximates the
    activation footprint but includes the gradients being freed and reallocated.
    `total` is the bare peak, so it carries the parameters and gradients too --
    an arm-independent constant of a gigabyte or two that pulls every ratio
    towards 1. Compare arms on `transient`; read `total` only to check that no
    arm is near the card.
    """
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
    peak = torch.cuda.max_memory_allocated()
    return {
        "ms": start.elapsed_time(end) / ITERATIONS,
        "transient": (peak - baseline) / 2**20,
        "total": peak / 2**20,
    }


def run_arm(model, name, canvas_batching, batches, spy=False):
    """One arm: forward+backward every batch it was given, gradients accumulated.

    Arm A gets N batches and B and D get one, which is the whole point -- the
    unit of work is the N samples, not the call.
    """
    resident = [on_cuda(batch) for batch in batches]

    def step():
        model.zero_grad(set_to_none=True)
        for batch in resident:
            model(**batch).loss.backward()

    with mode(model, canvas_batching):
        if spy:
            text_config = model.config.text_config
            with counting_varlen_calls() as calls, torch.no_grad():
                model(**resident[0])
            wide = sum(kind == "full_attention" for kind in text_config.layer_types)
            expected = {"flash": text_config.num_hidden_layers - wide, "wide": wide}
            if calls != expected:
                raise AssertionError(
                    f"the packed arm reached the varlen kernels {calls} times, not {expected}. A "
                    "`should_use_varlen_encoder` refusal falls back to a block-diagonal mask, which is "
                    "5.9-12.1x the kernel -- timing that and calling it packing is what this checks for."
                )
            varlen_calls = f"{calls['flash']}f+{calls['wide']}w"
        else:
            varlen_calls = None
        model.zero_grad(set_to_none=True)
        # Summed over the arm's batches, so the three arms' figures are the same
        # quantity: one number per unit of work, not per call.
        with torch.no_grad():
            loss = sum(float(model(**batch).loss) for batch in resident)
        result = measure(step)

    # Emptied before the next arm rather than left to go out of scope: `total` is an
    # absolute peak, so one arm's inputs still resident would be charged to the next.
    resident.clear()
    torch.cuda.empty_cache()
    result.update(arm=name, loss=loss, varlen_calls=varlen_calls)
    return result


def padding_ratios(specs, count):
    """What the padded arm pays extra on each axis, reported separately.

    `_pad_canvas_stack` pads the canvas axis to the batch's largest `K` and the
    dummy canvases run a forward and a backward, weighted zero. So the padded arm
    has two sources of waste, and quoting only the encoder one -- as the first
    version of this benchmark did -- attributes a saving on the canvas axis to
    the encoder's raggedness and gets the story backwards.
    """
    rows = [prompt + canvases * CANVAS for prompt, canvases in specs]
    canvases = [canvases for _, canvases in specs]
    return {
        "tokens": sum(rows),
        "encoder": count * max(rows) / sum(rows),
        "canvas": count * max(canvases) / sum(canvases),
    }


def run_case(model, specs, count, dtype):
    built = features(specs, model.config.text_config.vocab_size, dtype)
    padded = DiffusionGemmaCollator(pad_token_id=0)
    packing = DiffusionGemmaCollator(pad_token_id=0, pack_samples=True)

    alone = [padded([feature]) for feature in built]
    packed = packing(built)
    batch = padded(built)
    assert_one_decoder_path(
        {
            **{f"A[{index}]": one for index, one in enumerate(alone)},
            "B": packed,
            "D": batch,
        }
    )

    arms = [
        run_arm(model, "A flat_seq mbs=1", "flat_seq", alone),
        run_arm(model, "B flat_seq_pack", "flat_seq_pack", [packed], spy=True),
        run_arm(model, f"D flat_seq mbs={count}", "flat_seq", [batch]),
    ]
    return arms, padding_ratios(specs, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--case", default=None, choices=list(CASES), help="run one case instead of all")
    args = parser.parse_args()

    dtype = torch.bfloat16
    model, config = build_model(args.layers, args.experts, dtype)
    wide = sum(kind == "full_attention" for kind in config.text_config.layer_types)
    print(
        f"{torch.cuda.get_device_name()} | layers={args.layers} "
        f"({args.layers - wide} sliding + {wide} wide) experts={args.experts} canvas={CANVAS} dtype={dtype}\n"
        f"decoder: dg_prefix hole kernel on every arm | encoder: A/D sdpa, B varlen (inherent to the mode)\n"
        f"attn_ffpa off, unlike the recipe -- favours B, which cannot reach it | no self-conditioning\n"
    )

    cases = CASES if args.case is None else {args.case: CASES[args.case]}
    for name, (specs, count) in cases.items():
        arms, ratios = run_case(model, specs, count, dtype)
        baseline = arms[0]
        print(
            f"--- {name}: {ratios['tokens']} real tokens | padded arm pays "
            f"{ratios['encoder']:.2f}x on the encoder axis, {ratios['canvas']:.2f}x on the canvas axis ---"
        )
        print(f"{'arm':<22} {'ms':>9} {'vs A':>7} {'transient MB':>13} {'total MB':>10} {'loss':>9} {'varlen':>7}")
        for arm in arms:
            calls = "-" if arm["varlen_calls"] is None else str(arm["varlen_calls"])
            print(
                f"{arm['arm']:<22} {arm['ms']:>9.2f} {baseline['ms'] / arm['ms']:>6.2f}x "
                f"{arm['transient']:>13.0f} {arm['total']:>10.0f} {arm['loss']:>9.4f} {calls:>7}"
            )
        print()


if __name__ == "__main__":
    main()
