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
"""What does packing samples buy the DiffusionGemma encoder, and against what?

Not the removal of padding waste. The production recipe runs `micro_batch_size:
1`, so a micro-batch holds one sample and has nothing to pad against -- the
config says as much where it turns `dyn_bsz` off. What packing buys is a
micro-batch that can hold several samples *without* padding, which is the thing
a token budget needs to exist at all. So the baseline to beat is not a padded
batch; it is today's gradient accumulation.

Three arms over the same S samples and the same token count:

* **accumulate** -- S sequential forward/backward passes at batch 1, which is
  what the recipe does now. No padding, no waste, and S times the launch
  overhead and the small-matrix inefficiency.
* **packed** -- one forward/backward over a single row of `sum(L)` tokens through
  the varlen kernels.
* **padded** -- one forward/backward over S rows of `max(L)` tokens with a mask,
  which is what raising `micro_batch_size` would do today. Included because it is
  the option that already exists, and its waste is the `S*max(L)/sum(L)` ratio
  printed beside it.

Real attention geometry (2304 hidden, 25 sliding layers 256 wide with a 512
window to 5 full layers 512 wide) at reduced depth; time and memory both scale
close to linearly in layers, so read the ratios and not the absolutes. The
default text config is dense, and the 26B checkpoint is not -- an MoE stack
groups tokens by expert, so packing changes that grouping too. Constraint 33 has
the numbers for what that does to reproducibility; its effect on throughput is
not measured here.
"""

from __future__ import annotations

import argparse

import torch

from veomni.models.transformers.diffusion_gemma import modeling_diffusion_gemma as modeling
from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaTextConfig
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaEncoderTextModel


WARMUP = 2
ITERATIONS = 5

#: Encoder lengths the recipe actually produces. It carries the prompt plus
#: whichever clean canvases precede the supervised one, so block sampling spreads
#: the length by up to 7 * 256 within one `max_seq_len: 4096` budget -- which the
#: config names as the reason padding waste could appear at all.
CASES = [
    ("one canvas each, short", [512, 512, 512, 512]),
    ("mid, equal", [2048, 2048]),
    ("block sampling spread", [512, 1024, 1792, 2560]),
    ("wide spread, one long", [3584, 768, 512, 512]),
    ("many short samples", [384] * 8),
]


def build_encoder(layers: int, experts: int = 8, dtype=torch.bfloat16):
    """Production attention geometry, fewer layers, checkpointing as configured.

    Every layer carries a dense MLP *and* an MoE block, so the expert shape has
    to be supplied: the default text config leaves `num_experts` unset and cannot
    be instantiated at all. The counts here follow the toy config's ratios
    (`moe_intermediate_size` at half the hidden size, 8 experts, top 2) rather
    than the 26B checkpoint's, which is not on this machine. Attention geometry
    is the part that is faithful.
    """
    config = DiffusionGemmaTextConfig(
        num_hidden_layers=layers,
        num_experts=experts,
        top_k_experts=min(2, experts),
        moe_intermediate_size=1152,
        num_global_key_value_heads=2,
        vocab_size=8192,
    )
    config._attn_implementation = "sdpa"
    config.attn_ffpa = False
    torch.manual_seed(0)
    model = DiffusionGemmaEncoderTextModel(config).to(device="cuda", dtype=dtype)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train(), config


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


def backward_from(hidden_states, model):
    """One scalar off the whole row, so every token contributes to the backward."""
    model.zero_grad(set_to_none=True)
    hidden_states.float().pow(2).mean().backward()


def accumulate_step(model, tokens, lengths):
    """Today: one sample per micro-batch, S of them, gradients accumulating."""

    def step():
        model.zero_grad(set_to_none=True)
        start = 0
        for length in lengths:
            row = tokens[:, start : start + length]
            positions = torch.arange(length, device=row.device).unsqueeze(0)
            out = model(input_ids=row, position_ids=positions, use_cache=True)
            out.last_hidden_state.float().pow(2).mean().backward()
            start += length

    return step


def packed_step(model, tokens, lengths):
    positions = torch.cat([torch.arange(n, device=tokens.device) for n in lengths]).unsqueeze(0)

    def step():
        out = model(input_ids=tokens, position_ids=positions, pack_samples=True, use_cache=True)
        backward_from(out.last_hidden_state, model)

    return step


def assert_the_packed_arm_used_the_kernels(model, config, lengths):
    """Or the packed arm is quietly timing the block-diagonal mask instead.

    `should_use_varlen_encoder` refuses for a list of reasons (constraint 33) and
    falls back to a mask that constraint 32 measures 5.9-12.1x the other side of
    this comparison. A refusal would leave every number below intact and wrong,
    so count the calls rather than trusting the argument.

    Forward only, deliberately. Non-reentrant checkpointing replays the forward
    inside the backward, so a full step calls each layer's attention twice and
    the expected count would have to encode that -- an accounting detail from the
    other side of the question being asked.
    """
    calls = {"flash": 0, "wide": 0}
    real = modeling.varlen_encoder_attention

    def spy(query, *args, **kwargs):
        calls["wide" if query.shape[-1] > 256 else "flash"] += 1
        return real(query, *args, **kwargs)

    tokens = torch.randint(0, 1000, (1, sum(lengths)), device="cuda")
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths]).unsqueeze(0)
    modeling.varlen_encoder_attention = spy
    try:
        with torch.no_grad():
            model(input_ids=tokens, position_ids=positions, pack_samples=True, use_cache=True)
    finally:
        modeling.varlen_encoder_attention = real

    wide = sum(kind == "full_attention" for kind in config.layer_types)
    expected = {"flash": config.num_hidden_layers - wide, "wide": wide}
    if calls != expected:
        raise AssertionError(
            f"the packed arm reached the varlen kernels {calls} times, not {expected}. Something in "
            "`should_use_varlen_encoder` refused and the packed timings below would be a block-diagonal "
            "mask wearing the packed arm's name."
        )
    print(f"packed arm verified: {calls['flash']} flash + {calls['wide']} wide varlen calls per forward")


def padded_step(model, tokens, lengths):
    """Raising `micro_batch_size` instead: S rows left-padded to the longest."""
    width = max(lengths)
    rows, mask = [], []
    start = 0
    for length in lengths:
        pad = width - length
        rows.append(torch.nn.functional.pad(tokens[0, start : start + length], (pad, 0)))
        mask.append(torch.cat([torch.zeros(pad, device=tokens.device), torch.ones(length, device=tokens.device)]))
        start += length
    batch = torch.stack(rows)
    attention_mask = torch.stack(mask).long()

    def step():
        out = model(input_ids=batch, attention_mask=attention_mask, use_cache=True)
        backward_from(out.last_hidden_state, model)

    return step


def run_case(model, name, lengths):
    total = sum(lengths)
    tokens = torch.randint(0, 1000, (1, total), device="cuda")

    arms = {
        "accumulate": accumulate_step(model, tokens, lengths),
        "packed": packed_step(model, tokens, lengths),
        "padded": padded_step(model, tokens, lengths),
    }
    results = {}
    for arm, step in arms.items():
        try:
            results[arm] = measure(step)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[arm] = (float("nan"), float("nan"))

    waste = len(lengths) * max(lengths) / total
    print(f"\n{name}: {len(lengths)} samples, {total} tokens, padded would be {waste:.2f}x that")
    print(f"  {'arm':<12} {'ms':>9} {'transient MB':>14} {'vs accumulate':>15}")
    reference = results["accumulate"][0]
    for arm, (milliseconds, megabytes) in results.items():
        speedup = f"{reference / milliseconds:.2f}x" if milliseconds == milliseconds else "OOM"
        print(f"  {arm:<12} {milliseconds:>9.2f} {megabytes:>14.0f} {speedup:>15}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=12, help="keeps the 5:1 sliding/full pattern")
    parser.add_argument(
        "--experts",
        type=int,
        default=8,
        help="1 makes the MoE block trivial, separating the arms' real gap from the expert loop's overhead",
    )
    args = parser.parse_args()

    model, config = build_encoder(args.layers, args.experts)
    sliding = sum(kind == "sliding_attention" for kind in config.layer_types)
    print(f"{torch.cuda.get_device_name()}, bfloat16, gradient checkpointing on")
    print(
        f"{config.num_hidden_layers} layers ({sliding} sliding {config.head_dim} wide, window "
        f"{config.sliding_window}; {config.num_hidden_layers - sliding} full {config.global_head_dim} wide), "
        f"hidden {config.hidden_size}"
    )

    assert_the_packed_arm_used_the_kernels(model, config, [384, 512, 640])

    for name, lengths in CASES:
        run_case(model, name, lengths)


if __name__ == "__main__":
    main()
