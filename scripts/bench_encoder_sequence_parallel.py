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
"""What Ulysses buys the DiffusionGemma encoder, and what it leaves for the decoder.

    torchrun --standalone --nproc_per_node=2 scripts/bench_encoder_sequence_parallel.py

With ``--whole-model`` it runs `forward` as well, which adds the decoder reading
the cache. Two arms over the same packed row, on the same ranks:

* **sp=1** -- every rank runs the whole row, which is what a single-GPU step does.
  Timed on rank 0 and reported as the baseline.
* **sp=2** -- each rank holds ``L/sp`` tokens and the attention gathers back to
  full length while splitting the heads.

The encoder is the only half that runs under SP today, so the interesting number
is not just its speedup. It is what the *decoder* now has to deal with: the
encoder KV cache is sharded along the sequence, and a canvas needs its whole
visible interval of it. This prints the cache each rank holds beside a price for
the two ways of giving the decoder a full interval:

* **gather** -- all-gather the cache. Priced per layer, since doing it inside the
  checkpointed layer keeps one full-length copy at peak instead of thirty.
* **all-to-all** -- Ulysses it, so each rank holds full length for ``kv/sp``
  heads. Same bytes resident as the sharded cache, different traffic.

Both are paid twice per step under non-reentrant checkpointing -- once in the
forward and once in recompute -- and the projection below says so. Read the
ratios, not the absolutes: depth is reduced from the recipe's 30 layers and the
MoE shape follows the toy config, while the attention geometry is faithful.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from veomni.distributed.parallel_state import _init_parallel_state, use_parallel_state
from veomni.models.transformers.diffusion_gemma import modeling_diffusion_gemma as modeling
from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaTextConfig
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaEncoderTextModel


WARMUP = 2
ITERATIONS = 5

#: Row lengths, in tokens. The recipe's `data.max_seq_len` is 4096 and the reason
#: to want SP is the lengths beyond it, so the sweep starts there and doubles.
LENGTHS = [4096, 8192, 16384, 32768]

#: Samples per row. A packed row holds several, and the segment count is what the
#: varlen kernels see; it does not interact with the sharding, which cuts the row
#: by token count and not by sample.
SAMPLES = 4


def build_encoder(layers: int, experts: int, dtype=torch.bfloat16):
    """Production attention geometry, fewer layers, checkpointing as configured."""
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


def packed_row(total: int, samples: int, device):
    """One row of `total` tokens split into `samples` segments of equal length."""
    if total % samples:
        raise ValueError(f"{total} tokens do not split into {samples} equal samples")
    each = total // samples
    tokens = torch.randint(0, 1000, (1, total), device=device)
    positions = torch.cat([torch.arange(each, device=device) for _ in range(samples)]).unsqueeze(0)
    return tokens, positions


def measure(step):
    """Milliseconds per step and the memory allocated inside the timing window."""
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    dist.barrier()
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


def cache_megabytes(model, tokens, positions):
    """What one forward leaves behind for the decoder to read, on this rank."""
    with torch.no_grad():
        out = model(input_ids=tokens, position_ids=positions, pack_samples=True, use_cache=True)
    cache = out.past_key_values
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise AssertionError(f"the encoder returned {type(cache).__name__}, which has no layers to measure")
    total = 0
    for layer in layers:
        for tensor in (layer.keys, layer.values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total / 2**20


def step_for(model, tokens, positions):
    def step():
        model.zero_grad(set_to_none=True)
        out = model(input_ids=tokens, position_ids=positions, pack_samples=True, use_cache=False)
        out.last_hidden_state.float().pow(2).mean().backward()

    return step


def assert_the_arm_used_the_kernels(model, config, tokens, positions, arm: str):
    """Or the timing is a block-diagonal mask wearing the varlen arm's name.

    Under SP a refusal is worse than slow: `should_use_varlen_encoder` raises
    rather than fall back, so what this catches is the sp=1 arm silently taking
    the mask path and flattering the comparison.
    """
    calls = {"flash": 0, "wide": 0}
    real = modeling.varlen_encoder_attention

    def spy(query, *args, **kwargs):
        calls["wide" if query.shape[-1] > 256 else "flash"] += 1
        return real(query, *args, **kwargs)

    modeling.varlen_encoder_attention = spy
    try:
        with torch.no_grad():
            model(input_ids=tokens, position_ids=positions, pack_samples=True, use_cache=False)
    finally:
        modeling.varlen_encoder_attention = real

    wide = sum(kind == "full_attention" for kind in config.layer_types)
    expected = {"flash": config.num_hidden_layers - wide, "wide": wide}
    if calls != expected:
        raise AssertionError(f"the {arm} arm reached the varlen kernels {calls} times, not {expected}")


def price_the_collectives(config, total: int, sp_size: int, group, dtype=torch.bfloat16):
    """Milliseconds for the two ways of handing the decoder a full-length cache.

    One layer, one of K and V. The sliding layers are the majority and the
    narrower ones, so this uses their geometry and the projection below is a
    lower bound on the global layers' share.
    """
    local = total // sp_size
    heads, dim = config.num_key_value_heads, config.head_dim
    sharded = torch.empty(1, local, heads, dim, device="cuda", dtype=dtype)

    def gather():
        out = [torch.empty_like(sharded) for _ in range(sp_size)]
        dist.all_gather(out, sharded, group=group)
        return torch.cat(out, dim=1)

    # `all_to_all_single` splits along dim 0, so the head groups have to lead. Built
    # in that layout rather than permuted into it: what is being priced against the
    # gather is the traffic, and both calls then move their bytes and nothing else.
    scattered = (
        torch.empty(sp_size, local, heads // sp_size, dim, device="cuda", dtype=dtype)
        if heads % sp_size == 0
        else None
    )

    def all_to_all():
        if scattered is None:
            return None
        destination = torch.empty_like(scattered)
        dist.all_to_all_single(destination, scattered, group=group)
        return destination

    prices = {}
    for name, call in (("gather", gather), ("all-to-all", all_to_all)):
        if call() is None:
            prices[name] = float("nan")
            continue
        for _ in range(WARMUP):
            call()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERATIONS):
            call()
        end.record()
        torch.cuda.synchronize()
        prices[name] = start.elapsed_time(end) / ITERATIONS
    return prices


def run_length(model, config, total: int, samples: int, replicated, sharded, rank: int, sp_size: int):
    device = torch.device("cuda", rank)
    tokens, positions = packed_row(total, samples, device)

    results = {}
    caches = {}
    for arm, state in (("sp=1", replicated), (f"sp={sp_size}", sharded)):
        with use_parallel_state(state):
            try:
                assert_the_arm_used_the_kernels(model, config, tokens, positions, arm)
                results[arm] = measure(step_for(model, tokens, positions))
                caches[arm] = cache_megabytes(model, tokens, positions)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results[arm], caches[arm] = (float("nan"), float("nan")), float("nan")
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    with use_parallel_state(sharded):
        # `ulysses_group` rather than `sp_group`: see `dg_sequence_parallel._ulysses`.
        # Harmless on this thread, but the flattened dimension is a trap worth not
        # copying into new code.
        prices = price_the_collectives(config, total, sp_size, sharded.ulysses_group)

    if rank != 0:
        return
    baseline = results["sp=1"][0]
    print(f"\n{total} tokens in {samples} samples")
    print(f"  {'arm':<8} {'ms':>9} {'transient MB':>14} {'cache MB/rank':>15} {'vs sp=1':>9}")
    for arm, (milliseconds, megabytes) in results.items():
        ratio = f"{baseline / milliseconds:.2f}x" if milliseconds == milliseconds else "OOM"
        print(f"  {arm:<8} {milliseconds:>9.2f} {megabytes:>14.0f} {caches[arm]:>15.0f} {ratio:>9}")

    step = results[f"sp={sp_size}"][0]
    layers = config.num_hidden_layers
    print(f"  giving the decoder a full-length cache, projected over {layers} layers x 2 (forward + recompute):")
    for name, milliseconds in prices.items():
        # Two tensors per layer, K and V, twice per step.
        projected = milliseconds * 2 * layers * 2
        share = f"{projected / step:.1%} of the sp={sp_size} step" if step == step else "n/a"
        print(f"    {name:<12} {milliseconds:>7.3f} ms per K or V per layer -> {projected:>8.1f} ms, {share}")


def build_whole_model(layers: int, experts: int, canvases: int, dtype=torch.bfloat16):
    """The model both halves of Eq. 13 run on, in the configuration SP accepts.

    `flat_seq_pack` because SP shards the encoder row and the padded modes have
    no single row to shard, and `attn_dg_prefix` because the hole kernel is the
    decoder's only sharded path -- both are what the preflight requires and what
    the recipe sets.
    """
    from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaConfig,
        DiffusionGemmaTextConfig,
    )
    from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

    text_config = DiffusionGemmaTextConfig(
        num_hidden_layers=layers,
        num_experts=experts,
        top_k_experts=min(2, experts),
        moe_intermediate_size=1152,
        num_global_key_value_heads=2,
        vocab_size=8192,
    )
    text_config.attn_dg_prefix = True
    text_config.attn_ffpa = False
    config = DiffusionGemmaConfig(
        text_config=text_config,
        canvas_length=256,
        max_canvases=canvases,
        vision_config=None,
    )
    config.canvas_batching = "flat_seq_pack"
    config.encoder_ar_loss = True
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = DiffusionGemmaForBlockDiffusion(config).to(device="cuda", dtype=dtype)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train(), config


def whole_model_batch(config, total: int, samples: int, canvases: int, sp_size: int, device):
    """A packed row through the shipped collator, so the shapes are the real ones."""
    from veomni.data.data_collator import DiffusionGemmaCollator
    from veomni.utils.constants import IGNORE_INDEX

    canvas = config.canvas_length
    vocab = config.text_config.vocab_size
    each = total // samples
    if each <= canvases * canvas:
        raise ValueError(f"a {each}-token sample cannot hold {canvases} canvases of {canvas}")

    # Every canvas sees the whole of its own sample's row, which is the most work
    # the interval can ask for. The recipe gives a canvas only the prefix in front
    # of it, so the decoder's share here is an upper bound -- the same for both
    # arms, which is what the ratio needs.
    features = []
    for index in range(samples):
        torch.manual_seed(100 + index)
        labels = torch.full((canvases, canvas), IGNORE_INDEX, dtype=torch.long)
        labels[:, ::2] = torch.randint(3, vocab, (canvases, canvas // 2))
        weight = torch.zeros(canvases, canvas)
        weight[:, ::2] = 2.0 / canvas
        features.append(
            {
                "input_ids": torch.randint(3, vocab, (each,)),
                "encoder_prompt_mask": torch.zeros(each, dtype=torch.bool),
                "decoder_input_ids": torch.randint(3, vocab, (canvases, canvas)),
                "labels": labels,
                "loss_weight": weight,
                "decoder_context_mask": torch.ones(canvases, each, dtype=torch.bool),
            }
        )
    batch = DiffusionGemmaCollator(pack_samples=True, sp_size=sp_size)(features)
    return {key: value.to(device) for key, value in batch.items()}


def run_whole_model(args, replicated, sharded, rank: int, sp_size: int):
    """The same two arms over `forward`, which is where the decoder appears.

    Reported separately from the encoder sweep because the two answer different
    questions: the encoder scales with the row, while the decoder's queries are
    `K` canvases and do not -- so what SP does for the whole step is bounded by
    how much of it the encoder was.
    """
    device = torch.device("cuda", rank)
    # Built under the sharded state so `sequence_parallel_preflight` actually runs:
    # `_init_parallel_state` keeps the *first* call as the global (the replicated one
    # above), and a model constructed outside this block would see `ulysses_size=1`
    # and wave the check through -- which is exactly the opposite of what a
    # benchmark of the SP path should be verifying.
    with use_parallel_state(sharded):
        model, config = build_whole_model(args.layers, args.experts, args.canvases)
    if rank == 0:
        print(f"\nwhole model: {args.canvases} canvases of {config.canvas_length}, AR term on, attn_dg_prefix on")

    for total in args.lengths:
        # A sample has to hold its canvases and then some prompt, so short rows take
        # fewer samples rather than smaller samples: the canvas span is fixed by the
        # config and the encoder length is what this sweep varies.
        samples = min(args.samples, max(1, total // (2 * args.canvases * config.canvas_length)))
        results = {}
        for arm, state in (("sp=1", replicated), (f"sp={sp_size}", sharded)):
            with use_parallel_state(state):
                batch = whole_model_batch(config, total, samples, args.canvases, state.sp_size, device)

                def step(batch=batch):
                    model.zero_grad(set_to_none=True)
                    model(**batch, use_cache=False).loss.backward()

                try:
                    results[arm] = measure(step)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    results[arm] = (float("nan"), float("nan"))
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if rank != 0:
            continue
        baseline = results["sp=1"][0]
        canvas_tokens = args.canvases * config.canvas_length * samples
        print(f"\n  {total} encoder tokens in {samples} samples, {canvas_tokens} canvas tokens")
        print(f"    {'arm':<8} {'ms':>9} {'transient MB':>14} {'vs sp=1':>9}")
        for arm, (milliseconds, megabytes) in results.items():
            ratio = f"{baseline / milliseconds:.2f}x" if milliseconds == milliseconds else "OOM"
            print(f"    {arm:<8} {milliseconds:>9.2f} {megabytes:>14.0f} {ratio:>9}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=12, help="keeps the 5:1 sliding/full pattern")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--lengths", type=int, nargs="*", default=LENGTHS)
    parser.add_argument(
        "--whole-model",
        action="store_true",
        help="also run `forward`, which adds the decoder reading the sharded cache",
    )
    parser.add_argument("--canvases", type=int, default=4, help="canvases per sample in the whole-model arm")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")

    replicated = _init_parallel_state(dp_size=world_size, dp_shard_size=world_size, ulysses_size=1, name=None)
    sharded = _init_parallel_state(dp_size=1, dp_shard_size=1, ulysses_size=world_size, name=None)

    model, config = build_encoder(args.layers, args.experts)
    if rank == 0:
        sliding = sum(kind == "sliding_attention" for kind in config.layer_types)
        print(f"{torch.cuda.get_device_name()} x {world_size}, bfloat16, gradient checkpointing on")
        print(
            f"{config.num_hidden_layers} layers ({sliding} sliding {config.head_dim} wide, window "
            f"{config.sliding_window}; {config.num_hidden_layers - sliding} full {config.global_head_dim} "
            f"wide), hidden {config.hidden_size}, {config.num_key_value_heads} kv heads"
        )

    for total in args.lengths:
        run_length(model, config, total, args.samples, replicated, sharded, rank, world_size)
    del model
    torch.cuda.empty_cache()

    if args.whole_model:
        run_whole_model(args, replicated, sharded, rank, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
