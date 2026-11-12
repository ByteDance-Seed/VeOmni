"""What packing the canvases onto the sequence buys the decoder's attention.

Both `canvas_batching` packing modes run one decoder pass over K canvases; they
differ in what the attention has to build first.

  * `flat_batch` keeps a canvas in its own row, so DG_PREFIX gathers that
    canvas's visible prefix into a private copy and concatenates the canvas onto
    it. Summed over K that is `K*P + K^2*C/2` keys materialised per layer, twice
    (once for K, once for V), and held for the backward.
  * `flat_seq` hands the kernel each canvas's visible run as an interval, so the
    cache is read where the encoder left it: `P + K*C`, no copy.

Measured on one A800 at the 26B shapes (prompt 2048, canvas 256, bf16,
forward+backward, `flat_seq` against `flat_batch`):

    layer                    K=4          K=8         K=16
    sliding, head_dim 256    3.1x 1.2x    2.8x 1.3x    2.6x 1.4x
    full,    head_dim 512    1.1x 2.4x    1.1x 2.9x    1.2x 3.6x

    (speedup, peak-memory reduction)

The split is the opposite of what the materialisation counts alone suggest. A
sliding layer caps each prefix at `W-1`, so the gather is small -- but it is
still a gather, and DG_PREFIX builds it with a Python loop that synchronises on
`.item()` per segment. At these shapes that packing, not the attention, is what
the sliding layers are paying for. The `full_attention` layers do carry the
quadratic term, and there the two run at nearly the same speed because the
arithmetic is the same; what the interval saves is the copy, which is why the
memory column is where they separate.

    python scripts/bench_diffusion_gemma_hole.py

Requires a GPU.
"""

import torch

from veomni.models.transformers.diffusion_gemma.dg_prefix import dg_prefix_decoder_attention
from veomni.ops.kernels.dllm_hole_attention import dllm_hole_attention


# diffusiongemma-26B-A4B-it. The sliding layers are 25 of 30, the full ones 5.
LAYERS = {
    "sliding (head_dim 256, 25 layers)": dict(heads=16, kv_heads=4, head_dim=256, window=256),
    "full (head_dim 512, 5 layers)": dict(heads=16, kv_heads=2, head_dim=512, window=0),
}
CANVAS_LENGTH = 256
PROMPT_LENGTH = 2048
BATCH = 1  # both recipes ship micro_batch_size: 1
CANVAS_COUNTS = (4, 8, 16)
SCALING = 1.0  # Gemma folds the attention scale into q_norm
DTYPE = torch.bfloat16
WARMUP, ITERATIONS = 3, 10


def build(canvases, heads, kv_heads, head_dim, window, seed=0):
    """One layer's decoder inputs, plus the visibility in both representations."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cache_length = PROMPT_LENGTH + canvases * CANVAS_LENGTH
    tokens = canvases * CANVAS_LENGTH
    scale = head_dim**-0.25

    def randn(*shape):
        return (torch.randn(shape, generator=generator, device="cuda", dtype=DTYPE) * scale).requires_grad_(True)

    # Canvas k reads the prompt plus the k clean canvases before it, narrowed on a
    # sliding layer to the last W-1 of those.
    hi = torch.tensor([[PROMPT_LENGTH + k * CANVAS_LENGTH for k in range(canvases)]], dtype=torch.int32).cuda()
    lo = torch.zeros_like(hi) if not window else (hi - (window - 1)).clamp(min=0)
    keys = torch.arange(cache_length, device="cuda")
    visible = (keys >= lo.reshape(-1, 1)) & (keys < hi.reshape(-1, 1))

    return {
        "packed_query": randn(BATCH, heads, tokens, head_dim),
        "row_query": randn(BATCH * canvases, heads, CANVAS_LENGTH, head_dim),
        "cache_key": randn(BATCH, kv_heads, cache_length, head_dim),
        "cache_value": randn(BATCH, kv_heads, cache_length, head_dim),
        "packed_key": randn(BATCH, kv_heads, tokens, head_dim),
        "packed_value": randn(BATCH, kv_heads, tokens, head_dim),
        "row_key": randn(BATCH * canvases, kv_heads, CANVAS_LENGTH, head_dim),
        "row_value": randn(BATCH * canvases, kv_heads, CANVAS_LENGTH, head_dim),
        "lo": lo,
        "hi": hi,
        "visible": visible,
        "canvases": canvases,
    }


#: Which tensors each path differentiates. The cache is shared; the queries and the
#: canvas K/V differ only in how they are laid out.
FLAT_BATCH_INPUTS = ("row_query", "row_key", "row_value", "cache_key", "cache_value")
FLAT_SEQ_INPUTS = ("packed_query", "packed_key", "packed_value", "cache_key", "cache_value")


def run_flat_batch(inputs):
    return dg_prefix_decoder_attention(
        inputs["row_query"],
        inputs["row_key"],
        inputs["row_value"],
        inputs["cache_key"],
        inputs["cache_value"],
        inputs["visible"],
        sm_scale=SCALING,
        canvas_groups=inputs["canvases"],
    )


def run_flat_seq(inputs):
    return dllm_hole_attention(
        inputs["packed_query"],
        inputs["cache_key"],
        inputs["cache_value"],
        inputs["packed_key"],
        inputs["packed_value"],
        inputs["lo"],
        inputs["hi"],
        SCALING,
    )


def measure(run, inputs, keys):
    """Forward plus backward, since the gather's copies are held for the backward."""
    tensors = [inputs[key] for key in keys]

    def step():
        output = run(inputs)
        torch.autograd.grad(output, tensors, torch.ones_like(output))

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
    return start.elapsed_time(end) / ITERATIONS, (torch.cuda.max_memory_allocated() - baseline) / 2**20


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a GPU")
    print(f"{torch.cuda.get_device_name()}, batch {BATCH}, prompt {PROMPT_LENGTH}, canvas {CANVAS_LENGTH}\n")

    for name, layer in LAYERS.items():
        print(name)
        print(f"  {'K':>3}  {'flat_batch':>20}  {'flat_seq':>20}  {'speedup':>8}  {'peak':>8}")
        for canvases in CANVAS_COUNTS:
            inputs = build(canvases, **layer)
            batch_ms, batch_mb = measure(run_flat_batch, inputs, FLAT_BATCH_INPUTS)
            seq_ms, seq_mb = measure(run_flat_seq, inputs, FLAT_SEQ_INPUTS)
            print(
                f"  {canvases:>3}  {batch_ms:>10.2f} ms {batch_mb:>7.0f} MB"
                f"  {seq_ms:>10.2f} ms {seq_mb:>7.0f} MB"
                f"  {batch_ms / seq_ms:>7.2f}x  {batch_mb / max(seq_mb, 1e-9):>7.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
