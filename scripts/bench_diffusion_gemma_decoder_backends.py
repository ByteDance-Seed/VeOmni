"""Compare the decoder's attention backends against each other at the real 26B geometry.

`create_diffusion_decoder_attention_mask` builds `full_valid_keys` / `sliding_valid_keys` once
and only then branches on the backend: flex compresses them with `_diffusion_decoder_block_mask`,
sdpa and eager expand them to a 4D boolean. So the two paths can be held against each other with
no weights and no training run, which is what this does.

It answers two different questions, and they need each other:

  * Is flex's decoder path *correct*? Read the float32 rows. A mask fault shows there at any
    tolerance; bf16 would hide it.
  * Is a bf16 difference between backends *large*? There is no absolute answer, so the script
    also measures sdpa's memory-efficient kernel against sdpa's math kernel -- same arithmetic,
    different accumulation order, no possible fault -- and that is the floor everything else is
    read against. On this model it is not small, and `grad_norm` differences at 26B are larger
    still: see constraint 31.

    python scripts/bench_diffusion_gemma_decoder_backends.py
"""

import argparse
import itertools

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import (
    _diffusion_decoder_block_mask,
    flex_kernel_options_for,
    widen_flex_recompile_budget,
)


# Training runs flex through `torch.compile`; uncompiled flex is a different implementation
# (it materializes the score matrix) and measuring it would answer about the wrong path.
widen_flex_recompile_budget()
compiled_flex_attention = torch.compile(flex_attention)

CANVAS = 256
QUERY_HEADS = 16
# (head_dim, kv_heads): the 25 sliding layers and the 5 global ones.
LAYER_WIDTHS = ((256, 8), (512, 2))


def sdpa_reference(q, k, v, valid_keys, backend):
    mask = valid_keys[:, None, None, :].expand(q.shape[0], 1, q.shape[2], valid_keys.shape[-1])
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=k.shape[1] != q.shape[1])


def expand_kv(t, query_heads):
    """`repeat_kv` by hand: the memory-efficient kernel refuses `enable_gqa`, and it has to run
    on the same arithmetic as everything else for its number to mean anything."""
    return t.repeat_interleave(query_heads // t.shape[1], dim=1)


def relative(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-9)).item()


def compare(contender, cache_length, head_dim, kv_heads, left_padding, hidden_tail, dtype, device):
    torch.manual_seed(0)
    kv_length = cache_length + CANVAS
    valid_keys = torch.ones(1, kv_length, dtype=torch.bool, device=device)
    valid_keys[:, :left_padding] = False
    if hidden_tail:  # canvases the decoder is not allowed to read yet
        valid_keys[:, cache_length - hidden_tail : cache_length] = False

    q = torch.randn(1, QUERY_HEADS, CANVAS, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, kv_heads, kv_length, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, kv_heads, kv_length, head_dim, device=device, dtype=dtype, requires_grad=True)
    q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    if contender == "flex":
        out = compiled_flex_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            block_mask=_diffusion_decoder_block_mask(valid_keys, CANVAS),
            enable_gqa=True,
            kernel_options=flex_kernel_options_for(head_dim, device),
        )
    else:
        out = sdpa_reference(
            q, expand_kv(k, QUERY_HEADS), expand_kv(v, QUERY_HEADS), valid_keys, SDPBackend.EFFICIENT_ATTENTION
        )
    reference = sdpa_reference(q_ref, k_ref, v_ref, valid_keys, SDPBackend.MATH)

    cotangent = torch.randn_like(out)
    out.backward(cotangent)
    reference.backward(cotangent.clone())
    return {
        "out": relative(out, reference),
        "dq": relative(q.grad, q_ref.grad),
        "dk": relative(k.grad, k_ref.grad),
        "dv": relative(v.grad, v_ref.grad),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="summaries only, no per-row output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: flex and the memory-efficient kernel are both CUDA-only")
    device = torch.device("cuda")

    cases = [
        case
        for case in itertools.product(
            [1024, 1295, 2048, 4096], LAYER_WIDTHS, [0, 37, 512], [0, CANVAS], [torch.bfloat16, torch.float32]
        )
        # The float32 global layer has nowhere to run: `flex_tile_cap` sizes its 32x32 tiles for
        # bf16, and in float32 a 512-wide head needs 324KiB of shared memory against an A100's
        # 163KiB -- 16x16 tiles miss it too. Training is bf16, so this bounds the bench, not the
        # model. The float32 sliding rows still answer the correctness question.
        if case[2] + case[3] < case[0] and not (case[4] is torch.float32 and case[1][0] == 512)
    ]

    for contender in ("flex", "sdpa_memory_efficient"):
        print(f"\n=== {contender} vs sdpa math ===")
        if not args.quiet:
            print(f"{'cache':>6} {'head':>5} {'kvh':>4} {'pad':>5} {'hidden':>7} {'dtype':>9} "
                  f"{'out':>10} {'dq':>10} {'dk':>10} {'dv':>10}")
        totals = {}
        for cache_length, (head_dim, kv_heads), left_padding, hidden_tail, dtype in cases:
            result = compare(
                contender, cache_length, head_dim, kv_heads, left_padding, hidden_tail, dtype, device
            )
            bucket = totals.setdefault(dtype, {"n": 0, "sum": 0.0, "worst": 0.0})
            bucket["n"] += 1
            bucket["sum"] += max(result.values())
            bucket["worst"] = max(bucket["worst"], max(result.values()))
            if not args.quiet:
                print(f"{cache_length:>6} {head_dim:>5} {kv_heads:>4} {left_padding:>5} {hidden_tail:>7} "
                      f"{str(dtype).removeprefix('torch.'):>9} {result['out']:>10.2e} {result['dq']:>10.2e} "
                      f"{result['dk']:>10.2e} {result['dv']:>10.2e}")
        for dtype, bucket in totals.items():
            print(f"  {str(dtype).removeprefix('torch.'):>9}: {bucket['n']} cases, "
                  f"mean {bucket['sum'] / bucket['n']:.2e}, worst {bucket['worst']:.2e}")


if __name__ == "__main__":
    main()
