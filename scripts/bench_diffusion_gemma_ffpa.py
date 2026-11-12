"""Check whether FFPA is a safe and worthwhile stand-in for sdpa on DiffusionGemma's wide heads.

FFPA only covers head dims in [320, 1024], so in this model it can only ever serve the five
`full_attention` layers (`global_head_dim=512`); the twenty-five sliding layers are 256 wide
and fall straight back to sdpa. Two more things narrow it further, and this script measures
both at the real shapes:

  * FFPA delegates to sdpa whenever `8 <= Nq < 512` (`FFPAAttnMeta.fallback`). The decoder's
    canvas is 256 long, so only the *encoder* can ever run the FFPA kernel.
  * With a boolean mask the Triton kernel returns NaN when the first 64-wide key tile is
    entirely masked -- the running max starts at -inf and `exp(-inf - -inf)` is NaN. Prompts
    are left-padded (`DiffusionGemmaCollator`), so every padded batch hits this. Passing the
    same mask additively with `finfo.min` keeps the running max finite and is exact, which is
    what `ffpa_attention_forward` does.

Correctness is judged on gradients, not outputs: flex_attention was exact in the forward pass
and silently wrong in the backward one. Errors are reported against an fp32 reference next to
sdpa's own error, so bf16 rounding is not mistaken for a bug, and the loss is a random
projection of the output rather than `out.sum()`, which is degenerate enough to hide a bad dQ.

    python scripts/bench_diffusion_gemma_ffpa.py

Requires a GPU and `ffpa-attn` (pip install ffpa-attn).
"""

import torch
import torch.nn.functional as F


try:
    from ffpa_attn import TritonBackend, ffpa_attn_func
except ModuleNotFoundError:  # pragma: no cover - manual tool
    raise SystemExit("ffpa-attn is not installed: uv pip install --no-deps ffpa-attn") from None


# The five full_attention layers of diffusiongemma-26B-A4B-it.
NUM_HEADS = 16
NUM_KV_HEADS = 2  # num_global_key_value_heads
HEAD_DIM = 512  # global_head_dim
# Gemma folds the attention scale into q_norm, so the attention call must not rescale.
SCALING = 1.0

DTYPE = torch.bfloat16
DEVICE = "cuda"

# How much of a left-padded prompt is padding in these measurements. Wide enough to cover a
# whole 64-key tile, which is what the boolean-mask NaN needs.
PAD_FRACTION = 8


def make_inputs(batch, q_len, kv_len, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, generator=generator, device=DEVICE, dtype=DTYPE)

    # Scale down so scores stay in a sane range without the usual 1/sqrt(d) factor.
    scale = HEAD_DIM**-0.25
    qkv = [
        (randn(batch, NUM_HEADS, q_len, HEAD_DIM) * scale).requires_grad_(True),
        (randn(batch, NUM_KV_HEADS, kv_len, HEAD_DIM) * scale).requires_grad_(True),
        (randn(batch, NUM_KV_HEADS, kv_len, HEAD_DIM) * scale).requires_grad_(True),
    ]
    # A random cotangent: `out.sum()` gives every query the same dO, which cancels enough of
    # the softmax Jacobian to make a wrong dQ look right.
    return qkv, randn(batch, NUM_HEADS, q_len, HEAD_DIM)


def make_mask(batch, q_len, kv_len, kind):
    """The left-padding key mask the collator produces: every query sees the same keys."""
    if kind == "none":
        return None
    valid_keys = torch.ones(batch, kv_len, dtype=torch.bool, device=DEVICE)
    valid_keys[:, : kv_len // PAD_FRACTION] = False
    mask = valid_keys[:, None, None, :].expand(batch, 1, q_len, kv_len)
    if kind == "bool-expand":
        return mask
    if kind == "bool-dense":
        return mask.contiguous()
    if kind == "additive":
        return torch.zeros(mask.shape, dtype=DTYPE, device=DEVICE).masked_fill_(~mask, torch.finfo(DTYPE).min)
    raise ValueError(kind)


def sdpa_call(q, k, v, mask):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=SCALING, enable_gqa=True)


def ffpa_call(q, k, v, mask):
    return ffpa_attn_func(
        q,
        k,
        v,
        attn_mask=mask,
        scale=SCALING,
        enable_gqa=True,
        forward_backend=TritonBackend(),
        backward_backend=TritonBackend(),
    )


def outputs_and_grads(call, qkv, mask, cotangent):
    out = call(*qkv, mask)
    grads = torch.autograd.grad((out * cotangent).sum(), qkv)
    return out, grads


def rel_error(actual, expected):
    denominator = expected.float().norm()
    if denominator == 0:
        return float("nan")
    return ((actual.float() - expected.float()).norm() / denominator).item()


def reference(qkv, mask, cotangent):
    """fp32 sdpa, the yardstick both bf16 kernels are measured against.

    The fp32 copies have to be leaves of their own. Differentiating `q.float()` back to the
    bf16 leaf rounds the reference gradient to bf16, which makes bf16 sdpa look bit-exact and
    hides how much room the comparison actually has.
    """
    leaves = [tensor.detach().float().requires_grad_(True) for tensor in qkv]
    if mask is not None and mask.is_floating_point():
        mask = mask.float()
    out = sdpa_call(*leaves, mask)
    return out, torch.autograd.grad((out * cotangent.float()).sum(), leaves)


def time_call(call, qkv, mask, cotangent, iters=20):
    def once():
        out = call(*qkv, mask)
        torch.autograd.grad((out * cotangent).sum(), qkv)

    for _ in range(3):
        once()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**20


# (name, batch, q_len, kv_len). The decoder's canvas is 256 long and attends to the whole
# encoder cache plus itself; the encoder is square.
SHAPES = [
    ("encoder, prompt 2048", 1, 2048, 2048),
    ("encoder, prompt 4096", 1, 4096, 4096),
    ("encoder, prompt 2048, batch 4", 4, 2048, 2048),
    ("decoder, canvas 256 / cache 2048", 1, 256, 2304),
]

MASK_KINDS = ["none", "bool-expand", "bool-dense", "additive"]


def correctness():
    print("== correctness: relative error against fp32 sdpa (bf16 inputs, random cotangent) ==")
    header = f"{'shape':<34}{'mask':<14}{'kernel':<7}{'out':>10}{'dQ':>10}{'dK':>10}{'dV':>10}"
    print(header)
    print("-" * len(header))
    for name, batch, q_len, kv_len in SHAPES:
        for kind in MASK_KINDS:
            qkv, cotangent = make_inputs(batch, q_len, kv_len)
            mask = make_mask(batch, q_len, kv_len, kind)
            ref_out, ref_grads = reference(qkv, mask, cotangent)
            for kernel, call in [("sdpa", sdpa_call), ("ffpa", ffpa_call)]:
                try:
                    out, grads = outputs_and_grads(call, qkv, mask, cotangent)
                except Exception as error:  # noqa: BLE001 - report and keep going
                    print(f"{name:<34}{kind:<14}{kernel:<7}  {type(error).__name__}: {error}")
                    continue
                values = [rel_error(out, ref_out)] + [rel_error(g, r) for g, r in zip(grads, ref_grads)]
                cells = "".join(f"{value:>10.2e}" for value in values)
                print(f"{name:<34}{kind:<14}{kernel:<7}{cells}")
            del qkv, mask, ref_out, ref_grads, cotangent
            torch.cuda.empty_cache()


def speed():
    print("\n== speed: forward + backward, ms per call (peak MiB) ==")
    header = f"{'shape':<34}{'mask':<14}{'sdpa':>18}{'ffpa':>18}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    for name, batch, q_len, kv_len in SHAPES:
        for kind in ["none", "additive"]:
            qkv, cotangent = make_inputs(batch, q_len, kv_len)
            mask = make_mask(batch, q_len, kv_len, kind)
            try:
                sdpa_ms, sdpa_mib = time_call(sdpa_call, qkv, mask, cotangent)
                ffpa_ms, ffpa_mib = time_call(ffpa_call, qkv, mask, cotangent)
            except Exception as error:  # noqa: BLE001
                print(f"{name:<34}{kind:<14}  {type(error).__name__}: {error}")
                continue
            sdpa_cell = f"{sdpa_ms:.2f} ({sdpa_mib:.0f})"
            ffpa_cell = f"{ffpa_ms:.2f} ({ffpa_mib:.0f})"
            print(f"{name:<34}{kind:<14}{sdpa_cell:>18}{ffpa_cell:>18}{sdpa_ms / ffpa_ms:>9.2f}x")
            del qkv, mask, cotangent
            torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this script needs a GPU")
    print(f"device: {torch.cuda.get_device_name()}  heads {NUM_HEADS}/{NUM_KV_HEADS}  head_dim {HEAD_DIM}\n")
    correctness()
    speed()


if __name__ == "__main__":
    main()
