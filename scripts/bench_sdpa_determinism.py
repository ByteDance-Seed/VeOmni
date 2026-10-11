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
"""Which masked-attention backend repeats itself, and what the one that does costs.

A non-null ``attn_mask`` rules flash out, so a model that attends through a 4D mask gets
the memory-efficient backend, whose backward is nondeterministic. ``enable_full_determinism``
turns that backend off; this script is the evidence for both halves of that decision.

1. Where the drift is. Measured on one A800 at DiffusionGemma's sliding-layer shape -- 16
   heads, 1024 tokens, bf16, sliding causal mask as an additive tensor:

           head_dim 256    efficient  16/16 repeats differed     math  0/16
           head_dim 128    efficient   0/16                      math  0/16

   The boundary is ``head_dim``, not sequence length. Do not read that as a documented
   limit: 128 is where *this* kernel was observed to change behaviour, not a threshold it
   states. What matters is that a probe at 128 or below concludes the opposite of the truth
   for a model at 256.

2. What math costs, since it materialises the attention matrix the other kernel avoids.
   Transient peak of the call alone, inputs excluded:

           seq 1024        6.44 ms ->  7.66 ms      peak   80 MiB ->  312 MiB
           seq 4096       57.67 ms -> 105.93 ms     peak  249 MiB -> 4320 MiB

   Quadratic in sequence length against the memory-efficient kernel's linear, so a short
   context barely notices and a long one may not fit at all.

        python scripts/bench_sdpa_determinism.py

Requires a GPU.
"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


HEADS, WINDOW, REPEATS = 16, 512, 16
BACKENDS = (("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH))


def build(seq, head_dim):
    """A sliding causal mask, additive and 4D -- the shape the encoder's sliding layers pass."""
    torch.manual_seed(0)
    shape = (1, HEADS, seq, head_dim)
    leaves = [torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    upstream = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    positions = torch.arange(seq, device="cuda")
    visible = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - WINDOW)
    mask = torch.zeros(1, 1, seq, seq, device="cuda", dtype=torch.bfloat16)
    mask.masked_fill_(~visible, float("-inf"))
    return leaves, upstream, mask


def gradients(backend, leaves, upstream, mask):
    with sdpa_kernel(backend):
        out = torch.nn.functional.scaled_dot_product_attention(*leaves, attn_mask=mask)
    return torch.autograd.grad(out, leaves, upstream)


def count_drift(backend, seq, head_dim):
    inputs = build(seq, head_dim)
    first = gradients(backend, *inputs)
    differed = 0
    for _ in range(REPEATS):
        again = gradients(backend, *inputs)
        differed += any(not torch.equal(a, b) for a, b in zip(again, first))
    return differed


def measure(backend, seq, head_dim, repeats=20):
    """Time and transient peak of the call alone -- the inputs are built and warmed first."""
    inputs = build(seq, head_dim)
    for _ in range(3):
        gradients(backend, *inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(repeats):
        gradients(backend, *inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats, (torch.cuda.max_memory_allocated() - base) / 2**20


def main():
    if not torch.cuda.is_available():
        raise SystemExit("both the drift and the cost are properties of the device kernels")

    print("backward drift over", REPEATS, "repeats, seq 1024")
    for head_dim in (256, 128):
        for name, backend in BACKENDS:
            print(f"  head_dim={head_dim:<4} {name:<10} {count_drift(backend, 1024, head_dim)}/{REPEATS} differed")

    print("\nforward+backward cost, head_dim 256")
    for seq in (1024, 4096):
        for name, backend in BACKENDS:
            milliseconds, peak = measure(backend, seq, 256)
            print(f"  seq={seq:<5} {name:<10} {milliseconds:7.2f} ms   peak {peak:7.1f} MiB")


if __name__ == "__main__":
    main()
