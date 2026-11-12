"""Which BlockMask builder survives a second shape, and what the one that does costs.

`transformers.masking_utils.flex_attention_mask` asks `create_block_mask` for `_compile=True`,
which torch answers with `torch.compile(create_block_mask)`. DiffusionGemma hands that builder
four shapes a step -- encoder and decoder, sliding and full -- and from the second one on, the
sparsity it returns disagrees with the same mask built alone. `uncompiled_block_masks` in
`modeling_diffusion_gemma.py` takes the flag back off; this is the evidence for that decision.

Two things it shows, both of which decided the fix:

1. Which builders are correct. Six shapes in one process, each compared byte for byte against
   the same mask built by itself. Measured on one A800, torch 2.9.1:

       _compile=True                   wrong at 16x31 and 16x56
       torch.compile(create_block_mask) wrong at 40x40, 16x31 and 16x56
       torch.compile(dynamic=False)     correct
       _compile=False                   correct

   The second row is what torch's own deprecation warning on `_compile` recommends instead, so
   following the warning would not have fixed anything. The third is correct but recompiles per
   shape and exhausts Dynamo's recompile limit within a handful of lengths, after which it falls
   back to the fourth anyway.

   A wrong mask is not a crash. flex's forward reads the key side and stayed exact at every
   shape; its dK/dV kernel reads the transposed side and returned exactly zero, which reaches
   training as `k_proj` and `v_proj` simply not moving.

2. What correctness costs, since the uncompiled builder materialises the dense mask that the
   compiled one fuses away:

       40x40        1.34 ms -> 1.87 ms
       4096x4096    1.85 ms -> 1.95 ms

   Under a millisecond *added* per mask, against an attention call at 4096 that takes tens.

       python scripts/bench_diffusion_gemma_block_mask.py

Requires a GPU.
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask


SHAPES = ((40, 40), (16, 31), (16, 56), (1024, 1024), (256, 4096), (4096, 4096))
BATCH, WINDOW = 2, 512

SUGGESTED = torch.compile(create_block_mask)
STATIC = torch.compile(create_block_mask, dynamic=False)


def sliding_causal(batch, head, q, kv):
    del batch, head
    return (q >= kv) & (kv > q - WINDOW)


def builders(q_length, kv_length):
    arguments = (sliding_causal, BATCH, None, q_length, kv_length)
    return {
        "_compile=True": lambda: create_block_mask(*arguments, device="cuda", _compile=True),
        "torch.compile(create_block_mask)": lambda: SUGGESTED(*arguments, "cuda"),
        "torch.compile(dynamic=False)": lambda: STATIC(*arguments, "cuda"),
        "_compile=False": lambda: create_block_mask(*arguments, device="cuda", _compile=False),
    }


SPARSITY = (
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
    "q_num_blocks",
    "q_indices",
    "full_q_num_blocks",
    "full_q_indices",
)


def signature(block_mask):
    """Everything the attention kernels read off the mask, as comparable bytes.

    Both directions, because the forward and the backward walk different ones -- a mask that is
    wrong on the transposed side alone still passes a forward comparison.
    """
    parts = []
    for name in SPARSITY:
        tensor = getattr(block_mask, name, None)
        parts.append(None if tensor is None else (tuple(tensor.shape), tensor.cpu().numpy().tobytes()))
    return tuple(parts)


def measure(build, repeats=10):
    for _ in range(3):
        build()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(repeats):
        build()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    # Each shape built alone, before any other has been through a compiled builder.
    alone = {
        shape: signature(create_block_mask(sliding_causal, BATCH, None, *shape, device="cuda", _compile=False))
        for shape in SHAPES
    }

    names = list(builders(*SHAPES[0]))
    labels = [f"{q}x{kv}" for q, kv in SHAPES]

    def row(name, cells):
        print(f"  {name:<34}" + "  ".join(f"{cell:>{len(label)}}" for cell, label in zip(cells, labels)))

    print("correctness -- six shapes in one process, against the same mask built alone")
    row("", labels)
    for name in names:
        row(name, ["ok" if signature(builders(*shape)[name]()) == alone[shape] else "WRONG" for shape in SHAPES])

    print("\ncost -- milliseconds per mask")
    row("", labels)
    for name in names:
        row(name, [f"{measure(builders(*shape)[name]):.2f}" for shape in SHAPES])


if __name__ == "__main__":
    main()
