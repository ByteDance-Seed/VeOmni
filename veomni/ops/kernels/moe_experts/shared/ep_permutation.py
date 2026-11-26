"""Internal CUDA row maps for an exact expert/sender chunk permutation."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _chunk_row_maps(
    metadata, direct, inverse, rows, CHUNKS: tl.constexpr, SEARCH_STEPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = row < rows
    lo = tl.full((BLOCK,), 0, tl.int64)
    hi = tl.full((BLOCK,), CHUNKS, tl.int64)
    for _ in range(SEARCH_STEPS):
        mid = lo + (hi - lo) // 2
        end = tl.load(metadata + mid, mid < CHUNKS, other=rows)
        right = (mid < CHUNKS) & (end <= row)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    previous = tl.load(metadata + lo - 1, valid & (lo > 0), other=0)
    original = tl.load(metadata + CHUNKS + lo, valid & (lo < CHUNKS), other=0)
    source = original + row - previous
    tl.store(direct + row, source, valid)
    tl.store(inverse + source, row, valid)


def fill_chunk_row_maps(metadata, direct, inverse, chunks, rows):
    """Metadata is [ordered chunk ends, original chunk starts], both int64."""
    _chunk_row_maps[(triton.cdiv(rows, 256),)](
        metadata,
        direct,
        inverse,
        rows,
        chunks,
        chunks.bit_length() + 1,
        256,
    )
