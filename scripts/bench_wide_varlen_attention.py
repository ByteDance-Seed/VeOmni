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
"""What packing an encoder row costs on each of the three ways to do it.

At DiffusionGemma's global-layer geometry -- 16 heads over 2 KV heads,
``global_head_dim=512``, bf16, forward+backward -- over three paths:

* ``unpacked``  FFPA at ``batch=S``, one row per sample. The status quo. With
  equal lengths transformers hands the layer no mask and FFPA applies causality
  itself; with ragged ones the collator left-pads and a real 4-D mask appears.
* ``dense``     FFPA on the packed row with a block-diagonal mask. What packing
  costs *without* this kernel, and the row the ``S``-fold argument is about:
  the mask does not stop the kernel forming the off-diagonal blocks.
* ``varlen``    ``wide_varlen_attention`` on the packed row.

``ffpa_attn_func`` is called directly rather than through
``ffpa_attention_forward`` so that a silent fallback cannot be read as FFPA's
number -- the wrapper's shape and dtype gates are what would hide it, and here
the arguments are known to clear them.
"""

import torch

from veomni.models.transformers.diffusion_gemma.ffpa_attention import FFPA_AVAILABLE
from veomni.ops.kernels.wide_varlen_attention import wide_varlen_attention


HEADS = 16
KV_HEADS = 2
HEAD_DIM = 512
DTYPE = torch.bfloat16
#: Gemma folds the attention scale into `q_norm`, so the layer passes 1.0.
SCALING = 1.0

WARMUP = 3
ITERATIONS = 10

#: Equal lengths are the case length bucketing produces, and the case where the
#: dense penalty is at its `S`-fold worst. Ragged is where padding also bites.
CASES = [
    ("4 x 1024", [1024] * 4),
    ("4 x 2048", [2048] * 4),
    ("8 x 1024", [1024] * 8),
    ("ragged 2048/1280/768/512", [2048, 1280, 768, 512]),
]


def build(lengths, seed=0):
    torch.manual_seed(seed)
    total = sum(lengths)
    longest = max(lengths)

    def packed(heads):
        return torch.randn(1, heads, total, HEAD_DIM, dtype=DTYPE, device="cuda", requires_grad=True)

    def padded(heads):
        return torch.randn(len(lengths), heads, longest, HEAD_DIM, dtype=DTYPE, device="cuda", requires_grad=True)

    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(lengths, dtype=torch.int32, device="cuda").cumsum(0)
    return {
        "lengths": lengths,
        "cu": cu,
        "packed": (packed(HEADS), packed(KV_HEADS), packed(KV_HEADS)),
        "padded": (padded(HEADS), padded(KV_HEADS), padded(KV_HEADS)),
    }


def block_diagonal_mask(lengths, dtype):
    """The mask the dense path needs, additive as ``_as_additive_mask`` makes it."""
    total = sum(lengths)
    positions = torch.arange(total, device="cuda")
    segment = torch.zeros(total, dtype=torch.long, device="cuda")
    start = 0
    for index, length in enumerate(lengths):
        segment[start : start + length] = index
        start += length
    same = segment[:, None] == segment[None, :]
    causal = positions[:, None] >= positions[None, :]
    keep = (same & causal)[None, None]
    return torch.zeros(keep.shape, dtype=dtype, device="cuda").masked_fill_(~keep, torch.finfo(dtype).min)


def padded_causal_mask(lengths, dtype):
    """Left padding plus causality, which is what makes the status-quo mask 4-D."""
    longest = max(lengths)
    positions = torch.arange(longest, device="cuda")
    pads = torch.tensor([longest - length for length in lengths], device="cuda")
    live = positions[None, :] >= pads[:, None]
    causal = positions[:, None] >= positions[None, :]
    keep = (live[:, None, :] & causal[None])[:, None]
    return torch.zeros(keep.shape, dtype=dtype, device="cuda").masked_fill_(~keep, torch.finfo(dtype).min)


def ffpa(query, key, value, mask, is_causal):
    from ffpa_attn import TritonBackend, ffpa_attn_func

    return ffpa_attn_func(
        query,
        key,
        value,
        attn_mask=mask,
        scale=SCALING,
        is_causal=is_causal,
        enable_gqa=key.shape[1] != query.shape[1],
        forward_backend=TritonBackend(),
        backward_backend=TritonBackend(),
    )


def megabytes(tensors):
    """What a path holds before it runs a step, counted once per distinct tensor.

    Reported alongside the transient peak because for two of the three paths it
    *is* the memory story: the unpacked path holds padded copies sized by the
    longest segment, and the dense path holds a mask quadratic in the packed
    length. Both are allocated before the timing window and so are invisible to
    a peak measured inside it -- which is why the transient figure alone must not
    be quoted as the memory comparison.
    """
    seen = {tensor.data_ptr(): tensor.numel() * tensor.element_size() for tensor in tensors}
    return sum(seen.values()) / 2**20


def measure(step):
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


def backward_of(output, tensors):
    torch.autograd.grad(output, tensors, torch.ones_like(output))


def run_case(name, lengths):
    inputs = build(lengths)
    query, key, value = inputs["packed"]
    padded_query, padded_key, padded_value = inputs["padded"]
    equal = len(set(lengths)) == 1
    results = {}

    def varlen():
        backward_of(
            wide_varlen_attention(query, key, value, inputs["cu"], max_seqlen=max(lengths), sm_scale=SCALING),
            [query, key, value],
        )

    dense_mask = block_diagonal_mask(lengths, DTYPE)

    def dense():
        backward_of(ffpa(query, key, value, dense_mask, False), [query, key, value])

    # Equal lengths need no padding, so transformers hands the layer no mask and
    # FFPA takes its own causal path; ragged ones get the 4-D one.
    status_quo_mask = None if equal else padded_causal_mask(lengths, DTYPE)

    def unpacked():
        backward_of(
            ffpa(padded_query, padded_key, padded_value, status_quo_mask, status_quo_mask is None),
            [padded_query, padded_key, padded_value],
        )

    resident = {
        "unpacked": megabytes([padded_query, padded_key, padded_value] + ([] if equal else [status_quo_mask])),
        "dense": megabytes([query, key, value, dense_mask]),
        "varlen": megabytes([query, key, value, inputs["cu"]]),
    }

    for label, step in (("unpacked", unpacked), ("dense", dense), ("varlen", varlen)):
        try:
            results[label] = measure(step)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[label] = (float("nan"), float("nan"))
        except Exception as error:  # noqa: BLE001
            results[label] = (f"{type(error).__name__}", str(error)[:60])

    total = sum(lengths)
    useful = sum(length * length for length in lengths)
    mask_mb = dense_mask.numel() * dense_mask.element_size() / 2**20
    print(f"\n{name}  packed {total} tokens")
    print(
        f"  dense forms {total * total / useful:.2f}x the score entries it needs"
        f"; its mask alone is {mask_mb:.0f} MB per layer"
    )
    print(f"  {'path':>10}  {'time':>12}  {'transient':>12}  {'resident':>11}  {'total':>10}")
    for label in ("unpacked", "dense", "varlen"):
        value_pair = results[label]
        if isinstance(value_pair[0], str):
            print(f"  {label:>10}  {value_pair[0]:>12}  {value_pair[1]}")
        else:
            held = resident[label]
            print(
                f"  {label:>10}  {value_pair[0]:>9.2f} ms  {value_pair[1]:>9.0f} MB"
                f"  {held:>8.0f} MB  {value_pair[1] + held:>7.0f} MB"
            )
    return results


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a GPU")
    if not FFPA_AVAILABLE:
        raise SystemExit("this benchmark needs ffpa-attn: uv pip install --no-deps ffpa-attn")
    print(f"{torch.cuda.get_device_name()}, {HEADS} heads over {KV_HEADS} kv heads, head_dim {HEAD_DIM}, {DTYPE}")
    for name, lengths in CASES:
        run_case(name, lengths)


if __name__ == "__main__":
    main()
