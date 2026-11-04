# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Bucket layout and rolling hash for DeepSeek-V4.1's n-gram memory.

Ported from the released ``inference/engram.py``. Two deliberate differences:

* **No ``sympy``.** The prime draw is reproduced with trial division, which
  returns the same primes (primality is not implementation-dependent) without
  adding a dependency for 48 numbers near 16M.
* **The position cache is optional.** Training is prefill-only over the whole
  packed sequence, so there is nothing to carry across calls and the lookback
  reads the current window directly. The cache exists for the generation FSM,
  where decode appends one position at a time (see ``DeepseekV41NgramCache``).

The compressed-token map is NOT built here. It depends on the tokenizer, which
belongs to the text-encoder module; the convert step bakes it into this
module's checkpoint as a persistent buffer instead.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


DEAD = -1
"""Cache sentinel for a position that takes no part in an n-gram (an image span)."""


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True


def find_next_prime(start: int, seen: set) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_bucket_primes(n_layers: int, max_ngram_size: int, n_heads: int, vocab_size: int) -> List[List[List[int]]]:
    """Per ``[layer][n-gram size][head]`` bucket modulus.

    Drawn in order and never reused, which is what keeps the ranges disjoint:
    each (n-gram size, head) pair owns a contiguous slice of its layer's table.
    """
    primes: List[List[List[int]]] = []
    seen: set = set()
    for _ in range(n_layers):
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def build_hash_multipliers(layer_ids: Sequence[int], max_ngram_size: int, compressed_vocab_size: int) -> torch.Tensor:
    """One odd multiplier per ``(layer, lookback)``, from a per-layer RNG.

    Seeding by ``layer_id`` is what makes the two engram layers hash the same
    n-gram to unrelated buckets. The bound keeps ``token_id * multiplier``
    inside int64.
    """
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def build_bucket_offsets(primes: List[List[List[int]]]) -> torch.Tensor:
    """Start row of each ``(n-gram size, head)`` bucket within its layer's table."""
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
    return torch.tensor(np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat]))


def gather_lookback(
    compressed: torch.Tensor,
    max_ngram_size: int,
    history: Optional[torch.Tensor] = None,
    start_pos: int = 0,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack each position with the ``max_ngram_size - 1`` compressed ids before it.

    ``compressed`` is ``[B, L]`` in the compressed id space, already carrying
    :data:`DEAD` where the token is an image. ``history`` (``[B, start_pos]``,
    decode only) supplies positions before this call's window. ``position_ids``
    (``[B, L]``, packed training) restarts at zero for each sample so lookback
    does not cross sample boundaries.

    Returns ``(tokens, blocked)``: ``tokens`` is ``[B, L, max_ngram_size]`` with
    blocked slots left as :data:`DEAD` for the caller to fill with its pad id,
    and ``blocked`` is ``[B, L, max_ngram_size]``.

    Blocking accumulates along the lookback on purpose: once the walk hits the
    start of the sequence or a dead token, every deeper slot is blocked too, so
    an n-gram never spans an image or reaches past position 0.
    """
    batch, seqlen = compressed.shape
    window = compressed if history is None else torch.cat([history, compressed], dim=1)
    offset = 0 if history is None else history.shape[1]

    row_positions = torch.arange(start_pos, start_pos + seqlen, device=compressed.device).expand(batch, seqlen)
    sample_positions = row_positions if position_ids is None else position_ids.to(compressed.device)
    tokens, blocked_slots = [], []
    blocked = torch.zeros_like(row_positions, dtype=torch.bool)
    for shift in range(max_ngram_size):
        index = (row_positions - start_pos + offset - shift).clamp_min(0)
        source = window.gather(1, index)
        blocked = blocked | (sample_positions < shift) | (source == DEAD)
        tokens.append(source)
        blocked_slots.append(blocked)
    return torch.stack(tokens, dim=-1), torch.stack(blocked_slots, dim=-1)


def rolling_hash(
    tokens: torch.Tensor, multipliers: torch.Tensor, primes: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Hash ids of the n-grams ending at each position.

    ``tokens`` ``[B, L, max_ngram_size]`` -> ``[B, L, n_layers, n_hash_cols]``.

    The running XOR after step ``i`` is the hash of the ``(i+1)``-gram, so one
    pass yields the 2-gram .. ``max_ngram_size``-gram ids together. Each lands
    in its own prime-sized bucket range via ``% prime + offset``.
    """
    products = tokens.unsqueeze(2) * multipliers
    rolling, hashes = products[..., 0], []
    for i in range(1, tokens.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets
