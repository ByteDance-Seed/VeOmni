"""Packed (TND) layout of a multi-sample Cosmos3 micro-batch.

The reasoner pass runs on the packed ``und`` sequence (``und_cu`` on both sides).
The generation pass takes queries in packed ``gen`` order (``gen_cu``) and keys in
per-sample interleaved ``[und_i ; gen_i]`` order (``joint_cu``), so each sample's
keys form one contiguous range a varlen kernel can address.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import Tensor


def _cumulative(lengths: tuple[int, ...], *, device) -> Tensor:
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)


@dataclass(frozen=True)
class PackedLayout:
    und_lengths: tuple[int, ...]
    gen_lengths: tuple[int, ...]
    und_cu: Tensor
    gen_cu: Tensor
    joint_cu: Tensor

    @property
    def joint_lengths(self) -> tuple[int, ...]:
        return tuple(und + gen for und, gen in zip(self.und_lengths, self.gen_lengths))


def build_packed_layout(und_lengths, gen_lengths, *, device=None) -> PackedLayout:
    und = tuple(int(length) for length in und_lengths)
    gen = tuple(int(length) for length in gen_lengths)
    if len(und) != len(gen) or not und or min(und + gen) <= 0:
        raise ValueError(f"a pack needs one positive (und, gen) length pair per sample, got {und} and {gen}.")
    return PackedLayout(
        und_lengths=und,
        gen_lengths=gen,
        und_cu=_cumulative(und, device=device),
        gen_cu=_cumulative(gen, device=device),
        joint_cu=_cumulative(tuple(a + b for a, b in zip(und, gen)), device=device),
    )


def interleave_und_gen(und: Tensor, gen: Tensor, layout: PackedLayout) -> Tensor:
    """``[und_0..und_n]`` + ``[gen_0..gen_n]`` -> ``[und_0; gen_0; und_1; gen_1; ...]``."""
    parts: list[Tensor] = []
    for und_part, gen_part in zip(und.split(list(layout.und_lengths)), gen.split(list(layout.gen_lengths))):
        parts.extend((und_part, gen_part))
    return torch.cat(parts, dim=0)
