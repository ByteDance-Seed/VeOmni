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
"""Shared-prefix training for packed Qwen3.5 micro-batches.

In GRPO-style RL a micro-batch carries ``n`` rollouts of one prompt, each packed
as ``prompt + response``. Packed training computes the prompt ``n`` times. This
module detects sequences that share a token prefix, runs the decoder stack on a
*compact* row where every shared prefix appears once, and expands the final
hidden states back to the original packed layout, so callers see the same
shapes and gradients sum into the shared prefix through the expansion.

Token-wise work (projections, norms, MLP) runs unchanged on the compact row.
Only the three sequence mixers need the plan, and each runs a fixed number of
kernel calls per layer regardless of the number of groups or rollouts:

* full attention: one varlen call; a prefix attends to itself, a suffix's
  queries attend to ``[prefix | suffix]`` keys (bottom-right causal);
* causal conv1d: one varlen call; each suffix is preceded by the last
  ``kernel_size - 1`` prefix rows as look-back context, whose outputs are dropped;
* gated delta rule: two varlen calls. The prefix runs up to its last chunk
  boundary and exports its state; the remaining prefix tail is replayed in
  front of every suffix, which starts from that state. Exporting state only at
  chunk boundaries keeps the chunk partition identical to unshared training.

The design follows tree training in AReaL and its GDN extension HARTS
(AReaL PR #1765) and OpenPipe ART, specialised to the two-level tree that a
rollout group forms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


GDN_CHUNK_SIZE = 64


@dataclass
class _Group:
    """One shared prefix with its suffixes, in original packed coordinates."""

    starts: list[int]  # start of each member sequence in the packed row
    lengths: list[int]
    prefix_length: int  # 0 < prefix_length < min(lengths) when grouped; == length for a singleton

    @property
    def is_singleton(self) -> bool:
        return len(self.starts) == 1


class SharedPrefixPlan:
    """Index maps between the packed row and its compact form, and per-mixer layouts.

    All index tensors live on ``device``; the cumulative-length lists also have
    CPU copies because the Ascend kernels read sequence lengths on the host.
    """

    def __init__(self, groups: list[_Group], total_length: int, conv_kernel_size: int, device: torch.device):
        self.groups = groups
        self.total_length = total_length
        compact_src: list[int] = []
        expand: list[int] = [0] * total_length

        # attention: queries are the compact rows in order; keys are gathered.
        attn_kv: list[int] = []
        attn_q_lens: list[int] = []
        attn_k_lens: list[int] = []
        # conv: rows gathered from compact, outputs kept for compact rows in order.
        conv_rows: list[int] = []
        conv_lens: list[int] = []
        conv_keep: list[int] = []
        # gated delta rule
        a_rows: list[int] = []
        a_lens: list[int] = []
        a_keep: list[int] = []  # compact rows produced by round A, in round-A output order
        b_rows: list[int] = []
        b_lens: list[int] = []
        b_state: list[int] = []  # index into round-A final states, -1 for the zero state
        b_keep_pos: list[int] = []  # positions in round-B output that produce a compact row
        b_keep_row: list[int] = []

        history = conv_kernel_size - 1
        for group in groups:
            base = len(compact_src)
            prefix = group.prefix_length
            first = group.starts[0]
            compact_src.extend(range(first, first + prefix))
            for start in group.starts:
                for i in range(prefix):
                    expand[start + i] = base + i
            prefix_rows = list(range(base, base + prefix))
            suffix_rows = []
            for start, length in zip(group.starts, group.lengths):
                if group.is_singleton:
                    break
                row0 = len(compact_src)
                compact_src.extend(range(start + prefix, start + length))
                for i in range(length - prefix):
                    expand[start + prefix + i] = row0 + i
                suffix_rows.append(list(range(row0, row0 + length - prefix)))

            # attention
            attn_kv.extend(prefix_rows)
            attn_q_lens.append(prefix)
            attn_k_lens.append(prefix)
            for rows in suffix_rows:
                attn_kv.extend(prefix_rows + rows)
                attn_q_lens.append(len(rows))
                attn_k_lens.append(prefix + len(rows))

            # conv
            conv_keep.extend(range(len(conv_rows), len(conv_rows) + prefix))
            conv_rows.extend(prefix_rows)
            conv_lens.append(prefix)
            context = prefix_rows[-history:] if history else []
            for rows in suffix_rows:
                offset = len(conv_rows) + len(context)
                conv_rows.extend(context + rows)
                conv_lens.append(len(context) + len(rows))
                conv_keep.extend(range(offset, offset + len(rows)))

            # gated delta rule
            if group.is_singleton:
                a_keep.extend(prefix_rows)
                a_rows.extend(prefix_rows)
                a_lens.append(prefix)
                continue
            boundary = prefix // GDN_CHUNK_SIZE * GDN_CHUNK_SIZE
            state = -1
            if boundary:
                state = len(a_lens)
                a_keep.extend(prefix_rows[:boundary])
                a_rows.extend(prefix_rows[:boundary])
                a_lens.append(boundary)
            tail = prefix_rows[boundary:]
            if tail:
                b_keep_pos.extend(range(len(b_rows), len(b_rows) + len(tail)))
                b_keep_row.extend(tail)
                b_rows.extend(tail)
                b_lens.append(len(tail))
                b_state.append(state)
            for rows in suffix_rows:
                offset = len(b_rows) + len(tail)
                b_rows.extend(tail + rows)
                b_lens.append(len(tail) + len(rows))
                b_state.append(state)
                b_keep_pos.extend(range(offset, offset + len(rows)))
                b_keep_row.extend(rows)

        self.compact_length = len(compact_src)

        def tensor(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=device)

        def cumulative(lengths: list[int]) -> tuple[torch.Tensor, list[int]]:
            cu = [0]
            for length in lengths:
                cu.append(cu[-1] + length)
            return torch.tensor(cu, dtype=torch.int32, device=device), cu

        self.compact_src = tensor(compact_src)
        self.expand_index = tensor(expand)

        self.attn_kv_index = tensor(attn_kv)
        self.attn_cu_q, attn_cu_q = cumulative(attn_q_lens)
        self.attn_cu_k, attn_cu_k = cumulative(attn_k_lens)
        self.attn_max_q = max(attn_q_lens)
        self.attn_max_k = max(attn_k_lens)

        self.conv_index = tensor(conv_rows)
        self.conv_cu, _ = cumulative(conv_lens)
        self.conv_keep = tensor(conv_keep)

        self.gdn_a_index = tensor(a_rows)
        self.gdn_a_cu, self.gdn_a_cu_list = cumulative(a_lens)
        self.gdn_b_index = tensor(b_rows)
        self.gdn_b_cu, self.gdn_b_cu_list = cumulative(b_lens)
        self.gdn_b_state = tensor(b_state)
        # Round-A outputs followed by the kept round-B outputs, permuted to compact order.
        produced = a_keep + b_keep_row
        order = [0] * self.compact_length
        for position, row in enumerate(produced):
            order[row] = position
        self.gdn_b_keep = tensor(b_keep_pos)
        self.gdn_order = tensor(order)
        assert sorted(produced) == list(range(self.compact_length)), "every compact row must be produced once"

    # ------------------------------------------------------------------ layout

    def compact(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Packed ``[..., T, ...]`` -> compact ``[..., C, ...]`` along ``dim``."""
        return tensor.index_select(dim, self.compact_src)

    def expand(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Compact -> packed. Backward sums the gradients of every copy of a shared row."""
        return tensor.index_select(dim, self.expand_index)

    def lm_rows(self, shift_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rows the LM head needs: one per distinct ``(compact row, label)`` pair.

        A shared prefix row is scored once when every member predicts the same
        next token there; the last prefix row, whose next token differs per
        member, is scored once per distinct label. Returns ``(rows, labels,
        inverse)``: compact rows ``[L]``, their labels ``[L]``, and the map from
        each packed position to its row ``[T]``.
        """
        labels = shift_labels.reshape(-1).long()
        if labels.numel() != self.total_length:
            raise ValueError(f"shift_labels has {labels.numel()} tokens, the plan expects {self.total_length}")
        offset = labels.min().clamp_max(0)
        span = labels.max() - offset + 1
        keys = self.expand_index * span + (labels - offset)
        unique, inverse = torch.unique(keys, return_inverse=True)
        return unique // span, unique % span + offset, inverse

    # ------------------------------------------------------------------ mixers

    def attention_kv(self, key: torch.Tensor, value: torch.Tensor, seq_dim: int):
        """Gather ``[prefix | suffix]`` keys and values for the single varlen call."""
        return key.index_select(seq_dim, self.attn_kv_index), value.index_select(seq_dim, self.attn_kv_index)

    def attention_kwargs(self) -> dict:
        return {
            "cu_seq_lens_q": self.attn_cu_q,
            "cu_seq_lens_k": self.attn_cu_k,
            "max_length_q": self.attn_max_q,
            "max_length_k": self.attn_max_k,
        }

    def causal_conv1d(self, conv: Callable[..., torch.Tensor], x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run a varlen ``conv(x=[B, T, D], cu_seqlens=...)`` on the compact row ``x``."""
        rows = x.index_select(1, self.conv_index)
        out = conv(x=rows, cu_seqlens=self.conv_cu, **kwargs)
        out = out[0] if isinstance(out, tuple) else out
        return out.index_select(1, self.conv_keep)

    def gated_delta_rule(
        self,
        kernel: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Two varlen calls of ``kernel`` (FLA ``chunk_gated_delta_rule`` signature) over the compact row."""
        inputs = (query, key, value, g, beta)
        out_a, states = kernel(
            *(t.index_select(1, self.gdn_a_index) for t in inputs),
            initial_state=None,
            output_final_state=True,
            cu_seqlens=self.gdn_a_cu,
            **kwargs,
        )
        pieces = [out_a]
        if self.gdn_b_index.numel():
            zero = states.new_zeros((1, *states.shape[1:]))
            initial = torch.cat([states, zero]).index_select(0, self.gdn_b_state % (states.shape[0] + 1))
            out_b, _ = kernel(
                *(t.index_select(1, self.gdn_b_index) for t in inputs),
                initial_state=initial,
                output_final_state=False,
                cu_seqlens=self.gdn_b_cu,
                **kwargs,
            )
            pieces.append(out_b.index_select(1, self.gdn_b_keep))
        return torch.cat(pieces, dim=1).index_select(1, self.gdn_order)


def build_shared_prefix_plan(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    conv_kernel_size: int = 4,
    min_prefix_length: int = GDN_CHUNK_SIZE,
) -> SharedPrefixPlan | None:
    """Group packed sequences by their longest common token prefix.

    ``input_ids`` is ``[1, T]``; ``cu_seqlens`` delimits the packed sequences.
    Sequences are grouped when they share at least ``min_prefix_length`` leading
    tokens (and, if given, identical ``position_ids`` over that prefix). Every
    member keeps at least one suffix token. Returns ``None`` when nothing is shared.
    """
    ids = input_ids.reshape(-1).cpu()
    cu = cu_seqlens.cpu().tolist()
    pos = None
    if position_ids is not None:
        pos = position_ids.reshape(-1, position_ids.shape[-1]).cpu()

    buckets: dict[bytes, list[int]] = {}
    for i in range(len(cu) - 1):
        start, end = cu[i], cu[i + 1]
        if end - start > min_prefix_length:
            key = ids[start : start + min_prefix_length].numpy().tobytes()
            buckets.setdefault(key, []).append(i)

    def common_prefix(a: int, b: int) -> int:
        """Shared leading tokens (and positions) of packed sequences ``a`` and ``b``, leaving each a suffix."""
        span = min(cu[a + 1] - cu[a], cu[b + 1] - cu[b]) - 1
        same = ids[cu[a] : cu[a] + span] == ids[cu[b] : cu[b] + span]
        if pos is not None:
            same &= (pos[:, cu[a] : cu[a] + span] == pos[:, cu[b] : cu[b] + span]).all(0)
        return int(same.int().cumprod(0).sum())

    grouped: dict[int, _Group] = {}
    for members in buckets.values():
        if len(members) < 2:
            continue
        # Different prompts can share a head (e.g. a system prompt). Sorting puts sequences with longer
        # common prefixes next to each other; then split the sorted run into the groups that save the
        # most tokens, a group of size m with common prefix p saving (m - 1) * p.
        members = sorted(members, key=lambda i: ids[cu[i] : cu[i + 1]].numpy().tobytes())
        adjacent = [common_prefix(a, b) for a, b in zip(members[:-1], members[1:])]
        best = [0] * (len(members) + 1)
        cut = [0] * (len(members) + 1)
        for j in range(1, len(members) + 1):
            best[j], cut[j] = best[j - 1], j - 1
            prefix = None
            for i in range(j - 2, -1, -1):
                prefix = adjacent[i] if prefix is None else min(prefix, adjacent[i])
                if prefix < min_prefix_length:
                    break
                saved = best[i] + (j - i - 1) * prefix
                if saved > best[j]:
                    best[j], cut[j] = saved, i
        j = len(members)
        while j > 0:
            i = cut[j]
            if j - i > 1:
                run = members[i:j]
                prefix = min(adjacent[i : j - 1])
                grouped[run[0]] = _Group([cu[k] for k in run], [cu[k + 1] - cu[k] for k in run], prefix)
                for k in run[1:]:
                    grouped[k] = None  # absorbed
            j = i

    if not any(g is not None for g in grouped.values()):
        return None

    groups: list[_Group] = []
    for i in range(len(cu) - 1):
        if cu[i + 1] == cu[i]:
            continue
        if i in grouped:
            if grouped[i] is not None:
                groups.append(grouped[i])
        else:
            length = cu[i + 1] - cu[i]
            groups.append(_Group([cu[i]], [length], length))
    return SharedPrefixPlan(groups, cu[-1], conv_kernel_size, input_ids.device)
