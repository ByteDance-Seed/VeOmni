# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2026, Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: BSD-3-Clause
# Ported from MindSpeed RingP2P (BSD-3-Clause) for Open-VeOmni context parallel.
"""Ring isend/irecv helpers for context-parallel KV exchange."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch import Tensor


TensorOrList = Union[Tensor, List[Tensor]]


class RingP2P:
    """Even/odd ordered P2P send/recv on a ring of global ranks."""

    def __init__(
        self,
        ring_global_ranks: Sequence[int],
        group: dist.ProcessGroup,
        group_for_send_recv_overlap: Optional[dist.ProcessGroup] = None,
        is_backward: bool = False,
    ) -> None:
        self.group = group
        self.group_for_send_recv_overlap = (
            group if group_for_send_recv_overlap is None else group_for_send_recv_overlap
        )

        global_rank = dist.get_rank()
        ring_rank = list(ring_global_ranks).index(global_rank)
        ring_size = len(ring_global_ranks)
        self.next = ring_global_ranks[(ring_rank + 1) % ring_size]
        self.prev = ring_global_ranks[(ring_rank + ring_size - 1) % ring_size]
        self.ring_rank = ring_rank
        if is_backward:
            self.next, self.prev = self.prev, self.next

        self.send_recv_ops: list[dist.Work] = []
        self._packed_recv = None
        self._single_recv = None

    def async_send_recv(self, send_tensor: TensorOrList, recv_tensor: TensorOrList) -> None:
        """Launch even/odd isend/irecv.

        Besides a single tensor, a mutable non-empty tensor list is packed into
        one flat payload.  The latter keeps Ring attention generic when K/V
        shapes differ and lets backward circulate ``[K, V, dK, dV]`` without
        retaining every remote K/V shard.
        """
        self._packed_recv = None
        self._single_recv = None
        packed = isinstance(send_tensor, list)
        if packed:
            if not isinstance(recv_tensor, list) or not send_tensor or len(send_tensor) != len(recv_tensor):
                raise ValueError("Packed RingP2P requires equally sized non-empty mutable tensor lists.")
            for index, (send_item, recv_item) in enumerate(zip(send_tensor, recv_tensor)):
                if send_item.shape != recv_item.shape:
                    raise ValueError(
                        f"Shape mismatch in packed RingP2P tensor {index}: "
                        f"send={send_item.shape}, recv={recv_item.shape}."
                    )
                if send_item.dtype != send_tensor[0].dtype or recv_item.dtype != send_tensor[0].dtype:
                    raise ValueError("Packed RingP2P tensors must have one common dtype.")
            shapes = tuple(item.shape for item in send_tensor)
            numels = tuple(item.numel() for item in send_tensor)
            send_payload = torch.cat(tuple(item.reshape(-1) for item in send_tensor), dim=0).contiguous()
            recv_payload = torch.empty_like(send_payload)
        else:
            send_payload = send_tensor.contiguous()
            if recv_tensor.is_contiguous():
                recv_payload = recv_tensor
            else:
                recv_payload = torch.empty(recv_tensor.shape, dtype=recv_tensor.dtype, device=recv_tensor.device)
                self._single_recv = (recv_tensor, recv_payload)
            numels = shapes = None

        if self.ring_rank % 2 == 0:
            send_op = dist.isend(send_payload, self.next, self.group)
            recv_op = dist.irecv(recv_payload, self.prev, self.group_for_send_recv_overlap)
            self.send_recv_ops.extend((send_op, recv_op))
        else:
            recv_op = dist.irecv(recv_payload, self.prev, self.group)
            send_op = dist.isend(send_payload, self.next, self.group_for_send_recv_overlap)
            self.send_recv_ops.extend((recv_op, send_op))

        if packed:
            self._packed_recv = (recv_tensor, recv_payload, numels, shapes)

    def wait(self) -> int:
        """Wait for outstanding P2P ops. Returns 1 if work completed, else 0."""
        if not self.send_recv_ops:
            return 0
        for op in self.send_recv_ops:
            op.wait()
        self.send_recv_ops = []
        if self._single_recv is not None:
            recv_tensor, recv_payload = self._single_recv
            recv_tensor.copy_(recv_payload)
            self._single_recv = None
        if self._packed_recv is not None:
            recv_tensor, recv_payload, numels, shapes = self._packed_recv
            offset = 0
            for recv_item, numel, shape in zip(recv_tensor, numels, shapes):
                recv_item.copy_(recv_payload[offset : offset + numel].view(shape))
                offset += numel
            self._packed_recv = None
        return 1
