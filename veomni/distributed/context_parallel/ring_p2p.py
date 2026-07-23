# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Ported from MindSpeed RingP2P (BSD-3-Clause) for Open-VeOmni context parallel.
"""Ring isend/irecv helpers for context-parallel KV exchange."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch import Tensor


TensorOrPair = Union[Tensor, List[Tensor]]


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

    def async_send_recv(self, send_tensor: TensorOrPair, recv_tensor: TensorOrPair) -> None:
        """Launch even/odd isend/irecv. Supports a single tensor or a mutable [K, V] list."""
        self._packed_recv = None
        packed = isinstance(send_tensor, list)
        if packed:
            if not isinstance(recv_tensor, list) or len(send_tensor) != 2 or len(recv_tensor) != 2:
                raise ValueError("Packed RingP2P tensors must be length-2 mutable lists [K, V].")
            k_send, v_send = send_tensor
            k_recv, v_recv = recv_tensor
            if k_send.shape != k_recv.shape or v_send.shape != v_recv.shape:
                raise ValueError(
                    "Shape mismatch in KV tensors:\n"
                    f"  k_send: {k_send.shape} vs k_recv: {k_recv.shape}\n"
                    f"  v_send: {v_send.shape} vs v_recv: {v_recv.shape}"
                )
            k_shape, v_shape = k_send.shape, v_send.shape
            k_numel = k_send.numel()
            send_payload = torch.cat((k_send.reshape(-1), v_send.reshape(-1)), dim=0).contiguous()
            recv_payload = torch.empty_like(send_payload)
        else:
            send_payload = send_tensor.contiguous()
            recv_payload = recv_tensor
            k_numel = k_shape = v_shape = None

        if self.ring_rank % 2 == 0:
            send_op = dist.isend(send_payload, self.next, self.group)
            recv_op = dist.irecv(recv_payload, self.prev, self.group_for_send_recv_overlap)
            self.send_recv_ops.extend((send_op, recv_op))
        else:
            recv_op = dist.irecv(recv_payload, self.prev, self.group)
            send_op = dist.isend(send_payload, self.next, self.group_for_send_recv_overlap)
            self.send_recv_ops.extend((recv_op, send_op))

        if packed:
            self._packed_recv = (recv_tensor, recv_payload, k_numel, k_shape, v_shape)

    def wait(self) -> int:
        """Wait for outstanding P2P ops. Returns 1 if work completed, else 0."""
        if not self.send_recv_ops:
            return 0
        for op in self.send_recv_ops:
            op.wait()
        self.send_recv_ops = []
        if self._packed_recv is not None:
            recv_tensor, recv_payload, k_numel, k_shape, v_shape = self._packed_recv
            recv_tensor[0] = recv_payload[:k_numel].view(k_shape)
            recv_tensor[1] = recv_payload[k_numel:].view(v_shape)
            self._packed_recv = None
        return 1
