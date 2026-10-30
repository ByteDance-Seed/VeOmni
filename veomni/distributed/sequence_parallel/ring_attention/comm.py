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

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


class RingComm:
    """P2P ring communicator over a context-parallel process group."""

    def __init__(self, group: ProcessGroup):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.send_rank = dist.get_global_rank(group, (self.rank + 1) % self.world_size)
        self.recv_rank = dist.get_global_rank(group, (self.rank - 1) % self.world_size)
        self._ops = []
        self._reqs = None

    def send_recv(self, to_send: Tensor, recv_tensor: Optional[Tensor] = None) -> Tensor:
        result = torch.empty_like(to_send) if recv_tensor is None else recv_tensor
        self._ops.append(dist.P2POp(dist.isend, to_send.contiguous(), self.send_rank, group=self.group))
        self._ops.append(dist.P2POp(dist.irecv, result, self.recv_rank, group=self.group))
        return result

    def commit(self) -> None:
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        if self._reqs is not None:
            for request in self._reqs:
                request.wait()
        self._ops = []
        self._reqs = None
