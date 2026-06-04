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

"""Variable-shape tensor list flatten / unflatten helpers."""

from __future__ import annotations

import torch


def naflatten(
    hid: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.LongTensor]:
    """Concatenate variable-shape tensors; return flat tensor and per-element prefix shapes ``(b, n)``."""
    assert len(hid) > 0
    device = hid[0].device
    prefix_rows: list[torch.Tensor] = []
    pieces: list[torch.Tensor] = []
    for x in hid:
        if x.dim() == 1:
            prefix_rows.append(torch.tensor([x.numel()], device=device, dtype=torch.long))
            pieces.append(x.reshape(-1))
        else:
            prefix_rows.append(torch.tensor(x.shape[:-1], device=device, dtype=torch.long))
            pieces.append(x.flatten(0, -2))
    shape = torch.stack(prefix_rows)
    return torch.cat(pieces), shape


def unflatten(
    hid: torch.Tensor,
    hid_shape: torch.LongTensor,
) -> list[torch.Tensor]:
    """Split a flat tensor using prefix shapes from :func:`naflatten`."""
    hid_len = hid_shape.prod(-1)
    chunks = hid.split(hid_len.tolist())
    return [x.unflatten(0, s.tolist()) for x, s in zip(chunks, hid_shape, strict=True)]
