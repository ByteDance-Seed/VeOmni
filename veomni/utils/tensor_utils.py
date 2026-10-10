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
    """Concatenate variable-shape tensors; return flat tensor and per-element prefix shapes ``(b, n)``.

    The returned shape metadata is kept on **CPU** (it is pure host-side
    bookkeeping derived from ``.shape`` / ``.numel()``). This lets
    :func:`unflatten` split with ``.tolist()`` without forcing a device→host
    sync per segment — those syncs otherwise dominate the post-forward of the
    text encoder / backbone (they block on the still-running forward kernels).
    The flat data tensor stays on the input device.
    """
    assert len(hid) > 0
    prefix_rows: list[torch.Tensor] = []
    pieces: list[torch.Tensor] = []
    for x in hid:
        if x.dim() == 1:
            prefix_rows.append(torch.tensor([x.numel()], dtype=torch.long))
            pieces.append(x.reshape(-1))
        else:
            prefix_rows.append(torch.tensor(tuple(x.shape[:-1]), dtype=torch.long))
            pieces.append(x.flatten(0, -2))
    shape = torch.stack(prefix_rows)
    return torch.cat(pieces), shape


def unflatten(
    hid: torch.Tensor,
    hid_shape: torch.LongTensor,
) -> list[torch.Tensor]:
    """Split a flat tensor using prefix shapes from :func:`naflatten`.

    ``hid_shape`` is moved to CPU once (a single sync at most) so the per-segment
    ``.tolist()`` calls below never trigger a device→host sync; with the CPU
    shape produced by :func:`naflatten` this is already a no-op.
    """
    if hid_shape.device.type != "cpu":
        hid_shape = hid_shape.cpu()
    hid_len = hid_shape.prod(-1)
    chunks = hid.split(hid_len.tolist())
    return [x.unflatten(0, s.tolist()) for x, s in zip(chunks, hid_shape, strict=True)]
