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

from threading import Lock

import torch

from ...utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, create_event, get_device_type, get_torch_device


def _device_api(tensor: torch.Tensor):
    if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) or tensor.device.type != get_device_type():
        raise RuntimeError(f"Replica buffer stream tracking is unavailable for {tensor.device}.")
    return get_torch_device()


class ReplicaBufferLease:
    """Exclusive storage until release; callers must retain this owner, not just its tensor.

    Record use after every asynchronous consumer submission. Release is not a
    synchronization operation and must happen only after the last consumer has
    relinquished ownership, including retained autograd graphs.
    """

    def __init__(self, pool: "ReplicaBufferPool", tensor: torch.Tensor) -> None:
        self._pool = pool
        self._tensor = tensor
        self._events = []
        self._eligible = False
        self._failed = False

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            raise RuntimeError("Replica buffer lease has been released.")
        return self._tensor

    def record_use(self) -> None:
        """Record completion and allocator ownership on the tensor's current stream."""
        tensor = self.tensor
        try:
            if tensor.device.type != "cpu":
                api = _device_api(tensor)
                with api.device(tensor.device):
                    stream = api.current_stream()
                    tensor.record_stream(stream)
                    event = create_event()
                    event.record(stream)
                    self._events.append(event)
            self._eligible = not self._failed
        except Exception:
            self._failed = True
            self._eligible = False
            raise

    def release(self) -> None:
        """Return completed storage to the bounded cache, at most once."""
        if self._tensor is None:
            return
        tensor, self._tensor = self._tensor, None
        if self._eligible:
            self._pool._put(tensor, self._events)
        self._events = []

    def __del__(self) -> None:
        self.release()


class ReplicaBufferPool:
    """A per-layer bounded idle-storage cache, never a cache of weight values."""

    def __init__(self, max_cached: int = 2) -> None:
        if not isinstance(max_cached, int) or max_cached < 0:
            raise ValueError("max_cached must be a nonnegative integer.")
        self._max_cached = max_cached
        self._idle = []
        self._lock = Lock()

    def acquire(self, reference: torch.Tensor, shape: tuple[int, ...]) -> ReplicaBufferLease:
        """Borrow zeroed storage, waiting on its previous uses without a host wait."""
        shape = tuple(shape)
        tensor = None
        events = []
        with self._lock:
            for index, (candidate, _) in enumerate(self._idle):
                if (
                    candidate.shape == shape
                    and candidate.dtype == reference.dtype
                    and candidate.device == reference.device
                ):
                    tensor, events = self._idle.pop(index)
                    break
        # Inference forwards must not leave inference-only tensors in a cache
        # that a later training forward may borrow.
        with torch.inference_mode(False), torch.no_grad():
            if tensor is None:
                tensor = reference.new_empty(shape)
            if tensor.device.type != "cpu":
                api = _device_api(tensor)
                with api.device(tensor.device):
                    stream = api.current_stream()
                    tensor.record_stream(stream)
                    for event in events:
                        stream.wait_event(event)
                    tensor.zero_()
            else:
                tensor.zero_()
        return ReplicaBufferLease(self, tensor)

    def _put(self, tensor: torch.Tensor, events: list) -> None:
        with self._lock:
            if len(self._idle) < self._max_cached:
                self._idle.append((tensor, events))
