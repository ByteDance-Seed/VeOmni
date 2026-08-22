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

"""Forward-scoped taps for detached loss observability."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch


_ACTIVE_CHUNK_LOSS_CONSUMER: ContextVar[Callable[[torch.Tensor], None] | None] = ContextVar(
    "veomni_active_chunk_loss_consumer",
    default=None,
)


@contextmanager
def capture_chunk_loss_per_token(consumer: Callable[[torch.Tensor], None]) -> Iterator[None]:
    """Route the main chunk-loss kernel's detached per-token CE to ``consumer``."""

    token = _ACTIVE_CHUNK_LOSS_CONSUMER.set(consumer)
    try:
        yield
    finally:
        _ACTIVE_CHUNK_LOSS_CONSUMER.reset(token)


def get_chunk_loss_consumer() -> Callable[[torch.Tensor], None] | None:
    """Return the consumer active for the current model-forward context."""

    return _ACTIVE_CHUNK_LOSS_CONSUMER.get()
