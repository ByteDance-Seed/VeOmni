from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

import torch


AttentionBackwardPhaseCallback = Callable[[str], None]
_attention_backward_phase_callback: ContextVar[Optional[AttentionBackwardPhaseCallback]] = ContextVar(
    "attention_backward_phase_callback",
    default=None,
)


@contextmanager
def attention_backward_phase_callback(
    callback: Optional[AttentionBackwardPhaseCallback],
) -> Iterator[None]:
    token = _attention_backward_phase_callback.set(callback)
    try:
        yield
    finally:
        _attention_backward_phase_callback.reset(token)


class _AttentionBackwardBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, callback: AttentionBackwardPhaseCallback) -> torch.Tensor:
        ctx.callback = callback
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        ctx.callback("attention")
        return grad_output, None


def register_attention_backward_boundary(value: torch.Tensor) -> torch.Tensor:
    callback = _attention_backward_phase_callback.get()
    if callback is None or not value.requires_grad:
        return value
    return _AttentionBackwardBoundary.apply(value, callback)
