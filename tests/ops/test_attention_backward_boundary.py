import torch

from veomni.ops.kernels.attention.backward_boundary import (
    attention_backward_phase_callback,
    register_attention_backward_boundary,
)


def test_attention_boundary_runs_immediately_before_attention_backward() -> None:
    order = []

    class FakeAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value * 2

        @staticmethod
        def backward(ctx, grad_output):
            order.append("attention_backward")
            return grad_output * 2

    value = torch.ones(2, requires_grad=True)
    with attention_backward_phase_callback(lambda phase: order.append(phase)):
        attention_output = FakeAttention.apply(value)
        register_attention_backward_boundary(attention_output).sum().backward()

    assert order == ["attention", "attention_backward"]


def test_attention_boundary_is_noop_without_callback() -> None:
    value = torch.ones(2, requires_grad=True)

    assert register_attention_backward_boundary(value) is value
