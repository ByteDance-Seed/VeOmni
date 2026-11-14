"""Two-rank regression for the dense MLP split's residual precision."""

import pytest
import torch
import torch.distributed as dist
from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.launch_utils import torchrun
from torch import nn
from torch.distributed.fsdp import FSDPModule

from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.distributed.torch_parallelize import parallelize_model_fsdp2
from veomni.utils.device import get_device_type


class _ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Linear(16, 16, bias=False)
        self.mlp = nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(), nn.Linear(32, 16, bias=False))

    def forward(self, hidden):
        hidden = hidden + self.attention(hidden)
        return hidden + self.mlp(hidden)


class _ResidualModel(nn.Module):
    _no_split_modules = ["_ResidualBlock"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ResidualBlock(), _ResidualBlock()])

    def init_weights(self):
        # Constant global values initialize identical DTensors regardless of grouping.
        for index, param in enumerate(self.parameters()):
            nn.init.constant_(param, 0.03 + 0.01 * index)

    def forward(self, hidden):
        for block in self.layers:
            hidden = block(hidden)
        return hidden


def _check_precision():
    _init_parallel_state(dp_size=2)
    try:
        models = []
        for split in (False, True):
            with pytest.MonkeyPatch.context() as env:
                env.setenv("VEOMNI_DENSE_MLP_LAYERS", "layers" if split else "")
                env.setenv("VEOMNI_DENSE_MLP_PREFETCH", "1")
                env.setenv("VEOMNI_DENSE_MLP_PREFETCH_NEXT", "1")
                with torch.device("meta"):
                    model = _ResidualModel()
                models.append(
                    parallelize_model_fsdp2(
                        model,
                        init_device="meta",
                        mixed_precision=MixedPrecisionConfig(
                            param_dtype="bfloat16", reduce_dtype="float32", output_dtype="float32"
                        ),
                    )
                )
        assert not isinstance(models[0].layers[0].mlp, FSDPModule)
        assert isinstance(models[1].layers[0].mlp, FSDPModule)

        generator = torch.Generator().manual_seed(42 + dist.get_rank())
        hidden = torch.randn(2, 8, 16, generator=generator).to(get_device_type())
        inputs = [hidden.clone().requires_grad_() for _ in models]
        outputs = []
        for model, value in zip(models, inputs):
            output = model(value)
            assert output.dtype == torch.float32
            outputs.append(output.detach())
            output.square().mean().backward()

        # Casting the child MLP to FP32 moves residual addition out of BF16;
        # exact forward parity catches that even though the block returns FP32.
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=1e-5, atol=1e-6)
        reference = dict(models[0].named_parameters())
        actual = dict(models[1].named_parameters())
        assert reference.keys() == actual.keys()
        for name, param in reference.items():
            assert param.grad is not None, name
            assert actual[name].grad is not None, name
            torch.testing.assert_close(
                param.grad.full_tensor(), actual[name].grad.full_tensor(), rtol=1e-5, atol=1e-6, msg=name
            )
    finally:
        clear_parallel_state()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
def test_dense_mlp_bf16_fp32_output_parity():
    if runner_device_count() < 2:
        pytest.skip("Requires two accelerator devices")
    torchrun(_check_precision, world_size=2)
