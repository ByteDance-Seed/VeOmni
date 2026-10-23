"""Exact coalesced GDN permutations, including padding, subgroups and backward."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.distributed.sequence_parallel.gdn_exchange import coalesced_gather_aligned, pack


def _worker(rank, world, endpoint):
    """Compare with independent rank-major gather and inverse gradient slicing."""
    dist.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world, timeout=timedelta(seconds=90)
    )
    try:
        even = dist.new_group([0, 2])
        odd = dist.new_group([1, 3])
        for group in (dist.group.WORLD, even if rank % 2 == 0 else odd):
            size, local_rank = dist.get_world_size(group), dist.get_rank(group)
            for dtype in (torch.float32, torch.bfloat16):
                torch.manual_seed(rank + 3)
                shapes = [(2, 3, 8, 7)] * 2 + [(2, 3, 16, 5), (2, 3, 16), (2, 3, 16)]
                inputs = [torch.randn(shape, dtype=dtype, requires_grad=True) for shape in shapes]
                outputs = coalesced_gather_aligned(*inputs, group)
                gradients = []
                for value, actual in zip(inputs, outputs):
                    peers = [torch.empty_like(value) for _ in range(size)]
                    dist.all_gather(peers, value.detach(), group=group)
                    reference = torch.cat([peer.chunk(size, dim=2)[local_rank] for peer in peers], dim=1)
                    assert torch.equal(actual, reference)
                    assert actual.is_contiguous()
                    grad = torch.randn_like(actual)
                    gradients.append(grad)
                torch.autograd.backward(outputs, gradients)
                for value, grad in zip(inputs, gradients):
                    peers = [torch.empty_like(grad) for _ in range(size)]
                    dist.all_gather(peers, grad, group=group)
                    reference = torch.cat([peer.chunk(size, dim=1)[local_rank] for peer in peers], dim=2)
                    assert torch.equal(value.grad, reference)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_coalesced_exchange_exact_forward_backward():
    """Run world-size four and noncontiguous two-rank subgroups on CPU."""
    spawn_with_rendezvous(_worker, args=(4,), nprocs=4)


def test_mixed_dtype_is_not_silently_cast():
    """The model must use its separate-exchange fallback for mixed dtypes."""
    q = torch.ones(1, 2, 4, 8)
    v = torch.ones(1, 2, 8, 4)
    beta = torch.ones(1, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="identical projection dtypes"):
        pack(q, q, v, beta, beta)


@pytest.mark.parametrize("enabled", (False, True))
def test_composite_config_reaches_registered_model(enabled):
    """The documented top-level override must reach every text GDN layer."""
    from transformers import Qwen4ExpConfig

    from tests.models.compare import ops_config_scope
    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.registry import MODELING_REGISTRY

    config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json")
    config._attn_implementation = "eager"
    ops = make_eager_ops_config()
    ops.coalesced_gdn_exchange = enabled
    with ops_config_scope(ops):
        model = MODELING_REGISTRY["qwen4_exp"]("Qwen4ExpForConditionalGeneration")(config)
    layers = [module for module in model.modules() if hasattr(module, "coalesced_gdn_exchange")]
    assert layers
    assert all(module.coalesced_gdn_exchange is enabled for module in layers)
