"""Exact coalesced GDN permutations, including padding, subgroups and backward."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.distributed.sequence_parallel.gdn_exchange import coalesced_gather_aligned, pack
from veomni.models.configuration_base import as_veomni_config


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
                assert outputs[0].untyped_storage().data_ptr() == outputs[1].untyped_storage().data_ptr()
                assert not outputs[0].is_contiguous()
                gradients = []
                for index, (value, actual) in enumerate(zip(inputs, outputs)):
                    peers = [torch.empty_like(value) for _ in range(size)]
                    dist.all_gather(peers, value.detach(), group=group)
                    reference = torch.cat([peer.chunk(size, dim=2)[local_rank] for peer in peers], dim=1)
                    assert torch.equal(actual, reference)
                    if index >= 3:
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


def test_pack_rejects_mixed_projection_dtypes():
    """The packing helper rejects mixed dtypes without silently casting them."""
    q = torch.ones(1, 2, 4, 8)
    v = torch.ones(1, 2, 8, 4)
    beta = torch.ones(1, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="identical projection dtypes"):
        pack(q, q, v, beta, beta)


def _native_exchange_worker(rank, device_type, family, mixed_dtype, endpoint):
    """Use actual NCCL/HCCL for packet parity and a packed text-model comparison."""
    import importlib
    from unittest.mock import patch

    from transformers import AutoConfig

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.utils.device import get_dist_comm_backend, get_torch_device

    device_module = get_torch_device()
    device_module.set_device(rank)
    device = torch.device(device_type, rank)
    backend = get_dist_comm_backend()
    dist.init_process_group(backend, store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(37 + rank)
        shapes = [(2, 3, 4, 7)] * 2 + [(2, 3, 8, 5), (2, 3, 8), (2, 3, 8)]
        values = [torch.randn(s, device=device, dtype=torch.bfloat16, requires_grad=True) for s in shapes]
        actual = coalesced_gather_aligned(*values, dist.group.WORLD)
        grads = []
        for source, output in zip(values, actual):
            peers = [torch.empty_like(source) for _ in range(2)]
            dist.all_gather(peers, source.detach(), group=dist.group.WORLD)
            assert torch.equal(output, torch.cat([p.chunk(2, 2)[rank] for p in peers], 1))
            grads.append(torch.randn_like(output))
        torch.autograd.backward(actual, grads)
        for source, grad in zip(values, grads):
            peers = [torch.empty_like(grad) for _ in range(2)]
            dist.all_gather(peers, grad, group=dist.group.WORLD)
            assert torch.equal(source.grad, torch.cat([p.chunk(2, 1)[rank] for p in peers], 2))

        _init_parallel_state(ulysses_size=2, device_type=device_type)
        module = importlib.import_module(
            f"veomni.models.transformers.{family}.generated.patched_modeling_{family}_"
            + ("npu" if device_type == "npu" else "gpu")
        )
        config = AutoConfig.from_pretrained(f"tests/toy_config/{family}_toy/config.json").text_config
        config.num_hidden_layers = 1
        config.layer_types = ["linear_attention"]
        if family == "qwen4_exp":
            config.ple_layer_ids = []
        config.use_cache = False
        config._attn_implementation = "eager"
        kernel = "triton" if device_type == "npu" else "fla"
        models = []
        model_class = getattr(
            module, {"qwen4_exp": "Qwen4Exp", "qwen3_5": "Qwen3_5", "qwen3_5_moe": "Qwen3_5Moe"}[family] + "TextModel"
        )
        for _ in range(2):
            ops = make_eager_ops_config(
                causal_conv1d_implementation=kernel, chunk_gated_delta_rule_implementation=kernel
            )
            as_veomni_config(config, validate=False)._set_ops_implementation(ops)
            models.append(model_class(config).to(device=device, dtype=torch.bfloat16))
        for tensor in (*models[0].parameters(), *models[0].buffers()):
            dist.broadcast(tensor.detach(), src=0)
        models[1].load_state_dict(models[0].state_dict())
        if mixed_dtype:
            for model in models:
                model.layers[0].linear_attn.in_proj_b.register_forward_hook(
                    lambda _module, _inputs, output: output.float()
                )
        torch.manual_seed(39)
        ids = torch.randint(3, config.vocab_size, (1, 32), device=device)[:, rank * 16 : (rank + 1) * 16]
        positions = torch.cat((torch.arange(13), torch.arange(19))).to(device)
        positions = positions[rank * 16 : (rank + 1) * 16].view(1, 1, -1).expand(4, -1, -1)
        bounds = torch.tensor([0, 13, 32], device=device, dtype=torch.int32)
        outputs = []
        coalesced_calls = []
        separate_calls = []
        original_exchange = module.coalesced_gather_aligned
        original_separate = module.gather_seq_scatter_heads

        def record_exchange(*args, **kwargs):
            """Prove the default model actually enters the coalesced branch."""
            coalesced_calls.append(True)
            return original_exchange(*args, **kwargs)

        def record_separate(*args, **kwargs):
            separate_calls.append(True)
            return original_separate(*args, **kwargs)

        def separate_exchange(q, k, v, beta, decay, group):
            """Independent reference: the previous five-collective model path."""
            return tuple(
                module.gather_seq_scatter_heads(value, seq_dim=1, head_dim=2, group=group)
                for value in (q, k, v, beta, decay)
            )

        for model, exchange in zip(models, (separate_exchange, record_exchange), strict=True):
            separate_calls.clear()
            with (
                patch.object(module, "coalesced_gather_aligned", exchange),
                patch.object(module, "gather_seq_scatter_heads", record_separate),
            ):
                output = model(
                    input_ids=ids,
                    position_ids=positions,
                    use_cache=False,
                    cu_seq_lens_q=bounds,
                    packed_seq_lens=(13, 19),
                ).last_hidden_state
                output.float().square().mean().backward()
            outputs.append(output.detach())
        assert len(coalesced_calls) == (0 if mixed_dtype else 1)
        assert len(separate_calls) == (5 if mixed_dtype else 0)
        torch.testing.assert_close(outputs[0], outputs[1], atol=1e-3, rtol=1e-3)
        for (name, ref), (other_name, actual_param) in zip(models[0].named_parameters(), models[1].named_parameters()):
            assert name == other_name
            assert (ref.grad is None) == (actual_param.grad is None), name
            if ref.grad is not None:
                torch.testing.assert_close(actual_param.grad, ref.grad, atol=2e-2, rtol=2e-2, msg=name)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
@pytest.mark.parametrize("family", ("qwen4_exp", "qwen3_5", "qwen3_5_moe"))
@pytest.mark.parametrize("mixed_dtype", (False, True))
def test_native_coalesced_exchange_and_packed_model(family, mixed_dtype):
    """Native backend parity is separate from the CPU/Gloo qualification."""
    from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, get_torch_device

    device = get_device_type()
    if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) or get_torch_device().device_count() < 2:
        pytest.skip("Requires two native CUDA or Ascend devices")
    if IS_CUDA_AVAILABLE:
        pytest.importorskip("fla")
    else:
        pytest.importorskip("triton")
    spawn_with_rendezvous(_native_exchange_worker, args=(device, family, mixed_dtype), nprocs=2)
