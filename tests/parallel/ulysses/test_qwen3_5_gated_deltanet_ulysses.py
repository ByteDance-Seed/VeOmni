"""Tests for Ulysses SP in Qwen3_5GatedDeltaNet.

Validates:
1. Conv1d weight slicing correctness (single-GPU, no dist required)
2. SP forward/backward equivalence vs non-SP baseline (multi-GPU)
3. SP forward determinism (multi-GPU)
"""

import os
import random
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


# Only run in CI when ulysses SP or Qwen3.5 model code is touched.
# Skip entirely on NPU — FLA triton kernels are GPU-only.
# Skip if transformers < 5.0.0 — Qwen3.5 is a transformers v5 model.
pytestmark = [
    pytest.mark.qwen3_5_ulysses,
    pytest.mark.skipif(
        IS_NPU_AVAILABLE, reason="Qwen3.5 GatedDeltaNet Ulysses tests require GPU (FLA triton kernels)"
    ),
    pytest.mark.skipif(
        not is_transformers_version_greater_or_equal_to("5.0.0"),
        reason="Qwen3.5 requires transformers >= 5.0.0",
    ),
]


try:
    from fla.modules.convolution import causal_conv1d as causal_conv1d_fn
except Exception:
    causal_conv1d_fn = None

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

_PATCHED_MODULE = "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu"


@pytest.fixture(scope="module", autouse=True)
def _bind_qwen3_5_op_slots():
    """Bind the patched module's OpSlots to FLA before any test runs.

    ``build_foundation_model`` is what normally calls ``_bind_veomni_ops`` to
    resolve each ``OpSlot`` to a concrete kernel. These tests skip that path —
    they construct ``Qwen3_5GatedDeltaNet`` directly to isolate the SP layer —
    so without this fixture every slot stays unbound, ``bound_kernel()``
    returns ``None`` in ``__init__``, and the varlen guard in ``forward``
    raises ``RuntimeError``.

    These tests already require FLA (see the ``causal_conv1d_fn is None`` skip
    inside each test), so binding to the FLA defaults matches existing intent.
    """
    if causal_conv1d_fn is None:
        # No FLA installed → individual tests will skip; nothing to bind.
        return
    import importlib

    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops

    _bind_veomni_ops(importlib.import_module(_PATCHED_MODULE), OpsImplementationConfig())


def _set_deterministic(seed=42):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_device = get_torch_device()
    if torch_device.is_available():
        torch_device.manual_seed(seed)
        torch_device.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _causal_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=None,
        activation=None,
        seq_idx=None,
        backend="triton",
        cu_seqlens=None,
    )[0]
    return out


def _slice_mixed_qkv(
    mixed_qkv: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    sp_size: int,
    sp_rank: int,
) -> torch.Tensor:
    """Slice mixed QKV channels for a given Ulysses rank to match head sharding."""
    key_dim = num_k_heads * head_k_dim

    local_k_heads = num_k_heads // sp_size
    local_v_heads = num_v_heads // sp_size
    local_key_dim = local_k_heads * head_k_dim
    local_value_dim = local_v_heads * head_v_dim

    k_off = sp_rank * local_key_dim
    v_off = sp_rank * local_value_dim

    q = mixed_qkv[:, :, k_off : k_off + local_key_dim]
    k = mixed_qkv[:, :, key_dim + k_off : key_dim + k_off + local_key_dim]
    v = mixed_qkv[:, :, 2 * key_dim + v_off : 2 * key_dim + v_off + local_value_dim]
    return torch.cat([q, k, v], dim=-1)


@dataclass
class _TinyQwen3_5Config:
    hidden_size: int = 128
    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 4
    linear_value_head_dim: int = 2
    linear_conv_kernel_dim: int = 3
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16


def _assert_forward_deterministic(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    repeats: int = 100,
) -> None:
    with torch.no_grad():
        ref = layer(inputs, attention_mask=None, cu_seq_lens_q=None).detach()
        for _ in range(repeats - 1):
            out = layer(inputs, attention_mask=None, cu_seq_lens_q=None)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test 1: Conv1d weight slicing (single GPU, no distributed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bsz", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [5, 8, 256])
@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 4), (16, 32)])
@pytest.mark.parametrize("head_k_dim,head_v_dim", [(4, 4), (4, 2)])
@pytest.mark.parametrize("sp_size", [2])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_lasp_depthwise_conv1d_slicing_matches_full(
    bsz, seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim, sp_size, kernel_size
):
    """Validate local QKV slicing and conv1d weight slicing match full conv output."""
    if causal_conv1d_fn is None or not get_torch_device().is_available():
        pytest.skip("FLA causal_conv1d or accelerator not available")

    from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import Qwen3_5GatedDeltaNet

    _set_deterministic(42)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim

    device = get_device_type()
    mixed_qkv_full = torch.randn(bsz, seq_len, conv_dim, device=device)
    weight_full = torch.randn(conv_dim, kernel_size, device=device)

    config = _TinyQwen3_5Config(
        hidden_size=key_dim * 2 + value_dim,
        linear_num_value_heads=num_v_heads,
        linear_num_key_heads=num_k_heads,
        linear_key_head_dim=head_k_dim,
        linear_value_head_dim=head_v_dim,
        linear_conv_kernel_dim=kernel_size,
    )
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device)
    layer.conv1d.weight.data.copy_(weight_full.unsqueeze(1))

    out_full = _causal_depthwise_conv1d(mixed_qkv_full, weight_full)

    for sp_rank in range(sp_size):
        mixed_qkv_local = _slice_mixed_qkv(
            mixed_qkv_full, num_k_heads, num_v_heads, head_k_dim, head_v_dim, sp_size, sp_rank
        )
        local_key_dim = (num_k_heads // sp_size) * head_k_dim
        local_value_dim = (num_v_heads // sp_size) * head_v_dim
        weight_local = layer._get_local_conv1d_weight(sp_rank, local_key_dim, local_value_dim)

        out_local = _causal_depthwise_conv1d(mixed_qkv_local, weight_local)
        out_expected = _slice_mixed_qkv(out_full, num_k_heads, num_v_heads, head_k_dim, head_v_dim, sp_size, sp_rank)

        torch.testing.assert_close(out_local, out_expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test 2: SP forward/backward equivalence (multi-GPU)
# ---------------------------------------------------------------------------


def _run_gated_deltanet_sp_fw_bw(rank: int, world_size: int, init_file: str, bsz: int, seq_len: int) -> None:
    """Compare SP vs baseline forward outputs and parameter grads for Qwen3_5 GatedDeltaNet."""
    device_type = get_device_type()
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from veomni.distributed.parallel_state import init_parallel_state
    from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import Qwen3_5GatedDeltaNet

    init_parallel_state(dp_size=1, ulysses_size=world_size, device_type=device_type)

    _set_deterministic(42)
    config = _TinyQwen3_5Config()
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device_type)
    layer.train()

    hidden = config.hidden_size

    if rank == 0:
        full_input = torch.randn(bsz, seq_len, hidden, device=device_type)
    else:
        full_input = torch.empty(bsz, seq_len, hidden, device=device_type)
    dist.broadcast(full_input, src=0)

    shard_len = seq_len // world_size
    local_input = full_input[:, rank * shard_len : (rank + 1) * shard_len].contiguous()

    # Baseline forward/backward on rank 0 with SP disabled
    baseline_out = None
    baseline_param_grads = None
    if rank == 0:
        no_sp_state = SimpleNamespace(ulysses_enabled=False)
        with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
            baseline_out = layer(full_input, attention_mask=None, cu_seq_lens_q=None)
            baseline_loss = baseline_out.mean()
            baseline_loss.backward()
            baseline_param_grads = {
                name: param.grad.detach().clone() if param.grad is not None else None
                for name, param in layer.named_parameters()
            }
            layer.zero_grad(set_to_none=True)
            baseline_out = baseline_out.detach()

    dist.barrier()

    # SP forward/backward
    sp_out_local = layer(local_input, attention_mask=None, cu_seq_lens_q=None)
    total_numel = bsz * seq_len * sp_out_local.shape[-1]
    sp_loss = sp_out_local.sum() / total_numel
    sp_loss.backward()

    for param in layer.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

    out_list = [torch.empty_like(sp_out_local) for _ in range(world_size)]
    dist.all_gather(out_list, sp_out_local.detach())
    sp_out_full = torch.cat(out_list, dim=1)

    if rank == 0:
        for name, param in layer.named_parameters():
            baseline_grad = baseline_param_grads.get(name)
            if baseline_grad is None and param.grad is None:
                continue
            assert baseline_grad is not None and param.grad is not None, f"Missing grad for {name}"
            torch.testing.assert_close(
                param.grad,
                baseline_grad,
                rtol=0,
                atol=3e-6,
                msg=lambda msg, n=name: f"{msg}\nGradient mismatch for {n}",
            )

    if rank == 0:
        torch.testing.assert_close(sp_out_full, baseline_out, rtol=0, atol=2e-4)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("bsz", [1, 8, 16, 32])
@pytest.mark.parametrize("seq_len", [256, 2048])
def test_qwen3_5_gated_deltanet_sp_equivalence(world_size, bsz, seq_len):
    """Ensure SP partitioned forward outputs and grads match the non-SP baseline."""
    if causal_conv1d_fn is None or not get_torch_device().is_available():
        pytest.skip("FLA causal_conv1d or accelerator not available")
    if get_torch_device().device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} devices for SP equivalence test")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        init_file = tmp.name

    try:
        mp.spawn(
            _run_gated_deltanet_sp_fw_bw, args=(world_size, init_file, bsz, seq_len), nprocs=world_size, join=True
        )
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)


# ---------------------------------------------------------------------------
# Test 3: Forward determinism (multi-GPU with SP)
# ---------------------------------------------------------------------------


def _run_gated_deltanet_sp_determinism(rank: int, world_size: int, init_file: str, bsz: int, seq_len: int) -> None:
    """Verify deterministic SP forward per rank."""
    device_type = get_device_type()
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from veomni.distributed.parallel_state import init_parallel_state
    from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import Qwen3_5GatedDeltaNet

    init_parallel_state(dp_size=1, ulysses_size=world_size, device_type=device_type)

    _set_deterministic(42)
    config = _TinyQwen3_5Config()
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device_type)
    layer.train()

    hidden = config.hidden_size

    if rank == 0:
        full_input = torch.randn(bsz, seq_len, hidden, device=device_type)
    else:
        full_input = torch.empty(bsz, seq_len, hidden, device=device_type)
    dist.broadcast(full_input, src=0)

    shard_len = seq_len // world_size
    local_input = full_input[:, rank * shard_len : (rank + 1) * shard_len].contiguous()

    _assert_forward_deterministic(layer, local_input, repeats=100)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("bsz", [1, 4])
@pytest.mark.parametrize("seq_len", [8, 2048, 32768])
def test_qwen3_5_gated_deltanet_forward_deterministic_sp(world_size, bsz, seq_len):
    """Ensure repeated SP forward passes are deterministic per rank."""
    if causal_conv1d_fn is None or not get_torch_device().is_available():
        pytest.skip("FLA causal_conv1d or accelerator not available")
    if get_torch_device().device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} devices for SP determinism test")
    if seq_len % world_size != 0:
        pytest.skip(f"seq_len must be divisible by world_size (seq_len={seq_len}, world_size={world_size})")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        init_file = tmp.name

    try:
        mp.spawn(
            _run_gated_deltanet_sp_determinism,
            args=(world_size, init_file, bsz, seq_len),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)


# ---------------------------------------------------------------------------
# Test 4: Forward determinism (single GPU, no SP)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bsz", [1, 4])
@pytest.mark.parametrize("seq_len", [8, 2048, 32768])
def test_qwen3_5_gated_deltanet_forward_deterministic_no_sp(bsz, seq_len):
    """Ensure repeated non-SP forward passes are deterministic."""
    if causal_conv1d_fn is None or not get_torch_device().is_available():
        pytest.skip("FLA causal_conv1d or accelerator not available")

    from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import Qwen3_5GatedDeltaNet

    device_type = get_device_type()

    _set_deterministic(42)
    config = _TinyQwen3_5Config()
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device_type)
    layer.train()

    inputs = torch.randn(bsz, seq_len, config.hidden_size, device=device_type)

    no_sp_state = SimpleNamespace(ulysses_enabled=False)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
        _assert_forward_deterministic(layer, inputs, repeats=100)
