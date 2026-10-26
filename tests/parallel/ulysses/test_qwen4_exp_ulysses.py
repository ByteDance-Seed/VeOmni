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

"""Qwen4-Exp component correctness under Ulysses sequence sharding."""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import torchrun
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, get_torch_device


_CI_ACCELERATOR_2 = LaunchSpec(nproc=2, device="accelerator")

_CI_ACCELERATOR_4 = LaunchSpec(nproc=4, device="accelerator")


pytestmark = [pytest.mark.qwen4_exp_ulysses]
_CUDA_ONLY = pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="This reference case requires CUDA/FLA kernels")

if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE):
    pytest.skip("Qwen4-Exp Ulysses tests require CUDA or Ascend", allow_module_level=True)

_PATCHED_MODULE = "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_" + (
    "npu" if get_device_type() == "npu" else "gpu"
)
_TOY_CONFIG = "tests/toy_config/qwen4_exp_toy"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if get_device_type() == "cuda":
    try:
        from fla.modules.convolution import causal_conv1d as _causal_conv1d_fn  # noqa: F401
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _chunk_gated_delta_rule_fn  # noqa: F401
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func  # noqa: F401
    except ImportError:
        pytest.skip("Qwen4-Exp Ulysses tests require FLA and FlashAttention kernels", allow_module_level=True)


def _install_qwen4_exp_ops_config(qsa_attention_implementation: str = "eager") -> None:
    """Install the device's native GDN kernels; NPU CI selects the hybrid cases."""
    from veomni.ops.config import set_ops_config

    npu = get_device_type() == "npu"
    set_ops_config(
        SimpleNamespace(
            rms_norm_gated_implementation="eager" if npu else "fla",
            causal_conv1d_implementation="triton" if npu else "fla",
            chunk_gated_delta_rule_implementation="triton" if npu else "fla",
            qsa_attention_implementation=qsa_attention_implementation,
        )
    )


@pytest.fixture(scope="module", autouse=True)
def _configured_qwen4_exp_kernels():
    """Scope the default selections to this module's in-process tests."""
    from veomni.ops.config import get_ops_config, set_ops_config

    previous = get_ops_config()
    _install_qwen4_exp_ops_config()
    try:
        yield
    finally:
        set_ops_config(previous)


def _set_deterministic(seed: int = 42) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    get_torch_device().manual_seed(seed)
    get_torch_device().manual_seed_all(seed)


def _use_full_float32_matmuls() -> None:
    """Keep sharded/full-sequence equivalence checks off the TF32 path.

    The two paths use different matmul shapes, so TF32 rounding can exceed the
    tight numerical tolerances even though the implementations are equivalent.
    This must run inside each spawned worker.
    """
    torch.set_float32_matmul_precision("highest")


def _assert_forward_deterministic(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    repeats: int = 10,
) -> None:
    with torch.no_grad():
        reference = layer(inputs, attention_mask=None, cu_seq_lens_q=None).detach()
        for _ in range(repeats - 1):
            output = layer(inputs, attention_mask=None, cu_seq_lens_q=None)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)


def _broadcast_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def _gather_sequence(local: torch.Tensor) -> torch.Tensor:
    local = local.contiguous()
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return torch.cat(gathered, dim=1)


def _sum_parameter_gradients(module: torch.nn.Module, divisor: float = 1.0) -> None:
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad = parameter.grad.contiguous()
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(divisor)


def _qwen4_causal_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run Qwen4's module-level channels-first causal convolution."""
    modeling = importlib.import_module(_PATCHED_MODULE)
    return modeling.causal_conv1d_fn(x, weight, bias=None, activation=None)


def _slice_mixed_qkv(
    mixed_qkv: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    sp_size: int,
    sp_rank: int,
) -> torch.Tensor:
    """Slice QKV channels from a Qwen4 ``[B, D, S]`` convolution input."""
    key_dim = num_k_heads * head_k_dim
    local_key_dim = (num_k_heads // sp_size) * head_k_dim
    local_value_dim = (num_v_heads // sp_size) * head_v_dim
    key_offset = sp_rank * local_key_dim
    value_offset = sp_rank * local_value_dim

    query = mixed_qkv[:, key_offset : key_offset + local_key_dim, :]
    key = mixed_qkv[:, key_dim + key_offset : key_dim + key_offset + local_key_dim, :]
    value = mixed_qkv[:, 2 * key_dim + value_offset : 2 * key_dim + value_offset + local_value_dim, :]
    return torch.cat((query, key, value), dim=1)


def _slice_conv1d_weight(
    weight: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    sp_size: int,
    sp_rank: int,
) -> torch.Tensor:
    """Slice depthwise-convolution channels with the same Q/K/V offsets."""
    key_dim = num_k_heads * head_k_dim
    local_key_dim = (num_k_heads // sp_size) * head_k_dim
    local_value_dim = (num_v_heads // sp_size) * head_v_dim
    key_offset = sp_rank * local_key_dim
    value_offset = sp_rank * local_value_dim

    query = weight[key_offset : key_offset + local_key_dim]
    key = weight[key_dim + key_offset : key_dim + key_offset + local_key_dim]
    value = weight[2 * key_dim + value_offset : 2 * key_dim + value_offset + local_value_dim]
    return torch.cat((query, key, value), dim=0)


def _run_ple_halo_equivalence(packed: bool = False, parallel_mode: str = "ulysses") -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    qwen4_exp = importlib.import_module(_PATCHED_MODULE)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        sp_state = _init_parallel_state(
            dp_size=1,
            cp_size=world_size if parallel_mode == "cp" else 1,
            ulysses_size=world_size if parallel_mode == "ulysses" else 1,
            device_type=get_device_type(),
            extra_parallel_names=("ple",),
            extra_parallel_sizes=(1,),
            extra_parallel_placement_innermost=(False,),
            name="sp",
        )
        dp_state = _init_parallel_state(
            dp_size=world_size,
            cp_size=1,
            ulysses_size=1,
            device_type=get_device_type(),
            extra_parallel_names=("ple",),
            extra_parallel_sizes=(1,),
            extra_parallel_placement_innermost=(False,),
            name="dp",
        )
        config = AutoConfig.from_pretrained(_TOY_CONFIG).text_config
        torch.manual_seed(17)
        device = torch.device(get_device_type(), rank)
        ple = qwen4_exp.Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0).float().to(device)
        with torch.no_grad():
            ple.conv1d.weight.normal_(mean=0.0, std=0.2)
        _broadcast_module(ple)

        local_seq_len = max(ple.short_conv_state_len, ple.ple_embedding.context_len) + 2
        seq_len = local_seq_len * world_size
        full_ids = (torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0) + 3) % config.vocab_size
        # Exercise the packed-sample reset immediately before an SP boundary.
        full_ids[:, local_seq_len - 1] = config.eos_token_id
        local_slice = slice(rank * local_seq_len, (rank + 1) * local_seq_len)
        local_ids = full_ids[:, local_slice].contiguous()

        boundaries = [0, local_seq_len - 1, seq_len] if packed else None
        cu_seq_lens = torch.tensor(boundaries, dtype=torch.int32, device=device) if boundaries is not None else None
        with use_parallel_state(dp_state):
            expected_embeddings = ple.ple_embedding(
                full_ids,
                past_key_values=None,
                cu_seq_lens_q=cu_seq_lens,
            ).detach()

        with use_parallel_state(sp_state):
            actual_embeddings = ple.ple_embedding(
                local_ids,
                past_key_values=None,
                cu_seq_lens_q=cu_seq_lens,
            ).detach()
        torch.testing.assert_close(_gather_sequence(actual_embeddings), expected_embeddings)

        torch.manual_seed(29)
        full_hidden = torch.randn(
            1,
            seq_len,
            config.hidden_size * config.hc_count,
            device=device,
            requires_grad=True,
        )
        with use_parallel_state(dp_state):
            expected_output = ple._short_conv(
                full_hidden,
                past_key_values=None,
                cu_seq_lens_q=cu_seq_lens,
            )
            expected_output.sum().backward()
        expected_input_grad = full_hidden.grad.detach().clone()
        expected_weight_grad = ple.conv1d.weight.grad.detach().clone()
        ple.conv1d.weight.grad = None

        local_hidden = full_hidden.detach()[:, local_slice].contiguous().requires_grad_(True)
        with use_parallel_state(sp_state):
            actual_output = ple._short_conv(
                local_hidden,
                past_key_values=None,
                cu_seq_lens_q=cu_seq_lens,
            )
            actual_output.sum().backward()
        dist.all_reduce(ple.conv1d.weight.grad, op=dist.ReduceOp.SUM)

        torch.testing.assert_close(_gather_sequence(actual_output.detach()), expected_output.detach())
        torch.testing.assert_close(_gather_sequence(local_hidden.grad.detach()), expected_input_grad)
        torch.testing.assert_close(ple.conv1d.weight.grad, expected_weight_grad)
    finally:
        clear_parallel_state()


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_ple_token_and_hidden_halos_match_full_sequence():
    torchrun(_run_ple_halo_equivalence, world_size=_CI_ACCELERATOR_2.nproc)


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_4)
def test_qwen4_exp_ple_token_and_hidden_halos_match_full_sequence_sp4():
    torchrun(_run_ple_halo_equivalence, world_size=_CI_ACCELERATOR_4.nproc)


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_packed_ple_hidden_halos_match_full_sequence():
    torchrun(_run_ple_halo_equivalence, world_size=_CI_ACCELERATOR_2.nproc, packed=True)


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_ple_token_and_hidden_halos_match_full_sequence_cp():
    if get_device_type() == "npu":
        pytest.skip("Qwen4-Exp Context Parallelism is currently supported on GPU only.")
    torchrun(_run_ple_halo_equivalence, _CI_ACCELERATOR_2.nproc, False, "cp")


def _run_qsa_equivalence(
    num_attention_heads: int = 2,
    num_key_value_heads: int = 1,
    parallel_mode: str = "ulysses",
    qsa_implementation: str = "eager",
) -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    qwen4_exp = importlib.import_module(_PATCHED_MODULE)

    _use_full_float32_matmuls()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        sp_state = _init_parallel_state(
            dp_size=1,
            cp_size=2 if parallel_mode == "hybrid" else world_size if parallel_mode == "cp" else 1,
            ulysses_size=world_size // 2
            if parallel_mode == "hybrid"
            else world_size
            if parallel_mode == "ulysses"
            else 1,
            allow_hybrid_cp=parallel_mode == "hybrid",
            device_type=get_device_type(),
            name="sp",
        )
        dp_state = _init_parallel_state(
            dp_size=world_size,
            cp_size=1,
            ulysses_size=1,
            device_type=get_device_type(),
            name="dp",
        )
        _install_qwen4_exp_ops_config(qsa_attention_implementation=qsa_implementation)
        device = torch.device(get_device_type(), rank)
        dtype = torch.bfloat16 if qsa_implementation == "tilelang" else torch.float32
        config = AutoConfig.from_pretrained(_TOY_CONFIG).text_config
        config.num_attention_heads = num_attention_heads
        config.num_key_value_heads = num_key_value_heads
        if qsa_implementation == "tilelang":
            config.head_dim = 64
            config.indexer_head_dim = 32
        torch.manual_seed(31)
        attention = qwen4_exp.Qwen4ExpTextAttention(config, layer_idx=1).to(device=device, dtype=dtype)
        _broadcast_module(attention)

        local_seq_len = 4
        seq_len = local_seq_len * world_size
        if rank == 0:
            full_hidden = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=dtype)
        else:
            full_hidden = torch.empty(1, seq_len, config.hidden_size, device=device, dtype=dtype)
        dist.broadcast(full_hidden, src=0)
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        rotary_angles = torch.arange(seq_len, device=device).view(1, seq_len, 1) * 0.17
        rotary_angles = rotary_angles + torch.arange(rotary_dim, device=device).view(1, 1, rotary_dim) * 0.03
        position_embeddings = (
            rotary_angles.cos().to(dtype=dtype),
            rotary_angles.sin().to(dtype=dtype),
        )
        packed_boundary = local_seq_len - 1
        cu_seq_lens_q = torch.tensor([0, packed_boundary, seq_len], dtype=torch.int32, device=device)
        expected_output = None
        expected_input_grad = None
        expected_parameter_grads = None
        if rank == 0:
            baseline_input = full_hidden.detach().clone().requires_grad_(True)
            with use_parallel_state(dp_state):
                expected_output, _ = attention(
                    baseline_input,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    cu_seq_lens_q=cu_seq_lens_q,
                )
                (expected_output.sum() / expected_output.numel()).backward()
            expected_input_grad = baseline_input.grad.detach().clone()
            expected_parameter_grads = {
                name: parameter.grad.detach().clone() if parameter.grad is not None else None
                for name, parameter in attention.named_parameters()
            }
            expected_output = expected_output.detach()
            attention.zero_grad(set_to_none=True)

        dist.barrier()
        local_slice = slice(rank * local_seq_len, (rank + 1) * local_seq_len)
        local_hidden = full_hidden[:, local_slice].detach().contiguous().requires_grad_(True)
        local_position_embeddings = tuple(tensor[:, local_slice].contiguous() for tensor in position_embeddings)
        with use_parallel_state(sp_state):
            selected_indices = attention.indexer(
                local_hidden,
                local_position_embeddings,
                None,
                None,
                cu_seq_lens_q=cu_seq_lens_q,
            )
            assert selected_indices.shape[:2] == (1, local_seq_len)
            valid_indices = selected_indices >= 0
            query_positions = torch.arange(
                rank * local_seq_len,
                (rank + 1) * local_seq_len,
                device=device,
            ).view(1, local_seq_len, 1)
            assert torch.all(
                selected_indices[valid_indices] <= query_positions.expand_as(selected_indices)[valid_indices]
            )
            before_boundary = query_positions < packed_boundary
            assert torch.all(
                (selected_indices < packed_boundary)[valid_indices & before_boundary.expand_as(selected_indices)]
            )
            assert torch.all(
                (selected_indices >= packed_boundary)[valid_indices & ~before_boundary.expand_as(selected_indices)]
            )
            actual_output, _ = attention(
                local_hidden,
                position_embeddings=local_position_embeddings,
                attention_mask=None,
                cu_seq_lens_q=cu_seq_lens_q,
            )
            (actual_output.sum() / (seq_len * actual_output.shape[-1])).backward()
        _sum_parameter_gradients(attention)

        actual_output = _gather_sequence(actual_output.detach())
        actual_input_grad = _gather_sequence(local_hidden.grad.detach())
        rtol = 1e-2 if dtype == torch.bfloat16 else 2e-5
        atol = 1e-2 if dtype == torch.bfloat16 else 2e-5
        if rank == 0:
            torch.testing.assert_close(actual_output, expected_output, rtol=rtol, atol=atol)
            torch.testing.assert_close(actual_input_grad, expected_input_grad, rtol=rtol, atol=atol)
            for name, parameter in attention.named_parameters():
                expected_grad = expected_parameter_grads[name]
                if expected_grad is None:
                    assert parameter.grad is None, f"Unexpected gradient for {name}"
                else:
                    torch.testing.assert_close(
                        parameter.grad,
                        expected_grad,
                        rtol=rtol,
                        atol=atol,
                        msg=lambda msg, parameter_name=name: f"{msg}\nGradient mismatch for {parameter_name}",
                    )
    finally:
        _install_qwen4_exp_ops_config()
        clear_parallel_state()


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_compact_qsa_mqa_matches_full_sequence():
    torchrun(_run_qsa_equivalence, world_size=_CI_ACCELERATOR_2.nproc)


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_4)
def test_qwen4_exp_compact_qsa_mqa_matches_full_sequence_sp4():
    torchrun(_run_qsa_equivalence, _CI_ACCELERATOR_4.nproc, 4, 1)


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_4)
def test_qwen4_exp_compact_qsa_hybrid_u2cp2_matches_full_sequence():
    """Compare packed QSA outputs and all gradients under the unified four-rank group."""
    torchrun(_run_qsa_equivalence, _CI_ACCELERATOR_4.nproc, 4, 1, "hybrid")


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_compact_qsa_gqa_matches_full_sequence():
    torchrun(_run_qsa_equivalence, _CI_ACCELERATOR_2.nproc, 4, 2)


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_compact_qsa_cp_nondivisible_heads_matches_full_sequence():
    """Check packed CP-only QSA outputs and gradients on CUDA and Ascend."""
    torchrun(_run_qsa_equivalence, _CI_ACCELERATOR_2.nproc, 3, 1, "cp", "eager")


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_compact_qsa_cp_tilelang_matches_full_sequence():
    pytest.importorskip("tilelang")
    torchrun(_run_qsa_equivalence, _CI_ACCELERATOR_2.nproc, 4, 1, "cp", "tilelang")


@_CUDA_ONLY
@pytest.mark.parametrize("bsz", [1, 4])
@pytest.mark.parametrize("seq_len", [5, 256])
@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 4), (16, 32)])
@pytest.mark.parametrize("head_k_dim,head_v_dim", [(4, 4), (4, 2)])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_qwen4_exp_depthwise_conv1d_slicing_matches_full(
    bsz: int,
    seq_len: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    kernel_size: int,
) -> None:
    """Local channels and weights reproduce the matching full-conv channels."""
    Qwen4ExpTextGatedDeltaNet = importlib.import_module(_PATCHED_MODULE).Qwen4ExpTextGatedDeltaNet

    _install_qwen4_exp_ops_config()
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim
    device = get_device_type()
    mixed_qkv_full = torch.randn(bsz, conv_dim, seq_len, device=device)
    weight_full = torch.randn(conv_dim, kernel_size, device=device)

    config = AutoConfig.from_pretrained(_TOY_CONFIG).text_config
    config.hidden_size = conv_dim
    config.linear_num_value_heads = num_v_heads
    config.linear_num_key_heads = num_k_heads
    config.linear_key_head_dim = head_k_dim
    config.linear_value_head_dim = head_v_dim
    config.linear_conv_kernel_dim = kernel_size
    layer = Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).float().to(device)
    layer.conv1d.weight.data.copy_(weight_full.unsqueeze(1))

    output_full = _qwen4_causal_depthwise_conv1d(mixed_qkv_full, weight_full)
    sp_size = 2
    for sp_rank in range(sp_size):
        mixed_qkv_local = _slice_mixed_qkv(
            mixed_qkv_full,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            sp_size,
            sp_rank,
        )
        weight_local = _slice_conv1d_weight(
            layer.conv1d.weight.squeeze(1),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            sp_size,
            sp_rank,
        )

        output_local = _qwen4_causal_depthwise_conv1d(mixed_qkv_local, weight_local)
        output_expected = _slice_mixed_qkv(
            output_full,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            sp_size,
            sp_rank,
        )
        torch.testing.assert_close(output_local, output_expected, rtol=0, atol=0)


def _run_gated_deltanet_equivalence(
    packed: bool = False,
    num_k_heads: int | None = None,
    num_v_heads: int | None = None,
    parallel_mode: str = "ulysses",
) -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    qwen4_exp = importlib.import_module(_PATCHED_MODULE)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        sp_state = _init_parallel_state(
            dp_size=1,
            cp_size=2 if parallel_mode == "hybrid" else 1,
            ulysses_size=world_size // 2 if parallel_mode == "hybrid" else world_size,
            allow_hybrid_cp=parallel_mode == "hybrid",
            device_type=get_device_type(),
            name="sp",
        )
        dp_state = _init_parallel_state(
            dp_size=world_size,
            ulysses_size=1,
            device_type=get_device_type(),
            name="dp",
        )
        device = torch.device(get_device_type(), rank)
        config = AutoConfig.from_pretrained(_TOY_CONFIG).text_config
        if num_k_heads is not None:
            config.linear_num_key_heads = num_k_heads
        elif world_size == 4:
            config.linear_num_key_heads = 4
        if num_v_heads is not None:
            config.linear_num_value_heads = num_v_heads
        torch.manual_seed(43)
        _install_qwen4_exp_ops_config()
        # Ascend's native GDN kernel rejects FP32 inputs. Keep CUDA's existing
        # FP32 checks and exercise the actual BF16 training kernel on NPU.
        dtype = torch.bfloat16 if get_device_type() == "npu" else torch.float32
        layer = qwen4_exp.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).to(device=device, dtype=dtype)
        _broadcast_module(layer)

        local_seq_len = 4
        seq_len = local_seq_len * world_size
        boundaries = [0, local_seq_len - 1, seq_len] if packed else [0, seq_len]
        if rank == 0:
            full_hidden = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=dtype)
        else:
            full_hidden = torch.empty(1, seq_len, config.hidden_size, device=device, dtype=dtype)
        dist.broadcast(full_hidden, src=0)
        cu_seq_lens = torch.tensor(boundaries, dtype=torch.int32, device=device)

        expected_output = None
        expected_input_grad = None
        expected_parameter_grads = None
        if rank == 0:
            baseline_input = full_hidden.detach().clone().requires_grad_(True)
            with use_parallel_state(dp_state):
                expected_output = layer(
                    baseline_input,
                    cu_seq_lens_q=cu_seq_lens,
                )
                (expected_output.sum() / expected_output.numel()).backward()
            expected_input_grad = baseline_input.grad.detach().clone()
            expected_parameter_grads = {
                name: parameter.grad.detach().clone() if parameter.grad is not None else None
                for name, parameter in layer.named_parameters()
            }
            expected_output = expected_output.detach()
            layer.zero_grad(set_to_none=True)

        dist.barrier()
        local_slice = slice(rank * local_seq_len, (rank + 1) * local_seq_len)
        local_hidden = full_hidden[:, local_slice].detach().contiguous().requires_grad_(True)
        with use_parallel_state(sp_state):
            actual_output = layer(local_hidden, cu_seq_lens_q=cu_seq_lens)
            (actual_output.sum() / (seq_len * actual_output.shape[-1])).backward()
        _sum_parameter_gradients(layer)

        actual_output = _gather_sequence(actual_output.detach())
        actual_input_grad = _gather_sequence(local_hidden.grad.detach())
        if rank == 0:
            output_tolerance = dict(rtol=2e-2, atol=2e-5) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-5)
            grad_tolerance = dict(rtol=2e-2, atol=3e-5) if dtype == torch.bfloat16 else dict(rtol=3e-5, atol=3e-5)
            torch.testing.assert_close(actual_output, expected_output, **output_tolerance)
            torch.testing.assert_close(actual_input_grad, expected_input_grad, **grad_tolerance)
            for name, parameter in layer.named_parameters():
                expected_grad = expected_parameter_grads[name]
                assert expected_grad is not None and parameter.grad is not None, f"Missing gradient for {name}"
                torch.testing.assert_close(
                    parameter.grad,
                    expected_grad,
                    **grad_tolerance,
                    msg=lambda msg, parameter_name=name: f"{msg}\nGradient mismatch for {parameter_name}",
                )
    finally:
        clear_parallel_state()


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_gated_deltanet_gqa_matches_full_sequence():
    torchrun(_run_gated_deltanet_equivalence, world_size=_CI_ACCELERATOR_2.nproc, num_k_heads=2, num_v_heads=4)


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_4)
def test_qwen4_exp_gated_deltanet_matches_full_sequence_sp4():
    torchrun(_run_gated_deltanet_equivalence, world_size=_CI_ACCELERATOR_4.nproc)


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_4)
def test_qwen4_exp_gated_deltanet_hybrid_u2cp2_matches_full_sequence():
    """Compare packed GDN outputs and all gradients under the unified four-rank group."""
    torchrun(_run_gated_deltanet_equivalence, _CI_ACCELERATOR_4.nproc, True, 4, 8, "hybrid")


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_gated_deltanet_packed_varlen_matches_full_sequence():
    torchrun(
        _run_gated_deltanet_equivalence, world_size=_CI_ACCELERATOR_2.nproc, packed=True, num_k_heads=2, num_v_heads=4
    )


@_CUDA_ONLY
@pytest.mark.parametrize(("num_k_heads", "num_v_heads"), [(3, 4), (4, 3)])
def test_qwen4_exp_gated_deltanet_rejects_nondivisible_heads(num_k_heads, num_v_heads):
    qwen4_exp = importlib.import_module(_PATCHED_MODULE)

    config = AutoConfig.from_pretrained(_TOY_CONFIG).text_config
    config.linear_num_key_heads = num_k_heads
    config.linear_num_value_heads = num_v_heads
    layer = qwen4_exp.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).float()
    parallel_state = SimpleNamespace(ulysses_enabled=True, ulysses_group=None, ulysses_rank=0, ulysses_size=2)
    hidden_states = torch.randn(1, 4, config.hidden_size)

    with (
        patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=parallel_state),
        pytest.raises(ValueError, match="must divide Qwen4-Exp GatedDeltaNet key heads"),
    ):
        layer(hidden_states)


_QWEN4_EXP_DETERMINISM_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 8),
    (4, 8),
    (1, 2048),
    (4, 2048),
)


def _run_gated_deltanet_determinism() -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    qwen4_exp = importlib.import_module(_PATCHED_MODULE)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    shapes = list(_QWEN4_EXP_DETERMINISM_SHAPES)
    if os.environ.get("VEOMNI_REVERSE_SCENARIOS") == "1":
        shapes = list(reversed(shapes))
    try:
        _init_parallel_state(
            dp_size=1,
            ulysses_size=world_size,
            device_type=get_device_type(),
            name=None,
        )
        _install_qwen4_exp_ops_config()
        device = torch.device(get_device_type(), rank)
        config = AutoConfig.from_pretrained(_TOY_CONFIG).text_config
        for bsz, seq_len in shapes:
            _set_deterministic()
            layer = qwen4_exp.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).float().to(device)
            _broadcast_module(layer)

            if rank == 0:
                full_input = torch.randn(bsz, seq_len, config.hidden_size, device=device)
            else:
                full_input = torch.empty(bsz, seq_len, config.hidden_size, device=device)
            dist.broadcast(full_input, src=0)

            shard_len = seq_len // world_size
            local_input = full_input[:, rank * shard_len : (rank + 1) * shard_len].contiguous()
            _assert_forward_deterministic(layer, local_input)
            dist.barrier()
        dist.barrier()
    finally:
        clear_parallel_state()


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_gated_deltanet_forward_deterministic_sp() -> None:
    torchrun(_run_gated_deltanet_determinism, _CI_ACCELERATOR_2.nproc)


@_CUDA_ONLY
@pytest.mark.parametrize("bsz", [1, 4])
@pytest.mark.parametrize("seq_len", [8, 2048])
def test_qwen4_exp_gated_deltanet_forward_deterministic_no_sp(bsz: int, seq_len: int) -> None:
    qwen4_exp = importlib.import_module(_PATCHED_MODULE)

    _install_qwen4_exp_ops_config()
    _set_deterministic()
    device = torch.device(get_device_type())
    config = AutoConfig.from_pretrained(_TOY_CONFIG).text_config
    layer = qwen4_exp.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).float().to(device)
    inputs = torch.randn(bsz, seq_len, config.hidden_size, device=device)

    no_sp_state = SimpleNamespace(
        ulysses_enabled=False,
        ulysses_group=None,
        ulysses_rank=0,
        ulysses_size=1,
    )
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
        _assert_forward_deterministic(layer, inputs)


def _run_text_model_equivalence() -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    qwen4_exp = importlib.import_module(_PATCHED_MODULE)

    _use_full_float32_matmuls()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        sp_state = _init_parallel_state(
            dp_size=1,
            ulysses_size=world_size,
            device_type=get_device_type(),
            extra_parallel_names=("ple",),
            extra_parallel_sizes=(1,),
            extra_parallel_placement_innermost=(False,),
            name="sp",
        )
        dp_state = _init_parallel_state(
            dp_size=world_size,
            ulysses_size=1,
            device_type=get_device_type(),
            extra_parallel_names=("ple",),
            extra_parallel_sizes=(1,),
            extra_parallel_placement_innermost=(False,),
            name="dp",
        )
        device = torch.device(get_device_type(), rank)
        config = AutoConfig.from_pretrained(_TOY_CONFIG).text_config
        config._attn_implementation = "eager"
        _install_qwen4_exp_ops_config()
        torch.manual_seed(59)
        model = qwen4_exp.Qwen4ExpTextModel(config).float().to(device)
        _broadcast_module(model)

        local_seq_len = 4
        seq_len = local_seq_len * world_size
        full_input_ids = (torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0) + 5) % config.vocab_size
        packed_boundary = local_seq_len - 1
        full_input_ids[:, packed_boundary - 1] = config.eos_token_id
        full_position_ids = torch.cat(
            (
                torch.arange(packed_boundary, dtype=torch.long, device=device),
                torch.arange(seq_len - packed_boundary, dtype=torch.long, device=device),
            )
        ).unsqueeze(0)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
        cu_seq_lens = torch.tensor([0, packed_boundary, seq_len], dtype=torch.int32, device=device)

        expected_output = None
        expected_parameter_grads = None
        if rank == 0:
            with use_parallel_state(dp_state):
                expected_output = model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    position_ids=full_position_ids,
                    cu_seq_lens_q=cu_seq_lens,
                ).last_hidden_state
                (expected_output.sum() / expected_output.numel()).backward()
            expected_parameter_grads = {
                name: parameter.grad.detach().clone() if parameter.grad is not None else None
                for name, parameter in model.named_parameters()
            }
            expected_output = expected_output.detach()
            model.zero_grad(set_to_none=True)

        dist.barrier()
        local_slice = slice(rank * local_seq_len, (rank + 1) * local_seq_len)
        with use_parallel_state(sp_state):
            actual_output = model(
                input_ids=full_input_ids[:, local_slice].contiguous(),
                attention_mask=attention_mask,
                position_ids=full_position_ids[:, local_slice].contiguous(),
                cu_seq_lens_q=cu_seq_lens,
            ).last_hidden_state
            (actual_output.sum() / (seq_len * actual_output.shape[-1])).backward()
        _sum_parameter_gradients(model)
        actual_output = _gather_sequence(actual_output.detach())

        if rank == 0:
            torch.testing.assert_close(actual_output, expected_output, rtol=5e-5, atol=5e-5)
            for name, parameter in model.named_parameters():
                expected_grad = expected_parameter_grads[name]
                if expected_grad is None:
                    assert parameter.grad is None, f"Unexpected gradient for {name}"
                else:
                    torch.testing.assert_close(
                        parameter.grad,
                        expected_grad,
                        rtol=8e-5,
                        atol=8e-5,
                        msg=lambda msg, parameter_name=name: f"{msg}\nGradient mismatch for {parameter_name}",
                    )
    finally:
        clear_parallel_state()


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_packed_text_model_matches_full_sequence():
    torchrun(_run_text_model_equivalence, world_size=_CI_ACCELERATOR_2.nproc)


def _run_router_aux_loss_equivalence() -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    qwen4_exp = importlib.import_module(_PATCHED_MODULE)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        sp_state = _init_parallel_state(
            dp_size=1,
            ulysses_size=world_size,
            device_type=get_device_type(),
            extra_parallel_names=("ple",),
            extra_parallel_sizes=(1,),
            extra_parallel_placement_innermost=(False,),
            name="sp",
        )
        dp_state = _init_parallel_state(
            dp_size=world_size,
            ulysses_size=1,
            device_type=get_device_type(),
            extra_parallel_names=("ple",),
            extra_parallel_sizes=(1,),
            extra_parallel_placement_innermost=(False,),
            name="dp",
        )
        device = torch.device(get_device_type(), rank)
        config = AutoConfig.from_pretrained(_TOY_CONFIG)
        config.text_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "veomni_flash_attention_2"
        _install_qwen4_exp_ops_config()
        torch.manual_seed(67)
        model = qwen4_exp.Qwen4ExpForConditionalGeneration(config).float().to(device)
        model.model.visual.to(dtype=torch.bfloat16)
        _broadcast_module(model)

        local_seq_len = 4
        seq_len = local_seq_len * world_size
        full_input_ids = (torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0) + 11) % 100
        full_position_ids = torch.arange(seq_len, dtype=torch.long, device=device).view(1, 1, -1).expand(1, 3, -1)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
        empty_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        expected_aux_loss = None
        expected_parameter_grads = None
        if rank == 0:
            with use_parallel_state(dp_state):
                expected = model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    position_ids=full_position_ids,
                    image_mask=empty_mask,
                    video_mask=empty_mask,
                    qwen4_exp_position_ids_layout="batch_first",
                    cu_seq_lens_q=cu_seq_lens,
                    output_router_logits=True,
                    logits_to_keep=1,
                )
                local_aux_losses = []
                for shard_rank in range(world_size):
                    shard_slice = slice(shard_rank * local_seq_len, (shard_rank + 1) * local_seq_len)
                    local_router_logits = tuple(
                        layer_logits.reshape(1, seq_len, -1)[:, shard_slice].flatten(0, 1)
                        for layer_logits in expected.router_logits
                    )
                    local_aux_losses.append(
                        qwen4_exp.load_balancing_loss_func(
                            local_router_logits,
                            config.text_config.num_experts,
                            config.text_config.num_experts_per_tok,
                            attention_mask[:, shard_slice],
                        )
                    )
                expected_aux_loss = torch.stack(local_aux_losses).mean()
                expected_aux_loss.backward()
            expected_parameter_grads = {
                name: parameter.grad.detach().clone() if parameter.grad is not None else None
                for name, parameter in model.named_parameters()
            }
            model.zero_grad(set_to_none=True)

        dist.barrier()
        local_slice = slice(rank * local_seq_len, (rank + 1) * local_seq_len)
        with use_parallel_state(sp_state):
            actual = model(
                input_ids=full_input_ids[:, local_slice].contiguous(),
                attention_mask=attention_mask,
                position_ids=full_position_ids[..., local_slice].contiguous(),
                image_mask=empty_mask,
                video_mask=empty_mask,
                qwen4_exp_position_ids_layout="batch_first",
                cu_seq_lens_q=cu_seq_lens,
                output_router_logits=True,
                logits_to_keep=1,
            )
            actual.aux_loss.backward()
        actual_aux_loss = actual.aux_loss.detach().clone()
        dist.all_reduce(actual_aux_loss, op=dist.ReduceOp.SUM)
        actual_aux_loss.div_(world_size)
        # FSDP2/DDP averages replicated parameter gradients over the complete
        # dp_sp group; mirror that reduction after the test's plain modules.
        _sum_parameter_gradients(model, divisor=world_size)

        if rank == 0:
            torch.testing.assert_close(actual_aux_loss, expected_aux_loss, rtol=1e-5, atol=1e-5)
            for name, parameter in model.named_parameters():
                expected_grad = expected_parameter_grads[name]
                if expected_grad is None:
                    assert parameter.grad is None, f"Unexpected gradient for {name}"
                else:
                    torch.testing.assert_close(
                        parameter.grad,
                        expected_grad,
                        rtol=1e-4,
                        atol=1e-4,
                        msg=lambda msg, parameter_name=name: f"{msg}\nGradient mismatch for {parameter_name}",
                    )
    finally:
        clear_parallel_state()


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_router_aux_loss_matches_mean_of_local_shards():
    torchrun(_run_router_aux_loss_equivalence, world_size=_CI_ACCELERATOR_2.nproc)


def _run_vlm_placeholder_equivalence(parallel_mode: str = "ulysses") -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    qwen4_exp = importlib.import_module(_PATCHED_MODULE)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        sp_state = _init_parallel_state(
            dp_size=1,
            cp_size=world_size if parallel_mode == "cp" else 1,
            ulysses_size=world_size if parallel_mode == "ulysses" else 1,
            device_type=get_device_type(),
            extra_parallel_names=("ple",),
            extra_parallel_sizes=(1,),
            extra_parallel_placement_innermost=(False,),
            name="sp",
        )
        dp_state = _init_parallel_state(
            dp_size=world_size,
            cp_size=1,
            ulysses_size=1,
            device_type=get_device_type(),
            extra_parallel_names=("ple",),
            extra_parallel_sizes=(1,),
            extra_parallel_placement_innermost=(False,),
            name="dp",
        )
        device = torch.device(get_device_type(), rank)
        config = AutoConfig.from_pretrained(_TOY_CONFIG)
        config.text_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "veomni_flash_attention_2"
        _install_qwen4_exp_ops_config()
        torch.manual_seed(71)
        model = qwen4_exp.Qwen4ExpModel(config).to(device=device, dtype=torch.bfloat16)
        _broadcast_module(model)
        patch_embed_rows = []
        model.visual.patch_embed.register_forward_pre_hook(
            lambda _module, args: patch_embed_rows.append(args[0].shape[0])
        )

        local_seq_len = 4
        seq_len = local_seq_len * world_size
        full_input_ids = (torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0) + 7) % 100
        image_mask = torch.zeros_like(full_input_ids, dtype=torch.bool)
        image_mask[:, local_seq_len - 1] = True
        image_mask[:, local_seq_len] = True
        full_input_ids.masked_fill_(image_mask, 0)
        video_mask = torch.zeros_like(image_mask)
        full_position_ids = torch.arange(seq_len, dtype=torch.long, device=device).view(1, 1, -1).expand(1, 3, -1)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
        flattened_patch_size = (
            config.vision_config.in_channels
            * config.vision_config.temporal_patch_size
            * config.vision_config.patch_size**2
        )
        if rank == 0:
            full_pixels = torch.randn(8, flattened_patch_size, device=device, dtype=torch.bfloat16)
        else:
            full_pixels = torch.empty(8, flattened_patch_size, device=device, dtype=torch.bfloat16)
        dist.broadcast(full_pixels, src=0)
        image_grid_thw = torch.tensor([[1, 2, 4]], dtype=torch.long, device=device)
        cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        expected_output = None
        expected_pixel_grad = None
        expected_parameter_grads = None
        if rank == 0:
            baseline_pixels = full_pixels.detach().clone().requires_grad_(True)
            with use_parallel_state(dp_state):
                expected_output = model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    position_ids=full_position_ids,
                    pixel_values=baseline_pixels,
                    image_grid_thw=image_grid_thw,
                    image_mask=image_mask,
                    video_mask=video_mask,
                    qwen4_exp_position_ids_layout="batch_first",
                    cu_seq_lens_q=cu_seq_lens,
                ).last_hidden_state
                (expected_output.sum() / expected_output.numel()).backward()
            expected_pixel_grad = baseline_pixels.grad.detach().clone()
            expected_parameter_grads = {
                name: parameter.grad.detach().clone() if parameter.grad is not None else None
                for name, parameter in model.named_parameters()
            }
            expected_output = expected_output.detach()
            model.zero_grad(set_to_none=True)

        dist.barrier()
        patch_embed_rows.clear()
        local_slice = slice(rank * local_seq_len, (rank + 1) * local_seq_len)
        pixel_rows_per_rank = full_pixels.shape[0] // world_size
        pixel_slice = slice(rank * pixel_rows_per_rank, (rank + 1) * pixel_rows_per_rank)
        local_pixels = full_pixels[pixel_slice].detach().contiguous().requires_grad_(True)
        with use_parallel_state(sp_state):
            actual_output = model(
                input_ids=full_input_ids[:, local_slice].contiguous(),
                attention_mask=attention_mask,
                position_ids=full_position_ids[..., local_slice].contiguous(),
                pixel_values=local_pixels,
                image_grid_thw=image_grid_thw,
                image_mask=image_mask,
                video_mask=video_mask,
                qwen4_exp_position_ids_layout="batch_first",
                multimodal_metadata={
                    "image_grid_thw_list": [[1, 2, 4]],
                    "vit_image_cu_seqlens": torch.tensor([0, 8], dtype=torch.int32, device=device),
                    "vit_image_max_seqlen": 8,
                },
                cu_seq_lens_q=cu_seq_lens,
            ).last_hidden_state
            assert patch_embed_rows[0] == local_pixels.shape[0]
            (actual_output.sum() / (seq_len * actual_output.shape[-1])).backward()
        _sum_parameter_gradients(model)
        actual_output = _gather_sequence(actual_output.detach())
        actual_pixel_grad = _gather_sequence(local_pixels.grad.detach().unsqueeze(0)).squeeze(0)

        if rank == 0:
            torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(actual_pixel_grad, expected_pixel_grad, rtol=2e-2, atol=2e-2)
            for name, parameter in model.named_parameters():
                expected_grad = expected_parameter_grads[name]
                if expected_grad is None:
                    assert parameter.grad is None, f"Unexpected gradient for {name}"
                else:
                    torch.testing.assert_close(
                        parameter.grad,
                        expected_grad,
                        rtol=2e-2,
                        atol=2e-2,
                        msg=lambda msg, parameter_name=name: f"{msg}\nGradient mismatch for {parameter_name}",
                    )
    finally:
        clear_parallel_state()


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_vlm_placeholder_crossing_shards_matches_full_sequence():
    torchrun(_run_vlm_placeholder_equivalence, world_size=_CI_ACCELERATOR_2.nproc)


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_vlm_placeholder_crossing_shards_matches_full_sequence_cp():
    torchrun(_run_vlm_placeholder_equivalence, _CI_ACCELERATOR_2.nproc, "cp")


def _run_vision_real_image_cp_non_fa_rejection() -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    qwen4_exp = importlib.import_module(_PATCHED_MODULE)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        cp_state = _init_parallel_state(
            dp_size=1,
            cp_size=world_size,
            ulysses_size=1,
            device_type=get_device_type(),
            name="cp",
        )
        device = torch.device(get_device_type(), rank)
        config = AutoConfig.from_pretrained(_TOY_CONFIG).vision_config
        config._attn_implementation = "veomni_sdpa"
        _install_qwen4_exp_ops_config()
        vision = qwen4_exp.Qwen4ExpVisionModel(config).to(device=device, dtype=torch.bfloat16)
        _broadcast_module(vision)

        flattened_patch_size = config.in_channels * config.temporal_patch_size * config.patch_size**2
        local_pixels = torch.randn(8 // world_size, flattened_patch_size, device=device, dtype=torch.bfloat16)
        grid_thw = torch.tensor([[1, 2, 4]], dtype=torch.long, device=device)
        vit_metadata = {
            "grid_thw_list": [[1, 2, 4]],
            "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32, device=device),
            "max_seqlen": 8,
        }
        with (
            use_parallel_state(cp_state),
            pytest.raises(NotImplementedError, match="Qwen4-Exp vision sequence parallelism requires FlashAttention"),
        ):
            vision(local_pixels, grid_thw=grid_thw, vit_metadata=vit_metadata, return_dict=True)
    finally:
        clear_parallel_state()


@_CUDA_ONLY
@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_qwen4_exp_vision_real_image_cp_rejects_non_flash_attention():
    if get_device_type() == "npu":
        pytest.skip("Qwen4-Exp Context Parallelism is currently supported on GPU only.")
    torchrun(_run_vision_real_image_cp_non_fa_rejection, world_size=_CI_ACCELERATOR_2.nproc)


@_CUDA_ONLY
@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.ones(1, 8, dtype=torch.bool),
        torch.ones(1, 1, 6, 8, dtype=torch.bool),
    ],
)
def test_qwen4_exp_text_attention_rejects_invalid_sp_attention_mask_before_collective(
    bad_mask: torch.Tensor,
) -> None:
    qwen4_exp = importlib.import_module(_PATCHED_MODULE)
    config = AutoConfig.from_pretrained(_TOY_CONFIG).text_config
    config._attn_implementation = "eager"
    _install_qwen4_exp_ops_config()
    attn = qwen4_exp.Qwen4ExpTextAttention(config, layer_idx=1).float()
    hidden_states = torch.randn(1, 4, config.hidden_size)
    cos = torch.ones(1, 4, attn.head_dim)
    sin = torch.zeros(1, 4, attn.head_dim)
    cp_state = SimpleNamespace(
        cp_enabled=True,
        cp_group=object(),
        cp_rank=0,
        cp_size=2,
        ulysses_enabled=False,
    )

    with (
        patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=cp_state),
        patch(
            f"{_PATCHED_MODULE}.all_gather_compressed_rows",
            side_effect=AssertionError("all_gather_compressed_rows must not run"),
        ),
        patch(f"{_PATCHED_MODULE}.gather_outputs", side_effect=AssertionError("gather_outputs must not run")),
        patch(f"{_PATCHED_MODULE}.all_gather_kv", side_effect=AssertionError("all_gather_kv must not run")),
        pytest.raises(ValueError, match="Qwen4-Exp sequence parallelism expects a 3D/4D attention_mask"),
    ):
        attn(hidden_states, (cos, sin), attention_mask=bad_mask)


@_CUDA_ONLY
def test_qwen4_exp_vision_attention_projects_cp_state_to_ulysses_parallel_state() -> None:
    from veomni.distributed import parallel_state as parallel_state_module
    from veomni.distributed.parallel_state import (
        ParallelState,
        clear_parallel_state,
        get_parallel_state,
        use_parallel_state,
    )

    qwen4_exp = importlib.import_module(_PATCHED_MODULE)
    config = AutoConfig.from_pretrained(_TOY_CONFIG).vision_config
    config._attn_implementation = "veomni_flash_attention_2"
    _install_qwen4_exp_ops_config()
    attn = qwen4_exp.Qwen4ExpVisionAttention(config).float()
    hidden_states = torch.randn(4, config.hidden_size)
    head_dim = config.hidden_size // config.num_heads
    cos = torch.ones(4, head_dim)
    sin = torch.zeros(4, head_dim)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)

    class _FakeMesh:
        def __init__(self, mesh_dim_names, group, local_rank):
            self.mesh_dim_names = mesh_dim_names
            self._group = group
            self._local_rank = local_rank

        def __getitem__(self, _item):
            return SimpleNamespace(_flatten=lambda mesh_dim_name=None: None)

        def get_group(self, dim_name):
            return self._group if dim_name in self.mesh_dim_names or dim_name == "sp" else None

        def get_local_rank(self, dim_name):
            return self._local_rank if dim_name in self.mesh_dim_names or dim_name == "sp" else 0

    cp2_group = object()
    cp4_group = object()
    ulysses2_group = object()
    ulysses4_group = object()
    cp2_mesh = _FakeMesh(("dp_shard", "cp"), cp2_group, 1)
    cp4_mesh = _FakeMesh(("dp_shard", "cp"), cp4_group, 3)
    mesh_calls = []

    def fake_init_device_mesh(*, device_type, mesh_shape, mesh_dim_names):
        mesh_calls.append((device_type, mesh_shape, mesh_dim_names))
        if mesh_shape == (2, 2) and mesh_dim_names == ("dp_shard", "ulysses"):
            return _FakeMesh(mesh_dim_names, ulysses2_group, 1)
        if mesh_shape == (1, 4) and mesh_dim_names == ("dp_shard", "ulysses"):
            return _FakeMesh(mesh_dim_names, ulysses4_group, 3)
        raise AssertionError(f"Unexpected mesh request: {mesh_shape}, {mesh_dim_names}")

    captured = []

    def fake_veomni_attn(_module, query_states, *_args, **_kwargs):
        active_state = get_parallel_state()
        captured.append(
            {
                "is_parallel_state": isinstance(active_state, ParallelState),
                "dp_size": active_state.dp_size,
                "dp_shard_size": active_state.dp_shard_size,
                "ulysses_enabled": active_state.ulysses_enabled,
                "ulysses_group": active_state.ulysses_group,
                "ulysses_rank": active_state.ulysses_rank,
                "ulysses_size": active_state.ulysses_size,
                "cp_enabled": active_state.cp_enabled,
                "cp_group": active_state.cp_group,
                "cp_rank": active_state.cp_rank,
                "cp_size": active_state.cp_size,
                "fsdp_enabled": active_state.fsdp_enabled,
            }
        )
        return query_states, None

    attn.veomni_attn = fake_veomni_attn
    clear_parallel_state()
    try:
        with (
            patch.object(ParallelState, "world_size", 4),
            patch.object(parallel_state_module, "init_device_mesh", side_effect=fake_init_device_mesh),
        ):
            cp2_state = ParallelState(dp_size=2, dp_shard_size=2, cp_size=2, ulysses_size=1, device_mesh=cp2_mesh)
            cp4_state = ParallelState(dp_size=1, dp_shard_size=1, cp_size=4, ulysses_size=1, device_mesh=cp4_mesh)
            with use_parallel_state(cp2_state):
                output_cp2 = attn(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=(cos, sin),
                    vision_max_seqlen=8,
                )
                # Second call with the same topology hits the topology cache.
                attn(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=(cos, sin),
                    vision_max_seqlen=8,
                )
                assert get_parallel_state() is cp2_state

            with use_parallel_state(cp4_state):
                output_cp4 = attn(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=(cos, sin),
                    vision_max_seqlen=8,
                )
                assert get_parallel_state() is cp4_state
    finally:
        clear_parallel_state()

    assert output_cp2.shape == hidden_states.shape
    assert output_cp4.shape == hidden_states.shape
    assert len(mesh_calls) == 2
    assert (
        captured[0]
        == captured[1]
        == {
            "is_parallel_state": True,
            "dp_size": 2,
            "dp_shard_size": 2,
            "ulysses_enabled": True,
            "ulysses_group": ulysses2_group,
            "ulysses_rank": 1,
            "ulysses_size": 2,
            "cp_enabled": False,
            "cp_group": None,
            "cp_rank": -1,
            "cp_size": 1,
            "fsdp_enabled": True,
        }
    )
    assert captured[2] == {
        "is_parallel_state": True,
        "dp_size": 1,
        "dp_shard_size": 1,
        "ulysses_enabled": True,
        "ulysses_group": ulysses4_group,
        "ulysses_rank": 3,
        "ulysses_size": 4,
        "cp_enabled": False,
        "cp_group": None,
        "cp_rank": -1,
        "cp_size": 1,
        "fsdp_enabled": True,
    }
