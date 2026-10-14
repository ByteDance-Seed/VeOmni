import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE


@pytest.fixture(params=["qwen3_5", "qwen3_5_moe"])
def modeling(request):
    return importlib.import_module(
        f"veomni.models.transformers.{request.param}.generated.patched_modeling_{request.param}_gpu"
    )


def _make_layer(modeling, monkeypatch, implementation, group_size=2):
    prefix = "Qwen3_5Moe" if "qwen3_5_moe" in modeling.__name__ else "Qwen3_5"
    monkeypatch.setattr(modeling.veomni_gdn_implementation, "_value", implementation)
    monkeypatch.setattr(modeling.veomni_rms_norm_gated, "_kernel", None)
    monkeypatch.setattr(modeling.veomni_causal_conv1d, "_kernel", Mock(side_effect=lambda **kwargs: (kwargs["x"],)))
    kernel = Mock(side_effect=lambda query, key, value, **kwargs: (value, None))
    monkeypatch.setattr(modeling.veomni_chunk_gated_delta_rule, "_kernel", kernel)
    config = SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2 * group_size,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    with torch.device("meta"):
        layer = getattr(modeling, prefix + "GatedDeltaNet")(config, layer_idx=0)
    return layer, kernel


@pytest.mark.parametrize("implementation", ["flash_qla", "fla", "eager", "npu"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("group_size", [1, 2, 4])
def test_qk_head_layout_is_selected_without_numerical_execution(
    modeling, monkeypatch, implementation, packed, group_size
):
    layer, kernel = _make_layer(modeling, monkeypatch, implementation, group_size)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ulysses_enabled=False))
    hidden = torch.empty(1, 6, 16, device="meta")
    boundaries = torch.empty(3, dtype=torch.int32, device="meta") if packed else None

    output = layer(hidden, cu_seq_lens_q=boundaries)

    query, key, value = kernel.call_args.args
    expected_heads = 2 if implementation == "flash_qla" and packed else 2 * group_size
    assert query.shape == key.shape == (1, 6, expected_heads, 4)
    assert value.shape == (1, 6, 2 * group_size, 4)
    assert output.shape == hidden.shape
    assert kernel.call_args.kwargs["cu_seqlens"] is boundaries
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()


@pytest.mark.parametrize("first,second", [("flash_qla", "fla"), ("fla", "flash_qla")])
def test_backend_and_qk_layout_remain_instance_local(modeling, monkeypatch, first, second):
    original, original_kernel = _make_layer(modeling, monkeypatch, first)
    other, other_kernel = _make_layer(modeling, monkeypatch, second)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ulysses_enabled=False))
    hidden = torch.empty(1, 6, 16, device="meta")
    boundaries = torch.empty(3, dtype=torch.int32, device="meta")

    original(hidden, cu_seq_lens_q=boundaries)
    other(hidden, cu_seq_lens_q=boundaries)

    assert original.chunk_gated_delta_rule is original_kernel
    assert other.chunk_gated_delta_rule is other_kernel
    assert original_kernel.call_args.args[0].shape[2] == (2 if first == "flash_qla" else 4)
    assert other_kernel.call_args.args[0].shape[2] == (2 if second == "flash_qla" else 4)


def test_grouped_heads_keep_ulysses_local_head_layout(modeling, monkeypatch):
    layer, kernel = _make_layer(modeling, monkeypatch, "flash_qla")
    state = SimpleNamespace(ulysses_enabled=True, ulysses_size=2, ulysses_rank=1, ulysses_group=object())
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: state)

    def scatter_heads(tensor, **kwargs):
        return torch.empty(
            tensor.shape[0], tensor.shape[1] * 2, tensor.shape[2] // 2, *tensor.shape[3:], device="meta"
        )

    def scatter_sequence(tensor, **kwargs):
        return torch.empty(tensor.shape[0], tensor.shape[1] // 2, tensor.shape[2] * 2, tensor.shape[3], device="meta")

    monkeypatch.setattr(modeling, "gather_seq_scatter_heads", scatter_heads)
    monkeypatch.setattr(modeling, "gather_heads_scatter_seq", scatter_sequence)
    hidden = torch.empty(1, 3, 16, device="meta")
    boundaries = torch.empty(3, dtype=torch.int32, device="meta")

    output = layer(hidden, cu_seq_lens_q=boundaries)

    query, key, value = kernel.call_args.args
    assert query.shape == key.shape == (1, 6, 1, 4)
    assert value.shape == (1, 6, 2, 4)
    assert kernel.call_args.kwargs["g"].shape == (1, 6, 2)
    assert output.shape == hidden.shape


def test_cached_decode_remains_unsupported(modeling, monkeypatch):
    layer, kernel = _make_layer(modeling, monkeypatch, "flash_qla")
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ulysses_enabled=False))
    cache = SimpleNamespace(has_previous_state=True, conv_states=[None], recurrent_states=[None])
    with pytest.raises(NotImplementedError, match="use_precomputed_states=True"):
        layer(
            torch.empty(1, 1, 16, device="meta"),
            cache_params=cache,
            cache_position=torch.empty(1, dtype=torch.int64, device="meta"),
        )
    kernel.assert_not_called()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Generated qwen4_exp modeling imports CUDA-only MoE kernels.")
@pytest.mark.parametrize("implementation", ["flash_qla", "fla", "eager", "npu"])
@pytest.mark.parametrize("group_size", [1, 2, 4])
@pytest.mark.parametrize("bound_kernel", [False, True])
def test_qwen4_exp_grouped_heads_preserve_fallbacks(monkeypatch, implementation, group_size, bound_kernel):
    modeling = importlib.import_module("veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu")
    monkeypatch.setattr(modeling.veomni_gdn_implementation, "_value", implementation)
    kernel = Mock(side_effect=lambda query, key, value, **kwargs: (value, None))
    monkeypatch.setattr(modeling.veomni_chunk_gated_delta_rule, "_kernel", kernel if bound_kernel else None)
    monkeypatch.setattr(modeling, "torch_chunk_gated_delta_rule", kernel)
    monkeypatch.setattr(modeling.veomni_causal_conv1d, "_kernel", Mock(side_effect=lambda **kwargs: (kwargs["x"],)))
    config = SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2 * group_size,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        output_gate_type=None,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    boundaries = Mock()
    boundaries.diff.return_value.tolist.return_value = [2, 4]
    with torch.device("meta"):
        layer = modeling.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0)
    output = layer(torch.empty(1, 6, 16, device="meta"), linear_attn_cu_seq_lens_q=boundaries)

    query, key, value = kernel.call_args.args
    expected_heads = 2 if implementation == "flash_qla" and bound_kernel else 2 * group_size
    expected_sequence = 6 if bound_kernel else 4
    assert query.shape == key.shape == (1, expected_sequence, expected_heads, 4)
    assert value.shape == (1, expected_sequence, 2 * group_size, 4)
    assert output.shape == (1, 6, 16)
    with pytest.raises(ValueError, match="requires linear_attn_cu_seq_lens_q"):
        layer(torch.empty(1, 6, 16, device="meta"))
    with pytest.raises(ValueError, match="does not support recurrent cache"):
        layer(torch.empty(1, 6, 16, device="meta"), linear_attn_cu_seq_lens_q=boundaries, cache_params=object())
