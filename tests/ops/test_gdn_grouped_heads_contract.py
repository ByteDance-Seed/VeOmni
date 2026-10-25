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

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import (
    Qwen3_5GatedDeltaNet,
)
from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import (
    get_parallel_state as qwen3_5_get_parallel_state,
)
from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
    Qwen3_5MoeGatedDeltaNet,
)
from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
    get_parallel_state as qwen3_5_moe_get_parallel_state,
)
from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu import (
    Qwen4ExpTextGatedDeltaNet,
)
from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu import (
    get_parallel_state as qwen4_exp_get_parallel_state,
)
from veomni.ops.config import get_ops_config, set_ops_config


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _PassthroughConv1d:
    def __init__(self, impl: str):
        self.impl = impl

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x


class _RecordingGdnOp:
    def __init__(self, impl: str, calls: list[dict[str, object]]):
        self.impl = impl
        self._calls = calls

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        *,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        self._calls.append(
            {
                "q_shape": tuple(query.shape),
                "k_shape": tuple(key.shape),
                "v_shape": tuple(value.shape),
                "g_shape": tuple(g.shape),
                "beta_shape": tuple(beta.shape),
                "cu_seqlens": cu_seqlens,
                "output_final_state": output_final_state,
                "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
            }
        )
        state = None
        if output_final_state:
            num_seqs = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else query.shape[0]
            state = query.new_zeros((num_seqs, value.shape[2], query.shape[-1], value.shape[-1]))
        return value + (query.sum() + key.sum() + g.sum() + beta.sum()) * 0, state


def _make_config(model_name: str):
    common = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    if model_name == "qwen3_5":
        return Qwen3_5TextConfig(**common)
    if model_name == "qwen3_5_moe":
        return Qwen3_5MoeTextConfig(**common)
    if model_name == "qwen4_exp":
        return Qwen4ExpTextConfig(**common, moe_intermediate_size=64, num_experts=2, num_experts_per_tok=1)
    raise ValueError(f"Unsupported model_name={model_name!r}")


def _build_layer(model_name: str, impl: str, calls: list[dict[str, object]]):
    config = _make_config(model_name)
    prev_cfg = get_ops_config()
    set_ops_config(None)
    try:
        if model_name == "qwen3_5":
            layer = Qwen3_5GatedDeltaNet(config, layer_idx=0)
        elif model_name == "qwen3_5_moe":
            layer = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)
        elif model_name == "qwen4_exp":
            layer = Qwen4ExpTextGatedDeltaNet(config, layer_idx=0)
        else:
            raise ValueError(f"Unsupported model_name={model_name!r}")
    finally:
        set_ops_config(prev_cfg)

    layer.veomni_causal_conv1d = _PassthroughConv1d(impl=impl)
    layer.veomni_chunk_gated_delta_rule = _RecordingGdnOp(impl=impl, calls=calls)
    return layer


@pytest.mark.parametrize("model_name", ["qwen3_5", "qwen3_5_moe", "qwen4_exp"])
@pytest.mark.parametrize("impl", ["flash_qla", "fla", "eager"])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("ulysses", [False, True])
def test_gdn_grouped_heads_dispatch_contract(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    impl: str,
    packed: bool,
    ulysses: bool,
):
    if ulysses and not packed:
        pytest.skip("Ulysses sequence parallelism uses packed cu_seq_lens_q")
    if ulysses and model_name == "qwen4_exp" and impl == "eager":
        pytest.skip("Qwen4-Exp Ulysses/CP requires non-eager GatedDeltaNet kernels")

    calls: list[dict[str, object]] = []
    layer = _build_layer(model_name, impl, calls)

    if ulysses:
        fake_ps = SimpleNamespace(
            ulysses_enabled=True,
            ulysses_size=2,
            ulysses_rank=0,
            ulysses_group=object(),
            cp_enabled=False,
            cp_size=1,
            cp_rank=0,
            cp_group=None,
            sp_enabled=True,
            sp_size=2,
            sp_rank=0,
            sp_group=object(),
        )

        def scatter_heads(x, seq_dim, head_dim, group=None):
            return torch.chunk(x, 2, dim=head_dim)[0].repeat(1, 2, *([1] * (x.ndim - 2)))

        def scatter_seq(x, head_dim, seq_dim, group=None):
            return torch.chunk(x, 2, dim=seq_dim)[0].repeat(1, 1, 2, 1)

        if model_name == "qwen3_5":
            monkeypatch.setattr(
                "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu.get_parallel_state",
                lambda: fake_ps,
            )
            monkeypatch.setattr(
                "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu.gather_seq_scatter_heads",
                scatter_heads,
            )
            monkeypatch.setattr(
                "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu.gather_heads_scatter_seq",
                scatter_seq,
            )
        elif model_name == "qwen3_5_moe":
            monkeypatch.setattr(
                "veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu.get_parallel_state",
                lambda: fake_ps,
            )
            monkeypatch.setattr(
                "veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu.gather_seq_scatter_heads",
                scatter_heads,
            )
            monkeypatch.setattr(
                "veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu.gather_heads_scatter_seq",
                scatter_seq,
            )
        elif model_name == "qwen4_exp":
            monkeypatch.setattr(
                "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu.get_parallel_state",
                lambda: fake_ps,
            )
            monkeypatch.setattr(
                "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu.gather_seq_scatter_heads",
                scatter_heads,
            )
            monkeypatch.setattr(
                "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu.gather_heads_scatter_seq",
                scatter_seq,
            )
    else:
        assert not qwen3_5_get_parallel_state().ulysses_enabled
        assert not qwen3_5_moe_get_parallel_state().ulysses_enabled
        assert not qwen4_exp_get_parallel_state().sp_enabled

    hidden_states = torch.randn(1, 3 if ulysses else 6, 64, requires_grad=True)
    cu_seq_lens_q = torch.tensor([0, 3, 6], dtype=torch.int32) if packed else None
    out = layer(hidden_states, cu_seq_lens_q=cu_seq_lens_q)
    out.sum().backward()

    expected_calls = 2 if (model_name == "qwen4_exp" and packed and impl == "eager" and not ulysses) else 1
    assert len(calls) == expected_calls
    local_k_heads = 2 if ulysses else 4
    local_v_heads = 8 if ulysses else 16
    expected_qk_heads = local_k_heads if impl == "flash_qla" else local_v_heads

    for call in calls:
        expected_seq_len = 3 if expected_calls == 2 else 6
        assert call["q_shape"] == (1, expected_seq_len, expected_qk_heads, 8)
        assert call["k_shape"] == (1, expected_seq_len, expected_qk_heads, 8)
        assert call["v_shape"] == (1, expected_seq_len, local_v_heads, 8)
        assert call["g_shape"] == (1, expected_seq_len, local_v_heads)
        assert call["beta_shape"] == (1, expected_seq_len, local_v_heads)


def test_gdn_precomputed_states_always_expand_qk_heads(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []
    layer = _build_layer("qwen4_exp", "flash_qla", calls)

    recurrent_calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def fake_recurrent(query, key, value, g, beta, initial_state, output_final_state, **kwargs):
        recurrent_calls.append((tuple(query.shape), tuple(key.shape), tuple(value.shape)))
        return value, initial_state

    monkeypatch.setattr(
        "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu.causal_conv1d_update",
        lambda x, conv_state, weight, bias=None, activation=None: x,
    )
    monkeypatch.setattr(
        "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu.torch_recurrent_gated_delta_rule",
        fake_recurrent,
    )

    cache_params = SimpleNamespace(
        has_previous_state=lambda layer_idx, state_idx=0: True,
        update_recurrent_state=lambda state, layer_idx: None,
        layers=[
            SimpleNamespace(
                record_past=False,
                conv_states=[torch.zeros(1, 192, 4)],
                recurrent_states=[torch.zeros(1, 16, 8, 8)],
            )
        ],
    )
    hidden_states = torch.randn(1, 1, 64)
    _ = layer(
        hidden_states,
        cache_params=cache_params,
        cache_position=torch.tensor([1]),
        cu_seq_lens_q=None,
    )
    assert len(calls) == 0
    assert recurrent_calls == [((1, 1, 16, 8), (1, 1, 16, 8), (1, 1, 16, 8))]
