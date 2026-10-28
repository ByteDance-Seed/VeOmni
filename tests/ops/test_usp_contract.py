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

"""USP must preserve existing CP layouts and reject unsupported objectives."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import AcceleratorConfig
from veomni.models import auto
from veomni.ops.kernels.attention.standard import flash
from veomni.trainer.text_dpo_trainer import TextDPOTrainer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_usp_layout_is_explicit(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(NotImplementedError, match="cannot be combined"):
        AcceleratorConfig(cp_size=2, ulysses_size=2)
    config = AcceleratorConfig(cp_size=2, ulysses_size=2, cp_layout="zigzag")
    assert config.dp_size == 1


@pytest.mark.parametrize(
    "model_type,layout,backend,allowed",
    [
        ("deepseek_v4", "contiguous", "eager", True),
        ("deepseek_v4", "zigzag", "veomni_flash_attention_2", False),
        ("qwen3", "contiguous", "veomni_flash_attention_2", False),
        ("qwen3", "zigzag", "veomni_flash_attention_2", True),
        ("qwen3", "zigzag", "veomni_flash_attention_4", True),
        ("qwen3", "zigzag", "flash_attention_2", False),
        ("qwen3", "zigzag", "flash_attention_4", False),
        ("qwen3", "zigzag", "veomni_flex_attention", False),
        ("qwen3_next", "zigzag", "veomni_flash_attention_2", False),
    ],
)
def test_cp_model_and_backend_gate(monkeypatch, model_type, layout, backend, allowed):
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(
        auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout=layout, device_type="cuda")
    )
    config = SimpleNamespace(model_type=model_type, _attn_implementation=backend)
    if allowed:
        auto.check_context_parallel_supported(config)
    else:
        with pytest.raises(NotImplementedError):
            auto.check_context_parallel_supported(config)


@pytest.mark.parametrize(
    "modifier",
    [
        {"dropout": 0.1},
        {"deterministic": True},
        {"sliding_window": 4},
        {"softcap": 1.0},
        {"s_aux": torch.ones(2)},
    ],
)
def test_usp_rejects_attention_modifiers_before_collectives(monkeypatch, modifier):
    monkeypatch.setattr(
        flash, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag", ulysses_enabled=True)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported attention entered an Ulysses collective")

    monkeypatch.setattr(flash, "prepare_ulysses_qkv", forbidden)
    module = SimpleNamespace(config=SimpleNamespace(_attn_implementation="veomni_flash_attention_2"), is_causal=True)
    q = torch.zeros(1, 2, 8, 4, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="not support"):
        flash.flash_attention_forward(module, q, q, q, None, **modifier)


def test_dpo_rejects_zigzag_before_setup():
    args = SimpleNamespace(model=SimpleNamespace(accelerator=SimpleNamespace(cp_size=2, cp_layout="zigzag")))
    with pytest.raises(NotImplementedError, match="DPO"):
        TextDPOTrainer(args)


@pytest.mark.parametrize("extra_outputs", [(), (None, None)])
def test_fa4_forward_accepts_auxiliary_outputs(monkeypatch, extra_outputs):
    from veomni.distributed.sequence_parallel.ring_attention import gpu as ring

    q = torch.zeros(1, 8, 2, 4)
    lse = torch.zeros(1, 2, 8)
    monkeypatch.setattr(ring, "FA_BACKEND", "fa4")
    monkeypatch.setattr(ring, "_fa4_fwd", lambda *args, **kwargs: (q, lse, *extra_outputs), raising=False)
    out, actual_lse = ring._fa_forward(q, q, q, 0.5, True)
    assert out is q and actual_lse is lse
    cu = torch.tensor([0, 8], dtype=torch.int32)
    out, actual_lse = ring._fa_varlen_forward(q, q, q, cu, cu, 8, 8, 0.5, True)
    assert out is q and actual_lse is lse


def test_rl_rejects_zigzag_for_composed_trainers(monkeypatch):
    from veomni.trainer import base_rl_trainer

    monkeypatch.setattr(
        base_rl_trainer, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag")
    )
    trainer = base_rl_trainer.BaseRLTrainer.__new__(base_rl_trainer.BaseRLTrainer)
    with pytest.raises(NotImplementedError, match="RL postprocessing"):
        trainer._build_preforward_postforward()


def test_usp_rejects_upstream_modeling_backend(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag"))
    with pytest.raises(NotImplementedError, match="MODELING_BACKEND"):
        auto.check_context_parallel_supported(SimpleNamespace(model_type="qwen3"), "flash_attention_2")


@pytest.mark.parametrize("packed", [False, True])
def test_usp_rejects_deterministic_environment_before_collectives(monkeypatch, packed):
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    monkeypatch.setattr(flash, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag"))

    def forbidden(*args, **kwargs):
        pytest.fail("deterministic USP entered an Ulysses collective")

    monkeypatch.setattr(flash, "prepare_ulysses_qkv", forbidden)
    module = SimpleNamespace(config=SimpleNamespace(_attn_implementation="veomni_flash_attention_2"), is_causal=True)
    q = torch.zeros(1, 2, 8, 4, dtype=torch.bfloat16)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 8, 16], dtype=torch.int32)} if packed else {}
    with pytest.raises(NotImplementedError, match="deterministic"):
        flash.flash_attention_forward(module, q, q, q, None, **kwargs)


def test_usp_rejects_sequence_classification(monkeypatch):
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag"))
    config = SimpleNamespace(model_type="qwen3", architectures=["Qwen3ForSequenceClassification"])
    with pytest.raises(NotImplementedError, match="Qwen3ForCausalLM"):
        auto.check_context_parallel_supported(config, "veomni_flash_attention_2")


@pytest.mark.parametrize("layout", [{"cp_layout": "contiguous"}, {}])
def test_contiguous_cp_keeps_standard_flash_path(monkeypatch, layout):
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    monkeypatch.setattr(flash, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, **layout))
    monkeypatch.setattr(flash, "should_apply_ulysses", lambda **kwargs: False)
    calls = []

    def standard(query, key, value, mask, **kwargs):
        calls.append(kwargs["attn_implementation"])
        return query

    monkeypatch.setattr(flash, "_flash_attention_forward", standard)
    module = SimpleNamespace(config=SimpleNamespace(_attn_implementation="veomni_flash_attention_2"), is_causal=True)
    q = torch.zeros(1, 2, 8, 4, dtype=torch.bfloat16)
    out, _ = flash.flash_attention_forward(module, q, q, q, None)
    torch.testing.assert_close(out, q.transpose(1, 2))
    assert calls == ["flash_attention_2"]


@pytest.mark.parametrize("backend", ["flash_attention_2", "flash_attention_4"])
def test_installed_ops_cannot_bypass_usp_with_plain_attention(monkeypatch, backend):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(
        auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag", device_type="cuda")
    )
    monkeypatch.setattr(
        auto, "get_ops_config", lambda: SimpleNamespace(attn_implementation="veomni_flash_attention_2")
    )
    config = auto.PretrainedConfig()
    config.model_type = "qwen3"
    with pytest.raises(NotImplementedError, match="VeOmni FlashAttention"):
        auto.build_foundation_model(config, attn_implementation=backend, init_device="meta")


@pytest.mark.parametrize("kind", ["plain", "zigzag", "varlen"])
def test_ring_pins_backend_through_interleaved_backward(monkeypatch, kind):
    from veomni.distributed.sequence_parallel.ring_attention import gpu as ring

    monkeypatch.setattr(ring, "FA_BACKENDS", frozenset({"fa2", "fa4"}))
    monkeypatch.setattr(ring, "FA_BACKEND", "fa2")
    calls = []

    def forward(group, q, k, v, *args, backend):
        calls.append(("forward", backend))
        return q + k + v, torch.zeros(1)

    def backward(group, dout, q, k, v, *args, backend):
        calls.append(("backward", backend))
        return dout, dout, dout

    prefix = {"plain": "_ring", "zigzag": "_zigzag_ring", "varlen": "_zigzag_ring_varlen"}[kind]
    monkeypatch.setattr(ring, prefix + "_forward", forward)
    monkeypatch.setattr(ring, prefix + "_backward", backward)
    func = {
        "plain": ring.ring_flash_attn_func,
        "zigzag": ring.zigzag_ring_flash_attn_func,
        "varlen": ring.zigzag_ring_flash_attn_varlen_func,
    }[kind]
    shape = (8, 2, 4) if kind == "varlen" else (1, 8, 2, 4)
    inputs = [torch.ones(shape, requires_grad=True) for _ in range(3)]
    args = (torch.tensor([0, 8], dtype=torch.int32), 8) if kind == "varlen" else ()
    first = func(*inputs, *args, group=object(), backend="fa4")
    second = func(*inputs, *args, group=object(), backend="fa2")
    monkeypatch.setattr(ring, "FA_BACKEND", "fa4")
    second.sum().backward()
    first.sum().backward()
    assert calls == [("forward", "fa4"), ("forward", "fa2"), ("backward", "fa2"), ("backward", "fa4")]
    for tensor in inputs:
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 2.0))


@pytest.mark.parametrize("packed", [False, True])
def test_flash_selects_explicit_fa4_when_both_backends_exist(monkeypatch, packed):
    from veomni.distributed.sequence_parallel.ring_attention import gpu as ring

    monkeypatch.setattr(ring, "FA_BACKENDS", frozenset({"fa2", "fa4"}))
    monkeypatch.setattr(ring, "FA_BACKEND", "fa2")
    monkeypatch.setattr(
        flash,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag", cp_group=object(), cp_size=2, device_type="cuda"),
    )
    monkeypatch.setattr(flash, "should_apply_ulysses", lambda **kwargs: False)
    calls = []

    def selected(query, *args, backend, **kwargs):
        calls.append(backend)
        return query

    monkeypatch.setattr(ring, "zigzag_ring_flash_attn_func", selected)
    monkeypatch.setattr(ring, "zigzag_ring_flash_attn_varlen_func", selected)
    module = SimpleNamespace(config=SimpleNamespace(_attn_implementation="veomni_flash_attention_2"), is_causal=True)
    q = torch.zeros(1, 2, 8, 4, dtype=torch.bfloat16)
    kwargs = {}
    if packed:
        from veomni.distributed.sequence_parallel.data import prepare_usp_metadata

        cu = torch.tensor([0, 8, 16], dtype=torch.int32)
        kwargs.update(prepare_usp_metadata(cu, cp_size=2), cu_seq_lens_q=cu)
    flash.flash_attention_forward(module, q, q, q, None, implementation="veomni_flash_attention_4", **kwargs)
    assert calls == ["fa4"]


@pytest.mark.parametrize("primitive", ["forward", "backward"])
@pytest.mark.parametrize("missing", ["window_size_left", "window_size_right", "cu_seqlens_k", "max_seqlen_k"])
def test_fa4_probe_rejects_missing_wrapper_arguments(primitive, missing):
    import inspect

    from veomni.distributed.sequence_parallel.ring_attention import gpu as ring

    shared = {
        "softmax_scale",
        "causal",
        "window_size_left",
        "window_size_right",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
    }

    def forward():
        pass

    def backward():
        pass

    parameters = {"forward": shared | {"return_lse"}, "backward": shared | {"dq", "dk", "dv"}}

    def set_signatures():
        for name, function in (("forward", forward), ("backward", backward)):
            function.__signature__ = inspect.Signature(
                inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY) for key in sorted(parameters[name])
            )

    set_signatures()
    assert ring._supports_fa4_interface(forward, backward)
    parameters[primitive].remove(missing)
    set_signatures()
    assert not ring._supports_fa4_interface(forward, backward)


@pytest.mark.parametrize("packed", [False, True])
def test_cuda_backend_rejects_dropout(packed):
    from veomni.distributed.sequence_parallel.ring_attention import gpu

    q = torch.zeros(1, 8, 2, 4)
    with pytest.raises(NotImplementedError, match="dropout"):
        if packed:
            gpu.packed_forward(q, q, q, torch.tensor([0, 8], dtype=torch.int32), 8, dropout_p=0.1)
        else:
            gpu.forward(q, q, q, dropout_p=0.1)


def test_npu_usp_model_gate(monkeypatch):
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(
        auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag", device_type="npu")
    )
    auto.check_context_parallel_supported(SimpleNamespace(model_type="qwen3"), "veomni_flash_attention_2")
    with pytest.raises(NotImplementedError, match="Ascend USP"):
        auto.check_context_parallel_supported(SimpleNamespace(model_type="qwen3"), "veomni_flash_attention_4")


@pytest.mark.parametrize("packed", [False, True])
def test_npu_adapter_uses_shared_dispatch_without_cuda_backend(monkeypatch, packed):
    from veomni.distributed.sequence_parallel import ring_attention as ring
    from veomni.distributed.sequence_parallel.ring_attention import gpu

    monkeypatch.setattr(gpu, "FA_BACKENDS", frozenset())
    monkeypatch.setattr(
        flash,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, cp_layout="zigzag", cp_group=object(), cp_size=2, device_type="npu"),
    )
    monkeypatch.setattr(flash, "should_apply_ulysses", lambda **kwargs: False)
    calls = []

    def selected(query, *args, **kwargs):
        calls.append(kwargs)
        return query

    monkeypatch.setattr(ring, "ring_attention", selected)
    module = SimpleNamespace(config=SimpleNamespace(_attn_implementation="veomni_flash_attention_2"), is_causal=True)
    q = torch.zeros(1, 2, 8, 4, dtype=torch.bfloat16)
    kwargs = {}
    if packed:
        from veomni.distributed.sequence_parallel.data import prepare_usp_metadata

        cu = torch.tensor([0, 8, 16], dtype=torch.int32)
        kwargs.update(prepare_usp_metadata(cu, cp_size=2), cu_seq_lens_q=cu)
    flash.flash_attention_forward(module, q, q, q, None, **kwargs)
    assert calls[0]["device_type"] == "npu"
    assert calls[0]["backend_name"] is None
    assert (calls[0]["usp_metadata"] is not None) == packed


@pytest.mark.parametrize("layout", ["contiguous", "zigzag"])
@pytest.mark.parametrize("entry", ["arguments", "state"])
def test_async_cp_is_rejected_before_mesh_initialization(monkeypatch, layout, entry):
    from veomni.distributed.parallel_state import ParallelState

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(NotImplementedError, match="async"):
        if entry == "arguments":
            AcceleratorConfig(cp_size=2, cp_layout=layout, enable_async=True)
        else:
            ParallelState(cp_size=2, cp_layout=layout, async_enabled=True)


def test_async_without_cp_remains_supported(monkeypatch):
    from veomni.distributed.parallel_state import ParallelState

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert AcceleratorConfig(enable_async=True).enable_async
    assert ParallelState(async_enabled=True).async_enabled
