import copy
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig

from veomni.arguments.arguments_types import (
    AcceleratorConfig,
    DataArguments,
    ModelArguments,
    OffloadConfig,
    OpsImplementationConfig,
    TrainingArguments,
    VeOmniArguments,
)
from veomni.arguments.arguments_types import (
    TorchCompileConfig as ArgumentsTorchCompileConfig,
)
from veomni.distributed.parallel_state import ParallelState, use_parallel_state
from veomni.distributed.torch_compile import (
    CompileConfig,
    compile_decoder_blocks,
    mark_compile_step_begin,
    validate_compile_config_for_fsdp2,
    validate_compile_runtime,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def _run_merged_swiglu_compiled_first_case(entrypoint, dtype_name, swiglu_limit):
    """Exercise a cold fused activation in a fresh interpreter, before any eager call."""
    from veomni.ops import resolve_op
    from veomni.ops.kernels.moe_experts.shared import swiglu
    from veomni.ops.kernels.moe_experts.shared.ep import EPMergedFc1GroupGemm
    from veomni.ops.kernels.moe_experts.standard.triton import MergedFc1TritonFusedMoeExpertFunction

    assert "veomni.ops.kernels.moe_experts.shared._triton_swiglu" not in sys.modules
    torch.manual_seed(31)
    dtype = getattr(torch, dtype_name)
    tokens, experts, hidden, intermediate, top_k = 23, 4, 64, 128, 2

    if entrypoint == "helper":
        merged = torch.randn(tokens, 2 * intermediate, device="cuda", dtype=dtype)
        gate, up = merged.chunk(2, dim=-1)
        gate, up, _, _ = swiglu.apply_swiglu_clamp(gate, up, swiglu_limit)
        gate_before, up_before = gate.clone(), up.clone()
        compiled = torch.compile(swiglu.merged_swiglu_forward, backend="inductor", fullgraph=True)
        actual = compiled(gate, up)
        expected = torch.ops.aten.silu(gate) * up
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(gate, gate_before, rtol=0, atol=0)
        torch.testing.assert_close(up, up_before, rtol=0, atol=0)
    else:
        x = 0.4 * torch.randn(tokens, hidden, device="cuda", dtype=dtype)
        w1 = 0.4 * torch.randn(experts, 2 * intermediate, hidden, device="cuda", dtype=dtype)
        w2 = 0.1 * torch.randn(experts, hidden, intermediate, device="cuda", dtype=dtype)
        if entrypoint == "ep":
            cumsum = torch.tensor([6, 12, 18, tokens], device="cuda", dtype=torch.int32)
            values = (x, w1, w2)

            def forward(hidden_states, fc1, fc2):
                return EPMergedFc1GroupGemm.apply(hidden_states, cumsum, fc1, fc2, swiglu_limit)

        else:
            logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
            routing, selected = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
            values = (x, routing.to(dtype), w1, w2)
            if entrypoint == "registry":
                wrapper = resolve_op("moe_experts", "standard", "fused_triton").wrapper
                empty = torch.empty(0, device="cuda", dtype=dtype)

                def forward(hidden_states, weights, fc1, fc2):
                    return wrapper(
                        hidden_states,
                        weights,
                        selected,
                        empty,
                        empty,
                        fc2,
                        fc1,
                        num_experts=experts,
                        swiglu_limit=swiglu_limit,
                    )

            else:

                def forward(hidden_states, weights, fc1, fc2):
                    return MergedFc1TritonFusedMoeExpertFunction.apply(
                        experts, weights, selected, hidden_states, fc1, fc2, swiglu_limit
                    )

        inputs = [value.detach().clone().requires_grad_() for value in values]
        reference_inputs = [value.detach().clone() for value in values]
        state = SimpleNamespace(ep_enabled=False, async_enabled=False)
        with use_parallel_state(state):
            compiled = torch.compile(forward, backend="inductor", fullgraph=True)
            actual = compiled(*inputs)
            grad_output = torch.randn_like(actual)
            actual_grads = torch.autograd.grad(actual, inputs, grad_output)

            # Run the reference only after compiled forward and backward finish.
            original = swiglu.merged_swiglu_forward
            swiglu.merged_swiglu_forward = lambda gate, up: torch.ops.aten.silu(gate) * up
            try:
                expected = forward(*reference_inputs)
            finally:
                swiglu.merged_swiglu_forward = original
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
        assert torch.isfinite(actual).all()
        # This regression covers cold initialization and backward execution.
        # test_deepseek_v4_inductor_fixed_route_expert_vjp_matches_fp32_reference
        # supplies the compiled numerical gradient oracle.
        for grad, value in zip(actual_grads, inputs, strict=True):
            assert grad.shape == value.shape and grad.dtype == value.dtype and grad.device == value.device
            assert torch.isfinite(grad).all()
            assert torch.count_nonzero(grad) > 0

    assert "veomni.ops.kernels.moe_experts.shared._triton_swiglu" in sys.modules


@pytest.mark.parametrize("entrypoint", ["helper", "non_ep", "ep", "registry"])
@pytest.mark.parametrize("dtype_name,swiglu_limit", [("bfloat16", None), ("float16", 1.0)])
def test_merged_swiglu_compiled_first_in_fresh_process(tmp_path, entrypoint, dtype_name, swiglu_limit):
    """A first compiled call must work without an earlier eager kernel initialization."""
    from tests.ops.utils import require_nvidia_cuda

    require_nvidia_cuda("triton")
    script = (
        "import sys; "
        "from tests.distributed.test_torch_compile import _run_merged_swiglu_compiled_first_case; "
        "_run_merged_swiglu_compiled_first_case(sys.argv[1], sys.argv[2], "
        "None if sys.argv[3] == 'None' else float(sys.argv[3]))"
    )
    env = os.environ.copy()
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "inductor")
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton")
    subprocess.run(
        [sys.executable, "-c", script, entrypoint, dtype_name, str(swiglu_limit)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=True,
        timeout=300,
    )


def _non_sp_parallel_state() -> ParallelState:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return ParallelState(dp_size=world_size, dp_shard_size=world_size)


def _model_args(**kwargs) -> ModelArguments:
    return ModelArguments(
        config_path="dummy_config.json",
        ops_implementation=OpsImplementationConfig(load_balancing_loss_implementation="eager"),
        **kwargs,
    )


class ToyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


class ToyVisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


class ToyModel(nn.Module):
    _no_split_modules = ["ToyDecoderLayer"]

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([ToyDecoderLayer(), ToyDecoderLayer()])
        self.vision = ToyVisionBlock()
        self.lm_head = nn.Linear(4, 8)


class ToyQwen3VLModel(nn.Module):
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "ToyVisionBlock"]
    input_modalities = ("image", "text")

    def __init__(self, decoder_layer):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_vl", vision_config=SimpleNamespace())
        self.layer = decoder_layer
        self.vision = ToyVisionBlock()


def test_compile_decoder_blocks_compiles_only_decoder_layers(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)

        def wrapped(*args, **inner_kwargs):
            return fn(*args, **inner_kwargs)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)

    model = ToyModel()
    compiled = compile_decoder_blocks(
        model,
        CompileConfig(backend="inductor", mode="reduce-overhead", fullgraph=True, dynamic=False),
    )

    assert compiled == 2
    for layer in model.layers:
        assert layer._veomni_forward_compiled is True
        assert layer._veomni_original_forward is ToyDecoderLayer.forward
    assert not getattr(model.vision, "_veomni_forward_compiled", False)
    assert not getattr(model.lm_head, "_veomni_forward_compiled", False)
    assert not getattr(model.embed_tokens, "_veomni_forward_compiled", False)
    assert calls == [{"fullgraph": True, "dynamic": False, "backend": "inductor", "mode": "reduce-overhead"}] * 2
    assert model.layers[0](torch.ones(2, 4)).shape == (2, 4)


def test_compile_decoder_blocks_uses_no_split_modules(monkeypatch):
    calls = []

    class ToyOtherDecoderLayer(ToyDecoderLayer):
        pass

    class MixedDecoderModel(nn.Module):
        _no_split_modules = ["ToyDecoderLayer"]

        def __init__(self):
            super().__init__()
            self.selected = ToyDecoderLayer()
            self.unselected = ToyOtherDecoderLayer()

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    model = MixedDecoderModel()
    compiled = compile_decoder_blocks(model, CompileConfig())

    assert compiled == 1
    assert getattr(model.selected, "_veomni_forward_compiled", False)
    assert not getattr(model.unselected, "_veomni_forward_compiled", False)
    assert calls == [{"fullgraph": True, "dynamic": False, "backend": "inductor"}]


def test_compile_decoder_blocks_rejects_unvalidated_multimodal_model(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    model = ToyModel()
    model.config = SimpleNamespace(model_type="qwen2_vl", vision_config=SimpleNamespace())
    model.input_modalities = ("image", "text")

    with pytest.raises(RuntimeError, match="only for dense Qwen3-VL"):
        compile_decoder_blocks(model, CompileConfig())


def test_compile_decoder_blocks_rejects_qwen3_vl_dynamic_shapes(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    with pytest.raises(RuntimeError, match="model.accelerator.torch_compile.dynamic=False"):
        compile_decoder_blocks(
            ToyQwen3VLModel(ToyDecoderLayer()),
            CompileConfig(dynamic=True),
        )


@pytest.mark.parametrize(
    "compile_config",
    [CompileConfig(backend="inductor", mode="reduce-overhead"), CompileConfig(backend="cudagraphs")],
    ids=["reduce-overhead", "cudagraphs-backend"],
)
def test_compile_decoder_blocks_rejects_qwen3_vl_cuda_graphs(monkeypatch, compile_config):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    with pytest.raises(RuntimeError, match="does not support CUDA Graph replay"):
        compile_decoder_blocks(ToyQwen3VLModel(ToyDecoderLayer()), compile_config)


@pytest.mark.parametrize(
    ("sequence_parallel_enabled", "async_enabled"),
    [(True, False), (False, True)],
    ids=["sequence-or-context-parallel", "async-ulysses"],
)
def test_compile_decoder_blocks_rejects_qwen3_vl_parallel_attention_paths(
    monkeypatch,
    sequence_parallel_enabled,
    async_enabled,
):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    with pytest.raises(RuntimeError, match="ulysses_size=1.*cp_size=1.*enable_async=False"):
        compile_decoder_blocks(
            ToyQwen3VLModel(ToyDecoderLayer()),
            CompileConfig(),
            sequence_parallel_enabled=sequence_parallel_enabled,
            async_enabled=async_enabled,
        )


def test_compile_decoder_blocks_targets_qwen3_vl_text_layers_only(monkeypatch):
    from veomni.models import build_foundation_model

    from ..tools.training_utils import make_eager_ops_config

    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)
    model = build_foundation_model(
        config_path="tests/toy_config/qwen3vl_toy/config.json",
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=make_eager_ops_config(),
    )

    compiled = compile_decoder_blocks(model, CompileConfig())

    assert compiled == len(model.model.language_model.layers) == 2
    assert all(layer._veomni_forward_compiled for layer in model.model.language_model.layers)
    assert all(not getattr(block, "_veomni_forward_compiled", False) for block in model.model.visual.blocks)


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_qwen3_vl_decoder_traces_under_fullgraph(use_checkpoint):
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
        Qwen3VLTextConfig,
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextRotaryEmbedding,
    )

    torch.manual_seed(0)
    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000,
    )
    config._attn_implementation = "eager"

    eager_layer = Qwen3VLTextDecoderLayer(config, layer_idx=0)
    compiled_layer = Qwen3VLTextDecoderLayer(config, layer_idx=0)
    compiled_layer.load_state_dict(eager_layer.state_dict())
    if use_checkpoint:
        for layer in (eager_layer, compiled_layer):
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

    compiled_model = ToyQwen3VLModel(compiled_layer)
    hidden_states_eager = torch.randn(1, 7, config.hidden_size, requires_grad=True)
    hidden_states_compiled = hidden_states_eager.detach().clone().requires_grad_(True)
    position_ids = torch.arange(7, dtype=torch.long).view(1, 7)
    rotary_emb = Qwen3VLTextRotaryEmbedding(config)
    position_embeddings = rotary_emb(hidden_states_eager, position_ids)

    # An explicit state avoids the logging fallback in get_parallel_state(), which Dynamo cannot trace fullgraph.
    with use_parallel_state(_non_sp_parallel_state()):
        assert (
            compile_decoder_blocks(
                compiled_model,
                CompileConfig(enable=True, backend="eager", fullgraph=True, dynamic=False),
            )
            == 1
        )
        eager_output = eager_layer(hidden_states_eager, position_embeddings=position_embeddings)
        compiled_output = compiled_layer(hidden_states_compiled, position_embeddings=position_embeddings)
        eager_output.square().mean().backward()
        compiled_output.square().mean().backward()

    torch.testing.assert_close(compiled_output, eager_output)
    torch.testing.assert_close(hidden_states_compiled.grad, hidden_states_eager.grad)
    for eager_param, compiled_param in zip(eager_layer.parameters(), compiled_layer.parameters()):
        torch.testing.assert_close(compiled_param.grad, eager_param.grad)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires CUDA")
def test_qwen3_vl_compiled_decoder_matches_eager_packed_flash_attention():
    pytest.importorskip("flash_attn")

    from veomni.data.data_collator import add_flash_attention_kwargs_from_position_ids
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
        Qwen3VLTextConfig,
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextRotaryEmbedding,
    )

    torch.manual_seed(0)
    device = torch.device(get_device_type())
    dtype = torch.float16
    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000,
    )
    config._attn_implementation = "flash_attention_2"

    eager_layer = Qwen3VLTextDecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype)
    compiled_layer = Qwen3VLTextDecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype)
    compiled_layer.load_state_dict(eager_layer.state_dict())
    compiled_model = ToyQwen3VLModel(compiled_layer)
    hidden_states_eager = torch.randn(1, 7, config.hidden_size, device=device, dtype=dtype, requires_grad=True)
    hidden_states_compiled = hidden_states_eager.detach().clone().requires_grad_(True)
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], device=device, dtype=torch.long)
    rotary_emb = Qwen3VLTextRotaryEmbedding(config).to(device)
    position_embeddings = rotary_emb(hidden_states_eager, position_ids)
    batch = {"position_ids": position_ids}
    add_flash_attention_kwargs_from_position_ids(batch)
    flash_attention_kwargs = {
        key: batch[key] for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
    }

    # An explicit state avoids the logging fallback in get_parallel_state(), which Dynamo cannot trace fullgraph.
    with use_parallel_state(_non_sp_parallel_state()):
        assert (
            compile_decoder_blocks(
                compiled_model,
                CompileConfig(enable=True, backend="inductor", fullgraph=True, dynamic=False),
            )
            == 1
        )
        eager_output = eager_layer(
            hidden_states_eager,
            position_embeddings=position_embeddings,
            **flash_attention_kwargs,
        )
        compiled_output = compiled_layer(
            hidden_states_compiled,
            position_embeddings=position_embeddings,
            **flash_attention_kwargs,
        )
        eager_output.float().square().mean().backward()
        compiled_output.float().square().mean().backward()

    torch.testing.assert_close(compiled_output, eager_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(hidden_states_compiled.grad, hidden_states_eager.grad, rtol=2e-2, atol=2e-2)
    for eager_param, compiled_param in zip(eager_layer.parameters(), compiled_layer.parameters()):
        torch.testing.assert_close(compiled_param.grad, eager_param.grad, rtol=2e-2, atol=2e-2)


def test_compile_decoder_blocks_rejects_mode_with_cudagraphs_backend():
    with pytest.raises(ValueError, match="'cudagraphs' backend"):
        compile_decoder_blocks(
            ToyModel(),
            CompileConfig(backend="cudagraphs", mode="reduce-overhead"),
        )


def test_compile_decoder_blocks_accepts_cudagraphs_backend_without_mode(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    compile_decoder_blocks(
        ToyModel(),
        CompileConfig(backend="cudagraphs", mode=None, fullgraph=True, dynamic=False),
    )
    assert calls == [{"fullgraph": True, "dynamic": False, "backend": "cudagraphs"}] * 2


def test_compile_decoder_blocks_skips_already_compiled(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    model = ToyModel()
    first = compile_decoder_blocks(model, CompileConfig())
    second = compile_decoder_blocks(model, CompileConfig())

    assert first == 2
    assert second == 0


def test_compile_decoder_blocks_no_decoder_layers_returns_zero(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    class NoDecoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = ToyVisionBlock()

    assert compile_decoder_blocks(NoDecoderModel(), CompileConfig()) == 0


def test_mark_compile_step_begin_calls_torch_compiler_api(monkeypatch):
    calls = []

    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch, "compiler", SimpleNamespace(cudagraph_mark_step_begin=lambda: calls.append("mark")))

    mark_compile_step_begin(enable_compile=True)
    mark_compile_step_begin(enable_compile=False)

    assert calls == ["mark"]


def test_mark_compile_step_begin_skips_non_cuda(monkeypatch):
    calls = []

    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(torch, "compiler", SimpleNamespace(cudagraph_mark_step_begin=lambda: calls.append("mark")))

    mark_compile_step_begin(enable_compile=True)

    assert calls == []


def test_mark_compile_step_begin_skips_without_torch_compiler(monkeypatch):
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)
    monkeypatch.delattr(torch, "compiler", raising=False)

    mark_compile_step_begin(enable_compile=True)


def test_vlm_train_step_marks_each_compile_micro_batch(monkeypatch):
    from veomni.trainer.base import BaseTrainer
    from veomni.trainer.vlm_trainer import VLMTrainer

    marks = []
    monkeypatch.setattr("veomni.trainer.vlm_trainer.mark_compile_step_begin", marks.append)
    monkeypatch.setattr("veomni.trainer.vlm_trainer.count_loss_token", lambda _: 1)
    monkeypatch.setattr("veomni.trainer.vlm_trainer.reduce_global_loss_token", lambda token_count: token_count)

    trainer = VLMTrainer.__new__(VLMTrainer)
    model_args = SimpleNamespace(
        optimizer=SimpleNamespace(max_grad_norm=1.0),
        accelerator=SimpleNamespace(offload_config=OffloadConfig()),
    )
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(model=model_args),
        state=SimpleNamespace(global_step=0),
        model=SimpleNamespace(
            args=model_args,
            _veomni_compile_uses_cuda_graphs=True,
            clip_grad_norm=lambda: torch.tensor(0.0),
            optimizer=SimpleNamespace(step=lambda: None, zero_grad=lambda: None),
            lr_scheduler=SimpleNamespace(step=lambda: None),
        ),
        model_reshard=lambda *_: None,
        _configure_hsdp_allreduce=lambda *_: None,
        sync_before_train_step=lambda: None,
        forward_backward_step=lambda _: (torch.tensor(1.0), {}, {}),
        on_step_begin=lambda **_: None,
        on_step_end=lambda **_: None,
    )
    trainer.base._reset_async_activation_offload_if_enabled = lambda model: (
        BaseTrainer._reset_async_activation_offload_if_enabled(trainer.base, model)
    )

    trainer.train_step(iter([[{}, {}]]))

    assert marks == [True, True]


def test_vlm_runtime_rejects_unsupported_compile_model_while_building(monkeypatch):
    # The rejection has to land during the model build, which the runtime does on
    # construction — i.e. before the trainer touches data.
    from veomni.trainer.vlm_trainer import VLMModelRuntime

    calls = []

    def build_unsupported_model(runtime):
        calls.append("_build_model")
        runtime.model = ToyModel()
        runtime.model.config = SimpleNamespace(model_type="qwen2_5_vl", vision_config=SimpleNamespace())
        runtime.model.input_modalities = ("image", "text")
        runtime.model_config = runtime.model.config
        runtime._validate_torch_compile()

    # No mesh is registered here, and none is needed: the rejection happens
    # before anything reads one. Stubbing ``setup`` alone would leave the
    # build's own scope looking for a state that was never registered.
    monkeypatch.setattr(VLMModelRuntime, "setup", lambda _: None)
    monkeypatch.setattr("veomni.models.model_runtime.use_parallel_state", lambda _name: nullcontext())
    monkeypatch.setattr(VLMModelRuntime, "_build_model", build_unsupported_model)
    monkeypatch.setattr(VLMModelRuntime, "_freeze_model_module", lambda _: calls.append("_freeze_model_module"))

    args = SimpleNamespace(
        accelerator=SimpleNamespace(
            torch_compile=ArgumentsTorchCompileConfig(enable=True),
            ulysses_size=1,
            cp_size=1,
            enable_async=False,
        ),
    )
    with pytest.raises(RuntimeError, match="only for dense Qwen3-VL"):
        VLMModelRuntime(args, train=SimpleNamespace())

    assert calls == ["_build_model"]


def test_compile_config_detects_cuda_graphs():
    assert CompileConfig(backend="inductor", mode=None).uses_cuda_graphs() is False
    assert CompileConfig(backend="inductor", mode="reduce-overhead").uses_cuda_graphs() is True
    assert CompileConfig(backend="cudagraphs", mode=None).uses_cuda_graphs() is True


def test_validate_compile_config_rejects_cuda_graphs_with_forward_reshard():
    with pytest.raises(RuntimeError, match="reshard_after_forward=False"):
        validate_compile_config_for_fsdp2(
            CompileConfig(enable=True, backend="inductor", mode="reduce-overhead"),
            enable_reshard_after_forward=True,
        )


def test_validate_compile_config_accepts_inductor_default_mode_with_forward_reshard():
    validate_compile_config_for_fsdp2(
        CompileConfig(enable=True, backend="inductor", mode=None),
        enable_reshard_after_forward=True,
    )


def test_validate_compile_config_accepts_cuda_graphs_without_forward_reshard():
    validate_compile_config_for_fsdp2(
        CompileConfig(enable=True, backend="inductor", mode="reduce-overhead"),
        enable_reshard_after_forward=False,
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"device_type": "cpu"}, "CUDA-only"),
        ({"fsdp_enabled": False}, "requires FSDP2"),
        ({"fsdp_mode": "ddp"}, "fsdp_mode='fsdp2'"),
        ({"any_extra_parallel_enabled": True}, "does not support ExtraParallel"),
    ],
)
def test_validate_compile_runtime_rejects_unsupported_contracts(monkeypatch, overrides, error):
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    runtime = {
        "device_type": get_device_type(),
        "fsdp_enabled": True,
        "fsdp_mode": "fsdp2",
        "any_extra_parallel_enabled": False,
        "enable_reshard_after_forward": True,
    }
    runtime.update(overrides)

    with pytest.raises(RuntimeError, match=error):
        validate_compile_runtime(CompileConfig(enable=True), **runtime)


def test_validate_compile_runtime_rejects_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="CUDA-only"):
        validate_compile_runtime(
            CompileConfig(enable=True),
            device_type=get_device_type(),
            fsdp_enabled=True,
            fsdp_mode="fsdp2",
            any_extra_parallel_enabled=False,
            enable_reshard_after_forward=True,
        )


def test_validate_compile_runtime_rejects_cuda_graphs_with_forward_reshard(monkeypatch):
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="reshard_after_forward=False"):
        validate_compile_runtime(
            CompileConfig(enable=True, backend="inductor", mode="reduce-overhead"),
            device_type=get_device_type(),
            fsdp_enabled=True,
            fsdp_mode="fsdp2",
            any_extra_parallel_enabled=False,
            enable_reshard_after_forward=True,
        )


def test_torch_compile_config_defaults():
    cfg = ArgumentsTorchCompileConfig()
    assert cfg.enable is False
    assert cfg.backend == "inductor"
    assert cfg.mode is None
    assert cfg.fullgraph is True
    assert cfg.dynamic is False


def test_enable_compile_requires_dynamic_batching():
    with pytest.raises(ValueError, match="model.accelerator.torch_compile.enable requires train.dyn_bsz=True"):
        VeOmniArguments(
            model=_model_args(accelerator=AcceleratorConfig(torch_compile=ArgumentsTorchCompileConfig(enable=True))),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                dyn_bsz=False,
                pad_to_length=False,
            ),
        )


def test_enable_compile_requires_padding_for_dynamic_batching():
    with pytest.raises(
        ValueError,
        match="model.accelerator.torch_compile.enable requires train.dyn_bsz=True and train.pad_to_length=True",
    ):
        VeOmniArguments(
            model=_model_args(accelerator=AcceleratorConfig(torch_compile=ArgumentsTorchCompileConfig(enable=True))),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                dyn_bsz=True,
                pad_to_length=False,
            ),
        )


def test_enable_compile_accepts_static_padded_dynamic_batching():
    args = VeOmniArguments(
        model=_model_args(accelerator=AcceleratorConfig(torch_compile=ArgumentsTorchCompileConfig(enable=True))),
        data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
        train=TrainingArguments(
            dyn_bsz=True,
            pad_to_length=True,
            micro_batch_size=2,
        ),
    )

    assert args.train.pad_to_length == 16


@dataclass
class ToyMultimodalDataArguments(DataArguments):
    supports_torch_compile = False

    mm_configs: dict = field(default_factory=dict)


def test_enable_compile_rejects_unsupported_data_pipeline():
    with pytest.raises(ValueError, match="not supported by this data pipeline"):
        VeOmniArguments(
            model=_model_args(accelerator=AcceleratorConfig(torch_compile=ArgumentsTorchCompileConfig(enable=True))),
            data=ToyMultimodalDataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                dyn_bsz=True,
                pad_to_length=True,
                micro_batch_size=2,
            ),
        )


def test_enable_compile_accepts_vlm_static_padded_dynamic_batching():
    from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMMDataArguments

    args = VeOmniVLMArguments(
        model=_model_args(accelerator=AcceleratorConfig(torch_compile=ArgumentsTorchCompileConfig(enable=True))),
        data=VLMMDataArguments(train_path="dummy.jsonl", max_seq_len=8),
        train=TrainingArguments(
            dyn_bsz=True,
            pad_to_length=True,
            micro_batch_size=2,
        ),
    )

    assert args.train.pad_to_length == 16


@dataclass
class ToyTextDataArguments(DataArguments):
    extra_text_config: str = "text"


def test_enable_compile_accepts_text_data_argument_subclass():
    args = VeOmniArguments(
        model=_model_args(accelerator=AcceleratorConfig(torch_compile=ArgumentsTorchCompileConfig(enable=True))),
        data=ToyTextDataArguments(train_path="dummy.jsonl", max_seq_len=8),
        train=TrainingArguments(
            dyn_bsz=True,
            pad_to_length=True,
            micro_batch_size=2,
        ),
    )

    assert args.train.pad_to_length == 16


def _deepseek_v4_fused_moe_ops_config():
    from ..models.compare import eager_ops_config

    ops = eager_ops_config()
    ops.moe_implementation = "fused_triton"
    return ops


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek V4 fused MoE kernel requires CUDA")
@pytest.mark.parametrize("compile_backend", ["eager"])
def test_deepseek_v4_fused_decoder_fullgraph_forward_backward_matches_eager(compile_backend):
    from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
        DeepseekV4ForCausalLM,
    )

    from ..models.compare import ops_config_scope

    torch.manual_seed(0)
    device = torch.device(get_device_type())
    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    config._attn_implementation = "eager"

    with ops_config_scope(_deepseek_v4_fused_moe_ops_config()):
        eager_model = DeepseekV4ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
        compiled_model = DeepseekV4ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    compiled_model.load_state_dict(eager_model.state_dict())

    inputs = torch.randint(0, config.vocab_size, (2, 16), device=device)

    parallel_state = SimpleNamespace(
        async_enabled=False,
        cp_enabled=False,
        ulysses_enabled=False,
        sp_enabled=False,
        ep_enabled=False,
    )
    with use_parallel_state(parallel_state):
        assert (
            compile_decoder_blocks(
                compiled_model,
                CompileConfig(enable=True, backend=compile_backend, fullgraph=True, dynamic=False),
            )
            == config.num_hidden_layers
        )
        eager_loss = eager_model(input_ids=inputs, labels=inputs, use_cache=False).loss
        compiled_loss = compiled_model(input_ids=inputs, labels=inputs, use_cache=False).loss
        eager_loss.backward()
        compiled_loss.backward()

    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-4, atol=1e-4)
    for (name, eager_param), (_, compiled_param) in zip(
        eager_model.named_parameters(),
        compiled_model.named_parameters(),
        strict=True,
    ):
        if eager_param.grad is None and compiled_param.grad is None:
            continue
        assert eager_param.grad is not None and compiled_param.grad is not None, name
        torch.testing.assert_close(compiled_param.grad, eager_param.grad, rtol=1e-3, atol=1e-3, msg=name)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek V4 fused MoE kernel requires CUDA")
def test_deepseek_v4_inductor_fullgraph_executes_finite_optimizer_step():
    from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
        DeepseekV4ForCausalLM,
    )

    from ..models.compare import ops_config_scope

    torch.manual_seed(0)
    device = torch.device(get_device_type())
    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    config._attn_implementation = "eager"

    with ops_config_scope(_deepseek_v4_fused_moe_ops_config()):
        model = DeepseekV4ForCausalLM(config).to(device=device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batches = [
        torch.randint(0, config.vocab_size, (2, 16), device=device),
        torch.randint(0, config.vocab_size, (2, 16), device=device),
    ]

    parallel_state = SimpleNamespace(
        async_enabled=False,
        cp_enabled=False,
        ulysses_enabled=False,
        sp_enabled=False,
        ep_enabled=False,
    )
    with use_parallel_state(parallel_state):
        assert (
            compile_decoder_blocks(
                model,
                CompileConfig(enable=True, backend="inductor", fullgraph=True, dynamic=False),
            )
            == config.num_hidden_layers
        )
        losses = []
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=batch, labels=batch, use_cache=False).loss
            assert torch.isfinite(loss), f"non-finite DeepSeek V4 compiled loss: {loss}"
            loss.backward()
            grad_tensors = [param.grad for param in model.parameters() if param.grad is not None]
            assert grad_tensors, "expected gradients after backward"
            assert all(torch.isfinite(grad).all() for grad in grad_tensors)
            optimizer.step()
            losses.append(loss.detach())

    assert all(torch.isfinite(loss) for loss in losses)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek V4 fused MoE kernel requires CUDA")
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_deepseek_v4_inductor_fixed_route_expert_vjp_matches_fp32_reference(seed):
    from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
        DeepseekV4ForCausalLM,
    )
    from veomni.ops import VeomniOp

    from ..models.compare import ops_config_scope

    torch.manual_seed(seed)
    device = torch.device(get_device_type())
    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")

    with ops_config_scope(_deepseek_v4_fused_moe_ops_config()):
        model = DeepseekV4ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    experts = model.model.layers[2].mlp.experts

    num_tokens = 12
    x_init = torch.randn(num_tokens, config.hidden_size, device=device, dtype=torch.bfloat16)
    logits = torch.randn(num_tokens, config.n_routed_experts, device=device, dtype=torch.float32)
    weights, indices = torch.topk(torch.softmax(logits, dim=-1), k=config.num_experts_per_tok, dim=-1)
    grad_out = torch.randn_like(x_init)

    parallel_state = SimpleNamespace(
        async_enabled=False,
        cp_enabled=False,
        ulysses_enabled=False,
        sp_enabled=False,
        ep_enabled=False,
    )
    with use_parallel_state(parallel_state):
        compiled_experts = copy.deepcopy(experts)
        compiled_fn = torch.compile(compiled_experts, backend="inductor", fullgraph=True, dynamic=False)
        compiled_x = x_init.detach().clone().requires_grad_(True)
        compiled_route = weights.detach().clone().requires_grad_(True)
        compiled_out = compiled_fn(compiled_x, indices, compiled_route)
        (compiled_out * grad_out).sum().backward()

        eager_experts = copy.deepcopy(experts)
        eager_x = x_init.detach().clone().requires_grad_(True)
        eager_route = weights.detach().clone().requires_grad_(True)
        eager_out = eager_experts(eager_x, indices, eager_route)
        (eager_out * grad_out).sum().backward()

    reference = copy.deepcopy(experts).float()
    reference.veomni_moe = VeomniOp("moe_experts", "standard", "eager")
    reference_x = x_init.detach().float().requires_grad_(True)
    reference_route = weights.detach().float().requires_grad_(True)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        expected = reference(reference_x, indices, reference_route)
        (expected * grad_out.float()).sum().backward()
    finally:
        torch.set_float32_matmul_precision(precision)

    for actual, label in ((compiled_out, "compiled"), (eager_out, "eager")):
        torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=2e-2, msg=label)
    torch.testing.assert_close(compiled_out, eager_out, rtol=1e-2, atol=2e-2)

    for actual_grad, expected_grad, name in (
        (compiled_x.grad, reference_x.grad, "hidden_states"),
        (compiled_route.grad, reference_route.grad, "routing_weights"),
        (compiled_experts.gate_up_proj.grad, reference.gate_up_proj.grad, "gate_up_proj"),
        (compiled_experts.down_proj.grad, reference.down_proj.grad, "down_proj"),
    ):
        cosine = F.cosine_similarity(actual_grad.float().flatten(), expected_grad.flatten(), dim=0).item()
        assert cosine > 0.999, f"{name} cosine={cosine:.6f}"
        torch.testing.assert_close(actual_grad.float(), expected_grad, rtol=2e-2, atol=3e-2, msg=name)

    for compiled_grad, eager_grad, name in (
        (compiled_x.grad, eager_x.grad, "hidden_states"),
        (compiled_route.grad, eager_route.grad, "routing_weights"),
        (compiled_experts.gate_up_proj.grad, eager_experts.gate_up_proj.grad, "gate_up_proj"),
        (compiled_experts.down_proj.grad, eager_experts.down_proj.grad, "down_proj"),
    ):
        torch.testing.assert_close(compiled_grad, eager_grad, rtol=1e-2, atol=2e-2, msg=name)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek V4 fused MoE kernel requires CUDA")
@pytest.mark.parametrize("seed", [1, 2])
def test_deepseek_v4_compiled_decoder_gradients_track_the_fp32_oracle(seed):
    from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
        DeepseekV4ForCausalLM,
    )
    from veomni.ops.batch_invariant import set_batch_invariant_mode

    from ..models.compare import eager_ops_config, ops_config_scope

    torch.manual_seed(seed)
    device = torch.device(get_device_type())
    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    config._attn_implementation = "eager"

    with ops_config_scope(eager_ops_config()):
        oracle = DeepseekV4ForCausalLM(config).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for module in oracle.modules():
            if hasattr(module, "tid2eid"):
                table = module.tid2eid
                tokens = torch.arange(table.shape[0], device=table.device).view(-1, 1)
                slots = torch.arange(table.shape[1], device=table.device).view(1, -1)
                table.copy_((tokens + slots) % module.num_experts)
    base_state = {name: value.detach().cpu() for name, value in oracle.state_dict().items()}

    def build_bf16_model():
        with ops_config_scope(_deepseek_v4_fused_moe_ops_config()):
            model = DeepseekV4ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
        model.load_state_dict(
            {
                name: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                for name, value in base_state.items()
            }
        )
        oracle.load_state_dict(
            {
                name: value.detach().to(
                    device=device,
                    dtype=torch.float32 if value.is_floating_point() else value.dtype,
                )
                for name, value in model.state_dict().items()
            }
        )
        return model

    eager_model = build_bf16_model()
    compiled_model = build_bf16_model()
    generator = torch.Generator(device="cpu").manual_seed(seed + 100)
    inputs = torch.randint(0, config.vocab_size, (2, 16), generator=generator).to(device)

    parallel_state = SimpleNamespace(
        async_enabled=False,
        cp_enabled=False,
        ulysses_enabled=False,
        sp_enabled=False,
        ep_enabled=False,
    )
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with use_parallel_state(parallel_state):
            oracle_loss = oracle(input_ids=inputs, labels=inputs, use_cache=False).loss
            oracle_loss.backward()
    finally:
        torch.set_float32_matmul_precision(precision)

    with set_batch_invariant_mode(True):
        with use_parallel_state(parallel_state):
            assert (
                compile_decoder_blocks(
                    compiled_model,
                    CompileConfig(enable=True, backend="inductor", fullgraph=True, dynamic=False),
                )
                == config.num_hidden_layers
            )
            eager_loss = eager_model(input_ids=inputs, labels=inputs, use_cache=False).loss
            compiled_loss = compiled_model(input_ids=inputs, labels=inputs, use_cache=False).loss
            eager_loss.backward()
            compiled_loss.backward()

    torch.testing.assert_close(eager_loss.float(), oracle_loss, rtol=5e-3, atol=1e-2)
    torch.testing.assert_close(compiled_loss.float(), oracle_loss, rtol=5e-3, atol=1e-2)
    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-3, atol=2e-3)

    oracle_grads = {name: param.grad for name, param in oracle.named_parameters() if param.grad is not None}
    eager_grads = {name: param.grad for name, param in eager_model.named_parameters() if param.grad is not None}
    compiled_grads = {name: param.grad for name, param in compiled_model.named_parameters() if param.grad is not None}
    assert compiled_grads.keys() == eager_grads.keys() == oracle_grads.keys()

    def total_cosine(grads):
        dot = sum(
            torch.sum(grads[name].float() * oracle_grads[name].float())
            for name in oracle_grads
            if "indexer.weights_proj" not in name
        )
        left = torch.sqrt(
            sum(torch.sum(grads[name].float().square()) for name in oracle_grads if "indexer.weights_proj" not in name)
        )
        right = torch.sqrt(
            sum(
                torch.sum(oracle_grads[name].float().square())
                for name in oracle_grads
                if "indexer.weights_proj" not in name
            )
        )
        return (dot / (left * right)).item()

    eager_total_cosine = total_cosine(eager_grads)
    compiled_total_cosine = total_cosine(compiled_grads)
    assert eager_total_cosine > 0.995, f"eager total cosine={eager_total_cosine:.6f}"
    assert compiled_total_cosine > 0.995, f"compiled total cosine={compiled_total_cosine:.6f}"
    assert abs(compiled_total_cosine - eager_total_cosine) < 2e-3

    for name, oracle_grad in oracle_grads.items():
        if "indexer.weights_proj" in name or oracle_grad.norm().item() < 1e-4:
            continue
        eager_cos = F.cosine_similarity(eager_grads[name].float().flatten(), oracle_grad.flatten(), dim=0).item()
        compiled_cos = F.cosine_similarity(
            compiled_grads[name].float().flatten(),
            oracle_grad.flatten(),
            dim=0,
        ).item()
        assert compiled_cos > 0.97, f"{name}: compiled cosine={compiled_cos:.6f}, eager cosine={eager_cos:.6f}"
        assert compiled_cos >= eager_cos - 5e-3, (
            f"{name}: compiled cosine={compiled_cos:.6f} trailed eager cosine={eager_cos:.6f}"
        )


@pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
def test_deepseek_v4_rms_norm_compiled_backward_matches_eager(compile_backend):
    from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
        DeepseekV4RMSNorm,
    )

    from ..models.compare import eager_ops_config, ops_config_scope

    torch.manual_seed(0)
    with ops_config_scope(eager_ops_config()):
        eager_norm = DeepseekV4RMSNorm(64)
        compiled_norm = copy.deepcopy(eager_norm)

    x_eager = torch.randn(2, 16, 64, dtype=torch.float32, requires_grad=True)
    x_compiled = x_eager.detach().clone().requires_grad_(True)
    grad_out = torch.randn(2, 16, 64, dtype=torch.float32)

    eager_out = eager_norm(x_eager)
    compiled_out = torch.compile(compiled_norm, backend=compile_backend, fullgraph=True, dynamic=False)(x_compiled)
    eager_out.backward(grad_out)
    compiled_out.backward(grad_out)

    torch.testing.assert_close(compiled_out, eager_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_compiled.grad, x_eager.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(compiled_norm.weight.grad, eager_norm.weight.grad, rtol=1e-5, atol=1e-5)
