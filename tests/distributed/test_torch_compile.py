import copy
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from veomni.arguments.arguments_types import (
    AcceleratorConfig,
    ChunkMBSConfig,
    DataArguments,
    GradientCheckpointingConfig,
    ModelArguments,
    OpsImplementationConfig,
    TrainingArguments,
    VeOmniArguments,
)
from veomni.arguments.arguments_types import (
    TorchCompileConfig as ArgumentsTorchCompileConfig,
)
from veomni.distributed.parallel_state import use_parallel_state
from veomni.distributed.torch_compile import (
    CompileConfig,
    compile_decoder_blocks,
    mark_compile_step_begin,
    validate_compile_config_for_fsdp2,
    validate_compile_runtime,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


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
    compiled_layer = copy.deepcopy(eager_layer)
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
    with use_parallel_state(SimpleNamespace(async_enabled=False)):
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
    compiled_layer = copy.deepcopy(eager_layer)
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
    with use_parallel_state(SimpleNamespace(async_enabled=False)):
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
    from veomni.trainer.vlm_trainer import VLMTrainer

    marks = []
    monkeypatch.setattr("veomni.trainer.vlm_trainer.mark_compile_step_begin", marks.append)
    monkeypatch.setattr("veomni.trainer.vlm_trainer.count_loss_token", lambda _: 1)
    monkeypatch.setattr("veomni.trainer.vlm_trainer.reduce_global_loss_token", lambda token_count: token_count)

    trainer = VLMTrainer.__new__(VLMTrainer)
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(model=SimpleNamespace(optimizer=SimpleNamespace(max_grad_norm=1.0))),
        state=SimpleNamespace(global_step=0),
        model=SimpleNamespace(
            _veomni_compile_uses_cuda_graphs=True,
            clip_grad_norm=lambda: torch.tensor(0.0),
        ),
        model_reshard=lambda *_: None,
        _configure_hsdp_allreduce=lambda *_: None,
        sync_before_train_step=lambda: None,
        forward_backward_step=lambda _: (torch.tensor(1.0), {}),
        optimizer=SimpleNamespace(step=lambda: None, zero_grad=lambda: None),
        lr_scheduler=SimpleNamespace(step=lambda: None),
        on_step_begin=lambda **_: None,
        on_step_end=lambda **_: None,
    )

    trainer.train_step(iter([[{}, {}]]))

    assert marks == [True, True]


def test_vlm_runtime_rejects_unsupported_compile_model_while_building(monkeypatch):
    # The rejection has to land during the model build, which the runtime does on
    # construction — i.e. before the trainer touches data.
    from veomni.trainer.vlm_trainer import VLMModelRuntime

    calls = []

    def build_unsupported_model(runtime):
        calls.append("build_model")
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
    monkeypatch.setattr(VLMModelRuntime, "build_model", build_unsupported_model)
    monkeypatch.setattr(VLMModelRuntime, "freeze_model", lambda _: calls.append("freeze_model"))

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

    assert calls == ["build_model"]


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


def test_enable_compile_rejects_chunk_mbs():
    with pytest.raises(ValueError, match="model.accelerator.chunk_mbs_config.enable is not supported"):
        VeOmniArguments(
            model=_model_args(
                accelerator=AcceleratorConfig(
                    torch_compile=ArgumentsTorchCompileConfig(enable=True),
                    chunk_mbs_config=ChunkMBSConfig(enable=True),
                ),
            ),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                dyn_bsz=True,
                pad_to_length=False,
            ),
        )


def test_chunk_mbs_rejects_static_padding():
    with pytest.raises(ValueError, match="not supported with train.pad_to_length"):
        VeOmniArguments(
            model=_model_args(accelerator=AcceleratorConfig(chunk_mbs_config=ChunkMBSConfig(enable=True))),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                dyn_bsz=True,
                pad_to_length=True,
                micro_batch_size=2,
            ),
        )


def test_chunk_mbs_rejects_reentrant_gradient_checkpointing():
    with pytest.raises(ValueError, match="requires non-reentrant gradient checkpointing"):
        VeOmniArguments(
            model=_model_args(
                accelerator=AcceleratorConfig(
                    chunk_mbs_config=ChunkMBSConfig(enable=True),
                    gradient_checkpointing=GradientCheckpointingConfig(enable_reentrant=True),
                ),
            ),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(),
        )


def test_chunk_mbs_rejects_dpo_trainer():
    with pytest.raises(ValueError, match="not supported by the DPO trainer"):
        VeOmniArguments(
            model=_model_args(accelerator=AcceleratorConfig(chunk_mbs_config=ChunkMBSConfig(enable=True))),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8, data_type="dpo"),
            train=TrainingArguments(),
        )


def test_dpo_trainer_rejects_chunk_mbs_regardless_of_data_type():
    from veomni.trainer.text_dpo_trainer import TextDPOTrainer

    args = SimpleNamespace(
        model=SimpleNamespace(accelerator=SimpleNamespace(chunk_mbs_config=ChunkMBSConfig(enable=True)))
    )
    with pytest.raises(ValueError, match="not supported by the DPO trainer"):
        TextDPOTrainer(args)


def test_rl_trainer_rejects_chunk_mbs():
    from veomni.trainer.base_rl_trainer import BaseRLTrainer

    args = SimpleNamespace(
        model=SimpleNamespace(accelerator=SimpleNamespace(chunk_mbs_config=ChunkMBSConfig(enable=True)))
    )
    trainer = BaseRLTrainer.__new__(BaseRLTrainer)
    trainer.args = args
    with pytest.raises(ValueError, match="not supported by RL trainers"):
        trainer.setup_distributed(args)


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
