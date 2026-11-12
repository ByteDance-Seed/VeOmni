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

import builtins
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from veomni.arguments import AcceleratorConfig, Float8TrainingConfig, MixedPrecisionConfig
from veomni.arguments.parser import parse_args, save_args
from veomni.distributed.float8 import convert_float8_linears, validate_float8_training
from veomni.utils import device


cpu = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
TARGET = "model.layers.0.mlp.up_proj"


def _precision(*targets, recipe="tensorwise"):
    return MixedPrecisionConfig(float8=Float8TrainingConfig(enable=True, recipe=recipe, target_modules=list(targets)))


def _state(**overrides):
    values = dict(
        dp_mode="fsdp2", sp_enabled=False, any_extra_parallel_enabled=False, tp_size=1, pp_size=1, dp_replicate_size=1
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class DenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.up_proj = nn.Linear(32, 32, bias=False)
        self.lm_head = nn.Linear(32, 32, bias=False)

    @property
    def fc(self):
        return self.model.layers[0].mlp.up_proj


@cpu
def test_float8_disabled_never_imports_torchao(monkeypatch):
    """Disabled float8 config never imports torchao or mutates model layers."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("torchao")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    model = DenseModel()
    precision = MixedPrecisionConfig()
    original = model.fc
    validate_float8_training(precision, _state(dp_mode="ddp"), cpu_offload=True, compile_enabled=True)
    assert convert_float8_linears(model, precision.float8) == []
    assert model.fc is original
    assert not asdict(precision)["float8"]["enable"]


@cpu
@pytest.mark.parametrize("targets", [[], [""], "fc", [1]])
def test_float8_requires_explicit_module_patterns(targets):
    with pytest.raises(ValueError, match="target_modules"):
        Float8TrainingConfig(enable=True, target_modules=targets)


@cpu
def test_float8_rejects_unknown_recipe():
    with pytest.raises(ValueError, match="recipe"):
        Float8TrainingConfig(enable=True, recipe="invalid", target_modules=[TARGET])


@cpu
@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise", "rowwise_with_gw_hp"])
def test_float8_yaml_cli_and_saved_config_roundtrip(monkeypatch, tmp_path, recipe):
    path = tmp_path / "accelerator.yaml"
    path.write_text(
        f"fsdp_config:\n  mixed_precision:\n    float8:\n      enable: true\n      recipe: {recipe}\n"
        "      target_modules:\n        - model.layers.*.mlp.up_proj\n"
    )
    monkeypatch.setattr("sys.argv", ["test", str(path)])
    config = parse_args(AcceleratorConfig)
    assert isinstance(config.fsdp_config.mixed_precision.float8, Float8TrainingConfig)
    assert config.fsdp_config.mixed_precision.float8.enable
    assert config.fsdp_config.mixed_precision.float8.recipe == recipe
    assert config.fsdp_config.mixed_precision.float8.target_modules == ["model.layers.*.mlp.up_proj"]
    save_args(config, str(tmp_path / "saved"))
    monkeypatch.setattr("sys.argv", ["test", str(tmp_path / "saved" / "veomni_cli.yaml")])
    assert asdict(parse_args(AcceleratorConfig)) == asdict(config)


@cpu
@pytest.mark.parametrize(
    "changes, kwargs, message",
    [
        ({"dp_mode": "ddp"}, {}, "FSDP2"),
        ({"sp_enabled": True}, {}, "sequence"),
        ({"any_extra_parallel_enabled": True}, {}, "ExtraParallel"),
        ({"tp_size": 2}, {}, "tensor"),
        ({"pp_size": 2}, {}, "pipeline"),
        ({"dp_replicate_size": 2}, {}, "HSDP"),
        ({}, {"cpu_offload": True}, "offload"),
        ({}, {"compile_enabled": True}, "compile"),
        ({}, {"is_peft_model": True}, "LoRA"),
    ],
)
def test_float8_rejects_unvalidated_combinations(monkeypatch, changes, kwargs, message):
    monkeypatch.setattr(device, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(device, "get_gpu_compute_capability", lambda: 89)
    with pytest.raises(ValueError, match=message):
        validate_float8_training(_precision("fc"), _state(**changes), **kwargs)


@cpu
@pytest.mark.parametrize(
    "case, message",
    [
        ("missing", "matched no"),
        ("shape", "divisible"),
        ("head", "not a supported"),
        ("tied", "shared"),
        ("subclass", "plain"),
        ("model", "currently supports"),
        ("dtype", "FP32"),
    ],
)
def test_float8_selection_validates_before_optional_import(case, message):
    """Reject invalid target modules, shapes, dtypes, and model types before importing torchao."""
    model = DenseModel()
    targets = [TARGET]
    if case == "missing":
        targets.append("missing")
    elif case == "shape":
        model.model.layers[0].mlp.up_proj = nn.Linear(32, 8, bias=False)
    elif case == "head":
        targets = ["lm_head"]
    elif case == "tied":
        model.alias = model.fc
    elif case == "subclass":

        class CustomLinear(nn.Linear):
            pass

        model.model.layers[0].mlp.up_proj = CustomLinear(32, 32, bias=False)
    elif case == "model":
        model.config.model_type = "qwen3_moe"
    elif case == "dtype":
        model.fc.bfloat16()
    original = model.fc
    with pytest.raises(ValueError, match=message):
        convert_float8_linears(model, Float8TrainingConfig(enable=True, target_modules=targets))
    assert model.fc is original


@cpu
@pytest.mark.parametrize(
    ("installed", "message"),
    [
        (None, r"optional extra: uv sync --extra gpu --extra float8 --dev"),
        ("0.17.0", r"requires torchao==0\.18\.0; found 0\.17\.0"),
    ],
)
def test_float8_reports_a_missing_or_mismatched_torchao_extra(monkeypatch, installed, message):
    """torchao is the optional ``float8`` extra, so a missing package must name the extra to install."""
    from importlib.metadata import PackageNotFoundError

    import veomni.distributed.float8 as float8

    def fake_version(name):
        assert name == "torchao"
        if installed is None:
            raise PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(float8, "version", fake_version)
    model = DenseModel()
    original = model.fc
    with pytest.raises(ImportError, match=message):
        convert_float8_linears(model, Float8TrainingConfig(enable=True, target_modules=[TARGET]))
    assert model.fc is original


@cpu
def test_float8_dit_selection_allows_block_projections_and_rejects_heads():
    from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
        WanTransformer3DModelConfig,
    )
    from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import WanTransformer3DModel

    with torch.device("meta"):
        cfg = WanTransformer3DModelConfig(
            num_layers=1,
            num_attention_heads=4,
            attention_head_dim=32,
            in_channels=16,
            out_channels=16,
            text_dim=64,
            freq_dim=64,
            ffn_dim=256,
            attn_implementation="eager",
        )
        wan = WanTransformer3DModel(cfg).float()

    with pytest.raises(ValueError, match="not a supported"):
        convert_float8_linears(wan, Float8TrainingConfig(enable=True, target_modules=["proj_out"]))
    with pytest.raises(ValueError, match="not a supported"):
        convert_float8_linears(wan, Float8TrainingConfig(enable=True, target_modules=["condition_embedder.time_proj"]))


def _require_native_float8():
    """Skip if the host lacks an NVIDIA SM89+ CUDA device or torchao.float8."""
    if not device.IS_CUDA_AVAILABLE or torch.version.hip or device.get_gpu_compute_capability() < 89:
        pytest.skip("native float8 training requires NVIDIA SM89+")
    pytest.importorskip("torchao.float8")


@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise", "rowwise_with_gw_hp"])
@pytest.mark.parametrize("tokens", [16, 17, 33])
def test_float8_recipes_native_kernels_and_ragged_bf16_reference(recipe, tokens):
    """Verify native aten::_scaled_mm dispatch and BF16 error bounds across recipes and ragged token counts."""
    _require_native_float8()
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(13)
    model = DenseModel().cuda()
    original_weight = model.fc.weight
    reference = deepcopy(model.fc).bfloat16()
    untouched = model.lm_head
    assert convert_float8_linears(
        model, Float8TrainingConfig(enable=True, recipe=recipe, target_modules=[TARGET])
    ) == [TARGET]
    assert model.fc.weight is original_weight
    assert model.lm_head is untouched
    model.fc.bfloat16()
    inputs = torch.randn(tokens, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    grad_output = torch.randn(tokens, 32, device="cuda", dtype=torch.bfloat16)
    expected = reference(reference_inputs)
    expected.backward(grad_output)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        actual = model.fc(inputs)
        actual.backward(grad_output)
        device.synchronize()
    assert actual.shape == expected.shape
    for candidate, baseline, tolerance in (
        (actual, expected, 0.08),
        (inputs.grad, reference_inputs.grad, 0.15),
        (model.fc.weight.grad, reference.weight.grad, 0.15),
    ):
        assert torch.isfinite(candidate).all()
        relative_error = (candidate.float() - baseline.float()).norm() / baseline.float().norm()
        assert relative_error < tolerance
    expected_scaled_mm = 2 if recipe == "rowwise_with_gw_hp" else 3
    assert sum(event.count for event in trace.key_averages() if event.key == "aten::_scaled_mm") == expected_scaled_mm


@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise"])
def test_float8_wan_dit_fsdp2_forward_backward(tmp_path, recipe):
    """Verify WanTransformer3DModel DiT block conversion and FSDP2 forward/backward under float8."""
    _require_native_float8()
    import torch.distributed as dist
    from torch.profiler import ProfilerActivity, profile
    from torchao.float8.float8_linear import Float8Linear

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2
    from veomni.models.checkpoint.weights import init_empty_weights
    from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
        WanTransformer3DModelConfig,
    )
    from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
        WanTransformer3DModel,
        apply_veomni_wan_transformer_patch,
    )

    apply_veomni_wan_transformer_patch()
    wan_cfg = WanTransformer3DModelConfig(
        num_layers=1,
        num_attention_heads=4,
        attention_head_dim=32,
        in_channels=16,
        out_channels=16,
        text_dim=64,
        freq_dim=64,
        ffn_dim=256,
        attn_implementation="eager",
    )
    ckpt = tmp_path / "wan_hf"
    torch.manual_seed(0)
    orig = WanTransformer3DModel(wan_cfg).float()
    orig.save_pretrained(ckpt)
    del orig

    dist.init_process_group("cuda:nccl,cpu:gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    try:
        _init_parallel_state(dp_size=1, dp_mode="fsdp2")
        precision = _precision(
            "blocks.*.attn1.to_q",
            "blocks.*.attn1.to_k",
            "blocks.*.attn1.to_v",
            "blocks.*.attn1.to_out.0",
            "blocks.*.attn2.to_q",
            "blocks.*.attn2.to_k",
            "blocks.*.attn2.to_v",
            "blocks.*.attn2.to_out.0",
            "blocks.*.ffn.net.0.proj",
            "blocks.*.ffn.net.2",
            recipe=recipe,
        )
        with init_empty_weights():
            model = WanTransformer3DModel(wan_cfg).float()
        model = parallelize_model_fsdp2(
            model,
            weights_path=str(ckpt),
            init_device="meta",
            mixed_precision=precision,
            broadcast_model_weights_from_rank0=False,
        )
        assert sum(isinstance(m, Float8Linear) for m in model.modules()) == 10
        assert not isinstance(model.proj_out, Float8Linear)
        latents = [torch.randn(1, 16, 2, 8, 8, device="cuda", dtype=torch.bfloat16)]
        hs = [torch.randn(1, 16, 2, 8, 8, device="cuda", dtype=torch.bfloat16)]
        ts = [torch.tensor([500.0], device="cuda", dtype=torch.bfloat16)]
        enc = [torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16)]
        target = [torch.randn(1, 16, 2, 8, 8, device="cuda", dtype=torch.bfloat16)]
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            out = model(
                latents=latents,
                hidden_states=hs,
                timestep=ts,
                encoder_hidden_states=enc,
                training_target=target,
            )
            loss = out.loss["mse_loss"]
            assert torch.isfinite(loss)
            loss.backward()
            device.synchronize()
        assert sum(event.count for event in trace.key_averages() if event.key == "aten::_scaled_mm") == 30
        assert all(
            param.grad is not None and torch.isfinite(param.grad.to_local()).all()
            for param in model.parameters()
            if param.requires_grad
        )
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise"])
def test_float8_qwen3_fsdp2_dcp_resume_and_hf_export(tmp_path, monkeypatch, recipe):
    """Verify Qwen3 FSDP2 float8 training, DCP checkpoint continuation, and FP32 HF export."""
    _require_native_float8()
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torchao.float8.float8_linear import Float8Linear
    from transformers import Qwen3Config
    from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

    from veomni.checkpoint.dcp_checkpointer import ModelState, OptimizerState
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2
    from veomni.models.checkpoint.weights import init_empty_weights, save_model_weights
    from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    checkpoint = tmp_path / "hf"
    torch.manual_seed(0)
    original = Qwen3ForCausalLM(config).float()
    original.save_pretrained(checkpoint)
    del original

    dist.init_process_group("cuda:nccl,cpu:gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    try:
        _init_parallel_state(dp_size=1, dp_mode="fsdp2")
        precision = _precision("model.layers.*.mlp.*_proj", "model.layers.*.self_attn.*_proj", recipe=recipe)

        def build(resume=False):
            with init_empty_weights():
                mod = Qwen3ForCausalLM(config).float()
            return parallelize_model_fsdp2(
                mod,
                weights_path=None if resume else str(checkpoint),
                init_device="meta",
                mixed_precision=precision,
                broadcast_model_weights_from_rank0=False,
                should_skip_hf_weight_load=resume,
            )

        model = build()
        assert sum(isinstance(m, Float8Linear) for m in model.modules()) == 7
        inputs = torch.arange(17, device="cuda").reshape(1, 17)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)

        def step(mod, opt):
            opt.zero_grad()
            loss = mod(input_ids=inputs, labels=inputs, use_cache=False).loss
            loss.backward()
            opt.step()
            return loss.detach()

        step(model, optimizer)
        dcp.save(
            {"model": ModelState(model), "optimizer": OptimizerState(model, optimizer)},
            checkpoint_id=tmp_path / "dcp",
        )
        expected_loss = step(model, optimizer)
        expected_params = {n: p.full_tensor().detach().cpu().clone() for n, p in model.named_parameters()}

        resumed = build(resume=True)
        resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-3, foreach=False)
        dcp.load(
            {"model": ModelState(resumed), "optimizer": OptimizerState(resumed, resumed_opt, load=True)},
            checkpoint_id=tmp_path / "dcp",
        )
        torch.testing.assert_close(step(resumed, resumed_opt), expected_loss, rtol=0, atol=0)
        export = tmp_path / "export"
        save_model_weights(
            str(export), resumed.state_dict(), global_rank=0, save_dtype="float32", model_assets=[config]
        )
        exported = HFQwen3ForCausalLM.from_pretrained(export, dtype=torch.float32)
        for name, param in exported.named_parameters():
            torch.testing.assert_close(param, expected_params[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


def _two_rank_gloo_fsdp2_worker(rank: int, checkpoint_str: str, dcp_str: str, recipe: str, endpoint):
    """Execute 2-rank FSDP2 sharded Llama float8 training and DCP resume."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torchao.float8.float8_linear import Float8Linear
    from transformers import LlamaConfig

    from veomni.checkpoint.dcp_checkpointer import ModelState, OptimizerState
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2
    from veomni.models.checkpoint.weights import init_empty_weights
    from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaForCausalLM

    device.get_torch_device().set_device(rank)
    dist.init_process_group(backend="gloo", store=endpoint.connect(), rank=rank, world_size=2)
    try:
        _init_parallel_state(dp_size=2, dp_mode="fsdp2")
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            tie_word_embeddings=False,
            attn_implementation="eager",
        )
        precision = _precision("model.layers.*.mlp.*_proj", "model.layers.*.self_attn.*_proj", recipe=recipe)

        def build(resume=False):
            with init_empty_weights():
                mod = LlamaForCausalLM(config).float()
            return parallelize_model_fsdp2(
                mod,
                weights_path=None if resume else checkpoint_str,
                init_device="meta",
                mixed_precision=precision,
                broadcast_model_weights_from_rank0=False,
                should_skip_hf_weight_load=resume,
            )

        model = build()
        assert sum(isinstance(m, Float8Linear) for m in model.modules()) == 7
        up_weight = dict(model.named_parameters())["model.layers.0.mlp.up_proj.weight"]
        assert up_weight.to_local().shape == (64, 64)
        inputs = torch.arange(17, device="cuda").reshape(1, 17)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)

        def step(mod, opt):
            opt.zero_grad()
            loss = mod(input_ids=inputs, labels=inputs, use_cache=False).loss
            loss.backward()
            opt.step()
            return loss.detach().cpu()

        step(model, optimizer)
        dcp.save(
            {"model": ModelState(model), "optimizer": OptimizerState(model, optimizer)},
            checkpoint_id=dcp_str,
        )
        expected_loss = step(model, optimizer)
        expected_shards = {n: p.to_local().detach().cpu().clone() for n, p in model.named_parameters()}

        resumed = build(resume=True)
        resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-3, foreach=False)
        dcp.load(
            {"model": ModelState(resumed), "optimizer": OptimizerState(resumed, resumed_opt, load=True)},
            checkpoint_id=dcp_str,
        )
        torch.testing.assert_close(step(resumed, resumed_opt), expected_loss, rtol=0, atol=0)
        for name, param in resumed.named_parameters():
            torch.testing.assert_close(param.to_local().cpu(), expected_shards[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise"])
def test_float8_additional_dit_families(recipe):
    """Verify QwenImage, LTXVideo, and MiniMaxH3 DiT float8 projections dispatch native scaled GEMMs."""
    _require_native_float8()
    from torch.profiler import ProfilerActivity, profile
    from torchao.float8.float8_linear import Float8Linear

    from veomni.models.diffusers.ltx2_3.ltx_transformer.configuration_ltx2_3_transformer import (
        LTXVideoTransformerModelConfig,
    )
    from veomni.models.diffusers.ltx2_3.ltx_transformer.modeling_ltx2_3_transformer import LTXVideoTransformerModel
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer import (
        MiniMaxH3DiTModelConfig,
    )
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.configuration_qwen_image_transformer import (
        QwenImageTransformer2DModelConfig,
    )
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )

    for model, pattern in (
        (
            QwenImageTransformer2DModel(
                QwenImageTransformer2DModelConfig(
                    num_layers=1,
                    num_attention_heads=4,
                    attention_head_dim=32,
                    axes_dims_rope=(8, 12, 12),
                    joint_attention_dim=128,
                    attn_implementation="eager",
                )
            ).float(),
            "transformer_blocks.0.img_mlp.net.0.proj",
        ),
        (
            LTXVideoTransformerModel(
                LTXVideoTransformerModelConfig(
                    num_layers=1,
                    num_attention_heads=4,
                    attention_head_dim=32,
                    cross_attention_dim=128,
                    caption_channels=128,
                    attn_implementation="eager",
                )
            ).float(),
            "transformer_blocks.0.ff.net.0.proj",
        ),
        (
            MiniMaxH3DiTModel(
                MiniMaxH3DiTModelConfig(
                    hidden_size=128,
                    num_layers=1,
                    token_refiner_num_layers=1,
                    num_attention_heads=4,
                    attention_head_dim=32,
                    ffn_hidden_size=256,
                    text_dim=128,
                    time_embed_hidden_size=128,
                    time_embed_dim=64,
                    adaln_out_features=18 * 128,
                    final_adaln_out_features=2 * 128,
                    attn_implementation="eager",
                )
            ).float(),
            "dit.blocks.0.mlp.fc1",
        ),
    ):
        assert convert_float8_linears(
            model, Float8TrainingConfig(enable=True, recipe=recipe, target_modules=[pattern])
        ) == [pattern]
        leaf = dict(model.named_modules())[pattern].cuda().bfloat16()
        assert isinstance(leaf, Float8Linear)
        x = torch.randn(17, leaf.in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            leaf(x).square().mean().backward()
            device.synchronize()
        assert sum(e.count for e in trace.key_averages() if e.key == "aten::_scaled_mm") == 3


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise"])
def test_float8_two_rank_sharded_fsdp2(tmp_path, recipe):
    """Verify 2-rank FSDP2 sharded Llama float8 training and DCP checkpoint continuation."""
    if not device.IS_CUDA_AVAILABLE or torch.version.hip or runner_device_count() < 2:
        pytest.skip("Requires two NVIDIA CUDA GPUs.")
    if not all(device.get_gpu_compute_capability(rank) >= 89 for rank in range(2)):
        pytest.skip("native float8 training requires NVIDIA SM89+")
    pytest.importorskip("torchao.float8")
    from transformers import LlamaConfig

    from tests.tools.launch_utils import spawn_with_rendezvous
    from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaForCausalLM

    ckpt = tmp_path / "llama_hf"
    torch.manual_seed(0)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            tie_word_embeddings=False,
            attn_implementation="eager",
        )
    ).float().save_pretrained(ckpt)
    spawn_with_rendezvous(
        _two_rank_gloo_fsdp2_worker,
        nprocs=2,
        args=(str(ckpt), str(tmp_path / f"dcp_{recipe}"), recipe),
    )
