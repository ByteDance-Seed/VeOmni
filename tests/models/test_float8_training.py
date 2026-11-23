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

"""Float8 training tests through the unified ``veomni.ops`` ``torchao`` rows."""

import builtins
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from veomni.arguments import MixedPrecisionConfig, ModelArguments, OpsImplementationConfig
from veomni.arguments.parser import parse_args, save_args
from veomni.models.configuration_base import as_veomni_config
from veomni.ops import OpsSelection, VeomniOp
from veomni.ops.torchao import torchao_fp8_linear
from veomni.utils import device


cpu = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _fp8_selection(recipe: str = "tensorwise", *, emulated: bool | None = None, overrides=None) -> OpsSelection:
    kwargs: dict = {"dtype": "fp8", "recipe": recipe}
    if emulated is not None:
        kwargs["emulated"] = emulated
    return OpsSelection(name="torchao", kwargs=kwargs, overrides=list(overrides or ()))


def _attach_ops(config, ops):
    config = as_veomni_config(config)
    config._set_ops_implementation(ops)
    return config


@cpu
def test_mixed_precision_config_has_no_float8_patch():
    """Float8 selection lives in ``veomni.ops`` rows, not ``MixedPrecisionConfig``."""
    precision = MixedPrecisionConfig()
    assert not hasattr(precision, "float8")
    assert "float8" not in asdict(precision)


@cpu
def test_float8_disabled_never_imports_torchao(monkeypatch):
    """Default eager ops never import torchao or replace plain ``nn.Linear`` modules."""
    from transformers import Qwen3Config

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("torchao")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        attn_implementation="eager",
    )
    config = _attach_ops(config, make_eager_ops_config())
    model = Qwen3ForCausalLM(config)
    x = torch.randint(0, 64, (1, 4))
    out = model(input_ids=x, use_cache=False).logits
    assert out.shape == (1, 4, 64)
    assert type(model.model.layers[0].mlp.up_proj) is nn.Linear


@cpu
def test_float8_ops_yaml_cli_and_saved_config_roundtrip(monkeypatch, tmp_path):
    """Float8 op selections round-trip cleanly through YAML and ``save_args``."""
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    path = tmp_path / "model.yaml"
    path.write_text(
        "config_path: /tmp/qwen3\n"
        "ops_implementation:\n"
        "  linear_implementation:\n"
        "    name: torchao\n"
        "    kwargs: {dtype: fp8, recipe: rowwise}\n"
        "  gated_mlp_implementation:\n"
        "    name: torchao\n"
        "    kwargs: {dtype: fp8, recipe: rowwise}\n"
    )
    monkeypatch.setattr("sys.argv", ["test", str(path)])
    config = parse_args(ModelArguments)
    lin = config.ops_implementation.linear_implementation
    mlp = config.ops_implementation.gated_mlp_implementation
    assert isinstance(lin, OpsSelection)
    assert isinstance(mlp, OpsSelection)
    assert lin.resolve("layers.0.q_proj") == ("torchao", {"dtype": "fp8", "recipe": "rowwise"})
    assert mlp.resolve("layers.0.mlp") == ("torchao", {"dtype": "fp8", "recipe": "rowwise"})

    save_args(config, str(tmp_path / "saved"))
    monkeypatch.setattr("sys.argv", ["test", str(tmp_path / "saved" / "veomni_cli.yaml")])
    assert asdict(parse_args(ModelArguments).ops_implementation) == asdict(config.ops_implementation)


@cpu
def test_float8_rejects_invalid_recipe_shape_and_hardware_before_torchao_import(monkeypatch):
    """Invalid recipe, non-16-divisible shapes, and non-SM89 hardware fail before importing torchao."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("torchao")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    x = torch.randn(2, 32)
    w = torch.randn(32, 32)

    with pytest.raises(ValueError, match="float8 recipe"):
        torchao_fp8_linear(x, w, recipe="invalid")

    with pytest.raises(ValueError, match="divisible by 16"):
        torchao_fp8_linear(torch.randn(2, 24), torch.randn(16, 24))
    with pytest.raises(ValueError, match="divisible by 16"):
        torchao_fp8_linear(torch.randn(2, 32), torch.randn(8, 32))

    # DTensor operands are refused before importing torchao.
    from torch.distributed.tensor import DTensor

    class _FakeDTensor(DTensor):
        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.zeros(2, 32), require_grad=False)

    with pytest.raises(TypeError, match="DTensor"):
        torchao_fp8_linear(_FakeDTensor(), w)
    with pytest.raises(TypeError, match="DTensor"):
        torchao_fp8_linear(x, _FakeDTensor())

    # Native FP8 requires NVIDIA CUDA SM89+.
    monkeypatch.setattr(device, "IS_CUDA_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="SM89"):
        torchao_fp8_linear(x, w, emulated=False)

    monkeypatch.setattr(device, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(device, "get_gpu_compute_capability", lambda _dev=None: 80)
    with pytest.raises(RuntimeError, match="SM89"):
        torchao_fp8_linear(x, w, emulated=False)

    monkeypatch.setattr(device, "get_gpu_compute_capability", lambda _dev=None: 90)
    monkeypatch.setattr(torch.version, "hip", "6.0")
    with pytest.raises(RuntimeError, match="SM89"):
        torchao_fp8_linear(x, w, emulated=False)


@cpu
@pytest.mark.parametrize(
    ("installed", "message"),
    [
        (None, r"optional extra: uv sync --extra gpu --extra float8 --dev"),
        ("0.17.0", r"requires torchao==0\.18\.0; found 0\.17\.0"),
    ],
)
def test_float8_reports_missing_or_mismatched_torchao_extra(monkeypatch, installed, message):
    """Missing or mismatched ``torchao`` raises a clear ``ImportError`` naming the extra."""
    from importlib.metadata import PackageNotFoundError

    from veomni.ops import torchao as torchao_ops

    torchao_ops._float8_linear_configs.cache_clear()

    def fake_version(name):
        assert name == "torchao"
        if installed is None:
            raise PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(torchao_ops, "version", fake_version)
    with pytest.raises(ImportError, match=message):
        torchao_fp8_linear(torch.randn(2, 32), torch.randn(32, 32), emulated=True)


@cpu
def test_float8_linear_compiles_fullgraph_without_tracing_metadata_io(monkeypatch):
    """``_float8_linear_configs`` stays outside Dynamo so ``fullgraph=True`` traces cleanly."""
    import sys
    import types
    from dataclasses import dataclass

    import torch.nn.functional as F

    from veomni.ops import torchao as torchao_ops

    @dataclass(frozen=True)
    class _FakeGemmCfg:
        use_fast_accum: bool = False

    @dataclass(frozen=True)
    class _FakeFloat8Cfg:
        recipe: str
        pad_inner_dim: bool = False
        emulate: bool = False
        enable_fsdp_float8_all_gather: bool = False
        gemm_config_output: _FakeGemmCfg = _FakeGemmCfg()
        gemm_config_grad_input: _FakeGemmCfg = _FakeGemmCfg()
        gemm_config_grad_weight: _FakeGemmCfg = _FakeGemmCfg()

        @classmethod
        def from_recipe_name(cls, name: str):
            return cls(recipe=name)

    @dataclass(frozen=True)
    class _FakeScaledMMCfg:
        emulate: bool
        use_fast_accum: bool
        fp8_output: bool
        pad_inner_dim: bool

    @dataclass(frozen=True)
    class _FakeLinearMMCfg:
        output: _FakeScaledMMCfg
        grad_input: _FakeScaledMMCfg
        grad_weight: _FakeScaledMMCfg

    class _FakeMatmul(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w_t, _mm_cfg, _cfg):
            return F.linear(x, w_t.t())

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None, None, None

    fake_torchao = types.ModuleType("torchao")
    fake_float8 = types.ModuleType("torchao.float8")
    fake_cfg_mod = types.ModuleType("torchao.float8.config")
    fake_cfg_mod.Float8LinearConfig = _FakeFloat8Cfg
    fake_tensor_mod = types.ModuleType("torchao.float8.float8_training_tensor")
    fake_tensor_mod.LinearMMConfig = _FakeLinearMMCfg
    fake_tensor_mod.ScaledMMConfig = _FakeScaledMMCfg
    fake_linear_mod = types.ModuleType("torchao.float8.float8_linear")
    fake_linear_mod.matmul_with_hp_or_float8_args = _FakeMatmul
    fake_torchao.float8 = fake_float8
    fake_float8.config = fake_cfg_mod
    fake_float8.float8_training_tensor = fake_tensor_mod
    fake_float8.float8_linear = fake_linear_mod

    for name, mod in (
        ("torchao", fake_torchao),
        ("torchao.float8", fake_float8),
        ("torchao.float8.config", fake_cfg_mod),
        ("torchao.float8.float8_training_tensor", fake_tensor_mod),
        ("torchao.float8.float8_linear", fake_linear_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(torchao_ops, "version", lambda _pkg: torchao_ops.TORCHAO_REQUIRED_VERSION)
    torchao_ops._float8_linear_configs.cache_clear()

    compiled = torch.compile(lambda x, w: torchao_fp8_linear(x, w, recipe="rowwise", emulated=True), fullgraph=True)
    x = torch.randn(2, 16)
    w = torch.randn(16, 16)
    try:
        torch.testing.assert_close(compiled(x, w), F.linear(x, w))
    finally:
        torchao_ops._float8_linear_configs.cache_clear()


@cpu
def test_qwen3_and_llama_bind_float8_handles(monkeypatch):
    """Qwen3 and Llama bind torchao float8 handles while keeping every projection a plain ``nn.Linear``."""
    from transformers import LlamaConfig, Qwen3Config

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.transformers.llama.modeling_llama import LlamaForCausalLM
    from veomni.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from veomni.ops import registry as ops_registry

    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr(ops_registry, "is_package_available", lambda _pkg: True)
    monkeypatch.setattr(VeomniOp, "_intern", dict(VeomniOp._intern))

    ops = make_eager_ops_config()
    ops.linear_implementation = _fp8_selection("rowwise")
    ops.gated_mlp_implementation = _fp8_selection("tensorwise")

    for cfg_cls, model_cls in ((Qwen3Config, Qwen3ForCausalLM), (LlamaConfig, LlamaForCausalLM)):
        cfg = _attach_ops(
            cfg_cls(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                attn_implementation="eager",
            ),
            ops,
        )
        assert "torchao" in cfg.to_json_string()
        model = model_cls(cfg)
        l0 = model.model.layers[0]

        assert l0.self_attn.veomni_q_proj.impl == "torchao"
        assert l0.self_attn.veomni_q_proj.kwargs == {"dtype": "fp8", "recipe": "rowwise"}
        assert l0.mlp.veomni_gated_mlp.impl == "torchao"
        assert l0.mlp.veomni_gated_mlp.kwargs == {"dtype": "fp8", "recipe": "tensorwise"}
        assert type(l0.self_attn.q_proj) is nn.Linear
        assert type(l0.mlp.gate_proj) is nn.Linear
        assert type(model.lm_head) is nn.Linear


@cpu
def test_dpo_reference_builds_cleanly_without_trainer_float8_patch(monkeypatch):
    """Frozen DPO reference disables mixed precision without needing any float8 trainer patch."""
    from tests.tools.training_utils import make_eager_ops_config
    from veomni.trainer import text_dpo_trainer as dpo

    class _DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3")
            self.proj = nn.Linear(32, 32)

    ops = make_eager_ops_config()
    ops.linear_implementation = _fp8_selection(recipe="rowwise")
    ops.gated_mlp_implementation = _fp8_selection(recipe="rowwise")
    args = ModelArguments(config_path="/tmp/qwen3", ops_implementation=ops)
    monkeypatch.setattr(dpo.DPOReferenceModelRuntime, "setup", lambda self: None)
    monkeypatch.setattr(dpo, "use_parallel_state", lambda _: nullcontext())
    monkeypatch.setattr("veomni.models.auto.build_foundation_model", lambda **kwargs: _DummyModel())
    monkeypatch.setattr(dpo.DPOReferenceModelRuntime, "_build_parallelized_model", lambda self: None)

    reference = dpo.DPOReferenceModelRuntime(args, train=None)
    assert not reference.args.accelerator.fsdp_config.mixed_precision.enable
    assert not hasattr(reference.args.accelerator.fsdp_config.mixed_precision, "float8")


def _require_native_float8():
    """Skip if the host lacks an NVIDIA SM89+ CUDA device or ``torchao.float8``."""
    if not device.IS_CUDA_AVAILABLE or torch.version.hip or device.get_gpu_compute_capability() < 89:
        pytest.skip("native float8 training requires NVIDIA SM89+")
    pytest.importorskip("torchao.float8")


@pytest.mark.parametrize(
    ("recipe", "expected_linear_mms"),
    [
        ("tensorwise", 3),
        ("rowwise", 3),
        ("rowwise_with_gw_hp", 2),
    ],
)
def test_float8_linear_and_gated_mlp_dispatch_scaled_mm(recipe, expected_linear_mms):
    """Verify ``linear`` and ``gated_mlp`` ``torchao`` FP8 rows dispatch ``aten::_scaled_mm`` and match BF16."""
    _require_native_float8()
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    linear_op = VeomniOp("linear", "standard", _fp8_selection(recipe=recipe))
    ref_linear_op = VeomniOp("linear", "standard", "eager")
    for tokens in (16, 17, 33):
        linear = nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16)
        ref_linear = nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16)
        ref_linear.load_state_dict(linear.state_dict())
        x = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            out = linear_op(linear, x)
            out.square().mean().backward()
            device.synchronize()
        assert sum(e.count for e in trace.key_averages() if e.key == "aten::_scaled_mm") == expected_linear_mms

        ref_out = ref_linear_op(ref_linear, x_ref)
        ref_out.square().mean().backward()
        for actual, expected in (
            (out, ref_out),
            (x.grad, x_ref.grad),
            (linear.weight.grad, ref_linear.weight.grad),
        ):
            assert torch.isfinite(actual).all()
            rel_err = (actual.float() - expected.float()).norm() / expected.float().norm().clamp(min=1e-6)
            assert rel_err < 0.15

    mlp = SimpleNamespace(
        gate_proj=nn.Linear(64, 128, bias=False, device="cuda", dtype=torch.bfloat16),
        up_proj=nn.Linear(64, 128, bias=False, device="cuda", dtype=torch.bfloat16),
        down_proj=nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    ref_mlp = SimpleNamespace(
        gate_proj=nn.Linear(64, 128, bias=False, device="cuda", dtype=torch.bfloat16),
        up_proj=nn.Linear(64, 128, bias=False, device="cuda", dtype=torch.bfloat16),
        down_proj=nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    for attr in ("gate_proj", "up_proj", "down_proj"):
        getattr(ref_mlp, attr).load_state_dict(getattr(mlp, attr).state_dict())
    mlp_op = VeomniOp("gated_mlp", "standard", _fp8_selection(recipe=recipe))
    ref_mlp_op = VeomniOp("gated_mlp", "standard", "eager")
    x_mlp = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_mlp_ref = x_mlp.detach().clone().requires_grad_(True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        mlp_out = mlp_op(mlp, x_mlp, activation="silu")
        mlp_out.square().mean().backward()
        device.synchronize()
    assert sum(e.count for e in trace.key_averages() if e.key == "aten::_scaled_mm") == 3 * expected_linear_mms
    ref_mlp_out = ref_mlp_op(ref_mlp, x_mlp_ref, activation="silu")
    ref_mlp_out.square().mean().backward()
    assert torch.isfinite(mlp_out).all() and torch.isfinite(x_mlp.grad).all()
    rel_err = (mlp_out.float() - ref_mlp_out.float()).norm() / ref_mlp_out.float().norm().clamp(min=1e-6)
    assert rel_err < 0.20


@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise", "rowwise_with_gw_hp"])
def test_float8_qwen3_fsdp2_train_dcp_resume_and_hf_export(tmp_path, recipe):
    """Verify Qwen3 FSDP2 float8 training via ``ops_config``, DCP resume, and HF export."""
    _require_native_float8()
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from transformers import Qwen3Config
    from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.checkpoint.dcp_checkpointer import ModelState, OptimizerState
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2
    from veomni.models.checkpoint.weights import init_empty_weights, save_model_weights
    from veomni.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
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
    ops = make_eager_ops_config()
    ops.linear_implementation = _fp8_selection(recipe=recipe)
    ops.gated_mlp_implementation = _fp8_selection(recipe=recipe)
    config = _attach_ops(config, ops)

    ckpt = tmp_path / "hf"
    Qwen3ForCausalLM(config).float().save_pretrained(ckpt)

    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        init_method=f"file://{tmp_path / 'rdzv'}",
        rank=0,
        world_size=1,
    )
    try:
        _init_parallel_state(dp_size=1, dp_mode="fsdp2")
        precision = MixedPrecisionConfig(enable=True, param_dtype="bfloat16", reduce_dtype="float32")

        def build(resume=False):
            with init_empty_weights():
                mod = Qwen3ForCausalLM(config).float()
            return parallelize_model_fsdp2(
                mod,
                weights_path=None if resume else str(ckpt),
                init_device="meta",
                mixed_precision=precision,
                broadcast_model_weights_from_rank0=False,
                should_skip_hf_weight_load=resume,
            )

        model = build()
        assert type(model.model.layers[0].mlp.gate_proj) is nn.Linear
        assert model.model.layers[0].mlp.veomni_gated_mlp.impl == "torchao"
        assert model.model.layers[0].self_attn.veomni_q_proj.impl == "torchao"
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


def _two_rank_fsdp2_worker(rank: int, checkpoint_str: str, dcp_str: str, recipe: str, endpoint):
    """Execute 2-rank FSDP2 sharded Llama float8 training and DCP resume via ``ops_config``."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from transformers import LlamaConfig

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.checkpoint.dcp_checkpointer import ModelState, OptimizerState
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2
    from veomni.models.checkpoint.weights import init_empty_weights
    from veomni.models.transformers.llama.modeling_llama import LlamaForCausalLM

    device.get_torch_device().set_device(rank)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", store=endpoint.connect(), rank=rank, world_size=2)
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
        ops = make_eager_ops_config()
        ops.linear_implementation = _fp8_selection(recipe=recipe)
        ops.gated_mlp_implementation = _fp8_selection(recipe=recipe)
        config = _attach_ops(config, ops)
        precision = MixedPrecisionConfig(enable=True, param_dtype="bfloat16", reduce_dtype="float32")

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
        assert type(model.model.layers[0].mlp.up_proj) is nn.Linear
        assert model.model.layers[0].mlp.veomni_gated_mlp.impl == "torchao"
        assert model.model.layers[0].self_attn.veomni_q_proj.impl == "torchao"
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
    from veomni.models.transformers.llama.modeling_llama import LlamaForCausalLM

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
        _two_rank_fsdp2_worker,
        nprocs=2,
        args=(str(ckpt), str(tmp_path / f"dcp_{recipe}"), recipe),
    )


@cpu
def test_ops_implementation_config_accepts_float8_row_kwargs(monkeypatch):
    """``OpsImplementationConfig`` accepts ``dtype``, ``recipe``, and ``emulated`` from ``torchao``'s signature."""
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    ops = OpsImplementationConfig(
        linear_implementation={"name": "torchao", "kwargs": {"dtype": "fp8", "recipe": "rowwise", "emulated": False}},
        gated_mlp_implementation={"name": "torchao", "kwargs": {"dtype": "fp8", "recipe": "rowwise_with_gw_hp"}},
    )
    assert ops.linear_implementation.kwargs == {"dtype": "fp8", "recipe": "rowwise", "emulated": False}
    assert ops.gated_mlp_implementation.kwargs == {"dtype": "fp8", "recipe": "rowwise_with_gw_hp"}
