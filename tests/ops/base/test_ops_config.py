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

"""CPU tests for installed ops-config resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from veomni.ops.config import get_ops_config, resolve_op_impl, set_ops_config


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("initially_configured", [False, True], ids=["none", "existing-config"])
@pytest.mark.parametrize("fail", [False, True], ids=["normal-exit", "exception"])
def test_ops_config_scope_restores_nested_bindings(initially_configured, fail):
    previous = get_ops_config()
    initial = eager_ops_config() if initially_configured else None
    inner = eager_ops_config()
    inner.cross_entropy_loss_implementation = "chunk_loss"
    with ops_config_scope(initial):
        assert get_ops_config() is initial

        def nested_scope():
            with ops_config_scope(inner):
                assert get_ops_config() is inner
                with ops_config_scope(None):
                    assert get_ops_config() is None
                assert get_ops_config() is inner
                # The scoped code may itself install another config (as model builders do).
                set_ops_config(eager_ops_config())
                if fail:
                    raise RuntimeError("construction failed")

        if fail:
            with pytest.raises(RuntimeError, match="construction failed"):
                nested_scope()
        else:
            nested_scope()
        assert get_ops_config() is initial
    assert get_ops_config() is previous


def test_resolve_op_impl_defaults_to_eager():
    set_ops_config(None)
    assert resolve_op_impl("rms_norm_implementation") == "eager"


def test_resolve_op_impl_reads_ops_config():
    set_ops_config(SimpleNamespace(cross_entropy_loss_implementation="chunk_loss"))
    assert resolve_op_impl("cross_entropy_loss_implementation") == "chunk_loss"


def test_resolve_op_impl_returns_none_for_unset_field_and_veomni_op_resolves_variant_default(monkeypatch):
    from veomni.arguments import OpsImplementationConfig
    from veomni.ops import VeomniOp

    monkeypatch.setattr(VeomniOp, "_intern", {})
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr("veomni.ops.registry.get_device_type", lambda: "npu")
    monkeypatch.setattr("veomni.ops.platform.requirement.IS_NPU_AVAILABLE", True)

    cfg = OpsImplementationConfig()
    with ops_config_scope(cfg):
        assert resolve_op_impl("rms_norm_implementation") is None
        assert VeomniOp("rms_norm", "standard", resolve_op_impl("rms_norm_implementation")).impl == "torch_npu"
        assert VeomniOp("rms_norm", "unweighted", resolve_op_impl("rms_norm_implementation")).impl == "eager"


@pytest.mark.parametrize("impl", [None, "torch_npu", "npu", "eager", "liger_kernel"])
def test_resolve_op_impl_preserves_variant_scoped_npu_compatibility(impl):
    with ops_config_scope(SimpleNamespace(rms_norm_implementation=impl)):
        assert resolve_op_impl("rms_norm_implementation") == impl
        assert resolve_op_impl("rms_norm_implementation", variant="standard") == impl
        expected = "eager" if impl in ("torch_npu", "npu") else impl
        resolved = resolve_op_impl("rms_norm_implementation", variant="unweighted")
        assert resolved == expected
        if expected == "eager":
            from veomni.ops import VeomniOp

            assert VeomniOp("rms_norm", "unweighted", resolved).impl == "eager"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("parse_config", [False, True], ids=["dataclass", "parser"])
def test_deepgemm_rejects_qat_before_hardware_validation(monkeypatch, parse_config):
    from veomni.arguments import OpsImplementationConfig
    from veomni.arguments.parser import _instantiate_recursive

    # An invalid combination must report the config conflict even without CUDA.
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", False)
    values = {"moe_implementation": "deepgemm_fp8", "qat_implementation": "fp8_blockwise"}
    with pytest.raises(ValueError, match="deepgemm_fp8.*cannot be combined.*fp8_blockwise") as error:
        if parse_config:
            _instantiate_recursive(OpsImplementationConfig, values)
        else:
            OpsImplementationConfig(**values)
    assert "Set qat_implementation='none'" in str(error.value)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    "values",
    [
        {"moe_implementation": "deepgemm_fp8"},
        {"moe_implementation": "deepgemm_fp8", "qat_implementation": "none"},
        {"moe_implementation": None, "qat_implementation": "fp8_blockwise"},
        {"moe_implementation": "eager", "qat_implementation": "fp8_blockwise"},
        {"moe_implementation": "fused_triton", "qat_implementation": "fp8_blockwise"},
        {"moe_implementation": "fused_quack", "qat_implementation": "fp8_blockwise"},
    ],
)
def test_deepgemm_qat_guard_preserves_valid_selections(monkeypatch, values):
    from veomni.arguments import OpsImplementationConfig

    # Model a supported QAT host without initializing an accelerator context.
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.device.get_gpu_compute_capability", lambda _device: 90)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("torch.version.hip", None)

    config = OpsImplementationConfig(**values)
    assert config.moe_implementation == values["moe_implementation"]
    assert config.qat_implementation == values.get("qat_implementation", "none")


def test_upstream_llama_config_ops_config_save_and_reload(tmp_path):
    """Upstream PretrainedConfig subclasses support ops_implementation init, save, and reload."""
    import json

    from transformers import AutoConfig
    from transformers.models.llama.configuration_llama import LlamaConfig

    import veomni.models  # noqa: F401 - ensures PretrainedConfig patch is applied

    bare = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4)
    assert bare.ops_implementation == {}
    assert "ops_implementation" not in bare.to_diff_dict()
    assert bare.ops_config.attn_implementation == "veomni_flash_attention_2"
    assert bare.ops_config.rms_norm_implementation is None

    configured = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        ops_implementation={"rms_norm_implementation": "eager", "attn_implementation": "sdpa"},
    )
    assert configured.ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "veomni_sdpa",
    }
    assert configured.ops_config.rms_norm_implementation == "eager"
    assert configured.ops_config.attn_implementation == "veomni_sdpa"
    assert configured._attn_implementation == "veomni_sdpa"

    # Serialization must not fail on dataclass attributes and only writes explicit sparse keys.
    json_payload = json.loads(configured.to_json_string())
    assert json_payload["ops_implementation"] == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "veomni_sdpa",
    }

    configured.save_pretrained(tmp_path)
    reloaded = AutoConfig.from_pretrained(tmp_path)
    assert reloaded.ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "veomni_sdpa",
    }
    assert reloaded.ops_config.rms_norm_implementation == "eager"
    assert reloaded.ops_config.attn_implementation == "veomni_sdpa"
    assert reloaded._attn_implementation == "veomni_sdpa"

    overridden = LlamaConfig.from_pretrained(
        tmp_path,
        ops_implementation={"swiglu_mlp_implementation": "eager"},
    )
    assert overridden.ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "veomni_sdpa",
        "swiglu_mlp_implementation": "eager",
    }


def test_pretrained_config_attn_implementation_bidirectional_sync():
    """config._attn_implementation and config.ops_config.attn_implementation stay synchronized."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM

    import veomni.models  # noqa: F401

    # 1. Passing attn_implementation kwarg reflects in ops_config.
    cfg_hf_attn = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        attn_implementation="eager",
    )
    assert cfg_hf_attn._attn_implementation == "eager"
    assert cfg_hf_attn.ops_config.attn_implementation == "eager"

    # 2. When both attn_implementation and ops_implementation["attn_implementation"] are passed at init,
    # ops_implementation wins and updates _attn_implementation.
    cfg_both = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        attn_implementation="eager",
        ops_implementation={"attn_implementation": "flex_attention"},
    )
    assert cfg_both._attn_implementation == "veomni_flex_attention"
    assert cfg_both.ops_config.attn_implementation == "veomni_flex_attention"

    # 3. Updating _attn_implementation later (as _from_config(..., attn_implementation=...) does)
    # updates ops_implementation["attn_implementation"] and ops_config.attn_implementation.
    cfg_both._attn_implementation = "eager"
    assert cfg_both._attn_implementation == "eager"
    assert cfg_both.ops_implementation["attn_implementation"] == "eager"
    assert cfg_both.ops_config.attn_implementation == "eager"

    # 4. Bare config passed into PreTrainedModel.__init__ resolves None -> "sdpa" on both attributes.
    cfg_bare = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
    )
    HFLlamaForCausalLM(cfg_bare)
    assert cfg_bare._attn_implementation == "sdpa"
    assert cfg_bare.ops_config.attn_implementation == "sdpa"


def test_composite_qwen3_vl_config_subconfig_merge_isolation_and_save(tmp_path):
    """Nested sub_configs merge parent defaults under child overrides without sharing mutable state."""
    import json

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    import veomni.models  # noqa: F401

    cfg = Qwen3VLConfig(
        ops_implementation={"rms_norm_implementation": "eager", "attn_implementation": "sdpa"},
        vision_config={"ops_implementation": {"attn_implementation": "eager"}},
    )

    assert cfg.ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "veomni_sdpa",
    }
    assert cfg.text_config.ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "veomni_sdpa",
    }
    assert cfg.vision_config.ops_implementation == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "eager",
    }
    assert cfg.text_config._attn_implementation == "veomni_sdpa"
    assert cfg.vision_config._attn_implementation == "eager"
    assert cfg.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert cfg.vision_config.ops_config.attn_implementation == "eager"

    # Sub-configs must own distinct dicts so mutating one does not leak to siblings or root.
    assert cfg.text_config.ops_implementation is not cfg.ops_implementation
    assert cfg.vision_config.ops_implementation is not cfg.ops_implementation
    assert cfg.text_config.ops_implementation is not cfg.vision_config.ops_implementation

    # Saving writes only the child diff relative to parent and round-trips cleanly.
    cfg.save_pretrained(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert raw["ops_implementation"] == {
        "rms_norm_implementation": "eager",
        "attn_implementation": "veomni_sdpa",
    }
    assert "ops_implementation" not in raw["text_config"]
    assert raw["vision_config"]["ops_implementation"] == {"attn_implementation": "eager"}

    reloaded = Qwen3VLConfig.from_pretrained(tmp_path)
    assert reloaded.text_config.ops_config.rms_norm_implementation == "eager"
    assert reloaded.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert reloaded.vision_config.ops_config.rms_norm_implementation == "eager"
    assert reloaded.vision_config.ops_config.attn_implementation == "eager"
