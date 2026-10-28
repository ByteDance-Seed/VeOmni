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

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_resolve_impl_returns_field_and_remaps_eager_as():
    from veomni.arguments import OpsImplementationConfig

    cfg = OpsImplementationConfig.from_dict(
        {"attn_implementation": "eager", "cross_entropy_loss_implementation": "chunk_loss"},
        validate=False,
    )
    assert cfg.resolve_impl("cross_entropy_loss_implementation") == "chunk_loss"
    assert cfg.resolve_impl("attn_implementation") == "eager"
    assert cfg.resolve_impl("attn_implementation", eager_as="veomni_sdpa") == "veomni_sdpa"


def test_resolve_impl_returns_none_for_unset_field_and_veomni_op_resolves_variant_default(monkeypatch):
    from veomni.arguments import OpsImplementationConfig
    from veomni.ops import VeomniOp

    monkeypatch.setattr(VeomniOp, "_intern", {})
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr("veomni.ops.registry.get_device_type", lambda: "npu")
    monkeypatch.setattr("veomni.ops.platform.requirement.IS_NPU_AVAILABLE", True)

    cfg = OpsImplementationConfig()
    assert cfg.resolve_impl("rms_norm_implementation") is None
    assert VeomniOp("rms_norm", "standard", cfg.resolve_impl("rms_norm_implementation")).impl == "torch_npu"
    assert (
        VeomniOp("rms_norm", "unweighted", cfg.resolve_impl("rms_norm_implementation", variant="unweighted")).impl
        == "eager"
    )


@pytest.mark.parametrize("impl", [None, "torch_npu", "npu", "eager", "liger_kernel"])
def test_resolve_impl_preserves_variant_scoped_npu_compatibility(impl):
    from veomni.arguments import OpsImplementationConfig

    cfg = OpsImplementationConfig.from_dict(
        {"rms_norm_implementation": impl} if impl is not None else {},
        validate=False,
    )
    assert cfg.resolve_impl("rms_norm_implementation") == impl
    assert cfg.resolve_impl("rms_norm_implementation", variant="standard") == impl
    expected = "eager" if impl in ("torch_npu", "npu") else impl
    resolved = cfg.resolve_impl("rms_norm_implementation", variant="unweighted")
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


def test_three_level_nested_config_preserves_grandchild_overrides(tmp_path, monkeypatch):
    """Three-level configs (root -> thinker_config -> text_config) keep parent < child < grandchild priority."""
    import json

    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    import veomni.models  # noqa: F401

    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)

    cfg = Qwen3OmniMoeConfig(
        ops_implementation={"rms_norm_implementation": "eager", "attn_implementation": "sdpa"},
        thinker_config={
            "text_config": {
                "ops_implementation": {
                    "rms_norm_implementation": "liger_kernel",
                    "attn_implementation": "eager",
                }
            }
        },
    )

    assert cfg.ops_config.rms_norm_implementation == "eager"
    assert cfg.ops_config.attn_implementation == "veomni_sdpa"
    assert cfg.thinker_config.ops_config.rms_norm_implementation == "eager"
    assert cfg.thinker_config.ops_config.attn_implementation == "veomni_sdpa"
    assert cfg.thinker_config.vision_config.ops_config.rms_norm_implementation == "eager"
    assert cfg.thinker_config.vision_config.ops_config.attn_implementation == "veomni_sdpa"
    assert cfg.thinker_config.text_config.ops_config.rms_norm_implementation == "liger_kernel"
    assert cfg.thinker_config.text_config.ops_config.attn_implementation == "eager"
    assert cfg.thinker_config.text_config._attn_implementation == "eager"

    cfg.save_pretrained(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "ops_implementation" not in raw["thinker_config"]
    assert "ops_implementation" not in raw["thinker_config"]["vision_config"]
    assert raw["thinker_config"]["text_config"]["ops_implementation"] == {
        "rms_norm_implementation": "liger_kernel",
        "attn_implementation": "eager",
    }

    reloaded = Qwen3OmniMoeConfig.from_pretrained(tmp_path)
    assert reloaded.thinker_config.vision_config.ops_config.rms_norm_implementation == "eager"
    assert reloaded.thinker_config.vision_config.ops_config.attn_implementation == "veomni_sdpa"
    assert reloaded.thinker_config.text_config.ops_config.rms_norm_implementation == "liger_kernel"
    assert reloaded.thinker_config.text_config.ops_config.attn_implementation == "eager"
    assert reloaded.thinker_config.text_config._attn_implementation == "eager"


@pytest.mark.parametrize("sub_as_instance", [False, True], ids=["dict", "config_instance"])
@pytest.mark.parametrize("parent_via_hf_attn", [False, True], ids=["parent_ops_impl", "parent_hf_attn"])
def test_subconfig_hf_attn_implementation_merges_and_roundtrips(tmp_path, sub_as_instance, parent_via_hf_attn):
    """A child sub-config specifying HF attn_implementation updates both _attn_implementation and ops_config."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig

    import veomni.models  # noqa: F401

    vision_arg = (
        Qwen3VLVisionConfig(attn_implementation="eager") if sub_as_instance else {"attn_implementation": "eager"}
    )
    parent_kwargs = (
        {"attn_implementation": "sdpa", "ops_implementation": {"rms_norm_implementation": "eager"}}
        if parent_via_hf_attn
        else {"ops_implementation": {"attn_implementation": "sdpa", "rms_norm_implementation": "eager"}}
    )
    cfg = Qwen3VLConfig(vision_config=vision_arg, **parent_kwargs)

    assert cfg.text_config._attn_implementation == "veomni_sdpa"
    assert cfg.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert cfg.vision_config._attn_implementation == "eager"
    assert cfg.vision_config.ops_config.attn_implementation == "eager"
    assert cfg.vision_config.ops_implementation["attn_implementation"] == "eager"

    save_dir = tmp_path / f"{sub_as_instance}_{parent_via_hf_attn}"
    cfg.save_pretrained(save_dir)
    reloaded = Qwen3VLConfig.from_pretrained(save_dir)
    assert reloaded.text_config._attn_implementation == "veomni_sdpa"
    assert reloaded.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert reloaded.vision_config._attn_implementation == "eager"
    assert reloaded.vision_config.ops_config.attn_implementation == "eager"


def test_nested_attn_implementation_mapping_overrides_grandchild():
    """Nested HF attn_implementation dict on a 3-level root config overrides grandchild's initial selection."""
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    import veomni.models  # noqa: F401

    cfg = Qwen3OmniMoeConfig(
        thinker_config={
            "text_config": {"attn_implementation": "sdpa"},
            "vision_config": {"attn_implementation": "sdpa"},
        },
        attn_implementation={"thinker_config": {"text_config": "eager"}},
    )
    assert cfg.thinker_config.text_config._attn_implementation == "eager"
    assert cfg.thinker_config.text_config.ops_config.attn_implementation == "eager"
    assert cfg.thinker_config.text_config.ops_implementation["attn_implementation"] == "eager"
    assert cfg.thinker_config.vision_config._attn_implementation == "veomni_sdpa"
    assert cfg.thinker_config.vision_config.ops_config.attn_implementation == "veomni_sdpa"

    # A scalar value under an intermediate sub-config key overrides the entire subtree.
    cfg_subtree = Qwen3OmniMoeConfig(
        thinker_config={
            "text_config": {"attn_implementation": "sdpa"},
            "vision_config": {"attn_implementation": "sdpa"},
        },
        attn_implementation={"thinker_config": "eager"},
    )
    assert cfg_subtree.thinker_config._attn_implementation == "eager"
    assert cfg_subtree.thinker_config.ops_config.attn_implementation == "eager"
    assert cfg_subtree.thinker_config.text_config._attn_implementation == "eager"
    assert cfg_subtree.thinker_config.text_config.ops_config.attn_implementation == "eager"
    assert cfg_subtree.thinker_config.vision_config._attn_implementation == "eager"
    assert cfg_subtree.thinker_config.vision_config.ops_config.attn_implementation == "eager"


def test_sparse_load_override_preserves_subconfig_choices(tmp_path, monkeypatch):
    """Sparse ops_implementation overrides on from_pretrained only propagate caller-specified keys."""
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    import veomni.models  # noqa: F401

    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)

    vl_dir = tmp_path / "qwen3_vl"
    vl_cfg = Qwen3VLConfig(
        ops_implementation={"attn_implementation": "sdpa", "rms_norm_implementation": "eager"},
        vision_config={"ops_implementation": {"attn_implementation": "eager"}},
    )
    vl_cfg.save_pretrained(vl_dir)

    # Empty override dict must leave sub-config choices untouched.
    reloaded_empty = Qwen3VLConfig.from_pretrained(vl_dir, ops_implementation={})
    assert reloaded_empty.text_config._attn_implementation == "veomni_sdpa"
    assert reloaded_empty.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert reloaded_empty.vision_config._attn_implementation == "eager"
    assert reloaded_empty.vision_config.ops_config.attn_implementation == "eager"

    # Overriding only rms_norm_implementation must propagate RMSNorm without clobbering vision's attention.
    reloaded_rms = Qwen3VLConfig.from_pretrained(
        vl_dir,
        ops_implementation={"rms_norm_implementation": "liger_kernel"},
    )
    assert reloaded_rms.ops_config.rms_norm_implementation == "liger_kernel"
    assert reloaded_rms.text_config.ops_config.rms_norm_implementation == "liger_kernel"
    assert reloaded_rms.vision_config.ops_config.rms_norm_implementation == "liger_kernel"
    assert reloaded_rms.text_config._attn_implementation == "veomni_sdpa"
    assert reloaded_rms.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert reloaded_rms.vision_config._attn_implementation == "eager"
    assert reloaded_rms.vision_config.ops_config.attn_implementation == "eager"

    # Three-level config (root -> thinker_config -> text_config) preserves grandchild attention under sparse override.
    omni_dir = tmp_path / "qwen3_omni"
    omni_cfg = Qwen3OmniMoeConfig(
        ops_implementation={"attn_implementation": "sdpa", "rms_norm_implementation": "eager"},
        thinker_config={"text_config": {"ops_implementation": {"attn_implementation": "eager"}}},
    )
    omni_cfg.save_pretrained(omni_dir)

    omni_reloaded = Qwen3OmniMoeConfig.from_pretrained(
        omni_dir,
        ops_implementation={"rms_norm_implementation": "liger_kernel"},
    )
    assert omni_reloaded.thinker_config.vision_config.ops_config.attn_implementation == "veomni_sdpa"
    assert omni_reloaded.thinker_config.vision_config.ops_config.rms_norm_implementation == "liger_kernel"
    assert omni_reloaded.thinker_config.text_config._attn_implementation == "eager"
    assert omni_reloaded.thinker_config.text_config.ops_config.attn_implementation == "eager"
    assert omni_reloaded.thinker_config.text_config.ops_config.rms_norm_implementation == "liger_kernel"


def test_sparse_override_enforces_cross_field_validation(tmp_path, monkeypatch):
    """Sparse overrides on existing or loaded configs validate the merged state at every node."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    import veomni.models  # noqa: F401

    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.device.get_gpu_compute_capability", lambda _device: 90)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("torch.version.hip", None)

    llama_dir = tmp_path / "llama"
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        ops_implementation={"moe_implementation": "deepgemm_fp8"},
    )
    cfg.save_pretrained(llama_dir)

    with pytest.raises(ValueError, match="deepgemm_fp8.*cannot be combined.*fp8_blockwise"):
        cfg._set_ops_implementation({"qat_implementation": "fp8_blockwise"})
    assert cfg.ops_implementation == {"moe_implementation": "deepgemm_fp8"}

    with pytest.raises(ValueError, match="deepgemm_fp8.*cannot be combined.*fp8_blockwise"):
        LlamaConfig.from_pretrained(llama_dir, ops_implementation={"qat_implementation": "fp8_blockwise"})

    # Overriding moe_implementation away from deepgemm_fp8 at the same time is valid.
    reloaded = LlamaConfig.from_pretrained(
        llama_dir,
        ops_implementation={"moe_implementation": "fused_triton", "qat_implementation": "fp8_blockwise"},
    )
    assert reloaded.ops_implementation == {
        "moe_implementation": "fused_triton",
        "qat_implementation": "fp8_blockwise",
    }

    # Sub-config cross-field conflicts must also be rejected atomically.
    vl_dir = tmp_path / "qwen3_vl"
    vl_cfg = Qwen3VLConfig(
        text_config={"ops_implementation": {"moe_implementation": "deepgemm_fp8"}},
    )
    vl_cfg.save_pretrained(vl_dir)

    with pytest.raises(ValueError, match="deepgemm_fp8.*cannot be combined.*fp8_blockwise"):
        vl_cfg._set_ops_implementation({"qat_implementation": "fp8_blockwise"})
    assert vl_cfg.ops_implementation == {}
    assert vl_cfg.text_config.ops_implementation == {"moe_implementation": "deepgemm_fp8"}

    with pytest.raises(ValueError, match="deepgemm_fp8.*cannot be combined.*fp8_blockwise"):
        Qwen3VLConfig.from_pretrained(vl_dir, ops_implementation={"qat_implementation": "fp8_blockwise"})


def test_cross_device_load_applies_overrides_before_hardware_validation(tmp_path, monkeypatch):
    """Loading a GPU checkpoint on NPU succeeds when caller overrides GPU-only ops to NPU-supported ones."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    import veomni.models  # noqa: F401
    from veomni.models.seed_omni.configuration_omni import OmniConfig
    from veomni.models.seed_omni.modeling_omni import OmniModel
    from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
    from veomni.models.seed_omni.modules.fake_model.fake_module_a.modeling import FakeModuleA
    from veomni.models.seed_omni.modules.module_configuration_base import OmniModuleConfig

    # 1. Export configs on simulated GPU host with GPU-only liger_kernel.
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)

    llama_dir = tmp_path / "llama_gpu"
    LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        ops_implementation={"rms_norm_implementation": "liger_kernel"},
    ).save_pretrained(llama_dir)

    vl_dir = tmp_path / "vl_gpu"
    Qwen3VLConfig(
        ops_implementation={"rms_norm_implementation": "liger_kernel", "attn_implementation": "sdpa"},
        vision_config={"ops_implementation": {"attn_implementation": "eager"}},
    ).save_pretrained(vl_dir)

    module_dir = tmp_path / "omni_module_gpu"
    FakeModuleAConfig(
        hidden_size=8,
        ops_implementation={"rms_norm_implementation": "liger_kernel"},
    ).save_pretrained(module_dir)

    omni_dir = tmp_path / "omni_composed_gpu"
    omni_cfg = OmniConfig(
        _module_entries={"fake_module_a": {"model_path": "fake_module_a"}},
        training_graphs={"default": [{"from": "fake_module_a", "to": "end"}]},
        generation_graphs={},
    )
    OmniModel(
        omni_cfg,
        {
            "fake_module_a": FakeModuleA(
                FakeModuleAConfig(hidden_size=8, ops_implementation={"rms_norm_implementation": "liger_kernel"})
            )
        },
    ).save_pretrained(omni_dir, save_module_weights=False)

    # 2. Switch host to simulated Ascend NPU.
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", False)

    # Without an override, loading the GPU checkpoint on NPU must still fail validation.
    with pytest.raises(ValueError, match="rms_norm_implementation='liger_kernel'"):
        LlamaConfig.from_pretrained(llama_dir)
    with pytest.raises(ValueError, match="rms_norm_implementation='liger_kernel'"):
        Qwen3VLConfig.from_pretrained(vl_dir)
    with pytest.raises(ValueError, match="rms_norm_implementation='liger_kernel'"):
        FakeModuleAConfig.from_pretrained(module_dir)
    with pytest.raises(ValueError, match="rms_norm_implementation='liger_kernel'"):
        OmniModuleConfig.from_pretrained(module_dir)
    with pytest.raises(ValueError, match="rms_norm_implementation='liger_kernel'"):
        OmniConfig.from_pretrained(omni_dir)

    # With a valid NPU override, the override is applied before hardware validation.
    llama_npu = LlamaConfig.from_pretrained(
        llama_dir,
        ops_implementation={"rms_norm_implementation": "torch_npu"},
    )
    assert llama_npu.ops_config.rms_norm_implementation == "torch_npu"

    vl_npu = Qwen3VLConfig.from_pretrained(
        vl_dir,
        ops_implementation={"rms_norm_implementation": "torch_npu"},
    )
    assert vl_npu.ops_config.rms_norm_implementation == "torch_npu"
    assert vl_npu.text_config.ops_config.rms_norm_implementation == "torch_npu"
    assert vl_npu.vision_config.ops_config.rms_norm_implementation == "torch_npu"
    assert vl_npu.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert vl_npu.vision_config.ops_config.attn_implementation == "eager"

    mod_npu = FakeModuleAConfig.from_pretrained(
        module_dir,
        ops_implementation={"rms_norm_implementation": "torch_npu"},
    )
    assert mod_npu.ops_config.rms_norm_implementation == "torch_npu"

    mod_base_npu = OmniModuleConfig.from_pretrained(
        module_dir,
        ops_implementation={"rms_norm_implementation": "torch_npu"},
    )
    assert mod_base_npu.ops_config.rms_norm_implementation == "torch_npu"

    omni_npu = OmniConfig.from_pretrained(
        omni_dir,
        ops_implementation={"rms_norm_implementation": "torch_npu"},
    )
    assert omni_npu._module_configs["fake_module_a"].ops_config.rms_norm_implementation == "torch_npu"

    omni_slot_npu = OmniConfig.from_pretrained(
        omni_dir,
        **{"fake_module_a.rms_norm_implementation": "torch_npu"},
    )
    assert omni_slot_npu._module_configs["fake_module_a"].ops_config.rms_norm_implementation == "torch_npu"
