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

"""CPU tests for ``VeOmniPretrainedConfig`` and ``OpsImplementationConfig`` resolution."""

from __future__ import annotations

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.configuration_base import get_veomni_config_class


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_unset_field_uses_registry_variant_default(monkeypatch):
    from veomni.arguments import OpsImplementationConfig
    from veomni.ops import VeomniOp

    monkeypatch.setattr(VeomniOp, "_intern", {})
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    monkeypatch.setattr("veomni.ops.registry.get_device_type", lambda: "npu")
    monkeypatch.setattr("veomni.ops.platform.requirement.IS_NPU_AVAILABLE", True)

    cfg = OpsImplementationConfig()
    assert cfg.rms_norm_implementation is None
    assert VeomniOp("rms_norm", "standard", cfg.rms_norm_implementation).impl == "torch_npu"
    assert VeomniOp("rms_norm", "unweighted", cfg.rms_norm_implementation).impl == "eager"


@pytest.mark.parametrize("impl", ["torch_npu", "npu"])
def test_unweighted_rms_norm_rejects_unregistered_backend(monkeypatch, impl):
    from veomni.arguments import OpsImplementationConfig
    from veomni.ops import VeomniOp

    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)
    cfg = OpsImplementationConfig(rms_norm_implementation=impl)
    with pytest.raises(KeyError, match="Unknown op 'rms_norm' variant='unweighted'"):
        VeomniOp("rms_norm", "unweighted", cfg.rms_norm_implementation)


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
    """Adapted HF config subclasses support ops_implementation init, save, and reload."""
    import json

    from transformers.models.llama.configuration_llama import LlamaConfig

    from veomni.models import build_config

    LlamaConfig = get_veomni_config_class(LlamaConfig)

    import veomni.models  # noqa: F401

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
    reloaded = build_config(tmp_path)
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

    LlamaConfig = get_veomni_config_class(LlamaConfig)
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

    # 4. Model initialization writes the same canonical ops state.
    cfg_bare = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        attn_implementation="eager",
    )
    HFLlamaForCausalLM(cfg_bare)
    assert cfg_bare._attn_implementation == "eager"
    assert cfg_bare.ops_config.attn_implementation == "eager"


@pytest.mark.parametrize("base_kind", ["model", "omni_module"])
def test_custom_config_adapts_declared_subconfig_schema(tmp_path, base_kind):
    from transformers import PretrainedConfig, Qwen3VLConfig

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.configuration_base import VeOmniPretrainedConfig
    from veomni.models.seed_omni.modules.module_configuration_base import OmniModuleConfig

    native_children = dict(Qwen3VLConfig.sub_configs)
    base = VeOmniPretrainedConfig if base_kind == "model" else OmniModuleConfig

    class CustomConfig(base):
        sub_configs = {"backbone": Qwen3VLConfig}

        def __init__(self, backbone=None, **kwargs):
            self.backbone = self.sub_configs["backbone"](**(backbone or {}))
            super().__init__(**kwargs)

    assert issubclass(CustomConfig.sub_configs["backbone"], VeOmniPretrainedConfig)
    assert Qwen3VLConfig.sub_configs == native_children
    for child in native_children.values():
        assert not issubclass(child, VeOmniPretrainedConfig)
    config = CustomConfig(
        backbone={"vision_config": {"ops_implementation": {"attn_implementation": "eager"}}},
        ops_implementation=make_eager_ops_config(attn_implementation="sdpa"),
    )
    assert isinstance(config.backbone, PretrainedConfig)
    assert config.backbone.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert config.backbone.vision_config.ops_config.attn_implementation == "eager"
    config.save_pretrained(tmp_path)
    restored = CustomConfig.from_pretrained(tmp_path)
    assert restored.backbone.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert restored.backbone.vision_config.ops_config.attn_implementation == "eager"


@pytest.mark.parametrize("family", ["deepseek_v4", "qwen2_5_omni", "qwen3_omni_moe"])
def test_handwritten_hf_config_subtrees_roundtrip(tmp_path, family):
    import pickle

    from transformers import PretrainedConfig

    from tests.models import tiny_configs
    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.configuration_base import VeOmniPretrainedConfig

    config = getattr(tiny_configs, f"tiny_{family}_config")()
    config._set_ops_implementation(make_eager_ops_config())

    def check(node):
        assert isinstance(node, VeOmniPretrainedConfig)
        assert node.ops_config.attn_implementation == "eager"
        for key, child_class in node.sub_configs.items():
            if isinstance(child_class, type) and issubclass(child_class, PretrainedConfig):
                assert issubclass(child_class, VeOmniPretrainedConfig)
            child = getattr(node, key, None)
            if child is not None:
                assert isinstance(child, PretrainedConfig)
                check(child)

    check(config)
    config.save_pretrained(tmp_path)
    restored = type(config).from_pretrained(tmp_path)
    check(restored)
    assert type(pickle.loads(pickle.dumps(restored))) is type(config)


def test_composite_qwen3_vl_config_subconfig_merge_isolation_and_save(tmp_path):
    """Nested sub_configs merge parent defaults under child overrides without sharing mutable state."""
    import json

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    Qwen3VLConfig = get_veomni_config_class(Qwen3VLConfig)

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

    Qwen3OmniMoeConfig = get_veomni_config_class(Qwen3OmniMoeConfig)

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

    Qwen3VLConfig = get_veomni_config_class(Qwen3VLConfig)
    Qwen3VLVisionConfig = get_veomni_config_class(Qwen3VLVisionConfig)

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

    Qwen3OmniMoeConfig = get_veomni_config_class(Qwen3OmniMoeConfig)

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

    Qwen3OmniMoeConfig = get_veomni_config_class(Qwen3OmniMoeConfig)
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    Qwen3VLConfig = get_veomni_config_class(Qwen3VLConfig)

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

    LlamaConfig = get_veomni_config_class(LlamaConfig)
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    Qwen3VLConfig = get_veomni_config_class(Qwen3VLConfig)

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

    LlamaConfig = get_veomni_config_class(LlamaConfig)
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    Qwen3VLConfig = get_veomni_config_class(Qwen3VLConfig)

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

    # Explicit attn_implementation kwarg at load time ranks above ops_implementation["attn_implementation"]
    # and is merged before NPU hardware validation runs.
    llama_both_npu = LlamaConfig.from_pretrained(
        llama_dir,
        ops_implementation={"attn_implementation": "flash_attention_2_hub", "rms_norm_implementation": "torch_npu"},
        attn_implementation="sdpa",
    )
    assert llama_both_npu._attn_implementation == "sdpa"
    assert llama_both_npu.ops_config.attn_implementation == "sdpa"
    assert llama_both_npu.ops_config.rms_norm_implementation == "torch_npu"


def test_import_leaves_native_hf_config_methods_unchanged():
    """A fresh interpreter verifies the import boundary before any VeOmni import."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        from transformers import PretrainedConfig, LlamaConfig
        before = (PretrainedConfig.__post_init__, PretrainedConfig.from_dict.__func__,
                  PretrainedConfig.to_diff_dict, PretrainedConfig._attn_implementation)
        import veomni.models
        after = (PretrainedConfig.__post_init__, PretrainedConfig.from_dict.__func__,
                 PretrainedConfig.to_diff_dict, PretrainedConfig._attn_implementation)
        assert before == after
        native = LlamaConfig(attn_implementation='eager')
        assert not hasattr(native, 'ops_config')
        assert not hasattr(PretrainedConfig, '_set_ops_implementation')
        assert native._attn_implementation == 'eager'
    """)
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_adapter_config_instances_pickle_in_fresh_interpreter():
    """Every nested dynamic class reconstructs through the cached factory on spawn."""
    import pickle
    import subprocess
    import sys

    from transformers import Qwen3VLConfig

    config = get_veomni_config_class(Qwen3VLConfig)(
        ops_implementation={"attn_implementation": "sdpa"},
        vision_config={"ops_implementation": {"attn_implementation": "eager"}},
    )
    script = """
import pickle, sys
from veomni.models.configuration_base import VeOmniPretrainedConfig, get_veomni_config_class
from transformers import Qwen3VLConfig
config = pickle.loads(sys.stdin.buffer.read())
assert type(config) is get_veomni_config_class(Qwen3VLConfig)
assert isinstance(config.text_config, VeOmniPretrainedConfig)
assert config.text_config._attn_implementation == 'veomni_sdpa'
assert config.vision_config._attn_implementation == 'eager'
assert '_attn_implementation_internal' not in vars(config)
"""
    subprocess.run([sys.executable, "-c", script], input=pickle.dumps(config), check=True, capture_output=True)


def test_config_loader_preserves_nested_hf_attention_keywords(tmp_path):
    """The adapter must initialize before native HF parent attention propagation."""
    import json

    from transformers import Qwen3VLConfig

    from veomni.models import VeOmniPretrainedConfig, build_config

    payload = {
        "model_type": "qwen3_vl",
        "attn_implementation": "eager",
        "text_config": {"attn_implementation": "sdpa"},
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))
    loaded = build_config(tmp_path)
    assert type(loaded) is get_veomni_config_class(Qwen3VLConfig)
    assert isinstance(loaded.text_config, VeOmniPretrainedConfig)
    assert loaded._attn_implementation == "eager"
    assert loaded.text_config._attn_implementation == "veomni_sdpa"
    assert loaded.vision_config._attn_implementation == "eager"


def test_attention_has_one_storage_and_private_hf_write_is_local():
    from transformers import Qwen3VLConfig

    config = get_veomni_config_class(Qwen3VLConfig)(
        ops_implementation={"attn_implementation": "sdpa"},
        vision_config={"ops_implementation": {"attn_implementation": "eager"}},
    )
    config._attn_implementation_internal = "sdpa"
    assert config.ops_implementation["attn_implementation"] == "sdpa"
    assert config.ops_config.attn_implementation == config._attn_implementation == "sdpa"
    assert config.vision_config._attn_implementation == "eager"
    assert "_attn_implementation_internal" not in vars(config)
    config.ops_implementation["attn_implementation"] = "eager"
    assert config._attn_implementation_internal == "eager"
    config._attn_implementation = {"text_config": "eager"}
    assert config.text_config.ops_config.attn_implementation == "eager"


def test_omni_config_uses_module_config_base_and_preserves_root_ops():
    import pickle

    from veomni.models import VeOmniPretrainedConfig
    from veomni.models.seed_omni.configuration_omni import OmniConfig
    from veomni.models.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    config = OmniConfig(
        _module_entries={},
        training_graphs={},
        generation_graphs={},
        ops_implementation={"attn_implementation": "eager"},
    )
    assert isinstance(config, VeOmniPretrainedConfig)
    assert config.ops_implementation == {"attn_implementation": "eager"}
    # Stable handwritten subclasses of an adapter must retain their exact type.
    custom = DeepseekV4Config(attn_implementation="eager")
    assert type(pickle.loads(pickle.dumps(custom))) is DeepseekV4Config


@pytest.mark.parametrize("adaptation", ["factory", "in_place"])
@pytest.mark.parametrize("typed_override", [False, True])
def test_remote_config_export_preserves_native_source_and_reload_target(tmp_path, adaptation, typed_override):
    import json

    from transformers import AutoConfig

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models import build_config
    from veomni.models.configuration_base import VeOmniPretrainedConfig, as_veomni_config

    source = tmp_path / "source"
    source.mkdir()
    (source / "configuration_remote.py").write_text(
        "from transformers import PretrainedConfig\n"
        "from .remote_defaults import HIDDEN_SIZE\n"
        "class RemoteConfig(PretrainedConfig):\n"
        "    model_type = 'veomni_test_remote'\n"
        "    def __init__(self, hidden_size=HIDDEN_SIZE, **kwargs):\n"
        "        self.hidden_size = hidden_size\n"
        "        super().__init__(**kwargs)\n"
    )
    (source / "remote_defaults.py").write_text("HIDDEN_SIZE = 8\n")
    (source / "config.json").write_text(
        json.dumps(
            {"model_type": "veomni_test_remote", "auto_map": {"AutoConfig": "configuration_remote.RemoteConfig"}}
        )
    )
    native = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    if adaptation == "factory":
        config = get_veomni_config_class(type(native))(ops_implementation={"attn_implementation": "eager"})
    else:
        config = as_veomni_config(native)
        config._set_ops_implementation({"attn_implementation": "eager"})
    original_type = type(config)
    exported = tmp_path / "exported"
    config.save_pretrained(exported)
    saved = json.loads((exported / "config.json").read_text())
    assert saved["auto_map"]["AutoConfig"] == "configuration_remote.RemoteConfig"
    assert (exported / "configuration_remote.py").read_text() == (source / "configuration_remote.py").read_text()
    assert (exported / "remote_defaults.py").is_file()
    assert not (exported / "configuration_base.py").exists()
    assert type(config) is original_type
    assert isinstance(config, VeOmniPretrainedConfig)
    plain = AutoConfig.from_pretrained(exported, trust_remote_code=True, local_files_only=True)
    assert not isinstance(plain, VeOmniPretrainedConfig)
    overrides = {"ops_implementation": make_eager_ops_config()} if typed_override else {}
    restored = build_config(exported, trust_remote_code=True, local_files_only=True, **overrides)
    assert isinstance(restored, VeOmniPretrainedConfig)
    assert restored.hidden_size == 8
    assert restored.ops_config.attn_implementation == "eager"


@pytest.mark.parametrize("child_as_instance", [False, True])
def test_typed_parent_preserves_explicit_child_ops(child_as_instance):
    from transformers import Qwen3VLConfig, Qwen3VLVisionConfig

    from tests.tools.training_utils import make_eager_ops_config

    child = {"ops_implementation": {"attn_implementation": "eager"}}
    if child_as_instance:
        child = get_veomni_config_class(Qwen3VLVisionConfig)(**child)
    config = get_veomni_config_class(Qwen3VLConfig)(
        vision_config=child, ops_implementation=make_eager_ops_config(attn_implementation="sdpa")
    )
    assert config.ops_config.attn_implementation == "veomni_sdpa"
    assert config.text_config.ops_config.attn_implementation == "veomni_sdpa"
    assert config.vision_config.ops_config.attn_implementation == "eager"


def test_child_default_reset_survives_save_and_reload(tmp_path):
    from transformers import Qwen3VLConfig

    Config = get_veomni_config_class(Qwen3VLConfig)
    config = Config(ops_implementation={"attn_implementation": "eager"})
    config.vision_config._attn_implementation = None
    expected = config.vision_config.ops_config.attn_implementation
    assert expected != config.ops_config.attn_implementation
    config.save_pretrained(tmp_path)
    restored = Config.from_pretrained(tmp_path)
    assert restored.ops_config.attn_implementation == "eager"
    assert restored.text_config.ops_config.attn_implementation == "eager"
    assert restored.vision_config.ops_config.attn_implementation == expected


@pytest.mark.parametrize("entrypoint", ["class", "builder"])
@pytest.mark.parametrize("saved_root_ops", [False, True], ids=["empty-root", "eager-root"])
def test_native_child_load_override_receives_ops(tmp_path, entrypoint, saved_root_ops):
    from transformers import Qwen3VLConfig, Qwen3VLVisionConfig

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models import build_config
    from veomni.models.configuration_base import VeOmniPretrainedConfig

    Config = get_veomni_config_class(Qwen3VLConfig)
    Config(ops_implementation=make_eager_ops_config() if saved_root_ops else {}).save_pretrained(tmp_path)
    child = Qwen3VLVisionConfig(depth=1, hidden_size=32, num_heads=4)
    load = Config.from_pretrained if entrypoint == "class" else build_config
    config = load(tmp_path, vision_config=child, ops_implementation=make_eager_ops_config())
    assert isinstance(config.vision_config, VeOmniPretrainedConfig)
    assert config.vision_config.depth == 1
    assert config.vision_config.ops_config.attn_implementation == "eager"
    assert config.vision_config.ops_config.rotary_pos_emb_vision_implementation == "eager"


def test_ops_config_cache_refreshes_after_dictionary_updates(tmp_path):
    import json

    from transformers import LlamaConfig

    Config = get_veomni_config_class(LlamaConfig)
    config = Config(ops_implementation={"attn_implementation": "eager"})
    initial = config.ops_config
    assert config.ops_config is initial
    config.ops_implementation["attn_implementation"] = "veomni_sdpa"
    updated = config.ops_config
    assert updated is not initial
    assert config.ops_config is updated
    assert updated.attn_implementation == "veomni_sdpa"
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "_ops_config_cache" not in saved
    assert Config.from_pretrained(tmp_path).ops_config.attn_implementation == updated.attn_implementation


@pytest.mark.parametrize("family", ["llama", "flux", "ltx2_3"])
def test_direct_model_validates_final_ops_before_hf_initialization(monkeypatch, family):
    from tests.models.tiny_configs import tiny_flux_config, tiny_llama_config, tiny_ltx2_3_config
    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.diffusers.ltx2_3.ltx_transformer.modeling_ltx2_3_transformer import LTXVideoTransformerModel
    from veomni.models.transformers.flux.modeling_flux import FluxModel
    from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaForCausalLM

    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", True)
    monkeypatch.setattr("veomni.utils.import_utils.is_torch_mlu_available", lambda: False)

    def forbid_preload(*args, **kwargs):
        pytest.fail("Invalid final selections must fail before HF preloads attention")

    monkeypatch.setattr("transformers.modeling_utils.lazy_import_flash_attention", forbid_preload)
    make_config, model_class = {
        "llama": (tiny_llama_config, LlamaForCausalLM),
        "flux": (tiny_flux_config, FluxModel),
        "ltx2_3": (tiny_ltx2_3_config, LTXVideoTransformerModel),
    }[family]
    config = make_config()
    # Construction stages selections; the model boundary validates the final dictionary.
    config.ops_implementation = {
        **make_eager_ops_config().to_dict(),
        "attn_implementation": "flash_attention_2_hub",
    }
    with pytest.raises(ValueError, match="not supported on Ascend NPU"):
        model_class(config)


@pytest.mark.parametrize("entrypoint", ["constructor", "ops_override", "attention_override", "dict_update"])
def test_config_normalizes_deprecated_attention_aliases(tmp_path, entrypoint):
    from transformers import LlamaConfig

    from tests.tools.training_utils import make_eager_ops_config

    Config = get_veomni_config_class(LlamaConfig)
    legacy = "veomni_flash_attention_2_with_sp"
    values = {**make_eager_ops_config().to_dict(), "attn_implementation": legacy}
    if entrypoint == "constructor":
        config = Config(ops_implementation=values)
    else:
        config = Config(ops_implementation=make_eager_ops_config())
        config.save_pretrained(tmp_path)
        if entrypoint == "ops_override":
            config = Config.from_pretrained(tmp_path, ops_implementation=values)
        elif entrypoint == "attention_override":
            config = Config.from_pretrained(tmp_path, attn_implementation=legacy)
        else:
            config.ops_implementation["attn_implementation"] = legacy
    assert config.ops_config.attn_implementation == "veomni_flash_attention_2"
    assert config._attn_implementation == "veomni_flash_attention_2"
