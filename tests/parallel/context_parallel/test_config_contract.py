import inspect
from types import SimpleNamespace

import pytest

from veomni.arguments import arguments_types
from veomni.arguments.arguments_types import (
    DataArguments,
    ModelArguments,
    OpsImplementationConfig,
    VeOmniArguments,
    validate_context_parallel_config,
    validate_gdn_context_parallel_config,
)
from veomni.distributed.parallel_state import ParallelState


def test_gdn_cp_selector_rejects_unknown_value():
    with pytest.raises(ValueError, match="gdn_context_parallel_implementation must be one of"):
        OpsImplementationConfig(gdn_context_parallel_implementation="experimental")


@pytest.mark.parametrize("implementation", ["state_passing_lossless", "kcp"])
def test_lossless_gdn_cp_rejects_ascendc_without_initial_state_gradient(implementation):
    with pytest.raises(ValueError, match="does not implement that gradient"):
        OpsImplementationConfig(
            gdn_context_parallel_implementation=implementation,
            chunk_gated_delta_rule_implementation="npu_ascendc",
        )


@pytest.mark.parametrize(
    ("cp_size", "implementation", "dyn_bsz", "message"),
    [
        (1, "state_passing_lossless", True, "requires train.accelerator.cp_size > 1"),
        (1, "kcp", True, "requires train.accelerator.cp_size > 1"),
        (2, "state_passing_lossless", False, "requires train.dyn_bsz=True"),
        (2, "kcp", False, "requires train.dyn_bsz=True"),
        (3, "state_passing_lossless", True, "requires cp_size to be a power of two"),
        (3, "kcp", True, "requires cp_size to be a power of two"),
    ],
)
def test_root_config_fails_closed_before_runtime(cp_size, implementation, dyn_bsz, message):
    arguments = SimpleNamespace(
        train=SimpleNamespace(
            accelerator=SimpleNamespace(cp_size=cp_size),
            dyn_bsz=dyn_bsz,
        ),
        model=SimpleNamespace(
            ops_implementation=SimpleNamespace(gdn_context_parallel_implementation=implementation),
        ),
    )

    with pytest.raises(ValueError, match=message):
        VeOmniArguments.__post_init__(arguments)


@pytest.mark.parametrize("implementation", ["state_passing_lossless", "kcp"])
def test_root_config_accepts_typed_gdn_cp_implementations(implementation):
    validate_gdn_context_parallel_config(cp_size=2, implementation=implementation, dyn_bsz=True)


def test_generic_ring_context_parallel_uses_disabled_selector_for_non_gdn_models():
    validate_gdn_context_parallel_config(cp_size=2, implementation="disabled", dyn_bsz=True)
    validate_context_parallel_config(
        cp_size=2,
        implementation="disabled",
        dyn_bsz=True,
        attn_implementation="veomni_flash_attention_2_with_sp",
        data_type="conversation",
        model_type="llama",
    )

    with pytest.raises(ValueError, match="to be one of"):
        ParallelState(cp_size=2, gdn_context_parallel_implementation="experimental")


def test_generic_ring_selector_never_silently_enables_qwen_gdn():
    with pytest.raises(ValueError, match="requires explicit"):
        validate_context_parallel_config(
            cp_size=2,
            implementation="disabled",
            dyn_bsz=True,
            attn_implementation="veomni_flash_attention_2_with_sp",
            data_type="conversation",
            model_type="qwen3_5_moe",
        )


def test_combined_cp_ulysses_malformed_sp_mesh_fails_closed():
    class BrokenMesh:
        def get_group(self, name):
            raise KeyError(name)

        def get_local_rank(self, name):
            raise KeyError(name)

    state = object.__new__(ParallelState)
    object.__setattr__(state, "cp_size", 2)
    object.__setattr__(state, "ulysses_size", 2)
    object.__setattr__(state, "device_mesh", BrokenMesh())
    object.__setattr__(state, "_sp_mesh", None)
    with pytest.raises(RuntimeError, match="flattened SP mesh"):
        _ = state.sp_group
    with pytest.raises(RuntimeError, match="flattened SP mesh"):
        _ = state.sp_rank


@pytest.mark.parametrize("implementation", ["disabled", "state_passing_lossless", "kcp"])
def test_context_parallel_rejects_cuda_topology_before_mesh_initialization(implementation):
    with pytest.raises(NotImplementedError, match="supported on Ascend NPU only"):
        ParallelState(
            cp_size=2,
            gdn_context_parallel_implementation=implementation,
            device_type="cuda",
        )


def test_context_parallel_selector_preserves_existing_positional_api_order():
    from veomni.distributed.parallel_state import init_parallel_state

    state_parameters = list(inspect.signature(ParallelState).parameters)
    init_parameters = list(inspect.signature(init_parallel_state).parameters)
    old_state_prefix = [
        "dp_size",
        "dp_replicate_size",
        "dp_shard_size",
        "tp_size",
        "pp_size",
        "cp_size",
        "ulysses_size",
        "dp_mode",
        "device_type",
        "include_sp_in_fsdp",
        "device_mesh",
        "extra_parallel_names",
        "extra_parallel_sizes",
        "extra_parallel_fsdp_device_mesh",
        "async_enabled",
    ]
    old_init_prefix = [
        "dp_size",
        "dp_replicate_size",
        "dp_shard_size",
        "tp_size",
        "pp_size",
        "cp_size",
        "ulysses_size",
        "dp_mode",
        "device_type",
        "include_sp_in_fsdp",
        "extra_parallel_sizes",
        "extra_parallel_placement_innermost",
        "extra_parallel_names",
        "async_enabled",
        "name",
    ]
    assert state_parameters[: len(old_state_prefix)] == old_state_prefix
    assert init_parameters[: len(old_init_prefix)] == old_init_prefix
    assert state_parameters[len(old_state_prefix) :] == ["gdn_context_parallel_implementation", "_sp_mesh"]
    assert init_parameters[len(old_init_prefix) :] == ["gdn_context_parallel_implementation"]


@pytest.mark.parametrize("attention", ["eager", "sdpa", "flash_attention_2"])
def test_gdn_context_parallel_rejects_attention_without_ring_dispatch(attention):
    with pytest.raises(ValueError, match="requires a VeOmni FlashAttention SP backend"):
        validate_context_parallel_config(
            cp_size=2,
            implementation="state_passing_lossless",
            dyn_bsz=True,
            attn_implementation=attention,
            data_type="conversation",
            model_type="qwen3_5_text",
        )


def test_gdn_context_parallel_rejects_model_without_patched_capability():
    with pytest.raises(ValueError, match="implemented only for Qwen3.5"):
        validate_context_parallel_config(
            cp_size=2,
            implementation="state_passing_lossless",
            dyn_bsz=True,
            attn_implementation="veomni_flash_attention_2_with_sp",
            data_type="conversation",
            model_type="llama",
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"attention_dropout": 0.1}, "attention_dropout=0"),
        ({"sliding_window": 4096}, "does not support sliding-window"),
        ({"data_type": "diffusion"}, "text-only packed causal-LM"),
        ({"is_encoder_decoder": True}, "causal self-attention decoder"),
    ],
)
def test_gdn_context_parallel_rejects_unsupported_attention_contract(override, message):
    contract = {
        "cp_size": 2,
        "implementation": "state_passing_lossless",
        "dyn_bsz": True,
        "attn_implementation": "veomni_flash_attention_2_with_sp",
        "data_type": "conversation",
        "model_type": "qwen3_5_text",
        "attention_dropout": 0.0,
        "sliding_window": None,
        "is_encoder_decoder": False,
    }
    contract.update(override)
    with pytest.raises(ValueError, match=message):
        validate_context_parallel_config(**contract)


@pytest.mark.parametrize("implementation", ["state_passing_lossless", "kcp"])
def test_gdn_context_parallel_accepts_explicit_supported_contract(implementation):
    validate_context_parallel_config(
        cp_size=8,
        implementation=implementation,
        dyn_bsz=True,
        attn_implementation="veomni_flash_attention_4_with_sp",
        data_type="conversation",
        model_type="qwen3_5_moe_text",
        attention_dropout=0.0,
        sliding_window=None,
        is_encoder_decoder=False,
    )


def test_root_config_rejects_cp_before_collator_can_slice_tokens():
    ops = OpsImplementationConfig(load_balancing_loss_implementation="eager")
    arguments = VeOmniArguments(
        model=ModelArguments(config_path="unused-test-config", ops_implementation=ops),
        data=DataArguments(train_path="unused-test-data"),
    )
    arguments.train.accelerator.cp_size = 3
    with pytest.raises(ValueError, match="requires cp_size to be a power of two"):
        VeOmniArguments.__post_init__(arguments)


def test_root_config_skips_model_config_read_when_cp_disabled(tmp_path):
    invalid_config = tmp_path / "config.json"
    invalid_config.write_text("{not-json", encoding="utf-8")
    ops = OpsImplementationConfig(load_balancing_loss_implementation="eager")
    arguments = object.__new__(VeOmniArguments)
    arguments.model = ModelArguments(config_path=str(invalid_config), ops_implementation=ops)
    arguments.data = DataArguments(train_path="unused-test-data")
    arguments.train = arguments_types.TrainingArguments()
    VeOmniArguments.__post_init__(arguments)
