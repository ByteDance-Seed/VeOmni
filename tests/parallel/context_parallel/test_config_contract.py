from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize(
    ("cp_size", "implementation", "dyn_bsz", "message"),
    [
        (1, "state_passing_lossless", True, "requires train.accelerator.cp_size > 1"),
        (2, "state_passing_lossless", False, "requires train.dyn_bsz=True"),
        (3, "state_passing_lossless", True, "requires cp_size to be a power of two"),
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


def test_root_config_accepts_typed_gdn_cp_implementation():
    validate_gdn_context_parallel_config(cp_size=2, implementation="state_passing_lossless", dyn_bsz=True)


def test_generic_context_parallel_without_a_selector_fails_closed():
    with pytest.raises(ValueError, match="generic Ring CP is not a production configuration"):
        validate_gdn_context_parallel_config(cp_size=3, implementation="disabled", dyn_bsz=False)

    with pytest.raises(ValueError, match="generic Ring CP is not enabled"):
        ParallelState(cp_size=2)


def test_gdn_context_parallel_rejects_cuda_topology_before_mesh_initialization():
    with pytest.raises(NotImplementedError, match="supported on Ascend NPU only"):
        ParallelState(
            cp_size=2,
            gdn_context_parallel_implementation="state_passing_lossless",
            device_type="cuda",
        )


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


def test_gdn_context_parallel_accepts_explicit_supported_contract():
    validate_context_parallel_config(
        cp_size=8,
        implementation="state_passing_lossless",
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
    with pytest.raises(ValueError, match="generic Ring CP is not a production configuration"):
        VeOmniArguments.__post_init__(arguments)
