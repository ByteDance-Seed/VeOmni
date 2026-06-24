import sys

import pytest
import torch

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type
from veomni.utils.import_utils import is_quack_gemm_available


def _ops_config(moe_implementation: str = "eager") -> OpsImplementationConfig:
    return OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation=moe_implementation,
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
    )


def _bind_gpt_oss_moe(impl: str) -> None:
    """Rebind GPT-OSS' module-level MoE OpSlot before each comparison forward."""
    from veomni.models.transformers.gpt_oss.generated import patched_modeling_gpt_oss_gpu as gpt_oss_gen

    gpt_oss_gen.veomni_moe_experts_forward.bind("eager" if impl == "eager" else impl.removeprefix("fused_"))


@pytest.fixture(autouse=True)
def _reset_gpt_oss_moe_binding():
    yield
    module = sys.modules.get("veomni.models.transformers.gpt_oss.generated.patched_modeling_gpt_oss_gpu")
    if module is not None:
        module.veomni_moe_experts_forward.bind("eager")


def test_gpt_oss_veomni_forward_loss_contract(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    model = build_foundation_model(
        "tests/toy_config/gpt_oss_toy",
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_ops_config(),
    )

    input_ids = torch.randint(3, 64, (1, 6))
    output = model(input_ids=input_ids, labels=input_ids, use_cache=False)

    assert model.__class__.__module__ == ("veomni.models.transformers.gpt_oss.generated.patched_modeling_gpt_oss_gpu")
    assert isinstance(output.loss, torch.Tensor)
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.logits is None
    assert output.fused_linear_aux is None


def test_gpt_oss_uses_dedicated_moe_opslot_variant(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    model = build_foundation_model(
        "tests/toy_config/gpt_oss_toy",
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_ops_config(),
    )

    from veomni.models.transformers.gpt_oss.generated import patched_modeling_gpt_oss_gpu as gpt_oss_gen

    assert gpt_oss_gen.veomni_moe_experts_forward.variant == "gpt_oss"
    assert model.model.layers[0].mlp.experts.gate_up_proj.shape[-1] == 2 * model.config.intermediate_size


def test_gpt_oss_moe_unbound_slot_does_not_fallback_to_eager(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    model = build_foundation_model(
        "tests/toy_config/gpt_oss_toy",
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_ops_config(moe_implementation="eager"),
    )

    from veomni.models.transformers.gpt_oss.generated import patched_modeling_gpt_oss_gpu as gpt_oss_gen

    slot = gpt_oss_gen.veomni_moe_experts_forward
    monkeypatch.setattr(slot, "_kernel", None)
    monkeypatch.setattr(slot, "_impl_name", None)

    experts = model.model.layers[0].mlp.experts
    hidden_states = torch.randn(4, model.config.hidden_size)
    router_indices = torch.zeros(4, model.config.num_experts_per_tok, dtype=torch.long)
    routing_weights = torch.full((4, model.config.num_experts_per_tok), 1.0 / model.config.num_experts_per_tok)

    with pytest.raises(RuntimeError, match="moe_implementation='eager'"):
        experts(hidden_states, router_indices=router_indices, routing_weights=routing_weights)


def test_gpt_oss_rejects_triton_moe_backend(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    with pytest.raises(KeyError, match="Unknown kernel 'triton'.*variant='gpt_oss'"):
        build_foundation_model(
            "tests/toy_config/gpt_oss_toy",
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=_ops_config(moe_implementation="fused_triton"),
        )


def test_gpt_oss_fused_quack_without_sm90_raises(monkeypatch):
    from veomni.ops import kernel_registry

    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    monkeypatch.setattr(kernel_registry, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(kernel_registry, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(kernel_registry, "get_gpu_compute_capability", lambda: 80)

    with pytest.raises(RuntimeError, match="compute_capability>=90"):
        build_foundation_model(
            "tests/toy_config/gpt_oss_toy",
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=_ops_config(moe_implementation="fused_quack"),
        )


def test_gpt_oss_fused_moe_matches_eager_cuda(monkeypatch):
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for GPT-OSS fused_quack MoE.")
    if not is_quack_gemm_available():
        pytest.skip("Quack SM90+ GEMM kernels are required for GPT-OSS fused_quack MoE.")

    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torch.manual_seed(0)
    device = get_device_type()
    eager_model = build_foundation_model(
        "tests/toy_config/gpt_oss_toy",
        torch_dtype="bfloat16",
        init_device=device,
        ops_implementation=_ops_config(moe_implementation="eager"),
    ).eval()
    with torch.no_grad():
        for layer in eager_model.model.layers:
            layer.mlp.experts.gate_up_proj_bias.normal_(mean=0.0, std=0.02)
            layer.mlp.experts.down_proj_bias.normal_(mean=0.0, std=0.02)
    fused_model = build_foundation_model(
        "tests/toy_config/gpt_oss_toy",
        torch_dtype="bfloat16",
        init_device=device,
        ops_implementation=_ops_config(moe_implementation="fused_quack"),
    ).eval()
    fused_model.load_state_dict(eager_model.state_dict())

    input_ids = torch.randint(3, 64, (1, 8), device=device)
    with torch.no_grad():
        _bind_gpt_oss_moe("eager")
        eager_logits = eager_model(input_ids=input_ids, use_cache=False).logits
        _bind_gpt_oss_moe("fused_quack")
        fused_logits = fused_model(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(fused_logits, eager_logits, atol=8e-3, rtol=8e-3)


def test_gpt_oss_quack_fused_moe_matches_eager_cuda(monkeypatch):
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for GPT-OSS fused_quack MoE.")
    if not is_quack_gemm_available():
        pytest.skip("Quack SM90+ GEMM kernels are required for GPT-OSS fused_quack MoE.")

    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torch.manual_seed(0)
    device = get_device_type()
    eager_model = build_foundation_model(
        "tests/toy_config/gpt_oss_toy",
        torch_dtype="bfloat16",
        init_device=device,
        ops_implementation=_ops_config(moe_implementation="eager"),
    )
    with torch.no_grad():
        for layer in eager_model.model.layers:
            layer.mlp.experts.gate_up_proj_bias.normal_(mean=0.0, std=0.02)
            layer.mlp.experts.down_proj_bias.normal_(mean=0.0, std=0.02)

    input_ids = torch.randint(3, 64, (1, 8), device=device)
    labels = input_ids.clone()
    _bind_gpt_oss_moe("eager")
    eager_loss = eager_model(input_ids=input_ids, labels=labels, use_cache=False).loss

    fused_model = build_foundation_model(
        "tests/toy_config/gpt_oss_toy",
        torch_dtype="bfloat16",
        init_device=device,
        ops_implementation=_ops_config(moe_implementation="fused_quack"),
    )
    fused_model.load_state_dict(eager_model.state_dict())

    _bind_gpt_oss_moe("fused_quack")
    fused_loss = fused_model(input_ids=input_ids, labels=labels, use_cache=False).loss
    torch.testing.assert_close(fused_loss, eager_loss, atol=8e-3, rtol=8e-3)

    eager_loss.backward()
    fused_loss.backward()
    eager_grad = eager_model.model.layers[0].mlp.experts.gate_up_proj.grad
    fused_grad = fused_model.model.layers[0].mlp.experts.gate_up_proj.grad
    torch.testing.assert_close(fused_grad, eager_grad, atol=2e-2, rtol=2e-2)
