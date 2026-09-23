"""Qwen-Image transformer attention follows ``attn_implementation`` instead of always running SDPA."""

from types import SimpleNamespace

import pytest
from diffusers.models import attention_dispatch
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

from veomni.arguments import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.diffusers.qwen_image.qwen_image_transformer import modeling_qwen_image_transformer as qi_model
from veomni.models.diffusers.qwen_image.qwen_image_transformer.configuration_qwen_image_transformer import (
    QwenImageTransformer2DModelConfig,
)
from veomni.ops.kernels.attention import flash


def tiny_config():
    return QwenImageTransformer2DModelConfig(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
    )


def build(attn_implementation):
    ops = OpsImplementationConfig(
        attn_implementation=attn_implementation,
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        swiglu_mlp_implementation="eager",
        cross_entropy_loss_implementation="eager",
        moe_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    return build_foundation_model(tiny_config(), init_device="cpu", torch_dtype="float32", ops_implementation=ops)


def attention_backends(model):
    return {block.attn.processor._attention_backend for block in model.transformer_blocks}


@pytest.fixture
def warnings(monkeypatch):
    """Stub every Hub kernel download and record fallback warnings; ``set_attention_backend`` stays real."""
    kernel = SimpleNamespace(flash_attn_func=lambda *a, **k: None, flash_attn_varlen_func=lambda *a, **k: None)
    monkeypatch.setattr(flash, "_load_veomni_flash_kernel", lambda implementation: kernel)  # Transformers init
    monkeypatch.setattr(attention_dispatch, "_check_attention_backend_requirements", lambda backend: None)
    monkeypatch.setattr(attention_dispatch, "_maybe_download_kernel_for_backend", lambda backend: None)
    monkeypatch.setattr(_AttentionBackendRegistry, "_active_backend", AttentionBackendName.NATIVE)
    monkeypatch.setattr(qi_model, "diffusers_version", "0.40.0")
    recorded = []
    monkeypatch.setattr(qi_model.logger, "warning_once", recorded.append)
    return recorded


@pytest.mark.parametrize(
    "attn_implementation, backend",
    [
        ("flash_attention_2_hub", AttentionBackendName.FLASH_VARLEN_HUB),
        ("flash_attention_3_hub", AttentionBackendName._FLASH_3_VARLEN_HUB),
    ],
)
def test_hub_attn_implementation_selects_diffusers_backend(warnings, attn_implementation, backend):
    model = build(attn_implementation)
    assert attention_backends(model) == {backend}
    assert warnings == []
    # Only this model switches: diffusers' process-wide default stays as it was.
    assert _AttentionBackendRegistry._active_backend == AttentionBackendName.NATIVE


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_sdpa_binds_native_backend(warnings, monkeypatch, attn_implementation):
    # A process-wide flash_varlen default (e.g. DIFFUSERS_ATTN_BACKEND) keeps a key prefix and would
    # mishandle the padded text, so SDPA must not follow it.
    monkeypatch.setattr(_AttentionBackendRegistry, "_active_backend", AttentionBackendName.FLASH_VARLEN)
    model = build(attn_implementation)
    assert attention_backends(model) == {AttentionBackendName.NATIVE}
    assert warnings == []
    assert _AttentionBackendRegistry._active_backend == AttentionBackendName.FLASH_VARLEN


@pytest.mark.parametrize("attn_implementation", ["flash_attention_2", "flash_attention_3"])
def test_local_flash_falls_back_to_sdpa_with_warning(warnings, attn_implementation):
    model = build(attn_implementation)
    assert attention_backends(model) == {AttentionBackendName.NATIVE}
    assert len(warnings) == 1
    assert "native SDPA" in warnings[0]


def test_hub_backend_requires_diffusers_0_40(warnings, monkeypatch):
    model = build("eager")
    monkeypatch.setattr(qi_model, "diffusers_version", "0.37.0")
    with pytest.raises(ImportError, match=r"requires diffusers>=0\.40\.0 \(found 0\.37\.0\)"):
        model._configure_attention("veomni_flash_attention_3_hub_with_sp")
    assert attention_backends(model) == {AttentionBackendName.NATIVE}
