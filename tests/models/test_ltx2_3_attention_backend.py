from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import veomni.models.diffusers.ltx2_3.ltx_core  # noqa: F401  (puts the vendored ltx_core on sys.path)
from veomni.arguments import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.diffusers.ltx2_3.ltx_transformer import modeling_ltx2_3_transformer as ltx_model
from veomni.models.diffusers.ltx2_3.ltx_transformer.configuration_ltx2_3_transformer import (
    LTXVideoTransformerModelConfig,
)
from veomni.ops.kernels.attention import flash


_FLASH_BACKENDS = [
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_2_hub",
    "flash_attention_3_hub",
    "flash_attention_4",
]
_TINY = dict(
    in_channels=8,
    out_channels=8,
    num_attention_heads=2,
    attention_head_dim=24,
    num_layers=2,
    cross_attention_dim=48,
    caption_channels=24,
    with_audio=True,
    audio_num_attention_heads=2,
    audio_attention_head_dim=12,
    audio_in_channels=128,
    audio_out_channels=128,
    audio_cross_attention_dim=24,
)


def build(backend, dtype=torch.float32):
    ops = OpsImplementationConfig(
        attn_implementation=backend,
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        swiglu_mlp_implementation="eager",
        cross_entropy_loss_implementation="eager",
        moe_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    model = build_foundation_model(
        LTXVideoTransformerModelConfig(**_TINY), init_device="cpu", torch_dtype="float32", ops_implementation=ops
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
    return model.to(dtype)


def inputs(context_mask=None):
    generator = torch.Generator().manual_seed(1)
    rand = lambda *shape: torch.randn(*shape, generator=generator)  # noqa: E731
    return {
        "hidden_states": [rand(1, 8, 2, 4, 6)],
        "timestep": [torch.tensor(0.5)],
        "encoder_hidden_states": [rand(1, 5, 24)],
        "context_mask": [torch.ones(1, 5, dtype=torch.int64) if context_mask is None else context_mask],
        "audio_hidden_states": [rand(1, 8, 7, 16)],
        "audio_timestep": [torch.tensor(0.5)],
        "training_target": [rand(1, 8, 2, 4, 6)],
        "audio_training_target": [rand(1, 8, 7, 16)],
    }


def run(model, sample):
    model.zero_grad(set_to_none=True)
    out = model(**sample)
    out.loss["mse_loss"].backward()
    return out, {name: param.grad for name, param in model.named_parameters() if param.grad is not None}


def attention_calls(model):
    calls = []
    for module in model.modules():
        if isinstance(module, ltx_model.Attention):
            module.register_forward_hook(lambda *_: calls.append(1))
    return calls


@pytest.fixture
def recorded_flash(monkeypatch):
    """Stub VeOmni's local/Hub FA loader with an FA-layout SDPA varlen kernel that records each call."""
    calls, loads = [], []

    def varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        assert q.ndim == 3 and q.dtype == k.dtype == v.dtype == torch.bfloat16 and not causal
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32
        assert cu_seqlens_q[-1] == q.shape[0] and cu_seqlens_k[-1] == k.shape[0]
        calls.append((max_seqlen_q, max_seqlen_k))
        split = lambda t, n: t.unflatten(0, (-1, n)).transpose(1, 2)  # noqa: E731
        out = F.scaled_dot_product_attention(
            split(q, max_seqlen_q), split(k, max_seqlen_k), split(v, max_seqlen_k), scale=softmax_scale
        )
        return out.transpose(1, 2).flatten(0, 1), None  # FA3/FA4-style (out, lse); the adapter must unwrap it

    def loader(name):
        loads.append(name)
        return SimpleNamespace(flash_attn_func=None, flash_attn_varlen_func=varlen)

    monkeypatch.setattr(flash, "_load_veomni_flash_kernel", loader)
    return SimpleNamespace(calls=calls, loads=loads)


@pytest.mark.parametrize("backend", _FLASH_BACKENDS)
def test_flash_backends_drive_every_dit_attention(recorded_flash, backend):
    model = build(backend, torch.bfloat16)
    reference = build("eager", torch.bfloat16)
    invoked = attention_calls(model)
    recorded_flash.loads.clear()  # Transformers may already have resolved the backend at construction

    out, grads = run(model, inputs())
    assert len(recorded_flash.calls) == len(invoked) > 0  # the all-valid text mask is dropped, nothing is masked
    assert recorded_flash.loads == [f"veomni_{backend}_with_sp"]

    recorded_flash.calls.clear()
    expected, expected_grads = run(reference, inputs())
    assert recorded_flash.calls == []  # an eager model next to it stays on PyTorch SDPA
    for actual, want in zip(
        out.predictions + out.audio_predictions, expected.predictions + expected.audio_predictions
    ):
        torch.testing.assert_close(actual, want)
    assert grads.keys() == expected_grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, expected_grads[name], msg=name)
    run(model, inputs())
    assert recorded_flash.loads == [f"veomni_{backend}_with_sp"]  # loaded once per model


def test_flash_backend_rejects_partial_masks_and_fp32(recorded_flash):
    model = build("flash_attention_3", torch.bfloat16)
    with pytest.raises(NotImplementedError, match="partial context"):
        run(model, inputs(context_mask=torch.tensor([[1, 1, 1, 0, 0]])))
    with pytest.raises(ValueError, match="FP16/BF16"):
        run(build("flash_attention_3"), inputs())


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_sdpa_backends_pin_pytorch_sdpa(recorded_flash, backend):
    model = build(backend)
    run(model, inputs())
    run(model, inputs(context_mask=torch.tensor([[1, 1, 1, 0, 0]])))  # masked calls stay on SDPA
    assert recorded_flash.calls == []
    functions = {
        type(fn)
        for module in model.modules()
        if isinstance(module, ltx_model.Attention)
        for fn in (module.attention_function, module.masked_attention_function)
    }
    assert functions == {ltx_model.PytorchAttention}


@pytest.mark.parametrize("backend", ["flash_attention_2", "flash_attention_3", "flash_attention_3_hub"])
def test_direct_loading_with_a_public_flash_name_fails_closed(backend):
    """Only build_foundation_model resolves public flash names; direct loading must not silently pick SDPA."""
    config = LTXVideoTransformerModelConfig(**_TINY)
    config._attn_implementation = backend
    with pytest.raises((ValueError, ImportError)):
        ltx_model.LTXVideoTransformerModel._from_config(config)


@pytest.mark.parametrize("backend", ["flex_attention", "magi_attention"])
def test_unsupported_backends_are_rejected_at_construction(backend):
    with pytest.raises(ValueError):
        build(backend)
