"""Qwen-Image joint attention follows ``attn_implementation`` instead of always running SDPA."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import modeling_flash_attention_utils as hf_flash_utils

from veomni.arguments import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.diffusers.qwen_image.qwen_image_transformer import modeling_qwen_image_transformer as qi_model
from veomni.models.diffusers.qwen_image.qwen_image_transformer.configuration_qwen_image_transformer import (
    QwenImageTransformer2DModelConfig,
)
from veomni.ops.kernels.attention import flash


FLASH_CASES = [
    ("flash_attention_2", "veomni_flash_attention_2_with_sp", "flash_attention_2"),
    ("flash_attention_3", "veomni_flash_attention_3_with_sp", "flash_attention_3"),
    ("flash_attention_2_hub", "veomni_flash_attention_2_hub_with_sp", "veomni_flash_attention_2_hub_with_sp"),
    ("flash_attention_3_hub", "veomni_flash_attention_3_hub_with_sp", "veomni_flash_attention_3_hub_with_sp"),
]


def tiny_config():
    return QwenImageTransformer2DModelConfig(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
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


def processor(model):
    (proc,) = {block.attn.processor for block in model.transformer_blocks}
    return proc


@pytest.fixture
def flash_calls(monkeypatch):
    """Stub kernel loading and replace the flash kernel with an SDPA oracle of its varlen semantics."""
    unused = lambda *args, **kwargs: None  # noqa: E731 - preloaded by Transformers, never called here
    kernel = SimpleNamespace(flash_attn_func=unused, flash_attn_varlen_func=unused)
    monkeypatch.setattr(flash, "_load_veomni_flash_kernel", lambda implementation: kernel)  # Transformers init
    # Transformers caches the loaded kernel process-wide; restore it so the stub does not leak into other tests.
    cached = ("_loaded_implementation", "_flash_fn", "_flash_varlen_fn", "_flash_with_kvcache_fn")
    for name in (*cached, "_pad_fn", "_unpad_fn", "_process_flash_kwargs_fn"):
        monkeypatch.setattr(hf_flash_utils, name, getattr(hf_flash_utils, name))
    calls = []

    def varlen_oracle(query, key, value, attention_mask, **kwargs):  # packed q/k/v: [1, T, H, D]
        calls.append({"mask": attention_mask, **kwargs})
        if kwargs.get("cu_seq_lens_q") is None:
            return F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            ).transpose(1, 2)
        q_bounds, k_bounds = kwargs["cu_seq_lens_q"].tolist(), kwargs["cu_seq_lens_k"].tolist()
        segments = [
            F.scaled_dot_product_attention(
                query[:, q0:q1].transpose(1, 2), key[:, k0:k1].transpose(1, 2), value[:, k0:k1].transpose(1, 2)
            )
            for q0, q1, k0, k1 in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:])
        ]
        return torch.cat(segments, dim=2).transpose(1, 2)

    monkeypatch.setattr(flash, "_flash_attention_forward", varlen_oracle)
    return calls


@pytest.mark.parametrize("requested, resolved, kernel", FLASH_CASES)
def test_flash_names_select_the_flash_wrapper(flash_calls, requested, resolved, kernel):
    proc = processor(build(requested))
    assert proc.use_flash_attention
    assert proc.kernel_config._attn_implementation == resolved


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_sdpa_binds_native_backend(monkeypatch, attn_implementation):
    backends = []

    def record(*args, backend=None, **kwargs):
        backends.append(backend)
        return torch.zeros_like(args[0])

    model = build(attn_implementation)
    assert not processor(model).use_flash_attention
    monkeypatch.setattr(qi_model, "dispatch_attention_fn", record)
    run_attention(model)
    # None would follow diffusers' process-wide default (e.g. DIFFUSERS_ATTN_BACKEND=flash_varlen).
    assert backends == ["native"]


def test_flash_path_rejects_additive_masks():
    with pytest.raises(ValueError, match="boolean or 0/1 keep mask"):
        qi_model._joint_varlen_metadata(torch.zeros(2, 11))  # 0 = attend in an additive mask


@pytest.mark.parametrize("invalid_value", [-1, 2])
def test_flash_path_rejects_nonbinary_integer_masks(invalid_value):
    with pytest.raises(ValueError, match="boolean or 0/1 keep mask"):
        qi_model._joint_varlen_metadata(torch.tensor([[1, invalid_value, 0, 1]]))


def test_flash_path_accepts_integer_keep_masks():
    metadata = qi_model._joint_varlen_metadata(torch.tensor([[1, 1, 0, 1], [1, 0, 0, 0]]))
    assert metadata["indices"].tolist() == [0, 1, 3, 4]
    assert metadata["cu_seqlens"].tolist() == [0, 3, 4]
    assert metadata["max_seqlen"] == 3


@pytest.mark.parametrize("attn_implementation", ["flex_attention", "veomni_magi_attention_with_sp"])
def test_unsupported_attn_implementation_is_rejected(attn_implementation):
    with pytest.raises(ValueError, match="Qwen-Image does not support attn_implementation"):
        qi_model.QwenImageSPAttnProcessor(attn_implementation)


def run_attention(model, seed=0, masked=True):
    """Joint attention with optional text padding in the middle of the joint sequence."""
    generator = torch.Generator().manual_seed(seed)
    batch, text_len, image_len, dim = 2, 5, 6, 16
    image = torch.randn(batch, image_len, dim, generator=generator, requires_grad=True)
    text = torch.randn(batch, text_len, dim, generator=generator, requires_grad=True)
    text_mask = torch.ones(batch, text_len, dtype=torch.bool)
    if masked:
        text_mask[1, 3:] = False
    joint_mask = torch.cat([text_mask, torch.ones(batch, image_len, dtype=torch.bool)], dim=1) if masked else None
    attn = model.transformer_blocks[0].attn
    image_out, text_out = attn.processor(attn, image, encoder_hidden_states=text, attention_mask=joint_mask)
    return image, text, image_out, text_out, text_mask


@pytest.mark.parametrize("requested, resolved, kernel", FLASH_CASES)
@pytest.mark.parametrize("masked", [False, True])
def test_flash_path_matches_sdpa_on_valid_tokens(flash_calls, requested, resolved, kernel, masked):
    reference = build("eager")
    model = build(requested)
    model.load_state_dict(reference.state_dict())

    ref_image, ref_text, ref_image_out, ref_text_out, text_mask = run_attention(reference, masked=masked)
    image, text, image_out, text_out, _ = run_attention(model, masked=masked)

    (call,) = flash_calls
    assert call["attn_implementation"] == kernel
    assert call["mask"] is None  # explicit varlen metadata instead of per-layer unpadding from the mask
    if masked:
        assert call["cu_seq_lens_q"].tolist() == [0, 11, 20]  # sample 1: 3 valid text + 6 image tokens
        assert call["cu_seq_lens_k"].tolist() == [0, 11, 20]
        assert call["max_length_q"] == call["max_length_k"] == 11
    else:
        assert call.get("cu_seq_lens_q") is None
        assert call.get("cu_seq_lens_k") is None
    torch.testing.assert_close(image_out, ref_image_out)
    torch.testing.assert_close(text_out[text_mask], ref_text_out[text_mask])
    (image_out.square().sum() + text_out[text_mask].square().sum()).backward()
    (ref_image_out.square().sum() + ref_text_out[text_mask].square().sum()).backward()
    torch.testing.assert_close(image.grad, ref_image.grad)
    torch.testing.assert_close(text.grad, ref_text.grad)
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        assert (parameter.grad is None) == (ref_parameter.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, ref_parameter.grad)


def test_attention_selection_is_instance_local(flash_calls):
    hub_model = build("flash_attention_3_hub")
    local_model = build("flash_attention_2")
    native_model = build("eager")
    for model in (hub_model, native_model, local_model, hub_model):
        run_attention(model)
    assert [call["attn_implementation"] for call in flash_calls] == [
        "veomni_flash_attention_3_hub_with_sp",
        "flash_attention_2",
        "veomni_flash_attention_3_hub_with_sp",
    ]


@pytest.mark.parametrize("error_type", [ImportError, RuntimeError])
def test_flash_kernel_failure_does_not_fall_back(flash_calls, monkeypatch, error_type):
    model = build("flash_attention_3_hub")

    def unavailable(*args, **kwargs):
        raise error_type("requested flash kernel is unavailable")

    monkeypatch.setattr(flash, "_flash_attention_forward", unavailable)
    with pytest.raises(error_type, match="requested flash kernel is unavailable"):
        run_attention(model)


@pytest.mark.parametrize("mask_shape", [(1, 11), (2, 10)])
@pytest.mark.parametrize("precomputed", [False, True])
def test_flash_path_rejects_mismatched_joint_mask_shape(flash_calls, mask_shape, precomputed):
    model = build("flash_attention_3_hub")
    attn = model.transformer_blocks[0].attn
    mask = torch.ones(mask_shape, dtype=torch.bool)
    metadata = qi_model._joint_varlen_metadata(mask) if precomputed else None
    with pytest.raises(ValueError, match="joint mask shape"):
        attn.processor(
            attn,
            torch.randn(2, 6, 16),
            encoder_hidden_states=torch.randn(2, 5, 16),
            attention_mask=mask,
            joint_varlen_metadata=metadata,
        )
    assert not flash_calls


@pytest.mark.parametrize("invalid_value", [-1, 2, float("-inf")])
def test_precomputed_metadata_does_not_bypass_mask_validation(flash_calls, invalid_value):
    model = build("flash_attention_3_hub")
    attn = model.transformer_blocks[0].attn
    mask = torch.tensor([[1] * 11, [1] * 10 + [invalid_value]])
    metadata = qi_model._joint_varlen_metadata(torch.ones(2, 11, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean or 0/1 keep mask"):
        attn.processor(
            attn,
            torch.randn(2, 6, 16),
            encoder_hidden_states=torch.randn(2, 5, 16),
            attention_mask=mask,
            joint_varlen_metadata=metadata,
        )
    assert not flash_calls


@pytest.mark.parametrize("invalid_value", [-1, 2, float("-inf"), 1j])
def test_model_rejects_invalid_text_mask_before_normalization(flash_calls, invalid_value):
    model = build("flash_attention_3_hub")
    text_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, invalid_value, 0]])
    with pytest.raises(ValueError, match="boolean or 0/1 keep mask"):
        run_forward(model, text_mask)
    assert not flash_calls


@pytest.mark.parametrize("mask_shape", [(1, 5), (2, 4)])
def test_model_rejects_mismatched_text_mask_shape(flash_calls, mask_shape):
    model = build("flash_attention_3_hub")
    with pytest.raises(ValueError, match="text mask shape"):
        run_forward(model, torch.ones(mask_shape, dtype=torch.bool))
    assert not flash_calls


def run_forward(model, text_mask=None):
    """Full transformer forward through the public model entry point."""
    generator = torch.Generator().manual_seed(0)
    batch, text_len = 2, 5
    if text_mask is None:
        text_mask = torch.ones(batch, text_len, dtype=torch.long)
        text_mask[1, 3:] = 0
    return model(
        hidden_states=torch.randn(batch, 6, 16, generator=generator),
        encoder_hidden_states=torch.randn(batch, text_len, 16, generator=generator),
        encoder_hidden_states_mask=text_mask,
        timestep=torch.tensor([0.3, 0.7]),
        img_shapes=[[(1, 2, 3)]] * batch,
        return_dict=False,
    )[0]


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_varlen_metadata_is_computed_once_per_forward(flash_calls, monkeypatch, gradient_checkpointing):
    reference = build("eager")
    model = build("flash_attention_3_hub")
    model.load_state_dict(reference.state_dict())
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        reference.gradient_checkpointing_enable()
    derivations = []
    derive = qi_model._joint_varlen_metadata
    monkeypatch.setattr(qi_model, "_joint_varlen_metadata", lambda mask: derivations.append(mask) or derive(mask))

    output, ref_output = run_forward(model), run_forward(reference)
    torch.testing.assert_close(output, ref_output)
    assert len(flash_calls) == 2  # one per block
    output.square().sum().backward()
    ref_output.square().sum().backward()
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        assert (parameter.grad is None) == (ref_parameter.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, ref_parameter.grad)
    assert len(flash_calls) == (4 if gradient_checkpointing else 2)
    assert len(derivations) == 1  # also reused during checkpoint recomputation
