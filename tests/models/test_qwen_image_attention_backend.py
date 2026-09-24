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


def test_flash_path_accepts_integer_keep_masks():
    metadata = qi_model._joint_varlen_metadata(torch.tensor([[1, 1, 0, 1], [1, 0, 0, 0]]))
    assert metadata["indices"].tolist() == [0, 1, 3, 4]
    assert metadata["cu_seqlens"].tolist() == [0, 3, 4]
    assert metadata["max_seqlen"] == 3


@pytest.mark.parametrize("attn_implementation", ["flex_attention", "veomni_magi_attention_with_sp"])
def test_unsupported_attn_implementation_is_rejected(attn_implementation):
    with pytest.raises(ValueError, match="Qwen-Image does not support attn_implementation"):
        qi_model.QwenImageSPAttnProcessor(attn_implementation)


def run_attention(model, seed=0):
    """Joint attention on a batch where sample 1 has padded text in the middle of the joint sequence."""
    generator = torch.Generator().manual_seed(seed)
    batch, text_len, image_len, dim = 2, 5, 6, 16
    image = torch.randn(batch, image_len, dim, generator=generator, requires_grad=True)
    text = torch.randn(batch, text_len, dim, generator=generator)
    text_mask = torch.ones(batch, text_len, dtype=torch.bool)
    text_mask[1, 3:] = False
    joint_mask = torch.cat([text_mask, torch.ones(batch, image_len, dtype=torch.bool)], dim=1)
    attn = model.transformer_blocks[0].attn
    image_out, text_out = attn.processor(attn, image, encoder_hidden_states=text, attention_mask=joint_mask)
    return image, image_out, text_out, text_mask


@pytest.mark.parametrize("requested, resolved, kernel", FLASH_CASES)
def test_flash_path_matches_sdpa_on_valid_tokens(flash_calls, requested, resolved, kernel):
    reference = build("eager")
    model = build(requested)
    model.load_state_dict(reference.state_dict())

    ref_image, ref_image_out, ref_text_out, text_mask = run_attention(reference)
    image, image_out, text_out, _ = run_attention(model)

    (call,) = flash_calls
    assert call["attn_implementation"] == kernel
    assert call["mask"] is None  # explicit varlen metadata instead of per-layer unpadding from the mask
    assert call["cu_seq_lens_q"].tolist() == [0, 11, 20]  # sample 1: 3 valid text + 6 image tokens
    assert call["cu_seq_lens_k"].tolist() == [0, 11, 20]
    assert call["max_length_q"] == call["max_length_k"] == 11
    torch.testing.assert_close(image_out, ref_image_out)
    torch.testing.assert_close(text_out[text_mask], ref_text_out[text_mask])
    image_out.square().sum().backward()
    ref_image_out.square().sum().backward()
    torch.testing.assert_close(image.grad, ref_image.grad)


def run_forward(model):
    """Full transformer forward on a batch where sample 1 has padded text."""
    generator = torch.Generator().manual_seed(0)
    batch, text_len = 2, 5
    text_mask = torch.ones(batch, text_len, dtype=torch.long)
    text_mask[1, 3:] = 0
    return qi_model.QwenImageTransformer2DModel_forward(
        model,
        hidden_states=torch.randn(batch, 6, 16, generator=generator),
        encoder_hidden_states=torch.randn(batch, text_len, 16, generator=generator),
        encoder_hidden_states_mask=text_mask,
        timestep=torch.tensor([0.3, 0.7]),
        img_shapes=[[(1, 2, 3)]] * batch,
        return_dict=False,
    )[0]


def test_varlen_metadata_is_computed_once_per_forward(flash_calls, monkeypatch):
    reference = build("eager")
    model = build("flash_attention_3_hub")
    model.load_state_dict(reference.state_dict())
    derivations = []
    derive = qi_model._joint_varlen_metadata
    monkeypatch.setattr(qi_model, "_joint_varlen_metadata", lambda mask: derivations.append(mask) or derive(mask))

    torch.testing.assert_close(run_forward(model), run_forward(reference))
    assert len(flash_calls) == 2  # one per block
    assert len(derivations) == 1  # each derivation syncs with the host, so not per layer
