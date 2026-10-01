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

"""Qwen-Image models registry, condition, and Diffusers parity tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from diffusers import QwenImageTransformer2DModel as OfficialQwenImageTransformer2DModel
from torch import nn

from tests.models.compare import assert_outputs_and_grads_match, eager_ops_config, ops_config_scope
from tests.models.tiny_configs import tiny_qwen_image_condition_config as _tiny_condition_config
from tests.models.tiny_configs import tiny_qwen_image_config as _tiny_config


_OFFICIAL_FORWARD = OfficialQwenImageTransformer2DModel.forward


def _sdpa_ops_config() -> SimpleNamespace:
    """Portable Qwen-Image attention path. This family has no local eager forward."""
    ops = eager_ops_config()
    ops.attn_implementation = "sdpa"
    return ops


def test_qwen_image_condition_builds_from_registered_class_without_assets(monkeypatch):
    from veomni.models import MODELING_REGISTRY
    from veomni.models.diffusers.qwen_image.qwen_image_condition import modeling_qwen_image_condition

    monkeypatch.setattr(modeling_qwen_image_condition, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(modeling_qwen_image_condition, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0))
    monkeypatch.setattr(modeling_qwen_image_condition.QwenImageConditionModel, "_load_components", lambda self: None)

    model_class = MODELING_REGISTRY["QwenImageConditionModel"]()
    model = model_class._from_config(_tiny_condition_config())

    assert type(model) is modeling_qwen_image_condition.QwenImageConditionModel
    assert model.config.model_type == "QwenImageConditionModel"


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_qwen_image_public_forward_matches_official_without_sp(monkeypatch, training):
    """Compare public inference and ragged supervised batches with upstream math."""
    from veomni.models import get_model_class

    torch.manual_seed(0)
    config = _tiny_config()
    official = OfficialQwenImageTransformer2DModel(**config.to_diffuser_dict())
    monkeypatch.setattr(OfficialQwenImageTransformer2DModel, "forward", OfficialQwenImageTransformer2DModel.forward)
    with ops_config_scope(_sdpa_ops_config()):
        ours = get_model_class(config)(config)
    processor = ours.transformer_blocks[0].attn.processor
    assert processor.veomni_attn.impl == "sdpa"
    assert processor.veomni_attn_masked is processor.veomni_attn
    ours.load_state_dict(official.state_dict())
    samples = [
        {
            "hidden_states": torch.randn(1, height * width, 16, requires_grad=True),
            "encoder_hidden_states": torch.randn(1, text_len, 32, requires_grad=True),
            "encoder_hidden_states_mask": torch.tensor([[True] * (text_len - 1) + [False]]),
            "timestep": torch.tensor([timestep]),
            "img_shapes": [[(1, height, width)]],
        }
        for height, width, text_len, timestep in [(2, 2, 3, 0.25), (2, 3, 5, 0.75)]
    ]
    ours_samples = [
        {
            key: value.detach().clone().requires_grad_(value.requires_grad) if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        for sample in samples
    ]
    targets = [torch.randn_like(sample["hidden_states"]) + 2 * i for i, sample in enumerate(samples)]
    expected_predictions = []

    def call(model):
        if model is official:
            expected_predictions.extend(_OFFICIAL_FORWARD(model, **sample, return_dict=False)[0] for sample in samples)
            if not training:
                return tuple(expected_predictions)
            return torch.stack(
                [
                    F.mse_loss(prediction, target)
                    for prediction, target in zip(expected_predictions, targets, strict=True)
                ]
            ).mean()
        if training:
            output = model(
                **{key: [sample[key] for sample in ours_samples] for key in samples[0]},
                training_target=targets,
            )
            assert set(output.loss) == {"mse_loss"}
            predictions = output.predictions
        else:
            predictions = [model(**sample, return_dict=False)[0] for sample in ours_samples]
        assert len(predictions) == len(samples)
        for actual, expected in zip(predictions, expected_predictions, strict=True):
            torch.testing.assert_close(actual, expected)
        return output.loss["mse_loss"] if training else tuple(predictions)

    assert_outputs_and_grads_match(official, ours, call)
    for actual, expected in zip(ours_samples, samples, strict=True):
        for key in ("hidden_states", "encoder_hidden_states"):
            assert expected[key].grad is not None
            torch.testing.assert_close(actual[key].grad, expected[key].grad)


def test_qwen_image_joint_keep_mask_omits_all_valid_tokens():
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        _joint_keep_mask,
    )

    device = torch.device("cpu")
    all_true = torch.ones(2, 4, dtype=torch.bool)
    assert (
        _joint_keep_mask(
            batch_size=2,
            image_seq_len=8,
            txt_seq_len=4,
            device=device,
            encoder_hidden_states_mask=all_true,
        )
        is None
    )
    assert (
        _joint_keep_mask(
            batch_size=2,
            image_seq_len=8,
            txt_seq_len=4,
            device=device,
            encoder_hidden_states_mask=None,
        )
        is None
    )


def test_qwen_image_joint_keep_mask_keeps_padding_and_dropped_text():
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        _joint_keep_mask,
    )

    device = torch.device("cpu")
    text_mask = torch.tensor([[True, True, False]])
    dropped = _joint_keep_mask(
        batch_size=1,
        image_seq_len=4,
        txt_seq_len=3,
        device=device,
        encoder_hidden_states_mask=text_mask,
    )
    assert dropped is not None
    assert dropped.shape == (1, 7)
    assert dropped.tolist() == [[True, True, False, True, True, True, True]]

    padded = _joint_keep_mask(
        batch_size=1,
        image_seq_len=3,
        txt_seq_len=3,
        device=device,
        encoder_hidden_states_mask=None,
        img_pad=1,
        txt_pad=1,
    )
    assert padded is not None
    assert padded.shape == (1, 8)
    assert padded.tolist() == [[True, True, True, False, True, True, True, False]]


def _tiny_joint_attn(heads: int = 2, dim_head: int = 4) -> SimpleNamespace:
    """Bias-free projections only, so a zero raw attention row stays zero after the output projection."""
    inner = heads * dim_head
    return SimpleNamespace(
        heads=heads,
        inner=inner,
        to_q=nn.Linear(inner, inner, bias=False),
        to_k=nn.Linear(inner, inner, bias=False),
        to_v=nn.Linear(inner, inner, bias=False),
        add_q_proj=nn.Linear(inner, inner, bias=False),
        add_k_proj=nn.Linear(inner, inner, bias=False),
        add_v_proj=nn.Linear(inner, inner, bias=False),
        to_out=[nn.Linear(inner, inner, bias=False)],
        to_add_out=nn.Linear(inner, inner, bias=False),
        norm_q=None,
        norm_k=None,
        norm_added_q=None,
        norm_added_k=None,
        layer_idx=0,
    )


def _segmented_flash_forward(calls: list):
    """CPU stand-in for Transformers' ``_flash_attention_forward``: ``[B, S, H, D]`` in and out."""

    def forward(query, key, value, attention_mask, query_length, is_causal, softmax_scale=None, **kwargs):
        cu_seqlens = kwargs.get("cu_seq_lens_q")
        calls.append({"attention_mask": attention_mask, "cu_seqlens": cu_seqlens, **kwargs})
        assert attention_mask is None and not is_causal
        if cu_seqlens is None:
            q, k, v = (t.transpose(1, 2) for t in (query, key, value))
            return F.scaled_dot_product_attention(q, k, v, scale=softmax_scale).transpose(1, 2)
        assert query.shape[0] == 1 and torch.equal(cu_seqlens, kwargs["cu_seq_lens_k"])
        bounds = cu_seqlens.tolist()
        q, k, v = (t[0].transpose(0, 1) for t in (query, key, value))
        segments = [
            F.scaled_dot_product_attention(
                q[:, start:stop], k[:, start:stop], v[:, start:stop], scale=softmax_scale
            ).transpose(0, 1)
            for start, stop in zip(bounds[:-1], bounds[1:])
        ]
        assert max(stop - start for start, stop in zip(bounds[:-1], bounds[1:])) == kwargs["max_length_q"]
        return torch.cat(segments).unsqueeze(0)

    return forward


def _flash_ops_config() -> SimpleNamespace:
    ops = _sdpa_ops_config()
    ops.attn_implementation = "veomni_flash_attention_2"
    return ops


@pytest.mark.parametrize("checkpointing", [False, True])
def test_qwen_image_flash_keeps_configured_row_for_middle_text_padding(
    monkeypatch, available_nvidia_ops, checkpointing
):
    """A padded joint mask stays on the flash row and matches SDPA on every output and gradient.

    ``available_nvidia_ops`` passes the CUDA gates of the real registry rows; only the kernel
    below VeOmni's flash adapter is replaced by ``_segmented_flash_forward``.
    """
    from veomni.models import get_model_class
    from veomni.models.diffusers.qwen_image.qwen_image_transformer import modeling_qwen_image_transformer as modeling
    from veomni.ops.kernels.attention.standard import flash as flash_backend

    calls: list = []
    monkeypatch.setattr(flash_backend, "_flash_attention_forward", _segmented_flash_forward(calls))
    metadata_calls = []
    derive = modeling._joint_varlen_metadata
    monkeypatch.setattr(modeling, "_joint_varlen_metadata", lambda mask: metadata_calls.append(1) or derive(mask))

    torch.manual_seed(0)
    config = _tiny_config(num_layers=2)
    with ops_config_scope(_sdpa_ops_config()):
        reference = get_model_class(config)(config)
    with ops_config_scope(_flash_ops_config()):
        flash = get_model_class(config)(config)
    flash.load_state_dict(reference.state_dict())
    processor = flash.transformer_blocks[0].attn.processor
    assert processor.veomni_attn.impl == "veomni_flash_attention_2"
    assert processor.veomni_attn_masked is processor.veomni_attn
    if checkpointing:
        reference.gradient_checkpointing_enable()
        flash.gradient_checkpointing_enable()

    height, width, text_len = 2, 3, 5
    # Joint order is [text, image]: a dropped middle text token and trailing text padding both
    # sit in the middle of the joint sequence, not at a prefix-length boundary.
    text_mask = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 1, 0, 0]])
    hidden_states = torch.randn(2, height * width, 16)
    encoder_hidden_states = torch.randn(2, text_len, 32)
    weight = torch.randn(2, height * width, 16)

    def run(model):
        model.zero_grad(set_to_none=True)
        hs = hidden_states.clone().requires_grad_()
        enc = encoder_hidden_states.clone().requires_grad_()
        out = modeling.QwenImageTransformer2DModel_forward(
            model,
            hidden_states=hs,
            encoder_hidden_states=enc,
            encoder_hidden_states_mask=text_mask,
            timestep=torch.tensor([0.25, 0.75]),
            img_shapes=[[(1, height, width)]] * 2,
            return_dict=False,
        )[0]
        (out * weight).sum().backward()
        grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        return out, hs.grad, enc.grad, grads

    expected = run(reference)
    actual = run(flash)

    # Recomputation reuses the metadata derived once by the forward.
    assert len(calls) == config.num_layers * (1 + checkpointing)
    assert metadata_calls == [1]
    joint_len = text_len + height * width
    assert [call["cu_seqlens"].tolist() for call in calls] == [[0, joint_len - 1, 2 * joint_len - 3]] * len(calls)
    assert {call["attn_implementation"] for call in calls} == {"flash_attention_2"}
    for actual_tensor, expected_tensor in zip(actual[:3], expected[:3], strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    assert actual[3].keys() == expected[3].keys()
    for name, grad in expected[3].items():
        torch.testing.assert_close(actual[3][name], grad, msg=name)


def test_qwen_image_flash_processor_packs_mask_without_precomputed_metadata(monkeypatch, available_nvidia_ops):
    from veomni.models.diffusers.qwen_image.qwen_image_transformer import modeling_qwen_image_transformer as modeling
    from veomni.ops.kernels.attention.standard import flash as flash_backend

    calls: list = []
    monkeypatch.setattr(flash_backend, "_flash_attention_forward", _segmented_flash_forward(calls))
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    with ops_config_scope(_sdpa_ops_config()):
        sdpa = modeling.QwenImageSPAttnProcessor()
    with ops_config_scope(_flash_ops_config()):
        flash = modeling.QwenImageSPAttnProcessor()

    torch.manual_seed(0)
    attn = _tiny_joint_attn()
    hidden = torch.randn(2, 4, attn.inner)
    encoder = torch.randn(2, 3, attn.inner)
    mask = torch.tensor([[1, 0, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1]], dtype=torch.int64)

    expected_img, expected_txt = sdpa(attn, hidden, encoder, attention_mask=mask.bool())
    actual_img, actual_txt = flash(attn, hidden, encoder, attention_mask=mask)

    assert len(calls) == 1 and calls[0]["cu_seqlens"].tolist() == [0, 6, 12]
    torch.testing.assert_close(actual_img, expected_img)
    text_keep = mask[:, :3].bool()
    torch.testing.assert_close(actual_txt[text_keep], expected_txt[text_keep])
    assert not actual_txt[~text_keep].any()

    with pytest.raises(ValueError, match="Broadcast masks are not supported"):
        flash(attn, hidden, encoder, attention_mask=mask[:1])


def test_qwen_image_processor_uses_primary_handle_without_mask(monkeypatch):
    from veomni.models.diffusers.qwen_image.qwen_image_transformer import modeling_qwen_image_transformer as modeling

    monkeypatch.setattr(
        modeling,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=False, ulysses_group=None),
    )
    with ops_config_scope(_sdpa_ops_config()):
        processor = modeling.QwenImageSPAttnProcessor()

    used: dict[str, object] = {}

    def primary(_module, query, _key, _value, attention_mask=None, **_kwargs):
        used["handle"] = "primary"
        used["mask"] = attention_mask
        return query.transpose(1, 2), None

    def masked(_module, query, _key, _value, attention_mask=None, **_kwargs):
        used["handle"] = "masked"
        used["mask"] = attention_mask
        return query.transpose(1, 2), None

    processor.veomni_attn = primary
    processor.veomni_attn_masked = masked

    attn = _tiny_joint_attn()
    hidden = torch.randn(1, 4, attn.inner)
    encoder = torch.randn(1, 3, attn.inner)

    processor(attn, hidden, encoder, attention_mask=None)
    assert used["handle"] == "primary"
    assert used["mask"] is None

    processor(attn, hidden, encoder, attention_mask=torch.ones(1, 7, dtype=torch.bool))
    assert used["handle"] == "masked"
    assert used["mask"] is not None
