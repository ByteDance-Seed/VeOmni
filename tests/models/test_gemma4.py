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

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel as HFGemma4Audio
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as HFGemma4

from tests.tools.gemma4_utils import make_gemma4_batch
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_foundation_model
from veomni.models.transformers.gemma4.generated import patched_modeling_gemma4_gpu as gemma4_modeling
from veomni.models.transformers.gemma4.generated.patched_modeling_gemma4_gpu import (
    Gemma4ForConditionalGeneration as VeOmniGemma4,
)
from veomni.trainer.callbacks.channel_loss_callback import ChannelLossComputer


_TOY_CONFIG = Path(__file__).parents[1] / "toy_config" / "gemma4_toy"
_FLEX_IMPLEMENTATION = "veomni_flex_attention"


def _eager_ops_config(attn_implementation: str) -> OpsImplementationConfig:
    return OpsImplementationConfig(
        attn_implementation=attn_implementation,
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        rotary_pos_emb_vision_implementation="eager",
        load_balancing_loss_implementation="eager",
    )


def _toy_config() -> Gemma4Config:
    return Gemma4Config.from_pretrained(_TOY_CONFIG)


@pytest.mark.parametrize("backend", ["eager", "sdpa", _FLEX_IMPLEMENTATION])
def test_gemma4_audio_matches_boolean_mask_reference(backend):
    """Audio mask visibility, outputs and gradients must not depend on the LM backend."""
    torch.manual_seed(31)
    config = _toy_config()
    config.audio_config.attention_context_left = 4
    config.audio_config.attention_context_right = 2
    model = VeOmniGemma4._from_config(config, attn_implementation=backend)
    audio = model.model.audio_tower
    reference = HFGemma4Audio._from_config(deepcopy(config.audio_config), attn_implementation="sdpa")
    reference.load_state_dict(audio.state_dict())
    masks = []
    reference_masks = []

    def record_mask(module, args, kwargs):
        masks.append(kwargs["attention_mask"])

    def record_reference_mask(module, args, kwargs):
        reference_masks.append(kwargs["attention_mask"])

    audio.layers[0].self_attn.register_forward_pre_hook(record_mask, with_kwargs=True)
    reference.layers[0].self_attn.register_forward_pre_hook(record_reference_mask, with_kwargs=True)
    features = torch.randn(1, 20, config.audio_config.subsampling_conv_channels[0], requires_grad=True)
    reference_features = features.detach().clone().requires_grad_()
    feature_mask = torch.ones(1, 20, dtype=torch.bool)
    feature_mask[:, -4:] = False

    output = audio(features, attention_mask=feature_mask)
    expected = reference(reference_features, attention_mask=feature_mask)

    assert masks[0].dtype == torch.bool
    torch.testing.assert_close(masks[0], reference_masks[0])
    torch.testing.assert_close(output.attention_mask, expected.attention_mask)
    torch.testing.assert_close(output.last_hidden_state, expected.last_hidden_state)
    output.last_hidden_state[output.attention_mask].square().sum().backward()
    expected.last_hidden_state[expected.attention_mask].square().sum().backward()
    torch.testing.assert_close(features.grad, reference_features.grad)
    torch.testing.assert_close(
        audio.layers[0].self_attn.q_proj.linear.weight.grad, reference.layers[0].self_attn.q_proj.linear.weight.grad
    )


@pytest.mark.parametrize("entry", ["image", "video", "tower"])
def test_gemma4_vision_rejects_ulysses_before_tower_computation(monkeypatch, entry):
    """Unadapted vision layouts fail explicitly before any FSDP-sharded vision block."""
    model = VeOmniGemma4._from_config(_toy_config(), attn_implementation="eager")
    monkeypatch.setattr(
        gemma4_modeling, "get_parallel_state", lambda: SimpleNamespace(ulysses_enabled=True), raising=False
    )
    calls = []
    model.model.vision_tower.patch_embedder.register_forward_pre_hook(lambda *args: calls.append("patch_embedder"))
    positions = torch.cartesian_prod(torch.arange(4), torch.arange(4)).unsqueeze(0)
    pixels = torch.randn(1, 16, 48)

    with pytest.raises(NotImplementedError, match="Gemma 4 vision.*Ulysses"):
        if entry == "image":
            model.get_image_features(pixels, positions)
        elif entry == "video":
            model.model.get_video_features(pixels.unsqueeze(1), positions.unsqueeze(1))
        else:
            model.model.vision_tower(pixel_values=pixels, pixel_position_ids=positions)
    assert calls == []


@pytest.mark.parametrize(
    "modalities",
    [
        (),
        ("image",),
        ("video",),
        ("audio",),
        ("image", "video"),
        ("image", "audio"),
        ("video", "audio"),
        ("image", "video", "audio"),
    ],
)
def test_gemma4_fsdp_dummy_paths_preserve_outputs_and_gradients(monkeypatch, modalities):
    """Every modality combination visits the same tower order and retains zero-grad dependencies."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    model = build_foundation_model(
        _TOY_CONFIG, torch_dtype="float32", init_device="cpu", ops_implementation=_eager_ops_config("eager")
    )
    batch = make_gemma4_batch(model.config, modalities)
    with torch.no_grad():
        expected = model(**batch)
    calls = []
    model.model.vision_tower.register_forward_pre_hook(lambda *args: calls.append("vision"))
    model.model.audio_tower.register_forward_pre_hook(lambda *args: calls.append("audio"))
    monkeypatch.setattr(
        gemma4_modeling,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_enabled=True, ulysses_enabled=False),
    )

    output = model(**batch)
    output.loss.backward()

    assert calls == ["vision", "vision", "audio"]
    torch.testing.assert_close(output.logits, expected.logits)
    torch.testing.assert_close(output.loss, expected.loss)
    for prefix, active in (
        ("vision_tower", "image" in modalities or "video" in modalities),
        ("audio_tower", "audio" in modalities),
        ("embed_vision", "image" in modalities or "video" in modalities),
        ("embed_audio", "audio" in modalities),
    ):
        for name, parameter in getattr(model.model, prefix).named_parameters():
            assert parameter.grad is not None, f"Missing gradient for {prefix}.{name}"
            assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient for {prefix}.{name}"
            if not active:
                assert torch.count_nonzero(parameter.grad) == 0, f"Nonzero dummy gradient for {prefix}.{name}"


@pytest.mark.parametrize("tower_name", ["vision_tower", "audio_tower"])
def test_gemma4_dummy_inputs_follow_live_parameter_dtype(tower_name):
    """Repeated dummy calls follow compute-weight casts instead of cached input dtype."""
    model = VeOmniGemma4._from_config(_toy_config(), attn_implementation="eager")
    tower = getattr(model.model, tower_name)
    with torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16, torch.float32):
            tower.to(dtype=dtype)
            output = tower.dummy_forward()
            assert output.last_hidden_state.dtype == dtype
            assert torch.isfinite(output.last_hidden_state).all()


@pytest.mark.parametrize("fsdp_enabled", [False, True])
def test_gemma4_conditional_ulysses_rejection_is_rank_symmetric(monkeypatch, fsdp_enabled):
    """A text-only peer rejects unsupported vision SP before entering language collectives."""
    model = VeOmniGemma4._from_config(_toy_config(), attn_implementation="eager")
    monkeypatch.setattr(
        gemma4_modeling,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_enabled=fsdp_enabled, ulysses_enabled=True),
    )
    calls = []
    for name in ("vision_tower", "audio_tower", "language_model"):
        getattr(model.model, name).register_forward_pre_hook(lambda *args: calls.append("tower"))
    with pytest.raises(NotImplementedError, match="Gemma 4 vision.*Ulysses"):
        model(input_ids=torch.tensor([[3, 4, 5, 6]]), use_cache=False)
    assert calls == []


def test_gemma4_state_dict_matches_huggingface():
    config = _toy_config()
    with torch.device("meta"):
        hf_model = HFGemma4(config)
        veomni_model = VeOmniGemma4(config)

    hf_state = {name: tuple(tensor.shape) for name, tensor in hf_model.state_dict().items()}
    veomni_state = {name: tuple(tensor.shape) for name, tensor in veomni_model.state_dict().items()}
    assert veomni_state == hf_state


def test_gemma4_eager_training(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torch.manual_seed(17)
    model = build_foundation_model(
        _TOY_CONFIG,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_eager_ops_config("eager"),
    )
    input_ids = torch.randint(3, model.config.text_config.vocab_size, (1, 8))

    output = model(input_ids=input_ids, labels=input_ids.clone(), logits_to_keep=1, use_cache=False)
    output.loss.backward()

    assert output.logits.shape == (1, 8, model.config.text_config.vocab_size)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(model.model.language_model.layers[0].self_attn.q_proj.weight.grad).all()


def test_gemma4_text_eager_training_ignores_logits_to_keep_with_labels(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    config = _toy_config().text_config
    config.save_pretrained(tmp_path)
    model = build_foundation_model(
        tmp_path,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_eager_ops_config("eager"),
    )
    input_ids = torch.randint(3, config.vocab_size, (1, 8))

    output = model(input_ids=input_ids, labels=input_ids.clone(), logits_to_keep=1, use_cache=False)
    output.loss.backward()

    assert output.logits.shape == (1, 8, config.vocab_size)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all()


@pytest.mark.parametrize("text_only", [False, True], ids=["multimodal", "text"])
@pytest.mark.parametrize("loss_implementation", ["eager", "chunk_loss"])
@pytest.mark.parametrize("softcap", [None, 30.0], ids=["uncapped", "softcapped"])
def test_gemma4_training_loss_matches_capped_logits(monkeypatch, tmp_path, text_only, loss_implementation, softcap):
    """A non-eager loss selection must preserve softcapped loss and gradients."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torch.manual_seed(23)
    config = _toy_config()
    config.text_config.final_logit_softcapping = softcap
    if text_only:
        config = config.text_config
    config.save_pretrained(tmp_path)
    ops_config = _eager_ops_config("eager")
    # chunk_loss exercises real non-eager dispatch on CPU without CUDA mocks.
    ops_config.cross_entropy_loss_implementation = loss_implementation
    model = build_foundation_model(tmp_path, torch_dtype="float32", init_device="cpu", ops_implementation=ops_config)
    with torch.no_grad():
        model.lm_head.weight.mul_(100)
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    labels = input_ids.clone()
    labels[:, 3] = -100
    backbone = model.model if text_only else model.model.language_model
    hidden_states = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    raw_logits = model.lm_head(hidden_states)
    expected_logits = raw_logits if softcap is None else torch.tanh(raw_logits / softcap) * softcap
    expected_loss = F.cross_entropy(
        expected_logits[:, :-1].reshape(-1, expected_logits.shape[-1]), labels[:, 1:].reshape(-1)
    )
    parameters = (model.lm_head.weight, backbone.layers[0].self_attn.q_proj.weight)
    expected_gradients = torch.autograd.grad(expected_loss, parameters)

    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    output.loss.backward()

    torch.testing.assert_close(output.loss, expected_loss)
    for parameter, expected_gradient in zip(parameters, expected_gradients):
        torch.testing.assert_close(parameter.grad, expected_gradient, atol=1e-5, rtol=1e-4)
    if softcap is not None:
        assert not torch.allclose(expected_logits, raw_logits)
        torch.testing.assert_close(output.logits, expected_logits)
    elif loss_implementation == "chunk_loss":
        assert output.logits is None


@pytest.mark.parametrize("text_only", [False, True], ids=["multimodal", "text"])
@pytest.mark.parametrize("loss_implementation", ["eager", "chunk_loss"])
@pytest.mark.parametrize("softcap", [None, 30.0], ids=["uncapped", "softcapped"])
def test_gemma4_softcapping_preserves_channel_loss(monkeypatch, tmp_path, text_only, loss_implementation, softcap):
    """Channel totals observe the selected instance loss, including softcapping."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    config = _toy_config()
    config.text_config.final_logit_softcapping = softcap
    if text_only:
        config = config.text_config
    config.save_pretrained(tmp_path)
    ops = _eager_ops_config("eager")
    ops.cross_entropy_loss_implementation = loss_implementation
    model = build_foundation_model(tmp_path, torch_dtype="float32", init_device="cpu", ops_implementation=ops)
    with torch.no_grad():
        model.lm_head.weight.mul_(100)
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    labels = input_ids.clone()
    labels[:, 3] = -100
    reference = model(input_ids=input_ids, labels=labels, use_cache=False)
    reference_gradient = torch.autograd.grad(reference.loss, model.lm_head.weight)[0]
    computer = ChannelLossComputer()
    computer.install(model)
    computer.begin_step([[42]], [torch.arange(8).unsqueeze(0)], [torch.ones_like(input_ids)])
    computer.before_micro_step()
    try:
        with computer.capture():
            output = model(input_ids=input_ids, labels=labels, use_cache=False)
        computer.after_micro_step()
        output.loss.backward()

        assert 42 in computer.step_totals
        loss_sum, token_count = computer.step_totals[42]
        expected_sum = reference.loss.detach() * (labels[:, 1:] != -100).sum()
        assert token_count.item() == 6
        torch.testing.assert_close(loss_sum, expected_sum)
        torch.testing.assert_close(output.loss, reference.loss)
        torch.testing.assert_close(model.lm_head.weight.grad, reference_gradient)
    finally:
        computer.uninstall()


@pytest.mark.parametrize("text_only", [False, True], ids=["multimodal", "text"])
def test_gemma4_softcapping_rejects_uncapped_log_probs(monkeypatch, tmp_path, text_only):
    """The fused log-probs path cannot silently bypass logit softcapping."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    config = _toy_config()
    config.text_config.final_logit_softcapping = 30.0
    if text_only:
        config = config.text_config
    config.save_pretrained(tmp_path)
    model = build_foundation_model(
        tmp_path,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_eager_ops_config("eager"),
    )
    input_ids = torch.tensor([[3, 4, 5, 6]])

    with pytest.raises(ValueError, match="softcapping.*return_log_probs"):
        model(input_ids=input_ids, labels=input_ids.clone(), return_log_probs=True, use_cache=False)


@pytest.mark.parametrize("has_features,has_mask", [(False, False), (True, False), (False, True)])
def test_gemma4_incomplete_audio_inputs_return_no_audio_states(monkeypatch, has_features, has_mask):
    """Audio states are absent when either encoder input is missing."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    model = build_foundation_model(
        _TOY_CONFIG,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=_eager_ops_config("eager"),
    ).eval()
    input_ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False)
        output = model(
            input_ids=input_ids,
            input_features=torch.randn(1, 16, 128) if has_features else None,
            input_features_mask=torch.ones(1, 16, dtype=torch.bool) if has_mask else None,
            use_cache=False,
        )

    assert output.audio_hidden_states is None
    torch.testing.assert_close(output.logits, expected.logits)


def test_gemma4_builds_pack_aware_full_and_sliding_masks(monkeypatch):
    captured = []

    def fake_flex_adapter(module, query, key, value, attention_mask, **kwargs):
        captured.append((module.layer_type, attention_mask))
        return torch.zeros_like(query).transpose(1, 2), None

    monkeypatch.setitem(
        ALL_ATTENTION_FUNCTIONS._global_mapping,
        _FLEX_IMPLEMENTATION,
        fake_flex_adapter,
    )
    config = _toy_config()
    model = VeOmniGemma4._from_config(config, attn_implementation=_FLEX_IMPLEMENTATION).eval()
    input_ids = torch.randint(3, config.text_config.vocab_size, (1, 8))
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    cu_seq_lens_q = torch.tensor([0, 3, 8], dtype=torch.int32)

    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            use_cache=False,
        )

    assert [layer_type for layer_type, _ in captured] == ["sliding_attention", "full_attention"]
    zero = torch.tensor(0)
    first_token_in_second_pack = torch.tensor(3)
    last_token_in_first_pack = torch.tensor(2)
    for _, block_mask in captured:
        assert isinstance(block_mask, BlockMask)
        assert not block_mask.mask_mod(zero, zero, first_token_in_second_pack, last_token_in_first_pack)


def test_gemma4_multimodal_masks_preserve_packed_boundaries(monkeypatch):
    captured = {}

    def fake_flex_adapter(module, query, key, value, attention_mask, **kwargs):
        captured[module.layer_type] = attention_mask
        return torch.zeros_like(query).transpose(1, 2), None

    monkeypatch.setitem(
        ALL_ATTENTION_FUNCTIONS._global_mapping,
        _FLEX_IMPLEMENTATION,
        fake_flex_adapter,
    )
    config = _toy_config()
    config.text_config.use_bidirectional_attention = "vision"
    model = VeOmniGemma4._from_config(config, attn_implementation=_FLEX_IMPLEMENTATION).eval()
    input_ids = torch.randint(3, config.text_config.vocab_size, (1, 8))
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    mm_token_type_ids = torch.tensor([[0, 1, 1, 0, 0, 1, 1, 0]])
    cu_seq_lens_q = torch.tensor([0, 4, 8], dtype=torch.int32)

    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            use_cache=False,
        )

    zero = torch.tensor(0)
    for block_mask in captured.values():
        assert isinstance(block_mask, BlockMask)
        assert not block_mask.mask_mod(zero, zero, torch.tensor(5), torch.tensor(1))
    assert captured["sliding_attention"].mask_mod(zero, zero, torch.tensor(1), torch.tensor(2))


def test_gemma4_generated_children_and_packed_masks(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    config = _toy_config()
    model_class = VeOmniGemma4
    model = model_class._from_config(config, attn_implementation="eager")

    assert type(model.model.vision_tower).__module__ == model_class.__module__
    assert type(model.model.audio_tower).__module__ == model_class.__module__
    assert type(model.model.language_model).__module__ == model_class.__module__

    modeling_module = gemma4_modeling
    seen_cu_seq_lens = []

    def capture_mask(**kwargs):
        seen_cu_seq_lens.append(kwargs.get("cu_seq_lens_q"))
        return None

    monkeypatch.setattr(modeling_module, "_gemma4_create_mask", capture_mask)

    input_ids = torch.tensor([[2, 7, 11, 13, 17, 1]])
    cu_seq_lens_q = torch.tensor([0, 3, 6], dtype=torch.int32)
    model(input_ids=input_ids, cu_seq_lens_q=cu_seq_lens_q, use_cache=False)

    assert len(seen_cu_seq_lens) == 2
    assert all(torch.equal(value, cu_seq_lens_q) for value in seen_cu_seq_lens)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("vision_blocks", [False, True])
def test_gemma4_packed_outputs_match_independent_sequences(backend, vision_blocks):
    """Public packed builders isolate samples for both dense attention backends."""
    config = _toy_config()
    config.text_config.use_bidirectional_attention = "vision" if vision_blocks else None
    model = VeOmniGemma4._from_config(config, attn_implementation=backend).eval()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    token_types = torch.tensor([[0, 1, 1, 0, 0, 1, 1, 0]]) if vision_blocks else None
    with torch.no_grad():
        packed = model(
            input_ids=ids,
            position_ids=torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            cu_seq_lens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
            mm_token_type_ids=token_types,
            use_cache=False,
        ).logits
        independent = torch.cat(
            [
                model(
                    input_ids=ids[:, start : start + 4],
                    mm_token_type_ids=token_types[:, start : start + 4] if token_types is not None else None,
                    use_cache=False,
                ).logits
                for start in (0, 4)
            ],
            dim=1,
        )
    torch.testing.assert_close(packed, independent)


@pytest.mark.parametrize("text_only", [False, True], ids=["multimodal", "text"])
@pytest.mark.parametrize("loss_implementation", ["eager", "chunk_loss"])
@pytest.mark.parametrize("topk", [False, True], ids=["log_probs", "topk"])
def test_gemma4_uncapped_auxiliary_outputs(monkeypatch, tmp_path, text_only, loss_implementation, topk):
    """Uncapped auxiliary outputs retain hidden states even for eager CE selection."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    config = _toy_config()
    config.text_config.final_logit_softcapping = None
    if text_only:
        config = config.text_config
    config.save_pretrained(tmp_path)
    ops = _eager_ops_config("eager")
    ops.cross_entropy_loss_implementation = loss_implementation
    model = build_foundation_model(tmp_path, torch_dtype="float32", init_device="cpu", ops_implementation=ops)
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    labels = input_ids.clone()
    labels[:, 3] = -100
    logits = model(input_ids=input_ids, use_cache=False).logits
    log_distribution = logits[:, :-1].float().log_softmax(-1)
    targets = labels[:, 1:]
    valid = targets != -100
    expected_log_probs = log_distribution.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    expected_entropy = -(log_distribution.exp() * log_distribution).sum(-1)
    expected_log_probs = expected_log_probs.masked_fill(~valid, 0)
    expected_entropy = expected_entropy.masked_fill(~valid, 0)
    kwargs = {}
    if topk:
        teacher_log_probs = torch.randn_like(logits).log_softmax(-1)
        teacher_values, teacher_ids = teacher_log_probs.topk(4, dim=-1)
        kwargs = dict(teacher_topk_ids=teacher_ids, teacher_topk_log_probs=teacher_values)
        student_values = log_distribution.gather(-1, teacher_ids[:, 1:])
        expected_distill = (
            (teacher_values[:, 1:].exp() * (teacher_values[:, 1:] - student_values)).sum(-1).masked_fill(~valid, 0)
        )
        objective = expected_distill.sum()
    else:
        objective = expected_log_probs.sum()
    expected_gradient = torch.autograd.grad(objective, model.lm_head.weight)[0]
    output = model(input_ids=input_ids, labels=labels, return_log_probs=True, chunk_size=2, use_cache=False, **kwargs)
    assert output.loss is None
    assert output.logits is None
    aux = output.fused_linear_aux
    torch.testing.assert_close(aux.log_probs, F.pad(expected_log_probs, (0, 1)))
    torch.testing.assert_close(aux.entropy, F.pad(expected_entropy, (0, 1)))
    if topk:
        torch.testing.assert_close(aux.distillation_losses, F.pad(expected_distill, (0, 1)))
        aux.distillation_losses.sum().backward()
    else:
        aux.log_probs.sum().backward()
    torch.testing.assert_close(model.lm_head.weight.grad, expected_gradient, atol=1e-6, rtol=1e-4)
