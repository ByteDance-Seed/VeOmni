"""Native tiny H3 packing contracts, without pretrained weights or encoders."""

import copy
from types import SimpleNamespace

import pytest
import torch

from veomni.models.diffusers.minimax_h3.minimax_h3_condition.configuration_minimax_h3_condition import (
    MiniMaxH3ConditionModelConfig,
)
from veomni.models.diffusers.minimax_h3.minimax_h3_condition.modeling_minimax_h3_condition import (
    MiniMaxH3ConditionModel,
)
from veomni.models.diffusers.minimax_h3.minimax_h3_core import core, minimax_h3_dit, packed_sequence
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer import (
    MiniMaxH3DiTModelConfig,
)
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import MiniMaxH3DiTModel
from veomni.models.diffusers.packing import DiffusionBatchOutput, validate_diffusion_remove_padding_support
from veomni.trainer.dit_trainer import DiTDataCollator


def tiny_model():
    return MiniMaxH3DiTModel(
        MiniMaxH3DiTModelConfig(
            hidden_size=32,
            num_layers=2,
            token_refiner_num_layers=1,
            num_attention_heads=2,
            attention_head_dim=16,
            ffn_hidden_size=64,
            text_dim=32,
            timestep_input_dim=16,
            time_embed_hidden_size=32,
            time_embed_dim=16,
            adaln_out_features=576,
            final_adaln_out_features=64,
            rope_inv_freq_len=2,
        )
    )


def condition_model():
    return MiniMaxH3ConditionModel(MiniMaxH3ConditionModelConfig(skip_encoder_load=True, num_train_timesteps=16))


def raw_sample(text_len=3, task="fl2va", refs=None):
    geometry = dict(text_len=text_len, latent_t=2, latent_h=4, latent_w=6, audio_t=3, audio_channel=2)
    if task == "ref2va":
        refs = refs or [{"kind": "image", "latent_t": 1, "latent_h": 4, "latent_w": 4}]
        pk = packed_sequence.build_packed_ref2va(**geometry, ref_blocks=refs)
        anchor_key = "ref_visual_anchor"
    else:
        pk = packed_sequence.build_packed_fl2va(**geometry, keyframe_indices=[0])
        anchor_key = "keyframe_cond_anchor"
    return dict(
        input_latents=torch.randn(1, 24, 2, 4, 6),
        audio_input_latents=torch.randn(2, 32, 3),
        prompt_embeds=torch.randn(text_len, 32),
        packed=pk,
        **{anchor_key: torch.randn(pk["cond_rows"], 96)},
        use_gradient_checkpointing=False,
    )


def prepare(condition, raws):
    return condition.prepare_samples(**DiTDataCollator()(raws))


def serial(model, samples):
    return [model(**s["model_inputs"], **s["targets"], **s["metadata"]) for s in samples]


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    monkeypatch.setattr(minimax_h3_dit, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(core, "ATTENTION_IMPLEMENTATION", "torch")
    monkeypatch.setattr(minimax_h3_dit, "get_ulysses_sequence_parallel_group", lambda: None)


def test_h3_opts_in_without_parameter_or_global_mutation():
    validate_diffusion_remove_padding_support(MiniMaxH3DiTModel, MiniMaxH3ConditionModel)
    model, other = tiny_model(), tiny_model()
    state = copy.deepcopy(model.state_dict())
    original_forward = type(model).forward
    model.configure_remove_padding(attn_implementation="eager")
    assert type(model).forward is original_forward
    assert not other.use_remove_padding
    assert state.keys() == model.state_dict().keys()
    for key in state:
        torch.testing.assert_close(state[key], model.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="backend"):
        model.configure_remove_padding(attn_implementation="sage_attention")


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_native_foundation_loader_accepts_reference_backend(backend):
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model

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
        tiny_model().config, init_device="cpu", torch_dtype="float32", ops_implementation=ops
    )
    assert isinstance(model, MiniMaxH3DiTModel)
    assert not model.use_remove_padding
    model.configure_remove_padding(attn_implementation=ops.attn_implementation)
    out = model(sample_inputs=prepare(condition_model(), [raw_sample()]))
    assert isinstance(out, DiffusionBatchOutput)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_preparation_preserves_single_sample_rng_and_weights(task):
    cond = condition_model()
    raws = [raw_sample(3, task), raw_sample(7, task)]
    torch.manual_seed(12)
    expected = [cond.process_condition(**DiTDataCollator()([r])) for r in raws]
    expected_rng = torch.get_rng_state()
    torch.manual_seed(12)
    actual = prepare(cond, raws)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for s, ref in zip(actual, expected):
        restored = s["model_inputs"] | s["targets"] | s["metadata"]
        for key, val in ref.items():
            if torch.is_tensor(val):
                torch.testing.assert_close(restored[key], val, rtol=0, atol=0)
        assert restored["t_video"] == ref["t_video"]
        assert restored["t_audio"] == ref["t_audio"]


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_packed_outputs_losses_and_gradients_match_serial(task, checkpointing):
    torch.manual_seed(7)
    base = tiny_model()
    packed = copy.deepcopy(base)
    packed.configure_remove_padding(attn_implementation="sdpa")
    raws = [raw_sample(3, task), raw_sample(7, task)]
    for r in raws:
        r["use_gradient_checkpointing"] = checkpointing
    samples = prepare(condition_model(), raws)
    expected = serial(base, samples)
    block_entries = {id(block): 0 for block in packed.dit.blocks}

    def record_block(module, inputs):
        block_entries[id(module)] += 1

    block_handles = [block.register_forward_pre_hook(record_block) for block in packed.dit.blocks]
    calls = []
    handle = packed.dit.register_forward_hook(lambda *a: calls.append(1))
    actual = packed(sample_inputs=samples)
    handle.remove()
    assert len(calls) == 1
    assert set(block_entries.values()) == {1}
    assert isinstance(actual, DiffusionBatchOutput)
    for i, ref in enumerate(expected):
        for key, value in zip(("video", "audio"), ref.predictions):
            torch.testing.assert_close(actual.sample_predictions[i][key], value, rtol=2e-5, atol=2e-5)
        for key, value in ref.loss.items():
            torch.testing.assert_close(actual.sample_losses[key][i], value, rtol=2e-5, atol=2e-5)
    sum(sum(o.loss.values()) for o in expected).div(len(samples)).backward()
    sum(actual.mean_losses(batch_size=len(samples)).values()).backward()
    assert set(block_entries.values()) == {2 if checkpointing else 1}
    for block_handle in block_handles:
        block_handle.remove()
    for (name, p), (_, q) in zip(base.named_parameters(), packed.named_parameters()):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)


def test_sample_isolation_boundaries_and_nonzero_padding():
    torch.manual_seed(9)
    model = tiny_model()
    model.configure_remove_padding(attn_implementation="eager")
    samples = prepare(condition_model(), [raw_sample(3), raw_sample(9)])
    expected = model(sample_inputs=samples)
    altered = copy.deepcopy(samples)
    altered[1]["model_inputs"]["prompt_embeds"].add_(100)
    for s in altered:
        inp = s["model_inputs"]
        used = int(inp["packed_seq_params"]["cu_seqlens_q"][1])
        inp["x"][:, used:] = 123
        inp["audio_x"][:, used:] = 123
    captured = []
    hook = model.dit.register_forward_pre_hook(lambda module, args, kwargs: captured.append(kwargs), with_kwargs=True)
    actual = model(sample_inputs=altered)
    hook.remove()
    for key in ("video", "audio"):
        torch.testing.assert_close(expected.sample_predictions[0][key], actual.sample_predictions[0][key])
    inp = captured[0]
    lengths = [int(s["model_inputs"]["packed_seq_params"]["cu_seqlens_q"][1]) for s in samples]
    assert inp["x"].shape[1] == sum(lengths)
    assert inp["packed_seq_params"]["cu_seqlens_q"].tolist() == [0, lengths[0], sum(lengths)]
    assert inp["refiner_packed_seq_params"]["cu_seqlens_q"].tolist() == [0, 3, 12]
    zero = copy.deepcopy(samples)
    for s in zero:
        s["model_inputs"]["x"].zero_()
        s["model_inputs"]["audio_x"].zero_()
    assert model(sample_inputs=zero).sample_predictions[0]["video"].shape == (1, 24, 2, 4, 6)


def test_ref2va_variable_reference_layouts_and_target_only_loss():
    refs = [
        {"kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 6},
        {"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 4},
    ]
    samples = prepare(condition_model(), [raw_sample(2, "ref2va"), raw_sample(11, "ref2va", refs)])
    model = tiny_model()
    expected = serial(model, samples)
    model.configure_remove_padding(attn_implementation="eager")
    actual = model(sample_inputs=samples)
    for i in range(2):
        torch.testing.assert_close(
            actual.sample_predictions[i]["video"], expected[i].predictions[0], rtol=2e-5, atol=2e-5
        )
        assert actual.sample_predictions[i]["audio"].shape == (2, 32, 3)


def test_invalid_condition_columns_and_audio_refs_fail_closed():
    cond = condition_model()
    rows = DiTDataCollator()([raw_sample(), raw_sample()])
    rows["prompt_embeds"] = rows["prompt_embeds"][:1]
    with pytest.raises(ValueError, match="length"):
        cond.prepare_samples(**rows)
    row = raw_sample(task="ref2va")
    row["ref_audio_anchor"] = torch.ones(2, 32)
    with pytest.raises(NotImplementedError, match="audio"):
        prepare(cond, [row])
    with pytest.raises(NotImplementedError, match="audio"):
        raw_sample(task="ref2va", refs=[{"kind": "audio", "ref_audio_t": 2}])


@pytest.mark.parametrize("audio_channel", [1, 2])
def test_visual_ref_layout_matches_native_inference(audio_channel):
    from veomni.models.diffusers.minimax_h3.inference import MiniMaxH3Unit_PackedSequenceBuilder

    refs = [
        {"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 4},
        {"kind": "video", "latent_t": 3, "latent_h": 6, "latent_w": 4},
    ]
    kwargs = dict(
        text_len=7, latent_t=2, latent_h=4, latent_w=6, audio_t=3, ref_blocks=refs, audio_channel=audio_channel
    )
    expected = MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_ref2va(**kwargs)
    actual = packed_sequence.build_packed_ref2va(**kwargs)
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=1e-12)
        else:
            assert actual[key] == value


@pytest.mark.parametrize("backend", ["veomni_flash_attention_2_with_sp", "veomni_flash_attention_3_with_sp"])
def test_fused_backend_receives_one_varlen_call_per_layer(monkeypatch, backend):
    from veomni.ops.kernels.attention import flash

    calls = []

    def kernel(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        assert cu_seqlens_q.dtype == torch.int32
        assert cu_seqlens_k is cu_seqlens_q
        assert cu_seqlens_q[-1] == q.shape[0]
        assert max_seqlen_q == max_seqlen_k
        assert not causal
        calls.append(cu_seqlens_q.tolist())
        return minimax_h3_dit._sdpa_varlen_attention(q, k, v, tuple(cu_seqlens_q.tolist()), softmax_scale, True)

    def loader(name):
        return SimpleNamespace(flash_attn_varlen_func=kernel)

    monkeypatch.setattr(flash, "_load_veomni_local_flash_kernel", loader)
    model = tiny_model().bfloat16()
    model.configure_remove_padding(attn_implementation=backend)
    raws = [raw_sample(3), raw_sample(7)]
    for row in raws:
        for key, value in row.items():
            if torch.is_tensor(value) and value.is_floating_point():
                row[key] = value.bfloat16()
    samples = prepare(condition_model(), raws)
    out = model(sample_inputs=samples)
    sum(out.mean_losses(batch_size=2).values()).backward()
    assert len(calls) == 3  # one refiner and two main-DiT blocks, not one call per sample
    assert calls[0] == [0, 3, 10]
    assert calls[1] == calls[2]
    assert calls[1] != calls[0]


def test_packing_rejects_heterogeneous_target_shapes():
    samples = prepare(condition_model(), [raw_sample(), raw_sample()])
    samples[1]["metadata"]["video_latent_shape"] = (3, 2, 3)
    model = tiny_model()
    model.configure_remove_padding(attn_implementation="eager")
    with pytest.raises(ValueError, match="fixed target"):
        model(sample_inputs=samples)


def test_missing_condition_rows_and_precision_casts_fail_closed():
    row = raw_sample()
    row["keyframe_cond_anchor"] = None
    with pytest.raises(ValueError, match="anchor rows"):
        prepare(condition_model(), [row])
    samples = prepare(condition_model(), [raw_sample()])
    samples[0]["model_inputs"]["unique_timesteps"] = samples[0]["model_inputs"]["unique_timesteps"].bfloat16()
    model = tiny_model()
    model.configure_remove_padding(attn_implementation="eager")
    with pytest.raises(ValueError, match="cast_forward_inputs"):
        model(sample_inputs=samples)


def test_disabled_requires_legacy_inputs():
    model = tiny_model()
    samples = prepare(condition_model(), [raw_sample()])
    with pytest.raises(ValueError, match="use_remove_padding"):
        model(sample_inputs=samples)
    assert len(serial(model, samples)[0].predictions) == 2
