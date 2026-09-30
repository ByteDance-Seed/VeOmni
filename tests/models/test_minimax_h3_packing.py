"""Native H3 model-owned packing, without pretrained weights or encoders."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

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
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
    MiniMaxH3DiTModel,
    MiniMaxH3DiTOutput,
)
from veomni.trainer.dit_trainer import DiTDataCollator
from veomni.utils.import_utils import is_torch_npu_available


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


def raw_sample(text_len=3, task="fl2va", refs=None, latent_t=2, latent_h=4, latent_w=6, audio_t=3, keyframes=(0,)):
    geometry = dict(
        text_len=text_len, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, audio_t=audio_t, audio_channel=2
    )
    if task == "ref2va":
        refs = refs or [{"kind": "image", "latent_t": 1, "latent_h": 4, "latent_w": 4}]
        pk = packed_sequence.build_packed_ref2va(**geometry, ref_blocks=refs)
        anchor_key = "ref_visual_anchor"
    else:
        pk = packed_sequence.build_packed_fl2va(**geometry, keyframe_indices=list(keyframes))
        anchor_key = "keyframe_cond_anchor"
    row = dict(
        input_latents=torch.randn(1, 24, latent_t, latent_h, latent_w),
        audio_input_latents=torch.randn(2, 32, audio_t),
        prompt_embeds=torch.randn(text_len, 32),
        packed=pk,
        use_gradient_checkpointing=False,
    )
    if pk["cond_rows"]:
        row[anchor_key] = torch.randn(pk["cond_rows"], 96)
    return row


def legacy_tail(sample):
    """Append the pre-#1204 64-row sample tail to a prepared single sample."""
    legacy = copy.deepcopy(sample)
    length = sample["x"].shape[1]
    pad = (-length) % 64
    for key in ("x", "audio_x", "img_position_ids"):
        legacy[key] = F.pad(legacy[key], (0, 0, 0, pad))
    legacy["token_tags"] = F.pad(legacy["token_tags"], (0, pad), value=-1)
    legacy["inverse_indices"] = F.pad(legacy["inverse_indices"], (0, pad))
    legacy["packed_seq_params"]["cu_seqlens_q"] = torch.tensor([0, length, length + pad], dtype=torch.int32)
    legacy["packed_seq_params"]["cu_seqlens_host"] = (0, length, length + pad)
    return legacy


def prepare(condition, raws):
    columns = condition.process_condition(**DiTDataCollator()(raws))
    if len(raws) == 1:
        return [columns]
    assert all(isinstance(value, list) and len(value) == len(raws) for value in columns.values())
    return [{key: value[i] for key, value in columns.items()} for i in range(len(raws))]


def batch(samples):
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def serial(model, samples):
    return [model(**sample) for sample in samples]


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    monkeypatch.setattr(minimax_h3_dit, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(core, "ATTENTION_IMPLEMENTATION", "torch")
    monkeypatch.setattr(minimax_h3_dit, "get_ulysses_sequence_parallel_group", lambda: None)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_native_foundation_loader_uses_ordinary_batch_contract(backend):
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
    keys = set(model.state_dict())
    columns = condition_model().process_condition(**DiTDataCollator()([raw_sample(), raw_sample(7)]))
    out = model(**columns)
    assert isinstance(out, MiniMaxH3DiTOutput)
    assert all(value.ndim == 0 for value in out.loss.values())
    assert [p.shape for p in out.predictions[0]] == [(1, 24, 2, 4, 6)] * 2
    assert keys == set(model.state_dict())


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
    for sample, ref in zip(actual, expected):
        for key, val in ref.items():
            if torch.is_tensor(val):
                torch.testing.assert_close(sample[key], val, rtol=0, atol=0)
        assert sample["t_video"] == ref["t_video"]
        assert sample["t_audio"] == ref["t_audio"]


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_packed_outputs_losses_and_gradients_match_serial(task, checkpointing):
    torch.manual_seed(7)
    base = tiny_model()
    packed = copy.deepcopy(base)
    raws = [raw_sample(3, task), raw_sample(7, task)]
    for row in raws:
        row["use_gradient_checkpointing"] = checkpointing
    samples = prepare(condition_model(), raws)
    expected = serial(base, samples)
    entries = {id(block): 0 for block in packed.dit.blocks}

    def record(module, inputs):
        entries[id(module)] += 1

    handles = [block.register_forward_pre_hook(record) for block in packed.dit.blocks]
    calls = []
    handle = packed.dit.register_forward_hook(lambda *args: calls.append(1))
    actual = packed(**batch(samples))
    handle.remove()
    assert len(calls) == 1
    assert set(entries.values()) == {1}
    for i, ref in enumerate(expected):
        torch.testing.assert_close(actual.predictions[0][i], ref.predictions[0], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual.predictions[1][i], ref.predictions[1], rtol=2e-5, atol=2e-5)
    for key in actual.loss:
        torch.testing.assert_close(actual.loss[key], torch.stack([out.loss[key] for out in expected]).mean())
    sum(sum(out.loss.values()) for out in expected).div(len(samples)).backward()
    sum(actual.loss.values()).backward()
    assert set(entries.values()) == {2 if checkpointing else 1}
    for handle in handles:
        handle.remove()
    for (name, p), (_, q) in zip(base.named_parameters(), packed.named_parameters()):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_mixed_target_geometry_matches_serial(task):
    torch.manual_seed(11)
    base = tiny_model()
    packed = copy.deepcopy(base)
    raws = [
        raw_sample(3, task),
        raw_sample(7, task, latent_t=3, latent_h=6, latent_w=4, audio_t=5),
    ]
    samples = prepare(condition_model(), raws)
    expected = serial(base, samples)
    actual = packed(**batch(samples))
    assert [p.shape for p in actual.predictions[0]] == [(1, 24, 2, 4, 6), (1, 24, 3, 6, 4)]
    assert [p.shape for p in actual.predictions[1]] == [(2, 32, 3), (2, 32, 5)]
    for i, ref in enumerate(expected):
        for a, b in zip((actual.predictions[0][i], actual.predictions[1][i]), ref.predictions):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    for key in actual.loss:
        torch.testing.assert_close(actual.loss[key], torch.stack([out.loss[key] for out in expected]).mean())
    sum(sum(out.loss.values()) for out in expected).div(len(samples)).backward()
    sum(actual.loss.values()).backward()
    for (name, p), (_, q) in zip(base.named_parameters(), packed.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)


def test_mixed_tasks_and_keyframes_match_serial():
    torch.manual_seed(13)
    base = tiny_model()
    packed = copy.deepcopy(base)
    raws = [raw_sample(3), raw_sample(5, "ref2va"), raw_sample(4, keyframes=())]
    collated = DiTDataCollator()(raws)
    assert [anchor is None for anchor in collated["keyframe_cond_anchor"]] == [False, True, True]
    assert [anchor is None for anchor in collated["ref_visual_anchor"]] == [True, False, True]
    samples = prepare(condition_model(), raws)
    assert [sample["cond_rows"] for sample in samples][2] == 0
    expected = serial(base, samples)
    actual = packed(**batch(samples))
    for i, ref in enumerate(expected):
        for a, b in zip((actual.predictions[0][i], actual.predictions[1][i]), ref.predictions):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    for key in actual.loss:
        torch.testing.assert_close(actual.loss[key], torch.stack([out.loss[key] for out in expected]).mean())
    sum(sum(out.loss.values()) for out in expected).div(len(samples)).backward()
    sum(actual.loss.values()).backward()
    for (name, p), (_, q) in zip(base.named_parameters(), packed.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)


def test_get_condition_flags_silent_audio_placeholder(monkeypatch):
    cond = MiniMaxH3ConditionModel(
        MiniMaxH3ConditionModelConfig(skip_encoder_load=True, num_train_timesteps=16, use_keyframe_condition=False)
    )
    cond._video_vae = torch.nn.Linear(1, 1)
    monkeypatch.setattr(cond, "_encode_text", lambda prompt, images, device: (torch.randn(3, 32), None))
    monkeypatch.setattr(cond, "_encode_video", lambda video, device: torch.randn(1, 24, 2, 4, 6))
    monkeypatch.setattr(cond, "_encode_audio", lambda audio, frames, device: torch.randn(2, 32, 3))
    monkeypatch.setattr(cond, "_make_silent_audio_latent", lambda frames, device: torch.zeros(2, 32, 3))
    frames = [torch.zeros(3, 8, 8)] * 22
    out = cond.get_condition(inputs=["a", "b"], videos=[frames, frames], audios=[(torch.zeros(1, 8), 16000), None])
    assert out["has_audio"] == [True, False]


def test_audio_loss_skips_samples_without_audio():
    torch.manual_seed(17)
    base = tiny_model()
    packed = copy.deepcopy(base)
    raws = [raw_sample(3), raw_sample(5), raw_sample(4, "ref2va")]
    raws[1]["has_audio"] = False
    samples = prepare(condition_model(), raws)
    assert [sample["has_audio"] for sample in samples] == [True, False, True]
    expected = serial(base, samples)
    assert expected[1].loss["mse_audio"] == 0
    actual = packed(**batch(samples))
    torch.testing.assert_close(actual.loss["mse_video"], torch.stack([o.loss["mse_video"] for o in expected]).mean())
    torch.testing.assert_close(actual.loss["mse_audio"], torch.stack([o.loss["mse_audio"] for o in expected]).mean())
    (sum(o.loss["mse_video"] + o.loss["mse_audio"] for o in expected) / 3).backward()
    sum(actual.loss.values()).backward()
    for (name, p), (_, q) in zip(base.named_parameters(), packed.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)

    silent = [raw_sample(3), raw_sample(5)]
    for row in silent:
        row["has_audio"] = False
    silent_samples = prepare(condition_model(), silent)
    for out in (base(**silent_samples[0]), base(**batch(silent_samples))):
        assert out.loss["mse_audio"] == 0 and out.loss["mse_audio"].requires_grad


def test_accumulated_microbatches_weight_each_sample_like_single_sample():
    """With the trainer's /K, packed microbatches weight every sample 1/G whatever their audio mix."""
    torch.manual_seed(19)
    model = tiny_model()
    raws = [raw_sample(3), raw_sample(5), raw_sample(4, "ref2va"), raw_sample(6)]
    raws[1]["has_audio"] = False  # microbatches [audio, no-audio] and [audio, audio]
    samples = prepare(condition_model(), raws)
    reference = sum(sum(out.loss.values()) for out in serial(model, samples)) / len(samples)
    accumulated = sum(sum(model(**batch(part)).loss.values()) / 2 for part in (samples[:2], samples[2:]))
    torch.testing.assert_close(accumulated, reference)


def test_unsupported_backend_fails_only_on_packed_forward():
    model = tiny_model()
    model._configure_packed_attention("veomni_flash_attention_4_with_sp")  # what __init__ runs; must not raise
    samples = prepare(condition_model(), [raw_sample(3), raw_sample(5)])
    serial(model, samples)
    with pytest.raises(ValueError, match="Unsupported H3 packing backend"):
        model(**batch(samples))


def test_sample_isolation_boundaries_and_zero_valid_rows():
    model = tiny_model()
    samples = prepare(condition_model(), [raw_sample(3), raw_sample(9)])
    expected = model(**batch(samples))
    altered = copy.deepcopy(samples)
    altered[1]["prompt_embeds"].add_(100)
    captured = []
    hook = model.dit.register_forward_pre_hook(lambda module, args, kwargs: captured.append(kwargs), with_kwargs=True)
    actual = model(**batch(altered))
    hook.remove()
    for a, b in zip(expected.predictions, actual.predictions):
        torch.testing.assert_close(a[0], b[0])
    inp = captured[0]
    lengths = [sample["x"].shape[1] for sample in samples]
    assert inp["x"].shape[1] == sum(lengths)
    assert inp["packed_seq_params"]["cu_seqlens_q"].tolist() == [0, lengths[0], sum(lengths)]
    assert inp["refiner_packed_seq_params"]["cu_seqlens_q"].tolist() == [0, 3, 12]
    for sample in samples:
        sample["x"].zero_()
        sample["audio_x"].zero_()
    assert [p.shape for p in model(**batch(samples)).predictions[0]] == [(1, 24, 2, 4, 6)] * 2


def test_ref2va_variable_reference_layouts_and_target_only_loss():
    refs = [
        {"kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 6},
        {"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 4},
    ]
    samples = prepare(condition_model(), [raw_sample(2, "ref2va"), raw_sample(11, "ref2va", refs)])
    model = tiny_model()
    expected = serial(model, samples)
    actual = model(**batch(samples))
    for i in range(2):
        torch.testing.assert_close(actual.predictions[0][i], expected[i].predictions[0], rtol=2e-5, atol=2e-5)
    assert [p.shape for p in actual.predictions[1]] == [(2, 32, 3)] * 2


def test_audio_refs_fail_closed():
    cond = condition_model()
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
    assert actual["cu_seqlens"].tolist() == [0, actual["seq_len"]]


def test_host_bounds_match_packed_layouts():
    refs = [
        {"kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 6},
        {"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 4},
    ]
    rows = [
        raw_sample(3),
        raw_sample(5, keyframes=(0, -1)),
        raw_sample(4, keyframes=()),
        raw_sample(6, "ref2va", refs),
    ]
    for row in rows:
        pk = row["packed"]
        assert packed_sequence.host_cu_seqlens(pk) == tuple(pk["cu_seqlens"].tolist())
        sample = prepare(condition_model(), [row])[0]
        assert sample["packed_seq_params"]["cu_seqlens_host"] == tuple(pk["cu_seqlens"].tolist())
        assert sample["refiner_packed_seq_params"]["cu_seqlens_host"] == (0, pk["text_len"])
        assert sample["refiner_packed_seq_params"]["cu_seqlens_q"].tolist() == [0, pk["text_len"]]
    legacy = dict(rows[0]["packed"])
    used = legacy["seq_len"]
    legacy["seq_len"] = used + (-used) % 64
    assert packed_sequence.host_cu_seqlens(legacy) == (0, used, legacy["seq_len"])


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_single_sample_valid_outputs_match_legacy_64_tail(task):
    model = tiny_model()
    sample = prepare(condition_model(), [raw_sample(3, task)])[0]
    length = sample["x"].shape[1]
    assert length % 64 != 0
    assert sample["packed_seq_params"]["cu_seqlens_q"].tolist() == [0, length]
    actual, expected = model(**sample), model(**legacy_tail(sample))
    for a, b in zip(actual.predictions, expected.predictions):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    for key in actual.loss:
        torch.testing.assert_close(actual.loss[key], expected.loss[key])


def test_uncovered_sp_attention_tail_cannot_poison_gradients(monkeypatch):
    attention = tiny_model().dit.blocks[0].attn
    x = torch.randn(8, 32, requires_grad=True)
    monkeypatch.setattr(torch, "empty_like", lambda value: torch.full_like(value, float("nan")))
    output = attention(x, rope_cos=None, rope_sin=None, cu_seqlens=(0, 7), max_seqlen=7)
    output[:7].square().mean().backward()
    assert all(torch.isfinite(param.grad).all() for param in attention.parameters())
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_sequence_parallel_padding_remains_forward_local(monkeypatch, task):
    model = tiny_model()
    sample = prepare(condition_model(), [raw_sample(3, task)])[0]
    length = sample["x"].shape[1]
    assert length % 2 == 1
    monkeypatch.setattr(minimax_h3_dit, "get_ulysses_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(minimax_h3_dit.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(minimax_h3_dit.dist, "get_rank", lambda group: 0)
    seen = []

    def block(hidden, **kwargs):
        seen.append((hidden.shape[0], kwargs["rope_cos"].shape[0], kwargs["cu_seqlens"], kwargs["use_ulysses"]))
        return hidden

    for module in model.dit.blocks:
        monkeypatch.setattr(module, "forward", block)
    monkeypatch.setattr(minimax_h3_dit._Gather, "apply", lambda group, x, *args: x.repeat(2, 1))
    out = model(**sample)
    assert seen == [((length + 1) // 2, length + 1, (0, length), True)] * 2
    assert sample["x"].shape[1] == sample["img_position_ids"].shape[1] == length
    assert out.predictions[0].shape == (1, 24, 2, 4, 6)
    with pytest.raises(ValueError, match="sequence parallelism"):
        model(**batch([sample, sample]))


_NPU_REJECTS_HUB = pytest.mark.skipif(is_torch_npu_available(), reason="Hub attention is rejected on Ascend NPU.")


@pytest.mark.parametrize(
    "backend",
    [
        "flash_attention_2",
        pytest.param("flash_attention_2_hub", marks=_NPU_REJECTS_HUB),
        "flash_attention_3",
        pytest.param("flash_attention_3_hub", marks=_NPU_REJECTS_HUB),
    ],
)
def test_fused_dispatch_keeps_refiners_sample_local_and_single_sample_legacy(monkeypatch, backend):
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model
    from veomni.ops.kernels.attention import flash

    calls, loads = [], []

    config = tiny_model().config

    def kernel(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        assert q.ndim == 3 and q.shape[1:] == (config.num_attention_heads, config.attention_head_dim)
        assert q.is_contiguous() and k.shape == v.shape == q.shape
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k is cu_seqlens_q
        assert cu_seqlens_q[-1] == q.shape[0] and max_seqlen_q == max_seqlen_k
        assert not causal
        calls.append(cu_seqlens_q.tolist())
        return minimax_h3_dit._sdpa_varlen_attention(q, k, v, tuple(cu_seqlens_q.tolist()), softmax_scale, True)

    def loader(name):
        loads.append(name)
        return SimpleNamespace(flash_attn_varlen_func=kernel)

    monkeypatch.setattr(flash, "_load_veomni_flash_kernel", loader)
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
    ).bfloat16()
    raws = [raw_sample(3), raw_sample(7)]
    for row in raws:
        for key, value in row.items():
            if torch.is_tensor(value) and value.is_floating_point():
                row[key] = value.bfloat16()
    samples = prepare(condition_model(), raws)
    serial(model, samples)
    assert calls == []
    loads.clear()  # Transformers may already have resolved the backend at construction.
    out = model(**batch(samples))
    sum(out.loss.values()).backward()
    assert calls[:2] == [[0, 3], [0, 7]]
    assert len(calls) == 4 and calls[2] == calls[3] and len(calls[2]) == 3
    assert loads == [f"veomni_{backend}_with_sp"]


def test_flash_backend_defers_packed_kernel_until_multisample_forward(monkeypatch):
    import sys

    from transformers import modeling_flash_attention_utils as hf_flash

    from veomni.ops.kernels.attention import flash

    loads = []

    def unavailable(name):
        loads.append(name)
        raise ImportError("flash_attn unavailable")

    def npu_attention(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):
        raise AssertionError("construction must not run NPU attention")

    # Mirror Ascend: Transformers resolves its native NPU FA before VeOmni's loader.
    monkeypatch.setattr(flash, "_load_veomni_flash_kernel", unavailable)
    monkeypatch.setattr(hf_flash, "is_flash_attn_2_available", lambda: False)
    monkeypatch.setattr(hf_flash, "is_torch_npu_available", lambda: True)
    for name in (
        "_loaded_implementation",
        "_flash_fn",
        "_flash_varlen_fn",
        "_flash_with_kvcache_fn",
        "_pad_fn",
        "_unpad_fn",
        "_process_flash_kwargs_fn",
    ):
        monkeypatch.setattr(hf_flash, name, None if name == "_loaded_implementation" else getattr(hf_flash, name))
    monkeypatch.setitem(
        sys.modules,
        "transformers.integrations.npu_flash_attention",
        SimpleNamespace(
            npu_flash_attn_func=npu_attention,
            npu_flash_attn_varlen_func=npu_attention,
            npu_flash_attn_with_kvcache=npu_attention,
        ),
    )
    config = tiny_model().config
    config._attn_implementation = "veomni_flash_attention_2_with_sp"
    model = MiniMaxH3DiTModel(config)
    samples = prepare(condition_model(), [raw_sample(3), raw_sample(7)])

    serial(model, samples[:1])
    assert loads == []
    with pytest.raises(ImportError, match="flash_attn unavailable"):
        model(**batch(samples))
    assert loads == ["veomni_flash_attention_2_with_sp"]


@pytest.mark.parametrize("checkpointing", [False, True])
def test_packed_sdpa_slices_with_host_bounds(monkeypatch, checkpointing):
    sdpa = minimax_h3_dit._sdpa_varlen_attention
    bounds = []

    def record(q, k, v, cu_seqlens, softmax_scale, compatibility_mode=False):
        bounds.append(cu_seqlens)
        return sdpa(q, k, v, cu_seqlens, softmax_scale, compatibility_mode)

    monkeypatch.setattr(minimax_h3_dit, "_sdpa_varlen_attention", record)
    raws = [raw_sample(3), raw_sample(7)]
    for row in raws:
        row["use_gradient_checkpointing"] = checkpointing
    samples = prepare(condition_model(), raws)
    model = tiny_model()
    reads = []
    tolist = torch.Tensor.tolist
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self: reads.append(self.shape) or tolist(self))
    out = model(**batch(samples))
    sum(out.loss.values()).backward()
    assert reads == []  # packing and the DiT use host bounds; no device read-back

    assert len(bounds) == 4 + 2 * checkpointing
    assert all(type(bound) is tuple and all(type(value) is int for value in bound) for bound in bounds)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_refiner_preserves_linear_row_counts_with_main_dit_packed(task):
    model = tiny_model()
    samples = prepare(condition_model(), [raw_sample(3, task), raw_sample(9, task)])
    rows = {"out_proj": [], "fc2": [], "main": []}
    refiner = model.dit.token_refiner.blocks[0]
    modules = {"out_proj": refiner.attn.out_proj, "fc2": refiner.mlp.fc2, "main": model.dit.blocks[0]}
    handles = [
        module.register_forward_pre_hook(lambda mod, args, name=name: rows[name].append(args[0].shape[0]))
        for name, module in modules.items()
    ]
    try:
        sum(model(**batch(samples)).loss.values()).backward()
    finally:
        for handle in handles:
            handle.remove()
    assert rows["out_proj"] == rows["fc2"] == [3, 9]
    assert rows["main"] == [sum(sample["x"].shape[1] for sample in samples)]


@pytest.mark.parametrize("offload", [True, [False, True]])
def test_multisample_wrapper_rejects_checkpoint_offload(offload):
    inputs = batch(prepare(condition_model(), [raw_sample(), raw_sample()]))
    inputs["use_gradient_checkpointing_offload"] = offload
    with pytest.raises(ValueError, match="checkpoint offload"):
        tiny_model()(**inputs)


@pytest.mark.parametrize("offload", [False, True])
def test_inference_forwards_checkpoint_offload_to_core(offload):
    from veomni.models.diffusers.minimax_h3.inference import model_fn_minimax_h3

    row = raw_sample()
    model = tiny_model()

    def capture(module, args, kwargs):
        assert kwargs["use_gradient_checkpointing_offload"] is offload
        assert kwargs["packed_seq_params"]["cu_seqlens_host"] == tuple(row["packed"]["cu_seqlens"].tolist())
        assert kwargs["refiner_packed_seq_params"]["cu_seqlens_host"] == (0, 3)
        raise RuntimeError("inference offload forwarded")

    handle = model.dit.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="inference offload forwarded"):
            model_fn_minimax_h3(
                model,
                row["input_latents"],
                row["audio_input_latents"],
                row["packed"],
                row["prompt_embeds"],
                t_video=0.5,
                t_audio=0.5,
                keyframe_cond_anchor=row["keyframe_cond_anchor"],
                use_gradient_checkpointing_offload=offload,
            )
    finally:
        handle.remove()


def test_condition_preserves_checkpoint_offload_for_model_validation():
    raws = [raw_sample(), raw_sample()]
    for row in raws:
        row["use_gradient_checkpointing_offload"] = True
    samples = prepare(condition_model(), raws)
    assert all(sample["use_gradient_checkpointing_offload"] is True for sample in samples)
    model = tiny_model()
    with pytest.raises(ValueError, match="checkpoint offload"):
        model(**batch(samples))

    def capture(module, args, kwargs):
        assert kwargs["use_gradient_checkpointing_offload"] is True
        raise RuntimeError("single-sample offload forwarded")

    handle = model.dit.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="single-sample offload forwarded"):
            model(**samples[0])
    finally:
        handle.remove()


def test_invalid_precision_and_legacy_tail_fail_closed():
    model = tiny_model()
    samples = prepare(condition_model(), [raw_sample(), raw_sample()])
    changed = copy.deepcopy(samples)
    changed[0]["unique_timesteps"] = changed[0]["unique_timesteps"].bfloat16()
    with pytest.raises(ValueError, match="cast_forward_inputs"):
        model(**batch(changed))
    with pytest.raises(ValueError, match="tail padding"):
        model(**batch([legacy_tail(samples[0]), samples[1]]))
    row = raw_sample()
    row["keyframe_cond_anchor"] = None
    with pytest.raises(ValueError, match="anchor rows"):
        prepare(condition_model(), [row])
    with pytest.raises(ValueError, match="nonempty"):
        model(x=[])


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_h3_foundation_loader_preserves_official_fp32_islands(dtype):
    from veomni.models.auto import build_foundation_model

    from ..tools.training_utils import make_eager_ops_config

    model = build_foundation_model(
        tiny_model().config, init_device="cpu", torch_dtype=dtype, ops_implementation=make_eager_ops_config()
    )
    ordinary_dtype = getattr(torch, dtype)
    islands = (
        model.dit.video_patch_proj,
        model.dit.audio_patch_proj,
        model.dit.time_embedder.proj_in,
        model.dit.time_embedder.proj_out,
        model.dit.final_layer.video_out,
        model.dit.final_layer.audio_out,
    )
    assert all(p.dtype == torch.float32 for module in islands for p in module.parameters())
    assert model.dit.condition_proj.weight.dtype == ordinary_dtype
    assert model.dit.blocks[0].adaln_proj.linear.weight.dtype == ordinary_dtype
    expected = tiny_model()
    assert {k: v.shape for k, v in model.state_dict().items()} == {
        k: v.shape for k, v in expected.state_dict().items()
    }
    sample = prepare(condition_model(), [raw_sample(text_len=9)])[0]
    out = model(**sample)
    assert all(value.dtype == torch.float32 for value in out.predictions)
    sum(out.loss.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_h3_fsdp_policy_preserves_coordinates_and_timesteps():
    from torch.distributed.fsdp import MixedPrecisionPolicy

    model = tiny_model()
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
    actual = model.get_fsdp_mixed_precision_policy(policy, model)
    assert actual.cast_forward_inputs is False
    assert policy.cast_forward_inputs is True
    assert actual.param_dtype == policy.param_dtype
    assert actual.reduce_dtype == policy.reduce_dtype
    assert actual.output_dtype == policy.output_dtype
    assert model.get_fsdp_mixed_precision_policy(policy, model.dit.condition_proj).cast_forward_inputs is True


def test_h3_adaln_activates_before_casting_fp32_time_embedding():
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import MiniMaxH3AdalnProj

    torch.manual_seed(39)
    module = MiniMaxH3AdalnProj(8, 8, 48, expand_ratio=6, modality_num=1).bfloat16()
    time = torch.randn(5, 8, dtype=torch.float32, requires_grad=True)
    actual = torch.cat(module(time), dim=-1)
    expected = module.linear(F.silu(time).bfloat16())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    assert time.grad.dtype == torch.float32 and torch.isfinite(time.grad).all()


class _PrecisionControlModel(torch.nn.Module):
    """A no-hook model for the shared FSDP input-cast regression."""

    config = SimpleNamespace(tie_word_embeddings=False)

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, position):
        return self.linear(x.to(self.linear.weight.dtype)), position


def _run_h3_precision_fsdp(checkpointing, lora, weights_path, wrap_projections=False, mixed_precision=True):
    import os

    import torch.distributed as dist
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    from veomni.arguments import AcceleratorConfig, MixedPrecisionConfig
    from veomni.distributed.parallel_state import (
        clear_parallel_state,
        init_parallel_state_from_config,
        use_parallel_state,
    )
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    from ..tools.training_utils import make_eager_ops_config

    os.environ["LOCAL_RANK"] = str(dist.get_rank())
    os.environ["WORLD_SIZE"] = str(dist.get_world_size())
    core.ATTENTION_IMPLEMENTATION = "torch"
    device = torch.device(get_device_type(), dist.get_rank())
    init_parallel_state_from_config(AcceleratorConfig(), name="h3_precision")
    try:
        with use_parallel_state("h3_precision"):
            config = tiny_model().config
            model = build_foundation_model(
                config,
                init_device="meta",
                torch_dtype="float32" if mixed_precision else "bfloat16",
                ops_implementation=make_eager_ops_config(),
            )
            reference = build_foundation_model(
                copy.deepcopy(config),
                init_device="cpu",
                torch_dtype="bfloat16",
                ops_implementation=make_eager_ops_config(),
            )
            torch.manual_seed(87)
            for parameter in reference.parameters():
                torch.nn.init.normal_(parameter, std=0.1)
            if dist.get_rank() == 0:
                save_file(reference.dit.state_dict(), os.path.join(weights_path, "model.safetensors"))
            dist.barrier()
            if lora:
                lora_config = VeOmniLoraConfig(
                    r=4,
                    lora_alpha=8,
                    target_modules=[
                        "qkv_proj",
                        "out_proj",
                        "fc1",
                        "fc2",
                        "condition_proj",
                        "linear",
                        "video_patch_proj",
                        "audio_patch_proj",
                        "proj_in",
                        "proj_out",
                        "video_out",
                        "audio_out",
                    ],
                )
                model = VeOmniLoraModel(model, lora_config)
                reference = VeOmniLoraModel(reference, copy.deepcopy(lora_config))
                for name, parameter in reference.named_parameters():
                    if "lora_" in name:
                        torch.nn.init.normal_(parameter, std=0.05)
            weights = {name: value.detach().clone().to(device) for name, value in reference.state_dict().items()}
            model = build_parallelize_model(
                model,
                weights_path=weights_path,
                init_device="meta",
                mixed_precision=MixedPrecisionConfig(enable=mixed_precision, cast_forward_inputs=True),
                is_peft_model=lora,
                broadcast_model_weights_from_rank0=wrap_projections,
                enable_gradient_checkpointing=False,
                basic_modules=["Linear", "LoraLinear", "MiniMaxH3FP32Linear"] if wrap_projections else None,
            )
            for name, parameter in model.named_parameters():
                if "lora_" not in name:
                    actual_weight = parameter.full_tensor()
                    torch.testing.assert_close(actual_weight, weights[name].to(actual_weight.dtype), rtol=0, atol=0)
            if lora:
                set_model_state_dict(
                    model,
                    {name: value for name, value in weights.items() if "lora_" in name},
                    options=StateDictOptions(full_state_dict=True, strict=False),
                )
            reference = reference.to(device)
            torch.manual_seed(94)
            samples = prepare(condition_model(), [raw_sample(17, task="ref2va"), raw_sample(9, latent_t=3)])
            for i, sample in enumerate(samples):
                sample["img_position_ids"] += 1025.333333333
                sample["unique_timesteps"] = torch.full_like(sample["unique_timesteps"], 0.094339624 + i * 0.1)
                sample["use_gradient_checkpointing"] = checkpointing

            def to_device(value):
                if torch.is_tensor(value):
                    return value.to(device)
                if isinstance(value, dict):
                    return {key: to_device(item) for key, item in value.items()}
                return value

            samples = [to_device(sample) for sample in samples]
            inputs = batch(samples)
            seen, projection_dtypes, time_dtypes = [], [], []
            from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import MiniMaxH3FP32Linear

            for module in model.modules():
                if isinstance(module, MiniMaxH3FP32Linear):
                    module.register_forward_pre_hook(
                        lambda module, args: projection_dtypes.append(module.weight.dtype)
                    )
            inner = model.get_base_model() if lora else model
            inner.dit.blocks[0].register_forward_pre_hook(
                lambda module, args, kwargs: time_dtypes.append(kwargs["t_emb"].dtype), with_kwargs=True
            )
            hook = model.register_forward_pre_hook(
                lambda module, args, kwargs: seen.append((kwargs["img_position_ids"], kwargs["unique_timesteps"])),
                with_kwargs=True,
            )
            single = model(**samples[0])
            single_reference = reference(**samples[0])
            with torch.no_grad():
                one_element = model(**batch([samples[0]]))
            for output in (single, one_element):
                for prediction, baseline in zip(output.predictions, single_reference.predictions):
                    torch.testing.assert_close(prediction, baseline, rtol=0.02, atol=0.002)
            sum(single.loss.values()).backward()
            sum(single_reference.loss.values()).backward()
            grads, ref_grads = [], []
            ref_params = dict(reference.named_parameters())
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    grads.append(parameter.grad.full_tensor().float().flatten())
                    ref_grads.append(ref_params[name].grad.float().flatten())
            grads, ref_grads = torch.cat(grads), torch.cat(ref_grads)
            assert (grads - ref_grads).norm() / ref_grads.norm() < 0.05
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            actual = model(**inputs)
            hook.remove()
            expected = serial(reference, samples)
            for actual_positions, positions in zip(seen[-1][0], inputs["img_position_ids"]):
                torch.testing.assert_close(actual_positions, positions, rtol=0, atol=0)
            for actual_timesteps, timesteps in zip(seen[-1][1], inputs["unique_timesteps"]):
                torch.testing.assert_close(actual_timesteps, timesteps, rtol=0, atol=0)
            assert projection_dtypes and set(projection_dtypes) == {torch.float32}
            assert time_dtypes and set(time_dtypes) == {torch.float32}
            for i, output in enumerate(expected):
                for modality, baseline in enumerate(output.predictions):
                    prediction = actual.predictions[modality][i]
                    assert prediction.dtype == torch.float32
                    torch.testing.assert_close(prediction, baseline, rtol=0.02, atol=0.002)
            sum(actual.loss.values()).backward()
            torch.stack([sum(output.loss.values()) for output in expected]).mean().backward()
            actual_grads, expected_grads = [], []
            reference_params = dict(reference.named_parameters())
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                gradient = parameter.grad.full_tensor().float()
                assert torch.isfinite(gradient).all()
                actual_grads.append(gradient.flatten())
                expected_grads.append(reference_params[name].grad.float().flatten())
            actual_grads, expected_grads = torch.cat(actual_grads), torch.cat(expected_grads)
            relative = (actual_grads - expected_grads).norm() / expected_grads.norm()
            assert relative < 0.05, relative
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            before = {
                name: parameter.full_tensor().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            optimizer.step()
            changed = [
                not torch.equal(parameter.full_tensor(), before[name])
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            assert any(changed)
            if not checkpointing and not lora and not wrap_projections:
                import torch.distributed.checkpoint as dcp

                from veomni.checkpoint.dcp_checkpointer import ModelState

                checkpoint_state = ModelState(model)
                dcp_path = os.path.join(weights_path, "roundtrip")
                saved = {name: parameter.full_tensor().clone() for name, parameter in model.named_parameters()}
                dcp.save({"model": checkpoint_state}, checkpoint_id=dcp_path)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                dcp.load({"model": checkpoint_state}, checkpoint_id=dcp_path)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter.full_tensor(), saved[name], rtol=0, atol=0)
                control_path = os.path.join(weights_path, "control")
                control_state = {"linear.weight": torch.eye(4), "linear.bias": torch.zeros(4)}
                if dist.get_rank() == 0:
                    os.makedirs(control_path)
                    save_file(control_state, os.path.join(control_path, "model.safetensors"))
                dist.barrier()
                for enabled, cast_inputs in ((True, True), (True, False), (False, True)):
                    with torch.device("meta"):
                        control = _PrecisionControlModel()
                    control = build_parallelize_model(
                        control,
                        weights_path=control_path,
                        init_device="meta",
                        mixed_precision=MixedPrecisionConfig(enable=enabled, cast_forward_inputs=cast_inputs),
                        enable_gradient_checkpointing=False,
                    )
                    for name, parameter in control.named_parameters():
                        torch.testing.assert_close(parameter.full_tensor().cpu(), control_state[name], rtol=0, atol=0)
                    position = torch.tensor([1025.333333333], dtype=torch.float64, device=device)
                    output, actual_position = control(torch.ones(2, 4, device=device), position)
                    expected_dtype = torch.bfloat16 if enabled and cast_inputs else torch.float64
                    assert actual_position.dtype == expected_dtype
                    torch.testing.assert_close(actual_position, position.to(expected_dtype), rtol=0, atol=0)
                    output.float().sum().backward()
                    assert all(torch.isfinite(p.grad.full_tensor()).all() for p in control.parameters())
            dist.barrier()
    finally:
        clear_parallel_state()


def _run_h3_precision_gloo(
    rank, rendezvous, checkpointing, lora, weights_path, wrap_projections, mixed_precision=True
):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        _run_h3_precision_fsdp(checkpointing, lora, weights_path, wrap_projections, mixed_precision)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("wrap_projections", [False, True])
def test_h3_precision_through_production_fsdp(tmp_path, checkpointing, lora, wrap_projections):
    import torch.multiprocessing as mp

    from veomni.utils.device import get_device_type

    from ..tools.launch_utils import torchrun

    if get_device_type() == "cpu":
        mp.spawn(
            _run_h3_precision_gloo,
            args=(str(tmp_path / "rendezvous"), checkpointing, lora, str(tmp_path), wrap_projections),
            nprocs=2,
            join=True,
        )
    else:
        torchrun(
            _run_h3_precision_fsdp,
            world_size=2,
            checkpointing=checkpointing,
            lora=lora,
            weights_path=str(tmp_path),
            wrap_projections=wrap_projections,
        )


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_h3_fp32_heads_support_multistep_bf16_anchors(task):
    from veomni.models.auto import build_foundation_model
    from veomni.models.diffusers.minimax_h3.inference import model_fn_minimax_h3
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.flow_match_scheduler import FlowMatchScheduler

    from ..tools.training_utils import make_eager_ops_config

    model = build_foundation_model(
        tiny_model().config, init_device="cpu", torch_dtype="bfloat16", ops_implementation=make_eager_ops_config()
    )
    row = raw_sample(9, task=task)
    row = {key: value.bfloat16() if torch.is_tensor(value) else value for key, value in row.items()}
    anchor = row["keyframe_cond_anchor" if task == "fl2va" else "ref_visual_anchor"]
    video, audio = row["input_latents"], row["audio_input_latents"]
    scheduler = FlowMatchScheduler()
    scheduler.set_timesteps(3)
    for timestep in scheduler.timesteps[:2]:
        with torch.no_grad():
            video_pred, audio_pred = model_fn_minimax_h3(
                model,
                video,
                audio,
                row["packed"],
                row["prompt_embeds"],
                t_video=0.5,
                t_audio=0.5,
                **{"keyframe_cond_anchor" if task == "fl2va" else "ref_visual_anchor": anchor},
            )
        assert video_pred.dtype == audio_pred.dtype == torch.float32
        video = scheduler.step(video_pred, timestep, video)
        audio = scheduler.step(audio_pred, timestep, audio)
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    assert anchor.dtype == torch.bfloat16


@pytest.mark.parametrize("legacy", [False, True])
def test_h3_hf_export_reload_preserves_fp32_values(tmp_path, legacy, monkeypatch):
    from safetensors.torch import load_file

    from veomni.models.auto import build_foundation_model
    from veomni.models.module_utils import load_model_weights
    from veomni.utils import save_safetensor_utils as save_utils

    from ..tools.training_utils import make_eager_ops_config

    model = build_foundation_model(
        tiny_model().config, init_device="cpu", torch_dtype="bfloat16", ops_implementation=make_eager_ops_config()
    )
    with torch.no_grad():
        model.dit.video_patch_proj.weight.fill_(1.001)
        model.dit.time_embedder.proj_in.weight.fill_(0.094339624)
    if legacy:
        monkeypatch.setattr(save_utils, "ckpt_to_state_dict", lambda **kwargs: model.state_dict())
        save_utils._save_hf_safetensor_legacy("unused", str(tmp_path), None, "dcp", None, model=model)
    else:
        mapping = dict.fromkeys(model.dit.state_dict(), 1)
        if save_utils.IS_NPU_AVAILABLE:
            pytest.skip("NPU exports use the legacy route tested separately")
        save_utils.save_hf_safetensor(
            save_hf_safetensor_path=str(tmp_path),
            ckpt_manager="dcp",
            model=model,
            fqn_to_index_mapping=mapping,
            is_rank_0=True,
        )
    loaded = {}
    for shard in tmp_path.glob("*.safetensors"):
        loaded.update(load_file(shard))
    prefix = "dit."
    for name in ("video_patch_proj.weight", "time_embedder.proj_in.weight"):
        expected = model.dit.state_dict()[name]
        assert loaded[prefix + name].dtype == torch.float32
        torch.testing.assert_close(loaded[prefix + name], expected, rtol=0, atol=0)
    assert loaded[prefix + "condition_proj.weight"].dtype == torch.bfloat16
    restored = build_foundation_model(
        copy.deepcopy(model.config),
        init_device="cpu",
        torch_dtype="bfloat16",
        ops_implementation=make_eager_ops_config(),
    )
    load_model_weights(restored, str(tmp_path), "cpu")
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


def test_h3_export_opt_in_preserves_other_model_defaults():
    from veomni.utils.save_safetensor_utils import _export_state_with_precision

    model = torch.nn.Linear(4, 4)
    model._keep_in_fp32_modules = ["linear"]
    state = {
        "linear.weight": torch.full((4,), 1.001, dtype=torch.float32),
        "other": torch.ones(4, dtype=torch.float16),
        "indices": torch.arange(4),
    }
    default = _export_state_with_precision(model, state)
    assert default["linear.weight"].dtype == torch.bfloat16
    assert default["other"].dtype == torch.float16
    model._preserve_fp32_modules_on_export = True
    opted_in = _export_state_with_precision(model, state)
    assert opted_in["linear.weight"].dtype == torch.float32
    torch.testing.assert_close(opted_in["linear.weight"], state["linear.weight"], rtol=0, atol=0)
    legacy = _export_state_with_precision(model, state, legacy=True)
    assert legacy["other"].dtype == torch.bfloat16
    assert legacy["indices"].dtype == torch.int64


def test_h3_loaded_lora_with_mixed_precision_disabled(tmp_path):
    import torch.multiprocessing as mp

    from veomni.utils.device import get_device_type

    from ..tools.launch_utils import torchrun

    if get_device_type() == "cpu":
        mp.spawn(
            _run_h3_precision_gloo,
            args=(str(tmp_path / "rendezvous"), False, True, str(tmp_path), False, False),
            nprocs=2,
            join=True,
        )
    else:
        torchrun(
            _run_h3_precision_fsdp,
            world_size=2,
            checkpointing=False,
            lora=True,
            weights_path=str(tmp_path),
            mixed_precision=False,
        )


def test_h3_fp32_lora_math_and_export_preserve_precision(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
    from veomni.utils.save_safetensor_utils import save_lora_adapter_with_dcp

    torch.manual_seed(78)
    model = VeOmniLoraModel(tiny_model(), VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=["video_patch_proj"]))
    projection = model.get_base_model().dit.video_patch_proj
    with torch.no_grad():
        projection.lora_A["default"].weight.fill_(0.094339624)
        projection.lora_B["default"].weight.fill_(0.013711)
    x = torch.randn(5, 96, requires_grad=True)
    normal = projection(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = projection(x)
    torch.testing.assert_close(actual, normal, rtol=0, atol=0)
    actual.sum().backward()
    assert projection.lora_B["default"].weight.grad.dtype == torch.float32
    from veomni.checkpoint import dcp_checkpointer

    monkeypatch.setattr(dcp_checkpointer, "empty_cache", lambda: None)
    save_lora_adapter_with_dcp(model, str(tmp_path))
    state = load_file(tmp_path / "adapter_model.safetensors")
    for name, tensor in state.items():
        assert tensor.dtype == torch.float32
        live_name = name.replace("lora_A.weight", "lora_A.default.weight").replace(
            "lora_B.weight", "lora_B.default.weight"
        )
        torch.testing.assert_close(tensor, dict(model.named_parameters())[live_name], rtol=0, atol=0)
