"""Native H3 model-owned packing, without pretrained weights or encoders."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

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
    assert out.predictions[0].shape == (2, 24, 2, 4, 6)
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
        torch.testing.assert_close(actual.predictions[0][i : i + 1], ref.predictions[0], rtol=2e-5, atol=2e-5)
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
def test_ordinary_trainer_step_matches_native_serial_update(monkeypatch, task):
    from tests.trainer.test_dit_microbatch import _trainer
    from veomni.trainer.dit_trainer import DiTModelRuntime

    trainer = _trainer(monkeypatch, "offline_training", 2)
    model = tiny_model()
    reference = copy.deepcopy(model)
    condition = condition_model()
    runtime = DiTModelRuntime.__new__(DiTModelRuntime)
    runtime.model_name, runtime.model, runtime.condition_model = "base", model, condition
    runtime.optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    runtime.lr_scheduler = torch.optim.lr_scheduler.StepLR(runtime.optimizer, step_size=1, gamma=0.5)
    gradients = []

    def clip():
        gradients.append([param.grad.clone() for param in model.parameters()])
        return torch.stack([param.grad.norm() for param in model.parameters()]).norm()

    runtime.clip_grad_norm = clip
    trainer.base.model = runtime
    trainer.base.state = SimpleNamespace(global_step=0)
    trainer.base.model_fwd_context = nullcontext()
    trainer.base.model_bwd_context = nullcontext()
    for name in (
        "on_step_begin",
        "on_step_end",
        "sync_before_train_step",
        "_reset_async_activation_offload_if_enabled",
        "model_reshard",
        "_configure_hsdp_allreduce",
    ):
        setattr(trainer.base, name, Mock())
    raws = [raw_sample(length, task) for length in (3, 9, 5, 7)]
    torch.manual_seed(77)
    expected = serial(reference, prepare(condition, raws))
    expected_loss = sum(sum(out.loss.values()) for out in expected) / len(raws)
    expected_loss.backward()
    optimizer = torch.optim.SGD(reference.parameters(), lr=1e-4)
    optimizer.step()
    calls = []
    handle = model.register_forward_pre_hook(lambda module, args: calls.append(1))
    torch.manual_seed(77)
    trainer.train_step(iter([[DiTDataCollator()(raws[:2]), DiTDataCollator()(raws[2:])]]))
    handle.remove()
    assert len(calls) == 2
    assert trainer.base.state.global_step == runtime.lr_scheduler.last_epoch == 1
    assert runtime.optimizer.param_groups[0]["lr"] == 5e-5
    for grad, param, ref in zip(gradients[0], model.parameters(), reference.parameters()):
        torch.testing.assert_close(grad, ref.grad, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(param, ref, rtol=2e-4, atol=2e-5)
        assert param.grad is None
    torch.testing.assert_close(torch.tensor(trainer.base.on_step_end.call_args.kwargs["loss"]), expected_loss.detach())


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
    assert model(**batch(samples)).predictions[0].shape == (2, 24, 2, 4, 6)


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
        torch.testing.assert_close(actual.predictions[0][i : i + 1], expected[i].predictions[0], rtol=2e-5, atol=2e-5)
    assert actual.predictions[1].shape == (2, 2, 32, 3)


def test_invalid_condition_columns_and_audio_refs_fail_closed():
    cond = condition_model()
    rows = DiTDataCollator()([raw_sample(), raw_sample()])
    rows["prompt_embeds"] = rows["prompt_embeds"][:1]
    with pytest.raises(ValueError, match="length"):
        cond.process_condition(**rows)
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


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_single_sample_valid_outputs_match_legacy_64_tail(task):
    model = tiny_model()
    sample = prepare(condition_model(), [raw_sample(3, task)])[0]
    length = sample["x"].shape[1]
    assert length % 64 != 0
    assert sample["packed_seq_params"]["cu_seqlens_q"].tolist() == [0, length]
    legacy = copy.deepcopy(sample)
    pad = (-length) % 64
    for key in ("x", "audio_x", "img_position_ids"):
        legacy[key] = F.pad(legacy[key], (0, 0, 0, pad))
    legacy["token_tags"] = F.pad(legacy["token_tags"], (0, pad), value=-1)
    legacy["inverse_indices"] = F.pad(legacy["inverse_indices"], (0, pad))
    legacy["packed_seq_params"]["cu_seqlens_q"] = torch.tensor([0, length, length + pad], dtype=torch.int32)
    actual, expected = model(**sample), model(**legacy)
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


@pytest.mark.parametrize("backend", ["veomni_flash_attention_2_with_sp", "veomni_flash_attention_3_with_sp"])
def test_fused_dispatch_keeps_refiners_sample_local_and_single_sample_legacy(monkeypatch, backend):
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model
    from veomni.ops.kernels.attention import flash

    calls = []

    def kernel(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k is cu_seqlens_q
        assert cu_seqlens_q[-1] == q.shape[0] and max_seqlen_q == max_seqlen_k
        assert not causal
        calls.append(cu_seqlens_q.tolist())
        return minimax_h3_dit._sdpa_varlen_attention(q, k, v, tuple(cu_seqlens_q.tolist()), softmax_scale, True)

    monkeypatch.setattr(
        flash, "_load_veomni_local_flash_kernel", lambda name: SimpleNamespace(flash_attn_varlen_func=kernel)
    )
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
    out = model(**batch(samples))
    sum(out.loss.values()).backward()
    assert calls[:2] == [[0, 3], [0, 7]]
    assert len(calls) == 4 and calls[2] == calls[3] and len(calls[2]) == 3


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
    monkeypatch.setattr(flash, "_load_veomni_local_flash_kernel", unavailable)
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
    out = tiny_model()(**batch(prepare(condition_model(), raws)))
    sum(out.loss.values()).backward()

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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_modulation_preserves_upstream_gather_dtype_and_gradients(monkeypatch, dtype):
    original = torch.Tensor.index_select
    gather_dtypes = []

    def select(tensor, dim, index):
        gather_dtypes.append(tensor.dtype)
        return original(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", select)
    index = torch.arange(512) % 2
    x = torch.ones(512, 4, dtype=dtype)
    shift, scale, gate = (torch.randn(2, 4, dtype=dtype, requires_grad=True) for _ in range(3))
    expected = x * (1 + original(scale, 0, index)) + original(shift, 0, index)
    out = minimax_h3_dit._modulate_scale_shift(x, shift, scale, index)
    gated = minimax_h3_dit._modulate_gate(x, gate, x, index)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(gated, x + original(gate, 0, index), rtol=0, atol=0)
    (out.sum() + gated.sum()).backward()
    assert gather_dtypes == [dtype] * 3
    expected_grads = torch.autograd.grad(
        expected.sum() + (x + original(gate, 0, index) * x).sum(), (shift, scale, gate)
    )
    for param, expected_grad in zip((shift, scale, gate), expected_grads):
        torch.testing.assert_close(param.grad, expected_grad, rtol=0, atol=0)


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


def test_invalid_batch_shapes_precision_and_legacy_tail_fail_closed():
    model = tiny_model()
    samples = prepare(condition_model(), [raw_sample(), raw_sample()])
    changed = copy.deepcopy(samples)
    changed[1]["video_latent_shape"] = (3, 2, 3)
    with pytest.raises(ValueError, match="fixed target"):
        model(**batch(changed))
    changed = copy.deepcopy(samples)
    changed[0]["unique_timesteps"] = changed[0]["unique_timesteps"].bfloat16()
    with pytest.raises(ValueError, match="cast_forward_inputs"):
        model(**batch(changed))
    changed = copy.deepcopy(samples)
    changed[0]["packed_seq_params"]["cu_seqlens_q"] = torch.tensor([0, 27, 27], dtype=torch.int32)
    with pytest.raises(ValueError, match="tail padding"):
        model(**batch(changed))
    row = raw_sample()
    row["keyframe_cond_anchor"] = None
    with pytest.raises(ValueError, match="anchor rows"):
        prepare(condition_model(), [row])
    with pytest.raises(ValueError, match="nonempty"):
        model(x=[])
