"""CPU/NPU correctness checks; full pretrained trainer acceptance is separate."""

import os

import pytest
import torch

from veomni.models.diffusers.hunyuanvideo15.conditioning import FlowMatchingConditioner
from veomni.models.diffusers.hunyuanvideo15.configuration_hunyuanvideo15 import HunyuanVideo15Config
from veomni.models.diffusers.hunyuanvideo15.modeling_hunyuanvideo15 import HunyuanVideo15Model


@pytest.fixture(autouse=True)
def cpu_threads():
    from torch.utils.checkpoint import DefaultDeviceType

    previous_device = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
    DefaultDeviceType.set_device_type(previous_device)


def initialize_nonzero_linears(model):
    """Exercise attention/gates; native zero output initialization hides errors."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, std=0.01)


def tiny_config(glyph=False):
    return HunyuanVideo15Config(
        hidden_size=24,
        heads_num=2,
        rope_dim_list=[4, 4, 4],
        in_channels=2,
        out_channels=2,
        mm_double_blocks_depth=1,
        mm_single_blocks_depth=0,
        text_states_dim=16,
        glyph_byT5_v2=glyph,
        use_cond_type_embedding=glyph,
        vision_projection="linear" if glyph else "none",
        vision_states_dim=8,
    )


def batch(glyph=False):
    return dict(
        hidden_states=torch.randn(1, 5, 1, 2, 2),
        timestep=torch.tensor([450.0]),
        text_states=torch.randn(1, 3, 16),
        encoder_attention_mask=torch.tensor([[True, True, False]]),
        byt5_text_states=torch.randn(1, 2, 1472) if glyph else None,
        byt5_text_mask=torch.tensor([[True, False]]) if glyph else None,
        training_target=torch.randn(1, 2, 1, 2, 2),
    )


@pytest.mark.parametrize("checkpointing", [False, True])
def test_forward_backward_update_and_strict_reload(tmp_path, checkpointing):
    torch.manual_seed(123)
    model = HunyuanVideo15Model(tiny_config())
    initialize_nonzero_linears(model)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    data = batch()
    before = model.img_in.proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
    result = model(**{k: [v, v] if v is not None else None for k, v in data.items()})
    assert len(result.predictions) == 2
    assert result.predictions[0].shape == (1, 2, 1, 2, 2)
    result.loss["mse_loss"].backward()
    assert torch.isfinite(result.loss["mse_loss"])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.img_in.proj.weight)
    model.eval().save_pretrained(tmp_path)
    reloaded = HunyuanVideo15Model.from_pretrained(tmp_path, local_files_only=True).eval()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(**data).predictions, reloaded(**data).predictions, rtol=0, atol=0)


def test_masked_condition_tokens_do_not_change_output():
    torch.manual_seed(123)
    model = HunyuanVideo15Model(tiny_config(glyph=True)).eval()
    initialize_nonzero_linears(model)
    data = batch(glyph=True)
    first = model(**data).predictions
    data["text_states"][:, -1] = 10000
    data["byt5_text_states"][:, -1] = -10000
    torch.testing.assert_close(first, model(**data).predictions, rtol=0, atol=0)


def test_flow_objective_and_rng_continuation():
    noise = FlowMatchingConditioner(42, "cpu")
    posterior = torch.randn(1, 4, 1, 2, 2)
    latent = noise.sample_posterior(posterior)
    noise(latent)
    saved = noise.rng_state_dict()
    expected_latent = noise.sample_posterior(posterior)
    expected = noise(expected_latent)
    restored = FlowMatchingConditioner(999, "cpu")
    restored.load_rng_state_dict(saved)
    actual_latent = restored.sample_posterior(posterior)
    actual = restored(actual_latent)
    torch.testing.assert_close(actual_latent, expected_latent, rtol=0, atol=0)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    sigma = actual["timestep"].view(-1, 1, 1, 1, 1) / 1000
    clean_plus_velocity = actual_latent + sigma * actual["training_target"]
    torch.testing.assert_close(actual["hidden_states"][:, :2], clean_plus_velocity)
    assert not actual["hidden_states"][:, 2:].count_nonzero()


def test_reject_misaligned_batches_and_unsupported_scope():
    model = HunyuanVideo15Model(tiny_config())
    data = batch()
    data["hidden_states"] = [data["hidden_states"]]
    with pytest.raises(ValueError, match="same length"):
        model(**data)
    with pytest.raises(ValueError, match="480p T2V"):
        HunyuanVideo15Config(ideal_task="i2v")
    with pytest.raises(ValueError, match="dense SDPA"):
        HunyuanVideo15Model(tiny_config().from_dict(tiny_config().to_dict() | {"attn_mode": "flex-block-attn"}))


@pytest.mark.skipif(not os.environ.get("HUNYUANVIDEO15_SOURCE"), reason="Pinned upstream source not supplied")
def test_upstream_output_gradient_and_full_parameter_schema():
    from .hunyuanvideo15_reference import load_reference

    torch.manual_seed(123)
    reference_cls = load_reference(os.environ["HUNYUANVIDEO15_SOURCE"])
    config = tiny_config(glyph=True)
    reference = reference_cls(**config.to_native_dict()).eval()
    initialize_nonzero_linears(reference)
    model = HunyuanVideo15Model(config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    data = batch(glyph=True)
    expected = reference(
        data["hidden_states"],
        data["timestep"],
        data["text_states"],
        None,
        data["encoder_attention_mask"],
        extra_kwargs={k: data[k] for k in ("byt5_text_states", "byt5_text_mask")},
    )[0]
    actual = model(**data).predictions
    assert expected.abs().max().item() > 1e-4
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert model.double_blocks[0].img_attn_q.weight.grad.abs().max().item() > 0
    errors = []
    parameters = dict(model.named_parameters())
    for name, p in reference.named_parameters():
        q = parameters[name]
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, atol=2e-5, rtol=2e-4)
            errors.append((q.grad - p.grad).abs().max().item())
        else:
            assert q.grad is None
    print(
        "source output max_abs",
        (actual - expected).abs().max().item(),
        "relative",
        ((actual - expected).abs() / expected.abs().clamp_min(1e-8)).max().item(),
        "gradient max_abs",
        max(errors),
    )
    # Full production configuration, no real parameter storage or weight download.
    with torch.device("meta"):
        full_reference = reference_cls(**HunyuanVideo15Config().to_native_dict())
        full_model = HunyuanVideo15Model(HunyuanVideo15Config())

    def manifest(m):
        return {k: (tuple(v.shape), v.dtype) for k, v in m.state_dict().items()}

    assert manifest(full_reference) == manifest(full_model)
    print(
        "full schema",
        len(full_model.state_dict()),
        "tensors",
        sum(p.numel() for p in full_model.parameters()),
        "parameters",
    )


def test_runtime_rng_roundtrip(monkeypatch):
    from veomni.models.diffusers.hunyuanvideo15.configuration_hunyuanvideo15 import HunyuanVideo15ConditionConfig
    from veomni.models.diffusers.hunyuanvideo15.modeling_hunyuanvideo15_condition import HunyuanVideo15ConditionModel
    from veomni.trainer.dit_trainer import DiTModelRuntime

    # This tests real process_condition/RNG hooks without claiming encoder execution.
    monkeypatch.setattr(HunyuanVideo15ConditionModel, "_load_components", lambda self, device: None)
    config = HunyuanVideo15ConditionConfig(generator_device="cpu", dtype="float32")
    condition = HunyuanVideo15ConditionModel(config)
    runtime = object.__new__(DiTModelRuntime)
    runtime.condition_model = condition
    inputs = dict(
        latents=[torch.randn(1, 2, 1, 2, 2)],
        text_states=[torch.randn(1, 3, 16)],
        encoder_attention_mask=[torch.tensor([[True, True, False]])],
        byt5_text_states=[torch.zeros(1, 2, 1472)],
        byt5_text_mask=[torch.zeros(1, 2, dtype=torch.bool)],
    )
    condition.process_condition(**inputs)
    saved = runtime.extra_state()
    expected = condition.process_condition(**inputs)
    runtime.condition_model = HunyuanVideo15ConditionModel(config)
    runtime.load_extra_state(saved)
    actual = runtime.condition_model.process_condition(**inputs)
    for key in expected:
        torch.testing.assert_close(expected[key][0], actual[key][0], rtol=0, atol=0)
    assert not runtime.condition_model.train().training


def test_native_vae_encode_strict_reload(tmp_path):
    from veomni.models.diffusers.hunyuanvideo15.native.vae import AutoencoderKLConv3D

    model = AutoencoderKLConv3D(
        in_channels=3,
        out_channels=3,
        latent_channels=2,
        block_out_channels=(8, 16, 16),
        layers_per_block=1,
        ffactor_spatial=4,
        ffactor_temporal=4,
        sample_size=32,
        sample_tsize=5,
        scaling_factor=1.0,
    ).eval()
    pixels = torch.randn(1, 3, 5, 32, 32)
    parameters = model.encode(pixels).latent_dist.parameters
    assert parameters.shape == (1, 4, 2, 8, 8)
    assert torch.isfinite(parameters).all()
    model.save_pretrained(tmp_path)
    loaded, info = AutoencoderKLConv3D.from_pretrained(tmp_path, local_files_only=True, output_loading_info=True)
    assert not any(info.values())
    torch.testing.assert_close(parameters, loaded.encode(pixels).latent_dist.parameters, rtol=0, atol=0)


def test_foundation_loader_preserves_native_weights(tmp_path):
    from veomni.models import build_foundation_model

    from ..tools.training_utils import make_eager_ops_config

    original = HunyuanVideo15Model(tiny_config()).eval()
    original.save_pretrained(tmp_path)
    # build_foundation_model loads CPU weights on rank 0; use a real one-rank
    # Gloo group rather than the uninitialized (-1) rank.
    torch.distributed.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'gloo'}")
    try:
        loaded = build_foundation_model(
            config_path=str(tmp_path),
            weights_path=str(tmp_path),
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=make_eager_ops_config(),
        ).eval()
    finally:
        torch.distributed.destroy_process_group()

    assert set(original.state_dict()) == set(loaded.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    data = batch()
    torch.testing.assert_close(original(**data).predictions, loaded(**data).predictions, rtol=0, atol=0)


def test_glyph_checkpoint_extraction_is_strict(tmp_path, monkeypatch):
    from veomni.models.diffusers.hunyuanvideo15.native import glyph_encoder

    encoder = torch.nn.Linear(3, 2)
    reference = {key: value.clone() for key, value in encoder.state_dict().items()}
    checkpoint = tmp_path / "byt5.pt"
    torch.save({"state_dict": {"module.text_tower.encoder." + k: v for k, v in reference.items()}}, checkpoint)

    def base_loader(
        *,
        byt5_name,
        special_token,
        color_special_token,
        font_special_token,
        color_ann_path,
        font_ann_path,
        multilingual,
        huggingface_cache_dir,
        device,
    ):
        return torch.nn.Linear(3, 2), object()

    monkeypatch.setattr(glyph_encoder, "load_byt5_and_byt5_tokenizer", base_loader)
    args = {
        "byt5_max_length": 256,
        "byT5_google_path": "local-base",
        "byT5_ckpt_path": str(checkpoint),
        "multilingual_prompt_format_color_path": "colors.json",
        "multilingual_prompt_format_font_path": "fonts.json",
    }
    _, loaded, length = glyph_encoder.create_byt5(args, "cpu")
    assert length == 256
    assert not any(p.requires_grad for p in loaded.parameters())
    for key in reference:
        torch.testing.assert_close(reference[key], loaded.state_dict()[key], rtol=0, atol=0)
    torch.save({"state_dict": {"module.text_tower.encoder.weight": reference["weight"]}}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        glyph_encoder.create_byt5(args, "cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_npu_forward_backward_update_and_rng(dtype, checkpointing):
    import copy

    from veomni.utils.device import get_device_type, get_torch_device

    if get_device_type() != "npu" or not get_torch_device().is_available():
        pytest.skip("An accessible NPU is required")
    accelerator = get_torch_device()
    accelerator.set_device(0)
    torch.manual_seed(42)
    config = HunyuanVideo15Config(
        hidden_size=256,
        heads_num=2,
        mm_double_blocks_depth=2,
        text_states_dim=32,
        vision_states_dim=8,
    )
    reference = HunyuanVideo15Model(config)
    initialize_nonzero_linears(reference)
    reference = reference.to(dtype=dtype)
    model = copy.deepcopy(reference).to("npu").train()
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Distinct seeds avoid a degenerate noise == latent, zero velocity target.
    flow = FlowMatchingConditioner(314, "npu")
    latent = torch.randn(1, 32, 2, 4, 4, dtype=dtype).to("npu")
    data = flow(latent)
    data.update(
        text_states=torch.randn(1, 8, 32, dtype=dtype).to("npu"),
        encoder_attention_mask=torch.tensor([[True] * 4 + [False] * 4]).to("npu"),
        byt5_text_states=torch.randn(1, 4, 1472, dtype=dtype).to("npu"),
        byt5_text_mask=torch.tensor([[True, True, False, False]]).to("npu"),
    )
    assert data["training_target"].abs().max().item() > 0
    cpu_data = {k: v.cpu() for k, v in data.items()}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        expected = reference(**cpu_data)
    expected.loss["mse_loss"].backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
    before = model.img_in.proj.weight.detach().clone()
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(**data)
        loss = result.loss["mse_loss"]
        loss.backward()
        assert torch.isfinite(loss).item()
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        assert model.double_blocks[0].img_attn_q.weight.grad.abs().max().item() > 0
        if step == 0:
            atol, rtol = (2e-4, 2e-3) if dtype == torch.float32 else (1e-2, 5e-2)
            torch.testing.assert_close(result.predictions.cpu(), expected.predictions, atol=atol, rtol=rtol)
            print(
                "NPU parity",
                dtype,
                checkpointing,
                "output max_abs",
                (result.predictions.cpu() - expected.predictions).abs().max().item(),
            )
        optimizer.step()
        losses.append(loss.item())
    accelerator.synchronize()
    assert not torch.equal(before, model.img_in.proj.weight)
    saved = flow.rng_state_dict()
    next_condition = flow(latent)
    flow.load_rng_state_dict(saved)
    replay = flow(latent)
    for key in replay:
        torch.testing.assert_close(replay[key], next_condition[key], rtol=0, atol=0)
    print("NPU losses", dtype, checkpointing, losses)


@pytest.mark.skipif(os.environ.get("HUNYUANVIDEO15_FULL_NPU") != "1", reason="Opt-in full 8.33B NPU test")
def test_full_backbone_npu():
    """Needs at least 40 GiB free; reduced latent input, random weights, SGD."""
    from veomni.utils.device import get_device_type, get_torch_device

    if get_device_type() != "npu" or not get_torch_device().is_available():
        pytest.skip("An accessible NPU is required")
    accelerator = get_torch_device()
    accelerator.set_device(0)
    accelerator.empty_cache()
    accelerator.reset_peak_memory_stats()
    torch.manual_seed(42)
    with torch.device("npu:0"):
        model = HunyuanVideo15Model._from_config(
            HunyuanVideo15Config(), dtype=torch.bfloat16, attn_implementation="eager"
        )
        initialize_nonzero_linears(model)
    assert sum(p.numel() for p in model.parameters()) == 8326608160
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    flow = FlowMatchingConditioner(314, "npu")
    data = flow(torch.randn(1, 32, 2, 4, 4, device="npu", dtype=torch.bfloat16))
    data.update(
        text_states=torch.randn(1, 8, 3584, device="npu", dtype=torch.bfloat16),
        encoder_attention_mask=torch.tensor([[True] * 4 + [False] * 4], device="npu"),
        byt5_text_states=torch.randn(1, 4, 1472, device="npu", dtype=torch.bfloat16),
        byt5_text_mask=torch.tensor([[True, True, False, False]], device="npu"),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False)
    before = model.img_in.proj.weight.detach().clone()
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**data).loss["mse_loss"]
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        assert torch.isfinite(loss).item() and torch.isfinite(norm).item()
        assert model.double_blocks[0].img_attn_q.weight.grad.abs().max().item() > 0
        optimizer.step()
        losses.append(loss.item())
    accelerator.synchronize()
    assert not torch.equal(before, model.img_in.proj.weight)
    print("Full NPU losses", losses, "peak allocated GiB", accelerator.max_memory_allocated() / 2**30)
