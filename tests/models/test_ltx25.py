"""CPU contracts for LTX-2.5; real-weight and distributed validation are separate."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
from veomni.lora.weight_loading import load_lora_weights
from veomni.models.checkpoint.weights import load_model_weights
from veomni.models.diffusers.ltx2_3.ltx_condition.configuration_ltx2_3_condition import LTXVideoConditionModelConfig
from veomni.models.diffusers.ltx2_3.ltx_condition.ltx25 import DIT_PREFIX, prepare_training_directory
from veomni.models.diffusers.ltx2_3.ltx_condition.modeling_ltx2_3_condition import LTXVideoConditionModel
from veomni.models.diffusers.ltx2_3.ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
    build_embeddings_processor,
)
from veomni.models.diffusers.ltx2_3.ltx_transformer.configuration_ltx2_3_transformer import (
    LTXVideoTransformerModelConfig,
)
from veomni.models.diffusers.ltx2_3.ltx_transformer.modeling_ltx2_3_transformer import (
    LTXVideoTransformerModel,
    apply_veomni_ltx_transformer_patch,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def tiny_config(**overrides):
    """Build a reduced audio-video configuration with LTX-2.5 architecture flags."""
    values = dict(
        in_channels=8,
        out_channels=8,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        cross_attention_dim=32,
        caption_channels=32,
        with_audio=True,
        audio_num_attention_heads=2,
        audio_attention_head_dim=16,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_cross_attention_dim=32,
        caption_proj_before_connector=True,
        cross_attention_adaln=True,
        apply_gated_attention=True,
        ff_bias=False,
        use_keyframes_abs_pos_embedding=True,
        connector_ff_bias=False,
        connector_num_attention_heads=2,
        connector_attention_head_dim=16,
        connector_num_layers=1,
        audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=16,
        connector_apply_gated_attention=True,
    )
    values.update(overrides)
    return LTXVideoTransformerModelConfig(**values)


def tiny_model(config=None):
    """Create a deterministic small model with nonzero weights for CPU regressions."""
    torch.manual_seed(12)
    model = LTXVideoTransformerModel(config or tiny_config()).float()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.1)
    apply_veomni_ltx_transformer_patch()
    return model


def batch():
    """Create reproducible video, audio and text tensors for one training sample."""
    gen = torch.Generator().manual_seed(42)

    def randn(*shape):
        """Draw fixture tensors from the batch-local random generator."""
        return torch.randn(*shape, generator=gen)

    return dict(
        hidden_states=[randn(1, 8, 3, 2, 2)],
        timestep=[torch.tensor(0.5)],
        encoder_hidden_states=[randn(1, 128, 32)],
        training_target=[randn(1, 8, 3, 2, 2)],
        audio_hidden_states=[randn(1, 2, 7, 4)],
        audio_timestep=[torch.tensor(0.5)],
        audio_training_target=[randn(1, 2, 7, 4)],
        audio_encoder_hidden_states=[randn(1, 128, 32)],
    )


def sdpa_backend_flags():
    """Capture all SDPA backend switches for validation-harness isolation checks."""
    return (
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )


@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
@pytest.mark.parametrize("heads", [1, 2])
def test_e2e_math_attention_forward_backward(monkeypatch, mask_kind, heads):
    """Check the smoke harness uses only math SDPA and preserves outputs and gradients."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from tests.parallel.ulysses.ltx25_e2e.train_observed import MathSDPAAttention

    torch.manual_seed(18)
    inputs = [torch.randn(2, length, heads * 8, requires_grad=True) for length in (3, 5, 5)]
    reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    mask = None
    if mask_kind != "none":
        mask = torch.tensor([True, True, True, False, False]).expand(2, 1, 3, 5)
        if mask_kind == "additive":
            mask = torch.zeros(mask.shape).masked_fill(~mask, -torch.inf)

    sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def checked_sdpa(*args, **kwargs):
        """Inspect the effective backend switches at the actual SDPA call."""
        calls.append(sdpa_backend_flags())
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", checked_sdpa)
    before = sdpa_backend_flags()
    actual = MathSDPAAttention()(*inputs, heads, mask)
    actual.square().sum().backward()
    assert calls == [(True, False, False, False)]
    assert sdpa_backend_flags() == before

    q, k, v = [value.reshape(2, -1, heads, 8).transpose(1, 2) for value in reference_inputs]
    with sdpa_kernel(SDPBackend.MATH):
        expected = sdpa(q, k, v, attn_mask=mask, is_causal=False).transpose(1, 2).reshape(2, 3, heads * 8)
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    for value, reference in zip(inputs, reference_inputs, strict=True):
        torch.testing.assert_close(value.grad, reference.grad)


def test_e2e_math_attention_restores_backends_on_error(monkeypatch):
    """A failed validation call must not leave global attention switches changed."""
    from tests.parallel.ulysses.ltx25_e2e.train_observed import MathSDPAAttention

    def fail(*args, **kwargs):
        """Raise inside SDPA after checking the backend override."""
        assert sdpa_backend_flags() == (True, False, False, False)
        raise RuntimeError("test attention failure")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fail)
    before = sdpa_backend_flags()
    value = torch.randn(1, 3, 8)
    with pytest.raises(RuntimeError, match="test attention failure"):
        MathSDPAAttention()(value, value, value, 1)
    assert sdpa_backend_flags() == before


def test_e2e_math_attention_pins_transformer_and_connectors():
    """Cover both attention routes in the DiT and frozen text connectors."""
    from tests.parallel.ulysses.ltx25_e2e.train_observed import MathSDPAAttention, pin_math_attention
    from veomni.models.diffusers.ltx2_3.ltx_transformer.modeling_ltx2_3_transformer import Attention

    model = tiny_model()
    processor = build_embeddings_processor({"transformer": model.config.to_dict()}).float().eval()
    roots = (model, processor)
    modules = [[module for module in root.modules() if isinstance(module, Attention)] for root in roots]
    assert all(modules)
    assert pin_math_attention(*roots) == sum(map(len, modules))
    for group in modules:
        for module in group:
            assert isinstance(module.attention_function, MathSDPAAttention)
            assert isinstance(module.masked_attention_function, MathSDPAAttention)
    inputs = batch()
    with torch.no_grad():
        video, audio, _ = processor.create_embeddings(
            inputs["encoder_hidden_states"][0],
            inputs["audio_encoder_hidden_states"][0],
            torch.zeros(1, 1, 1, 128),
        )
    inputs["encoder_hidden_states"] = [video]
    inputs["audio_encoder_hidden_states"] = [audio]
    loss = model(**inputs).loss["mse_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)


def make_pack(path, model):
    """Write a miniature official-layout transformer and connector checkpoint."""
    metadata = {"transformer": model.config.to_dict()}
    processor = build_embeddings_processor(metadata)
    state = {DIT_PREFIX + k: v.detach().contiguous() for k, v in model.state_dict().items()}
    for prefix, connector in (
        ("video_embeddings_connector.", processor.video_connector),
        ("audio_embeddings_connector.", processor.audio_connector),
    ):
        state.update({DIT_PREFIX + prefix + k: v.detach().contiguous() for k, v in connector.state_dict().items()})
    save_file(state, str(path), metadata={"config": json.dumps(metadata)})
    return state, metadata


@pytest.mark.parametrize("conditioning", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_keyframe_semantics_and_backward(conditioning, checkpointing):
    """Check keyframe embeddings, clean-frame timesteps and finite backward gradients."""
    model = tiny_model()
    model.set_gradient_checkpointing(checkpointing)
    inputs = batch()
    if conditioning:
        inputs["video_loss_mask"] = [torch.cat([torch.zeros(4), torch.ones(8)])]
    projected = []
    original = model.video_args_preprocessor.prepare

    def capture(*args, **kwargs):
        """Capture projected video tokens without changing preprocessor behavior."""
        result = original(*args, **kwargs)
        projected.append((args[0], result.x.detach()))
        return result

    model.video_args_preprocessor.prepare = capture
    result = model(**inputs)
    result.loss["mse_loss"].backward()
    modality, actual = projected[0]
    expected = model.patchify_proj(modality.latent).detach()
    expected[:, :4] += model.keyframes_abs_pos_embedding.detach()
    torch.testing.assert_close(actual, expected)
    assert torch.equal(modality.timesteps[:, :4] == 0, torch.full((1, 4), conditioning))
    assert model.transformer_blocks[0].ff.net[0].proj.bias is None
    assert model.transformer_blocks[0].audio_ff.net[0].proj.bias is not None
    assert torch.isfinite(result.loss["mse_loss"])
    assert model.keyframes_abs_pos_embedding.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_23_defaults_preserve_structure_and_output():
    """Verify omitted LTX-2.5 flags retain LTX-2.3 parameters and predictions."""
    implicit = tiny_config().to_dict()
    for key in ("ff_bias", "audio_ff_bias", "use_keyframes_abs_pos_embedding"):
        implicit.pop(key)
    old = tiny_model(LTXVideoTransformerModelConfig(**implicit))
    explicit = tiny_model(tiny_config(ff_bias=True, audio_ff_bias=True, use_keyframes_abs_pos_embedding=False))
    assert old.state_dict().keys() == explicit.state_dict().keys()
    assert "keyframes_abs_pos_embedding" not in old.state_dict()
    assert old.transformer_blocks[0].ff.net[0].proj.bias is not None
    torch.testing.assert_close(old(**batch()).predictions[0], explicit(**batch()).predictions[0], rtol=0, atol=0)


def test_official_pack_prepare_load_and_condition(tmp_path):
    """Verify split-pack preparation, exact weight reload and condition-model backward."""
    model = tiny_model()
    pack = tmp_path / "official.safetensors"
    make_pack(pack, model)
    directory = tmp_path / "prepared"
    prepare_training_directory(str(pack), str(directory))
    loaded_config = LTXVideoTransformerModelConfig.from_pretrained(directory)
    reloaded = tiny_model(loaded_config)
    load_model_weights(reloaded, str(directory), init_device="cpu")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)
    condition_config = LTXVideoConditionModelConfig.from_pretrained(
        str(directory), with_audio=True, first_frame_conditioning_p=0
    )
    condition = LTXVideoConditionModel(condition_config, meta_init=True).float()
    assert condition.embeddings_processor.video_connector.transformer_1d_blocks[0].ff.net[0].proj.bias is None
    assert condition.embeddings_processor.audio_connector.transformer_1d_blocks[0].ff.net[0].proj.bias is None
    raw = batch()
    processed = condition.process_condition(
        latents=raw["hidden_states"],
        context=raw["encoder_hidden_states"],
        context_mask=[torch.ones(1, 128, dtype=torch.long)],
        audio_context=raw["audio_encoder_hidden_states"],
        audio_latents=raw["audio_hidden_states"],
    )
    reloaded(**processed).loss["mse_loss"].backward()
    assert reloaded.keyframes_abs_pos_embedding.grad.norm() > 0
    with pytest.raises(FileExistsError):
        prepare_training_directory(str(pack), str(directory))


def test_prepare_refuses_missing_weights(tmp_path):
    """Reject incomplete transformer weights before creating the output directory."""
    pack = tmp_path / "bad.safetensors"
    state, metadata = make_pack(pack, tiny_model())
    state.pop(DIT_PREFIX + "keyframes_abs_pos_embedding")
    save_file(state, str(pack), metadata={"config": json.dumps(metadata)})
    with pytest.raises(ValueError, match="missing=.*keyframes_abs_pos_embedding"):
        prepare_training_directory(str(pack), str(tmp_path / "prepared"))
    assert not (tmp_path / "prepared").exists()


def test_native_lora_update_save_reload(tmp_path):
    """Verify a native LoRA update survives adapter reload and weight merging."""
    base = tiny_model()
    config = VeOmniLoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    model = VeOmniLoraModel(copy.deepcopy(base), config)
    inputs = batch()
    before = model(**inputs).predictions[0].detach()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model(**inputs).loss["mse_loss"].backward()
    optimizer.step()
    after = model(**inputs).predictions[0].detach()
    assert not torch.equal(before, after)
    model.save_pretrained(str(tmp_path))
    reloaded = VeOmniLoraModel.from_pretrained(copy.deepcopy(base), str(tmp_path))
    load_lora_weights(reloaded, str(tmp_path), init_device="cpu")
    torch.testing.assert_close(reloaded(**inputs).predictions[0], after, rtol=0, atol=0)
    merged = reloaded.merge_and_unload()
    torch.testing.assert_close(merged(**inputs).predictions[0], after, rtol=1e-5, atol=1e-6)


def make_gemma4_pack(tmp_path, model_type="gemma4_unified_text", layout="flat", metadata_assets=False):
    """Build a tiny Gemma-4 pack with configurable text keys and tokenizer asset storage."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import AutoModel, PreTrainedTokenizerFast
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config = CONFIG_MAPPING[model_type](
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        layer_types=["sliding_attention", "full_attention"],
        max_position_embeddings=128,
        sliding_window=16,
        hidden_size_per_layer_input=0,
    )
    source = AutoModel.from_config(config).eval()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "hello": 4}, unk_token="<unk>")
        ),
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )
    tokenizer.backend_tokenizer.pre_tokenizer = WhitespaceSplit()
    assets = tmp_path / "assets"
    tokenizer.save_pretrained(assets)
    prefix = "model." if layout == "flat" else "model.language_model."
    state = {prefix + k: v.detach().contiguous() for k, v in source.state_dict().items()}
    # Multimodal tower tensors must not enter the caption-only model.
    state["model.embed_audio.unused"] = torch.ones(1)
    state["vision_model.unused"] = torch.ones(1)
    text_config = config.to_dict()
    text_config["dtype"] = "bfloat16"
    metadata = {"gemma_config": json.dumps({"text_config": text_config})}
    for filename, key in (
        ("tokenizer.json", "tokenizer_json"),
        ("tokenizer_config.json", "hf_asset__tokenizer_config.json"),
    ):
        payload = (assets / filename).read_bytes()
        if metadata_assets and filename == "tokenizer_config.json":
            metadata[filename] = payload.decode()
        else:
            state[key] = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
            if metadata_assets:
                state[key] = state[key].view(torch.int8)
    for stream, width in (("video", 32), ("audio", 16)):
        state[f"text_embedding_projection.{stream}_aggregate_embed.weight"] = torch.randn(width, 96)
        state[f"text_embedding_projection.{stream}_aggregate_embed.bias"] = torch.randn(width)
    pack = tmp_path / "gemma.safetensors"
    save_file(state, str(pack), metadata=metadata)
    return source, state, pack, metadata


@pytest.mark.parametrize("model_type", ["gemma4_text", "gemma4_unified_text"])
@pytest.mark.parametrize("layout", ["flat", "nested"])
@pytest.mark.parametrize("metadata_assets", [False, True])
def test_packed_gemma4_and_projection_roundtrip(tmp_path, model_type, layout, metadata_assets):
    """Compare packed text encoding and projections with independently built references."""
    from veomni.models.diffusers.ltx2_3.ltx_condition.ltx25 import PackedGemma4Encoder, load_packed_feature_extractor

    source, state, pack, _ = make_gemma4_pack(tmp_path, model_type, layout, metadata_assets)
    encoder = PackedGemma4Encoder(str(pack), torch.device("cpu"), torch.float32, max_length=8)
    assert type(encoder.model) is type(source)
    for name, buffer in source.named_buffers():
        torch.testing.assert_close(encoder.model.get_buffer(name), buffer, rtol=0, atol=0)
    assert not encoder.model.training and all(not p.requires_grad for p in encoder.parameters())
    assert all(not b.is_meta and torch.isfinite(b).all() for b in encoder.model.buffers())
    hidden, mask = encoder.encode("hello")
    reference = source(
        input_ids=torch.tensor([[0, 0, 0, 0, 0, 0, 2, 4]]),
        attention_mask=mask,
        output_hidden_states=True,
        use_cache=False,
    )
    for a, b in zip(hidden, reference.hidden_states):
        torch.testing.assert_close(a, b)
    assert mask.tolist() == [[0, 0, 0, 0, 0, 0, 1, 1]]
    blank, blank_mask = encoder.encode("")
    assert blank_mask.sum() == 1
    assert all(torch.isfinite(h).all() for h in blank)
    extractor = load_packed_feature_extractor(str(pack), torch.device("cpu"), torch.float32)
    video, audio = extractor(hidden, mask)
    assert video.shape == (1, 8, 32) and audio.shape == (1, 8, 16)
    encoded = torch.stack(hidden, dim=-1)
    expected = encoded * torch.rsqrt(encoded.square().mean(dim=2, keepdim=True) + 1e-6)
    expected = expected.flatten(2).masked_fill(~mask.bool().unsqueeze(-1), 0)
    torch.testing.assert_close(
        video,
        torch.nn.functional.linear(
            expected,
            state["text_embedding_projection.video_aggregate_embed.weight"],
            state["text_embedding_projection.video_aggregate_embed.bias"],
        ),
    )

    # Compare all hidden states against an independently constructed text model,
    # including BOS idempotence, whitespace/blank captions, special tokens and truncation.
    for prompt, ids in (
        ("  hello  ", [2, 4]),
        ("   ", [2]),
        ("<bos> hello", [2, 4]),
        ("<eos>", [2, 1]),
        ("hello " * 12, [2, *([4] * 7)]),
    ):
        hidden, mask = encoder.encode(prompt)
        padded = torch.tensor([[0] * (8 - len(ids)) + ids])
        expected_mask = padded.ne(0).long()
        torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
        with torch.no_grad():
            expected = source(padded, attention_mask=expected_mask, output_hidden_states=True, use_cache=False)
        for actual, reference in zip(hidden, expected.hidden_states, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["tokenizer_json", "hf_asset__tokenizer_config.json", "model.norm.weight"])
def test_packed_gemma4_rejects_missing_assets_or_weights(tmp_path, missing):
    """Reject packed Gemma checkpoints missing tokenizer assets or text weights."""
    from veomni.models.diffusers.ltx2_3.ltx_condition.ltx25 import PackedGemma4Encoder

    _, state, pack, metadata = make_gemma4_pack(tmp_path)
    del state[missing]
    save_file(state, str(pack), metadata=metadata)
    with pytest.raises((ValueError, RuntimeError), match="missing|Missing"):
        PackedGemma4Encoder(str(pack), torch.device("cpu"), torch.float32)


@pytest.mark.parametrize("ltx25", [False, True])
def test_replicated_connectors_ignore_ambient_sp(monkeypatch, ltx25):
    """Verify replicated text connectors avoid sequence-parallel collectives."""
    from veomni.models.diffusers.ltx2_3.ltx_transformer import modeling_ltx2_3_transformer as modeling

    config = tiny_config(
        caption_proj_before_connector=ltx25,
        connector_ff_bias=not ltx25,
        connector_apply_gated_attention=ltx25,
    )
    torch.manual_seed(7)
    processor = build_embeddings_processor({"transformer": config.to_dict()}).float().eval()
    apply_veomni_ltx_transformer_patch()
    video = torch.randn(1, 128, 32)
    audio = torch.randn(1, 128, 32)
    mask = torch.cat([torch.full((1, 1, 1, 64), -torch.inf), torch.zeros(1, 1, 1, 64)], dim=-1)
    with torch.no_grad():
        monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
        expected = processor.create_embeddings(video, audio, mask)
        # Replicated connectors must not consult or use an SP process group.
        monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))
        actual = processor.create_embeddings(video, audio, mask)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.parametrize("component", ["encoder", "decoder"])
@pytest.mark.parametrize("prefix", ["vae.", ""])
def test_video_vae_checkpoint_layouts(component, prefix):
    """Verify split and prefixed VAE keys select only the requested component."""
    from veomni.models.diffusers.ltx2_3.ltx_core.model.video_vae.model_configurator import (
        VAE_DECODER_COMFY_KEYS_FILTER,
        VAE_ENCODER_COMFY_KEYS_FILTER,
    )

    ops = VAE_ENCODER_COMFY_KEYS_FILTER if component == "encoder" else VAE_DECODER_COMFY_KEYS_FILTER
    assert ops.apply_to_key(f"{prefix}{component}.conv_in.weight") == "conv_in.weight"
    assert ops.apply_to_key(f"{prefix}per_channel_statistics.std-of-means") == "per_channel_statistics.std-of-means"
    other = "decoder" if component == "encoder" else "encoder"
    assert ops.apply_to_key(f"{prefix}{other}.conv_in.weight") is None
    assert ops.apply_to_key("model.diffusion_model.patchify_proj.weight") is None


@pytest.mark.parametrize(
    ("failure", "retry_after", "expected_delay"),
    [
        ("http", "Wed, 21 Oct 2026 07:28:00 GMT", 5),
        ("http", "invalid", 5),
        ("http", "120", 60),
        ("http", "-1", 0),
        ("http", "2", 2),
        ("incomplete", None, 5),
    ],
)
def test_smoke_download_retries_transient_failures(tmp_path, monkeypatch, failure, retry_after, expected_delay):
    """Retry transient responses and verify the completed file before publishing it."""
    import hashlib
    import http.client
    import io
    import urllib.error

    from tests.parallel.ulysses.ltx25_e2e import range_download

    payload = b"complete checkpoint bytes"
    calls, delays = [], []

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}"}

        def read(self, size=-1):
            """Simulate a connection dropping during the first response body."""
            if failure == "incomplete" and len(calls) == 1:
                raise http.client.IncompleteRead(b"partial", len(payload))
            return super().read(size)

    def open_response(request, timeout):
        """Fail the first request or return an exact byte-range response."""
        calls.append(request.get_header("Range"))
        if failure == "http" and len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", {"Retry-After": retry_after}, None)
        return Response(payload)

    monkeypatch.setattr(range_download.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(range_download.time, "sleep", delays.append)
    manifest = [
        dict(
            path="model.bin",
            url="https://example.invalid/model",
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    ]
    paths = range_download.download_all(manifest, tmp_path, lambda *args, **kwargs: None, workers=1)
    assert paths[0].read_bytes() == payload
    assert not paths[0].with_suffix(".bin.part").exists()
    assert calls == [f"bytes=0-{len(payload) - 1}"] * 2
    assert delays == [expected_delay]


@pytest.mark.parametrize(("status", "attempts"), [(404, 1), (503, 4)])
def test_smoke_download_bounds_retries(tmp_path, monkeypatch, status, attempts):
    """Stop on permanent errors or retry exhaustion without publishing a partial file."""
    import urllib.error

    from tests.parallel.ulysses.ltx25_e2e import range_download

    calls, delays = [], []

    def fail_request(request, timeout):
        """Return the same HTTP failure on every request."""
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, status, "Failure", {}, None)

    monkeypatch.setattr(range_download.urllib.request, "urlopen", fail_request)
    monkeypatch.setattr(range_download.time, "sleep", delays.append)
    manifest = [dict(path="model.bin", url="https://example.invalid/model", size=8, sha256="unused")]
    with pytest.raises(urllib.error.HTTPError):
        range_download.download_all(manifest, tmp_path, lambda *args, **kwargs: None, workers=1)
    assert len(calls) == attempts
    assert delays == [5, 10, 20][: attempts - 1]
    assert not (tmp_path / "model.bin").exists()


@pytest.mark.parametrize("failure", ["size", "SHA256"])
def test_smoke_download_integrity_checks_survive_python_optimization(tmp_path, failure):
    """Reject corrupt assets under python -O without replacing an existing destination."""
    import subprocess
    import sys
    import textwrap

    destination = tmp_path / "model.bin"
    destination.write_bytes(b"existing verified file")
    script = textwrap.dedent("""\
        import hashlib
        import io
        import sys
        from pathlib import Path
        from unittest.mock import patch
        from tests.parallel.ulysses.ltx25_e2e import range_download

        root, failure = Path(sys.argv[1]), sys.argv[2]
        payload = b"checkpoint"
        events = []
        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": f"bytes 0-{len(payload)-1}/{len(payload)}"}

        def emit(event, **kwargs):
            events.append(event)
            if event == "download_progress" and failure == "size":
                (root / "model.bin.part").write_bytes(payload[:-1])

        digest = hashlib.sha256(payload if failure == "size" else b"other bytes").hexdigest()
        manifest = [dict(path="model.bin", url="https://example.invalid/model",
                         size=len(payload), sha256=digest)]
        with patch.object(range_download.urllib.request, "urlopen",
                          return_value=Response(payload)):
            try:
                range_download.download_all(manifest, root, emit, workers=1)
            except ValueError as error:
                if f"Weight {failure} mismatch" not in str(error):
                    raise
            else:
                raise RuntimeError("Corrupt asset passed integrity verification")
        if "weight_verified" in events:
            raise RuntimeError("Corrupt asset was reported as verified")
        """)
    result = subprocess.run(
        [sys.executable, "-O", "-c", script, str(tmp_path), failure], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert destination.read_bytes() == b"existing verified file"


@pytest.mark.parametrize("interrupted", [False, True])
def test_smoke_download_cancels_queued_chunks_and_waits_for_active_writes(tmp_path, monkeypatch, interrupted):
    """Cancel queued chunks on failure while active workers finish before file handles close."""
    import io
    import threading
    import urllib.error

    from tests.parallel.ulysses.ltx25_e2e import range_download

    active = threading.Event()
    release = threading.Event()
    started, written, shutdowns = [], [], []
    error = KeyboardInterrupt() if interrupted else urllib.error.HTTPError("unused", 404, "Missing", {}, None)
    executor = range_download.concurrent.futures.ThreadPoolExecutor
    write = range_download.os.pwrite

    class ControlledExecutor(executor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            """Keep active responses blocked until the real queue has been cancelled or drained."""
            shutdowns.append(cancel_futures)
            super().shutdown(wait=False, cancel_futures=cancel_futures)
            release.set()
            super().shutdown(wait=wait)

    class Response(io.BytesIO):
        status = 206

        def read(self, size=-1):
            """Hold an active response while the failed future is consumed."""
            assert release.wait(5), "executor did not shut down"
            return super().read(size)

    def open_response(request, timeout):
        """Fail the first chunk only after another worker has started."""
        start = int(request.get_header("Range").split("=")[1].split("-")[0])
        started.append(start)
        if start == 0:
            assert active.wait(5), "second worker did not start"
            raise error
        active.set()
        response = Response(b"x")
        response.headers = {"Content-Range": f"bytes {start}-{start}/12"}
        return response

    def record_write(fd, data, offset):
        """Use the real descriptor so a premature close causes the test to fail."""
        result = write(fd, data, offset)
        written.append(offset)
        return result

    monkeypatch.setattr(range_download.concurrent.futures, "ThreadPoolExecutor", ControlledExecutor)
    monkeypatch.setattr(range_download.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(range_download.os, "pwrite", record_write)
    events = []
    manifest = [dict(path="model.bin", url="https://example.invalid/model", size=12, sha256="unused")]
    with pytest.raises(type(error)) as caught:
        range_download.download_all(
            manifest, tmp_path, lambda event, **kwargs: events.append(event), workers=2, chunk_size=1
        )
    assert caught.value is error
    assert shutdowns[0] is True
    assert 1 < len(started) < 12
    assert sorted(written) == sorted(start for start in started if start != 0)
    assert not (tmp_path / "model.bin").exists()
    assert "weight_verified" not in events
