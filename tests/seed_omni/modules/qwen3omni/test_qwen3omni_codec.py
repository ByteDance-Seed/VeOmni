"""Unit tests for the Qwen3-Omni audio codec OmniModule."""

import numpy as np
import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata
from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.qwen3omni.carrier import TALKER_CODES_KEY
from veomni.models.seed_omni.modules.qwen3omni.codec.accelerated.accelerated import Qwen3OmniCodecAccelerated
from veomni.models.seed_omni.modules.qwen3omni.codec.configuration import Qwen3OmniCodecConfig
from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import (
    SIGNAL_AUDIO_DONE,
    TALKER_CODE_LABELS,
    Qwen3OmniCodec,
    _as_mono_batch,
)
from veomni.models.seed_omni.modules.qwen3omni.codec.processing import Qwen3OmniCodecPreprocessor
from veomni.models.seed_omni.utils.conversation import ConversationItem


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


NUM_QUANTIZERS = 16
CODEBOOK_SIZE = 128
SAMPLING_RATE = 24000

TINY_CODE2WAV_CONFIG = dict(
    codebook_size=CODEBOOK_SIZE,
    semantic_codebook_size=CODEBOOK_SIZE,
    num_quantizers=NUM_QUANTIZERS,
    num_semantic_quantizers=1,
    codebook_dim=16,
    vector_quantization_hidden_dimension=16,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    decoder_dim=32,
    upsample_rates=[2, 2],
    upsampling_ratios=[4],
    sliding_window=8,
    max_position_embeddings=128,
)

TINY_MIMI_CONFIG = dict(
    codebook_size=CODEBOOK_SIZE,
    num_quantizers=20,
    num_semantic_quantizers=1,
    hidden_size=32,
    upsample_groups=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    intermediate_size=32,
    sliding_window=8,
    max_position_embeddings=128,
    num_filters=8,
    num_residual_layers=1,
    upsampling_ratios=[4, 2],
    codebook_dim=16,
    vector_quantization_hidden_dimension=16,
    sampling_rate=SAMPLING_RATE,
    audio_channels=1,
)


def _tiny_config(**overrides) -> Qwen3OmniCodecConfig:
    kwargs = dict(
        ops_implementation=make_eager_ops_config(),
        code2wav_config=TINY_CODE2WAV_CONFIG,
        mimi_config=TINY_MIMI_CONFIG,
        chunk_size=8,
        left_context_size=2,
    )
    kwargs.update(overrides)
    return Qwen3OmniCodecConfig(**kwargs)


def _tiny_module() -> Qwen3OmniCodec:
    torch.manual_seed(0)
    return Qwen3OmniCodec(_tiny_config()).eval()


def _accelerated() -> Qwen3OmniCodecAccelerated:
    torch.manual_seed(0)
    return Qwen3OmniCodecAccelerated(_tiny_config()).eval()


def _codes(frames: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, CODEBOOK_SIZE, (NUM_QUANTIZERS, frames))


def _assistant_audio(seconds: float, rate: int = SAMPLING_RATE) -> ConversationItem:
    torch.manual_seed(2)
    return ConversationItem(
        type="audio",
        value=torch.randn(int(rate * seconds)) * 0.1,
        role="assistant",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=rate, num_samples=int(rate * seconds))},
    )


def _run_encode(module, conversation_list):
    inputs = module.pre_forward(method="encode", conversation_list=conversation_list)
    outputs = module.encode(**inputs)
    return module.post_forward(method="encode", **outputs)


def test_codec_config_derivation_and_validation():
    config = _tiny_config(chunk_size=300, left_context_size=25)
    assert config.model_type == "qwen3omni_codec"
    assert (config.chunk_size, config.left_context_size) == (300, 25)
    assert (config.num_quantizers, config.sampling_rate, config.tie_word_embeddings) == (
        NUM_QUANTIZERS,
        SAMPLING_RATE,
        False,
    )
    assert config.frame_rate == config.mimi_config.frame_rate

    # Real checkpoint pair: 24000 Hz / 12.5 Hz == 1920 samples/frame.
    real_rates = _tiny_config(
        code2wav_config={**TINY_CODE2WAV_CONFIG, "upsample_rates": [8, 5, 4, 3], "upsampling_ratios": [2, 2]},
        mimi_config={**TINY_MIMI_CONFIG, "upsampling_ratios": [8, 6, 5, 4]},
    )
    assert (real_rates.decoder_samples_per_frame, real_rates.encoder_samples_per_frame, real_rates.frame_rate) == (
        1920,
        1920,
        12.5,
    )

    # Repeating a derived value is accepted; contradicting it or mismatched halves is rejected.
    assert _tiny_config(sampling_rate=SAMPLING_RATE, num_quantizers=NUM_QUANTIZERS).sampling_rate == SAMPLING_RATE
    with pytest.raises(ValueError, match="derived from the two halves"):
        _tiny_config(sampling_rate=16000)
    with pytest.raises(ValueError, match="disagree on codebook size"):
        _tiny_config(mimi_config={**TINY_MIMI_CONFIG, "codebook_size": CODEBOOK_SIZE * 2})
    with pytest.raises(ValueError, match="only 8 quantizers"):
        _tiny_config(mimi_config={**TINY_MIMI_CONFIG, "num_quantizers": 8})
    with pytest.raises(ValueError, match="how much audio one code frame spans"):
        _tiny_config(code2wav_config={**TINY_CODE2WAV_CONFIG, "upsampling_ratios": [8]})


def test_codec_encode_contracts_and_rvq_truncation():
    module = _tiny_module()
    waveform = torch.randn(1, SAMPLING_RATE)
    with torch.no_grad():
        for shape in ((SAMPLING_RATE,), (1, SAMPLING_RATE), (1, 1, SAMPLING_RATE)):
            codes = module.encode_wav(torch.randn(*shape), sampling_rate=SAMPLING_RATE)
            assert codes.shape[0] == NUM_QUANTIZERS and codes.dtype == torch.long
            assert 0 <= int(codes.min()) and int(codes.max()) < CODEBOOK_SIZE
        # Truncating deeper RVQ output to 16 quantizers is exact.
        shallow = module.mimi.encode(waveform.unsqueeze(0), num_quantizers=NUM_QUANTIZERS).audio_codes
        deep = module.mimi.encode(waveform.unsqueeze(0), num_quantizers=TINY_MIMI_CONFIG["num_quantizers"]).audio_codes
    assert torch.equal(deep[:, :NUM_QUANTIZERS], shallow)

    with pytest.raises(ValueError, match="16000 Hz audio but this codec operates at 24000 Hz"):
        module.encode_wav(torch.randn(16000), sampling_rate=16000)
    with pytest.raises(ValueError, match="expected mono audio"):
        module.encode_wav(torch.randn(2, SAMPLING_RATE), sampling_rate=SAMPLING_RATE)
    with pytest.raises(ValueError, match="single mono clip"):
        module.encode_wav(torch.randn(2, 1, SAMPLING_RATE), sampling_rate=SAMPLING_RATE)
    with pytest.raises(ValueError, match="cannot read a waveform"):
        module.encode_wav(torch.randn(1, 1, 1, SAMPLING_RATE), sampling_rate=SAMPLING_RATE)


@pytest.mark.parametrize("frames", [4, 8, 20])
def test_decode_matches_upstream_and_roundtrips(frames):
    module = _tiny_module()
    upstream_config = Qwen3OmniMoeCode2WavConfig(**TINY_CODE2WAV_CONFIG)
    upstream_config.ops_implementation = make_eager_ops_config().to_dict()
    upstream = Qwen3OmniMoeCode2Wav._from_config(upstream_config).eval()
    upstream.load_state_dict(
        {k.removeprefix("code2wav."): v for k, v in module.state_dict().items() if k.startswith("code2wav.")},
        strict=True,
    )

    codes = _codes(frames)
    with torch.no_grad():
        mine = module.decode_codes(codes)
        batched = module.decode_codes(codes.unsqueeze(0))
        reference = upstream.chunked_decode(codes.unsqueeze(0), chunk_size=8, left_context_size=2).float()
        roundtrip = module.decode_codes(
            module.encode_wav(torch.randn(SAMPLING_RATE // 4), sampling_rate=SAMPLING_RATE)
        )
    torch.testing.assert_close(mine, reference, rtol=0, atol=1e-6)
    assert torch.equal(mine, batched) and mine.dim() == 3
    assert roundtrip.dim() == 3 and roundtrip.size(-1) > 0

    with pytest.raises(ValueError, match="expected 16 RVQ code groups"):
        module.decode_codes(codes.transpose(0, 1))


def test_generate_appends_audio_item_and_validates_codes():
    module = _tiny_module()
    conversation = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="output", value=_codes(8), role="assistant", meta={TALKER_CODES_KEY: True}),
    ]
    with torch.no_grad():
        outputs = module.generate(conversation_list=conversation)

    assert outputs[FSM_SIGNAL_KEY] == SIGNAL_AUDIO_DONE
    tail = outputs["conversation_list"][-1]
    assert (tail.type, tail.role) == ("audio", "assistant")
    assert tail.meta[AUDIO_METADATA_KEY].sampling_rate == SAMPLING_RATE
    assert tail.meta[AUDIO_METADATA_KEY].num_samples == tail.value.numel()

    normalized = OmniModel._normalize_generated(outputs["generated"])
    assert normalized is not None and normalized["type"] == "audio"
    assert normalized["meta"][AUDIO_METADATA_KEY].sampling_rate == SAMPLING_RATE
    assert torch.equal(tail.value, normalized["value"])

    with pytest.raises(ValueError, match="no output row stamped 'qwen3omni_talker_codes'"):
        module.generate(conversation_list=[ConversationItem(type="text", value="hi", role="user")])


def test_accelerated_encode_stashes_isolated_assistant_targets():
    module = _accelerated()
    user = ConversationItem(
        type="audio",
        value=torch.randn(1200),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=16000, num_samples=1200)},
    )
    short, long = _assistant_audio(0.04), _assistant_audio(0.09)
    with torch.no_grad():
        out = _run_encode(module, [[user, short], [long]])
        alone = module.encode_wav(short.value, sampling_rate=SAMPLING_RATE)
        decoded = module.decode_codes(short.meta[TALKER_CODE_LABELS])

    assert out["conversation_list"][0][1] is short
    assert TALKER_CODE_LABELS not in user.meta
    assert torch.equal(short.meta[TALKER_CODE_LABELS], alone)
    assert short.meta[TALKER_CODE_LABELS].shape[1] < long.meta[TALKER_CODE_LABELS].shape[1]
    assert decoded.dim() == 3 and decoded.size(0) == 1

    # No assistant audio is a no-op; missing or mismatched rate raises.
    text = ConversationItem(type="text", value="hi", role="user")
    with torch.no_grad():
        assert _run_encode(module, [[text]])["conversation_list"] == [[text]]

    missing_rate = _assistant_audio(0.05)
    del missing_rate.meta[AUDIO_METADATA_KEY]
    with pytest.raises(ValueError, match="states no meta"):
        _run_encode(module, [[missing_rate]])
    with pytest.raises(ValueError, match="but this codec operates at"):
        _run_encode(module, [[_assistant_audio(0.05, rate=16000)]])


def test_codec_preprocessor_resamples_assistant_only_and_validates_metadata():
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=SAMPLING_RATE)

    retuned = _assistant_audio(1.0, 22050)
    already_24k = _assistant_audio(1.0, SAMPLING_RATE)
    already_24k.value = np.linspace(-1, 1, SAMPLING_RATE, dtype=np.float32)
    original_24k = already_24k.value.copy()
    user = ConversationItem(
        type="audio",
        value=np.zeros(22050, dtype=np.float32),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=22050, num_samples=22050)},
    )

    preprocessor.preprocess_conversations([[user, retuned, already_24k]])

    assert retuned.meta[AUDIO_METADATA_KEY].sampling_rate == SAMPLING_RATE
    assert retuned.meta[AUDIO_METADATA_KEY].num_samples == retuned.value.numel()
    assert retuned.meta[AUDIO_METADATA_KEY].duration_seconds == pytest.approx(1.0, rel=1e-3)
    assert torch.equal(already_24k.value, torch.from_numpy(original_24k))
    for item in (retuned, already_24k):
        assert isinstance(item.value, torch.Tensor) and _as_mono_batch(item.value).shape[:2] == (1, 1)
    assert user.meta[AUDIO_METADATA_KEY].sampling_rate == 22050 and user.value.size == 22050

    # Inference is a no-op; missing rate on assistant audio raises.
    infer_item = _assistant_audio(1.0, 22050)
    preprocessor.preprocess_conversations([[infer_item]], inference=True)
    assert infer_item.meta[AUDIO_METADATA_KEY].sampling_rate == 22050
    with pytest.raises(ValueError, match="states no meta"):
        preprocessor.preprocess_conversations(
            [[ConversationItem(type="audio", value=np.zeros(1000, dtype=np.float32), role="assistant", meta={})]]
        )

    # AudioMetadata.resampled_to validates duration and bounds.
    original = AudioMetadata(sampling_rate=22050, num_samples=22050)
    assert original.resampled_to(24000, 24001).duration_seconds == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError, match="would retime it"):
        original.resampled_to(24000, 48000)
    with pytest.raises(ValueError, match="sampling_rate must be positive"):
        original.resampled_to(0, 24000)
    with pytest.raises(ValueError, match="num_samples must be non-negative"):
        original.resampled_to(24000, -1)


@pytest.fixture
def single_rank_group(tmp_path):
    """One rank on the local accelerator for the FSDP2 quantizer test."""
    import torch.distributed as dist

    from veomni.distributed.parallel_state import clear_parallel_state
    from veomni.utils.device import (
        IS_CUDA_AVAILABLE,
        IS_NPU_AVAILABLE,
        get_dist_comm_backend,
        get_torch_device,
    )

    if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE):
        pytest.skip("needs an accelerator for the FSDP2 path")

    get_torch_device().set_device(0)
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{tmp_path / 'rendezvous'}",
        world_size=1,
        rank=0,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
def test_encode_works_with_the_quantizer_sharded(single_rank_group):
    """MimiResidualVectorQuantizer is invoked by method call, so FSDP2 units must stay unsharded."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    from veomni.distributed.utils import sort_fqn_by_submodule_first
    from veomni.models.seed_omni.accelerated.utils.dispatch import call_graph_endpoint
    from veomni.utils.device import get_device_type

    device = f"{get_device_type()}:0"
    module = _accelerated().to(device)
    mesh = init_device_mesh(get_device_type(), (1,))

    wrap_targets = set(module._no_split_modules)
    by_fqn = {fqn: mod for fqn, mod in module.named_modules() if type(mod).__name__ in wrap_targets}
    units = [by_fqn[fqn] for fqn in sort_fqn_by_submodule_first(list(by_fqn))]
    for unit in units:
        fully_shard(unit, mesh=mesh)
    fully_shard(module, mesh=mesh)

    quantizers = [m for m in units if "VectorQuantizer" in type(m).__name__]
    assert quantizers and all(any(isinstance(p, DTensor) for p in q.parameters()) for q in quantizers)

    item = _assistant_audio(0.05)
    item.value = item.value.to(device)
    with torch.no_grad():
        inputs = module.pre_forward(method="encode", conversation_list=[[item]])
        outputs = call_graph_endpoint(module, module, method="encode", kwargs=inputs)
        out = module.post_forward(method="encode", **outputs)

    codes = out["conversation_list"][0][0].meta[TALKER_CODE_LABELS]
    assert codes.shape[0] == NUM_QUANTIZERS and not isinstance(codes, DTensor)
