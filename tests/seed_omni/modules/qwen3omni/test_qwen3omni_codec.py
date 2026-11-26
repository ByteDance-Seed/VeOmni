"""Unit tests for the Qwen3-Omni audio codec OmniModule.

Hermetic tests only: structure, contracts and the wrapper's fidelity to
upstream, all on random weights. A random-weight quantizer collapses (every
frame encodes to the same index), so the claim this module actually rests on —
that Mimi's encoder and Qwen's decoder share an index space — cannot be tested
here. That evidence needs both real checkpoints and lives in
``scripts/seed_omni/qwen3omni_audio_check.py``, which also carries the control
that separates intrinsic RVQ residual drift from a genuine codec mismatch.
"""

import numpy as np
import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

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
)
from veomni.models.seed_omni.modules.qwen3omni.codec.processing import Qwen3OmniCodecPreprocessor
from veomni.models.seed_omni.utils.conversation import ConversationItem


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
    # 2*2*4 == 16 == 24000 / the tiny Mimi's 1500 Hz frame rate, so the two
    # halves agree on samples-per-code-frame the way the real pair does at 1920.
    upsample_rates=[2, 2],
    upsampling_ratios=[4],
    sliding_window=8,
    max_position_embeddings=128,
)

# Deeper than the decoder needs (20 > 16) so the truncation property below is
# actually exercised rather than trivially true, mirroring the real pair (32
# quantizers encoded, 16 consumed).
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


def _codes(frames: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, CODEBOOK_SIZE, (NUM_QUANTIZERS, frames))


def test_config_carries_decode_window_and_derives_audio_rates():
    config = _tiny_config(chunk_size=300, left_context_size=25)
    assert config.model_type == "qwen3omni_codec"
    # upstream's hardcoded call-site values, now overridable from a YAML
    assert (config.chunk_size, config.left_context_size) == (300, 25)
    # Depth is the decoder's (what the talker emits); rates are the encoder's
    # (the Qwen checkpoint declares neither).
    assert config.num_quantizers == NUM_QUANTIZERS
    assert config.sampling_rate == SAMPLING_RATE
    assert config.frame_rate == config.mimi_config.frame_rate
    assert config.tie_word_embeddings is False


def test_mismatched_codebook_sizes_are_rejected():
    """Indices produced against one codebook are meaningless against another."""
    with pytest.raises(ValueError, match="disagree on codebook size"):
        _tiny_config(mimi_config={**TINY_MIMI_CONFIG, "codebook_size": CODEBOOK_SIZE * 2})


def test_encoder_shallower_than_decoder_is_rejected():
    """An encoder with fewer quantizers cannot fill every code group."""
    with pytest.raises(ValueError, match="only 8 quantizers"):
        _tiny_config(mimi_config={**TINY_MIMI_CONFIG, "num_quantizers": 8})


def test_disagreeing_samples_per_code_frame_is_rejected():
    """The one mismatch that would never raise on its own.

    Nothing downstream notices halves that disagree on frame duration: the
    decode succeeds, the carrier is tagged with the encoder's sampling rate, and
    the audio simply plays at the wrong speed.
    """
    with pytest.raises(ValueError, match="how much audio one code frame spans"):
        _tiny_config(code2wav_config={**TINY_CODE2WAV_CONFIG, "upsampling_ratios": [8]})


def test_real_checkpoint_rates_agree():
    """Pins the arithmetic the check relies on, at the real pair's values.

    24000 Hz / 12.5 Hz == 8*5*4*3 * 2*2 == 1920 samples per code frame.
    """
    config = _tiny_config(
        code2wav_config={**TINY_CODE2WAV_CONFIG, "upsample_rates": [8, 5, 4, 3], "upsampling_ratios": [2, 2]},
        mimi_config={**TINY_MIMI_CONFIG, "upsampling_ratios": [8, 6, 5, 4]},
    )
    assert config.decoder_samples_per_frame == 1920
    assert config.encoder_samples_per_frame == 1920
    assert config.frame_rate == 12.5


@pytest.mark.parametrize("key,value", [("sampling_rate", SAMPLING_RATE), ("num_quantizers", NUM_QUANTIZERS)])
def test_a_config_repeating_a_derived_value_is_accepted(key, value):
    """``sampling_rate`` was a stored field before the merge.

    A config still carrying it must not die on the read-only property with an
    ``AttributeError`` that names the symptom instead of the cause.
    """
    assert getattr(_tiny_config(**{key: value}), key) == value


def test_a_config_contradicting_a_derived_value_is_rejected():
    with pytest.raises(ValueError, match="derived from the two halves"):
        _tiny_config(sampling_rate=16000)


def test_encode_returns_decoder_depth_codes():
    module = _tiny_module()
    with torch.no_grad():
        codes = module.encode_wav(torch.randn(SAMPLING_RATE), sampling_rate=SAMPLING_RATE)
    assert codes.dim() == 2
    assert codes.size(0) == NUM_QUANTIZERS
    assert codes.dtype == torch.long
    assert 0 <= int(codes.min()) and int(codes.max()) < CODEBOOK_SIZE


def test_truncating_a_deeper_encode_is_exact():
    """Why taking Mimi's first 16 of 32 quantizers is exact, not approximate.

    RVQ is sequential: each quantizer encodes the residual left by the previous
    ones, so the first n codes do not depend on how many follow. If this ever
    broke, every talker target would be subtly wrong.
    """
    module = _tiny_module()
    waveform = torch.randn(1, SAMPLING_RATE)
    with torch.no_grad():
        shallow = module.mimi.encode(waveform.unsqueeze(0), num_quantizers=NUM_QUANTIZERS).audio_codes
        deep = module.mimi.encode(waveform.unsqueeze(0), num_quantizers=TINY_MIMI_CONFIG["num_quantizers"]).audio_codes
    assert torch.equal(deep[:, :NUM_QUANTIZERS], shallow)


@pytest.mark.parametrize(
    "shape", [(SAMPLING_RATE,), (1, SAMPLING_RATE), (1, 1, SAMPLING_RATE)], ids=["flat", "2d", "3d"]
)
def test_encode_accepts_the_usual_mono_layouts(shape):
    module = _tiny_module()
    with torch.no_grad():
        codes = module.encode_wav(torch.randn(*shape), sampling_rate=SAMPLING_RATE)
    assert codes.size(0) == NUM_QUANTIZERS


def test_encode_rejects_a_mismatched_sampling_rate():
    """Silent resampling would stretch the code timeline and misalign every target."""
    module = _tiny_module()
    with pytest.raises(ValueError, match="16000 Hz audio but this codec operates at 24000 Hz"):
        module.encode_wav(torch.randn(16000), sampling_rate=16000)
    # The matching rate is accepted, so the guard is not just always-on.
    with torch.no_grad():
        module.encode_wav(torch.randn(SAMPLING_RATE), sampling_rate=SAMPLING_RATE)


def test_encode_rejects_multichannel_and_batched_audio():
    module = _tiny_module()
    with pytest.raises(ValueError, match="expected mono audio"):
        module.encode_wav(torch.randn(2, SAMPLING_RATE), sampling_rate=SAMPLING_RATE)
    with pytest.raises(ValueError, match="single mono clip"):
        module.encode_wav(torch.randn(2, 1, SAMPLING_RATE), sampling_rate=SAMPLING_RATE)
    with pytest.raises(ValueError, match="cannot read a waveform"):
        module.encode_wav(torch.randn(1, 1, 1, SAMPLING_RATE), sampling_rate=SAMPLING_RATE)


def test_encode_then_decode_runs_end_to_end():
    """Both halves compose on one module — shape/plumbing only, not fidelity.

    Random weights collapse the quantizer, so nothing here says the audio is
    reconstructed; see the module docstring for where that is measured.
    """
    module = _tiny_module()
    with torch.no_grad():
        codes = module.encode_wav(torch.randn(SAMPLING_RATE // 4), sampling_rate=SAMPLING_RATE)
        waveform = module.decode_codes(codes)
    assert waveform.dim() == 3  # (batch, channel, samples)
    assert waveform.size(-1) > 0


@pytest.mark.parametrize("frames", [4, 8, 20])
def test_decode_matches_upstream(frames):
    """Same weights, same window, same device/dtype -> the same samples.

    Spans the chunking regimes: shorter than one chunk, exactly one chunk, and
    several chunks with left context (chunk_size=8, left_context_size=2).

    Compared with a tolerance rather than ``torch.equal``: this module delegates
    the whole decode to upstream's own ``chunked_decode``, so the two runs differ
    only in how the compiler reassociated the same floating-point operations
    (measured at ~1e-10 absolute on CPU). A bit-identity assertion here would
    pin the build's reduction order, not this module's behaviour — and it does
    fail exactly that way on a host whose BLAS splits the GEMM differently.
    """
    module = _tiny_module()
    # The reference is the *same class* our module wraps, so it has to be built
    # with the same op selections — on CPU that means eager attention, otherwise
    # `_from_config` reaches for the flash-attention kernel and fails there
    # rather than in the code under test.
    upstream_config = Qwen3OmniMoeCode2WavConfig(**TINY_CODE2WAV_CONFIG)
    upstream_config.ops_implementation = make_eager_ops_config().to_dict()
    upstream = Qwen3OmniMoeCode2Wav._from_config(upstream_config).eval()
    upstream.load_state_dict(
        {key[len("code2wav.") :]: value for key, value in module.state_dict().items() if key.startswith("code2wav.")},
        strict=True,
    )

    codes = _codes(frames)
    with torch.no_grad():
        mine = module.decode_codes(codes)
        reference = upstream.chunked_decode(codes.unsqueeze(0), chunk_size=8, left_context_size=2).float()
    torch.testing.assert_close(mine, reference, rtol=0, atol=1e-6)


def test_decode_accepts_batched_and_unbatched_codes():
    module = _tiny_module()
    codes = _codes(8)
    with torch.no_grad():
        unbatched = module.decode_codes(codes)
        batched = module.decode_codes(codes.unsqueeze(0))
    assert torch.equal(unbatched, batched)
    assert unbatched.dim() == 3  # (batch, channel, samples)


def test_wrong_code_group_count_is_rejected():
    """A transposed or truncated code tensor would otherwise decode to noise."""
    module = _tiny_module()
    with pytest.raises(ValueError, match="expected 16 RVQ code groups"):
        module.decode_codes(_codes(8).transpose(0, 1))


def test_generate_appends_an_audio_item_and_signals_done():
    module = _tiny_module()
    conversation = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="output", value=_codes(8), role="assistant", meta={TALKER_CODES_KEY: True}),
    ]
    with torch.no_grad():
        outputs = module.generate(conversation_list=conversation)

    assert outputs[FSM_SIGNAL_KEY] == SIGNAL_AUDIO_DONE
    tail = outputs["conversation_list"][-1]
    assert tail.type == "audio"
    assert tail.role == "assistant"
    assert tail.meta[AUDIO_METADATA_KEY].sampling_rate == SAMPLING_RATE
    # Synthesised, so this is the only place the clip's length is known;
    # the inferencer reads both halves to write a wav.
    assert tail.meta[AUDIO_METADATA_KEY].num_samples == tail.value.numel()

    # `generated` has to survive OmniModel's normalizer, which keeps only
    # `{type, value}` dicts. Handing back the bare waveform silently drops the
    # audio and leaves a graph that ran to completion with nothing to show.
    normalized = OmniModel._normalize_generated(outputs["generated"])
    assert normalized is not None
    assert normalized["type"] == "audio"
    assert normalized["meta"][AUDIO_METADATA_KEY].sampling_rate == SAMPLING_RATE
    assert torch.equal(tail.value, normalized["value"])


def test_generate_without_codes_raises():
    """An empty waveform out of a TTS graph reads as a quality problem, not a wiring bug."""
    module = _tiny_module()
    with pytest.raises(ValueError, match="no output row stamped 'qwen3omni_talker_codes'"):
        module.generate(conversation_list=[ConversationItem(type="text", value="hi", role="user")])


# ── training node: assistant speech -> talker code targets ────────────────────


def _accelerated() -> Qwen3OmniCodecAccelerated:
    torch.manual_seed(0)
    return Qwen3OmniCodecAccelerated(_tiny_config()).eval()


def _assistant_audio(seconds: float, rate: int = SAMPLING_RATE) -> ConversationItem:
    torch.manual_seed(2)
    return ConversationItem(
        type="audio",
        value=torch.randn(int(rate * seconds)) * 0.1,
        role="assistant",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=rate, num_samples=int(rate * seconds))},
    )


def _run_encode(module, conversation_list):
    """Drive the node the way OmniModel._run_train_node does."""
    inputs = module.pre_forward(method="encode", conversation_list=conversation_list)
    outputs = module.encode(**inputs)
    return module.post_forward(method="encode", **outputs)


def test_encode_stashes_code_targets_on_the_assistant_audio_item():
    module = _accelerated()
    item = _assistant_audio(0.05)
    with torch.no_grad():
        out = _run_encode(module, [[item]])

    codes = item.meta[TALKER_CODE_LABELS]
    assert out["conversation_list"][0][0] is item
    assert codes.shape[0] == NUM_QUANTIZERS
    assert codes.dtype == torch.long
    # Targets annotate the audio rather than becoming another turn: a row of
    # their own would put codes inside the conversation the chat template walks.
    assert [i.type for i in out["conversation_list"][0]] == ["audio"]


def test_encode_never_touches_the_user_side_audio():
    """The user's speech is a 16kHz *input* to the thinker, not a target."""
    module = _accelerated()
    user = ConversationItem(
        type="audio",
        value=torch.randn(1200),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=16000, num_samples=1200)},
    )
    assistant = _assistant_audio(0.05)
    with torch.no_grad():
        _run_encode(module, [[user, assistant]])

    assert TALKER_CODE_LABELS not in user.meta
    assert TALKER_CODE_LABELS in assistant.meta


def test_encode_is_a_noop_when_a_batch_has_no_assistant_audio():
    # The codec is frozen, so unlike a trained encoder it needs no dummy anchor
    # to keep FSDP producing a gradient for it.
    module = _accelerated()
    text = ConversationItem(type="text", value="hi", role="user")
    with torch.no_grad():
        out = _run_encode(module, [[text]])
    assert out["conversation_list"] == [[text]]


def test_encode_refuses_audio_that_does_not_declare_its_rate():
    module = _accelerated()
    item = _assistant_audio(0.05)
    del item.meta[AUDIO_METADATA_KEY]
    with pytest.raises(ValueError, match="states no meta"):
        _run_encode(module, [[item]])


def test_encode_refuses_audio_loaded_at_the_wrong_rate():
    # The failure this prevents: a 16kHz corpus tokenized as if it were 24kHz
    # yields targets whose timeline runs 1.5x fast, which trains fine and only
    # shows up as speech at the wrong speed.
    module = _accelerated()
    item = _assistant_audio(0.05, rate=16000)
    with pytest.raises(ValueError, match="but this codec operates at"):
        _run_encode(module, [[item]])


def test_each_clip_is_encoded_alone_so_padding_cannot_leak_into_targets():
    """A short clip batched beside a long one must not gain codes for silence."""
    module = _accelerated()
    short, long = _assistant_audio(0.04), _assistant_audio(0.09)
    with torch.no_grad():
        _run_encode(module, [[short], [long]])
        alone = module.encode_wav(short.value, sampling_rate=SAMPLING_RATE)

    together = short.meta[TALKER_CODE_LABELS]
    assert torch.equal(together, alone)
    assert together.shape[1] < long.meta[TALKER_CODE_LABELS].shape[1]


def test_code_targets_round_trip_through_the_vocoder():
    """The targets must be exactly what decode_codes accepts, shape and depth."""
    module = _accelerated()
    item = _assistant_audio(0.05)
    with torch.no_grad():
        _run_encode(module, [[item]])
        waveform = module.decode_codes(item.meta[TALKER_CODE_LABELS])
    assert waveform.dim() == 3 and waveform.size(0) == 1


# ── FSDP2: the quantizer is reached by method, not by __call__ ────────────────


@pytest.fixture
def single_rank_group(tmp_path):
    """One rank on the local accelerator, enough to drive the FSDP2 wrap."""
    import torch.distributed as dist

    from veomni.distributed.parallel_state import clear_parallel_state
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

    if not torch.accelerator.is_available():
        pytest.skip("needs an accelerator for the FSDP2 path")

    get_torch_device().set_device(f"{get_device_type()}:0")
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


def test_encode_works_with_the_quantizer_sharded(single_rank_group):
    """The one FSDP2 shape this module has to survive, exercised end to end.

    Wraps exactly what ``fsdp_scope: module`` wraps — every module whose class
    is named in the codec's own ``_no_split_modules`` — rather than a
    hand-picked module. That is the point of the test: those entries name
    ``MimiResidualVectorQuantizer``, and every path to one is a method call, so
    FSDP2's ``__call__`` hooks never fire. Without the module's
    ``register_fsdp_forward_method`` calls this raises a bare mixed-tensor error
    from the quantizer's ``input_proj``. Deriving the wrap list from the product
    is what makes the test fail when the registration targets the wrong module,
    which a hand-picked wrap does not.

    Wrapped bottom-up through ``sort_fqn_by_submodule_first``, as
    ``parallelize_model_fsdp2`` does, because the effective targets nest: HF
    ``post_init`` unions in ``MimiModel._no_split_modules``, which names the
    enclosing ``MimiSplitResidualVectorQuantizer``. Sharding a parent first
    absorbs the children's parameters into its group, and ``fully_shard`` on
    the child then takes its TP/EP branch and dies inside
    ``DeviceMesh._concatenate`` on a mesh with no ``mesh_dim_names``.

    Driven through ``call_graph_endpoint`` rather than through ``_run_encode``:
    the trampoline is what the graph uses, and it is what unshards the module's
    *own* unit (which owns Mimi's convolutional encoder). Calling ``encode`` as
    a plain method would fail on the convs before reaching the quantizer.
    """
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh
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

    # The preconditions the test rests on: without them the encode below would
    # pass for the trivial reason that nothing relevant was sharded.
    quantizers = [m for m in units if "VectorQuantizer" in type(m).__name__]
    assert quantizers, f"no quantizer unit among {wrap_targets}"
    assert all(any(isinstance(p, DTensor) for p in q.parameters()) for q in quantizers)

    item = _assistant_audio(0.05)
    item.value = item.value.to(device)
    with torch.no_grad():
        inputs = module.pre_forward(method="encode", conversation_list=[[item]])
        outputs = call_graph_endpoint(module, module, method="encode", kwargs=inputs)
        out = module.post_forward(method="encode", **outputs)

    codes = out["conversation_list"][0][0].meta[TALKER_CODE_LABELS]
    assert codes.shape[0] == NUM_QUANTIZERS
    assert not isinstance(codes, DTensor)


def test_the_codec_retunes_assistant_speech_and_moves_the_timeline_with_it():
    """``encode_wav`` refuses any other rate, and this is what satisfies it.

    Payload and metadata move together: a row left claiming 22.05 kHz x 22050
    samples would read as 0.92 s once the payload became 24 kHz.
    """
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = _assistant_audio(1.0, 22050)

    preprocessor.preprocess_conversations([[item]])

    metadata = item.meta[AUDIO_METADATA_KEY]
    assert metadata.sampling_rate == 24000
    assert metadata.num_samples == item.value.numel()
    assert metadata.duration_seconds == pytest.approx(1.0, rel=1e-3)


def test_what_the_codec_leaves_on_the_row_is_what_encode_wav_can_read():
    """A tensor, not the numpy array the data layer loaded.

    ``encode_pre`` hands ``item.value`` to ``encode_wav`` verbatim and its shape
    normaliser calls ``.dim()``.
    """
    from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import _as_mono_batch

    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    items = [_assistant_audio(1.0, 22050), _assistant_audio(1.0, 24000)]

    preprocessor.preprocess_conversations([items])

    for item in items:
        assert isinstance(item.value, torch.Tensor), f"{item.value.__class__.__name__} would break encode_wav"
        assert _as_mono_batch(item.value).shape[:2] == (1, 1)


def test_a_clip_already_at_the_codec_rate_is_not_resampled():
    """Not merely "same length" — resampling 24k to 24k would still filter it."""
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = _assistant_audio(1.0, 24000)
    item.value = np.linspace(-1, 1, 24000, dtype=np.float32)
    original = item.value.copy()

    preprocessor.preprocess_conversations([[item]])

    assert torch.equal(item.value, torch.from_numpy(original))


def test_the_codec_never_touches_the_user_side():
    """Same row type, opposite meaning — the audio tower owns the user's side."""
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = ConversationItem(
        type="audio",
        value=np.zeros(22050, dtype=np.float32),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=22050, num_samples=22050)},
    )

    preprocessor.preprocess_conversations([[item]])

    assert item.meta[AUDIO_METADATA_KEY].sampling_rate == 22050
    assert item.value.size == 22050


def test_the_codec_refuses_assistant_audio_that_declares_no_rate():
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = ConversationItem(type="audio", value=np.zeros(1000, dtype=np.float32), role="assistant", meta={})

    with pytest.raises(ValueError, match="states no meta"):
        preprocessor.preprocess_conversations([[item]])


def test_resampled_to_refuses_a_conversion_that_retimed_the_clip():
    """Retuning must not retime it: the codes are scored against text, so a clip
    that silently got longer trains the talker against shifted targets.

    Both fields move together, and the duration is checked across them — the one
    caller (the codec preprocessor) resamples with librosa, whose length is
    rounded, so the tolerance is a single sample at the coarser rate.
    """
    original = AudioMetadata(sampling_rate=22050, num_samples=22050)  # 1.0 s

    # 24 kHz for the same second, rounded to a whole sample: accepted.
    assert original.resampled_to(24000, 24000).duration_seconds == pytest.approx(1.0)
    assert original.resampled_to(24000, 24001).duration_seconds == pytest.approx(1.0, abs=1e-3)
    # A count that only makes sense as a *different* clip: refused.
    with pytest.raises(ValueError, match="would retime it"):
        original.resampled_to(24000, 48000)

    for sampling_rate in (0, -1):
        with pytest.raises(ValueError, match="sampling_rate must be positive"):
            original.resampled_to(sampling_rate, 24000)
    with pytest.raises(ValueError, match="num_samples must be non-negative"):
        original.resampled_to(24000, -1)


def test_the_codec_does_nothing_at_inference():
    """The assistant's speech is generation's output there, not its input."""
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = _assistant_audio(1.0, 22050)

    preprocessor.preprocess_conversations([[item]], inference=True)

    assert item.meta[AUDIO_METADATA_KEY].sampling_rate == 22050
