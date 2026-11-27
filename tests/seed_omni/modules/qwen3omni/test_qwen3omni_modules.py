"""Unit tests for Qwen3-Omni OmniModules (audio, text_encoder, vision, llm, talker, codec)."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers.masking_utils import find_packed_sequence_indices
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeVisionEncoder,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration as _Ref,
)

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.hf_paths import hf_local_or_remote
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.seed_omni.utils.media_metadata import (
    AUDIO_METADATA_KEY,
    VIDEO_METADATA_KEY,
    AudioMetadata,
    VideoMetadata,
)
from veomni.data.seed_omni.utils.video import VideoInputs
from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.base.llm_packing import scatter_llm_hidden_states
from veomni.models.seed_omni.modules.base.sp_shard import sp_local_len, sp_valid_in_window
from veomni.models.seed_omni.modules.qwen3omni.audio.accelerated.accelerated import _scatter_audio_window
from veomni.models.seed_omni.modules.qwen3omni.audio.configuration import Qwen3OmniAudioEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.audio.modeling import (
    Qwen3OmniAudioEncoder,
    audio_output_lengths,
    encode_qwen3omni_audio_items,
)
from veomni.models.seed_omni.modules.qwen3omni.audio.processing import _OMNI_MEL_LEN, Qwen3OmniAudioPreprocessor
from veomni.models.seed_omni.modules.qwen3omni.carrier import TALKER_CODES_KEY, is_thinker_input
from veomni.models.seed_omni.modules.qwen3omni.codec.accelerated.accelerated import Qwen3OmniCodecAccelerated
from veomni.models.seed_omni.modules.qwen3omni.codec.configuration import Qwen3OmniCodecConfig
from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import (
    SIGNAL_AUDIO_DONE,
    TALKER_CODE_LABELS,
    Qwen3OmniCodec,
    _as_mono_batch,
)
from veomni.models.seed_omni.modules.qwen3omni.codec.processing import Qwen3OmniCodecPreprocessor
from veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated import (
    Qwen3OmniLlmAccelerated,
    _fold_fsdp_dummy_anchors,
    _qwen3omni_modality_groups,
)
from veomni.models.seed_omni.modules.qwen3omni.llm.configuration import Qwen3OmniLlmConfig
from veomni.models.seed_omni.modules.qwen3omni.llm.modeling import (
    THINKER_EMBED,
    THINKER_HIDDEN,
    Qwen3OmniLlm,
    pack_qwen3omni_conversations_for_forward,
    qwen3omni_vision_position_ids,
)
from veomni.models.seed_omni.modules.qwen3omni.talker.accelerated.accelerated import (
    _talker_prompt,
    _target_codes,
    _thinker_traces,
)
from veomni.models.seed_omni.modules.qwen3omni.talker.configuration import Qwen3OmniTalkerConfig
from veomni.models.seed_omni.modules.qwen3omni.talker.modeling import (
    SIGNAL_CODES_DONE,
    Qwen3OmniTalker,
    _chatml_turns,
    _speech_prefix_length,
    _speech_span_end,
    collect_media_metadata,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.accelerated.accelerated import (
    Qwen3OmniTextEncoderAccelerated,
    _stash_talker_prompts,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.accelerated.accelerated import (
    TrainingMixin as Qwen3OmniTrainingMixin,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.chat_template import Qwen3OmniChatTemplate
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.configuration import Qwen3OmniTextEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import (
    PromptSegment,
    Qwen3OmniTextEncoder,
    talker_prompt_segments,
    talker_supervised_mask,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.processing import Qwen3OmniTextEncoderPreprocessor
from veomni.models.seed_omni.modules.qwen3omni.vision.configuration import Qwen3OmniVisionEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.vision.modeling import (
    Qwen3OmniVisionEncoder,
    build_qwen3omni_vit_metadata,
    scatter_qwen3omni_visual_embeds,
)
from veomni.models.seed_omni.modules.qwen3omni.vision.processing import (
    _OMNI_GRID,
    Qwen3OmniVisionPreprocessor,
    _second_per_grid,
)
from veomni.models.seed_omni.modules.text_encoder.accelerated.accelerated import TrainingMixin as BaseTrainingMixin
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem
from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as _Upstream,
)
from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths
from veomni.models.utils.mask_utils import _should_use_packed_mask
from veomni.ops import VeomniOp
from veomni.utils.tensor_utils import unflatten


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
CONFIG_DIR = Path(__file__).parents[4] / "tests" / "toy_config" / "qwen3omni_toy"
TOKENIZER = hf_local_or_remote("Qwen/Qwen3-Omni-30B-A3B-Instruct")
requires_tokenizer = pytest.mark.skipif(
    not os.path.isdir(TOKENIZER), reason=f"needs a Qwen3-Omni tokenizer at {TOKENIZER}"
)

# Shared tiny constants
NUM_MEL_BINS, AUDIO_SR, POS_PER_SEC = 128, 16000, 13
CODEC_QUANTIZERS, CODEC_CODEBOOK, CODEC_SR = 16, 128, 24000
THINKER_HIDDEN_SIZE = 24
IM_START, SYSTEM, USER, ASSISTANT = 1000, 1001, 1002, 1003
IM_END, AUDIO_START, AUDIO_END, AUDIO_PAD = 1004, 1005, 1006, 1007
TTS_BOS, TTS_EOS, TTS_PAD = 1008, 1009, 1010
ROLE_TOKEN_IDS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
HIDDEN, MERGE = 8, 2
VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD = 151652, 151653, 151655, 151656
UP_AUDIO_START, UP_AUDIO_PAD = 151669, 151675
_WTE = None

TINY_TEXT_CONFIG = dict(
    num_hidden_layers=6,
    hidden_size=64,
    intermediate_size=64,
    moe_intermediate_size=32,
    num_experts=4,
    num_experts_per_tok=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=64,
    dtype="float32",
)

TINY_CODE2WAV_CONFIG = dict(
    codebook_size=CODEC_CODEBOOK,
    semantic_codebook_size=CODEC_CODEBOOK,
    num_quantizers=CODEC_QUANTIZERS,
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
    codebook_size=CODEC_CODEBOOK,
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
    sampling_rate=CODEC_SR,
    audio_channels=1,
)


# ── audio & text_encoder ─────────────────────────────────────────────────────


def _tiny_audio_config() -> Qwen3OmniAudioEncoderConfig:
    return Qwen3OmniAudioEncoderConfig(
        ops_implementation=make_eager_ops_config(),
        audio_config={
            "num_mel_bins": NUM_MEL_BINS,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "d_model": 32,
            "encoder_ffn_dim": 64,
            "downsample_hidden_size": 16,
            "output_dim": 48,
            "n_window": 50,
            "n_window_infer": 100,
        },
    )


def test_audio_preprocessor_encoder_and_sp_scatter():
    pytest.importorskip("librosa")
    prep = Qwen3OmniAudioPreprocessor(
        WhisperFeatureExtractor(feature_size=NUM_MEL_BINS, sampling_rate=AUDIO_SR), n_window=50
    )

    np_item = ConversationItem(type="audio", value=np.zeros(AUDIO_SR * 2, dtype=np.float32), role="user")
    pt_item = ConversationItem(type="audio", value=torch.zeros(1, AUDIO_SR * 2), role="user")
    resampled = ConversationItem(
        type="audio",
        value=np.zeros(24000 * 4, dtype=np.float32),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=24000, num_samples=24000 * 4)},
    )
    text = ConversationItem(type="text", value="hello", role="user")
    prep.preprocess_conversations([[np_item, pt_item, resampled, text]], inference=True)

    mel_len, mel_bins = np_item.value.shape
    assert mel_bins == NUM_MEL_BINS and mel_len < 3000 and tuple(pt_item.value.shape) == (mel_len, mel_bins)
    assert np_item.meta[_OMNI_MEL_LEN] == mel_len and text.value == "hello"
    assert np_item.meta["_omni_token_len"] == 2 * POS_PER_SEC < mel_len
    assert abs(int(audio_output_lengths(torch.tensor([resampled.meta[_OMNI_MEL_LEN]]), 50)[0]) - 4 * POS_PER_SEC) <= 1

    with pytest.raises(ValueError, match="mono"):
        prep.preprocess_conversations(
            [[ConversationItem(type="audio", value=np.zeros((AUDIO_SR, 2), dtype=np.float32), role="user")]],
            inference=True,
        )
    with pytest.raises(ValueError, match="silently truncate"):
        prep.preprocess_conversations(
            [
                [
                    ConversationItem(
                        type="audio",
                        value=np.zeros(40 * AUDIO_SR, dtype=np.float32),
                        role="user",
                        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=AUDIO_SR, num_samples=40 * AUDIO_SR)},
                    )
                ]
            ],
            inference=True,
        )

    # Encoder length parity, empty forward, attn passthrough, generate skipping assistant speech.
    torch.manual_seed(0)
    model = Qwen3OmniAudioEncoder(_tiny_audio_config()).eval()
    lengths = torch.tensor([1, 13, 37, 100, 201, 1000])
    assert audio_output_lengths(lengths, 50).tolist() == _get_feat_extract_output_lengths(lengths).tolist()
    assert model._check_and_adjust_attn_implementation("sdpa") == "sdpa"
    assert model.forward(input_features=None, feature_lens=None) == {"audio_embeds": None}

    items = [
        ConversationItem(type="audio", value=torch.randn(m, NUM_MEL_BINS), role="user", meta={_OMNI_MEL_LEN: m})
        for m in (13, 100, 201)
    ]
    assistant = ConversationItem(
        type="audio", value=torch.randn(100, NUM_MEL_BINS), role="assistant", meta={_OMNI_MEL_LEN: 100}
    )
    dummy = ConversationItem(type="audio", value=torch.zeros(8), role="user", is_dummy=True)
    assert is_thinker_input(items[0]) and not is_thinker_input(assistant) and not is_thinker_input(dummy)

    with torch.no_grad():
        encode_qwen3omni_audio_items(model, items)
        encode_qwen3omni_audio_items(model, [])
        conv = [
            assistant,
            ConversationItem(
                type="audio", value=torch.randn(100, NUM_MEL_BINS), role="user", meta={_OMNI_MEL_LEN: 100}
            ),
        ]
        assert model.generate(conversation_list=conv)["conversation_list"] is conv
    assert [it.value.size(0) for it in items] == audio_output_lengths(torch.tensor([13, 100, 201]), 50).tolist()
    assert (
        assistant.value.shape == (100, NUM_MEL_BINS) and conv[1].value.size(-1) == model.config.audio_config.output_dim
    )

    bad_cfg = _tiny_audio_config()
    bad_cfg.audio_config.n_window = 25
    with pytest.raises(ValueError, match="hardcodes 100 frames"):
        Qwen3OmniAudioEncoder(bad_cfg)

    # Audio SP window scatter covers all real rows without overflow.
    mel_lens = [200, 300]
    full = [int(v) for v in audio_output_lengths(torch.tensor(mel_lens), 100).tolist()]
    window, covered = (sum(full) + (-sum(full) % 2)) // 2, 0
    for rank in range(2):
        sp_items = [
            ConversationItem(type="audio", value=torch.zeros(n, 4), role="user", meta={_OMNI_MEL_LEN: m})
            for n, m in zip(full, mel_lens)
        ]
        _scatter_audio_window(sp_items, torch.randn(window, 4), n_window=100, sp_rank=rank, sp_size=2)
        rows = sum(it.value.size(0) for it in sp_items)
        assert rows <= window
        covered += rows
    assert covered == sum(full)


def test_text_encoder_untied_tts_embeds_and_decode():
    cfg = Qwen3OmniTextEncoderConfig(vocab_size=64, hidden_size=32, tie_word_embeddings=False)
    assert (cfg.model_type, cfg.lm_head_bias) == ("qwen3omni_text_encoder", False)
    assert Qwen3OmniTextEncoder.preprocessor_class is Qwen3OmniTextEncoderPreprocessor
    assert inspect.signature(Qwen3OmniTextEncoder.generate).parameters["generation_kwargs"].default is None

    module = Qwen3OmniTextEncoder(cfg)
    assert isinstance(module.lm_head, torch.nn.Linear) and module.lm_head.bias is None
    assert module.encode(torch.tensor([[1, 2, 3]]))["inputs_embeds"].shape == (3, 32)
    assert not any("_embed_tokens" in cls.__dict__ for cls in type(module).__mro__)
    module._chat_template = mock.MagicMock(tts_bos_token_id=1, tts_eos_token_id=2, tts_pad_token_id=3)
    with torch.no_grad():
        assert torch.equal(module.tts_embeds(), module.embed_tokens(torch.tensor([[1, 2, 3]])))

    assert "_prepare_decode_inputs" in Qwen3OmniTrainingMixin.__dict__
    assert Qwen3OmniTrainingMixin._prepare_decode_inputs is not BaseTrainingMixin._prepare_decode_inputs
    accel = Qwen3OmniTextEncoderAccelerated(
        Qwen3OmniTextEncoderConfig(vocab_size=64, hidden_size=8, tie_word_embeddings=False)
    )
    sample = [
        ConversationItem(
            type="text", value=torch.zeros(2, 8), role="user", meta={"labels": torch.zeros(2, dtype=torch.long)}
        ),
        ConversationItem(type="audio", value=torch.zeros(2400), role="assistant"),
        ConversationItem(
            type="text", value=torch.zeros(3, 8), role="assistant", meta={"labels": torch.zeros(3, dtype=torch.long)}
        ),
    ]
    hidden, shift_labels = accel._prepare_decode_inputs([sample])
    assert hidden.shape[0] == 5 and shift_labels.shape[0] == 5


# ── vision tower ─────────────────────────────────────────────────────────────


def _tiny_vision() -> Qwen3OmniVisionEncoder:
    with open(os.path.join(CONFIG_DIR, "config.json")) as handle:
        tiny = dict(json.load(handle)["thinker_config"]["vision_config"])
    tiny.update(
        depth=4,
        hidden_size=32,
        out_hidden_size=64,
        num_heads=4,
        intermediate_size=32,
        deepstack_visual_indexes=[1, 2],
        dtype="float32",
    )
    torch.manual_seed(0)
    return Qwen3OmniVisionEncoder(
        Qwen3OmniVisionEncoderConfig(vision_config=tiny, ops_implementation=make_eager_ops_config())
    ).eval()


def _patch_item(item_type, grid, cfg, meta=None) -> ConversationItem:
    t, h, w = grid
    pixel_row = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    torch.manual_seed(sum(grid))
    return ConversationItem(
        type=item_type,
        value=torch.randn(t * h * w, pixel_row),
        role="user",
        meta={_OMNI_GRID: list(grid), **(meta or {})},
    )


def test_vision_encode_generate_preprocessor_and_sp_scatter():
    module = _tiny_vision()
    cfg = module.config.vision_config
    assert module._check_and_adjust_attn_implementation("sdpa") == "sdpa"

    reference = Qwen3OmniMoeVisionEncoder._from_config(cfg).eval()
    missing, unexpected = reference.load_state_dict(module.visual.state_dict(), strict=False)
    assert not missing and not unexpected

    grids = [(1, 4, 4), (2, 2, 2)]
    together = [
        _patch_item("image", grids[0], cfg),
        _patch_item("video", grids[1], cfg, meta={"second_per_grid": 1.0}),
    ]
    alone = [
        [_patch_item("image", grids[0], cfg)],
        [_patch_item("video", grids[1], cfg, meta={"second_per_grid": 1.0})],
    ]
    pixel_values = torch.cat([it.value for it in together], dim=0)
    grid_thw = torch.tensor([list(g) for g in grids], dtype=torch.long)

    with torch.no_grad():
        embeds, deepstack = module._encode(pixel_values, grid_thw, None)
        expected = reference(hidden_states=pixel_values.type(reference.dtype), grid_thw=grid_thw)
        module.generate(conversation_list=list(together))
        for single in alone:
            module.generate(conversation_list=single)

    torch.testing.assert_close(embeds, expected.pooler_output, rtol=0, atol=0)
    for got, want in zip(deepstack, expected.deepstack_features, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(together[0].value, alone[0][0].value)
    torch.testing.assert_close(together[1].value, alone[1][0].value)
    assert len(together[0].meta["deepstack"]) == len(cfg.deepstack_visual_indexes)
    assert together[1].meta["second_per_grid"] == 1.0 and _OMNI_GRID not in together[0].meta

    text_only = [ConversationItem(type="text", value=torch.zeros(3), role="user")]
    assert module.generate(conversation_list=text_only)["conversation_list"] is text_only

    metadata = build_qwen3omni_vit_metadata([[2, 2, 2], [1, 4, 4]], spatial_merge_size=2)
    assert metadata["cu_seqlens"].tolist() == [0, 4, 8, 24] and metadata["max_seqlen"] == 16

    # Preprocessor loads dir config and computes second_per_grid off actual kept frames.
    preprocessor = Qwen3OmniVisionPreprocessor.from_pretrained(CONFIG_DIR)
    assert preprocessor._video_processor.do_sample_frames is False
    vid_meta = VideoMetadata(fps=25.0, total_num_frames=49, frames_indices=[0, 12, 24, 36, 48], requested_fps=2.0)
    assert _second_per_grid(
        ConversationItem(type="video", value=None, role="user", meta={VIDEO_METADATA_KEY: vid_meta}), 1
    ) == pytest.approx(12.0 / 25.0)
    vid_item = ConversationItem(
        type="video",
        value=VideoInputs(video=torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8)),
        role="user",
        meta={
            VIDEO_METADATA_KEY: VideoMetadata(
                fps=8.0, total_num_frames=7, frames_indices=[0, 2, 4, 6], requested_fps=4.0
            )
        },
    )
    img_item = ConversationItem(type="image", value=torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8), role="user")
    preprocessor.preprocess_conversations([[img_item, vid_item]], inference=True)
    assert vid_item.meta["second_per_grid"] == pytest.approx(preprocessor._video_processor.temporal_patch_size * 0.25)
    assert "second_per_grid" not in img_item.meta and len(img_item.meta[_OMNI_GRID]) == 3

    # Vision SP scatter slices tokens and DeepStack together and covers all rows.
    class _Slot:
        def __init__(self, num_merged, item, grid):
            self.num_merged, self.item, self.grid = num_merged, item, grid

    sizes, covered = [4, 2], 0
    window = (sum(sizes) + (-sum(sizes) % 2)) // 2
    for rank in range(2):
        sp_items = [ConversationItem(type="image", value=None, role="user", meta={}) for _ in sizes]
        slots = [_Slot(n, it, torch.tensor((1, 2, 2))) for n, it in zip(sizes, sp_items)]
        scatter_qwen3omni_visual_embeds(
            slots, torch.randn(window, 8), [torch.randn(window, 8), torch.randn(window, 8)], sp_rank=rank, sp_size=2
        )
        assert all(
            it.value.size(0) == len(it.meta["deepstack"][0]) and len(it.meta["deepstack"]) == 2 for it in sp_items
        )
        rows = sum(it.value.size(0) for it in sp_items)
        assert rows <= window
        covered += rows
    assert covered == sum(sizes)


# ── thinker LLM ──────────────────────────────────────────────────────────────


def _text(n: int) -> ConversationItem:
    return ConversationItem(type="text", value=torch.zeros(n, HIDDEN), role="user")


def _media(kind: str, n: int, grid_thw, num_deepstack: int = 2, **meta) -> ConversationItem:
    return ConversationItem(
        type=kind,
        value=torch.zeros(n, HIDDEN),
        role="user",
        meta={
            "grid_thw": torch.tensor(grid_thw),
            "deepstack": [torch.zeros(n, HIDDEN) for _ in range(num_deepstack)],
            **meta,
        },
    )


def _pack(sample):
    return pack_qwen3omni_conversations_for_forward([sample], torch.device("cpu"), MERGE, POS_PER_SEC)


def _accelerated_llm(**config_overrides) -> Qwen3OmniLlmAccelerated:
    torch.manual_seed(0)
    text_config = {**TINY_TEXT_CONFIG, **config_overrides.pop("text_config", {})}
    return Qwen3OmniLlmAccelerated(
        Qwen3OmniLlmConfig(ops_implementation=make_eager_ops_config(), text_config=text_config, **config_overrides)
    ).eval()


def _text_sample(module, lengths):
    hidden = module.config.text_config.hidden_size
    torch.manual_seed(3)
    return [ConversationItem(type="text", value=torch.randn(n, hidden), role="user") for n in lengths]


def _run_forward_node(module, conversation_list):
    inputs = module.pre_forward(method="forward", conversation_list=conversation_list)
    outputs = module.forward(**inputs)
    return module.post_forward(method="forward", **outputs), inputs


@pytest.mark.parametrize(("grid", "second_per_grid"), [((1, 4, 6), 1.0), ((2, 4, 4), 0.5), ((3, 4, 4), 2 / 25)])
def test_llm_vision_positions_match_upstream(grid, second_per_grid):
    grid_t, grid_h, grid_w = grid
    media_len = grid_t * grid_h * grid_w // (MERGE**2)
    is_video = second_per_grid != 1.0
    pad_token = VIDEO_PAD if is_video else IMAGE_PAD

    ns = SimpleNamespace(
        spatial_merge_size=MERGE,
        config=SimpleNamespace(
            image_token_id=IMAGE_PAD,
            video_token_id=VIDEO_PAD,
            audio_token_id=UP_AUDIO_PAD,
            vision_start_token_id=VISION_START,
            audio_start_token_id=UP_AUDIO_START,
            position_id_per_seconds=POS_PER_SEC,
        ),
    )
    ns.get_llm_pos_ids_for_vision = _Upstream.get_llm_pos_ids_for_vision.__get__(ns)
    ns.get_rope_index = _Upstream.get_rope_index.__get__(ns)

    input_ids = torch.tensor([[1, 1, 1, VISION_START, *([pad_token] * media_len), VISION_END, 2, 2]], dtype=torch.long)
    grid_thw = torch.tensor([grid], dtype=torch.long)
    upstream_pos, _ = ns.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=None if is_video else grid_thw,
        video_grid_thw=grid_thw if is_video else None,
        audio_seqlens=torch.tensor([0], dtype=torch.long) if is_video else None,
        second_per_grids=torch.tensor([second_per_grid]) if is_video else None,
    )
    packed = _pack(
        [
            _text(3),
            _text(1),
            _media("video" if is_video else "image", media_len, grid, second_per_grid=second_per_grid),
            _text(3),
        ]
    )
    assert packed["position_ids"].dtype == torch.float and torch.equal(packed["position_ids"], upstream_pos)


def test_llm_pack_tmrope_thinker_states_and_router():
    # Audio + video share wall-clock axis; multi-sample restarts positions; empty & dummy segments preserve contracts.
    audio_len, second_per_grid, grid_t = int(2.0 * POS_PER_SEC), 2 / 25, 5
    av = _pack(
        [
            ConversationItem(type="audio", value=torch.zeros(audio_len, HIDDEN), role="user"),
            _media("video", grid_t * 4 * 4 // (MERGE**2), (grid_t, 4, 4), second_per_grid=second_per_grid),
        ]
    )
    seconds = av["position_ids"][0, 0] / POS_PER_SEC
    assert torch.allclose(
        torch.unique(seconds[audio_len:], sorted=True),
        seconds[audio_len - 1] + 1.0 / POS_PER_SEC + torch.arange(grid_t) * second_per_grid,
    )

    vpos = qwen3omni_vision_position_ids(5, torch.tensor((1, 4, 6)), MERGE, position_id_per_seconds=POS_PER_SEC)
    img_packed = _pack([_text(5), _media("image", 6, (1, 4, 6)), _text(3)])
    assert img_packed["visual_pos_masks"].tolist() == [[False] * 5 + [True] * 6 + [False] * 3]
    resume = float(vpos.max()) + 1.0
    assert img_packed["position_ids"][:, 0, 11:].tolist() == [[resume, resume + 1, resume + 2]] * 3

    multi = pack_qwen3omni_conversations_for_forward([[_text(3)], [_text(5)]], torch.device("cpu"), MERGE, POS_PER_SEC)
    assert multi["cu_seq_lens"].tolist() == [0, 3, 8] and multi["position_ids"][0, 0].tolist() == [
        0,
        1,
        2,
        0,
        1,
        2,
        3,
        4,
    ]
    with pytest.raises(ValueError, match="second_per_grid"):
        _pack([_text(2), _media("video", 4, (1, 4, 4))])

    empty_sample = [_text(2), _text(0), _text(3)]
    packed_empty = _pack(empty_sample)
    scatter_llm_hidden_states(
        [empty_sample], unflatten(packed_empty["inputs_embeds"][0], packed_empty["inputs_embeds_shape"])
    )
    assert [it.value.shape[1] for it in empty_sample] == [HIDDEN] * 3

    dummy = _media("image", 4, (1, 4, 4))
    dummy.is_dummy = True
    with_dummy = _pack([_text(3), dummy])
    assert with_dummy["inputs_embeds"].shape == (1, 3, HIDDEN) and with_dummy["visual_pos_masks"] is None
    assert len(with_dummy["deepstack_visual_embeds"]) == 2

    # Thinker state collection (inference + training) & expert converter registration.
    assert hasattr(Qwen3OmniLlm, "_create_checkpoint_tensor_converter")
    assert hasattr(Qwen3OmniLlm, "_convert_fqn_to_index_mapping")
    infer_mod = Qwen3OmniLlm(
        Qwen3OmniLlmConfig(
            ops_implementation=make_eager_ops_config(),
            text_config=TINY_TEXT_CONFIG,
            accept_hidden_layer=3,
            collect_thinker_states=True,
        )
    ).eval()
    assert infer_mod._check_and_adjust_attn_implementation("sdpa") == "sdpa"
    torch.manual_seed(0)
    hidden = infer_mod.config.text_config.hidden_size
    prompt, steps = torch.randn(5, hidden), [torch.randn(1, hidden) for _ in range(2)]
    infer_mod.reset_global_inference_state()
    conv = [ConversationItem(type="text", value=prompt, role="user")]
    with torch.no_grad():
        infer_mod.generate(conversation_list=conv)
        for emb in steps:
            conv[-1].value = emb
            infer_mod.generate(conversation_list=conv)
        states = infer_mod.flush_thinker_states()
        full = torch.cat([prompt, *steps], dim=0).unsqueeze(0)
        ref = infer_mod.language_model(inputs_embeds=full, use_cache=False, output_hidden_states=True)
    assert torch.equal(states["thinker_embed"], full)
    assert torch.allclose(states["thinker_hidden"], ref.hidden_states[3], atol=1e-5)

    train_mod = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    sample = _text_sample(train_mod, [4, 3])
    originals = [i.value.clone() for i in sample]
    _, inputs = _run_forward_node(train_mod, [sample])
    with torch.no_grad():
        all_h = train_mod.forward(**inputs)["all_hidden_states"]
    assert torch.equal(sample[0].meta[THINKER_EMBED], originals[0]) and torch.equal(
        sample[0].meta[THINKER_HIDDEN], all_h[2][0, :4]
    )
    assert not sample[0].meta[THINKER_EMBED].requires_grad and not sample[0].meta[THINKER_HIDDEN].requires_grad

    # Single-sample causal mask + router balancing loss.
    router_mod = _accelerated_llm(text_config={"output_router_logits": True, "router_aux_loss_coef": 0.01})
    r_inputs = router_mod.pre_forward(method="forward", conversation_list=[_text_sample(router_mod, [3, 4])])
    assert not _should_use_packed_mask(r_inputs["cu_seq_lens_q"], "eager", 1)
    assert find_packed_sequence_indices(torch.tensor([[0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0]])) is not None
    assert isinstance(router_mod.veomni_load_balancing_loss, VeomniOp)
    logits = (torch.randn(4, router_mod.config.text_config.num_experts),)
    with torch.no_grad():
        assert torch.isfinite(router_mod.router_balancing_loss(logits))
    assert router_mod._full_sequence_router_logits(logits) is logits


def test_llm_sp_sharded_handoff_regressions():
    module = _accelerated_llm()
    hidden = module.config.text_config.hidden_size

    # 1. Empty dummy SP window stays finite (sum() * 0, not mean() * 0 -> NaN).
    embeds = torch.ones(1, 4, 8)
    empty_dummy = ConversationItem(
        type="image", value=torch.zeros(0, 8), role="user", is_dummy=True, meta={_OMNI_TOKEN_LEN: 0}
    )
    folded = _fold_fsdp_dummy_anchors(embeds, [[empty_dummy]])
    assert torch.isfinite(folded).all() and torch.allclose(folded, embeds)

    # 2. _qwen3omni_modality_groups includes user audio and skips assistant audio.
    text_item = ConversationItem(type="text", value=torch.zeros(2, hidden), role="user")
    user_audio = ConversationItem(type="audio", value=torch.zeros(3, hidden), role="user")
    assistant_audio = ConversationItem(type="audio", value=torch.zeros(5, hidden), role="assistant")
    assert _qwen3omni_modality_groups([[text_item, user_audio, assistant_audio]]) == [[text_item], [user_audio]]

    # 3. Mixed-modality batch regathers producers before packing.
    mixed = [
        *_text_sample(module, [4]),
        ConversationItem(
            type="image",
            value=torch.zeros(4, hidden),
            role="user",
            meta={"grid_thw": torch.tensor((1, 4, 4)), "deepstack": []},
        ),
    ]
    gathered = {"called": False}
    with (
        mock.patch(
            "veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.gather_sp_sharded_groups",
            side_effect=lambda *_, **__: gathered.__setitem__("called", True),
        ),
        mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state") as ps,
        mock.patch.object(module, "_normalize_visual_inputs", side_effect=lambda e, m, d, **_: (e, m, d)),
    ):
        ps.return_value.sp_size, ps.return_value.sp_group = 2, object()
        ps.return_value.skip_sp_gather_between_modules, ps.return_value.fsdp_enabled = True, False
        try:
            module.forward_pre(method="forward", conversation_list=[mixed])
        except Exception:
            pass
    assert gathered["called"]

    # 4. forward_post emits rank-local windows under skip_sp_gather_between_modules, including 1-token trace shards.
    full = 7
    window = sp_local_len(full, 2)
    sample = [
        ConversationItem(
            type="text",
            value=torch.randn(sp_valid_in_window(full, i, 2), hidden),
            role="user",
            meta={_OMNI_TOKEN_LEN: n},
        )
        for i, n in enumerate((4, 3))
    ]
    module._pack_inputs_embeds_shape = torch.tensor([[4], [3]])
    for rank in range(2):
        with mock.patch(
            "veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state"
        ) as ps:
            ps.return_value.sp_size, ps.return_value.sp_rank, ps.return_value.sp_group = 2, rank, object()
            ps.return_value.skip_sp_gather_between_modules = True
            module._conversation_carrier = [sample]
            res = module.forward_post(
                hidden_states=torch.randn(1, window, hidden), router_logits=None, router_attention_mask=None
            )
        assert sum(it.value.size(0) for it in res["conversation_list"][0]) == sp_valid_in_window(full, rank, 2)

    traced_mod = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=1)
    one_tok = [ConversationItem(type="text", value=torch.randn(1, hidden), role="user", meta={_OMNI_TOKEN_LEN: 2})]
    traced_mod._conversation_carrier, traced_mod._pack_inputs_embeds_shape = [one_tok], torch.tensor([[2]])
    with mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state") as ps:
        ps.return_value.sp_size, ps.return_value.sp_rank, ps.return_value.sp_group = 2, 0, object()
        ps.return_value.skip_sp_gather_between_modules = True
        traced_mod.forward_post(
            hidden_states=torch.randn(1, 1, hidden),
            all_hidden_states=(torch.randn(1, 1, hidden), torch.randn(1, 1, hidden)),
            router_logits=None,
            router_attention_mask=None,
        )
    assert one_tok[0].meta[THINKER_EMBED].shape == one_tok[0].meta[THINKER_HIDDEN].shape == (1, hidden)

    # 5. forward_pre excludes assistant speech from full_len.
    seen = {}
    with (
        mock.patch.object(
            module,
            "_normalize_visual_inputs",
            side_effect=lambda e, m, d, *, full_len: (
                seen.__setitem__("full_len", full_len) or e,
                torch.zeros(1, full_len, dtype=torch.bool),
                None,
            ),
        ),
        mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated.get_parallel_state") as ps,
    ):
        ps.return_value.sp_size, ps.return_value.fsdp_enabled, ps.return_value.skip_sp_gather_between_modules = (
            1,
            False,
            False,
        )
        module.forward_pre(
            method="forward",
            conversation_list=[
                [
                    ConversationItem(
                        type="text", value=torch.randn(4, hidden), role="user", meta={_OMNI_TOKEN_LEN: 4}
                    ),
                    ConversationItem(
                        type="text", value=torch.randn(3, hidden), role="assistant", meta={_OMNI_TOKEN_LEN: 3}
                    ),
                    ConversationItem(
                        type="audio", value=torch.zeros(24000, 8), role="assistant", meta={_OMNI_TOKEN_LEN: 24000}
                    ),
                ]
            ],
        )
    assert seen["full_len"] == 7


# ── talker ───────────────────────────────────────────────────────────────────


def _thinker_wte():
    global _WTE
    if _WTE is None:
        torch.manual_seed(1)
        _WTE = torch.nn.Embedding(1200, THINKER_HIDDEN_SIZE)
    return _WTE


def _shrink(raw, layers=2):
    small = dict(raw)
    small.update(
        num_hidden_layers=layers,
        hidden_size=32,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        dtype="float32",
    )
    if small.get("layer_types"):
        small["layer_types"] = small["layer_types"][:layers]
    return small


def _tiny_talker() -> Qwen3OmniTalker:
    with open(os.path.join(CONFIG_DIR, "config.json")) as handle:
        raw = json.load(handle)["talker_config"]
    tiny = {
        **raw,
        "text_config": {
            **_shrink(raw["text_config"]),
            "moe_intermediate_size": 16,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        },
        "code_predictor_config": _shrink(raw["code_predictor_config"]),
        "thinker_hidden_size": THINKER_HIDDEN_SIZE,
    }
    torch.manual_seed(0)
    module = Qwen3OmniTalker(
        Qwen3OmniTalkerConfig(talker_config=tiny, ops_implementation=make_eager_ops_config())
    ).eval()
    for name, param in module.named_parameters():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            with torch.no_grad():
                param.normal_(mean=0.0, std=0.02)
    return module


def _talker_case(history: bool):
    rows = [
        ("text", "system", [IM_START, SYSTEM, 20, 21, IM_END]),
        ("text", "user", [IM_START, USER, 30, AUDIO_START]),
        ("audio", "user", [AUDIO_PAD] * 7),
        ("text", "user", [AUDIO_END, 31, 32, IM_END]),
    ]
    surviving = 15
    if history:
        rows += [
            ("text", "assistant", [IM_START, ASSISTANT, 40, 41, IM_END]),
            ("text", "user", [IM_START, USER, 50, 51, IM_END]),
        ]
        surviving += 5
    rows.append(("text", "assistant", [IM_START, ASSISTANT, 60]))
    prompt_ids = torch.tensor([tok for _, _, toks in rows for tok in toks], dtype=torch.long)
    segments = [PromptSegment(role, kind, len(toks)) for kind, role, toks in rows]
    torch.manual_seed(2)
    return SimpleNamespace(
        prompt_ids=prompt_ids,
        segments=segments,
        thinker_embed=torch.randn(1, prompt_ids.numel() + 5, THINKER_HIDDEN_SIZE),
        thinker_hidden=torch.randn(1, prompt_ids.numel() + 5, THINKER_HIDDEN_SIZE),
        surviving=surviving,
    )


def _talker_payload(prompt_ids, segments):
    return {
        "input_ids": prompt_ids,
        "segments": segments,
        "supervised": torch.zeros(sum(s.length for s in segments), dtype=torch.bool),
        "tts_embeds": _thinker_wte()(torch.tensor([[TTS_BOS, TTS_EOS, TTS_PAD]], dtype=torch.long)),
        "tts_pad_token_id": TTS_PAD,
        "im_start_token_id": IM_START,
        "im_end_token_id": IM_END,
        "role_token_ids": ROLE_TOKEN_IDS,
    }


def _make_talker_prompt(module: Qwen3OmniTalker, length: int = 6, trailing: int = 3) -> dict:
    torch.manual_seed(1)
    hidden, vocab = (
        module.config.talker_config.text_config.hidden_size,
        module.config.talker_config.text_config.vocab_size,
    )
    return {
        "inputs_embeds": torch.randn(1, length, hidden),
        "talker_input_ids": torch.randint(0, vocab - 1024, (1, length)),
        "trailing_text_hidden": torch.randn(1, trailing, hidden),
        "tts_pad_embed": torch.randn(1, 1, hidden),
    }


def _decode_reference(module: Qwen3OmniTalker, prompt: dict, max_new_tokens: int) -> tuple:
    talker, config = module.talker, module.config.talker_config
    predictor, residual_logits = talker.code_predictor, []
    original = predictor.generate

    def taped(*args, **kwargs):
        kwargs["output_logits"] = True
        res = original(*args, **kwargs)
        residual_logits.append(torch.stack(res.logits, dim=0).float())
        return res

    predictor.generate = taped
    try:
        torch.manual_seed(7)
        with torch.no_grad():
            output = talker.generate(
                inputs_embeds=prompt["inputs_embeds"],
                talker_input_ids=prompt["talker_input_ids"],
                trailing_text_hidden=prompt["trailing_text_hidden"],
                tts_pad_embed=prompt["tts_pad_embed"],
                attention_mask=torch.ones(prompt["inputs_embeds"].shape[:2], dtype=torch.long),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=config.codec_eos_token_id,
                output_hidden_states=True,
                output_logits=True,
                return_dict_in_generate=True,
            )
    finally:
        predictor.generate = original
    codes = torch.stack([s[-1] for s in output.hidden_states if s[-1] is not None], dim=1).transpose(1, 2).squeeze(0)
    return codes, torch.stack(output.logits, dim=0).float(), residual_logits


@pytest.mark.parametrize("history", [False, True], ids=["single_turn", "multi_turn"])
def test_talker_prompt_assembly_matches_upstream(history):
    module = _tiny_talker()
    case = _talker_case(history)
    speaker_id = module.resolve_speaker_id("ethan")

    shim = SimpleNamespace(
        talker=module.talker,
        config=SimpleNamespace(talker_config=module.config.talker_config, tts_pad_token_id=TTS_PAD),
    )
    shim._get_talker_user_parts = _Ref._get_talker_user_parts.__get__(shim)
    shim._get_talker_assistant_parts = _Ref._get_talker_assistant_parts.__get__(shim)

    with torch.no_grad():
        tts_bos, tts_eos, tts_pad = module.talker.text_projection(
            _thinker_wte()(torch.tensor([[TTS_BOS, TTS_EOS, TTS_PAD]], dtype=torch.long))
        ).chunk(3, dim=1)
        mm_mask = (case.prompt_ids == AUDIO_PAD).unsqueeze(0)
        boundaries = torch.nonzero(case.prompt_ids == IM_START).reshape(-1).tolist() + [case.thinker_embed.size(1)]
        embeds, ids, trailing = [], [], None
        for idx in range(len(boundaries) - 1):
            start, end = boundaries[idx], boundaries[idx + 1]
            role = int(case.prompt_ids[start + 1])
            if role == SYSTEM or (role == ASSISTANT and idx != len(boundaries) - 2):
                continue
            if role == USER:
                embeds.append(
                    shim._get_talker_user_parts(start, end, mm_mask, case.thinker_hidden, case.thinker_embed)
                )
                ids.append(case.prompt_ids[start:end].unsqueeze(0))
            elif role == ASSISTANT:
                se, si, trailing = shim._get_talker_assistant_parts(
                    start, end, speaker_id, case.thinker_embed, tts_pad, tts_bos, tts_eos
                )
                embeds.append(se)
                ids.append(si)

        actual = module.build_prompt(
            thinker_embed=case.thinker_embed,
            thinker_hidden=case.thinker_hidden,
            talker_prompt=_talker_payload(case.prompt_ids, case.segments),
            speaker="ethan",
        )

    torch.testing.assert_close(actual["inputs_embeds"], torch.cat(embeds, dim=1), rtol=0, atol=0)
    torch.testing.assert_close(actual["trailing_text_hidden"], trailing, rtol=0, atol=0)
    assert torch.equal(actual["talker_input_ids"], torch.cat(ids, dim=1))
    assert actual["inputs_embeds"].size(1) == case.surviving + 9


def test_talker_generate_fsdp_converter_and_carrier_helpers(monkeypatch):
    module = _tiny_talker()
    case = _talker_case(history=False)

    assert module._check_and_adjust_attn_implementation("sdpa") == "sdpa"
    assert set(module._no_split_modules) == {
        "Qwen3OmniMoeTalkerDecoderLayer",
        "Qwen3OmniMoeTalkerCodePredictorDecoderLayer",
    }
    assert hasattr(Qwen3OmniTalker, "_create_checkpoint_tensor_converter")
    assert hasattr(Qwen3OmniTalker, "_convert_fqn_to_index_mapping")

    # generate forwards media metadata and stamps TALKER_CODES_KEY.
    seen = {}
    num_groups = module.config.talker_config.num_code_groups
    monkeypatch.setattr(
        module,
        "generate_codes",
        lambda prompt, kwargs, media_metadata=None: seen.__setitem__("meta", media_metadata)
        or torch.randint(0, 100, (num_groups, 12)),
    )
    conv = [
        ConversationItem(type="text", value=torch.zeros(1), role="user"),
        ConversationItem(type="audio", value=torch.zeros(7, 4), role="user", meta={"audio_mel_len": 28}),
        ConversationItem(type="image", value=torch.zeros(4, 4), role="user", meta={"grid_thw": [1, 4, 4]}),
        ConversationItem(
            type="video", value=torch.zeros(8, 4), role="user", meta={"grid_thw": [2, 4, 4], "second_per_grid": 0.5}
        ),
    ]
    out = module.generate(
        conversation_list=conv,
        thinker_embed=case.thinker_embed,
        thinker_hidden=case.thinker_hidden,
        talker_prompt=_talker_payload(case.prompt_ids, case.segments),
    )
    assert out[FSM_SIGNAL_KEY] == SIGNAL_CODES_DONE and out["conversation_list"][-1].meta[TALKER_CODES_KEY] is True
    assert seen["meta"]["audio_feature_lengths"].tolist() == [28] and seen["meta"][
        "video_second_per_grid"
    ].tolist() == [0.5]
    assert all(
        collect_media_metadata([ConversationItem(type="text", value=torch.zeros(3), role="user")])[k] is None
        for k in ("audio_feature_lengths", "image_grid_thw", "video_grid_thw", "video_second_per_grid")
    )

    # Carrier helpers: _thinker_traces, _target_codes, _stash_talker_prompts, _speech_prefix_length.
    def traced(rows, role="user"):
        return ConversationItem(
            type="text",
            value=torch.zeros(rows, dtype=torch.long),
            role=role,
            meta={THINKER_EMBED: torch.randn(rows, 8), THINKER_HIDDEN: torch.randn(rows, 8)},
        )

    def coded(frames):
        return ConversationItem(
            type="audio",
            value=torch.zeros(1),
            role="assistant",
            meta={TALKER_CODE_LABELS: torch.zeros(4, frames, dtype=torch.long)},
        )

    s0, s1 = traced(3), traced(2, "assistant")
    dummy = ConversationItem(type="image", value=torch.zeros(4, 8), role="user", is_dummy=True)
    embed, hidden = _thinker_traces([s0, dummy, s1])
    assert embed.shape == hidden.shape == (1, 5, 8)
    c0 = coded(9)
    assert _target_codes([s0, c0]) is c0.meta[TALKER_CODE_LABELS] and _target_codes([s0]) is None
    with pytest.raises(ValueError, match="one utterance per prompt"):
        _target_codes([coded(3), coded(5)])
    samples = [[traced(3)], [traced(2), traced(4, "assistant")]]
    _stash_talker_prompts(samples, [{"which": 0}, {"which": 1}])
    assert [_talker_prompt(s) for s in samples] == [{"which": 0}, {"which": 1}]

    sup = torch.zeros(30, dtype=torch.bool)
    sup[4:9] = sup[25:] = True
    assert _speech_prefix_length(sup, start=22, sequence_length=30) == 3

    # FSDP2 mixed precision: projections in bf16 while sharded decoder reports fp32.
    module.talker.text_projection.to(torch.bfloat16)
    module.talker.hidden_projection.to(torch.bfloat16)
    module.talker.get_input_embeddings().to(torch.bfloat16)
    assert module.talker.dtype == torch.float32 and module.compute_dtype == torch.bfloat16
    assert (
        module.build_prompt(
            thinker_embed=case.thinker_embed,
            thinker_hidden=case.thinker_hidden,
            talker_prompt=_talker_payload(case.prompt_ids, case.segments),
            speaker="ethan",
        )["inputs_embeds"].dtype
        == torch.bfloat16
    )


@requires_tokenizer
def test_talker_training_prompt_leaves_response_to_be_spoken():
    template = Qwen3OmniChatTemplate(AutoTokenizer.from_pretrained(TOKENIZER))
    conv = template.tokenize_conversation(
        [
            ConversationItem(type="text", value="Please say hello out loud.", role="user"),
            ConversationItem(type="text", value="Hello there, nice to meet you!", role="assistant"),
            ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant"),
        ]
    )
    segments, supervised, input_ids = (
        talker_prompt_segments(conv),
        talker_supervised_mask(conv),
        template.pack_talker_input_ids(conv),
    )
    seq_len = sum(s.length for s in segments)
    assert input_ids.numel() == seq_len == supervised.numel() and all(s.type != "audio" for s in segments)
    _, start, _ = _chatml_turns(
        input_ids,
        total_length=seq_len,
        im_start_token_id=template.im_start_token_id,
        role_token_ids=template.role_token_ids,
    )[-1]
    prefix_len = _speech_prefix_length(supervised, start, seq_len)
    speech_end = _speech_span_end(
        input_ids, start, seq_len, trace_length=seq_len, im_end_token_id=template.im_end_token_id
    )
    assert template.tokenizer.decode(input_ids[start + prefix_len : speech_end]) == "Hello there, nice to meet you!"


def test_talker_teacher_forcing_and_batched_loss():
    module = _tiny_talker()
    prompt = _make_talker_prompt(module)
    codes, ref_semantic, ref_residual = _decode_reference(module, prompt, max_new_tokens=6)
    prompt_len, frames, groups = (
        prompt["inputs_embeds"].shape[1],
        codes.shape[1],
        module.config.talker_config.num_code_groups,
    )

    sequence = torch.cat(
        (prompt["inputs_embeds"], module.embed_code_frames(codes) + module._text_rows(prompt, frames)), dim=1
    )
    with torch.no_grad():
        hidden = module.talker.model(
            input_ids=None,
            attention_mask=None,
            position_ids=module._teacher_forced_positions(prompt, frames),
            past_key_values=None,
            inputs_embeds=sequence,
            use_cache=False,
        ).last_hidden_state
    semantic_logits = module.talker.codec_head(
        hidden[:, prompt_len - 1 : prompt_len - 1 + ref_semantic.shape[0]]
    ).float()
    torch.testing.assert_close(semantic_logits[0], ref_semantic[:, 0], rtol=1e-4, atol=1e-4)

    predictor = module.talker.code_predictor
    tables = predictor.get_input_embeddings()
    rows = [
        hidden[0, prompt_len - 1 : prompt_len - 1 + frames].unsqueeze(1),
        module.talker.get_input_embeddings()(codes[0]).unsqueeze(1),
        *(tables[g - 1](codes[g]).unsqueeze(1) for g in range(1, groups - 1)),
    ]
    with torch.no_grad():
        pred_h = predictor.model(inputs_embeds=torch.cat(rows, dim=1), use_cache=False).last_hidden_state
    residual_logits = torch.stack([predictor.lm_head[g - 1](pred_h[:, g]) for g in range(1, groups)], dim=1).float()
    torch.testing.assert_close(
        residual_logits, torch.cat(ref_residual, dim=1).permute(1, 0, 2)[:frames], rtol=1e-4, atol=1e-4
    )

    # Single-utterance & batched loss reduction and gradient flow.
    a_prompt, b_prompt = _make_talker_prompt(module), _make_talker_prompt(module, length=8, trailing=2)
    a, _, _ = _decode_reference(module, a_prompt, max_new_tokens=3)
    b, _, _ = _decode_reference(module, b_prompt, max_new_tokens=5)
    parts_a, parts_b = module._utterance_losses(a_prompt, a), module._utterance_losses(b_prompt, b)
    expected = float(parts_a.semantic_sum.detach() + parts_b.semantic_sum.detach()) / (
        parts_a.semantic_count + parts_b.semantic_count
    ) + float(parts_a.residual_sum.detach() + parts_b.residual_sum.detach()) / (
        parts_a.residual_count + parts_b.residual_count
    )
    handed = iter((a_prompt, b_prompt))
    with mock.patch.object(module, "build_prompt", side_effect=lambda **_: next(handed)):
        batched = module.forward(
            thinker_embeds=[None, None], thinker_hiddens=[None, None], talker_prompts=[{}, {}], target_codes=[a, b]
        )["loss"]
    torch.testing.assert_close(batched, torch.tensor(expected, dtype=batched.dtype), rtol=1e-5, atol=1e-5)
    batched.backward()
    assert module.talker.codec_head.weight.grad.abs().sum() > 0
    assert module.talker.code_predictor.lm_head[0].weight.grad.abs().sum() > 0


# ── codec ────────────────────────────────────────────────────────────────────


def _tiny_codec_config(**overrides) -> Qwen3OmniCodecConfig:
    kwargs = dict(
        ops_implementation=make_eager_ops_config(),
        code2wav_config=TINY_CODE2WAV_CONFIG,
        mimi_config=TINY_MIMI_CONFIG,
        chunk_size=8,
        left_context_size=2,
    )
    kwargs.update(overrides)
    return Qwen3OmniCodecConfig(**kwargs)


def _assistant_audio(seconds: float, rate: int = CODEC_SR) -> ConversationItem:
    torch.manual_seed(2)
    return ConversationItem(
        type="audio",
        value=torch.randn(int(rate * seconds)) * 0.1,
        role="assistant",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=rate, num_samples=int(rate * seconds))},
    )


def test_codec_config_encode_decode_generate_and_preprocessor():
    cfg = _tiny_codec_config(chunk_size=300, left_context_size=25)
    assert (cfg.model_type, cfg.num_quantizers, cfg.sampling_rate, cfg.tie_word_embeddings) == (
        "qwen3omni_codec",
        CODEC_QUANTIZERS,
        CODEC_SR,
        False,
    )
    with pytest.raises(ValueError, match="disagree on codebook size"):
        _tiny_codec_config(mimi_config={**TINY_MIMI_CONFIG, "codebook_size": CODEC_CODEBOOK * 2})

    torch.manual_seed(0)
    module = Qwen3OmniCodec(_tiny_codec_config()).eval()
    assert module._check_and_adjust_attn_implementation("sdpa") == "sdpa"

    # Encode RVQ truncation parity + upstream Code2Wav decode parity + roundtrip + generate.
    upstream_cfg = Qwen3OmniMoeCode2WavConfig(**TINY_CODE2WAV_CONFIG)
    upstream_cfg.ops_implementation = make_eager_ops_config().to_dict()
    upstream = Qwen3OmniMoeCode2Wav._from_config(upstream_cfg).eval()
    upstream.load_state_dict(
        {k.removeprefix("code2wav."): v for k, v in module.state_dict().items() if k.startswith("code2wav.")},
        strict=True,
    )

    torch.manual_seed(1)
    codes = torch.randint(0, CODEC_CODEBOOK, (CODEC_QUANTIZERS, 12))
    wav = torch.randn(1, CODEC_SR // 4)
    with torch.no_grad():
        shallow = module.mimi.encode(wav.unsqueeze(0), num_quantizers=CODEC_QUANTIZERS).audio_codes
        deep = module.mimi.encode(wav.unsqueeze(0), num_quantizers=TINY_MIMI_CONFIG["num_quantizers"]).audio_codes
        mine = module.decode_codes(codes)
        ref = upstream.chunked_decode(codes.unsqueeze(0), chunk_size=8, left_context_size=2).float()
        rt = module.decode_codes(module.encode_wav(wav, sampling_rate=CODEC_SR))
        gen = module.generate(
            conversation_list=[
                ConversationItem(type="text", value="hi", role="user"),
                ConversationItem(type="output", value=codes, role="assistant", meta={TALKER_CODES_KEY: True}),
            ]
        )
    assert torch.equal(deep[:, :CODEC_QUANTIZERS], shallow)
    torch.testing.assert_close(mine, ref, rtol=0, atol=1e-6)
    assert rt.dim() == 3 and rt.size(-1) > 0
    assert gen[FSM_SIGNAL_KEY] == SIGNAL_AUDIO_DONE
    assert OmniModel._normalize_generated(gen["generated"])["type"] == "audio"

    # Preprocessor resamples assistant speech only; accelerated encode stashes isolated targets.
    prep = Qwen3OmniCodecPreprocessor(sampling_rate=CODEC_SR)
    retuned = _assistant_audio(1.0, 22050)
    user = ConversationItem(
        type="audio",
        value=np.zeros(22050, dtype=np.float32),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=22050, num_samples=22050)},
    )
    prep.preprocess_conversations([[user, retuned]])
    assert retuned.meta[AUDIO_METADATA_KEY].sampling_rate == CODEC_SR and _as_mono_batch(retuned.value).shape[:2] == (
        1,
        1,
    )
    assert user.meta[AUDIO_METADATA_KEY].sampling_rate == 22050

    assert AudioMetadata(sampling_rate=22050, num_samples=22050).resampled_to(
        24000, 24001
    ).duration_seconds == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError, match="would retime it"):
        AudioMetadata(sampling_rate=22050, num_samples=22050).resampled_to(24000, 48000)

    accel = Qwen3OmniCodecAccelerated(_tiny_codec_config()).eval()
    short, long = _assistant_audio(0.04), _assistant_audio(0.09)
    with torch.no_grad():
        inputs = accel.pre_forward(method="encode", conversation_list=[[user, short], [long]])
        out = accel.post_forward(method="encode", **accel.encode(**inputs))
        alone = accel.encode_wav(short.value, sampling_rate=CODEC_SR)
    assert out["conversation_list"][0][1] is short and TALKER_CODE_LABELS not in user.meta
    assert torch.equal(short.meta[TALKER_CODE_LABELS], alone)
    assert short.meta[TALKER_CODE_LABELS].shape[1] < long.meta[TALKER_CODE_LABELS].shape[1]


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
def test_codec_encode_with_fsdp2_quantizer_sharded(tmp_path):
    """MimiResidualVectorQuantizer is invoked by method call, so FSDP2 units must stay unsharded."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    from veomni.distributed.parallel_state import clear_parallel_state
    from veomni.distributed.utils import sort_fqn_by_submodule_first
    from veomni.models.seed_omni.accelerated.utils.dispatch import call_graph_endpoint
    from veomni.utils.device import (
        IS_CUDA_AVAILABLE,
        IS_NPU_AVAILABLE,
        get_device_type,
        get_dist_comm_backend,
        get_torch_device,
    )

    if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE):
        pytest.skip("needs an accelerator for the FSDP2 path")

    get_torch_device().set_device(0)
    dist.init_process_group(
        backend=get_dist_comm_backend(), init_method=f"file://{tmp_path / 'rendezvous'}", world_size=1, rank=0
    )
    try:
        device = f"{get_device_type()}:0"
        module = Qwen3OmniCodecAccelerated(_tiny_codec_config()).eval().to(device)
        mesh = init_device_mesh(get_device_type(), (1,))
        wrap_targets = set(module._no_split_modules)
        by_fqn = {fqn: mod for fqn, mod in module.named_modules() if type(mod).__name__ in wrap_targets}
        units = [by_fqn[fqn] for fqn in sort_fqn_by_submodule_first(list(by_fqn))]
        for unit in units:
            fully_shard(unit, mesh=mesh)
        fully_shard(module, mesh=mesh)
        assert any(
            "VectorQuantizer" in type(m).__name__ and any(isinstance(p, DTensor) for p in m.parameters())
            for m in units
        )

        item = _assistant_audio(0.05)
        item.value = item.value.to(device)
        with torch.no_grad():
            inputs = module.pre_forward(method="encode", conversation_list=[[item]])
            out = module.post_forward(
                method="encode", **call_graph_endpoint(module, module, method="encode", kwargs=inputs)
            )
        codes = out["conversation_list"][0][0].meta[TALKER_CODE_LABELS]
        assert codes.shape[0] == CODEC_QUANTIZERS and not isinstance(codes, DTensor)
    finally:
        dist.destroy_process_group()
        clear_parallel_state()
