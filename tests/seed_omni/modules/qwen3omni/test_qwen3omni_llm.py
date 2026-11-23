"""Unit tests for the Qwen3-Omni thinker backbone packing (TMRoPE position ids).

The interesting property here is that the segment-model packer must produce the
*same* position ids as upstream's token-level ``get_rope_index``, even though it
never materialises the ``<|image_pad|>`` / ``<|audio_pad|>`` runs. These tests
pin that equivalence against the real upstream implementation rather than
against hand-written expectations.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.tools.training_utils import make_eager_ops_config
from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata
from veomni.models.seed_omni.modules.qwen3omni.llm.accelerated.accelerated import Qwen3OmniLlmAccelerated
from veomni.models.seed_omni.modules.qwen3omni.llm.configuration import Qwen3OmniLlmConfig
from veomni.models.seed_omni.modules.qwen3omni.llm.modeling import (
    THINKER_EMBED,
    THINKER_HIDDEN,
    Qwen3OmniLlm,
    pack_qwen3omni_conversations_for_forward,
    qwen3omni_vision_position_ids,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as _Upstream,
)


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


HIDDEN = 8
MERGE = 2
POS_PER_SEC = 13

VISION_START, VISION_END = 151652, 151653
IMAGE_PAD, VIDEO_PAD = 151655, 151656
AUDIO_START, AUDIO_END, AUDIO_PAD = 151669, 151670, 151675


def _upstream_self():
    """Bind upstream ``get_rope_index`` to a config-carrying namespace.

    Mirrors how the v1 data pipeline calls it (see ``get_position_id`` in the
    patchgen-generated modeling) — no weights needed.
    """
    ns = SimpleNamespace(
        spatial_merge_size=MERGE,
        config=SimpleNamespace(
            image_token_id=IMAGE_PAD,
            video_token_id=VIDEO_PAD,
            audio_token_id=AUDIO_PAD,
            vision_start_token_id=VISION_START,
            audio_start_token_id=AUDIO_START,
            position_id_per_seconds=POS_PER_SEC,
        ),
    )
    ns.get_llm_pos_ids_for_vision = _Upstream.get_llm_pos_ids_for_vision.__get__(ns)
    ns.get_rope_index = _Upstream.get_rope_index.__get__(ns)
    return ns


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


def test_pack_concatenates_embeds_and_marks_visual_positions():
    grid = (1, 4, 6)
    image_len = 4 * 6 // (MERGE**2)
    sample = [_text(3), _media("image", image_len, grid), _text(2)]
    packed = _pack(sample)

    total = 3 + image_len + 2
    assert packed["inputs_embeds"].shape == (1, total, HIDDEN)
    assert packed["position_ids"].shape == (3, 1, total)
    assert packed["visual_pos_masks"].tolist() == [[False] * 3 + [True] * image_len + [False] * 2]
    # one entry per deepstack layer, each covering the image rows
    assert len(packed["deepstack_visual_embeds"]) == 2
    assert packed["deepstack_visual_embeds"][0].shape == (image_len, HIDDEN)
    assert packed["cu_seq_lens"].tolist() == [0, total]
    assert packed["max_length"] == total


def test_text_and_audio_positions_are_plain_sequential():
    """TMRoPE degenerates to 1-D for text + standalone audio.

    Upstream takes its non-vision branch here (``attention_mask.cumsum``), so
    all three rows must be identical ``arange`` — the same thing the packer
    emits for any non-visual segment.
    """
    audio_len = 7
    sample = [
        _text(4),
        ConversationItem(type="audio", value=torch.zeros(audio_len, HIDDEN), role="user"),
        _text(2),
    ]
    packed = _pack(sample)
    total = 4 + audio_len + 2

    expected = torch.arange(total, dtype=torch.float).view(1, 1, -1).expand(3, 1, -1)
    assert torch.equal(packed["position_ids"], expected)
    # a text/audio-only sample has no DeepStack rows to inject
    assert packed["visual_pos_masks"] is None
    assert packed["deepstack_visual_embeds"] is None


def test_audio_and_video_share_one_wall_clock_position_axis():
    """The temporal row is ``position_id_per_seconds * elapsed_seconds`` for both modalities.

    This is the whole point of TMRoPE and it spans two modules: the audio tower
    fixes the token rate (13 tokens/second, pinned in ``test_qwen3omni_audio``)
    while this packer fixes the position rate -- 1 position per audio token, but
    ``second_per_grid * 13`` per video grid step. If either side drifts, audio
    and video silently describe different timelines and nothing raises. So
    decode the position ids back into seconds and check both modalities land
    where their content actually sits in time.
    """
    audio_seconds = 2.0
    audio_len = int(audio_seconds * POS_PER_SEC)
    # temporal_patch_size=2 sampled at 25fps -> a grid step spans 0.08s.
    second_per_grid = 2 / 25
    grid_t = 5
    grid = (grid_t, 4, 4)
    video_len = grid_t * 4 * 4 // (MERGE**2)

    sample = [
        ConversationItem(type="audio", value=torch.zeros(audio_len, HIDDEN), role="user"),
        _media("video", video_len, grid, second_per_grid=second_per_grid),
    ]
    temporal = _pack(sample)["position_ids"][0, 0]
    seconds = temporal / POS_PER_SEC

    audio_seconds_axis = seconds[:audio_len]
    expected_audio = torch.arange(audio_len, dtype=torch.float) / POS_PER_SEC
    assert torch.allclose(audio_seconds_axis, expected_audio)

    # The video starts one position unit after the audio's last, so its own
    # timeline is offset by the audio's duration -- and from there each grid
    # step must advance the clock by exactly second_per_grid seconds.
    video_seconds_axis = torch.unique(seconds[audio_len:], sorted=True)
    frame_offset = audio_seconds_axis[-1] + 1.0 / POS_PER_SEC
    expected_video = frame_offset + torch.arange(grid_t, dtype=torch.float) * second_per_grid
    assert torch.allclose(video_seconds_axis, expected_video)


@pytest.mark.parametrize(
    ("grid", "second_per_grid"),
    [
        ((1, 4, 6), 1.0),  # still image
        ((1, 2, 2), 1.0),  # single merged token
        ((3, 4, 4), 1.0),  # multi-frame, image convention (t * 1 * pos_per_sec)
        ((3, 4, 4), 0.5),  # video at 2 fps — block max stays integral
        ((2, 4, 4), 0.5),  # video at 2 fps, fractional block max (6.5)
        ((3, 4, 4), 2 / 25),  # video at 12.5 fps, temporal_patch_size=2
    ],
)
def test_vision_positions_match_upstream_get_rope_index(grid, second_per_grid):
    """Segment-level packing must equal upstream's token-level TMRoPE."""
    grid_t, grid_h, grid_w = grid
    media_len = grid_t * grid_h * grid_w // (MERGE**2)
    is_video = second_per_grid != 1.0
    pad_token = VIDEO_PAD if is_video else IMAGE_PAD
    head, tail = 3, 2

    input_ids = torch.tensor(
        [[1] * head + [VISION_START] + [pad_token] * media_len + [VISION_END] + [2] * tail],
        dtype=torch.long,
    )
    grid_thw = torch.tensor([grid], dtype=torch.long)
    ups = _upstream_self()
    upstream_pos, _ = ups.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=None if is_video else grid_thw,
        video_grid_thw=grid_thw if is_video else None,
        # VeOmni's per-video convention: a zero-length audio entry means
        # "this video has no audio track" (see Patch.1 on get_rope_index).
        audio_seqlens=torch.tensor([0], dtype=torch.long) if is_video else None,
        second_per_grids=torch.tensor([second_per_grid]) if is_video else None,
    )

    # Carrier view: <|vision_start|> / <|vision_end|> are ordinary text rows, and
    # the media item stands in for the whole <|*_pad|> run.
    sample = [
        _text(head),
        _text(1),
        _media(
            "video" if is_video else "image",
            media_len,
            grid,
            second_per_grid=second_per_grid,
        ),
        _text(1 + tail),
    ]
    packed = _pack(sample)

    # No dtype coercion on either side: upstream keeps TMRoPE positions in float
    # because ``second_per_grid * position_id_per_seconds`` is usually fractional.
    # Rounding here would shift every position after the clip.
    assert packed["position_ids"].dtype == torch.float
    assert torch.equal(packed["position_ids"], upstream_pos)


def test_vision_position_ids_advance_cursor_by_max_plus_one():
    """The next segment starts at ``previous.max() + 1`` (upstream's rule).

    Getting this wrong is silent: the model still runs, but every token after an
    image sits at the wrong RoPE offset.
    """
    grid = (1, 4, 6)
    image_len = 4 * 6 // (MERGE**2)
    start = 5
    vision_pos = qwen3omni_vision_position_ids(start, torch.tensor(grid), MERGE, position_id_per_seconds=POS_PER_SEC)
    assert vision_pos.shape == (3, image_len)

    sample = [_text(start), _media("image", image_len, grid), _text(3)]
    packed = _pack(sample)
    pos = packed["position_ids"][:, 0, :]
    assert torch.equal(pos[:, start : start + image_len], vision_pos)
    resume = float(vision_pos.max()) + 1.0
    assert pos[:, start + image_len :].tolist() == [[resume, resume + 1, resume + 2]] * 3


def test_multiple_samples_restart_positions_and_extend_cu_seqlens():
    """Varlen packing: each sample keeps its own position origin."""
    packed = pack_qwen3omni_conversations_for_forward(
        [[_text(3)], [_text(5)]], torch.device("cpu"), MERGE, POS_PER_SEC
    )
    assert packed["cu_seq_lens"].tolist() == [0, 3, 8]
    assert packed["max_length"] == 5
    assert packed["position_ids"][0, 0, :].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]


def test_video_without_second_per_grid_is_rejected():
    """Defaulting a video's timeline to 1.0 would be silently wrong, so refuse it.

    ``second_per_grid`` is ``temporal_patch_size / fps`` and only the vision
    preprocessor knows it. A still image is genuinely one instant, so it keeps
    the 1.0 default.
    """
    video = _media("video", 4, (1, 4, 4))
    with pytest.raises(ValueError, match="second_per_grid"):
        _pack([_text(2), video])

    # same grid as an image: allowed, no metadata needed
    assert _pack([_text(2), _media("image", 4, (1, 4, 4))])["position_ids"].shape == (3, 1, 6)


def test_a_clip_with_sound_packs_like_its_silent_twin():
    """Without the time interleave the packer places a clip by its frames alone.

    Built with the data layer's own key and type, so a sound track is stated the
    way a decoded clip states it: the packer reads the frame timeline and skips
    the track, leaving the positions exactly those of the silent clip.
    """
    silent = _media("video", 4, (1, 4, 4), second_per_grid=0.5)
    sounding = _media(
        "video",
        4,
        (1, 4, 4),
        second_per_grid=0.5,
        **{AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=16_000, num_samples=8)},
    )

    assert torch.equal(_pack([_text(2), sounding])["position_ids"], _pack([_text(2), silent])["position_ids"])


def test_a_silent_video_reaches_the_packer(tmp_path):
    """The other half of the guard: it must not refuse a clip with no sound."""
    del tmp_path
    video = _media("video", 4, (1, 4, 4), second_per_grid=0.5)

    assert _pack([_text(2), video])["position_ids"].shape == (3, 1, 6)


def test_zero_length_segments_are_skipped():
    """An empty text row (e.g. an empty assistant turn) must not break packing.

    It carries no position — so it contributes no rows to the packed sequence —
    but it keeps its *slot* in the shape metadata: the hidden-state scatter pairs
    rows to items by position, so an item the packer dropped while the scatter
    still visited it would take the next item's slice and raise ``StopIteration``
    on the one after, leaving the carrier half-overwritten.
    """
    packed = _pack([_text(2), _text(0), _text(3)])
    assert packed["inputs_embeds"].shape == (1, 5, HIDDEN)
    assert packed["position_ids"][0, 0, :].tolist() == [0, 1, 2, 3, 4]
    # Three segments out, one of them empty — not two.
    assert packed["inputs_embeds_shape"].tolist() == [[2], [0], [3]]


def test_an_empty_segment_survives_the_round_trip_to_the_carrier():
    """Packing and the scatter must agree on how many segments there are."""
    from veomni.models.seed_omni.modules.base.llm_packing import scatter_llm_hidden_states
    from veomni.utils.tensor_utils import unflatten

    sample = [_text(2), _text(0), _text(3)]
    packed = _pack(sample)
    segments = unflatten(packed["inputs_embeds"][0], packed["inputs_embeds_shape"])

    scatter_llm_hidden_states([sample], segments)

    assert [tuple(item.value.shape) for item in sample] == [(2, HIDDEN), (0, HIDDEN), (3, HIDDEN)]


def test_thinker_state_collection_matches_a_one_shot_run():
    """The talker reads layer ``accept_hidden_layer``, not the final hidden state.

    Upstream builds its two thinker traces by concatenating ``hidden_states[0]``
    and ``hidden_states[accept_hidden_layer]`` over every generation step. The
    FSM only ever sees one step at a time, so the backbone accumulates them —
    and the result has to equal a single teacher-forced pass over the same
    sequence, otherwise the talker is conditioned on the wrong states.
    """
    accept = 3
    config = Qwen3OmniLlmConfig(
        ops_implementation=make_eager_ops_config(),
        text_config=TINY_TEXT_CONFIG,
        accept_hidden_layer=accept,
        collect_thinker_states=True,
    )
    module = Qwen3OmniLlm(config).eval()
    torch.manual_seed(0)

    prompt_len, decode_steps = 5, 3
    prompt = torch.randn(prompt_len, config.text_config.hidden_size)
    fed_back = [torch.randn(1, config.text_config.hidden_size) for _ in range(decode_steps)]

    module.reset_global_inference_state()
    conversation = [ConversationItem(type="text", value=prompt, role="user")]
    with torch.no_grad():
        module.generate(conversation_list=conversation)
        for embed in fed_back:
            conversation[-1].value = embed  # what the text encoder writes back
            module.generate(conversation_list=conversation)
    states = module.flush_thinker_states()

    full = torch.cat([prompt, *fed_back], dim=0).unsqueeze(0)
    with torch.no_grad():
        reference = module.language_model(inputs_embeds=full, use_cache=False, output_hidden_states=True)

    covered = prompt_len + decode_steps
    assert states["thinker_embed"].shape == (1, covered, config.text_config.hidden_size)
    assert torch.equal(states["thinker_embed"], full)
    assert torch.allclose(states["thinker_hidden"], reference.hidden_states[accept], atol=1e-5)
    # guard the whole point of the exercise: the accept layer is not the last one
    assert not torch.allclose(reference.hidden_states[accept], reference.last_hidden_state)
    # flush clears, so a second conversation cannot inherit the first one's trace
    assert module._accept_hidden_cache == []


def test_thinker_states_are_not_collected_by_default():
    """All-layer hidden states are a memory cost text-only inference should not pay."""
    config = Qwen3OmniLlmConfig(ops_implementation=make_eager_ops_config(), text_config=TINY_TEXT_CONFIG)
    assert config.collect_thinker_states is False

    module = Qwen3OmniLlm(config).eval()
    module.reset_global_inference_state()
    conversation = [ConversationItem(type="text", value=torch.randn(4, config.text_config.hidden_size), role="user")]
    with torch.no_grad():
        module.generate(conversation_list=conversation)
    assert module._accept_hidden_cache == []


def test_generate_refuses_sequence_parallelism():
    """`forward_pre` slices the sample per rank; `generate` has no such hook.

    So under Ulysses the exchange would rebuild the full sequence from a prompt
    every rank already holds (4 rows in, 8 out) while `cu_seq_lens` still
    described the original — a wrong answer rather than a crash. The shared
    `SimpleArGenerationMixin` refuses the same configuration.
    """
    from unittest import mock

    module = _accelerated_llm()
    module.reset_global_inference_state()
    conversation = [ConversationItem(type="text", value=torch.randn(4, HIDDEN), role="user")]

    fake_ps = mock.MagicMock()
    fake_ps.sp_size = 2
    with mock.patch("veomni.models.seed_omni.modules.qwen3omni.llm.modeling.get_parallel_state", return_value=fake_ps):
        with pytest.raises(ValueError, match="sp_size=1"):
            module.generate(conversation_list=conversation)


def test_dummy_items_are_excluded_but_donate_deepstack():
    """FSDP dummy rows must not enter the sequence, yet still anchor DeepStack.

    Without the anchor the vision merger that produced the dummy features gets
    no gradient on this rank and FSDP's reduce-scatter desyncs.
    """
    dummy = _media("image", 4, (1, 4, 4))
    dummy.is_dummy = True
    packed = _pack([_text(3), dummy])

    assert packed["inputs_embeds"].shape == (1, 3, HIDDEN)
    assert packed["visual_pos_masks"] is None
    assert len(packed["deepstack_visual_embeds"]) == 2


# ── training path: handing the talker its two traces via the carrier ──────────


def _accelerated_llm(**config_overrides) -> "Qwen3OmniLlmAccelerated":
    torch.manual_seed(0)
    config = Qwen3OmniLlmConfig(
        ops_implementation=make_eager_ops_config(), text_config=TINY_TEXT_CONFIG, **config_overrides
    )
    return Qwen3OmniLlmAccelerated(config).eval()


def _run_forward_node(module, conversation_list):
    """Drive the node the way OmniModel._run_train_node does."""
    inputs = module.pre_forward(method="forward", conversation_list=conversation_list)
    outputs = module.forward(**inputs)
    return module.post_forward(method="forward", **outputs), inputs


def _text_sample(module, lengths):
    hidden = module.config.text_config.hidden_size
    torch.manual_seed(3)
    return [ConversationItem(type="text", value=torch.randn(n, hidden), role="user") for n in lengths]


def test_training_leaves_the_thinker_traces_on_the_carrier_items():
    """The talker reads its prompt off the carrier, so the thinker must write it there.

    Inference cannot do this -- it accumulates per-step traces and hands them
    over through the FSM's ctx -- but training is one pass over the packed
    sequence, so the same per-item split that returns hidden states to
    ``item.value`` returns these to ``item.meta``.
    """
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    sample = _text_sample(module, [4, 3])
    originals = [item.value.clone() for item in sample]

    with torch.no_grad():
        out, _ = _run_forward_node(module, [sample])

    assert out["conversation_list"][0] is sample
    for item, original in zip(sample, originals, strict=True):
        # index 0 of the layer stack IS the packed input embedding, so the
        # published trace must be exactly what this item fed in.
        assert torch.equal(item.meta[THINKER_EMBED], original)
        assert item.meta[THINKER_HIDDEN].shape == original.shape
        # item.value stays the final hidden state -- that belongs to the text head.
        assert item.value.shape == original.shape
        assert not torch.equal(item.value, original)


def test_the_published_hidden_trace_is_the_layer_the_talker_accepts():
    """Not the last layer. The lm_head reads that one; the talker reads layer 24."""
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    sample = _text_sample(module, [5])

    with torch.no_grad():
        _, inputs = _run_forward_node(module, [sample])
        reference = module.forward(**inputs)["all_hidden_states"]

    published = sample[0].meta[THINKER_HIDDEN]
    accept = reference[2].squeeze(0)
    final = reference[-1].squeeze(0)
    assert torch.equal(published, accept)
    assert not torch.equal(accept, final), "tiny model made the check vacuous"


def test_thinker_traces_are_detached_from_the_thinker():
    """This round trains the talker against a frozen stack.

    A live graph here would keep every thinker activation alive to produce
    parameter gradients that are masked off anyway.
    """
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    sample = _text_sample(module, [4])
    _run_forward_node(module, [sample])

    assert not sample[0].meta[THINKER_EMBED].requires_grad
    assert not sample[0].meta[THINKER_HIDDEN].requires_grad


def test_thinker_only_training_publishes_nothing():
    """Retaining every layer is a memory cost a graph without a talker should not pay."""
    module = _accelerated_llm()
    sample = _text_sample(module, [4])

    with torch.no_grad():
        _, inputs = _run_forward_node(module, [sample])

    assert inputs["output_hidden_states"] is False
    assert THINKER_EMBED not in sample[0].meta
    assert THINKER_HIDDEN not in sample[0].meta


def test_traces_skip_dummy_items_like_the_hidden_state_scatter_does():
    """Dummies never enter the packed sequence, so they have no slice to receive."""
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=2)
    hidden = module.config.text_config.hidden_size
    real = _text_sample(module, [4])[0]
    dummy = ConversationItem(
        type="image",
        value=torch.zeros(4, hidden),
        role="user",
        is_dummy=True,
        meta={"grid_thw": torch.tensor((1, 4, 4)), "deepstack": [torch.zeros(4, hidden) for _ in range(2)]},
    )

    with torch.no_grad():
        _run_forward_node(module, [[real, dummy]])

    assert THINKER_EMBED in real.meta
    assert THINKER_EMBED not in dummy.meta


def test_an_accept_layer_past_the_backbone_depth_says_so():
    """Bare indexing raises IndexError from inside a post hook, naming nothing."""
    module = _accelerated_llm(collect_thinker_states=True, accept_hidden_layer=99)
    sample = _text_sample(module, [4])
    with pytest.raises(ValueError, match="accept_hidden_layer=99"):
        _run_forward_node(module, [sample])
