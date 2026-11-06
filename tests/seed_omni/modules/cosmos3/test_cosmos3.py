"""Cosmos3-Nano on SeedOmni: checkpoint names, the freeze switch, the graph against the
monolithic formula, packing, the loss reduction, the CPU preprocessors and the JSONL adapter."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.seed_omni.utils.media_metadata import VIDEO_METADATA_KEY, VideoMetadata
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.cosmos3.convert_model import route_weight
from veomni.models.seed_omni.modules.cosmos3.core import (
    add_time_conditioning,
    build_latent_grid,
    build_packed_layout,
    build_segmented_latent_position_ids,
    build_segmented_text_position_ids,
    flow_matching_loss,
    latent_temporal_positions,
    patchify_latent,
    unpatchify_latent,
)
from veomni.models.seed_omni.modules.cosmos3.flow_connector.configuration import Cosmos3FlowConnectorConfig
from veomni.models.seed_omni.modules.cosmos3.item_meta import FPS_KEY, LATENT_KEY
from veomni.models.seed_omni.modules.cosmos3.mot.configuration import Cosmos3MotConfig
from veomni.models.seed_omni.modules.cosmos3.mot.modeling import TRAINABLE_NAME_KEYS
from veomni.models.seed_omni.modules.cosmos3.text_encoder.chat_template import Cosmos3ChatTemplate
from veomni.models.seed_omni.modules.cosmos3.text_encoder.configuration import Cosmos3TextEncoderConfig
from veomni.models.seed_omni.modules.cosmos3.vae.configuration import Cosmos3VaeConfig
from veomni.models.seed_omni.modules.cosmos3.vae.processing import Cosmos3VaePreprocessor, prefix_frame_indices
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

MODULE_TYPES = ("cosmos3_text_encoder", "cosmos3_vae", "cosmos3_mot", "cosmos3_flow_connector")

# The released checkpoint's per-layer and network-level keys, minus the dropped vision / audio / action ones.
LAYER_KEY_SUFFIXES = frozenset(
    {
        "self_attn.to_q.weight",
        "self_attn.to_k.weight",
        "self_attn.to_v.weight",
        "self_attn.to_out.weight",
        "self_attn.norm_q.weight",
        "self_attn.norm_k.weight",
        "self_attn.add_q_proj.weight",
        "self_attn.add_k_proj.weight",
        "self_attn.add_v_proj.weight",
        "self_attn.to_add_out.weight",
        "self_attn.norm_added_q.weight",
        "self_attn.norm_added_k.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp_moe_gen.gate_proj.weight",
        "mlp_moe_gen.up_proj.weight",
        "mlp_moe_gen.down_proj.weight",
        "input_layernorm.weight",
        "input_layernorm_moe_gen.weight",
        "post_attention_layernorm.weight",
        "post_attention_layernorm_moe_gen.weight",
    }
)
TOP_LEVEL_KEYS = frozenset(
    {
        "embed_tokens.weight",
        "lm_head.weight",
        "norm.weight",
        "norm_moe_gen.weight",
        "proj_in.weight",
        "proj_in.bias",
        "proj_out.weight",
        "proj_out.bias",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
    }
)

EAGER_OPS = {
    "attn_implementation": "eager",
    "rms_norm_implementation": "eager",
    "rotary_pos_emb_implementation": "eager",
}
TINY_MOT = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 96,
    "mrope_section": (4, 2, 2),
    "unified_3d_mrope_temporal_modality_margin": 15,
}
LATENT_CHANNELS = 2  # patch_latent_dim = 2 * 2 * 2


def _modules(attn_implementation: str = "eager", accelerated: bool = True, **mot_overrides):
    torch.manual_seed(0)
    registry = OMNI_ACCELERATED_MODEL_REGISTRY if accelerated else OMNI_MODEL_REGISTRY
    ops = {**EAGER_OPS, "attn_implementation": attn_implementation}
    text_encoder = registry["cosmos3_text_encoder"]()(
        Cosmos3TextEncoderConfig(vocab_size=128, hidden_size=64, generation_marker_ids=[1, 2])
    )
    mot = registry["cosmos3_mot"]()(Cosmos3MotConfig(**{**TINY_MOT, **mot_overrides}, ops_implementation=ops))
    connector = registry["cosmos3_flow_connector"]()(
        Cosmos3FlowConnectorConfig(hidden_size=64, patch_latent_dim=4 * LATENT_CHANNELS, loss_scale=1.0)
    )
    for module in (text_encoder, mot, connector):
        module.eval()
    if accelerated:
        text_encoder.tokenizer = SimpleNamespace()
    return {"cosmos3_text_encoder": text_encoder, "cosmos3_mot": mot, "cosmos3_flow_connector": connector}


def _samples() -> list[dict]:
    """Two samples of different prompt length, latent geometry and fps."""
    generator = torch.Generator().manual_seed(1234)
    return [
        {
            "ids": torch.randint(0, 128, (length,), generator=generator),
            "latent": torch.randn(shape, generator=generator),
            "fps": fps,
        }
        for length, shape, fps in (
            (7, (LATENT_CHANNELS, 3, 4, 6), 5.0),
            (3, (LATENT_CHANNELS, 2, 2, 4), None),
        )
    ]


def _conversation(samples: list[dict]) -> list[list[ConversationItem]]:
    """What the CPU preprocessors and the VAE leave on the batch: tokenized prompts and latents."""
    return [
        [
            ConversationItem(type="text", value=sample["ids"].clone(), role="user"),
            ConversationItem(
                type="video",
                value=sample["latent"].clone(),
                role="assistant",
                meta={LATENT_KEY: True, FPS_KEY: sample["fps"]},
            ),
        ]
        for sample in samples
    ]


def _graph_loss(modules, samples) -> torch.Tensor:
    """Loss through the SeedOmni graph, the VAE step replaced by ready latents."""
    edges = [
        {"from": "cosmos3_text_encoder.encode", "to": "cosmos3_mot"},
        {"from": "cosmos3_flow_connector.embed_latent", "to": "cosmos3_mot"},
        {"from": "cosmos3_mot", "to": "cosmos3_flow_connector.decode_velocity"},
        {"from": "cosmos3_flow_connector.decode_velocity", "to": "end"},
    ]
    config = OmniConfig(
        _module_entries={name: {"model_path": name} for name in modules},
        training_graphs={"default": edges},
        generation_graphs={},
    )
    return OmniModel(config, modules).forward({"conversation_list": _conversation(samples)})["loss"]


def _reference_loss(modules, samples) -> torch.Tensor:
    """The DiT port's monolithic forward, written out over the same parameters and the same draws."""
    text_encoder, mot, connector = (
        modules[name] for name in ("cosmos3_text_encoder", "cosmos3_mot", "cosmos3_flow_connector")
    )
    noisy, targets, masks, timesteps = connector.sample_flow([sample["latent"] for sample in samples])
    patch = connector.patch_size
    und_lengths = [int(sample["ids"].numel()) for sample in samples]
    hidden_und = text_encoder.embed_tokens(torch.cat([sample["ids"] for sample in samples]))
    rows = [patchify_latent(latent, patch) for latent in noisy]
    gen_lengths = [int(row.shape[0]) for row in rows]
    hidden_gen = connector.proj_in(torch.cat(rows))
    row_weight = torch.cat(
        [
            (1.0 - mask.reshape(-1)).repeat_interleave(length // mask.numel()).reshape(-1, 1)
            for mask, length in zip(masks, gen_lengths)
        ]
    )
    sigma = timesteps[0] if len(samples) == 1 else torch.cat(timesteps).repeat_interleave(torch.tensor(gen_lengths))
    hidden_gen = add_time_conditioning(
        hidden_gen, sigma, connector.time_embedder, connector.config.timestep_scale, row_weight=row_weight
    )
    grids = [build_latent_grid(latent.shape, (1, *patch)) for latent in noisy]
    config = mot.config
    und_rope = mot.rotary_emb(build_segmented_text_position_ids(und_lengths))
    gen_rope = mot.rotary_emb(
        build_segmented_latent_position_ids(
            grids,
            fps=[config.base_fps if sample["fps"] is None else sample["fps"] for sample in samples],
            base_fps=config.base_fps,
            temporal_compression_factor=config.temporal_compression_factor,
            enable_fps_modulation=config.enable_fps_modulation,
            temporal_offsets=[length + config.unified_3d_mrope_temporal_modality_margin for length in und_lengths],
        )
    )
    layout = build_packed_layout(und_lengths, gen_lengths) if len(samples) > 1 else None
    for layer in mot.layers:
        hidden_und, hidden_gen = layer(hidden_und, hidden_gen, und_rope=und_rope, gen_rope=gen_rope, layout=layout)
    out = connector.proj_out(mot.norm_moe_gen(hidden_gen))
    losses = [
        flow_matching_loss(
            unpatchify_latent(part, tuple(latent.shape), patch).unsqueeze(0),
            target.unsqueeze(0),
            condition_mask=[mask.unsqueeze(0)],
            loss_scale=connector.config.loss_scale,
        )
        for part, latent, target, mask in zip(out.split(gen_lengths), noisy, targets, masks)
    ]
    return torch.stack(losses).mean()


def test_every_module_is_registered():
    for model_type in MODULE_TYPES:
        for registry in (OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_ACCELERATED_MODEL_REGISTRY):
            assert model_type in registry.valid_keys(), (registry, model_type)


def test_parameter_names_are_the_checkpoint_keys_of_their_own_module():
    """Union equality with the release, and every name routes back to the module that holds it."""
    modules = _modules(accelerated=False)
    expected = set(TOP_LEVEL_KEYS)
    for index in range(TINY_MOT["num_hidden_layers"]):
        expected |= {f"layers.{index}.{suffix}" for suffix in LAYER_KEY_SUFFIXES}
    names = set()
    for module_name, module in modules.items():
        for name, _ in module.named_parameters():
            assert route_weight(name) == module_name, name
            names.add(name)
    assert names == expected, sorted(names ^ expected)


def test_dropped_and_unknown_checkpoint_keys():
    assert route_weight("blocks.0.attn.qkv.weight") is None
    assert route_weight("action_proj_in.weight") is None
    with pytest.raises(ValueError, match="no route"):
        route_weight("visual.patch_embed.proj.weight")


def test_freezing_the_reasoner_keeps_only_the_generation_tower_trainable():
    modules = _modules(accelerated=False)
    for module in modules.values():
        if hasattr(module, "freeze_model"):
            module.freeze_model()
    mot = modules["cosmos3_mot"]
    for name, parameter in mot.named_parameters():
        assert parameter.requires_grad == any(key in name for key in TRAINABLE_NAME_KEYS), name
    assert mot.layers[0].self_attn.add_q_proj.weight.requires_grad
    assert mot.norm_moe_gen.weight.requires_grad
    assert not mot.layers[0].self_attn.to_q.weight.requires_grad
    assert not any(p.requires_grad for p in modules["cosmos3_text_encoder"].parameters())
    assert all(p.requires_grad for p in modules["cosmos3_flow_connector"].parameters())


def test_joint_training_releases_everything_but_lm_head():
    torch.manual_seed(0)
    mot = OMNI_MODEL_REGISTRY["cosmos3_mot"]()(
        Cosmos3MotConfig(**TINY_MOT, train_joint_reasoner=True, ops_implementation=EAGER_OPS)
    )
    text_encoder = OMNI_MODEL_REGISTRY["cosmos3_text_encoder"]()(
        Cosmos3TextEncoderConfig(vocab_size=128, hidden_size=64, train_joint_reasoner=True)
    )
    mot.freeze_model()
    text_encoder.freeze_model()
    assert all(p.requires_grad for p in mot.parameters())
    assert text_encoder.embed_tokens.weight.requires_grad
    assert not text_encoder.lm_head.weight.requires_grad


def test_joint_training_is_rejected_under_ddp(monkeypatch):
    from veomni.models.seed_omni.modules.cosmos3.mot import modeling

    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(dp_mode="ddp"))
    mot = OMNI_MODEL_REGISTRY["cosmos3_mot"]()(
        Cosmos3MotConfig(**TINY_MOT, train_joint_reasoner=True, ops_implementation=EAGER_OPS)
    )
    with pytest.raises(NotImplementedError, match="fsdp2"):
        mot.freeze_model()


@pytest.mark.parametrize("count", [1, 2])
def test_graph_matches_the_monolithic_forward(count):
    """The four-module split, items and all, computes the DiT port's loss bit for bit."""
    samples = _samples()[:count]
    graph_modules = _modules()
    reference_modules = _modules()
    with torch.no_grad():
        graph = _graph_loss(graph_modules, samples)
        reference = _reference_loss(reference_modules, samples)
    torch.testing.assert_close(graph, reference, rtol=0, atol=0)


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_packed_forward_matches_per_sample_forward(attn_implementation):
    """Packing must isolate samples: each keeps its own positions, fps and attention range."""
    mot = _modules(attn_implementation, accelerated=False)["cosmos3_mot"]
    generator = torch.Generator().manual_seed(7)
    und_lengths, grids, fps = [7, 3], [(3, 2, 3), (2, 1, 2)], [5.0, 24.0]
    und = [torch.randn(length, 64, generator=generator) for length in und_lengths]
    gen = [torch.randn(t * h * w, 64, generator=generator) for t, h, w in grids]
    with torch.no_grad():
        packed = mot(torch.cat(und), torch.cat(gen), und_lengths=und_lengths, latent_grids=grids, fps=fps)[
            "hidden_gen"
        ].split([t * h * w for t, h, w in grids])
        for index in range(2):
            alone = mot(
                und[index],
                gen[index],
                und_lengths=und_lengths[index : index + 1],
                latent_grids=grids[index : index + 1],
                fps=fps[index : index + 1],
            )["hidden_gen"]
            torch.testing.assert_close(packed[index], alone, rtol=1e-5, atol=1e-5)


def test_loss_is_a_per_sample_mean():
    """The trainer divides by the micro-batch count, so a micro-batch must not sum its samples."""
    connector = _modules(accelerated=False)["cosmos3_flow_connector"]
    shape = (LATENT_CHANNELS, 3, 4, 6)
    rows = torch.randn(3 * 2 * 3, 64)
    target, mask = torch.randn(shape), torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    single = connector.decode_velocity(rows, [shape], [target], [mask])["_loss"]
    doubled = connector.decode_velocity(torch.cat([rows, rows]), [shape] * 2, [target] * 2, [mask] * 2)["_loss"]
    torch.testing.assert_close(doubled, single)


def test_sampling_stream_resumes_from_its_state():
    connector = _modules(accelerated=False)["cosmos3_flow_connector"]
    latents = [sample["latent"] for sample in _samples()]
    state = connector.rng_state_dict()
    first = connector.sample_flow(latents)
    connector.load_rng_state_dict(state)
    again = connector.sample_flow(latents)
    for a, b in zip(first, again):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_gradient_checkpointing_keeps_the_gradients():
    generator = torch.Generator().manual_seed(3)
    und, gen = torch.randn(5, 64, generator=generator), torch.randn(12, 64, generator=generator)
    grads = []
    for checkpointing in (False, True):
        mot = _modules(accelerated=False, train_joint_reasoner=True)["cosmos3_mot"]
        mot.freeze_model()
        mot.train()
        if checkpointing:
            mot.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        mot(und, gen, und_lengths=[5], latent_grids=[(2, 2, 3)], fps=[5.0])["hidden_gen"].square().sum().backward()
        grads.append({name: p.grad for name, p in mot.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], msg=name)


def test_latent_temporal_step_is_base_fps_over_fps():
    """A 5 fps clip at base_fps=24 advances 4.8 per latent frame, not 1.0."""
    positions = latent_temporal_positions(6, fps=5.0)
    assert (positions[1:] - positions[:-1]).tolist() == pytest.approx([4.8] * 5, abs=1e-6)


class _ChatTokenizer:
    """``apply_chat_template`` that emits ``[10, <one id per char>, 11]``."""

    eos_token_id = None
    bos_token_id = None

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs["add_generation_prompt"] and kwargs["tokenize"]
        return [10, *(100 + ord(char) % 20 for char in conversation[0]["content"]), 11]


def test_chat_template_truncates_then_appends_the_markers():
    template = Cosmos3ChatTemplate(_ChatTokenizer(), generation_marker_ids=[1, 2], max_sequence_length=4)
    sample = [
        ConversationItem(type="text", value="abcdef", role="user"),
        ConversationItem(type="video", value=None, role="assistant"),
    ]
    parts = template.tokenize_conversation(sample)
    prompt = parts[0]
    assert prompt.value.tolist()[:4] == [10, 100 + ord("a") % 20, 100 + ord("b") % 20, 100 + ord("c") % 20]
    assert prompt.value.tolist()[4:] == [1, 2]
    assert (prompt.meta["labels"] == -100).all()
    assert prompt.meta[_OMNI_TOKEN_LEN] == 6
    assert parts[1].type == "video"


@pytest.mark.parametrize(("prompt", "price"), [("abcdefghi", 3), ("", 1)])
def test_prompt_priced_by_chars_like_the_dit_port(prompt, price):
    """The DiT port's online estimate: ceil(chars / 4), at least 1; the ids are unchanged."""
    template = Cosmos3ChatTemplate(_ChatTokenizer(), generation_marker_ids=[1, 2], price_prompt_by_chars=True)
    part = template.tokenize_conversation([ConversationItem(type="text", value=prompt, role="user")])[0]
    assert part.meta[_OMNI_TOKEN_LEN] == price
    assert part.value.numel() == len(prompt) + 4


def test_text_preprocessor_leaves_cached_prompts_alone():
    from veomni.models.seed_omni.modules.cosmos3.text_encoder.processing import Cosmos3TextEncoderPreprocessor

    template = Cosmos3ChatTemplate(_ChatTokenizer(), generation_marker_ids=[1, 2])
    preprocessor = Cosmos3TextEncoderPreprocessor(template)
    fresh = [ConversationItem(type="text", value="ab", role="user")]
    cached = [ConversationItem(type="text", value=torch.tensor([5, 6]), role="user")]
    preprocessor.preprocess_conversations([fresh, cached])
    assert fresh[0].value.tolist() == [10, 100 + ord("a") % 20, 100 + ord("b") % 20, 11, 1, 2]
    assert cached[0].value.tolist() == [5, 6]


@pytest.mark.parametrize(
    ("total", "video_fps", "indices", "pad"),
    [
        (40, 5.0, list(range(37)), 0),  # 40 frames at the sampling rate: the 4k+1 prefix
        (90, 15.0, list(range(0, 90, 3))[:29], 0),  # 6 s at 15 fps: 30 -> 29 frames, stride 3
        (11, 30.0, [0, 2, 4, 6, 8], 0),  # shorter than min_frames: 5 frames, stride round(11/5)
        (2, 5.0, [0, 1], 3),  # a 2-frame clip pads up to 5
        # Capped at 117, but stride round(1000/117)=9 reaches only 112 frames; the rest pad.
        (1000, 5.0, list(range(0, 1000, 9)), 5),
    ],
)
def test_prefix_frame_indices_match_the_dit_port(total, video_fps, indices, pad):
    """The DiT port's calculate_frame_indices(frame_selection="prefix") with the recipe's settings."""
    assert prefix_frame_indices(total, video_fps, fps=5.0, min_frames=5, max_frames=117, factor=4, remainder=1) == (
        indices,
        pad,
    )


def test_prefix_frame_count_is_calculate_frame_indices():
    try:
        from veomni.data.multimodal import video_utils
    except ImportError:
        pytest.skip("veomni.data.multimodal.video_utils needs PyAV")
    for total in range(1, 400, 7):
        for video_fps in (5.0, 7.5, 15.0, 29.97, 30.0):
            spread, pad = video_utils.calculate_frame_indices(
                total_frames=total,
                video_fps=video_fps,
                fps=5.0,
                frame_factor=4,
                frame_factor_remainder=1,
                min_frames=5,
                max_frames=117,
            )
            indices, prefix_pad = prefix_frame_indices(
                total, video_fps, fps=5.0, min_frames=5, max_frames=117, factor=4, remainder=1
            )
            assert len(indices) + prefix_pad == len(spread) + pad


def _clip_item(frames: int, height: int, width: int, stride: int = 1, fps: float = 30.0) -> ConversationItem:
    video = torch.randint(0, 256, (frames, 3, height, width), dtype=torch.uint8)
    metadata = VideoMetadata(
        fps=fps, total_num_frames=frames * stride, frames_indices=list(range(0, frames * stride, stride))
    )
    return ConversationItem(
        type="video", value=SimpleNamespace(video=video), role="assistant", meta={VIDEO_METADATA_KEY: metadata}
    )


def test_vae_preprocessor_cuts_resizes_and_prices_the_clip():
    preprocessor = Cosmos3VaePreprocessor(
        Cosmos3VaeConfig(vae_config={"scale_factor_spatial": 16, "scale_factor_temporal": 4})
    )
    long_clip, short_clip = _clip_item(11, 70, 90), _clip_item(2, 64, 64)
    source = long_clip.value.video.clone()
    cached = ConversationItem(type="video", value=torch.zeros(48, 1, 2, 2), role="assistant", meta={LATENT_KEY: True})
    preprocessor.preprocess_conversations([[long_clip], [short_clip], [cached]])

    # 11 frames at 30 fps keep 5, every other one; 70x90 resizes to the 32-grid 64x96.
    assert tuple(long_clip.value.shape) == (3, 5, 64, 96)
    assert long_clip.value.min() >= -1.0 and long_clip.value.max() <= 1.0
    expected = Cosmos3VaePreprocessor.to_vae_input(preprocessor, source[[0, 2, 4, 6, 8]])
    assert torch.equal(long_clip.value, expected)
    # Priced on the frames before the resize: 2 latent frames of ceil(4/2) x ceil(5/2) patches, plus three markers.
    assert long_clip.meta[_OMNI_TOKEN_LEN] == 2 * 2 * 3 + 3
    # The mRoPE spaces latent frames by the source clip's rate.
    assert long_clip.meta[FPS_KEY] == 30.0
    # 2 frames pad up to 5 by repeating the last one.
    assert short_clip.value.shape[1] == 5
    torch.testing.assert_close(short_clip.value[:, 4], short_clip.value[:, 1])
    assert cached.value.shape == (48, 1, 2, 2) and _OMNI_TOKEN_LEN not in cached.meta


def test_vae_preprocessor_needs_every_source_frame():
    preprocessor = Cosmos3VaePreprocessor(
        Cosmos3VaeConfig(vae_config={"scale_factor_spatial": 16, "scale_factor_temporal": 4})
    )
    with pytest.raises(ValueError, match="mm_configs.fps: null"):
        preprocessor.preprocess_conversations([[_clip_item(11, 64, 64, stride=6)]])


def test_sft_jsonl_preprocess():
    from veomni.data.seed_omni.preprocess import cosmos3_sft_jsonl_preprocess

    row = {"vision_path": "videos/a.mp4", "t2w_windows": [{"caption_json": {"subjects": ["a bridge"]}}]}
    constructed, media = cosmos3_sft_jsonl_preprocess(row, row, data_dir="/data")
    assert constructed == [["user", ("text", '{"subjects": ["a bridge"]}')], ["assistant", ("video", None)]]
    assert media == {"video": [os.path.join("/data", "videos/a.mp4")]}

    dense = {"vision_path": "/abs/a.mp4", "t2w_windows": [{"caption": "a bridge"}]}
    constructed, media = cosmos3_sft_jsonl_preprocess(dense, dense, data_dir="/data")
    assert constructed[0][1] == ("text", "a bridge.") and media == {"video": ["/abs/a.mp4"]}
    assert cosmos3_sft_jsonl_preprocess(row, row, cfg_dropout_rate=1.0)[0][0][1] == ("text", "")

    # A blank structured caption falls back to the dense ones; a URI is not joined to data_dir.
    blank = {
        "vision_path": "https://host/a.mp4",
        "t2w_windows": [{"caption_json": " ", "caption": " "}],
        "caption": "a bridge",
    }
    constructed, media = cosmos3_sft_jsonl_preprocess(blank, blank, data_dir="/data")
    assert constructed[0][1] == ("text", "a bridge.") and media == {"video": ["https://host/a.mp4"]}


@pytest.mark.parametrize("name", ["modules_train.yaml", "modules_offline_embedding.yaml"])
def test_only_the_mot_recomputes(name):
    """The recipes turn recompute on globally; modules without decoder blocks must opt out."""
    import yaml

    path = Path(__file__).parents[4] / "configs/seed_omni/Cosmos3/cosmos3_nano/train" / name
    modules = yaml.safe_load(path.read_text())
    for module, entry in modules.items():
        enabled = entry.get("accelerator", {}).get("gradient_checkpointing", {}).get("enable", True)
        assert enabled == (module == "cosmos3_mot"), module
