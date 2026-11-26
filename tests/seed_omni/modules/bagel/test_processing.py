"""BAGEL preprocessing, MoT packing, and offline VAE cache round-trip tests."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.seed_omni.modules.bagel.helpers import (
    bagel_template,
    config_cls,
    model_cls,
    tiny_bagel_qwen2_cfg,
)
from veomni.data.dataset import get_length_by_labels_fn, get_length_fn
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import accelerated as mot_acc
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated.accelerated import TrainingMixin
from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import preprocess_mot_inputs
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    _OMNI_TOKEN_LEN as BAGEL_SIGLIP_TOKEN_LEN,
)
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    BagelSiglipNavitPreprocessor,
    BagelSiglipNavitProcessor,
)
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_CONTEXT_KEY,
    BAGEL_PHASE_KEY,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_START_TOKEN,
    BAGEL_VAE_CONTEXT,
)
from veomni.models.seed_omni.modules.bagel.text_encoder.processing import BagelTextEncoderPreprocessor
from veomni.models.seed_omni.modules.bagel.vae.processing import (
    BAGEL_VAE_PIXEL_SHAPE,
    BAGEL_VAE_POSTERIOR,
    BagelVAEPreprocessor,
    BagelVAEProcessor,
)
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, _OMNI_TOKEN_LEN, ConversationItem


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _tiny_vae(meta: bool = False, **config_overrides):
    """Construct a tiny ``BagelVAE`` on CPU or meta device with a bound processor."""
    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    config_kwargs = dict(
        resolution=8,
        ch=32,
        ch_mult=[1],
        num_res_blocks=1,
        z_channels=2,
        max_image_size=8,
        min_image_size=4,
        image_stride=4,
        max_pixels=64,
        downsample=1,
    )
    config_kwargs.update(config_overrides)
    with torch.device("meta" if meta else "cpu"):
        model = BagelVAE(BagelVAEConfig(**config_kwargs))
    model._image_processor = BagelVAEProcessor.from_config(model.config)
    return model


def _build_preprocessors():
    return (
        BagelVAEPreprocessor(
            BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
            dtype=torch.bfloat16,
            dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
            dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
        ),
        BagelSiglipNavitPreprocessor(
            BagelSiglipNavitProcessor(
                patch_size=2,
                image_size=4,
                min_image_size=2,
                max_pixels=16,
                vit_max_num_patch_per_side=2,
            ),
            dtype=torch.bfloat16,
            dummy_pixel_values=torch.zeros(1, 12, dtype=torch.bfloat16),
        ),
        BagelTextEncoderPreprocessor(bagel_template()),
    )


def test_bagel_preprocessors_route_training_and_inference_contexts() -> None:
    """Verify training (``edit``/``gen`` + text-only dummy injection + ``dyn_bsz`` token lens) and inference routing."""
    vae_pre, siglip_pre, text_pre = _build_preprocessors()
    user_image = torch.full((3, 4, 4), 7, dtype=torch.uint8)
    assistant_image = torch.full((3, 4, 4), 9, dtype=torch.uint8)

    # 1. Training batch: sample 0 has edit + gen images; sample 1 is text-only (receives both VAE and SigLIP dummies)
    train_batch = [
        [
            ConversationItem(type="text", value="edit", role="user"),
            ConversationItem(type="image", value=user_image.clone(), role="user", meta={_IMG_TAG_KEY: "edit"}),
            ConversationItem(
                type="image", value=assistant_image.clone(), role="assistant", meta={_IMG_TAG_KEY: "gen"}
            ),
        ],
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="text", value="ok", role="assistant"),
        ],
    ]
    for pre in (vae_pre, siglip_pre, text_pre):
        pre({"conversation_list": train_batch})

    edit_sample, text_sample = train_batch
    assert [item.meta.get(BAGEL_CONTEXT_KEY) for item in edit_sample if item.type == "image"] == [
        BAGEL_VAE_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
        BAGEL_VAE_CONTEXT,
    ]
    assert [item.meta.get(_IMG_TAG_KEY) for item in edit_sample if item.type == "image"] == ["edit", "edit", "gen"]
    dummies = [item for item in text_sample if item.is_dummy and item.type == "image"]
    assert {item.meta.get(BAGEL_CONTEXT_KEY) for item in dummies} == {BAGEL_VAE_CONTEXT, BAGEL_SIGLIP_CONTEXT}
    for sample in train_batch:
        assert all(
            isinstance(item.meta.get(_OMNI_TOKEN_LEN), int) and item.meta[_OMNI_TOKEN_LEN] > 0 for item in sample
        )
    assert (
        get_length_fn({"conversation_list": edit_sample})
        > get_length_by_labels_fn({"conversation_list": edit_sample})
        > 0
    )

    # 2. Inference routing: infer_und routes to SigLIP + appends BAGEL_START_TOKEN; infer_edit expands to VAE + SigLIP
    und_batch = [[ConversationItem(type="image", value=user_image.clone(), role="user")]]
    for pre in (vae_pre, siglip_pre, text_pre):
        pre({"conversation_list": und_batch}, inference=True, generation_kwargs={"infer_type": "infer_und"})
    assert [item.meta.get(BAGEL_CONTEXT_KEY) for item in und_batch[0] if item.type == "image"] == [
        BAGEL_SIGLIP_CONTEXT
    ]
    assert und_batch[0][1].meta[BAGEL_SIGLIP_TOKEN_LEN] == 4
    assert und_batch[0][-1].meta.get(BAGEL_PHASE_KEY) == BAGEL_START_TOKEN


def test_bagel_mot_packing_sp_padding_and_scatter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ``preprocess_mot_inputs`` expert routing, ``forward_pre`` SP padding, and ``forward_post`` triplet scatter."""
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg())).train()
    hidden_size = int(model.config.hidden_size)

    monkeypatch.setattr(
        mot_acc,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_size=4, cp_size=1, ulysses_size=4, sp_group=object()),
    )

    def _fake_sp_pad(tensor, dim, pad_value):
        pad_shape = list(tensor.shape)
        pad_shape[dim] = 3
        padding = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
        return torch.cat((tensor, padding), dim=dim)

    monkeypatch.setattr(mot_acc, "sp_pad", _fake_sp_pad)
    monkeypatch.setattr(mot_acc, "slice_input_tensor", lambda tensor, **kwargs: tensor)

    text = ConversationItem(type="text", value=torch.ones(2, hidden_size), role="user")
    image = ConversationItem(
        type="image",
        value=torch.ones(3, hidden_size),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "gen"},
    )
    inputs = model.forward_pre(conversation_list=[[text, image]])
    assert torch.equal(inputs["packed_token_type_ids"], torch.tensor([0, 0, 1, 1, 1, -1, -1, -1]))
    assert torch.equal(inputs["packed_attention_metadata"][2, 5:], torch.tensor([-1, -1, -1], dtype=torch.int32))

    # Triplet scatter in forward_post
    def _marker(context: str) -> ConversationItem:
        return ConversationItem(
            type="text",
            value=torch.zeros(1, 4),
            role="user",
            meta={"labels": torch.full((1,), -100), BAGEL_CONTEXT_KEY: context},
        )

    latent_item = ConversationItem(
        type="image",
        value=torch.zeros(2, 4),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "edit"},
    )
    siglip_item = ConversationItem(
        type="image",
        value=torch.zeros(3, 4),
        role="user",
        meta={BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT, _IMG_TAG_KEY: "edit"},
    )
    sample = [
        _marker(BAGEL_VAE_CONTEXT),
        latent_item,
        _marker(BAGEL_VAE_CONTEXT),
        _marker(BAGEL_SIGLIP_CONTEXT),
        siglip_item,
        _marker(BAGEL_SIGLIP_CONTEXT),
    ]
    packed = preprocess_mot_inputs([sample], device=torch.device("cpu"), dtype=torch.float32, hidden_size=4)
    assert packed is not None
    assert torch.equal(packed.packed_token_type_ids, torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0]))
    hidden_states = torch.arange(9 * 4, dtype=torch.float32).reshape(9, 4)
    monkeypatch.setattr(
        mot_acc,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_size=1, cp_size=1, ulysses_size=1, sp_group=None),
    )
    stub = SimpleNamespace(device=torch.device("cpu"), _conversation_carrier=[sample], _packed_training=packed)
    TrainingMixin.forward_post(stub, hidden_states)
    assert torch.equal(latent_item.value, hidden_states[1:3])
    assert torch.equal(siglip_item.value, hidden_states[5:8])


def test_bagel_vae_encode_and_offline_cache_roundtrip(tmp_path: Path) -> None:
    """Verify ragged-shape VAE ``encode``/``offline_encode`` and parquet cache replay into ``BagelFlowConnector``."""
    from datasets import load_dataset

    from veomni.data.seed_omni.seedomni_transform import process_seedomni_cached_example
    from veomni.models.seed_omni.utils.offline_cache import SeedOmniOfflineCacheWriter

    vae = _tiny_vae(ch_mult=[1, 1], downsample=2)
    meta_vae = _tiny_vae(meta=True, ch_mult=[1, 1], downsample=2)

    edit_img = ConversationItem(
        type="image",
        value=torch.zeros(3, 4, 8),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "edit", BAGEL_VAE_PIXEL_SHAPE: torch.tensor([4, 8])},
    )
    gen_img = ConversationItem(
        type="image",
        value=torch.ones(3, 8, 4),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "gen", BAGEL_VAE_PIXEL_SHAPE: torch.tensor([8, 4])},
    )

    # 1. Ragged micro-batch offline_encode matches single-item posteriors without spatial padding
    single_edit = copy.deepcopy([[edit_img]])
    vae.post_forward(
        "offline_encode", **vae.offline_encode(**vae.pre_forward("offline_encode", conversation_list=single_edit))
    )
    single_gen = copy.deepcopy([[gen_img]])
    vae.post_forward(
        "offline_encode", **vae.offline_encode(**vae.pre_forward("offline_encode", conversation_list=single_gen))
    )

    batch = [[copy.deepcopy(edit_img), copy.deepcopy(gen_img)]]
    pre = vae.pre_forward("offline_encode", conversation_list=batch)
    encoded = vae.post_forward("offline_encode", **vae.offline_encode(**pre))["conversation_list"]
    torch.testing.assert_close(encoded[0][0].value, single_edit[0][0].value)
    torch.testing.assert_close(encoded[0][1].value, single_gen[0][0].value)

    # 2. Parquet round-trip -> meta-built online_process -> BagelFlowConnector.embed_latent_pre
    writer = SeedOmniOfflineCacheWriter(str(tmp_path))
    writer.open()
    writer.save_conversation_list(encoded)
    writer.flush()
    dataset = load_dataset("parquet", data_files=[str(tmp_path / "shard_000000.parquet")], split="train")
    replayed = [process_seedomni_cached_example(dataset[0])[0]["conversation_list"]]
    assert all(item.meta[BAGEL_VAE_POSTERIOR] for item in replayed[0])

    assert meta_vae.encoder.conv_in.weight.is_meta
    processed = meta_vae.post_forward(
        "online_process",
        **meta_vae.online_process(**meta_vae.pre_forward("online_process", conversation_list=replayed)),
    )["conversation_list"]
    assert [item.meta[_IMG_TAG_KEY] for item in processed[0]] == ["edit", "gen"]

    flow = model_cls("bagel_flow_connector")(
        config_cls("bagel_flow_connector")(
            hidden_size=4,
            z_channels=2,
            latent_patch_size=1,
            patch_latent_dim=2,
            max_latent_size=4,
            timestep_frequency_embedding_size=4,
            timestep_shift=3.0,
        )
    )
    edit_item, gen_item = processed[0]
    inputs = flow.embed_latent_pre(conversation_list=processed)
    assert flow._embed_lengths == [8, 8]
    assert torch.equal(inputs["timesteps"][:8], torch.zeros(8))
    assert "flow_velocity_target" not in edit_item.meta
    assert gen_item.meta["flow_velocity_target"].shape == (8, 2)
    assert torch.all(gen_item.meta["timestep"] == gen_item.meta["timestep"][0])
