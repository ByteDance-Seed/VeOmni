"""End-to-end BAGEL generation tests across ``infer_und``, ``infer_gen``, and ``infer_edit``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.seed_omni.modules.bagel.helpers import (
    FakeBagelTokenizer,
    bagel_cfg_dir,
    bagel_template,
    config_cls,
    load_omni_config,
    native_model_cls,
    tiny_bagel_qwen2_cfg,
)
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated.accelerated import BagelQwen2MoTAccelerated
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    BagelSiglipNavitPreprocessor,
    BagelSiglipNavitProcessor,
)
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_CONTEXT_KEY,
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_PHASE_KEY,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_START_TOKEN,
    BAGEL_VAE_CONTEXT,
)
from veomni.models.seed_omni.modules.bagel.text_encoder.processing import BagelTextEncoderPreprocessor
from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEPreprocessor, BagelVAEProcessor
from veomni.models.seed_omni.processing_omni import OmniProcessor
from veomni.models.seed_omni.utils.conversation import ConversationItem


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _build_tiny_bagel_omni_model_and_processor(infer_graph_rel: str) -> tuple[OmniModel, OmniProcessor]:
    """Build a tiny 5-module real BAGEL ``OmniModel`` and ``OmniProcessor`` for ``infer_graph_rel``."""
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "train/modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "train/graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / infer_graph_rel,
    )
    cfg.training_graphs = {}

    hidden_size = 8
    qwen_kwargs = {
        **tiny_bagel_qwen2_cfg(),
        "hidden_size": hidden_size,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
    }
    text_encoder = native_model_cls("bagel_text_encoder")(
        config_cls("bagel_text_encoder")(vocab_size=int(qwen_kwargs["vocab_size"]), hidden_size=hidden_size)
    ).eval()
    text_encoder._tokenizer = FakeBagelTokenizer()
    text_encoder._chat_template = bagel_template()

    siglip = native_model_cls("bagel_siglip_navit")(
        config_cls("bagel_siglip_navit")(
            hidden_size=hidden_size,
            output_size=hidden_size,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=4,
            patch_size=2,
            rope=False,
            ops_implementation={"attn_implementation": "eager", "rotary_pos_emb_vision_implementation": "eager"},
        )
    ).eval()

    qwen_mot = native_model_cls("bagel_qwen2_mot")(config_cls("bagel_qwen2_mot")(**qwen_kwargs)).eval()

    flow = native_model_cls("bagel_flow_connector")(
        config_cls("bagel_flow_connector")(
            hidden_size=hidden_size,
            z_channels=2,
            latent_patch_size=1,
            patch_latent_dim=2,
            max_latent_size=4,
            timestep_frequency_embedding_size=4,
            resolution=4,
        )
    ).eval()

    vae_cfg = config_cls("bagel_vae")(
        resolution=4,
        ch=32,
        ch_mult=[1],
        num_res_blocks=1,
        z_channels=2,
        max_image_size=4,
        min_image_size=4,
        image_stride=2,
        max_pixels=16,
        downsample=1,
    )
    vae = native_model_cls("bagel_vae")(vae_cfg).eval()
    vae._image_processor = BagelVAEProcessor.from_config(vae_cfg)

    model = OmniModel(
        cfg,
        {
            "bagel_text_encoder": text_encoder,
            "bagel_siglip_navit": siglip,
            "bagel_qwen2_mot": qwen_mot,
            "bagel_flow_connector": flow,
            "bagel_vae": vae,
        },
    ).eval()
    preprocessors = {
        "bagel_vae": BagelVAEPreprocessor(
            BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
            dtype=torch.float32,
        ),
        "bagel_siglip_navit": BagelSiglipNavitPreprocessor(
            BagelSiglipNavitProcessor(
                patch_size=2,
                image_size=4,
                min_image_size=2,
                max_pixels=16,
                vit_max_num_patch_per_side=2,
            ),
            dtype=torch.float32,
        ),
        "bagel_text_encoder": BagelTextEncoderPreprocessor(bagel_template()),
    }
    return model, OmniProcessor(preprocessors, infer_type=model.config.infer_type)


def test_bagel_e2e_generation_across_und_gen_and_edit_scenarios(tmp_path: Path) -> None:
    """Verify real 5-module BAGEL ``OmniProcessor`` + ``OmniModel.generate`` across ``infer_und``, ``infer_gen``, and ``infer_edit``."""
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(12, 34, 56)).save(str(image_path))

    # 1. infer_und: routes image to SigLIP only, appends BAGEL_START_TOKEN, prefills + decodes text
    und_model, und_processor = _build_tiny_bagel_omni_model_and_processor("infer/graph_infer_und.yaml")
    assert und_model.config.infer_type == "infer_und"
    und_request = und_processor(text="hi", images=str(image_path))
    assert [item.meta.get(BAGEL_CONTEXT_KEY) for item in und_request["conversation_list"] if item.type == "image"] == [
        BAGEL_SIGLIP_CONTEXT
    ]
    assert und_request["conversation_list"][-1].meta.get(BAGEL_PHASE_KEY) == BAGEL_START_TOKEN
    und_model.reset()
    und_generated = und_model.generate(
        und_request,
        generation_kwargs={"max_new_tokens": 2, "do_sample": False},
    )
    assert any(item["type"] == "text" for item in und_generated)
    assert und_model.modules_dict["bagel_qwen2_mot"]._generation_state.main.cache is not None

    # 2. infer_gen: text prompt -> CFG flow denoise loop -> decoded PIL image
    gen_model, gen_processor = _build_tiny_bagel_omni_model_and_processor("infer/graph_infer_gen.yaml")
    assert gen_model.config.infer_type == "infer_gen"
    gen_request = gen_processor(text="draw")
    gen_model.reset()
    gen_generated = gen_model.generate(
        gen_request,
        generation_kwargs={
            "num_timesteps": 2,
            "image_height": 4,
            "image_width": 4,
            "latent_downsample": 2,
            "cfg_text_scale": 2.0,
            "cfg_img_scale": 1.0,
        },
    )
    assert any(item["type"] == "image" and isinstance(item["value"], Image.Image) for item in gen_generated)

    # 3. infer_edit: routes user image to both VAE and SigLIP context -> 3-branch CFG denoise -> decoded PIL image
    edit_model, edit_processor = _build_tiny_bagel_omni_model_and_processor("infer/graph_infer_edit.yaml")
    assert edit_model.config.infer_type == "infer_edit"
    edit_request = edit_processor(text="edit", images=str(image_path))
    assert [
        item.meta.get(BAGEL_CONTEXT_KEY) for item in edit_request["conversation_list"] if item.type == "image"
    ] == [BAGEL_VAE_CONTEXT, BAGEL_SIGLIP_CONTEXT]
    edit_model.reset()
    edit_generated = edit_model.generate(
        edit_request,
        generation_kwargs={
            "num_timesteps": 2,
            "image_height": 4,
            "image_width": 4,
            "latent_downsample": 2,
            "cfg_text_scale": 2.0,
            "cfg_img_scale": 1.5,
        },
    )
    assert any(item["type"] == "image" and isinstance(item["value"], Image.Image) for item in edit_generated)


def test_bagel_qwen2_mot_accelerated_denoise_branch_packs_cfg_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify accelerated ``denoise_branch`` stacks active CFG branches into one packed ``forward_inference`` call."""
    model = BagelQwen2MoTAccelerated(config_cls("bagel_qwen2_mot")(**tiny_bagel_qwen2_cfg()))
    state = model._generation_state
    for branch, length, next_pos in ((state.main, 2, 3), (state.cfg_text, 0, 7), (state.cfg_img, 3, 11)):
        cache = model._new_empty_cache()
        if length > 0:
            cache.key_cache[0] = torch.ones(length, 1, 1)
            cache.value_cache[0] = torch.ones(length, 1, 1)
        branch.install_cache(
            cache=cache,
            cache_len=length,
            next_position_id=torch.tensor(next_pos),
            device=model.device,
        )

    captured: dict[str, object] = {}

    def _capture_forward_inference(self, **kwargs):
        del self
        captured.update(kwargs)
        return {"hidden_states": kwargs["packed_query_sequence"]}

    monkeypatch.setattr(type(model), "forward_inference", _capture_forward_inference)
    tail = ConversationItem(
        type="output",
        value=torch.zeros(5, int(model.config.hidden_size)),
        role="assistant",
        meta={BAGEL_PHASE_KEY: BAGEL_FLOW_QUERY, "timestep": 0.5},
    )
    model.denoise_branch([tail], generation_kwargs={"cfg_text_scale": 2.0, "cfg_img_scale": 1.5})

    assert captured["query_lens"].tolist() == [5, 5, 5]
    assert int(captured["packed_query_sequence"].shape[0]) == 15
    assert tail.meta.get(BAGEL_PHASE_KEY) == BAGEL_FLOW_HIDDEN
    assert int(tail.value.shape[0]) == 15
