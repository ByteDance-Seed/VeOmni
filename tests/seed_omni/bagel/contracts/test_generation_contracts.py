from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from PIL import Image

from tests.seed_omni.bagel.contracts.helpers import (
    bagel_cfg_dir,
    config_cls,
    load_omni_config,
    model_cls,
    tiny_bagel_qwen2_cfg,
)
from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.mixins.modulemixin import ModuleMixin
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.bagel.qwen2_mot.generation_state import MotGenerationState
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_FLOW_VELOCITY,
    BAGEL_GENERATED_LATENT,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_VAE_CONTEXT,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


def test_bagel_infer_gen_denoise_signal_smoke():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_modules=bagel_cfg_dir() / "modules_infer_eager.yaml",
        infer_graph_path=bagel_cfg_dir() / "graph_infer_gen.yaml",
    )
    model = OmniModel(
        cfg,
        {
            "bagel_text_encoder": _InferGenTextEncoder(),
            "bagel_siglip_navit": _NoopBagelSiglip(),
            "bagel_qwen2_mot": _InferGenBagelQwen(),
            "bagel_flow_connector": _InferGenBagelFlow(),
            "bagel_vae": _InferGenBagelVAE(),
        },
    ).eval()
    trace: list[str] = []
    ctx = model.generate(
        {"conversation_list": [ConversationItem(type="text", value="prompt", role="user")]},
        trace=trace,
        generation_kwargs={
            "max_new_tokens": 8,
            "do_sample": False,
            "image_height": 64,
            "image_width": 64,
        },
    )

    assert any("transition: prompt_encode -> query_denoise" in entry for entry in trace)
    assert any("transition: query_denoise -> velocity_collect" in entry for entry in trace)
    assert any("transition: velocity_collect -> image_decode" in entry for entry in trace)
    assert any("transition: image_decode -> done" in entry for entry in trace)
    assert any(item["type"] == "image" for item in model.generated)
    assert "timestep" not in ctx["conversation_list"][-1].meta


def test_bagel_infer_edit_defaults_to_denoise_signal_smoke():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_modules=bagel_cfg_dir() / "modules_infer_eager.yaml",
        infer_graph_path=bagel_cfg_dir() / "graph_infer_edit.yaml",
    )
    model = OmniModel(
        cfg,
        {
            "bagel_text_encoder": _InferGenTextEncoder(),
            "bagel_siglip_navit": _NoopBagelSiglip(),
            "bagel_qwen2_mot": _InferGenBagelQwen(),
            "bagel_flow_connector": _InferEditBagelFlow(),
            "bagel_vae": _InferEditBagelVAE(),
        },
    ).eval()
    trace: list[str] = []
    ctx = model.generate(
        {
            "conversation_list": [
                ConversationItem(
                    type="image",
                    value=Image.new("RGB", (1, 1)),
                    role="user",
                    source=BAGEL_VAE_CONTEXT,
                ),
                ConversationItem(
                    type="image",
                    value=Image.new("RGB", (1, 1)),
                    role="user",
                    source=BAGEL_SIGLIP_CONTEXT,
                ),
                ConversationItem(type="text", value="prompt", role="user"),
            ]
        },
        trace=trace,
        generation_kwargs={
            "max_new_tokens": 8,
            "do_sample": False,
            "image_height": 64,
            "image_width": 64,
        },
    )

    assert any("transition: prompt_encode -> query_denoise" in entry for entry in trace)
    assert not any("transition: prompt_encode -> text_ar" in entry for entry in trace)
    assert any("transition: image_decode -> done" in entry for entry in trace)
    assert any(item["type"] == "image" for item in model.generated)
    assert "timestep" not in ctx["conversation_list"][-1].meta


def test_bagel_infer_gen_cfg_text_branch_signal_smoke():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_modules=bagel_cfg_dir() / "modules_infer_eager.yaml",
        infer_graph_path=bagel_cfg_dir() / "graph_infer_gen.yaml",
    )
    model = OmniModel(
        cfg,
        {
            "bagel_text_encoder": _InferGenTextEncoder(),
            "bagel_siglip_navit": _NoopBagelSiglip(),
            "bagel_qwen2_mot": _InferGenBagelQwen(),
            "bagel_flow_connector": _InferGenBagelFlow(),
            "bagel_vae": _InferGenBagelVAE(),
        },
    ).eval()
    trace: list[str] = []
    ctx = model.generate(
        {"conversation_list": [ConversationItem(type="text", value="prompt", role="user")]},
        trace=trace,
        generation_kwargs={
            "max_new_tokens": 8,
            "do_sample": False,
            "image_height": 64,
            "image_width": 64,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
        },
    )

    assert not any("module_signal(need_denoise_branch)" in entry for entry in trace)
    assert any(
        "transition: velocity_collect -> image_decode [module_signal(image_complete)]" in entry for entry in trace
    )
    assert any(item["type"] == "image" for item in model.generated)
    assert "timestep" not in ctx["conversation_list"][-1].meta


def test_bagel_infer_gen_cfg_text_and_image_branch_signal_smoke():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_modules=bagel_cfg_dir() / "modules_infer_eager.yaml",
        infer_graph_path=bagel_cfg_dir() / "graph_infer_gen.yaml",
    )
    model = OmniModel(
        cfg,
        {
            "bagel_text_encoder": _InferGenTextEncoder(),
            "bagel_siglip_navit": _NoopBagelSiglip(),
            "bagel_qwen2_mot": _InferGenBagelQwen(),
            "bagel_flow_connector": _InferGenBagelFlow(),
            "bagel_vae": _InferGenBagelVAE(),
        },
    ).eval()
    trace: list[str] = []
    ctx = model.generate(
        {"conversation_list": [ConversationItem(type="text", value="prompt", role="user")]},
        trace=trace,
        generation_kwargs={
            "max_new_tokens": 12,
            "do_sample": False,
            "image_height": 64,
            "image_width": 64,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.5,
        },
    )

    assert not any("module_signal(need_denoise_branch)" in entry for entry in trace)
    assert any(
        "transition: velocity_collect -> image_decode [module_signal(image_complete)]" in entry for entry in trace
    )
    assert any(item["type"] == "image" for item in model.generated)
    assert "timestep" not in ctx["conversation_list"][-1].meta


def test_bagel_qwen2_mot_cfg_text_context_snapshot_is_internal():
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))

    model._generation_state.cfg_text.snapshot(
        cache=None,
        key_values_lens=None,
        packed_key_value_indexes=None,
        next_position_id=torch.tensor(7),
        empty_cache_factory=model._new_empty_cache,
        device=model.device,
    )
    cfg_text_context = model._generation_state.cfg_text
    assert cfg_text_context.cache is not None
    assert cfg_text_context.cache_len() == 0
    assert cfg_text_context.repeated_position_ids(3, device=model.device).tolist() == [7, 7, 7]
    assert cfg_text_context.key_values_lens.tolist() == [0]
    assert cfg_text_context.packed_key_value_indexes.numel() == 0

    cache = model._new_empty_cache()
    model._generation_state.cfg_text.snapshot(
        cache=cache,
        key_values_lens=torch.tensor([5], dtype=torch.int32),
        packed_key_value_indexes=torch.arange(5),
        next_position_id=torch.tensor(11),
        empty_cache_factory=model._new_empty_cache,
        device=model.device,
    )

    assert cfg_text_context.cache is not cache
    assert cfg_text_context.cache_len() == 5
    assert cfg_text_context.repeated_position_ids(2, device=model.device).tolist() == [11, 11]
    assert cfg_text_context.packed_key_value_indexes.tolist() == [0, 1, 2, 3, 4]


def test_bagel_qwen2_mot_cfg_img_context_accessors_are_internal():
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))

    model._generation_state.cfg_img.ensure_empty(empty_cache_factory=model._new_empty_cache, device=model.device)
    cfg_img_context = model._generation_state.cfg_img
    assert cfg_img_context.cache is not None
    assert cfg_img_context.cache_len() == 0
    assert cfg_img_context.repeated_position_ids(3, device=model.device).tolist() == [0, 0, 0]
    assert cfg_img_context.key_values_lens.tolist() == [0]
    assert cfg_img_context.packed_key_value_indexes.numel() == 0


def test_bagel_qwen2_mot_cfg_img_requires_text_cfg():
    with pytest.raises(ValueError, match="cfg_img_scale > 1.0 requires cfg_text_scale > 1.0"):
        MotGenerationState().validate_cfg_request({"cfg_text_scale": 1.0, "cfg_img_scale": 1.5})


def test_bagel_qwen2_mot_branch_batch_indexes_use_per_branch_attention_offsets():
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))
    state = model._generation_state
    state.main.snapshot(
        cache=_fake_cache(model, torch.tensor([10.0, 11.0])),
        key_values_lens=torch.tensor([2], dtype=torch.int32),
        packed_key_value_indexes=torch.arange(2),
        next_position_id=torch.tensor(3),
        empty_cache_factory=model._new_empty_cache,
        device=model.device,
    )
    state.cfg_text.snapshot(
        cache=None,
        key_values_lens=None,
        packed_key_value_indexes=None,
        next_position_id=torch.tensor(7),
        empty_cache_factory=model._new_empty_cache,
        device=model.device,
    )
    state.cfg_img.snapshot(
        cache=_fake_cache(model, torch.tensor([20.0, 21.0, 22.0])),
        key_values_lens=torch.tensor([3], dtype=torch.int32),
        packed_key_value_indexes=torch.arange(3),
        next_position_id=torch.tensor(11),
        empty_cache_factory=model._new_empty_cache,
        device=model.device,
    )

    inputs = state.preprocess_parallel_denoise_inputs(
        torch.zeros(5, int(model.config.hidden_size)),
        {"cfg_text_scale": 2.0, "cfg_img_scale": 1.5},
        timestep=0.5,
        empty_cache_factory=model._new_empty_cache,
        device=model.device,
        dtype=model.dtype,
    )

    assert inputs["query_lens"].tolist() == [5, 5, 5]
    assert inputs["key_values_lens"].tolist() == [2, 0, 3]
    assert inputs["packed_query_indexes"].tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19]
    assert inputs["packed_key_value_indexes"].tolist() == [0, 1, 12, 13, 14]
    assert inputs["packed_text_indexes"].tolist() == [0, 4, 5, 9, 10, 14]
    assert inputs["packed_vae_token_indexes"].tolist() == [1, 2, 3, 6, 7, 8, 11, 12, 13]
    assert inputs["packed_query_position_ids"].tolist() == [3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11]
    assert inputs["past_key_values"].key_cache[0].reshape(-1).tolist() == [10.0, 11.0, 20.0, 21.0, 22.0]


@pytest.mark.parametrize(
    ("branches", "generation_kwargs", "timestep", "renorm_type"),
    [
        (("main",), {"cfg_text_scale": 1.0, "cfg_img_scale": 1.0}, 0.5, "global"),
        (("main", "cfg_text"), {"cfg_text_scale": 2.0, "cfg_img_scale": 1.0}, 0.5, "global"),
        (("main", "cfg_text", "cfg_img"), {"cfg_text_scale": 2.0, "cfg_img_scale": 1.5}, 0.5, "global"),
        (
            ("main",),
            {"cfg_text_scale": 2.0, "cfg_img_scale": 1.5, "cfg_interval": [0.2, 0.8]},
            0.1,
            "global",
        ),
        (("main", "cfg_text"), {"cfg_text_scale": 2.0, "cfg_img_scale": 1.0}, 0.5, "channel"),
        (("main", "cfg_text", "cfg_img"), {"cfg_text_scale": 2.0, "cfg_img_scale": 1.5}, 0.5, "text_channel"),
    ],
)
def test_bagel_qwen2_mot_collects_stacked_cfg_velocity_like_serial_oracle(
    branches: tuple[str, ...],
    generation_kwargs: dict[str, float],
    timestep: float,
    renorm_type: str,
):
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))
    main = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    cfg_text = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    cfg_img = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
    velocity_by_branch = {"main": main, "cfg_text": cfg_text, "cfg_img": cfg_img}
    generation_kwargs = {
        **generation_kwargs,
        "cfg_interval": generation_kwargs.get("cfg_interval", [0.0, 1.0]),
        "cfg_renorm_type": renorm_type,
        "cfg_renorm_min": 0.0,
    }
    conversation = [
        ConversationItem(
            type="output",
            value=torch.cat([_marker_wrap(velocity_by_branch[branch]) for branch in branches], dim=0),
            role="assistant",
            source=BAGEL_FLOW_VELOCITY,
            meta={"timestep": torch.tensor(timestep)},
        )
    ]
    state = model._generation_state
    state.main.ensure_empty(empty_cache_factory=model._new_empty_cache, device=model.device)
    if "cfg_text" in branches:
        state.cfg_text.ensure_empty(empty_cache_factory=model._new_empty_cache, device=model.device)
    if "cfg_img" in branches:
        state.cfg_img.ensure_empty(empty_cache_factory=model._new_empty_cache, device=model.device)
    state.preprocess_parallel_denoise_inputs(
        torch.zeros(4, int(model.config.hidden_size)),
        generation_kwargs,
        timestep=torch.tensor(timestep),
        empty_cache_factory=model._new_empty_cache,
        device=model.device,
        dtype=model.dtype,
    )
    out = model.collect_velocity(
        conversation_list=conversation,
        generation_kwargs=generation_kwargs,
    )

    expected = _serial_cfg_velocity_oracle(
        main,
        cfg_text if "cfg_text" in branches else None,
        cfg_img if "cfg_img" in branches else None,
        generation_kwargs,
    )

    assert FSM_SIGNAL_KEY not in out
    torch.testing.assert_close(conversation[-1].value, expected)


def _fake_cache(model: nn.Module, values: torch.Tensor):
    cache = model._new_empty_cache()
    cache.key_cache[0] = values.reshape(-1, 1, 1)
    cache.value_cache[0] = (values + 100.0).reshape(-1, 1, 1)
    return cache


def _marker_wrap(velocity: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros_like(velocity[:1]), velocity, torch.zeros_like(velocity[:1])], dim=0)


def _serial_cfg_velocity_oracle(
    main: torch.Tensor,
    cfg_text: torch.Tensor | None,
    cfg_img: torch.Tensor | None,
    generation_kwargs: dict[str, object],
) -> torch.Tensor:
    if cfg_text is None:
        return main

    cfg_text_scale = float(generation_kwargs.get("cfg_text_scale", 1.0))
    cfg_img_scale = float(generation_kwargs.get("cfg_img_scale", 1.0))
    cfg_renorm_min = float(generation_kwargs.get("cfg_renorm_min", 0.0))
    cfg_renorm_type = str(generation_kwargs.get("cfg_renorm_type", "global"))

    guided = cfg_text + cfg_text_scale * (main - cfg_text)
    if cfg_renorm_type == "text_channel":
        scale = (torch.norm(main, dim=-1, keepdim=True) / (torch.norm(guided, dim=-1, keepdim=True) + 1e-8)).clamp(
            min=cfg_renorm_min,
            max=1.0,
        )
        merged = guided * scale
        if cfg_img_scale > 1.0:
            assert cfg_img is not None
            merged = cfg_img + cfg_img_scale * (merged - cfg_img)
        return merged

    if cfg_img_scale > 1.0:
        assert cfg_img is not None
        guided = cfg_img + cfg_img_scale * (guided - cfg_img)

    if cfg_renorm_type == "global":
        norm_main = torch.norm(main)
        norm_guided = torch.norm(guided)
    elif cfg_renorm_type == "channel":
        norm_main = torch.norm(main, dim=-1, keepdim=True)
        norm_guided = torch.norm(guided, dim=-1, keepdim=True)
    else:
        raise NotImplementedError(cfg_renorm_type)
    return guided * (norm_main / (norm_guided + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)


def _fake_cfg_branch_count(generation_kwargs: dict | None) -> int:
    branch_count = 1
    if float((generation_kwargs or {}).get("cfg_text_scale", 1.0)) > 1.0:
        branch_count += 1
    if float((generation_kwargs or {}).get("cfg_img_scale", 1.0)) > 1.0:
        branch_count += 1
    return branch_count


class _NoopBagelSiglip(ModuleMixin, nn.Module):
    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        return {"conversation_list": conversation_list}


class _InferGenTextEncoder(ModuleMixin, nn.Module):
    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        return {"conversation_list": conversation_list}

    def encode_image_markers(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        for item in conversation_list:
            if item.type not in {"image", "output"} or not torch.is_tensor(item.value) or item.value.dim() != 2:
                continue
            item.value = torch.cat([torch.zeros(1, 8), item.value, torch.ones(1, 8)], dim=0)
        return {"conversation_list": conversation_list}


class _InferGenBagelQwen(ModuleMixin, nn.Module):
    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict | None = None,
        **kwargs,
    ):
        del kwargs
        assert conversation_list is not None
        if not conversation_list or conversation_list[-1].type != "output":
            return {"conversation_list": conversation_list}
        tail = conversation_list[-1]
        if tail.source == BAGEL_FLOW_QUERY:
            tail.source = BAGEL_FLOW_HIDDEN
            tail.value = tail.value.repeat(_fake_cfg_branch_count(generation_kwargs), 1)
            return {"conversation_list": conversation_list}
        if tail.source == BAGEL_FLOW_VELOCITY:
            tail.value = torch.zeros(16, 4)
            return {"conversation_list": conversation_list}
        return {"conversation_list": conversation_list}

    def denoise_branch(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict | None = None,
        **kwargs,
    ):
        del kwargs
        assert conversation_list is not None
        tail = conversation_list[-1]
        assert tail.source == BAGEL_FLOW_QUERY
        tail.source = BAGEL_FLOW_HIDDEN
        tail.value = tail.value.repeat(_fake_cfg_branch_count(generation_kwargs), 1)
        return {"conversation_list": conversation_list}

    def collect_velocity(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict | None = None,
        **kwargs,
    ):
        del kwargs
        assert conversation_list is not None
        tail = conversation_list[-1]
        assert tail.source == BAGEL_FLOW_VELOCITY
        assert tail.value.shape[0] == 18 * _fake_cfg_branch_count(generation_kwargs)
        tail.value = torch.zeros(16, 4)
        return {"conversation_list": conversation_list}


class _InferGenBagelFlow(ModuleMixin, nn.Module):
    def prepare_denoise_query(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        if (
            conversation_list
            and conversation_list[-1].type == "output"
            and not torch.is_tensor(conversation_list[-1].value)
        ):
            item = conversation_list[-1]
        else:
            item = conversation_list[-1] if conversation_list and conversation_list[-1].type == "output" else None
        if item is None:
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=torch.zeros(16, 8),
                    role="assistant",
                    source=BAGEL_FLOW_QUERY,
                    meta={"timestep": torch.tensor(0.5)},
                )
            )
        else:
            item.value = torch.zeros(16, 8)
            item.source = BAGEL_FLOW_QUERY
            item.meta = {"timestep": torch.tensor(0.5)}
        return {"conversation_list": conversation_list}

    def decode_velocity_from_hidden(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        item = conversation_list[-1]
        assert item.source == BAGEL_FLOW_HIDDEN
        assert item.value.shape[0] % 18 == 0
        item.value = torch.zeros(item.value.shape[0], 4)
        item.source = BAGEL_FLOW_VELOCITY
        return {"conversation_list": conversation_list}

    def advance_denoise(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        item = conversation_list[-1]
        item.value = torch.zeros(1, 4, 4)
        item.source = BAGEL_GENERATED_LATENT
        item.meta.pop("timestep", None)
        return {"conversation_list": conversation_list, FSM_SIGNAL_KEY: "image_complete"}


class _InferGenBagelVAE(ModuleMixin, nn.Module):
    def decode_generated(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        assert "timestep" not in conversation_list[-1].meta
        return {
            "conversation_list": conversation_list,
            "generated": {"type": "image", "value": Image.new("RGB", (1, 1)), "meta": {}},
        }


class _InferEditBagelVAE(_InferGenBagelVAE):
    def encode_context(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        for item in conversation_list:
            if item.type == "image" and item.source == BAGEL_VAE_CONTEXT:
                item.value = torch.zeros(4, 4, 4)
        return {"conversation_list": conversation_list}


class _InferEditBagelFlow(_InferGenBagelFlow):
    def embed_context_latents(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        for item in conversation_list:
            if item.type == "image" and item.source == BAGEL_VAE_CONTEXT and torch.is_tensor(item.value):
                item.value = torch.zeros(16, 8)
                item.meta.clear()
        return {"conversation_list": conversation_list}
