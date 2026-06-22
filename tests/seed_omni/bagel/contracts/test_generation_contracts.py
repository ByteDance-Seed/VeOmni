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
from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.modeling_omni import OmniModel


def test_bagel_qwen2_mot_gen_mode_requires_token_indexes():
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))

    with pytest.raises(ValueError, match="mode='gen' requires"):
        model._forward_packed_inference(
            packed_query_sequence=torch.randn(1, 64),
            query_lens=torch.tensor([1], dtype=torch.int32),
            packed_query_position_ids=torch.tensor([0], dtype=torch.long),
            packed_query_indexes=torch.tensor([0], dtype=torch.long),
            mode="gen",
        )


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

    assert any("transition: prompt_encode -> denoise_query" in entry for entry in trace)
    assert any("transition: denoise_query -> velocity_collect" in entry for entry in trace)
    assert any("transition: velocity_collect -> denoise_advance" in entry for entry in trace)
    assert any("transition: denoise_advance -> image_decode" in entry for entry in trace)
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
                ConversationItem(type="image", value=Image.new("RGB", (1, 1)), role="user"),
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

    assert any("transition: prompt_encode -> denoise_query" in entry for entry in trace)
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

    assert any(
        "transition: velocity_collect -> denoise_query [module_signal(need_denoise_branch)]" in entry
        for entry in trace
    )
    assert any(
        "transition: velocity_collect -> denoise_advance [module_signal(velocity_ready)]" in entry for entry in trace
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

    branch_transitions = [
        entry
        for entry in trace
        if "transition: velocity_collect -> denoise_query [module_signal(need_denoise_branch)]" in entry
    ]
    assert len(branch_transitions) == 2
    assert any(
        "transition: velocity_collect -> denoise_advance [module_signal(velocity_ready)]" in entry for entry in trace
    )
    assert any(item["type"] == "image" for item in model.generated)
    assert "timestep" not in ctx["conversation_list"][-1].meta


def test_bagel_qwen2_mot_cfg_text_context_snapshot_is_internal():
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))

    model._snapshot_cfg_text_context(
        past_key_values=None,
        key_values_lens=None,
        packed_key_value_indexes=None,
        next_position_id=torch.tensor(7),
    )
    model._generation_state.denoise_branch = "cfg_text"

    cfg_text_context = model._generation_state.cfg_text
    assert cfg_text_context.cache is not None
    assert cfg_text_context.cache_len() == 0
    assert cfg_text_context.position_ids(3, device=model.device).tolist() == [7, 7, 7]
    assert cfg_text_context.key_values_lens.tolist() == [0]
    assert cfg_text_context.packed_key_value_indexes.numel() == 0

    cache = model._new_empty_cache()
    model._snapshot_cfg_text_context(
        past_key_values=cache,
        key_values_lens=torch.tensor([5], dtype=torch.int32),
        packed_key_value_indexes=torch.arange(5),
        next_position_id=torch.tensor(11),
    )

    assert cfg_text_context.cache is not cache
    assert cfg_text_context.cache_len() == 5
    assert cfg_text_context.position_ids(2, device=model.device).tolist() == [11, 11]
    assert cfg_text_context.packed_key_value_indexes.tolist() == [0, 1, 2, 3, 4]


def test_bagel_qwen2_mot_cfg_img_context_accessors_are_internal():
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))

    model._ensure_cfg_img_context()
    model._generation_state.denoise_branch = "cfg_img"

    cfg_img_context = model._generation_state.cfg_img
    assert cfg_img_context.cache is not None
    assert cfg_img_context.cache_len() == 0
    assert cfg_img_context.position_ids(3, device=model.device).tolist() == [0, 0, 0]
    assert cfg_img_context.key_values_lens.tolist() == [0]
    assert cfg_img_context.packed_key_value_indexes.numel() == 0


def test_bagel_qwen2_mot_cfg_img_requires_text_cfg():
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import validate_cfg_request

    with pytest.raises(ValueError, match="cfg_img_scale > 1.0 requires cfg_text_scale > 1.0"):
        validate_cfg_request({"cfg_text_scale": 1.0, "cfg_img_scale": 1.5})


def test_bagel_qwen2_mot_cfg_text_image_velocity_collection_and_merge():
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))
    conversation = [
        ConversationItem(
            type="output",
            value=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            role="assistant",
            meta={"timestep": torch.tensor(0.5)},
        )
    ]
    generation_kwargs = {
        "cfg_text_scale": 2.0,
        "cfg_img_scale": 1.5,
        "cfg_interval": [0.0, 1.0],
        "cfg_renorm_type": "global",
        "cfg_renorm_min": 0.0,
    }

    out = model._collect_or_merge_velocity(
        conversation_list=conversation,
        conversation=conversation,
        generation_kwargs=generation_kwargs,
    )
    assert out[FSM_SIGNAL_KEY] == "need_denoise_branch"
    assert model._generation_state.denoise_branch == "cfg_text"

    conversation[-1].value = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    out = model._collect_or_merge_velocity(
        conversation_list=conversation,
        conversation=conversation,
        generation_kwargs=generation_kwargs,
    )
    assert out[FSM_SIGNAL_KEY] == "need_denoise_branch"
    assert model._generation_state.denoise_branch == "cfg_img"

    conversation[-1].value = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
    out = model._collect_or_merge_velocity(
        conversation_list=conversation,
        conversation=conversation,
        generation_kwargs=generation_kwargs,
    )

    main = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    cfg_text = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    cfg_img = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
    text_guided = cfg_text + 2.0 * (main - cfg_text)
    guided = cfg_img + 1.5 * (text_guided - cfg_img)
    expected = guided * (torch.norm(main) / (torch.norm(guided) + 1e-8)).clamp(min=0.0, max=1.0)

    assert out[FSM_SIGNAL_KEY] == "velocity_ready"
    assert model._generation_state.denoise_branch == "main"
    assert model._generation_state.velocity_buffer == {}
    torch.testing.assert_close(conversation[-1].value, expected)


class _NoopBagelSiglip(nn.Module):
    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        return {"conversation_list": conversation_list}


class _InferGenTextEncoder(nn.Module):
    def prompt_encode(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        return {"conversation_list": conversation_list}

    def encode_image_query_markers(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        for item in conversation_list:
            if item.type not in {"image", "output"} or not torch.is_tensor(item.value) or item.value.dim() != 2:
                continue
            item.value = torch.cat([torch.zeros(1, 8), item.value, torch.ones(1, 8)], dim=0)
        return {"conversation_list": conversation_list}


class _InferGenBagelQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity_collect_count = 0

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
        if torch.is_tensor(tail.value) and tail.value.dim() == 2 and tail.value.shape[-1] == 4:
            branch_count = 0
            if float((generation_kwargs or {}).get("cfg_text_scale", 1.0)) > 1.0:
                branch_count += 1
            if float((generation_kwargs or {}).get("cfg_img_scale", 1.0)) > 1.0:
                branch_count += 1
            if self.velocity_collect_count < branch_count:
                self.velocity_collect_count += 1
                tail.value = None
                return {"conversation_list": conversation_list, FSM_SIGNAL_KEY: "need_denoise_branch"}
            return {"conversation_list": conversation_list, FSM_SIGNAL_KEY: "velocity_ready"}
        return {"conversation_list": conversation_list}


class _InferGenBagelFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = "prepare"

    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        if self.phase == "advance" and not torch.is_tensor(conversation_list[-1].value):
            self.phase = "prepare"
        if self.phase == "prepare":
            item = conversation_list[-1] if conversation_list and conversation_list[-1].type == "output" else None
            if item is None:
                conversation_list.append(
                    ConversationItem(
                        type="output",
                        value=torch.zeros(16, 8),
                        role="assistant",
                        meta={"timestep": torch.tensor(0.5)},
                    )
                )
            else:
                item.value = torch.zeros(16, 8)
                item.meta = {"timestep": torch.tensor(0.5)}
            self.phase = "decode"
            return {"conversation_list": conversation_list}
        if self.phase == "decode":
            item = conversation_list[-1]
            assert item.value.shape == (18, 8)
            item.value = torch.zeros(16, 4)
            self.phase = "advance"
            return {"conversation_list": conversation_list}
        item = conversation_list[-1]
        item.value = torch.zeros(1, 4, 4)
        item.meta.pop("timestep", None)
        self.phase = "done"
        return {"conversation_list": conversation_list, FSM_SIGNAL_KEY: "image_complete"}


class _InferGenBagelVAE(nn.Module):
    def decode(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        assert "timestep" not in conversation_list[-1].meta
        return {
            "conversation_list": conversation_list,
            "generated": {"type": "image", "value": Image.new("RGB", (1, 1)), "meta": {}},
        }


class _InferEditBagelVAE(_InferGenBagelVAE):
    def encode(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        for index, item in enumerate(conversation_list):
            if item.type == "image" and item.role == "user":
                conversation_list.insert(
                    index,
                    ConversationItem(type="output", value=torch.zeros(4, 4, 4), role="assistant", meta={}),
                )
                break
        return {"conversation_list": conversation_list}


class _InferEditBagelFlow(_InferGenBagelFlow):
    def embed_latent(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        for item in conversation_list:
            if item.type == "output" and torch.is_tensor(item.value) and item.value.dim() == 3:
                item.value = torch.zeros(16, 8)
                item.meta.clear()
        return {"conversation_list": conversation_list}
