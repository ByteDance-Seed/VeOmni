"""BAGEL generation graph smoke and eager/accelerated denoise structure."""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image

from tests.seed_omni.bagel.helpers import (
    bagel_cfg_dir,
    config_cls,
    load_omni_config,
    native_model_cls,
    tiny_bagel_qwen2_cfg,
)
from veomni.models.seed_omni.accelerator import OmniModelRuntime
from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.mixins.inference_module_mixin import InferenceModuleMixin
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_FLOW_VELOCITY,
    BAGEL_GENERATED_LATENT,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_VAE_CONTEXT,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.seed_omni.utils.graph_profiler import GraphProfiler


def _make_veomni_runtime(cfg, modules):
    model = OmniModel(cfg, modules).eval()
    return OmniModelRuntime(model), model


def test_bagel_infer_gen_denoise_signal_smoke():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / "graph_infer_gen.yaml",
    )
    runtime, _model = _make_veomni_runtime(
        cfg,
        {
            "bagel_text_encoder": _InferGenTextEncoder(),
            "bagel_siglip_navit": _NoopBagelSiglip(),
            "bagel_qwen2_mot": _InferGenBagelQwen(),
            "bagel_flow_connector": _InferGenBagelFlow(),
            "bagel_vae": _InferGenBagelVAE(),
        },
    )
    profiler = GraphProfiler()
    request = {"conversation_list": [ConversationItem(type="text", value="prompt", role="user")]}
    generated = runtime.generate(
        request,
        profiler=profiler,
        generation_kwargs={
            "max_new_tokens": 8,
            "do_sample": False,
            "image_height": 64,
            "image_width": 64,
        },
    )

    trace = profiler.save_records()
    assert any("transition: prompt_encode -> query_denoise" in entry for entry in trace)
    assert any("transition: query_denoise -> velocity_collect" in entry for entry in trace)
    assert any("transition: velocity_collect -> image_decode" in entry for entry in trace)
    assert any("transition: image_decode -> done" in entry for entry in trace)
    assert any(item["type"] == "image" for item in generated)
    assert "timestep" not in request["conversation_list"][-1].meta


def test_bagel_infer_gen_user_image_runs_siglip_context_only():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / "graph_infer_gen.yaml",
    )
    siglip = _CountingInferGenBagelSiglip()
    runtime, _model = _make_veomni_runtime(
        cfg,
        {
            "bagel_text_encoder": _InferGenTextEncoder(),
            "bagel_siglip_navit": siglip,
            "bagel_qwen2_mot": _InferGenBagelQwen(),
            "bagel_flow_connector": _InferGenBagelFlow(),
            "bagel_vae": _InferGenBagelVAE(),
        },
    )
    request = {
        "conversation_list": [
            ConversationItem(
                type="image",
                value=Image.new("RGB", (1, 1)),
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            ),
            ConversationItem(type="text", value="prompt", role="user"),
        ]
    }
    generated = runtime.generate(
        request,
        generation_kwargs={
            "max_new_tokens": 8,
            "do_sample": False,
            "image_height": 64,
            "image_width": 64,
        },
    )

    assert siglip.calls == 1
    assert all(item.source != BAGEL_VAE_CONTEXT for item in request["conversation_list"])
    assert torch.equal(request["conversation_list"][0].value[0], torch.zeros(8))
    assert torch.equal(request["conversation_list"][0].value[1:], torch.ones(3, 8))
    assert any(item["type"] == "image" for item in generated)


def test_bagel_infer_edit_defaults_to_denoise_signal_smoke():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / "graph_infer_edit.yaml",
    )
    runtime, _model = _make_veomni_runtime(
        cfg,
        {
            "bagel_text_encoder": _InferGenTextEncoder(),
            "bagel_siglip_navit": _NoopBagelSiglip(),
            "bagel_qwen2_mot": _InferGenBagelQwen(),
            "bagel_flow_connector": _InferEditBagelFlow(),
            "bagel_vae": _InferEditBagelVAE(),
        },
    )
    profiler = GraphProfiler()
    request = {
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
    }
    generated = runtime.generate(
        request,
        profiler=profiler,
        generation_kwargs={
            "max_new_tokens": 8,
            "do_sample": False,
            "image_height": 64,
            "image_width": 64,
        },
    )

    trace = profiler.save_records()
    assert any("transition: prompt_encode -> query_denoise" in entry for entry in trace)
    assert not any("transition: prompt_encode -> text_ar" in entry for entry in trace)
    assert any("transition: image_decode -> done" in entry for entry in trace)
    assert any(item["type"] == "image" for item in generated)
    assert "timestep" not in request["conversation_list"][-1].meta


def _fake_cache(model: nn.Module, values: torch.Tensor):
    cache = model._new_empty_cache()
    cache.key_cache[0] = values.reshape(-1, 1, 1)
    cache.value_cache[0] = (values + 100.0).reshape(-1, 1, 1)
    return cache


def _install_three_branch_caches(model: nn.Module) -> None:
    state = model._generation_state
    state.main.install_cache(
        cache=_fake_cache(model, torch.tensor([10.0, 11.0])),
        cache_len=2,
        next_position_id=torch.tensor(3),
        device=model.device,
    )
    state.cfg_text.install_cache(
        cache=model._new_empty_cache(),
        cache_len=0,
        next_position_id=torch.tensor(7),
        device=model.device,
    )
    state.cfg_img.install_cache(
        cache=_fake_cache(model, torch.tensor([20.0, 21.0, 22.0])),
        cache_len=3,
        next_position_id=torch.tensor(11),
        device=model.device,
    )


def test_bagel_qwen2_mot_eager_denoise_branch_runs_serial_forward_inference(monkeypatch):
    BagelQwen2MoT = native_model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))
    _install_three_branch_caches(model)
    query = torch.zeros(5, int(model.config.hidden_size))
    calls: list[dict[str, object]] = []

    def _capture_forward_inference(self, **kwargs):
        del self
        calls.append(kwargs)
        return {"hidden_states": kwargs["packed_query_sequence"]}

    monkeypatch.setattr(type(model), "forward_inference", _capture_forward_inference)
    tail = ConversationItem(
        type="output",
        value=query,
        role="assistant",
        source=BAGEL_FLOW_QUERY,
        meta={"timestep": 0.5},
    )
    model.denoise_branch([tail], generation_kwargs={"cfg_text_scale": 2.0, "cfg_img_scale": 1.5})

    assert len(calls) == 3
    assert tail.source == BAGEL_FLOW_HIDDEN
    assert int(tail.value.shape[0]) == 15
    assert calls[0]["query_lens"].tolist() == [5]
    assert calls[1]["query_lens"].tolist() == [5]
    assert calls[2]["query_lens"].tolist() == [5]


def test_bagel_qwen2_mot_accelerated_denoise_branch_packs_cfg_branches(monkeypatch):
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import BagelQwen2MoTAccelerated

    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoTAccelerated(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))
    _install_three_branch_caches(model)
    query = torch.zeros(5, int(model.config.hidden_size))
    captured: dict[str, object] = {}

    def _capture_forward_inference(self, **kwargs):
        del self
        captured.update(kwargs)
        return {"hidden_states": kwargs["packed_query_sequence"]}

    monkeypatch.setattr(type(model), "forward_inference", _capture_forward_inference)
    tail = ConversationItem(
        type="output",
        value=query,
        role="assistant",
        source=BAGEL_FLOW_QUERY,
        meta={"timestep": 0.5},
    )
    model.denoise_branch([tail], generation_kwargs={"cfg_text_scale": 2.0, "cfg_img_scale": 1.5})

    assert captured["query_lens"].tolist() == [5, 5, 5]
    assert int(captured["packed_query_sequence"].shape[0]) == 15
    assert tail.source == BAGEL_FLOW_HIDDEN
    assert int(tail.value.shape[0]) == 15


def _fake_cfg_branch_count(generation_kwargs: dict | None) -> int:
    branch_count = 1
    if float((generation_kwargs or {}).get("cfg_text_scale", 1.0)) > 1.0:
        branch_count += 1
    if float((generation_kwargs or {}).get("cfg_img_scale", 1.0)) > 1.0:
        branch_count += 1
    return branch_count


class _NoopBagelSiglip(BaseMixin, InferenceModuleMixin, nn.Module):
    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        return {"conversation_list": conversation_list}


class _CountingInferGenBagelSiglip(BaseMixin, InferenceModuleMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        self.calls += 1
        assert not any(item.type == "image" and item.source == BAGEL_VAE_CONTEXT for item in conversation_list)
        for item in conversation_list:
            if item.type == "image" and item.source == BAGEL_SIGLIP_CONTEXT:
                item.value = torch.ones(2, 8)
        return {"conversation_list": conversation_list}


class _InferGenTextEncoder(BaseMixin, InferenceModuleMixin, nn.Module):
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


class _InferGenBagelQwen(BaseMixin, InferenceModuleMixin, nn.Module):
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


class _InferGenBagelFlow(BaseMixin, InferenceModuleMixin, nn.Module):
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


class _InferGenBagelVAE(BaseMixin, InferenceModuleMixin, nn.Module):
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
