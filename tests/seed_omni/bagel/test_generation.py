"""BAGEL generation graph smoke and eager/accelerated denoise structure."""

from __future__ import annotations

from types import SimpleNamespace

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
from veomni.models.seed_omni.accelerated import OmniModelRuntime
from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.mixins.inference_module_mixin import InferenceModuleMixin
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_CONTEXT_KEY,
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_FLOW_VELOCITY,
    BAGEL_GENERATED_LATENT,
    BAGEL_PHASE_KEY,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_START_TOKEN,
    BAGEL_VAE_CONTEXT,
)
from veomni.models.seed_omni.modules.module_configuration_base import OmniModuleConfig
from veomni.models.seed_omni.modules.module_modeling_base import PretrainedOmniModule
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.seed_omni.utils.graph_profiler import GraphProfiler


def _make_veomni_runtime(cfg, modules):
    # The stubs implement generation endpoints only; OmniModel checks every graph's methods.
    cfg.training_graphs = {}
    model = OmniModel(cfg, modules).eval()
    return OmniModelRuntime(model), model


def test_bagel_text_encoder_injects_raw_bos_for_understanding() -> None:
    BagelTextEncoder = native_model_cls("bagel_text_encoder")
    BagelTextEncoderConfig = config_cls("bagel_text_encoder")
    model = BagelTextEncoder(BagelTextEncoderConfig(vocab_size=8, hidden_size=4)).eval()
    model._chat_template = SimpleNamespace(bos_token_id=3)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))

    prompt = ConversationItem(
        type="text",
        value=torch.tensor([1, 2]),
        role="user",
        meta={"_omni_tokenized": True, "input_ids": torch.tensor([1, 2])},
    )
    conversation = model.generate([prompt], generation_kwargs={"infer_type": "infer_und"})["conversation_list"]

    assert len(conversation) == 2
    assert conversation[-1].type == "output"
    assert conversation[-1].meta[BAGEL_PHASE_KEY] == BAGEL_START_TOKEN
    assert conversation[-1].meta["input_ids"].tolist() == [3]
    torch.testing.assert_close(conversation[-1].value, model.embed_tokens.weight[3].reshape(1, 4))


def test_bagel_qwen_first_understanding_step_prefills_prompt_then_decodes_bos(monkeypatch) -> None:
    BagelQwen2MoT = native_model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg())).eval()
    hidden_size = int(model.config.hidden_size)
    prompt = ConversationItem(type="text", value=torch.ones(2, hidden_size), role="user")
    bos = ConversationItem(
        type="output",
        value=torch.full((1, hidden_size), 2.0),
        role="assistant",
        meta={BAGEL_PHASE_KEY: BAGEL_START_TOKEN},
    )
    calls: list[tuple[str, list[ConversationItem]]] = []

    def _prefill(self, conversation_list, generation_kwargs):
        del self, generation_kwargs
        calls.append(("prefill", list(conversation_list)))
        return torch.full((2, hidden_size), 3.0)

    def _decode(self, conversation_list):
        del self
        calls.append(("decode", list(conversation_list)))
        return torch.full((1, hidden_size), 4.0)

    monkeypatch.setattr(type(model), "_prefill_prompt", _prefill)
    monkeypatch.setattr(type(model), "_decode_next_token", _decode)

    conversation = model.generate(
        [prompt, bos],
        generation_kwargs={"infer_type": "infer_und"},
    )["conversation_list"]

    assert calls == [("prefill", [prompt]), ("decode", [prompt, bos])]
    assert conversation == [prompt, bos]
    assert BAGEL_PHASE_KEY not in bos.meta
    torch.testing.assert_close(bos.value, torch.full((1, hidden_size), 4.0))


def test_bagel_infer_gen_denoise_signal_smoke():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "train/modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "train/graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / "infer/graph_infer_gen.yaml",
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
        modules_path=bagel_cfg_dir() / "train/modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "train/graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / "infer/graph_infer_gen.yaml",
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
                meta={BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT},
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
    assert all(item.meta.get(BAGEL_CONTEXT_KEY) != BAGEL_VAE_CONTEXT for item in request["conversation_list"])
    assert torch.equal(request["conversation_list"][0].value[0], torch.zeros(8))
    assert torch.equal(request["conversation_list"][0].value[1:], torch.ones(3, 8))
    assert any(item["type"] == "image" for item in generated)


def test_bagel_infer_edit_defaults_to_denoise_signal_smoke():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "train/modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "train/graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / "infer/graph_infer_edit.yaml",
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
                meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT},
            ),
            ConversationItem(
                type="image",
                value=Image.new("RGB", (1, 1)),
                role="user",
                meta={BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT},
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
        meta={BAGEL_PHASE_KEY: BAGEL_FLOW_QUERY, "timestep": 0.5},
    )
    model.denoise_branch([tail], generation_kwargs={"cfg_text_scale": 2.0, "cfg_img_scale": 1.5})

    assert len(calls) == 3
    assert tail.meta.get(BAGEL_PHASE_KEY) == BAGEL_FLOW_HIDDEN
    assert int(tail.value.shape[0]) == 15
    assert calls[0]["query_lens"].tolist() == [5]
    assert calls[1]["query_lens"].tolist() == [5]
    assert calls[2]["query_lens"].tolist() == [5]


def test_bagel_qwen2_mot_accelerated_denoise_branch_packs_cfg_branches(monkeypatch):
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated.accelerated import BagelQwen2MoTAccelerated

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
        meta={BAGEL_PHASE_KEY: BAGEL_FLOW_QUERY, "timestep": 0.5},
    )
    model.denoise_branch([tail], generation_kwargs={"cfg_text_scale": 2.0, "cfg_img_scale": 1.5})

    assert captured["query_lens"].tolist() == [5, 5, 5]
    assert int(captured["packed_query_sequence"].shape[0]) == 15
    assert tail.meta.get(BAGEL_PHASE_KEY) == BAGEL_FLOW_HIDDEN
    assert int(tail.value.shape[0]) == 15


def _fake_cfg_branch_count(generation_kwargs: dict | None) -> int:
    branch_count = 1
    if float((generation_kwargs or {}).get("cfg_text_scale", 1.0)) > 1.0:
        branch_count += 1
    if float((generation_kwargs or {}).get("cfg_img_scale", 1.0)) > 1.0:
        branch_count += 1
    return branch_count


class _StubConfig(OmniModuleConfig):
    model_type = "bagel_generation_stub"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _StubModule(PretrainedOmniModule, BaseMixin, InferenceModuleMixin):
    """Weightless graph participant: ``OmniModel`` only takes ``PretrainedOmniModule``."""

    config_class = _StubConfig

    def __init__(self):
        super().__init__(_StubConfig())


class _NoopBagelSiglip(_StubModule):
    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        return {"conversation_list": conversation_list}


class _CountingInferGenBagelSiglip(_StubModule):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        self.calls += 1
        assert not any(
            item.type == "image" and item.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_VAE_CONTEXT
            for item in conversation_list
        )
        for item in conversation_list:
            if item.type == "image" and item.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_SIGLIP_CONTEXT:
                item.value = torch.ones(2, 8)
        return {"conversation_list": conversation_list}


class _InferGenTextEncoder(_StubModule):
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


class _InferGenBagelQwen(_StubModule):
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
        if tail.meta.get(BAGEL_PHASE_KEY) == BAGEL_FLOW_QUERY:
            tail.meta[BAGEL_PHASE_KEY] = BAGEL_FLOW_HIDDEN
            tail.value = tail.value.repeat(_fake_cfg_branch_count(generation_kwargs), 1)
            return {"conversation_list": conversation_list}
        if tail.meta.get(BAGEL_PHASE_KEY) == BAGEL_FLOW_VELOCITY:
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
        assert tail.meta.get(BAGEL_PHASE_KEY) == BAGEL_FLOW_QUERY
        tail.meta[BAGEL_PHASE_KEY] = BAGEL_FLOW_HIDDEN
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
        assert tail.meta.get(BAGEL_PHASE_KEY) == BAGEL_FLOW_VELOCITY
        assert tail.value.shape[0] == 18 * _fake_cfg_branch_count(generation_kwargs)
        tail.value = torch.zeros(16, 4)
        return {"conversation_list": conversation_list}


class _InferGenBagelFlow(_StubModule):
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
                    meta={BAGEL_PHASE_KEY: BAGEL_FLOW_QUERY, "timestep": torch.tensor(0.5)},
                )
            )
        else:
            item.value = torch.zeros(16, 8)
            item.meta = {BAGEL_PHASE_KEY: BAGEL_FLOW_QUERY, "timestep": torch.tensor(0.5)}
        return {"conversation_list": conversation_list}

    def decode_velocity_from_hidden(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        item = conversation_list[-1]
        assert item.meta.get(BAGEL_PHASE_KEY) == BAGEL_FLOW_HIDDEN
        assert item.value.shape[0] % 18 == 0
        item.value = torch.zeros(item.value.shape[0], 4)
        item.meta[BAGEL_PHASE_KEY] = BAGEL_FLOW_VELOCITY
        return {"conversation_list": conversation_list}

    def advance_denoise(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        item = conversation_list[-1]
        item.value = torch.zeros(1, 4, 4)
        item.meta[BAGEL_PHASE_KEY] = BAGEL_GENERATED_LATENT
        item.meta.pop("timestep", None)
        return {"conversation_list": conversation_list, FSM_SIGNAL_KEY: "image_complete"}


class _InferGenBagelVAE(_StubModule):
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
            if item.type == "image" and item.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_VAE_CONTEXT:
                item.value = torch.zeros(4, 4, 4)
        return {"conversation_list": conversation_list}


class _InferEditBagelFlow(_InferGenBagelFlow):
    def embed_context_latents(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        for item in conversation_list:
            if (
                item.type == "image"
                and item.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_VAE_CONTEXT
                and torch.is_tensor(item.value)
            ):
                item.value = torch.zeros(16, 8)
                item.meta = {BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT}
        return {"conversation_list": conversation_list}
