"""BAGEL generation graph smoke and eager/accelerated denoise structure."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.seed_omni.modules.bagel.helpers import (
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


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _make_veomni_runtime(cfg, modules):
    # The stubs implement generation endpoints only; OmniModel checks every graph's methods.
    cfg.training_graphs = {}
    model = OmniModel(cfg, modules).eval()
    return OmniModelRuntime(model), model


def test_bagel_text_encoder_injects_raw_bos_for_understanding() -> None:
    """Verify ``BagelTextEncoder.generate`` appends an embedded BOS output item on the first understanding step."""
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
    """Verify ``BagelQwen2MoT.generate`` prefills the prompt on ``prompt_encode`` and decodes the tail BOS token on ``text_ar``."""
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
        del generation_kwargs
        calls.append(("prefill", list(conversation_list)))
        self._generation_state.main.install_cache(
            cache=self._new_empty_cache(),
            cache_len=2,
            next_position_id=torch.tensor(2),
            device=self.device,
        )
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
    assert calls == [("prefill", [prompt])]
    assert conversation[-1].meta.get(BAGEL_PHASE_KEY) == BAGEL_START_TOKEN
    torch.testing.assert_close(bos.value, torch.full((1, hidden_size), 2.0))

    conversation = model.generate(
        conversation,
        generation_kwargs={"infer_type": "infer_und"},
    )["conversation_list"]
    assert calls == [("prefill", [prompt]), ("decode", [prompt, bos])]
    assert conversation == [prompt, bos]
    assert BAGEL_PHASE_KEY not in bos.meta
    torch.testing.assert_close(bos.value, torch.full((1, hidden_size), 4.0))


def test_bagel_infer_gen_denoise_signal_smoke():
    """Verify the ``infer_gen`` FSM transitions through ``prompt_encode``, ``query_denoise``, ``velocity_collect``, and ``image_decode``."""
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
    """Verify ``infer_gen`` encodes user images through SigLIP without creating VAE context items."""
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
    """Verify ``infer_edit`` encodes both VAE and SigLIP context and runs the denoise loop to completion."""
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
    """Verify eager ``denoise_branch`` runs one serial ``forward_inference`` call per active CFG branch."""
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
    """Verify accelerated ``denoise_branch`` stacks all active CFG branches into one packed ``forward_inference`` call."""
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
    """Return the number of active CFG branches implied by ``generation_kwargs``."""
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

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        return attn_implementation

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


def test_bagel_native_omni_model_and_processor_propagate_resolved_infer_type_without_duplication(tmp_path):
    """Verify native ``OmniProcessor`` and ``OmniModel.generate`` propagate ``config.infer_type`` to preprocessing and module calls without caller duplication."""
    from tests.seed_omni.modules.bagel.test_processing import _bagel_template, _FakeBagelTokenizer
    from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
        BagelSiglipNavitPreprocessor,
        BagelSiglipNavitProcessor,
    )
    from veomni.models.seed_omni.modules.bagel.text_encoder.processing import BagelTextEncoderPreprocessor
    from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEPreprocessor, BagelVAEProcessor
    from veomni.models.seed_omni.processing_omni import OmniProcessor

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(str(image_path))

    def _build_preprocessors():
        return {
            "bagel_vae": BagelVAEPreprocessor(
                BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
                dtype=torch.float32,
            ),
            "bagel_siglip_navit": BagelSiglipNavitPreprocessor(
                BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
                dtype=torch.float32,
            ),
            "bagel_text_encoder": BagelTextEncoderPreprocessor(_bagel_template()),
        }

    # 1. Default converted checkpoint scenario (infer_und): real BagelTextEncoder + BagelQwen2MoT
    und_cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "train/modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "train/graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / "infer/graph_infer_und.yaml",
    )
    und_cfg.training_graphs = {}
    assert und_cfg.infer_type == "infer_und"

    BagelTextEncoder = native_model_cls("bagel_text_encoder")
    BagelTextEncoderConfig = config_cls("bagel_text_encoder")
    BagelQwen2MoT = native_model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")

    qwen_kwargs = {
        **tiny_bagel_qwen2_cfg(),
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
    }
    hidden_size = int(qwen_kwargs["hidden_size"])

    def _make_real_text_and_mot():
        text_enc = BagelTextEncoder(
            BagelTextEncoderConfig(vocab_size=int(qwen_kwargs["vocab_size"]), hidden_size=hidden_size)
        ).eval()
        text_enc._tokenizer = _FakeBagelTokenizer()
        text_enc._chat_template = _bagel_template()
        text_enc._sample_token = lambda hidden_states, **kwargs: text_enc._chat_template.eos_token_id
        mot = BagelQwen2MoT(BagelQwen2MoTConfig(**qwen_kwargs)).eval()
        return text_enc, mot

    class _DimAlignedSiglip(_NoopBagelSiglip):
        def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
            del kwargs
            assert conversation_list is not None
            for item in conversation_list:
                if item.type == "image" and item.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_SIGLIP_CONTEXT:
                    item.value = torch.zeros(4, hidden_size)
            return {"conversation_list": conversation_list}

    und_text_encoder, und_qwen_mot = _make_real_text_and_mot()
    und_model = OmniModel(
        und_cfg,
        {
            "bagel_text_encoder": und_text_encoder,
            "bagel_siglip_navit": _DimAlignedSiglip(),
            "bagel_qwen2_mot": und_qwen_mot,
            "bagel_flow_connector": _InferGenBagelFlow(),
            "bagel_vae": _InferGenBagelVAE(),
        },
    ).eval()
    und_processor = OmniProcessor(_build_preprocessors(), infer_type=und_model.config.infer_type)

    # Caller passes neither infer_type to processor(...) nor infer_type in generation_kwargs
    und_request = und_processor(text="hi", images=str(image_path))
    assert [item.meta.get(BAGEL_CONTEXT_KEY) for item in und_request["conversation_list"] if item.type == "image"] == [
        BAGEL_SIGLIP_CONTEXT
    ]
    und_model.reset()
    und_generated = und_model.generate(und_request, generation_kwargs={"max_new_tokens": 4})
    assert any(item["type"] == "text" for item in und_generated)
    assert und_qwen_mot._generation_state.main.cache is not None

    # 2. Image edit scenario (infer_edit): processor routes both VAE + SigLIP and real text_encoder + qwen_mot run edit FSM
    edit_cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "train/modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "train/graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / "infer/graph_infer_edit.yaml",
    )
    edit_cfg.training_graphs = {}
    assert edit_cfg.infer_type == "infer_edit"

    edit_text_encoder, edit_qwen_mot = _make_real_text_and_mot()
    edit_model = OmniModel(
        edit_cfg,
        {
            "bagel_text_encoder": edit_text_encoder,
            "bagel_siglip_navit": _DimAlignedSiglip(),
            "bagel_qwen2_mot": edit_qwen_mot,
            "bagel_flow_connector": _InferEditBagelFlow(),
            "bagel_vae": _InferEditBagelVAE(),
        },
    ).eval()
    edit_processor = OmniProcessor(_build_preprocessors(), infer_type=edit_model.config.infer_type)

    # Caller passes neither infer_type to processor(...) nor infer_type in generation_kwargs
    edit_request = edit_processor(text="edit", images=str(image_path))
    assert [
        item.meta.get(BAGEL_CONTEXT_KEY) for item in edit_request["conversation_list"] if item.type == "image"
    ] == [
        BAGEL_VAE_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
    ]
    edit_model.reset()
    edit_generated = edit_model.generate(
        edit_request,
        generation_kwargs={"max_new_tokens": 8, "do_sample": False, "image_height": 64, "image_width": 64},
    )
    assert any(item["type"] == "image" for item in edit_generated)
    assert edit_qwen_mot._generation_state.main.cache is not None
