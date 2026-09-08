"""CPU unit tests for Janus packed tokens/masks and model-level FSDP config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from veomni.arguments.arguments_types import FSDPConfig
from veomni.models.seed_omni.graphs.training_graph import TrainingGraph
from veomni.models.seed_omni.modules.janus.packing import (
    GEN_IMAGE_MASK,
    PACKED_INPUT_IDS,
    PACKED_LABELS,
    PIXEL_VALUES_UND,
    UND_IMAGE_MASK,
    UND_NUM_REAL,
    fold_dummy_anchor,
    masked_scatter_embeds,
    pack_janus_conversations,
    shift_packed_labels,
    teacher_force_vq_hidden,
)
from veomni.models.seed_omni.modules.janus.text_encoder.chat_template import JanusChatTemplate
from veomni.models.seed_omni.modules.janus.text_encoder.processing import JanusTextEncoderPreprocessor
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.constants import IGNORE_INDEX


def _janus_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "Janus" / "janus_1.3b"


class _FakeTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    boi_token = "<boi>"
    eoi_token = "<eoi>"
    bos_token_id = 1
    eos_token_id = 2
    boi_token_id = 3
    eoi_token_id = 4
    pad_token_id = 0
    special_tokens = {"<s>": 1, "</s>": 2, "<boi>": 3, "<eoi>": 4}

    def convert_tokens_to_ids(self, token):
        return self.special_tokens.get(token, -1)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        index = 0
        while index < len(text):
            matched = False
            for token, token_id in sorted(self.special_tokens.items(), key=lambda item: len(item[0]), reverse=True):
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(ord(text[index]) % 50 + 10)
                index += 1
        return {"input_ids": ids}


def test_fsdp_scope_defaults_to_module():
    assert FSDPConfig().fsdp_scope == "module"


def test_packed_train_graph_is_a_dag_of_pack_nodes():
    spec = yaml.safe_load((_janus_cfg_dir() / "packed/graph_train.yaml").read_text())
    graph = TrainingGraph(spec)
    names = [node.name for node in graph.iter_nodes()]
    assert names[0] == "janus_text_encoder.pack_encode"
    assert "janus_siglip.pack_encode" in names
    assert "janus_vqvae.pack_encode" in names
    assert "janus_llama.pack_forward" in names
    assert "janus_text_encoder.pack_decode" in names
    assert "janus_vqvae.pack_decode" in names


def test_pack_janus_conversations_expands_images_and_skips_dummies():
    und = torch.ones(3, 4, 4)
    dummy = torch.zeros(3, 4, 4)
    sample = [
        ConversationItem(
            type="text",
            value=torch.tensor([1, 2, 3]),
            role="user",
            meta={"labels": torch.tensor([-100, -100, -100])},
        ),
        ConversationItem(type="image", value=und, role="user", source="janus_siglip"),
        ConversationItem(
            type="text",
            value=torch.tensor([4, 5]),
            role="assistant",
            meta={"labels": torch.tensor([4, 5])},
        ),
        ConversationItem(type="image", value=dummy, role="dummy", source="janus_vqvae"),
    ]
    packed = pack_janus_conversations([sample], pad_token_id=0, num_image_tokens=4)
    # 3 text + 4 image + 2 text = 9 (dummy image is not in the sequence)
    assert packed[PACKED_INPUT_IDS].shape[-1] == 9
    assert int(packed[UND_IMAGE_MASK].sum()) == 4
    assert int(packed[GEN_IMAGE_MASK].sum()) == 0
    assert packed[UND_NUM_REAL] == 1
    assert packed[PIXEL_VALUES_UND].shape[0] == 1  # real und only; gen dummy is on gen stack
    assert packed[PACKED_LABELS][0, 3:7].tolist() == [IGNORE_INDEX] * 4


def test_masked_scatter_and_teacher_force_vq():
    packed = torch.zeros(1, 6, 2)
    mask = torch.tensor([[False, False, True, True, False, False]])
    embeds = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    out = masked_scatter_embeds(packed, mask, embeds)
    assert out[0, 2].tolist() == [1.0, 1.0]
    assert out[0, 3].tolist() == [2.0, 2.0]
    hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    selected = teacher_force_vq_hidden(hidden, mask)
    # tokens at positions 2,3 are gen-image; predictors are hidden[1] and hidden[2]
    assert selected.reshape(-1).tolist() == [1.0, 2.0]


def test_fold_dummy_anchor_is_zero_valued_but_connected():
    target = torch.ones(2, 3, requires_grad=True)
    dummy = torch.full((1, 3), 4.0, requires_grad=True)
    out = fold_dummy_anchor(target, dummy)
    assert torch.equal(out, target)
    out.sum().backward()
    assert dummy.grad is not None
    assert torch.count_nonzero(dummy.grad) == 0


def test_shift_packed_labels_pads_ignore():
    labels = torch.tensor([[1, 2, 3, IGNORE_INDEX]])
    shifted = shift_packed_labels(labels)
    assert shifted.tolist() == [[2, 3, IGNORE_INDEX, IGNORE_INDEX]]


def test_janus_packed_preprocessor_writes_batch_keys():
    preprocessor = JanusTextEncoderPreprocessor(
        JanusChatTemplate(_FakeTokenizer()), packed_preprocess=True, num_image_tokens=4
    )
    pixels = torch.ones(3, 4, 4)
    conversation = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="image", value=pixels, role="user", source="janus_siglip"),
            ConversationItem(type="text", value="ok", role="assistant"),
            ConversationItem(type="image", value=torch.zeros(3, 4, 4), role="dummy", source="janus_vqvae"),
        ]
    ]
    batch = {"conversation_list": conversation}
    preprocessor(batch, inference=False)
    assert PACKED_INPUT_IDS in batch
    assert UND_IMAGE_MASK in batch
    assert batch[UND_NUM_REAL] == 1
    # The carrier is dropped once packed: its per-item pixel tensors are already
    # copied into pixel_values_und/gen, so keeping it would ship them twice.
    assert "conversation_list" not in batch


def test_janus_packed_preprocessor_keeps_conversation_list_on_inference():
    preprocessor = JanusTextEncoderPreprocessor(
        JanusChatTemplate(_FakeTokenizer()), packed_preprocess=True, num_image_tokens=4
    )
    conversation = [[ConversationItem(type="text", value="hi", role="user")]]
    batch = {"conversation_list": conversation}
    preprocessor(batch, inference=True)
    assert batch["conversation_list"] is conversation


def test_janus_packed_preprocessor_skips_pack_on_inference():
    preprocessor = JanusTextEncoderPreprocessor(
        JanusChatTemplate(_FakeTokenizer()), packed_preprocess=True, num_image_tokens=4
    )
    conversation = [[ConversationItem(type="text", value="hi", role="user")]]
    batch = {"conversation_list": conversation}
    preprocessor(batch, inference=True)
    assert PACKED_INPUT_IDS not in batch


def _defer_runtime(name: str, model, *, ulysses_size: int = 1):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from veomni.distributed.torch_compile import CompileConfig

    rt = MagicMock()
    rt._defer_parallelize = True
    rt.skip_hf_weight_load = False
    rt.model = model
    rt.args = SimpleNamespace(
        model_path=f"/tmp/{name}",
        lora_config=None,
        basic_modules=[],
        optimizer=SimpleNamespace(type="adamw", muon_expert_zero_comm=False),
        accelerator=SimpleNamespace(
            init_device="meta",
            ulysses_size=ulysses_size,
            enable_async=False,
            cp_size=1,
            tp_size=1,
            pp_size=1,
            broadcast_model_weights_from_rank0=False,
            ep_sharded_stream_load=False,
            torch_compile=CompileConfig(),
            gradient_checkpointing=SimpleNamespace(enable=False, enable_reentrant=False, early_stop=True),
            fsdp_config=SimpleNamespace(
                fsdp_scope="model",
                fsdp_mode="fsdp2",
                reshard_after_forward=True,
                mixed_precision=SimpleNamespace(enable=False),
                forward_prefetch=False,
                offload=False,
                offload_pin_memory=False,
                max_load_broadcast_size=20.0,
            ),
        ),
    )
    rt._scoped.return_value = nullcontext()
    return rt


def _omni_runtime_args(*, accelerator=None, optimizer=None):
    from types import SimpleNamespace

    composer = _defer_runtime("composer", torch.nn.Linear(2, 2)).args
    return SimpleNamespace(
        accelerator=accelerator or composer.accelerator,
        optimizer=optimizer if optimizer is not None else composer.optimizer,
    )


def test_composed_wrap_scopes_child_no_split_modules(monkeypatch: pytest.MonkeyPatch):
    """Each child's ``_no_split_modules`` is prefixed with that child's name."""
    from unittest.mock import MagicMock

    from veomni.models.seed_omni.accelerator.omni_model_runtime import OmniModelRuntime

    class _ChildA(torch.nn.Module):
        _no_split_modules = ["LayerA"]

    class _ChildB(torch.nn.Module):
        _no_split_modules = []

    captured: dict = {}

    def _fake_build(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr("veomni.distributed.torch_parallelize.build_parallelize_model", _fake_build)

    child_a, child_b = _ChildA(), _ChildB()
    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = ["LayerA"]
    omni.module_runtimes = {
        "a": _defer_runtime("a", child_a),
        "b": _defer_runtime("b", child_b),
    }
    omni.omni_model_runtime_args = _omni_runtime_args()
    omni._parallelize_composed_model()

    assert captured["basic_modules"] == ["a.LayerA"]
    assert omni.model._no_split_modules == ["a.LayerA"]
    assert captured["weights_path"] == {"a": "/tmp/a", "b": "/tmp/b"}
    assert "_ChildA" not in captured["basic_modules"]
    assert "_ChildB" not in captured["basic_modules"]


def test_composed_wrap_scopes_embedding_to_owning_child(monkeypatch: pytest.MonkeyPatch):
    """TextEncoder ``Embedding`` must not also match a sibling VQ codebook."""
    from unittest.mock import MagicMock

    from veomni.models.seed_omni.accelerator.omni_model_runtime import OmniModelRuntime

    class _TextEnc(torch.nn.Module):
        _no_split_modules = ["Embedding"]

    class _Vqvae(torch.nn.Module):
        _no_split_modules = ["JanusVQVAEVectorQuantizer"]

    class _Llama(torch.nn.Module):
        _no_split_modules = ["LlamaDecoderLayer"]

    captured: dict = {}

    def _fake_build(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr("veomni.distributed.torch_parallelize.build_parallelize_model", _fake_build)

    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = ["Embedding", "JanusVQVAEVectorQuantizer"]
    omni.module_runtimes = {
        "text": _defer_runtime("text", _TextEnc()),
        "vqvae": _defer_runtime("vqvae", _Vqvae()),
        "llama": _defer_runtime("llama", _Llama()),
    }
    omni.omni_model_runtime_args = _omni_runtime_args()
    omni._parallelize_composed_model()

    assert captured["basic_modules"] == [
        "text.Embedding",
        "vqvae.JanusVQVAEVectorQuantizer",
        "llama.LlamaDecoderLayer",
    ]
    assert omni.model._no_split_modules == captured["basic_modules"]
    assert "Embedding" not in captured["basic_modules"]
    assert "_TextEnc" not in captured["basic_modules"]
    assert "_Vqvae" not in captured["basic_modules"]


def test_composed_wrap_does_not_inspect_module_level_sp():
    """Wrap uses the already-built runtimes; module SP overlays are not compared."""
    from unittest.mock import MagicMock, patch

    from veomni.models.seed_omni.accelerator.omni_model_runtime import OmniModelRuntime

    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = []
    omni.module_runtimes = {
        "a": _defer_runtime("a", torch.nn.Linear(2, 2), ulysses_size=1),
        "b": _defer_runtime("b", torch.nn.Linear(2, 2), ulysses_size=4),
    }
    omni.omni_model_runtime_args = _omni_runtime_args()
    with patch(
        "veomni.distributed.torch_parallelize.build_parallelize_model", side_effect=lambda model, **kwargs: model
    ):
        omni._parallelize_composed_model()


def test_composed_wrap_uses_composer_accelerator_not_module_overlay(monkeypatch: pytest.MonkeyPatch):
    """Wrap knobs come from the OmniModel accelerator / optimizer, not a module overlay."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from veomni.models.seed_omni.accelerator.omni_model_runtime import OmniModelRuntime

    captured: dict = {}

    def _fake_build(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr("veomni.distributed.torch_parallelize.build_parallelize_model", _fake_build)

    module = _defer_runtime("a", torch.nn.Linear(2, 2))
    module.args.accelerator.init_device = "cuda"
    module.args.optimizer.type = "adamw"
    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = []
    omni.module_runtimes = {"a": module}
    omni.omni_model_runtime_args = _omni_runtime_args(
        optimizer=SimpleNamespace(type="muon", muon_expert_zero_comm=True),
    )
    omni._parallelize_composed_model()

    assert captured["init_device"] == "meta"
    assert captured["muon_expert_zero_comm"] is True


def test_fsdp_wrap_target_scopes_class_to_child_prefix():
    """A scoped target wraps its class under that child only; bare stays global."""
    from veomni.distributed.torch_parallelize import _is_fsdp_wrap_target

    targets = {"text.Embedding", "llama.LlamaDecoderLayer"}
    assert _is_fsdp_wrap_target("text.embed_tokens", "Embedding", targets)
    assert _is_fsdp_wrap_target("llama.language_model.layers.0", "LlamaDecoderLayer", targets)
    # The VQ codebook is the same class under a different child.
    assert not _is_fsdp_wrap_target("vqvae.vqmodel.quantize.embedding", "Embedding", targets)
    # A prefix must match on a path boundary, not a string prefix.
    assert not _is_fsdp_wrap_target("text_encoder.embed_tokens", "Embedding", targets)
    assert _is_fsdp_wrap_target("layers.0", "LlamaDecoderLayer", {"LlamaDecoderLayer"})


def test_composed_wrap_skips_when_fsdp_scope_is_module(monkeypatch: pytest.MonkeyPatch):
    from unittest.mock import MagicMock

    from veomni.models.seed_omni.accelerator.omni_model_runtime import OmniModelRuntime

    called = []
    monkeypatch.setattr(
        "veomni.distributed.torch_parallelize.build_parallelize_model",
        lambda model, **kwargs: called.append(kwargs) or model,
    )

    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = []
    omni.module_runtimes = {"a": _defer_runtime("a", torch.nn.Linear(2, 2))}
    composer = _defer_runtime("composer", torch.nn.Linear(2, 2)).args.accelerator
    composer.fsdp_config.fsdp_scope = "module"
    omni.omni_model_runtime_args = _omni_runtime_args(accelerator=composer)
    omni._parallelize_composed_model()

    assert called == []
