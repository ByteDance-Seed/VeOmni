"""Unit tests for SeedOmni ``modules/base`` and ``veomni.utils.tensor_utils``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.seed_omni.modules.helpers import load_from_omni, save_as_omni
from veomni.models.seed_omni import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_MODEL_REGISTRY,
    BaseMixin,
)
from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY
from veomni.models.seed_omni.modules.module_modeling_base import PretrainedOmniModule
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.tensor_utils import naflatten, unflatten


def _config_cls(model_type: str):
    return OMNI_CONFIG_REGISTRY[model_type]()


def _model_cls(model_type: str):
    return OMNI_MODEL_REGISTRY[model_type]()


def _accelerated_model_cls(model_type: str):
    return OMNI_ACCELERATED_MODEL_REGISTRY[model_type]()


def _patch_local_loss_reducer(monkeypatch):
    """Keep module-only tests out of the distributed loss collective."""
    import veomni.distributed.sequence_parallel as sp_utils

    def _local_reduce(loss, num_valid_tokens, group=None):
        del group
        return torch.where(num_valid_tokens > 0, loss, torch.zeros_like(loss))

    monkeypatch.setattr(sp_utils, "reduce_sequence_parallel_loss", _local_reduce)
    return sp_utils


def test_naflatten_and_unflatten_roundtrip():
    tensors_3d = [torch.randn(2, 3, 8), torch.randn(1, 5, 8)]
    flat_3d, shape_3d = naflatten(tensors_3d)
    assert shape_3d.device.type == "cpu"
    restored_3d = unflatten(flat_3d, shape_3d)
    for a, b in zip(tensors_3d, restored_3d, strict=True):
        torch.testing.assert_close(a, b)

    tensors_1d = [torch.randn(4), torch.randn(7)]
    flat_1d, shape_1d = naflatten(tensors_1d)
    assert shape_1d.device.type == "cpu"
    restored_1d = unflatten(flat_1d, shape_1d)
    for a, b in zip(tensors_1d, restored_1d, strict=True):
        torch.testing.assert_close(a, b)


def test_all_registered_classes_are_module_mixins():
    accelerated_keys = set(OMNI_ACCELERATED_MODEL_REGISTRY.valid_keys())
    for name in OMNI_MODEL_REGISTRY.valid_keys():
        native_cls = OMNI_MODEL_REGISTRY[name]()
        assert name in accelerated_keys, f"{name} must register OMNI_ACCELERATED_MODEL_REGISTRY"
        assert issubclass(native_cls, PretrainedOmniModule), f"{name} native must inherit PretrainedOmniModule"
        assert not issubclass(native_cls, BaseMixin), f"{name} native must not inherit BaseMixin"
        accelerated_cls = OMNI_ACCELERATED_MODEL_REGISTRY[name]()
        assert issubclass(accelerated_cls, BaseMixin), f"{name} accelerated must inherit BaseMixin"


def test_text_encoder_save_reload_via_registry(tmp_path: Path):
    """Verifies config/model round-trip through OmniModel and the OMNI registry."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")

    te = TextEncoder(TextEncoderConfig(vocab_size=128, hidden_size=64, tie_word_embeddings=True))
    save_as_omni(tmp_path, "text_encoder", te)

    cfg, te2 = load_from_omni(tmp_path, "text_encoder")
    assert cfg.model_type == "text_encoder"
    assert isinstance(te2, TextEncoder)
    assert te2.config.vocab_size == 128


def test_text_encoder_decode_returns_single_loss_key():
    """Single-loss protocol: ``post_forward`` maps ``loss`` → ``_loss``."""
    TextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16))
    h = torch.randn(2, 4, 16)
    labels = torch.randint(0, 64, (2, 4))
    out = te.decode(hidden_states=h, labels=labels)
    assert out["loss"] is not None and out["loss"].dim() == 0
    graph_out = te.post_forward("decode", **out)
    assert "_loss" in graph_out and graph_out["_loss"].dim() == 0
    assert "lm_loss" not in graph_out


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_text_encoder_accelerated_decode_matches_native_with_unshifted_and_masked_labels(
    monkeypatch, tie_word_embeddings
):
    """Unshifted ``labels`` (including masked prompt positions) must match native ``TextEncoder.decode``."""
    _patch_local_loss_reducer(monkeypatch)
    import veomni.models.seed_omni.modules.base.text_encoder.accelerated.accelerated as text_encoder_accelerated

    monkeypatch.setattr(
        text_encoder_accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_group=None),
    )

    NativeTextEncoder = _model_cls("text_encoder")
    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")

    cfg = TextEncoderConfig(
        vocab_size=64,
        hidden_size=16,
        tie_word_embeddings=tie_word_embeddings,
        ops_implementation={"cross_entropy_loss_implementation": "eager"},
    )
    torch.manual_seed(0)
    native = NativeTextEncoder(cfg)
    acc = AccTextEncoder(cfg)
    acc.load_state_dict(native.state_dict())

    h = torch.randn(2, 5, 16)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, IGNORE_INDEX, 3, 7, 11],
            [IGNORE_INDEX, 5, IGNORE_INDEX, 9, 13],
        ],
        dtype=torch.long,
    )
    native_loss = native.decode(hidden_states=h, labels=labels)["loss"]
    acc_loss = acc.decode(hidden_states=h, labels=labels)["loss"]
    torch.testing.assert_close(acc_loss, native_loss)


def test_text_encoder_decode_pre_masks_sample_boundary_labels():
    """Sample k's first label is IGNORE regardless of what the chat template emitted."""
    TextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=8))

    def _text(labels: list[int]) -> ConversationItem:
        return ConversationItem(
            type="text",
            value=torch.randn(len(labels), 8),
            role="assistant",
            meta={"labels": torch.tensor(labels)},
        )

    conversation_list = [[_text([1, 2, 3])], [_text([4, 5])]]
    out = te.decode_pre(conversation_list=conversation_list)

    assert out["shift_labels"].tolist() == [2, 3, IGNORE_INDEX, 5, IGNORE_INDEX]
    assert conversation_list[1][0].meta["labels"].tolist() == [4, 5]


def test_text_encoder_decode_inference_returns_logits_only():
    """Base ``TextEncoder.decode`` without labels returns logits only."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16))
    h = torch.randn(2, 4, 16)
    out = te.decode(hidden_states=h)
    assert out["logits"] is not None and out["logits"].shape == (2, 4, 64)
    assert out["loss"] is None


def test_text_encoder_decode_dispatches_pre_shifted_loss(monkeypatch):
    """Untied, bias-free heads delegate loss policy to the configured ops wrapper."""
    import veomni.models.seed_omni.modules.base.text_encoder.accelerated.accelerated as text_encoder_accelerated

    reduction_group = object()
    captured = {}
    monkeypatch.setattr(
        text_encoder_accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_group=reduction_group),
    )

    TextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=False))

    def fake_loss_function(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"].sum() * 0, None, None

    te.loss_function = fake_loss_function
    hidden_states = torch.randn(5, 16)
    shift_labels = torch.tensor([1, 2, -100, 3, -100])
    out = te.decode(hidden_states=hidden_states, shift_labels=shift_labels)

    assert out["loss"].dim() == 0
    assert out["logits"] is None
    assert captured["hidden_states"].shape == (3, 16)
    assert torch.equal(captured["hidden_states"], hidden_states[[0, 1, 3]])
    assert torch.equal(captured["shift_labels"], torch.tensor([1, 2, 3]))
    assert torch.equal(captured["labels"], torch.tensor([1, 2, 3]))
    assert captured["weights"] is te.lm_head.weight
    assert captured["loss_reduction_group"] is reduction_group


def test_text_encoder_tied_head_uses_explicit_eager_loss(monkeypatch):
    """Tied heads must not enter a globally selected fused-linear loss."""
    _patch_local_loss_reducer(monkeypatch)
    TextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=True))

    def unexpected_loss_function(**kwargs):
        raise AssertionError("tied head must not call the globally selected loss function")

    te.loss_function = unexpected_loss_function
    hidden_states = torch.randn(5, 16)
    shift_labels = torch.tensor([1, 2, -100, 3, -100])
    out = te.decode(hidden_states=hidden_states, shift_labels=shift_labels)

    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (3, 64)


def test_text_encoder_decode_all_masked_span_scores_zero(monkeypatch):
    """An unsupervised span scores 0.0, never NaN (constraint 7d clamped denominator)."""
    _patch_local_loss_reducer(monkeypatch)
    TextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=False))

    def unexpected_loss_function(**kwargs):
        raise AssertionError("an all-masked span must not enter the fused loss (0/0)")

    te.loss_function = unexpected_loss_function
    out = te.decode(hidden_states=torch.randn(5, 16), shift_labels=torch.full((5,), -100))

    assert float(out["loss"].detach()) == 0.0
    assert out["loss"].requires_grad


def test_training_module_mixin_forward_requires_override_without_native_impl():
    class CodecOnlyAccelerated(TrainingModuleMixin, BaseMixin):
        pass

    with pytest.raises(NotImplementedError, match="forward\\(\\*\\*kwargs\\) is not implemented"):
        CodecOnlyAccelerated().forward()


def test_text_encoder_fsdp_no_split_modules_preserved():
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    text_encoder = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=False))
    assert text_encoder._no_split_modules == {"ShardedEmbedding"}


def test_text_encoder_selects_ops_from_module_config_not_global_ops_config():
    """Each OmniModule resolves its VeomniOp handles from its own config."""
    from veomni.ops.registry import OP_REGISTRY

    TextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")

    te_default = TextEncoder(
        TextEncoderConfig(
            vocab_size=64,
            hidden_size=16,
            tie_word_embeddings=False,
        )
    )
    te_eager = TextEncoder(
        TextEncoderConfig(
            vocab_size=64,
            hidden_size=16,
            tie_word_embeddings=False,
            ops_implementation={"cross_entropy_loss_implementation": "eager"},
        )
    )
    te_chunk = TextEncoder(
        TextEncoderConfig(
            vocab_size=64,
            hidden_size=16,
            tie_word_embeddings=False,
            ops_implementation={"cross_entropy_loss_implementation": "chunk_loss"},
        )
    )

    assert te_default.veomni_ce.impl == OP_REGISTRY.default_impl("cross_entropy_loss", "standard")
    assert te_eager.veomni_ce.impl == "eager"
    assert te_chunk.veomni_ce.impl == "chunk_loss"


def test_text_encoder_preprocessor_from_pretrained_forwards_and_tolerates_processor_kwargs():
    from veomni.models.seed_omni.modules.base.text_encoder.processing import TextEncoderPreprocessor

    fake_template = SimpleNamespace(tokenizer="tok")

    class _BaseSub(TextEncoderPreprocessor):
        @classmethod
        def build_chat_template(cls, module_path: str, *, config_overrides=None):
            return fake_template

    class _CustomSub(_BaseSub):
        def __init__(self, chat_template, *, packed_preprocess: bool = False, **kwargs):
            super().__init__(chat_template, **kwargs)
            self.packed_preprocess = packed_preprocess

    base_proc = _BaseSub.from_pretrained("/tmp/fake", extra_unused_kwarg=True)
    assert base_proc._tokenizer == "tok"

    custom_proc = _CustomSub.from_pretrained("/tmp/fake", packed_preprocess=True, extra_unused_kwarg=1)
    assert custom_proc.packed_preprocess is True
