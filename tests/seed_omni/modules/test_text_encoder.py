"""Unit tests for SeedOmni ``TextEncoder`` (``veomni/models/seed_omni/modules/text_encoder/``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.seed_omni.modules.helpers import load_from_omni, save_as_omni
from veomni.models.seed_omni import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_MODEL_REGISTRY,
)
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY
from veomni.models.seed_omni.modules.text_encoder.configuration import TextEncoderConfig
from veomni.models.seed_omni.modules.text_encoder.modeling import TextEncoder
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.constants import IGNORE_INDEX


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

_VOCAB, _HIDDEN = 8, 4


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


def test_text_encoder_save_reload_via_registry(tmp_path: Path):
    """Verifies config/model round-trip through OmniModel and the OMNI registry."""
    TextEncoderCls = _model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")

    te = TextEncoderCls(TextEncoderCfg(vocab_size=128, hidden_size=64, tie_word_embeddings=True))
    save_as_omni(tmp_path, "text_encoder", te)

    cfg, te2 = load_from_omni(tmp_path, "text_encoder")
    assert cfg.model_type == "text_encoder"
    assert isinstance(te2, TextEncoderCls)
    assert te2.config.vocab_size == 128


def test_text_encoder_decode_returns_single_loss_key():
    """Single-loss protocol: ``post_forward`` maps ``loss`` → ``_loss``."""
    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    te = AccTextEncoder(TextEncoderCfg(vocab_size=64, hidden_size=16))
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
    import veomni.models.seed_omni.modules.text_encoder.accelerated.accelerated as text_encoder_accelerated

    monkeypatch.setattr(
        text_encoder_accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_group=None),
    )

    NativeTextEncoder = _model_cls("text_encoder")
    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")

    cfg = TextEncoderCfg(
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
    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    te = AccTextEncoder(TextEncoderCfg(vocab_size=64, hidden_size=8))

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


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_text_encoder_decode_pre_keeps_non_text_boundary_row(modality: str):
    """Every non-text segment keeps its final row with a masked label so the
    following text segment's first label shifts onto the media boundary row."""
    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    te = AccTextEncoder(TextEncoderCfg(vocab_size=64, hidden_size=8))

    prompt = ConversationItem(
        type="text",
        value=torch.full((2, 8), 1.0),
        role="user",
        meta={"labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX])},
    )
    media = ConversationItem(
        type=modality,
        value=torch.tensor([[2.0] * 8, [3.0] * 8, [4.0] * 8]),
        role="user",
    )
    reply = ConversationItem(
        type="text",
        value=torch.tensor([[5.0] * 8, [6.0] * 8]),
        role="assistant",
        meta={"labels": torch.tensor([10, 11])},
    )

    out = te.decode_pre(conversation_list=[[prompt, media, reply]])
    # Rows: 2 prompt + 1 media tail + 2 reply = 5 rows.
    # Unshifted labels: [-100, -100, -100 (media), 10, 11] -> shifted: [-100, -100, 10, 11, -100].
    # Thus row 2 (media[-1]) predicts label 10 (reply[0]), not row 1 (prompt[-1]).
    assert out["hidden_states"].shape == (5, 8)
    torch.testing.assert_close(out["hidden_states"][2], media.value[-1])
    assert out["shift_labels"].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 10, 11, IGNORE_INDEX]


def test_text_encoder_native_decode_fp16_casts_logits_to_fp32():
    """Native ``TextEncoder.decode`` casts logits to FP32 before ``reduction='sum'`` so
    half-precision long sequences do not overflow to ``inf``."""
    NativeTextEncoder = _model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    te = NativeTextEncoder(TextEncoderCfg(vocab_size=8, hidden_size=4, tie_word_embeddings=False)).half()
    with torch.no_grad():
        te.lm_head.weight.zero_()
    h = torch.zeros(1, 40000, 4, dtype=torch.float16)
    labels = torch.zeros(1, 40000, dtype=torch.long)

    out_shift = te.decode(hidden_states=h, shift_labels=labels)
    out_unshifted = te.decode(hidden_states=h, labels=labels)
    assert out_shift["loss"].dtype == torch.float32
    assert out_unshifted["loss"].dtype == torch.float32
    assert torch.isfinite(out_shift["loss"])
    assert torch.isfinite(out_unshifted["loss"])


def test_text_encoder_decode_inference_returns_logits_only():
    """Base ``TextEncoder.decode`` without labels returns logits only."""
    NativeTextEncoder = _model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    te = NativeTextEncoder(TextEncoderCfg(vocab_size=64, hidden_size=16))
    h = torch.randn(2, 4, 16)
    out = te.decode(hidden_states=h)
    assert out["logits"] is not None and out["logits"].shape == (2, 4, 64)
    assert out["loss"] is None


def test_text_encoder_decode_dispatches_pre_shifted_loss(monkeypatch):
    """Untied, bias-free heads delegate loss policy to the configured ops wrapper."""
    import veomni.models.seed_omni.modules.text_encoder.accelerated.accelerated as text_encoder_accelerated

    reduction_group = object()
    captured = {}
    monkeypatch.setattr(
        text_encoder_accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_group=reduction_group),
    )

    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    te = AccTextEncoder(TextEncoderCfg(vocab_size=64, hidden_size=16, tie_word_embeddings=False))

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
    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    te = AccTextEncoder(TextEncoderCfg(vocab_size=64, hidden_size=16, tie_word_embeddings=True))

    def unexpected_loss_function(**kwargs):
        raise AssertionError("tied head must not call the globally selected loss function")

    te.loss_function = unexpected_loss_function
    hidden_states = torch.randn(5, 16)
    shift_labels = torch.tensor([1, 2, -100, 3, -100])
    out = te.decode(hidden_states=hidden_states, shift_labels=shift_labels)

    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (3, 64)


def test_text_encoder_decode_all_masked_span_scores_zero(monkeypatch):
    """An unsupervised span scores 0.0, never NaN (clamped denominator)."""
    _patch_local_loss_reducer(monkeypatch)
    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    te = AccTextEncoder(TextEncoderCfg(vocab_size=64, hidden_size=16, tie_word_embeddings=False))

    def unexpected_loss_function(**kwargs):
        raise AssertionError("an all-masked span must not enter the fused loss (0/0)")

    te.loss_function = unexpected_loss_function
    out = te.decode(hidden_states=torch.randn(5, 16), shift_labels=torch.full((5,), -100))

    assert float(out["loss"].detach()) == 0.0
    assert out["loss"].requires_grad
    assert out["logits"].shape == (1, 64)


def test_text_encoder_fsdp_no_split_modules_preserved():
    NativeTextEncoder = _model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    text_encoder = NativeTextEncoder(TextEncoderCfg(vocab_size=64, hidden_size=16, tie_word_embeddings=False))
    assert text_encoder._no_split_modules == {"ShardedEmbedding"}


def test_text_encoder_selects_ops_from_module_config_not_global_ops_config():
    """Each OmniModule resolves its VeomniOp handles from its own config."""
    from veomni.ops.registry import OP_REGISTRY

    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")

    te_default = AccTextEncoder(
        TextEncoderCfg(
            vocab_size=64,
            hidden_size=16,
            tie_word_embeddings=False,
        )
    )
    te_eager = AccTextEncoder(
        TextEncoderCfg(
            vocab_size=64,
            hidden_size=16,
            tie_word_embeddings=False,
            ops_implementation={"cross_entropy_loss_implementation": "eager"},
        )
    )
    te_chunk = AccTextEncoder(
        TextEncoderCfg(
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
    from veomni.models.seed_omni.modules.text_encoder.processing import TextEncoderPreprocessor

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


@pytest.mark.parametrize("pad_token_id", [None, 3])
def test_text_encoder_passes_pad_token_id_to_sharded_embedding(pad_token_id):
    text_encoder = TextEncoder(TextEncoderConfig(vocab_size=_VOCAB, hidden_size=_HIDDEN, pad_token_id=pad_token_id))
    assert text_encoder.embed_tokens.padding_idx == pad_token_id


def test_text_encoder_tied_head_projects_through_sharded_embedding():
    text_encoder = TextEncoder(TextEncoderConfig(vocab_size=_VOCAB, hidden_size=_HIDDEN, tie_word_embeddings=True))
    ids = torch.tensor([[0, 2, 5]], dtype=torch.long)
    embeds = text_encoder.encode(input_ids=ids)["inputs_embeds"]
    logits = text_encoder._project(embeds)
    expected = F.linear(embeds, text_encoder.embed_tokens.weight)
    torch.testing.assert_close(logits, expected)


def test_text_encoder_ddp_allowed_when_ep_enabled_and_emb_disabled(monkeypatch):
    """A DDP TextEncoder with ``emb=1`` must not be rejected when the job mesh enables ``ep>1``."""
    from veomni.distributed import torch_parallelize

    AccTextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderCfg = _config_cls("text_encoder")
    te = AccTextEncoder(TextEncoderCfg(vocab_size=_VOCAB, hidden_size=_HIDDEN, tie_word_embeddings=False))
    monkeypatch.setattr(torch_parallelize, "DDP", lambda module, **kwargs: module)

    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: SimpleNamespace(
            local_rank=0,
            dp_group=None,
            any_extra_parallel_enabled=True,
            extra_parallel_sizes={"ep": 4, "emb": 1},
            extra_parallel_enabled=lambda para: para == "ep",
        ),
    )
    assert torch_parallelize.parallelize_model_ddp(te) is te

    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: SimpleNamespace(
            local_rank=0,
            dp_group=None,
            any_extra_parallel_enabled=True,
            extra_parallel_sizes={"ep": 1, "emb": 2},
            extra_parallel_enabled=lambda para: para == "emb",
        ),
    )
    with pytest.raises(RuntimeError, match="requires fsdp_mode='fsdp2'"):
        torch_parallelize.parallelize_model_ddp(te)
