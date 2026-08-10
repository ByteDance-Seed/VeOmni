import types

import pytest
import torch

from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.token_accounting import (
    TOKEN_ACCOUNTING_KEY,
    TokenAccounting,
    build_token_accounting,
    metrics_from_totals,
    pop_token_accounting,
)


def _parallel_state(*, sp_size: int, sp_rank: int = 0):
    return types.SimpleNamespace(
        sp_enabled=sp_size > 1,
        sp_size=sp_size,
        sp_rank=sp_rank,
        cp_enabled=False,
    )


def _documents():
    return [
        {
            "input_ids": torch.arange(5),
            "attention_mask": torch.ones(5, dtype=torch.long),
            "labels": torch.arange(5),
        },
        {
            "input_ids": torch.arange(7),
            "attention_mask": torch.ones(7, dtype=torch.long),
            "labels": torch.arange(7),
        },
    ]


def _collate(monkeypatch, *, sp_size: int):
    import veomni.data.data_collator as data_collator

    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: _parallel_state(sp_size=sp_size))
    batch = data_collator.MainCollator(pad_to_length=16)(_documents())
    assert TOKEN_ACCOUNTING_KEY in batch
    return pop_token_accounting(batch)


def test_token_accounting_validates_ordering():
    stats = build_token_accounting(
        physical_window_tokens=16,
        aligned_compute_tokens=12,
        source_input_tokens=12,
        loss_tokens=10,
        num_documents=2,
        max_document_length=7,
        sum_document_len_squared=74,
    )
    assert stats.source_input_tokens == 12

    with pytest.raises(ValueError, match="ordering violated"):
        build_token_accounting(
            physical_window_tokens=10,
            aligned_compute_tokens=12,
            source_input_tokens=12,
            loss_tokens=10,
            num_documents=2,
            max_document_length=7,
        )


def test_training_step_gate_rejects_schema_only_zero_counters():
    with pytest.raises(ValueError, match="training-step gate violated"):
        TokenAccounting(16, 12, 0, 0, 0, 0).validate_training_step()

    TokenAccounting(16, 12, 12, 10, 2, 7, 74).validate_training_step()


def test_observability_validation_is_nonfatal():
    from veomni.utils.helper import _safe_token_accounting_metrics

    metrics, valid = _safe_token_accounting_metrics(
        TokenAccounting(16, 12, 0, 0, 0, 0),
        delta_time=1.0,
    )

    assert valid is False
    assert metrics == {}



def test_metrics_distinguish_capacity_source_and_loss():
    totals = TokenAccounting(16, 12, 12, 10, 2, 7, 74)
    metrics = metrics_from_totals(totals, delta_time=2.0)

    assert metrics["token_accounting/source_fill"] == pytest.approx(0.75)
    assert metrics["token_accounting/loss_density"] == pytest.approx(10 / 12)
    assert metrics["token_accounting/capacity_tokens_per_second(M)"] == pytest.approx(8e-6)
    assert metrics["token_accounting/source_tokens_per_second(M)"] == pytest.approx(6e-6)


def test_collator_accounting_is_sequence_parallel_invariant(monkeypatch):
    without_sp = _collate(monkeypatch, sp_size=1)
    with_sp = _collate(monkeypatch, sp_size=2)

    assert without_sp == with_sp
    assert with_sp.physical_window_tokens == 16
    assert with_sp.aligned_compute_tokens == 12
    assert with_sp.source_input_tokens == 12
    assert with_sp.loss_tokens == 10
    assert with_sp.num_documents == 2
    assert with_sp.max_document_length == 7
    assert with_sp.sum_document_len_squared == 74


def test_compute_seqlens_prefers_global_linear_attention_boundaries(monkeypatch):
    from veomni.utils import helper

    state = types.SimpleNamespace(cp_enabled=True, sp_size=4)
    monkeypatch.setattr(helper, "get_parallel_state", lambda: state)
    micro_batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 4], dtype=torch.int32),
        "linear_attn_cu_seq_lens_q": torch.tensor([0, 12, 16], dtype=torch.int32),
        "tail_padding_length": torch.tensor(4, dtype=torch.int32),
    }

    assert helper._compute_seqlens(micro_batch) == [12]


def test_compute_seqlens_does_not_scale_rank_local_documents_under_cp(monkeypatch):
    from veomni.utils import helper

    monkeypatch.setattr(
        helper,
        "get_parallel_state",
        lambda: types.SimpleNamespace(cp_enabled=True, sp_size=4),
    )
    micro_batch = {"cu_seq_lens_q": torch.tensor([0, 3, 7], dtype=torch.int32)}

    # CP boundaries are rank-local fragments. Multiplying each fragment would
    # inflate the quadratic attention term used by the FLOPs estimator.
    assert helper._compute_seqlens(micro_batch) == [3, 4]


def test_environ_meter_consumes_accounting_before_model_forward(monkeypatch):
    from veomni.utils import helper

    meter = object.__new__(helper.EnvironMeter)
    meter.config = types.SimpleNamespace(condition_model_type=None)
    meter.enable_multisource = False
    meter.batch_token_accountings = []
    meter.batch_accounting_expected = 0
    meter.batch_accounting_observed = 0
    meter.batch_seqlens = []
    meter.images_seqlens = []
    monkeypatch.setattr(helper, "_compute_seqlens", lambda _: [12])
    monkeypatch.setattr(helper, "_compute_image_seqlens", lambda _: [])

    batch = {TOKEN_ACCOUNTING_KEY: TokenAccounting(16, 12, 12, 10, 2, 7, 74).to_dict()}
    meter.add(batch)

    assert TOKEN_ACCOUNTING_KEY not in batch
    assert meter.batch_accounting_expected == 1
    assert meter.batch_accounting_observed == 1
    assert meter.batch_token_accountings == [TokenAccounting(16, 12, 12, 10, 2, 7, 74)]


def test_loss_count_excludes_padding_and_pack_boundaries(monkeypatch):
    accounting = _collate(monkeypatch, sp_size=1)
    assert accounting.loss_tokens < accounting.source_input_tokens
    assert accounting.loss_tokens == 10
    assert IGNORE_INDEX == -100
