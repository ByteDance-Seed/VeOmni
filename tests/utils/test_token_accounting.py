import types

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.token_accounting import (
    TOKEN_ACCOUNTING_KEY,
    TokenAccounting,
    build_token_accounting,
    metrics_from_totals,
    pop_token_accounting,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


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
    batch = data_collator.MainCollator(pad_to_length=16, collect_token_accounting=True)(_documents())
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


def test_collator_accounting_validation_is_nonfatal():
    import veomni.data.data_collator as data_collator

    batch = {"position_ids": torch.arange(8).unsqueeze(0)}
    data_collator._finalize_token_accounting(
        batch,
        partial={
            "source_input_tokens": 12,
            "num_documents": 2,
            "max_document_length": 7,
            "sum_document_len_squared": 74,
        },
        loss_tokens=10,
        tail_padding_length=0,
    )

    assert TOKEN_ACCOUNTING_KEY not in batch


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
    monkeypatch.setattr(helper, "_compute_image_seqlens", lambda _, config: [])

    batch = {TOKEN_ACCOUNTING_KEY: TokenAccounting(16, 12, 12, 10, 2, 7, 74).to_dict()}
    meter.add(batch)

    assert TOKEN_ACCOUNTING_KEY not in batch
    assert meter.batch_accounting_expected == 1
    assert meter.batch_accounting_observed == 1
    assert meter.batch_token_accountings == [TokenAccounting(16, 12, 12, 10, 2, 7, 74)]


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("with_grid", [False, True])
def test_environ_meter_counts_tokens_and_gemma3_images_together(as_list, with_grid):
    from veomni.utils import helper

    meter = object.__new__(helper.EnvironMeter)
    meter.config = types.SimpleNamespace(model_type="gemma3", vision_config=types.SimpleNamespace(patch_size=14))
    meter.enable_multisource = False
    meter.batch_token_accountings = []
    meter.batch_accounting_expected = 0
    meter.batch_accounting_observed = 0
    meter.batch_seqlens = []
    meter.images_seqlens = []
    accounting = TokenAccounting(4, 4, 4, 3, 1, 4, 16)
    batch = {
        TOKEN_ACCOUNTING_KEY: accounting.to_dict(),
        "attention_mask": torch.ones((1, 4), dtype=torch.int64),
        "pixel_values": torch.empty(2, 3, 28, 28),
    }
    if with_grid:
        batch["image_grid_thw"] = torch.tensor([[1, 2, 3], [1, 2, 4]])

    meter.add([batch] if as_list else batch)

    assert TOKEN_ACCOUNTING_KEY not in batch
    assert meter.batch_accounting_expected == meter.batch_accounting_observed == 1
    assert meter.batch_token_accountings == [accounting]
    assert meter.batch_seqlens == [4]
    assert meter.images_seqlens == ([6, 8] if with_grid else [4, 4])


@pytest.mark.parametrize(("sp_size", "sp_rank"), [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize(("per_segment", "expected"), [(False, 4), (True, 3)])
def test_supervised_token_accounting_respects_objective(monkeypatch, sp_size, sp_rank, per_segment, expected):
    import veomni.data.data_collator as data_collator
    from veomni.trainer.text_dpo_trainer import _build_dpo_labels_list

    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: _parallel_state(sp_size=sp_size, sp_rank=sp_rank))
    feature = {
        "input_ids": torch.tensor([1, 2, 3, 4, 5]),
        "attention_mask": torch.ones(5, dtype=torch.long),
        "labels": torch.tensor([1, 2, 3, 4, 5]),
        "position_ids": torch.tensor([0, 1, 2, 0, 1]),
    }
    batch = data_collator.MainCollator(per_segment_loss_accounting=per_segment, collect_token_accounting=True)(
        [feature]
    )

    assert pop_token_accounting(batch).loss_tokens == expected
    if per_segment:
        labels = feature["labels"]
        if sp_size > 1:
            labels = torch.nn.functional.pad(labels[1:], (0, 1), value=IGNORE_INDEX)
        actual_objective_labels = _build_dpo_labels_list(labels, [3, 2], sp_enabled=sp_size > 1)
        assert sum(int((segment != IGNORE_INDEX).sum()) for segment in actual_objective_labels) == expected
    # The accounting policy must never modify the training labels.
    expected_labels = [1, 2, 3, 4, 5] if sp_size == 1 else ([2, 3, 4] if sp_rank == 0 else [5, -100, -100])
    assert batch["labels"].reshape(-1).tolist() == expected_labels


@pytest.mark.parametrize("sp_size", [1, 2])
def test_standalone_collator_returns_model_inputs_without_accounting(monkeypatch, sp_size):
    import veomni.data.data_collator as data_collator

    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: _parallel_state(sp_size=sp_size))
    batch = data_collator.MainCollator()(_documents())

    assert TOKEN_ACCOUNTING_KEY not in batch
    assert "_token_accounting_partial" not in batch


@pytest.mark.parametrize("trainer_kind", ["text", "dpo", "vlm"])
def test_standard_trainer_collators_enable_accounting(monkeypatch, trainer_kind):
    import veomni.data.data_collator as data_collator
    from veomni.trainer.base import BaseTrainer
    from veomni.trainer.vlm_trainer import VLMTrainer

    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: _parallel_state(sp_size=1))
    base = types.SimpleNamespace(
        args=types.SimpleNamespace(
            data=types.SimpleNamespace(data_type="conversation"),
            train=types.SimpleNamespace(pad_to_length=False),
        ),
        model=types.SimpleNamespace(unwrapped_module=torch.nn.Identity()),
    )
    if trainer_kind == "vlm":
        VLMTrainer._build_collate_fn(types.SimpleNamespace(base=base))
    else:
        BaseTrainer._build_collate_fn(base, per_segment_loss_accounting=trainer_kind == "dpo")

    batch = base.collate_fn(_documents())
    assert pop_token_accounting(batch).loss_tokens == 10


@pytest.mark.parametrize("use_rl", [False, True])
def test_preforward_never_passes_accounting_metadata_to_model(monkeypatch, use_rl):
    import veomni.data.data_collator as data_collator
    from veomni.trainer.base import BaseTrainer
    from veomni.trainer.base_rl_trainer import BaseRLTrainer

    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: _parallel_state(sp_size=1))
    trainer = object.__new__(BaseRLTrainer if use_rl else BaseTrainer)
    trainer.device = torch.device("cpu")
    trainer.LOG_SAMPLE = False
    collator = data_collator.MainCollator(collect_token_accounting=True)
    if use_rl:
        trainer.pre_forward = collator
        prepared = trainer.preforward(_documents())
    else:
        prepared = trainer.preforward(collator(_documents()))

    assert TOKEN_ACCOUNTING_KEY not in prepared
    assert prepared["input_ids"].shape[-1] == 12


def test_loss_count_excludes_padding_and_pack_boundaries(monkeypatch):
    accounting = _collate(monkeypatch, sp_size=1)
    assert accounting.loss_tokens < accounting.source_input_tokens
    assert accounting.loss_tokens == 10
    assert IGNORE_INDEX == -100
