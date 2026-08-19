from types import SimpleNamespace

import pytest
import torch

import veomni.utils.loss_utils as loss_utils
from veomni.utils.constants import IGNORE_INDEX


def test_reduce_global_loss_token_reduces_each_denominator_once(monkeypatch):
    calls = []

    def fake_all_reduce(value, op="mean", group=None):
        calls.append((value, op, group))
        return value + 10

    monkeypatch.setattr(loss_utils, "all_reduce", fake_all_reduce)

    reduced = loss_utils.reduce_global_loss_token(
        {
            "foundation_tokens": torch.tensor(3),
            "image_decoder_tokens": torch.tensor(0),
        }
    )

    assert reduced == {"foundation_tokens": 13, "image_decoder_tokens": 10}
    assert calls == [(3, "sum", None), (0, "sum", None)]


def test_mean_global_loss_reuses_pre_reduced_denominator(monkeypatch):
    def fail_all_reduce(*args, **kwargs):
        raise AssertionError("denominator should already be reduced")

    monkeypatch.setattr(loss_utils, "all_reduce", fail_all_reduce)
    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=False, fsdp_size=2),
    )

    loss_dict = loss_utils.mean_global_loss(
        {"foundation_loss": torch.tensor(4.0)},
        {"foundation_tokens": torch.tensor(3)},
        {"foundation_tokens": torch.tensor(5)},
        {"foundation_tokens": 6},
    )

    assert loss_dict["foundation_loss"].item() == pytest.approx(4.0)


def test_mean_global_loss_falls_back_to_reducing_denominator(monkeypatch):
    calls = []

    def fake_all_reduce(value, op="mean", group=None):
        calls.append((value, op, group))
        return value * 2

    monkeypatch.setattr(loss_utils, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=False, fsdp_size=1),
    )

    loss_dict = loss_utils.mean_global_loss(
        {"foundation_loss": torch.tensor(4.0)},
        {"foundation_tokens": torch.tensor(2)},
        {"foundation_tokens": torch.tensor(5)},
    )

    assert loss_dict["foundation_loss"].item() == pytest.approx(0.8)
    assert calls == [(5, "sum", None)]


def test_count_loss_token_accumulates_extra_label_keys_across_micro_batches():
    micro_batch = {
        "labels": torch.tensor([[1, 2, 3, IGNORE_INDEX]]),
        "mtp_labels": torch.tensor([[3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]),
    }

    single = loss_utils.count_loss_token(micro_batch)
    assert single["foundation_tokens"].item() == 3
    assert single["mtp_tokens"].item() == 1

    accumulated = loss_utils.count_loss_token([micro_batch, micro_batch, micro_batch])
    assert accumulated["foundation_tokens"].item() == 9
    assert accumulated["mtp_tokens"].item() == 3


def test_count_loss_token_handles_several_extra_label_keys():
    token_len = loss_utils.count_loss_token(
        {
            "labels": torch.tensor([[1, 2]]),
            "mtp_labels": torch.tensor([[2, IGNORE_INDEX]]),
            "image_labels": torch.tensor([[7, 8]]),
        }
    )
    assert token_len["mtp_tokens"].item() == 1
    assert token_len["image_tokens"].item() == 2
