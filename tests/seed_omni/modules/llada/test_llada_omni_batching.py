"""Unit tests for the LLaDA Omni module's batching and sample counting.

Both cover a defect that the non-Omni path would not have caught, because the
Omni path assembles its own batch inside the graph instead of inheriting
``MainCollator``'s packing.

* The batch must be **packed**: the configured backend is ``flash_attention_2``,
  which takes sample boundaries from ``cu_seq_lens_*`` and receives no attention
  mask, so a ``[B, L]`` block would let every row attend across the batch.
* A draw that masked nothing must still count as one sample of the mean, or the
  loss is biased up by the fraction of such draws.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from veomni.models.seed_omni.modules.llada.accelerated.accelerated import TrainingMixin
from veomni.models.seed_omni.utils.conversation import ConversationItem


def _sample(length: int, answer_at: list[int], supervised_everywhere: bool = True) -> list[ConversationItem]:
    """One carrier sample holding the tensors ``process_llada_example`` produces."""
    labels = torch.full((length,), -100, dtype=torch.long)
    for pos in answer_at:
        labels[pos] = 10 + pos
    weight = torch.zeros(length, dtype=torch.float32)
    if supervised_everywhere:
        weight.fill_(0.5)
    value = {
        "input_ids": torch.arange(length),
        "attention_mask": torch.ones(length, dtype=torch.long),
        "labels": labels,
        "loss_weight": weight,
        "answer_mask": (labels != -100).long(),
    }
    return [ConversationItem(type="text", value=value, role="assistant", meta={})]


def _module() -> TrainingMixin:
    module = TrainingMixin()
    module.config = SimpleNamespace(pad_token_id=0, router_aux_loss_coef=0.0)
    # `batch_pre` reads the device off the embedding weight; a real module gets it
    # from the built model, which a CPU unit test has no reason to construct.
    module.get_input_embeddings = lambda: SimpleNamespace(weight=torch.zeros(1))
    return module


def test_two_samples_are_packed_into_one_row_with_a_boundary_each():
    """Two samples of different lengths become one row, and the second's first
    token is a boundary. That is what keeps the varlen backend from attending
    across them."""
    batch = _module().batch_pre(conversation_list=[_sample(4, [1, 2]), _sample(3, [1])])

    assert batch["input_ids"].shape == (1, 7)
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2]]
    assert batch["cu_seq_lens_q"].tolist() == [0, 4, 7]
    assert batch["max_length_q"] == 4


def test_a_single_sample_still_produces_varlen_boundaries():
    """One sample needs the same kwargs: the backend reads lengths from them even
    when there is nothing to separate."""
    batch = _module().batch_pre(conversation_list=[_sample(5, [2])])

    assert batch["input_ids"].shape == (1, 5)
    assert batch["cu_seq_lens_q"].tolist() == [0, 5]


def test_the_answer_mask_does_not_reach_the_model():
    """It exists only for counting; forwarding it would pass an unexpected keyword
    down to the decoder layers."""
    batch = _module().batch_pre(conversation_list=[_sample(4, [1])])

    assert "answer_mask" not in batch


def test_a_masked_nothing_draw_is_still_one_sample_of_the_mean():
    """LLaDA masks nothing in about ``1/(N+1)`` of draws. Those contribute zero to
    the sum but are still one sample, so counting non-zero weights would bias the
    loss up by that fraction."""
    micro_batch = {"conversation_list": [_sample(5, [2], supervised_everywhere=False)]}

    assert TrainingMixin._micro_batch_sample_count(micro_batch) == 1


def test_packed_samples_are_counted_per_segment():
    """Under packing a row is not a sample; the segment is."""
    packed = {
        "labels": torch.tensor([[-100, 5, 6, -100, 7]]),
        "loss_weight": torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.5]]),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
    }
    micro_batch = {"conversation_list": [[ConversationItem(type="text", value=packed, role="assistant", meta={})]]}

    assert TrainingMixin._micro_batch_sample_count(micro_batch) == 2


@pytest.mark.parametrize("missing", [None, []])
def test_a_batch_without_samples_counts_zero_rather_than_raising(missing):
    """``on_micro_batches_ready`` runs on every step, including a dummy pass."""
    assert TrainingMixin._micro_batch_sample_count({"conversation_list": missing}) == 0


def test_sequence_parallelism_is_refused_rather_than_mis_computed(monkeypatch):
    """``batch_pre`` hands every rank the whole packed row, so under Ulysses each
    would supervise the full sequence and the gradient would then be scaled by the
    group size. The loss would be wrong without any shape error, which is why the
    module refuses the topology instead of running it."""
    from veomni.models.seed_omni.modules.llada.accelerated import accelerated

    monkeypatch.setattr(accelerated, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True, sp_size=2))

    with pytest.raises(NotImplementedError, match="does not implement sequence parallelism"):
        _module().batch_pre(conversation_list=[_sample(4, [1])])
