"""Weighted chunk CE agrees with a dense reference, including zero mass."""

import pytest
import torch
import torch.nn.functional as F

from veomni.ops import resolve_op


@pytest.mark.parametrize("weights", [[0.0, 0.5, 1.0, 0.2, 1.0], [0.0] * 5, [0.01] * 5])
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_weighted_chunk_loss_and_grads(weights, chunk_size):
    """Match weighted chunk loss and gradients to dense cross-entropy."""
    torch.manual_seed(12)
    hidden = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    labels = torch.tensor([1, 2, -100, 4, 3])
    token_weights = torch.tensor(weights)
    op = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper
    actual = op(hidden, labels, weight, loss_weights=token_weights, chunk_size=chunk_size)
    actual_grad = torch.autograd.grad(actual, (hidden, weight))
    valid_weights = token_weights * (labels != -100)
    denominator = valid_weights.sum()
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    expected = (
        F.cross_entropy(F.linear(hidden, weight), labels, ignore_index=-100, reduction="none") * valid_weights
    ).sum() / denominator
    expected_grad = torch.autograd.grad(expected, (hidden, weight))
    torch.testing.assert_close(actual, expected)
    for actual_g, expected_g in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_g, expected_g, atol=2e-7, rtol=1e-5)


def test_causal_weighting_shifts_with_targets_and_matches_dense():
    """Shift token weights with causal targets and compare the dense loss."""
    from veomni.models.loss_utils.cross_entropy_loss import ForCausalLMLoss
    from veomni.ops import VeomniOp

    torch.manual_seed(42)
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    labels = torch.tensor([[-100, 2, 3, 4, 1]])
    weights = torch.tensor([[0.0, 0.2, 0.0, 1.0, 0.5]])
    actual, _, _ = ForCausalLMLoss(
        hidden_states=hidden,
        weights=weight,
        labels=labels,
        loss_weights=weights,
        op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss"),
        chunk_size=2,
    )
    target = F.pad(labels, (0, 1), value=-100)[..., 1:]
    aligned_weights = F.pad(weights, (0, 1), value=0)[..., 1:]
    ce = F.cross_entropy(
        F.linear(hidden, weight).reshape(-1, 7), target.reshape(-1), reduction="none", ignore_index=-100
    ).reshape_as(target)
    expected = (ce * aligned_weights).sum() / aligned_weights.sum()
    ag = torch.autograd.grad(actual, (hidden, weight), retain_graph=True)
    eg = torch.autograd.grad(expected, (hidden, weight))
    torch.testing.assert_close(actual, expected)
    for a, e in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, e, atol=2e-7, rtol=1e-5)


def test_chatml_message_weights_and_legacy_binary():
    """Preserve binary masks while emitting fractional message weights."""
    from types import SimpleNamespace

    from veomni.data.chat_template import ChatmlTemplate

    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode()))
    template = ChatmlTemplate(tokenizer)
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "partial", "loss_weight": 0.2},
        {"role": "assistant", "content": "skip", "loss_weight": 0.0},
        {"role": "assistant", "content": "full"},
    ]
    out = template.encode_messages(messages)
    assert len(out["input_ids"]) == len(out["loss_weights"]) == len(out["labels"])
    partial = template.encode_messages(messages[1:2])
    assert all(abs(w - 0.2) < 1e-6 for w in partial["loss_weights"])
    skip = template.encode_messages(messages[2:3])
    assert all(label == -100 for label in skip["labels"])
    assert "loss_weights" not in template.encode_messages(messages[-1:])


def test_chatml_nullable_mask_matches_role_defaults():
    from types import SimpleNamespace

    from veomni.data.chat_template import ChatmlTemplate

    template = ChatmlTemplate(SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode())))
    default = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "loss_weight": 0.25}]
    nullable = [{**message, "loss_mask": None} for message in default]
    assert template.encode_messages(nullable) == template.encode_messages(default)


def test_truncated_fractional_turn_does_not_change_unweighted_normalization(monkeypatch):
    from types import SimpleNamespace

    from veomni.data.chat_template import ChatmlTemplate
    from veomni.utils import loss_utils

    template = ChatmlTemplate(SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode())))
    tail = {"role": "assistant", "content": "normal tail"}
    expected = template.encode_messages([tail])
    actual = template.encode_messages(
        [{"role": "assistant", "content": "removed", "loss_weight": 0.25}, tail],
        max_seq_len=len(expected["input_ids"]),
    )
    assert actual == expected and "loss_weights" not in actual
    first_only = template.encode_messages(
        [{"role": "assistant", "content": "removed", "loss_weight": 0.25}, tail],
        max_seq_len=len(expected["input_ids"]) + 1,
    )
    assert first_only["labels"][0] != -100 and "loss_weights" not in first_only
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(loss_utils, "all_reduce", lambda value, op: value)
    batches = [{"labels": torch.tensor(actual["labels"])}, {"labels": torch.tensor([5, 6, 7])}]
    weighted, _, totals = loss_utils.prepare_loss_token_counts(batches)
    assert not weighted and totals["foundation_tokens"] == len(actual["labels"]) + 3


def test_all_zero_weighted_step_is_rejected_before_optimizer(monkeypatch):
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(loss_utils, "all_reduce", lambda value, op: value)
    batch = {"labels": torch.tensor([-100, -100]), "loss_weights": torch.tensor([0.0, 0.0])}
    with pytest.raises(ValueError, match="no supervised target mass"):
        loss_utils.prepare_loss_token_counts([batch])
    with pytest.raises(ValueError, match="no supervised target mass"):
        loss_utils.prepare_loss_token_counts([{"labels": torch.tensor([-100, -100])}])
    _, _, totals = loss_utils.prepare_loss_token_counts(
        [{"labels": torch.tensor([-100, -100]), "image_labels": torch.tensor([1, -100])}]
    )
    assert totals["foundation_tokens"] == 0 and totals["image_tokens"] == 1


@pytest.mark.parametrize("template_name", ["DefaultTemplate", "TokenizerTemplate"])
def test_conversation_rejects_unsupported_loss_weight_template(template_name):
    from types import SimpleNamespace

    from veomni.data import chat_template
    from veomni.data.data_transform import process_conversation_example

    template = getattr(chat_template, template_name)(SimpleNamespace())
    messages = [{"role": "assistant", "content": "answer", "loss_mask": 1, "loss_weight": 0.25}]
    with pytest.raises(ValueError, match="does not support loss_weight"):
        process_conversation_example({"messages": messages}, template, max_seq_len=64)


def test_conversation_preserves_chatml_loss_weight():
    from types import SimpleNamespace

    from veomni.data.chat_template import ChatmlTemplate
    from veomni.data.data_transform import process_conversation_example

    template = ChatmlTemplate(SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode())))
    messages = [{"role": "assistant", "content": "answer", "loss_weight": 0.25}]
    result = process_conversation_example({"messages": messages}, template, max_seq_len=64)[0]
    assert torch.all(result["loss_weights"] == 0.25)


def test_loss_mass_uses_weights_only_on_valid_targets():
    from veomni.utils.loss_utils import count_loss_token

    result = count_loss_token(
        [
            {"labels": torch.tensor([-100, 2, 3]), "loss_weights": torch.tensor([5.0, 0.1, 0.2])},
            {"labels": torch.tensor([1, -100])},
        ]
    )
    torch.testing.assert_close(result["foundation_tokens"], torch.tensor(1.3))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sp_enabled", [False, True])
def test_long_sequence_weight_denominator_accumulates_in_fp32(monkeypatch, dtype, sp_enabled):
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=sp_enabled))
    labels = torch.ones(65_537, dtype=torch.long)
    weights = torch.ones(labels.shape, dtype=dtype)
    actual = loss_utils.count_loss_token({"labels": labels, "loss_weights": weights})["foundation_tokens"]
    assert actual.dtype == torch.float32
    assert actual.item() == (65_537 if sp_enabled else 65_536)


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1, 1e-45])
def test_weight_preflight_rejects_invalid_custom_batch(monkeypatch, value):
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    batch = {"labels": torch.tensor([1, 2]), "loss_weights": torch.tensor([1.0, value])}
    with pytest.raises(ValueError, match="normal float32"):
        loss_utils.prepare_loss_token_counts([batch])


def test_packing_rejects_per_sample_weight_misalignment_even_if_total_matches(monkeypatch):
    from types import SimpleNamespace

    from veomni.data import data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    samples = [
        {"labels": torch.tensor([1, 2, 3]), "loss_weights": torch.tensor([0.5, 0.5])},
        {"labels": torch.tensor([4, 5, 6]), "loss_weights": torch.tensor([0.5, 0.5, 0.5, 0.5])},
    ]
    with pytest.raises(ValueError, match="corresponding labels before packing"):
        m.PackingCollator()(samples)


def test_packing_nullable_weights_match_missing_weights(monkeypatch):
    from types import SimpleNamespace

    from veomni.data import data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    first = {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([-100, 2]), "position_ids": torch.arange(2)}
    second = {
        "input_ids": torch.tensor([3, 4]),
        "labels": torch.tensor([-100, 4]),
        "position_ids": torch.arange(2),
        "loss_weights": torch.tensor([0.0, 0.25]),
    }
    baseline = m.MainCollator()([first.copy(), second.copy()])
    nullable = m.MainCollator()([{**first, "loss_weights": None}, second.copy()])
    assert torch.equal(nullable["labels"], baseline["labels"])
    assert torch.equal(nullable["loss_weights"], baseline["loss_weights"])
    all_null = m.MainCollator()([{**first, "loss_weights": None}])
    assert "loss_weights" not in all_null


def test_collator_packs_mixed_weighted_and_unweighted_samples(monkeypatch):
    from types import SimpleNamespace

    from veomni.data import data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    c = m.MainCollator()
    samples = [
        {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([-100, 2]),
            "position_ids": torch.tensor([0, 1]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "loss_weights": torch.tensor([0.0, 0.2]),
        },
        {
            "input_ids": torch.tensor([3, 4]),
            "labels": torch.tensor([-100, 4]),
            "position_ids": torch.tensor([0, 1]),
            "attention_mask": torch.ones(2, dtype=torch.long),
        },
    ]
    result = c(samples)
    torch.testing.assert_close(result["loss_weights"], torch.tensor([[0.0, 0.2, 0.0, 1.0]]))


def _check_sp_weighted_reduce(rank, path):
    """Check SP reduction against weighted and zero-mass references."""
    import torch.distributed as dist

    from veomni.distributed.sequence_parallel.loss import reduce_sequence_parallel_loss

    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2)
    try:
        local_loss = torch.tensor(2.0 if rank == 0 else 4.0, requires_grad=True)
        local_mass = torch.tensor(0.1 if rank == 0 else 0.2)
        loss = reduce_sequence_parallel_loss(local_loss, local_mass, group=dist.group.WORLD)
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(10.0 / 3.0))
        torch.testing.assert_close(local_loss.grad, torch.tensor(2 / 3 if rank == 0 else 4 / 3))
        zero_loss = torch.tensor(0.0, requires_grad=True)
        output = reduce_sequence_parallel_loss(zero_loss, torch.tensor(0.0), group=dist.group.WORLD)
        output.backward()
        assert output.item() == zero_loss.grad.item() == 0.0
    finally:
        dist.destroy_process_group()


def test_sp_weighted_reduce_handles_mass_below_one_and_all_zero(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(_check_sp_weighted_reduce, args=(str(tmp_path / "rdzv"),), nprocs=2, start_method="spawn")


def _check_sp_causal(rank, path):
    from types import SimpleNamespace

    import torch.distributed as dist

    from veomni.models.loss_utils import cross_entropy_loss as causal
    from veomni.ops import VeomniOp

    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        all_hidden = torch.randn(4, 3)
        initial_weight = torch.randn(5, 3)
        labels_all = torch.tensor([1, 2, 3, 4])
        weights_all = torch.tensor([0.2, 0.3, 0.0, 0.1])
        local_hidden = all_hidden[rank * 2 : rank * 2 + 2].clone().requires_grad_()
        local_weight = initial_weight.clone().requires_grad_()
        local_labels = labels_all[rank * 2 : rank * 2 + 2]
        local_weights = weights_all[rank * 2 : rank * 2 + 2]
        causal.get_parallel_state = lambda: SimpleNamespace(sp_enabled=True)
        actual, _, _ = causal.ForCausalLMLoss(
            hidden_states=local_hidden,
            labels=local_labels,
            weights=local_weight,
            loss_weights=local_weights,
            op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss"),
            loss_reduction_group=dist.group.WORLD,
        )
        actual.backward()
        dist.all_reduce(local_weight.grad)
        local_weight.grad /= 2
        ref_hidden = all_hidden.clone().requires_grad_()
        ref_weight = initial_weight.clone().requires_grad_()
        expected = (
            F.cross_entropy(F.linear(ref_hidden, ref_weight), labels_all, reduction="none") * weights_all
        ).sum() / weights_all.sum()
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(local_weight.grad, ref_weight.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            local_hidden.grad / 2, ref_hidden.grad[rank * 2 : rank * 2 + 2], atol=1e-6, rtol=1e-5
        )
    finally:
        dist.destroy_process_group()


def test_sp_causal_weighted_loss_and_gradients_match_unsharded(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(_check_sp_causal, args=(str(tmp_path / "rdzv"),), nprocs=2, start_method="spawn")


def test_gradient_accumulation_uses_weight_mass_not_microbatch_count(monkeypatch):
    """Normalize accumulated loss by supervised weight, not batch count."""
    from types import SimpleNamespace

    from veomni.models.loss_utils.cross_entropy_loss import ForCausalLMLoss
    from veomni.ops import VeomniOp
    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False, fsdp_size=1))
    torch.manual_seed(21)
    h1 = torch.randn(1, 4, 3)
    h2 = torch.randn(1, 4, 3)
    labels1 = torch.tensor([[-100, 0, 1, 2]])
    labels2 = torch.tensor([[-100, 1, 2, 3]])
    w1 = torch.tensor([[0.0, 0.1, 0.2, 0.1]])
    w2 = torch.tensor([[0.0, 1.0, 0.0, 0.3]])
    batches = [{"labels": labels1, "loss_weights": w1}, {"labels": labels2, "loss_weights": w2}]
    lengths = loss_utils.count_loss_token(batches)
    param = torch.randn(5, 3, requires_grad=True)
    total = 0
    for hidden, batch in zip((h1, h2), batches, strict=True):
        local, _, _ = ForCausalLMLoss(
            hidden_states=hidden,
            weights=param,
            labels=batch["labels"],
            loss_weights=batch["loss_weights"],
            op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss"),
        )
        totals = loss_utils.mean_global_loss(
            local,
            loss_utils.count_loss_token(batch),
            lengths,
            {"foundation_tokens": lengths["foundation_tokens"].item()},
        )
        total = total + totals["foundation_loss"]
    reference = 0
    for hidden, batch in zip((h1, h2), batches, strict=True):
        targets = F.pad(batch["labels"], (0, 1), value=-100)[..., 1:]
        weight = F.pad(batch["loss_weights"], (0, 1), value=0)[..., 1:]
        token_ce = F.cross_entropy(
            F.linear(hidden, param).reshape(-1, 5), targets.reshape(-1), ignore_index=-100, reduction="none"
        )
        reference = reference + (token_ce * weight.flatten()).sum()
    reference = reference / lengths["foundation_tokens"]
    (actual_grad,) = torch.autograd.grad(total, param, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(reference, param)
    torch.testing.assert_close(total, reference)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)


def test_sp_packing_shifts_and_slices_weights_with_labels(monkeypatch):
    """Keep packed weights aligned with shifted and sharded targets."""
    from types import SimpleNamespace

    from veomni.data import data_collator as m

    def make_samples():
        return [
            {
                "input_ids": torch.tensor([11, 12, 13]),
                "labels": torch.tensor([-100, 12, 13]),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "loss_weights": torch.tensor([0.0, 0.2, 0.3]),
            },
            {
                "input_ids": torch.tensor([21, 22]),
                "labels": torch.tensor([21, 22]),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "loss_weights": torch.tensor([0.4, 0.5]),
            },
        ]

    for rank, expected_labels, expected_weights in [
        (0, [12, 13, -100, 22], [0.2, 0.3, 0.4, 0.5]),
        (1, [-100, -100, -100, -100], [0.0, 0.0, 0.0, 0.0]),
    ]:
        monkeypatch.setattr(
            m, "get_parallel_state", lambda rank=rank: SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=rank)
        )
        out = m.MainCollator(pad_to_length=8)(make_samples())
        assert out["labels"].tolist() == [expected_labels]
        torch.testing.assert_close(out["loss_weights"], torch.tensor([expected_weights]))


def test_chatml_binary_explicit_weights_keep_legacy_output_shape():
    """Keep the legacy ChatML output layout for binary supervision."""
    from types import SimpleNamespace

    from veomni.data.chat_template import ChatmlTemplate

    template = ChatmlTemplate(SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode())))
    out = template.encode_messages(
        [
            {"role": "user", "content": "q", "loss_weight": 0},
            {"role": "assistant", "content": "skip", "loss_weight": 0},
            {"role": "assistant", "content": "a", "loss_weight": 1},
        ]
    )
    assert "loss_weights" not in out
    masked = template.encode_messages([{"role": "assistant", "content": "skip", "loss_weight": 0}])
    assert all(label == -100 for label in masked["labels"])


@pytest.mark.parametrize("invalid", [-0.1, 1.01, 1e38, 1e-40, float("nan"), float("inf")])
def test_collator_rejects_invalid_token_weights(monkeypatch, invalid):
    from types import SimpleNamespace

    from veomni.data import data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    samples = [
        {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([-100, 2]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "loss_weights": torch.tensor([0.0, invalid]),
        }
    ]
    with pytest.raises(ValueError, match="loss_weights"):
        m.MainCollator()(samples)


def test_mixed_nonsp_microbatches_exclude_unsupervised_first_causal_target(monkeypatch):
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    batches = [
        {"labels": torch.tensor([7, 1, 2]), "loss_weights": torch.tensor([0.4, 0.2, 0.3])},
        {"labels": torch.tensor([5, 3, 4])},
    ]
    mass = loss_utils.count_loss_token(batches, weighted_normalization=True)["foundation_tokens"]
    torch.testing.assert_close(mass, torch.tensor(2.5))
    assert loss_utils.has_weighted_loss_in_step(batches)


def _check_dp_mixed_weighted(rank, path):
    """Compare mixed weighted and unweighted DP gradients to dense CE."""
    from types import SimpleNamespace

    import torch.distributed as dist

    from veomni.models.loss_utils import cross_entropy_loss as causal
    from veomni.ops import VeomniOp
    from veomni.utils import loss_utils

    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2)
    try:
        ps = SimpleNamespace(sp_enabled=False, fsdp_size=2)
        loss_utils.get_parallel_state = lambda: ps
        causal.get_parallel_state = lambda: ps

        def all_reduce_cpu(value, op="sum", **kwargs):
            tensor = torch.tensor(value, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            return tensor.item() if tensor.numel() == 1 else tensor.tolist()

        loss_utils.all_reduce = all_reduce_cpu
        torch.manual_seed(19)
        hidden_all = torch.randn(2, 3, 4)
        labels_all = torch.tensor([[3, 2, 1], [4, 1, 3]])
        param = torch.randn(5, 4, requires_grad=True)
        batch = {"labels": labels_all[rank : rank + 1]}
        if rank == 0:
            batch["loss_weights"] = torch.tensor([[0.3, 0.2, 0.4]])
        weighted, counts, global_counts = loss_utils.prepare_loss_token_counts([batch])
        assert weighted
        unweighted, _, unweighted_global = loss_utils.prepare_loss_token_counts(
            [{"labels": labels_all[rank : rank + 1]}]
        )
        assert not unweighted
        assert unweighted_global["foundation_tokens"] == 6
        op = VeomniOp("cross_entropy_loss", "standard", "chunk_loss")
        local, _, _ = causal.ForCausalLMLoss(
            hidden_states=hidden_all[rank : rank + 1],
            labels=batch["labels"],
            weights=param,
            op=op,
            **({"loss_weights": batch["loss_weights"]} if rank == 0 else {}),
        )
        out = loss_utils.mean_global_loss(local, counts, counts, global_counts)["foundation_loss"]
        out.backward()
        dist.all_reduce(param.grad)
        param.grad /= 2
        ref_param = param.detach().clone().requires_grad_()
        targets = labels_all[:, 1:].reshape(-1)
        logits = F.linear(hidden_all[:, :-1], ref_param).reshape(-1, 5)
        reference_weights = torch.tensor([0.2, 0.4, 1.0, 1.0])
        ref = (F.cross_entropy(logits, targets, reduction="none") * reference_weights).sum() / reference_weights.sum()
        ref.backward()
        torch.testing.assert_close(param.grad, ref_param.grad, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def test_dp_mixed_weighted_and_unweighted_matches_dense_gradient(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(_check_dp_mixed_weighted, args=(str(tmp_path / "rdzv"),), nprocs=2, start_method="spawn")


@pytest.mark.parametrize("invalid", [-1, 1.01, 1e38, 1e-100, float("nan"), float("inf")])
@pytest.mark.parametrize("loss_mask", [0, 1])
def test_chatml_rejects_out_of_range_weights(invalid, loss_mask):
    from types import SimpleNamespace

    from veomni.data.chat_template import ChatmlTemplate

    template = ChatmlTemplate(SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode())))
    with pytest.raises(ValueError, match="loss_weight"):
        template.encode_messages(
            [{"role": "assistant", "content": "x", "loss_weight": invalid, "loss_mask": loss_mask}]
        )


def test_qwen3vl_remap_preserves_weight_and_masks_header():
    from types import SimpleNamespace

    from veomni.data.chat_template import Qwen3VLChatTemplate

    template = object.__new__(Qwen3VLChatTemplate)
    template.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode()))
    template.image_token_id = -999
    template.video_token_id = -998
    messages = [
        {"role": "user", "content": "hi", "loss_mask": 0},
        {"role": "assistant", "content": "response", "loss_mask": 1, "loss_weight": 0.25},
        {"role": "assistant", "content": "skip", "loss_mask": 1, "loss_weight": 0.0},
    ]
    out = template._tokenize_and_remap(messages)
    assert out["loss_weights"].dtype == torch.float32
    assert torch.all(out["loss_weights"][out["labels"] != -100] == 0.25)
    assert torch.all(out["loss_weights"][out["labels"] == -100] == 0)
    assert "loss_weights" not in template._tokenize_and_remap(messages[:1])


def test_multimodal_preprocess_rejects_weight_instead_of_dropping_it():
    from veomni.data.multimodal.preprocess import conv_preprocess

    with pytest.raises(ValueError, match="not supported"):
        conv_preprocess("doom", [{"from": "gpt", "value": "x", "loss_weight": 0.25}])


def test_multimodal_preprocess_nested_numpy_and_nullable_weights():
    import numpy as np

    from veomni.data.multimodal.preprocess import conv_preprocess

    records = np.array([{"from": "gpt", "value": "x", "loss_weight": 0.25}], dtype=object)
    with pytest.raises(ValueError, match="not supported"):
        conv_preprocess("doom", {"messages": None, "conversations": records})
    plain = [{"from": "human", "value": "q", "loss_weight": None}, {"from": "gpt", "value": "a"}]
    assert conv_preprocess("sharegpt4v_sft", plain) == [["user", ("text", "q")], ["assistant", ("text", "a")]]


def test_weighted_chunk_with_projected_logit_bias_and_gradients():
    """Match dense loss and gradients when the output projection has bias."""
    torch.manual_seed(7)
    hidden = torch.randn(5, 4, requires_grad=True)
    head = torch.randn(7, 4, requires_grad=True)
    bias = torch.randn(5, 3, requires_grad=True)
    bias_head = torch.randn(7, 3, requires_grad=True)
    labels = torch.tensor([1, 2, -100, 4, 3])
    weights = torch.tensor([0.25, 1.0, 0.0, 0.5, 1.0])
    op = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper
    actual = op(
        hidden, labels, head, logits_bias=bias, logits_bias_weight=bias_head, loss_weights=weights, chunk_size=2
    )
    params = (hidden, head, bias, bias_head)
    actual_grads = torch.autograd.grad(actual, params)
    logits = F.linear(hidden, head) + F.linear(bias, bias_head)
    per_token = F.cross_entropy(logits, labels, ignore_index=-100, reduction="none")
    expected = (per_token * weights).sum() / weights.sum()
    expected_grads = torch.autograd.grad(expected, params)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want, atol=2e-7, rtol=1e-5)


def test_weighted_chunk_rejects_ambiguous_external_denominator():
    op = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper
    with pytest.raises(ValueError, match="num_items_in_batch"):
        op(
            torch.randn(2, 4),
            torch.tensor([1, 2]),
            torch.randn(7, 4),
            loss_weights=torch.tensor([0.25, 1.0]),
            num_items_in_batch=2,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_weights_use_fp32_sum_on_long_sequence(dtype):
    op = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper
    hidden = torch.zeros(65536, 1)
    head = torch.tensor([[0.0], [1.0]])
    labels = torch.zeros(65536, dtype=torch.long)
    weights = torch.ones(65536, dtype=dtype)
    result = op(hidden, labels, head, loss_weights=weights, chunk_size=2048)
    torch.testing.assert_close(result, torch.tensor(2.0).log())


def test_collator_keeps_weights_and_mtp_labels_for_trainer_rejection(monkeypatch):
    """Retain both fields so the trainer can reject the unsupported pair."""
    from types import SimpleNamespace

    from veomni.data import data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    feature = {
        "input_ids": torch.tensor([1, 2]),
        "labels": torch.tensor([-100, 2]),
        "attention_mask": torch.ones(2, dtype=torch.long),
        "position_ids": torch.arange(2),
        "mtp_labels": torch.tensor([[-100, -100]]),
        "loss_weights": torch.tensor([0.0, 0.25]),
    }
    result = m.MainCollator()([feature])
    assert "loss_weights" in result and "mtp_labels" in result


@pytest.mark.parametrize("template_cls", ["Qwen2VLChatTemplate", "Qwen3VLChatTemplate"])
def test_sharegpt_vlm_preserves_turn_masks_and_fractional_weights(template_cls):
    from types import SimpleNamespace

    from veomni.data.chat_template import Qwen2VLChatTemplate, Qwen3VLChatTemplate
    from veomni.data.multimodal.preprocess import conv_preprocess

    template = object.__new__(
        {"Qwen2VLChatTemplate": Qwen2VLChatTemplate, "Qwen3VLChatTemplate": Qwen3VLChatTemplate}[template_cls]
    )
    template.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode()))
    template.image_token_id = -999
    template.video_token_id = -998
    template.image_pattern = lambda count: "I" * count
    messages = [
        {"from": "human", "value": "<image>q", "loss_mask": 0},
        {"from": "gpt", "value": "downweighted", "loss_mask": 1, "loss_weight": 0.25},
        {"from": "gpt", "value": "untrained", "loss_mask": 0, "loss_weight": 0.5},
        {"from": "gpt", "value": "full", "loss_mask": 1, "loss_weight": 1.0},
    ]
    conversations = conv_preprocess("sharegpt4v_sft", messages, preserve_loss_supervision=True)
    result = template.encode_messages(conversations, num_tokens={"image": [3]})
    weights, labels = result["loss_weights"], result["labels"]
    assert ((weights == 0.25) & (labels != -100)).any()
    assert ((weights == 1.0) & (labels != -100)).any()
    assert torch.all(weights[labels == -100] == 0)
    assert messages[1]["loss_weight"] == 0.25
    masked_messages = [{"from": "human", "value": "q", "loss_mask": 0}, messages[2]]
    masked_only = conv_preprocess("sharegpt4v_sft", masked_messages, preserve_loss_supervision=True)
    masked_result = template.encode_messages(masked_only)
    assert "loss_weights" not in masked_result
    assert torch.all(masked_result["labels"] == -100)
    unweighted_messages = [
        masked_messages[0],
        {key: value for key, value in messages[2].items() if key != "loss_weight"},
    ]
    baseline = template.encode_messages(
        conv_preprocess("sharegpt4v_sft", unweighted_messages, preserve_loss_supervision=True)
    )
    assert torch.equal(masked_result["input_ids"], baseline["input_ids"])
    assert torch.equal(masked_result["labels"], baseline["labels"])


def test_chatml_masked_fractional_weight_is_excluded():
    """A mask excludes a valid weight instead of rejecting the sample."""
    from types import SimpleNamespace

    from veomni.data.chat_template import ChatmlTemplate

    template = object.__new__(ChatmlTemplate)
    template.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode()))
    messages = [
        {"role": "assistant", "content": "masked", "loss_mask": 0, "loss_weight": 0.5},
        {"role": "assistant", "content": "trained", "loss_mask": 1, "loss_weight": 0.25},
    ]
    masked = template.encode_messages(messages[:1])
    assert all(label == -100 for label in masked["labels"])
    assert "loss_weights" not in masked
    combined = template.encode_messages(messages)
    assert all(weight == 0 for weight in combined["loss_weights"][: len(masked["labels"])])
    assert all(weight == 0.25 for weight in combined["loss_weights"][len(masked["labels"]) :])


def test_qwen_vl_standard_transform_keeps_weighted_assistant_turns(monkeypatch):
    from types import SimpleNamespace

    from veomni.data import data_transform
    from veomni.data.chat_template import Qwen3VLChatTemplate
    from veomni.data.multimodal import image_utils

    template = object.__new__(Qwen3VLChatTemplate)
    template.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(text.encode()))
    template.image_token_id = -999
    template.video_token_id = -998
    template.image_pattern = lambda count: "I" * count
    monkeypatch.setattr(image_utils, "fetch_images", lambda images, **kwargs: images)

    class ImageProcessor:
        merge_size = 1

        def __call__(self, **kwargs):
            return {"image_grid_thw": torch.tensor([[1, 1, 2]]), "pixel_values": torch.ones(2, 2)}

    def position_ids(**kwargs):
        length = kwargs["input_ids"].shape[-1]
        return {"position_ids": torch.arange(length).view(1, 1, length).expand(1, 3, length)}

    sample = {
        "source": "sharegpt4v_sft",
        "images": ["dummy"],
        "conversations": [
            {"from": "human", "value": "<image>q"},
            {"from": "gpt", "value": "weighted", "loss_weight": 0.25},
            {"from": "gpt", "value": "masked", "loss_mask": 0},
        ],
    }
    result = data_transform._process_sample_qwen_vl_base(
        sample, SimpleNamespace(image_processor=ImageProcessor()), template, position_ids
    )[0]
    labels, weights = result["labels"], result["loss_weights"]
    assert ((weights == 0.25) & (labels != -100)).any()
    assert torch.all(weights[labels == -100] == 0)


def test_multimodal_supervision_is_scoped_to_opted_in_consumer():
    from veomni.data.multimodal.preprocess import conv_preprocess

    plain = [{"from": "human", "value": "q", "loss_mask": 0}, {"from": "gpt", "value": "a", "loss_mask": 0}]
    assert conv_preprocess("sharegpt4v_sft", plain) == [["user", ("text", "q")], ["assistant", ("text", "a")]]
    with pytest.raises(ValueError, match="not supported"):
        conv_preprocess("sharegpt4v_sft", [{"from": "gpt", "value": "a", "loss_weight": 0.25}])


def test_weighted_loss_rejects_unsupported_models(monkeypatch):
    from veomni.utils.loss_utils import validate_weighted_loss_model

    validate_weighted_loss_model(True, "qwen4_exp")
    validate_weighted_loss_model(False, "qwen3_vl")
    with pytest.raises(ValueError, match="VeOmni qwen4_exp"):
        validate_weighted_loss_model(True, "qwen3_vl")
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    with pytest.raises(ValueError, match="VeOmni qwen4_exp"):
        validate_weighted_loss_model(True, "qwen4_exp")


def test_weighted_router_aux_rejected_consistently_before_forward(monkeypatch):
    """Reject weighted router loss before executing the model."""
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    seen = []

    def collective(value, op):
        seen.append((value, op))
        return [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # another rank requested router logits

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(loss_utils, "all_reduce", collective)
    with pytest.raises(ValueError, match="router auxiliary"):
        loss_utils.prepare_loss_token_counts(
            [{"labels": torch.tensor([[1, 2]]), "loss_weights": torch.tensor([[0.0, 0.0]])}]
        )
    assert len(seen) == 1 and seen[0][1] == "sum"


def test_weighted_qwen4_direct_forward_rejects_router_aux_before_model_execution():
    from veomni.models.transformers.qwen4_exp.qwen4_exp_gpu_patch_gen_config import (
        qwen4_exp_for_conditional_generation_forward_patched,
    )

    with pytest.raises(ValueError, match="router auxiliary"):
        qwen4_exp_for_conditional_generation_forward_patched(
            object(), labels=torch.tensor([[1]]), loss_weights=torch.tensor([[0.0]]), output_router_logits=True
        )


def _check_weighted_router_agreement(rank, path):
    """Check coordinated rejection of incompatible per-rank batch metadata."""
    from datetime import timedelta
    from types import SimpleNamespace

    import torch.distributed as dist

    from veomni.utils import loss_utils

    dist.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        loss_utils.get_parallel_state = lambda: SimpleNamespace(sp_enabled=False)

        def all_reduce_cpu(value, op="sum", **kwargs):
            tensor = torch.tensor(value, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.MAX)
            return tensor.item() if tensor.numel() == 1 else tensor.tolist()

        loss_utils.all_reduce = all_reduce_cpu
        batch = {"labels": torch.tensor([[1, 2]])}
        if rank == 0:
            batch["loss_weights"] = torch.tensor([[0.0, 0.0]])
        else:
            batch["output_router_logits"] = True
        with pytest.raises(ValueError, match="router auxiliary"):
            loss_utils.prepare_loss_token_counts([batch])
        batch = {"labels": torch.tensor([[1, 2]])}
        if rank == 0:
            batch["loss_weights"] = torch.tensor([[0.5, 0.5]])
        else:
            batch["mtp_labels"] = torch.tensor([[[1, -100]]])
        with pytest.raises(ValueError, match="MTP labels"):
            loss_utils.prepare_loss_token_counts([batch])
        batches = [
            {"labels": torch.tensor([[1, 2]]), "loss_weights": torch.tensor([[0.5, 0.5]])},
            {"labels": torch.tensor([[1, 2]]), "mtp_labels": torch.tensor([[[1, -100]]])},
        ]
        with pytest.raises(ValueError, match="MTP labels"):
            loss_utils.prepare_loss_token_counts(batches)
        uncollated = [{"labels": torch.tensor([1, 2])}]
        if rank == 0:
            uncollated[0]["loss_weights"] = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="uncollated list-valued micro-batches"):
            loss_utils.prepare_loss_token_counts([uncollated])
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2
        from veomni.data import data_collator

        data_collator.get_parallel_state = lambda: SimpleNamespace(sp_enabled=False)

        def feature():
            return {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([-100, 2]),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "position_ids": torch.arange(2),
            }

        first, second = feature(), feature()
        if rank == 0:
            first["loss_weights"] = torch.tensor([0.0, 0.5])
            second["mtp_labels"] = torch.tensor([[-100, 2]])
        batch = data_collator.MainCollator()([first, second])
        if rank == 0:
            assert batch["mtp_labels"].shape[-1] == batch["labels"].shape[-1]
        with pytest.raises(ValueError, match="MTP labels"):
            loss_utils.prepare_loss_token_counts([batch])
        from veomni.models.model_runtime import VeOmniModelRuntime

        class WeightedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.veomni_ce = SimpleNamespace(impl="chunk_loss")

            def forward(self, value):
                return value * self.weight

        wrapped = torch.nn.parallel.DistributedDataParallel(WeightedModel())
        real_runtime = object.__new__(VeOmniModelRuntime)
        real_runtime.model = wrapped
        real_runtime.model_config = SimpleNamespace(model_type="qwen4_exp")
        batch = {"labels": torch.tensor([[1, 2]]), "loss_weights": torch.tensor([[0.5, 0.5]])}
        weighted, _, _ = loss_utils.prepare_loss_token_counts([batch], real_runtime)
        assert weighted
    finally:
        dist.destroy_process_group()


def test_weighted_router_aux_error_is_agreed_across_ranks(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _check_weighted_router_agreement, args=(str(tmp_path / "router_rdzv"),), nprocs=2, start_method="spawn"
    )


def test_prepare_counts_uses_one_collective_for_unweighted(monkeypatch):
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    calls = []

    def reduce_once(value, op):
        calls.append((value, op))
        return value

    monkeypatch.setattr(loss_utils, "all_reduce", reduce_once)
    weighted, local, global_counts = loss_utils.prepare_loss_token_counts([{"labels": torch.tensor([[3, 4, -100]])}])
    assert not weighted
    assert local["foundation_tokens"].item() == global_counts["foundation_tokens"] == 2
    assert len(calls) == 1


def test_unweighted_scalar_label_keeps_legacy_count(monkeypatch):
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(loss_utils, "all_reduce", lambda value, op: value)
    weighted, local, totals = loss_utils.prepare_loss_token_counts([{"labels": torch.tensor(1)}])
    assert not weighted
    assert local["foundation_tokens"].item() == totals["foundation_tokens"] == 1


def test_prepare_counts_handles_uncollated_rl_micro_batches(monkeypatch):
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(loss_utils, "all_reduce", lambda value, op: value)
    batches = [[{"labels": torch.tensor([-100, 2, 3])}, {"labels": torch.tensor([-100, 4])}]]
    weighted, local, global_counts = loss_utils.prepare_loss_token_counts(batches)
    assert not weighted
    assert local["foundation_tokens"].item() == global_counts["foundation_tokens"] == 3
    assert not loss_utils.has_weighted_loss_in_step(batches)


def test_prepare_counts_rejects_weighted_uncollated_micro_batches(monkeypatch):
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(loss_utils, "all_reduce", lambda value, op: value)
    batches = [[{"labels": torch.tensor([-100, 2]), "loss_weights": torch.tensor([0.0, 0.5])}]]
    assert loss_utils.has_weighted_loss_in_step(batches)
    with pytest.raises(ValueError, match="uncollated list-valued micro-batches"):
        loss_utils.prepare_loss_token_counts(batches)


def test_prepare_counts_checks_weighted_model_locally(monkeypatch):
    from types import SimpleNamespace

    from veomni.utils import loss_utils

    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(loss_utils, "all_reduce", lambda value, op: value)
    batch = {"labels": torch.tensor([1, 2]), "loss_weights": torch.tensor([1.0, 0.5])}
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(model_type="qwen4_exp"),
        unwrapped_module=SimpleNamespace(veomni_ce=SimpleNamespace(impl="eager")),
    )
    with pytest.raises(ValueError, match="qwen4_exp backend with chunk_loss"):
        loss_utils.prepare_loss_token_counts([batch], runtime)
