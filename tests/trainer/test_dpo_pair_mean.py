# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Check actual DPO optimizer gradients against the same global pair mean."""

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel
from transformers.utils import ModelOutput

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.data.data_collator import MainCollator, PostCollator
from veomni.data.data_transform import dpo_producer_identity, process_dpo_example
from veomni.data.dpo_collator import DPOProvenanceCollator
from veomni.distributed.clip_grad_norm import _allreduce_ddp_sp_grads
from veomni.trainer import text_dpo_trainer as dpo
from veomni.utils.device import IS_NPU_AVAILABLE
from veomni.utils.dpo_provenance import DPO_FEATURE_KEY


@dataclass
class _AuxOutput(ModelOutput):
    log_probs: torch.Tensor | None = None


@dataclass
class _ModelOutput(ModelOutput):
    fused_linear_aux: _AuxOutput | None = None
    marker: int | None = None


class _TinyModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, input_ids, **kwargs):
        return _ModelOutput(fused_linear_aux=_AuxOutput(log_probs=input_ids.float() * self.weight), marker=0)


class _FSDPTinyModel(_TinyModel):
    def __init__(self, value):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor([value, 0.0, 0.0, 0.0], dtype=torch.float32))

    def forward(self, input_ids, **kwargs):
        return _ModelOutput(fused_linear_aux=_AuxOutput(log_probs=input_ids.float() * self.weight[0]), marker=0)


class _CaptureOptimizer:
    def __init__(self, model, *, require_dtensor=False):
        self.model = model
        self.require_dtensor = require_dtensor
        self.gradient = None
        self.gradient_vector = None
        self.steps = 0

    def step(self):
        gradient = self.model.weight.grad
        if self.require_dtensor:
            assert isinstance(gradient, DTensor), "FSDP test requires an actually sharded DTensor gradient"
        if isinstance(gradient, DTensor):
            gradient = gradient.full_tensor()
        if gradient.ndim:
            self.gradient_vector = gradient.detach().clone()
            gradient = gradient[0]
        self.gradient = gradient.detach().clone()
        self.steps += 1

    def zero_grad(self):
        pass


class _Runtime:
    def __init__(self, value, state, *, policy, distributed, fsdp_mesh=None, device="cpu"):
        self.parallel_state = state
        self.model_config = SimpleNamespace(vocab_size=256)
        self.module = _FSDPTinyModel(value) if policy and fsdp_mesh is not None else _TinyModel(value)
        self.module.to(device)
        if policy and fsdp_mesh is not None:
            from torch.distributed._composable.fsdp import fully_shard

            self.model = fully_shard(self.module, mesh=fsdp_mesh)
        elif policy and distributed:
            self.model = DistributedDataParallel(self.module, process_group=getattr(state, "dp_group", None))
        else:
            self.model = self.module
        self.fsdp = fsdp_mesh is not None and policy
        if policy:
            self.optimizer = _CaptureOptimizer(self.module, require_dtensor=self.fsdp)
            self.lr_scheduler = SimpleNamespace(step=lambda: None)

    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def clip_grad_norm(self):
        if self.fsdp:
            return torch.tensor(0.0)
        if self.parallel_state.sp_enabled:
            _allreduce_ddp_sp_grads(self.model, self.parallel_state)
        return self.module.weight.grad.detach().norm()


def _microbatch(coefficients, sp_size=1, sp_rank=0):
    ids, labels = [], []
    for value in coefficients:
        ids.extend((max(value, 0), 0, max(-value, 0), 0))
        labels.extend((-100, 1, -100, 1))
    ids = torch.tensor([ids], dtype=torch.long)
    labels = torch.tensor([labels], dtype=torch.long)
    cu = torch.arange(0, ids.shape[-1] + 1, 2, dtype=torch.int32)
    if sp_size > 1:
        labels = F.pad(labels[..., 1:], (0, 1), value=-100)
        width = ids.shape[-1] // sp_size
        ids = ids[:, sp_rank * width : (sp_rank + 1) * width]
        labels = labels[:, sp_rank * width : (sp_rank + 1) * width]
    return {"input_ids": ids, "labels": labels, "cu_seq_lens_q": cu}


def _build_trainer(state, *, distributed, monkeypatch=None, fsdp_mesh=None, device="cpu"):
    set_attribute = setattr if monkeypatch is None else monkeypatch.setattr
    set_attribute(dpo, "get_parallel_state", lambda: state)
    set_attribute(dpo, "use_parallel_state", lambda *_: nullcontext())
    set_attribute(dpo, "set_batch_invariant_mode", lambda *_: nullcontext())
    set_attribute(dpo, "mark_compile_step_begin", lambda *_: None)
    trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
    cfg = SimpleNamespace(
        explicit_segment_provenance=False,
        average_log_prob=False,
        beta=0.1,
        label_smoothing=0.0,
        loss_type="sigmoid",
        reference_free=False,
        refer_model_precision="bfloat16",
    )
    captured = {}
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(dpo_config=cfg, train=SimpleNamespace(enable_batch_invariant_mode=False)),
        state=SimpleNamespace(global_step=0, epoch=0),
        _callbacks=[],
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
        preforward=lambda batch: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
        _reset_async_activation_offload_if_enabled=lambda *_: None,
        on_step_begin=lambda **kwargs: None,
        sync_before_train_step=lambda: None,
        model_reshard=lambda *args: None,
        _configure_hsdp_allreduce=lambda *args: None,
        on_step_end=lambda **kwargs: captured.update(kwargs),
    )
    trainer.policy_model = _Runtime(
        -0.7, state, policy=True, distributed=distributed, fsdp_mesh=fsdp_mesh, device=device
    )
    trainer.reference_model = _Runtime(0.0, state, policy=False, distributed=distributed, device=device)
    trainer.post_forward = PostCollator()
    return trainer, captured


def _expected(coefficients):
    weight = torch.tensor(-0.7, requires_grad=True)
    rewards = torch.tensor(coefficients, dtype=torch.float32) * weight * 0.1
    loss = -F.logsigmoid(rewards).mean()
    loss.backward()
    return loss.detach(), weight.grad.detach(), rewards.detach().mean()


def _run(trainer, groups, state):
    batches = [_microbatch(group, state.sp_size, state.sp_rank if state.sp_enabled else 0) for group in groups]
    trainer.train_step(iter([batches]))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("groups", [[[1, 3, 3, 3]], [[1], [3, 3, 3]]])
def test_single_process_pair_weighted_gradient_and_metrics(groups, monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    _run(trainer, groups, state)
    expected_loss, expected_grad, expected_chosen = _expected([1, 3, 3, 3])
    torch.testing.assert_close(trainer.policy_model.optimizer.gradient, expected_grad)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-7)
    assert metrics["loss_dict"]["chosen_rewards"] == pytest.approx(expected_chosen.item(), abs=1e-7)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_reward_accuracy_is_pair_weighted_not_microbatch_weighted(monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    _run(trainer, [[-2], [1, 3, 3]], state)
    assert metrics["loss_dict"]["reward_accuracy"] == pytest.approx(0.25, abs=1e-7)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("accepts_kwargs", [False, True])
def test_legacy_backward_override_fails_before_optimizer(monkeypatch, accepts_kwargs):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    if accepts_kwargs:
        trainer.forward_backward_step = lambda micro_batch, **kwargs: None
    else:
        trainer.forward_backward_step = lambda micro_batch: None
    with pytest.raises(RuntimeError, match="must accept pair_normalization"):
        _run(trainer, [[1]], state)
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_unknown_tail_and_malformed_labels_fail_before_optimizer(monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    batch = _microbatch([1])
    batch["labels"] = batch["labels"][:, :-1]
    with pytest.raises(RuntimeError, match="same nonempty"):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0
    batch = _microbatch([1])
    batch["input_ids"] = F.pad(batch["input_ids"], (0, 2))
    batch["labels"] = F.pad(batch["labels"], (0, 2), value=-100)
    batch["cu_seq_lens_q"] = torch.tensor([0, 2, 4, 5, 6], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="trailing one-token segments are ambiguous"):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    "case", ["float_ids", "negative_ids", "negative_labels", "ids_outside_vocab", "labels_outside_vocab"]
)
def test_invalid_token_values_fail_before_optimizer(monkeypatch, case):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    batch = _microbatch([1])
    if case == "float_ids":
        batch["input_ids"] = batch["input_ids"].float()
    elif case == "negative_ids":
        batch["input_ids"][0, 0] = -1
    elif case == "negative_labels":
        batch["labels"][0, 0] = -2
    elif case == "ids_outside_vocab":
        batch["input_ids"][0, 0] = 256
    else:
        batch["labels"][0, 0] = 256
    with pytest.raises(RuntimeError, match="token ID"):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("case", ["key_boundaries", "linear_boundaries", "maximum", "attention_shape"])
def test_inconsistent_attention_metadata_fails_before_optimizer(monkeypatch, case):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    batch = _microbatch([1])
    if case == "key_boundaries":
        batch["cu_seq_lens_k"] = torch.tensor([0, 1, 4], dtype=torch.int32)
    elif case == "linear_boundaries":
        batch["linear_attn_cu_seq_lens_q"] = torch.tensor([0, 1, 4], dtype=torch.int32)
    elif case == "maximum":
        batch["max_length_q"] = 3
    else:
        batch["attention_mask"] = torch.ones((1, 3), dtype=torch.long)
    with pytest.raises(RuntimeError):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_unproven_prepadded_feature_fails_before_optimizer(monkeypatch):
    import veomni.data.data_collator as collator_module

    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    feature = {
        "input_ids": torch.tensor([11, 12, 13, 21, 22, 23, 0, 0]),
        "labels": torch.tensor([11, 12, 13, 21, 22, 23, -100, -100]),
        "position_ids": torch.tensor([0, 1, 2, 0, 1, 2, 0, 0]),
        "attention_mask": torch.ones(8, dtype=torch.long),
    }
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer.base.args.dpo_config.explicit_segment_provenance = True
    trainer.base.args.data = SimpleNamespace(enable_multisource=False)
    batch = DPOProvenanceCollator(MainCollator())([feature])
    with pytest.raises(RuntimeError, match="provenance"):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_standard_zero_tail_rejects_unscoreable_one_token_pair(monkeypatch):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        dpo_unpadded_output_contract = True
        dpo_unpadded_output_contract_id = "test-tokenizer-v1"

        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    features = [
        process_dpo_example(
            {"chosen": chosen, "rejected": rejected}, tokenizer=Tokenizer(), record_segment_provenance=True
        )[0]
        for chosen, rejected in [("Bx", "Bx"), ("B", "B")]
    ]
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer.base.args.dpo_config.explicit_segment_provenance = True
    trainer.base.args.data = SimpleNamespace(enable_multisource=False)
    batch = DPOProvenanceCollator(MainCollator())(features)
    assert batch["cu_seq_lens_q"].tolist() == [0, 2, 4, 5, 6]
    with pytest.raises(RuntimeError, match="causal prediction target"):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_exhausted_dpo_dataloader_does_not_advance_checkpoint_step(monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer.base.args.dpo_config.explicit_segment_provenance = True
    trainer.base.args.data = SimpleNamespace(enable_multisource=False)
    trainer.state.global_step = 5
    with pytest.raises(StopIteration):
        trainer.train_step(iter([]))
    assert trainer.state.global_step == 5
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_short_dpo_epochs_save_actual_epoch_positions(monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer.base.args.train.local_rank = 0
    trainer.base.args.train.num_train_epochs = 2
    trainer.base.args.train_steps = 3
    trainer.base.args.data = SimpleNamespace(
        enable_multisource=False, dataloader=SimpleNamespace(use_background_prefetcher=False, drop_last=True)
    )
    trainer.start_epoch, trainer.start_step = 0, 0
    trainer.base.train_dataloader = [[_microbatch([1])]]
    trainer.base.destroy_distributed = lambda: None
    positions = []
    trainer.on_train_begin = trainer.on_train_end = trainer.on_epoch_begin = lambda: None
    trainer.on_epoch_end = lambda: positions.append(
        (trainer.state.epoch, trainer._dpo_epoch_step, trainer.state.global_step, trainer._dpo_epoch_complete)
    )
    monkeypatch.setattr(dpo, "synchronize", lambda: None)
    monkeypatch.setattr(dpo.helper, "print_device_mem_info", lambda *_: None)
    trainer.train()
    assert positions == [(0, 1, 1, True), (1, 1, 2, True)]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_dpo_resume_replays_pending_epoch_end_with_real_progress_callback(monkeypatch):
    from veomni.trainer.callbacks.trace_callback import TqdmCallback

    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer.base.args.train.local_rank = 0
    trainer.base.args.train.num_train_epochs = 1
    trainer.base.args.train_steps = 3
    trainer.base.args.data = SimpleNamespace(
        enable_multisource=False, dataloader=SimpleNamespace(use_background_prefetcher=True, drop_last=True)
    )
    trainer.start_epoch, trainer.start_step = 1, 0
    trainer._dpo_pending_epoch_end = (0, 3)
    trainer.state.global_step = 3
    trainer.base.train_dataloader = []
    trainer.base.destroy_distributed = lambda: None
    trainer.on_train_begin = trainer.on_train_end = lambda: None
    progress = TqdmCallback(trainer)
    seen = []

    def epoch_end():
        progress.on_epoch_end(trainer.state)
        seen.append((trainer.state.epoch, trainer._dpo_epoch_step, trainer._dpo_epoch_complete))

    trainer.on_epoch_end = epoch_end
    monkeypatch.setattr(dpo, "synchronize", lambda: None)
    trainer.train()
    assert seen == [(0, 3, True)]
    assert trainer._dpo_pending_epoch_end is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_dpo_rejects_duplicated_padding_microbatch_before_optimizer(monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    batch = _microbatch([1])
    batch["padding_flag"] = True
    with pytest.raises((ValueError, RuntimeError), match="duplicated padding microbatches"):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_known_nonzero_collator_tail_is_trimmed_without_sp(monkeypatch):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        dpo_unpadded_output_contract = True
        dpo_unpadded_output_contract_id = "test-tokenizer-v1"

        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    features = [process_dpo_example({"chosen": "Bx", "rejected": "Ax"}, tokenizer=Tokenizer())[0]]
    results = []
    for collator in (MainCollator(), MainCollator(pad_to_length=8)):
        trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
        trainer.train_step(iter([[collator(features)]]))
        results.append((trainer.policy_model.optimizer.gradient.clone(), metrics["loss"]))
    torch.testing.assert_close(results[0][0], results[1][0])
    assert results[0][1] == pytest.approx(results[1][1], abs=1e-7)


def _real_collator_sp_worker(rank, world_size, endpoint):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        dpo_unpadded_output_contract = True
        dpo_unpadded_output_contract_id = "test-tokenizer-v1"

        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=world_size)
    try:
        dp_groups = [dist.new_group([0]), dist.new_group([1])]
        state = SimpleNamespace(
            sp_enabled=True,
            sp_size=2,
            sp_rank=rank,
            sp_group=dist.group.WORLD,
            dp_group=dp_groups[rank],
            fsdp_group=dist.group.WORLD,
        )
        collator_module.get_parallel_state = lambda: state
        trainer, _ = _build_trainer(state, distributed=True)
        trainer.base.args.dpo_config.explicit_segment_provenance = True
        trainer.base.args.data = SimpleNamespace(enable_multisource=False)
        features = [
            process_dpo_example(
                {"chosen": chosen, "rejected": rejected}, tokenizer=Tokenizer(), record_segment_provenance=True
            )[0]
            for chosen, rejected in [("Bx", "Bx"), ("B", "B")]
        ]
        batch = DPOProvenanceCollator(MainCollator())(features)
        assert batch["cu_seq_lens_q"].tolist() == [0, 2, 4, 5, 6]
        with pytest.raises(RuntimeError, match="causal prediction target"):
            trainer.train_step(iter([[batch]]))
        assert trainer.policy_model.optimizer.steps == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_standard_zero_tail_rejects_unscoreable_one_token_pair_with_sp():
    spawn_with_rendezvous(_real_collator_sp_worker, args=(2,), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_legacy_callback_can_replace_valid_batches_before_weighting(monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer.base.on_step_begin = lambda micro_batches, **kwargs: micro_batches.__setitem__(
        0, _microbatch([1, 3, 3, 3])
    )
    _run(trainer, [[1]], state)
    expected_loss, expected_grad, _ = _expected([1, 3, 3, 3])
    torch.testing.assert_close(trainer.policy_model.optimizer.gradient, expected_grad)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-7)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_provenance_enabled_uneven_microbatches_match_pair_mean(monkeypatch):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        dpo_unpadded_output_contract = True
        dpo_unpadded_output_contract_id = "test-tokenizer-v1"

        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer.base.args.dpo_config.explicit_segment_provenance = True
    trainer.base.args.data = SimpleNamespace(enable_multisource=False)
    trainer.policy_model.model_config = SimpleNamespace(vocab_size=256)
    collator = DPOProvenanceCollator(MainCollator(pad_to_length=24))

    def feature(coefficient):
        return process_dpo_example(
            {"chosen": "!" + chr(ord("A") + coefficient) + "x", "rejected": "!Ax"},
            tokenizer=Tokenizer(),
            record_segment_provenance=True,
        )[0]

    batches = [collator([feature(1)]), collator([feature(3), feature(3), feature(3)])]
    trainer.train_step(iter([batches]))
    expected_loss, expected_grad, _ = _expected([1, 3, 3, 3])
    torch.testing.assert_close(trainer.policy_model.optimizer.gradient, expected_grad, rtol=1e-6, atol=1e-7)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)


def _init_test_group(rank, world_size, native_npu, endpoint):
    if native_npu:
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
    device = f"npu:{rank}" if native_npu else "cpu"
    dist.init_process_group(
        "hccl" if native_npu else "gloo",
        store=endpoint.connect(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    return device


def _distributed_worker(rank, world_size, scenario, native_npu, endpoint):
    device = _init_test_group(rank, world_size, native_npu, endpoint)
    try:
        if scenario == "dp2_sp2":
            groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
            dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
            state = SimpleNamespace(
                sp_enabled=True,
                sp_size=2,
                sp_rank=rank % 2,
                sp_group=groups[rank // 2],
                dp_group=dp_groups[rank % 2],
                fsdp_group=dist.group.WORLD,
            )
            batches = [[1], [2]] if rank < 2 else [[3, 3], [4, 4, 4]]
            expected = [1, 2, 3, 3, 4, 4, 4]
        else:
            state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0, fsdp_group=dist.group.WORLD)
            batches = [[1], [2]] if rank == 0 else [[3, 3], [4, 4, 4]]
            expected = [1, 2, 3, 3, 4, 4, 4]
        trainer, metrics = _build_trainer(state, distributed=True, device=device)
        _run(trainer, batches, state)
        expected_loss, expected_grad, expected_chosen = _expected(expected)
        torch.testing.assert_close(trainer.policy_model.optimizer.gradient.cpu(), expected_grad, rtol=1e-6, atol=1e-7)
        assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
        assert metrics["loss_dict"]["chosen_rewards"] == pytest.approx(expected_chosen.item(), abs=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=lambda context: LaunchSpec(device="cpu", nproc=context.params["world_size"]))
@pytest.mark.parametrize(("world_size", "scenario"), [(2, "dp2"), (4, "dp2_sp2")])
def test_distributed_unequal_pair_counts_match_global_reference(world_size, scenario):
    spawn_with_rendezvous(_distributed_worker, args=(world_size, scenario, False), nprocs=world_size)


def _provenance_dp_sp_worker(rank, world_size, native_npu, endpoint):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        dpo_unpadded_output_contract = True
        dpo_unpadded_output_contract_id = "test-tokenizer-v1"

        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    device = _init_test_group(rank, world_size, native_npu, endpoint)
    try:
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        dp_rank = rank // 2
        state = SimpleNamespace(
            sp_enabled=True,
            sp_size=2,
            sp_rank=rank % 2,
            sp_group=groups[dp_rank],
            dp_group=dp_groups[rank % 2],
            fsdp_group=dist.group.WORLD,
        )
        collator_module.get_parallel_state = lambda: state
        trainer, metrics = _build_trainer(state, distributed=True, device=device)
        trainer._dpo_provenance_required = True
        trainer.base.args.dpo_config.explicit_segment_provenance = True
        trainer._dpo_objective_parameters = dpo._dpo_objective_parameters(trainer.base.args.dpo_config)
        trainer.base.args.data = SimpleNamespace(enable_multisource=False)
        collator = DPOProvenanceCollator(MainCollator())

        def feature(coefficient):
            return process_dpo_example(
                {"chosen": "!" + chr(ord("A") + coefficient) + "x", "rejected": "!Ax"},
                tokenizer=Tokenizer(),
                record_segment_provenance=True,
            )[0]

        microbatches = [[1], [3, 3]] if dp_rank == 0 else [[2], [4, 5, 6]]
        trainer.train_step(iter([[collator([feature(value) for value in group]) for group in microbatches]]))
        expected_loss, expected_grad, _ = _expected([1, 3, 3, 2, 4, 5, 6])
        torch.testing.assert_close(trainer.policy_model.optimizer.gradient.cpu(), expected_grad, rtol=1e-5, atol=1e-6)
        assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_provenance_enabled_unequal_pairs_with_dp_and_sp():
    spawn_with_rendezvous(_provenance_dp_sp_worker, args=(4, False), nprocs=4)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=4))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 4, reason="requires four Ascend NPUs and HCCL")
def test_native_hccl_provenance_dp_sp_pair_weighting():
    spawn_with_rendezvous(_provenance_dp_sp_worker, args=(4, True), nprocs=4)


def _fsdp_worker(rank, world_size, hsdp, sp, native_npu, endpoint):
    from torch.distributed.device_mesh import init_device_mesh

    device = _init_test_group(rank, world_size, native_npu, endpoint)
    try:
        shape = (2, world_size // 2) if hsdp else (world_size,)
        names = ("replicate", "shard") if hsdp else ("shard",)
        mesh = init_device_mesh("npu" if native_npu else "cpu", shape, mesh_dim_names=names)
        if sp:
            sp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
            state = SimpleNamespace(
                sp_enabled=True,
                sp_size=2,
                sp_rank=rank % 2,
                sp_group=sp_groups[rank // 2],
                fsdp_group=dist.group.WORLD,
            )
        else:
            state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0, fsdp_group=dist.group.WORLD)
        trainer, metrics = _build_trainer(state, distributed=True, fsdp_mesh=mesh, device=device)
        if hsdp:

            def configure_hsdp(micro_step, num_micro_steps, runtime):
                if micro_step == 0:
                    runtime.model.set_requires_all_reduce(False)
                elif micro_step == num_micro_steps - 1:
                    runtime.model.set_requires_all_reduce(True)

            trainer.base._configure_hsdp_allreduce = configure_hsdp
        if sp:
            groups = [[1], [2]] if rank < 2 else [[3, 3], [4, 4, 4]]
            all_values = [1, 2, 3, 3, 4, 4, 4]
        else:
            groups = [[rank + 1], [rank + 2] * (rank + 1)]
            all_values = [
                item
                for owner in range(world_size)
                for group in ([owner + 1], [owner + 2] * (owner + 1))
                for item in group
            ]
        _run(trainer, groups, state)
        expected_loss, expected_grad, expected_chosen = _expected(all_values)
        torch.testing.assert_close(trainer.policy_model.optimizer.gradient.cpu(), expected_grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            trainer.policy_model.optimizer.gradient_vector[1:].cpu(), torch.zeros(3), rtol=0, atol=0
        )
        assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
        assert metrics["loss_dict"]["chosen_rewards"] == pytest.approx(expected_chosen.item(), abs=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=lambda context: LaunchSpec(device="cpu", nproc=context.params["world_size"]))
@pytest.mark.parametrize(
    ("world_size", "hsdp", "sp"), [(2, False, False), (4, True, False), (4, False, True), (4, True, True)]
)
def test_fsdp2_and_hsdp_pair_weighted_gradients(world_size, hsdp, sp):
    spawn_with_rendezvous(_fsdp_worker, args=(world_size, hsdp, sp, False), nprocs=world_size)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=4))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 4, reason="requires four Ascend NPUs and HCCL")
def test_native_hccl_ddp_sp_pair_weighting():
    assert torch.npu.device_count() >= 4
    spawn_with_rendezvous(_distributed_worker, args=(4, "dp2_sp2", True), nprocs=4)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=4))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 4, reason="requires four Ascend NPUs and HCCL")
@pytest.mark.parametrize("hsdp", [False, True])
def test_native_hccl_fsdp_sp_pair_weighting(hsdp):
    assert torch.npu.device_count() >= 4
    spawn_with_rendezvous(_fsdp_worker, args=(4, hsdp, True, True), nprocs=4)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("average_log_prob", [False, True])
def test_multiturn_completion_only_loss_and_gradient_match_reference(monkeypatch, average_log_prob):
    import veomni.data.data_collator as collator_module
    from veomni.data.chat_template import ChatmlTemplate

    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer._dpo_provenance_required = True
    trainer.base.args.dpo_config.explicit_segment_provenance = True
    trainer.base.args.dpo_config.average_log_prob = average_log_prob
    trainer._dpo_objective_parameters = dpo._dpo_objective_parameters(trainer.base.args.dpo_config)
    trainer.base.args.data = SimpleNamespace(enable_multisource=False)
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "second"},
    ]
    feature = process_dpo_example(
        {
            "chosen": history + [{"role": "assistant", "content": "preferred response"}],
            "rejected": history + [{"role": "assistant", "content": "short"}],
        },
        chat_template=ChatmlTemplate(_CheckpointTokenizer()),
        record_segment_provenance=True,
    )[0]
    marker = feature[DPO_FEATURE_KEY]
    batch = DPOProvenanceCollator(MainCollator())([feature])
    trainer.train_step(iter([[batch]]))

    weight = torch.tensor(-0.7, requires_grad=True)
    scores = []
    for start, end in ((0, marker.chosen), (marker.chosen, marker.chosen + marker.rejected)):
        ids = feature["input_ids"][start:end].float()
        active = feature["labels"][start + 1 : end] != -100
        score = (ids[:-1] * weight * active).sum()
        if average_log_prob:
            score = score / active.sum()
        scores.append(score)
    expected_loss = -F.logsigmoid(0.1 * (scores[0] - scores[1]))
    expected_loss.backward()
    torch.testing.assert_close(trainer.policy_model.optimizer.gradient, weight.grad, rtol=1e-6, atol=1e-6)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)


def _replace_with_adamw(trainer):
    trainer.policy_model.optimizer = torch.optim.AdamW(trainer.policy_model.module.parameters(), lr=0.01)
    trainer.policy_model.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.policy_model.optimizer, step_size=1)


class _CheckpointTokenizer:
    dpo_unpadded_output_contract = True
    dpo_unpadded_output_contract_id = "pair-test-tokenizer-v1"

    def encode(self, text, add_special_tokens=True):
        return [ord(value) for value in text]


def _install_recovery_geometry(trainer):
    from tests.data.test_dpo_provenance_checkpoint import _trainer as checkpoint_fixture

    geometry_args = checkpoint_fixture(True).base.args
    trainer.base.args.train.__dict__.update(vars(geometry_args.train))
    trainer.base.args.data = geometry_args.data
    trainer.base.args.data.enable_multisource = False
    trainer.base.args.model = geometry_args.model
    trainer.base.args.train_steps = geometry_args.train_steps
    trainer._dpo_recovery_geometry = dpo._dpo_recovery_geometry(trainer.base.args)


def _run_strict(trainer, groups, state):
    from unittest.mock import patch

    import veomni.data.data_collator as collator_module

    trainer._dpo_provenance_required = True
    trainer.base.args.dpo_config.explicit_segment_provenance = True
    trainer._dpo_objective_parameters = dpo._dpo_objective_parameters(trainer.base.args.dpo_config)
    _install_recovery_geometry(trainer)
    trainer.policy_model.tokenizer = _CheckpointTokenizer()
    trainer.policy_model.chat_template = None
    trainer._dpo_producer_identity = dpo_producer_identity(trainer.policy_model.tokenizer, None, None)
    trainer._dpo_transform_max_seq_len = 32

    def feature(value):
        return process_dpo_example(
            {"chosen": "!" + chr(ord("A") + value) + "x", "rejected": "!Ax"},
            tokenizer=trainer.policy_model.tokenizer,
            record_segment_provenance=True,
        )[0]

    with patch.object(collator_module, "get_parallel_state", return_value=state):
        collator = DPOProvenanceCollator(MainCollator())
        trainer.train_step(iter([[collator([feature(value) for value in group]) for group in groups]]))


def _cold_resume_worker(rank, checkpoint_path, result_path):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False)
    trainer.base.args.dpo_config.explicit_segment_provenance = True
    trainer._dpo_objective_parameters = dpo._dpo_objective_parameters(trainer.base.args.dpo_config)
    trainer.policy_model.tokenizer = _CheckpointTokenizer()
    trainer.policy_model.chat_template = None
    trainer._dpo_producer_identity = dpo_producer_identity(trainer.policy_model.tokenizer, None, None)
    trainer._dpo_transform_max_seq_len = 32
    _install_recovery_geometry(trainer)
    _replace_with_adamw(trainer)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer.validate_dpo_provenance_checkpoint(saved["dpo_protocol"])
    trainer.policy_model.module.load_state_dict(saved["model"])
    trainer.policy_model.optimizer.load_state_dict(saved["optimizer"])
    trainer.policy_model.lr_scheduler.load_state_dict(saved["scheduler"])
    trainer.state.global_step = saved["step"]
    _run_strict(trainer, [[2], [4, 5]], state)
    torch.save(
        {
            "model": trainer.policy_model.module.state_dict(),
            "optimizer": trainer.policy_model.optimizer.state_dict(),
            "metrics": metrics,
        },
        result_path,
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_pair_mean_checkpoint_fresh_process_matches_uninterrupted(tmp_path, monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    _replace_with_adamw(trainer)
    _run_strict(trainer, [[1], [2, 3]], state)
    checkpoint = tmp_path / "step1.pt"
    torch.save(
        {
            "model": trainer.policy_model.module.state_dict(),
            "optimizer": trainer.policy_model.optimizer.state_dict(),
            "scheduler": trainer.policy_model.lr_scheduler.state_dict(),
            "dpo_protocol": trainer.dpo_provenance_checkpoint_state(),
            "step": trainer.state.global_step,
        },
        checkpoint,
    )
    _run_strict(trainer, [[2], [4, 5]], state)
    result = tmp_path / "cold.pt"
    mp.spawn(_cold_resume_worker, args=(str(checkpoint), str(result)), nprocs=1, join=True)
    restored = torch.load(result, map_location="cpu", weights_only=False)
    torch.testing.assert_close(restored["model"]["weight"], trainer.policy_model.module.weight, rtol=0, atol=0)
    expected_state = next(iter(trainer.policy_model.optimizer.state_dict()["state"].values()))
    restored_state = next(iter(restored["optimizer"]["state"].values()))
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(restored_state[key], expected_state[key], rtol=0, atol=0)
    assert restored["metrics"]["loss"] == metrics["loss"]


def _invalid_worker(rank, world_size, scenario, endpoint):
    dist.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        if scenario in ("sp_boundaries", "sp_tail_divergence"):
            dp_groups = [dist.new_group([0]), dist.new_group([1])]
            state = SimpleNamespace(
                sp_enabled=True,
                sp_size=2,
                sp_rank=rank,
                sp_group=dist.group.WORLD,
                dp_group=dp_groups[rank],
                fsdp_group=dist.group.WORLD,
            )
        else:
            state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0, fsdp_group=dist.group.WORLD)
        trainer, _ = _build_trainer(state, distributed=True)
        if scenario == "mismatched_microsteps":
            batches = [_microbatch([1])] * (1 if rank == 0 else 2)
        elif scenario in ("one_eof", "all_eof", "loader_error", "eof_and_loader_error", "callback_error"):
            batches = [_microbatch([1])]
            if scenario == "callback_error" and rank == 1:

                def fail_callback(**kwargs):
                    raise ValueError("injected callback error")

                trainer.base.on_step_begin = fail_callback
        else:
            if scenario == "sp_tail_divergence":
                local_tokens = 3 if rank == 0 else 4
                batch = {
                    "input_ids": torch.zeros((1, local_tokens), dtype=torch.long),
                    "labels": torch.full((1, local_tokens), -100, dtype=torch.long),
                    "cu_seq_lens_q": torch.tensor([0, 2, 4, 2 * local_tokens], dtype=torch.int32),
                    "tail_padding_length": torch.tensor(2 * local_tokens - 4, dtype=torch.int32),
                }
            else:
                batch = _microbatch([1], state.sp_size, state.sp_rank)
            if rank == 1 and scenario == "sp_boundaries":
                batch["cu_seq_lens_q"] = torch.tensor([0, 1, 4], dtype=torch.int32)
            if rank == 1 and scenario == "bad_labels":
                batch["labels"] = batch["labels"][:, :-1]
            if rank == 1 and scenario == "bad_token":
                batch["input_ids"][0, 0] = 256
            batches = [batch]
        if (
            scenario == "all_eof"
            or (scenario == "one_eof" and rank == 1)
            or (scenario == "eof_and_loader_error" and rank == 0)
        ):
            iterator = iter([])
        elif scenario in ("loader_error", "eof_and_loader_error") and rank == 1:

            def fail_loader():
                raise ValueError("injected loader error")
                yield  # pragma: no cover

            iterator = fail_loader()
        else:
            iterator = iter([batches])
        exception = StopIteration if scenario == "all_eof" else RuntimeError
        with pytest.raises(exception, match="injected loader error" if scenario == "eof_and_loader_error" else None):
            trainer.train_step(iterator)
        assert trainer.policy_model.optimizer.steps == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize(
    "scenario",
    [
        "mismatched_microsteps",
        "sp_boundaries",
        "sp_tail_divergence",
        "one_eof",
        "all_eof",
        "loader_error",
        "eof_and_loader_error",
        "callback_error",
        "bad_labels",
        "bad_token",
    ],
)
def test_incompatible_distributed_batches_fail_before_optimizer(scenario):
    spawn_with_rendezvous(_invalid_worker, args=(2, scenario), nprocs=2)


def _mutated_provenance_worker(rank, world_size, phase, native_npu, endpoint):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        dpo_unpadded_output_contract = True
        dpo_unpadded_output_contract_id = "test-tokenizer-v1"

        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    if native_npu:
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
    device = f"npu:{rank}" if native_npu else "cpu"
    dist.init_process_group(
        "hccl" if native_npu else "gloo", store=endpoint.connect(), rank=rank, world_size=world_size
    )
    try:
        state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0, fsdp_group=dist.group.WORLD)
        collator_module.get_parallel_state = lambda: state
        trainer, _ = _build_trainer(state, distributed=True, device=device)
        trainer._dpo_provenance_required = True
        trainer.base.args.dpo_config.explicit_segment_provenance = True
        trainer._dpo_objective_parameters = dpo._dpo_objective_parameters(trainer.base.args.dpo_config)
        trainer.base.args.data = SimpleNamespace(enable_multisource=False)
        feature = process_dpo_example(
            {"chosen": "!Bx", "rejected": "!Ax"}, tokenizer=Tokenizer(), record_segment_provenance=True
        )[0]
        batch = DPOProvenanceCollator(MainCollator())([feature])
        if rank == 1 and phase == "before":
            trainer.base.args.dpo_config.explicit_segment_provenance = False
        if rank == 1 and phase == "callback":

            def disable_provenance(**kwargs):
                trainer.base.args.dpo_config.explicit_segment_provenance = False
                kwargs["micro_batches"][0] = _microbatch([1])

            trainer.base.on_step_begin = disable_provenance
        if rank == 1 and phase == "objective_before":
            trainer.base.args.dpo_config.beta = 0.2
        if rank == 1 and phase == "objective_callback":

            def change_objective(**_kwargs):
                trainer.base.args.dpo_config.beta = 0.2

            trainer.base.on_step_begin = change_objective
        if rank == 1 and phase == "audit_before":
            trainer.base.args.dpo_config.audited_unpadded_tokenizer_sha256 = "a" * 64
        if rank == 1 and phase == "audit_callback":

            def change_audit(**_kwargs):
                trainer.base.args.dpo_config.audited_unpadded_tokenizer_sha256 = "a" * 64

            trainer.base.on_step_begin = change_audit
        if rank == 1 and phase == "malformed_before":
            trainer.base.args.dpo_config = None
        if rank == 1 and phase == "malformed_callback":

            def invalidate_config(**_kwargs):
                trainer.base.args.dpo_config = None

            trainer.base.on_step_begin = invalidate_config
        if phase.startswith(("objective_", "audit_")):
            expected = "objective parameters cannot change"
        elif phase.startswith("malformed_"):
            expected = "AttributeError"
        else:
            expected = "provenance cannot be disabled"
        with pytest.raises(RuntimeError, match=expected):
            trainer.train_step(iter([[batch]]))
        assert trainer.policy_model.optimizer.steps == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize(
    "phase",
    [
        "before",
        "callback",
        "objective_before",
        "objective_callback",
        "audit_before",
        "audit_callback",
        "malformed_before",
        "malformed_callback",
    ],
)
def test_rank_local_provenance_disable_is_coordinated(phase):
    spawn_with_rendezvous(_mutated_provenance_worker, args=(2, phase, False), nprocs=2)


def _different_producers_worker(rank, world_size, native_npu, endpoint):
    class Tokenizer:
        dpo_unpadded_output_contract = True

        def dpo_unpadded_output_contract_id(self):
            return f"tokenizer-{rank}"

    if native_npu:
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
    dist.init_process_group(
        "hccl" if native_npu else "gloo", store=endpoint.connect(), rank=rank, world_size=world_size
    )
    try:
        trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
        trainer._dpo_provenance_required = True
        trainer.base = SimpleNamespace(
            args=SimpleNamespace(dpo_config=dpo.DPOConfig(), data=SimpleNamespace(max_seq_len=32))
        )
        trainer.policy_model = SimpleNamespace(tokenizer=Tokenizer(), chat_template=None)
        with pytest.raises(RuntimeError, match="producer identity does not match across ranks"):
            trainer._build_data_transform()
    finally:
        dist.destroy_process_group()


def _different_objectives_worker(rank, world_size, native_npu, endpoint):
    if native_npu:
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
    dist.init_process_group(
        "hccl" if native_npu else "gloo", store=endpoint.connect(), rank=rank, world_size=world_size
    )
    try:
        params = dpo._dpo_objective_parameters(dpo.DPOConfig(beta=0.1 if rank == 0 else 0.2))
        with pytest.raises(RuntimeError, match="objective parameters does not match across ranks"):
            dpo._dpo_check_consistent_across_ranks(params, context="objective parameters")
    finally:
        dist.destroy_process_group()


def _invalid_dpo_cursor_worker(rank, world_size, scenario, native_npu, endpoint):
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    if native_npu:
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
    dist.init_process_group(
        "hccl" if native_npu else "gloo", store=endpoint.connect(), rank=rank, world_size=world_size
    )
    try:
        state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0, fsdp_group=dist.group.WORLD)
        trainer, _ = _build_trainer(state, distributed=True, device=f"npu:{rank}" if native_npu else "cpu")
        trainer.base.args.dpo_config.explicit_segment_provenance = True
        trainer._dpo_objective_parameters = dpo._dpo_objective_parameters(trainer.base.args.dpo_config)
        trainer.policy_model.tokenizer = _CheckpointTokenizer()
        trainer.policy_model.chat_template = None
        trainer._dpo_producer_identity = dpo_producer_identity(trainer.policy_model.tokenizer, None, None)
        trainer._dpo_transform_max_seq_len = 32
        _install_recovery_geometry(trainer)
        trainer.base.args.train.checkpoint = SimpleNamespace(load_path="/unused")
        trainer.base.args.train.global_rank = rank
        trainer.base.start_step = 1

        class Loader:
            def load_state_dict(self, cursor):
                if rank == 1 and scenario == "malformed":
                    raise ValueError("injected malformed cursor")
                self.cursor = cursor

            def __iter__(self):
                if rank == 1 and scenario == "lazy":
                    raise ValueError("injected lazy cursor failure")
                return self

            def __next__(self):
                raise StopIteration

        trainer.base.train_dataloader = Loader()
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        checkpoint = {
            "global_step": 3,
            "dpo_position": {"epoch": 0, "step_in_epoch": 3, "epoch_complete": False},
            "dpo_provenance_protocol": trainer.dpo_provenance_checkpoint_state(),
        }
        if rank != 1 or scenario != "missing":
            checkpoint["train_dataloader"] = {"cursor": 7}
        if rank == 1 and scenario == "empty":
            checkpoint["train_dataloader"] = {}
        if rank == 1 and scenario == "global_missing":
            del checkpoint["global_step"]
        if rank == 1 and scenario == "global_invalid":
            checkpoint["global_step"] = "3"
        if rank == 1 and scenario == "global_mismatch":
            checkpoint["global_step"] = 4
        if rank == 1 and scenario == "position_mismatch":
            checkpoint["dpo_position"]["step_in_epoch"] = 2
        callback._read_current = lambda _: checkpoint
        with pytest.raises((ValueError, RuntimeError)):
            callback.load_global_state()
        assert trainer.policy_model.optimizer.steps == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_rank_local_producer_identity_mismatch_is_coordinated():
    spawn_with_rendezvous(_different_producers_worker, args=(2, False), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_rank_local_objective_mismatch_is_coordinated():
    spawn_with_rendezvous(_different_objectives_worker, args=(2, False), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize(
    "scenario",
    [
        "missing",
        "empty",
        "malformed",
        "lazy",
        "global_missing",
        "global_invalid",
        "global_mismatch",
        "position_mismatch",
    ],
)
def test_rank_local_dpo_checkpoint_cursor_failure_is_coordinated(scenario):
    spawn_with_rendezvous(_invalid_dpo_cursor_worker, args=(2, scenario, False), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=2))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 2, reason="requires two Ascend NPUs and HCCL")
@pytest.mark.parametrize(
    "phase",
    [
        "before",
        "callback",
        "objective_before",
        "objective_callback",
        "audit_before",
        "audit_callback",
        "malformed_before",
        "malformed_callback",
    ],
)
def test_native_hccl_rank_local_provenance_disable_is_coordinated(phase):
    spawn_with_rendezvous(_mutated_provenance_worker, args=(2, phase, True), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=2))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 2, reason="requires two Ascend NPUs and HCCL")
def test_native_hccl_rank_local_producer_identity_mismatch_is_coordinated():
    spawn_with_rendezvous(_different_producers_worker, args=(2, True), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=2))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 2, reason="requires two Ascend NPUs and HCCL")
def test_native_hccl_rank_local_objective_mismatch_is_coordinated():
    spawn_with_rendezvous(_different_objectives_worker, args=(2, True), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=2))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 2, reason="requires two Ascend NPUs and HCCL")
@pytest.mark.parametrize(
    "scenario",
    [
        "missing",
        "empty",
        "malformed",
        "lazy",
        "global_missing",
        "global_invalid",
        "global_mismatch",
        "position_mismatch",
    ],
)
def test_native_hccl_rank_local_dpo_checkpoint_cursor_failure_is_coordinated(scenario):
    spawn_with_rendezvous(_invalid_dpo_cursor_worker, args=(2, scenario, True), nprocs=2)
