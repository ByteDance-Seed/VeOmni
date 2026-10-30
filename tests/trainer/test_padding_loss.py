from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils import loss_utils


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_padding_has_no_loss_tokens_or_gradient(monkeypatch):
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False, fsdp_size=1))
    real = {"labels": torch.tensor([3, -100])}
    padding = {"labels": torch.tensor([3, -100]), "image_labels": torch.tensor([1, 2]), "padding_flag": True}
    assert loss_utils.count_loss_token(padding)["foundation_tokens"].item() == 0
    assert loss_utils.count_loss_token(padding)["image_tokens"].item() == 0
    total_tokens = loss_utils.count_loss_token([real, padding])
    assert total_tokens["foundation_tokens"].item() == 1
    weight = torch.tensor(2.0, requires_grad=True)
    total_loss = 0
    for batch in (real, padding):
        scaled = loss_utils.mean_global_loss(
            weight.square(), loss_utils.count_loss_token(batch), total_tokens, {"foundation_tokens": 1}
        )
        total_loss += scaled["foundation_loss"]
    total_loss.backward()
    assert weight.grad.item() == 4.0


def _sp_bf16_padding_worker(rank, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    original = loss_utils.get_parallel_state
    try:
        loss_utils.get_parallel_state = lambda: SimpleNamespace(
            sp_enabled=True, sp_group=dist.group.WORLD, sp_size=2, fsdp_size=2
        )
        real = {"labels": torch.tensor([3])}
        padding = {"labels": torch.tensor([3]), "padding_flag": True}
        gradients = []
        for batches in ([real], [real, padding]):
            weight = torch.tensor(2.0, dtype=torch.bfloat16, requires_grad=True)
            total_tokens = loss_utils.count_loss_token(batches)
            for batch in batches:
                result = loss_utils.mean_global_loss(
                    weight.square().float(),
                    loss_utils.count_loss_token(batch),
                    total_tokens,
                    {"foundation_tokens": 2},
                )
                result["foundation_loss"].backward()
            gradients.append(weight.grad.detach().clone())
        assert torch.equal(gradients[0], gradients[1])
        assert gradients[0].item() == 4.0
    finally:
        loss_utils.get_parallel_state = original
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_sp_bf16_padding_gradient_matches_no_padding():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_sp_bf16_padding_worker, args=(), nprocs=2)


def _asymmetric_distributed_padding_worker(rank, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        real = {"labels": torch.tensor([1] * (rank + 1))}
        assert loss_utils.reduce_global_loss_token(loss_utils.count_loss_token(real)) == {"foundation_tokens": 3}
        batch = (
            {"labels": torch.tensor([1]), "mtp_labels": torch.tensor([1]), "padding_flag": True}
            if rank == 0
            else {"labels": torch.tensor([1])}
        )
        error = None
        try:
            loss_utils.reduce_global_loss_token(
                loss_utils.count_loss_token(batch), contains_padding_batch=loss_utils.contains_padding(batch)
            )
        except RuntimeError as exc:
            error = str(exc)
        failures = [None, None]
        dist.all_gather_object(failures, error)
        assert all(message and "does not support padded" in message for message in failures)

        from veomni.trainer.base import BaseTrainer

        trainer = object.__new__(BaseTrainer)
        trainer.state = SimpleNamespace(global_step=3)
        entered = []
        trainer.on_step_begin = lambda **_kwargs: entered.append(True)
        batch = {"padding_flag": True} if rank == 0 else {"labels": torch.tensor([1])}
        error = None
        try:
            trainer.train_step(iter([[batch]]))
        except RuntimeError as exc:
            error = str(exc)
        results = [None, None]
        dist.all_gather_object(results, (error, trainer.state.global_step, entered))
        assert all(
            message and "does not support padded" in message and step == 3 and not seen
            for message, step, seen in results
        )
        batches = [] if rank == 0 else [{"labels": torch.tensor([1])}]
        error = None
        try:
            trainer.train_step(iter([batches]))
        except RuntimeError as exc:
            error = str(exc)
        dist.all_gather_object(results, (error, trainer.state.global_step, entered))
        assert all(message and "empty step" in message and step == 3 and not seen for message, step, seen in results)

        batch = {"padding_flag": torch.tensor([1, 0])} if rank == 0 else {"labels": torch.tensor([1])}
        error = None
        try:
            trainer.train_step(iter([[batch]]))
        except RuntimeError as exc:
            error = str(exc)
        dist.all_gather_object(results, (error, trainer.state.global_step, entered))
        assert all(
            message and "Invalid padding metadata" in message and step == 3 and not seen
            for message, step, seen in results
        )

        batches = [{"labels": torch.tensor([1])} for _ in range(rank + 1)]
        error = None
        try:
            trainer.train_step(iter([batches]))
        except RuntimeError as exc:
            error = str(exc)
        dist.all_gather_object(results, (error, trainer.state.global_step, entered))
        assert all(
            message and "unequal microbatch counts" in message and step == 3 and not seen
            for message, step, seen in results
        )

        trainer.model = SimpleNamespace()
        trainer._reset_async_activation_offload_if_enabled = lambda _model: None
        trainer.on_step_begin = lambda micro_batches: micro_batches.pop() if rank == 0 else None
        batches = [{"labels": torch.tensor([1])}, {"labels": torch.tensor([2])}]
        error = None
        try:
            trainer.train_step(iter([batches]))
        except RuntimeError as exc:
            error = str(exc)
        dist.all_gather_object(results, (error, trainer.state.global_step))
        assert all(
            message and "callback changes to the microbatch count" in message and step == 3
            for message, step in results
        )
        trainer.on_step_begin = lambda micro_batches: micro_batches[0].pop("labels") if rank == 0 else None
        batches = [{"labels": torch.tensor([1])}]
        error = None
        try:
            trainer.train_step(iter([batches]))
        except RuntimeError as exc:
            error = str(exc)
        dist.all_gather_object(results, (error, trainer.state.global_step))
        assert all(message and step == 3 for message, step in results)

        def fail_step_begin(micro_batches):
            if rank == 0:
                raise StopIteration("not a dataloader EOF")

        trainer.on_step_begin = fail_step_begin
        error = None
        try:
            trainer.train_step(iter([[{"labels": torch.tensor([1])}]]))
        except RuntimeError as exc:
            error = str(exc)
        dist.all_gather_object(results, (error, trainer.state.global_step))
        assert all(message and step == 3 for message, step in results)

        batch = {"labels": torch.tensor([1])}
        if rank == 0:
            batch.update(image_labels=torch.tensor([1, 2]), audio_labels=torch.tensor([1]))
        else:
            batch.update(audio_labels=torch.tensor([1, 2, 3]), image_labels=torch.tensor([1]))
        counts = loss_utils.count_loss_token_synchronized([batch], 1)
        totals = loss_utils.reduce_global_loss_token(counts)
        assert totals["image_tokens"] == 3 and totals["audio_tokens"] == 4

        batch = {"labels": torch.tensor([1])}
        batch.update({"image_labels" if rank == 0 else "audio_labels": torch.tensor([1, 2])})
        counts = loss_utils.count_loss_token_synchronized([batch], 1)
        totals = loss_utils.reduce_global_loss_token(counts)
        assert totals["image_tokens"] == 2 and totals["audio_tokens"] == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_asymmetric_distributed_padding_fails_on_all_ranks():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_asymmetric_distributed_padding_worker, args=(), nprocs=2)


def _asymmetric_objective_padding_worker(rank, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        batch = {"padding_flag": True} if rank == 0 else {"labels": torch.tensor([1])}
        for objective in ("DPO", "Omni training", "DiT"):
            failure = None
            try:
                loss_utils.reject_objective_padding([batch], objective)
            except ValueError as exc:
                failure = str(exc)
            failures = [None, None]
            dist.all_gather_object(failures, failure)
            assert all(message and objective in message for message in failures)
            failure = None
            try:
                loss_utils.reject_objective_padding([] if rank == 0 else [batch], objective)
            except ValueError as exc:
                failure = str(exc)
            dist.all_gather_object(failures, failure)
            assert all(message and objective in message and "empty step" in message for message in failures)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_asymmetric_dpo_omni_dit_padding_fails_on_all_ranks():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_asymmetric_objective_padding_worker, args=(), nprocs=2)


def _composed_post_callback_padding_worker(rank, endpoint):
    import importlib.util
    import sys
    import types
    from datetime import timedelta

    import torch.distributed as dist

    if importlib.util.find_spec("av") is None:
        av = types.ModuleType("av")
        av.container = SimpleNamespace(Container=object)
        av.audio = SimpleNamespace(AudioStream=object)
        av.AudioFrame = object
        sys.modules["av"] = av
    import veomni.trainer.dit_trainer as dit_module
    from veomni.trainer.omni.omni_trainer import OmniTrainer
    from veomni.trainer.text_dpo_trainer import TextDPOTrainer

    dit_module.use_parallel_state = lambda _state: nullcontext()
    dit_module.get_parallel_state = lambda: SimpleNamespace(sp_enabled=False)
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        for kind in ("dpo", "omni", "omni_offline", "dit"):
            state = SimpleNamespace(global_step=3)
            if kind == "dpo":
                trainer = object.__new__(TextDPOTrainer)
                trainer.state = state
                trainer.policy_model = object()
                trainer.base = SimpleNamespace(_reset_async_activation_offload_if_enabled=lambda _model: None)
            elif kind.startswith("omni"):
                trainer = object.__new__(OmniTrainer)
                trainer.state = state
                trainer.offline_cache_writer = object()
                trainer._reset_async_activation_offload_if_enabled = lambda: None
            else:
                trainer = object.__new__(dit_module.DiTTrainer)
                trainer.base = SimpleNamespace(
                    state=state,
                    model=SimpleNamespace(parallel_state=object()),
                    _reset_async_activation_offload_if_enabled=lambda _model: None,
                )

            def step_begin(micro_batches):
                if rank == 0:
                    micro_batches[0]["padding_flag"] = True

            trainer.on_step_begin = step_begin
            fn = trainer.offline_cache_step if kind == "omni_offline" else trainer.train_step
            error = None
            try:
                fn(iter([[{"labels": torch.tensor([1])}]]))
            except ValueError as exc:
                error = str(exc)
            results = [None, None]
            dist.all_gather_object(results, (error, state.global_step))
            assert all(message and "padded micro-batches" in message and step == 3 for message, step in results)

            def fail_step_begin(micro_batches):
                if rank == 0:
                    raise StopIteration("not a dataloader EOF")

            trainer.on_step_begin = fail_step_begin
            error = None
            try:
                fn(iter([[{"labels": torch.tensor([1])}]]))
            except RuntimeError as exc:
                error = str(exc)
            dist.all_gather_object(results, (error, state.global_step))
            assert all(message and "preflight failed" in message and step == 3 for message, step in results)

            def invalid_step_begin(micro_batches):
                if rank == 0:
                    raise StopIteration("not a dataloader EOF")
                micro_batches[0]["padding_flag"] = torch.tensor([1, 0])

            trainer.on_step_begin = invalid_step_begin
            error = None
            try:
                fn(iter([[{"labels": torch.tensor([1])}]]))
            except RuntimeError as exc:
                error = str(exc)
            dist.all_gather_object(results, (error, state.global_step))
            assert all(message and "preflight failed" in message and step == 3 for message, step in results)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_composed_trainers_synchronize_post_callback_padding():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_composed_post_callback_padding_worker, args=(), nprocs=2)


@pytest.mark.parametrize("objective", ["DPO", "Omni training", "DiT"])
def test_objective_preflight_preserves_single_rank_exception(objective):
    batch = [{"labels": torch.tensor([1])}]
    with pytest.raises(ValueError, match="callback error"):
        loss_utils.reject_objective_padding(batch, objective, preflight_error=ValueError("callback error"))
    with pytest.raises(RuntimeError, match="preflight failed"):
        loss_utils.reject_objective_padding(batch, objective, preflight_error=StopIteration())


def test_exactly_full_flush_does_not_emit_padding_only_step():
    from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader

    class Strategy:
        def __init__(self):
            self.values = [1, 2, 3, 4]

        def empty(self):
            return not self.values

        def is_ready_for_micro_batch(self):
            return bool(self.values)

        def get_micro_batch(self, _step):
            return {"labels": torch.tensor([self.values.pop(0)])}

        def put_item(self, _item):
            pass

    loader = DynamicBatchSizeDataLoader(iter([{}]), Strategy(), num_micro_batch=2, length=1, drop_last=False)
    batches = list(loader)
    assert len(batches) == 2
    assert all(not sample.get("padding_flag", False) for batch in batches for sample in batch)


def test_unpacked_dynamic_tail_rejected_explicitly():
    from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader

    class ListStrategy:
        def __init__(self):
            self.values = [1, 2, 3]

        def empty(self):
            return not self.values

        def is_ready_for_micro_batch(self):
            return bool(self.values)

        def get_micro_batch(self, _step):
            return [{"labels": torch.tensor([self.values.pop(0)])}]

        def put_item(self, _item):
            pass

    loader = DynamicBatchSizeDataLoader(iter([{}]), ListStrategy(), num_micro_batch=2, length=1, drop_last=False)
    iterator = iter(loader)
    next(iterator)
    with pytest.raises(RuntimeError, match="requires collated dictionary"):
        next(iterator)


def test_distributed_dynamic_tail_is_explicitly_rejected(monkeypatch):
    import veomni.data.data_loader as loaders

    monkeypatch.setattr(loaders, "get_parallel_state", lambda: SimpleNamespace(dp_size=2, sp_size=1))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="requires drop_last=True"):
        loaders.build_native_dataloader(
            dataset=[],
            micro_batch_size=1,
            global_batch_size=2,
            dataloader_batch_size=1,
            max_seq_len=8,
            train_steps=1,
            dyn_bsz=True,
            dyn_bsz_runtime="main",
            drop_last=False,
        )


@pytest.mark.parametrize("trainer_name", ["dpo", "omni"])
def test_composed_trainers_reject_padding_before_updating(trainer_name):
    from veomni.trainer.omni.omni_trainer import OmniTrainer
    from veomni.trainer.text_dpo_trainer import TextDPOTrainer

    trainer_type = TextDPOTrainer if trainer_name == "dpo" else OmniTrainer
    trainer = object.__new__(trainer_type)
    if trainer_name == "dpo":
        trainer.base = SimpleNamespace()
    trainer.state = SimpleNamespace(global_step=3)
    with pytest.raises(ValueError, match="does not support padded micro-batches"):
        trainer.train_step(iter([[{"padding_flag": True}]]))
    assert trainer.state.global_step == 3
    with pytest.raises(ValueError, match="does not support padded micro-batches"):
        trainer.forward_backward_step({"padding_flag": True})
    if trainer_name == "omni":
        trainer.offline_cache_writer = object()
        with pytest.raises(ValueError, match="does not support padded micro-batches"):
            trainer.offline_cache_step(iter([[{"padding_flag": True}]]))
        assert trainer.state.global_step == 3


@pytest.mark.parametrize("trainer_name", ["base", "text", "vlm"])
def test_all_padding_step_does_not_advance_training_state(monkeypatch, trainer_name):
    import veomni.trainer.base as base_module
    import veomni.trainer.text_trainer as text_module
    import veomni.trainer.vlm_trainer as vlm_module

    module, cls = {
        "base": (base_module, base_module.BaseTrainer),
        "text": (text_module, text_module.TextTrainer),
        "vlm": (vlm_module, vlm_module.VLMTrainer),
    }[trainer_name]
    monkeypatch.setattr(
        base_module,
        "reduce_global_loss_token",
        lambda counts, **kwargs: {key: value.item() for key, value in counts.items()},
    )
    trainer = object.__new__(cls)
    state = SimpleNamespace(global_step=3)
    updates = []
    model = SimpleNamespace(
        optimizer=SimpleNamespace(step=lambda: updates.append("optimizer")),
        lr_scheduler=SimpleNamespace(step=lambda: updates.append("scheduler")),
    )
    if trainer_name == "base":
        trainer.state = state
        trainer.model = model
    else:
        trainer.base = object.__new__(base_module.BaseTrainer)
        trainer.base.state = state
        trainer.base.model = model
    with pytest.raises(ValueError, match="at least one real micro-batch"):
        trainer.train_step(iter([[{"labels": torch.tensor([1]), "padding_flag": True}]]))
    assert state.global_step == 3
    assert updates == []


def test_dit_rejects_padding_before_model_or_step(monkeypatch):
    import importlib.util
    import sys
    import types

    if importlib.util.find_spec("av") is None:
        # This control-flow test never touches video decoding; CI has the real dependency.
        av = types.ModuleType("av")
        av.container = SimpleNamespace(Container=object)
        av.audio = SimpleNamespace(AudioStream=object)
        av.AudioFrame = object
        monkeypatch.setitem(sys.modules, "av", av)
    import veomni.trainer.dit_trainer as dit_module

    monkeypatch.setattr(dit_module, "use_parallel_state", lambda _state: nullcontext())
    monkeypatch.setattr(dit_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    trainer = object.__new__(dit_module.DiTTrainer)
    state = SimpleNamespace(global_step=3)
    trainer.base = SimpleNamespace(state=state, model=SimpleNamespace(parallel_state=object()))
    with pytest.raises(ValueError, match="DiT does not support padded micro-batches"):
        trainer.train_step(iter([[{"padding_flag": True}]]))
    with pytest.raises(ValueError, match="DiT does not support padded micro-batches"):
        trainer.forward_backward_step({"padding_flag": True})
    assert state.global_step == 3


def test_environment_meter_counts_only_real_samples():
    from veomni.trainer.callbacks.trace_callback import EnvironMeterCallback

    recorded = []
    callback = object.__new__(EnvironMeterCallback)
    callback.trainer = SimpleNamespace(environ_meter=SimpleNamespace(add=recorded.append))
    real = {"labels": torch.tensor([1, 2]), "source_name": "a"}
    padding = {**real, "padding_flag": True}
    callback.on_step_begin(SimpleNamespace(global_step=1), micro_batches=[real, padding])
    assert recorded == [real]


def test_environment_meter_accepts_unpacked_rl_batches():
    from veomni.trainer.callbacks.trace_callback import EnvironMeterCallback

    recorded = []
    callback = object.__new__(EnvironMeterCallback)
    callback.trainer = SimpleNamespace(environ_meter=SimpleNamespace(add=recorded.append))
    rl_batch = [{"labels": torch.tensor([1, 2])}]
    callback.on_step_begin(SimpleNamespace(global_step=1), micro_batches=[rl_batch])
    assert recorded == [rl_batch]


def test_base_rl_forward_still_accepts_list_microbatch(monkeypatch):
    import veomni.trainer.base as base_module
    from veomni.trainer.base_rl_trainer import BaseRLTrainer

    monkeypatch.setattr(base_module, "use_parallel_state", lambda _state: nullcontext())
    monkeypatch.setattr(base_module, "set_batch_invariant_mode", lambda _enabled: nullcontext())

    class StrictModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.parallel_state = object()

        def forward(self, x, use_cache=False):
            return SimpleNamespace(loss=self.weight * x)

    trainer = object.__new__(BaseRLTrainer)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.args = SimpleNamespace(train=SimpleNamespace(enable_batch_invariant_mode=False))
    trainer.model = StrictModel()
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.channel_loss_callback = None
    trainer.preforward = lambda batches: {"x": sum(batch["x"] for batch in batches)}
    trainer.postforward = lambda outputs, _batch: (outputs.loss, {}, {})
    trainer.forward_backward_step([{"x": torch.tensor(2.0)}, {"x": torch.tensor(3.0)}])
    assert trainer.model.weight.grad.item() == 5.0


def test_padding_skips_rng_and_does_not_initialize_conditional_parameter(monkeypatch):
    import veomni.trainer.base as base_module

    monkeypatch.setattr(base_module, "use_parallel_state", lambda _state: nullcontext())
    monkeypatch.setattr(base_module, "set_batch_invariant_mode", lambda _enabled: nullcontext())

    class ConditionalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.main = torch.nn.Parameter(torch.tensor(1.0))
            self.conditional = torch.nn.Parameter(torch.tensor(1.0))
            self.parallel_state = object()

        def forward(self, conditional, use_cache=False):
            torch.rand(())
            value = self.conditional if conditional else self.main
            return SimpleNamespace(loss=value.square())

    trainer = object.__new__(base_module.BaseTrainer)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.args = SimpleNamespace(train=SimpleNamespace(enable_batch_invariant_mode=False))
    trainer.model = ConditionalModel()
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.channel_loss_callback = None
    trainer.preforward = lambda batch: batch
    trainer.postforward = lambda outputs, _batch: (outputs.loss, {}, {})
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.1, weight_decay=0.1)
    trainer.forward_backward_step({"conditional": False})
    saved_rng = torch.get_rng_state()
    trainer.forward_backward_step({"conditional": True, "padding_flag": True})
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert trainer.model.conditional.grad is None
    optimizer.step()
    assert trainer.model.conditional.item() == 1.0


def test_padding_strips_model_flag_and_skips_capture(monkeypatch):
    import veomni.trainer.base as base_module
    import veomni.trainer.callbacks.base as callback_module
    from veomni.arguments.arguments_types import ChannelLossConfig
    from veomni.trainer.callbacks.channel_loss_callback import ChannelLossCallback

    monkeypatch.setattr(base_module, "use_parallel_state", lambda _state: nullcontext())
    monkeypatch.setattr(base_module, "set_batch_invariant_mode", lambda _enabled: nullcontext())
    monkeypatch.setattr(callback_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    class StrictModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.parallel_state = object()

        def forward(self, x, use_cache=False):
            computer = self.channel.computer
            if computer.capture_active and computer._source_ids:
                computer._result = [
                    {"source_id": computer._source_ids[0], "loss_sum": x.detach(), "token_count": torch.tensor(1)}
                ]
            return SimpleNamespace(loss=self.weight * x)

    trainer = object.__new__(base_module.BaseTrainer)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            enable_batch_invariant_mode=False, channel_loss=ChannelLossConfig(enable=True, interval=1, strict=True)
        )
    )
    trainer.model = StrictModel()
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.channel_loss_callback = ChannelLossCallback(trainer)
    trainer.preforward = lambda batch: batch
    trainer.postforward = lambda outputs, _batch: (outputs.loss * trainer.scale, {}, {"metric": torch.tensor(5.0)})
    callback = trainer.channel_loss_callback
    trainer.model.channel = callback
    original_capture = callback.model_forward_context
    captures = []

    @contextmanager
    def capture():
        captures.append(True)
        with original_capture():
            yield

    callback.model_forward_context = capture
    real_a = {"x": torch.tensor(2.0), "ds_idx": torch.tensor([3])}
    padding = {"padding_flag": True}
    real_b = {"x": torch.tensor(4.0), "ds_idx": torch.tensor([4]), "padding_flag": False}
    callback.on_step_begin(trainer.state, micro_batches=[real_a, padding, real_b])
    reported = []
    for batch, scale in ((real_a, 1), (padding, 0), (real_b, 1)):
        trainer.scale = scale
        reported.append(base_module.BaseTrainer.forward_backward_step(trainer, batch)[2])
    assert trainer.model.weight.grad.item() == 6.0
    assert callback.computer._micro_step == 3
    assert len(captures) == 2
    assert {key: float(value[0]) for key, value in callback.computer.step_totals.items()} == {3: 2.0, 4: 4.0}
    assert reported[0]["metric"].item() == 5.0
    assert reported[1] == {}
    callback.config.interval = 2
    callback.on_step_begin(
        trainer.state, micro_batches=[{"ds_idx": torch.tensor([3])}, padding, {"ds_idx": torch.tensor([4])}]
    )


@pytest.mark.parametrize("kind", ["sft", "omni"])
def test_environ_meter_includes_preflight_time(monkeypatch, kind):
    from veomni.trainer.callbacks import trace_callback
    from veomni.trainer.callbacks.omni_callbacks import step_metrics_callback

    recorded = []
    trainer = SimpleNamespace(environ_meter=SimpleNamespace(add=recorded.append), _step_preflight_start_time=100.0)
    callback_type = (
        trace_callback.EnvironMeterCallback if kind == "sft" else step_metrics_callback.OmniStepMetricsCallback
    )
    callback = object.__new__(callback_type)
    callback.trainer = trainer
    monkeypatch.setattr(trace_callback.time, "time", lambda: 130.0)
    callback.on_step_begin(SimpleNamespace(), micro_batches=[{"labels": torch.tensor([1])}])
    assert callback.start_time == 100.0
    assert len(recorded) == 1
    assert trainer._step_preflight_start_time is None


def test_dpo_wrapper_reads_and_clears_base_preflight_clock_across_steps(monkeypatch):
    from veomni.trainer.callbacks import trace_callback

    class Wrapper:
        def __init__(self):
            self.base = SimpleNamespace(environ_meter=SimpleNamespace(add=lambda _batch: None))

        def __getattr__(self, name):
            return getattr(self.base, name)

    trainer = Wrapper()
    callback = object.__new__(trace_callback.EnvironMeterCallback)
    callback.trainer = trainer
    monkeypatch.setattr(trace_callback.time, "time", lambda: 300.0)
    for start in (100.0, 200.0):
        trainer.base._step_preflight_start_time = start
        callback.on_step_begin(SimpleNamespace(), micro_batches=[{"labels": torch.tensor([1])}])
        assert callback.start_time == start
        assert trainer.base._step_preflight_start_time is None
        assert "_step_preflight_start_time" not in trainer.__dict__


def test_packed_train_step_and_adamw_update_match_without_padding(monkeypatch):
    import copy

    import veomni.data.data_collator as collator_module
    import veomni.trainer.base as base_module
    from veomni.trainer.callbacks.base import TrainerState

    parallel_state = SimpleNamespace(sp_enabled=False, fsdp_size=1)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(loss_utils, "all_reduce", lambda value, **_kwargs: value)
    monkeypatch.setattr(base_module, "use_parallel_state", lambda _state: nullcontext())
    monkeypatch.setattr(base_module, "set_batch_invariant_mode", lambda _enabled: nullcontext())
    monkeypatch.setattr(base_module, "mark_compile_step_begin", lambda *_args: None)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))
            self.parallel_state = parallel_state

        def forward(self, input_ids, labels, use_cache=False, **_kwargs):
            targets = labels[..., 1:]
            logits = self.weight * input_ids[..., :-1].float().unsqueeze(-1) * torch.arange(8, dtype=torch.float32)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 8), targets.reshape(-1), ignore_index=-100)
            return SimpleNamespace(loss=loss)

    collator = collator_module.PackingCollator()
    packed = collator(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([-100, 2, 3]),
                "position_ids": torch.tensor([0, 1, 2]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([4, 5]),
                "position_ids": torch.tensor([0, 1]),
            },
        ]
    )
    assert packed["labels"].flatten().tolist() == [-100, 2, 3, -100, 5]
    packed_other = collator(
        [{"input_ids": torch.tensor([6, 7]), "labels": torch.tensor([-100, 7]), "position_ids": torch.tensor([0, 1])}]
    )

    def run(pad, mask_mode=None):
        model = TinyModel()
        model.optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.1)
        model.lr_scheduler = SimpleNamespace(step=lambda: None)
        recorded = {}
        model.clip_grad_norm = lambda: recorded.setdefault("gradient", model.weight.grad.detach().clone())
        trainer = object.__new__(base_module.BaseTrainer)
        trainer.state = TrainerState(global_step=0)
        trainer.args = SimpleNamespace(train=SimpleNamespace(enable_batch_invariant_mode=False))
        trainer.model = model
        trainer.model_fwd_context = nullcontext()
        trainer.model_bwd_context = nullcontext()
        trainer.channel_loss_callback = None
        trainer.preforward = lambda batch: batch
        trainer.model_reshard = lambda *_args: None
        trainer._configure_hsdp_allreduce = lambda *_args: None
        trainer._reset_async_activation_offload_if_enabled = lambda *_args: None
        trainer.sync_before_train_step = lambda: None

        def step_begin(micro_batches):
            if mask_mode == "callback":
                micro_batches[0]["labels"][0, 1] = -100

        trainer.on_step_begin = step_begin
        trainer.on_step_end = lambda **kwargs: recorded.update(kwargs)
        batches = [copy.deepcopy(packed), copy.deepcopy(packed_other)] + ([{"padding_flag": True}] if pad else [])
        if mask_mode == "before":
            batches[0]["labels"][0, 1] = -100
        base_module.BaseTrainer.train_step(trainer, iter([batches]))
        return recorded, model.weight.detach().clone()

    reference, reference_weight = run(False)
    actual, actual_weight = run(True)
    assert torch.equal(actual["gradient"], reference["gradient"])
    assert torch.equal(actual_weight, reference_weight)
    assert actual["loss"] == reference["loss"]

    masked_reference, masked_reference_weight = run(False, "before")
    masked_callback, masked_callback_weight = run(False, "callback")
    assert torch.equal(masked_callback["gradient"], masked_reference["gradient"])
    assert torch.equal(masked_callback_weight, masked_reference_weight)
    assert masked_callback["loss"] == masked_reference["loss"]
