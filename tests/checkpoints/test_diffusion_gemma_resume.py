# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A resumed DiffusionGemma run must retrace the self-conditioning draw.

The Bernoulli that picks which canvases get the self-conditioning signal comes from a
generator `GlobalStateCallback` does not persist -- it saves the default CPU and device
RNG and nothing about a trainer's own generators. So the draw has to be
reconstructible from the state that *is* saved, and this module checks that end to end:
the masks, losses and parameter gradients of the steps after a resume against the same
steps of an uninterrupted run.

What that buys, precisely: `save_global_state` / `load_global_state` run for real, so a future
generator that needs persisting fails here rather than in production; and the draw's
keying on `global_step` plus the within-step counter reset is exercised under gradient
accumulation. What it does not buy: that the draw stays off the default generators, and
that consecutive micro-batches of one step differ -- neither is visible in a comparison
of two runs that both restore the same RNG, and both live in `TestSelfConditioning` in
tests/models/test_diffusion_gemma.py.

`deepcopy` of the `state_dict` stands in for DCP's transport of the model, optimizer and
lr scheduler only. DCP shards and reassembles these same objects, and sharding wants a
process group this test has no reason to build.
"""

import contextlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.models.test_diffusion_gemma import build_toy_batch, build_toy_model, toy_runtime
from veomni.checkpoint import layout
from veomni.models.transformers.diffusion_gemma import modeling_diffusion_gemma
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback
from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer


TOTAL_STEPS = 6
RESUME_AFTER = 3
MICRO_BATCHES_PER_STEP = 2
CONDITIONING_PROBABILITY = 0.5
#: Steps per epoch, only so `load_global_state` derives a mid-epoch resume.
TRAIN_STEPS = 100
#: Draws before the resume point, i.e. how much of `masks` belongs to the first run.
DRAWS_BEFORE_RESUME = RESUME_AFTER * MICRO_BATCHES_PER_STEP


@pytest.fixture(scope="module", autouse=True)
def single_rank_process_group(tmp_path_factory):
    """`dllm_mean_global_loss` all-reduces unconditionally and VeOmni's `all_reduce`
    fails closed without a process group, so the loss needs one even single-process.
    (The Kmax alignment does not -- it checks `dist.is_initialized()` first.) One gloo
    rank over a file store keeps the reduction on its real path without picking a port.

    Note that `all_reduce` allocates on `get_device_type()`, so on a CUDA box this
    reduces a CUDA tensor over a gloo group. That works today and is the same thing the
    rest of the CPU suite does.
    """
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            pytest.skip("the loss reduction below divides by the group size")
        yield
        return
    store = dist.FileStore(str(tmp_path_factory.mktemp("process_group") / "store"), 1)
    dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


class _StatefulStub:
    """Stands in for the data-iterator and environ-meter halves of the global state."""

    def __init__(self, name):
        self.name = name
        self.loaded = None

    def state_dict(self):
        return {"stub": self.name}

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


class _Harness:
    """The parts of the training loop the self-conditioning draw actually depends on.

    `DllmTrainer.train_step` advances `global_step`, calls `on_micro_batches_ready`, then
    loops `forward_backward_step` over the micro-batches. Clipping, resharding, callbacks
    and the environ meter cannot reach the draw, so this rebuilds only that spine and
    keeps the real `forward_backward_step`.
    """

    def __init__(self, model, save_path, device="cpu"):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=0.5, total_iters=4)
        self.masks = []
        self.data_iterator = _StatefulStub("data_iterator")
        self.train_dataloader = _StatefulStub("train_dataloader")
        self.environ_meter = _StatefulStub("environ_meter")

        base = BaseTrainer.__new__(BaseTrainer)
        base.device = torch.device(device)
        base.LOG_SAMPLE = False
        base.args = SimpleNamespace(
            train=SimpleNamespace(
                local_rank=0,
                global_rank=0,
                world_size=1,
                enable_batch_invariant_mode=False,
                seed=42,
                checkpoint=SimpleNamespace(save_path=str(save_path), load_path=None),
            ),
            train_steps=TRAIN_STEPS,
        )
        base.state = TrainerState()
        base.model = toy_runtime(model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
        base.model_fwd_context = contextlib.nullcontext()
        base.model_bwd_context = contextlib.nullcontext()
        # `state_dict` saves `data_iterator` when the trainer has one, which is the
        # branch production takes; `BaseTrainer` only sets it inside `train()`.
        base.data_iterator = self.data_iterator
        base.train_dataloader = self.train_dataloader
        base.environ_meter = self.environ_meter

        trainer = DiffusionGemmaTrainer.__new__(DiffusionGemmaTrainer)
        trainer.base = base
        # The mask is attached inside `preforward` and the micro-batch is dropped before
        # `forward_backward_step` returns, so record it on the way through.
        inner_preforward = trainer.preforward

        def recording_preforward(micro_batch):
            micro_batch = inner_preforward(micro_batch)
            self.masks.append(micro_batch["self_conditioning_mask"].tolist())
            return micro_batch

        trainer.preforward = recording_preforward
        self.trainer = trainer

        self.callback = GlobalStateCallback.__new__(GlobalStateCallback)
        self.callback.trainer = base
        self.callback._last_saved_step = -1

    @property
    def global_step(self):
        return self.trainer.base.state.global_step

    def step(self, micro_batches):
        """One optimizer step over its micro-batches, in `train_step`'s order."""
        self.trainer.base.state.global_step += 1
        self.trainer.on_micro_batches_ready(micro_batches)
        losses = []
        for micro_batch in micro_batches:
            loss, _ = self.trainer.forward_backward_step({k: v.clone() for k, v in micro_batch.items()})
            losses.append(float(loss.detach()))
        gradients = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return losses, gradients

    def snapshot(self):
        state = self.trainer.base.state
        self.callback.save_global_state(state)
        return {
            "model": deepcopy(self.model.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "lr_scheduler": deepcopy(self.lr_scheduler.state_dict()),
            "step_dir": layout.step_dir(self.trainer.base.args.train.checkpoint.save_path, state.global_step),
        }

    def restore(self, snapshot):
        self.model.load_state_dict(snapshot["model"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.lr_scheduler.load_state_dict(snapshot["lr_scheduler"])
        self.trainer.base.args.train.checkpoint.load_path = snapshot["step_dir"]
        assert self.callback.load_global_state() is not None


def _toy_run(dtype=torch.float32, dg_prefix=False):
    """A model with self-conditioning on, and every micro-batch fixed up front.

    The batches are built before training because `build_toy_batch` reseeds the default
    generator; drawing them inside the loop would reset the RNG that `extra_state`
    carries and hide whether it is restored.
    """
    model = build_toy_model(dtype=dtype).train()
    model.config.self_conditioning_prob = CONDITIONING_PROBABILITY
    model.config.text_config.attn_dg_prefix = dg_prefix
    batches = [
        [
            build_toy_batch(model.config, batch_size=4, seed=100 + step * MICRO_BATCHES_PER_STEP + index)
            for index in range(MICRO_BATCHES_PER_STEP)
        ]
        for step in range(TOTAL_STEPS)
    ]
    return model, batches


def _run(harness, batches, snapshot_after=None):
    records, snapshot = [], None
    for index, micro_batches in enumerate(batches):
        records.append(harness.step(micro_batches))
        if index + 1 == snapshot_after:
            snapshot = harness.snapshot()
    return records, snapshot


def _assert_identical(left, right, label):
    assert left, f"{label}: nothing to compare"
    assert left.keys() == right.keys(), label
    for name in left:
        assert torch.equal(left[name], right[name]), f"{label}: {name}"


class TestResumeReproducesTheSelfConditioningDraw:
    def test_the_masks_losses_and_gradients_match_an_uninterrupted_run(self, tmp_path):
        model, batches = _toy_run()
        continuous = _Harness(model, tmp_path)
        records, snapshot = _run(continuous, batches, snapshot_after=RESUME_AFTER)

        resumed_model, _ = _toy_run()
        resumed = _Harness(resumed_model, tmp_path)
        resumed.restore(snapshot)
        assert resumed.global_step == RESUME_AFTER
        # `state_dict` reads `data_iterator` and `load_global_state` writes
        # `train_dataloader`. In production that is one payload -- `VeOmniIter.state_dict`
        # delegates to the dataloader it wraps -- so the two stubs are separate here only to
        # show which branch was taken, and the assertion is that the load ran at all.
        assert resumed.train_dataloader.loaded == {"stub": "data_iterator"}
        resumed_records, _ = _run(resumed, batches[RESUME_AFTER:])

        assert len(resumed.masks) == (TOTAL_STEPS - RESUME_AFTER) * MICRO_BATCHES_PER_STEP
        assert continuous.masks[DRAWS_BEFORE_RESUME:] == resumed.masks, (
            f"resumed {resumed.masks} against {continuous.masks[DRAWS_BEFORE_RESUME:]}"
        )
        for offset, (expected, actual) in enumerate(zip(records[RESUME_AFTER:], resumed_records, strict=True)):
            step = RESUME_AFTER + offset + 1
            assert expected[0] == actual[0], f"loss diverged at step {step}"
            _assert_identical(expected[1], actual[1], f"gradients at step {step}")

    def test_the_draw_carries_information(self, tmp_path):
        """A mask that is constant would make the comparison above pass on nothing."""
        model, batches = _toy_run()
        harness = _Harness(model, tmp_path)
        _run(harness, batches)
        assert len(harness.masks) == TOTAL_STEPS * MICRO_BATCHES_PER_STEP
        flags = {flag for mask in harness.masks for flag in mask}
        assert flags == {True, False}, harness.masks

    def test_a_resume_that_loses_the_step_counter_diverges(self, tmp_path):
        """The counter is what the comparison rests on, so show that losing it is visible.

        This is the failure the old carried generator produced: the draw restarted from
        `train.seed` and replayed the first steps' masks.
        """
        model, batches = _toy_run()
        continuous = _Harness(model, tmp_path)
        _, snapshot = _run(continuous, batches, snapshot_after=RESUME_AFTER)

        rewound_model, _ = _toy_run()
        rewound = _Harness(rewound_model, tmp_path)
        rewound.restore(snapshot)
        rewound.trainer.base.state.global_step = 0
        _run(rewound, batches[RESUME_AFTER:])

        assert continuous.masks[DRAWS_BEFORE_RESUME:] != rewound.masks

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="the DG_PREFIX Triton kernel needs a GPU")
    def test_the_mask_sequence_ignores_the_device_and_the_kernel(self, monkeypatch, tmp_path):
        """Same masks from CPU/float32/sdpa and GPU/bfloat16/DG_PREFIX.

        Constraint 24's failure mode is Triton compile / autotune burning the default CPU
        *and* CUDA generators, which desyncs ranks running different kernels. So the GPU
        run here both goes through the Triton kernel -- asserted, because a silent fall
        back to sdpa would make this vacuous -- and starts from disturbed defaults, which
        is what makes a draw that reads either of them fail this.
        """
        cpu_model, batches = _toy_run()
        cpu = _Harness(cpu_model, tmp_path)
        _run(cpu, batches)

        gpu_model, gpu_batches = _toy_run(dtype=torch.bfloat16, dg_prefix=True)
        gpu = _Harness(gpu_model.cuda(), tmp_path, device="cuda")
        kernel_calls = []
        kernel = modeling_diffusion_gemma.dg_prefix_decoder_attention

        def counting_kernel(*args, **kwargs):
            kernel_calls.append(None)
            return kernel(*args, **kwargs)

        monkeypatch.setattr(modeling_diffusion_gemma, "dg_prefix_decoder_attention", counting_kernel)
        torch.randn(4096)
        torch.randn(4096, device="cuda")
        _run(gpu, [[{k: v.cuda() for k, v in mb.items()} for mb in step] for step in gpu_batches])

        assert kernel_calls, "attn_dg_prefix fell back to sdpa, so this compares two sdpa runs"
        assert cpu.masks == gpu.masks, f"gpu {gpu.masks} against cpu {cpu.masks}"
