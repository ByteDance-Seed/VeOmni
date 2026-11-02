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

"""DiT resume equivalence: the noise/timestep stream must continue, not restart.

A diffusion trainer draws its noise and timestep ids inside
``condition_model.process_condition``, either from a per-run ``torch.Generator``
(Wan, Qwen-Image) or from the accelerator's default RNG (LTX2, MiniMax-H3). Neither is model state, so the job-level checkpoint has to carry both
across a resume. When it does not, the resumed process — whose condition model
was just constructed and re-seeded — replays the stream from the *start* of the
run at the first resumed step.

This runs the real ``DiTTrainer`` twice against the same dummy dataset:

* **run A** trains ``TOTAL_STEPS`` steps uninterrupted and records a signature of
  the noise/timestep drawn at every step;
* **run B** loads run A's mid-run checkpoint, so it starts past the checkpoint
  and its condition model is built fresh in a new process, never having seen the
  steps it resumes past.

Run B's signatures must equal run A's on the resumed steps. The stand-in
condition model is what makes this observable: the real Wan one needs gated
VAE/text-encoder weights, but the property under test is the RNG contract, so
the stand-in reproduces it — a generator seeded at construction, plus per-step
draws from both that generator and the device default RNG.

Dropout is on by default (``VEOMNI_ALIGN_DROPOUT_P``) in both the condition model
and the forward path. That is deliberate, because it is the case most likely to
look like a hole in the fix:

* Dropout adds device-RNG consumers, but it does **not** break resume on its own.
  The checkpoint restores the whole device generator *state*, so as long as the
  two runs consume the stream in the same order, the dropout masks come back
  identical. Verified with ``p=0.5`` in both places.
* What does break it is an *asymmetric* consumer: anything that draws from the
  device stream after the restore point without a counterpart at the same
  logical position in the uninterrupted run (a random buffer init, a random
  sampler or augmentation, an extra warmup forward). Measured with such an
  injected draw: the generator-backed ``timestep_id`` still matched (that stream
  is restored exactly), while the dropout-affected ``noise_sum``/``device_sum``
  diverged.

So the invariant this guards is consumption-order symmetry from the restore
point onward — not the absence of dropout.

Note that ``tests/train_scripts/train_dit_test.py`` sets the condition model to
``None`` and skips ``process_condition`` entirely, which is why the resume path
had no coverage before.
"""

import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict

import pytest
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.data_generators import DummyDataset  # noqa: E402
from tools.launch_utils import find_free_port  # noqa: E402


TOTAL_STEPS = 6
SAVE_STEPS = 3
# 1-based global_step numbering, so these are the steps after the checkpoint.
RESUMED_STEPS = list(range(SAVE_STEPS + 1, TOTAL_STEPS + 1))
SIGNATURE_FILE = "step_signatures.json"

# Mirrors ``WanTransformer3DConditionModel``: same seed derivation, same
# generator, and the same two randomness sources per step.
STANDIN_SEED = 2024
TIMESTEP_CHOICES = 1000

# Dropout is the interesting second-order case: it is never disabled on the
# frozen condition model (only the model itself is put in ``train()`` and the
# condition model is merely ``requires_grad_(False)``), and every active dropout
# consumes the *device* RNG on top of the per-step ``process_condition`` draws.
# Restoring the device RNG state is therefore not enough on its own — the two
# runs must also consume the stream in the same order. Keeping dropout on here
# makes the test guard that, not just the "one draw per step" case.
DROPOUT_P = float(os.environ.get("VEOMNI_ALIGN_DROPOUT_P", "0.5"))


class _AlignConditionModel:
    """Stands in for a diffusion condition model's noise/timestep sampling."""

    def __init__(self, seed: int, dp_rank: int = 0) -> None:
        self.generator = torch.Generator(device=torch.device("cpu"))
        self.generator.manual_seed(seed + dp_rank)
        self.dropout = torch.nn.Dropout(DROPOUT_P)

    def rng_state_dict(self) -> Dict[str, torch.Tensor]:
        return {"generator": self.generator.get_state()}

    def load_rng_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator"])

    def process_condition(self, hidden_states=None, encoder_hidden_states=None, latents=None, **kwargs):
        # The DiT model consumes one tensor per sample (it zips these lists in
        # its forward), so the outputs keep the same list layout as the real
        # condition models.
        reference = hidden_states if hidden_states is not None else latents
        context = encoder_hidden_states if encoder_hidden_states is not None else [None] * len(reference)

        noises, timesteps, device_sums = [], [], []
        for sample in reference:
            noise = torch.randn(sample.shape, dtype=sample.dtype, generator=self.generator).to(sample.device)
            timestep_id = int(torch.randint(0, TIMESTEP_CHOICES, (1,), generator=self.generator).item())
            # A second draw from the device default RNG, so the accelerator
            # stream is part of the signature too. This is the branch that
            # LTX2 / MiniMax-H3 rely on.
            device_draw = torch.randn(4, device=sample.device)
            # Active dropout (the condition model is never ``eval()``-ed) draws
            # another slice out of the same device stream, before backward.
            noise = self.dropout(noise)
            noises.append(noise)
            timesteps.append(torch.full((sample.shape[0],), timestep_id / TIMESTEP_CHOICES, device=sample.device))
            device_sums.append(float(device_draw.float().sum()))

        return {
            "latents": latents,
            "hidden_states": noises,
            "training_target": noises,
            "timestep": timesteps,
            "encoder_hidden_states": list(context),
            "_signature": {
                "noise_sum": float(sum(noise.float().sum() for noise in noises)),
                "noise_first": float(noises[0].flatten()[0].float()),
                "timestep_id": int(round(timesteps[0][0].item() * TIMESTEP_CHOICES)),
                "device_sum": float(sum(device_sums)),
            },
        }


def _build_trainer_class():
    """Import the trainer lazily so pytest can import this module standalone."""
    from veomni.trainer.callbacks import Callback, TrainerState
    from veomni.trainer.dit_trainer import DiTModelRuntime, DiTTrainer, VeOmniDiTArguments, VeOmniModelRuntime

    class SignatureCallback(Callback):
        def __init__(self, trainer) -> None:
            super().__init__(trainer)
            self.signatures: Dict[str, Dict[str, float]] = {}

        def on_step_end(self, state: TrainerState, **kwargs) -> None:
            signature = getattr(self.trainer, "_last_signature", None)
            if signature is not None:
                self.signatures[str(state.global_step)] = signature
                self.trainer._last_signature = None

        def on_train_end(self, state: TrainerState, **kwargs) -> None:
            if self.trainer.args.train.global_rank != 0:
                return
            output_dir = self.trainer.args.train.checkpoint.output_dir
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, SIGNATURE_FILE), "w") as handle:
                json.dump(self.signatures, handle, indent=2, sort_keys=True)

    class AlignDiTModelRuntime(DiTModelRuntime):
        """A DiT runtime whose condition model is the RNG stand-in.

        The condition model is built by the runtime, not the trainer, so
        swapping in the stand-in means overriding here. Its per-rank seed
        derivation and the ``rng_state_dict``/``load_rng_state_dict`` pair are
        what the resume path under test actually persists.
        """

        def _build_condition_model(self, condition_model_type: str) -> None:
            from veomni.distributed.parallel_state import get_parallel_state

            # Same per-rank seed derivation as the real condition models.
            self.condition_model = _AlignConditionModel(seed=STANDIN_SEED, dp_rank=get_parallel_state().dp_rank)

        def _freeze_model_module(self) -> None:
            """Freeze the DiT only.

            The real runtime freezes the condition model alongside it, but the
            stand-in is deliberately not an ``nn.Module`` and owns no parameters.
            """
            VeOmniModelRuntime._freeze_model_module(self)

    class ResumeAlignDiTTrainer(DiTTrainer):
        def __init__(self, args: "VeOmniDiTArguments"):
            args.train.training_task = "offline_training"
            super().__init__(args)
            # Callbacks are handed the reused ``BaseTrainer`` instance, so the
            # per-step signature has to live there to be observable.
            self.base._last_signature = None
            self._signature_callback = SignatureCallback(self.base)
            # Stands in for dropout inside the DiT transformer: another device-RNG
            # consumer on every forward pass.
            self._model_dropout = torch.nn.Dropout(DROPOUT_P)

        def _build_model_runtime(self) -> DiTModelRuntime:
            return AlignDiTModelRuntime(self.base.args.model, "base", train=self.base.args.train)

        def _build_data_transform(self) -> None:
            def process_dummy_example(example: dict, **kwargs):
                return [{key: torch.tensor(value) for key, value in example.items()}]

            self.base.data_transform = process_dummy_example

        def forward_backward_step(self, micro_batch: Dict[str, Any]) -> tuple:
            from veomni.distributed.parallel_state import use_parallel_state

            micro_batch = self.preforward(micro_batch)
            with torch.no_grad():
                micro_batch = self.condition_model.process_condition(**micro_batch)
            self.base._last_signature = micro_batch.pop("_signature")

            # Model-side dropout, i.e. a device-RNG consumer in the forward path.
            micro_batch["hidden_states"] = [self._model_dropout(h) for h in micro_batch["hidden_states"]]

            with use_parallel_state("base"), self.base.model_fwd_context:
                outputs = self.base.model(**micro_batch)
            loss, loss_dict = self.postforward(outputs, micro_batch)
            with use_parallel_state("base"), self.base.model_bwd_context:
                loss.backward()
            return loss, loss_dict

        def on_step_end(self, loss=None, loss_dict=None, grad_norm=None, aux_metrics=None):
            self._signature_callback.on_step_end(self.base.state)
            super().on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm, aux_metrics=aux_metrics)

        def on_train_end(self):
            super().on_train_end()
            self._signature_callback.on_train_end(self.base.state)

    return ResumeAlignDiTTrainer


def main():
    from veomni.arguments import parse_args
    from veomni.trainer.dit_trainer import VeOmniDiTArguments

    args: VeOmniDiTArguments = parse_args(VeOmniDiTArguments)
    _build_trainer_class()(args).train()


def _output_dir(run: str) -> str:
    return os.path.abspath(f"./test_dit_resume_align_{run}")


def _checkpoint_dir() -> str:
    return os.path.join(_output_dir("full"), "checkpoints", f"global_step_{SAVE_STEPS}")


def _materialize_toy_dit_weights(target_dir: str) -> None:
    """Write HF-format weights for the toy Wan transformer.

    The FSDP2 path materializes a meta-init model through ``init_weights()``,
    which under transformers v5 reaches ``tie_weights`` and trips over the
    diffusers-based model's ``_tied_weights_keys`` (v5 renamed it to
    ``all_tied_weights_keys``). Supplying a ``model_path`` sidesteps that
    entirely: the weights are loaded instead of randomly initialized. Building
    the same model directly (no meta device, no FSDP) does not hit the issue.
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model
    from veomni.ops import apply_ops_config

    # Wan's ``rope_apply`` has a non-standard signature, so its device patch
    # explicitly disables the liger RoPE backend. The framework default for
    # ``rotary_pos_emb_implementation`` is ``liger_kernel``, which would raise
    # here; every Wan YAML pins ``eager`` for the same reason.
    apply_ops_config(OpsImplementationConfig(rotary_pos_emb_implementation="eager"))
    model = build_foundation_model(
        config_path="tests/toy_config/wan_t2v_toy/config.json",
        weights_path=None,
        torch_dtype="float32",
        init_device="cpu",
    )
    model.save_pretrained(target_dir)


def _repo_root() -> str:
    # <repo>/tests/checkpoints/test_dit_resume_align.py -> <repo>
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _child_env() -> Dict[str, str]:
    """Environment for the trainer subprocess, pinned to this checkout.

    ``torch.distributed.run`` puts the *script's* directory (not the working
    directory) on the child's ``sys.path[0]``, so an unrelated ``veomni``
    installed in the environment — an editable install pointing at some other
    checkout, say — would be imported instead of the one under test. Putting the
    repo root first on ``PYTHONPATH`` makes the test validate the code it lives
    in, whatever the ambient environment says.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _repo_root() + (os.pathsep + existing if existing else "")
    return env


def _run_trainer(run: str, train_path: str, weights_dir: str, load_path: str | None) -> None:
    output_dir = _output_dir(run)
    shutil.rmtree(output_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=2",
        f"--master-port={find_free_port()}",
        os.path.abspath(__file__),
        "--model.config_path=tests/toy_config/wan_t2v_toy/config.json",
        f"--model.model_path={weights_dir}",
        f"--data.train_path={train_path}",
        "--data.max_seq_len=64",
        "--train.training_task=offline_training",
        # FSDP needs world_size > 1 (``fsdp_enabled`` is ``fsdp_size > 1``), so
        # keep the global batch at the dp size.
        "--train.global_batch_size=2",
        "--train.micro_batch_size=1",
        f"--train.max_steps={TOTAL_STEPS}",
        "--train.bsz_warmup_ratio=0",
        "--train.enable_full_determinism=True",
        "--train.checkpoint.manager=dcp",
        f"--train.checkpoint.save_steps={SAVE_STEPS}",
        "--train.checkpoint.save_epochs=0",
        "--train.checkpoint.save_hf_weights=False",
        f"--train.checkpoint.output_dir={output_dir}",
        # Wan's RoPE signature is non-standard, so the liger backend is
        # explicitly disabled for it; pin the eager backend the Wan YAMLs use.
        "--model.ops_implementation.rotary_pos_emb_implementation=eager",
        "--model.accelerator.fsdp_config.fsdp_mode=fsdp2",
        "--model.accelerator.init_device=meta",
        # Gradient checkpointing is orthogonal to the RNG question, and its
        # non-reentrant path forwards ``early_stop`` into the Wan block, which
        # does not accept it.
        "--model.accelerator.gradient_checkpointing.enable=False",
    ]
    if load_path is not None:
        cmd.append(f"--train.checkpoint.load_path={load_path}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=_child_env())
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "trainer.log"), "w") as handle:
        handle.write(result.stdout)
        handle.write("\n----- stderr -----\n")
        handle.write(result.stderr)
    if result.returncode != 0:
        raise AssertionError(
            f"trainer run {run!r} failed (exit {result.returncode})\n"
            f"--- stdout tail ---\n{result.stdout[-8000:]}\n"
            f"--- stderr tail ---\n{result.stderr[-8000:]}"
        )


def _read_signatures(run: str) -> Dict[str, Dict[str, float]]:
    with open(os.path.join(_output_dir(run), SIGNATURE_FILE)) as handle:
        return json.load(handle)


def test_dit_resume_continues_noise_and_timestep_stream():
    """A resumed DiT run draws the same noise/timestep as the uninterrupted one."""
    from veomni.utils.device import get_device_type

    if get_device_type() == "cpu":
        pytest.skip("needs an accelerator: the trainer cannot run on CPU")

    dummy = DummyDataset(seq_len=64, dataset_type="wan_t2v")
    weights_dir = os.path.abspath("./test_dit_resume_align_weights")
    try:
        shutil.rmtree(weights_dir, ignore_errors=True)
        _materialize_toy_dit_weights(weights_dir)
        _run_trainer("full", dummy.save_path, weights_dir, load_path=None)
        assert os.path.isdir(dummy.save_path), f"dataset vanished after run A: {dummy.save_path}"
        _run_trainer("resumed", dummy.save_path, weights_dir, load_path=_checkpoint_dir())

        full = _read_signatures("full")
        resumed = _read_signatures("resumed")

        # Guard the test's own premise: the resumed run must not have replayed
        # the steps before the checkpoint, or nothing about a resume is exercised.
        pre_checkpoint = {str(step) for step in range(1, SAVE_STEPS + 1)}
        assert pre_checkpoint.isdisjoint(resumed), (
            f"resumed run replayed pre-checkpoint steps, so it did not resume: {sorted(resumed, key=int)}"
        )
        assert sorted(int(step) for step in resumed) == RESUMED_STEPS, (
            f"resumed run covered steps {sorted(resumed, key=int)}, expected {RESUMED_STEPS}"
        )
        assert all(str(step) in full for step in RESUMED_STEPS), (
            f"uninterrupted run did not record steps {RESUMED_STEPS}: {sorted(full, key=int)}"
        )

        for step in RESUMED_STEPS:
            assert full[str(step)] == resumed[str(step)], (
                f"step {step}: the resumed run diverged from the uninterrupted run\n"
                f"  uninterrupted = {full[str(step)]}\n"
                f"  resumed       = {resumed[str(step)]}"
            )
    finally:
        if os.environ.get("VEOMNI_KEEP_ALIGN_LOGS"):
            print(f"kept: {_output_dir('full')}, {_output_dir('resumed')}, {weights_dir}")
        else:
            shutil.rmtree(_output_dir("full"), ignore_errors=True)
            shutil.rmtree(_output_dir("resumed"), ignore_errors=True)
            shutil.rmtree(weights_dir, ignore_errors=True)
        dummy.clean_cache()


if __name__ == "__main__":
    main()
