"""
LoRA checkpoint save/load test using BaseTrainer + toy Qwen3 model (FSDP2).
Runs on both GPU and NPU CI (the NPU unit-test workflow runs this too).

Flow:
  1. Train 3 steps with LoRA adapter.
  2. After step 1: capture golden LoRA params and save DCP checkpoint.
  3. Continue training (steps 2-3 further mutate the adapter weights).
  4. At train_end: reload the step-1 DCP checkpoint into the live model,
     then assert each LoRA (lora_A / lora_B) local shard matches the golden.

Run (4 GPUs):
    torchrun --nproc_per_node=4 tests/lora/test_lora_trainer_saveload.py \\
        --model.config_path tests/toy_config/qwen3_toy/config.json \\
        --model.lora_config '{"rank": 8, "alpha": 16, "lora_modules": ["q_proj", "v_proj"]}' \\
        --model.ops_implementation.attn_implementation flash_attention_2 \\
        --model.ops_implementation.cross_entropy_loss_implementation eager \\
        --model.ops_implementation.rms_norm_implementation eager \\
        --model.ops_implementation.swiglu_mlp_implementation eager \\
        --model.ops_implementation.rotary_pos_emb_implementation eager \\
        --data.train_path dummy \\
        --data.max_seq_len 128 \\
        --train.checkpoint.output_dir /tmp/test_lora_saveload \\
        --train.accelerator.fsdp_config.fsdp_mode fsdp2 \\
        --train.init_device meta \\
        --train.global_batch_size 4 \\
        --train.micro_batch_size 1 \\
        --train.max_steps 3 \\
        --train.checkpoint.save_steps 0 \\
        --train.checkpoint.save_hf_weights false
"""

import os
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.checkpoint import FileSystemReader

from veomni.arguments import VeOmniArguments, parse_args
from veomni.data import build_dummy_dataset
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks.base import Callback, TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointerCallback, HFLoraCkptCallback
from veomni.utils import helper


os.environ["NCCL_DEBUG"] = "OFF"
# Prevent DCP "too many open files" on some systems.
torch.multiprocessing.set_sharing_strategy("file_system")

logger = helper.create_logger(__name__)

_LORA_KEYS = ("lora_A", "lora_B")
# Arbitrary non-zero sentinel for the frozen-base canary perturbation; any
# non-zero value works — we only need to detect whether load overwrote it.
_CANARY_PERTURBATION = 3.14159


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard of a DTensor, or the tensor itself."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


# ---------------------------------------------------------------------------
# Fake environment meter (no-op) — keeps EnvironMeterCallback happy
# ---------------------------------------------------------------------------


class _FakeEnvironMeter:
    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        pass


class _EnvironMeterCallbackTest(Callback):
    def __init__(self, trainer: "LoraTrainerSaveLoadTest"):
        super().__init__(trainer)
        self.trainer.environ_meter = _FakeEnvironMeter()


# ---------------------------------------------------------------------------
# Custom CheckpointerCallback
# ---------------------------------------------------------------------------


class _LoraCheckpointerCallback(CheckpointerCallback):
    trainer: "LoraTrainerSaveLoadTest"

    # Do not load on train_begin (no prior checkpoint).
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        if state.global_step == 1:
            # Snapshot LoRA local shards before any further weight updates.
            self.trainer.golden_lora = {
                n: _local(p).detach().clone()
                for n, p in self.trainer.model.named_parameters()
                if any(k in n for k in _LORA_KEYS)
            }
            assert self.trainer.golden_lora, "No LoRA parameters found in model!"

            # Snapshot frozen base local shards + pick a canary param.
            # Used at train_end to verify that loading a LoRA-only DCP does NOT
            # overwrite frozen base weights (``StateDictOptions(strict=False)``).
            self.trainer.golden_base = {
                n: _local(p).detach().clone() for n, p in self.trainer.model.named_parameters() if not p.requires_grad
            }
            assert self.trainer.golden_base, "No frozen base parameters found in model!"
            self.trainer.canary_name = next(iter(self.trainer.golden_base.keys()))

            self._save_checkpoint(state)
            self.trainer.dcp_ckpt_path = os.path.join(
                self.trainer.args.train.checkpoint.save_path,
                f"global_step_{state.global_step}",
            )
            logger.info_rank0(f"[test] step-1 checkpoint saved → {self.trainer.dcp_ckpt_path}")

            # Invariant for trainable_only=True: the DCP must contain only LoRA
            # tensors under the ``model.*`` namespace; any frozen base key would
            # silently get restored on load and defeat the size optimisation.
            model_keys = [
                k
                for k in FileSystemReader(self.trainer.dcp_ckpt_path).read_metadata().state_dict_metadata
                if k.startswith("model.")
            ]
            base_keys_in_dcp = [k for k in model_keys if "lora_" not in k]
            assert not base_keys_in_dcp, (
                f"trainable_only=True should drop frozen base from the DCP, "
                f"but found {len(base_keys_in_dcp)} non-LoRA model key(s); "
                f"first 5: {base_keys_in_dcp[:5]}"
            )
            assert any("lora_" in k for k in model_keys), f"DCP contains no LoRA tensors: {model_keys[:5]}"
            logger.info_rank0(f"[test] DCP holds {len(model_keys)} model tensors, all LoRA (no frozen base leaked).")


# ---------------------------------------------------------------------------
# Stub HF LoRA callback (no saves during this test)
# ---------------------------------------------------------------------------


class _NoopHFLoraCkptCallback(HFLoraCkptCallback):
    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass

    def _save_model_assets(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Verification callback
# ---------------------------------------------------------------------------


class _LoraCheckCallback(Callback):
    trainer: "LoraTrainerSaveLoadTest"

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        canary_name = self.trainer.canary_name
        canary_param = dict(self.trainer.model.named_parameters())[canary_name]
        with torch.no_grad():
            _local(canary_param).add_(_CANARY_PERTURBATION)
        expected_canary = self.trainer.golden_base[canary_name] + _CANARY_PERTURBATION

        # Restore model to the step-1 checkpoint.
        self.trainer.args.train.checkpoint.load_path = self.trainer.dcp_ckpt_path
        self.trainer.checkpointer_callback._load_checkpoint()

        if not torch.allclose(_local(canary_param), expected_canary, atol=0, rtol=0):
            raise AssertionError(
                f"strict=False failed: frozen base param '{canary_name}' was "
                f"overwritten by _load_checkpoint (the in-place perturbation "
                f"of size {_CANARY_PERTURBATION} should have survived the load)."
            )

        # Compare each LoRA local shard with the golden snapshot.
        mismatches = []
        for n, p in self.trainer.model.named_parameters():
            if not any(k in n for k in _LORA_KEYS):
                continue
            current = _local(p)
            golden = self.trainer.golden_lora[n]
            if not torch.allclose(current, golden, atol=0, rtol=0):
                mismatches.append(n)

        if mismatches:
            raise AssertionError(f"LoRA weight mismatch after checkpoint reload for: {mismatches}")

        logger.info_rank0(
            f"[test] PASS — {len(self.trainer.golden_lora)} LoRA tensors restored from DCP; "
            f"frozen base canary '{canary_name}' preserved across load."
        )


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------


class LoraTrainerSaveLoadTest(BaseTrainer):
    # Set in _LoraCheckpointerCallback.on_step_end
    golden_lora: Dict[str, torch.Tensor]
    golden_base: Dict[str, torch.Tensor]
    canary_name: str
    dcp_ckpt_path: str

    # -- dataset / asset overrides -----------------------------------------

    def _build_model_assets(self) -> None:
        self.model_assets = [self.model_config]

    def _build_data_transform(self) -> None:
        pass

    def _build_dataset(self) -> None:
        args: VeOmniArguments = self.args
        self.train_dataset = build_dummy_dataset(task_type="text", size=64, max_seq_len=args.data.max_seq_len)
        args.compute_train_steps()
        self.train_steps = args.train_steps

    # -- callbacks ----------------------------------------------------------

    def _init_callbacks(self) -> None:
        self.environ_meter_callback = _EnvironMeterCallbackTest(self)
        self.checkpointer_callback = _LoraCheckpointerCallback(self)
        self.hf_ckpt_callback = _NoopHFLoraCkptCallback(self)
        self.check_callback = _LoraCheckCallback(self)
        self.state = TrainerState()

    def on_train_begin(self) -> None:
        self.environ_meter_callback.on_train_begin(self.state)
        self.checkpointer_callback.on_train_begin(self.state)
        self.hf_ckpt_callback.on_train_begin(self.state)
        self.check_callback.on_train_begin(self.state)

    def on_train_end(self) -> None:
        self.environ_meter_callback.on_train_end(self.state)
        self.checkpointer_callback.on_train_end(self.state)
        self.hf_ckpt_callback.on_train_end(self.state)
        self.check_callback.on_train_end(self.state)

    def on_epoch_begin(self) -> None:
        self.environ_meter_callback.on_epoch_begin(self.state)
        self.checkpointer_callback.on_epoch_begin(self.state)
        self.hf_ckpt_callback.on_epoch_begin(self.state)
        self.check_callback.on_epoch_begin(self.state)

    def on_epoch_end(self) -> None:
        self.environ_meter_callback.on_epoch_end(self.state)
        self.checkpointer_callback.on_epoch_end(self.state)
        self.hf_ckpt_callback.on_epoch_end(self.state)
        self.check_callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        self.environ_meter_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.checkpointer_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.hf_ckpt_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.check_callback.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs) -> None:
        self.environ_meter_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpointer_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.hf_ckpt_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.check_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args: VeOmniArguments = parse_args(VeOmniArguments)
    trainer = LoraTrainerSaveLoadTest(args)
    trainer.train()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


def test_lora_trainer_saveload():
    """Run the LoRA save/load test via torchrun and assert it passes."""
    import shutil
    import subprocess

    import pytest

    from veomni.utils.device import get_torch_device

    from ..tools.launch_utils import find_free_port

    torch_device = get_torch_device()
    device_count = torch_device.device_count() if torch_device.is_available() else 0
    if device_count < 1:
        pytest.skip("Requires at least 1 accelerator device")

    nproc = min(device_count, 4)
    output_dir = "/tmp/test_lora_saveload"
    port = find_free_port()

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        __file__,
        "tests/lora/qwen3_toy_lora.yaml",  # positional config_file
        f"--train.checkpoint.output_dir={output_dir}",
    ]

    try:
        result = subprocess.run(cmd, check=True)
        assert result.returncode == 0
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
