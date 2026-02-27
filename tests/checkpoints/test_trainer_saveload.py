import copy
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import pytest
import torch
import yaml


try:
    from .checkpoint_verification_utils import verify_dcp_to_hf_conversion
    from .utils import (
        get_checkpoint_dir,
        get_checkpoint_test_command,
        get_hf_output_dir,
        get_merge_dcp_to_hf_command,
        get_output_dir,
    )
except Exception as _:
    from checkpoint_verification_utils import verify_dcp_to_hf_conversion
from veomni.arguments import parse_args
from veomni.data import build_dummy_dataset
from veomni.trainer.base import BaseTrainer, VeOmniArguments
from veomni.trainer.callbacks.base import Callback, TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointerCallback, HuggingfaceCkptCallback
from veomni.utils import helper


os.environ["NCCL_DEBUG"] = "OFF"

# To prevent DCP from complaining "too many open files"
# see: https://github.com/pytorch/pytorch/issues/11201
torch.multiprocessing.set_sharing_strategy("file_system")

logger = helper.create_logger(__name__)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def check_state_dict(lhs_dict, rhs_dict, need_flatten=False, tied_weight_key: Optional[list[str]] = None):
    if need_flatten:
        lhs_dict = flatten_dict(lhs_dict)
        rhs_dict = flatten_dict(rhs_dict)

    for k, v in rhs_dict.items():
        if "step" in k or "param_groups" in k:
            continue
        if tied_weight_key and k in tied_weight_key:
            logger.info_rank0(f"skipping tied_weights_key: {k}")
            continue

        lhs, rhs = lhs_dict[k], v
        logger.info_rank0(f"checking {k}...")
        # unwrap to local if available
        lhs_val = lhs.to_local() if hasattr(lhs, "to_local") else lhs
        rhs_val = rhs.to_local() if hasattr(rhs, "to_local") else rhs

        torch.testing.assert_close(rhs_val, lhs_val)


def read_output_dir_from_yaml(yaml_path: str) -> str:
    """Read output_dir from yaml config file."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config.get("train", {}).get("output_dir", None)


def capture_param_dtypes(model):
    """Capture a snapshot of {param_name: dtype} for the model."""
    return {name: param.data.dtype for name, param in model.named_parameters()}


def assert_param_dtypes_unchanged(model, original_dtypes):
    """Assert that model parameter dtypes have not been mutated."""
    for name, param in model.named_parameters():
        current_dtype = param.data.dtype
        expected_dtype = original_dtypes[name]
        assert current_dtype == expected_dtype, (
            f"Parameter '{name}' dtype was mutated: expected {expected_dtype}, got {current_dtype}. "
            f"This indicates an in-place dtype conversion side effect during save."
        )


class TrainerTest(BaseTrainer):
    # The reference state dicts to compare with state dicts restored from DCP later
    golden_model_sd: Dict[str, torch.Tensor]
    golden_optim_sd: Dict[str, torch.Tensor]

    # Paths to DCP and HF checkpoints for verification
    dcp_global_step: int
    dcp_weights_path: str
    hf_weights_path: str

    def _build_model_assets(self):
        self.model_assets = [self.model_config]

    def _build_data_transform(self):
        pass

    def _build_dataset(self):
        args: VeOmniArguments = self.args
        self.train_dataset = build_dummy_dataset(task_type="text", size=8192, max_seq_len=args.data.max_seq_len)
        args.compute_train_steps()
        self.train_steps = args.train_steps

    def _init_callbacks(self):
        self.environ_meter_callback = EnvironMeterCallbackTest(self)
        self.checkpointer_callback = CheckpointerCallbackTest(self)
        self.hf_ckpt_callback = HuggingfaceCkptCallbackTest(self)
        self.check_callback = CheckCallback(self)
        self.state = TrainerState()

    def on_train_begin(self):
        self.environ_meter_callback.on_train_begin(self.state)
        self.checkpointer_callback.on_train_begin(self.state)
        self.hf_ckpt_callback.on_train_begin(self.state)
        self.check_callback.on_train_begin(self.state)

    def on_train_end(self):
        self.environ_meter_callback.on_train_end(self.state)
        self.checkpointer_callback.on_train_end(self.state)
        self.hf_ckpt_callback.on_train_end(self.state)
        self.check_callback.on_train_end(self.state)

    def on_epoch_begin(self):
        self.environ_meter_callback.on_epoch_begin(self.state)
        self.checkpointer_callback.on_epoch_begin(self.state)
        self.hf_ckpt_callback.on_epoch_begin(self.state)
        self.check_callback.on_epoch_begin(self.state)

    def on_epoch_end(self):
        self.environ_meter_callback.on_epoch_end(self.state)
        self.checkpointer_callback.on_epoch_end(self.state)
        self.hf_ckpt_callback.on_epoch_end(self.state)
        self.check_callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        self.environ_meter_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.checkpointer_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.hf_ckpt_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.check_callback.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs) -> None:
        self.environ_meter_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpointer_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.hf_ckpt_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.check_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)


class FakeEnvironMeter:
    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        pass


class EnvironMeterCallbackTest(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self.trainer.environ_meter = FakeEnvironMeter()


class CheckpointerCallbackTest(CheckpointerCallback):
    trainer: TrainerTest

    def on_step_end(self, state: TrainerState, **kwargs):
        pass

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if state.epoch == 0:
            self.trainer.golden_model_sd = copy.deepcopy(self.trainer.model.state_dict())
            self.trainer.golden_optim_sd = copy.deepcopy(self.trainer.optimizer.state_dict())
            self._save_checkpoint(state)
            self.trainer.dcp_weights_path = os.path.join(
                self.trainer.args.train.save_checkpoint_path, f"global_step_{state.global_step}"
            )
            self.trainer.dcp_global_step = state.global_step

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass


class HuggingfaceCkptCallbackTest(HuggingfaceCkptCallback):
    trainer: TrainerTest

    def on_step_end(self, state: TrainerState, **kwargs):
        pass

    def on_epoch_end(self, state: TrainerState, **kwargs):
        state.global_step = self.trainer.dcp_global_step
        dtypes_before_hf_save = capture_param_dtypes(self.trainer.model)
        self._save_checkpoint(state)
        assert_param_dtypes_unchanged(self.trainer.model, dtypes_before_hf_save)
        self.trainer.hf_weights_path = os.path.join(
            self.trainer.args.train.save_checkpoint_path, f"global_step_{state.global_step}", "hf_ckpt"
        )

    def on_train_end(self, state: TrainerState, **kwargs):
        pass


class CheckCallback(Callback):
    trainer: TrainerTest

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.args.train.global_rank == 0:
            # Verify HF checkpoint by comparing with DCP checkpoint
            logger.info("Verifying HF checkpoint conversion...")
            assert verify_dcp_to_hf_conversion(
                dcp_checkpoint_dir=self.trainer.dcp_weights_path,
                hf_checkpoint_dir=self.trainer.hf_weights_path,
                safe_serialization=True,
            ), "HF checkpoint verification failed"

        self.trainer.args.train.load_checkpoint_path = self.trainer.dcp_weights_path
        self.trainer.checkpointer_callback._load_checkpoint()

        tied_weights_keys = None
        if hasattr(self.trainer.model, "_tied_weights_keys"):
            tied_weights_keys = self.trainer.model._tied_weights_keys

        check_state_dict(self.trainer.golden_model_sd, self.trainer.model.state_dict(), tied_weights_keys)
        check_state_dict(self.trainer.golden_optim_sd, self.trainer.optimizer.state_dict(), need_flatten=True)


def main():
    args: VeOmniArguments = parse_args(VeOmniArguments)
    trainer = TrainerTest(args)
    trainer.fit()


if __name__ == "__main__":
    main()


def _run_trainer_saveload_and_verify(model_name: str, ep_size: int):
    exec_command = get_checkpoint_test_command(model_name, ep_size)
    merge_command = get_merge_dcp_to_hf_command(model_name, ep_size)

    exec_result = subprocess.run(exec_command, shell=True, check=True)
    assert exec_result.returncode == 0

    merge_result = subprocess.run(merge_command, shell=True, check=True)
    assert merge_result.returncode == 0

    assert verify_dcp_to_hf_conversion(
        dcp_checkpoint_dir=get_checkpoint_dir(model_name, ep_size),
        hf_checkpoint_dir=get_hf_output_dir(model_name, ep_size),
        safe_serialization=True,
    ), f"Save and Load Checkpoint failed for `{model_name}` with ep_size `{ep_size}`"

    shutil.rmtree(get_output_dir(model_name, ep_size))


def _run_trainer_save_hf_safetensor(model_name: str, ep_size: int):
    exec_command = get_checkpoint_test_command(model_name, ep_size, save_hf_weights=True)
    exec_result = subprocess.run(exec_command, shell=True, check=True)
    assert exec_result.returncode == 0

    shutil.rmtree(get_output_dir(model_name, ep_size))


TEST_MODELS = ["qwen3_moe", "deepseek_v3"]
TEST_EP_SIZES = [1, 4, 8]


@pytest.mark.parametrize("model_name,ep_size", [(model, ep) for model in TEST_MODELS for ep in TEST_EP_SIZES])
def test_trainer_saveload(model_name: str, ep_size: int):
    _run_trainer_saveload_and_verify(model_name, ep_size)


@pytest.mark.parametrize("ep_size", TEST_EP_SIZES)
def test_trainer_save_hf_safetensor(ep_size: int):
    # only test save hf safetensor on qwen3_moe to save resources
    _run_trainer_save_hf_safetensor("qwen3_moe", ep_size)
