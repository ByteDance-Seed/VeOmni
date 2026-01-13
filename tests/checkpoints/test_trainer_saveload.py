import copy
import os
import shutil
import subprocess
from typing import Dict, Optional

import torch
import yaml
from checkpoint_verification_utils import verify_dcp_to_hf_conversion

from veomni.data import build_dummy_dataset
from veomni.trainer.base import Arguments, BaseTrainer
from veomni.trainer.callbacks.base import Callback, CallbackHandler, TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointerCallback, HuggingfaceCkptCallback
from veomni.utils import helper
from veomni.utils.arguments import parse_args


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


class TestTrainer(BaseTrainer):
    # The reference state dicts to compare with state dicts restored from DCP later
    golden_model_sd: Dict[str, torch.Tensor]
    golden_optim_sd: Dict[str, torch.Tensor]

    # Paths to DCP and HF checkpoints for verification
    dcp_global_step: int
    dcp_weights_path: str
    hf_weights_path: str

    def build_model_assets(self):
        return []

    def build_training_dataset(self):
        args: Arguments = self.args
        self.train_dataset = build_dummy_dataset(task_type="text", size=8192, max_seq_len=args.data.max_seq_len)
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        self.train_steps = args.train.train_steps

    def _init_callbacks(self):
        self.callbacks = CallbackHandler(
            [
                TestEnvironMeterCallback(self),
                TestCheckpointerCallback(self),
                TestHuggingfaceCkptCallback(self),
                CheckCallback(self),
            ]
        )
        self.state = TrainerState()


class FakeEnvironMeter:
    def state_dict(self):
        return {}

    def load_state_dict(self, **kwargs):
        pass


class TestEnvironMeterCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self.trainer.environ_meter = FakeEnvironMeter()


class TestCheckpointerCallback(CheckpointerCallback):
    trainer: TestTrainer

    def on_step_end(self, state: TrainerState, **kwargs):
        if state.global_step == 1:
            self.trainer.golden_model_sd = copy.deepcopy(self.trainer.model.state_dict())
            self.trainer.golden_optim_sd = copy.deepcopy(self.trainer.optimizer.state_dict())
            self._save_checkpoint(state)
            self.trainer.dcp_weights_path = os.path.join(
                self.trainer.args.train.save_checkpoint_path, f"global_step_{state.global_step}"
            )
            self.trainer.dcp_global_step = state.global_step

    def on_epoch_end(self, state: TrainerState, **kwargs):
        dcp_state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {},
        }

        if getattr(self.trainer.checkpointer, "save_future", None) is not None:  # async save
            self.trainer.checkpointer.save_future.result()

        self.trainer.checkpointer.load(self.trainer.dcp_weights_path, dcp_state)

        # compare resumed model & optimizer
        tied_weights_keys = None
        if hasattr(self.trainer.model, "_tied_weights_keys"):
            tied_weights_keys = self.trainer.model._tied_weights_keys

        check_state_dict(self.trainer.golden_model_sd, self.trainer.model.state_dict(), tied_weights_keys)
        check_state_dict(self.trainer.golden_optim_sd, self.trainer.optimizer.state_dict(), need_flatten=True)

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass


class TestHuggingfaceCkptCallback(HuggingfaceCkptCallback):
    trainer: TestTrainer

    def on_step_end(self, state: TrainerState, **kwargs):
        pass

    def on_epoch_end(self, state: TrainerState, **kwargs):
        state.global_step = self.trainer.dcp_global_step
        self._save_checkpoint(state)
        self.trainer.hf_weights_path = os.path.join(
            self.trainer.args.train.save_checkpoint_path, f"global_step_{state.global_step}", "hf_ckpt"
        )

    def on_train_end(self, state: TrainerState, **kwargs):
        pass


class CheckCallback(Callback):
    trainer: TestTrainer

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.args.train.global_rank == 0:
            # Verify HF checkpoint by comparing with DCP checkpoint
            logger.info("Verifying HF checkpoint conversion...")
            assert verify_dcp_to_hf_conversion(
                dcp_checkpoint_dir=self.trainer.dcp_weights_path,
                hf_checkpoint_dir=self.trainer.hf_weights_path,
                safe_serialization=True,
            ), "HF checkpoint verification failed"

            shutil.rmtree(self.trainer.args.train.output_dir)


def main():
    args: Arguments = parse_args(Arguments)
    trainer = TestTrainer(args)
    trainer.fit()


def verify_dcp_hf(
    yaml_config_path: str,
):
    # Run merge_dcp_to_hf script after training completes
    output_dir = read_output_dir_from_yaml(yaml_config_path)
    assert output_dir is not None, f"output_dir not found in {yaml_config_path}"
    checkpoint_dir = os.path.join(output_dir, "checkpoints", "global_step_1")
    hf_output_dir = os.path.join(output_dir, "hf_ckpt")

    merge_command = [
        "python",
        "scripts/merge_dcp_to_hf.py",
        "--load-dir",
        checkpoint_dir,
        "--save-dir",
        hf_output_dir,
        "--model-assets-dir",
        os.path.join(output_dir, "model_assets"),
    ]

    logger.info(f"Running merge_dcp_to_hf script: {' '.join(merge_command)}")
    merge_result = subprocess.run(merge_command, check=True, capture_output=True, text=True)
    assert merge_result.returncode == 0

    # Verify the output exists
    assert os.path.exists(hf_output_dir), f"HF checkpoint directory not found: {hf_output_dir}"
    files = os.listdir(hf_output_dir)
    logger.info(f"HF checkpoint files: {files}")
    assert len(files) > 0, "No files generated in HF checkpoint directory"
    assert "config.json" in files, "config.json not found in HF checkpoint directory"

    # Verify HF checkpoint by comparing with DCP checkpoint
    logger.info("Verifying HF checkpoint conversion...")
    assert verify_dcp_to_hf_conversion(
        dcp_checkpoint_dir=checkpoint_dir,
        hf_checkpoint_dir=hf_output_dir,
        safe_serialization=True,
    ), "HF checkpoint verification failed"


def test_trainer_saveload_ep8():
    yaml_config_path = "tests/checkpoints/ep8.yaml"
    ep8_command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--master_port=4321",
        "tests/checkpoints/test_trainer_saveload.py",
        yaml_config_path,
    ]
    ep8_result = subprocess.run(ep8_command, check=True)
    assert ep8_result.returncode == 0


def test_trainer_saveload_ep4():
    yaml_config_path = "tests/checkpoints/ep4.yaml"
    ep4_command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--master_port=4321",
        "tests/checkpoints/test_trainer_saveload.py",
        yaml_config_path,
    ]
    ep4_result = subprocess.run(ep4_command, check=True)
    assert ep4_result.returncode == 0


def test_trainer_saveload_no_ep():
    yaml_config_path = "tests/checkpoints/no_ep.yaml"
    no_ep_command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--master_port=4321",
        "tests/checkpoints/test_trainer_saveload.py",
        yaml_config_path,
    ]
    no_ep_result = subprocess.run(no_ep_command, check=True)
    assert no_ep_result.returncode == 0


if __name__ == "__main__":
    main()
