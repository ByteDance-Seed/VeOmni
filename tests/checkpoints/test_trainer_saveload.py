import copy
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

import pytest
import torch
import yaml

from scripts.ci.accelerator.launch_plan import runner_device_count


try:
    from .checkpoint_verification_utils import verify_dcp_to_hf_conversion
    from .utils import (
        get_checkpoint_test_command,
        get_dcp_weights_dir,
        get_hf_output_dir,
        get_merge_dcp_to_hf_command,
        get_output_dir,
    )
except Exception as _:
    from checkpoint_verification_utils import verify_dcp_to_hf_conversion

from scripts.ci.accelerator.launch_plan import LaunchSpec, from_parameter
from tests.checkpoints.utils import get_checkpoint_test_nproc
from veomni.arguments import parse_args
from veomni.checkpoint.layout import weights_dir
from veomni.data import build_dummy_dataset
from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
from veomni.models.seed_omni.modules.fake_model.fake_module_a.modeling import FakeModuleA
from veomni.trainer.base import BaseTrainer, VeOmniArguments
from veomni.trainer.callbacks.base import Callback, TrainerState
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
from veomni.utils import helper
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


_CI_LAUNCH = LaunchSpec(nproc=2)


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
    train = config.get("train", {})
    return train.get("checkpoint", {}).get("output_dir") or train.get("output_dir")


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

    def _build_data_transform(self):
        pass

    def _build_dataset(self):
        args: VeOmniArguments = self.args
        self.train_dataset = build_dummy_dataset(task_type="text", size=8192, max_seq_len=args.data.max_seq_len)
        args.compute_train_steps()
        self.train_steps = args.train_steps

    def _init_callbacks(self):
        self.environ_meter_callback = EnvironMeterCallbackTest(self)
        self.checkpoint_callback = CheckpointCallbackTest(self)
        self.check_callback = CheckCallback(self)
        self.state = TrainerState()

    def on_train_begin(self):
        self.environ_meter_callback.on_train_begin(self.state)
        self.checkpoint_callback.on_train_begin(self.state)
        self.check_callback.on_train_begin(self.state)

    def on_train_end(self):
        self.environ_meter_callback.on_train_end(self.state)
        self.checkpoint_callback.on_train_end(self.state)
        self.check_callback.on_train_end(self.state)

    def on_epoch_begin(self):
        self.environ_meter_callback.on_epoch_begin(self.state)
        self.checkpoint_callback.on_epoch_begin(self.state)
        self.check_callback.on_epoch_begin(self.state)

    def on_epoch_end(self):
        self.environ_meter_callback.on_epoch_end(self.state)
        self.checkpoint_callback.on_epoch_end(self.state)
        self.check_callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        self.environ_meter_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.checkpoint_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.check_callback.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs) -> None:
        self.environ_meter_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpoint_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
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


class CheckpointCallbackTest(CheckpointCallback):
    trainer: TrainerTest

    def on_step_end(self, state: TrainerState, **kwargs):
        if state.global_step == 1:
            self.trainer.golden_model_sd = copy.deepcopy(self.trainer.model.state_dict())
            self.trainer.golden_optim_sd = copy.deepcopy(self.trainer.model.optimizer.state_dict())
            flat_optim = flatten_dict(self.trainer.golden_optim_sd)
            assert any("step" not in k and "param_groups" not in k for k in flat_optim), (
                "Expected non-empty optimizer state before saving step-1 DCP checkpoint"
            )
            self._save_dcp(state)
            # Two different paths: resume takes the step directory, while a
            # reader of raw shards wants the DCP directory inside it.
            self.trainer.step_ckpt_path = os.path.join(
                self.trainer.args.train.checkpoint.save_path, f"global_step_{state.global_step}"
            )
            self.trainer.dcp_weights_path = weights_dir(self.trainer.step_ckpt_path)
            self.trainer.dcp_global_step = state.global_step
            dtypes_before_hf_save = capture_param_dtypes(self.trainer.model)
            self._save_hf(state)
            assert_param_dtypes_unchanged(self.trainer.model, dtypes_before_hf_save)
            self.trainer.hf_weights_path = os.path.join(
                self.trainer.args.train.checkpoint.save_path, f"global_step_{state.global_step}", "hf_ckpt"
            )

    def on_epoch_end(self, state: TrainerState, **kwargs):
        pass

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass


class CheckCallback(Callback):
    trainer: TrainerTest

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        assert state.global_step > self.trainer.dcp_global_step, (
            f"Expected training to continue past saved step {self.trainer.dcp_global_step}, "
            f"ended at step {state.global_step}"
        )
        if self.trainer.args.train.global_rank == 0:
            # Verify HF checkpoint by comparing with DCP checkpoint
            logger.info("Verifying HF checkpoint conversion...")
            assert verify_dcp_to_hf_conversion(
                dcp_checkpoint_dir=self.trainer.dcp_weights_path,
                hf_checkpoint_dir=self.trainer.hf_weights_path,
                safe_serialization=True,
            ), "HF checkpoint verification failed"

        # Confirm post-save training mutated model weights before restoring step-1 DCP.
        post_step2_sd = self.trainer.model.state_dict()
        changed_params = [
            k
            for k, v in post_step2_sd.items()
            if k in self.trainer.golden_model_sd
            and not torch.equal(
                v.to_local() if hasattr(v, "to_local") else v,
                self.trainer.golden_model_sd[k].to_local()
                if hasattr(self.trainer.golden_model_sd[k], "to_local")
                else self.trainer.golden_model_sd[k],
            )
        ]
        assert changed_params, "Expected step-2 optimizer update to mutate model weights after step-1 save"

        self.trainer.args.train.checkpoint.load_path = self.trainer.step_ckpt_path
        self.trainer.load()

        tied_weights_keys = None
        if hasattr(self.trainer.model, "_tied_weights_keys"):
            tied_weights_keys = self.trainer.model._tied_weights_keys

        check_state_dict(self.trainer.golden_model_sd, self.trainer.model.state_dict(), tied_weights_keys)
        check_state_dict(self.trainer.golden_optim_sd, self.trainer.model.optimizer.state_dict(), need_flatten=True)


def main():
    args: VeOmniArguments = parse_args(VeOmniArguments)
    trainer = TrainerTest(args)
    trainer.train()


if __name__ == "__main__":
    main()


def _run_trainer_saveload_and_verify(
    model_name: str,
    ep_size: int,
    nproc: int,
    dp_replicate_size: Optional[int] = None,
):
    if nproc > min(get_checkpoint_test_nproc(), runner_device_count()):
        pytest.skip(f"Requires nproc={nproc}, exceeds configured/available devices")
    # Every round-trip also exports HF weights, checks dtypes, and verifies both
    # direct export and the independent DCP-to-HF CLI conversion.
    exec_command = get_checkpoint_test_command(
        model_name, ep_size, save_hf_weights=True, dp_replicate_size=dp_replicate_size, nproc=nproc
    )
    merge_command = get_merge_dcp_to_hf_command(model_name, ep_size, dp_replicate_size=dp_replicate_size, nproc=nproc)

    exec_result = subprocess.run(exec_command, shell=True, check=True)
    assert exec_result.returncode == 0

    merge_result = subprocess.run(merge_command, shell=True, check=True)
    assert merge_result.returncode == 0

    assert verify_dcp_to_hf_conversion(
        dcp_checkpoint_dir=get_dcp_weights_dir(model_name, ep_size, dp_replicate_size, nproc=nproc),
        hf_checkpoint_dir=get_hf_output_dir(model_name, ep_size, dp_replicate_size, nproc=nproc),
        safe_serialization=True,
    ), (
        f"Save and Load Checkpoint failed for `{model_name}` with ep_size `{ep_size}`, "
        f"nproc `{nproc}`, and dp_replicate_size `{dp_replicate_size}`"
    )

    shutil.rmtree(get_output_dir(model_name, ep_size, dp_replicate_size, nproc=nproc))


# MoE save/load coverage on 4 GPUs:
# - qwen3_moe ep=1: 1D FSDP (no EP)
# - qwen3_moe ep=2: 2D expert mesh (ep_fsdp=2, ep=2) with both axes active
# - qwen3_moe ep=4: 2D expert mesh with degenerate FSDP axis (ep_fsdp=1, ep=4)
# - deepseek_v3 ep=2: DeepSeek-V3 converter + 2D expert mesh (ep_fsdp=2, ep=2)
# - deepseek_v4 ep=1: hash-MoE tid2eid + fused experts + mHC DCP resume
_SAVELOAD_CASES = [
    pytest.param("qwen3_moe", 1, 4, id="qwen3_moe-1"),
    pytest.param("qwen3_moe", 2, 4, id="qwen3_moe-2"),
    pytest.param("qwen3_moe", 4, 4, id="qwen3_moe-4"),
    pytest.param("deepseek_v3", 2, 4, id="deepseek_v3-2"),
    pytest.param("deepseek_v4", 1, 4, id="deepseek_v4-1"),
]


@pytest.mark.launch_plan(provider=from_parameter("nproc"))
@pytest.mark.parametrize("model_name,ep_size,nproc", _SAVELOAD_CASES)
def test_trainer_saveload(model_name: str, ep_size: int, nproc: int):
    _run_trainer_saveload_and_verify(model_name, ep_size, nproc=nproc)


# HSDP save/load coverage:
# - nproc=8, ep=2, dp_replicate=2 -> 3D expert mesh (2, 2, 2), all three dims active (ep_fsdp > 1).
# - nproc=4, ep=2, dp_replicate=2 -> 3D expert mesh (2, 1, 2), degenerate ep_fsdp=1 boundary.
_HSDP_SAVELOAD_CASES = [
    pytest.param(2, 8, id="ep2-8gpu"),
    pytest.param(2, 4, id="ep2-4gpu"),
]


@pytest.mark.launch_plan(provider=from_parameter("nproc"))
@pytest.mark.parametrize("ep_size,nproc", _HSDP_SAVELOAD_CASES)
def test_trainer_saveload_hsdp(ep_size: int, nproc: int):
    _run_trainer_saveload_and_verify("qwen3_moe", ep_size, nproc=nproc, dp_replicate_size=2)


@pytest.mark.launch_plan(provider=_CI_LAUNCH)
@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) or runner_device_count() < 2,
    reason="requires at least 2 accelerator devices",
)
def test_frozen_omni_module_resume_loads_hf_weights(tmp_path):
    from tests.tools.launch_utils import find_free_port, run_bounded_process_group

    snapshot_dir = tmp_path / "frozen_module"
    output_dir = tmp_path / "output"
    resume_root = output_dir / "checkpoints" / "global_step_1"
    resume_root.mkdir(parents=True)

    config = FakeModuleAConfig(hidden_size=64, freeze=True)
    model = FakeModuleA(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)
    model.save_pretrained(snapshot_dir)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={_CI_LAUNCH.nproc}",
        f"--master-port={find_free_port()}",
        "tests/checkpoints/omni_frozen_resume_runner.py",
        # `model.model_path` is the split-checkpoint ROOT; the runner resolves
        # `frozen_module` (the module name) as this module's subfolder under it.
        "--model.model_path",
        str(tmp_path),
        "--model.ops_implementation.attn_implementation",
        "eager",
        "--model.ops_implementation.cross_entropy_loss_implementation",
        "eager",
        "--model.ops_implementation.rms_norm_implementation",
        "eager",
        "--model.ops_implementation.swiglu_mlp_implementation",
        "eager",
        "--model.ops_implementation.rotary_pos_emb_implementation",
        "eager",
        "--model.ops_implementation.load_balancing_loss_implementation",
        "eager",
        # SeedOmni keeps the parallel plan on `model.accelerator`, not `train.*`.
        "--model.accelerator.fsdp_config.fsdp_mode",
        "fsdp2",
        "--model.accelerator.fsdp_config.forward_prefetch",
        "False",
        "--model.broadcast_model_weights_from_rank0",
        "False",
        "--model.accelerator.init_device",
        "meta",
        "--model.accelerator.gradient_checkpointing.enable",
        "False",
        "--data.train_path",
        "dummy",
        "--train.checkpoint.load_path",
        str(resume_root),
        "--train.checkpoint.output_dir",
        str(output_dir),
        "--train.checkpoint.manager",
        "dcp",
        "--train.global_batch_size",
        "2",
        "--train.micro_batch_size",
        "1",
    ]
    run_bounded_process_group(command)
