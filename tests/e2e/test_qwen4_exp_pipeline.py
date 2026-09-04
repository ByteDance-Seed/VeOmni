import json
import os
import subprocess

import pytest
import torch

from veomni.utils.device import get_torch_device

from ..tools import DummyDataset, ParallelConfig, build_torchrun_cmd, materialize_weights


_ACCELERATOR = get_torch_device()
_TOY_CONFIG = "./tests/toy_config/qwen4_exp_toy/config.json"
_TRAIN_SCRIPT = "tests/train_scripts/train_qwen4_exp_pipeline_test.py"
_EAGER_OP_ARGS = [
    "--model.ops_implementation.attn_implementation=eager",
    "--model.ops_implementation.cross_entropy_loss_implementation=eager",
    "--model.ops_implementation.rms_norm_implementation=eager",
    "--model.ops_implementation.swiglu_mlp_implementation=eager",
    "--model.ops_implementation.rotary_pos_emb_implementation=eager",
    "--model.ops_implementation.rotary_pos_emb_vision_implementation=eager",
    "--model.ops_implementation.load_balancing_loss_implementation=eager",
    "--model.ops_implementation.rms_norm_gated_implementation=eager",
    "--model.ops_implementation.causal_conv1d_implementation=eager",
    "--model.ops_implementation.chunk_gated_delta_rule_implementation=eager",
]
_PLE_ARGS = [
    "--train.accelerator.extra_parallel_names=ple",
    "--train.accelerator.extra_parallel_sizes=2",
    "--train.accelerator.extra_parallel_placement_innermost=false",
    "--train.broadcast_model_weights_from_rank0=false",
    "--train.ep_sharded_stream_load=true",
]


def _read_result(output_dir):
    with open(output_dir / "qwen4_exp_pipeline_result.json") as f:
        result = json.load(f)
    assert result["losses"]
    assert torch.isfinite(torch.tensor(result["losses"])).all()
    assert result["ple_grad_seen"]
    assert result["ple_grad_finite"]
    assert result["ple_parameter_changed"]
    assert result["expert_grad_seen"]
    assert result["expert_grad_finite"]
    assert result["expert_parameter_changed"]
    assert {"ple", "ep", "non_extra_parallel"}.issubset(result["optimizer_groups"])
    assert result["optimizer_state_entries"] > 0
    return result


@pytest.mark.skipif(
    not _ACCELERATOR.is_available() or _ACCELERATOR.device_count() < 2,
    reason="Qwen4-Exp VLM SFT pipeline smoke requires two CUDA or NPU devices",
)
def test_qwen4_exp_two_device_training_dcp_resume(tmp_path):
    """Exercise toy VLM SFT with PLE=2, EP=2, optimizer update, and DCP resume."""
    model_path = tmp_path / "model"
    writer_output = tmp_path / "writer"
    resume_output = tmp_path / "resume"
    materialize_weights(_TOY_CONFIG, str(model_path), save_original_format=False)

    dummy_dataset = DummyDataset(
        num_samples=8,
        seq_len=64,
        dataset_type="qwen4exp",
        cache_name=f"qwen4_exp_pipeline_{tmp_path.name}",
    )
    try:
        common_args = [
            *_EAGER_OP_ARGS,
            *_PLE_ARGS,
            "--data.max_seq_len=64",
            "--data.dataloader.num_workers=0",
            # Two micro-batches per rank exercise gradient accumulation before
            # every optimizer step: 4 / (2 ranks * micro_batch_size 1) = 2.
            "--train.global_batch_size=4",
            "--train.gradient_checkpointing.enable=false",
            "--train.optimizer.lr=0.01",
            "--train.checkpoint.save_steps=1",
        ]
        writer_cmd = build_torchrun_cmd(
            script=_TRAIN_SCRIPT,
            config_path=_TOY_CONFIG,
            model_path=str(model_path),
            train_path=dummy_dataset.save_path,
            output_dir=str(writer_output),
            parallel_config=ParallelConfig(sp_size=1, ep_size=2, fsdp_mode="fsdp2"),
            nproc=2,
            extra_args=common_args,
            model_name="qwen4_exp",
        )
        writer_env = dict(os.environ, QWEN4_EXP_EXPECT_START_STEP="0")
        subprocess.run(writer_cmd, check=True, env=writer_env)

        writer_result = _read_result(writer_output)
        assert writer_result["start_global_step"] == 0
        assert writer_result["end_global_step"] == 2

        resume_checkpoint = writer_output / "checkpoints" / "global_step_1"
        assert resume_checkpoint.is_dir()
        for rank in range(2):
            assert (resume_checkpoint / f"qwen4_exp_ple_rank_{rank}.pt").is_file()
            assert (resume_checkpoint / f"qwen4_exp_expert_rank_{rank}.pt").is_file()

        resume_cmd = build_torchrun_cmd(
            script=_TRAIN_SCRIPT,
            config_path=_TOY_CONFIG,
            model_path=str(model_path),
            train_path=dummy_dataset.save_path,
            output_dir=str(resume_output),
            parallel_config=ParallelConfig(sp_size=1, ep_size=2, fsdp_mode="fsdp2"),
            nproc=2,
            extra_args=[
                *common_args,
                "--train.checkpoint.save_steps=0",
                f"--train.checkpoint.load_path={resume_checkpoint}",
            ],
            model_name="qwen4_exp",
        )
        resume_env = dict(os.environ, QWEN4_EXP_EXPECT_START_STEP="1")
        subprocess.run(resume_cmd, check=True, env=resume_env)

        resume_result = _read_result(resume_output)
        assert resume_result["start_global_step"] == 1
        assert resume_result["end_global_step"] == 2
        assert resume_result["optimizer_state_restored"]
    finally:
        dummy_dataset.clean_cache()
