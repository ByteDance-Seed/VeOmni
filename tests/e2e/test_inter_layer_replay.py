import os

import pytest

from veomni.utils.device import IS_NPU_AVAILABLE, get_torch_device

from ..tools import DummyDataset, ParallelConfig, compare_metrics, materialize_weights, run_training_config


_CONFIG_PATH = "./tests/toy_config/qwen3_moe_ilp_toy/config.json"
_TRAIN_SCRIPT = "tests/train_scripts/train_text_test.py"


pytestmark = [
    pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="ILP integration requires Ascend NPU"),
    pytest.mark.skipif(
        os.getenv("VEOMNI_RUN_ILP_E2E") != "1",
        reason="Set VEOMNI_RUN_ILP_E2E=1 to reserve two NPUs for the ILP integration test",
    ),
]


def test_qwen3_moe_fsdp2_ilp_matches_native_checkpoint(tmp_path) -> None:
    if get_torch_device().device_count() < 2:
        pytest.skip("ILP EP2 integration requires two NPUs")

    model_path = tmp_path / "model"
    materialize_weights(_CONFIG_PATH, str(model_path))
    dataset = DummyDataset(num_samples=8, seq_len=128)
    parallel = ParallelConfig(sp_size=1, ep_size=2, fsdp_mode="fsdp2")
    common_args = [
        "--train.global_batch_size=2",
        "--train.max_steps=2",
        "--train.gradient_checkpointing.enable=True",
        "--train.gradient_checkpointing.enable_reentrant=False",
        "--train.accelerator.fsdp_config.mixed_precision.enable=True",
        "--train.accelerator.fsdp_config.mixed_precision.param_dtype=bfloat16",
        "--train.accelerator.fsdp_config.mixed_precision.reduce_dtype=float32",
    ]

    try:
        baseline = run_training_config(
            script=_TRAIN_SCRIPT,
            config_path=_CONFIG_PATH,
            model_path=str(model_path),
            train_path=dataset.save_path,
            output_dir=str(tmp_path),
            task_name="baseline",
            parallel_config=parallel,
            nproc=2,
            extra_args=common_args,
            model_name="qwen3_moe",
        )
        ilp = run_training_config(
            script=_TRAIN_SCRIPT,
            config_path=_CONFIG_PATH,
            model_path=str(model_path),
            train_path=dataset.save_path,
            output_dir=str(tmp_path),
            task_name="ilp",
            parallel_config=parallel,
            nproc=2,
            extra_args=[
                *common_args,
                "--train.inter_layer_replay.enable=True",
                "--train.inter_layer_replay.window_size=2",
            ],
            model_name="qwen3_moe",
        )

        compare_metrics({"baseline": baseline, "ilp": ilp}, rtol=1e-2, atol=1e-2, keys=["loss", "grad_norm"])
    finally:
        dataset.clean_cache()
