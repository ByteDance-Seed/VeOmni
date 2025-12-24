import os


CI_MODEL_DIR = os.getenv("CI_MODEL_DIR", ".")
CI_DATASET_DIR = os.getenv("CI_DATASET_DIR", ".")


def qwen3_0p6b_base_tulu_sft_script():
    params = [
        "torchrun --nnodes=1 --nproc_per_node=8 --master-port=4321",
        "tasks/train_torch.py",
        "configs/sft/qwen3_sft.yaml",
        f"--model.model_path {os.path.join(CI_MODEL_DIR, 'Qwen3-0.6B-Base')}",
        f"--data.train_path {os.path.join(CI_DATASET_DIR, 'tulu-3-sft-mixture/data')}",
        "--train.output_dir Qwen3-0.6B-Base-sft",
        "--train.enable_full_determinism true",
        "--train.num_train_epochs 1",
        "--train.max_steps 20",
        "--train.use_wandb false $@ 2>&1",
    ]

    exec_script = " \\\n".join(params)

    return exec_script


SFT_SCRIPT = {
    "qwen3_0p6b_base_tulu_sft": qwen3_0p6b_base_tulu_sft_script(),
}

E2E_TEST_SCRIPT = {**SFT_SCRIPT}
