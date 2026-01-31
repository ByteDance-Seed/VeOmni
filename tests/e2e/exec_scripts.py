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


def qwen3_0p6b_base_tulu_sft_no_reshard_script():
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
        "--train.enable_reshard_after_forward false",
        "--train.enable_reshard_after_backward false",
        "--train.use_wandb false $@ 2>&1",
    ]

    exec_script = " \\\n".join(params)

    return exec_script


def qwen3_0p6b_base_tulu_sft_rmpad_with_pos_ids_padded_script():
    params = [
        "torchrun --nnodes=1 --nproc_per_node=8 --master-port=4322",
        "tasks/train_torch.py",
        "configs/sft/qwen3_sft.yaml",
        f"--model.model_path {os.path.join(CI_MODEL_DIR, 'Qwen3-0.6B-Base')}",
        f"--data.train_path {os.path.join(CI_DATASET_DIR, 'tulu-3-sft-mixture/data')}",
        "--train.output_dir Qwen3-0.6B-Base-sft-rmpad-pos-padded",
        "--train.enable_full_determinism true",
        "--train.num_train_epochs 1",
        "--train.max_steps 20",
        "--train.rmpad false",
        "--train.rmpad_with_pos_ids true",
        "--train.pad_packed_input true",
        "--train.use_wandb false $@ 2>&1",
    ]

    exec_script = " \\\n".join(params)

    return exec_script


SFT_SCRIPT = {
    "qwen3_0p6b_base_tulu_sft": qwen3_0p6b_base_tulu_sft_script(),
    "qwen3_0p6b_base_tulu_sft_no_reshard": qwen3_0p6b_base_tulu_sft_no_reshard_script(),
    "qwen3_0p6b_base_tulu_sft_rmpad_with_pos_ids_padded": (
        qwen3_0p6b_base_tulu_sft_rmpad_with_pos_ids_padded_script()
    ),
}

E2E_TEST_SCRIPT = {**SFT_SCRIPT}
