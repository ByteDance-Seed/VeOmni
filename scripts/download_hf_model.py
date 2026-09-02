import argparse
import os

from huggingface_hub import snapshot_download


"""
python3 scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-8B-Instruct --local_dir Qwen3-VL-8B-Instruct
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--local_dir", type=str, default="./Qwen3-VL-8B-Instruct")
    parser.add_argument("--local_dir_use_symlinks", type=bool, default=False)
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = args.local_dir
    local_dir_use_symlinks = args.local_dir_use_symlinks

    snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(local_dir, repo_id.split("/")[1]),
        local_dir_use_symlinks=local_dir_use_symlinks,
    )
