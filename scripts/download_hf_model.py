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

import argparse
import os

from huggingface_hub import snapshot_download


"""
python3 scripts/download_hf_model.py --repo_id deepseek-ai/Janus-1.3B --local_dir Janus-1.3B
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="deepseek-ai/Janus-1.3B")
    parser.add_argument("--local_dir", type=str, default="./Janus-1.3B")
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
