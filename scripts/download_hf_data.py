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

from huggingface_hub import snapshot_download


"""
python3 scripts/download_hf_data.py --repo_id HuggingFaceFW/fineweb --local_dir ./fineweb/ --allow_patterns sample/10BT/*
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--local_dir", type=str, default="./fineweb/")
    parser.add_argument("--allow_patterns", type=str, default=None)
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = args.local_dir
    allow_patterns = args.allow_patterns

    folder = snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
