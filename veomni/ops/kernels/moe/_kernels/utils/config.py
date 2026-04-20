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

import json
import os
from typing import Any, Dict

from .path import (
    get_bpex_root,
    get_config_dedicated_file_for,
    get_config_path_prefix_for,
)


def load_all_configs(path_prefix: str) -> Dict:
    """Load all configs in specified directory and merge them into a single dictionary."""
    res = {}

    try:
        with open(f"{path_prefix}.bpex") as f:
            res = json.loads(f.read())
    except FileNotFoundError:
        pass

    try:
        dir = path_prefix
        algos = [
            f[: -len(".bpex")] for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(".bpex")
        ]
        for algo_key in algos:
            with open(f"{dir}/{algo_key}.bpex") as f:
                t = json.loads(f.read())
            res.update({algo_key: t})
    except FileNotFoundError:
        pass

    return res


def load_all_configs_for(kernel: Any) -> Dict:
    """Load configs for all pre-tuned algo-key for a given kernel and merge them into a single
    dictionary. Device and Triton version is assumed the same as the calling environment.
    """
    path_prefix = get_config_path_prefix_for(kernel)
    configs = load_all_configs(path_prefix)
    return configs


def write_config_into_dedicated_file_for(dir_prefix: str, kernel: Any, algo_key: str, configs: Dict):
    """Write config for the given kernel and algo_key into `dir_prefix`. Internal directory
    hierarchy used by bpex is preserved inside `dir_prefix`."""
    rel = os.path.relpath(get_config_dedicated_file_for(kernel, algo_key), get_bpex_root())
    path = f"{dir_prefix}/{rel}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w+") as f:
        f.write(format_config_to_str(configs))


def format_config_to_str(configs: Dict):
    return json.dumps(configs, indent=2, sort_keys=True) + "\n"
