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

from . import logging


logger = logging.get_logger(__name__)


ENV_DEFAULTS = {
    "MODELING_BACKEND": "veomni",
    "VEOMNI_USE_LIGER_KERNEL": "1",
    "USE_GROUP_GEMM": "1",
}


def get_env(name: str):
    try:
        default = ENV_DEFAULTS[name]
    except KeyError:
        raise KeyError(f"Env var `{name}` not defined in ENV_DEFAULTS")

    return os.environ.get(name, default)


def format_envs() -> str:
    lines = []
    lines.append("\n========== Environment Variables ==========")

    for name in sorted(ENV_DEFAULTS):
        raw = os.environ.get(name)
        value = raw if raw is not None else ENV_DEFAULTS[name]
        source = "env" if raw is not None else "default"
        lines.append(f"{name}={value} (source={source})")

    lines.append("===========================================")
    return "\n".join(lines)
