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

"""Per-module checkpoint path resolution for a SeedOmni V2 model.

A SeedOmni V2 checkpoint is a root folder with one subfolder per OmniModule,
so a module's ``model_path`` is normally a bare name joined under that root.
"""

from __future__ import annotations

import os
from typing import Any

from ....utils.fs import is_non_local


def resolve_model_path(
    model_path: str | os.PathLike,
    modules_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Join a relative per-module ``model_path`` under the checkpoint root.

    A remote-scheme path counts as absolute: ``os.path.isabs("hdfs://ns/x")`` is
    ``False``, so without the extra check a fully-qualified remote path would be
    silently joined under the root as ``/local/root/hdfs://ns/x``.
    """
    if not modules_config:
        return {}
    checkpoint_root = str(model_path)
    for mod_cfg in modules_config.values():
        if not isinstance(mod_cfg, dict):
            continue
        resolved = mod_cfg.get("model_path") or mod_cfg.get("weights_path")
        if resolved is None:
            continue
        if not os.path.isabs(resolved) and not is_non_local(resolved):
            resolved = os.path.join(checkpoint_root, resolved)
        mod_cfg["model_path"] = resolved
    return modules_config
