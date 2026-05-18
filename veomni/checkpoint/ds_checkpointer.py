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

"""DeepSpeed-format checkpointer."""

import os
from typing import Any, Dict, Optional

from ..utils import logging
from .checkpointer import CheckpointerBase


logger = logging.get_logger(__name__)


class DeepSpeedCheckpointer(CheckpointerBase):
    """Checkpointer that uses DeepSpeed engine.save_checkpoint / load_checkpoint."""

    def save(
        self,
        path: str,
        state: Dict[str, Any],
        save_async: Optional[bool] = None,
        **kwargs,
    ):
        engine = state["model"]  # In DS mode, model IS the engine
        tag = os.path.basename(path)  # e.g. "global_step_100"
        os.makedirs(path, exist_ok=True)

        # DeepSpeed save_checkpoint stores model + optimizer + lr_scheduler
        # We pass extra state as client_state
        engine.save_checkpoint(path, tag=tag, client_state=state.get("extra_state", {}))
        logger.info_rank0(f"DeepSpeed checkpoint saved at {path}")

    def load(self, path: str, state: Dict[str, Any], **kwargs):
        engine = state["model"]  # In DS mode, model IS the engine
        tag = os.path.basename(path)

        _, client_state = engine.load_checkpoint(path, tag=tag)
        if client_state is not None:
            state["extra_state"] = client_state

        logger.info_rank0(f"DeepSpeed checkpoint loaded from {path}")
