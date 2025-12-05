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


"""Training metrics logger supporting wandb and tensorboard."""

import os
from typing import Any, Dict, Optional

import torch

from veomni.utils import logging

logger = logging.get_logger(__name__)


class Logger:
    """
    Unified logger supporting both wandb and tensorboard.
    """

    def __init__(
        self,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        use_tensorboard: bool = False,
        tensorboard_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.wandb_writer = None
        self.tensorboard_writer = None

        if self.use_wandb:
            try:
                import wandb

                self.wandb_writer = wandb
                self.wandb_writer.init(
                    project=wandb_project,
                    name=wandb_name,
                    config=config,
                )
            except ImportError:
                logger.warning("wandb is not installed, skipping wandb initialization")
                self.use_wandb = False

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                if tensorboard_dir is None:
                    raise ValueError("tensorboard_dir must be provided when use_tensorboard is True")

                os.makedirs(tensorboard_dir, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
                logger.info(f"TensorBoard logging initialized at {tensorboard_dir}")
            except ImportError:
                logger.warning("tensorboard is not installed, skipping tensorboard initialization")
                self.use_tensorboard = False

    def log(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics to both wandb and tensorboard if enabled.

        Args:
            metrics: Dictionary of metrics to log
            step: Global step number
        """
        if self.use_wandb and self.wandb_writer:
            self.wandb_writer.log(metrics, step=step)

        if self.use_tensorboard and self.tensorboard_writer:
            for key, value in metrics.items():
                # Convert tensor to scalar if needed
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        value = value.item()
                    else:
                        value = value.mean().item()

                # Log scalar values
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, step)

    def close(self):
        """Close all loggers."""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()

