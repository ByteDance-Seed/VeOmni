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

"""OmniModuleInferencer — per-module inference builder for SeedOmni V2.

Each sub-module is built by its own :class:`OmniModuleInferencer` (default:
eager ``from_pretrained`` + ``device_map='auto'``).  Per-module FSDP2 is opt-in
via the module's YAML ``train.accelerator.fsdp_config`` block — see
``infer_*.yaml`` ``modules:`` overrides deep-merged into ``train.yaml``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ...arguments import VeOmniArguments
from ...distributed.parallel_state import use_parallel_state
from ...utils import helper
from ...utils.device import get_device_type
from ..base import BaseTrainer
from .omni_module_trainer import OmniModuleTrainer


logger = helper.create_logger(__name__)


class OmniModuleInferencer(OmniModuleTrainer):
    """Per-module inference builder — extends :class:`OmniModuleTrainer`.

    * **Default (eager)** — ``from_pretrained(..., device_map='auto')`` on one
      device; no distributed init required.
    * **Optional FSDP** — when the module's YAML carries
      ``train.accelerator.fsdp_config.fsdp_mode: fsdp2``, reuses the training
      meta-init → FSDP2 → weight-load path from :class:`OmniModuleTrainer`
      (without checkpoint callbacks).
    """

    def __init__(
        self,
        args: VeOmniArguments,
        subfolder_name: str,
    ):
        self.subfolder_name = subfolder_name
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        if args.train.accelerator.fsdp_config.fsdp_mode == "eager":
            self._build_model_eager()
            self._build_model_assets()
        else:
            # FSDP / extra-parallel (e.g. ep / emb) inference: reuse the training build
            # (per-module ParallelState -> meta-init -> FSDP2 wrap + weight load),
            # in eval mode and without optimizer / checkpoint callbacks.

            # TODO(WJC)
            # Disable mixed precision so FSDP params stay bf16 (no float32 master)
            # — matching the eager (bf16) modules; otherwise the float32 embeds
            # collide with the eager backbone's bf16 weights (dtype mismatch).
            args.train.accelerator.fsdp_config.mixed_precision.enable = False

            # ``_setup`` builds this module's ParallelState without mutating the
            # global current state, so scope the meta-init + FSDP/DDP wrap (which
            # read ``get_parallel_state()``) to it.
            self._setup()
            with use_parallel_state(self.parallel_state):
                self.base._build_model()
                self._build_model_assets()
                self._freeze_model_module()
                self.base._build_parallelized_model()
                self.base.model.eval()

    @property
    def model(self) -> torch.nn.Module:
        return self.base.model

    def _build_model_eager(self) -> None:
        """Eager ``from_pretrained`` load with a launch-aware ``device_map``.

        * **Single-process** (no torchrun) — ``device_map='auto'`` lets accelerate
          spread the one model across all visible GPUs (and offload if needed).
        * **Distributed** (torchrun, mixed with FSDP/emb modules) — every rank
          sees all GPUs. Pin a **full replica to this rank's own device**
          (``{"": "<device>:<local_rank>"}``; the ``""`` key targets the whole
          model), co-located with this rank's sharded modules.
        """
        args: VeOmniArguments = self.base.args
        assert args.train.accelerator.fsdp_config.fsdp_mode == "eager"
        from ...models.seed_omni import OMNI_MODEL_REGISTRY, read_model_type

        model_path = args.model.model_path
        overrides = dict(args.model.model_config or {})
        model_type = read_model_type(model_path)
        cls = OMNI_MODEL_REGISTRY[model_type]()
        if dist.is_initialized():
            # Distributed launch: one full replica pinned to this rank's device.
            device_map = {"": f"{get_device_type()}:{int(os.getenv('LOCAL_RANK', 0))}"}
        else:
            device_map = "auto"
        logger.info_rank0(
            f"OmniModuleInferencer '{self.subfolder_name}': eager load "
            f"(model_type={model_type}, cls={cls.__name__}, device_map={device_map}) from {model_path}"
        )
        self.base.model = cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            **overrides,
        ).eval()


__all__ = [
    "OmniModuleInferencer",
]
