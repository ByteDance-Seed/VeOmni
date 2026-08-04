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

"""Model-driven helpers for :class:`~veomni.data.data_collator.SeedOmniCollator`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator, Sequence, Union

import torch.nn as nn

from ...data.data_collator import SeedOmniCollator
from ...utils import logging
from .graphs.dispatch import unwrap_module_chain
from .modeling_omni import OmniModel


if TYPE_CHECKING:
    from .accelerator.omni_model_runtime import OmniModelRuntime

logger = logging.get_logger(__name__)

# Either SeedOmni model handle: the bare HF ``OmniModel`` or the VeOmni runtime,
# which forwards ``config`` / ``modules_dict`` to the model it composes.
ComposedModel = Union[OmniModel, "OmniModelRuntime"]


def iter_compose_modules(model: ComposedModel) -> Iterator[tuple[str, nn.Module]]:
    """Yield ``(module_name, module)`` in config declaration order.

    Under VeOmni the composed model already holds each :class:`ModuleRuntime`'s
    (FSDP/DDP-wrapped) module, so both handles enumerate the same objects.
    """
    modules = model.modules_dict
    for name in model.config.module_names:
        yield name, modules[name]


def collect_cpu_preprocessors(model: ComposedModel) -> tuple[Callable[..., None], ...]:
    """Collect each graph module's optional worker-side CPU preprocessor.

    Preprocessors run in **fixed serial order** — the config ``modules:`` declaration
    order (``model.config.module_names``).  A module whose prep depends on an earlier
    module's output must be declared after that module.
    """
    preprocessors: list[Callable[..., None]] = []
    for name, module in iter_compose_modules(model):
        raw = unwrap_module_chain(module)
        builder = getattr(raw, "build_cpu_preprocessor", None)
        preprocessor = builder() if builder is not None else None
        if preprocessor is not None:
            preprocessors.append(preprocessor)
            logger.info_rank0(
                f"SeedOmni collator: module '{name}' contributes worker-side "
                f"CPU preprocessor {type(preprocessor).__name__}."
            )
    return tuple(preprocessors)


def build_seed_omni_collator(model: ComposedModel) -> SeedOmniCollator:
    """Build a list-only :class:`SeedOmniCollator` from a composed model handle."""
    from .processing import OmniProcessor

    processor = OmniProcessor.from_composed_model(model)
    logger.info_rank0(f"SeedOmniCollator with {len(processor._preprocessors)} worker-side CPU preprocessor(s).")
    return SeedOmniCollator(processor=processor)


def build_inference_cpu_preprocessors(model: ComposedModel) -> tuple[Callable[..., None], ...]:
    """Collect CPU preprocessors once for inference (re-run each turn / request)."""
    preprocessors = collect_cpu_preprocessors(model)
    logger.info_rank0(f"Inference CPU preprocessors: collected {len(preprocessors)} for per-turn reuse.")
    return preprocessors


def run_cpu_preprocessors(
    preprocessors: Sequence[Callable[..., None]],
    conversation_batches: Sequence[list],
    *,
    inference: bool = False,
    generation_kwargs: dict | None = None,
) -> None:
    """Run pre-collected preprocessors over batched ``conversation_list`` payloads in order."""
    for preprocessor in preprocessors:
        preprocessor(conversation_batches, inference=inference, generation_kwargs=generation_kwargs)


__all__ = [
    "ComposedModel",
    "build_inference_cpu_preprocessors",
    "build_seed_omni_collator",
    "collect_cpu_preprocessors",
    "iter_compose_modules",
    "run_cpu_preprocessors",
]
