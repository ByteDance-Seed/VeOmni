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

"""VeOmni kernel registry and the families registered on import.

``compound`` holds nested-handle helpers. Importing this package registers
families under ``_kernels``: ``rms_norm``, ``rope``, ``rope_vision``,
``async_ulysses_*``, ``dsa_attention`` / ``dsa_indexer``, ``swiglu_mlp``,
``moe_experts``, ``loss`` (LB + CE), ``gated_delta_rule``, and
``attention``. ``apply_kernel_patch`` registers ``veomni_*`` names on
``ALL_ATTENTION_FUNCTIONS``.
"""

from ..utils import logging
from ..utils.env import get_env
from . import _kernels as _kernel_families  # noqa: F401
from .registry import KERNEL_REGISTRY, VeomniKernel, register_kernel, resolve_kernel


logger = logging.get_logger(__name__)


def apply_kernel_patch() -> None:
    """Register ``veomni_*`` attention names on HF dicts.

    No-op when ``MODELING_BACKEND=hf``. Safe to call more than once.
    """
    if get_env("MODELING_BACKEND") == "hf":
        logger.info_rank0("Skip applying kernel patch. Using huggingface transformers backend.")
        return
    from ._kernels.attention.install import apply_veomni_attention_patch

    apply_veomni_attention_patch()


apply_kernel_patch()


__all__ = [
    "KERNEL_REGISTRY",
    "VeomniKernel",
    "apply_kernel_patch",
    "register_kernel",
    "resolve_kernel",
]
