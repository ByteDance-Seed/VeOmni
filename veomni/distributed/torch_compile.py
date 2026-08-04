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

import types
from dataclasses import dataclass
from typing import Collection, Optional

import torch
import torch.nn as nn

from ..utils import logging
from ..utils.device import IS_CUDA_AVAILABLE


logger = logging.get_logger(__name__)


@dataclass
class CompileConfig:
    """Runtime options for compiling FSDP2 decoder blocks."""

    enable: bool = False
    backend: Optional[str] = "inductor"
    mode: Optional[str] = None
    fullgraph: bool = True
    dynamic: bool = False

    def uses_cuda_graphs(self) -> bool:
        """Return whether this config asks torch.compile to use graph replay."""

        return self.backend == "cudagraphs" or self.mode == "reduce-overhead"


def _decoder_block_class_names(model: nn.Module) -> set[str]:
    no_split_modules = getattr(model, "_no_split_modules", None) or getattr(type(model), "_no_split_modules", None)
    if no_split_modules is None:
        return set()
    return {name for name in no_split_modules if isinstance(name, str) and name.endswith("DecoderLayer")}


def _is_decoder_block(module: nn.Module, decoder_block_class_names: Optional[Collection[str]] = None) -> bool:
    """Check decoder blocks using the model's ``_no_split_modules`` list.

    HuggingFace model classes typically list the decoder layer class in
    ``_no_split_modules`` for FSDP/Accelerate. Multimodal models may include
    vision/audio blocks in the same list, so the compile path keeps only
    entries that are decoder layers.
    """

    if decoder_block_class_names is None:
        decoder_block_class_names = _decoder_block_class_names(module)
    return type(module).__name__ in decoder_block_class_names


def validate_compile_config_for_fsdp2(compile_config: CompileConfig, enable_reshard_after_forward: bool) -> None:
    if not compile_config.enable:
        return
    if compile_config.uses_cuda_graphs() and enable_reshard_after_forward:
        raise RuntimeError(
            "train.torch_compile with CUDA Graphs requires "
            "train.accelerator.fsdp_config.reshard_after_forward=False. "
            "Set train.torch_compile.mode=None or disable forward resharding before enabling graph replay."
        )


def compile_decoder_blocks(model: nn.Module, compile_config: CompileConfig) -> int:
    """Compile forward of every decoder block inside ``model`` in place.

    Compiling the forward method (rather than wrapping the whole module)
    preserves module identity for FSDP2 — pre/post-forward all-gather and
    reshard hooks stay outside the compiled region.
    """

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "train.torch_compile.enable requires torch.compile, but this PyTorch build has no torch.compile."
        )

    compile_kwargs = {
        "fullgraph": compile_config.fullgraph,
        "dynamic": compile_config.dynamic,
    }
    if compile_config.backend is not None:
        compile_kwargs["backend"] = compile_config.backend
    if compile_config.mode is not None:
        if compile_config.backend == "cudagraphs":
            raise ValueError(
                "train.torch_compile.mode is not accepted by the 'cudagraphs' backend. "
                "Leave mode=None with backend='cudagraphs', or switch backend to 'inductor'."
            )
        compile_kwargs["mode"] = compile_config.mode

    decoder_block_class_names = _decoder_block_class_names(model)
    compiled = 0
    for fqn, module in model.named_modules():
        if not _is_decoder_block(module, decoder_block_class_names):
            continue
        if getattr(module, "_veomni_forward_compiled", False):
            continue

        original_forward = module.forward
        if hasattr(original_forward, "__func__"):
            module._veomni_original_forward = original_forward.__func__
            compiled_forward = torch.compile(original_forward.__func__, **compile_kwargs)
            module.forward = types.MethodType(compiled_forward, module)
        else:
            module._veomni_original_forward = original_forward
            module.forward = torch.compile(original_forward, **compile_kwargs)
        module._veomni_forward_compiled = True
        module._veomni_compile_config = dict(compile_kwargs)
        logger.info_rank0(f"Compiled decoder block forward for {fqn} with torch.compile({compile_kwargs}).")
        compiled += 1

    logger.info_rank0(f"Compiled {compiled} decoder blocks with torch.compile.")
    return compiled


def mark_compile_step_begin(enable_compile: bool) -> None:
    """Mark a new training step for CUDA Graph Trees managed by torch.compile."""

    if not enable_compile or not IS_CUDA_AVAILABLE:
        return
    mark_step_begin = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if mark_step_begin is not None:
        mark_step_begin()
