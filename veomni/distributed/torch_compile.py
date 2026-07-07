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
from typing import Optional

import torch
import torch.nn as nn

from ..utils import logging
from ..utils.device import get_device_type


logger = logging.get_logger(__name__)


@dataclass
class CompileConfig:
    """Runtime options for compiling FSDP2 decoder blocks."""

    enable: bool = False
    backend: Optional[str] = "inductor"
    mode: Optional[str] = "reduce-overhead"
    fullgraph: bool = True
    dynamic: bool = False


def _is_decoder_block(module: nn.Module) -> bool:
    """A decoder block is identified by its class name ending in ``DecoderLayer``.

    This intentionally excludes ViT / vision blocks (``VisionBlock``,
    ``VLVisionBlock``, ``VisionEncoderLayer``, ``VLDecoderLayer``'s vision
    siblings) and the LM head / embedding modules; only the transformer
    decoder blocks of the language model are compiled.
    """

    return type(module).__name__.endswith("DecoderLayer")


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

    compiled = 0
    for fqn, module in model.named_modules():
        if not _is_decoder_block(module):
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

    if not enable_compile or get_device_type() != "cuda":
        return
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
