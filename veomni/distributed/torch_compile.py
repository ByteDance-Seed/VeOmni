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
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from ..utils import logging
from ..utils.device import get_device_type


logger = logging.get_logger(__name__)


def select_leaf_compile_modules(target_modules: Iterable[Tuple[str, nn.Module]]) -> list[Tuple[str, nn.Module]]:
    """Keep only target modules that do not contain another target module.

    If both a parent and one of its children are FSDP targets, compiling the
    parent forward can pull the child's FSDP collectives into the compiled
    region. Leaf targets keep FSDP communication at module boundaries.
    """

    modules = list(target_modules)
    parent_fqns = set()
    for module_fqn, _ in modules:
        if module_fqn == "":
            continue
        parts = module_fqn.split(".")
        for idx in range(len(parts)):
            parent_fqns.add(".".join(parts[:idx]))

    leaf_modules = [(module_fqn, module) for module_fqn, module in modules if module_fqn not in parent_fqns]

    skipped = len(modules) - len(leaf_modules)
    if skipped:
        logger.info_rank0(f"Skip compiling {skipped} non-leaf target modules to keep FSDP collectives outside.")
    return leaf_modules


def compile_module_forward(
    module: nn.Module,
    *,
    module_fqn: str,
    backend: Optional[str] = None,
    mode: Optional[str] = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> bool:
    """Compile one module's forward method in place.

    Compiling the forward method instead of wrapping the whole module preserves
    module identity for FSDP2. FSDP's pre/post-forward all-gather and reshard
    hooks therefore stay outside the compiled region, matching the per-block
    direction discussed in https://github.com/ByteDance-Seed/VeOmni/issues/401.
    """

    if getattr(module, "_veomni_forward_compiled", False):
        logger.warning_rank0(f"Skip compiling {module_fqn}: forward is already compiled.")
        return False

    if not hasattr(torch, "compile"):
        raise RuntimeError("train.enable_compile requires torch.compile, but this PyTorch build has no torch.compile.")

    compile_kwargs = {
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    if backend is not None:
        compile_kwargs["backend"] = backend
    if mode is not None:
        compile_kwargs["mode"] = mode

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
    logger.info_rank0(f"Compiled forward for {module_fqn} with torch.compile({compile_kwargs}).")
    return True


def compile_module_forwards(
    target_modules: Iterable[Tuple[str, nn.Module]],
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> int:
    compiled = 0
    seen_modules = set()
    for module_fqn, module in target_modules:
        if id(module) in seen_modules:
            continue
        seen_modules.add(id(module))
        if compile_module_forward(
            module,
            module_fqn=module_fqn,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        ):
            compiled += 1

    logger.info_rank0(f"Compiled {compiled} module forwards with torch.compile.")
    return compiled


def mark_compile_step_begin(enable_compile: bool) -> None:
    """Mark a new training step for CUDA Graph Trees managed by torch.compile."""

    if not enable_compile or get_device_type() != "cuda":
        return
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
