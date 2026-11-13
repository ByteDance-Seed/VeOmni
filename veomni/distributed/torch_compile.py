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
from typing import Collection, Optional, Sequence

import torch
import torch.nn as nn

from ..utils import device, logging


logger = logging.get_logger(__name__)


_SUPPORTED_MULTIMODAL_DECODER_BLOCKS = {
    "qwen3_vl": "Qwen3VLTextDecoderLayer",
}


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


def _decoder_block_specs(model: nn.Module) -> list[tuple[str, str | None]]:
    """``(class name, child scope)`` for each decoder layer in ``_no_split_modules``.

    An entry is either a bare class name (``LlamaDecoderLayer``, scope ``None``,
    matches anywhere) or scoped to a child (``janus_llama.LlamaDecoderLayer`` —
    see ``_scoped_no_split_modules``, matches only under that child).

    Keeping the scope — rather than reducing every entry to its class name — is
    what stops two children that happen to use the same layer class from both
    being compiled: the composed-model contract says a scoped target covers that
    child only. Only the scope is stripped from the *name* used for the class
    comparison, because :func:`_is_decoder_block` matches
    ``type(module).__name__``, which never carries one.

    Nothing upstream catches a bare-name mismatch: :func:`validate_compile_model`
    returns early for :class:`OmniModel` (it inherits
    ``input_modalities = "text"`` and its config has no ``vision_config`` /
    ``audio_config``, so :func:`_is_multimodal_model` is ``False``), so an
    unusable entry lets the call through and :func:`compile_decoder_blocks` then
    matches zero blocks — a ``RuntimeError`` from ``build_parallelize_model``
    after the FSDP wrap has already started.
    """
    no_split_modules = getattr(model, "_no_split_modules", None) or getattr(type(model), "_no_split_modules", None)
    if no_split_modules is None:
        return []
    specs: list[tuple[str, str | None]] = []
    for entry in no_split_modules:
        if not isinstance(entry, str):
            continue
        scope, _, class_name = entry.rpartition(".")
        if class_name.endswith("DecoderLayer"):
            specs.append((class_name, scope or None))
    return specs


def _decoder_block_class_names(model: nn.Module) -> set[str]:
    """Decoder-layer class names in ``_no_split_modules`` (scope stripped)."""
    return {class_name for class_name, _ in _decoder_block_specs(model)}


def _is_multimodal_model(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is not None and any(
        getattr(config, config_name, None) is not None for config_name in ("vision_config", "audio_config")
    ):
        return True

    input_modalities = getattr(model, "input_modalities", None) or getattr(type(model), "input_modalities", ())
    if isinstance(input_modalities, str):
        input_modalities = (input_modalities,)
    return any(modality != "text" for modality in input_modalities)


def validate_compile_model(
    model: nn.Module,
    compile_config: CompileConfig,
    sequence_parallel_enabled: bool = False,
    async_enabled: bool = False,
) -> None:
    """Validate model-specific contracts for per-block compilation.

    Text models keep the generic decoder-layer path. Multimodal models are
    fail-closed because their towers and cross-modal injection points have
    different shape and lifecycle contracts.
    """

    if not _is_multimodal_model(model):
        return

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    expected_decoder_block = _SUPPORTED_MULTIMODAL_DECODER_BLOCKS.get(model_type)
    if expected_decoder_block is None:
        raise RuntimeError(
            "model.accelerator.torch_compile.enable currently supports multimodal training only for dense Qwen3-VL "
            f"(model_type='qwen3_vl'); got model_type={model_type!r}."
        )
    if compile_config.dynamic:
        raise RuntimeError(
            "model.accelerator.torch_compile.enable for Qwen3-VL requires model.accelerator.torch_compile.dynamic=False."
        )
    if compile_config.uses_cuda_graphs():
        raise RuntimeError(
            "model.accelerator.torch_compile.enable for Qwen3-VL does not support CUDA Graph replay yet because packed "
            "FlashAttention metadata can vary between micro-batches; use backend='inductor' and mode=None."
        )
    if sequence_parallel_enabled or async_enabled:
        raise RuntimeError(
            "model.accelerator.torch_compile.enable for Qwen3-VL does not support Ulysses sequence parallelism, "
            "Ring Attention context parallelism, or async Ulysses yet; set model.accelerator.ulysses_size=1, "
            "model.accelerator.cp_size=1, and model.accelerator.enable_async=False."
        )

    decoder_block_class_names = _decoder_block_class_names(model)
    if expected_decoder_block not in decoder_block_class_names:
        raise RuntimeError(
            f"model.accelerator.torch_compile.enable for Qwen3-VL requires {expected_decoder_block!r} in model._no_split_modules."
        )


def _is_decoder_block(
    module: nn.Module,
    decoder_block_class_names: Optional[Collection[str]] = None,
    *,
    fqn: str | None = None,
    decoder_block_specs: Optional[Sequence[tuple[str, str | None]]] = None,
) -> bool:
    """Check decoder blocks using the model's ``_no_split_modules`` list.

    HuggingFace model classes typically list the decoder layer class in
    ``_no_split_modules`` for FSDP/Accelerate. Multimodal models may include
    vision/audio blocks in the same list, so the compile path keeps only
    entries that are decoder layers.

    When ``fqn`` and ``decoder_block_specs`` are given, a *scoped* entry
    (``janus_llama.LlamaDecoderLayer``) matches only modules under that child, so
    a sibling reusing the same layer class is left alone. A bare entry still
    matches anywhere.
    """

    if decoder_block_specs is not None:
        class_name = type(module).__name__
        if fqn is None:
            return any(name == class_name for name, _ in decoder_block_specs)
        return any(
            name == class_name and (scope is None or fqn == scope or fqn.startswith(scope + "."))
            for name, scope in decoder_block_specs
        )

    if decoder_block_class_names is None:
        decoder_block_class_names = _decoder_block_class_names(module)
    return type(module).__name__ in decoder_block_class_names


def validate_compile_config_for_fsdp2(compile_config: CompileConfig, enable_reshard_after_forward: bool) -> None:
    if not compile_config.enable:
        return
    if compile_config.uses_cuda_graphs() and enable_reshard_after_forward:
        raise RuntimeError(
            "model.accelerator.torch_compile with CUDA Graphs requires "
            "model.accelerator.fsdp_config.reshard_after_forward=False. "
            "Set model.accelerator.torch_compile.mode=None or disable forward resharding before enabling graph replay."
        )


def validate_compile_runtime(
    compile_config: CompileConfig,
    *,
    device_type: str,
    fsdp_enabled: bool,
    fsdp_mode: str,
    any_extra_parallel_enabled: bool,
    enable_reshard_after_forward: bool,
) -> None:
    """Validate runtime contracts before compile-specific setup begins."""

    if not compile_config.enable:
        return
    if not device.IS_CUDA_AVAILABLE or device_type != device.get_device_type():
        raise RuntimeError("model.accelerator.torch_compile.enable is CUDA-only for now.")
    if not fsdp_enabled:
        raise RuntimeError(
            "model.accelerator.torch_compile.enable requires FSDP2; compile without FSDP is not supported."
        )
    if fsdp_mode != "fsdp2":
        raise RuntimeError("model.accelerator.torch_compile.enable requires fsdp_mode='fsdp2'; DDP is not supported.")
    if any_extra_parallel_enabled:
        raise RuntimeError(
            "model.accelerator.torch_compile.enable currently does not support ExtraParallel models because EP all-to-all "
            "communication may be captured inside compiled blocks."
        )
    validate_compile_config_for_fsdp2(compile_config, enable_reshard_after_forward)


def compile_decoder_blocks(
    model: nn.Module,
    compile_config: CompileConfig,
    sequence_parallel_enabled: bool = False,
    async_enabled: bool = False,
) -> int:
    """Compile forward of every decoder block inside ``model`` in place.

    Compiling the forward method (rather than wrapping the whole module)
    preserves module identity for FSDP2 — pre/post-forward all-gather and
    reshard hooks stay outside the compiled region.
    """

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "model.accelerator.torch_compile.enable requires torch.compile, but this PyTorch build has no torch.compile."
        )

    validate_compile_model(model, compile_config, sequence_parallel_enabled, async_enabled)

    compile_kwargs = {
        "fullgraph": compile_config.fullgraph,
        "dynamic": compile_config.dynamic,
    }
    if compile_config.backend is not None:
        compile_kwargs["backend"] = compile_config.backend
    if compile_config.mode is not None:
        if compile_config.backend == "cudagraphs":
            raise ValueError(
                "model.accelerator.torch_compile.mode is not accepted by the 'cudagraphs' backend. "
                "Leave mode=None with backend='cudagraphs', or switch backend to 'inductor'."
            )
        compile_kwargs["mode"] = compile_config.mode

    decoder_block_specs = _decoder_block_specs(model)
    compiled = 0
    for fqn, module in model.named_modules():
        if not _is_decoder_block(module, fqn=fqn, decoder_block_specs=decoder_block_specs):
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

    if not enable_compile or not device.IS_CUDA_AVAILABLE:
        return
    mark_step_begin = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if mark_step_begin is not None:
        mark_step_begin()
