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


def _decoder_block_specs(model: nn.Module, *, all_classes: bool = False) -> list[tuple[str, str | None]]:
    """``(class name, child scope)`` for each compilable block in ``_no_split_modules``.

    An entry is a bare class name (``LlamaDecoderLayer``, scope ``None``, matches
    anywhere) or — on a SeedOmni composed model, where the children share one
    FSDP tree — scoped to a child (``janus_llama.LlamaDecoderLayer``, matches
    only under that child; see ``_scoped_no_split_modules`` in
    ``models/seed_omni/accelerated/omni_model/omni_model_runtime.py``).

    ``type(module).__name__`` never carries a scope, so the name half is what
    ``_is_decoder_block`` compares and the scope half is what it matches against
    the fqn.

    ``all_classes`` decides which entries qualify. The default keeps only
    ``*DecoderLayer``, which is what every single model wants: a Qwen2.5-Omni
    lists its vision and audio towers alongside the decoder layers, and those
    must stay eager. A SeedOmni run passes ``True`` because its modules declare
    whatever their own blocks are called (``JanusVQVAEAttnBlock``), and the
    per-module
    :attr:`~...modules.module_modeling_base.PretrainedOmniModule.supports_torch_compile`
    flag — not the class name — is what decides whether the module is compiled
    at all.
    """
    no_split_modules = getattr(model, "_no_split_modules", None) or getattr(type(model), "_no_split_modules", None)
    if no_split_modules is None:
        return []
    specs: list[tuple[str, str | None]] = []
    for entry in no_split_modules:
        if not isinstance(entry, str):
            continue
        scope, _, class_name = entry.rpartition(".")
        if all_classes or class_name.endswith("DecoderLayer"):
            specs.append((class_name, scope or None))
    return specs


def _decoder_block_class_names(model: nn.Module) -> set[str]:
    """Decoder-layer class names in ``_no_split_modules``, scopes stripped.

    Kept for :func:`validate_compile_model`, which only asks whether a class is
    declared at all.
    """
    return {class_name for class_name, _ in _decoder_block_specs(model)}


def _is_composed_model(model: nn.Module) -> bool:
    """Whether ``model`` is a SeedOmni composed :class:`OmniModel`.

    Recognised by the per-module config map, which no single model carries. A
    composed model presents as text-only to :func:`_is_multimodal_model`: it
    inherits ``input_modalities = "text"`` from ``PreTrainedModel`` and its
    ``OmniConfig`` has no ``vision_config`` / ``audio_config`` — the media
    modalities live in ``_module_configs``.
    """
    config = getattr(model, "config", None)
    return bool(getattr(config, "_module_configs", None))


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

    if _is_composed_model(model):
        # Each child is compiled block-by-block under the same rules as a single
        # model, but the cross-module graph and the per-module op selections make
        # the CUDA-graph and sequence-parallel paths unverified: replay would need
        # one mark per micro-batch across the composed forward, and Ulysses
        # would slice activations the graph edges reorder. Keep them off.
        #
        # ``dynamic`` is allowed, and is the setting to use when the packed
        # sequence length moves between steps (a sample with one image against a
        # sample with four). Measured with dynamic=True over six lengths spanning
        # 300-2816 tokens: every shape matched eager to the same tolerance as a
        # static compile (2.4e-2 relative on a bf16 stack, saturating at four
        # layers and identical for forward and gradients), six lengths kept four
        # graphs, and the variable-shape step was 1.41x faster than eager.
        if compile_config.uses_cuda_graphs():
            raise RuntimeError(
                "model.accelerator.torch_compile.enable does not support CUDA Graph replay for a composed "
                "SeedOmni OmniModel; use backend='inductor' and mode=None."
            )
        if sequence_parallel_enabled or async_enabled:
            raise RuntimeError(
                "model.accelerator.torch_compile.enable does not support sequence parallelism or async Ulysses "
                "for a composed SeedOmni OmniModel; set model.accelerator.ulysses_size=1, "
                "model.accelerator.cp_size=1, and model.accelerator.enable_async=False."
            )
        return

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
    fqn: Optional[str] = None,
    decoder_block_specs: Optional[Sequence[tuple[str, str | None]]] = None,
) -> bool:
    """Check decoder blocks using the model's ``_no_split_modules`` list.

    HuggingFace model classes typically list the decoder layer class in
    ``_no_split_modules`` for FSDP/Accelerate. Multimodal models may include
    vision/audio blocks in the same list, so the compile path keeps only
    entries that are decoder layers.

    Given ``decoder_block_specs``, a scoped entry matches only modules under
    that child, so a sibling that reuses the same layer class is left alone; a
    bare entry still matches anywhere. The rule mirrors
    ``_is_fsdp_wrap_target``, so the wrap plan and the compile plan agree on
    what a block is.
    """

    if decoder_block_specs is not None:
        class_name = type(module).__name__
        return any(
            name == class_name and (scope is None or _under_scope(fqn, scope)) for name, scope in decoder_block_specs
        )

    if decoder_block_class_names is None:
        decoder_block_class_names = _decoder_block_class_names(module)
    return type(module).__name__ in decoder_block_class_names


def _under_scope(fqn: Optional[str], scope: str) -> bool:
    """Whether ``fqn`` is ``scope`` itself or a descendant of it."""
    if fqn is None:
        return False
    return fqn == scope or fqn.startswith(scope + ".")


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


#: Op rows whose kernels emit dtypes Inductor cannot lower.
#:
#: ``torchao_mxfp4`` materializes ``float8_e8m0fnu`` scale tensors, and Inductor's
#: codegen has no lowering for that dtype. A runtime check cannot catch it, so the
#: selection is refused here, by config field. Formats whose scales use a dtype
#: Inductor knows (``torchao_nvfp4`` uses ``float8_e4m3fn`` plus an FP32
#: per-tensor scale) stay out of this table.
_UNCOMPILABLE_IMPLS: dict[str, str] = {
    "linear_implementation": "torchao_mxfp4",
    "gated_mlp_implementation": "torchao_mxfp4",
}


def _uncompilable_selections(ops) -> list[str]:
    """Return the offending ``field=impl`` strings on one ops config, if any."""
    if ops is None:
        return []
    return [
        f"{field}={getattr(ops, field)!r}"
        for field, impl in _UNCOMPILABLE_IMPLS.items()
        if getattr(ops, field, None) == impl
    ]


def _reject_incompatible_ops(model: nn.Module) -> None:
    """Reject op selections ``torch.compile`` cannot lower, at any config depth.

    The walk covers nested configs because that is how the selections bind: a
    multimodal root carries its text tower's ops under ``sub_configs``, and a
    module resolves the selection from the config it was built with. A child-only
    selection is real and must be caught, and the message names the path so the
    offending level is unambiguous.
    """
    config = getattr(model, "config", None)
    offending = _uncompilable_at(config, prefix="")
    if offending:
        raise RuntimeError(
            "model.accelerator.torch_compile cannot be combined with "
            f"{', '.join(offending)}: those rows emit tensors of a dtype Inductor cannot lower "
            "(float8_e8m0fnu for torchao_mxfp4). Disable compilation or select a different row."
        )


def _uncompilable_at(config, *, prefix: str) -> list[str]:
    """Collect offending selections on ``config`` and every sub-config below it.

    A SeedOmni composed ``OmniConfig`` files its children in ``_module_configs``
    rather than ``sub_configs`` (the latter describes one HF model's nesting), so
    both are walked — otherwise a child's ``torchao_mxfp4`` selection would never
    be seen and would fail at runtime instead, inside Inductor.
    """
    if config is None:
        return []
    found = [f"{prefix}{item}" for item in _uncompilable_selections(getattr(config, "ops_config", None))]
    for name in getattr(config, "sub_configs", {}) or {}:
        child = getattr(config, name, None)
        if child is not None:
            found.extend(_uncompilable_at(child, prefix=f"{prefix}{name}."))
    for name, child in (getattr(config, "_module_configs", None) or {}).items():
        found.extend(_uncompilable_at(child, prefix=f"{prefix}{name}."))
    return found


def compile_decoder_blocks(
    model: nn.Module,
    compile_config: CompileConfig,
    sequence_parallel_enabled: bool = False,
    async_enabled: bool = False,
    scopes: Optional[Collection[str]] = None,
) -> int:
    """Compile forward of every decoder block inside ``model`` in place.

    Compiling the forward method (rather than wrapping the whole module)
    preserves module identity for FSDP2 — pre/post-forward all-gather and
    reshard hooks stay outside the compiled region.

    ``scopes`` narrows the set to blocks under those children. A SeedOmni
    composed model passes the modules that opted into ``torch.compile``, so a
    sibling that did not opt in keeps running eagerly inside the same run.
    ``None`` keeps every declared decoder block.
    """

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "model.accelerator.torch_compile.enable requires torch.compile, but this PyTorch build has no torch.compile."
        )

    validate_compile_model(model, compile_config, sequence_parallel_enabled, async_enabled)
    _reject_incompatible_ops(model)

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

    # ``scopes`` implies a SeedOmni run, where a module's blocks are named
    # whatever its family calls them and the per-module flag decides who is
    # compiled. Without it, keep the HF convention of ``*DecoderLayer`` only.
    decoder_block_specs = _decoder_block_specs(model, all_classes=scopes is not None)
    compiled = 0
    for fqn, module in model.named_modules():
        if not _is_decoder_block(module, fqn=fqn, decoder_block_specs=decoder_block_specs):
            continue
        if scopes is not None and not any(_under_scope(fqn, scope) for scope in scopes):
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
