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
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..utils import device, logging


logger = logging.get_logger(__name__)


#: Which callers this module serves, and the signal each is recognised by.
#:
#: ``validate_compile_model`` gates four kinds of model, and the branch each one
#: takes is decided from the model's *topology*, not from a single flag — the
#: signals differ because the models differ:
#:
#: =========================  ==========================================  ====================================
#: caller                     recognised by                               what it gets
#: =========================  ==========================================  ====================================
#: base trainer, text model   ``_is_multimodal_model`` is ``False``       the generic ``*DecoderLayer`` path
#: base trainer, multimodal   ``_is_multimodal_model`` is ``True``        fail-closed: dense Qwen3-VL only
#: SeedOmni module            ``_is_omni_module`` (``isinstance``)        composed guards (no CUDA graph, no SP)
#: SeedOmni composed          ``_is_composed_model`` (``_module_configs``)  composed guards
#: =========================  ==========================================  ====================================
#:
#: Why the multimodal branch is narrower than the SeedOmni ones: a single HF
#: multimodal model holds its towers **and** the cross-modal injection between
#: them in one ``forward``, so its block boundaries and lifecycle are not the
#: plain decoder-block contract. Only dense Qwen3-VL has been validated, and
#: ``_SUPPORTED_MULTIMODAL_DECODER_BLOCKS`` names what that validation covered.
#: A SeedOmni module holds one component and no injection point, so it gets the
#: composed guards instead — even though its config carries a ``vision_config``
#: and so looks multimodal to ``_is_multimodal_model``.
#:
#: A new caller normally adds a branch here. The alternative — moving the answer
#: onto the model as a capability flag, the way
#: :attr:`~...modules.module_modeling_base.PretrainedOmniModule.supports_torch_compile`
#: already works for SeedOmni — would collapse this to one check plus the shared
#: runtime constraints, and retire ``_SUPPORTED_MULTIMODAL_DECODER_BLOCKS`` (the
#: table exists only to express "we validated this one"). That refactor has not
#: been done because re-validating the existing Qwen3-VL support needs a real
#: training run.

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
        """Return whether this config asks torch.compile to use graph replay.

        Both ``mode`` spellings are needed: ``reduce-overhead`` and
        ``max-autotune`` each turn on Triton graph capture in torch's own mode
        table (``torch/_inductor/__init__.py``), while
        ``max-autotune-no-cudagraphs`` deliberately does not. Missing
        ``max-autotune`` would let a run past the SeedOmni replay guard and record
        ``_veomni_compile_uses_cuda_graphs=False``, so the per-step
        ``cudagraph_mark_step_begin`` would never be called for a graph that does
        replay.
        """

        return self.backend == "cudagraphs" or self.mode == "reduce-overhead" or self.mode == "max-autotune"


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


def _is_omni_module(model: nn.Module) -> bool:
    """Whether ``model`` is one SeedOmni module, possibly behind a wrapper.

    Under ``fsdp_scope='module'`` each module is wrapped on its own, so the wrap
    sees a single component rather than a whole multimodal model. That matters
    for the guards below: a SeedOmni tower carries a ``vision_config`` because
    that is how its own config is shaped, not because it also holds an LLM and
    the cross-modal injection between them. A single HF multimodal model
    (Qwen2.5-Omni, Qwen3-VL-MoE, ...) does hold all of that and stays
    fail-closed.

    The wrapper is looked through. A LoRA module is a
    :class:`~veomni.lora.VeOmniLoraModel` (a plain ``nn.Module``) holding the
    SeedOmni module on ``get_base_model()`` / ``.model``, so an ``isinstance``
    against the outer object would miss it and send the tower through the
    single-model multimodal rule instead — which rejects every SeedOmni family
    except dense Qwen3-VL.
    """
    from ..models.seed_omni.accelerated.utils.dispatch import unwrap_module_chain
    from ..models.seed_omni.modules.module_modeling_base import PretrainedOmniModule

    return isinstance(unwrap_module_chain(model), PretrainedOmniModule)


def validate_compile_model(
    model: nn.Module,
    compile_config: CompileConfig,
    sequence_parallel_enabled: bool = False,
    async_enabled: bool = False,
) -> None:
    """Validate model-specific contracts for per-block compilation.

    Four callers, four branches — see the module docstring for the table and for
    why the single-model multimodal rule is narrower than the SeedOmni ones.
    Text models keep the generic decoder-layer path. A single multimodal model is
    fail-closed because its towers and cross-modal injection points have
    different shape and lifecycle contracts. A SeedOmni module or composed
    ``OmniModel`` is compiled block by block and gets the composed guards.
    """

    # A composed ``OmniModel`` and a single SeedOmni module share these guards:
    # both compile block by block, and the two paths that stay refused are
    # refused for reasons that do not depend on which of the two it is. The
    # parse-time twin lives in ``_validate_omni_accelerator`` — it runs for the
    # top-level accelerator and every resolved module, so this is the backstop
    # for a caller that built a config without going through the launcher.
    #
    # ``dynamic`` is allowed and is what a run wants when the packed length moves
    # between steps. Measured over six lengths spanning 300-2816 tokens: every
    # shape matched eager to the same tolerance a static compile gives (2.4e-2
    # relative on a bf16 stack, saturating at four layers and identical for
    # forward and gradients), six lengths kept four graphs, and the variable-shape
    # step was 1.41x faster than eager.
    if _is_composed_model(model) or _is_omni_module(model):
        kind = "a composed SeedOmni OmniModel" if _is_composed_model(model) else "a SeedOmni module"
        if compile_config.uses_cuda_graphs():
            raise RuntimeError(
                f"model.accelerator.torch_compile.enable does not support CUDA Graph replay for {kind}; "
                "use backend='inductor' and mode=None."
            )
        if sequence_parallel_enabled or async_enabled:
            raise RuntimeError(
                f"model.accelerator.torch_compile.enable does not support sequence parallelism or async Ulysses "
                f"for {kind}; set model.accelerator.ulysses_size=1, model.accelerator.cp_size=1, "
                "and model.accelerator.enable_async=False."
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
    *,
    fqn: Optional[str] = None,
    decoder_block_specs: Sequence[tuple[str, str | None]],
) -> bool:
    """Whether ``module`` is one of the blocks named in ``_no_split_modules``.

    HuggingFace model classes list the decoder layer class there for
    FSDP/Accelerate. A scoped entry (``janus_llama.LlamaDecoderLayer``) matches
    only modules under that child, so a sibling that reuses the same layer class
    is left alone; a bare entry matches anywhere. The rule mirrors
    ``_is_fsdp_wrap_target``, so the wrap plan and the compile plan agree on what
    a block is.
    """

    class_name = type(module).__name__
    return any(
        name == class_name and (scope is None or _under_scope(fqn, scope)) for name, scope in decoder_block_specs
    )


def _under_scope(fqn: Optional[str], scope: str) -> bool:
    """Whether ``fqn`` is ``scope`` itself or a descendant of it.

    An empty scope means "this whole tree", which is how a standalone module
    (``fsdp_scope='module'``) says its own root is the scope.
    """
    if not scope:
        return True
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


#: Op rows whose kernels emit dtypes Inductor cannot lower, keyed by
#: ``(config field, row name)`` and giving the format that triggers it.
#:
#: ``mxfp4`` materializes ``float8_e8m0fnu`` scale tensors, and Inductor's codegen
#: has no lowering for that dtype. A runtime check cannot catch it, so the selection
#: is refused here, from the config. It is keyed on the *format* rather than the row
#: because the row is ``torchao`` for both FP4 layouts: a config may compile with
#: ``dtype: nvfp4`` (float8_e4m3fn scales plus an FP32 per-tensor scale, both of
#: which Inductor knows) and must not with ``dtype: mxfp4``.
_UNCOMPILABLE_FORMATS: dict[str, tuple[str, str, str]] = {
    "linear_implementation": ("linear", "torchao", "mxfp4"),
    "gated_mlp_implementation": ("gated_mlp", "torchao", "mxfp4"),
}


def _bound_uncompilable_ops(model: nn.Module) -> tuple[bool, list[str]]:
    """Inspect concrete ``VeomniOp`` handles bound on ``model`` for uncompilable formats."""
    from ..ops.registry import VeomniOp

    op_to_rule = {op: (field, impl, dtype) for field, (op, impl, dtype) in _UNCOMPILABLE_FORMATS.items()}
    has_bound_ops = False
    offending: list[str] = []
    for module in model.modules():
        for attr in vars(module).values():
            if isinstance(attr, VeomniOp):
                has_bound_ops = True
                if attr.op in op_to_rule:
                    field, impl, dtype = op_to_rule[attr.op]
                    if attr.impl == impl and attr.kwargs.get("dtype") == dtype:
                        entry = f"{field}={impl}(dtype={dtype!r})"
                        if entry not in offending:
                            offending.append(entry)
    return has_bound_ops, offending


def _selection_reaches(value, target_impl: str, target_dtype: str) -> bool:
    """Return whether any ordered subset of overrides on ``value`` reaches ``(target_impl, target_dtype)``."""
    base_name = getattr(value, "name", value)
    base_kwargs = getattr(value, "kwargs", {}) or {}
    states: set[tuple[str | None, str | None]] = {(base_name, base_kwargs.get("dtype"))}
    for rule in getattr(value, "overrides", ()) or ():
        rule_kwargs = rule.kwargs or {}
        next_states = set(states)
        for name, dtype in states:
            if rule.name is not None and name is not None and rule.name != name:
                next_states.add((rule.name, rule_kwargs.get("dtype")))
            else:
                next_states.add((rule.name if rule.name is not None else name, rule_kwargs.get("dtype", dtype)))
        states = next_states
    return (target_impl, target_dtype) in states


def _uncompilable_selections(ops) -> list[str]:
    """Return the offending ``field=row(dtype=...)`` strings on one ops config, if any."""
    if ops is None:
        return []
    offending: list[str] = []
    for field, (_op, impl, dtype) in _UNCOMPILABLE_FORMATS.items():
        if _selection_reaches(getattr(ops, field, None), impl, dtype):
            offending.append(f"{field}={impl}(dtype={dtype!r})")
    return offending


def _reject_incompatible_ops(model: nn.Module) -> None:
    """Reject op selections ``torch.compile`` cannot lower, at any config depth.

    On a concrete model that binds ``VeomniOp`` handles, inspect only those
    resolved handles so unmatched overrides or op families the model never
    binds (such as ``linear_implementation`` on Gemma3) do not falsely block
    compilation. Fall back to walking ``config`` / ``sub_configs`` when no
    handles are bound on ``model`` — that path is what a SeedOmni module takes.

    Called once per compiled module, so a sibling that stayed eager never reaches
    this — its selection is irrelevant because Inductor never sees it.
    """
    has_bound_ops, offending = _bound_uncompilable_ops(model)
    if not has_bound_ops:
        config = getattr(model, "config", None)
        offending = _uncompilable_at(config, prefix="")
    if offending:
        raise RuntimeError(
            "model.accelerator.torch_compile cannot be combined with "
            f"{', '.join(offending)}: those selections emit tensors of a dtype Inductor cannot lower "
            "(float8_e8m0fnu for the mxfp4 layout). Disable compilation or select a different dtype."
        )


def _uncompilable_at(config, *, prefix: str) -> list[str]:
    """Collect offending selections on ``config`` and every sub-config below it.

    A SeedOmni composed ``OmniConfig`` files its children in ``_module_configs``
    rather than ``sub_configs`` (the latter describes one HF model's nesting), so
    both are walked. Under the per-module design this only ever sees one module's
    config, but the walk stays general so a single HF multimodal model is still
    covered.
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
    all_classes: bool = False,
) -> int:
    """Compile forward of every block inside ``model`` in place.

    Compiling the forward method (rather than wrapping the whole module)
    preserves module identity for FSDP2 — pre/post-forward all-gather and
    reshard hooks stay outside the compiled region.

    ``all_classes`` decides which ``_no_split_modules`` entries qualify. The
    default keeps ``*DecoderLayer``, which is what a single HF model wants: a
    Qwen2.5-Omni lists its vision and audio towers alongside its decoder layers,
    and those must stay eager. SeedOmni passes ``True`` because its modules name
    whatever their own blocks are called (``JanusVQVAEAttnBlock``) and already
    opted in via ``supports_torch_compile`` — the module is the unit of
    compilation, so there is no subtree left to select.
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

    decoder_block_specs = _decoder_block_specs(model, all_classes=all_classes)
    compiled = 0
    for fqn, module in model.named_modules():
        if not _is_decoder_block(module, fqn=fqn, decoder_block_specs=decoder_block_specs):
            continue
        if getattr(module, "_veomni_forward_compiled", False):
            continue

        # One config for this call: a SeedOmni module passes its own, so `dynamic`
        # is already the answer for this module's blocks.
        block_kwargs = compile_kwargs

        original_forward = module.forward
        if hasattr(original_forward, "__func__"):
            module._veomni_original_forward = original_forward.__func__
            compiled_forward = torch.compile(original_forward.__func__, **block_kwargs)
            module.forward = types.MethodType(compiled_forward, module)
        else:
            module._veomni_original_forward = original_forward
            module.forward = torch.compile(original_forward, **block_kwargs)
        module._veomni_forward_compiled = True
        module._veomni_compile_config = dict(block_kwargs)
        logger.info_rank0(f"Compiled decoder block forward for {fqn} with torch.compile({block_kwargs}).")
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
