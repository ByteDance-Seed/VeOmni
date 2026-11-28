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

from ..utils import device, logging


logger = logging.get_logger(__name__)


@dataclass
class CompileConfig:
    """Runtime options for compiling FSDP2 decoder blocks."""

    enable: bool = False
    backend: Optional[str] = "inductor"
    mode: Optional[str] = None
    fullgraph: bool = True
    #: Passed straight to ``torch.compile``. ``None`` is torch's own default:
    #: the first call compiles a static graph and a shape change promotes it to
    #: a symbolic one, which then covers the rest of the run.
    #:
    #: ``False`` is not the safe choice here. With ``fullgraph=True`` a re-cache
    #: is a hard failure, not a silent recompile: one new shape per step
    #: exhausts ``torch._dynamo.config.cache_size_limit`` (8) and raises
    #: ``FailOnRecompileLimitHit`` — 8 patch counts are fine, the 9th is fatal.
    #:
    #: ``True`` also survives a shape that moves, but pays for it on the first
    #: call even when the shape never changes, and it makes Dynamo symbolically
    #: trace every custom autograd ``Function``. A ``Function`` that chains two
    #: Triton kernels then fails in ``overwrite_tensor_vt_proxy``: the first
    #: kernel's launch scalars come back as constants while the outer graph still
    #: holds symbols for them. The fused-Triton MoE experts are built that way.
    #: ``None`` keeps the first compile static and identical to ``False``; only a
    #: real second shape reaches the symbolic path.
    dynamic: Optional[bool] = None

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


def _block_fqns(model: nn.Module) -> list[str]:
    """Module paths of the blocks named in ``_no_split_modules``.

    That list is the wrap-unit list FSDP already uses, so compiling it verbatim
    keeps the compiled set and the sharded set identical — the whole model, not a
    subset picked by class name.

    An entry is a bare class name (``LlamaDecoderLayer``) matching anywhere, or —
    on a SeedOmni composed model, whose children share one FSDP tree — scoped to
    a child (``janus_llama.LlamaDecoderLayer``, see ``_scoped_no_split_modules``)
    matching only under it.
    """
    entries = getattr(model, "_no_split_modules", None) or getattr(type(model), "_no_split_modules", None)
    if not entries:
        return []
    wanted = []
    for entry in entries:
        if isinstance(entry, str):
            scope, _, class_name = entry.rpartition(".")
            wanted.append((class_name, scope))
    return [
        fqn
        for fqn, module in model.named_modules()
        if fqn
        and any(
            class_name == type(module).__name__ and (not scope or fqn == scope or fqn.startswith(scope + "."))
            for class_name, scope in wanted
        )
    ]


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
    sequence_parallel_enabled: bool = False,
    async_parallel_enabled: bool = False,
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
    # Ulysses/CP slice activations along the sequence dimension, which the graph
    # edges of a compiled block reorder; async Ulysses pipelines the same
    # exchanges across steps. Both are refused for the same reason under SeedOmni
    # (``_validate_omni_accelerator``), so the single-model path refuses them too
    # rather than silently compiling a block that cannot run.
    if sequence_parallel_enabled:
        raise RuntimeError(
            "model.accelerator.torch_compile.enable does not support sequence parallelism: "
            "set accelerator.ulysses_size=1 and accelerator.cp_size=1."
        )
    if async_parallel_enabled:
        raise RuntimeError(
            "model.accelerator.torch_compile.enable does not support async Ulysses: "
            "set accelerator.enable_async=False."
        )
    validate_compile_config_for_fsdp2(compile_config, enable_reshard_after_forward)


# The torchao row selects a quantization format per op family. Preserve the
# existing MXFP4 restrictions; the MXFP8 gated MLP restriction is Inductor-only.
_MXFP4_FORMATS = (("linear", "mxfp4"), ("gated_mlp", "mxfp4"))
_INDUCTOR_FORMATS = (*_MXFP4_FORMATS, ("gated_mlp", "mxfp8"))


def _bound_uncompilable_ops(model: nn.Module, formats: tuple[tuple[str, str], ...]) -> tuple[bool, list[str]]:
    """Inspect concrete ``VeomniOp`` handles bound on ``model`` for uncompilable formats."""
    from ..ops.registry import VeomniOp

    has_bound_ops = False
    offending: list[str] = []
    for module in model.modules():
        for attr in vars(module).values():
            if isinstance(attr, VeomniOp):
                has_bound_ops = True
                dtype = attr.kwargs.get("dtype")
                if attr.impl == "torchao" and (attr.op, dtype) in formats:
                    entry = f"{attr.op}_implementation=torchao(dtype={dtype!r})"
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


def _uncompilable_selections(ops, formats: tuple[tuple[str, str], ...]) -> list[str]:
    """Return the offending ``field=row(dtype=...)`` strings on one ops config, if any."""
    if ops is None:
        return []
    offending: list[str] = []
    for op, dtype in formats:
        field = f"{op}_implementation"
        if _selection_reaches(getattr(ops, field, None), "torchao", dtype):
            offending.append(f"{field}=torchao(dtype={dtype!r})")
    return offending


def _reject_incompatible_ops(model: nn.Module, backend: str | None = "inductor") -> None:
    """Reject op selections incompatible with the requested compiler backend, at any config depth.

    On a concrete model that binds ``VeomniOp`` handles, inspect only those
    resolved handles so unmatched overrides or op families the model never
    binds (such as ``linear_implementation`` on Gemma3) do not falsely block
    compilation. Fall back to walking ``config`` / ``sub_configs`` when no
    handles are bound on ``model`` — that path is what a SeedOmni module takes.

    Called once per compiled module, so a sibling that stayed eager never reaches
    this — its selection is irrelevant because Inductor never sees it.
    """
    # None leaves torch.compile on its default Inductor backend.
    formats = _INDUCTOR_FORMATS if backend in (None, "inductor") else _MXFP4_FORMATS
    has_bound_ops, offending = _bound_uncompilable_ops(model, formats)
    if not has_bound_ops:
        config = getattr(model, "config", None)
        offending = _uncompilable_at(config, prefix="", formats=formats)
    if offending:
        reasons = []
        if any("dtype='mxfp4'" in entry for entry in offending):
            reasons.append("MXFP4 uses float8_e8m0fnu scales that Inductor cannot lower")
        if any("dtype='mxfp8'" in entry for entry in offending):
            reasons.append(
                "on B200 with PyTorch 2.11, Inductor changed the down-projection weight gradient "
                "in both native and emulated MXFP8 gated MLPs. Use backend='aot_eager' for this row"
            )
        raise RuntimeError(
            f"model.accelerator.torch_compile cannot be combined with {', '.join(offending)}: "
            f"{' and '.join(reasons)}. Disable compilation or select a different dtype."
        )


def _uncompilable_at(config, *, prefix: str, formats: tuple[tuple[str, str], ...] = _INDUCTOR_FORMATS) -> list[str]:
    """Collect offending selections on ``config`` and every sub-config below it.

    A SeedOmni composed ``OmniConfig`` files its children in ``_module_configs``
    rather than ``sub_configs`` (the latter describes one HF model's nesting), so
    both are walked. Under the per-module design this only ever sees one module's
    config, but the walk stays general so a single HF multimodal model is still
    covered.
    """
    if config is None:
        return []
    found = [f"{prefix}{item}" for item in _uncompilable_selections(getattr(config, "ops_config", None), formats)]
    for name in getattr(config, "sub_configs", {}) or {}:
        child = getattr(config, name, None)
        if child is not None:
            found.extend(_uncompilable_at(child, prefix=f"{prefix}{name}.", formats=formats))
    for name, child in (getattr(config, "_module_configs", None) or {}).items():
        found.extend(_uncompilable_at(child, prefix=f"{prefix}{name}.", formats=formats))
    return found


def compile_decoder_blocks(model: nn.Module, compile_config: CompileConfig) -> int:
    """Compile forward of every block inside ``model`` in place.

    Compiling the forward method (rather than wrapping the whole module)
    preserves module identity for FSDP2 — pre/post-forward all-gather and
    reshard hooks stay outside the compiled region.

    The blocks are the ones ``_no_split_modules`` names, so the compiled set and
    the FSDP wrap set are the same (see :func:`_block_fqns`). Nothing here asks
    what kind of model it is; the runtime contracts are checked once by the
    caller through :func:`validate_compile_runtime`.
    """

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "model.accelerator.torch_compile.enable requires torch.compile, but this PyTorch build has no torch.compile."
        )

    _reject_incompatible_ops(model, compile_config.backend)

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

    from ..ops.torchao import iter_mxfp8_projections, validate_mxfp8_projection_hooks

    blocks = [(fqn, model.get_submodule(fqn)) for fqn in _block_fqns(model)]
    # Dynamo can reuse one block's graph for a sibling with different hooks.
    # Inspect each selected projection before handing any block to the compiler.
    for fqn, block in blocks:
        for path, projection, _ in iter_mxfp8_projections(block):
            if projection is not None:
                validate_mxfp8_projection_hooks(projection, f"{fqn}.{path}")

    compiled = 0
    for fqn, module in blocks:
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
        logger.info_rank0(f"Compiled block forward for {fqn} with torch.compile({compile_kwargs}).")
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
