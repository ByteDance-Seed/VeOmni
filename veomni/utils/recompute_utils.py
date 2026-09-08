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

import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

from veomni.utils import helper


logger = helper.create_logger(__name__)


def string_to_op(op_string: str) -> Any:
    """
    Convert a single operation string to PyTorch operation object

    Args:
        op_string: e.g. "aten.addmm.default" or "torch.ops.flash_attn._flash_attn_forward.default"

    Returns:
        PyTorch operation object
    """
    global torch
    # Clean the string
    clean_string = op_string.strip()

    # Remove torch.ops. prefix (if exists)
    if clean_string.startswith("torch.ops."):
        clean_string = clean_string[len("torch.ops.") :]

    # Split path and access level by level
    parts = clean_string.split(".")

    # Check if torch.ops is available
    if not hasattr(torch, "ops"):
        raise AttributeError("torch.ops not available in this PyTorch version")

    current = torch.ops

    # Special handling: ensure accessing aten operations by first trying to trigger registration
    if parts[0] == "aten":
        try:
            # Try to access a basic aten operation to trigger module loading
            _ = torch.ops.aten.add
        except AttributeError:
            # If cannot access aten, may need to import related modules
            try:
                import torch._C._dispatch
            except ImportError:
                pass

    for i, part in enumerate(parts):
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            # More detailed error information, including current path
            current_path = ".".join(parts[:i])
            available_attrs = dir(current) if hasattr(current, "__dict__") else []
            raise AttributeError(
                f"Operation '{op_string}' not found. "
                f"Missing attribute: '{part}' at path 'torch.ops.{current_path}'. "
                f"Available attributes: {available_attrs[:10]}{'...' if len(available_attrs) > 10 else ''}"
            )

    return current


def convert_ops_to_objects(ops_strings: List[str]) -> List[Any]:
    """
    Convert operation string list to operation object list
    Args:
        ops_strings: String list

    Returns:
        PyTorch operation object list
    """
    ops_objects = []
    failed_ops = []

    # First perform environment check
    _check_torch_ops_availability()

    for op_str in ops_strings:
        try:
            op_obj = string_to_op(op_str)
            ops_objects.append(op_obj)
            logger.info_rank0(f"✓ Conversion successful: {op_str}")
            assert isinstance(op_obj, torch._ops.OpOverload), "Please check if the ops is end with .default"
        except (AttributeError, TypeError) as e:
            logger.info_rank0(f"✗ Conversion failed: {op_str} - {e}")
            failed_ops.append(op_str)
        except Exception as e:
            logger.info_rank0(f"✗ Conversion failed: {op_str} - {e}")
            raise e

    if failed_ops:
        logger.info_rank0(f"\nWarning: {len(failed_ops)} operations failed to convert")
        logger.info_rank0("Possible reasons:")
        logger.info_rank0("1. PyTorch version does not support certain operations")
        logger.info_rank0("2. Missing related extension modules (e.g. flash_attn)")
        logger.info_rank0("3. Operation name spelling error")

    return ops_objects


# ---------------------------------------------------------------------------
# Unified block-level checkpoint entry (framework side)
# ---------------------------------------------------------------------------
#
# Wrap every checkpoint point (e.g. one DiT block per call) with
# ``checkpoint_forward`` — it picks the active strategy from the process-level
# switch, so model code never re-implements checkpoint dispatch:
#
#     hidden = checkpoint_forward(block, use_gradient_checkpointing,
#                                 use_gradient_checkpointing_offload, hidden, ...)
#
# Priority (torch-native only, no deepspeed): offload (save_on_cpu) > SAC
# (selective) > full checkpoint > direct call. ``context_fn`` (kw-only)
# overrides the process-level selective switch.


def _create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)

    return custom_forward


def checkpoint_forward(
    model,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
    *args,
    context_fn=None,
    layer_index: Optional[int] = None,
    **kwargs,
):
    """Gradient checkpoint wrapper (framework entry point).

    Torch-native (no deepspeed); direct call when checkpointing is off.
    Priority: offload > selective (SAC) > full checkpoint > direct call.

    kw-only semantics:
    - ``context_fn``: given = use it instead of the process-level SAC switch.
    - ``layer_index``: block index of the wrapped module. With a positive
      ``gradient_checkpoint_layers`` selection, indices outside it bypass
      checkpointing (direct call); with a positive
      ``selective_gradient_checkpoint_layers`` subset, layers outside it fall
      back to full recomputation (no SAC context). ``None`` never filters.
    """
    # Positive layer selection: only the configured block indices are wrapped
    # in checkpointing (any variant); everything else runs directly.
    layers = _state["layers"]
    if layer_index is not None and layers is not None and layer_index not in layers:
        return model(*args, **kwargs)

    if use_gradient_checkpointing and not use_gradient_checkpointing_offload:
        if context_fn is None:
            context_fn = get_context_fn(layer_index)
        if context_fn is not None:
            return torch.utils.checkpoint.checkpoint(
                _create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
                context_fn=context_fn,
            )
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(
                _create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    if use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            _create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    return model(*args, **kwargs)


def _check_torch_ops_availability():
    global torch
    # Check if torch.ops is available
    if not hasattr(torch, "ops"):
        raise RuntimeError("torch.ops is not available in current PyTorch version")

    # Check basic aten operations
    try:
        _ = torch.ops.aten.add
        logger.info_rank0("✓ torch.ops.aten available")
    except AttributeError as e:
        logger.info_rank0(f"✗ torch.ops.aten not available: {e}")
        logger.info_rank0("Trying to import necessary modules...")
        try:
            import torch._C._dispatch

            logger.info_rank0("✓ Successfully imported torch._C._dispatch")
        except ImportError as e:
            logger.info_rank0(f"✗ Cannot import torch._C._dispatch: {e}")


# ---------------------------------------------------------------------------
# Selective activation checkpointing (SAC)
# ---------------------------------------------------------------------------
#
# Wraps PyTorch 2.9 native SAC
# (``torch.utils.checkpoint.create_selective_checkpoint_contexts``): a per-op
# policy decides which forward outputs are saved (not recomputed in backward).
# Default policy (Megatron/Flash selective style): attention ops keep outputs
# (``MUST_SAVE``), the rest are recomputed.
#
# Process-level switch via ``configure``; ``checkpoint_forward`` reads it
# through ``get_context_fn`` — no model signature changes.
#
#     configure(enabled=True)  # default attention set
#     configure(enabled=True, extra_op_names=["aten.aten_mm.default"])
#
# Known restrictions (degrades silently, see ``configure``): activation
# offload, Ulysses sequence parallel, reentrant checkpointing, torch.compile.

#: Substring tokens matched against an operator's ``namespace::name`` as a
#: fallback when exact ``OpOverload`` resolution misses an operator (e.g. a
#: backend custom op not registered at configure time). Over-matching only
#: saves more activations (memory), never changes numerics.
DEFAULT_SELECTIVE_TOKENS: Tuple[str, ...] = (
    "_scaled_dot_product",  # aten SDPA family (fused or math variants)
    "npu_fusion_attention",  # torch_npu fused flash attention
    "flash_attn",  # flash-attn-2/3 namespaces
    "flash_attention",
    "sageattn",
    "memory_efficient_attention",  # xformers
)

#: Substrings that must never be saved even if they match a token (e.g. the
#: decomposed efficient-attention helpers materialize the attention matrix,
#: which is exactly what we do not want to persist).
_SELECTIVE_IGNORE_TOKENS: Tuple[str, ...] = ("_efficient_attention",)

#: Candidate operator namespaces probed for custom attention kernels.
_SELECTIVE_NS_CANDIDATES: Tuple[str, ...] = (
    "npu",
    "npu_extension",
    "flashattn",
    "flash_attn",
    "sageattention",
    "xformers",
)


def _op_qualname(op: Any) -> str:
    """Best-effort ``namespace::name`` for an OpOverload / HOP / raw op."""
    schema = getattr(op, "_schema", None)
    if schema is not None and hasattr(schema, "name"):
        return str(schema.name)
    return str(op).replace("torch.ops.", "").split(".", 1)[-1]


def _token_matches_qualname(qualname: str, tokens: Sequence[str]) -> bool:
    lowered = qualname.lower()
    return any(t in lowered for t in tokens) and not any(i in lowered for i in _SELECTIVE_IGNORE_TOKENS)


#: Process-level SAC state.
_state = {
    "enabled": False,
    "context_fn": None,  # functools.partial(create_selective_checkpoint_contexts, ...)
    "exact_ops": [],  # list[OpOverload]
    "prefix_mode": False,  # True -> substring policy instead of exact list
    "layers": None,  # Optional[set[int]]; None -> recompute every layer
    "sac_layers": None,  # Optional[set[int]]; None -> SAC on every recomputed layer
    "warned": False,
}


def reset() -> None:
    """Disable and drop all resolved state. Idempotent."""
    _state.update(
        {
            "enabled": False,
            "context_fn": None,
            "exact_ops": [],
            "prefix_mode": False,
            "layers": None,
            "sac_layers": None,
            "warned": False,
        }
    )


def _warn_once(message: str) -> None:
    if not _state["warned"]:
        _state["warned"] = True
        logger.warning("selective checkpointing: %s", message)


def resolve_exact_ops(extra_op_names: Optional[Sequence[str]] = None) -> Tuple[List[Any], List[str]]:
    """Resolve the default attention operators plus user extras to OpOverloads.

    Returns ``(resolved_ops, failed_names)``; never raises. ``failed_names``
    are extra names that could not be resolved (missing registration) — callers
    may fall back to substring matching.
    """
    ops: List[Any] = []
    failed: List[str] = []

    # aten SDPA family: every packet whose name starts with _scaled_dot_product.
    for name in sorted(dir(torch.ops.aten)):
        if not name.startswith("_scaled_dot_product"):
            continue
        packet = getattr(torch.ops.aten, name)
        default = getattr(packet, "default", None)
        if default is not None:
            ops.append(default)

    # Custom namespaces: probe candidate namespaces for token-matching ops.
    for ns_name in _SELECTIVE_NS_CANDIDATES:
        try:
            ns = getattr(torch.ops, ns_name)
        except AttributeError:
            continue
        for op_name in dir(ns):
            if not _token_matches_qualname(op_name, DEFAULT_SELECTIVE_TOKENS):
                continue
            packet = getattr(ns, op_name)
            default = getattr(packet, "default", None)
            if default is not None:
                ops.append(default)

    # User extras: full "torch.ops.<ns>.<name>.default" strings.
    if extra_op_names:
        for op_str in extra_op_names:
            try:
                op = string_to_op(op_str)
                assert isinstance(op, torch._ops.OpOverload), "op string must end with an overload name"
                ops.append(op)
            except (AssertionError, AttributeError, TypeError) as exc:
                logger.warning("selective checkpointing: cannot resolve extra op %r (%s)", op_str, exc)
                failed.append(op_str)

    # De-duplicate while preserving order (OpOverloads are hashable singletons).
    seen = set()
    unique = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            unique.append(op)
    return unique, failed


def _build_policy(exact_ops: Sequence[Any], prefix_mode: bool) -> Callable[[Any, Any, tuple, dict], Any]:
    """Policy fn for ``create_selective_checkpoint_contexts``.

    Returns MUST_SAVE for resolved attention ops (exact identity) or, in prefix
    mode, for any op whose qualname matches a default token; otherwise
    PREFER_RECOMPUTE. Both directions of mismatch are numerically safe (they
    only change how much is saved vs recomputed).
    """
    op_set = set(exact_ops)

    def policy(ctx, op, *args, **kwargs):
        del ctx, args, kwargs  # unused
        if prefix_mode:
            if _token_matches_qualname(_op_qualname(op), DEFAULT_SELECTIVE_TOKENS):
                return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
        elif op in op_set:
            return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
        return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

    return policy


def _parse_layer_spec(spec: Optional[Sequence[Union[int, str]]]) -> Optional[set]:
    """Normalize a layer-selection spec into a ``set[int]`` (None = all layers).

    Entry = int or inclusive ``"a-b"`` range string. Invalid entries are
    logged and skipped (never raised); empty/None spec returns None.
    """
    if not spec:
        return None
    layers: set = set()
    for item in spec:
        if isinstance(item, bool):
            logger.warning("selective checkpointing: skip invalid layer spec %r (bool)", item)
        elif isinstance(item, int):
            layers.add(item)
        elif isinstance(item, str):
            parts = item.split("-")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                start, end = int(parts[0]), int(parts[1])
                if start <= end:
                    layers.update(range(start, end + 1))  # inclusive both ends
                    continue
                logger.warning("selective checkpointing: skip invalid layer range %r (start > end)", item)
            else:
                logger.warning("selective checkpointing: skip invalid layer spec %r (want int or 'a-b')", item)
        else:
            logger.warning("selective checkpointing: skip invalid layer spec %r (type %s)", item, type(item).__name__)
    return layers


def configure(
    *,
    enabled: bool,
    extra_op_names: Optional[Sequence[str]] = None,
    prefix_mode: bool = False,
    gradient_checkpoint_layers: Optional[Sequence[Union[int, str]]] = None,
    selective_gradient_checkpoint_layers: Optional[Sequence[Union[int, str]]] = None,
) -> None:
    """Enable/disable selective checkpointing process-wide.

    Args:
        enabled: master switch; ``False`` resets all state (keeps
            ``gradient_checkpoint_layers`` filter — it gates full
            checkpointing too).
        extra_op_names: extra op strings to MUST_SAVE beyond the default
            attention set, e.g. ``["aten._scaled_dot_product_attention.default"]``.
        prefix_mode: match ops by name substring instead of exact OpOverload
            identity (auto-fallback when exact resolution finds nothing).
        gradient_checkpoint_layers: only these block indices are wrapped in
            checkpointing, the rest run directly. Empty/None = every layer.
            Entries are ints or inclusive ``"a-b"`` range strings.
        selective_gradient_checkpoint_layers: of the recomputed layers, only
            these run SAC; the rest fall back to full recomputation. Empty/None
            = SAC on every recomputed layer. Entries as above, e.g.
            ``[10, "11-19"]`` on a 20-layer model: layers 0-9 full recompute,
            10-19 SAC. The caller folds ``selective``/``enable`` into ``enabled``.
    """
    layers = _parse_layer_spec(gradient_checkpoint_layers)
    sac_layers = _parse_layer_spec(selective_gradient_checkpoint_layers)
    if not enabled:
        if gradient_checkpoint_layers is None:
            reset()
        else:
            reset()  # keep layer filter: it gates full checkpointing too
            if not layers:
                _warn_once("gradient_checkpoint_layers given but no valid layer parsed; recomputing every layer")
            _state["layers"] = layers or None
        return

    exact_ops, failed = resolve_exact_ops(extra_op_names)
    if not exact_ops and not prefix_mode:
        _warn_once(
            "no attention operators resolved (failed extras: %s); falling back to name-substring policy",
            failed or "none",
        )
        prefix_mode = True

    _state["exact_ops"] = exact_ops
    _state["prefix_mode"] = prefix_mode
    if prefix_mode:
        policy = _build_policy(exact_ops, prefix_mode=True)
        _state["context_fn"] = functools.partial(torch.utils.checkpoint.create_selective_checkpoint_contexts, policy)
    else:
        _state["context_fn"] = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, list(exact_ops)
        )
    if gradient_checkpoint_layers is not None and not layers:
        _warn_once("gradient_checkpoint_layers given but no valid layer parsed; recomputing every layer")
    if selective_gradient_checkpoint_layers is not None and not sac_layers:
        _warn_once(
            "selective_gradient_checkpoint_layers given but no valid layer parsed; SAC on every recomputed layer"
        )
    _state["layers"] = layers or None
    _state["sac_layers"] = sac_layers  # None = SAC on every recomputed layer
    _state["enabled"] = True
    logger.info(
        "selective checkpointing: enabled (%d exact ops, prefix_mode=%s, sac_layers=%s)",
        len(exact_ops),
        prefix_mode,
        None if sac_layers is None else sorted(sac_layers),
    )


def get_context_fn(layer_index: Optional[int] = None) -> Optional[Callable[[], Tuple[Any, Any]]]:
    """context_fn to pass to ``torch.utils.checkpoint.checkpoint``.

    Returns None unless ``configure(enabled=True)`` succeeded (callers then
    fall back to full checkpointing). When a positive ``selective_gradient_checkpoint_layers``
    subset was configured, layers outside it also get None (full
    recomputation); ``layer_index=None`` (old callers) never filters.
    """
    if not _state["enabled"]:
        return None
    sac_layers = _state["sac_layers"]
    if layer_index is not None and sac_layers is not None and layer_index not in sac_layers:
        return None  # full recomputation for layers outside the SAC subset
    return _state["context_fn"]
