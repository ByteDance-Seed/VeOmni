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

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Literal, Tuple

import torch
from torch import nn


try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from ..utils import logging
from ..utils.device import IS_CUDA_AVAILABLE


logger = logging.get_logger(__name__)


_NodeSpec = Tuple[Any, ...]
_CUDA_GRAPH_ATTR = "_veomni_cuda_graph_forward"

CudaGraphScope = Literal["auto", "layer", "attn"]

_NON_TEXT_CONFIG_FIELDS = (
    "vision_config",
    "visual_config",
    "image_config",
    "video_config",
    "audio_config",
    "talker_config",
    "thinker_config",
    "code2wav_config",
)
_NON_TEXT_TOKEN_FIELDS = (
    "image_token_id",
    "video_token_id",
    "audio_token_id",
    "vision_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "image_token_index",
    "video_token_index",
    "audio_token_index",
)
_NON_TEXT_NAME_MARKERS = (
    "vl",
    "vision",
    "visual",
    "image",
    "video",
    "audio",
    "omni",
    "wan",
    "janus",
)


class _UnsupportedCudaGraphInput(Exception):
    pass


@dataclass
class _GraphState:
    warmup_calls: int = 0
    sample_args: tuple[torch.Tensor, ...] | None = None
    graph_module: Any | None = None
    graphed_callable: Callable[..., Any] | None = None
    disabled: bool = False
    warned: bool = False


@dataclass(frozen=True)
class _StateTensor:
    name: str
    tensor: torch.Tensor


def _is_cuda_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda


def _is_supported_static(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device))


def _static_key(value: Any) -> Any:
    if not _is_supported_static(value):
        raise _UnsupportedCudaGraphInput(f"unsupported static argument type: {type(value).__name__}")
    return ("static", type(value).__name__, str(value))


def _flatten_obj(value: Any, leaves: list[torch.Tensor]) -> _NodeSpec:
    if isinstance(value, torch.Tensor):
        index = len(leaves)
        leaves.append(value)
        return ("tensor", index)
    if isinstance(value, tuple):
        return ("tuple", tuple(_flatten_obj(item, leaves) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_flatten_obj(item, leaves) for item in value))
    if isinstance(value, dict):
        return ("dict", tuple((key, _flatten_obj(value[key], leaves)) for key in sorted(value.keys())))
    if not _is_supported_static(value):
        raise _UnsupportedCudaGraphInput(f"unsupported static argument type: {type(value).__name__}")
    return ("static", value)


def _rebuild_obj(spec: _NodeSpec, leaves: Iterator[torch.Tensor]) -> Any:
    kind = spec[0]
    if kind == "tensor":
        return next(leaves)
    if kind == "tuple":
        return tuple(_rebuild_obj(item, leaves) for item in spec[1])
    if kind == "list":
        return [_rebuild_obj(item, leaves) for item in spec[1]]
    if kind == "dict":
        return {key: _rebuild_obj(item, leaves) for key, item in spec[1]}
    if kind == "static":
        return spec[1]
    raise RuntimeError(f"unknown CUDA graph spec kind: {kind}")


def _tensor_key(tensor: torch.Tensor) -> Any:
    return (
        "tensor",
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        tensor.device.type,
        tensor.device.index,
        tensor.requires_grad,
    )


def _spec_key(spec: _NodeSpec, leaves: tuple[torch.Tensor, ...]) -> Any:
    kind = spec[0]
    if kind == "tensor":
        return _tensor_key(leaves[spec[1]])
    if kind in ("tuple", "list"):
        return (kind, tuple(_spec_key(item, leaves) for item in spec[1]))
    if kind == "dict":
        return (kind, tuple((key, _spec_key(item, leaves)) for key, item in spec[1]))
    if kind == "static":
        return _static_key(spec[1])
    raise RuntimeError(f"unknown CUDA graph spec kind: {kind}")


def _clone_sample_arg(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.layout is not torch.strided:
        raise _UnsupportedCudaGraphInput(f"unsupported tensor layout: {tensor.layout}")
    sample = torch.empty_strided(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    with torch.no_grad():
        sample.copy_(tensor)
    if tensor.requires_grad:
        sample.requires_grad_(True)
    return sample


def _string_contains_non_text_marker(value: Any) -> bool:
    value = str(value).lower()
    return any(marker in value for marker in _NON_TEXT_NAME_MARKERS)


def _get_model_config(model: nn.Module) -> Any:
    candidates = [model, getattr(model, "module", None), getattr(model, "_orig_module", None)]
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None:
            return config
    return None


def _cuda_graph_non_text_markers(model: nn.Module) -> list[str]:
    markers: list[str] = []
    config = _get_model_config(model)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type is not None and _string_contains_non_text_marker(model_type):
            markers.append(f"model_type={model_type!r}")

        class_name = getattr(config, "_class_name", None)
        if class_name is not None and _string_contains_non_text_marker(class_name):
            markers.append(f"_class_name={class_name!r}")

        architectures = getattr(config, "architectures", None) or ()
        for architecture in architectures:
            if _string_contains_non_text_marker(architecture):
                markers.append(f"architecture={architecture!r}")
                break

        for name in _NON_TEXT_CONFIG_FIELDS:
            if getattr(config, name, None) is not None:
                markers.append(name)
        for name in _NON_TEXT_TOKEN_FIELDS:
            if getattr(config, name, None) is not None:
                markers.append(name)

    root = getattr(model, "module", model)
    for name in ("visual", "vision_tower", "audio_tower", "talker", "thinker"):
        if getattr(root, name, None) is not None:
            markers.append(name)
    return markers


def _validate_cuda_graph_text_only_model(model: nn.Module) -> None:
    markers = _cuda_graph_non_text_markers(model)
    if markers:
        unique_markers = ", ".join(dict.fromkeys(markers))
        raise ValueError(
            "train.cuda_graph.enable=True is currently supported only for text-only models. "
            f"Found non-text model marker(s): {unique_markers}."
        )


def _iter_text_decoder_layers(model: nn.Module) -> Iterable[nn.Module]:
    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
    ]
    roots = [model]
    wrapped_module = getattr(model, "module", None)
    if wrapped_module is not None:
        roots.append(wrapped_module)

    seen: set[int] = set()
    for root in roots:
        for path in candidates:
            obj: Any = root
            for name in path:
                obj = getattr(obj, name, None)
                if obj is None:
                    break
            if obj is None:
                continue
            for layer in obj:
                layer_id = id(layer)
                if layer_id in seen:
                    continue
                seen.add(layer_id)
                yield layer


def _has_moe_experts(module: nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    if "moe" in class_name:
        return True
    if hasattr(module, "experts") and hasattr(module, "gate"):
        return True
    return False


def _is_moe_decoder_layer(layer: nn.Module) -> bool:
    if "moe" in layer.__class__.__name__.lower():
        return True
    mlp = getattr(layer, "mlp", None)
    return isinstance(mlp, nn.Module) and _has_moe_experts(mlp)


def _moe_expert_input_is_padded(layer: nn.Module) -> bool:
    """Return whether the MoE expert dispatch has static capacity padding.

    Whole-layer CUDA graph capture for MoE requires expert dispatch to have a
    static per-expert capacity. VeOmni does not currently expose those
    attributes, so this returns false for today's MoE layers while leaving a
    narrow compatibility hook for future configs.
    """
    candidates: list[Any] = [layer, getattr(layer, "mlp", None), getattr(layer, "config", None)]
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        candidates.append(getattr(mlp, "config", None))

    pad_names = (
        "moe_pad_expert_input_to_capacity",
        "pad_expert_input_to_capacity",
        "moe_pad_expert_input",
    )
    capacity_names = (
        "moe_expert_capacity_factor",
        "expert_capacity_factor",
        "moe_expert_capacity",
        "expert_capacity",
    )
    for candidate in candidates:
        if candidate is None:
            continue
        has_padding = any(bool(getattr(candidate, name, False)) for name in pad_names)
        has_capacity = any(getattr(candidate, name, None) is not None for name in capacity_names)
        if has_padding and has_capacity:
            return True
    return False


def _iter_attention_modules(layer: nn.Module) -> Iterable[nn.Module]:
    candidates = ("self_attn", "linear_attn", "attention", "self_attention")
    seen: set[int] = set()
    for name in candidates:
        module = getattr(layer, name, None)
        if not isinstance(module, nn.Module):
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        yield module


class _GraphForwardModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        original_forward: Callable[..., Any],
        args_spec: _NodeSpec,
        kwargs_spec: _NodeSpec,
        state_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.call_module = _OriginalForwardModule(module, original_forward)
        self.args_spec = args_spec
        self.kwargs_spec = kwargs_spec
        self.state_names = state_names
        self.output_spec: _NodeSpec | None = None

    def _flatten_output(self, output: Any) -> tuple[torch.Tensor, ...]:
        leaves: list[torch.Tensor] = []
        spec = _flatten_obj(output, leaves)
        if self.output_spec is None:
            self.output_spec = spec
        elif self.output_spec != spec:
            raise _UnsupportedCudaGraphInput("CUDA graph output structure changed between calls")
        return tuple(leaves)

    def rebuild_output(self, flat_outputs: Any) -> Any:
        if self.output_spec is None:
            raise RuntimeError("CUDA graph output spec is not initialized")
        if isinstance(flat_outputs, torch.Tensor):
            leaves = (flat_outputs,)
        else:
            leaves = tuple(flat_outputs)
        return _rebuild_obj(self.output_spec, iter(leaves))

    def forward(self, *flat_tensor_args):
        if self.state_names:
            user_tensor_args = flat_tensor_args[: -len(self.state_names)]
            state_tensors = flat_tensor_args[-len(self.state_names) :]
        else:
            user_tensor_args = flat_tensor_args
            state_tensors = ()

        tensor_iter = iter(user_tensor_args)
        rebuilt_args = _rebuild_obj(self.args_spec, tensor_iter)
        rebuilt_kwargs = _rebuild_obj(self.kwargs_spec, tensor_iter)
        if not self.state_names:
            return self._flatten_output(self.call_module(*rebuilt_args, **rebuilt_kwargs))

        state = dict(zip(self.state_names, state_tensors))
        return self._flatten_output(functional_call(self.call_module, state, rebuilt_args, rebuilt_kwargs))


class _GraphForwardCallable:
    def __init__(
        self,
        module: nn.Module,
        original_forward: Callable[..., Any],
        args_spec: _NodeSpec,
        kwargs_spec: _NodeSpec,
        state_names: tuple[str, ...],
    ) -> None:
        self.call_module = _OriginalForwardModule(module, original_forward)
        self.args_spec = args_spec
        self.kwargs_spec = kwargs_spec
        self.state_names = state_names
        self.output_spec: _NodeSpec | None = None

    def _flatten_output(self, output: Any) -> tuple[torch.Tensor, ...]:
        leaves: list[torch.Tensor] = []
        spec = _flatten_obj(output, leaves)
        if self.output_spec is None:
            self.output_spec = spec
        elif self.output_spec != spec:
            raise _UnsupportedCudaGraphInput("CUDA graph output structure changed between calls")
        return tuple(leaves)

    def rebuild_output(self, flat_outputs: Any) -> Any:
        if self.output_spec is None:
            raise RuntimeError("CUDA graph output spec is not initialized")
        if isinstance(flat_outputs, torch.Tensor):
            leaves = (flat_outputs,)
        else:
            leaves = tuple(flat_outputs)
        return _rebuild_obj(self.output_spec, iter(leaves))

    def __call__(self, *flat_tensor_args):
        user_tensor_args = flat_tensor_args[: -len(self.state_names)]
        state_tensors = flat_tensor_args[-len(self.state_names) :]
        tensor_iter = iter(user_tensor_args)
        rebuilt_args = _rebuild_obj(self.args_spec, tensor_iter)
        rebuilt_kwargs = _rebuild_obj(self.kwargs_spec, tensor_iter)
        state = dict(zip(self.state_names, state_tensors))
        return self._flatten_output(functional_call(self.call_module, state, rebuilt_args, rebuilt_kwargs))


class _OriginalForwardModule(nn.Module):
    def __init__(self, module: nn.Module, original_forward: Callable[..., Any]) -> None:
        super().__init__()
        self.module = module
        self.original_forward = original_forward

    def forward(self, *args, **kwargs):
        return self.original_forward(*args, **kwargs)


def _named_parameters(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    try:
        yield from module.named_parameters(recurse=True, remove_duplicate=False)
    except TypeError:
        yield from module.named_parameters(recurse=True)


def _named_buffers(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    try:
        yield from module.named_buffers(recurse=True, remove_duplicate=False)
    except TypeError:
        yield from module.named_buffers(recurse=True)


def _iter_module_state_tensors(module: nn.Module) -> tuple[_StateTensor, ...]:
    state_tensors: list[_StateTensor] = []
    for name, tensor in _named_parameters(module):
        state_tensors.append(_StateTensor(f"module.{name}", tensor))
    for name, tensor in _named_buffers(module):
        state_tensors.append(_StateTensor(f"module.{name}", tensor))
    return tuple(state_tensors)


class _LazyCudaGraphForward:
    def __init__(
        self,
        module: nn.Module,
        original_forward: Callable[..., Any],
        *,
        num_warmup_steps: int,
        max_graphs: int,
        strict: bool,
        capture_module_state: bool,
    ):
        self.module = module
        self.original_forward = original_forward
        self.num_warmup_steps = num_warmup_steps
        self.max_graphs = max_graphs
        self.strict = strict
        self.capture_module_state = capture_module_state
        self.graphs: dict[Any, _GraphState] = {}

    def _warn_once(self, state: _GraphState, message: str) -> None:
        if state.warned:
            return
        state.warned = True
        logger.warning_rank0(message)

    def clear(self) -> None:
        self.graphs.clear()

    def _fallback(self, *args, **kwargs):
        return self.original_forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if not IS_CUDA_AVAILABLE or not torch.is_grad_enabled() or not self.module.training:
            return self._fallback(*args, **kwargs)

        leaves: list[torch.Tensor] = []
        try:
            args_spec = _flatten_obj(args, leaves)
            kwargs_spec = _flatten_obj(kwargs, leaves)
            tensor_args = tuple(leaves)
            if not tensor_args or any(not _is_cuda_tensor(tensor) for tensor in tensor_args):
                return self._fallback(*args, **kwargs)
            state_tensors = _iter_module_state_tensors(self.module) if self.capture_module_state else ()
            if any(not _is_cuda_tensor(item.tensor) for item in state_tensors):
                return self._fallback(*args, **kwargs)
            state_key = tuple((item.name, _tensor_key(item.tensor)) for item in state_tensors)
            key = (_spec_key(args_spec, tensor_args), _spec_key(kwargs_spec, tensor_args), state_key)
        except _UnsupportedCudaGraphInput as exc:
            if self.strict:
                raise
            state = self.graphs.setdefault(("unsupported", str(exc)), _GraphState(disabled=True))
            self._warn_once(
                state,
                f"Disable CUDA graph for {self.module.__class__.__name__}: {exc}.",
            )
            return self._fallback(*args, **kwargs)

        state = self.graphs.get(key)
        if state is None:
            if len(self.graphs) >= self.max_graphs:
                return self._fallback(*args, **kwargs)
            state = _GraphState()
            self.graphs[key] = state

        if state.disabled:
            return self._fallback(*args, **kwargs)

        if state.graphed_callable is not None:
            flat_outputs = state.graphed_callable(*tensor_args, *(item.tensor for item in state_tensors))
            return state.graph_module.rebuild_output(flat_outputs)

        if state.warmup_calls < self.num_warmup_steps:
            state.warmup_calls += 1
            return self._fallback(*args, **kwargs)

        graph_tensor_args = tensor_args + tuple(item.tensor for item in state_tensors)
        state_names = tuple(item.name for item in state_tensors)
        graph_forward_cls = _GraphForwardCallable if state_names else _GraphForwardModule
        state.graph_module = graph_forward_cls(
            self.module,
            self.original_forward,
            args_spec,
            kwargs_spec,
            state_names,
        )

        try:
            state.sample_args = tuple(_clone_sample_arg(tensor) for tensor in graph_tensor_args)
            state.graphed_callable = torch.cuda.make_graphed_callables(
                state.graph_module,
                state.sample_args,
                num_warmup_iters=self.num_warmup_steps,
                allow_unused_input=True,
            )
        except Exception as exc:
            if self.strict:
                raise
            state.disabled = True
            self._warn_once(
                state,
                f"Disable CUDA graph for {self.module.__class__.__name__}: "
                f"capture failed with {type(exc).__name__}: {exc}.",
            )
            return self._fallback(*args, **kwargs)

        flat_outputs = state.graphed_callable(*graph_tensor_args)
        return state.graph_module.rebuild_output(flat_outputs)


def _wrap_cuda_graph_forward(
    module: nn.Module,
    *,
    num_warmup_steps: int,
    max_graphs: int,
    strict: bool = False,
    capture_module_state: bool = False,
) -> bool:
    if hasattr(module, _CUDA_GRAPH_ATTR):
        return False
    original_forward = module.forward
    graph_forward = _LazyCudaGraphForward(
        module,
        original_forward,
        num_warmup_steps=num_warmup_steps,
        max_graphs=max_graphs,
        strict=strict,
        capture_module_state=capture_module_state,
    )

    @functools.wraps(original_forward)
    def wrapped_forward(*args, __graph_forward=graph_forward, **kwargs):
        return __graph_forward(*args, **kwargs)

    module.forward = wrapped_forward
    setattr(module, _CUDA_GRAPH_ATTR, graph_forward)
    return True


def _wrap_attention_modules(
    layer: nn.Module,
    *,
    num_warmup_steps: int,
    max_graphs_per_layer: int,
    strict: bool,
    capture_module_state: bool,
) -> int:
    wrapped = 0
    for module in _iter_attention_modules(layer):
        if _wrap_cuda_graph_forward(
            module,
            num_warmup_steps=num_warmup_steps,
            max_graphs=max_graphs_per_layer,
            strict=strict,
            capture_module_state=capture_module_state,
        ):
            wrapped += 1
    return wrapped


def apply_layer_cuda_graph(
    model: nn.Module,
    *,
    num_warmup_steps: int,
    max_graphs_per_layer: int,
    strict: bool = False,
    scope: CudaGraphScope = "auto",
    capture_module_state: bool = False,
) -> int:
    if scope not in ("auto", "layer", "attn"):
        raise ValueError(f"Unsupported CUDA graph scope: {scope}")
    _validate_cuda_graph_text_only_model(model)
    if not IS_CUDA_AVAILABLE:
        logger.warning_rank0("train.cuda_graph.enable=True is ignored because CUDA is unavailable.")
        return 0

    wrapped = 0
    skipped_moe_layers = 0
    attention_wrapped_layers = 0
    for layer in _iter_text_decoder_layers(model):
        is_unpadded_moe_layer = _is_moe_decoder_layer(layer) and not _moe_expert_input_is_padded(layer)

        if scope == "layer":
            if is_unpadded_moe_layer:
                raise ValueError(
                    "train.cuda_graph.scope='layer' cannot capture an MoE decoder layer unless expert "
                    "capacity padding is enabled. Use train.cuda_graph.scope='auto' or 'attn' to graph only "
                    "the static attention submodule, or add MoE expert-input padding first."
                )
            if _wrap_cuda_graph_forward(
                layer,
                num_warmup_steps=num_warmup_steps,
                max_graphs=max_graphs_per_layer,
                strict=strict,
                capture_module_state=capture_module_state,
            ):
                wrapped += 1
            continue

        attention_only = scope == "attn" or is_unpadded_moe_layer or (scope == "auto" and capture_module_state)
        if attention_only:
            count = _wrap_attention_modules(
                layer,
                num_warmup_steps=num_warmup_steps,
                max_graphs_per_layer=max_graphs_per_layer,
                strict=strict,
                capture_module_state=capture_module_state,
            )
            if is_unpadded_moe_layer:
                skipped_moe_layers += 1
            if count:
                attention_wrapped_layers += 1
                wrapped += count
            continue

        if _wrap_cuda_graph_forward(
            layer,
            num_warmup_steps=num_warmup_steps,
            max_graphs=max_graphs_per_layer,
            strict=strict,
            capture_module_state=capture_module_state,
        ):
            wrapped += 1

    if wrapped:
        message = f"Enabled CUDA graph capture for {wrapped} text decoder module(s)"
        if attention_wrapped_layers:
            message += f" ({attention_wrapped_layers} attention-only layer(s))"
        if skipped_moe_layers:
            message += (
                f"; skipped whole-layer capture for {skipped_moe_layers} unpadded MoE layer(s) because "
                "expert routing has dynamic shapes."
            )
        logger.info_rank0(message)
    else:
        logger.warning_rank0("train.cuda_graph.enable=True found no text decoder modules to wrap.")
    return wrapped


def clear_cuda_graphs(model: nn.Module) -> int:
    cleared = 0
    for module in model.modules():
        graph_forward = getattr(module, _CUDA_GRAPH_ATTR, None)
        if graph_forward is None or not isinstance(graph_forward, _LazyCudaGraphForward):
            continue
        graph_forward.clear()
        module.forward = graph_forward.original_forward
        delattr(module, _CUDA_GRAPH_ATTR)
        cleared += 1

    if cleared and IS_CUDA_AVAILABLE:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return cleared
