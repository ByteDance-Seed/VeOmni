# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Private replicated-dummy sequence-parallel scope.

Only ``Qwen3_5VisionModel.dummy_forward`` may activate this sentinel. Flash
treats the active scope as a replicated/local path. Flex fail-closes. Ordinary
callers cannot turn the bypass on through public arguments or kwargs.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch.nn as nn


_PUBLIC_BYPASS_ERROR = (
    "skip_sequence_parallel is not a public argument; the replicated dummy path is a private scoped context."
)
_FORGED_SCOPE_ERROR = "forged activation of the replicated dummy sequence-parallel scope"
_FLEX_SCOPE_ERROR = (
    "FlexAttention does not support the replicated dummy sequence-parallel bypass; "
    "the private dummy scope is Flash-only because a global BlockMask is layout-unsafe."
)

_ACTIVE: ContextVar[bool] = ContextVar("veomni_replicated_dummy_sp", default=False)


class _ReplicatedDummySPToken:
    __slots__ = ()


_DUMMY_SP_TOKEN = _ReplicatedDummySPToken()


def is_replicated_dummy_sequence_parallel() -> bool:
    return bool(_ACTIVE.get())


def reject_public_sequence_parallel_bypass(kwargs: dict) -> None:
    if "skip_sequence_parallel" in kwargs:
        raise TypeError(_PUBLIC_BYPASS_ERROR)


@contextmanager
def _replicated_dummy_sequence_parallel(token: object) -> Iterator[None]:
    """Private scoped sentinel. Auto-restores on exit, including exceptions."""
    if token is not _DUMMY_SP_TOKEN:
        raise RuntimeError(_FORGED_SCOPE_ERROR)
    reset_token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(reset_token)


def _call_replicated_dummy_checkpointed_module(
    token: object,
    module: nn.Module,
    *args,
    **kwargs,
):
    """Call one dummy-vision block while preserving scope across AC replay.

    Hugging Face's ``GradientCheckpointingLayer`` captures the callable passed
    to ``_gradient_checkpointing_func`` and invokes it later during backward.
    By then the outer ``dummy_forward`` scope has exited. Build the captured
    callable here so every invocation (initial forward and replay) re-enters
    the private scope, without mutating the module or its checkpoint function.

    The callable exposes ``__self__`` because VeOmni's reentrant checkpoint
    shim uses it to find the block's FSDP state.
    """

    if token is not _DUMMY_SP_TOKEN or not is_replicated_dummy_sequence_parallel():
        raise RuntimeError(_FORGED_SCOPE_ERROR)

    if bool(getattr(module, "gradient_checkpointing", False)) and module.training:
        checkpoint_fn = getattr(module, "_gradient_checkpointing_func", None)
        if checkpoint_fn is None:
            raise RuntimeError("dummy vision block has gradient checkpointing enabled without a checkpoint function")

        def scoped_module_call(*module_args):
            with _replicated_dummy_sequence_parallel(token):
                return nn.Module.__call__(module, *module_args, **kwargs)

        scoped_module_call.__self__ = module
        return checkpoint_fn(scoped_module_call, *args)

    return module(*args, **kwargs)
