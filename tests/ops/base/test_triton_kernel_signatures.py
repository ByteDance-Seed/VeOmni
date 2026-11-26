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

"""Static check for kernel calls only a Triton build would reject.

A ``@triton.jit`` body is not executed as Python: it is traced. A call to a method
``tl.tensor`` does not have is therefore valid Python and fails only when Triton
compiles the kernel, on hardware. Every numerical test of such a kernel skips
without a matching device, so a CPU-only host and CI's device-less half both pass
while the GPU job fails:

    AttributeError: 'tensor' object has no attribute 'minimum'

which is exactly what happened to the FP4 E8M0 clamp -- 25 minutes into
``gpu_unit_tests_v5``, after a fully green local run.

Triton exposes most of ``triton.language`` as free functions and binds only *some*
of them onto ``tl.tensor`` as methods (``core._tensor_member_fn`` /
``core.core._binary_op_type_checking_impl``). This walks the AST instead of running
anything, so it needs no device.

**Scope:** only the kernels under ``veomni/ops/kernels/`` that this package owns.
``vendor/`` is third-party code copied in under its own review; scanning it would
report pre-existing hits (e.g. ``g_diff.clamp(...)`` in the FLA gated-delta-rule
port, where ``tl.clamp`` is a free function with no ``tensor`` binding) that this
change did not introduce and must not silently "fix".
"""

from __future__ import annotations

import ast
import pathlib

import pytest


triton_language = pytest.importorskip("triton.language", reason="Triton is a GPU-extra dependency")


def _free_functions_only() -> frozenset[str]:
    """``triton.language`` names that are *not* also bound onto ``tl.tensor``.

    Derived from the installed Triton rather than hard-coded, so a version that
    starts binding one of these onto ``tensor`` (as some already are) does not
    produce a false positive.
    """
    return frozenset(
        name
        for name in dir(triton_language)
        if not name.startswith("_")
        and callable(getattr(triton_language, name))
        and not hasattr(triton_language.tensor, name)
    )


#: Receiver names for other modules whose functions share a name with a
#: ``triton.language`` one (``torch.tensor``, ``libdevice.div_rn``, ``desc.load``).
_NOT_A_TRITON_VALUE = frozenset(
    {"tl", "triton", "language", "torch", "libdevice", "np", "numpy", "math", "F", "nn", "desc", "self"}
)


def _owned_kernel_files(root: pathlib.Path) -> list[pathlib.Path]:
    kernels = root / "veomni" / "ops" / "kernels"
    return sorted(p for p in kernels.rglob("*.py") if "vendor" not in p.parts and "@triton.jit" in p.read_text())


def _is_jit_decorator(node: ast.expr) -> bool:
    """Whether a decorator is ``@triton.jit`` / ``@jit`` / ``@triton.jit(...)``."""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr == "jit"
    return isinstance(target, ast.Name) and target.id == "jit"


def _jit_functions(tree: ast.Module):
    """Yield the body statements of every ``@triton.jit`` function in ``tree``.

    Only these bodies are traced by Triton and therefore only here that a call to a
    non-existent ``tl.tensor`` method is an error. Scanning the whole module would
    also read ordinary host code, where several ``triton.language`` free functions
    share a name with a ``torch.Tensor`` method (``clamp``, ``sqrt``, ``sigmoid``,
    ``floor``, ``sort``, ``where``), so a normal ``t.clamp(...)`` in a helper beside
    a kernel would be reported as a kernel bug.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(_is_jit_decorator(d) for d in node.decorator_list):
                yield from ast.walk(node)


def _offenders(paths: list[pathlib.Path], free: frozenset[str]) -> list[str]:
    out = []
    for path in paths:
        for node in _jit_functions(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            if name not in free:
                continue
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id in _NOT_A_TRITON_VALUE:
                continue
            out.append(f"{path}:{node.lineno} calls {name}() as a method; use tl.{name}(...)")
    return out


def test_no_triton_free_function_is_called_as_a_tensor_method() -> None:
    """Catches the class of error that no device-less test run can see."""
    root = pathlib.Path(__file__).resolve().parents[3]
    paths = _owned_kernel_files(root)
    assert paths, "the scan should find the kernels it is meant to inspect"

    found = _offenders(paths, _free_functions_only())
    assert not found, "Triton free functions called as tensor methods:\n" + "\n".join(found)
