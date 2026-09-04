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

"""Every call inside a generated module must fit the signature it calls.

``patchgen --check`` regenerates and text-diffs, and the ruff pass it runs
reports undefined names, not argument mismatches -- so a patched caller can pass
a keyword its patched callee does not accept and every gate stays green. On
DeepSeek-V4 that happened three times in a row, always the same way: the NPU
config hand-forks a method the GPU config also patches, the GPU side gains a
parameter, and the fork keeps the old signature. Nothing catches it until the
model runs, and only on the hardware that selects the forked module.

The check reads source rather than importing, so a module for an accelerator
this host does not have is covered exactly like the local one.
"""

import ast
import inspect
from functools import lru_cache
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMERS_ROOT = REPO_ROOT / "veomni" / "models_kernel" / "transformers"
GENERATED_MODULES = sorted(TRANSFORMERS_ROOT.glob("*/generated/patched_modeling_*.py"))

_UNKNOWN = object()


def _signature_from_ast(node: ast.FunctionDef, drop_self: bool) -> inspect.Signature:
    a = node.args
    params: list[inspect.Parameter] = []

    def add(arg: ast.arg, kind: inspect._ParameterKind, has_default: bool) -> None:
        params.append(inspect.Parameter(arg.arg, kind, default=None if has_default else inspect.Parameter.empty))

    positional = a.posonlyargs + a.args
    n_defaults = len(a.defaults)
    first_default = len(positional) - n_defaults
    for i, arg in enumerate(a.posonlyargs):
        add(arg, inspect.Parameter.POSITIONAL_ONLY, i >= first_default)
    for i, arg in enumerate(a.args, start=len(a.posonlyargs)):
        add(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, i >= first_default)
    if a.vararg is not None:
        params.append(inspect.Parameter(a.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        add(arg, inspect.Parameter.KEYWORD_ONLY, default is not None)
    if a.kwarg is not None:
        params.append(inspect.Parameter(a.kwarg.arg, inspect.Parameter.VAR_KEYWORD))

    if drop_self and params and params[0].name == "self":
        params = params[1:]
    return inspect.Signature(params)


@lru_cache(maxsize=None)
def _parse_module(dotted: str) -> ast.Module | None:
    for candidate in (
        REPO_ROOT / (dotted.replace(".", "/") + ".py"),
        REPO_ROOT / dotted.replace(".", "/") / "__init__.py",
    ):
        if candidate.exists():
            return ast.parse(candidate.read_text())
    return None


def _resolve_function(dotted: str, name: str, depth: int = 0) -> inspect.Signature | None:
    """Find a first-party function by reading source, following re-exports.

    Reading rather than importing is what lets this run anywhere: several
    veomni modules import `torch_npu` at module scope, so an import-based
    resolver would be unable to check the very NPU call sites that keep
    breaking.
    """
    if depth > 4:
        return None
    tree = _parse_module(dotted)
    if tree is None:
        return None
    is_package = (REPO_ROOT / dotted.replace(".", "/") / "__init__.py").exists()
    package = dotted if is_package else dotted.rsplit(".", 1)[0]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return _signature_from_ast(node, drop_self=False)
        if isinstance(node, ast.ClassDef) and node.name == name:
            return None
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) != name:
                    continue
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                if node.level:
                    target = f"{base}.{node.module}" if node.module else base
                else:
                    target = node.module or ""
                if not target.startswith("veomni"):
                    return None
                return _resolve_function(target, alias.name, depth + 1)
    return None


def _imported_veomni_signatures(tree: ast.Module) -> dict[str, inspect.Signature]:
    """Signatures for names the module imports from veomni.

    Restricted to first-party modules: third-party signatures are not ours to
    keep in sync, and tracking them here would turn this into a test of their
    releases rather than of our patches.
    """
    found: dict[str, inspect.Signature] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("veomni"):
            continue
        for alias in node.names:
            signature = _resolve_function(node.module, alias.name)
            if signature is not None:
                found[alias.asname or alias.name] = signature
    return found


def _attribute_owner_classes(tree: ast.Module, classes: set[str]) -> dict[str, set[str]]:
    """Map ``self.<attr>`` to the classes it is constructed from.

    Covers a direct ``self.x = SomeClass(...)`` and the dispatch-table form
    ``self.x = TABLE[key](...)``, which is how DeepSeek-V4 picks a compressor
    per layer type. Anything else stays unmapped and is simply not checked.
    """
    tables: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            names = {v.id for v in node.value.values if isinstance(v, ast.Name) and v.id in classes}
            if names:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        tables[target.id] = names

    owners: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
            ):
                continue
            for call in ast.walk(node.value):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if isinstance(func, ast.Name) and func.id in classes:
                    owners.setdefault(target.attr, set()).add(func.id)
                elif isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name) and func.value.id in tables:
                    owners.setdefault(target.attr, set()).update(tables[func.value.id])
    return owners


def _rejection(signature: inspect.Signature, n_positional: int, keywords: list[str]) -> str | None:
    """Why the signature would reject this argument list, or None if it accepts."""
    try:
        signature.bind(*[_UNKNOWN] * n_positional, **dict.fromkeys(keywords, _UNKNOWN))
    except TypeError as exc:
        return str(exc)
    return None


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())

    functions = {n.name: _signature_from_ast(n, drop_self=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    functions.update(_imported_veomni_signatures(tree))

    methods: dict[str, dict[str, inspect.Signature]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods[node.name] = {
                m.name: _signature_from_ast(m, drop_self=True) for m in node.body if isinstance(m, ast.FunctionDef)
            }
    owners = _attribute_owner_classes(tree, set(methods))

    problems = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        # A splat hides the real argument list, so there is nothing to check.
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
            continue
        keywords = [k.arg for k in call.keywords]

        func = call.func
        if isinstance(func, ast.Name) and func.id in functions:
            reason = _rejection(functions[func.id], len(call.args), keywords)
            if reason:
                problems.append(f"{path.name}:{call.lineno} calls {func.id}(...): {reason}")
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and func.attr in owners
        ):
            # One attribute can hold any of several classes, and sibling classes
            # may legitimately disagree: DeepSeek-V4's hash and top-k routers
            # take different arguments, and the caller branches on which is
            # installed. So the bar is that the call fits *some* class it could
            # reach, not every one.
            candidates = [
                (cls, methods[cls]["forward"]) for cls in sorted(owners[func.attr]) if "forward" in methods[cls]
            ]
            reasons = {cls: _rejection(signature, len(call.args), keywords) for cls, signature in candidates}
            if candidates and all(reasons.values()):
                detail = "; ".join(f"{cls}.forward: {reason}" for cls, reason in reasons.items())
                problems.append(f"{path.name}:{call.lineno} calls self.{func.attr}(...) — {detail}")
    return problems


def test_generated_modules_exist():
    """A silent zero-module glob would make every check below vacuously pass."""
    assert len(GENERATED_MODULES) > 20, GENERATED_MODULES


@pytest.mark.parametrize("path", GENERATED_MODULES, ids=lambda p: p.stem)
def test_generated_call_sites_match_their_callees(path: Path):
    problems = _violations(path)
    assert not problems, "\n".join(problems)
