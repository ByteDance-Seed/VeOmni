# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unified ``patchgen`` console-script entry.

Reads ``[tool.patchgen]`` from the nearest ``pyproject.toml``, builds a
:class:`DiscoveryConfig`, and dispatches to the codegen runner or the
drift checker depending on whether ``--check`` is present in ``argv``.

``--check`` is the only flag this layer owns; everything else
(``--list / --all / <module> [--diff / -v / -o / -c / --dry-run / --fix]``)
flows straight through to the underlying CLI built by
:func:`patchgen.run_codegen.build_cli` / :func:`patchgen.check_patchgen.build_cli`.

Example downstream ``pyproject.toml``::

    [tool.patchgen]
    search_root = "raskel/trainer/multilora/models"
    package_prefix = "raskel.trainer.multilora.models"
    ruff_isolated = true
    ruff_extra_ignore = ["E501"]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import tomllib  # py>=3.11
except ImportError:  # pragma: no cover - py<3.11
    import tomli as tomllib  # type: ignore[no-redef]

from . import DiscoveryConfig, build_check_cli, build_run_codegen_cli


_PROG = "patchgen"


def _find_pyproject_with_section(start: Path) -> Optional[tuple[Path, dict]]:
    """Walk up from ``start`` looking for a ``pyproject.toml`` containing a
    ``[tool.patchgen]`` table. Returns ``(pyproject_path, section_dict)`` or
    ``None``.

    Walking up rather than checking only CWD lets ``patchgen --check`` work
    from any subdirectory of the consuming repo (pre-commit hooks run from
    the repo root anyway, but Makefile recipes and ad-hoc invocations don't
    always).
    """
    for parent in (start, *start.parents):
        candidate = parent / "pyproject.toml"
        if not candidate.is_file():
            continue
        with candidate.open("rb") as fp:
            data = tomllib.load(fp)
        section = data.get("tool", {}).get("patchgen")
        if section is not None:
            return candidate, section
    return None


def _discovery_from_section(section: dict, repo_root: Path) -> DiscoveryConfig:
    try:
        search_root_str = section["search_root"]
        package_prefix = section["package_prefix"]
    except KeyError as e:
        raise SystemExit(
            f"patchgen: [tool.patchgen] is missing required key {e.args[0]!r}. "
            "Both `search_root` and `package_prefix` are required."
        ) from e
    return DiscoveryConfig(
        search_root=(repo_root / search_root_str).resolve(),
        package_prefix=package_prefix,
        legacy_patches_prefix=section.get("legacy_patches_prefix"),
        ruff_extra_ignore=tuple(section.get("ruff_extra_ignore", ())),
        ruff_isolated=bool(section.get("ruff_isolated", False)),
    )


def _load_discovery() -> DiscoveryConfig:
    found = _find_pyproject_with_section(Path.cwd().resolve())
    if found is None:
        raise SystemExit(
            "patchgen: no [tool.patchgen] section found in any pyproject.toml "
            "from the current directory up. Add a section like:\n\n"
            "  [tool.patchgen]\n"
            '  search_root = "<dir containing *_patch_gen_config.py>"\n'
            '  package_prefix = "<dotted import prefix>"\n'
        )
    pyproject_path, section = found
    return _discovery_from_section(section, pyproject_path.parent)


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    discovery = _load_discovery()
    if "--check" in argv:
        argv = [a for a in argv if a != "--check"]
        return build_check_cli(discovery, prog_name=_PROG)(argv)
    return build_run_codegen_cli(discovery, prog_name=_PROG)(argv)


if __name__ == "__main__":
    sys.exit(main())
