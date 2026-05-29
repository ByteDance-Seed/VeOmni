# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Shared post-generation normalization step.

``run_codegen`` (writing a fresh artifact) and ``check_patchgen`` (validating
the committed artifact) must apply the **same** ruff normalization so the two
paths agree byte-for-byte. Without this, a contributor running ``run_codegen``
could commit a file that immediately fails ``check_patchgen`` whenever ruff
would rewrite a single line of the raw codegen output.

The ``--ignore`` set mirrors the per-file-ignores VeOmni's ``pyproject.toml``
declares for ``veomni/models/transformers/**/generated/*.py``:

- ``E402`` — generated files paste external imports at original class
  positions (e.g. ``create_patch_from_external`` inline aliases), not at
  the top of the file.
- ``B007`` — upstream Transformers occasionally has unused loop variables.

``E501`` is already in the project-wide ``[tool.ruff].ignore`` list and so is
not repeated here. Downstream projects whose ``pyproject.toml`` does NOT
ignore ``E501`` globally should pass their own ``extra_ignore`` instead of
relying on this default.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


DEFAULT_IGNORE = ("E402", "B007")


def ruff_fix_and_format(
    path: Path,
    *,
    extra_ignore: Optional[tuple[str, ...]] = None,
    isolated: bool = False,
) -> None:
    """Run ``ruff check --fix`` + ``ruff format`` on *path*.

    Args:
        path: file to normalize in-place.
        extra_ignore: additional ruff codes to suppress on top of
            :data:`DEFAULT_IGNORE`. Downstream callers whose project ``ruff``
            config differs from VeOmni's (e.g. no global ``E501`` ignore) can
            pass ``("E501",)`` here so the temp file the drift checker writes
            is normalized against the same effective rule set as the
            checked-in generated file.
        isolated: if True, pass ``--isolated`` to ``ruff`` so the project's
            ``pyproject.toml`` is ignored — ruff falls back to its built-in
            defaults (line-length 88, no per-file-ignores). This gives
            location-independent output: the same temp file produces the
            same normalized form regardless of where ``ruff`` happens to
            discover a ``pyproject.toml``. Recommended for dependent
            projects whose generated files were originally produced under
            ``--isolated``. VeOmni's own files were generated without
            ``--isolated``, so the upstream CLI keeps the False default to
            preserve byte-identity with the checked-in artifacts.
    """
    ignore = list(DEFAULT_IGNORE)
    if extra_ignore:
        ignore.extend(extra_ignore)
    check_cmd = ["ruff", "check", "--fix", "--quiet"]
    format_cmd = ["ruff", "format", "--quiet"]
    if isolated:
        check_cmd.append("--isolated")
        format_cmd.append("--isolated")
    check_cmd.extend(["--ignore", ",".join(ignore), str(path)])
    format_cmd.append(str(path))
    try:
        subprocess.run(check_cmd, check=True, capture_output=True)
        subprocess.run(format_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        # ``capture_output=True`` swallows ruff's diagnostics into the
        # exception object, where they're invisible by default. Re-raise
        # with stdout + stderr inlined so syntax errors in generated files
        # (or a missing ruff binary path quirk) show up directly in the
        # caller's traceback.
        stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(
            f"ruff normalization failed (exit {exc.returncode})\n"
            f"command: {' '.join(exc.cmd)}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        ) from exc
