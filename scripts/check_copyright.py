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

"""Check that all source files have correct Copyright headers."""

from __future__ import annotations

import re
import sys
from pathlib import Path


# Standard Bytedance copyright block (template from auto.py)
STANDARD_COPYRIGHT_LINES = [
    "# Copyright 2025 Bytedance Ltd. and/or its affiliates",
    "#",
    '# Licensed under the Apache License, Version 2.0 (the "License");',
    "# you may not use this file except in compliance with the License.",
    "# You may obtain a copy of the License at",
    "#",
    "#     http://www.apache.org/licenses/LICENSE-2.0",
    "#",
    "# Unless required by applicable law or agreed to in writing, software",
    '# distributed under the License is distributed on an "AS IS" BASIS,',
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "# See the License for the specific language governing permissions and",
    "# limitations under the License.",
]

# For modified files: must contain these in header
MODIFIED_MARKER = "This file has been modified by Bytedance Ltd. and/or its affiliates"
ORIGINAL_URL_MARKER = "The original version found at:"
BYTEDANCE_COPYRIGHT = "Copyright 2025 Bytedance Ltd. and/or its affiliates"
APACHE_LICENSE_REF = "http://www.apache.org/licenses/LICENSE-2.0"


def get_files_to_check(root: Path) -> list[Path]:
    """Collect all files that should have copyright (exclude configs and .venv)."""
    exclude_dirs = {".venv", ".git", "__pycache__", "_build", ".ruff_cache"}
    include_suffixes = (".py", ".sh")
    include_names = ("Makefile",)
    include_name_prefix = ("Dockerfile",)

    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in path.parts for part in exclude_dirs):
            continue
        if path.suffix in (".json", ".yaml", ".yml", ".md", ".txt", ".lock", ".toml"):
            continue
        if path.suffix in include_suffixes:
            files.append(path)
        elif path.name in include_names:
            files.append(path)
        elif path.name.startswith(include_name_prefix[0]):
            files.append(path)
    return sorted(files)


def read_header(path: Path, max_lines: int = 25) -> str:
    """Read first max_lines of file as header."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[:max_lines])


def check_standard_copyright(header: str) -> bool:
    """Check that header contains standard Bytedance + Apache 2.0 block."""
    if BYTEDANCE_COPYRIGHT not in header:
        return False
    if APACHE_LICENSE_REF not in header:
        return False
    if "Licensed under the Apache License" not in header:
        return False
    return True


def check_modified_copyright(header: str) -> bool:
    """Check that header is valid modified-file format (original + Bytedance + URL + Apache)."""
    if BYTEDANCE_COPYRIGHT not in header:
        return False
    if MODIFIED_MARKER not in header:
        return False
    if ORIGINAL_URL_MARKER not in header:
        return False
    if APACHE_LICENSE_REF not in header:
        return False
    if "Licensed under the Apache License" not in header:
        return False
    # Should have something like "https://" or "http://" for original URL
    if not re.search(r"https?://\S+", header):
        return False
    return True


def check_file(path: Path, root: Path) -> tuple[bool, str]:
    """Check one file. Returns (ok, message)."""
    header = read_header(path)
    if not header.strip():
        return False, "empty or unreadable"

    if check_modified_copyright(header):
        return True, "modified-format"
    if check_standard_copyright(header):
        return True, "standard"

    if "# Copyright" in header and "Bytedance" not in header:
        return False, "has copyright but missing Bytedance"
    if "Copyright" in header:
        return False, "copyright present but missing Apache 2.0 or Bytedance"
    return False, "missing Copyright header"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    files = get_files_to_check(root)
    failed: list[tuple[Path, str]] = []

    for path in files:
        rel = path.relative_to(root)
        ok, msg = check_file(path, root)
        if not ok:
            failed.append((rel, msg))

    if failed:
        print("Copyright check failed. The following files have missing or invalid headers:\n")
        for rel, msg in failed:
            print(f"  {rel}: {msg}")
        print(f"\nTotal: {len(failed)} file(s).")
        print("\nStandard header template (see veomni/models/auto.py):")
        for line in STANDARD_COPYRIGHT_LINES[:3]:
            print(f"  {line}")
        print("  ...")
        return 1

    print(f"Copyright check passed for {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
