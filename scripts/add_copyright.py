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

"""One-off script to add standard Bytedance copyright to files that lack it."""

from pathlib import Path


STANDARD_HEADER = """# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    # Same list as check_copyright.py
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
        if (
            path.suffix in include_suffixes
            or path.name in include_names
            or any(path.name.startswith(p) for p in include_name_prefix)
        ):
            files.append(path)

    added = 0
    for path in sorted(files):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "Copyright 2025 Bytedance" in text and "http://www.apache.org/licenses/LICENSE-2.0" in text:
            continue
        if "This file has been modified by Bytedance" in text:
            continue

        lines = text.splitlines(keepends=True)
        out_lines: list[str] = []
        if lines and lines[0].strip().startswith("#!"):
            out_lines.append(lines[0])
            if len(lines) > 1 and lines[1].strip() == "":
                out_lines.append(lines[1])
                rest = lines[2:]
            else:
                out_lines.append("\n")
                rest = lines[1:]
        else:
            rest = lines

        out_lines.append(STANDARD_HEADER)
        if rest and rest[0].strip() == "":
            rest = rest[1:]
        out_lines.extend(rest)

        path.write_text("".join(out_lines), encoding="utf-8")
        print(path.relative_to(root))
        added += 1

    print(f"Added copyright to {added} files.")


if __name__ == "__main__":
    main()
