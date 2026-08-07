#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2>=3.1",
#     "pyyaml>=6.0",
# ]
# ///
"""Render docker/{cuda,ascend}/Dockerfile.* from docker/matrix.yaml + docker/templates/*.j2.

Usage::

    uv run docker/generate.py            # regenerate every entry in matrix.yaml
    uv run docker/generate.py --check    # exit 1 if any output would change

A bare ``python docker/generate.py`` also works if ``jinja2`` and ``pyyaml`` are
installed in the active environment (CI uses ``uv run`` which auto-installs the
PEP 723 dependencies above).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined


DOCKER_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCKER_DIR.parent
MATRIX_FILE = DOCKER_DIR / "matrix.yaml"
TEMPLATES_DIR = DOCKER_DIR / "templates"


def build_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )


def build_uv_sync_command(ctx: dict) -> str:
    """Render a multi-line ``uv sync ...`` shell command with ``\\``
    continuation. ``uv_extra_flags`` (optional, per-image) appends raw flags
    — Ascend images use it for ``--allow-insecure-host`` (CANN base CA bundle
    can't validate github.com / pythonhosted.org under ``--locked``)."""
    extras = ctx.get("uv_extras") or []
    dev = bool(ctx.get("uv_dev", False))
    extra_flags = ctx.get("uv_extra_flags") or []
    parts = ["uv sync", "--locked", "--all-packages"]
    parts.extend(f"--extra {e}" for e in extras)
    if dev:
        parts.append("--dev")
    parts.extend(extra_flags)
    return " \\\n    ".join(parts)


def resolve_extras(image: dict, presets: dict[str, list[str]]) -> list[str]:
    """Look up the image's `uv_extras_preset` against the top-level
    `uv_extras_presets` table. We use named presets (rather than inlining a
    list per image) so the cuda/ascend/etc. extras sets stay declared in one
    place and a future backend (e.g. ascend uv migration) only adds a preset
    key + an `images:` entry."""
    name = image.get("name", "?")
    preset_name = image.get("uv_extras_preset")
    if preset_name is None:
        raise RuntimeError(f"image {name!r} is missing required field `uv_extras_preset` (one of: {sorted(presets)})")
    if preset_name not in presets:
        raise RuntimeError(
            f"image {name!r} references unknown uv_extras_preset={preset_name!r}; defined presets: {sorted(presets)}"
        )
    return presets[preset_name]


def render(
    env: Environment,
    image: dict,
    defaults: dict,
    extras_presets: dict[str, list[str]],
) -> tuple[Path, str]:
    ctx = {**defaults, **image}
    ctx["uv_extras"] = resolve_extras(image, extras_presets)
    ctx["uv_sync_command"] = build_uv_sync_command(ctx)
    template = env.get_template(image["template"])
    out_path = REPO_ROOT / image["output"]
    return out_path, template.render(**ctx)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 (without writing) if any output would differ from on-disk content.",
    )
    args = parser.parse_args()

    matrix = yaml.safe_load(MATRIX_FILE.read_text())
    defaults = matrix.get("defaults", {}) or {}
    images = matrix.get("images", []) or []
    extras_presets = matrix.get("uv_extras_presets", {}) or {}
    if not images:
        print(f"no images defined in {MATRIX_FILE}", file=sys.stderr)
        return 1
    if not extras_presets:
        print(f"no uv_extras_presets defined in {MATRIX_FILE}", file=sys.stderr)
        return 1
    if not defaults.get("uv_version"):
        print(f"defaults.uv_version is required in {MATRIX_FILE}", file=sys.stderr)
        return 1

    env = build_env()
    drift: list[Path] = []
    for image in images:
        out_path, rendered = render(env, image, defaults, extras_presets)
        rel = out_path.relative_to(REPO_ROOT)

        if args.check:
            current = out_path.read_text() if out_path.exists() else ""
            if current != rendered:
                drift.append(rel)
                print(f"drift: {rel}", file=sys.stderr)
            else:
                print(f"ok:    {rel}")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered)
            print(f"wrote: {rel}")

    if args.check and drift:
        print(
            "\nRegenerate by running:\n    uv run docker/generate.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
