#!/bin/bash
# CI build entrypoint for the `patchgen` Python wheel.
#
# Distinct from the root-level ``VeOmni/build.sh`` (which builds the veomni
# wheel). Point your CI/CD here when releasing patchgen as a standalone
# distribution.
#
# Output layout: builds the patchgen sdist from ``patchgen-pkg/`` and
# extracts its contents into ``<repo-root>/output/`` (the typical CI
# product-output directory). Extracting — rather than copying the
# ``.tar.gz`` itself — matches the convention where the platform re-tars
# whatever lives under ``output/`` into the final distribution artifact;
# leaving the sdist tarball in place would produce a tarball-in-tarball.
#
# Pass ``$OUTPUT_PATH`` to override the destination for local smoke
# tests, e.g.:
#
#   OUTPUT_PATH=/tmp/x bash patchgen-pkg/build.sh
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_PATH:-$REPO_ROOT/output}"

cd "$SCRIPT_DIR"

python3 -m pip install --upgrade pip build
python3 -m build --sdist

mkdir -p "$OUTPUT_DIR"
tar -xzf dist/patchgen-*.tar.gz -C "$OUTPUT_DIR" --strip-components=1

ls -la "$OUTPUT_DIR/"
