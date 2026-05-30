#!/bin/bash
# CI build entrypoint for the `patchgen` Python wheel.
#
# Distinct from the root-level ``VeOmni/build.sh`` (which builds the veomni
# wheel). Point your CI/CD here when releasing patchgen as a standalone
# distribution.
#
# Honors ``$OUTPUT_PATH`` (the convention many CI platforms use to mark the
# artifact upload directory) and falls back to ``./output`` for local
# smoke tests:
#
#   OUTPUT_PATH=/tmp/x bash patchgen-pkg/build.sh
#
# Produces both an sdist (``patchgen-<version>.tar.gz``) and a
# ``py3-none-any`` wheel.
set -euxo pipefail

cd "$(dirname "$0")"

python3 -m pip install --upgrade pip build
python3 -m build --sdist --wheel

mkdir -p "${OUTPUT_PATH:-output}"
cp dist/*.tar.gz dist/*.whl "${OUTPUT_PATH:-output}/"

ls -la "${OUTPUT_PATH:-output}/"
