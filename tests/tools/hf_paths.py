"""Helpers for resolving HuggingFace hub identifiers against a local mirror.

CI runners bind-mount a shared NFS (see `github_runner/`) at the path named by
the `CI_HF_MODELS_DIR` env var. Models that have been pre-staged on the share
live under `$CI_HF_MODELS_DIR/<org>/<name>/` and look like an extracted HF
snapshot (config.json + tokenizer + weights), i.e. they can be fed straight to
`from_pretrained` as a directory path.

We intentionally fall back to the original hub id when the env var is unset or
the expected subdirectory is missing, so the code keeps working in dev
environments and in CI jobs that haven't been migrated yet (those runs still
hit `HF_ENDPOINT=https://hf-mirror.com`).
"""

from __future__ import annotations

import os


_HF_MODELS_DIR_ENV = "CI_HF_MODELS_DIR"


def hf_local_or_remote(hub_id: str) -> str:
    """Map a HuggingFace hub id to a local path when the shared NFS has it.

    Returns ``$CI_HF_MODELS_DIR/<hub_id>`` if that env var is set and the
    directory exists; otherwise returns ``hub_id`` unchanged so callers can
    still reach the hub (via ``HF_ENDPOINT`` or the public endpoint).
    """
    base = os.environ.get(_HF_MODELS_DIR_ENV)
    if base:
        local = os.path.join(base, hub_id)
        if os.path.isdir(local):
            return local
    return hub_id
