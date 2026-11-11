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

"""Finish an HF export whose row-sharded ``emb`` tables were left split by rank.

A training-time HF export keeps each oversized embedding-parallel table as the
per-rank shards DCP wrote, and records the files they belong in. This writes
those files and removes the shards. Pass one export directory, or any root
above several (a step's ``hf_ckpt/``): every directory holding a marker below
it is merged.

    python scripts/merge_hf_export.py <step>/hf_ckpt
"""

import argparse
import os

from veomni.checkpoint.layout import HF_PENDING_MERGE_FILENAME, HF_SHARDED_DIRNAME
from veomni.utils import helper
from veomni.utils.save_safetensor_utils import merge_pending_hf_shards


logger = helper.create_logger(__name__)


def scan(root: str) -> tuple[list[str], list[str]]:
    """Exports below ``root`` holding a marker, and shard directories with no marker beside them."""
    pending, orphaned = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        if HF_PENDING_MERGE_FILENAME in filenames:
            pending.append(dirpath)
        elif HF_SHARDED_DIRNAME in dirnames:
            orphaned.append(os.path.join(dirpath, HF_SHARDED_DIRNAME))
        dirnames[:] = [d for d in dirnames if d != HF_SHARDED_DIRNAME]
    return sorted(pending), sorted(orphaned)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="An HF export directory, or a root containing several.")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Output files written at once per export (default: all of them).",
    )
    args = parser.parse_args()

    exports, orphaned = scan(args.path)
    for shard_dir in orphaned:
        logger.warning(
            f"{shard_dir} has no {HF_PENDING_MERGE_FILENAME} beside it: left over from an interrupted save or "
            "merge. Check the export loads, then delete it."
        )
    if not exports:
        logger.info(f"No {HF_PENDING_MERGE_FILENAME} under {args.path}: nothing to merge.")
    failed = []
    for export in exports:
        logger.info(f"Merging {export}")
        try:
            merge_pending_hf_shards(export, num_threads=args.num_threads)
        except Exception:
            logger.exception(f"Could not merge {export}; its marker and shards are kept.")
            failed.append(export)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(exports)} export(s) not merged: {', '.join(failed)}")


if __name__ == "__main__":
    main()
