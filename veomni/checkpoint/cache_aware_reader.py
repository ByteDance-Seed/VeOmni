# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Opt-in, per-rank source-page advice after a DCP read has completed."""

import errno
import os
import warnings
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader


class CacheAwareFileSystemReader(FileSystemReader):
    """Advise away the checkpoint byte ranges this rank has finished reading.

    Advice is local to this host and does not guarantee eviction. Other ranks
    may still be reading the same files, so this can increase I/O; it is opt-in.
    """

    def read_data(self, plan, planner):
        future = super().read_data(plan, planner)

        def advise_completed(completed):
            completed.wait()
            if not hasattr(os, "posix_fadvise") or not hasattr(os, "O_NOFOLLOW"):
                raise RuntimeError("Checkpoint cache advice requires POSIX fadvise and O_NOFOLLOW")
            requests = {}
            for item in plan.items:
                info = self.storage_data[item.storage_index]
                relative = Path(info.relative_path)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid checkpoint shard path")
                if info.offset < 0 or info.length <= 0:
                    raise ValueError("Invalid checkpoint shard range")
                requests.setdefault(relative, set()).add((info.offset, info.length))
            for relative, ranges in requests.items():
                try:
                    fd = os.open(Path(self.path) / relative, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                except OSError as error:
                    if error.errno != errno.ELOOP:
                        raise
                    warnings.warn("Skipping checkpoint cache advice for a symlink shard", stacklevel=2)
                    continue
                try:
                    for offset, length in sorted(ranges):
                        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            return None

        return future.then(advise_completed)
