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


def cache_aware_checkpointer(checkpointer):
    """Enable cache advice on a manager without changing save ownership."""
    import inspect

    from .dcp_checkpointer import DistributedCheckpointer

    if not isinstance(checkpointer, type) or not issubclass(checkpointer, DistributedCheckpointer):
        raise ValueError("DCP source cache advice requires the distributed checkpointer")
    load_signature = inspect.signature(checkpointer.load)
    cache_parameter = load_signature.parameters.get("evict_source_cache")
    if cache_parameter is None or cache_parameter.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise ValueError("DCP source cache advice requires an explicit keyword-capable evict_source_cache parameter")

    class CheckpointerWithCacheAdvice:
        def load(self, *args, **kwargs):
            bound = load_signature.bind_partial(*args, **kwargs)
            if "evict_source_cache" not in bound.arguments:
                kwargs["evict_source_cache"] = True
            return checkpointer.load(*args, **kwargs)

        def __getattr__(self, name):
            # In particular, save and pending-save ownership remain on the
            # original class rather than a new per-manager subclass.
            return getattr(checkpointer, name)

    return CheckpointerWithCacheAdvice()
