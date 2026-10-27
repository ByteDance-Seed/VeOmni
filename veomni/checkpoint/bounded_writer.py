# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License").

"""Bound the tensor payload in each serial DCP filesystem write."""

import gc
import math
import queue
from collections import Counter

import torch
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX
from torch.distributed.checkpoint.planner import WriteItemType


def _buckets(items, max_bytes):
    """Limit tensor bytes per file; a single oversized item gets its own file."""
    if max_bytes <= 0:
        raise ValueError("DCP file bucket size must be positive")
    bucket, size = [], 0
    for item in items:
        if item.type == WriteItemType.BYTE_IO:
            # PyTorch writes BYTE_IO entries serially and does not retain the
            # whole set. Keeping them together avoids one file per scalar
            # optimizer-group setting; one opaque value is not size bounded.
            bucket.append(item)
            continue
        if item.tensor_data is None:
            raise ValueError("DCP tensor write is missing its metadata")
        meta = item.tensor_data
        # ``size`` is the global shape for DTensor/ShardedTensor; DCP copies
        # this write item's local chunk. Using the global shape would create
        # thousands of unnecessary tiny files for a sharded model.
        item_bytes = (
            math.prod(meta.chunk.sizes) * torch.empty((), dtype=meta.properties.dtype, device="meta").element_size()
        )
        if bucket and size + item_bytes > max_bytes:
            yield bucket
            bucket, size = [], 0
        bucket.append(item)
        size += item_bytes
        if size >= max_bytes:
            yield bucket
            bucket, size = [], 0
    if bucket:
        yield bucket


class BoundedFileSystemWriter(FileSystemWriter):
    """Write bounded-size buckets with the ordinary PyTorch DCP serialization.

    PyTorch 2.10 holds tensor copies in one file until that file closes. Some
    runtimes also retain temporary storage through Python reference cycles. Run
    one file at a time and collect cycles at its boundary, so those copies do
    not accumulate across files. This bounds tensor payload in each file to the
    largest of ``max_bytes`` or one tensor item. It does not bound unrelated
    allocator caches, filesystem page cache, or opaque BYTE_IO payloads. DCP's
    planner and metadata retain the exact tensor values and offsets. Existing
    checkpoints are not overwritten by default. As with FileSystemWriter,
    ``sync_files=False`` does not guarantee durability across a host crash.
    """

    def __init__(self, path, *, max_bytes=256 * 1024**2, sync_files=False, overwrite=False):
        if max_bytes <= 0:
            raise ValueError("DCP file bucket size must be positive")
        self.max_bucket_bytes = max_bytes
        super().__init__(
            path,
            single_file_per_rank=True,
            sync_files=sync_files,
            thread_count=1,
            per_thread_copy_ahead=0,
            overwrite=overwrite,
        )

    def write_data(self, plan, planner):
        prefix = getattr(plan.storage_data, "prefix", None)
        if not isinstance(prefix, str):
            raise RuntimeError("Unsupported PyTorch DCP filesystem storage plan")
        if self.thread_count != 1 or self.per_thread_copy_ahead != 0:
            raise RuntimeError("Bounded DCP writes require the serial CPU loader")
        results = []
        for index, bucket in enumerate(_buckets(plan.items, self.max_bucket_bytes)):
            file_name = f"{prefix}{index}{DEFAULT_SUFFIX}"
            file_queue = queue.Queue()
            file_queue.put((self.fs.concat_path(self.path, file_name), file_name, bucket))
            try:
                # FileSystemWriter is synchronous with thread_count=1.  Let
                # its complete write frame (including the tensor dictionary)
                # go out of scope before collecting unreachable storage cycles.
                file_results = self._write_data(planner, file_queue).wait()
                # PyTorch's queue worker catches queue.Empty around its entire
                # write loop. If a planner or stream raises it unexpectedly,
                # _write_data can return success without writing every item.
                # Never publish DCP metadata for a partial or duplicate file.
                if len(file_results) != len(bucket) or Counter(r.index for r in file_results) != Counter(
                    item.index for item in bucket
                ):
                    raise RuntimeError(f"Incomplete DCP write for {file_name}")
                results.extend(file_results)
            finally:
                gc.collect()
        future = torch.futures.Future()
        future.set_result(results)
        return future
