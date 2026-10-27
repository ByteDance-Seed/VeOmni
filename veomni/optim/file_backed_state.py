"""Local, lossless scratch storage for FP32 optimizer tensors.

Scratch files are not checkpoints. A new process must reconstruct them from a
completed DCP checkpoint. Each tensor has independent storage so that DCP does
not clone a view of a larger pooled storage while saving.
"""

import fcntl
import mmap
import os
import re
import stat
import tempfile
import threading
import weakref
from pathlib import Path

import torch


def _filesystem_type(path):
    """Resolve bind/overlay mounts; memory-backed and remote mounts fail closed."""
    path = str(Path(path).resolve())
    selected = None
    with open("/proc/self/mountinfo", encoding="utf-8") as handle:
        for line in handle:
            mount, separator, filesystem = line.partition(" - ")
            if not separator:
                continue
            mountpoint = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), mount.split()[4])
            if os.path.commonpath((path, mountpoint)) == mountpoint:
                if selected is None or len(mountpoint) > len(selected[0]):
                    selected = (mountpoint, filesystem.split()[0])
    if selected is None:
        raise RuntimeError("Could not determine optimizer scratch filesystem type")
    return selected[1]


class FileBackedFP32State:
    """Own unnamed MAP_SHARED scratch files until their tensor aliases die."""

    def __init__(self, root, *, minimum_free_bytes=64 * 1024**3):
        root = Path(root)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("Optimizer scratch root must be an existing absolute directory")
        if _filesystem_type(root) not in {"ext4", "xfs", "btrfs"}:
            raise ValueError("Optimizer scratch must use a supported local disk filesystem")
        info = root.stat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("Optimizer scratch root must be private and owned by the training user")
        if minimum_free_bytes < 0:
            raise ValueError("Scratch free-space reserve cannot be negative")
        if not all(hasattr(os, name) for name in ("posix_fallocate", "posix_fadvise")) or not hasattr(
            mmap.mmap, "madvise"
        ):
            raise RuntimeError("File-backed optimizer state requires Linux fallocate/fadvise/madvise")
        self.directory = root
        self.minimum_free_bytes = minimum_free_bytes
        self._entries = {}
        self._lock = threading.RLock()

    def zeros(self, shape):
        numel = 1
        for dim in shape:
            if dim < 0:
                raise ValueError("File-backed optimizer tensor shape cannot be negative")
            numel *= dim
        if numel == 0:
            return torch.zeros(shape, dtype=torch.float32)
        size = numel * 4
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
        lock_fd = os.open(self.directory / ".veomni-fp32-reserve.lock", flags, 0o600)
        file = None
        mapping = None
        try:
            # All ranks using this root coordinate the free-space check and
            # actual allocation. Other filesystem writers are not controlled.
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                stat = os.statvfs(self.directory)
                if stat.f_bavail * stat.f_frsize < size + self.minimum_free_bytes:
                    raise RuntimeError("Insufficient local disk space for FP32 optimizer scratch")
                # TemporaryFile uses O_TMPFILE or unlinks immediately. A killed
                # process releases the reserved inode and blocks automatically.
                file = tempfile.TemporaryFile(mode="w+b", dir=self.directory)
                os.posix_fallocate(file.fileno(), 0, size)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            mapping = mmap.mmap(file.fileno(), size, access=mmap.ACCESS_WRITE)
            tensor = torch.frombuffer(mapping, dtype=torch.float32, count=numel).view(shape)
            storage = tensor.untyped_storage()
            identity = storage._cdata
            with self._lock:
                self._entries[identity] = (mapping, file)
            weakref.finalize(storage, self._release, identity)
            return tensor
        except BaseException:
            if mapping is not None:
                mapping.close()
            if file is not None:
                file.close()
            raise
        finally:
            os.close(lock_fd)

    def owns(self, tensor):
        with self._lock:
            return tensor.device.type == "cpu" and (
                (tensor.dtype == torch.float32 and tensor.numel() == 0)
                or tensor.untyped_storage()._cdata in self._entries
            )

    @staticmethod
    def _sync_and_evict(entry):
        mapping, file = entry
        fd = file.fileno()
        mapping.flush()
        os.fsync(fd)
        mapping.madvise(mmap.MADV_DONTNEED)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def evict_range(self, tensor, start, count):
        """Flush and discard the page-aligned range touched by one update chunk."""
        if not tensor.is_contiguous() or start < 0 or count < 0 or start + count > tensor.numel():
            raise ValueError("Invalid file-backed state range")
        with self._lock:
            if not self.owns(tensor):
                return False
            if not count:
                return True
            mapping, file = self._entries[tensor.untyped_storage()._cdata]
            begin = (tensor.storage_offset() + start) * tensor.element_size()
            end = begin + count * tensor.element_size()
            begin = begin // mmap.PAGESIZE * mmap.PAGESIZE
            end = min(len(mapping), ((end + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE)
            mapping.flush(begin, end - begin)
            os.fsync(file.fileno())
            mapping.madvise(mmap.MADV_DONTNEED, begin, end - begin)
            os.posix_fadvise(file.fileno(), begin, end - begin, os.POSIX_FADV_DONTNEED)
            return True

    def evict(self, tensor):
        """Synchronize all writes before discarding this tensor's cached pages."""
        with self._lock:
            if not self.owns(tensor):
                return False
            if tensor.numel() == 0:
                return True
            self._sync_and_evict(self._entries[tensor.untyped_storage()._cdata])
            return True

    def evict_all(self):
        with self._lock:
            for entry in self._entries.values():
                self._sync_and_evict(entry)

    def _release(self, identity):
        with self._lock:
            entry = self._entries.pop(identity, None)
            if entry is not None:
                self._release_entry(entry)

    @staticmethod
    def _release_entry(entry):
        mapping, file = entry
        # The storage finalizer can run before its buffer export is released.
        # In that case mmap owns itself until Python releases the last export.
        try:
            mapping.close()
        except BufferError:
            pass
        file.close()
