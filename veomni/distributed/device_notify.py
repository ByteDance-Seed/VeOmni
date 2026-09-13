# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import ctypes
from threading import Lock
from typing import Any, Optional


class _AclrtNotifyRuntime:
    _instance: Optional["_AclrtNotifyRuntime"] = None
    _instance_lock = Lock()

    @classmethod
    def get(cls) -> "_AclrtNotifyRuntime":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtCreateNotify.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64]
        acl.aclrtCreateNotify.restype = ctypes.c_int
        acl.aclrtDestroyNotify.argtypes = [ctypes.c_void_p]
        acl.aclrtDestroyNotify.restype = ctypes.c_int
        acl.aclrtRecordNotify.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        acl.aclrtRecordNotify.restype = ctypes.c_int
        acl.aclrtWaitAndResetNotify.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        acl.aclrtWaitAndResetNotify.restype = ctypes.c_int
        self._acl = acl

    @staticmethod
    def _check(code: int, operation: str, name: str) -> None:
        if code != 0:
            raise RuntimeError(f"aclrtNotify {operation} failed for ILP dependency {name}: error {code}.")

    def create(self, name: str) -> Any:
        handle = ctypes.c_void_p()
        self._check(self._acl.aclrtCreateNotify(ctypes.byref(handle), 0), "create", name)
        return handle

    def record(self, handle: Any, stream_handle: int, name: str) -> None:
        self._check(
            self._acl.aclrtRecordNotify(handle, ctypes.c_void_p(stream_handle)),
            "record",
            name,
        )

    def wait_and_reset(
        self,
        handle: Any,
        stream_handle: int,
        timeout_ms: int,
        name: str,
    ) -> None:
        self._check(
            self._acl.aclrtWaitAndResetNotify(
                handle,
                ctypes.c_void_p(stream_handle),
                timeout_ms,
            ),
            "wait-and-reset",
            name,
        )

    def destroy(self, handle: Any, name: str) -> None:
        self._check(self._acl.aclrtDestroyNotify(handle), "destroy", name)


class DeviceNotify:
    """One-shot device dependency that permits wait-before-record ordering."""

    def __init__(self, name: str, runtime: Optional[Any] = None, timeout_ms: int = 120_000) -> None:
        self.name = name
        self._runtime = runtime or _AclrtNotifyRuntime.get()
        self._timeout_ms = timeout_ms
        self._lock = Lock()
        self._handle = self._runtime.create(name)
        self._wait_enqueued = False
        self._record_enqueued = False

    @staticmethod
    def _stream_handle(stream: Any) -> int:
        stream_handle = getattr(stream, "npu_stream", None)
        if stream_handle is None:
            raise TypeError("ILP aclrtNotify requires a torch_npu stream with an npu_stream handle.")
        return int(stream_handle)

    def wait(self, stream: Any) -> None:
        with self._lock:
            if self._handle is None:
                raise RuntimeError(f"ILP dependency {self.name} was waited after close.")
            if self._wait_enqueued:
                raise RuntimeError(f"ILP dependency {self.name} was waited twice.")
            self._runtime.wait_and_reset(
                self._handle,
                self._stream_handle(stream),
                self._timeout_ms,
                self.name,
            )
            self._wait_enqueued = True

    def record(self, stream: Any) -> None:
        with self._lock:
            if self._handle is None:
                raise RuntimeError(f"ILP dependency {self.name} was recorded after close.")
            if self._record_enqueued:
                raise RuntimeError(f"ILP dependency {self.name} was recorded twice.")
            self._runtime.record(self._handle, self._stream_handle(stream), self.name)
            self._record_enqueued = True

    def close(self) -> None:
        with self._lock:
            if self._handle is None:
                return
            self._runtime.destroy(self._handle, self.name)
            self._handle = None
