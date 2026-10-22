"""Allocate rendezvous ports without reusing a pending launch's port."""

import fcntl
import json
import os
import socket
from pathlib import Path


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def find_free_port(registry=None):
    registry = registry or os.environ.get("VEOMNI_CI_PORT_REGISTRY")
    if not registry:
        return _free_port()
    path = Path(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        claimed = set(json.loads(path.read_text())) if path.exists() else set()
        for _ in range(128):
            port = _free_port()
            if port in claimed:
                continue
            claimed.add(port)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(sorted(claimed)))
            temporary.replace(path)
            return port
    raise RuntimeError("Could not allocate a distinct CI rendezvous port")
