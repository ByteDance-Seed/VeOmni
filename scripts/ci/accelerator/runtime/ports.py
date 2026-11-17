"""Own live rendezvous servers; retain legacy cooperative port allocation."""

import fcntl
import json
import os
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


@dataclass(frozen=True)
class RendezvousEndpoint:
    """Picklable client configuration for one parent-owned TCPStore server."""

    host: str
    port: int
    prefix: str
    timeout: timedelta

    def connect(self):
        """Connect as a client, including on rank zero; never rebind the port."""
        from torch.distributed import PrefixStore, TCPStore

        client = TCPStore(self.host, self.port, is_master=False, timeout=self.timeout)
        return PrefixStore(self.prefix, client)


@contextmanager
def rendezvous_endpoint(*, timeout=timedelta(minutes=30)):
    """Keep a bound TCPStore alive until the caller has joined its workers.

    Port zero lets the kernel choose and bind atomically. Only the endpoint is
    yielded: the parent retains the sole Python reference to the server, so
    leaving the context closes it. Workers use explicit client stores rather
    than env://, where rank zero would try to bind another server.
    """
    from torch.distributed import TCPStore

    server = TCPStore("127.0.0.1", 0, is_master=True, timeout=timeout, wait_for_workers=False)
    endpoint = RendezvousEndpoint("127.0.0.1", server.port, uuid.uuid4().hex, timeout)
    try:
        yield endpoint
    finally:
        del server


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def find_free_port(registry=None):
    """Return a port number, not an OS reservation; prefer a live rendezvous.

    The registry prevents reuse by cooperating callers in one CI run. Other
    sockets can still claim the port after the probe closes it.
    """
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
