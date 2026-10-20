# tests/utils.py
import os
import signal
import socket
import subprocess

import psutil
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _collect_process_tree(root_pid: int) -> tuple[list[psutil.Process], set[int]]:
    """Return all live processes and process-group IDs in the tree rooted at ``root_pid``."""
    procs: list[psutil.Process] = []
    pgids: set[int] = set()
    try:
        pgids.add(os.getpgid(root_pid))
    except ProcessLookupError:
        pass
    try:
        root = psutil.Process(root_pid)
        procs.append(root)
        for child in root.children(recursive=True):
            procs.append(child)
            try:
                pgids.add(os.getpgid(child.pid))
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass
    except psutil.NoSuchProcess:
        pass
    return procs, pgids


def _signal_process_groups(pgids: set[int], sig: int) -> None:
    """Send ``sig`` to every process group in ``pgids``, ignoring already-exited groups."""
    for pgid in pgids:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass


def run_bounded_process_group(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 180.0,
    grace_timeout: float = 35.0,
) -> None:
    """Run ``command`` in a new session and terminate the launcher + worker groups on timeout.

    ``torchrun`` spawns each worker in its own session (``start_new_session=True``), so
    killing only ``process.pid``'s process group before ``torchrun``'s 30s shutdown window
    completes can orphan ``SIGTERM``-ignoring workers. We snapshot all descendant worker
    process groups before signaling, give ``torchrun`` ``grace_timeout`` (>= 30s by default)
    to finish its own shutdown, and then send ``SIGKILL`` to every recorded process group.
    """
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        procs, pgids = _collect_process_tree(process.pid)
        _signal_process_groups(pgids, signal.SIGTERM)
        try:
            process.wait(timeout=grace_timeout)
        except subprocess.TimeoutExpired:
            more_procs, more_pgids = _collect_process_tree(process.pid)
            procs.extend(more_procs)
            pgids.update(more_pgids)
        finally:
            launcher_pgid = os.getpgid(process.pid) if process.poll() is None else None
            worker_pgids = {pgid for pgid in pgids if pgid != launcher_pgid}
            worker_procs = [proc for proc in procs if proc.pid != process.pid]
            # Kill descendant worker groups first so the still-alive torchrun
            # launcher can reap its direct children before exiting itself.
            _signal_process_groups(worker_pgids, signal.SIGKILL)
            for proc in worker_procs:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(worker_procs, timeout=5.0)
            _signal_process_groups(pgids, signal.SIGKILL)
            if process.poll() is None:
                process.kill()
            process.wait()
        raise

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _dist_worker_entry(rank, world_size, port, func, args, kwargs):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if get_torch_device().is_available() and get_torch_device().device_count() >= world_size:
        backend = get_dist_comm_backend()
        get_torch_device().set_device(rank)
    else:
        backend = "gloo"

    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        func(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def torchrun(func, world_size: int = 4, *args, **kwargs):
    if get_torch_device().is_available() and get_torch_device().device_count() < world_size:
        pytest.skip(f"Requires {world_size} {get_device_type()} devices")

    port = find_free_port()

    mp.spawn(_dist_worker_entry, args=(world_size, port, func, args, kwargs), nprocs=world_size, join=True)
