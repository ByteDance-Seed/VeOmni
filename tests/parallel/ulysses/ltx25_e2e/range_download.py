"""Download immutable publisher assets with validated HTTP byte ranges."""

import concurrent.futures
import hashlib
import http.client
import os
import time
import urllib.error
import urllib.request


def download_all(manifest, root, emit, workers=64, chunk_size=64 * 1024**2):
    """Download manifest assets in parallel and verify size and SHA256 before publishing."""
    paths = [root / item["path"] for item in manifest]
    handles = []
    chunks = []
    total = sum(item["size"] for item in manifest)
    try:
        for i, (item, path) in enumerate(zip(manifest, paths)):
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            fd = os.open(partial, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            os.ftruncate(fd, item["size"])
            handles.append(fd)
            chunks.extend(
                (i, start, min(start + chunk_size, item["size"]) - 1) for start in range(0, item["size"], chunk_size)
            )

        def fetch(chunk):
            """Fetch and positionally write one validated byte range with bounded retries."""
            i, start, end = chunk
            item = manifest[i]
            expected_range = f"bytes {start}-{end}/{item['size']}"
            for attempt in range(4):
                try:
                    request = urllib.request.Request(item["url"], headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                            raise ValueError("Publisher returned an unexpected byte range")
                        offset = start
                        while block := response.read(min(1024**2, end - offset + 1)):
                            if offset + len(block) > end + 1:
                                raise ValueError("Range exceeds expected size")
                            view = memoryview(block)
                            while view:
                                written = os.pwrite(handles[i], view, offset)
                                if written <= 0:
                                    raise OSError("No progress writing downloaded range")
                                offset += written
                                view = view[written:]
                            if offset == end + 1:
                                break
                        if offset != end + 1:
                            raise OSError("Incomplete HTTP range")
                    return end - start + 1
                except urllib.error.HTTPError as error:
                    if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                        raise
                    try:
                        delay = int(error.headers.get("Retry-After", 5 * 2**attempt))
                    except ValueError:
                        delay = 5 * 2**attempt
                    time.sleep(min(60, max(0, delay)))
                except (OSError, TimeoutError, http.client.HTTPException):
                    if attempt == 3:
                        raise
                    time.sleep(5 * 2**attempt)

        emit("ranged_download", status="start", bytes=total, workers=workers, chunks=len(chunks))
        completed = 0
        last_report = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                futures = [pool.submit(fetch, chunk) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    completed += future.result()
                    if time.monotonic() - last_report > 30 or completed == total:
                        emit("download_progress", completed_bytes=completed, total_bytes=total)
                        last_report = time.monotonic()
            except BaseException:
                # Stop queued downloads, but let active writes finish before closing their descriptors.
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    finally:
        for fd in handles:
            os.close(fd)
    for item, path in zip(manifest, paths):
        partial = path.with_suffix(path.suffix + ".part")
        digest = hashlib.sha256()
        with partial.open("rb") as stream:
            for block in iter(lambda: stream.read(16 * 1024**2), b""):
                digest.update(block)
        if partial.stat().st_size != item["size"]:
            raise ValueError(f"Weight size mismatch: {item['path']}")
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Weight SHA256 mismatch: {item['path']}")
        partial.replace(path)
        emit("weight_verified", file=item["path"], size=item["size"], sha256=digest.hexdigest())
    return paths
