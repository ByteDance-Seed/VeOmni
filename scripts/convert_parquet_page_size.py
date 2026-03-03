#!/usr/bin/env python3
"""Convert LIBERO parquet files to use smaller page size and page index.

Rewrites each parquet file with:
- 64KB data page size (default is ~1MB) for finer-grained row access
- Page index (column index + offset index) for efficient page skipping
- Preserves the original schema, row group structure, and compression

Usage:
    python scripts/convert_parquet_page_size.py \
        --src /mnt/local/localcache00/libero \
        --dst /mnt/local/localcache00/libero_64KB \
        --page-size 65536 \
        --num-workers 16
"""

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


def convert_single_file(
    src_path: str,
    dst_path: str,
    page_size: int,
) -> str:
    """Rewrite a single parquet file with the target page size and page index.

    Args:
        src_path: Path to the source parquet file.
        dst_path: Path to write the converted parquet file.
        page_size: Target data page size in bytes.

    Returns:
        A status message string.
    """
    src_file = pq.ParquetFile(src_path)
    table = src_file.read()

    # Preserve original row group size (num rows per row group)
    # by writing one row group at a time
    row_group_sizes = []
    meta = src_file.metadata
    for i in range(meta.num_row_groups):
        row_group_sizes.append(meta.row_group(i).num_rows)

    dst_p = Path(dst_path)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(
        dst_path,
        schema=table.schema,
        compression="snappy",
        data_page_size=page_size,
        write_page_index=True,
        # Keep data_page_version='2.0' for better page index support
        data_page_version="2.0",
    )

    offset = 0
    for rg_size in row_group_sizes:
        chunk = table.slice(offset, rg_size)
        writer.write_table(chunk)
        offset += rg_size
    writer.close()

    return f"OK: {src_path} -> {dst_path}"


def main() -> None:
    """Convert all parquet files in a LIBERO dataset directory."""
    parser = argparse.ArgumentParser(
        description="Convert LIBERO parquet files to use smaller page size and page index."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source LIBERO dataset root directory.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Destination directory for converted dataset.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=65536,
        help="Target data page size in bytes (default: 65536 = 64KB).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16).",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    # Copy meta directory as-is
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    if src_meta.exists():
        if dst_meta.exists():
            shutil.rmtree(dst_meta)
        shutil.copytree(src_meta, dst_meta)
        print(f"Copied meta/ directory to {dst_meta}")

    # Discover all parquet files under data/
    src_data = src_root / "data"
    parquet_files = sorted(src_data.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to convert")

    # Build (src, dst) pairs
    tasks = []
    for src_path in parquet_files:
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        tasks.append((str(src_path), str(dst_path)))

    # Convert in parallel
    t0 = time.time()
    done = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(convert_single_file, src, dst, args.page_size): src for src, dst in tasks}
        for future in as_completed(futures):
            src = futures[future]
            try:
                result = future.result()
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    print(f"[{done}/{len(tasks)}] {elapsed:.1f}s - {result}")
            except Exception as e:
                errors.append((src, str(e)))
                print(f"ERROR: {src}: {e}")

    elapsed = time.time() - t0
    print(f"\nDone: {done}/{len(tasks)} files in {elapsed:.1f}s")
    if errors:
        print(f"Errors ({len(errors)}):")
        for src, err in errors:
            print(f"  {src}: {err}")


if __name__ == "__main__":
    main()
