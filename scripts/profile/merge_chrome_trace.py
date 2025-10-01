import argparse
import glob
import gzip
import json
import os
from typing import List


def merge_traces_direct(input_patterns: List[str], output: str):
    """
    Directly merge PyTorch trace files without timeline alignment.
    Each rank will maintain its original timestamps.
    """
    merged = {"schemaVersion": 1, "deviceProperties": [], "traceEvents": []}

    for pattern in input_patterns:
        files = sorted(glob.glob(pattern))

        for i, file_path in enumerate(files):
            print(f"Processing {file_path}...")

            # Extract rank from filename
            rank = None
            if "rank" in file_path:
                rank_part = file_path.split("rank")[1].split("_")[0]
                try:
                    rank = int(rank_part)
                except ValueError:
                    rank = i
            else:
                rank = i

            # Open gzipped or regular JSON
            if file_path.endswith(".gz"):
                with gzip.open(file_path, "rt") as f:
                    data = json.load(f)
            else:
                with open(file_path) as f:
                    data = json.load(f)

            # Merge device properties
            if "deviceProperties" in data:
                for device in data["deviceProperties"]:
                    device_copy = device.copy()
                    device_copy["name"] = f"Rank {rank} - {device.get('name', 'Unknown Device')}"
                    merged["deviceProperties"].append(device_copy)

            # Add trace events with rank identification
            if "traceEvents" in data:
                for event in data["traceEvents"]:
                    event_copy = event.copy()

                    # Add rank information to event names
                    if "name" in event_copy:
                        event_copy["name"] = f"[R{rank}] {event_copy['name']}"

                    # Modify process/thread IDs to separate ranks
                    if "pid" in event_copy:
                        if isinstance(event_copy["pid"], str):
                            event_copy["pid"] = f"rank{rank}_{event_copy['pid']}"
                        else:
                            event_copy["pid"] = int(event_copy["pid"]) + (rank * 1000)
                    else:
                        event_copy["pid"] = rank * 1000

                    if "tid" in event_copy:
                        if isinstance(event_copy["tid"], str):
                            event_copy["tid"] = f"rank{rank}_{event_copy['tid']}"
                        else:
                            event_copy["tid"] = int(event_copy["tid"]) + (rank * 1000)
                    else:
                        event_copy["tid"] = rank * 1000

                    merged["traceEvents"].append(event_copy)

    # Sort events by timestamp
    merged["traceEvents"].sort(key=lambda x: x.get("ts", 0))

    print(f"Writing merged trace to {output}...")
    print(f"Total events: {len(merged['traceEvents'])}")

    # Write compressed JSON
    if not output.endswith(".gz"):
        output += ".gz"

    with gzip.open(output, "wt") as f:
        json.dump(merged, f, separators=(",", ":"))

    print(f"Merged trace saved as {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PyTorch profiler traces.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input trace files")
    parser.add_argument(
        "--pattern",
        type=str,
        default="veomni_rank*.pt.trace.json.gz",
        help="Filename pattern for trace files (default: veomni_rank*.pt.trace.json.gz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_trace.json.gz",
        help="Output merged JSON file (default: merged_trace.json)",
    )

    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, args.pattern)
    merge_traces_direct([input_path], args.output)
