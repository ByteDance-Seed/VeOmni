"""Convert aligned Tom-and-Jerry caption/video lists to VeOmni DiT JSONL."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare_dataset(root, output):
    root, output = Path(root).resolve(), Path(output)
    captions = (root / "captions.txt").read_text(encoding="utf-8").splitlines()
    videos = (root / "videos.txt").read_text(encoding="utf-8").splitlines()
    if not captions or len(captions) != len(videos):
        raise ValueError(f"Caption/video count mismatch: {len(captions)} / {len(videos)}")
    records = []
    seen = set()
    for line, (caption, relative) in enumerate(zip(captions, videos), 1):
        path = (root / relative.strip()).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid video at line {line}: {relative}")
        if path in seen or not caption.strip():
            raise ValueError(f"Duplicate video or empty caption at line {line}")
        seen.add(path)
        records.append({"id": line - 1, "prompt": caption, "video_bytes": str(path)})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the complete source before creating an output; never overwrite it.
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "dataset": "Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset",
        "samples": len(records),
        "source_list_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("captions.txt", "videos.txt")
        },
        "jsonl_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    # Sidecar must not be mistaken for another dataset shard by directory loading.
    output.with_suffix(".manifest").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.dataset_root, args.output), indent=2))
