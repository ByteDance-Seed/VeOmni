"""Encode aligned LQ/HQ media pairs for VeOmni's existing dit_offline transform.

Each JSONL line: {"video": "/path/to/lq.mp4", "target": "/path/to/hq.mp4"}.
The generated Parquet contains pickled tensors, matching the existing trainer.
Only consume datasets produced by trusted preprocessing code.
"""

import argparse
import json
import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from veomni.models.diffusers.seedvr2.conditioning_seedvr2 import SeedVR2ConditionConfig, SeedVR2ConditionModel
from veomni.models.diffusers.seedvr2.inference import preprocess_video, read_media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    if min(args.height, args.width) < 16:
        parser.error("height and width must be at least 16")
    torch.manual_seed(args.seed)
    condition = SeedVR2ConditionModel(SeedVR2ConditionConfig(base_model_path=str(args.weights)))
    condition.eval().to(device=args.device, dtype=getattr(torch, args.dtype))
    args.output.mkdir(parents=True)
    count = 0
    with args.manifest.open(encoding="utf-8") as manifest:
        for line in manifest:
            if not line.strip():
                continue
            pair = json.loads(line)
            low, low_fps = read_media(args.manifest.parent / pair["video"])
            high, high_fps = read_media(args.manifest.parent / pair["target"])
            if low.shape[0] != high.shape[0] or low_fps != high_fps:
                raise ValueError("Pairs must have matching frame counts and frame rates.")
            low = preprocess_video(low, args.height * args.width)
            high = preprocess_video(high, args.height * args.width)
            if low.shape != high.shape:
                raise ValueError("Paired media must have matching aspect ratios and aligned content.")
            with torch.inference_mode():
                embeddings = condition.get_condition(videos=[low], target_videos=[high])
            row = {key: pickle.dumps(value[0].detach().cpu()) for key, value in embeddings.items()}
            pq.write_table(pa.Table.from_pylist([row]), args.output / f"pair-{count:06d}.parquet")
            count += 1
    if not count:
        raise ValueError("Manifest contains no pairs.")
    print(
        json.dumps(
            {
                "pairs": count,
                "seed": args.seed,
                "output": str(args.output),
                "objective": "supervised one-step velocity; not official adversarial training",
            }
        )
    )


if __name__ == "__main__":
    main()
