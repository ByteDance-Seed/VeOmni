"""Download trl-lib/ultrafeedback_binarized and save as local parquet.

Usage:
    python scripts/prepare_ultrafeedback.py --output_dir data/ultrafeedback_binarized
"""

import argparse
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/ultrafeedback_binarized")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    output_path = os.path.join(args.output_dir, "train.parquet")
    dataset.to_parquet(output_path)
    print(f"Saved {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()
