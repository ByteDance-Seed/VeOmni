"""Generate local image/text fixtures for Qwen-Image pipeline smoke tests."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def generate_dataset(output_dir: Path, samples: int = 16, size: int = 64) -> Path:
    if samples < 1 or size < 16:
        raise ValueError("samples must be positive and size must be at least 16.")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    colors = ["red", "green", "blue", "orange"]
    for index in range(samples):
        color = colors[index % len(colors)]
        image = Image.new("RGB", (size + (index % 2) * size // 2, size), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((size // 4, size // 4, 3 * size // 4, 3 * size // 4), fill=color)
        image_name = f"sample_{index:04d}.png"
        image.save(image_dir / image_name)
        records.append({"prompt": f"A {color} square on a white background.", "image_path": image_name})
    data_path = output_dir / "train.jsonl"
    data_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return data_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    print(generate_dataset(args.output_dir, args.samples, args.size).resolve())
