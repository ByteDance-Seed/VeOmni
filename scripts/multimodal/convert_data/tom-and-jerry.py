# pip install soundfile, librosa
import argparse
import math
import os

from datasets import Dataset

from veomni.data.multimodal.video_utils import (
    load_video_bytes_from_path,
)


def load_dataset(dataset_path: str):
    captions_file = os.path.join(dataset_path, "captions.txt")
    videos_file = os.path.join(dataset_path, "videos.txt")

    with open(captions_file, encoding="utf-8") as f:
        captions = f.readlines()

    with open(videos_file, encoding="utf-8") as f:
        video_paths = f.readlines()

    captions = [caption.strip() for caption in captions]
    video_paths = [video_path.strip() for video_path in video_paths]

    assert len(captions) == len(video_paths), (
        f"captions.txt {len(captions)} and {len(video_paths)}videos.txt line not match"
    )

    data = {"text": captions, "video": video_paths}

    dataset = Dataset.from_dict(data)
    return dataset


def tom_and_jerry(dataset_path: str, output_dir: str):
    NUM_SHARD = 30
    NUM_PROC = 32

    dataset = load_dataset(dataset_path)

    os.makedirs(output_dir, exist_ok=True)
    total_len = len(dataset)
    batch_len = math.ceil(total_len / NUM_SHARD)
    print(f"Total length: {total_len}, batch length: {batch_len}")

    index = 0
    for i in range(0, total_len, batch_len):
        print(f"Generating {index}th parquet file")
        end_idx = min(i + batch_len, total_len)
        chunk_ds = dataset.select(range(i, end_idx))
        chunk_num_proc = min(NUM_PROC, len(chunk_ds))

        def process_example(example):
            video_bytes = load_video_bytes_from_path(os.path.join(dataset_path, example["video"]))
            return {
                "prompt": example["text"],
                "video_bytes": video_bytes,
                "source": "Tom-and-Jerry-VideoGeneration-Dataset",
            }

        ds = chunk_ds.map(
            process_example,
            num_proc=chunk_num_proc,
            remove_columns=chunk_ds.column_names,
            keep_in_memory=True,
            desc=f"Processing shard {index}",
        )
        ds.to_parquet(os.path.join(output_dir, f"{index}.parquet"))
        index += 1


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args = args.parse_args()
    tom_and_jerry(args.dataset_path, args.output_dir)
