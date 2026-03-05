# pip install soundfile, librosa
import math
import os

from datasets import Dataset, load_dataset

from veomni.data.multimodal.video_utils import load_video_bytes_from_path


def voice_assistant():
    NUM_SHARD = 500

    dataset = load_dataset("TempoFunk/webvid-10M", split="train")

    def generate_data(examples):
        for index in range(len(examples["videoid"])):
            video_path = examples["contentUrl"][index]
            video_bytes = load_video_bytes_from_path(video_path)
            prompt = examples["name"][index]
            yield {
                "prompt": prompt,
                "video_bytes": video_bytes,
                "source": "webvid-10M",
            }

    output_dir = "/mnt/hdfs/veomni/dataset/video_generation/webvid-10M"
    os.makedirs(output_dir, exist_ok=True)
    total_len = len(dataset)
    batch_len = math.ceil(total_len / NUM_SHARD)
    print(f"Total length: {total_len}, batch length: {batch_len}")

    index = 0
    for i in range(0, total_len, batch_len):
        print(f"Generating {index}th parquet file")
        ds = Dataset.from_generator(
            generate_data,
            gen_kwargs={"examples": dataset.select(range(i, i + batch_len))},
            keep_in_memory=True,
            num_proc=64,
        )
        ds.to_parquet(os.path.join(output_dir, f"{index}.parquet"))
        index += 1


if __name__ == "__main__":
    voice_assistant()
