"""Dummy dataset generator for integration and e2e tests."""

import math
import os

import torch.distributed as dist
from datasets import Dataset

from veomni.data.dummy_dataset import build_dummy_dataset
from veomni.utils.helper import get_cache_dir


class DummyDataset:
    """Generate and save dummy parquet datasets for training tests.

    Wraps ``build_dummy_dataset`` and writes sharded parquet files to a
    cache directory so that the VeOmni data pipeline can load them.
    """

    def __init__(self, num_samples=16, seq_len=8192, dataset_type: str = "text") -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_shard = 2

        self.save_path = get_cache_dir(f"./{dataset_type}")

        if not dist.is_initialized() or dist.get_rank() == 0:
            self.dataset = build_dummy_dataset(dataset_type, self.num_samples, self.seq_len)
            self.build_dummy_dataset()

        if dist.is_initialized():
            dist.barrier()

    def generate_data(self):
        num_samples = len(self.dataset)
        for index in range(num_samples):
            item = self.dataset[index][0]
            yield item

    def build_dummy_dataset(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        batch_len = math.ceil(self.num_samples / self.num_shard)
        print(f"Total length: {self.num_samples}, batch length: {batch_len}")

        index = 0
        for _i in range(0, self.num_samples, batch_len):
            print(f"Generating {index}th parquet file")
            ds = Dataset.from_generator(
                self.generate_data,
                keep_in_memory=True,
                num_proc=1,
            )
            ds.to_parquet(os.path.join(self.save_path, f"{index}.parquet"))
            index += 1

    def clean_cache(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            if os.path.exists(self.save_path):
                os.system(f"rm -rf {self.save_path}")

    def __del__(self):
        self.clean_cache()
