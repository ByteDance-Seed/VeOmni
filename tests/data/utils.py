import math
import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset

from veomni.utils import helper
from veomni.utils.device import get_device_type
from veomni.utils.helper import get_cache_dir


logger = helper.create_logger(__name__)


class DummyDataset:
    def __init__(self, size=100, num_shard=2, dataset_name: str = "test_dataset") -> None:
        self.size = size
        self.num_shard = num_shard

        self.save_path = get_cache_dir(f"./{dataset_name}")

        if not dist.is_initialized() or dist.get_rank() == 0:
            self.build_dummy_dataset()

        if dist.is_initialized():
            dist.barrier()

    def generate_data(self, index_list: List):
        for index in index_list:
            input_ids = [index + 1] * (index + 1)
            yield {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": input_ids}

    def build_dummy_dataset(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        batch_len = math.ceil(self.size / self.num_shard)
        print(f"Total length: {self.size}, batch length: {batch_len}")

        index = 0
        for i in range(0, self.size, batch_len):
            print(f"Generating {index}th parquet file")
            ds = Dataset.from_generator(
                self.generate_data,
                gen_kwargs={"index_list": list(range(i + 1, i + batch_len + 1))},
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


def process_dummy_example(
    example: Dict[str, Any],
    max_seq_len: int,
    rmpad_with_pos_ids: bool = False,
    source_name: str = None,
) -> List[Dict[str, "torch.Tensor"]]:
    tokenized_example = {}
    for k, v in example.items():
        if k == "ds_idx" or k == "source_name":
            continue
        else:
            tokenized_example[k] = torch.tensor(v[:max_seq_len], dtype=torch.long)
    if rmpad_with_pos_ids:  # precompute position_ids
        tokenized_example["position_ids"] = torch.arange(0, len(tokenized_example["input_ids"]), dtype=torch.long)
    return [tokenized_example]


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn = nn.Linear(1, 1)


def compare_items(item, rank, group_size, group):
    item = item.to(get_device_type())
    item_list = [torch.empty_like(item) for _ in range(group_size)]

    dist.all_gather(item_list, item, group=group)

    for i in range(0, group_size):
        if not torch.equal(item, item_list[i]):
            logger.info(f"[rank{rank}]: group_rank {i} item is not equal to item {rank}")
            return False

    return True


def compare_global_batch(global_batch_list, global_batch_resume_list):
    for global_batch, global_batch_resume in zip(global_batch_list, global_batch_resume_list):
        for micro_batch, micro_batch_resume in zip(global_batch, global_batch_resume):
            for key in micro_batch.keys():
                if torch.is_tensor(micro_batch[key]):
                    assert torch.all(micro_batch[key] == micro_batch_resume[key])


def compare_metrics(metrics, metrics_resume):
    assert metrics["consume_tokens(M)"] == metrics_resume["consume_tokens(M)"]
