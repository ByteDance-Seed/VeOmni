from functools import partial

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

from veomni.data import (
    MappingDataset,
    build_chat_template,
    build_dataloader,
)
from veomni.data.data_transform import process_sft_example
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.models import build_tokenizer
from veomni.utils import helper
from veomni.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    init_parallel_state(
        dp_size=4,
        tp_size=1,
        ep_size=1,
        pp_size=1,
        cp_size=1,
        ulysses_size=1,
        dp_mode="dp",
    )
    dataset_path = "glaiveai/glaive-code-assistant-v2"
    dataset_iterator = load_dataset(dataset_path, split="train").select(range(1000))
    dataset_iterator = dataset_iterator.map(
        lambda x: {
            "messages": [{"role": "user", "content": x["question"]}, {"role": "assistant", "content": x["answer"]}]
        }
    )
    dataset_iterator = dataset_iterator.add_column("id", range(len(dataset_iterator)))
    tokenizer = build_tokenizer("Qwen/Qwen2.5-Coder-32B-Instruct")
    chat_template = build_chat_template("chatml", tokenizer)
    transform = partial(
        process_sft_example,
        chat_template=chat_template,
        max_seq_len=2048,
        text_keys=["messages"],
    )
    train_dataset = MappingDataset(data=dataset_iterator, transform=transform)
    total_train_dataset = len(train_dataset)
    total_tokens = 0

    for item in train_dataset:
        for sub_item in item:
            total_tokens += len(sub_item["input_ids"])

    train_steps = total_tokens / (4096 * 128)

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        micro_batch_size=4,
        global_batch_size=128,
        dataloader_batch_size=4,
        seed=44,
        max_seq_len=4096,
        train_steps=train_steps,
        rmpad=True,
        rmpad_with_pos_ids=True,
        bsz_warmup_ratio=0.0,
        bsz_warmup_init_mbtoken=0,
        dyn_bsz_margin=0,
        dyn_bsz_buffer_size=0,
        num_workers=2,
        drop_last=True,
        pin_memory=False,
        prefetch_factor=2,
    )

    total_samples_data_loader = 0
    list_ids = []
    for batch in tqdm(train_dataloader):
        assert len(batch) == 128 // (4 * 4), "len(batch) should be equally 128 / world_size / micro_batch_size"
        for micro_batch in batch:
            position_ids = micro_batch["position_ids"]
            # we will calculate the total number of samples in the data loader by summing the total number of 0 in the position_ids
            total_zeros = (position_ids == 0).sum()
            total_samples_data_loader += total_zeros
            list_ids.extend(micro_batch["id"].tolist())

    # How to collect the ids from all the workers?
    all_list_ids = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_list_ids, list_ids, group=get_parallel_state().fsdp_group)
    all_list_ids = sum(all_list_ids, [])
    all_list_ids = sum(all_list_ids, [])
    total_unique_ids = len(set(all_list_ids))
    total_samples_data_loader = float(total_samples_data_loader)
    total_samples_data_loader = all_reduce(
        total_samples_data_loader, op="sum", group=get_parallel_state().fsdp_group, device=torch.device("cpu")
    )  # we get the total number of samples in the data loader

    if dist.get_rank() == 0:
        print(f"Total samples in the data loader: {total_samples_data_loader}")
        print(f"Total samples in the dataset: {total_train_dataset}")
        print(f"Total unique ids: {total_unique_ids}")

# Before fix the bug, the result is:
# Total samples in the data loader: 1226.0
# Total samples in the dataset: 1000
# Total unique ids: 248

# After fix the bug, the result is:
# Total samples in the data loader: 1237.0
# Total samples in the dataset: 1000
# Total unique ids: 992
