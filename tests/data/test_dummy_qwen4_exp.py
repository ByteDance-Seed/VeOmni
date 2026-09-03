import torch

from veomni.data.dummy_dataset import build_dummy_dataset
from veomni.utils.constants import IGNORE_INDEX


def test_dummy_qwen4_exp_dataset_matches_toy_vlm_inputs():
    dataset = build_dummy_dataset(task_type="qwen4exp", size=2, max_seq_len=64)
    example = dataset[0][0]
    multimodal_mask = example["image_mask"] | example["video_mask"]

    assert len(dataset) == 2
    assert example["input_ids"].shape == (64,)
    assert example["position_ids"].shape == (3, 64)
    assert example["pixel_values"].shape[1] == 4 * 4 * 2 * 3
    assert example["pixel_values_videos"].shape[1] == 4 * 4 * 2 * 3
    assert torch.all(example["input_ids"] >= 0)
    assert torch.all(example["input_ids"] < 120)
    assert torch.all(example["input_ids"][multimodal_mask] == 0)
    assert torch.all(example["labels"][multimodal_mask] == IGNORE_INDEX)
    assert torch.any(example["labels"][~multimodal_mask] != IGNORE_INDEX)
