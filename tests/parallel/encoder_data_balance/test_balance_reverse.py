import os
import random
import subprocess
import sys

import torch
import torch.distributed as dist

from veomni.distributed.parallel_state import init_parallel_state
from veomni.utils.data_balance.data_balance import Qwen3VLEncoderDataBalance
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


def construct_data():
    batch_size = random.randint(3, 15)
    image_grid_thw = torch.ones((batch_size, 3), dtype=torch.long, device=get_device_type())
    for i in range(batch_size):
        image_grid_thw[i, 1] = random.randint(10, 50)
        image_grid_thw[i, 2] = random.randint(10, 50)
    pixel_lengths = torch.prod(image_grid_thw, dim=1)
    pixel_values = torch.cat([torch.randn((pl, 1152), device=get_device_type()) for pl in pixel_lengths])

    return pixel_values, image_grid_thw


def check_recover_precision(pixel_values, image_grid_thw):
    """check the persision of recover balance"""
    if torch.distributed.get_rank() == 0:
        print("check the persision of recover balance")
    # initialize Qwen3VLEncoderDataBalance
    databalance = Qwen3VLEncoderDataBalance(spatial_merge_unit=1)
    # balance
    balanced_pixel_values, balanced_image_grid_thw = databalance.balance_data(pixel_values, image_grid_thw)
    # recover
    re_pixel_values, re_deepstack_feat_list = databalance.data_bridge(
        hidden_state=balanced_pixel_values, deepstack_feature_lists=[balanced_pixel_values, balanced_pixel_values]
    )

    # check pixel_values recover percision
    assert pixel_values.equal(re_pixel_values), (
        f"pixel_values != re_pixel_values, rank: {torch.distributed.get_rank()}, "
        f"check failed, in check_balance_performance_and_recover_precision()"
    )
    dist.barrier()
    if dist.get_rank() == 0:
        print("pixel_values check pass")

    # check deepstack feature recover percision
    for i, ds_feat in enumerate(re_deepstack_feat_list):
        assert pixel_values.equal(ds_feat), (
            f"pixel_values != re_deepstack_feat_list[{i}], rank: {torch.distributed.get_rank()}, "
            f"check failed, in check_balance_performance_and_recover_precision()"
        )
    dist.barrier()
    if dist.get_rank() == 0:
        print("deepstack feature check pass")


def main():
    get_torch_device().set_device(f"{get_device_type()}:{os.getenv('RANK')}")
    dist.init_process_group(backend=get_dist_comm_backend())
    init_parallel_state(
        dp_size=int(os.getenv("WORLD_SIZE")),
        dp_mode="fsdp2",
    )

    # Construct fake data
    pixel_values, image_grid_thw = construct_data()
    # spatial_merge_unit = 1
    check_recover_precision(pixel_values, image_grid_thw)

    print("all test passed")

    dist.barrier()
    dist.destroy_process_group()


def test_encoder_balance():
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=8",
        "--node-rank=0",
        "--master_addr=localhost",
        "--master_port=12345",
        "tests/parallel/encoder_data_balance/test_balance_reverse.py",
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise


if __name__ == "__main__":
    main()
