import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import find_free_port
from veomni.distributed.parallel_state import _init_parallel_state
from veomni.utils.data_balance.data_balance import Qwen3VLEncoderDataBalance
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


_CI_LAUNCH = LaunchSpec(nproc=4)

# Deterministic imbalanced per-rank image grids (t, h, w):
# - Rank 0 & 2 hold many large/mixed images and scatter to multiple target ranks;
# - Rank 1 & 3 hold fewer images than world_size (1 and 2 images), guaranteeing
#   empty all-to-all send buckets alongside multi-source receives.
_IMBALANCED_GRIDS_BY_RANK = (
    ((1, 48, 44), (1, 42, 40), (1, 38, 36), (1, 34, 32), (1, 28, 24), (1, 20, 18)),
    ((1, 10, 10),),
    ((1, 46, 42), (1, 36, 30), (1, 24, 22), (1, 16, 14), (1, 12, 10)),
    ((1, 30, 28), (1, 14, 12)),
)


def construct_data():
    rank = dist.get_rank()
    device = get_device_type()
    grids = _IMBALANCED_GRIDS_BY_RANK[rank % len(_IMBALANCED_GRIDS_BY_RANK)]
    image_grid_thw = torch.tensor(grids, dtype=torch.long, device=device)
    pixel_lengths = torch.prod(image_grid_thw, dim=1)
    generator = torch.Generator(device="cpu").manual_seed(1000 + rank)
    pixel_values = torch.cat(
        [torch.randn((int(pl.item()), 1152), generator=generator).to(device) for pl in pixel_lengths]
    )

    return pixel_values, image_grid_thw


def check_recover_precision(pixel_values, image_grid_thw):
    """check the persision of recover balance"""
    if torch.distributed.get_rank() == 0:
        print("check the persision of recover balance")
    # initialize Qwen3VLEncoderDataBalance
    databalance = Qwen3VLEncoderDataBalance(spatial_merge_unit=1)
    # balance
    balanced_pixel_values, balanced_image_grid_thw = databalance.balance_data(pixel_values, image_grid_thw)
    # Imbalanced inputs must redistribute across ranks (including empty send buckets on sparse ranks)
    assert not (
        balanced_image_grid_thw.shape == image_grid_thw.shape and balanced_image_grid_thw.equal(image_grid_thw)
    ), f"rank {dist.get_rank()}: expected cross-rank redistribution for imbalanced inputs"
    if image_grid_thw.shape[0] < dist.get_world_size():
        assert any(length == 0 for (length,) in databalance.state_buffer["image"]["pixel_values_origin"]), (
            f"rank {dist.get_rank()}: expected at least one empty send bucket when image count < world_size"
        )
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
    _init_parallel_state(
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


@pytest.mark.launch_plan(provider=_CI_LAUNCH)
def test_encoder_balance():
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={_CI_LAUNCH.nproc}",
        "--node-rank=0",
        "--master_addr=localhost",
        f"--master_port={find_free_port()}",
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
