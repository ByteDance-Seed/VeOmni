import os
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from veomni.distributed.parallel_state import _init_parallel_state
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
    _init_parallel_state(
        dp_size=int(os.getenv("WORLD_SIZE")),
        dp_mode="fsdp2",
    )

    # Construct fake data
    pixel_values, image_grid_thw = construct_data()
    # spatial_merge_unit = 1
    check_recover_precision(pixel_values, image_grid_thw)
    check_merged_bridge_gradients()

    print("all test passed")

    dist.barrier()
    dist.destroy_process_group()


def check_merged_bridge_gradients():
    """Keep legacy image/video entry points while sharing inverse and VJP routing."""
    rank = dist.get_rank()
    grid = torch.tensor([[1, 2, 2]] * (rank + 1), device=get_device_type(), dtype=torch.long)
    pixels = torch.arange((rank + 1) * 4 * 3, device=get_device_type(), dtype=torch.float32).reshape(-1, 3)
    pixels = pixels + 1000 * rank
    balancer = Qwen3VLEncoderDataBalance(spatial_merge_unit=4)
    balanced_image, _ = balancer.balance_data(pixels, grid, data_type="image")
    # A later video invocation must not overwrite the image inverse plan.
    balancer.balance_data(pixels + 17, grid, data_type="video")
    merged = balanced_image.reshape(-1, 4, 3).mean(dim=1)
    hidden = merged.detach().clone().requires_grad_()
    features = [merged.detach().clone().requires_grad_() for _ in range(2)]
    restored, deepstack = balancer.data_bridge(hidden, features, data_type="image")
    expected = pixels.reshape(-1, 4, 3).mean(dim=1)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    for feature in deepstack:
        torch.testing.assert_close(feature, expected, rtol=0, atol=0)
    loss = (restored * expected).sum()
    for index, feature in enumerate(deepstack, start=2):
        loss = loss + (feature * expected * index).sum()
    loss.backward()
    torch.testing.assert_close(hidden.grad, merged, rtol=0, atol=0)
    for index, feature in enumerate(features, start=2):
        torch.testing.assert_close(feature.grad, merged * index, rtol=0, atol=0)
    video, _ = balancer.data_bridge((merged + 17).detach().requires_grad_(), [], require_grad=False, data_type="video")
    assert not video.requires_grad
    torch.testing.assert_close(video, expected + 17, rtol=0, atol=0)
    with torch.no_grad():
        recovered, _ = balancer.data_bridge(hidden, [], data_type="image")
    assert not recovered.requires_grad


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("outer_no_grad", [False, True])
def test_singleton_bridge_disabled_grad(monkeypatch, empty, outer_no_grad):
    monkeypatch.setattr(
        "veomni.utils.data_balance.data_balance.get_parallel_state", lambda: SimpleNamespace(dp_group=None)
    )
    count = 0 if empty else 2
    balancer = Qwen3VLEncoderDataBalance(spatial_merge_unit=1)
    balancer.balance_data(torch.ones(count, 3), torch.ones(count, 3, dtype=torch.long))
    hidden = torch.ones(count, 3, requires_grad=True)
    with torch.set_grad_enabled(not outer_no_grad):
        restored, features = balancer.data_bridge(hidden, [hidden], require_grad=outer_no_grad)
    assert not restored.requires_grad and not features[0].requires_grad
    torch.testing.assert_close(restored, hidden, rtol=0, atol=0)


def test_encoder_balance():
    world_size = int(os.getenv("VEOMNI_BALANCE_WORLD_SIZE", "8"))
    if world_size < 1:
        raise ValueError("VEOMNI_BALANCE_WORLD_SIZE must be positive.")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
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
