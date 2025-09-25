import os
from functools import partial

import torch

from veomni.utils.device import get_device_type
from veomni.utils.import_utils import is_torch_npu_available

from .launch import torchrun
from .utils import build_model_optim, load_omnistore, save_omnistore, train_one_step


TEST_SAVE_LOAD_DIR = "test_save_load"
TOYMODEL_CONFIG_PATH = "./tests/model_config/qwen3moe_toy.json"
TOLERANCE = {"rtol": 1e-3, "atol": 1e-5} if is_torch_npu_available() else {"rtol": 0, "atol": 0}


def model_save_load(fsdp_size: int, ep_size: int, ulysses_size: int = 1):
    res = {}
    save_path = os.path.join(TEST_SAVE_LOAD_DIR, f"fsdp{fsdp_size}_ep{ep_size}_sp{ulysses_size}")

    model, optimizer = build_model_optim(
        config_path=TOYMODEL_CONFIG_PATH,
        dp_size=fsdp_size,
        ep_size=ep_size,
        ulysses_size=ulysses_size,
        init_device=get_device_type(),
    )

    # train 1 step
    train_one_step(model, optimizer, seed=42)

    # save
    save_omnistore(model, optimizer, save_path=save_path)

    # train 2 steps
    res["origin_step1"] = train_one_step(model, optimizer, seed=42)
    res["origin_step2"] = train_one_step(model, optimizer, seed=43)

    # load
    load_omnistore(model, optimizer, load_path=save_path)

    # train 2 steps
    res["load_step1"] = train_one_step(model, optimizer, seed=42)
    res["load_step2"] = train_one_step(model, optimizer, seed=43)

    # compare step 1/2 loss & grad norm
    rtol, atol = TOLERANCE["rtol"], TOLERANCE["atol"]
    torch.testing.assert_close(res["origin_step1"][0], res["load_step1"][0], rtol=rtol, atol=atol)
    torch.testing.assert_close(res["origin_step1"][1], res["load_step1"][1], rtol=rtol, atol=atol)
    torch.testing.assert_close(res["origin_step2"][0], res["load_step2"][0], rtol=rtol, atol=atol)
    torch.testing.assert_close(res["origin_step2"][1], res["load_step2"][1], rtol=rtol, atol=atol)


# pytest tasks
test_model_save_load_fsdp8_ep1_sp1 = partial(torchrun, 8, model_save_load, 8, 1, 1)
test_model_save_load_fsdp8_ep2_sp1 = partial(torchrun, 8, model_save_load, 8, 2, 1)
test_model_save_load_fsdp8_ep4_sp1 = partial(torchrun, 8, model_save_load, 8, 4, 1)
test_model_save_load_fsdp8_ep8_sp1 = partial(torchrun, 8, model_save_load, 8, 8, 1)
