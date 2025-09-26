from functools import partial

from veomni.utils.device import get_device_type

from .launch import torchrun
from .utils import TOY_MODEL_MAP, build_model_optim


TOYMODEL_CONFIG_PATH = TOY_MODEL_MAP["qwen3_moe"]["path"]
TOYMODEL_NPARAM = TOY_MODEL_MAP["qwen3_moe"]["n_param"]


def distributed_init(fsdp_size: int, ep_size: int, ulysses_size: int = 1):
    model, optimizer = build_model_optim(
        config_path=TOYMODEL_CONFIG_PATH,
        dp_size=fsdp_size,
        ep_size=ep_size,
        ulysses_size=ulysses_size,
        init_device=get_device_type(),
    )

    nparam = sum(p.numel() for p in model.parameters())
    assert nparam == TOYMODEL_NPARAM // fsdp_size


test_distributed_init = partial(torchrun, 4, distributed_init, 4, 1, 1)
