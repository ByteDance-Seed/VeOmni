import copy
import gc
import os
from typing import Dict

import pytest
import torch

from veomni import _safe_apply_patches
from veomni.arguments import DataArguments, ModelArguments, TrainingArguments
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.trainer.base import BaseTrainer, VeOmniArguments
from veomni.utils.device import empty_cache, get_device_type, synchronize
from veomni.utils.loss_utils import count_loss_token

from ..tools.common_utils import print_device_mem_info
from .utils import (
    ModelMode,
    compare_multi_items,
    prepare_data,
    prepare_model_modes,
    print_all_values,
    set_environ_param,
)
from .weight_sync_adapters import get_sync_weight_func


os.environ["NCCL_DEBUG"] = "OFF"


def _release_device_memory():
    synchronize()
    gc.collect()
    empty_cache()


class TrainerTest(BaseTrainer):
    def __init__(self, hf_model_mode: ModelMode, trainer_config: VeOmniArguments):
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        set_environ_param(hf_model_mode)
        _safe_apply_patches()
        super().__init__(trainer_config)

    def _init_callbacks(self):
        pass

    def build_model_assets(self):
        return []

    def _build_data(self):
        # use dummy micro_batch in this ci
        self.args.compute_train_steps(100)
        self.train_steps = self.args.train_steps
        pass

    def build_parallelize_model(self):
        # no parallel in this ci
        pass

    def forward_backward_step(self, state_dict: Dict[str, torch.Tensor], model_mode: ModelMode, dataloader):
        set_environ_param(model_mode)
        _safe_apply_patches()

        model_name = self.model_config.model_type
        self.args.model.attn_implementation = model_mode.attn_implementation
        self.args.model.moe_implementation = model_mode.moe_implementation

        self._build_model()
        self._build_optimizer_and_scheduler()
        print_device_mem_info(f"[Memory Info] after building model {model_name}:")

        # Sync weights
        if model_mode.sync_weight_func is None:
            self.model.load_state_dict(state_dict)
        else:
            model_mode.sync_weight_func(self.model_config, state_dict, self.model)

        print(f"{'-' * 10} {model_name}_{model_mode} {'-' * 10}")
        args: VeOmniArguments = self.args

        loss: torch.Tensor
        loss_dict: Dict[str, torch.Tensor]

        data_iter = iter(dataloader)
        batch = next(data_iter)
        self.micro_batches_token_len = count_loss_token(batch)
        self.micro_batch_token_len = count_loss_token(batch)

        loss, loss_dict = super().forward_backward_step(batch)
        grad_norm = veomni_clip_grad_norm(self.model, args.train.max_grad_norm)

        _release_device_memory()
        print_device_mem_info(f"[Memory Info] after model {model_name} train_one_step:")

        result_metrics = {k: v.item() for k, v in loss_dict.items()}
        result_metrics["gnorm"] = grad_norm
        return result_metrics


# Test case: (config_path, is_moe, rtol, atol). id= must match weight_sync_adapters key if the model needs custom sync.
# rtol/atol: tolerances for compare_multi_items; can be set per case.
_DEFAULT_RTOL = 1e-2
_DEFAULT_ATOL = 1e-2

test_cases = [
    pytest.param(
        "./tests/toy_config/llama31_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="llama3.1",
    ),
    pytest.param(
        "./tests/toy_config/qwen25_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen2.5",
    ),
    pytest.param(
        "./tests/toy_config/qwen3_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3",
    ),
    pytest.param(
        "./tests/toy_config/qwen3_moe_toy",
        True,
        0.5,
        0.02,
        id="qwen3_moe",
    ),
    pytest.param(
        "./tests/toy_config/seed_oss_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="seed_oss",
    ),
    pytest.param(
        "./tests/toy_config/deepseek_v3_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="deepseek_v3",
    ),
]


@pytest.mark.parametrize("config_path, is_moe, rtol, atol", test_cases)
def test_models_patch_fwd_bwd(
    request: pytest.FixtureRequest,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
):
    case_id = request.node.callspec.id
    sync_weight_func = get_sync_weight_func(case_id)
    hf_model_modes, veomni_model_modes = prepare_model_modes(is_moe=is_moe, sync_weight_func=sync_weight_func)

    # delete flash_attention_3 mode for hf deepseek_v3.
    # TODO: transformers v5 fixed this, remove this after veomni support transformers v5.
    if case_id == "deepseek_v3":
        hf_model_modes = [mode for mode in hf_model_modes if mode.attn_implementation != "flash_attention_3"]

    model_config = ModelArguments(config_path=config_path)
    data_config = DataArguments(train_path="")
    training_config = TrainingArguments(
        output_dir="./test_models_patch",
        enable_mixed_precision=False,
        enable_full_determinism=True,
        init_device=get_device_type(),
    )

    trainer_config = VeOmniArguments(
        model=model_config,
        data=data_config,
        train=training_config,
    )

    trainer = TrainerTest(hf_model_modes[0], trainer_config)

    state_dict = copy.deepcopy(trainer.model.state_dict())

    del trainer.model, trainer.optimizer, trainer.lr_scheduler

    dummy_data_loader = prepare_data(max_seq_len=1024)

    res = {}
    log_keys = []
    # Train HF backend models
    for idx, mode in enumerate(hf_model_modes):
        result_metrics = trainer.forward_backward_step(state_dict, mode, dummy_data_loader)
        if not log_keys:
            log_keys = set(result_metrics.keys())
        else:
            assert set(result_metrics.keys()) == set(log_keys)
        res[mode] = result_metrics
    # Train VeOmni backend models
    for idx, mode in enumerate(veomni_model_modes):
        result_metrics = trainer.forward_backward_step(state_dict, mode, dummy_data_loader)
        assert set(result_metrics.keys()) == set(log_keys)
        res[mode] = result_metrics

    assert len(res) == len(hf_model_modes) + len(veomni_model_modes)

    for key in log_keys:
        print_all_values(res, key, case_id)

    compare_multi_items(res, rtol=rtol, atol=atol)

    _release_device_memory()
    print_device_mem_info("[Memory Info] after running train_compare_models:")
