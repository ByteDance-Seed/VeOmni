import copy
import gc
import os
import sys
from typing import Dict

import pytest
import torch

from veomni import _apply_patches
from veomni.arguments import CheckpointConfig, DataArguments, ModelArguments, TrainingArguments
from veomni.data.data_collator import MainCollator
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.trainer.base import BaseTrainer, VeOmniArguments
from veomni.utils.device import IS_NPU_AVAILABLE, empty_cache, get_device_type, synchronize
from veomni.utils.env import get_env
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to
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
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# enable_full_determinism(42)


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
        _apply_patches()
        super().__init__(trainer_config)

    def _init_callbacks(self):
        pass

    def _build_model_assets(self):
        self.model_assets = []

    # Op names whose OpSlot state should match use_liger_kernel.
    _LIGER_OP_NAMES = {"rms_norm", "swiglu_mlp", "apply_rotary_pos_emb", "cross_entropy_loss"}

    def _verify_opslot_state(self, model_mode: ModelMode):
        """Assert OpSlot binding matches use_liger_kernel after model build."""
        from veomni.ops.dispatch import OpSlot

        modeling_module = sys.modules.get(self.model.__class__.__module__)
        if modeling_module is None:
            return
        for name, obj in vars(modeling_module).items():
            if not isinstance(obj, OpSlot) or obj.op_name not in self._LIGER_OP_NAMES:
                continue
            if model_mode.use_liger_kernel:
                assert obj.has_kernel, (
                    f"OpSlot {name} ({obj.op_name}/{obj.variant}) should have kernel when use_liger_kernel=True"
                )
            else:
                assert not obj.has_kernel, (
                    f"OpSlot {name} ({obj.op_name}/{obj.variant}) should be eager when use_liger_kernel=False"
                )

    def _build_data_transform(self):
        pass

    def _build_dataset(self):
        # use dummy micro_batch in this ci
        self.args.compute_train_steps(100)
        self.train_steps = self.args.train_steps
        pass

    def _build_collate_fn(self):
        data_collate_info = {}
        if self.model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            data_collate_info = {
                "audio_feature_lengths": (0, False, None, None),
                "input_features": (0, True, 0, 1),
                "audio_mask": (-1, False, 0, 1),
                # Pack along dim 0 to match input_features (4, 400, 128): (2,400)+(2,400) -> (4, 400).
                # HF expects feature_attention_mask (num_audios, seq_len) for indexing.
                "feature_attention_mask": (0, False, None, None),
            }
        self.collate_fn = MainCollator(seq_classification=False, data_collate_info=data_collate_info)

    def _build_dataloader(self):
        pass

    def _build_parallelize_model(self):
        # no parallel in this ci
        pass

    def forward_backward_step(self, state_dict: Dict[str, torch.Tensor], model_mode: ModelMode, dataloader):
        set_environ_param(model_mode)
        _apply_patches()

        model_name = self.model_config.model_type
        self.args.model.ops_implementation.attn_implementation = model_mode.attn_implementation
        self.args.model.ops_implementation.moe_implementation = model_mode.moe_implementation

        if model_mode.use_liger_kernel:
            self.args.model.ops_implementation.rms_norm_implementation = "liger"
            self.args.model.ops_implementation.swiglu_mlp_implementation = "liger"
            self.args.model.ops_implementation.apply_rotary_pos_emb_implementation = "liger"
            self.args.model.ops_implementation.cross_entropy_loss_implementation = "liger_fused"
        else:
            self.args.model.ops_implementation.rms_norm_implementation = "eager"
            self.args.model.ops_implementation.swiglu_mlp_implementation = "eager"
            self.args.model.ops_implementation.apply_rotary_pos_emb_implementation = "eager"
            self.args.model.ops_implementation.cross_entropy_loss_implementation = "eager"

        self._build_model()
        self._verify_opslot_state(model_mode)
        self._build_optimizer()
        self._build_lr_scheduler()
        print_device_mem_info(f"[Memory Info] after building model {model_name}:")

        # Sync weights
        if model_mode.sync_weight_func is None:
            self.model.load_state_dict(state_dict)
        else:
            model_mode.sync_weight_func(self.model_config, state_dict, self.model)

        if self.model_config.model_type in ["qwen2_5_omni", "qwen3_omni_moe"]:
            self.model.disable_talker()
            self.model = self.model.thinker

        print(f"{'-' * 10} {model_name}_{model_mode} {'-' * 10}")
        args: VeOmniArguments = self.args

        loss: torch.Tensor
        loss_dict: Dict[str, torch.Tensor]

        data_iter = iter(dataloader)
        raw_features = next(data_iter)
        batch = self.collate_fn(raw_features)
        self.micro_batches_token_len = count_loss_token(batch)
        self.micro_batch_token_len = count_loss_token(batch)

        if self.model_config.model_type in ["qwen2_5_omni", "qwen3_omni_moe"] and get_env("MODELING_BACKEND") == "hf":
            audio_feature_lengths = batch["audio_feature_lengths"]
            # qwen omni got strange logic in audio_forward
            batch["input_features"] = (
                batch["input_features"]
                .reshape(len(audio_feature_lengths), audio_feature_lengths.max(), -1)
                .permute(0, 2, 1)
                .to(dtype=self.model.dtype)
            )
        elif self.model_config.model_type in ["qwen3_omni_moe"] and get_env("MODELING_BACKEND") == "veomni":
            batch["input_features"] = batch["input_features"].to(
                dtype=self.model.dtype
            )  # qwen3 omni didn't handle dtype in audio_forward

        if batch["position_ids"].dim() == 3 and batch["position_ids"].shape[1] == 3:
            batch["position_ids"] = batch["position_ids"].transpose(0, 1).contiguous()

        loss, loss_dict = super().forward_backward_step(batch)
        grad_norm = veomni_clip_grad_norm(self.model, args.train.optimizer.max_grad_norm)

        _release_device_memory()
        print_device_mem_info(f"[Memory Info] after model {model_name} train_one_step:")

        result_metrics = {k: v.item() for k, v in loss_dict.items()}
        result_metrics["gnorm"] = grad_norm
        return result_metrics


# Test case: (config_path, is_moe, rtol, atol). id= must match weight_sync_adapters key if the model needs custom sync.
# rtol/atol: tolerances for compare_multi_items; can be set per case.
_DEFAULT_RTOL = 1e-2
_DEFAULT_ATOL = 1e-2

_TEST_CASES_TRANSFORMERS_V4 = [
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
    pytest.param(
        "./tests/toy_config/qwen2vl_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen2_vl",
    ),
    pytest.param(
        "./tests/toy_config/qwen25vl_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen2_5_vl",
    ),
    pytest.param(
        "./tests/toy_config/qwen3vl_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_vl",
    ),
    pytest.param(
        "./tests/toy_config/qwen25omni_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen2_5_omni",
    ),
    pytest.param(
        "./tests/toy_config/qwen3vlmoe_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_vl_moe",
    ),
    pytest.param(
        "./tests/toy_config/qwen3omni_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_omni_moe",
    ),
]

_TEST_CASES_TRANSFORMERS_V5 = [
    pytest.param(
        "./tests/toy_config/qwen3_5_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_5",
    ),
    pytest.param(
        "./tests/toy_config/qwen3_5_moe_toy/config.json",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_5_moe",
    ),
    pytest.param(
        "./tests/toy_config/qwen2vl_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen2_vl",
    ),
    pytest.param(
        "./tests/toy_config/qwen2_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen2",
    ),
]

if is_transformers_version_greater_or_equal_to("5.0.0"):
    test_cases = _TEST_CASES_TRANSFORMERS_V5
    print("[test_models_patch] Using transformers v5 test cases.")
else:
    test_cases = _TEST_CASES_TRANSFORMERS_V4
    print("[test_models_patch] Using transformers v4 test cases.")


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

    # hf qwen2_5_omni fa3 error
    if case_id == "qwen2_5_omni":
        hf_model_modes = [mode for mode in hf_model_modes if mode.attn_implementation != "flash_attention_3"]

        if IS_NPU_AVAILABLE:
            # npu not support torch.kaiser_window init in Token2WavBigVGANModel
            return

    # Qwen3.5 compatibility:
    # - HF backend doesn't support the test's position_ids test cases.
    # - VeOmni backend doesn't support the padded_bsh cases as we only support packed sequence case.
    if case_id in ("qwen3_5", "qwen3_5_moe"):
        #    hf_model_modes = [mode for mode in hf_model_modes if mode.attn_case != "position_ids"]
        hf_model_modes = [mode for mode in hf_model_modes if mode.attn_implementation != "flash_attention_3"]
        veomni_model_modes = [
            mode for mode in veomni_model_modes if mode.attn_implementation != "veomni_flash_attention_3_with_sp"
        ]

    model_config = ModelArguments(config_path=config_path)
    data_config = DataArguments(train_path="")
    training_config = TrainingArguments(
        checkpoint=CheckpointConfig(output_dir="./test_models_patch"),
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

    model_config = trainer.model_config
    dummy_data_loader = prepare_data(case_id, max_seq_len=1024, model_config=model_config)

    res = {}
    log_keys = []
    # Train HF backend models
    for _idx, mode in enumerate(hf_model_modes):
        result_metrics = trainer.forward_backward_step(state_dict, mode, dummy_data_loader)
        if not log_keys:
            log_keys = set(result_metrics.keys())
        else:
            assert set(result_metrics.keys()) == set(log_keys)
        res[mode] = result_metrics
    # Train VeOmni backend models
    for _idx, mode in enumerate(veomni_model_modes):
        result_metrics = trainer.forward_backward_step(state_dict, mode, dummy_data_loader)
        assert set(result_metrics.keys()) == set(log_keys)
        res[mode] = result_metrics

    assert len(res) == len(hf_model_modes) + len(veomni_model_modes)

    for key in log_keys:
        print_all_values(res, key, case_id)

    compare_multi_items(res, rtol=rtol, atol=atol)

    _release_device_memory()
    print_device_mem_info("[Memory Info] after running train_compare_models:")
