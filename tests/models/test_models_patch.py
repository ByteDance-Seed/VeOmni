import copy
import functools
import gc
import importlib
import os
import sys
from typing import Dict

import pytest
import torch

from veomni import _apply_patches
from veomni.arguments import (
    AcceleratorConfig,
    CheckpointConfig,
    DataArguments,
    FSDPConfig,
    MixedPrecisionConfig,
    ModelArguments,
    TrainingArguments,
)
from veomni.data.data_collator import MainCollator
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.trainer.base import BaseTrainer, VeOmniArguments
from veomni.utils.device import IS_NPU_AVAILABLE, empty_cache, get_device_type, synchronize
from veomni.utils.env import get_env
from veomni.utils.loss_utils import count_loss_token

from ..tools.common_utils import print_device_mem_info
from ..tools.training_utils import make_eager_ops_config
from .utils import (
    ModelMode,
    compare_multi_items,
    prepare_data,
    prepare_model_modes,
    print_all_values,
    set_environ_param,
)


os.environ["NCCL_DEBUG"] = "OFF"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# enable_full_determinism(42)


# ────────────────────────────────────────────────────────────────────────────
# Test-scope shim for a transformers 5.9 upstream bug in qwen-family multimodal
# `Model.forward(**kwargs)` paths.
#
# The bug: upstream Qwen*VLModel / Qwen*OmniModel-style `forward` accepts
# `**kwargs` and forwards them as-is into `self.visual(..., **kwargs)`. When
# the outer caller has populated `FlashAttentionKwargs` (`cu_seq_lens_q/k`,
# `max_length_q/k`) for the LM packed-sequence path — which VeOmni's data
# pipeline always does — those LM-level keys ride through the ViT block
# chain and reach `*VisionAttention.forward`, which then does:
#
#     attention_interface(..., cu_seq_lens_q=cu_seqlens, ..., **kwargs)
#
# Same key on both sides → `TypeError: got multiple values for keyword
# argument 'cu_seq_lens_q'` at the Python call site, before any function body
# even runs.
#
# We do NOT hit this in production. VeOmni's own patched `Model.forward`
# (loaded via patchgen at `veomni/models/transformers/<m>/generated/`) builds
# a clean `image_vit_kwargs = {"vit_metadata": ...}` and never forwards the
# LM kwargs into the visual call — see e.g.
# `qwen2_vl_gpu_patch_gen_config.py:409` (`image_vit_kwargs`). So the
# `MODELING_BACKEND=veomni` mode in this test passes without any patching.
#
# But this test ALSO exercises `MODELING_BACKEND=hf` (raw upstream modeling)
# for HF↔VeOmni numerics parity, and that path imports the upstream modeling
# files directly — patchgen never runs. To keep the `hf` arm running on
# transformers 5.9, we wrap the upstream ViT classes' `.forward` here and pop
# the leaked keys at the ViT entrypoint. The wrap is scoped to this test
# module — the runtime VeOmni stack (`veomni/ops/...`) ships with no such
# monkey-patch, on purpose (see `docs/transformers_v5/patchgen.md`: VeOmni
# avoids runtime monkey-patching of upstream model code).
#
# Drop this shim when upstream stops leaking the LM-level FlashAttentionKwargs
# into the visual call (it would also need to filter them out of the
# `Model.forward(**kwargs)` → `self.visual(**kwargs)` chain).
# ────────────────────────────────────────────────────────────────────────────
_HF_VIT_FORWARDS_PATCHED = False
_LEAKED_LM_FLASH_KWARGS = (
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "max_length_q",
    "max_length_k",
)
# Both the visual and (for omni models) audio encoder forwards leak — the
# `forward` of each receives `**kwargs` from the outer `Model.forward` and
# threads them into the per-block attention call. Wrap both at the encoder
# entrypoint.
_HF_VIT_CLASSES_TO_STRIP: tuple[tuple[str, str], ...] = (
    ("transformers.models.qwen2_vl.modeling_qwen2_vl", "Qwen2VisionTransformerPretrainedModel"),
    ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VisionTransformerPretrainedModel"),
    ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLVisionModel"),
    ("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe", "Qwen3VLMoeVisionModel"),
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5VisionModel"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeVisionModel"),
    ("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni", "Qwen2_5OmniVisionEncoder"),
    ("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni", "Qwen2_5OmniAudioEncoder"),
    ("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe", "Qwen3OmniMoeVisionEncoder"),
    ("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe", "Qwen3OmniMoeAudioEncoder"),
)


def _patch_hf_vit_forwards_for_5_9_flash_kwargs_leak() -> None:
    """Wrap upstream qwen-family ViT ``forward`` to pop LM-level flash kwargs.

    See the module-level comment block above for the full rationale. Idempotent
    via ``_HF_VIT_FORWARDS_PATCHED`` and a per-method marker, and a no-op for
    VeOmni's patchgen-generated classes (they are different Python objects).
    """
    global _HF_VIT_FORWARDS_PATCHED
    if _HF_VIT_FORWARDS_PATCHED:
        return

    def _strip_leaked(forward):
        @functools.wraps(forward)
        def wrapped(*args, **kwargs):
            for key in _LEAKED_LM_FLASH_KWARGS:
                kwargs.pop(key, None)
            return forward(*args, **kwargs)

        wrapped._hf_vit_kwargs_strip = True
        return wrapped

    for module_path, class_name in _HF_VIT_CLASSES_TO_STRIP:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue
        cls = getattr(mod, class_name, None)
        if cls is None:
            continue
        if getattr(cls.forward, "_hf_vit_kwargs_strip", False):
            continue
        cls.forward = _strip_leaked(cls.forward)

    _HF_VIT_FORWARDS_PATCHED = True


# Apply at module import — test-scope, idempotent.
_patch_hf_vit_forwards_for_5_9_flash_kwargs_leak()


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
                assert obj.use_non_eager_impl, (
                    f"OpSlot {name} ({obj.op_name}/{obj.variant}) should have kernel when use_liger_kernel=True"
                )
            else:
                assert not obj.use_non_eager_impl, (
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
        # Aggressive teardown of any model / optimizer / lr_scheduler from the
        # previous mode iteration. Without this the prior FSDP-wrapped model
        # plus its optimizer states stay pinned across `_build_model` calls
        # (Python GC alone is not enough because FSDP / lazy-init hold cross
        # references), and on multi-mode runs over qwen3_5 we accumulate
        # 5+ GiB per mode, eventually OOM'ing on the embedding tensor for the
        # next model build (manifested as ``Process X has 43.80 GiB``).
        # ``hasattr`` returns True for class-level descriptors that ``delattr``
        # can't remove (e.g. inherited properties on BaseTrainer), so use the
        # instance ``__dict__`` directly.
        for _attr in ("model", "optimizer", "lr_scheduler"):
            self.__dict__.pop(_attr, None)
        _release_device_memory()

        set_environ_param(model_mode)
        _apply_patches()

        model_name = self.model_config.model_type
        from .utils import _build_ops_config_for_mode

        self.args.model.ops_implementation = _build_ops_config_for_mode(model_mode)

        if model_mode.use_liger_kernel:
            self.args.model.ops_implementation.rms_norm_implementation = "liger_kernel"
            self.args.model.ops_implementation.swiglu_mlp_implementation = "liger_kernel"
            self.args.model.ops_implementation.rotary_pos_emb_implementation = "liger_kernel"
            # qwen3_5 / qwen3_5_moe have a large vocab and the fused Liger
            # cross-entropy materializes the full [B, S, V] logits buffer
            # (~5 GiB on the toy config), which OOMs on shared L20 runners
            # where another job is still holding part of the card. Use
            # chunk_loss for those two models — it processes the vocab in
            # chunks so peak allocation stays modest; the other liger ops
            # (rms_norm / rotary / swiglu) are still exercised.
            if model_name in ("qwen3_5", "qwen3_5_moe"):
                self.args.model.ops_implementation.cross_entropy_loss_implementation = "chunk_loss"
            else:
                self.args.model.ops_implementation.cross_entropy_loss_implementation = "liger_kernel"
        else:
            self.args.model.ops_implementation.rms_norm_implementation = "eager"
            self.args.model.ops_implementation.swiglu_mlp_implementation = "eager"
            self.args.model.ops_implementation.rotary_pos_emb_implementation = "eager"
            self.args.model.ops_implementation.cross_entropy_loss_implementation = "eager"

        self._build_model()
        self._verify_opslot_state(model_mode)
        self._build_optimizer()
        self._build_lr_scheduler()
        print_device_mem_info(f"[Memory Info] after building model {model_name}:")

        # Sync weights — every model that test_models_patch covers ships a
        # patchgen layout that matches HF's in-memory state dict, so a
        # straight ``load_state_dict`` is sufficient. When loading from a real
        # on-disk HF safetensors checkpoint, the per-expert → fused merge
        # still happens, but at the runtime-converter layer (e.g.
        # ``DeepseekV3CheckpointTensorConverter``); that path is exercised by
        # ``test_logits_bitwise_equal_v5_via_loader`` in
        # ``test_models_logits_equal.py``.
        self.model.load_state_dict(state_dict)

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


# Test case: (config_path, is_moe, rtol, atol).
# rtol/atol: tolerances for compare_multi_items; can be set per case.
_DEFAULT_RTOL = 1e-2
_DEFAULT_ATOL = 1e-2

# Models without a patchgen path are not covered here. Migrate them via
# ``/veomni-migrate-transformers-v5`` to bring them back into this list.
TEST_CASES = [
    pytest.param(
        "./tests/toy_config/llama31_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="llama3_1",
    ),
    pytest.param(
        "./tests/toy_config/qwen3_5_toy/config.json",
        False,
        # qwen3_5* uses chunk_loss in liger mode (see forward_backward_step
        # — fused Liger CE OOMs on the qwen3_5 full vocab). chunk_loss is
        # bit-exact with eager but drifts ~3% from HF native CE in bfloat16
        # via the GatedDeltaNet linear-attention path. Bump tolerance so
        # the HF↔VeOmni gnorm comparison stays in band.
        0.05,
        0.05,
        id="qwen3_5",
    ),
    pytest.param(
        "./tests/toy_config/qwen3_5_moe_toy/config.json",
        True,
        # Same chunk_loss/HF drift as qwen3_5 above; MoE path adds
        # routing-loss noise on top so allow a slightly wider envelope.
        0.05,
        0.05,
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
    pytest.param(
        "./tests/toy_config/qwen25vl_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen2_5_vl",
    ),
    pytest.param(
        "./tests/toy_config/qwen3vl_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_vl",
    ),
    pytest.param(
        "./tests/toy_config/qwen3vlmoe_toy/config.json",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_vl_moe",
    ),
    pytest.param(
        "./tests/toy_config/qwen25omni_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen2_5_omni",
    ),
    pytest.param(
        "./tests/toy_config/qwen3omni_toy/config.json",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_omni_moe",
    ),
    pytest.param(
        "./tests/toy_config/seed_oss_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="seed_oss",
    ),
    pytest.param(
        "./tests/toy_config/deepseek_v3_toy/config.json",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="deepseek_v3",
    ),
]


@pytest.mark.parametrize("config_path, is_moe, rtol, atol", TEST_CASES)
def test_models_patch_fwd_bwd(
    request: pytest.FixtureRequest,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
):
    case_id = request.node.callspec.id
    hf_model_modes, veomni_model_modes = prepare_model_modes(is_moe=is_moe)

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
        if IS_NPU_AVAILABLE:
            # Qwen3.5 GatedDeltaNet has no NPU backend (no FLA / FlashQLA on
            # Ascend); the OpSlot bind for fla raises on NPU.
            return
        #    hf_model_modes = [mode for mode in hf_model_modes if mode.attn_case != "position_ids"]
        hf_model_modes = [mode for mode in hf_model_modes if mode.attn_implementation != "flash_attention_3"]
        veomni_model_modes = [
            mode for mode in veomni_model_modes if mode.attn_implementation != "veomni_flash_attention_3_with_sp"
        ]

    # The actual ops backend used per test case is set by ``set_environ_param``
    # inside ``TrainerTest`` (which calls ``apply_ops_config`` with a
    # mode-specific config). The ops_implementation on this ModelArguments is
    # never consumed at training time, so we pin it to all-eager — without
    # this the public ``OpsImplementationConfig()`` defaults (liger_kernel /
    # fused_triton / triton) would fail validation on NPU before the test
    # even runs.
    model_config = ModelArguments(config_path=config_path, ops_implementation=make_eager_ops_config())
    data_config = DataArguments(train_path="")
    training_config = TrainingArguments(
        checkpoint=CheckpointConfig(output_dir="./test_models_patch"),
        accelerator=AcceleratorConfig(
            fsdp_config=FSDPConfig(
                fsdp_mode="ddp",
                mixed_precision=MixedPrecisionConfig(enable=False),
            ),
        ),
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
