# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One training step of each toy model: HF modeling vs VeOmni modeling with its kernels.

Each case trains one step through ``BaseTrainer`` (MainCollator packing, varlen
FlashAttention kwargs, loss, backward, grad clipping) once per ``ModelMode`` from
the same initial weights, then compares loss and grad norm across modes. The HF
modes run upstream modeling with ``MODELING_BACKEND=hf``; the VeOmni modes run the
patchgen modeling with the device's kernels on or off and, for MoE models, the
device's fused experts.
"""

from __future__ import annotations

import functools
import importlib
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from rich.console import Console
from rich.table import Table
from transformers import set_seed

from tests.tools.ci_config import ci_config_path
from tests.tools.common_utils import print_device_mem_info
from tests.tools.launch_utils import find_free_port
from tests.tools.training_utils import make_eager_ops_config, release_device_memory
from veomni.arguments import (
    AcceleratorConfig,
    CheckpointConfig,
    DataArguments,
    FSDPConfig,
    MixedPrecisionConfig,
    ModelArguments,
    TrainingArguments,
)
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.data.data_collator import MainCollator
from veomni.data.dummy_dataset import build_dummy_dataset
from veomni.distributed.parallel_state import clear_parallel_state
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.ops import OP_REGISTRY, VeomniOp, apply_ops_patch
from veomni.trainer.base import BaseTrainer, VeOmniArguments
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, is_sm90_or_above
from veomni.utils.env import get_env
from veomni.utils.import_utils import is_liger_kernel_available, is_package_available
from veomni.utils.loss_utils import count_loss_token


pytestmark = pytest.mark.skipif(not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE), reason="needs a CUDA GPU or Ascend NPU")

_TOY_CONFIG_DIR = Path(__file__).resolve().parents[1] / "toy_config"


# ────────────────────────────────────────────────────────────────────────────
# Test-scope shim for a transformers upstream bug in qwen-family multimodal
# ``Model.forward(**kwargs)`` paths, needed by the ``MODELING_BACKEND=hf`` arm only.
#
# Upstream Qwen*VLModel / Qwen*OmniModel ``forward`` forwards ``**kwargs`` as-is
# into ``self.visual(..., **kwargs)``. When the caller populated the LM-level
# ``FlashAttentionKwargs`` (``cu_seq_lens_q/k``, ``max_length_q/k``) for packed
# sequences, those keys reach ``*VisionAttention.forward`` next to its own
# ``cu_seq_lens_q=cu_seqlens`` and raise ``TypeError: got multiple values``.
# VeOmni's patchgen ``Model.forward`` builds clean ViT kwargs, so the VeOmni arm
# needs no shim. Drop this once upstream stops leaking the LM kwargs.
# ────────────────────────────────────────────────────────────────────────────
_LEAKED_LM_FLASH_KWARGS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
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


@pytest.fixture(scope="module", autouse=True)
def _strip_leaked_lm_flash_kwargs_from_hf_encoders():
    def _strip_leaked(forward):
        @functools.wraps(forward)
        def wrapped(*args, **kwargs):
            for key in _LEAKED_LM_FLASH_KWARGS:
                kwargs.pop(key, None)
            return forward(*args, **kwargs)

        return wrapped

    originals = []
    for module_path, class_name in _HF_VIT_CLASSES_TO_STRIP:
        try:
            cls = getattr(importlib.import_module(module_path), class_name, None)
        except ImportError:
            continue
        if cls is None:
            continue
        originals.append((cls, cls.forward))
        cls.forward = _strip_leaked(cls.forward)
    yield
    for cls, forward in originals:
        cls.forward = forward


# --- Modes -------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMode:
    modeling_backend: str
    attn_implementation: str
    moe_implementation: str = "eager"
    use_kernels: bool = False

    def __str__(self) -> str:
        return (
            f"{self.modeling_backend}_[attn-{self.attn_implementation}]"
            f"_[moe-{self.moe_implementation}]_[kernels-{self.use_kernels}]"
        )


_KERNEL_FIELDS = (
    "rms_norm_implementation",
    "swiglu_mlp_implementation",
    "rotary_pos_emb_implementation",
    "rotary_pos_emb_vision_implementation",
    "cross_entropy_loss_implementation",
)
# NPU kernels are the torch_npu rows. On GPU they are Liger, which the NPU runner lacks.
_USE_KERNELS = [True, False] if IS_NPU_AVAILABLE or is_liger_kernel_available() else [False]
# The Triton load-balancing-loss row is CUDA-only.
_LOAD_BALANCING_LOSS_IMPL = "triton" if IS_CUDA_AVAILABLE and is_package_available("triton") else "eager"
_FUSED_MOE_IMPL = "fused_npu" if IS_NPU_AVAILABLE else "fused_triton"

_HF_ATTN = ("flash_attention_2", "flash_attention_3")
_VEOMNI_ATTN = ("veomni_flash_attention_2", "veomni_flash_attention_3")


def _kernel_ops(model_type: str) -> dict[str, str]:
    """Config overrides for ``model_type`` when ``use_kernels=True`` (unset fields use ``OP_REGISTRY.default_impl``)."""
    if not IS_NPU_AVAILABLE and model_type in ("qwen3_5", "qwen3_5_moe"):
        # Liger CE materializes the full-vocab logits of these toys (~5 GiB);
        # chunk_loss keeps the step within shared CI cards.
        return {"cross_entropy_loss_implementation": "chunk_loss"}
    return {}


def _ops_config_for_mode(mode: ModelMode, model_type: str) -> OpsImplementationConfig:
    """Build the mode's ops config. Call after ``MODELING_BACKEND`` is set for the mode."""
    kernel_ops = _kernel_ops(model_type) if mode.use_kernels else dict.fromkeys(_KERNEL_FIELDS, "eager")
    return OpsImplementationConfig(
        attn_implementation=mode.attn_implementation,
        moe_implementation=mode.moe_implementation,
        load_balancing_loss_implementation=_LOAD_BALANCING_LOSS_IMPL,
        **kernel_ops,
    )


def _skip_fa3(attn_impl: str) -> bool:
    """FA3 needs Hopper; NPU has no FA3 port."""
    if attn_impl not in ("flash_attention_3", "veomni_flash_attention_3"):
        return False
    return IS_NPU_AVAILABLE or not is_sm90_or_above()


def prepare_model_modes(is_moe: bool) -> tuple[list[ModelMode], list[ModelMode]]:
    hf_modes = [ModelMode("hf", attn) for attn in _HF_ATTN if not _skip_fa3(attn)]
    moe_impls = ("eager", _FUSED_MOE_IMPL) if is_moe else ("eager",)
    veomni_modes = [
        ModelMode("veomni", attn, moe_implementation=moe_impl, use_kernels=use_kernels)
        for moe_impl in moe_impls
        for attn in _VEOMNI_ATTN
        if not _skip_fa3(attn)
        for use_kernels in _USE_KERNELS
    ]
    return hf_modes, veomni_modes


# --- Selected-op checks ------------------------------------------------------------

_KERNEL_OP_FIELDS = {
    "rms_norm": ("rms_norm_implementation",),
    "swiglu_mlp": ("swiglu_mlp_implementation",),
    "rope": ("rotary_pos_emb_implementation", "rotary_pos_emb_vision_implementation"),
    "cross_entropy_loss": ("cross_entropy_loss_implementation",),
}


def _op_handles(model: torch.nn.Module) -> dict[str, list[VeomniOp]]:
    handles: dict[str, list[VeomniOp]] = defaultdict(list)
    for module in model.modules():
        for value in vars(module).values():
            if isinstance(value, VeomniOp):
                handles[value.op].append(value)
    return handles


def _verify_selected_ops(model: torch.nn.Module, mode: ModelMode, config: OpsImplementationConfig) -> None:
    """Fail when the built model does not run the kernels this mode is meant to compare."""
    handles = _op_handles(model)
    if mode.modeling_backend == "hf":
        assert not handles, f"[{mode}] HF modeling must not hold VeOmni ops, found {sorted(handles)}"
        return

    for op, field_names in _KERNEL_OP_FIELDS.items():
        if not handles[op]:
            continue
        impls = sorted({handle.impl for handle in handles[op]})
        if not mode.use_kernels:
            assert impls == ["eager"], f"[{mode}] {op} should be eager with kernels off, got {impls}"
            continue
        configured = [getattr(config, name) for name in field_names if getattr(config, name) is not None]
        for handle in handles[op]:
            expected = set(configured) if configured else {OP_REGISTRY.default_impl(op, handle.variant)}
            assert handle.impl in expected, (
                f"[{mode}] {op}/{handle.variant} should use {sorted(expected)}, got {handle.impl!r}"
            )

    moe_impls = sorted({handle.impl for handle in handles["moe_experts"]})
    if handles["moe_experts"]:
        assert moe_impls == [mode.moe_implementation], (
            f"[{mode}] moe_experts should use {mode.moe_implementation!r}, got {moe_impls}"
        )
    attn_impls = {handle.impl for handle in handles["attention"]}
    assert mode.attn_implementation in attn_impls, (
        f"[{mode}] no attention handle uses {mode.attn_implementation!r}, got {sorted(attn_impls)}"
    )


# --- Data and comparison -----------------------------------------------------------

MODEL_TO_DATASET = {
    "qwen3_vl": "qwen3vl",
    "qwen3_5": "qwen3vl",
    "qwen3_5_moe": "qwen3vl",
    "qwen3_vl_moe": "qwen3vl",
    "qwen2_vl": "qwen2vl",
    "qwen2_5_vl": "qwen2vl",
    "qwen2_5_omni": "qwen2omni",
    "qwen3_omni_moe": "qwen3omni",
}


def _token_ids(model_config) -> dict[str, int]:
    if model_config.model_type not in MODEL_TO_DATASET:
        return {}
    if model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
        thinker = model_config.thinker_config
        return {
            "image_token_id": thinker.image_token_id,
            "video_token_id": thinker.video_token_id,
            "audio_token_id": thinker.audio_token_id,
        }
    return {"image_token_id": model_config.image_token_id, "video_token_id": model_config.video_token_id}


def _to_feature_dict(example: list, token_ids: dict[str, int]) -> dict[str, torch.Tensor]:
    """Turn a dummy dataset item into a MainCollator feature with the model's multimodal token ids."""
    example = {
        key: (value.clone() if isinstance(value, torch.Tensor) else torch.tensor(value))
        for key, value in example[0].items()
    }
    for mask_key, token_key in (
        ("image_mask", "image_token_id"),
        ("video_mask", "video_token_id"),
        ("audio_mask", "audio_token_id"),
    ):
        if mask_key in example and token_key in token_ids:
            example["input_ids"] = example["input_ids"].masked_fill(example[mask_key], token_ids[token_key])
    return example


def prepare_data(case_id: str, max_seq_len: int, model_config, num_samples: int) -> list[dict[str, torch.Tensor]]:
    dataset = build_dummy_dataset(MODEL_TO_DATASET.get(case_id, "text"), num_samples, max_seq_len)
    token_ids = _token_ids(model_config)
    set_seed(42)
    return [_to_feature_dict(dataset[index], token_ids) for index in range(num_samples)]


def print_all_values(outputs: dict[ModelMode, dict[str, Any]], value_key: str, model_type: str = "") -> None:
    table = Table(title=f"Alignment Result: [bold magenta]{model_type} {value_key}[/bold magenta]")
    mode_fields = [f.name for f in fields(ModelMode)]
    for field_name in mode_fields:
        table.add_column(field_name, style="cyan", justify="left")
    table.add_column(value_key.upper(), style="bold green", justify="right")
    for mode, output in outputs.items():
        value = output.get(value_key, "N/A")
        value = value.item() if hasattr(value, "item") else value
        table.add_row(*(str(asdict(mode)[name]) for name in mode_fields), f"{value:.8f}")
    Console().print(table)


def compare_multi_items(outputs: dict[ModelMode, dict[str, Any]], rtol: float, atol: float) -> None:
    base_mode = next(iter(outputs))
    base_output = outputs[base_mode]
    for mode, output in outputs.items():
        if mode == base_mode:
            continue
        for key in output:
            try:
                torch.testing.assert_close(output[key], base_output[key], rtol=rtol, atol=atol)
            except AssertionError as e:
                print_all_values(outputs, key)
                raise AssertionError(f"{key} of {mode} does not match {base_mode}") from e


# --- Trainer -----------------------------------------------------------------------


def _set_modeling_backend(mode: ModelMode) -> None:
    os.environ["MODELING_BACKEND"] = mode.modeling_backend
    apply_ops_patch()


class _NoParallelRuntime(VeOmniModelRuntime):
    """Kernel alignment compares the unwrapped module."""

    def _build_parallelized_model(self) -> None:
        pass


@pytest.fixture
def single_process_trainer_env(monkeypatch: pytest.MonkeyPatch):
    """One-rank launch env for ``BaseTrainer``, torn down so later ``tests/models`` cases see none of it."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(find_free_port()))
    monkeypatch.setenv("MODELING_BACKEND", get_env("MODELING_BACKEND"))
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        clear_parallel_state()


class TrainerTest(BaseTrainer):
    def __init__(self, first_mode: ModelMode, trainer_config: VeOmniArguments):
        _set_modeling_backend(first_mode)
        super().__init__(trainer_config)

    def _build_model_runtime(self, model_name: str = "base") -> VeOmniModelRuntime:
        return _NoParallelRuntime(self.args.model, model_name=model_name, train=self.args.train)

    def _init_callbacks(self, trainer=None):
        pass

    def _build_data_transform(self):
        pass

    def _build_dataset(self):
        self.args.compute_train_steps(100)
        self.train_steps = self.args.train_steps

    def _build_collate_fn(self):
        data_collate_info = {}
        if self.model.model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            data_collate_info = {
                "audio_feature_lengths": (0, False, None, None),
                "input_features": (0, True, 0, 1),
                "audio_mask": (-1, False, 0, 1),
                # Pack along dim 0 to match input_features: (2, 400) + (2, 400) -> (4, 400).
                # HF expects feature_attention_mask (num_audios, seq_len) for indexing.
                "feature_attention_mask": (0, False, None, None),
            }
        self.collate_fn = MainCollator(seq_classification=False, data_collate_info=data_collate_info)

    def _build_dataloader(self):
        pass

    def _release_model(self) -> None:
        # Drop the module, optimizer and scheduler of the previous mode; the runtime
        # handle is reused, and its recorded ``model_config`` names the next build.
        self.model.model = None
        self.model.optimizer = None
        self.model.lr_scheduler = None
        release_device_memory()

    def forward_backward_step(
        self, state_dict: dict[str, torch.Tensor], mode: ModelMode, features: list[dict[str, torch.Tensor]]
    ) -> dict[str, float]:
        self._release_model()
        _set_modeling_backend(mode)

        model_type = self.model.model_config.model_type
        ops_config = _ops_config_for_mode(mode, model_type)
        self.args.model.ops_implementation = ops_config
        self.model._build_model()
        _verify_selected_ops(self.model.model, mode, ops_config)
        self.model._build_optimizer()
        self._build_lr_scheduler()
        print_device_mem_info(f"[Memory Info] after building model {model_type}:")

        self.model.load_state_dict(state_dict)
        if model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.model.disable_talker()
            self.model.model = self.model.thinker

        print(f"{'-' * 10} {model_type}_{mode} {'-' * 10}")
        batch = self.collate_fn([{key: value.clone() for key, value in feature.items()} for feature in features])
        self.micro_batches_token_len = count_loss_token(batch)
        self.micro_batch_token_len = count_loss_token(batch)

        modeling_backend = get_env("MODELING_BACKEND")
        if model_type in ("qwen2_5_omni", "qwen3_omni_moe") and modeling_backend == "hf":
            # Upstream omni audio_forward expects (num_audios, mel, frames).
            audio_feature_lengths = batch["audio_feature_lengths"]
            batch["input_features"] = (
                batch["input_features"]
                .reshape(len(audio_feature_lengths), audio_feature_lengths.max(), -1)
                .permute(0, 2, 1)
                .to(dtype=self.model.dtype)
            )
        elif model_type == "qwen3_omni_moe" and modeling_backend == "veomni":
            batch["input_features"] = batch["input_features"].to(dtype=self.model.dtype)

        if batch["position_ids"].dim() == 3 and batch["position_ids"].shape[1] == 3:
            batch["position_ids"] = batch["position_ids"].transpose(0, 1).contiguous()

        _, loss_dict, _ = super().forward_backward_step(batch)
        grad_norm = self.model.clip_grad_norm()

        release_device_memory()
        print_device_mem_info(f"[Memory Info] after model {model_type} train_one_step:")

        result_metrics = {key: value.item() for key, value in loss_dict.items()}
        result_metrics["gnorm"] = grad_norm
        return result_metrics


# --- Cases -------------------------------------------------------------------------

_DEFAULT_RTOL = 1e-2
_DEFAULT_ATOL = 1e-2

# (toy config dir, is_moe, rtol, atol). Models without a patchgen path are not covered.
TEST_CASES = [
    pytest.param("llama31_toy", False, _DEFAULT_RTOL, _DEFAULT_ATOL, id="llama3_1"),
    # chunk_loss is bit-exact with eager but drifts ~3% from HF native CE in bf16
    # through the GatedDeltaNet path; the MoE variant adds routing-loss noise.
    pytest.param("qwen3_5_toy", False, 0.05, 0.05, id="qwen3_5"),
    pytest.param("qwen3_5_moe_toy", True, 0.05, 0.05, id="qwen3_5_moe"),
    pytest.param("qwen2vl_toy", False, _DEFAULT_RTOL, _DEFAULT_ATOL, id="qwen2_vl"),
    pytest.param("qwen2_toy", False, _DEFAULT_RTOL, _DEFAULT_ATOL, id="qwen2"),
    pytest.param("qwen25vl_toy", False, _DEFAULT_RTOL, _DEFAULT_ATOL, id="qwen2_5_vl"),
    pytest.param("qwen3vl_toy", False, _DEFAULT_RTOL, _DEFAULT_ATOL, id="qwen3_vl"),
    pytest.param("qwen3vlmoe_toy", True, _DEFAULT_RTOL, _DEFAULT_ATOL, id="qwen3_vl_moe"),
    pytest.param("qwen25omni_toy", False, _DEFAULT_RTOL, _DEFAULT_ATOL, id="qwen2_5_omni"),
    pytest.param("qwen3omni_toy", True, _DEFAULT_RTOL, _DEFAULT_ATOL, id="qwen3_omni_moe"),
    pytest.param("seed_oss_toy", False, _DEFAULT_RTOL, _DEFAULT_ATOL, id="seed_oss"),
    pytest.param("deepseek_v3_toy", True, _DEFAULT_RTOL, _DEFAULT_ATOL, id="deepseek_v3"),
    pytest.param("glm_moe_dsa_toy", True, _DEFAULT_RTOL, _DEFAULT_ATOL, id="glm_moe_dsa"),
    pytest.param("deepseek_v4_toy", True, _DEFAULT_RTOL, _DEFAULT_ATOL, id="deepseek_v4"),
]


def _case_modes(case_id: str, is_moe: bool) -> tuple[list[ModelMode], list[ModelMode]]:
    hf_modes, veomni_modes = prepare_model_modes(is_moe=is_moe)

    if case_id == "qwen2_5_omni":
        if IS_NPU_AVAILABLE:
            pytest.skip("torch.kaiser_window in Token2WavBigVGANModel init is unsupported on NPU")
        # Upstream qwen2_5_omni fails under FA3.
        hf_modes = [mode for mode in hf_modes if mode.attn_implementation != "flash_attention_3"]

    if case_id == "deepseek_v4":
        # DeepSeek-V4 attention is eager-only (head_dim=512 exceeds FA's cap, SDPA has
        # no per-head sink, Flex cannot resize the BlockMask after the compressor
        # concat). Keep attention eager and exercise the clamp-aware fused experts,
        # which receive ``swiglu_limit``. On NPU the kernel mode also covers the
        # torch_npu DeepSeek-V4 RMSNorm.
        hf_modes = [ModelMode("hf", "eager")]
        veomni_modes = [
            ModelMode("veomni", "eager", moe_implementation=_FUSED_MOE_IMPL, use_kernels=use_kernels)
            for use_kernels in (_USE_KERNELS if IS_NPU_AVAILABLE else [False])
        ]

    if case_id == "glm_moe_dsa":
        # Under the default ``dsa_attention_implementation=eager``, GLM-MoE-DSA folds
        # its DSA top-k selection into the attention mask, which only ``eager`` /
        # ``sdpa`` support; flash attention would discard it and raises. Compare HF
        # eager against VeOmni's eager expert loop and its fused MoE path, the one
        # expert parallelism runs on.
        hf_modes = [ModelMode("hf", "eager")]
        veomni_modes = [
            ModelMode("veomni", "eager"),
            ModelMode("veomni", "eager", moe_implementation=_FUSED_MOE_IMPL),
        ]

    if case_id in ("qwen3_5", "qwen3_5_moe"):
        if IS_NPU_AVAILABLE:
            pytest.skip("Qwen3.5 GatedDeltaNet keeps the fla default, which this case does not run on NPU")
        hf_modes = [mode for mode in hf_modes if mode.attn_implementation != "flash_attention_3"]
        veomni_modes = [mode for mode in veomni_modes if mode.attn_implementation != "veomni_flash_attention_3"]

    return hf_modes, veomni_modes


@pytest.mark.usefixtures("single_process_trainer_env")
@pytest.mark.parametrize("toy_config, is_moe, rtol, atol", TEST_CASES)
def test_models_patch_fwd_bwd(
    request: pytest.FixtureRequest, tmp_path: Path, toy_config: str, is_moe: bool, rtol: float, atol: float
):
    case_id = request.node.callspec.id
    hf_modes, veomni_modes = _case_modes(case_id, is_moe)

    # The ops config consumed per mode is rebuilt in ``forward_backward_step``;
    # this one only has to validate on every device for the first build.
    model_config = ModelArguments(
        config_path=ci_config_path(_TOY_CONFIG_DIR / toy_config / "config.json"),
        ops_implementation=make_eager_ops_config(),
        accelerator=AcceleratorConfig(
            init_device=get_device_type(),
            fsdp_config=FSDPConfig(fsdp_mode="ddp", mixed_precision=MixedPrecisionConfig(enable=False)),
        ),
    )
    trainer_config = VeOmniArguments(
        model=model_config,
        data=DataArguments(train_path=""),
        train=TrainingArguments(
            checkpoint=CheckpointConfig(output_dir=str(tmp_path / "test_models_patch")),
            enable_full_determinism=True,
        ),
    )
    trainer = TrainerTest(hf_modes[0], trainer_config)

    # Keep the shared initial weights off the accelerator while each mode holds its own model.
    state_dict = {key: value.detach().to("cpu", copy=True) for key, value in trainer.model.state_dict().items()}
    built_config = trainer.model.model_config
    trainer._release_model()

    # Upstream DeepSeek-V4 eager attention ignores packed cu-seqlens, so two packed
    # samples would compare different attention semantics; keep V4 to one sequence.
    num_samples = 1 if case_id == "deepseek_v4" else 2
    features = prepare_data(case_id, max_seq_len=1024, model_config=built_config, num_samples=num_samples)

    results: dict[ModelMode, dict[str, Any]] = {}
    for mode in [*hf_modes, *veomni_modes]:
        metrics = trainer.forward_backward_step(state_dict, mode, features)
        if results:
            assert set(metrics) == set(next(iter(results.values()))), f"[{mode}] reported different metrics"
        results[mode] = metrics

    for key in next(iter(results.values())):
        print_all_values(results, key, case_id)
    compare_multi_items(results, rtol=rtol, atol=atol)

    trainer._release_model()
    print_device_mem_info("[Memory Info] after running train_compare_models:")
