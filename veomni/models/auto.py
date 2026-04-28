# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import functools
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from ..arguments.arguments_types import OpsImplementationConfig
from ..distributed.parallel_state import get_parallel_state
from ..ops.dispatch import OpSlot
from ..utils import logging
from ..utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_gpu_compute_capability, is_torch_npu_available
from ..utils.import_utils import is_transformers_version_greater_or_equal_to
from .loader import BaseModelLoader, get_loader, get_model_class, get_model_config, get_model_processor


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class _ResolvedMoeImplementation:
    config_value: Literal["eager", "fused"]
    kernel_name: Optional[str]


def _select_moe_kernel_by_device(moe_implementation: str) -> _ResolvedMoeImplementation:
    """Select the concrete MoE kernel for the user-facing implementation value."""
    if moe_implementation == "eager":
        return _ResolvedMoeImplementation(
            config_value="eager",
            kernel_name=None,
        )

    if moe_implementation == "fused":
        if IS_NPU_AVAILABLE:
            kernel_name = "npu"
        elif IS_CUDA_AVAILABLE:
            kernel_name = "quack" if get_gpu_compute_capability() >= 100 else "triton"
        else:
            raise RuntimeError("moe_implementation='fused' requires a CUDA GPU or NPU device.")
    elif moe_implementation == "fused_triton":
        kernel_name = "triton"
    elif moe_implementation == "fused_quack":
        kernel_name = "quack"
    elif moe_implementation == "fused_npu":
        kernel_name = "npu"
    else:
        raise ValueError(
            f"Invalid moe_implementation: {moe_implementation!r}. "
            "Expected one of: 'eager', 'fused', 'fused_triton', 'fused_quack', 'fused_npu'."
        )

    return _ResolvedMoeImplementation(
        config_value="fused",
        kernel_name=kernel_name,
    )


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)


def build_processor(processor_path: str, **kwargs) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return get_model_processor(processor_path, padding_side="right", trust_remote_code=True, **kwargs)


def build_config(config_path: str, **config_kwargs) -> "PretrainedConfig":
    """
    Builds the model config.
    """
    trust_remote_code = config_kwargs.pop("trust_remote_code", True)
    return get_model_config(config_path, trust_remote_code=trust_remote_code, **config_kwargs)


def apply_moe_patch_transformers_v4(config, moe_implementation: str):
    """Legacy MoE dispatch path for models that don't use OpSlot.

    User-facing ``moe_implementation`` is a single field with values like
    ``"eager"``/``"fused"``/``"fused_triton"``/``"fused_quack"``. Legacy
    modeling code (qwen3_moe, deepseek_v3, etc.) only checks
    ``config._moe_implementation`` for the coarse ``"eager"``/``"fused"``
    distinction, so we split the user value into (mode, kernel) here and bind
    the fused kernel separately.
    """
    resolved = _select_moe_kernel_by_device(moe_implementation)
    if resolved.config_value == "eager":
        logger.warning_rank0("You are using eager moe implementation, expect this to be VERY SLOW!")
        config._moe_implementation = "eager"
        return

    fused_moe_kernel = resolved.kernel_name
    assert fused_moe_kernel is not None
    logger.info_rank0(f"MoE implementation: fused (kernel={fused_moe_kernel})")
    config._moe_implementation = "fused"

    from ..ops.kernels.moe import apply_veomni_fused_moe_patch

    apply_veomni_fused_moe_patch(fused_moe_kernel=fused_moe_kernel)


def _bind_veomni_ops(
    modeling_module,
    ops_config: OpsImplementationConfig,
    moe_implementation: Optional[str] = None,
) -> bool:
    """Bind all OpSlot instances found in *modeling_module*.

    Returns ``True`` if at least one OpSlot was found (and bound).
    """
    found = False
    resolved_moe: Optional[_ResolvedMoeImplementation] = None
    for name in dir(modeling_module):
        obj = getattr(modeling_module, name, None)
        if isinstance(obj, OpSlot):
            # `moe_experts` is the one op whose user-facing config field is not
            # `moe_experts_implementation` but `moe_implementation`. Resolve
            # auto/explicit user values into the registry's concrete kernel names.
            if obj.op_name == "moe_experts":
                if resolved_moe is None:
                    resolved_moe = _select_moe_kernel_by_device(moe_implementation or ops_config.moe_implementation)
                impl_name = resolved_moe.kernel_name or "eager"
            else:
                impl_name = getattr(ops_config, f"{obj.op_name}_implementation", "eager")
            obj.bind(impl_name)
            logger.info_rank0(f"OpSlot '{name}' bound to '{impl_name}' -> {obj}")
            found = True
    return found


def _module_has_opslot(modeling_module, op_name: str) -> bool:
    """Return True if *modeling_module* declares an OpSlot for *op_name*."""
    if modeling_module is None:
        return False
    for name in dir(modeling_module):
        obj = getattr(modeling_module, name, None)
        if isinstance(obj, OpSlot) and obj.op_name == op_name:
            return True
    return False


def _module_imports_global_fused_moe(modeling_module) -> bool:
    """Return True if *modeling_module* imports the global fused_moe_forward."""
    return modeling_module is not None and hasattr(modeling_module, "fused_moe_forward")


def _module_uses_opslot_moe_dispatch(modeling_module) -> bool:
    """Return True when MoE experts should be driven by an OpSlot."""
    return is_transformers_version_greater_or_equal_to("5.0.0") and _module_has_opslot(modeling_module, "moe_experts")


def build_foundation_model(
    config_path: Union[str, PretrainedConfig],
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[
        Literal[
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
            "veomni_flash_attention_2_with_sp",
            "veomni_flash_attention_3_with_sp",
            "veomni_flash_attention_4_with_sp",
            "native-sparse",
        ]
    ] = "veomni_flash_attention_2_with_sp",
    moe_implementation: Optional[str] = "fused",
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    encoder_data_balance: Optional[bool] = False,
    encoder_data_balance_sorting_algo: Optional[str] = "post_mbs_balancing_greedy_without_pad",
) -> "PreTrainedModel":
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.

    Contract: ``apply_ops_config(ops_implementation)`` must run before this
    function so that ``LOSS_MAPPING`` contains VeOmni's loss wrappers.
    ``BaseTrainer`` does this automatically; standalone scripts (tests, eval
    harnesses) that call ``build_foundation_model`` directly should call
    ``apply_ops_config`` themselves. If we detect that nobody did, we install
    defaults with a warning rather than letting the model fail later with a
    cryptic kwargs mismatch inside HuggingFace's stock loss wrapper.
    """
    from ..ops import apply_ops_config
    from ..ops.config.singleton import get_ops_config

    if get_ops_config() is None:
        logger.warning_rank0(
            "build_foundation_model was called before apply_ops_config. VeOmni "
            "assumes training goes through BaseTrainer, which installs the ops "
            "config for you. Installing OpsImplementationConfig() defaults now "
            "so self.loss_function() does not trip on missing kwargs. If you "
            "are running a standalone script, call apply_ops_config(...) "
            "yourself before build_foundation_model to pick kernel backends."
        )
        apply_ops_config(OpsImplementationConfig())

    if config_kwargs is None:
        config_kwargs = {}

    if isinstance(config_path, PretrainedConfig):
        config = config_path
    else:
        config = build_config(config_path, **config_kwargs)

    if encoder_data_balance:
        if config.model_type == "qwen3_vl_moe":
            if get_parallel_state().sp_enabled:
                logger.warning_rank0(
                    "Warning: Qwen3VLEncoderDataBalance currently does not support sequence parallelism. "
                    "The configuration of 'encoder_data_balance' is reset to False. "
                    "This issue will be addressed in a future release."
                )
                config.encoder_data_balance = False
            else:
                config.encoder_data_balance = encoder_data_balance
                config.encoder_data_balance_sorting_algo = encoder_data_balance_sorting_algo
        else:
            logger.warning_rank0(
                f"Encoder data balance currently supported only for Qwen3-VL MoE, "
                f"current model type: {config.model_type}, reset encoder_data_balance = False"
            )
            config.encoder_data_balance = False
    else:
        config.encoder_data_balance = False

    loader: Optional[BaseModelLoader] = get_loader(config)

    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
    }

    if attn_implementation == "flash_attention_4" and not is_transformers_version_greater_or_equal_to("5.0.0"):
        raise RuntimeError(
            f"attn_implementation '{attn_implementation}' bare name requires Transformers>=5.0.0. "
            'For Transformers v4, please use attn_implementation="veomni_flash_attention_4_with_sp".'
        )

    if attn_implementation not in (
        "veomni_flash_attention_2_with_sp",
        "veomni_flash_attention_3_with_sp",
        "veomni_flash_attention_4_with_sp",
    ):
        logger.warning_rank0(
            f"building foundation model with attn_implementation: {attn_implementation}.. you are missing sequence parallelism support. Please use veomni_flash_attention_2_with_sp or veomni_flash_attention_3_with_sp for SP."
        )

    if (init_device == "cpu" and get_parallel_state().global_rank != 0) or init_device == "meta":
        empty_init = True
    else:
        empty_init = False

    # ── Pre-load: legacy MoE config patch ─────────────────────────────────
    # Models like qwen3_moe capture ``config._moe_implementation`` inside
    # ``PatchQwen3MoeExperts.__init__``, so the legacy patch has to land on
    # *config* before ``loader.load_model`` instantiates the experts.
    # Transformers v4 monkey patches stay on the legacy path. Transformers v5
    # models use OpSlot only when the generated module declares a ``moe_experts``
    # slot; v5 models still using ``_moe_implementation`` remain on the legacy
    # global-pointer path.
    #
    # TODO(kernel-registry): migrate the remaining legacy ``_moe_implementation``
    # users to the OpSlot("moe_experts", …) + KERNEL_REGISTRY pattern that
    # qwen3_5_moe already uses; once they do, drop this whole pre-load block.
    # Current holdouts: qwen3_moe, qwen3_vl_moe, qwen3_omni_moe, deepseek_v3.
    effective_moe_implementation = moe_implementation or get_ops_config().moe_implementation
    pre_load_modeling_module = sys.modules.get(get_model_class(config).__module__)
    if _module_imports_global_fused_moe(pre_load_modeling_module):
        if _module_uses_opslot_moe_dispatch(pre_load_modeling_module):
            logger.info_rank0(
                f"Model has a 'moe_experts' OpSlot; using OpSlot MoE dispatch for {effective_moe_implementation!r}."
            )
        else:
            apply_moe_patch_transformers_v4(config, effective_moe_implementation)

    model = loader.load_model(
        init_kwargs=init_kwargs,
        weights_path=weights_path,
        empty_init=empty_init,
        init_device=init_device,
    )

    # ── Post-load: OpSlot binding ─────────────────────────────────────────
    # OpSlots are module-level singletons, so binding them after the model
    # is constructed is fine (and necessary — the patched modeling module is
    # only guaranteed to be importable once the loader has instantiated the
    # model class).
    modeling_module = sys.modules.get(model.__class__.__module__)
    if modeling_module is not None:
        if _bind_veomni_ops(modeling_module, get_ops_config(), moe_implementation=effective_moe_implementation):
            logger.info_rank0("OpSlot-based kernel dispatch active.")

    if is_torch_npu_available():
        # We override the forward method (on NPU devices) instead of passing CPU FA kwargs directly to the model in the trainer,
        # due to the behavior in https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_fully_shard/_fsdp_state.py#L130
        logger.info_rank0(
            "We override the model’s forward method on NPU devices to ensure that the FA kwargs are on CPU, since the npu_fused_attention requires cpu FA kwargs"
        )
        original_forward = model.forward

        @functools.wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            if "cu_seq_lens_q" in kwargs and kwargs["cu_seq_lens_q"] is not None:
                kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_q"].cpu()
            if "cu_seq_lens_k" in kwargs and kwargs["cu_seq_lens_k"] is not None:
                kwargs["cu_seq_lens_k"] = kwargs["cu_seq_lens_k"].cpu()
            return original_forward(*args, **kwargs)

        model.forward = wrapped_forward

    if is_transformers_version_greater_or_equal_to("5.0.0"):
        assert not getattr(model, "use_kernels", False), (
            "Still evaluating HF kernels hub integration with VeOmni patches; keep use_kernels disabled for now "
            "to avoid unexpected kernel loading side effects."
        )

    model_class_path = f"{model.__class__.__module__}.{model.__class__.__name__}"
    logger.info_rank0(f"Built foundation model class: {model_class_path}")

    return model
