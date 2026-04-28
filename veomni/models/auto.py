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
from ..utils.device import is_torch_npu_available
from ..utils.import_utils import is_transformers_version_greater_or_equal_to
from .loader import BaseModelLoader, get_loader, get_model_class, get_model_config, get_model_processor


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


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


_MOE_EXPERT_COUNT_FIELDS = (
    # Names used by HF MoE configs for the routed-experts dimension. Any one
    # of these being present (and > 0) means the model has MoE experts.
    "num_experts",
    "num_local_experts",
    "moe_num_experts",
    "n_routed_experts",
)


def _config_is_moe(config) -> bool:
    """Return True if *config* declares any MoE expert dimension.

    The legacy MoE patch (``apply_moe_patch_transformers_v4``) binds a fused
    MoE kernel via ``apply_veomni_fused_moe_patch``. With the GPU-reasonable
    ``moe_implementation`` default flipped from ``"eager"`` to
    ``"fused_triton"``, that patch would otherwise fire for *every* model
    (Llama / plain Qwen / etc.) and crash on hosts without triton (or NPU
    where ``fused_triton`` is GPU-only). We short-circuit it for non-MoE
    configs so dense models pay no cost from the new default.

    Walks the top-level config plus any ``text_config`` / ``language_config``
    sub-config (used by VLM wrappers like Qwen2-VL / Qwen3-VL) since those
    place the MoE fields on the language sub-config rather than the wrapper.
    """
    candidates = [config]
    for sub_attr in ("text_config", "language_config"):
        sub = getattr(config, sub_attr, None)
        if sub is not None:
            candidates.append(sub)
    for cand in candidates:
        for field_name in _MOE_EXPERT_COUNT_FIELDS:
            value = getattr(cand, field_name, None)
            if value is not None and value > 0:
                return True
    return False


def apply_moe_patch_transformers_v4(config, moe_implementation: str):
    """Legacy MoE dispatch path for models that don't use OpSlot.

    User-facing ``moe_implementation`` is a single field with values
    ``"eager"``/``"fused_triton"``/``"fused_quack"``. Legacy modeling code
    (qwen3_moe, deepseek_v3, etc.) only checks ``config._moe_implementation``
    for the coarse ``"eager"``/``"fused"`` distinction, so we split the user
    value into (mode, kernel) here and bind the fused kernel separately.
    """
    if moe_implementation == "eager":
        logger.warning_rank0("You are using eager moe implementation, expect this to be VERY SLOW!")
        config._moe_implementation = "eager"
        return

    if not moe_implementation.startswith("fused_"):
        raise ValueError(
            f"Invalid moe_implementation: {moe_implementation!r}. "
            "Expected 'eager' or a 'fused_<kernel>' name. OSS kernels: "
            "'fused_triton', 'fused_quack', 'fused_npu'. Third-party backends "
            "may register additional 'fused_<name>' values; if you set one, "
            "make sure the corresponding kernel is installed and registered."
        )

    fused_moe_kernel = moe_implementation.removeprefix("fused_")
    logger.info_rank0(f"MoE implementation: fused (kernel={fused_moe_kernel})")
    config._moe_implementation = "fused"

    from ..ops.kernels.moe import apply_veomni_fused_moe_patch

    apply_veomni_fused_moe_patch(fused_moe_kernel=fused_moe_kernel)


def _bind_veomni_ops(modeling_module, ops_config: OpsImplementationConfig) -> bool:
    """Bind all OpSlot instances found in *modeling_module*.

    Returns ``True`` if at least one OpSlot was found (and bound).
    """
    found = False
    for name in dir(modeling_module):
        obj = getattr(modeling_module, name, None)
        if isinstance(obj, OpSlot):
            # `moe_experts` is the one op whose user-facing config field is not
            # `moe_experts_implementation` but `moe_implementation`, and its
            # values carry a `fused_` prefix (e.g. `fused_triton`) that the
            # KERNEL_REGISTRY entries don't. Translate here so the registry
            # lookup finds the kernel and the HardwareRequirement check fires.
            if obj.op_name == "moe_experts":
                impl_name = (
                    "eager"
                    if ops_config.moe_implementation == "eager"
                    else ops_config.moe_implementation.removeprefix("fused_")
                )
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
    moe_implementation: Optional[str] = None,
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
            "config for you. Installing OpsImplementationConfig.eager_defaults() "
            "now so self.loss_function() does not trip on missing kwargs and "
            "the build does not require liger-kernel / triton / fused MoE. If "
            "you are running a standalone script, call apply_ops_config(...) "
            "yourself before build_foundation_model to pick kernel backends."
        )
        apply_ops_config(OpsImplementationConfig.eager_defaults())

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
    # We look up the modeling module via ``get_model_class`` (config → class)
    # so we can skip the legacy patch for models that own a ``moe_experts``
    # OpSlot (qwen3_5_moe and friends).
    #
    # TODO(kernel-registry): migrate the remaining legacy ``_moe_implementation``
    # users to the OpSlot("moe_experts", …) + KERNEL_REGISTRY pattern that
    # qwen3_5_moe already uses; once they do, drop this whole pre-load block.
    # Current holdouts: qwen3_moe, qwen3_vl_moe, qwen3_omni_moe, deepseek_v3.
    if moe_implementation is not None:
        pre_load_modeling_module = sys.modules.get(get_model_class(config).__module__)
        if _module_has_opslot(pre_load_modeling_module, "moe_experts"):
            logger.info_rank0(
                f"Model has a 'moe_experts' OpSlot; ignoring legacy moe_implementation={moe_implementation!r}."
            )
        elif not _config_is_moe(config):
            # Dense models (Llama, plain Qwen, etc.) don't have experts. With
            # the new ``moe_implementation`` default of ``"fused_triton"``,
            # falling through to the legacy patch would import / bind a fused
            # MoE kernel that the model never uses, and would outright crash on
            # hosts without triton (or on NPU where ``fused_triton`` is
            # GPU-only). Skip the patch so dense models pay zero cost from the
            # new default.
            logger.info_rank0(
                f"Config has no MoE expert dimension; skipping legacy MoE patch (moe_implementation={moe_implementation!r})."
            )
        else:
            apply_moe_patch_transformers_v4(config, moe_implementation)

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
        if _bind_veomni_ops(modeling_module, get_ops_config()):
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
