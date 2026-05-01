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
from .loader import BaseModelLoader, get_loader, get_model_config, get_model_processor


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


def _build_safe_fallback_ops_config() -> OpsImplementationConfig:
    """Conservative all-eager OpsImplementationConfig for the silent fallback path.

    Used by ``build_foundation_model`` when the caller passes neither
    ``ops_implementation`` nor a prior ``apply_ops_config`` call. The public
    ``OpsImplementationConfig()`` defaults are GPU-optimal (Liger / Triton /
    fused_triton) and raise on hosts without those packages, but standalone
    callers (inference scripts, weight-materialization helpers, dummy-forward
    tests) need a baseline that works everywhere — so we use ``"eager"`` for
    every per-op field. ``attn_implementation`` keeps the GPU-friendly
    ``"flash_attention_2"`` default; ``MODELING_BACKEND`` rewrites it to the
    SP-aware variant in ``__post_init__`` and ``flash_attention_2`` itself
    has graceful HF fallbacks (sdpa) when the package is missing.
    """
    return OpsImplementationConfig(
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
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


def _bind_veomni_ops(modeling_module, ops_config: OpsImplementationConfig) -> bool:
    """Bind all OpSlot instances found in *modeling_module*.

    Returns ``True`` if at least one OpSlot was found (and bound).
    """
    found = False
    moe_experts_kernel: Optional[str] = None
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
                if impl_name != "eager":
                    moe_experts_kernel = impl_name
            else:
                impl_name = getattr(ops_config, f"{obj.op_name}_implementation", "eager")
            obj.bind(impl_name)
            logger.info_rank0(f"OpSlot '{name}' bound to '{impl_name}' -> {obj}")
            found = True

    # The OpSlot only acts as an eager-vs-fused guard
    # (``slot.use_non_eager_impl``). Inside the fused branch, modeling code
    # calls ``veomni.ops.fused_moe_forward(...)``, which dispatches through
    # the module-level pointer ``veomni.ops.kernels.moe._fused_moe_forward``.
    # Bind that pointer here to the kernel matching the slot so the two stay
    # in sync. Eager bindings leave the pointer untouched.
    if moe_experts_kernel is not None:
        from ..ops.kernels.moe import apply_veomni_fused_moe_patch

        apply_veomni_fused_moe_patch(fused_moe_kernel=moe_experts_kernel)

    return found


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
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    encoder_data_balance: Optional[bool] = False,
    encoder_data_balance_sorting_algo: Optional[str] = "post_mbs_balancing_greedy_without_pad",
    ops_implementation: Optional[OpsImplementationConfig] = None,
) -> "PreTrainedModel":
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.

    Ops dispatch is owned by this function: when ``ops_implementation`` is
    provided we run ``apply_ops_config`` before constructing the model, and the
    resolved ``attn_implementation`` is read from it (the explicit
    ``attn_implementation`` kwarg is ignored in that case). Trainers always
    pass ``ops_implementation``; standalone scripts that omit it (e.g.
    ``tasks/infer/*``, ``tests/e2e/test_e2e_parallel._materialize_weights_dir``,
    ``tests/distributed/test_dummy_forward``) get an *all-eager* fallback
    config installed automatically — unless something earlier (e.g. a
    ``DiTTrainer`` building a condition model first) already installed one,
    in which case we leave it alone.

    The fallback is intentionally conservative (every per-op field set to
    ``"eager"``, no Liger / Triton / fused kernels) so it works on every
    accelerator without depending on optional packages. Production training
    paths route through trainers that pass the user's
    ``OpsImplementationConfig`` explicitly, which carries the GPU-optimal
    defaults.
    """
    from ..ops import apply_ops_config
    from ..ops.config.singleton import get_ops_config

    if ops_implementation is not None:
        apply_ops_config(ops_implementation)
        attn_implementation = ops_implementation.attn_implementation
    elif get_ops_config() is None:
        apply_ops_config(_build_safe_fallback_ops_config())

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

    # ── Pre-init: OpSlot binding ──────────────────────────────────────────
    # ``get_loader`` -> ``get_model_class`` -> ``MODELING_REGISTRY[...]()``
    # has already imported the patched modeling module, so ``loader.model_cls``
    # is in ``sys.modules`` and we can resolve OpSlot bindings *before*
    # the model is constructed. This matters for slots consumed inside
    # ``__init__`` (e.g. Qwen3.5's GatedDeltaNet picks between
    # ``Qwen3_5RMSNormGated`` and ``FusedRMSNormGated`` at init time based
    # on ``veomni_rms_norm_gated.use_non_eager_impl``); slots consumed only
    # in ``forward`` would also work post-init, but binding once, here, keeps
    # the timing uniform. Assumes ``loader.model_cls`` is final at this point —
    # i.e. no loader rewrites it between here and ``loader.load_model()`` below.
    model_cls = getattr(loader, "model_cls", None) if loader is not None else None
    modeling_module = sys.modules.get(model_cls.__module__) if model_cls is not None else None
    if modeling_module is not None:
        if _bind_veomni_ops(modeling_module, get_ops_config()):
            logger.info_rank0("OpSlot-based kernel dispatch active.")

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

    model = loader.load_model(
        init_kwargs=init_kwargs,
        weights_path=weights_path,
        empty_init=empty_init,
        init_device=init_device,
    )

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
