import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable, Dict, Optional

import torch
from rich.console import Console
from rich.table import Table
from transformers import set_seed

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.data.dummy_dataset import build_dummy_dataset
from veomni.ops import apply_ops_config
from veomni.utils.import_utils import is_liger_kernel_available, is_package_available, is_torch_npu_available


_LIGER_KERNEL = "liger_kernel"
_EAGER = "eager"
# Only include the Liger-backed mode when the package is actually installed
# (it's unavailable on the NPU CI runner); otherwise every parametrized case
# would fail in OpsImplementationConfig.__post_init__ with a ValueError.
_USE_LIGER_KERNEL = [True, False] if is_liger_kernel_available() else [False]
# NPU ships triton-ascend (not mainline ``triton``); the fused Triton
# load-balancing-loss kernel at ``veomni/ops/kernels/load_balancing_loss/triton.py``
# imports ``triton`` unconditionally, so fall back to the pure-PyTorch backend
# when the mainline package is missing.
_LOAD_BALANCING_LOSS_IMPL = "triton" if is_package_available("triton") else "eager"


@dataclass(frozen=True)
class ModelMode:
    modeling_backend: str
    attn_implementation: str
    sync_weight_func: Optional[Callable] = None
    moe_implementation: str = "eager"
    use_liger_kernel: bool = False

    def __str__(self):
        return (
            f"{self.modeling_backend}_[attn-{self.attn_implementation}]"
            f"_[moe-{self.moe_implementation}]_[ligerkernel-{self.use_liger_kernel}]"
        )


# HF uses _HF_ATTN, VeOmni uses _VEOMNI_ATTN x _USE_LIGER_KERNEL.
# On NPU skip FA3.
_HF_ATTN = ["flash_attention_2", "flash_attention_3"]
_VEOMNI_ATTN = [
    "veomni_flash_attention_2_with_sp",
    "veomni_flash_attention_3_with_sp",
]


def _skip_fa3_npu(attn_impl: str) -> bool:
    """Skip FA3 on NPU."""
    if not is_torch_npu_available():
        return False
    return attn_impl in ("flash_attention_3", "veomni_flash_attention_3_with_sp")


def _append_veomni_modes(modes: list, moe_implementation: str = "eager"):
    """Append VeOmni modes for case; every attn uses _USE_LIGER_KERNEL (True/False)."""
    for veomni_attn in _VEOMNI_ATTN:
        if _skip_fa3_npu(veomni_attn):
            continue
        for use_liger in _USE_LIGER_KERNEL:
            modes.append(
                ModelMode(
                    "veomni",
                    veomni_attn,
                    moe_implementation=moe_implementation,
                    use_liger_kernel=use_liger,
                )
            )


def _base_model_modes():
    """Base (non-MoE) model modes; all use sync_weight_func=None by default."""
    modes = []
    for hf_attn in _HF_ATTN:
        if _skip_fa3_npu(hf_attn):
            continue
        modes.append(ModelMode("hf", hf_attn))
    _append_veomni_modes(modes)
    return modes


def _moe_model_modes():
    """MoE model modes: same attn variants with moe_implementation=fused."""
    modes = []
    for hf_attn in _HF_ATTN:
        if _skip_fa3_npu(hf_attn):
            continue
    _append_veomni_modes(modes, moe_implementation="fused")
    return modes


def prepare_model_modes(
    is_moe: bool = False,
    sync_weight_func: Optional[Callable] = None,
):
    """
    Build model modes for patch tests.

    Args:
        is_moe: If True, include MoE-specific modes (e.g. fused MoE).
        sync_weight_func: Optional callable(config, state_dict, model) used only for
            VeOmni backend modes when HF/VeOmni state dict layouts differ. Will be
            removed in a future version when layouts align; pass None for normal models.
    """
    base_modes = _base_model_modes()
    moe_modes = _moe_model_modes()
    final_models_modes = base_modes + moe_modes if is_moe else base_modes

    if sync_weight_func is not None:
        final_models_modes = [
            replace(mode, sync_weight_func=sync_weight_func) if mode.modeling_backend == "veomni" else mode
            for mode in final_models_modes
        ]

    hf_model_modes = [m for m in final_models_modes if m.modeling_backend == "hf"]
    veomni_model_modes = [m for m in final_models_modes if m.modeling_backend == "veomni"]
    return hf_model_modes, veomni_model_modes


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

UNSQUEECE_KEYS = ["input_ids", "attention_mask", "labels", "position_ids", "image_mask", "video_mask", "audio_mask"]


def parse_token_id_from_config(model_config):
    if model_config.model_type not in MODEL_TO_DATASET:
        return {}
    if model_config.model_type in ["qwen2_5_omni", "qwen3_omni_moe"]:
        token_ids_dict = {
            "image_token_id": model_config.thinker_config.image_token_id,
            "video_token_id": model_config.thinker_config.video_token_id,
            "audio_token_id": model_config.thinker_config.audio_token_id,
        }
    else:
        token_ids_dict = {
            "image_token_id": model_config.image_token_id,
            "video_token_id": model_config.video_token_id,
        }
    return token_ids_dict


def _to_feature_dict(example: list, token_ids_dict: dict) -> dict:
    """Convert dataset item to feature dict for MainCollator (no batch dim, no precomputed cu_seqlens)."""
    example = example[0]
    example = {key: (v.clone() if isinstance(v, torch.Tensor) else torch.tensor(v)) for key, v in example.items()}

    if "image_mask" in example and "image_token_id" in token_ids_dict:
        example["input_ids"] = example["input_ids"].masked_fill(
            example["image_mask"], token_ids_dict["image_token_id"]
        )
    if "video_mask" in example and "video_token_id" in token_ids_dict:
        example["input_ids"] = example["input_ids"].masked_fill(
            example["video_mask"], token_ids_dict["video_token_id"]
        )
    if "audio_mask" in example and "audio_token_id" in token_ids_dict:
        example["input_ids"] = example["input_ids"].masked_fill(
            example["audio_mask"], token_ids_dict["audio_token_id"]
        )

    return example


def prepare_data(model_name: str, max_seq_len: int, model_config):
    """
    Build a DummyDataLoader that yields raw features (list of 2 dicts).
    MainCollator packing + cu_seqlens precompute is done in the trainer (TrainerTest).
    """
    from veomni.data.dummy_dataset import build_dummy_dataset

    dataset_name = MODEL_TO_DATASET.get(model_name, "text")
    dataset = build_dummy_dataset(dataset_name, 2, max_seq_len)

    token_ids_dict = parse_token_id_from_config(model_config)

    class DummyDataLoader:
        def __iter__(self):
            set_seed(42)
            features = [
                _to_feature_dict(dataset[0], token_ids_dict),
                _to_feature_dict(dataset[1], token_ids_dict),
            ]
            yield features

    return DummyDataLoader()


def print_all_values(output_dict, value_key: str, model_type: str = ""):
    console = Console()
    first_mode = next(iter(output_dict.keys()))

    table = Table(title=f"Alignment Result: [bold magenta]{model_type} {value_key}[/bold magenta]")
    mode_fields = [f.name for f in fields(first_mode) if f.name != "sync_weight_func"]

    for field in mode_fields:
        table.add_column(field, style="cyan", justify="left")

    table.add_column(value_key.upper(), style="bold green", justify="right")

    for mode, output in output_dict.items():
        mode_data = asdict(mode)
        row_cells = []

        for field in mode_fields:
            row_cells.append(str(mode_data[field]))

        val_obj = output.get(value_key, "N/A")
        val_str = f"{val_obj.item() if hasattr(val_obj, 'item') else val_obj:.8f}"
        row_cells.append(val_str)

        table.add_row(*row_cells)

    console.print(table)


def compare_multi_items(outputs_dict: Dict, rtol=0.01, atol=0.01):
    base_task = next(iter(outputs_dict))
    base_output = outputs_dict[base_task]

    for task, output in outputs_dict.items():
        if task == base_task:
            continue
        for key in output.keys():
            try:
                torch.testing.assert_close(
                    output[key],
                    base_output[key],
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError as e:
                print_all_values(outputs_dict, key)
                raise AssertionError(f"{key} not match") from e


# ---------------------------------------------------------------------------
# Pristine snapshots of HF model classes taken at module import — before any
# veomni import has a chance to monkey-patch them. `apply_veomni_hf_unpatch`
# restores them so a second test in the same process can build an HF model
# that is genuinely un-patched. Keep the module imports inside this block
# so that util users who never need them are not forced to pay the cost.
# ---------------------------------------------------------------------------
import transformers.models.deepseek_v3.modeling_deepseek_v3 as _hf_ds3  # noqa: E402
import transformers.models.qwen3.modeling_qwen3 as _hf_qwen3  # noqa: E402
import transformers.models.qwen3_moe.modeling_qwen3_moe as _hf_qwen3_moe  # noqa: E402


# Cover every patch site reachable from apply_veomni_*_patch() + the Liger
# and Triton branches of apply_veomni_*_gpu_patch(). We capture:
# - forward methods (undo in-class forward replacement, used by both the
#   always-on patch and the Triton branch's _patch_rms_norm / RoPE forward)
# - whole classes (undo module-level class swap done by the Liger branch
#   and by DeepseekV3MoE / Qwen3MoeSparseMoeBlock replacements)
# - module-level functions (apply_rotary_pos_emb is replaced by Liger)
#
# Restore order matters: we first reset in-class attributes on the pristine
# class objects, then restore the module-level names. That way, even if a
# prior test mutated `Class.forward` and then swapped in a Liger class at
# the module level, both layers are reverted.
_PRISTINE_HF = {
    # qwen3
    "qwen3.CausalLM.forward": _hf_qwen3.Qwen3ForCausalLM.forward,
    "qwen3.SeqCls.forward": _hf_qwen3.Qwen3ForSequenceClassification.forward,
    "qwen3.apply_rotary_pos_emb": _hf_qwen3.apply_rotary_pos_emb,
    "qwen3.RMSNorm.cls": _hf_qwen3.Qwen3RMSNorm,
    "qwen3.RMSNorm.forward": _hf_qwen3.Qwen3RMSNorm.forward,
    "qwen3.MLP.cls": _hf_qwen3.Qwen3MLP,
    # qwen3_moe
    "qwen3_moe.CausalLM.forward": _hf_qwen3_moe.Qwen3MoeForCausalLM.forward,
    "qwen3_moe.MoeBlock.cls": _hf_qwen3_moe.Qwen3MoeSparseMoeBlock,
    "qwen3_moe.PreTrained.init_weights": _hf_qwen3_moe.Qwen3MoePreTrainedModel._init_weights,
    "qwen3_moe.apply_rotary_pos_emb": _hf_qwen3_moe.apply_rotary_pos_emb,
    "qwen3_moe.RMSNorm.cls": _hf_qwen3_moe.Qwen3MoeRMSNorm,
    "qwen3_moe.RMSNorm.forward": _hf_qwen3_moe.Qwen3MoeRMSNorm.forward,
    "qwen3_moe.MLP.cls": _hf_qwen3_moe.Qwen3MoeMLP,
    # deepseek_v3
    "ds3.Attention.forward": _hf_ds3.DeepseekV3Attention.forward,
    "ds3.CausalLM.forward": _hf_ds3.DeepseekV3ForCausalLM.forward,
    "ds3.MoE.cls": _hf_ds3.DeepseekV3MoE,
    "ds3.PreTrained.init_weights": _hf_ds3.DeepseekV3PreTrainedModel._init_weights,
    "ds3.apply_rotary_pos_emb": _hf_ds3.apply_rotary_pos_emb,
    "ds3.RotaryEmb.cls": _hf_ds3.DeepseekV3RotaryEmbedding,
    "ds3.RotaryEmb.forward": _hf_ds3.DeepseekV3RotaryEmbedding.forward,
    "ds3.RMSNorm.cls": _hf_ds3.DeepseekV3RMSNorm,
    "ds3.RMSNorm.forward": _hf_ds3.DeepseekV3RMSNorm.forward,
    "ds3.MLP.cls": _hf_ds3.DeepseekV3MLP,
}


def apply_veomni_hf_unpatch():
    """Undo in-place veomni monkey-patches on HF model modules.

    `apply_veomni_*_patch()` in each of `qwen3/`, `qwen3_moe/`, `deepseek_v3/`
    mutates the HF model modules directly (forward swaps, class swaps, new
    `get_parallel_plan` methods). Without this restore, the first parametrize
    case to build a veomni model leaks its patches into every subsequent HF
    build in the same test session.
    """
    # Step 1: restore in-class forward methods on pristine class objects.
    # This handles both the always-on forward swaps and the Triton branch's
    # in-class mutations (DeepseekV3RotaryEmbedding.forward / DeepseekV3RMSNorm.forward).
    _PRISTINE_HF["qwen3.RMSNorm.cls"].forward = _PRISTINE_HF["qwen3.RMSNorm.forward"]
    _PRISTINE_HF["qwen3_moe.RMSNorm.cls"].forward = _PRISTINE_HF["qwen3_moe.RMSNorm.forward"]
    _PRISTINE_HF["ds3.RotaryEmb.cls"].forward = _PRISTINE_HF["ds3.RotaryEmb.forward"]
    _PRISTINE_HF["ds3.RMSNorm.cls"].forward = _PRISTINE_HF["ds3.RMSNorm.forward"]

    _hf_qwen3.Qwen3ForCausalLM.forward = _PRISTINE_HF["qwen3.CausalLM.forward"]
    _hf_qwen3.Qwen3ForSequenceClassification.forward = _PRISTINE_HF["qwen3.SeqCls.forward"]
    _hf_qwen3_moe.Qwen3MoeForCausalLM.forward = _PRISTINE_HF["qwen3_moe.CausalLM.forward"]
    _hf_qwen3_moe.Qwen3MoePreTrainedModel._init_weights = _PRISTINE_HF["qwen3_moe.PreTrained.init_weights"]
    _hf_ds3.DeepseekV3Attention.forward = _PRISTINE_HF["ds3.Attention.forward"]
    _hf_ds3.DeepseekV3ForCausalLM.forward = _PRISTINE_HF["ds3.CausalLM.forward"]
    _hf_ds3.DeepseekV3PreTrainedModel._init_weights = _PRISTINE_HF["ds3.PreTrained.init_weights"]

    # Step 2: restore module-level names for classes / functions that the
    # Liger branch swaps out wholesale, plus the class swaps done by the
    # always-on patch (Qwen3MoeSparseMoeBlock, DeepseekV3MoE).
    _hf_qwen3.apply_rotary_pos_emb = _PRISTINE_HF["qwen3.apply_rotary_pos_emb"]
    _hf_qwen3.Qwen3RMSNorm = _PRISTINE_HF["qwen3.RMSNorm.cls"]
    _hf_qwen3.Qwen3MLP = _PRISTINE_HF["qwen3.MLP.cls"]
    _hf_qwen3_moe.Qwen3MoeSparseMoeBlock = _PRISTINE_HF["qwen3_moe.MoeBlock.cls"]
    _hf_qwen3_moe.apply_rotary_pos_emb = _PRISTINE_HF["qwen3_moe.apply_rotary_pos_emb"]
    _hf_qwen3_moe.Qwen3MoeRMSNorm = _PRISTINE_HF["qwen3_moe.RMSNorm.cls"]
    _hf_qwen3_moe.Qwen3MoeMLP = _PRISTINE_HF["qwen3_moe.MLP.cls"]
    _hf_ds3.DeepseekV3MoE = _PRISTINE_HF["ds3.MoE.cls"]
    _hf_ds3.apply_rotary_pos_emb = _PRISTINE_HF["ds3.apply_rotary_pos_emb"]
    _hf_ds3.DeepseekV3RotaryEmbedding = _PRISTINE_HF["ds3.RotaryEmb.cls"]
    _hf_ds3.DeepseekV3RMSNorm = _PRISTINE_HF["ds3.RMSNorm.cls"]
    _hf_ds3.DeepseekV3MLP = _PRISTINE_HF["ds3.MLP.cls"]

    # Step 3: remove `get_parallel_plan` methods injected onto HF causal-LM
    # classes by apply_veomni_*_patch.
    for cls in (_hf_qwen3_moe.Qwen3MoeForCausalLM, _hf_ds3.DeepseekV3ForCausalLM):
        if "get_parallel_plan" in cls.__dict__:
            delattr(cls, "get_parallel_plan")


def apply_veomni_loss_unpatch():
    from transformers.loss.loss_utils import LOSS_MAPPING, ForCausalLMLoss

    from veomni.ops.kernels import cross_entropy

    cross_entropy._cross_entropy = None

    LOSS_MAPPING["ForCausalLM"] = ForCausalLMLoss
    LOSS_MAPPING["ForConditionalGeneration"] = ForCausalLMLoss


def apply_veomni_moe_unpatch():
    from veomni.ops.kernels import moe

    moe._fused_moe_forward = None


def _build_ops_config_for_mode(model_mode: ModelMode) -> OpsImplementationConfig:
    """Build an OpsImplementationConfig from a ModelMode for testing."""
    liger_impl = _LIGER_KERNEL if model_mode.use_liger_kernel else _EAGER
    return OpsImplementationConfig(
        attn_implementation=model_mode.attn_implementation,
        moe_implementation=model_mode.moe_implementation if model_mode.moe_implementation != "eager" else None,
        cross_entropy_loss_implementation=liger_impl,
        rms_norm_implementation=liger_impl,
        swiglu_mlp_implementation=liger_impl,
        rotary_pos_emb_implementation=liger_impl,
        load_balancing_loss_implementation=_LOAD_BALANCING_LOSS_IMPL,
    )


def set_environ_param(model_mode: ModelMode):
    apply_veomni_loss_unpatch()
    apply_veomni_moe_unpatch()
    if model_mode.modeling_backend == "veomni":
        os.environ["MODELING_BACKEND"] = "veomni"
    else:
        os.environ["MODELING_BACKEND"] = "hf"

    ops_config = _build_ops_config_for_mode(model_mode)
    apply_ops_config(ops_config)
