import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable, Dict, Optional

import torch
from rich.console import Console
from rich.table import Table
from transformers import set_seed

from veomni.data.dummy_dataset import build_dummy_dataset
from veomni.utils.import_utils import is_torch_npu_available


@dataclass(frozen=True)
class ModelMode:
    modeling_backend: str
    attn_implementation: str
    sync_weight_func: Optional[Callable] = None
    moe_implementation: str = "eager"  # 修正类型匹配
    use_liger_kernel: bool = False

    def __str__(self):
        return f"{self.modeling_backend}_[attn-{self.attn_implementation}]_[moe-{self.moe_implementation}]_[ligerkernel-{self.use_liger_kernel}]]"


# HF uses _HF_ATTN, VeOmni uses _VEOMNI_ATTN × _USE_LIGER_KERNEL.
# On NPU skip FA3.
_HF_ATTN = ["eager", "flash_attention_2", "flash_attention_3"]
_VEOMNI_ATTN = [
    "eager",
    "veomni_flash_attention_2_with_sp",
    "veomni_flash_attention_3_with_sp",
]
_USE_LIGER_KERNEL = [True, False]


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


def prepare_data(model_name: str, max_seq_len: int, model_config):
    dataset_name = MODEL_TO_DATASET.get(model_name, "text")
    dataset = build_dummy_dataset(dataset_name, 1, max_seq_len)

    token_ids_dict = parse_token_id_from_config(model_config)

    class DummyDataLoader:
        def process(self, example):
            example = example[0]
            example = {key: torch.tensor(v) for key, v in example.items()}
            for key in UNSQUEECE_KEYS:
                if key not in example:
                    continue
                example[key] = example[key].unsqueeze(0)

            if "image_mask" in example:
                example["input_ids"].masked_fill_(example["image_mask"], token_ids_dict["image_token_id"])
            if "video_mask" in example:
                example["input_ids"].masked_fill_(example["video_mask"], token_ids_dict["video_token_id"])
            if "audio_mask" in example:
                example["input_ids"].masked_fill_(example["audio_mask"], token_ids_dict["audio_token_id"])

            if example["position_ids"].dim() == 3:
                example["position_ids"] = example["position_ids"].transpose(0, 1).contiguous()

            return example

        def __iter__(self):
            set_seed(42)
            yield self.process(dataset[0])

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
        val_str = f"{val_obj.item() if hasattr(val_obj, 'item') else val_obj:.8f}"  # 这里加上了.4f保留小数
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
            except AssertionError:
                print_all_values(outputs_dict, key)
                raise AssertionError(f"{key} not match")


def apply_veomni_loss_unpatch():
    from transformers.loss.loss_utils import LOSS_MAPPING, ForCausalLMLoss

    from veomni.ops import fused_cross_entropy

    fused_cross_entropy._cross_entropy = None

    LOSS_MAPPING["ForCausalLM"] = ForCausalLMLoss
    LOSS_MAPPING["ForConditionalGeneration"] = ForCausalLMLoss


def apply_veomni_moe_unpatch():
    from veomni.ops import fused_moe

    fused_moe._fused_moe_forward = None


def set_environ_param(model_mode: ModelMode):
    apply_veomni_loss_unpatch()
    apply_veomni_moe_unpatch()
    if model_mode.modeling_backend == "veomni":
        os.environ["MODELING_BACKEND"] = "veomni"
    else:
        os.environ["MODELING_BACKEND"] = "hf"

    if model_mode.use_liger_kernel:
        os.environ["VEOMNI_USE_LIGER_KERNEL"] = "1"
    else:
        os.environ["VEOMNI_USE_LIGER_KERNEL"] = "0"
