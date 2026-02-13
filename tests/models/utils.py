import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from transformers import set_seed

from veomni.models import build_foundation_model
from veomni.optim import build_optimizer
from veomni.utils.device import get_device_type
from veomni.utils.import_utils import is_torch_npu_available


def build_base_model_optim(
    config_path: str,
    attn_implementation: str = "eager",
    moe_implementation: str = "eager",
):
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="bfloat16",
        attn_implementation=attn_implementation,
        moe_implementation=moe_implementation,
        init_device=get_device_type(),
    )

    optimizer = build_optimizer(
        model,
        lr=0.0001,
        weight_decay=0,
        fused=True,
        optimizer_type="adamw",
        no_decay_modules=[],
        no_decay_params=[],
    )

    return model, optimizer


@dataclass(frozen=True)
class ModelMode:
    modeling_backend: str
    attn_implementation: str
    attn_case: str
    sync_weight_func: Optional[Callable] = None
    moe_implementation: str = "eager"  # 修正类型匹配
    use_liger_kernel: bool = False

    def __str__(self):
        return f"{self.modeling_backend}_[attn-{self.attn_implementation}]_[moe-{self.moe_implementation}]_[ligerkernel-{self.use_liger_kernel}]_[{self.attn_case}]"


# For each attn_case: HF uses _HF_ATTN, VeOmni uses _VEOMNI_ATTN × _USE_LIGER_KERNEL.
# On NPU skip FA3.
_BASE_ATTN_CASES = ["padded_bsh", "position_ids"]
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


def _append_veomni_modes(modes: list, attn_case: str, moe_implementation: str = "eager"):
    """Append VeOmni modes for one attn_case; every attn uses _USE_LIGER_KERNEL (True/False)."""
    for veomni_attn in _VEOMNI_ATTN:
        if _skip_fa3_npu(veomni_attn):
            continue
        for use_liger in _USE_LIGER_KERNEL:
            modes.append(
                ModelMode(
                    "veomni",
                    veomni_attn,
                    attn_case,
                    moe_implementation=moe_implementation,
                    use_liger_kernel=use_liger,
                )
            )


def _base_model_modes():
    """Base (non-MoE) model modes; all use sync_weight_func=None by default."""
    modes = []
    for attn_case in _BASE_ATTN_CASES:
        for hf_attn in _HF_ATTN:
            if _skip_fa3_npu(hf_attn):
                continue
            modes.append(ModelMode("hf", hf_attn, attn_case))
        _append_veomni_modes(modes, attn_case)
    return modes


def _moe_model_modes():
    """MoE model modes: same attn variants with moe_implementation=fused."""
    modes = []
    for attn_case in _BASE_ATTN_CASES:
        for hf_attn in _HF_ATTN:
            if _skip_fa3_npu(hf_attn):
                continue
        _append_veomni_modes(modes, attn_case, moe_implementation="fused")
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


def prepare_data(bsz, max_seq_len, seq_lens):
    def _get_dummy_inputs(data_type, bsz, max_seq_len, seq_lens, seed=42):
        if seq_lens.ndim != 1 or seq_lens.shape[0] != bsz:
            raise ValueError("seq_lens shape must be (batch_size,)")
        if torch.any(seq_lens > max_seq_len):
            raise ValueError(f"seq_lens must not contain elements > {max_seq_len}. {max_seq_len=}")

        set_seed(seed)
        input_ids = torch.randint(0, 1024, (bsz, max_seq_len))
        attention_mask = torch.ones_like(input_ids)
        positions = torch.arange(max_seq_len).expand(bsz, -1)
        padding_cutoff = (max_seq_len - seq_lens).unsqueeze(1)
        # left padding
        attention_mask[positions < padding_cutoff] = 0

        if data_type == "cu_seqlens":
            input_ids = torch.cat([input_ids[i, :l] for i, l in enumerate(seq_lens)])
            cu_seqlens = F.pad(seq_lens, pad=(1, 0)).cumsum_(-1).int()

            return {
                "input_ids": input_ids,
                "cu_seqlens": cu_seqlens,
                "attention_mask": torch.ones_like(input_ids),
                "labels": input_ids.clone(),
            }

        elif data_type == "position_ids":
            position_ids_list = []
            for i in range(input_ids.size(0)):
                valid_token_count = attention_mask[i].sum().item()
                position_ids = torch.arange(valid_token_count)
                position_ids_list.append(position_ids)
            position_ids = torch.cat(position_ids_list).unsqueeze(0)
            input_ids = torch.cat([input_ids[i, :l] for i, l in enumerate(seq_lens)]).unsqueeze(0)

            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "labels": input_ids.clone(),
            }

        elif data_type == "padded_bsh":
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }

        else:
            raise ValueError(f"Invalid data_type: {data_type}")

    dummy_data = {
        "cu_seqlens": _get_dummy_inputs(
            data_type="cu_seqlens", bsz=bsz, max_seq_len=max_seq_len, seq_lens=seq_lens, seed=42
        ),
        "position_ids": _get_dummy_inputs(
            data_type="position_ids", bsz=bsz, max_seq_len=max_seq_len, seq_lens=seq_lens, seed=42
        ),
        "padded_bsh": _get_dummy_inputs(
            data_type="padded_bsh", bsz=bsz, max_seq_len=max_seq_len, seq_lens=seq_lens, seed=42
        ),
    }

    return dummy_data


def train_one_step(model, optimizer, inputs):
    for k, v in inputs.items():
        inputs[k] = v.to(get_device_type())

    optimizer.zero_grad()
    loss = model(**inputs, use_cache=False).loss.mean()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    optimizer.step()

    return loss, gnorm


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


def compare_multi_items(outputs_dict: Dict, rtol=0.01, atol=0.01, gnorm_rtol=None, gnorm_atol=None):
    base_task = next(iter(outputs_dict))
    base_output = outputs_dict[base_task]

    # Use separate tolerances for gnorm if provided, otherwise fall back to loss tolerances.
    _gnorm_rtol = gnorm_rtol if gnorm_rtol is not None else rtol
    _gnorm_atol = gnorm_atol if gnorm_atol is not None else atol

    for task, output in outputs_dict.items():
        if task == base_task:
            continue
        try:
            torch.testing.assert_close(
                output["loss"],
                base_output["loss"],
                rtol=rtol,
                atol=atol,
            )
        except AssertionError:
            print_all_values(outputs_dict, "loss")
            raise AssertionError("Loss not match")

        try:
            torch.testing.assert_close(
                output["gnorm"],
                base_output["gnorm"],
                rtol=_gnorm_rtol,
                atol=_gnorm_atol,
            )
        except AssertionError:
            print_all_values(outputs_dict, "gnorm")
            raise AssertionError("Gnorm not match")


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
        os.environ["USE_LIGER_KERNEL"] = "1"
    else:
        os.environ["USE_LIGER_KERNEL"] = "0"
