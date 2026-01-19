import os
from dataclasses import asdict, dataclass, fields
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


def prepare_model_modes(is_moe: bool = False):
    base_model_modes = [
        ModelMode(modeling_backend="hf", attn_implementation="eager", attn_case="padded_bsh"),
        ModelMode(modeling_backend="veomni", attn_implementation="eager", attn_case="padded_bsh"),
        ModelMode(modeling_backend="hf", attn_implementation="flash_attention_2", attn_case="position_ids"),
        ModelMode(
            modeling_backend="veomni",
            attn_implementation="veomni_flash_attention_2_with_sp",
            attn_case="position_ids",
        ),
    ]
    if not is_torch_npu_available():
        base_model_modes.extend(
            [
                ModelMode(modeling_backend="hf", attn_implementation="flash_attention_3", attn_case="position_ids"),
                ModelMode(
                    modeling_backend="veomni",
                    attn_implementation="veomni_flash_attention_3_with_sp",
                    attn_case="position_ids",
                    use_liger_kernel=True,
                ),
                ModelMode(
                    modeling_backend="veomni",
                    attn_implementation="veomni_flash_attention_3_with_sp",
                    attn_case="position_ids",
                    use_liger_kernel=False,
                ),
            ]
        )

    moe_model_modes = [
        ModelMode(
            modeling_backend="hf",
            attn_implementation="eager",
            attn_case="position_ids",
            moe_implementation="fused",
        ),
        ModelMode(
            modeling_backend="veomni",
            attn_implementation="eager",
            attn_case="position_ids",
            moe_implementation="fused",
        ),
        ModelMode(
            modeling_backend="hf",
            attn_implementation="flash_attention_2",
            attn_case="position_ids",
            moe_implementation="fused",
        ),
        ModelMode(
            modeling_backend="veomni",
            attn_implementation="veomni_flash_attention_2_with_sp",
            attn_case="position_ids",
            moe_implementation="fused",
        ),
    ]
    if not is_torch_npu_available():
        moe_model_modes.extend(
            [
                ModelMode(
                    modeling_backend="hf",
                    attn_implementation="flash_attention_3",
                    attn_case="position_ids",
                    moe_implementation="fused",
                ),
                ModelMode(
                    modeling_backend="veomni",
                    attn_implementation="veomni_flash_attention_3_with_sp",
                    attn_case="position_ids",
                    moe_implementation="fused",
                    use_liger_kernel=True,
                ),
                ModelMode(
                    modeling_backend="veomni",
                    attn_implementation="veomni_flash_attention_3_with_sp",
                    attn_case="position_ids",
                    moe_implementation="fused",
                    use_liger_kernel=False,
                ),
            ]
        )

    final_models_modes = base_model_modes + moe_model_modes if is_moe else base_model_modes
    hf_model_modes = [model_mode for model_mode in final_models_modes if model_mode.modeling_backend == "hf"]
    veomni_model_modes = [model_mode for model_mode in final_models_modes if model_mode.modeling_backend == "veomni"]

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
    loss = model(**inputs).loss.mean()
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


def compare_multi_items(outputs_dict: Dict, rtol=1e-3, atol=1e-5):
    base_task = next(iter(outputs_dict))
    base_output = outputs_dict[base_task]

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
                rtol=rtol,
                atol=atol,
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
