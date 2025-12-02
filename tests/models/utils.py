from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import set_seed

from veomni.models import build_foundation_model
from veomni.optim import build_optimizer
from veomni.utils.device import get_device_type


def build_base_model_optim(
    config_path: str,
    force_use_huggingface: bool = True,
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
        force_use_huggingface=force_use_huggingface,
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


@dataclass
class ModelMode:
    force_use_huggingface: bool
    attn_implementation: str
    attn_case: str
    sync_weight_func: any = None
    moe_implementation: bool = "eager"


def prepare_models_modes(is_moe: bool = False):
    base_model_modes = [
        ModelMode(force_use_huggingface=True, attn_implementation="eager", attn_case="padded_bsh"),
        ModelMode(force_use_huggingface=False, attn_implementation="eager", attn_case="padded_bsh"),
        ModelMode(force_use_huggingface=True, attn_implementation="flash_attention_2", attn_case="position_ids"),
        ModelMode(force_use_huggingface=False, attn_implementation="flash_attention_2", attn_case="position_ids"),
    ]

    moe_model_modes = [
        ModelMode(
            force_use_huggingface=True,
            attn_implementation="eager",
            attn_case="position_ids",
            moe_implementation="fused",
        ),
        ModelMode(
            force_use_huggingface=True,
            attn_implementation="eager",
            attn_case="position_ids",
            moe_implementation="fused",
        ),
        ModelMode(
            force_use_huggingface=True,
            attn_implementation="flash_attention_2",
            attn_case="position_ids",
            moe_implementation="fused",
        ),
        ModelMode(
            force_use_huggingface=False,
            attn_implementation="flash_attention_2",
            attn_case="position_ids",
            moe_implementation="fused",
        ),
    ]

    return base_model_modes + moe_model_modes if is_moe else base_model_modes


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


def print_all_values(output_dict, value_key):
    max_key_length = max(len(key) for key in output_dict.keys()) + len(value_key) + 1

    for key, output in output_dict.items():
        value = output[value_key]
        value_str = f"{value.item() if hasattr(value, 'item') else value}"

        print(f"  {(key + '.' + value_key).rjust(max_key_length)}: {value_str}")


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
