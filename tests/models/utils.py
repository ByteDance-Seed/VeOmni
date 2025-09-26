from typing import Dict

import torch
import torch.nn.functional as F
from transformers import set_seed

from veomni.models import build_foundation_model
from veomni.optim import build_optimizer
from veomni.utils.device import get_device_type, get_torch_device


def build_base_model_optim(
    config_path: str,
    force_use_huggingface: bool = True,
    attn_implementation: str = "eager",
    moe_implementation: str = "eager",
):
    set_seed(42)
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


def get_dummy_data(data_type, bsz, seqlen, seed=42):
    set_seed(seed)
    input_ids = torch.randint(0, 1024, (bsz, seqlen))
    attention_mask = torch.ones_like(input_ids)
    if data_type == "cu_seqlens":
        seqlens = attention_mask.sum(-1)
        input_ids = torch.cat([input_ids[i, :l] for i, l in enumerate(seqlens)])
        cu_seqlens = F.pad(seqlens, pad=(1, 0)).cumsum_(-1).int()

        return {
            "input_ids": input_ids,
            "cu_seqlens": cu_seqlens,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }
    elif data_type == "position_ids":
        seqlens = attention_mask.sum(-1)
        position_ids_list = []
        for i in range(input_ids.size(0)):
            valid_token_count = attention_mask[i].sum().item()
            position_ids = torch.arange(valid_token_count)
            position_ids_list.append(position_ids)
        position_ids = torch.cat(position_ids_list).unsqueeze(0)
        input_ids = torch.cat([input_ids[i, :l] for i, l in enumerate(seqlens)]).unsqueeze(0)

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


def train_one_step(model, optimizer, inputs):
    for k, v in inputs.items():
        inputs[k] = v.to(get_torch_device())

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
