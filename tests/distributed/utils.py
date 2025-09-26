import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import set_seed

from veomni.checkpoint.checkpointer import DistributedCheckpointer
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.optim import build_optimizer
from veomni.utils.device import get_device_type, get_torch_device


TOY_MODEL_MAP = {"qwen3_moe": {"path": "./tests/model_config/qwen3moe_toy.json", "n_param": 224104896}}


def save_dcp(model, optimizer, save_path, global_steps: int = None):
    state = {
        "model": model,
        "optimizer": optimizer,
        "extra_state": {
            "cpu": torch.random.get_rng_state(),
            get_device_type(): get_torch_device().get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
    }

    DistributedCheckpointer.save(save_path, state, global_steps=global_steps)


def load_dcp(model, optimizer, load_path):
    state = {"model": model, "optimizer": optimizer, "extra_state": {}}
    DistributedCheckpointer.load(load_path, state)

    get_torch_device().random.set_rng_state(state["extra_state"][get_device_type()])
    torch.random.set_rng_state(state["extra_state"]["cpu"])
    np.random.set_state(state["extra_state"]["numpy"])
    random.setstate(state["extra_state"]["random"])


def build_model_optim(config_path: str, dp_size: int, ep_size: int, ulysses_size: int = 1, init_device: str = "meta"):
    dp_size = dp_size
    ep_size = ep_size
    ulysses_size = ulysses_size

    tp_size = 1
    pp_size = 1
    cp_size = 1

    force_use_huggingface = False

    init_parallel_state(
        dp_size=dp_size,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ulysses_size=ulysses_size,
        dp_mode="fsdp1",
    )

    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="flash_attention_2",
        moe_implementation="eager",
        init_device=init_device,
        force_use_huggingface=force_use_huggingface,
    )

    model: FSDP = build_parallelize_model(
        model,
        init_device=init_device,
        weights_path=None,
        enable_full_shard=True,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True,
        enable_fsdp_offload=False,
        basic_modules=model._no_split_modules,
        enable_reentrant=False,
        enable_forward_prefetch=True,
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


def get_dummy_data(data_type="cu_seqlens", seed=42):
    set_seed(seed)
    input_ids = torch.randint(0, 1024, (1, 128))
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
        input_ids = torch.cat([input_ids[i, :l] for i, l in enumerate(seqlens)])

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": input_ids.clone(),
        }
    else:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def train_one_step(model, optimizer, data_type="cu_seqlens", seed=42):
    inputs = get_dummy_data(data_type=data_type, seed=seed)

    optimizer.zero_grad()
    loss = model(**inputs).loss.mean()
    loss.backward()

    gnorm = model.clip_grad_norm_(max_norm=1.0)
    optimizer.step()
    if dist.get_rank() == 0:
        print(f"loss: {loss}, gnorm: {gnorm}")
    return loss, gnorm
