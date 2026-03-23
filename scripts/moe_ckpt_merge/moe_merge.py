import os
from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob
from typing import Generator, List, Tuple

import torch
from safetensors.torch import safe_open
from tqdm import tqdm
from transformers import AutoConfig

from veomni.models import build_processor, build_tokenizer, save_model_weights


@dataclass
class StateDictIterator:
    filepath: str

    def __iter__(self) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]


@dataclass
class MoEGroup:
    prefix: str  # e.g. "model.layers", "thinker.model.layers", "talker.model.layers"
    num_hidden_layers: int
    num_experts: int
    moe_layer_start_idx: int = 0


def _get_moe_config(config) -> Tuple[int, int]:
    """Extract num_experts and first_k_dense_replace from a config object."""
    if hasattr(config, "num_experts"):
        num_experts = config.num_experts
    elif hasattr(config, "n_routed_experts"):
        num_experts = config.n_routed_experts
    else:
        raise RuntimeError(f"could not find num_experts in config: {type(config)}")

    if hasattr(config, "first_k_dense_replace"):
        moe_layer_start_idx = config.first_k_dense_replace
    else:
        moe_layer_start_idx = 0

    return num_experts, moe_layer_start_idx


def _detect_moe_groups(config) -> List[MoEGroup]:
    """Detect MoE groups from config, supporting both flat and nested (omni) models."""
    groups = []

    # Case 1: Qwen3-Omni style with thinker_config / talker_config
    if hasattr(config, "thinker_config") or hasattr(config, "talker_config"):
        if hasattr(config, "thinker_config"):
            thinker_cfg = config.thinker_config
            text_cfg = thinker_cfg.text_config if hasattr(thinker_cfg, "text_config") else thinker_cfg
            num_experts, moe_start = _get_moe_config(text_cfg)
            groups.append(
                MoEGroup(
                    prefix="thinker.model.layers",
                    num_hidden_layers=text_cfg.num_hidden_layers,
                    num_experts=num_experts,
                    moe_layer_start_idx=moe_start,
                )
            )

        if hasattr(config, "talker_config"):
            talker_cfg = config.talker_config
            text_cfg = talker_cfg.text_config if hasattr(talker_cfg, "text_config") else talker_cfg
            if hasattr(text_cfg, "num_experts") or hasattr(text_cfg, "n_routed_experts"):
                num_experts, moe_start = _get_moe_config(text_cfg)
                groups.append(
                    MoEGroup(
                        prefix="talker.model.layers",
                        num_hidden_layers=text_cfg.num_hidden_layers,
                        num_experts=num_experts,
                        moe_layer_start_idx=moe_start,
                    )
                )

        return groups

    # Case 2: Flat model (qwen3moe, deepseek, etc.)
    text_cfg = config.text_config if hasattr(config, "text_config") else config
    num_experts, moe_start = _get_moe_config(text_cfg)
    num_hidden_layers = (
        text_cfg.num_hidden_layers if hasattr(text_cfg, "num_hidden_layers") else config.num_hidden_layers
    )
    groups.append(
        MoEGroup(
            prefix="model.layers",
            num_hidden_layers=num_hidden_layers,
            num_experts=num_experts,
            moe_layer_start_idx=moe_start,
        )
    )
    return groups


def _merge_experts_for_group(state_dict: dict, group: MoEGroup):
    """Merge per-expert weights into stacked tensors for one MoE group."""
    prefix = group.prefix
    print(
        f"Merging experts for '{prefix}': layers {group.moe_layer_start_idx}-{group.num_hidden_layers - 1}, "
        f"{group.num_experts} experts"
    )

    for i in range(group.moe_layer_start_idx, group.num_hidden_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensors = []
            for j in range(group.num_experts):
                key = f"{prefix}.{i}.mlp.experts.{j}.{proj}.weight"
                tensors.append(state_dict.pop(key))
            state_dict[f"{prefix}.{i}.mlp.experts.{proj}"] = torch.stack(tensors)


def main(raw_hf_path, merge_hf_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(merge_hf_path, exist_ok=True)

    config = AutoConfig.from_pretrained(raw_hf_path)
    tokenizer = build_tokenizer(raw_hf_path)
    try:
        processor = build_processor(raw_hf_path)
    except Exception:
        processor = None

    safetensor_files = list(glob(os.path.join(raw_hf_path, "*.safetensors")))
    safetensor_files.sort()
    state_dict_iterators = [StateDictIterator(shard_file) for shard_file in safetensor_files]
    new_state_dict = {}
    for state_dict_iterator in tqdm(state_dict_iterators, desc="Loading checkpoint shards"):
        for name, tensor in state_dict_iterator:
            new_state_dict[name] = tensor.cpu()

    print(new_state_dict.keys())

    moe_groups = _detect_moe_groups(config)
    print(f"Detected {len(moe_groups)} MoE group(s)")
    for group in moe_groups:
        _merge_experts_for_group(new_state_dict, group)

    model_assets = [asset for asset in [config, tokenizer, processor] if asset is not None]
    save_model_weights(merge_hf_path, new_state_dict, model_assets=model_assets)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_hf_path", type=str, required=True)
    parser.add_argument("--merge_hf_path", type=str, required=True)
    args = parser.parse_args()
    main(args.raw_hf_path, args.merge_hf_path)
