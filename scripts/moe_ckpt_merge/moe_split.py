"""
Reverse process of moe_merge.py - splits merged MoE expert weights back to individual experts.

This script takes a HF checkpoint with stacked/fused expert weights and splits them back to
the original per-expert format expected by HuggingFace safetensors.

Supported input formats:
  - v4 veomni format (from moe_merge.py):
      model.layers.{i}.mlp.experts.gate_proj  [E, I, H]
      model.layers.{i}.mlp.experts.up_proj    [E, I, H]
      model.layers.{i}.mlp.experts.down_proj  [E, H, I]

  - v5 fused format (from training with transformers v5 patchgen modeling):
      model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
      model.layers.{i}.mlp.experts.down_proj     [E, H, I]

Output format (original HF per-expert):
    model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [I, H]
    model.layers.{i}.mlp.experts.{j}.up_proj.weight    [I, H]
    model.layers.{i}.mlp.experts.{j}.down_proj.weight  [H, I]

Usage: python moe_split.py --merge_hf_path <merged_checkpoint> --split_hf_path <output_dir>
"""

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob
from typing import Generator, Tuple

import torch
from safetensors.torch import safe_open
from tqdm import tqdm
from transformers import AutoConfig

from veomni.models import build_tokenizer, save_model_weights


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


def main(merge_hf_path, split_hf_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(split_hf_path, exist_ok=True)

    config = AutoConfig.from_pretrained(merge_hf_path)
    tokenizer = build_tokenizer(merge_hf_path)

    safetensor_files = list(glob(os.path.join(merge_hf_path, "*.safetensors")))
    safetensor_files.sort()
    state_dict_iterators = [StateDictIterator(shard_file) for shard_file in safetensor_files]
    new_state_dict = {}
    for state_dict_iterator in tqdm(state_dict_iterators, desc="Loading checkpoint shards"):
        for name, tensor in state_dict_iterator:
            new_state_dict[name] = tensor.cpu()

    num_experts = config.num_experts
    num_hidden_layers = config.num_hidden_layers
    for i in range(num_hidden_layers):
        print(f"Converting layer {i}")

        # Handle v5 fused gate_up_proj [E, 2*I, H] -> gate_proj + up_proj per expert
        gate_up_key = f"model.layers.{i}.mlp.experts.gate_up_proj"
        if gate_up_key in new_state_dict:
            gate_up_tensor = new_state_dict.pop(gate_up_key)  # [E, 2*I, H]
            for j in range(num_experts):
                gate, up = gate_up_tensor[j].chunk(2, dim=0)  # each [I, H]
                new_state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = gate
                new_state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = up

        # Handle v4 separate gate_proj [E, I, H] and up_proj [E, I, H] per expert
        for proj_name in ["gate_proj", "up_proj"]:
            stacked_key = f"model.layers.{i}.mlp.experts.{proj_name}"
            if stacked_key in new_state_dict:
                stacked_tensor = new_state_dict.pop(stacked_key)
                for j in range(num_experts):
                    expert_key = f"model.layers.{i}.mlp.experts.{j}.{proj_name}.weight"
                    new_state_dict[expert_key] = stacked_tensor[j]

        # Handle down_proj [E, H, I] (same key name in both v4 and v5)
        down_key = f"model.layers.{i}.mlp.experts.down_proj"
        if down_key in new_state_dict:
            stacked_tensor = new_state_dict.pop(down_key)
            for j in range(num_experts):
                expert_key = f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"
                new_state_dict[expert_key] = stacked_tensor[j]

    model_assets = [config, tokenizer]

    print("Saving to safetensors")
    save_model_weights(split_hf_path, new_state_dict, model_assets=model_assets)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--merge_hf_path", type=str, required=True)
    parser.add_argument("--split_hf_path", type=str, required=True)
    args = parser.parse_args()
    main(args.merge_hf_path, args.split_hf_path)
