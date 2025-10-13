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

REG_MERGE = {
    "Qwen3-30B-A3B": "qwen",
    "Kimi-VL-A3B-Thinking-2506": "kimi",
    "Kimi-VL-A3B-Thinking": "kimi",
    "Kimi-VL-A3B-Instruct": "kimi",
}
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


def main(raw_hf_path, merge_hf_path):
    model_type = REG_MERGE[raw_hf_path]
    if model_type == 'qwen':
        merge_qwen(raw_hf_path, merge_hf_path)
    elif model_type == 'kimi':
        merge_kimi(raw_hf_path, merge_hf_path)
    else:
        raise NotImplementedError("Model type not supported!")

def merge_qwen(raw_hf_path, merge_hf_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(merge_hf_path, exist_ok=True)

    config = AutoConfig.from_pretrained(raw_hf_path)
    tokenizer = build_tokenizer(raw_hf_path)

    safetensor_files = list(glob(os.path.join(raw_hf_path, "*.safetensors")))
    safetensor_files.sort()
    state_dict_iterators = [StateDictIterator(shard_file) for shard_file in safetensor_files]
    new_state_dict = {}
    for state_dict_iterator in tqdm(state_dict_iterators, desc="Loading checkpoint shards"):
        for name, tensor in state_dict_iterator:
            new_state_dict[name] = tensor.cpu()

    num_experts = config.num_experts
    num_hidden_layers = config.num_hidden_layers
    for i in range(num_hidden_layers):
        gate_proj = []
        for j in range(num_experts):
            gate_proj.append(new_state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"))

        new_state_dict[f"model.layers.{i}.mlp.experts.gate_proj"] = torch.stack(gate_proj)
        up_proj = []
        for j in range(num_experts):
            up_proj.append(new_state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"))

        new_state_dict[f"model.layers.{i}.mlp.experts.up_proj"] = torch.stack(up_proj)
        down_proj = []
        for j in range(num_experts):
            down_proj.append(new_state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"))

        new_state_dict[f"model.layers.{i}.mlp.experts.down_proj"] = torch.stack(down_proj)

    model_assets = [config, tokenizer]
    save_model_weights(merge_hf_path, new_state_dict, model_assets=model_assets)

def merge_kimi(raw_hf_path, merge_hf_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(merge_hf_path, exist_ok=True)

    config = AutoConfig.from_pretrained(raw_hf_path, trust_remote_code=True)
    tokenizer = build_tokenizer(raw_hf_path)

    safetensor_files = list(glob(os.path.join(raw_hf_path, "*.safetensors")))
    safetensor_files.sort()
    state_dict_iterators = [StateDictIterator(shard_file) for shard_file in safetensor_files]
    new_state_dict = {}
    for state_dict_iterator in tqdm(state_dict_iterators, desc="Loading checkpoint shards"):
        for name, tensor in state_dict_iterator:
            new_state_dict[name] = tensor.cpu()

    num_experts = config.text_config.n_routed_experts
    num_hidden_layers = config.text_config.num_hidden_layers
    moe_layer_idxs = [i for i in range(num_hidden_layers) if (i >= config.text_config.first_k_dense_replace and i%config.text_config.moe_layer_freq==0)]
    print(new_state_dict.keys())
    for i in moe_layer_idxs:
        gate_proj = []
        for j in range(num_experts):
            gate_proj.append(new_state_dict.pop(f"language_model.model.layers.{i}.mlp.experts.{j}.gate_proj.weight"))

        new_state_dict[f"language_model.model.layers.{i}.mlp.experts.gate_proj"] = torch.stack(gate_proj)
        up_proj = []
        for j in range(num_experts):
            up_proj.append(new_state_dict.pop(f"language_model.model.layers.{i}.mlp.experts.{j}.up_proj.weight"))

        new_state_dict[f"language_model.model.layers.{i}.mlp.experts.up_proj"] = torch.stack(up_proj)
        down_proj = []
        for j in range(num_experts):
            down_proj.append(new_state_dict.pop(f"language_model.model.layers.{i}.mlp.experts.{j}.down_proj.weight"))

        new_state_dict[f"language_model.model.layers.{i}.mlp.experts.down_proj"] = torch.stack(down_proj)

    model_assets = [config, tokenizer]
    save_model_weights(merge_hf_path, new_state_dict, model_assets=model_assets)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_hf_path", type=str, required=True)
    parser.add_argument("--merge_hf_path", type=str, required=True)
    args = parser.parse_args()
    main(args.raw_hf_path, args.merge_hf_path)
