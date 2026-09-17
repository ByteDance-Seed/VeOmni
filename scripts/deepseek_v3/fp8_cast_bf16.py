# adapted from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py
import json
import os
from argparse import ArgumentParser
from glob import glob

import torch
from kernel import weight_dequant
from safetensors.torch import load_file
from tqdm import tqdm

from veomni.models import save_model_weights
from veomni.utils.device import empty_cache, get_device_type


def main(fp8_path, bf16_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file) as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device=get_device_type())
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    new_state_dict = {}
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device=get_device_type())
        loaded_files[file_name] = current_state_dict

        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv).cpu()
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight.cpu()
            else:
                new_state_dict[weight_name] = weight.cpu()

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            empty_cache()

    num_experts = 256
    start_layer, end_layer = 3, 61
    for i in range(start_layer, end_layer):
        gate_proj = []
        for j in range(num_experts):
            gate_proj.append(new_state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"))

        new_state_dict[f"model.layers.{i}.mlp.experts.fc1_1"] = torch.stack(gate_proj)
        up_proj = []
        for j in range(num_experts):
            up_proj.append(new_state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"))

        new_state_dict[f"model.layers.{i}.mlp.experts.fc1_2"] = torch.stack(up_proj)
        down_proj = []
        for j in range(num_experts):
            down_proj.append(new_state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"))

        new_state_dict[f"model.layers.{i}.mlp.experts.fc2"] = torch.stack(down_proj)

    weight_names = list(new_state_dict.keys())
    for k in weight_names:
        if f"model.layers.{end_layer}" in k:
            new_state_dict.pop(k)

    save_model_weights(bf16_path, new_state_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
