# Copyright 2024-2025 The Black-forest-labs Authors. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from safetensors import safe_open


refiners = []


def encode_prompt_using_clip(prompt, text_encoder, tokenizer, max_length, device):
    input_ids = tokenizer(
        prompt, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True
    ).input_ids.to(device)
    pooled_prompt_emb, _ = text_encoder(input_ids)
    return pooled_prompt_emb


def encode_prompt_using_t5(prompt, text_encoder, tokenizer, max_length, device):
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    prompt_emb = text_encoder(input_ids)
    return prompt_emb


@torch.no_grad()
def process_prompt(prompt, positive=True):
    if isinstance(prompt, list):
        prompt = [process_prompt(prompt_, positive=positive) for prompt_ in prompt]
    else:
        for refiner in refiners:
            prompt = refiner(prompt, positive=positive)
    return prompt


def encode_prompt(
    prompt,
    positive=True,
    device="cuda",
    t5_sequence_length=512,
    text_encoder_1=None,
    tokenizer_1=None,
    text_encoder_2=None,
    tokenizer_2=None,
):
    """_summary_

    Args:
        prompt (_type_): _description_
        positive (bool, optional): _description_. Defaults to True.
        device (str, optional): _description_. Defaults to "cuda".
        t5_sequence_length (int, optional): _description_. Defaults to 512.
        text_encoder_1 (_type_, optional): _description_. Defaults to None.
        tokenizer_1 (_type_, optional): _description_. Defaults to None.
        text_encoder_2 (_type_, optional): _description_. Defaults to None.
        tokenizer_2 (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    prompt = process_prompt(prompt=prompt, positive=positive)

    # CLIP
    pooled_prompt_emb = encode_prompt_using_clip(prompt, text_encoder_1, tokenizer_1, 77, device)

    # T5
    prompt_emb = encode_prompt_using_t5(prompt, text_encoder_2, tokenizer_2, t5_sequence_length, device)

    # text_ids
    text_ids = torch.zeros(prompt_emb.shape[0], prompt_emb.shape[1], 3).to(device=device, dtype=prompt_emb.dtype)

    return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}
    # return prompt_emb, pooled_prompt_emb, text_ids


def load_state_dict_from_folder(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in ["safetensors", "bin", "ckpt", "pth", "pt"]:
            state_dict.update(
                load_state_dict_(os.path.join(file_path, file_name), torch_dtype=torch_dtype, device=device)
            )
    return state_dict


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=device) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def load_state_dict_(file_path, torch_dtype=None, device="cpu"):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_model(file_path, device=None, torch_dtype=None):
    print(f"Loading models from: {file_path}")
    if device is None:
        device = device
    if torch_dtype is None:
        torch_dtype = torch_dtype
    if isinstance(file_path, list):
        state_dict = {}
        for path in file_path:
            state_dict.update(load_state_dict_(path, torch_dtype, device))
    elif os.path.isfile(file_path):
        state_dict = load_state_dict_(file_path, torch_dtype, device)
    else:
        state_dict = None

    return state_dict


def from_diffusers(state_dict):
    rename_dict = {
        "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
        "text_model.embeddings.position_embedding.weight": "position_embeds",
        "text_model.final_layer_norm.weight": "final_layer_norm.weight",
        "text_model.final_layer_norm.bias": "final_layer_norm.bias",
    }
    attn_rename_dict = {
        "self_attn.q_proj": "attn.to_q",
        "self_attn.k_proj": "attn.to_k",
        "self_attn.v_proj": "attn.to_v",
        "self_attn.out_proj": "attn.to_out",
        "layer_norm1": "layer_norm1",
        "layer_norm2": "layer_norm2",
        "mlp.fc1": "fc1",
        "mlp.fc2": "fc2",
    }
    state_dict_ = {}
    for name in state_dict:
        if name in rename_dict:
            param = state_dict[name]
            if name == "text_model.embeddings.position_embedding.weight":
                param = param.reshape((1, param.shape[0], param.shape[1]))
            state_dict_[rename_dict[name]] = param
        elif name.startswith("text_model.encoder.layers."):
            param = state_dict[name]
            names = name.split(".")
            layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
            name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
            state_dict_[name_] = param
    return state_dict_


def load_model_from_huggingface_folder(file_path, model_classes, torch_dtype, device):
    if torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        model = model_classes.from_pretrained(file_path, torch_dtype=torch_dtype).eval()
    else:
        model = model_classes.from_pretrained(file_path).eval().to(dtype=torch_dtype)
    if torch_dtype == torch.float16 and hasattr(model, "half"):
        model = model.half()
    return model


def load_model_from_single_file(state_dict, model_class, model_resource, torch_dtype, device):
    state_dict_converter = model_class.state_dict_converter()
    if model_resource == "civitai":
        state_dict_results = state_dict_converter.from_civitai(state_dict)
    elif model_resource == "diffusers":
        state_dict_results = state_dict_converter.from_diffusers(state_dict)
    return state_dict_results
