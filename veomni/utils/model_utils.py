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


import numpy as np
import torch.nn as nn

from . import logging


logger = logging.get_logger(__name__)


def pretty_print_trainable_parameters(model: nn.Module):
    trainable_parameters = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_parameters.append(n)

    printable_results = {}
    for p in trainable_parameters:
        param_split = p.split(".")
        param_name = ""
        digit_index = 0
        layer_index_list = []
        for split_item in param_split:
            if split_item.isdigit():
                param_name += f"<{digit_index}>."
                layer_index_list.append(int(split_item))
                digit_index += 1
            else:
                param_name += f"{split_item}."
        param_name = param_name[:-1]

        if param_name not in printable_results:
            printable_results[param_name] = []
        printable_results[param_name].append(layer_index_list)

    train_param_info = "\n**** trainable parameters ****"
    for param_key in printable_results.keys():
        layer_idxs = np.array(printable_results[param_key])
        if layer_idxs.shape[-1] == 0:
            train_param_info += "\n" + param_key
            continue
        layer_min = layer_idxs.min(axis=0)
        layer_max = layer_idxs.max(axis=0)
        print_pattern = param_key
        for index in range(len(layer_min)):
            if layer_min[index] == layer_max[index]:
                print_pattern = print_pattern.replace(f"<{index}>", f"[{layer_min[index]}]")
            else:
                print_pattern = print_pattern.replace(f"<{index}>", f"[{layer_min[index]}-{layer_max[index]}]")
        train_param_info += "\n" + print_pattern
    train_param_info += "\n**** trainable parameters ****"
    logger.info_rank0(train_param_info)
