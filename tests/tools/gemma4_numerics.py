# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Test-only Gemma precision fixtures and explicit FP32 attention reference."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from tests.models.tiny_configs import tiny_gemma4_text_config
from veomni.ops.kernels.attention.standard import flex as flex_backend
from veomni.ops.kernels.attention.standard.flash import bind_flash_attention_forward


def precision_config(bidirectional=False):
    """Retain the historical four-layer PLE/KV-sharing stress fixture."""
    values = tiny_gemma4_text_config().to_dict()
    values.update(
        head_dim=256,
        global_head_dim=256,
        sliding_window=32,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention"] * 2,
        use_bidirectional_attention="all" if bidirectional else None,
    )
    return Gemma4TextConfig.from_dict(values)


def visibility(config, length, cu, *, sliding, flash=False, device="cpu"):
    """Independent visibility oracle; Flash's native radius is window minus one."""
    positions = torch.arange(length, device=device)
    documents = torch.bucketize(positions, cu[1:].to(device=device), right=True)
    distance = positions[:, None] - positions[None, :]
    visible = documents[:, None] == documents[None, :]
    causal = getattr(config, "is_causal", True)
    if causal:
        visible = visible & (distance >= 0)
    if sliding:
        window = config.sliding_window
        if causal:
            visible = visible & (distance < window)
        elif flash:
            if length > window:
                visible = visible & (distance.abs() <= window - 1)
        else:
            visible = visible & (distance.abs() <= window)
    return visible[None, None]


def dense_masks(config, length, cu, *, backend="sdpa", dtype=torch.float32, device="cpu", flash=False):
    """Construct masks independently of the production Gemma/public mask builders."""
    result = {}
    for layer_type in ("full_attention", "sliding_attention"):
        mask = visibility(config, length, cu, sliding=layer_type == "sliding_attention", flash=flash, device=device)
        if backend == "eager":
            mask = torch.where(mask, torch.zeros((), dtype=dtype, device=device), torch.finfo(dtype).min)
        result[layer_type] = mask
    return result


def fp32_attention(module, query, key, value, attention_mask, *, scaling=None, **kwargs):
    """Bypass VeOmni SDPA's nested backend context and low-precision accumulations."""
    del module, kwargs
    if isinstance(attention_mask, BlockMask):
        raise TypeError("The FP32 oracle requires an explicit dense mask")
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            query.float(),
            key.float(),
            value.float(),
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=scaling,
            enable_gqa=True,
        )
    return output.to(query.dtype).transpose(1, 2), None


def relative_l2(actual, expected):
    """Report a norm metric, preserving failures near zero with explicit absolute checks."""
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


def attention_module(config, sliding):
    module = torch.nn.Module()
    module.config = config
    module.is_causal = getattr(config, "is_causal", True)
    module.is_sliding = sliding
    module.sliding_window = config.sliding_window if sliding else None
    module.num_key_value_groups = 2
    return module


def run_kernel(module, qkv, mask, cu, length, backend):
    kwargs = dict(
        scaling=1.0,
        sliding_window=module.sliding_window,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=length // 2,
        max_length_k=length // 2,
    )
    if backend == "flex_attention":
        return flex_backend.flex_attention_forward(
            module, *qkv, mask, kernel_options={"BACKEND": "TRITON", "num_stages": 1}, **kwargs
        )[0]
    return bind_flash_attention_forward("veomni_flash_attention_2")(module, *qkv, None, deterministic=True, **kwargs)[
        0
    ]
