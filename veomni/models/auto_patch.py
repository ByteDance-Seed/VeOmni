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


import torch
import types

ATTENTION_KEYWORDS = ("attention", "Attention", "ATTENTION")


def _is_attention_module(module):
    name = module.__class__.__name__
    return any(keyword in name for keyword in ATTENTION_KEYWORDS)


def _wrap_attention_forward(module):
    """Patch forward to move cu_seq_lens_x to CPU only on NPU."""
    if hasattr(module, "_original_forward_patched"):
        # Avoid double patch
        return

    original_forward = module.forward

    def wrapped_forward(self, *args, **kwargs):
        # Only patch for NPU fused-attention case
        if torch.npu.is_available():
            for key in ("cu_seq_lens_q", "cu_seq_lens_k"):
                if key in kwargs and kwargs[key] is not None:
                    # Avoid unnecessary sync: only convert if tensor on NPU
                    v = kwargs[key]
                    if isinstance(v, torch.Tensor) and v.device.type == "npu":
                        kwargs[key] = v.cpu()

        return original_forward(*args, **kwargs)

    # Monkey patch
    module.forward = types.MethodType(wrapped_forward, module)
    module._original_forward_patched = True


def auto_patch_npu_attention(model):
    """
    Automatically find all attention modules in the model
    and patch their forward() so that cu_seq_lens_x stays on CPU.

    Args:
        model: torch.nn.Module
    """
    for name, module in model.named_modules():
        if _is_attention_module(module):
            _wrap_attention_forward(module)


__all__ = ["auto_patch_attention"]