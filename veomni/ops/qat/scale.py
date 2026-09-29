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
# See the License for the specific language governing limitations
# under the License.

"""Shared FP8 scale-storage pairing checks for QAT quantizers."""

from __future__ import annotations

import torch


def validate_fp8_scale_pairing(scale_fmt: str | None, scale_dtype: torch.dtype) -> None:
    """Reject scale storage that cannot represent the scale the kernel divides by.

    E8M0 has no mantissa. Storing an unrounded FP32 scale as E8M0 and later
    dequantizing with the unrounded value makes ``q * stored_scale`` drift
    from the fused dequant path.
    """
    if scale_dtype not in (torch.float32, torch.float8_e8m0fnu):
        raise AssertionError(f"FP8 quantizers support float32 and float8_e8m0fnu scales, got {scale_dtype}")
    if scale_dtype == torch.float8_e8m0fnu and scale_fmt is None:
        raise AssertionError(
            'float8_e8m0fnu scales only represent powers of two: pass scale_fmt (DeepSeek V4 uses "ue8m0")'
        )
