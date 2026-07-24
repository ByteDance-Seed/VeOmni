# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HunyuanImage 3 2D RoPE reference (model-local).

The 2D frequency layout follows Tencent's ``build_2d_rope`` implementation at
revision ``6e9113a692a27a0751d82aba3b2015a876646c03`` -- kept in the model
directory because it is a bit-for-bit reproduction of the upstream model's
positional encoding scheme, not a generic RoPE primitive.

The dense GCA reference attention forward, which used to live here, has moved
to :mod:`veomni.ops.kernels.attention.reference` (it is model-agnostic and
usable as a parity oracle for any GQA attention kernel).
"""

import torch


def build_reference_2d_rope(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    rope_theta: float = 10000.0,
    base_rescale_factor: float = 1.0,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build official-layout cos/sin tensors from compiled ``[B, 2, T]`` coordinates."""
    if position_ids.ndim != 3 or position_ids.shape[1] != 2:
        raise ValueError("position_ids must have shape [batch, 2, sequence_length].")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("position_ids must use an integer dtype.")
    if head_dim <= 0 or head_dim % 4:
        raise ValueError("Hunyuan Image 3 attention head_dim must be positive and divisible by four.")
    if rope_theta <= 0 or base_rescale_factor <= 0:
        raise ValueError("rope_theta and base_rescale_factor must be positive.")

    base = float(rope_theta)
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (head_dim / (head_dim - 2))
    inverse_frequency = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=position_ids.device) / head_dim)
    )
    theta = inverse_frequency.reshape(1, 1, head_dim // 4, 2)
    coordinates = position_ids.transpose(1, 2).unsqueeze(-2).to(torch.float32)
    angles = (coordinates * theta).reshape(position_ids.shape[0], position_ids.shape[2], head_dim // 2)
    angles = angles.repeat(1, 1, 2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    if dtype is not None:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
    return cos, sin


__all__ = ["build_reference_2d_rope"]
