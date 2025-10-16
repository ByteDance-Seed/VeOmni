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
import torch.nn.functional as F


def len2culen(seqlens: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the sequence lengths to cumulative sequence lengths.

    NOTE: flash attention only accepts int32 cu_seqlens.
    """
    return F.pad(torch.cumsum(seqlens, dim=0), (1, 0)).type(torch.int32)


def culen2len(cu_seqlens: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the cumulative sequence lengths to sequence lengths.
    """
    return cu_seqlens.diff()


def pos2culen(position_ids: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the position ids to cumulative sequence lengths.
    """
    if position_ids.dim() == 3:  # (batch_size, dim, seq_length):
        position_ids = position_ids[:, 0, :]

    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), dtype=torch.int32, device=position_ids.device)
    return F.pad(indices_q[position_ids == 0], (0, 1), "constant", position_ids.size(0))


def culen2pos(cu_seqlens: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the cumulative sequence lengths to position ids.
    """
    seqlens = culen2len(cu_seqlens).cpu()
    position_ids = torch.cat([torch.arange(length, dtype=torch.long, device=cu_seqlens.device) for length in seqlens])
    return position_ids.unsqueeze(0)


def prepare_fa_kwargs_from_position_ids(position_ids):
    """
    Copy from transformers/modeling_flash_attention_utils.py 354567d955fbc5fbd70fc841b7a7bcc654bea3f1
    This function returns all the necessary kwargs to call `flash_attn_varlen_func` extracted from position_ids.

    Arguments:
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into
            ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query,
            `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.view(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )
    cu_seq_lens_k = cu_seq_lens_q

    # https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/flash_attn/flash_attn_interface.py#L1423-L1424
    # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always increasing
    # for some models (e.g. qwen2-vl).
    max_length_q = cu_seq_lens_q.diff().max()
    # NOTE: With torch compile, this will cause a graph break if you don't set
    # `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` in the environment or call
    # `torch._dynamo.config.capture_scalar_outputs = True` before doing the forward pass.
    # This is a limitation of flash attention API, as the function `flash_attn_varlen_func`
    # requires `max_length_q`, `max_length_k` to be passed as `int` and not `torch.Tensor`.
    max_length_q = max_length_q.item()
    max_length_k = max_length_q

    return (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k)
