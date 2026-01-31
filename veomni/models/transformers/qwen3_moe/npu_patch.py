# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
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
import torch_npu
import transformers.models.qwen3_moe.modeling_qwen3_moe as hf_qwen3_moe

from ....ops.group_gemm.kernel.npu_group_gemm import GmmFunction
from ....ops.npu_patch import npu_fused_operator


def qwen3_moe_sparse_moe_block_forward_npu(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """NPU optimized implementation for `forward` in Qwen3MoeSparseMoeBlock."""
    # hidden_states: (batch_size, sequence_length, hidden_size)
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # Loop over all available experts in the model and perform the computation on each expert
    # Concat all weights
    input_dtype = hidden_states.dtype
    # For the weight is merged
    w1 = self.experts.up_proj.transpose(1, 2).to(input_dtype)
    w2 = self.experts.gate_proj.transpose(1, 2).to(input_dtype)
    w3 = self.experts.down_proj.transpose(1, 2).to(input_dtype)

    permuted_tokens, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    tokens_per_expert = torch.histc(selected_experts, bins=self.num_experts, min=0, max=self.num_experts)

    up_res = GmmFunction.apply(permuted_tokens, w1, tokens_per_expert)
    gate_res = GmmFunction.apply(permuted_tokens, w2, tokens_per_expert)
    act_res = torch_npu.npu_swiglu(torch.cat([gate_res, up_res], dim=-1))
    down_res = GmmFunction.apply(act_res, w3, tokens_per_expert)

    final_hidden_states = torch_npu.npu_moe_token_unpermute(down_res, row_ids_map, probs=routing_weights)

    return final_hidden_states, router_logits


def apply_qwen3_moe_npu_patch():
    # Patches for Qwen3 MoE Model
    hf_qwen3_moe.Qwen3MoeRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
    # hf_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = qwen3_moe_sparse_moe_block_forward_npu
    hf_qwen3_moe.apply_rotary_pos_emb = npu_fused_operator.apply_rotary_pos_emb_npu
