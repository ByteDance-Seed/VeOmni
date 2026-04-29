import torch

from ..utils.import_utils import is_fused_moe_available


if is_fused_moe_available():
    from .group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
    from .group_gemm.kernel.moe import expert_histogram, moe_gather, moe_scatter


class FusedMoeExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_experts: int,
        gate_weights: torch.Tensor,
        expert_index: torch.Tensor,
        hidden_states: torch.Tensor,
        gate_up_proj: torch.Tensor,
        gate_up_proj_bias: torch.Tensor,
        down_proj: torch.Tensor,
        down_proj_bias: torch.Tensor,
        alpha: float,
        limit: float,
    ):
        # MOE Step 3: 分发输入 tokens 到专家（与原逻辑一致）
        splits = expert_histogram(expert_index, num_experts)
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)
        scatter_output = moe_scatter(hidden_states, scatter_index)  # (total_tokens, hidden_size)
        cumsum_t = torch.cumsum(splits, dim=0)
        max_M = scatter_output.shape[0]

        # MOE Step 4: 合并计算 Gate 和 Up 投影（新增逻辑）
        # 输入: scatter_output (total_tokens, hidden_size)
        # 权重: gate_up_proj (num_experts, hidden_size, 2*expert_dim)
        # 偏置: gate_up_proj_bias (num_experts, 2*expert_dim)
        # 输出: gate_up (total_tokens, 2*expert_dim)
        gate_up = group_gemm_same_nk(
            a=scatter_output,
            b=gate_up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        gate_up_bias = gate_up_proj_bias.index_select(0, expert_index.flatten()).view(gate_up.shape)
        scattered_gate_up_bias = torch.empty_like(gate_up_bias)
        scattered_gate_up_bias[scatter_index.flatten()] = gate_up_bias
        gate_up += scattered_gate_up_bias

        # 拆分 Gate 和 Up 分量（新增逻辑）
        gate = gate_up[..., ::2]  # 取偶数列作为 gate
        up = gate_up[..., 1::2]   # 取奇数列作为 up

        # MOE Step 5: 应用修正的 SwiGLU 激活（新增逻辑）
        up = up.clamp(min=-limit, max=limit)            # up 双向截断
        gate = gate.clamp(min=None, max=limit)            # gate 单向截断
        gate_activation = gate * torch.sigmoid(gate * alpha)        # 带 alpha 缩放的 sigmoid
        gate_up_activation = (up + 1) * gate_activation                   # (up + 1) 修正项

        # MOE Step 8: compute the the weighted linear layer 1 result
        # MOE Step 8-1: compute scattered_gate_weight, shape is (batch_size * sequence_len * topk)
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        # MOE Step 8-2: multiply activate with scattered_gate_weight
        # fc1_weighted_output shape is (batch_size * sequence_len * topk, ffn_dim)
        gate_up_output = gate_up_activation

        down_output = group_gemm_same_nk(
            a=gate_up_output,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        down_output_bias = down_proj_bias.index_select(0, expert_index.flatten()).view(down_output.shape)
        scattered_down_output_bias = torch.empty_like(down_output_bias)
        scattered_down_output_bias[scatter_index.flatten()] = down_output_bias
        down_output += scattered_down_output_bias

        expert_output = moe_gather(down_output * scattered_gate_weight, scatter_index)  # 聚合 top-k 专家结果
        output = expert_output.reshape(hidden_states.shape)
    
        # 保存反向传播所需张量
        ctx.save_for_backward(
            gate_weights, 
            hidden_states, 
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            scatter_index,
            scatter_output,
            cumsum_t,
            gate_up,
            gate_up_output,
            gate_up_activation,
            scattered_gate_weight,
            expert_index,
            down_output,
        )
        ctx.alpha = alpha
        ctx.limit = limit
        ctx.num_experts = num_experts

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 恢复保存的张量和超参数
        (
            gate_weights, 
            hidden_states, 
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            scatter_index,
            scatter_output,
            cumsum_t,
            gate_up,
            gate_up_output,
            gate_up_activation,
            scattered_gate_weight,
            expert_index,
            down_output,
        ) = ctx.saved_tensors
        alpha = ctx.alpha
        limit = ctx.limit
        num_experts = ctx.num_experts
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # 反向传播核心逻辑（按前向步骤逆序）
        grad_down_output = moe_scatter(grad_output, scatter_index)  # Step 7 逆过程

        grad_scattered_gate_weight = torch.sum(grad_down_output * down_output, dim=-1)
        grad_down_output *= scattered_gate_weight

        # dgrad: 计算 gate_up_output 的梯度 (对应 down_proj 反向)
        grad_gate_up_weight_output = group_gemm_same_nk(
            a=grad_down_output,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=True,
        )

        # wgrad: 计算 down_proj 的梯度 (对应 gate_up_output 反向)
        grad_down_weight = None
        if down_proj.requires_grad:
            grad_down_weight = torch.empty_like(down_proj)
            group_gemm_same_mn(
                a=gate_up_output, # grad_down_output,
                b=grad_down_output, # gate_up_output,
                c=grad_down_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # SwiGLU 激活反向传播（Step 5 逆过程）
        # 2. 计算 gate 和 up 的梯度
        gate = gate_up[..., ::2]
        up = gate_up[..., 1::2]
        up_clamp = up.clamp(min=-limit, max=limit) 
        gate_clamped = gate.clamp(min=None, max=limit)
        sigmoid_gate_alpha = torch.sigmoid(gate_clamped * alpha)

        # 2.1 up 梯度 (含截断)
        grad_up = grad_gate_up_weight_output * gate_clamped * sigmoid_gate_alpha
        grad_up[up < -limit] = 0
        grad_up[up > limit] = 0

        # 2.2 gate 梯度 (含截断和 sigmoid 导数)
        grad_gate_clamped = grad_gate_up_weight_output * (up_clamp + 1) * sigmoid_gate_alpha
        grad_sigmoid = gate_clamped * grad_gate_up_weight_output * (up_clamp + 1)
        grad_gate = grad_sigmoid * (sigmoid_gate_alpha * (1 - sigmoid_gate_alpha) * alpha) + grad_gate_clamped
        grad_gate[gate > limit] = 0  # 截断区域梯度置零

        # 3. 合并 gate/up 梯度并计算 gate_up_proj 梯度
        grad_gate_up = torch.zeros_like(gate_up)
        grad_gate_up[..., ::2] = grad_gate
        grad_gate_up[..., 1::2] = grad_up

        grad_scatter_output = group_gemm_same_nk(
            a=grad_gate_up,
            b=gate_up_proj,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=True,
        )

        # 聚合梯度并恢复形状
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        # 计算权重梯度（gate_up_proj 和 down_proj）
        grad_gate_up_proj = None
        if gate_up_proj.requires_grad:
            grad_gate_up_proj = torch.empty_like(gate_up_proj)
            group_gemm_same_mn(
                a=scatter_output,
                b=grad_gate_up,
                c=grad_gate_up_proj,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # 偏置梯度（直接求和）
        grad_gate_up_proj_bias = torch.zeros_like(gate_up_proj_bias)
        grad_gate_up_proj_bias.index_add_(0, expert_index.flatten(), grad_gate_up[scatter_index.flatten()])

        grad_down_proj_bias = torch.zeros_like(down_proj_bias)
        grad_down_proj_bias.index_add_(0, expert_index.flatten(), grad_down_output[scatter_index.flatten()])

        # 路由权重梯度
        grad_gate_weights = grad_scattered_gate_weight[scatter_index.flatten()].reshape(gate_weights.shape)

        return (
            None,  # num_experts
            grad_gate_weights,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_gate_up_proj,  # gate_up_proj
            grad_gate_up_proj_bias,  # gate_up_proj_bias
            grad_down_weight,  # down_proj
            grad_down_proj_bias,  # down_proj_bias
            None,  # alpha（标量无梯度）
            None,  # limit（标量无梯度）
        )


def fused_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states,
    gate_up_proj,  # [num_experts, hidden, 2*ffn_dim]，交错布局
    gate_up_proj_bias,
    down_proj,  # [num_experts, ffn_dim, hidden]
    down_proj_bias,
    limit,
    alpha,
):
    routing_weights = routing_weights.bfloat16()
    hidden_states = hidden_states.bfloat16()
    final_hidden_states = FusedMoeExpertFunction.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        alpha,
        limit,
    )
    return final_hidden_states
