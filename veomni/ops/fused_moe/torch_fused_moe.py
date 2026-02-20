import torch
import torch.nn.functional as F

from ..group_gemm.kernel.moe import expert_histogram, moe_gather, moe_scatter
from ..group_gemm.torch_moe_utils.utils import indices_padding_wrapper


def torch_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,  # expert_index for each token
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
):
    # TODO: support EP
    routing_weights = routing_weights.bfloat16()
    hidden_states = hidden_states.bfloat16()
    # Prepare inputs for Grouped GEMM
    # we assume top-k has been performed before this func
    # num_tokens_per_expert = torch.histc(selected_experts.view(-1), bins=num_experts, min=0, max=num_experts)
    num_tokens_per_expert = expert_histogram(selected_experts, num_experts)

    # Reorder the token indices to match the order of the experts
    # token_indices_experts_sorted shape (bs*slen*top_k,)
    # token_indices_experts_sorted = torch.argsort(selected_experts.view(-1), stable=True)
    token_indices_experts_sorted = (
        selected_experts.flatten().argsort(stable=True).argsort().int().view(selected_experts.shape)
    )

    # select tokens by scatter_index, and put them together
    # token_indices_experts_sorted shape (batch_size * sequence_len * topk, hidden_size)
    # token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1, 1).expand(-1, hidden_states.size(-1))
    # routed_input = torch.gather(hidden_states, dim=0, index=token_indices_experts_sorted)
    routed_input = moe_scatter(hidden_states, token_indices_experts_sorted)

    reshaped_gate_weight = routing_weights.reshape(-1, 1)
    scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
    scattered_gate_weight[token_indices_experts_sorted.flatten()] = reshaped_gate_weight

    # padd inputs for alignment
    run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)

    # _run_experts_grouped_mm expects (w1, w2, w3) where w2 is the output proj.
    routed_outputs = run_experts_fn(
        fc1_1_weight,
        fc2_weight,
        fc1_2_weight,
        routed_input,
        num_tokens_per_expert=num_tokens_per_expert,
        gate_weights=scattered_gate_weight,
    )

    output = moe_gather(routed_outputs, token_indices_experts_sorted)
    output = output.reshape(hidden_states.shape)

    return output


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor | None,
    gate_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = F.silu(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
    h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
    if gate_weights is not None:
        h = h * gate_weights
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out
