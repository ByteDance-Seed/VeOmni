import torch
import torch.nn.functional as F

from ..group_gemm.kernel.moe import expert_histogram
from ..group_gemm.torch_moe_utils.utils import indices_padding_wrapper


def torch_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
):
    """
    torch._grouped_mm based fused moe forward using pure torch token reorder/combine.
    This path relies on native autograd end-to-end (no custom autograd.Function needed).
    """
    routing_weights = routing_weights.bfloat16()
    hidden_states = hidden_states.bfloat16()

    use_merged_fc1 = fc1_1_2_weight is not None
    if use_merged_fc1:
        if fc1_1_weight is not None or fc1_2_weight is not None:
            raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
    else:
        if fc1_1_weight is None or fc1_2_weight is None:
            raise ValueError("Split fc1 mode requires both fc1_1_weight and fc1_2_weight.")

    num_tokens_per_expert = expert_histogram(selected_experts, num_experts)

    # Sort flattened (token, topk slot) entries by expert id.
    token_indices_experts_sorted = torch.argsort(selected_experts.view(-1), stable=True)

    topk = selected_experts.shape[1]

    # Route tokens to grouped expert input order.
    routed_input = hidden_states[token_indices_experts_sorted // topk]
    top_scores_experts_sorted = routing_weights.view(-1)[token_indices_experts_sorted].reshape(-1, 1)

    if use_merged_fc1:
        run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm_merged_fc1)
        routed_outputs = run_experts_fn(
            fc1_1_2_weight,
            fc2_weight,
            routed_input,
            num_tokens_per_expert=num_tokens_per_expert,
            gate_weights=top_scores_experts_sorted,
        )
    else:
        run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
        routed_outputs = run_experts_fn(
            fc1_1_weight,
            fc2_weight,
            fc1_2_weight,
            routed_input,
            num_tokens_per_expert=num_tokens_per_expert,
            gate_weights=top_scores_experts_sorted,
        )

    # Unsort and reduce top-k expert outputs back to token order.
    routed_output_unsorted = torch.zeros(
        (hidden_states.shape[0] * topk, hidden_states.shape[1]),
        dtype=routed_outputs.dtype,
        device=routed_outputs.device,
    )
    routed_output_unsorted[token_indices_experts_sorted] = routed_outputs
    output = routed_output_unsorted.view(hidden_states.shape[0], topk, hidden_states.shape[1]).sum(dim=1)
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


def _run_experts_grouped_mm_merged_fc1(
    w1_2: torch.Tensor,
    w2: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor | None,
    gate_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h12 = torch._grouped_mm(x.bfloat16(), w1_2.bfloat16().transpose(-2, -1), offs=offsets)
    intermediate_dim = h12.shape[-1] // 2
    h = F.silu(h12[..., :intermediate_dim]) * h12[..., intermediate_dim:]
    if gate_weights is not None:
        h = h * gate_weights
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
    return out
