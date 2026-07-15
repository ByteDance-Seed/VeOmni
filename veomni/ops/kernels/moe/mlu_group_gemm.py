import torch
from apex.contrib.grouped_gemm.ops import backend as apex_backend

from ....distributed.moe import preprocess, token_pre_all2all, tokens_post_all2all
from ....distributed.moe.moe_layer import _apply_swiglu_clamp
from ....distributed.parallel_state import get_parallel_state

class MLUGroupGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        swiglu_limit=None,
    ):
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]
        # compute batch_sizes from cumsum (must be on CPU and int64 for apex backend.gmm)
        cumsum_cpu = cumsum.cpu().to(torch.int64)
        batch_sizes = torch.cat([cumsum_cpu[:1], cumsum_cpu[1:] - cumsum_cpu[:-1]])

        # compute linear layer fc1-1
        fc1_1_output = apex_backend.gmm(
            a=permute_tokens,
            b=fc1_1_weight,
            batch_sizes=batch_sizes,
            trans_a=False,
            trans_b=True,
        )

        # compute linear layer fc1-2
        fc1_2_output = apex_backend.gmm(
            a=permute_tokens,
            b=fc1_2_weight,
            batch_sizes=batch_sizes,
            trans_a=False,
            trans_b=True,
        )

        # gpt-oss / DeepSeek-V4 style clamped SwiGLU pre-activation. No-op when
        # swiglu_limit is None (legacy MoE models) — masks are None and the
        # ``if swiglu_limit is not None`` guards in backward are skipped.
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = _apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )

        # compute the actication of linear layer fc1-1
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # compute final result of linear layer fc1
        fc1_output = fc1_1_activation * fc1_2_output

        # weighted projection is outside this function
        # compute linear layer fc2
        fc2_output = apex_backend.gmm(
            a=fc1_output,
            b=fc2_weight,
            batch_sizes=batch_sizes,
            trans_a=False,
            trans_b=True,
        )

        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [tokens, hidden_dim]
        (
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]
        cumsum_cpu = cumsum.cpu().to(torch.int64)
        batch_sizes = torch.cat([cumsum_cpu[:1], cumsum_cpu[1:] - cumsum_cpu[:-1]])

        # dgrad fc1
        grad_fc1_output = apex_backend.gmm(
            a=grad_output,
            b=fc2_weight,
            batch_sizes=batch_sizes,
            trans_a=False,
            trans_b=False,
        )

        # recompute
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output

        # wgrad fc2: grad_output.T @ fc1_output
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = apex_backend.gmm(
                a=grad_output,
                b=fc1_output,
                batch_sizes=batch_sizes,
                trans_a=True,
                trans_b=False,
            )

        grad_fc1_2_output = fc1_1_activation * grad_fc1_output
        grad_fc1_1_activation = grad_fc1_output * fc1_2_output

        if swiglu_limit is not None:
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

        # dgrad output 2: grad_fc1_2_output @ fc1_2_weight
        grad_scatter_output_2 = apex_backend.gmm(
            a=grad_fc1_2_output,
            b=fc1_2_weight,
            batch_sizes=batch_sizes,
            trans_a=False,
            trans_b=False,
        )
        # wgrad fc1-2: grad_fc1_2_output.T @ permute_tokens
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = apex_backend.gmm(
                a=grad_fc1_2_output,
                b=permute_tokens,
                batch_sizes=batch_sizes,
                trans_a=True,
                trans_b=False,
            )

        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)

        # dgrad output 1: grad_fc1_1_output @ fc1_1_weight
        grad_scatter_output_1 = apex_backend.gmm(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            batch_sizes=batch_sizes,
            trans_a=False,
            trans_b=False,
        )

        # wgrad fc1-1: grad_fc1_1_output.T @ permute_tokens
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = apex_backend.gmm(
                a=grad_fc1_1_output,
                b=permute_tokens,
                batch_sizes=batch_sizes,
                trans_a=True,
                trans_b=False,
            )

        # grad input
        grad_permute_tokens = grad_scatter_output_1 + grad_scatter_output_2

        return (
            grad_permute_tokens,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
            None,  # swiglu_limit
        )


class MLUMergedFc1GroupGemm(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_2_weight,
        fc2_weight,
        swiglu_limit=None,
    ):
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]
        assert fc1_1_2_weight.shape[1] % 2 == 0, (
            f"Merged fc1_1_2_weight dim 1 must be even, got {fc1_1_2_weight.shape[1]}"
        )

        cumsum_cpu = cumsum.cpu().to(torch.int64)
        batch_sizes = torch.cat([cumsum_cpu[:1], cumsum_cpu[1:] - cumsum_cpu[:-1]])

        # Single fc1 gemm: output shape [T, 2I]
        fc1_output = apex_backend.gmm(
            a=permute_tokens,
            b=fc1_1_2_weight,
            batch_sizes=batch_sizes,
            trans_a=False,
            trans_b=True,
        )

        # chunk is a view, no copy
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)

        # gpt-oss / DeepSeek-V4 style clamped SwiGLU pre-activation. ``_apply_swiglu_clamp``
        # creates new tensors when ``swiglu_limit is not None`` so the saved halves are
        # independent of ``fc1_output`` storage; otherwise it is a no-op.
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = _apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )

        # compute the activation of linear layer fc1-1
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # compute final result of linear layer fc1
        fc1_result = fc1_1_activation * fc1_2_output

        # compute linear layer fc2
        fc2_output = apex_backend.gmm(
            a=fc1_result,
            b=fc2_weight,
            batch_sizes=batch_sizes,
            trans_a=False,
            trans_b=True,
        )

        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit

        # recompute
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output

        cumsum_cpu = cumsum.cpu().to(torch.int64)
        batch_sizes = torch.cat([cumsum_cpu[:1], cumsum_cpu[1:] - cumsum_cpu[:-1]])

        # dgrad fc2
        grad_fc1_result = apex_backend.gmm(
            a=grad_output,
            b=fc2_weight,
            batch_sizes=batch_sizes,
            trans_b=False,
        )

        # wgrad fc2
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = apex_backend.gmm(
                a=grad_output,
                b=fc1_result,
                batch_sizes=batch_sizes,
                trans_a=True,
                trans_b=False,
            )

        # gate gradients
        grad_fc1_2_output = fc1_1_activation * grad_fc1_result
        grad_fc1_1_activation = grad_fc1_result * fc1_2_output
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

        # Merge grads back to [T, 2I]
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)

        # single dgrad for merged fc1
        grad_permute_tokens = apex_backend.gmm(
            a=grad_fc1_output,
            b=fc1_1_2_weight,
            batch_sizes=batch_sizes,
            trans_b=False,
        )

        # single wgrad for merged fc1
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = apex_backend.gmm(
                a=grad_fc1_output,
                b=permute_tokens,
                batch_sizes=batch_sizes,
                trans_a=True,
                trans_b=False,
            )

        return (
            grad_permute_tokens,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_2_weight,  # fc1_1_2_weight
            grad_fc2_weight,  # fc2_weight
            None,  # swiglu_limit
        )


def mlu_group_gemm_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
):
    """Triton grouped-gemm fused MoE forward pass.

    Accepts either split fc1 weights (fc1_1_weight, fc1_2_weight) or a merged
    fc1_1_2_weight tensor.

    ``swiglu_limit``: gpt-oss / DeepSeek-V4 style clamp on the SwiGLU
    pre-activations (``gate.clamp(max=L)``, ``up.clamp(min=-L, max=L)``).
    ``None`` disables the clamp (default, zero overhead — used by every legacy
    MoE model).
    """
    if fc1_1_2_weight is not None:
        if fc1_1_weight is not None or fc1_2_weight is not None:
            raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
    if get_parallel_state().ep_enabled:
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        # preprocess, permute token for ep
        input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
            preprocess(
                expert_mask=expert_mask,
                num_experts=num_experts,
                ep_group=get_parallel_state().ep_group,
            )
        )
        permute_tokens, routing_map, local_input_permutation_mapping, org_hidden_states_shape = token_pre_all2all(
            hidden_states=hidden_states,
            expert_mask=expert_mask,
            num_experts=num_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
            ep_group=get_parallel_state().ep_group,
        )

        cumsum = torch.cumsum(num_global_sum_tokens_per_local_expert, dim=0)
    else:
        from apex.contrib.permute import permute
        from apex.contrib.unpermute import unpermute
        permute_tokens, sorted_indice = permute(
            hidden_states, selected_experts, -1
        ) 
        splits = torch.bincount(selected_experts.view(-1), minlength=num_experts)
        cumsum = torch.cumsum(splits, dim=0)

    if fc1_1_2_weight is not None:
        final_permute_tokens = MLUMergedFc1GroupGemm.apply(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            swiglu_limit,
        )
    else:
        final_permute_tokens = MLUGroupGemm.apply(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            swiglu_limit,
        )

    if get_parallel_state().ep_enabled:
        # unpermute with routing_weight
        final_hidden_states = tokens_post_all2all(
            expert_outputs=final_permute_tokens,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            num_experts=num_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
            routing_map=routing_map,
            local_input_permutation_mapping=local_input_permutation_mapping,
            org_hidden_states_shape=org_hidden_states_shape,
            ep_group=get_parallel_state().ep_group,
        )
    else:
        final_hidden_states = unpermute(
            final_permute_tokens, sorted_indice, routing_weights
        )
    return final_hidden_states
