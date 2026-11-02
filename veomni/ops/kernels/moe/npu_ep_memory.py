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

"""NPU balanced GMM storage and canonical owner weight gradients."""

import torch
import torch.distributed as dist
import torch_npu


def _owner_token_counts(host_counts, plan, rank):
    """Recover original owner demand without modifying physical execution counts."""
    counts = host_counts[: plan.num_local_experts].to(dtype=torch.int64).clone()
    for replica in plan.replicas:
        if replica.source_rank == rank:
            counts[replica.source_local_expert] += replica.moved_tokens
    return counts


def restore_owner_wgrad_inputs(tokens, grad_output, plan, group, device_counts=None):
    """Return replica prefixes to owners in the original logical token order.

    The planner allocates successive global sender-major prefixes, in replica
    order. Combining independently rounded partial wgrads cannot preserve the
    native OFF GMM's reduction, even when the partials themselves use FP32.
    """
    counts = plan.tokens_per_local_physical_expert.sum(0).tolist()
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    rank = dist.get_rank(group)
    received = {}
    retained = []
    operations = []
    for replica in plan.replicas:
        if rank == replica.target_rank:
            slot = plan.num_local_experts + replica.target_slot
            start, end = offsets[slot : slot + 2]
            rows = torch.cat((tokens[start:end], grad_output[start:end]), dim=1).contiguous()
            retained.append(rows)
            operations.append(
                dist.P2POp(dist.isend, rows, dist.get_global_rank(group, replica.source_rank), group=group)
            )
        elif rank == replica.source_rank:
            rows = tokens.new_empty((replica.moved_tokens, tokens.shape[1] + grad_output.shape[1]))
            received.setdefault(replica.source_local_expert, []).append(rows)
            operations.append(
                dist.P2POp(dist.irecv, rows, dist.get_global_rank(group, replica.target_rank), group=group)
            )
    pending = dist.batch_isend_irecv(operations) if operations else []
    for work in pending:
        work.wait()

    owner_tokens, owner_grad, owner_counts = [], [], []
    for expert in range(plan.num_local_experts):
        start, end = offsets[expert : expert + 2]
        prefixes = received.get(expert, ())
        owner_tokens.extend(rows[:, : tokens.shape[1]] for rows in prefixes)
        owner_grad.extend(rows[:, tokens.shape[1] :] for rows in prefixes)
        owner_tokens.append(tokens[start:end])
        owner_grad.append(grad_output[start:end])
        owner_counts.append(end - start + sum(rows.shape[0] for rows in prefixes))
    return (
        torch.cat(owner_tokens, dim=0),
        torch.cat(owner_grad, dim=0),
        device_counts
        if device_counts is not None
        else torch.tensor(owner_counts, dtype=torch.int64, device=tokens.device),
    )


class _BalancedGroupGemm(torch.autograd.Function):
    """Keep native wgrad and owner restoration inside one autograd boundary."""

    @staticmethod
    def forward(ctx, tokens, local_weight, replica_weight, counts, host_counts, plan, group, transfer, memory, gemm):
        ctx.plan, ctx.group = plan, group
        # Legacy backward consumes the saved combined copy. Only Memory needs
        # the borrowed replica storage and its owner for the graph lifetime.
        ctx.transfer = transfer if memory else None
        ctx.memory, ctx.gemm = memory, gemm
        ctx.local_tokens = int(host_counts[: plan.num_local_experts].sum())
        host_owner_counts = owner_counts = None
        if ctx.needs_input_grad[1]:
            # These immutable counts are known before backward. Upload them
            # here rather than synchronizing after its P2P and activation cats.
            host_owner_counts = _owner_token_counts(host_counts, plan, dist.get_rank(group))
            if tokens.device.type == "npu":
                host_owner_counts = host_owner_counts.pin_memory()
            owner_counts = host_owner_counts.to(tokens.device, non_blocking=True)
        combined = None if memory else torch.cat((local_weight, replica_weight))
        # Retain the CPU source along with its asynchronous device copy.
        ctx.save_for_backward(
            tokens, local_weight, replica_weight if memory else None, counts, combined, host_owner_counts, owner_counts
        )
        # The enclosing MLP records both waited transfers in one finally block,
        # including when the first recording fails.
        if memory:
            return segmented_group_gemm(tokens, local_weight, replica_weight, counts, ctx.local_tokens, gemm)
        return (
            gemm(tokens, combined.transpose(1, 2), counts)
            if tokens.shape[0]
            else tokens.new_empty((0, local_weight.shape[1]))
        )

    @staticmethod
    def backward(ctx, grad_output):
        tokens, local_weight, replica_weight, counts, combined, _host_owner_counts, device_owner_counts = (
            ctx.saved_tensors
        )
        grad_tokens = grad_weight = None
        try:
            if ctx.needs_input_grad[0]:
                if ctx.memory:
                    grad_tokens = segmented_group_gemm(
                        grad_output,
                        local_weight.transpose(1, 2),
                        replica_weight.transpose(1, 2),
                        counts,
                        ctx.local_tokens,
                        ctx.gemm,
                    )
                else:
                    grad_tokens = (
                        ctx.gemm(grad_output, combined, counts) if tokens.shape[0] else torch.empty_like(tokens)
                    )
            if ctx.needs_input_grad[1]:
                owner_tokens, owner_grad, owner_counts = restore_owner_wgrad_inputs(
                    tokens, grad_output, ctx.plan, ctx.group, device_counts=device_owner_counts
                )
                if owner_tokens.shape[0]:
                    grad_weight = torch_npu.npu_grouped_matmul(
                        [owner_tokens.T],
                        [owner_grad],
                        bias=None,
                        group_list=owner_counts,
                        split_item=3,
                        group_type=2,
                        group_list_type=1,
                    )[0].transpose(1, 2)
                else:
                    grad_weight = torch.zeros_like(local_weight)
        finally:
            if ctx.transfer is not None:
                ctx.transfer.record_stream_use()
        return grad_tokens, grad_weight, None, None, None, None, None, None, None, None


def balanced_group_gemm(
    tokens, local_weight, replica_weight, counts, host_counts, plan, group, transfer, memory, gemm
):
    return _BalancedGroupGemm.apply(
        tokens, local_weight, replica_weight, counts, host_counts, plan, group, transfer, memory, gemm
    )


class _EmptyGroupGemm(torch.autograd.Function):
    """Keep zero-token groups in autograd without invoking an empty native GMM."""

    @staticmethod
    def forward(ctx, tokens, weight):
        ctx.layouts = [(tensor.shape, tensor.dtype, tensor.device) for tensor in (tokens, weight)]
        return tokens.new_empty((0, weight.shape[-1]))

    @staticmethod
    def backward(ctx, grad_output):
        return tuple(
            torch.zeros(shape, dtype=dtype, device=device) if needed else None
            for (shape, dtype, device), needed in zip(ctx.layouts, ctx.needs_input_grad, strict=True)
        )


def segmented_group_gemm(tokens, local_weight, replica_weight, counts, local_tokens, group_gemm):
    """Execute original and replica token segments using the existing NPU kernel."""
    groups = local_weight.shape[0]
    outputs = []
    for segment, weight, segment_counts in (
        (tokens[:local_tokens], local_weight, counts[:groups]),
        (tokens[local_tokens:], replica_weight, counts[groups:]),
    ):
        transposed_weight = weight.transpose(1, 2)
        if segment.shape[0]:
            outputs.append(group_gemm(segment, transposed_weight, segment_counts))
        else:
            outputs.append(_EmptyGroupGemm.apply(segment, transposed_weight))
    return torch.cat(outputs, dim=0)
