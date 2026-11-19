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
"""Opt-in switches that adopt NeMo AutoModel's training semantics.

Two places where the two DiffusionGemma stacks compute the same *function* but hand
the optimizer a different problem, so their first step already diverges. Both are
off by default -- VeOmni's own semantics are unchanged unless the environment asks
otherwise -- and both exist to attribute a training difference between the stacks,
not to ship. Constraint 43 records the parity these switches close.

``VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD=1``
    Fold ``router.per_expert_scale`` into ``experts.down_proj`` at load time and
    freeze the scale at one, which is what NeMo's state-dict adapter does. The
    forward is unchanged (``s * (W @ h)`` and ``(s * W) @ h`` agree); what changes
    is that the optimizer now sees one product matrix instead of a matrix and a
    multiplier, and AdamW's per-coordinate step, weight decay and clipping all read
    that organization. See ``fold_per_expert_scale``.

``VEOMNI_ALIGN_NEMO_SELF_COND_EMBED_GRAD=1``
    Let the self-conditioning soft embeddings keep a gradient path to the embedding
    table. Both stacks stop the final loss from reaching pass 1, but they cut at
    different points: VeOmni freezes the whole ``u = p @ E``, NeMo freezes only the
    prediction ``p`` and recomputes ``p @ E`` inside the differentiable pass. See
    ``nemo_aligned_self_conditioning_embeddings``.
"""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from ....utils import logging


logger = logging.get_logger(__name__)


def fold_expert_scale_enabled() -> bool:
    """Whether to adopt NeMo's folded expert parameterization. See the module docstring."""
    return os.environ.get("VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD", "0") == "1"


def self_cond_embed_grad_enabled() -> bool:
    """Whether self-conditioning may update the embedding table. See the module docstring."""
    return os.environ.get("VEOMNI_ALIGN_NEMO_SELF_COND_EMBED_GRAD", "0") == "1"


@torch.no_grad()
def fold_per_expert_scale(model: nn.Module) -> int:
    """Rewrite every expert as ``W' = s * W`` with ``s`` frozen at one.

    Returns the number of expert tensors folded, so a caller can assert the switch
    reached something. A no-op, returning zero, unless the switch is on.

    Must run after the weights land and before the optimizer is built: it is the
    parameter list that this changes, and an optimizer already holding
    ``per_expert_scale`` would carry a moment for a tensor that no longer moves.

    Encoder and decoder share one transformer stack, so the same ``down_proj``
    object is reachable from two layer modules; the fold is keyed on tensor
    identity to avoid squaring the scale on the shared pair. The multiply runs at no
    less than fp32 and casts back, which is one rounding rather than the two a bf16
    product written to an fp32 parameter would take -- the defect constraint 44
    records on the NeMo side.

    Works on a sharded model; ``_fold_one`` carries the sharding argument.
    """
    if not fold_expert_scale_enabled():
        return 0

    plan = model.get_parallel_plan() if hasattr(model, "get_parallel_plan") else None

    # Keyed on *storage*, collecting every name each tensor answers to. Both matter.
    #
    # Encoder and decoder share one transformer stack, so the same weights are reachable
    # from two layer modules and folding per module would square the scale -- silently,
    # since the forward stays self-consistent and only the value is wrong. Object
    # identity is the obvious key and is not enough: FSDP2 can rebind a parameter it has
    # all-gathered and re-sharded, leaving two objects over one buffer, so `is` starts
    # answering no to a pair that is still one tensor. The address answers the question
    # that matters, which is whether writing through one is visible through the other.
    #
    # The names are what let the parallel plan recognize the tensor; it knows only the
    # decoder's spelling.
    folded: dict[int, tuple[torch.Tensor, torch.Tensor, list[str]]] = {}
    for name, module in model.named_modules():
        router = getattr(module, "router", None)
        experts = getattr(module, "experts", None)
        if router is None or experts is None:
            continue
        scale = getattr(router, "per_expert_scale", None)
        down_proj = getattr(experts, "down_proj", None)
        if not isinstance(scale, torch.Tensor) or not isinstance(down_proj, torch.Tensor):
            continue
        storage = (down_proj.to_local() if isinstance(down_proj, DTensor) else down_proj).data_ptr()
        seen = folded.setdefault(storage, (down_proj, scale, []))
        kept = seen[1]
        kept_storage = (kept.to_local() if isinstance(kept, DTensor) else kept).data_ptr()
        scale_storage = (scale.to_local() if isinstance(scale, DTensor) else scale).data_ptr()
        if kept_storage != scale_storage:
            raise RuntimeError(
                f"`{name}` shares its experts with another layer but not its `per_expert_scale`; "
                "folding would apply one of the two scales to weights the other also owns."
            )
        seen[2].append(f"{name}.experts.down_proj")

    for down_proj, scale, fqns in folded.values():
        _fold_one(down_proj, scale, fqns, plan)
        scale.fill_(1.0)
        scale.requires_grad_(False)

    if folded:
        logger.info_rank0(
            f"DiffusionGemma: folded `per_expert_scale` into {len(folded)} expert down-projection(s) and "
            "froze the scale, adopting NeMo AutoModel's parameterization "
            "(VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD=1)."
        )
    return len(folded)


def _local_expert_scale(down_proj: torch.Tensor, scale: torch.Tensor, fqns, plan) -> torch.Tensor:
    """The slice of ``scale`` that lines up with this rank's slice of ``down_proj``.

    Shaped to broadcast over ``down_proj``'s local shard. Neither sharding is
    reconstructed here, on purpose: a shard offset recomputed a second way is one
    that can disagree with the first, and a fold that pairs expert ``e``'s weights
    with expert ``e'``'s scale produces a perfectly self-consistent forward. Nothing
    downstream would report it; the run would just train a different model. So both
    splits are delegated to whoever already owns them.

    The scale is one float per expert, small enough to replicate outright, so it
    starts whole. Expert parallel's cut comes from ``ParallelPlan.shard_tensor`` --
    the *same* call, on the same plan, under the same parameter name that the loader
    used to cut ``down_proj`` itself. It cannot select a different block than the
    weights got without also having loaded the weights wrong. FSDP's cut on top
    comes from ``compute_local_shape_and_global_offset``, so DTensor's own sharding
    rule states which experts are local rather than this guessing it.
    """
    whole = scale.full_tensor() if isinstance(scale, DTensor) else scale
    whole = whole.detach().reshape(-1)

    experts_here = down_proj.shape[0]
    if whole.numel() != experts_here:
        if plan is None:
            raise RuntimeError(
                f"`down_proj` holds {experts_here} of `per_expert_scale`'s {whole.numel()} experts, so the "
                "experts are sharded, but the model exposes no parallel plan to say how."
            )
        # The plan keys on the *decoder* path; the encoder reaches the same tensor
        # under a name it does not recognize, which would silently decline to slice.
        sharded = [fqn for fqn in fqns if plan._get_shard_parameter_groupname(fqn)]
        if not sharded:
            raise RuntimeError(
                f"`down_proj` holds {experts_here} of `per_expert_scale`'s {whole.numel()} experts, but none "
                f"of its names {sorted(fqns)} is in the parallel plan, so there is nothing to say which "
                "experts are local."
            )
        whole = plan.shard_tensor(whole, sharded[0], tuple(down_proj.shape))
        if whole.numel() != experts_here:
            # `_slice_shard_tensor` returns the tensor untouched when it cannot slice,
            # including on an internal error it has swallowed.
            raise RuntimeError(
                f"the parallel plan did not reduce `per_expert_scale` to this rank's {experts_here} experts "
                f"(it returned {whole.numel()}); folding it would pair the wrong experts."
            )

    shaped = whole.reshape(-1, 1, 1).to(device=down_proj.device, dtype=down_proj.dtype)
    if isinstance(down_proj, DTensor):
        # Only the expert dimension is being indexed, and FSDP does not necessarily
        # shard it: VeOmni gives DiffusionGemma's experts `Shard(1)` on the ep_fsdp
        # mesh, so each rank holds *every* local expert and a slice of the
        # intermediate dimension instead. Asking `distribute_tensor` for the whole
        # placement therefore tried to spread a length-1 axis over the mesh, which
        # left the ranks holding different shapes -- some raised, the rest waited on
        # a collective that was never issued, and the fold deadlocked.
        local_shape, offsets = compute_local_shape_and_global_offset(
            down_proj.shape, down_proj.device_mesh, down_proj.placements
        )
        shaped = shaped[offsets[0] : offsets[0] + local_shape[0]]
    return shaped


def _fold_one(down_proj: torch.Tensor, scale: torch.Tensor, fqns, plan) -> None:
    """``down_proj[e] *= scale[e]`` on whatever slice of the experts is local.

    There is no numeric post-condition here, and that is deliberate rather than an
    omission. The obvious one -- each expert's norm must grow by its own scale -- is
    a tautology: it would recompute the expectation from the very vector it just
    multiplied by, so ``||W_e * s|| == |s| * ||W_e||`` holds for any ``s``, right or
    wrong. What guards the pairing is structural, in ``_local_expert_scale``.
    """
    local_scale = _local_expert_scale(down_proj, scale, fqns, plan)
    # A DTensor reports its global shape, so `down_proj` has to be brought down to the
    # shard before the two are compared; otherwise this reads 8 experts against a
    # 2-expert shard and, worse, compares equal in the unsharded cases.
    local = down_proj.to_local() if isinstance(down_proj, DTensor) else down_proj
    if local_scale.shape[0] != local.shape[0]:
        raise RuntimeError(
            f"the scale shard has {local_scale.shape[0]} experts against `down_proj`'s {local.shape[0]}; "
            "the two are sharded differently and multiplying them would broadcast across experts."
        )
    if local.numel() == 0:
        return
    # `promote_types` rather than `.float()`: the intent is *at least* fp32, and
    # `.float()` would narrow an fp64 parameter on the way through.
    wide = torch.promote_types(local.dtype, torch.float32)
    local.copy_((local.to(wide) * local_scale.to(wide)).to(local.dtype))


class _SelfConditioningEmbeddings(torch.autograd.Function):
    """``softmax(lm_head(h)) @ E * embed_scale`` differentiable in ``E`` alone.

    ``h`` and the head arrive detached -- pass 1 is frozen on both stacks -- so the
    only gradient to return is the embedding table's, and the probabilities need not
    be kept for it: the backward walks the same token chunks and recomputes them.
    That is what lets this match NeMo's semantics at VeOmni's memory, instead of
    retaining the ``[tokens, vocab]`` fp32 tensor a plain autograd graph would save.
    """

    @staticmethod
    def forward(ctx, hidden, head_weight, embedding_weight, embed_scale, softcapping, chunk_size):
        ctx.save_for_backward(hidden, head_weight, embed_scale)
        ctx.softcapping = softcapping
        ctx.chunk_size = chunk_size
        chunks = [
            torch.matmul(_probabilities(chunk, head_weight, softcapping).to(embedding_weight.dtype), embedding_weight)
            for chunk in torch.split(hidden, chunk_size, dim=0)
        ]
        embeddings = torch.cat(chunks, dim=0)
        return embeddings * embed_scale.to(device=embeddings.device, dtype=embeddings.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, head_weight, embed_scale = ctx.saved_tensors
        if not ctx.needs_input_grad[2]:
            return None, None, None, None, None, None
        grad_output = grad_output * embed_scale.to(device=grad_output.device, dtype=grad_output.dtype)
        grad_embedding = None
        for chunk, grad_chunk in zip(
            torch.split(hidden, ctx.chunk_size, dim=0),
            torch.split(grad_output, ctx.chunk_size, dim=0),
            strict=True,
        ):
            probabilities = _probabilities(chunk, head_weight, ctx.softcapping).to(grad_chunk.dtype)
            contribution = torch.matmul(probabilities.transpose(0, 1), grad_chunk)
            grad_embedding = contribution if grad_embedding is None else grad_embedding.add_(contribution)
        return None, None, grad_embedding, None, None, None


def _probabilities(hidden: torch.Tensor, head_weight: torch.Tensor, softcapping: float | None) -> torch.Tensor:
    logits = torch.nn.functional.linear(hidden, head_weight).float()
    if softcapping is not None:
        logits = torch.tanh(logits / softcapping) * softcapping
    return logits.softmax(dim=-1)


def nemo_aligned_self_conditioning_embeddings(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    embedding_weight: torch.Tensor,
    embed_scale: torch.Tensor,
    softcapping: float | None = None,
    logit_elements_per_chunk: int = 2**26,
) -> torch.Tensor:
    """``chunked_self_conditioning_embeddings`` that the embedding table can learn from.

    Same value, and the same bound on the temporary logit slice. The difference is
    where the stop-gradient sits: the caller runs pass 1 under ``no_grad``, and this
    re-enables grad for the ``p @ E`` product alone, leaving pass 1's prediction
    frozen. NeMo reaches the same place from the other side -- it detaches pass 1's
    logits and recomputes the product inside pass 2's differentiable decode.

    The published formulation does not fix this boundary, so this is a choice between
    two readings rather than a correction of one.
    """
    if hidden_states.shape[-1] != head_weight.shape[-1]:
        raise ValueError(f"hidden size {hidden_states.shape[-1]} does not match LM head {tuple(head_weight.shape)}.")
    if head_weight.shape != embedding_weight.shape:
        raise ValueError(
            f"LM head and embedding weights must have the same shape, got "
            f"{tuple(head_weight.shape)} and {tuple(embedding_weight.shape)}."
        )
    if logit_elements_per_chunk < 1:
        raise ValueError("`logit_elements_per_chunk` must be positive.")

    leading_shape = hidden_states.shape[:-1]
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).detach()
    chunk_size = max(1, logit_elements_per_chunk // head_weight.shape[0])
    # The reshape has to stay inside: a view taken under the caller's `no_grad` keeps
    # `requires_grad` but loses `grad_fn`, which severs the path this exists to hold
    # open and does so silently -- the gradient then reads as exactly zero.
    with torch.enable_grad():
        embeddings = _SelfConditioningEmbeddings.apply(
            flat_hidden,
            head_weight.detach(),
            embedding_weight,
            embed_scale,
            softcapping,
            chunk_size,
        )
        return embeddings.reshape(*leading_shape, embedding_weight.shape[-1])
