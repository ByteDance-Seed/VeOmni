from typing import List

import torch
import torch.distributed as dist

from .modeling_seed_omni import SeedOmniModel


def get_embed_token_grad_mask_func(
    model: SeedOmniModel, trained_embedding_indices: List[int], global_rank: int, world_size: int
):
    embed_dim = model.encoder.text_encoder.weight.shape[-1]
    train_indices = []
    for indice in trained_embedding_indices:
        train_indices.append((indice * embed_dim, indice * embed_dim + embed_dim))

    def mask_embedding_grad(embedding: torch.nn.Embedding):
        grad = embedding.weight.grad
        device = embedding.weight.device
        if grad is None:
            local_numel = torch.tensor([0], device=device)
        else:
            local_numel = torch.tensor([grad.numel()], device=device)

        all_numel = [torch.zeros_like(local_numel) for _ in range(world_size)]
        dist.all_gather(all_numel, local_numel)

        if grad is None:
            return
        shard_sizes = [x.item() for x in all_numel]
        offsets = [0]
        for sz in shard_sizes:
            offsets.append(offsets[-1] + sz)

        start_offset = offsets[global_rank]
        end_offset = offsets[global_rank + 1]

        local_mask = torch.zeros_like(grad, dtype=torch.bool, device=device)
        for start, end in train_indices:
            if end >= start_offset and start < end_offset:
                local_mask[max(start, start_offset) - start_offset : min(end, end_offset) - start_offset] = True
        grad.mul_(local_mask)

    return mask_embedding_grad
