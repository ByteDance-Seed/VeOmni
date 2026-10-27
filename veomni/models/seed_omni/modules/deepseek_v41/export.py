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

"""Write a trained DeepSeek-V4.1 llm back in the release's own formats.

The export hook of the ``deepseek_v41`` HF layout (``export_state``). It
receives the llm's full state dict under module keys and returns the tensors
the release stores, still under module keys -- the layout renames them back:

* a dense weight the release ships in fp8 is requantized in the release's
  blocks for it (``(32, 32)`` today), and its ue8m0 scale is returned as
  ``<weight>_scale``;
* each fused expert slab is split into the release's per-expert ``w1`` /
  ``w3`` (the two halves of ``gate_up_proj``) and ``w2``, each requantized to
  packed fp4 with its ``scale``;
* everything else passes through, and the export casts it between float
  dtypes as needed.

Under FSDP and EP the work is spread so that no rank requantizes the model: a
dense weight is gathered on every rank but requantized and written by one,
chosen round-robin by key; an expert slab is gathered only over the mesh
dimensions that do not split its experts, and each rank splits its own
experts, one replica per slab doing the work. A tensor returned as a plain
tensor is written by the rank that returns it.
"""

from __future__ import annotations

import re
from typing import Mapping

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from .....models.checkpoint.convert import split_moe_params
from .....models.transformers.deepseek_v4.checkpoint_tensor_converter import (
    validate_deepseek_v4_export_quant_support,
)
from .dequant import quant_fp4_blockwise, quant_fp8_blockwise
from .llm.configuration import RELEASED_NUM_HIDDEN_LAYERS
from .released_names import module_scale_name


_FUSED_EXPERT_KEY = re.compile(
    r"^(?P<prefix>model\.(?:layers|mtp)\.\d+\.mlp\.experts)\.(?P<fused>gate_up_proj|down_proj)$"
)


def _rank_and_world() -> tuple[int, int]:
    if not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _local_experts(tensor: torch.Tensor, slab_index: int) -> tuple[torch.Tensor | None, int]:
    """This rank's experts of a fused slab and the global index of the first, or ``None`` when another replica owns them.

    A DTensor keeps its dim-0 sharding -- the expert one under EP, FSDP's own
    without EP, which splits the experts just the same -- and is gathered over
    every other mesh dimension; of the replicas that then hold the same
    experts, slab ``slab_index`` goes to replica ``slab_index % replicas``.
    """
    if not isinstance(tensor, DTensor):
        rank, _ = _rank_and_world()
        return (tensor if rank == 0 else None), 0
    mesh = tensor.device_mesh
    expert_dims = [i for i, placement in enumerate(tensor.placements) if placement == Shard(0) and mesh.size(i) > 1]
    if len(expert_dims) > 1:
        # The checkpointer composes (ep_fsdp, ep) on the same dim in the opposite
        # order to the physical one (Muon zero-comm), so the global view is
        # block-transposed and no redistribute recovers the expert order.
        raise NotImplementedError(
            f"export_llm_state: experts split on dim 0 by several mesh dims {tuple(tensor.placements)} "
            f"(e.g. train.optimizer.muon_expert_zero_comm) cannot be exported to the release."
        )
    kept = expert_dims[0] if expert_dims else None
    placements = [Shard(0) if i == kept else Replicate() for i in range(mesh.ndim)]
    local = tensor.redistribute(mesh, placements).to_local()
    replica, replicas = 0, 1
    for i in range(mesh.ndim):
        if i == kept:
            continue
        replica = replica * mesh.size(i) + mesh.get_local_rank(i)
        replicas *= mesh.size(i)
    if slab_index % replicas != replica:
        return None, 0
    if kept is None:
        return local, 0
    # Shard(0) splits as torch.chunk does: ceil-sized chunks, the last ones short or empty.
    chunk = -(-tensor.shape[0] // mesh.size(kept))
    return local, mesh.get_local_rank(kept) * chunk


def _split_experts(prefix: str, fused: str, experts: torch.Tensor, start: int, block: int) -> dict[str, torch.Tensor]:
    """``[E_local, out, in]`` -> the release's per-expert tensors, requantized to fp4 in ``block`` columns."""
    out: dict[str, torch.Tensor] = {}
    halves = ("w1", "w3") if fused == "gate_up_proj" else ("w2",)
    for local in range(experts.shape[0]):
        parts = experts[local].chunk(len(halves), dim=0)
        for proj, weight in zip(halves, parts, strict=True):
            packed, scale = quant_fp4_blockwise(weight, block)
            out[f"{prefix}.{start + local}.{proj}.weight"] = packed.cpu()
            out[f"{prefix}.{start + local}.{proj}.scale"] = scale.cpu()
    return out


def _split_packed_experts(
    prefix: str, fused: str, packed: torch.Tensor, scale: torch.Tensor, start: int
) -> dict[str, torch.Tensor]:
    """A quantized module's slabs are already in the release format: only split them."""
    out: dict[str, torch.Tensor] = {}
    halves = ("w1", "w3") if fused == "gate_up_proj" else ("w2",)
    for local in range(packed.shape[0]):
        weights = packed[local].chunk(len(halves), dim=0)
        scales = scale[local].chunk(len(halves), dim=0)
        for proj, weight, weight_scale in zip(halves, weights, scales, strict=True):
            out[f"{prefix}.{start + local}.{proj}.weight"] = weight.contiguous().cpu()
            out[f"{prefix}.{start + local}.{proj}.scale"] = weight_scale.contiguous().cpu()
    return out


def export_llm_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    fp8_blocks: Mapping[str, tuple[int, int]],
    fp4_block: int,
) -> dict[str, torch.Tensor]:
    """The release-format tensors of ``state``.

    ``fp8_blocks`` maps each module key the release stores in fp8 to its
    ``(rows, cols)`` block, and ``fp4_block`` is the experts' column block,
    both as the release's own scales have them.

    Refuses a shrunk stack: its layers are a selection of the release's, and
    exporting it as the release would write a model with the wrong depth.
    """
    config = model.config
    if config.num_hidden_layers != RELEASED_NUM_HIDDEN_LAYERS:
        raise NotImplementedError(
            f"Exporting a {config.num_hidden_layers}-layer DeepSeek-V4.1 llm in the release layout is not "
            f"supported: the release has {RELEASED_NUM_HIDDEN_LAYERS} layers, and a shrunk stack is a "
            "selection of them. Export the full model, or save the split checkpoint instead."
        )
    rank, world = _rank_and_world()
    out: dict[str, torch.Tensor] = {}
    dense_index = slab_index = 0
    for key in sorted(state):
        tensor = state[key]
        if key.endswith("_scale") and key[: -len("_scale")] in state:
            continue
        match = _FUSED_EXPERT_KEY.match(key)
        if match is not None:
            local, start = _local_experts(tensor, slab_index)
            scale = state.get(module_scale_name(key))
            local_scale = _local_experts(scale, slab_index)[0] if scale is not None else None
            slab_index += 1
            if local is None:
                continue
            if local_scale is not None:
                out.update(_split_packed_experts(match["prefix"], match["fused"], local, local_scale, start))
            else:
                out.update(_split_experts(match["prefix"], match["fused"], local, start, fp4_block))
            continue
        if key not in fp8_blocks:
            out[key] = tensor
            continue
        scale = state.get(module_scale_name(key))
        full = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
        full_scale = scale.full_tensor() if isinstance(scale, DTensor) else scale
        owner = dense_index % world
        dense_index += 1
        if rank != owner:
            continue
        if full_scale is not None:
            weight, weight_scale = full, full_scale
        else:
            weight, weight_scale = quant_fp8_blockwise(full, fp8_blocks[key])
        out[key] = weight.cpu()
        out[module_scale_name(key)] = weight_scale.cpu()
    return out


def refuse_ngram_export(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The n-gram tables are frozen by design; a trained copy has no release-format export yet."""
    raise NotImplementedError(
        "Exporting a trained deepseek_v41_ngram in the release layout is not supported: its two tables are "
        "91.6 GiB each and would have to be requantized row block by row block across the embedding-parallel "
        "ranks. Keep the ngram module frozen (the export then copies the released tables), or save the split "
        "checkpoint."
    )


__all__ = [
    "export_llm_state",
    "refuse_ngram_export",
    "split_moe_params",
    "validate_deepseek_v4_export_quant_support",
]
