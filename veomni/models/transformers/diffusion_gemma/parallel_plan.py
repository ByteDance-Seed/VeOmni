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
"""Expert-parallel plan for DiffusionGemma.

Shards the fused MoE expert weights over the ``ep`` group with the same layout as
Qwen3-MoE: ``gate_up_proj`` ``[E, 2*I, H]`` and ``down_proj``
``[E, H, I]``, both sharded on the expert dimension.

DiffusionGemma needs one extra step that no other VeOmni model does. Its encoder and
decoder *share* every transformer parameter, and ``ParallelPlan.apply`` walks the
deduplicated ``model.named_parameters()``. Only the decoder path is yielded
(``DiffusionGemmaModel`` registers ``self.decoder`` first so that the canonical names
match the checkpoint), and the encoder's identical tensor is skipped as a duplicate. But
``apply`` shards by rebinding a *new* parameter object at that FQN, which leaves the
encoder pointing at the unsharded ``[E, ...]`` tensor: the tie is broken, and the orphan
is not covered by ``fqn2spec_info``, so it would be missed by the extra-parallel grad-norm
grouping too. Restoring the tie is therefore not optional -- it happens in ``post_shard``,
which runs while ``apply`` can still see the result.
"""

from torch import nn
from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan
from ....distributed.parallel_state import get_parallel_state
from ....utils import logging


logger = logging.get_logger(__name__)


def _sharing_enabled(model: nn.Module) -> bool:
    """Whether encoder/decoder sharing is allowed by ``config.tie_word_embeddings``."""
    config = getattr(model, "config", None)
    return bool(getattr(config, "tie_word_embeddings", True))


def retie_encoder_to_decoder(model: nn.Module) -> int:
    """Re-point the encoder's shared parameters at the decoder's (possibly sharded) ones.

    Returns the number of parameters that had to be re-tied, so callers can assert the
    tie was actually restored.

    ``tie_word_embeddings=False`` disables every encoder/decoder tie, so a model loaded
    with independently trained encoder weights must keep them.
    """
    if not _sharing_enabled(model):
        return 0

    base = getattr(model, "model", model)
    encoder = base.encoder.language_model
    decoder = base.decoder

    # ``encoder.language_model`` and ``decoder`` expose the same parameter names
    # (``layers.*``, ``norm.weight``, ``embed_tokens.weight``), so one walk covers every
    # entry of ``DiffusionGemmaModel._tied_weights_keys``. Enumerating the sub-stacks
    # individually instead is how ``norm.weight`` got missed before. Decoder-private
    # parameters (``self_conditioning``) have no encoder counterpart and are skipped;
    # the encoder-private ``layer_scalar`` tensors are buffers, so they never show up here.
    decoder_params = dict(decoder.named_parameters(recurse=True))

    retied = 0
    for name, encoder_param in list(encoder.named_parameters(recurse=True)):
        decoder_param = decoder_params.get(name)
        if decoder_param is None or decoder_param is encoder_param:
            continue
        module_path, _, attr = name.rpartition(".")
        target = encoder.get_submodule(module_path) if module_path else encoder
        setattr(target, attr, decoder_param)
        retied += 1

    # ``lm_head`` is tied to the embedding only when the config asks for it; VeOmni's
    # loader re-applies that tie itself, so this is just a no-op safeguard for the
    # expert-sharding path, which runs before the loader.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and getattr(model.config, "tie_word_embeddings", False):
        decoder_embed_weight = decoder.embed_tokens.weight
        if lm_head.weight is not decoder_embed_weight:
            lm_head.weight = decoder_embed_weight
            retied += 1

    return retied


def share_expert_modules(model: nn.Module) -> int:
    """Make the encoder reuse the decoder's ``experts`` *modules*, not just their tensors.

    Only correct (and only needed) when expert parallel is on. ``parallelize_model_fsdp2``
    calls ``fully_shard`` on the decoder's ``experts`` module by itself, on the EP mesh with
    ``Shard(1)``. The encoder's ``experts`` module is not in the EP map, so the very same
    parameter objects also get claimed by the root unit on the dp_shard mesh with
    ``Shard(0)``; two units disagreeing about one tensor's layout dies in FSDP2's
    ``reset_sharded_param`` ("start (0) + length (E/ep) exceeds dimension size"). Collapsing
    the two modules into one leaves exactly one owner, and the root skips it because it is
    already an ``FSDPModule``.

    Without EP the decoder's ``experts`` never becomes its own unit, and doing this would
    instead make the root walk into an already-sharded module through the encoder and fail
    on ``('dp_shard', 'dp_shard')`` mesh composition -- hence the guard in ``post_shard``.

    Safe because ``DiffusionGemmaTextExperts`` holds no per-side state: the encoder's
    private tensors are the layer's ``layer_scalar`` buffers, which live on the layer.
    Returns 0 when ``tie_word_embeddings=False`` disables the shared expert weights.
    """
    if not _sharing_enabled(model):
        return 0

    base = getattr(model, "model", model)
    shared = 0
    for encoder_layer, decoder_layer in zip(base.encoder.language_model.layers, base.decoder.layers):
        if getattr(encoder_layer, "experts", None) is not getattr(decoder_layer, "experts", None):
            encoder_layer.experts = decoder_layer.experts
            shared += 1
    return shared


class DiffusionGemmaParallelPlan(ParallelPlan):
    def post_shard(self, model):
        if get_parallel_state().extra_parallel_enabled("ep"):
            shared = share_expert_modules(model)
            if shared:
                logger.info_rank0(f"DiffusionGemma: collapsed {shared} encoder/decoder expert module pairs for EP.")
        retied = retie_encoder_to_decoder(model)
        if retied:
            logger.info_rank0(f"DiffusionGemma: re-tied {retied} encoder/decoder parameters after expert sharding.")


def get_parallel_plan():
    ep_plan = {
        "model.decoder.layers.*.experts.gate_up_proj": Shard(0),
        "model.decoder.layers.*.experts.down_proj": Shard(0),
    }
    parallel_plan = DiffusionGemmaParallelPlan(
        extra_parallel_plan={
            "ep": ep_plan,
        }
    )
    return parallel_plan
