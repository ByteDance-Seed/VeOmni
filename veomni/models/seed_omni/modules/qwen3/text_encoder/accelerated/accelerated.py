"""VeOmni-accelerated Qwen3TextEncoder — training graph hooks.

FSM ``generate`` now lives natively on :class:`~.modeling.Qwen3TextEncoder`;
this file only owns the SP-aware training pre/forward/post hooks plus the
image-mode vision-token freeze and its row-wise weight decay (all genuinely
accelerated-only).
"""

from typing import Any, Dict, Optional

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from veomni.distributed.emb_parallel import ShardedEmbedding
from veomni.distributed.parallel_state import get_parallel_state

from .....mixins.training_module_mixin import post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....base.text_encoder.accelerated import (
    TrainingMixin as BaseTrainingMixin,
)
from ....base.text_encoder.accelerated import (
    VeOmniMixin as BaseVeOmniMixin,
)
from ....qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from ..chat_template import Qwen3ChatTemplate
from ..configuration import Qwen3TextEncoderConfig
from ..modeling import Qwen3TextEncoder


class TrainingMixin(BaseTrainingMixin):
    config: Qwen3TextEncoderConfig
    device: torch.device
    dtype: torch.dtype
    _tokenizer: Any
    embed_tokens: ShardedEmbedding
    _trainable_row_mask: Optional[torch.Tensor]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._trainable_row_mask: Optional[torch.Tensor] = None

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().encode_pre(conversation_list, **kwargs)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().encode_post(**outputs)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().decode_pre(conversation_list, **kwargs)

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().decode_post(**outputs)

    def freeze_model(self) -> None:
        if not self._enable_image:
            return  # fully trainable (default text-only behaviour)
        # The user can't know the vision special-token ids, but the module can:
        # resolve them from its own tokenizer so only those rows stay trainable.
        ids = [int(self._tokenizer.convert_tokens_to_ids(tok)) for tok in self._VISION_SPECIAL_TOKENS]
        keep = torch.zeros(self.embed_tokens.num_embeddings, dtype=torch.bool)
        keep[ids] = True
        self._trainable_row_mask = keep
        self.embed_tokens.weight.requires_grad_(True)
        # The weight parameter is replaced by parallelization (FSDP2 shards it, the
        # emb plan splits its rows), so a hook on it now would be lost. FSDP2's own
        # pre-forward hook is prepended, so on the first lookup this one sees the
        # rows the lookup reads. The tied ``project`` does not run module hooks; it
        # is masked because it reads that same unsharded parameter, which is why
        # ``encode`` must run before the first ``decode``.
        self.embed_tokens.register_forward_pre_hook(self._mask_frozen_embedding_rows)

    def _emb_row_offset(self, rows: int) -> int:
        """First table row of a weight holding ``rows`` rows: this rank's ``emb`` slice, if split."""
        if rows == self.embed_tokens.num_embeddings:
            return 0
        return get_parallel_state().extra_parallel_rank("emb") * rows

    def _mask_frozen_embedding_rows(self, embed_tokens: torch.nn.Module, args: tuple) -> None:
        weight = embed_tokens.weight
        if not weight.requires_grad or getattr(weight, "_frozen_rows_masked", False):
            return
        rows = weight.shape[0]
        start = self._emb_row_offset(rows)
        mask = self._trainable_row_mask[start : start + rows].to(device=weight.device, dtype=weight.dtype)
        weight.register_hook(lambda grad: grad * mask.unsqueeze(1))
        # FSDP2 keeps one unsharded parameter across steps, so hook it once.
        weight._frozen_rows_masked = True

    def configure_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Apply the table's weight decay to the vision rows only.

        AdamW decays a whole parameter outside the gradient, so the masked rows
        would still shrink. Their group's ``weight_decay`` is taken over: set to 0
        for AdamW, and applied to the trainable rows by a step pre-hook, the same
        ``p *= 1 - lr * wd`` before the update that AdamW would do.
        """
        if self._trainable_row_mask is None:
            return
        weight = self.embed_tokens.weight
        sub_optimizers = getattr(optimizer, "optimizers_dict", {None: optimizer}).values()
        owner = next((o for o in sub_optimizers if _param_group_of(o, weight) is not None), None)
        if owner is None:
            return
        group = _param_group_of(owner, weight)
        self._row_weight_decay = group["weight_decay"]
        if not self._row_weight_decay:
            return
        if len(group["params"]) > 1:
            group["params"] = [p for p in group["params"] if p is not weight]
            owner.add_param_group({**group, "params": [weight], "weight_decay": 0.0})
        else:
            group["weight_decay"] = 0.0

        local_rows = weight.shape[0]
        start = self._emb_row_offset(local_rows)
        if isinstance(weight, DTensor):
            local_shape, offset = compute_local_shape_and_global_offset(
                weight.shape, weight.device_mesh, weight.placements
            )
            start += offset[0]
            local_rows = local_shape[0]
        rows = self._trainable_row_mask[start : start + local_rows].nonzero().flatten()
        self._decayed_rows = rows.to(weight.device)
        self._decayed_weight = weight
        optimizer.register_step_pre_hook(self._decay_trainable_rows)

    def _decay_trainable_rows(self, optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict) -> None:
        weight = self._decayed_weight
        group = _param_group_of(optimizer, weight)
        if group is None or weight.grad is None:
            return
        # Loading a checkpoint rebuilds the param groups with their saved weight decay.
        group["weight_decay"] = 0.0
        local = weight.to_local() if isinstance(weight, DTensor) else weight
        with torch.no_grad():
            local[self._decayed_rows] *= 1 - group["lr"] * self._row_weight_decay


def _param_group_of(optimizer: torch.optim.Optimizer, param: torch.Tensor) -> Optional[Dict[str, Any]]:
    return next((g for g in optimizer.param_groups if any(p is param for p in g["params"])), None)


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    """Qwen3 ChatML text encoder, optionally image-aware — accelerated wrapper.

    Only the chat-template selection and the image-mode freeze are genuinely
    accelerated-specific; the encode/decode plumbing and ChatML ``generate``
    FSM live on the native :class:`~.modeling.Qwen3TextEncoder`.
    """

    config: Qwen3TextEncoderConfig
    _chat_template: Qwen3ChatTemplate | Qwen3VLChatTemplate

    # Vision special tokens whose embedding rows bootstrap image understanding;
    # ids are resolved from the tokenizer at freeze time (see :meth:`freeze_model`).
    _VISION_SPECIAL_TOKENS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")

    @property
    def _enable_image(self) -> bool:
        return self.config.enable_image

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        # Only the template differs: image mode reuses the Qwen3-VL ChatML template
        # (adds the vision wrap tokens); otherwise the text-only Qwen3 ChatML.
        if self._enable_image:
            self._chat_template = Qwen3VLChatTemplate(tokenizer)
        else:
            self._chat_template = Qwen3ChatTemplate(tokenizer)


class Qwen3TextEncoderAccelerated(VeOmniMixin, Qwen3TextEncoder):
    pass


__all__ = ["Qwen3TextEncoderAccelerated"]
