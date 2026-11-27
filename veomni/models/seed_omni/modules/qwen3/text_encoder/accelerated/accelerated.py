"""VeOmni-accelerated Qwen3TextEncoder — training graph hooks.

FSM ``generate`` lives natively on :class:`~.modeling.Qwen3TextEncoder`;
this file owns the SP-aware training pre/forward/post hooks, ChatML template
binding, and the image-mode vision-token freeze with its row-wise weight decay.
"""

import hashlib
from typing import Any, Dict, Optional

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from veomni.distributed.emb_parallel import ShardedEmbedding
from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils import logging

from .....mixins.training_module_mixin import post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from ....text_encoder.accelerated import (
    TrainingMixin as BaseTrainingMixin,
)
from ....text_encoder.accelerated import (
    VeOmniMixin as BaseVeOmniMixin,
)
from ..chat_template import Qwen3ChatTemplate
from ..configuration import Qwen3TextEncoderConfig
from ..modeling import Qwen3TextEncoder


logger = logging.get_logger(__name__)


class TrainingMixin(BaseTrainingMixin):
    config: Qwen3TextEncoderConfig
    device: torch.device
    dtype: torch.dtype
    _tokenizer: Any
    embed_tokens: ShardedEmbedding
    _trainable_row_mask: Optional[torch.Tensor]

    def __init__(self, *args, **kwargs) -> None:
        """Forward to the base encoder; start with no embedding-row restriction."""
        super().__init__(*args, **kwargs)
        # Set by `freeze_model` in image mode; `None` means "no row restriction".
        self._trainable_row_mask: Optional[torch.Tensor] = None

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Pack the carrier's token ids for the shared text-encoder forward."""
        return super().encode_pre(conversation_list, **kwargs)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        """Scatter packed embeddings back onto the carrier's text segments."""
        return super().encode_post(**outputs)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Stack hidden states and shifted labels for the LM-head loss."""
        return super().decode_pre(conversation_list, **kwargs)

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        """Rename the node's scalar ``loss`` to the graph's ``_loss`` key."""
        return super().decode_post(**outputs)

    def freeze_model(self) -> None:
        """In image mode, train only the vision special-token embedding rows.

        Bootstrapping a text-only LLM onto a vision tower must not re-train the
        whole vocabulary: ``freeze_model`` keeps the tied table's
        ``requires_grad`` on (the tied head needs it) but masks every row except
        the vision special tokens. The ids are resolved from the module's own
        tokenizer — a user cannot know them, but the module can.
        """
        if not self._enable_image:
            return  # fully trainable (default text-only behaviour)
        ids = [self._resolve_vision_row(token) for token in self._VISION_SPECIAL_TOKENS]
        keep = torch.zeros(self.embed_tokens.num_embeddings, dtype=torch.bool)
        keep[ids] = True
        self._trainable_row_mask = keep
        self.embed_tokens.weight.requires_grad_(True)
        # A frozen untied head would leave the module training three embedding rows
        # and a randomly-initialised output projection, i.e. unable to emit tokens
        # at all — a silent failure. Tied heads need no action: the mask covers the
        # shared table, which is why ``encode`` must run before the first ``decode``.
        if not self.config.tie_word_embeddings:
            raise ValueError(
                "Qwen3 image-mode freezes the LM head so only the vision special-token "
                "embedding rows train, which needs the head tied to that table. "
                f"Got tie_word_embeddings=False ({type(self).__name__})."
            )
        # The weight parameter is replaced by parallelization (FSDP2 shards it, the
        # emb plan splits its rows), so a hook on it now would be lost. FSDP2's own
        # pre-forward hook is prepended, so on the first lookup this one sees the
        # rows the lookup reads. The tied ``project`` does not run module hooks; it
        # is masked because it reads that same unsharded parameter, which is why
        # ``encode`` must run before the first ``decode``.
        self.embed_tokens.register_forward_pre_hook(self._mask_frozen_embedding_rows)

    def _resolve_vision_row(self, token: str) -> int:
        """Vocabulary row of ``token``, validated against the table it indexes.

        ``convert_tokens_to_ids`` fails in ways that would silently train the wrong
        rows: ``None`` for an unknown token, ``unk_token_id`` on a fast tokenizer
        (which would train ``<unk>`` instead), a negative surrogate id (which Python
        wraps to the *end* of the table), or an id past ``num_embeddings``.
        """
        row = self._tokenizer.convert_tokens_to_ids(token)
        num_rows = self.embed_tokens.num_embeddings
        if row is None:
            raise ValueError(f"Qwen3 image mode: the tokenizer does not know {token!r}.")
        row = int(row)
        unk = getattr(self._tokenizer, "unk_token_id", None)
        if unk is not None and row == int(unk):
            raise ValueError(
                f"Qwen3 image mode: {token!r} resolves to the unk id ({row}), so the vision "
                "rows cannot be identified in this tokenizer."
            )
        if not 0 <= row < num_rows:
            raise ValueError(
                f"Qwen3 image mode: {token!r} resolves to row {row}, outside the embedding table ([0, {num_rows}))."
            )
        return row

    def _emb_row_offset(self, rows: int) -> int:
        """First table row of a weight holding ``rows`` rows: this rank's ``emb`` slice, if split."""
        if rows == self.embed_tokens.num_embeddings:
            return 0
        parallel_state = get_parallel_state()
        if not parallel_state.extra_parallel_enabled("emb"):
            raise RuntimeError(
                f"{type(self).__name__}: the embedding table has {rows} of "
                f"{self.embed_tokens.num_embeddings} rows but the active ParallelState has no "
                "'emb' dimension, so the vision-row mask cannot be sliced to it."
            )
        return parallel_state.extra_parallel_rank("emb") * rows

    def _mask_frozen_embedding_rows(self, embed_tokens: torch.nn.Module, args: tuple) -> None:
        """Zero the gradient of every row outside ``_trainable_row_mask``.

        Registered as an ``embed_tokens`` forward pre-hook so it runs after FSDP2 has
        unsharded the table (FSDP's own hook is prepended), and slices the mask to
        this rank's rows when an ``emb`` plan split the table. Hooked once per
        parameter: FSDP2 keeps one unsharded Parameter across steps.
        """
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
        """Take the embedding table's weight decay over, applying it to the vision rows only.

        AdamW decays a whole parameter outside the gradient, so the row mask alone
        would still let it shrink the frozen rows — and with them the tied output
        head. This hook gives the table a decay-free group and applies the module's
        configured ``weight_decay`` to the trainable rows by a step pre-hook, the
        same ``p *= 1 - lr * wd`` AdamW would do, before the update.

        Giving the table its own group changes the optimizer's param-group count,
        which is part of the checkpoint key space (a saved optimizer state is
        rejected when the counts differ). It is therefore done here, right after
        the optimizer is built — before the lr-scheduler or any checkpoint load —
        so every save in a run has the same layout. Resuming from a checkpoint
        written by a run *without* this hook is not supported for this recipe.

        Nonzero row-wise decay with an FP32-master optimizer is rejected before
        modifying its groups or hooks; the compute-copy hook cannot update that
        optimizer's authoritative master weight. The runtime coordinates local
        optimizer-build failures across ranks before training starts. With zero
        decay, a group identity binds the row mask for exact optimizer resume.
        """
        if self._trainable_row_mask is None:
            return
        weight = self.embed_tokens.weight
        sub_optimizers = getattr(optimizer, "optimizers_dict", {None: optimizer}).values()
        owners = [
            (leaf, group)
            for leaf in sub_optimizers
            for group in leaf.param_groups
            if any(param is weight for param in group["params"])
        ]
        if len(owners) > 1:
            raise RuntimeError("Qwen3 image-mode embedding has multiple optimizer owners")
        owner, group = owners[0] if owners else (None, None)
        if owner is None:
            # Fail closed: silently leaving the group's decay in place is exactly the
            # state that shrinks the frozen rows, and nothing downstream would notice.
            logger.warning_rank0(
                f"{type(self).__name__}: the embedding table is not in any optimizer param group, so the "
                "vision-row weight decay cannot be applied; its group decay (if any) stays in effect."
            )
            return
        from veomni.optim.master_fp32_adamw import MasterFP32AdamW

        if isinstance(owner, MasterFP32AdamW):
            if owner.state.get(weight):
                raise RuntimeError("Qwen3 image-mode row masking requires a fresh FP32 master optimizer")
            if group["weight_decay"]:
                raise RuntimeError(
                    "Qwen3 image-mode row-wise weight decay is unsupported with FP32 master optimizers; "
                    "use AdamW or set weight_decay=0."
                )
            mask = self._trainable_row_mask
            if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != self.embed_tokens.num_embeddings:
                raise RuntimeError("Qwen3 image-mode trainable-row mask is invalid")
            mask_bytes = mask.detach().to(device="cpu").contiguous().numpy().tobytes()
            positions = [index for index, param in enumerate(group["params"]) if param is weight]
            if len(positions) != 1:
                raise RuntimeError("Qwen3 image-mode embedding must occur exactly once in its optimizer group")
            identity = f"v1:{hashlib.sha256(mask_bytes).hexdigest()}:{positions[0]}"
            marker = "qwen3_image_row_mask"
            if marker in group and group[marker] != identity:
                raise RuntimeError("Qwen3 image-mode optimizer group has a conflicting row-mask identity")
            group[marker] = identity
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
        # Plain attribute assignment would register the Parameter again and leak
        # a duplicate ``_decayed_weight`` key into ``state_dict()``.
        self.__dict__["_decayed_weight"] = weight
        optimizer.register_step_pre_hook(self._decay_trainable_rows)

    def _decay_trainable_rows(self, optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict) -> None:
        """Apply ``_row_weight_decay`` to the trainable rows, before this step's update.

        Re-zeroes the group's ``weight_decay`` every step because loading a checkpoint
        rebuilds the param groups from their saved hyperparameters.
        """
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
    """The param group holding ``param`` (by identity), or ``None``."""
    return next((g for g in optimizer.param_groups if any(p is param for p in g["params"])), None)


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    """Qwen3 ChatML text encoder, optionally image-aware — accelerated wrapper.

    Only the chat-template selection and the image-mode freeze are genuinely
    accelerated-specific; the encode/decode plumbing and ChatML ``generate`` FSM
    live on the native :class:`~.modeling.Qwen3TextEncoder`.
    """

    config: Qwen3TextEncoderConfig
    _chat_template: Qwen3ChatTemplate | Qwen3VLChatTemplate

    # Vision special tokens whose embedding rows bootstrap image understanding;
    # ids are resolved from the tokenizer at freeze time (see :meth:`freeze_model`).
    _VISION_SPECIAL_TOKENS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")

    @property
    def _enable_image(self) -> bool:
        """Whether this module runs in image mode (the recipe's ``enable_image``)."""
        return self.config.enable_image

    @property
    def tokenizer(self) -> Any:
        """The module's tokenizer, bound by ``bind_module_assets``."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        """Bind the tokenizer and rebuild the chat template it implies.

        Only the template differs: image mode reuses the Qwen3-VL ChatML template
        (adds the vision wrap tokens and per-frame video segments); otherwise the
        text-only Qwen3 ChatML.
        """
        self._tokenizer = tokenizer
        self._chat_template = Qwen3VLChatTemplate(tokenizer) if self._enable_image else Qwen3ChatTemplate(tokenizer)


class Qwen3TextEncoderAccelerated(VeOmniMixin, Qwen3TextEncoder):
    """Text encoder with the SP-aware training hooks and image-mode row masking."""


__all__ = ["Qwen3TextEncoderAccelerated"]
