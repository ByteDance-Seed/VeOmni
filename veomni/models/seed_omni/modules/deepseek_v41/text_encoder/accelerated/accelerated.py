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

"""Training-graph hooks for the DeepSeek-V4.1 text encoder.

Encode / decode plumbing -- the SP slice and gather around ``embed_tokens``,
the fused linear+CE head, the vocab-parallel lookup -- is the shared base's.
V4.1 adds the chat template (``thinking_mode`` / ``reasoning_effort``) and,
when the backbone scattered ``meta['dspark_hidden']`` / ``meta['dspark_markov_bias']``,
the DSpark slot-0 CE of official ``forward_head``: row ``j`` holds token ``x[j]``
in the draft slot, so it predicts ``x[j+1]`` -- the same causal labels as the
main head -- with the Markov residual of ``x[j]`` added to the shared LM head's
logits.
"""

from typing import Any, Dict, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.loss_utils import ForCausalLMLoss
from veomni.ops import VeomniOp
from veomni.utils.constants import IGNORE_INDEX

from .....mixins.training_module_mixin import post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....base.llm_packing import item_token_len
from ....text_encoder.accelerated import TrainingMixin as BaseTrainingMixin
from ....text_encoder.accelerated import VeOmniMixin as BaseVeOmniMixin
from ..chat_template import DeepseekV41ChatTemplate
from ..configuration import DeepseekV41TextEncoderConfig
from ..modeling import DeepseekV41TextEncoder


#: Rows of fp32 ``[rows, vocab]`` logits the draft CE holds at once.
_DSPARK_CHUNK_SIZE = 2048


class TrainingMixin(BaseTrainingMixin):
    config: DeepseekV41TextEncoderConfig
    device: torch.device
    _chat_template: DeepseekV41ChatTemplate

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.veomni_dspark_ce = VeomniOp("cross_entropy_loss", "standard", "chunk_loss")

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # The shared batch still holds ngram_embeds from the sibling node.
        del kwargs
        return super().encode_pre(conversation_list)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().encode_post(**outputs)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        packed = super().decode_pre(conversation_list)
        ps = get_parallel_state()
        sp_sharded = ps.sp_size > 1 and ps.skip_sp_gather_between_modules
        dspark_hidden = self._pack_dspark_rows(conversation_list, "dspark_hidden", sp_sharded=sp_sharded)
        if dspark_hidden is not None:
            packed["dspark_hidden_states"] = dspark_hidden
            packed["dspark_markov_bias"] = self._pack_dspark_rows(
                conversation_list, "dspark_markov_bias", sp_sharded=sp_sharded
            )
            for sample in conversation_list or []:
                for part in sample:
                    if not part.is_dummy and "dspark_markov_head_weight" in part.meta:
                        packed["dspark_markov_head_weight"] = part.meta.pop("dspark_markov_head_weight")
                        break
            if sp_sharded:
                packed["mtp_labels"] = self._mask_sharded_dspark_labels(packed["shift_labels"], conversation_list, ps)
            else:
                packed["mtp_labels"] = self._mask_dspark_labels(packed["shift_labels"], conversation_list)
        return packed

    def _mask_sharded_dspark_labels(
        self,
        local_shift_labels: torch.Tensor,
        conversation_list: Optional[list[list[ConversationItem]]],
        ps: Any,
    ) -> torch.Tensor:
        """Mask the draft-CE rows this rank's CP window scores which decode never drafts from.

        The window holds only real rows (padding never enters the carrier), so the
        mask is built over the padded global row and then narrowed to the same
        ``[shard_start, shard_start + rows)`` span the labels were — never the
        padded width, which would be one item longer on the tail rank.
        """
        from ....base.sp_shard import sp_local_len, sp_valid_in_window

        unsupervised = []
        for sample in conversation_list or []:
            sample_start = len(unsupervised)
            for part in sample:
                if part.is_dummy:
                    continue
                if part.type == "text":
                    unsupervised.extend([False] * int(part.meta["labels"].shape[0]))
                elif part.type in ("image", "video", "audio"):
                    unsupervised.extend([True] * item_token_len(part))
            if len(unsupervised) > sample_start:
                unsupervised[sample_start] = True
        valid_len = len(unsupervised)
        local_len = sp_local_len(valid_len, ps.sp_size)
        rows = sp_valid_in_window(valid_len, ps.sp_rank, ps.sp_size)
        shard_start = ps.sp_rank * local_len
        if local_len * ps.sp_size > valid_len:
            unsupervised.extend([True] * (local_len * ps.sp_size - valid_len))
        local_mask = torch.tensor(unsupervised[shard_start : shard_start + rows], dtype=torch.bool).to(
            local_shift_labels.device, non_blocking=True
        )
        return local_shift_labels.masked_fill(local_mask, IGNORE_INDEX)

    def _mask_dspark_labels(
        self,
        shift_labels: torch.Tensor,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> torch.Tensor:
        """The causal labels minus the rows decode never drafts from.

        A sample's first row sees no main KV (decode always has at least the
        verified token), and a vision row puts an image embedding in the slot
        where decode always has a sampled text token.
        """
        unsupervised = []
        for sample in conversation_list or []:
            sample_start = len(unsupervised)
            for part in sample:
                if part.is_dummy:
                    continue
                if part.type == "text":
                    unsupervised.extend([False] * part.meta["labels"].shape[0])
                elif part.type in ("image", "video", "audio"):
                    # Decode packs exactly one row per media segment (see
                    # `_prepare_decode_inputs`); draft CE never targets it.
                    unsupervised.append(True)
            if len(unsupervised) > sample_start:
                unsupervised[sample_start] = True
        mask = torch.tensor(unsupervised, dtype=torch.bool).to(shift_labels.device, non_blocking=True)
        if mask.numel() != shift_labels.numel():
            raise RuntimeError(
                f"DSpark mask walked {mask.numel()} rows but the packed decode labels have "
                f"{shift_labels.numel()}; the two walks must cover the same non-dummy parts."
            )
        return shift_labels.masked_fill(mask, IGNORE_INDEX)

    def _pack_dspark_rows(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
        meta_key: str,
        *,
        sp_sharded: bool = False,
    ) -> Optional[torch.Tensor]:
        """Same walk as ``_prepare_decode_inputs``, reading ``meta[meta_key]``."""
        chunks = []
        for sample in conversation_list or []:
            for part in sample:
                if part.is_dummy:
                    continue
                rows = part.meta.pop(meta_key, None)
                if rows is None:
                    return None
                if rows.dim() == 3:
                    rows = rows.squeeze(0)
                if part.type == "text" or sp_sharded:
                    chunks.append(rows)
                elif part.type in ("image", "video", "audio"):
                    # One draft row per media segment, matching
                    # `_prepare_decode_inputs` (audio spans are packed as one).
                    chunks.append(rows[-1:])
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    def decode(
        self,
        hidden_states: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        shift_labels: torch.LongTensor | None = None,
        dspark_hidden_states: torch.Tensor | None = None,
        dspark_markov_bias: torch.Tensor | None = None,
        dspark_markov_head_weight: torch.Tensor | None = None,
        mtp_labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> dict:
        if dspark_hidden_states is not None and hidden_states is not None:
            for name, rows in (("hidden", dspark_hidden_states), ("Markov bias", dspark_markov_bias)):
                if rows is not None and rows.shape[:-1] != hidden_states.shape[:-1]:
                    raise RuntimeError(
                        f"DSpark {name} packed to {tuple(rows.shape[:-1])} but the LM head packed "
                        f"{tuple(hidden_states.shape[:-1])}. Scatter and decode must walk "
                        "the same non-dummy conversation parts."
                    )
        outputs = super().decode(hidden_states=hidden_states, labels=labels, shift_labels=shift_labels, **kwargs)
        if dspark_hidden_states is None or outputs.get("loss") is None:
            return outputs
        if dspark_markov_bias is None or mtp_labels is None:
            raise RuntimeError(
                "DSpark hidden states reached decode without dspark_markov_bias / mtp_labels; "
                "the draft CE is only defined with the Markov residual on the draft-slot labels."
            )
        outputs["mtp_loss"] = self._dspark_loss(
            dspark_hidden_states, dspark_markov_bias, mtp_labels, dspark_markov_head_weight
        )
        return outputs

    def _dspark_loss(
        self,
        hidden_states: torch.Tensor,
        markov_bias: torch.Tensor,
        labels: torch.LongTensor,
        markov_head_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``CE(lm_head(h) + markov(x), labels)``, token-mean over the ``dp_sp`` mesh like the main head.

        A bias-free untied head goes through ``chunk_loss`` with the Markov
        residual as its ``logits_bias``, which projects only the supervised
        rows and keeps one chunk of their fp32 logits live. Any other head
        materializes the supervised rows' logits.
        """
        num_supervised_tokens = (labels != IGNORE_INDEX).sum()
        if (
            num_supervised_tokens > 0
            and not self.config.tie_word_embeddings
            and self.lm_head is not None
            and self.lm_head.bias is None
        ):
            loss, _, _ = ForCausalLMLoss(
                logits=None,
                labels=None,
                shift_labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                chunk_size=_DSPARK_CHUNK_SIZE,
                ignore_index=IGNORE_INDEX,
                loss_reduction_group=get_parallel_state().fsdp_group,
                logits_bias=markov_bias,
                logits_bias_weight=markov_head_weight,
                op=self.veomni_dspark_ce,
            )
            return loss
        if num_supervised_tokens > 0:
            supervised = labels != IGNORE_INDEX
            hidden_states, markov_bias, labels = hidden_states[supervised], markov_bias[supervised], labels[supervised]
        else:
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))[:1]
            markov_bias = markov_bias.reshape(-1, markov_bias.size(-1))[:1]
            labels = labels.reshape(-1)[:1]
        if markov_head_weight is not None:
            markov_bias = torch.nn.functional.linear(markov_bias, markov_head_weight)
        loss, _, _ = ForCausalLMLoss(
            logits=self._project(hidden_states) + markov_bias,
            labels=None,
            shift_labels=labels,
            vocab_size=self.config.vocab_size,
            num_items_in_batch=num_supervised_tokens.clamp(min=1),
            loss_reduction_group=get_parallel_state().fsdp_group,
        )
        return loss

    @post_forward("decode")
    def decode_post(self, mtp_loss: Optional[torch.Tensor] = None, **outputs: Any) -> Dict[str, Any]:
        forwarded = super().decode_post(**outputs)
        ce_loss = forwarded.get("_loss")
        if ce_loss is not None:
            loss_dict = {"ce_loss": ce_loss}
            if mtp_loss is not None:
                loss_dict["mtp_loss"] = mtp_loss
            forwarded["_loss"] = loss_dict
        elif mtp_loss is not None:
            forwarded["_loss"] = {"mtp_loss": mtp_loss}
        return forwarded


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    _chat_template: DeepseekV41ChatTemplate

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = DeepseekV41ChatTemplate(
            tokenizer,
            thinking_mode=self.config.thinking_mode,
            reasoning_effort=self.config.reasoning_effort,
        )


class DeepseekV41TextEncoderAccelerated(VeOmniMixin, DeepseekV41TextEncoder):
    pass


__all__ = ["DeepseekV41TextEncoderAccelerated"]
