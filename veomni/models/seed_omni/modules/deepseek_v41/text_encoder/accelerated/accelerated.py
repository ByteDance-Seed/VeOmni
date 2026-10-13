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
import torch.nn.functional as F

from veomni.distributed.parallel_state import get_parallel_state
from veomni.ops.kernels.cross_entropy import ForCausalLMLoss
from veomni.ops.kernels.cross_entropy.eager import eager_cross_entropy
from veomni.utils.constants import IGNORE_INDEX

from .....mixins.training_module_mixin import post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....base.text_encoder.accelerated import TrainingMixin as BaseTrainingMixin
from ....base.text_encoder.accelerated import VeOmniMixin as BaseVeOmniMixin
from ..chat_template import DeepseekV41ChatTemplate
from ..configuration import DeepseekV41TextEncoderConfig
from ..modeling import DeepseekV41TextEncoder


class _MarkovResidualCrossEntropy(torch.autograd.Function):
    """Summed ``CE(h @ W.T + bias, labels)`` over the supervised rows, gradients taken in forward.

    Under SP every rank scores the gathered full row, so a 32k-token sample would
    otherwise hold ``[S, vocab]`` fp32 logits and their softmax gradient at once.
    Here only one chunk of fp32 logits is live; the full-size tensors left are the
    bias itself and its gradient.
    """

    @staticmethod
    def forward(ctx, hidden, weight, bias, labels, chunk_size):
        rows = (labels != IGNORE_INDEX).nonzero().squeeze(-1)
        need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
        grad_hidden = torch.zeros_like(hidden) if need_hidden else None
        grad_weight = torch.zeros_like(weight) if need_weight else None
        grad_bias = torch.zeros_like(bias) if need_bias else None
        loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for start in range(0, rows.numel(), chunk_size):
            index = rows[start : start + chunk_size]
            h = hidden.index_select(0, index)
            target = labels.index_select(0, index)
            logits = (F.linear(h, weight) + bias.index_select(0, index)).float()
            lse = torch.logsumexp(logits, dim=-1)
            loss += (lse - logits.gather(-1, target[:, None]).squeeze(-1)).sum()
            grad = logits.sub_(lse[:, None]).exp_()
            grad.scatter_add_(-1, target[:, None], grad.new_full((index.numel(), 1), -1.0))
            grad = grad.to(hidden.dtype)
            if need_bias:
                grad_bias.index_copy_(0, index, grad)
            if need_hidden:
                grad_hidden.index_copy_(0, index, grad @ weight)
            if need_weight:
                grad_weight.addmm_(grad.t(), h)
        ctx.save_for_backward(*(g if g is not None else torch.empty(0) for g in (grad_hidden, grad_weight, grad_bias)))
        ctx.has_grads = (need_hidden, need_weight, need_bias)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grads = [g * grad_output.to(g.dtype) if has else None for g, has in zip(ctx.saved_tensors, ctx.has_grads)]
        return *grads, None, None


def _markov_residual_cross_entropy(
    logits, labels, vocab_size, num_items_in_batch, ignore_index=IGNORE_INDEX, **kwargs
) -> tuple[torch.Tensor, None]:
    """``cross_entropy_fn`` for ``ForCausalLMLoss``: the linear, the Markov add and the CE fused per chunk."""
    del logits, ignore_index
    loss = _MarkovResidualCrossEntropy.apply(
        kwargs["hidden_states"],
        kwargs["weights"],
        kwargs["markov_bias"].reshape(-1, vocab_size),
        labels,
        kwargs.get("markov_chunk_size", 2048),
    )
    return loss / num_items_in_batch, None


class TrainingMixin(BaseTrainingMixin):
    config: DeepseekV41TextEncoderConfig
    device: torch.device
    _chat_template: DeepseekV41ChatTemplate

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
        dspark_hidden = self._pack_dspark_rows(conversation_list, "dspark_hidden")
        if dspark_hidden is not None:
            packed["dspark_hidden_states"] = dspark_hidden
            packed["dspark_markov_bias"] = self._pack_dspark_rows(conversation_list, "dspark_markov_bias")
            packed["mtp_labels"] = self._mask_dspark_labels(packed["shift_labels"], conversation_list)
        return packed

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
                elif part.type in ("image", "video"):
                    unsupervised.append(True)
            if len(unsupervised) > sample_start:
                unsupervised[sample_start] = True
        mask = torch.tensor(unsupervised, dtype=torch.bool).to(shift_labels.device, non_blocking=True)
        return shift_labels.masked_fill(mask, IGNORE_INDEX)

    def _pack_dspark_rows(
        self, conversation_list: Optional[list[list[ConversationItem]]], meta_key: str
    ) -> Optional[torch.Tensor]:
        """Same walk as ``_prepare_decode_inputs``, reading ``meta[meta_key]``."""
        chunks = []
        for sample in conversation_list or []:
            for part in sample:
                if part.is_dummy:
                    continue
                rows = part.meta.get(meta_key)
                if rows is None:
                    return None
                if rows.dim() == 3:
                    rows = rows.squeeze(0)
                if part.type == "text":
                    chunks.append(rows)
                elif part.type in ("image", "video"):
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
        outputs["mtp_loss"] = self._dspark_loss(dspark_hidden_states, dspark_markov_bias, mtp_labels)
        return outputs

    def _dspark_loss(
        self, hidden_states: torch.Tensor, markov_bias: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        """``CE(lm_head(h) + markov(x), labels)``, token-mean over the ``dp_sp`` mesh like the main head.

        The shared fused linear+CE kernels take a weight and no additive term, so
        a bias-free untied head goes through the chunked
        :class:`_MarkovResidualCrossEntropy`; any other head materializes the
        supervised rows' logits.
        """
        num_supervised_tokens = (labels != IGNORE_INDEX).sum()
        if not self.config.tie_word_embeddings and self.lm_head is not None and self.lm_head.bias is None:
            loss, _, _ = ForCausalLMLoss(
                logits=None,
                labels=None,
                shift_labels=labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=num_supervised_tokens.clamp(min=1),
                loss_reduction_group=get_parallel_state().fsdp_group,
                cross_entropy_fn=_markov_residual_cross_entropy,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                markov_bias=markov_bias,
            )
            return loss
        if num_supervised_tokens > 0:
            supervised = labels != IGNORE_INDEX
            hidden_states, markov_bias, labels = hidden_states[supervised], markov_bias[supervised], labels[supervised]
        loss, _, _ = ForCausalLMLoss(
            logits=self._project(hidden_states) + markov_bias,
            labels=None,
            shift_labels=labels,
            vocab_size=self.config.vocab_size,
            num_items_in_batch=num_supervised_tokens.clamp(min=1),
            loss_reduction_group=get_parallel_state().fsdp_group,
            cross_entropy_fn=eager_cross_entropy,
        )
        return loss

    @post_forward("decode")
    def decode_post(self, mtp_loss: Optional[torch.Tensor] = None, **outputs: Any) -> Dict[str, Any]:
        forwarded = super().decode_post(**outputs)
        if mtp_loss is not None:
            forwarded["_loss"]["mtp_loss"] = mtp_loss
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
