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

# Patch https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3/modeling_qwen3.py


from typing import Optional, Union

import torch
import transformers.models.qwen3.modeling_qwen3 as hf_qwen3
from transformers import Qwen3ForCausalLM, Qwen3ForSequenceClassification
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    can_return_tuple,
)

from ....utils import logging
from ....utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from ....utils.import_utils import is_transformers_version_greater_or_equal_to


logger = logging.get_logger(__name__)


# ================================================================
# PATCH: Qwen3ForCausalLM.forward
# 1. Support use with fuse cross_entropy loss function.
# ================================================================
@can_return_tuple
def qwen3forcausallm_forward(
    self: Qwen3ForCausalLM,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

    >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    # --- Patch.1 ---
    loss = None
    logits = None
    if labels is not None:
        loss, logits = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    # --- Patch.1 ---

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# ================================================================
# PATCH: Qwen3ForSequenceClassification.forward
# 1. Support SP
# ================================================================
@can_return_tuple
def qwen3forSequenceClassification_forward(
    self: Qwen3ForSequenceClassification,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> SequenceClassifierOutputWithPast:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the classification loss.

            This head uses a single-label cross-entropy loss. In our setup, labels typically follow the
            "token-level labels" convention: positions not supervised should be set to `-100`, and only the
            supervised token(s) (e.g., the last valid token of each sample) carry a real class id in
            `[0, ..., num_labels - 1]`. Tokens with label `-100` are ignored.

            Note: `labels` should be provided for classification training tasks.

    Returns:

    """
    outputs: BaseModelOutputWithPast = self.model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state

    loss = None
    logits = None
    if labels is not None:
        loss, logits = self.loss_function(
            logits=logits,
            labels=labels,
            num_labels=self.num_labels,
            hidden_states=hidden_states,
            weights=self.score.weight,
            **kwargs,
        )
    else:
        logits = self.score(hidden_states)

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def apply_veomni_qwen3_patch():
    logger.info_rank0("Apply VeOmni patch to Qwen3.")

    hf_qwen3.Qwen3ForCausalLM.forward = qwen3forcausallm_forward
    hf_qwen3.Qwen3ForSequenceClassification.forward = qwen3forSequenceClassification_forward

    if IS_CUDA_AVAILABLE:
        from .gpu_patch import apply_veomni_qwen3_gpu_patch

        apply_veomni_qwen3_gpu_patch()
    elif IS_NPU_AVAILABLE and is_transformers_version_greater_or_equal_to("4.50.4"):
        from .npu_patch import apply_qwen3_npu_patch

        apply_qwen3_npu_patch()
    else:
        logger.warning_rank0(
            "Qwen3ForCausalLM in VeOmni only support CUDA or NPU with transformers version >= 4.50.4."
        )
