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

"""
Pristine HuggingFace class snapshots + restoration helpers.

Tests that build both an HF and a VeOmni model in the same process need to
run the HF build *before* anything triggers ``apply_veomni_*_patch``, since
those functions monkey-patch HF module classes process-wide. When pytest
runs multiple cases in the same process this is hard to guarantee, so
``apply_veomni_hf_unpatch()`` restores the pristine class attributes
captured at this module's import time.

This module deliberately avoids importing from ``veomni.data`` so test
files that only need the unpatch helper don't pull in heavyweight optional
dependencies (``av``, ``torchcodec``, ...). Keep the imports here narrow.
"""

import transformers.models.deepseek_v3.modeling_deepseek_v3 as _hf_ds3
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _hf_qwen25omni
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _hf_qwen25vl
import transformers.models.qwen2_vl.modeling_qwen2_vl as _hf_qwen2vl
import transformers.models.qwen3.modeling_qwen3 as _hf_qwen3
import transformers.models.qwen3_moe.modeling_qwen3_moe as _hf_qwen3_moe
import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as _hf_q3omnimoe
import transformers.models.qwen3_vl.modeling_qwen3_vl as _hf_qwen3vl
import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as _hf_qwen3vlmoe


# Cover every patch site reachable from apply_veomni_*_patch() + the Liger
# and Triton branches of apply_veomni_*_gpu_patch(). We capture:
# - forward methods (undo in-class forward replacement, used by both the
#   always-on patch and the Triton branch's _patch_rms_norm / RoPE forward)
# - whole classes (undo module-level class swap done by the Liger branch
#   and by DeepseekV3MoE / Qwen3MoeSparseMoeBlock replacements)
# - module-level functions (apply_rotary_pos_emb is replaced by Liger)
#
# Restore order matters: we first reset in-class attributes on the pristine
# class objects, then restore the module-level names. That way, even if a
# prior test mutated `Class.forward` and then swapped in a Liger class at
# the module level, both layers are reverted.
_PRISTINE_HF = {
    # qwen3
    "qwen3.CausalLM.forward": _hf_qwen3.Qwen3ForCausalLM.forward,
    "qwen3.SeqCls.forward": _hf_qwen3.Qwen3ForSequenceClassification.forward,
    "qwen3.apply_rotary_pos_emb": _hf_qwen3.apply_rotary_pos_emb,
    "qwen3.RMSNorm.cls": _hf_qwen3.Qwen3RMSNorm,
    "qwen3.RMSNorm.forward": _hf_qwen3.Qwen3RMSNorm.forward,
    "qwen3.MLP.cls": _hf_qwen3.Qwen3MLP,
    # qwen3_moe
    "qwen3_moe.CausalLM.forward": _hf_qwen3_moe.Qwen3MoeForCausalLM.forward,
    "qwen3_moe.MoeBlock.cls": _hf_qwen3_moe.Qwen3MoeSparseMoeBlock,
    "qwen3_moe.PreTrained.init_weights": _hf_qwen3_moe.Qwen3MoePreTrainedModel._init_weights,
    "qwen3_moe.apply_rotary_pos_emb": _hf_qwen3_moe.apply_rotary_pos_emb,
    "qwen3_moe.RMSNorm.cls": _hf_qwen3_moe.Qwen3MoeRMSNorm,
    "qwen3_moe.RMSNorm.forward": _hf_qwen3_moe.Qwen3MoeRMSNorm.forward,
    "qwen3_moe.MLP.cls": _hf_qwen3_moe.Qwen3MoeMLP,
    # deepseek_v3
    "ds3.Attention.forward": _hf_ds3.DeepseekV3Attention.forward,
    "ds3.CausalLM.forward": _hf_ds3.DeepseekV3ForCausalLM.forward,
    "ds3.MoE.cls": _hf_ds3.DeepseekV3MoE,
    "ds3.PreTrained.init_weights": _hf_ds3.DeepseekV3PreTrainedModel._init_weights,
    "ds3.apply_rotary_pos_emb": _hf_ds3.apply_rotary_pos_emb,
    "ds3.RotaryEmb.cls": _hf_ds3.DeepseekV3RotaryEmbedding,
    "ds3.RotaryEmb.forward": _hf_ds3.DeepseekV3RotaryEmbedding.forward,
    "ds3.RMSNorm.cls": _hf_ds3.DeepseekV3RMSNorm,
    "ds3.RMSNorm.forward": _hf_ds3.DeepseekV3RMSNorm.forward,
    "ds3.MLP.cls": _hf_ds3.DeepseekV3MLP,
    # qwen2_vl (apply_veomni_qwen2vl_patch)
    "qwen2_vl.CausalLM.cls": _hf_qwen2vl.Qwen2VLForConditionalGeneration,
    "qwen2_vl.Model.cls": _hf_qwen2vl.Qwen2VLModel,
    "qwen2_vl.VisionPretrained.cls": _hf_qwen2vl.Qwen2VisionTransformerPretrainedModel,
    "qwen2_vl.DecoderLayer.cls": _hf_qwen2vl.Qwen2VLDecoderLayer,
    "qwen2_vl.VisionAttention.forward": _hf_qwen2vl.VisionAttention.forward,
    # qwen2_5_vl (apply_veomni_qwen25_vl_patch)
    "qwen2_5_vl.VisionAttention.forward": _hf_qwen25vl.Qwen2_5_VLVisionAttention.forward,
    "qwen2_5_vl.VisionPretrained.forward": _hf_qwen25vl.Qwen2_5_VisionTransformerPretrainedModel.forward,
    "qwen2_5_vl.Model.cls": _hf_qwen25vl.Qwen2_5_VLModel,
    "qwen2_5_vl.CausalLM.cls": _hf_qwen25vl.Qwen2_5_VLForConditionalGeneration,
    # qwen3_vl (apply_veomni_qwen3vl_patch)
    "qwen3_vl.VisionAttention.forward": _hf_qwen3vl.Qwen3VLVisionAttention.forward,
    "qwen3_vl.TextAttention.cls": _hf_qwen3vl.Qwen3VLTextAttention,
    "qwen3_vl.TextModel._deepstack_process": _hf_qwen3vl.Qwen3VLTextModel._deepstack_process,
    "qwen3_vl.VisionModel.fast_pos_embed_interpolate": _hf_qwen3vl.Qwen3VLVisionModel.fast_pos_embed_interpolate,
    "qwen3_vl.VisionModel.rot_pos_emb": _hf_qwen3vl.Qwen3VLVisionModel.rot_pos_emb,
    "qwen3_vl.VisionModel.forward": _hf_qwen3vl.Qwen3VLVisionModel.forward,
    "qwen3_vl.Model.cls": _hf_qwen3vl.Qwen3VLModel,
    "qwen3_vl.CausalLM.cls": _hf_qwen3vl.Qwen3VLForConditionalGeneration,
    # qwen3_vl_moe (apply_veomni_qwen3vlmoe_patch)
    "qwen3_vl_moe.CausalLM.cls": _hf_qwen3vlmoe.Qwen3VLMoeForConditionalGeneration,
    "qwen3_vl_moe.Model.cls": _hf_qwen3vlmoe.Qwen3VLMoeModel,
    "qwen3_vl_moe.TextModel._deepstack_process": _hf_qwen3vlmoe.Qwen3VLMoeTextModel._deepstack_process,
    "qwen3_vl_moe.VisionModel.forward": _hf_qwen3vlmoe.Qwen3VLMoeVisionModel.forward,
    "qwen3_vl_moe.VisionAttention.forward": _hf_qwen3vlmoe.Qwen3VLMoeVisionAttention.forward,
    "qwen3_vl_moe.TextSparseMoeBlock.forward": _hf_qwen3vlmoe.Qwen3VLMoeTextSparseMoeBlock.forward,
    "qwen3_vl_moe.TextExperts.cls": _hf_qwen3vlmoe.Qwen3VLMoeTextExperts,
    # qwen2_5_omni (apply_veomni_qwen25omni_patch)
    "qwen2_5_omni.CausalLM.cls": _hf_qwen25omni.Qwen2_5OmniForConditionalGeneration,
    "qwen2_5_omni.ThinkerCausalLM.cls": _hf_qwen25omni.Qwen2_5OmniThinkerForConditionalGeneration,
    "qwen2_5_omni.VisionEncoder.cls": _hf_qwen25omni.Qwen2_5OmniVisionEncoder,
    "qwen2_5_omni.AudioEncoder.cls": _hf_qwen25omni.Qwen2_5OmniAudioEncoder,
    "qwen2_5_omni.VisionAttention.forward": _hf_qwen25omni.Qwen2_5OmniVisionAttention.forward,
    "qwen2_5_omni.PreTrainedForCondGen.get_rope_index": (
        _hf_qwen25omni.Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index
    ),
    # qwen3_omni_moe (apply_veomni_qwen3_omni_moe_patch). `_no_split_modules`
    # is a list class attribute; copy it so a future in-place mutation can't
    # corrupt the snapshot.
    "qwen3_omni_moe.PreTrained._no_split_modules": list(_hf_q3omnimoe.Qwen3OmniMoePreTrainedModel._no_split_modules),
    "qwen3_omni_moe.PreTrainedForCondGen.get_rope_index": (
        _hf_q3omnimoe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index
    ),
    "qwen3_omni_moe.PreTrained.init_weights": _hf_q3omnimoe.Qwen3OmniMoePreTrainedModel._init_weights,
    "qwen3_omni_moe.VisionAttention.forward": _hf_q3omnimoe.Qwen3OmniMoeVisionAttention.forward,
    "qwen3_omni_moe.VisionEncoder.cls": _hf_q3omnimoe.Qwen3OmniMoeVisionEncoder,
    "qwen3_omni_moe.AudioEncoder.cls": _hf_q3omnimoe.Qwen3OmniMoeAudioEncoder,
    "qwen3_omni_moe.ThinkerTextModel.cls": _hf_q3omnimoe.Qwen3OmniMoeThinkerTextModel,
    "qwen3_omni_moe.ThinkerTextSparseMoeBlock.cls": _hf_q3omnimoe.Qwen3OmniMoeThinkerTextSparseMoeBlock,
    "qwen3_omni_moe.ThinkerCausalLM.cls": _hf_q3omnimoe.Qwen3OmniMoeThinkerForConditionalGeneration,
    "qwen3_omni_moe.CausalLM.cls": _hf_q3omnimoe.Qwen3OmniMoeForConditionalGeneration,
}


def apply_veomni_hf_unpatch():
    """Undo in-place veomni monkey-patches on HF model modules.

    `apply_veomni_*_patch()` in each of `qwen3/`, `qwen3_moe/`, `deepseek_v3/`,
    `qwen2_vl/`, `qwen2_5vl/`, `qwen3_vl/`, `qwen3_vl_moe/`, `qwen2_5_omni/`
    and `qwen3_omni_moe/` mutates the HF model modules directly (forward swaps,
    class swaps, new `get_parallel_plan` / `dummy_forward` methods). Without
    this restore, the first parametrize case to build a veomni model leaks
    its patches into every subsequent HF build in the same test session,
    silently turning HF-vs-VeOmni comparisons into VeOmni-vs-VeOmni.
    """
    # Step 1: restore in-class forward methods on pristine class objects.
    # This handles both the always-on forward swaps and the Triton branch's
    # in-class mutations (DeepseekV3RotaryEmbedding.forward / DeepseekV3RMSNorm.forward).
    _PRISTINE_HF["qwen3.RMSNorm.cls"].forward = _PRISTINE_HF["qwen3.RMSNorm.forward"]
    _PRISTINE_HF["qwen3_moe.RMSNorm.cls"].forward = _PRISTINE_HF["qwen3_moe.RMSNorm.forward"]
    _PRISTINE_HF["ds3.RotaryEmb.cls"].forward = _PRISTINE_HF["ds3.RotaryEmb.forward"]
    _PRISTINE_HF["ds3.RMSNorm.cls"].forward = _PRISTINE_HF["ds3.RMSNorm.forward"]

    _hf_qwen3.Qwen3ForCausalLM.forward = _PRISTINE_HF["qwen3.CausalLM.forward"]
    _hf_qwen3.Qwen3ForSequenceClassification.forward = _PRISTINE_HF["qwen3.SeqCls.forward"]
    _hf_qwen3_moe.Qwen3MoeForCausalLM.forward = _PRISTINE_HF["qwen3_moe.CausalLM.forward"]
    _hf_qwen3_moe.Qwen3MoePreTrainedModel._init_weights = _PRISTINE_HF["qwen3_moe.PreTrained.init_weights"]
    _hf_ds3.DeepseekV3Attention.forward = _PRISTINE_HF["ds3.Attention.forward"]
    _hf_ds3.DeepseekV3ForCausalLM.forward = _PRISTINE_HF["ds3.CausalLM.forward"]
    _hf_ds3.DeepseekV3PreTrainedModel._init_weights = _PRISTINE_HF["ds3.PreTrained.init_weights"]

    # qwen2_vl: VisionAttention.forward only — class swaps are handled in step 2.
    _hf_qwen2vl.VisionAttention.forward = _PRISTINE_HF["qwen2_vl.VisionAttention.forward"]
    # qwen2_5_vl: vision-tower forwards on classes that aren't themselves swapped.
    _hf_qwen25vl.Qwen2_5_VLVisionAttention.forward = _PRISTINE_HF["qwen2_5_vl.VisionAttention.forward"]
    _hf_qwen25vl.Qwen2_5_VisionTransformerPretrainedModel.forward = _PRISTINE_HF["qwen2_5_vl.VisionPretrained.forward"]
    # qwen3_vl: vision-tower + text-model in-class methods.
    _hf_qwen3vl.Qwen3VLVisionAttention.forward = _PRISTINE_HF["qwen3_vl.VisionAttention.forward"]
    _hf_qwen3vl.Qwen3VLTextModel._deepstack_process = _PRISTINE_HF["qwen3_vl.TextModel._deepstack_process"]
    _hf_qwen3vl.Qwen3VLVisionModel.fast_pos_embed_interpolate = _PRISTINE_HF[
        "qwen3_vl.VisionModel.fast_pos_embed_interpolate"
    ]
    _hf_qwen3vl.Qwen3VLVisionModel.rot_pos_emb = _PRISTINE_HF["qwen3_vl.VisionModel.rot_pos_emb"]
    _hf_qwen3vl.Qwen3VLVisionModel.forward = _PRISTINE_HF["qwen3_vl.VisionModel.forward"]
    # qwen3_vl_moe: vision + text-MoE in-class methods.
    _hf_qwen3vlmoe.Qwen3VLMoeTextModel._deepstack_process = _PRISTINE_HF["qwen3_vl_moe.TextModel._deepstack_process"]
    _hf_qwen3vlmoe.Qwen3VLMoeVisionModel.forward = _PRISTINE_HF["qwen3_vl_moe.VisionModel.forward"]
    _hf_qwen3vlmoe.Qwen3VLMoeVisionAttention.forward = _PRISTINE_HF["qwen3_vl_moe.VisionAttention.forward"]
    _hf_qwen3vlmoe.Qwen3VLMoeTextSparseMoeBlock.forward = _PRISTINE_HF["qwen3_vl_moe.TextSparseMoeBlock.forward"]
    # qwen2_5_omni: vision-attention forward + rope-index method.
    _hf_qwen25omni.Qwen2_5OmniVisionAttention.forward = _PRISTINE_HF["qwen2_5_omni.VisionAttention.forward"]
    _hf_qwen25omni.Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index = _PRISTINE_HF[
        "qwen2_5_omni.PreTrainedForCondGen.get_rope_index"
    ]
    # qwen3_omni_moe: vision-attention forward + a few PreTrained-class attributes.
    _hf_q3omnimoe.Qwen3OmniMoePreTrainedModel._no_split_modules = list(
        _PRISTINE_HF["qwen3_omni_moe.PreTrained._no_split_modules"]
    )
    _hf_q3omnimoe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index = _PRISTINE_HF[
        "qwen3_omni_moe.PreTrainedForCondGen.get_rope_index"
    ]
    _hf_q3omnimoe.Qwen3OmniMoePreTrainedModel._init_weights = _PRISTINE_HF["qwen3_omni_moe.PreTrained.init_weights"]
    _hf_q3omnimoe.Qwen3OmniMoeVisionAttention.forward = _PRISTINE_HF["qwen3_omni_moe.VisionAttention.forward"]

    # Step 2: restore module-level names for classes / functions that the
    # Liger branch swaps out wholesale, plus the class swaps done by the
    # always-on patch (Qwen3MoeSparseMoeBlock, DeepseekV3MoE, every VLM/Omni
    # *ForConditionalGeneration / *Model, etc.).
    _hf_qwen3.apply_rotary_pos_emb = _PRISTINE_HF["qwen3.apply_rotary_pos_emb"]
    _hf_qwen3.Qwen3RMSNorm = _PRISTINE_HF["qwen3.RMSNorm.cls"]
    _hf_qwen3.Qwen3MLP = _PRISTINE_HF["qwen3.MLP.cls"]
    _hf_qwen3_moe.Qwen3MoeSparseMoeBlock = _PRISTINE_HF["qwen3_moe.MoeBlock.cls"]
    _hf_qwen3_moe.apply_rotary_pos_emb = _PRISTINE_HF["qwen3_moe.apply_rotary_pos_emb"]
    _hf_qwen3_moe.Qwen3MoeRMSNorm = _PRISTINE_HF["qwen3_moe.RMSNorm.cls"]
    _hf_qwen3_moe.Qwen3MoeMLP = _PRISTINE_HF["qwen3_moe.MLP.cls"]
    _hf_ds3.DeepseekV3MoE = _PRISTINE_HF["ds3.MoE.cls"]
    _hf_ds3.apply_rotary_pos_emb = _PRISTINE_HF["ds3.apply_rotary_pos_emb"]
    _hf_ds3.DeepseekV3RotaryEmbedding = _PRISTINE_HF["ds3.RotaryEmb.cls"]
    _hf_ds3.DeepseekV3RMSNorm = _PRISTINE_HF["ds3.RMSNorm.cls"]
    _hf_ds3.DeepseekV3MLP = _PRISTINE_HF["ds3.MLP.cls"]

    _hf_qwen2vl.Qwen2VLForConditionalGeneration = _PRISTINE_HF["qwen2_vl.CausalLM.cls"]
    _hf_qwen2vl.Qwen2VLModel = _PRISTINE_HF["qwen2_vl.Model.cls"]
    _hf_qwen2vl.Qwen2VisionTransformerPretrainedModel = _PRISTINE_HF["qwen2_vl.VisionPretrained.cls"]
    _hf_qwen2vl.Qwen2VLDecoderLayer = _PRISTINE_HF["qwen2_vl.DecoderLayer.cls"]

    _hf_qwen25vl.Qwen2_5_VLModel = _PRISTINE_HF["qwen2_5_vl.Model.cls"]
    _hf_qwen25vl.Qwen2_5_VLForConditionalGeneration = _PRISTINE_HF["qwen2_5_vl.CausalLM.cls"]

    _hf_qwen3vl.Qwen3VLTextAttention = _PRISTINE_HF["qwen3_vl.TextAttention.cls"]
    _hf_qwen3vl.Qwen3VLModel = _PRISTINE_HF["qwen3_vl.Model.cls"]
    _hf_qwen3vl.Qwen3VLForConditionalGeneration = _PRISTINE_HF["qwen3_vl.CausalLM.cls"]

    _hf_qwen3vlmoe.Qwen3VLMoeForConditionalGeneration = _PRISTINE_HF["qwen3_vl_moe.CausalLM.cls"]
    _hf_qwen3vlmoe.Qwen3VLMoeModel = _PRISTINE_HF["qwen3_vl_moe.Model.cls"]
    _hf_qwen3vlmoe.Qwen3VLMoeTextExperts = _PRISTINE_HF["qwen3_vl_moe.TextExperts.cls"]

    _hf_qwen25omni.Qwen2_5OmniForConditionalGeneration = _PRISTINE_HF["qwen2_5_omni.CausalLM.cls"]
    _hf_qwen25omni.Qwen2_5OmniThinkerForConditionalGeneration = _PRISTINE_HF["qwen2_5_omni.ThinkerCausalLM.cls"]
    _hf_qwen25omni.Qwen2_5OmniVisionEncoder = _PRISTINE_HF["qwen2_5_omni.VisionEncoder.cls"]
    _hf_qwen25omni.Qwen2_5OmniAudioEncoder = _PRISTINE_HF["qwen2_5_omni.AudioEncoder.cls"]

    _hf_q3omnimoe.Qwen3OmniMoeVisionEncoder = _PRISTINE_HF["qwen3_omni_moe.VisionEncoder.cls"]
    _hf_q3omnimoe.Qwen3OmniMoeAudioEncoder = _PRISTINE_HF["qwen3_omni_moe.AudioEncoder.cls"]
    _hf_q3omnimoe.Qwen3OmniMoeThinkerTextModel = _PRISTINE_HF["qwen3_omni_moe.ThinkerTextModel.cls"]
    _hf_q3omnimoe.Qwen3OmniMoeThinkerTextSparseMoeBlock = _PRISTINE_HF["qwen3_omni_moe.ThinkerTextSparseMoeBlock.cls"]
    _hf_q3omnimoe.Qwen3OmniMoeThinkerForConditionalGeneration = _PRISTINE_HF["qwen3_omni_moe.ThinkerCausalLM.cls"]
    _hf_q3omnimoe.Qwen3OmniMoeForConditionalGeneration = _PRISTINE_HF["qwen3_omni_moe.CausalLM.cls"]

    # Step 3: remove `get_parallel_plan` / `dummy_forward` methods that the
    # patches inject onto HF classes (no native version to restore to).
    for cls in (_hf_qwen3_moe.Qwen3MoeForCausalLM, _hf_ds3.DeepseekV3ForCausalLM):
        if "get_parallel_plan" in cls.__dict__:
            delattr(cls, "get_parallel_plan")
    for cls in (_hf_qwen3vlmoe.Qwen3VLMoePreTrainedModel, _hf_q3omnimoe.Qwen3OmniMoePreTrainedModel):
        if "get_parallel_plan" in cls.__dict__:
            delattr(cls, "get_parallel_plan")
    for cls in (
        _hf_qwen25vl.Qwen2_5_VisionTransformerPretrainedModel,
        _hf_qwen3vl.Qwen3VLVisionModel,
        _hf_qwen3vlmoe.Qwen3VLMoeVisionModel,
    ):
        if "dummy_forward" in cls.__dict__:
            delattr(cls, "dummy_forward")
