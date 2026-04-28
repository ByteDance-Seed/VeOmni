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
import transformers.models.qwen3.modeling_qwen3 as _hf_qwen3
import transformers.models.qwen3_moe.modeling_qwen3_moe as _hf_qwen3_moe


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
}


def apply_veomni_hf_unpatch():
    """Undo in-place veomni monkey-patches on HF model modules.

    `apply_veomni_*_patch()` in each of `qwen3/`, `qwen3_moe/`, `deepseek_v3/`
    mutates the HF model modules directly (forward swaps, class swaps, new
    `get_parallel_plan` methods). Without this restore, the first parametrize
    case to build a veomni model leaks its patches into every subsequent HF
    build in the same test session.
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

    # Step 2: restore module-level names for classes / functions that the
    # Liger branch swaps out wholesale, plus the class swaps done by the
    # always-on patch (Qwen3MoeSparseMoeBlock, DeepseekV3MoE).
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

    # Step 3: remove `get_parallel_plan` methods injected onto HF causal-LM
    # classes by apply_veomni_*_patch.
    for cls in (_hf_qwen3_moe.Qwen3MoeForCausalLM, _hf_ds3.DeepseekV3ForCausalLM):
        if "get_parallel_plan" in cls.__dict__:
            delattr(cls, "get_parallel_plan")
