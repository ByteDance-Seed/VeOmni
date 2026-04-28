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
from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen3_omni_moe")
def register_qwen3_omni_moe_config():
    # The veomni subclass forces tie_word_embeddings=False to match reality: the
    # top-level Qwen3OmniMoeForConditionalGeneration is a container over
    # `thinker`/`talker` with no container-level `embed_tokens` or `lm_head`, so
    # post-load embedding tying must be a no-op. Applied on both v4 and v5
    # branches — upstream HF keeps the default True, which would drive
    # post_process_after_weight_loading into an unresolvable get_input_embeddings
    # fallback. See configuration_qwen3_omni_moe.py for the rationale.
    from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig, apply_veomni_qwen3_omni_moe_patch

    apply_veomni_qwen3_omni_moe_patch()
    return Qwen3OmniMoeConfig


@MODELING_REGISTRY.register("qwen3_omni_moe")
def register_qwen3_omni_moe_modeling(architecture: str):
    if is_transformers_version_greater_or_equal_to("5.2.0"):
        # Talker classes are not subclassed locally; they live only in upstream
        # transformers and are not trained via VeOmni's training path.
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerForConditionalGeneration,
            Qwen3OmniMoeTalkerModel,
        )

        from .checkpoint_tensor_converter import create_qwen3_omni_moe_checkpoint_tensor_converter
        from .generated.patched_modeling_qwen3_omni_moe_gpu import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerTextModel,
        )

        # The thinker text submodel is also loadable standalone (e.g. when the
        # registry dispatches on architecture == "...ThinkerTextModel"), so the
        # converter must be attached to each class that may be the load entry.
        for model_cls in (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerTextModel,
        ):
            model_cls._create_checkpoint_tensor_converter = staticmethod(
                create_qwen3_omni_moe_checkpoint_tensor_converter
            )
    else:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerForConditionalGeneration,
            Qwen3OmniMoeTalkerModel,
        )

        from .checkpoint_tensor_converter_v4 import create_qwen3_omni_moe_v4_checkpoint_tensor_converter
        from .modeling_qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerTextModel,
            apply_veomni_qwen3_omni_moe_patch,
        )

        apply_veomni_qwen3_omni_moe_patch()

        # Stack per-expert HF weights into the thinker's three 3-D nn.Parameters at
        # load time (Qwen3OmniMoeThinkerExperts.gate_proj/up_proj/down_proj). The
        # factory returns None when `_moe_implementation != "fused"`, so eager-mode
        # loads (which use nn.ModuleList for the thinker too) are unaffected.
        for model_cls in (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerTextModel,
        ):
            model_cls._create_checkpoint_tensor_converter = staticmethod(
                create_qwen3_omni_moe_v4_checkpoint_tensor_converter
            )

    if "ThinkerTextModel" in architecture:
        return Qwen3OmniMoeThinkerTextModel
    if "ThinkerForConditionalGeneration" in architecture:
        return Qwen3OmniMoeThinkerForConditionalGeneration
    if "TalkerModel" in architecture:
        return Qwen3OmniMoeTalkerModel
    if "TalkerForConditionalGeneration" in architecture:
        return Qwen3OmniMoeTalkerForConditionalGeneration
    if "ForConditionalGeneration" in architecture:
        return Qwen3OmniMoeForConditionalGeneration
    return Qwen3OmniMoeForConditionalGeneration


@MODEL_PROCESSOR_REGISTRY.register("Qwen3OmniMoeProcessor")
def register_qwen3_omni_moe_processor():
    # The veomni subclass is required on both v4 and v5 branches: VeOmni's data
    # pipeline calls the processor with `audios=` (plural) and passes empty
    # lists for missing modalities, while upstream's signature is `audio=`
    # (singular) with `if audio is not None` checks. These are data-format
    # patches, independent of transformers version.
    from .processing_qwen3_omni_moe import Qwen3OmniMoeProcessor, apply_veomni_qwen3_omni_moe_patch

    apply_veomni_qwen3_omni_moe_patch()
    return Qwen3OmniMoeProcessor
