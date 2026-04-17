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
    if is_transformers_version_greater_or_equal_to("5.2.0"):
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

        return Qwen3OmniMoeConfig

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

        # Fix an upstream HF typo: `Qwen3OmniMoePreTrainedModel._no_split_modules`
        # references `Qwen3OmniMoeDecoderLayer`, which does not exist (the real
        # class is `Qwen3OmniMoeThinkerTextDecoderLayer`). Override on the
        # top-level wrapper so FSDP's no-split map resolves to a real class.
        Qwen3OmniMoeForConditionalGeneration._no_split_modules = [
            "Qwen3OmniMoeThinkerTextDecoderLayer",
            "Qwen3OmniMoeVisionBlock",
            "Qwen3OmniMoeAudioEncoderLayer",
        ]

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

        from .modeling_qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerTextModel,
            apply_veomni_qwen3_omni_moe_patch,
        )

        apply_veomni_qwen3_omni_moe_patch()

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
    if is_transformers_version_greater_or_equal_to("5.2.0"):
        from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

        return Qwen3OmniMoeProcessor

    from .processing_qwen3_omni_moe import Qwen3OmniMoeProcessor, apply_veomni_qwen3_omni_moe_patch

    apply_veomni_qwen3_omni_moe_patch()
    return Qwen3OmniMoeProcessor
