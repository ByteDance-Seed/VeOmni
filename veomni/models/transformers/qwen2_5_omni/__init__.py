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
from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen2_5_omni")
def register_qwen2_5_omni_config():
    # The veomni Qwen2_5OmniConfig forces tie_word_embeddings=False to match
    # reality: the top-level Qwen2_5OmniForConditionalGeneration is a container
    # over thinker / talker / token2wav with no container-level embed_tokens or
    # lm_head, so post-load embedding tying must be a no-op. Upstream HF keeps
    # the default True, which would drive post_process_after_weight_loading
    # into an unresolvable get_input_embeddings fallback. See
    # configuration_qwen2_5_omni.py for the full rationale.
    from .configuration_qwen2_5_omni import Qwen2_5OmniConfig

    return Qwen2_5OmniConfig


@MODELING_REGISTRY.register("qwen2_5_omni")
def register_qwen2_5_omni_modeling(architecture: str):
    # Talker classes are not subclassed locally; they live only in upstream
    # transformers and are not trained via VeOmni's training path. The
    # generated module excludes them via ``config.exclude_from_output``
    # (see qwen2_5_omni_gpu_patch_gen_config.py).
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniTalkerForConditionalGeneration,
        Qwen2_5OmniTalkerModel,
    )

    from .generated.patched_modeling_qwen2_5_omni_gpu import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    if "TalkerModel" in architecture:
        return Qwen2_5OmniTalkerModel
    if "TalkerForConditionalGeneration" in architecture:
        return Qwen2_5OmniTalkerForConditionalGeneration
    if "ThinkerForConditionalGeneration" in architecture:
        return Qwen2_5OmniThinkerForConditionalGeneration
    if "ForConditionalGeneration" in architecture:
        return Qwen2_5OmniForConditionalGeneration
    return Qwen2_5OmniForConditionalGeneration


@MODEL_PROCESSOR_REGISTRY.register("Qwen2_5OmniProcessor")
def register_qwen2_5_omni_processor():
    # The veomni subclass is required because VeOmni's data pipeline calls the
    # processor with `audios=` (plural) and passes empty lists for missing
    # modalities, while upstream's signature is `audio=` (singular) with
    # `if audio is not None` checks. These are data-format patches, independent
    # of transformers version.
    from .processing_qwen2_5_omni import Qwen2_5OmniProcessor

    return Qwen2_5OmniProcessor
