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

from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("hunyuan_image_3_moe")
def register_hunyuan_image_3_config():
    from .configuration_hunyuan_image_3 import HunyuanImage3Config

    return HunyuanImage3Config


@MODEL_PROCESSOR_REGISTRY.register("HunyuanImage3Processor")
def register_hunyuan_image_3_processor():
    # The official checkpoint ships no ``preprocessor_config.json`` and no
    # ``AutoProcessor`` entry in ``auto_map``, so ``AutoProcessor.from_pretrained``
    # cannot build one. The loader falls back to this registry, keyed by the name
    # ``HunyuanImage3Config.processor_class`` advertises.
    from .processing_hunyuan_image_3 import HunyuanImage3Processor

    return HunyuanImage3Processor


@MODELING_REGISTRY.register("hunyuan_image_3_moe")
def register_hunyuan_image_3_modeling(architecture: str):
    if IS_NPU_AVAILABLE:
        raise RuntimeError("The initial HunyuanImage 3 implementation supports NVIDIA GPU only.")
    if architecture != "HunyuanImage3ForCausalMM":
        raise ValueError(f"Unsupported HunyuanImage 3 architecture: {architecture}.")

    # No data-transform import needed here: ``process_sample_hunyuan_image_3``
    # registers itself from ``veomni/data/data_transform.py``, which
    # ``veomni.data.__init__`` already imports.
    from .checkpoint_tensor_converter import create_hunyuan_image_3_checkpoint_tensor_converter
    from .generated.patched_modeling_hunyuan_image_3_gpu import HunyuanImage3ForCausalMM

    HunyuanImage3ForCausalMM._create_checkpoint_tensor_converter = staticmethod(
        create_hunyuan_image_3_checkpoint_tensor_converter
    )
    return HunyuanImage3ForCausalMM


__all__ = [
    "register_hunyuan_image_3_config",
    "register_hunyuan_image_3_modeling",
    "register_hunyuan_image_3_processor",
]
