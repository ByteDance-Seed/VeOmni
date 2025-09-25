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

from ..utils.import_utils import (
    is_veomni_patch_available,
)
from .chat_template import build_chat_template
from .data_collator import (
    CollatePipeline,
    DataCollatorWithPacking,
    DataCollatorWithPadding,
    DataCollatorWithPositionIDs,
    MakeMicroBatchCollator,
    TextSequenceShardCollator,
    UnpackDataCollator,
)
from .data_loader import build_dataloader
from .dataset import (
    build_energon_dataset,
    build_interleave_dataset,
    build_iterative_dataset,
    build_mapping_dataset,
)
from .dummy_dataset import build_dummy_dataset
from .multimodal.data_collator import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
)
from .multimodal.multimodal_chat_template import build_multimodal_chat_template


if is_veomni_patch_available():
    # for internal use only
    from veomni_patch.data.streaming import (
        build_byted_dataset,
        build_multisource_dataset,
        build_streaming_dataloader,
        build_vanilla_streaming_dataloader,
    )
else:

    def build_byted_dataset(*args, **kwargs):
        raise NotImplementedError("build_byted_dataset is not available, please install veomni_patch")

    def build_multisource_dataset(*args, **kwargs):
        raise NotImplementedError("build_multisource_dataset is not available, please install veomni_patch")

    def build_streaming_dataloader(*args, **kwargs):
        raise NotImplementedError("build_streaming_dataloader is not available, please install veomni_patch")

    def build_vanilla_streaming_dataloader(*args, **kwargs):
        raise NotImplementedError("build_vanilla_streaming_dataloader is not available, please install veomni_patch")


__all__ = [
    "build_chat_template",
    "build_dataloader",
    "build_byted_dataset",
    "build_multisource_dataset",
    "build_streaming_dataloader",
    "build_vanilla_streaming_dataloader",
    "build_dummy_dataset",
    "build_iterative_dataset",
    "build_mapping_dataset",
    "build_energon_dataset",
    "build_multimodal_chat_template",
    "CollatePipeline",
    "DataCollatorWithPacking",
    "DataCollatorWithPadding",
    "DataCollatorWithPositionIDs",
    "MakeMicroBatchCollator",
    "TextSequenceShardCollator",
    "UnpackDataCollator",
    "OmniDataCollatorWithPacking",
    "OmniDataCollatorWithPadding",
    "OmniSequenceShardCollator",
    "build_interleave_dataset",
]
