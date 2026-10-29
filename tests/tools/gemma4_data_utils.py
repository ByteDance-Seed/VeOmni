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

"""Offline native Gemma4 processor with the existing toy model's token IDs."""

import string

from transformers import GemmaTokenizer
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor


CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<|turn>' + ('model' if message.role == 'assistant' else message.role) + '\n' }}"
    "{% if message.content is string %}{{ message.content | trim }}{% else %}"
    "{% for item in message.content %}"
    "{% if item.type == 'text' %}{{ item.text | trim }}"
    "{% elif item.type == 'image' %}{{ '<|image|>' }}"
    "{% elif item.type == 'video' %}{{ '<|video|>' }}"
    "{% elif item.type == 'audio' %}{{ '<|audio|>' }}{% endif %}"
    "{% endfor %}{% endif %}{{ '<turn|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}"
)


def gemma4_processor():
    """Exercise HF patchification, audio features and placeholder expansion offline."""
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
    tokens += list(string.ascii_letters + string.digits + "▁\n.,:!?-")
    tokens += [f"<unused_{i}>" for i in range(len(tokens), 128)]
    special = {
        "sot_token": (113, "<|turn>"),
        "eot_token": (114, "<turn|>"),
        "boi_token": (119, "<|image>"),
        "boa_token": (120, "<|audio>"),
        "image_token": (121, "<|image|>"),
        "eoi_token": (122, "<image|>"),
        "audio_token": (123, "<|audio|>"),
        "eoa_token": (124, "<audio|>"),
    }
    for index, token in special.values():
        tokens[index] = token
    tokens[125] = "<|video|>"
    tokenizer = GemmaTokenizer(
        vocab={token: index for index, token in enumerate(tokens)},
        merges=[],
        **{name: token for name, (_, token) in special.items()},
        extra_special_tokens=["<|video|>"],
    )
    return Gemma4Processor(
        tokenizer=tokenizer,
        image_processor=Gemma4ImageProcessor(patch_size=4, pooling_kernel_size=2, max_soft_tokens=70, do_resize=False),
        video_processor=Gemma4VideoProcessor(
            patch_size=4, pooling_kernel_size=2, max_soft_tokens=70, do_resize=False, do_sample_frames=False
        ),
        feature_extractor=Gemma4AudioFeatureExtractor(feature_size=8),
        chat_template=CHAT_TEMPLATE,
    )
