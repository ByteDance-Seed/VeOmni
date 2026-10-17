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

"""Config-derived Gemma 4 batches shared by CPU and asymmetric FSDP tests."""

import torch


def make_gemma4_batch(config, modalities, *, device="cpu", dtype=torch.float32):
    """Build real image/video/audio features and matching placeholder tokens."""
    tokens = list(range(3, 11))
    batch = {}
    if "image" in modalities or "video" in modalities:
        grid_size = 2 * config.vision_config.pooling_kernel_size
        num_patches = grid_size**2
        num_tokens = num_patches // config.vision_config.pooling_kernel_size**2
        positions = torch.cartesian_prod(
            torch.arange(grid_size, device=device), torch.arange(grid_size, device=device)
        ).unsqueeze(0)
        pixel_dim = 3 * config.vision_config.patch_size**2
        if "image" in modalities:
            batch["pixel_values"] = torch.rand(1, num_patches, pixel_dim, dtype=dtype, device=device)
            batch["image_position_ids"] = positions
            tokens.extend([config.image_token_id] * num_tokens)
        if "video" in modalities:
            batch["pixel_values_videos"] = torch.rand(1, 2, num_patches, pixel_dim, dtype=dtype, device=device)
            batch["video_position_ids"] = positions.unsqueeze(1).expand(-1, 2, -1, -1).contiguous()
            tokens.extend([config.video_token_id] * (2 * num_tokens))
    if "audio" in modalities:
        feature_length = 4 * config.audio_config.attention_chunk_size
        feature_width = config.audio_config.subsampling_conv_channels[0]
        batch["input_features"] = torch.randn(1, feature_length, feature_width, dtype=dtype, device=device)
        batch["input_features_mask"] = torch.ones(1, feature_length, dtype=torch.bool, device=device)
        tokens.extend([config.audio_token_id] * config.audio_config.attention_chunk_size)
    tokens.extend([11, 12])
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    labels = input_ids.clone()
    for token_id in (config.image_token_id, config.video_token_id, config.audio_token_id):
        labels[labels == token_id] = -100
    batch.update(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(len(tokens), device=device).unsqueeze(0),
        use_cache=False,
    )
    return batch
