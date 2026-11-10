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

"""Optional callbacks for model-specific FLOPs metadata."""

from . import logging


logger = logging.get_logger(__name__)


def _compute_wan_seqlens(micro_batch):
    """Compute DiT sequence lengths from latent shapes."""
    if "latents" not in micro_batch:
        return []
    seq_lens = []
    for latents in micro_batch["latents"]:
        latent_shape = latents.shape
        if len(latent_shape) == 3:
            batch_size, seq_len, _channels = latent_shape
            seq_lens.append(batch_size * seq_len)
            continue
        batch_size = latent_shape[0] if len(latent_shape) == 5 else 1
        _channels, time, height, width = latent_shape[-4:]
        time_out = int((time - 1) / 1 + 1)
        height_out = int((height - 2) / 2 + 1)
        width_out = int((width - 2) / 2 + 1)
        seq_lens.append(batch_size * time_out * height_out * width_out)
    return seq_lens


def _extract_default_dit_metadata(micro_batch):
    """Adapt the existing DiT lengths to the metadata callback interface."""
    return _compute_wan_seqlens(micro_batch), {}


def _extract_minimax_h3_metadata(micro_batch):
    """Return main and text lengths for usable offline H3 inputs."""
    seq_lens, text_lens = [], []
    for sample in micro_batch if isinstance(micro_batch, list) else [micro_batch]:
        packed = sample.get("packed")
        if isinstance(packed, dict):
            packed = [packed]
        if not isinstance(packed, (list, tuple)) or not packed:
            continue
        sample_seq_lens, sample_text_lens = [], []
        for item in packed:
            if not isinstance(item, dict):
                break
            seq_len, text_len = item.get("seq_len"), item.get("text_len")
            positions = [item.get(key) for key in ("text_pos", "img_pos", "audio_pos")]
            if any(value is None for value in (seq_len, text_len, *positions)):
                logger.warning_once(
                    "MiniMax-H3 FLOPs metadata is incomplete or invalid; MFU and token metrics may be underestimated."
                )
                break
            seq_len, text_len = int(seq_len), int(text_len)
            used_len = sum(position.numel() for position in positions)
            if used_len <= 0 or used_len > seq_len or text_len < 0 or text_len > used_len:
                logger.warning_once(
                    "MiniMax-H3 FLOPs metadata is incomplete or invalid; MFU and token metrics may be underestimated."
                )
                break
            sample_seq_lens.append(used_len)
            sample_text_lens.append(text_len)
        else:
            seq_lens.extend(sample_seq_lens)
            text_lens.extend(sample_text_lens)
    return seq_lens, {"h3_text_seqlens": text_lens}


_DEFAULT_DIT = "__default_dit__"
FLOPS_METADATA_CALLBACKS = {
    _DEFAULT_DIT: _extract_default_dit_metadata,
    "MiniMaxH3DiTModel": _extract_minimax_h3_metadata,
}


def get_flops_metadata_callback(config):
    """Select a DiT callback; HF models keep their existing metadata path."""
    if getattr(config, "condition_model_type", None) is None:
        return None
    return FLOPS_METADATA_CALLBACKS.get(getattr(config, "model_type", None), FLOPS_METADATA_CALLBACKS[_DEFAULT_DIT])
