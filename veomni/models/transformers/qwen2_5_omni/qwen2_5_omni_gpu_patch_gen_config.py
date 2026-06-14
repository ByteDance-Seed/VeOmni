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
Patch configuration for Qwen2.5-Omni transformers>=5.9.0 code generation.

Covers the thinker training path (text + vision + audio, dense — no MoE):
  - PreTrained.get_rope_index with per-video use_audio_in_video derived from
    audio_seqlens (interleaved video-with-audio / video-without-audio in the
    same batch) and tolerant of attention_mask=None
  - Vision attention dispatch via VARLEN_ATTENTION_TYPES (covers the
    veomni_flash_attention_* custom names)
  - Vision encoder forward with SP gather/slice (full + windowed cu_seqlens
    extended for the SP pad tail) and a new dummy_forward for FSDP rank
    asymmetry
  - Audio encoder _prepare_attention_mask returning None on VARLEN paths,
    forward with input_features (len, mel) permute + SP gather/strip + SP
    slice + cu_seqlens extension, plus a new dummy_forward
  - Thinker.get_audio_features simplified to VeOmni's flat (len, mel) inputs
    (no feature_attention_mask)
  - Thinker.get_position_id_func — multiprocessing-safe per-sample closure
    that converts VeOmni multimodal token ids into 3D position-ids at data
    preprocessing time
  - Thinker.forward: pre-computed image/video/audio masks (popped from
    kwargs), SP-aware embed gather+scatter, FSDP dummy ViT/audio forward on
    ranks without modality, fused loss via OpSlot + self.loss_function,
    precomputed multimodal position-ids transposed from (bs, 3, L) to
    (3, bs, L), filtered zero-length audio_feature_lengths
  - ForConditionalGeneration.__init__: force has_talker=False, pin
    _no_split_modules to {ThinkerDecoderLayer, VisionBlock, AudioEncoderLayer},
    drop talker.*/token2wav.* keys on state-dict load
  - ForConditionalGeneration.enable_talker: raise NotImplementedError (the
    talker / token2wav modules are excluded from the generated file)
  - ForConditionalGeneration.forward: delegate to thinker only
  - ForConditionalGeneration.get_position_id_func: delegate to thinker

Regen command:
patchgen veomni.models.transformers.qwen2_5_omni.qwen2_5_omni_gpu_patch_gen_config -o veomni/models/transformers/qwen2_5_omni/generated --diff
"""

import copy
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    pad_tensor,
    slice_input_tensor,
    unpad_tensor,
)
from veomni.models.transformers.attention_utils import VARLEN_ATTENTION_TYPES
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import (
    AUDIO_INPUT_INDEX,
    IGNORE_INDEX,
    IMAGE_INPUT_INDEX,
    VIDEO_INPUT_INDEX,
)
from veomni.utils.model_outputs import Qwen2_5OmniThinkerCausalLMOutputWithLogProbs


config = PatchConfig(
    source_module="transformers.models.qwen2_5_omni.modeling_qwen2_5_omni",
    target_file="patched_modeling_qwen2_5_omni_gpu.py",
    description="Qwen2.5-Omni thinker with VeOmni v5 compatibility (SP + FSDP + fused loss)",
)


# ================================================================
# Additional imports needed by the patched methods in the generated file
# ================================================================
config.add_import("copy", is_from_import=False)
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("torch.nn.functional", alias="F", is_from_import=False)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=[
        "gather_outputs",
        "pad_tensor",
        "slice_input_tensor",
        "unpad_tensor",
    ],
)
config.add_import("veomni.models.transformers.attention_utils", names=["VARLEN_ATTENTION_TYPES"])
# Surface ``Qwen2_5OmniThinkerCausalLMOutputWithLogProbs`` so the patched
# ``Qwen2_5OmniThinkerForConditionalGeneration.forward`` can return per-token
# log-probs / entropy as constructor fields while preserving ``rope_deltas``.
# Mutating ``output.log_probs`` / ``output.entropy`` after the base-class
# constructor would bypass ``ModelOutput`` pytree flattening, breaking FSDP2's
# pre-backward unshard hook on ``lm_head`` and triggering
# ``setStorage … storage of size 0`` in ``chunk_logprobs.backward``.
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "Qwen2_5OmniThinkerCausalLMOutputWithLogProbs"],
)
config.drop_import_names("Qwen2_5OmniThinkerCausalLMOutputWithPast")

config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Only the Thinker forward is in VeOmni's training path (Talker / Token2Wav
    # are inference-only speech paths excluded from the generated file).
    from veomni.ops.dispatch import OpSlot
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    """
)
config.add_import(
    "veomni.utils.constants",
    names=["AUDIO_INPUT_INDEX", "IGNORE_INDEX", "IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"],
)


# ================================================================
# Drop talker + token2wav from generated output.
#
# The patched `Qwen2_5OmniForConditionalGeneration.__init__` forces
# `has_talker=False` and never calls `enable_talker()`, so the talker /
# token2wav modules are never instantiated on the training path. But HF's
# `PreTrainedModel.post_init` aggregates `_no_split_modules` by walking the
# class hierarchy, so leaving `Qwen2_5OmniTalkerDecoderLayer`,
# `Qwen2_5OmniToken2Wav*`, etc. in the generated module still lets them
# contribute to the aggregation and can pull dead layer names into the FSDP
# no-split set. Excluding them here also trims the generated file by ~1500
# lines and removes an import-time footprint that isn't exercised.
#
# Excluding `Qwen2_5OmniTalkerForConditionalGeneration` /
# `Qwen2_5OmniTalkerModel` means the generated module no longer exports
# them, but `__init__.py` continues to fetch the *upstream* talker classes
# directly from `transformers.models.qwen2_5_omni.modeling_qwen2_5_omni`
# so the registry remains fully populated.
# ================================================================
config.exclude_from_output(
    # Talker (speech LM) — inference-only, never trained
    "Qwen2_5OmniTalkerCausalLMOutputWithPast",
    "Qwen2_5OmniTalkerForConditionalGeneration",
    "Qwen2_5OmniTalkerModel",
    # Token2Wav DiT (speech-token -> mel) — inference-only
    "Qwen2_5OmniToken2WavModel",
    "Qwen2_5OmniToken2WavDiTModel",
    "Qwen2_5OmniToken2WavBigVGANModel",
    "Qwen2_5OmniDiTRotaryEmbedding",
    "DiTAttention",
    "DiTCodecEmbedding",
    "DiTDecoderLayer",
    "DiTInputEmbedding",
    "DiTMLP",
    "DiTTimestepEmbedding",
    "Qwen2_5_OmniAdaLayerNormZero",
    "Qwen2_5_OmniAdaLayerNormZero_Final",
    "SinusPositionEmbedding",
    "RungeKutta4ODESolver",
    # BigVGAN vocoder (speech-mel -> waveform) — inference-only
    "AMPBlock",
    "AttentiveStatisticsPooling",
    "DownSample1d",
    "UpSample1d",
    "ECAPA_TimeDelayNet",
    "Res2NetBlock",
    "SqueezeExcitationBlock",
    "SqueezeExcitationRes2NetBlock",
    "TimeDelayNetBlock",
    "TorchActivation1d",
    "SnakeBeta",
)


# ================================================================
# Module-level helper emitted into the generated file so multiprocessing
# dataloaders can pickle the per-sample position-id closure.
# ================================================================
config.add_post_import_block(
    '''
def get_position_id(main_func, self, **kwargs):
    """Per-sample position-ids for VeOmni dataloader workers.

    Invoked inside the data pipeline (bs=1 per sample). Wraps the HF
    get_rope_index so it can be partial-bound with a SimpleNamespace carrying
    the model config and helpers, then shipped across multiprocessing workers.
    """
    position_ids, rope_deltas = main_func(self, **kwargs)
    assert len(position_ids.shape) == 3 and position_ids.shape[1] == 1
    assert len(rope_deltas.shape) == 2 and rope_deltas.shape[0] == 1
    return {"position_ids": position_ids.squeeze(1), "rope_deltas": rope_deltas.squeeze(0)}
'''
)


# ================================================================
# Patch: Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index
# 1. [PosID] support interleaved video-with-audio vs video-without-audio in
#    one batch. v5 upstream uses a single `use_audio_in_video` boolean flag
#    applied globally; we use `audio_seqlens[audio_idx] == 0` (inherited
#    from the v4 Qwen2.5-Omni convention and matches qwen3_omni_moe) to
#    decide per-video, consuming the zero placeholder to keep `audio_idx`
#    aligned.
# 2. [Mask] tolerate `attention_mask=None` at the data boundary.
# ================================================================
@config.override_method(
    "Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index",
    description="Per-video use_audio_in_video via audio_seqlens + None attention_mask tolerance",
)
def qwen2_5_omni_get_rope_index_patched(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    # --- Patch.1 ---
    # use_audio_in_video removed from the signature; decided per-video below.
    # --- Patch.1 ---
    audio_seqlens: Optional[torch.LongTensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_merge_size = self.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    audio_token_id = self.config.audio_token_id
    vision_start_token_id = self.config.vision_start_token_id
    audio_start_token_id = self.config.audio_start_token_id
    position_id_per_seconds = self.config.position_id_per_seconds
    seconds_per_chunk = self.config.seconds_per_chunk
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        # --- Patch.2 ---
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        # --- Patch.2 ---
        attention_mask = attention_mask == 1
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_idx, video_idx, audio_idx = 0, 0, 0
        for i, input_ids in enumerate(total_input_ids):
            # --- Patch.2 ---
            input_ids = input_ids[attention_mask[i]]
            # --- Patch.2 ---
            image_nums, video_nums, audio_nums = 0, 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)

            # --- Patch.1 ---
            audio_start_indices = torch.argwhere(input_ids == audio_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = torch.sum(
                input_ids[audio_start_indices - 1] != vision_start_token_id
            )  # audio but not in <video><audio>
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum() + (vision_tokens == audio_start_token_id).sum()
            # --- Patch.1 ---

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = image_nums + video_nums + audio_nums

            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                min_ed = min(ed_image, ed_video, ed_audio)
                if min_ed == ed_audio:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + audio_len + eos_len
                    audio_idx += 1
                    remain_audios -= 1

                elif min_ed == ed_image:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                    llm_pos_ids = self.get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + image_len + eos_len
                    image_idx += 1
                    remain_images -= 1

                elif min_ed == ed_video:
                    # --- Patch.1 ---
                    if audio_seqlens[audio_idx] == 0:
                        use_audio_in_video = False
                        audio_idx += 1  # consume zero-length placeholder
                    else:
                        use_audio_in_video = True
                    # --- Patch.1 ---

                    if not use_audio_in_video:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    else:
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )

                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        sub_len = 0
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]
                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                                )
                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]
                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                                )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            # --- Patch.2 ---
            position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
            # --- Patch.2 ---
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)

        return position_ids, mrope_position_deltas
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas


# ================================================================
# Patch: Qwen2_5OmniPreTrainedModel._init_weights
# Upstream branches on `isinstance(module, UpSample1d)` and
# `isinstance(module, DownSample1d)`; both classes are excluded from the
# generated file (training never instantiates the BigVGAN vocoder), so
# those name lookups would fail at runtime during `post_init`. Drop the
# UpSample1d / DownSample1d branches — everything else matches upstream
# verbatim.
# ================================================================
# These names (`init`, `np`, `SinusoidsPositionEmbedding`,
# `Qwen2_5_VisionRotaryEmbedding`) are resolved from the generated file's
# module namespace at import time, not from this patch config — patchgen
# lifts the function body verbatim into the generated class.
@config.override_method(
    "Qwen2_5OmniPreTrainedModel._init_weights",
    description="Drop UpSample1d / DownSample1d branches since both classes are excluded from the generated file",
)
@torch.no_grad()
def qwen2_5_omni_pretrained_init_weights_patched(self, module):
    super()._init_weights(module)
    if isinstance(module, SinusoidsPositionEmbedding):  # noqa: F821
        log_timescale_increment = np.log(module.max_timescale) / (module.channels // 2 - 1)  # noqa: F821
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(module.channels // 2).float())
        scaled_time = torch.arange(module.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]  # noqa: F821
        init.copy_(module.positional_embedding, torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1))  # noqa: F821
    elif isinstance(module, Qwen2_5_VisionRotaryEmbedding):  # noqa: F821
        inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
        init.copy_(module.inv_freq, inv_freq)  # noqa: F821


# ================================================================
# Patch: Qwen2_5OmniAudioEncoder._prepare_attention_mask
# 1. Return None for any VARLEN_ATTENTION_TYPES backend (including
#    veomni_flash_attention_*); upstream only checks the HF built-in
#    flash-attention name via `is_flash_attention_requested`.
# ================================================================
@config.override_method(
    "Qwen2_5OmniAudioEncoder._prepare_attention_mask",
    description="Return None on all VARLEN attention backends (veomni_flash_attention_* included)",
)
def qwen2_5_omni_audio_prepare_attention_mask_patched(
    self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    # --- Patch.1 ---
    if self.config._attn_implementation in VARLEN_ATTENTION_TYPES:
        return None
    # --- Patch.1 ---

    seq_length = inputs_tensor.shape[0]
    attention_mask = torch.full(
        [1, 1, seq_length, seq_length],
        torch.finfo(inputs_tensor.dtype).min,
        device=inputs_tensor.device,
        dtype=inputs_tensor.dtype,
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    return attention_mask


# ================================================================
# Patch: Qwen2_5OmniAudioEncoder.forward
# 1. [Data] VeOmni ships input_features as (len, num_mel_bins); permute to
#    (num_mel_bins, len) to match the HF contract.
# 2. [SP] Gather along time, strip SP-padding so chunking is deterministic.
# 3. [SP] Slice hidden_states along seq dim before encoder layers; extend
#    cu_seqlens for the padded tail on the last rank.
# 4. [SP] Gather token_audio output, strip residual SP-padding, then re-slice
#    so the caller receives an SP-sharded tensor with consistent layout.
# ================================================================
@config.override_method(
    "Qwen2_5OmniAudioEncoder.forward",
    description="Permute VeOmni (len, mel) input + SP gather/strip + SP slice + extend cu_seqlens",
)
def qwen2_5_omni_audio_forward_patched(self, input_features, feature_lens=None, aftercnn_lens=None, **kwargs):
    r"""
    feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
        mel length
    aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
        mel length after cnn
    """
    # The LM-level FlashAttentionKwargs (``cu_seq_lens_q/k``, ``max_length_q/k``)
    # are injected by the data collator for packed-sequence attention and ride
    # through Thinker.forward's ``**kwargs``. They must not reach the audio
    # encoder layers — upstream's ``Qwen2_5OmniAudioAttention.forward`` passes
    # vision-local ``cu_seq_lens_q=cu_seqlens`` explicitly AND forwards
    # ``**kwargs``, which would otherwise raise ``TypeError: got multiple
    # values for keyword argument 'cu_seq_lens_q'`` (mirrors the strip we do
    # in the qwen3_vl / qwen2_5_vl / qwen3_omni_moe ViT forwards).
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        kwargs.pop(key, None)

    # --- Patch.1 ---
    input_features = input_features.permute(1, 0)  # (len, num_mel_bins) -> (num_mel_bins, len)
    # --- Patch.1 ---

    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

    chunk_lengths = torch.tensor(
        [self.n_window * 2] * chunk_num.sum(),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

    # --- Patch.2 ---
    if get_parallel_state().sp_enabled:
        unpadded_dim_len = torch.sum(chunk_lengths)
        input_features = gather_outputs(input_features, gather_dim=1, group=get_parallel_state().sp_group)
        sp_padding_size = input_features.size(1) - unpadded_dim_len
        if sp_padding_size > 0:
            input_features = unpad_tensor(input_features, dim=1, padding_size=sp_padding_size)
    # --- Patch.2 ---

    chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
    padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
        chunk_list, chunk_lengths, padding_value=0, padding_side="right"
    )
    padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
    padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

    padded_embed = padded_embed + self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(
        0
    ).to(padded_embed.dtype)
    hidden_states = padded_embed[padded_mask_after_cnn]
    cu_seqlens = torch.cat(
        (
            torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )
    ).to(torch.int32)
    # --- Patch.3 ---
    if get_parallel_state().sp_enabled:
        unpadded_dim_len = cu_seqlens[-1]
        hidden_states = slice_input_tensor(hidden_states, dim=0, group=get_parallel_state().sp_group)
        pad_seq_len = hidden_states.size(0) * get_parallel_state().sp_size - unpadded_dim_len
        if pad_seq_len > 0:
            # Add this extra sequence to cu_seqlens with the padding length so the
            # varlen kernel attends within the padding chunk rather than across
            # samples on the last rank.
            new_cumsum = cu_seqlens[-1] + pad_seq_len
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
    # --- Patch.3 ---

    for encoder_layer in self.layers:
        # NB: ``attention_mask`` is intentionally not forwarded here. The
        # audio attention path is varlen (cu_seqlens-driven, flash-attn)
        # and does not consume an explicit 4D mask; upstream encoder_layer
        # in 5.9 also calls ``encoder_layer(hidden_states, cu_seqlens=...)``
        # without an ``attention_mask`` kwarg. Passing it here used to
        # work pre-5.9 but now collides with ``attention_mask=None``
        # explicitly passed by ``*VisionAttention.forward`` in
        # ``attention_interface(..., attention_mask=None, ..., **kwargs)``,
        # raising ``TypeError: got multiple values for keyword argument
        # 'attention_mask'``.
        layer_outputs = encoder_layer(
            hidden_states,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        hidden_states = layer_outputs[0]

    # --- Patch.4 ---
    if get_parallel_state().sp_enabled:
        hidden_states = gather_outputs(hidden_states, gather_dim=0, group=get_parallel_state().sp_group)
        sp_padding_size = hidden_states.size(0) - unpadded_dim_len
        if sp_padding_size > 0:
            hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
    # --- Patch.4 ---

    hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
    token_audio_list = []
    for each_audio_states in hidden_states_list:
        each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
        each_audio_states = self.ln_post(each_audio_states)
        each_audio_states = self.proj(each_audio_states)
        token_audio_list.append(each_audio_states)
    token_audio = torch.cat(token_audio_list, dim=0)

    # --- Patch.4 ---
    if get_parallel_state().sp_enabled:
        token_audio = slice_input_tensor(token_audio, dim=0, group=get_parallel_state().sp_group)
    # --- Patch.4 ---

    return BaseModelOutputWithPooling(last_hidden_state=token_audio)


# ================================================================
# Patch: Qwen2_5OmniAudioEncoder.dummy_forward (NEW method)
# [FSDP] Drive a synthetic forward so reduce-scatter stays in sync on ranks
# with no audio data. Dtype is pulled from `self.conv1.weight.dtype` at call
# time — under FSDP2 + MixedPrecision, `self.dtype` (via
# `next(self.parameters()).dtype`) may still report the sharded full
# precision (fp32) while the per-module compute dtype has already been cast
# to bf16; using the conv weight's live dtype keeps the dummy inputs matched
# to whatever the conv actually runs in. We do NOT cache (`_dummy_data`)
# because the cached tensor would stay at first-call dtype and break on
# subsequent calls that enter a different mixed-precision context.
# ================================================================
@config.override_method(
    "Qwen2_5OmniAudioEncoder.dummy_forward",
    description="FSDP dummy forward with conv-weight dtype lookup (no caching) to stay bf16-safe",
)
def qwen2_5_omni_audio_dummy_forward_patched(self):
    # Minimum valid length is one chunk (n_window * 2 mel frames).
    min_len = self.n_window * 2
    dtype = self.conv1.weight.dtype
    input_features = torch.zeros((min_len, self.config.num_mel_bins), dtype=dtype, device=self.device)
    feature_lens = torch.tensor([min_len], dtype=torch.long, device=self.device)
    aftercnn_lens, _ = self._get_feat_extract_output_lengths(feature_lens)
    return self(input_features=input_features, feature_lens=feature_lens, aftercnn_lens=aftercnn_lens)


# ================================================================
# Patch: Qwen2_5OmniVisionAttention.forward
# 1. [SP] Dispatch via VARLEN_ATTENTION_TYPES (covers veomni_flash_attention_*
#    custom names) instead of v5 upstream `is_flash_attention_requested`,
#    which only recognizes HF built-in flash-attention names. Without this
#    the SP-appended cu_seqlens padding entry would run through the
#    non-varlen split branch and size-mismatch.
# ================================================================
@config.override_method(
    "Qwen2_5OmniVisionAttention.forward",
    description="Route through VARLEN_ATTENTION_TYPES so veomni_flash_attention_* with cu_seqlens works",
)
def qwen2_5_omni_vision_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    key_states = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    value_states = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
    key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # --- Patch.1 ---
    if self.config._attn_implementation in VARLEN_ATTENTION_TYPES:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        # Other implementations: process each chunk separately.
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)
    # --- Patch.1 ---

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


# ================================================================
# Patch: Qwen2_5OmniVisionEncoder.forward
# 1. [SP] all_gather hidden_states / rotary_pos_emb across the SP group so
#    window indexing operates on the full sequence, strip the SP pad, then
#    re-slice the post-window tensors so each rank holds its shard for the
#    encoder loop.
# 2. [SP] Extend `cu_seqlens` / `cu_window_seqlens` with the padded tail on
#    the last rank, so the varlen kernel attends within the padding chunk
#    rather than spanning sample boundaries.
# 3. [SP] After the encoder loop, gather merged hidden_states, strip pad,
#    apply reverse_indices, then slice back to the per-rank shard.
# ================================================================
@config.override_method(
    "Qwen2_5OmniVisionEncoder.forward",
    description="SP gather/slice around window-attention reshape + extend cu_seqlens for SP pad tail",
)
def qwen2_5_omni_vision_forward_patched(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | BaseModelOutputWithPooling:
    r"""
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `BaseModelOutputWithPooling`: last_hidden_state and pooler_output.
    """
    # Precomputed ViT metadata — a per-modality sub-dict the Thinker forward
    # selects from `multimodal_metadata` and passes as the single
    # `vit_metadata` kwarg. qwen2.5-omni's window-attention ViT consumes
    # cu_seqlens / cu_window_seqlens / window_index (the get_window_index
    # permutation) — all derived host-side by the collator hook. (Unlike
    # qwen2.5-VL it needs no precomputed max_seqlen: the vision attention
    # computes `.max()` on-device, no host sync.) All .get() below fall back
    # to None for callers bypassing MainCollator, in which case the in-forward
    # derivation runs unchanged. See .agents/knowledge/multimodal_metadata.md.
    vit_metadata = kwargs.pop("vit_metadata", None) or {}
    precomputed_grid_thw_list = vit_metadata.get("grid_thw_list")
    precomputed_cu_seqlens = vit_metadata.get("cu_seqlens")
    precomputed_cu_window_seqlens = vit_metadata.get("cu_window_seqlens")
    precomputed_window_index = vit_metadata.get("window_index")
    use_precompute = precomputed_cu_seqlens is not None

    hidden_states = self.patch_embed(hidden_states)

    # `grid_thw_list` — host-side list reused for rotary position-id
    # construction, unpadded_dim_size, and the cu_seqlens fallback below.
    # Materialise once here so the rotary path shares the same tolist sync.
    grid_thw_list = precomputed_grid_thw_list
    if grid_thw_list is None:
        grid_thw_list = grid_thw.tolist()

    # Build ``position_ids`` host-driven, mirroring upstream's
    # ``get_vision_position_ids``. Replaces ``self.rot_pos_emb(grid_thw)`` which
    # in transformers 5.9 is a deprecated shim that (a) emits a per-forward
    # ``FutureWarning`` and (b) re-runs ``grid_thw.tolist()`` internally.
    ms = self.spatial_merge_size
    device = hidden_states.device
    position_ids_list = []
    for t, h, w in grid_thw_list:
        hpos_ids = torch.arange(h, device=device).reshape(h, 1).expand(h, w)
        hpos_ids = hpos_ids.reshape(h // ms, ms, w // ms, ms).transpose(1, 2).flatten()
        wpos_ids = torch.arange(w, device=device).reshape(1, w).expand(h, w)
        wpos_ids = wpos_ids.reshape(h // ms, ms, w // ms, ms).transpose(1, 2).flatten()
        position_ids_list.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    rotary_pos_emb = self.rotary_pos_emb(torch.cat(position_ids_list, dim=0))

    if use_precompute:
        # Collator-precomputed: window_index / cu_seqlens / cu_window_seqlens
        # are CPU tensors (the latter two already carry any SP-pad tail and
        # are unique_consecutive'd). Move to device non-blocking — an H2D
        # copy, not a host-device sync. FA2 needs int32; tracing needs grid's.
        window_index = precomputed_window_index.to(hidden_states.device, non_blocking=True)
        cu_seqlens = precomputed_cu_seqlens.to(
            hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            non_blocking=True,
        )
        cu_window_seqlens = precomputed_cu_window_seqlens.to(
            hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            non_blocking=True,
        )
    else:
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # `unpadded_dim_size` (pre-SP-pad patch count) — derived host-side from
    # the `grid_thw_list` materialised above so the SP padding math stays a
    # Python int and doesn't read `cu_seqlens[-1]` (a 0-D GPU scalar → sync).
    # Precomputed `cu_seqlens` already carries the SP-pad tail, so
    # `cu_seqlens[-1]` is no longer the unpadded count anyway.
    unpadded_dim_size = sum(t * h * w for t, h, w in grid_thw_list)

    # --- Patch.1 ---
    # all_gather hidden_states across SP ranks: window-indexing assumes the
    # full per-image sequence is present locally.
    if get_parallel_state().sp_enabled:
        hidden_states = gather_outputs(hidden_states, gather_dim=0, group=get_parallel_state().sp_group)
        sp_padding_size = hidden_states.size(0) - unpadded_dim_size
        if sp_padding_size > 0:
            hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
    # --- Patch.1 ---

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

    # --- Patch.1 ---
    # Re-pad + slice along seq dim so each rank holds an evenly-sized shard
    # heading into the encoder loop. Extend the cu_seqlens lists so the
    # varlen kernel treats the pad as its own chunk on the last rank.
    if get_parallel_state().sp_enabled:
        if sp_padding_size > 0:
            hidden_states = pad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
            rotary_pos_emb = pad_tensor(rotary_pos_emb, dim=0, padding_size=sp_padding_size)
            # Precomputed cu_seqlens / cu_window_seqlens already carry the
            # SP-pad tail entry; only the fallback path extends them here.
            if not use_precompute:
                new_cumsum = cu_seqlens[-1] + sp_padding_size
                cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
                cu_window_seqlens = torch.cat([cu_window_seqlens, new_cumsum.unsqueeze(0)], dim=0)

        hidden_states = slice_input_tensor(hidden_states, dim=0, group=get_parallel_state().sp_group)
        rotary_pos_emb = slice_input_tensor(rotary_pos_emb, dim=0)
    # --- Patch.1 ---

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            rotary_pos_emb=rotary_pos_emb,
            **kwargs,
        )

    merged_hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)

    # --- Patch.3 ---
    if get_parallel_state().sp_enabled:
        sp_padding_size = merged_hidden_states.size(0) - unpadded_dim_size
        merged_hidden_states = gather_outputs(merged_hidden_states, gather_dim=0, group=get_parallel_state().sp_group)
        if sp_padding_size > 0:
            merged_hidden_states = unpad_tensor(merged_hidden_states, dim=0, padding_size=sp_padding_size)
    # --- Patch.3 ---

    merged_hidden_states = merged_hidden_states[reverse_indices, :]

    # --- Patch.3 ---
    if get_parallel_state().sp_enabled:
        if sp_padding_size > 0:
            merged_hidden_states = pad_tensor(merged_hidden_states, dim=0, padding_size=sp_padding_size)
        merged_hidden_states = slice_input_tensor(merged_hidden_states, dim=0, group=get_parallel_state().sp_group)
    # --- Patch.3 ---

    return BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
    )


# ================================================================
# Patch: Qwen2_5OmniVisionEncoder.dummy_forward (NEW method)
# [FSDP] Drive a synthetic forward so reduce-scatter stays in sync on ranks
# with no pixel_values. See the audio counterpart above for the dtype
# rationale.
# ================================================================
@config.override_method(
    "Qwen2_5OmniVisionEncoder.dummy_forward",
    description="FSDP dummy forward with patch-embed dtype lookup to stay bf16-safe under MixedPrecision",
)
def qwen2_5_omni_vision_dummy_forward_patched(self):
    # Build the `vit_metadata` sub-dict host-side from the Python-int dummy
    # grid: dummy_forward runs inside the Thinker forward (FSDP path for ranks
    # with no real images), so the collator can't precompute it — but the
    # dummy grid is plain ints here, so the host-side derivation runs
    # sync-free. The dummy grid is SP-divisible, so there is no sp-pad tail.
    def _dummy_vit_metadata(t, h, w):
        return _qwen2_5_vit_metadata(  # noqa: F821 defined via add_helper
            [[t, h, w]],
            0,
            self.window_size,
            self.spatial_merge_size,
            self.patch_size,
        )

    dtype = self.patch_embed.proj.weight.dtype
    if get_parallel_state().sp_enabled:
        sp_size = get_parallel_state().sp_size
        pixel_values = torch.zeros((16, 3 * 2 * 14 * 14), dtype=dtype, device=self.device)
        grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
        vit_metadata = _dummy_vit_metadata(1, 4 * sp_size, 4)
    else:
        pixel_values = torch.zeros((16, 3 * 2 * 14 * 14), dtype=dtype, device=self.device)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
        vit_metadata = _dummy_vit_metadata(1, 4, 4)
    return self(hidden_states=pixel_values, grid_thw=grid_thw, vit_metadata=vit_metadata)


# ================================================================
# Patch: Qwen2_5OmniThinkerForConditionalGeneration.get_audio_features
# Simplified to the VeOmni training path: input_features is already the flat
# (len, num_mel_bins) tensor (after the collator strips feature padding),
# and feature_attention_mask is not carried in training. Return the raw
# last_hidden_state to keep the forward body terse.
# ================================================================
@config.override_method(
    "Qwen2_5OmniThinkerForConditionalGeneration.get_audio_features",
    description="Simplify get_audio_features for VeOmni flat (len, mel) inputs — no feature_attention_mask",
)
def qwen2_5_omni_thinker_get_audio_features_patched(
    self,
    input_features,
    audio_feature_lengths=None,
):
    r"""
    audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
        The length of feature shape of each audio in LLM.
    """
    audio_feat_lengths, _audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
        audio_feature_lengths
    )
    audio_outputs = self.audio_tower(
        input_features,
        feature_lens=audio_feature_lengths,
        aftercnn_lens=audio_feat_lengths,
    )
    return audio_outputs.last_hidden_state


# ================================================================
# Patch: Qwen2_5OmniThinkerForConditionalGeneration.get_position_id_func (NEW)
# Returns a per-sample closure that converts VeOmni's multimodal tokens
# (IMAGE_INPUT_INDEX / VIDEO_INPUT_INDEX / AUDIO_INPUT_INDEX) into 3D
# position_ids at data-preprocessing time. SimpleNamespace + unbound
# methods avoid pickling the full model across dataloader workers.
# get_llm_pos_ids_for_vision / get_chunked_index live on the
# Qwen2_5OmniPreTrainedModelForConditionalGeneration parent, so we walk
# the MRO via `type(self)` (which resolves to the Thinker class).
# ================================================================
@config.override_method(
    "Qwen2_5OmniThinkerForConditionalGeneration.get_position_id_func",
    description="Multiprocessing-safe per-sample position-id closure with VeOmni multimodal token ids",
)
def qwen2_5_omni_thinker_get_position_id_func_patched(self):
    fake_config = copy.copy(self.config)
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    fake_config.audio_token_id = AUDIO_INPUT_INDEX
    fake_model = SimpleNamespace(
        config=fake_config,
        spatial_merge_size=self.spatial_merge_size,
        get_llm_pos_ids_for_vision=partial(type(self).get_llm_pos_ids_for_vision, None),
        get_chunked_index=partial(type(self).get_chunked_index, None),
    )
    return partial(
        get_position_id,  # noqa: F821 — defined at module scope via add_post_import_block
        type(self).get_rope_index,
        fake_model,
    )


# ================================================================
# Patch: Qwen2_5OmniThinkerForConditionalGeneration.forward
# 1. [Mask] Pop pre-computed image/video/audio masks from kwargs (set by
#    VeOmni's process_sample) — avoids extra all_gather for full-mask
#    information when using SP.
# 2. [SP] gather_outputs on input/image/video/audio embeddings to perform
#    the multimodal fill-back on the full sequence, then slice back.
# 3. [FSDP] dummy ViT/audio forward on ranks where the modality input is
#    None so reduce-scatter stays in sync.
# 4. [PosIDs] Transpose precomputed position_ids from (bs, 3, L) to
#    (3, bs, L) so the model layer's mrope handler sees the canonical axis
#    order.
# 5. [Loss] Delegate loss to OpSlot-guarded `veomni_causal_lm_loss` first,
#    then fall back to `self.loss_function` (VeOmni's patched LOSS_MAPPING
#    returns `(loss, logits, fused_linear_aux)`).
# 6. [Data] Filter zero-length audio_feature_lengths (placeholder entries
#    for videos without audio) before forwarding the audio tower.
# 7. [LogProbs] Return Qwen2_5OmniThinkerCausalLMOutputWithLogProbs so
#    per-token log-probs / entropy ride along as constructor fields —
#    mutating after the constructor would break FSDP2's lm_head unshard hook
#    on backward.
# ================================================================
@config.override_method(
    "Qwen2_5OmniThinkerForConditionalGeneration.forward",
    description="VeOmni SP + FSDP + precomputed masks + fused loss + log-probs / entropy",
)
def qwen2_5_omni_thinker_forward_patched(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    # --- Patch.1 ---
    # feature_attention_mask removed: VeOmni's collator already produces flat
    # audio_feature_lengths so this signature drops the redundant mask.
    # --- Patch.1 ---
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    # --- Patch.1 ---
    # use_audio_in_video removed: handled per-video in get_rope_index via
    # the audio_seqlens[audio_idx] == 0 convention.
    # --- Patch.1 ---
    cache_position: Optional[torch.LongTensor] = None,
    video_second_per_grid: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen2_5OmniThinkerCausalLMOutputWithLogProbs:
    r"""
    cache_position (`torch.LongTensor`, *optional*):
        Indices describing the positions of the input sequence tokens in the cache.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
        The length of feature shape of each audio in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
        Number of seconds per grid for each video, used for temporal feature mapping.
    """
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.1 ---
    assert "image_mask" in kwargs, "image_mask should have already been computed in process_sample"
    assert "video_mask" in kwargs, "video_mask should have already been computed in process_sample"
    assert "audio_mask" in kwargs, "audio_mask should have already been computed in process_sample"
    image_mask = kwargs.pop("image_mask")
    video_mask = kwargs.pop("video_mask")
    audio_mask = kwargs.pop("audio_mask")
    # --- Patch.1 ---

    # --- Patch.7 ---
    # Bundle the per-modality ViT metadata from `multimodal_metadata`
    # (collator-precomputed; see .agents/knowledge/multimodal_metadata.md)
    # into one `vit_metadata` sub-dict per `get_*_features` call. The patched
    # window-attention ViT.forward reads grid_thw_list / cu_seqlens /
    # cu_window_seqlens / window_index from it, with runtime fallback when
    # absent.
    multimodal_metadata = kwargs.pop("multimodal_metadata", None) or {}
    image_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("image_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_image_cu_seqlens"),
            "cu_window_seqlens": multimodal_metadata.get("vit_image_cu_window_seqlens"),
            "window_index": multimodal_metadata.get("vit_image_window_index"),
        }
    }
    video_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("video_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_video_cu_seqlens"),
            "cu_window_seqlens": multimodal_metadata.get("vit_video_cu_window_seqlens"),
            "window_index": multimodal_metadata.get("vit_video_window_index"),
        }
    }
    # --- Patch.7 ---

    # --- Patch.2 ---
    if self.training and get_parallel_state().sp_enabled:
        inputs_embeds = gather_outputs(inputs_embeds, gather_dim=1, group=get_parallel_state().sp_group)
    # --- Patch.2 ---

    # --- Patch.6 ---
    if input_features is not None:
        valid_mask = audio_feature_lengths != 0
        audio_feature_lengths = audio_feature_lengths[valid_mask]
        if input_features.shape[0] == 0:
            # input_features is (0, dim) when no audio in all videos; skip the
            # audio tower entirely.
            input_features = None
    # --- Patch.6 ---

    if input_features is not None:
        audio_features = self.get_audio_features(
            input_features,
            audio_feature_lengths=audio_feature_lengths,
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.2 ---
        if self.training and get_parallel_state().sp_enabled:
            audio_features = gather_outputs(audio_features, gather_dim=0, group=get_parallel_state().sp_group)
        # --- Patch.2 ---
        audio_features = audio_features[: audio_mask.sum()]
        audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.3 ---
        fake_audio = self.audio_tower.dummy_forward().last_hidden_state.mean() * 0.0
        fake_audio = fake_audio.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_audio
        # --- Patch.3 ---

    if pixel_values is not None:
        image_embeds = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True, **image_vit_kwargs
        ).pooler_output
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.2 ---
        if self.training and get_parallel_state().sp_enabled:
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)
        # --- Patch.2 ---
        # `masked_scatter` consumes exactly `image_mask.sum()` leading rows of
        # `image_embeds`; image-placeholder positions are in vision-token order
        # and the collator pads the vision sequence only at the end, so padded
        # rows are trailing and unused. No `[:n]` slice — drops a host sync.
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.3 ---
        fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.3 ---

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True, **video_vit_kwargs
        ).pooler_output
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.2 ---
        if self.training and get_parallel_state().sp_enabled:
            video_embeds = gather_outputs(video_embeds, gather_dim=0, group=get_parallel_state().sp_group)
        # --- Patch.2 ---
        # As with the image branch: masked_scatter uses exactly
        # `video_mask.sum()` leading rows — no `[:n]` slice, no host sync.
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.3 ---
        fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.3 ---

    # --- Patch.2 ---
    if self.training and get_parallel_state().sp_enabled:
        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=get_parallel_state().sp_group)
    # --- Patch.2 ---

    if attention_mask is not None and position_ids is None:
        if (
            cache_position is None
            or (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
        ):
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                audio_feature_lengths,
                video_second_per_grid,
            )
            rope_deltas = rope_deltas - delta0
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length = input_ids.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    elif position_ids is not None:
        # --- Patch.4 ---
        if position_ids.ndim == 3 and position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()
        # --- Patch.4 ---

    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs[0]

    # --- Patch.5 ---
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss (chunked fused CE
        # when bound, falls back to ``self.loss_function`` otherwise).
        if veomni_causal_lm_loss.use_non_eager_impl:  # noqa: F821 — declared via add_post_import_block
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(  # noqa: F821
                logits=logits,
                labels=labels,
                vocab_size=self.config.get_text_config().vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                ignore_index=IGNORE_INDEX,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            # Modification: VeOmni's patched ``loss_function`` (via
            # LOSS_MAPPING) returns ``(loss, logits, fused_linear_aux)``;
            # unpack to match the OpSlot branch above.
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.get_text_config().vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                ignore_index=IGNORE_INDEX,
                **kwargs,
            )
            if fused_linear_aux is not None:
                # fused_linear_aux path empties loss/logits slots; clear the local 3D
                # logits so output mirrors the OpSlot branch's contract.
                logits = None
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.5 ---

    # --- Patch.7 ---
    return Qwen2_5OmniThinkerCausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
        fused_linear_aux=fused_linear_aux,
    )
    # --- Patch.7 ---


# ================================================================
# Patch: Qwen2_5OmniForConditionalGeneration.__init__
# 1. [Talker] Force `has_talker=False` — VeOmni's training path only forwards
#    through `thinker` (see the patched `forward` below), so constructing
#    the talker and token2wav would only add unused parameters and drag
#    talker / token2wav layers into the FSDP `_no_split_modules`
#    aggregation (HF recursively merges children's `_no_split_modules` at
#    `post_init`, see transformers `modeling_utils.PreTrainedModel.post_init`).
#    Unused talker / token2wav layers FSDP-wrapped but never forwarded cause
#    a rank-desync hang during asymmetric-modality forward.
# 2. [FSDP] After `post_init`, replace the aggregated `_no_split_modules`
#    with the exact VeOmni target set. The upstream top-level
#    `_no_split_modules = ["Qwen2_5OmniTalkerForConditionalGeneration",
#    "Qwen2_5OmniToken2WavModel"]` is now empty after Step 1; we replace it
#    with the three real training-targets so FSDP keeps decoder / vision /
#    audio layers as the unsplittable boundaries.
# 3. [State-dict] Drop talker / token2wav keys on state-dict load.
# ================================================================
@config.override_method(
    "Qwen2_5OmniForConditionalGeneration.__init__",
    description="Skip talker, pin _no_split_modules, drop talker / token2wav keys on state-dict load",
)
def qwen2_5_omni_for_conditional_generation_init_patched(self, config):
    super().__init__(config)
    self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)
    # --- Patch.1 ---
    self.has_talker = False
    # --- Patch.1 ---
    self.speaker_map = {}
    self.post_init()
    # --- Patch.2 ---
    self._no_split_modules = [
        "Qwen2_5OmniDecoderLayer",
        "Qwen2_5OmniVisionBlock",
        "Qwen2_5OmniAudioEncoderLayer",
    ]
    # --- Patch.2 ---

    # --- Patch.3 ---
    # Training builds the model with ``has_talker=False`` and excludes
    # ``Qwen2_5OmniTalker*`` / ``Qwen2_5OmniToken2Wav*`` classes from the
    # generated module entirely. Full pretrained checkpoints and HF-backend
    # state_dicts still carry ``talker.*`` / ``token2wav.*`` keys, and the
    # default strict load_state_dict raises on them. Strip those keys at the
    # top-level prefix before the strict check fires — the rest of the
    # state_dict (thinker.*) loads unchanged.
    def _drop_talker_and_token2wav_keys(
        module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if prefix:  # only filter at the top-level module
            return
        for k in list(state_dict.keys()):
            if k.startswith("talker.") or k.startswith("token2wav."):
                del state_dict[k]

    self.register_load_state_dict_pre_hook(_drop_talker_and_token2wav_keys)
    # --- Patch.3 ---


# ================================================================
# Patch: Qwen2_5OmniForConditionalGeneration.enable_talker
# The talker + token2wav classes are excluded from the generated file
# (training never instantiates them), so the upstream body
# ``self.talker = Qwen2_5OmniTalkerForConditionalGeneration(...)`` would
# fail at import-time static analysis and at call time. Replace with an
# explicit NotImplementedError so the reason is clear if anything reaches
# here.
# ================================================================
@config.override_method(
    "Qwen2_5OmniForConditionalGeneration.enable_talker",
    description="Disable talker / token2wav path in the training modeling (excluded classes)",
)
def qwen2_5_omni_enable_talker_patched(self):
    raise NotImplementedError(
        "talker / token2wav are not available in the VeOmni training modeling. "
        "Use the upstream transformers implementation for TTS generation."
    )


# ================================================================
# Patch: Qwen2_5OmniForConditionalGeneration.generate
# Upstream's body unconditionally instantiates the talker / token2wav and
# walks `self.speaker_map`, both of which are absent in the VeOmni training
# build (has_talker=False, no enable_talker call). It also carries a
# mutable default arg `talker_eos_token_id: list[int] = [8292, 8294]` that
# trips B006 once the body is regenerated in our generated file. Replace
# with an explicit NotImplementedError so anyone calling generate from the
# training entry-point sees a clear message (and to silence B006).
# ================================================================
@config.override_method(
    "Qwen2_5OmniForConditionalGeneration.generate",
    description="Disable TTS generate in the training modeling (talker / token2wav are excluded)",
)
def qwen2_5_omni_generate_patched(self, *args, **kwargs):
    raise NotImplementedError(
        "Qwen2_5OmniForConditionalGeneration.generate is disabled in the VeOmni training modeling "
        "(talker / token2wav are excluded). Use the upstream transformers implementation for TTS generation."
    )


# ================================================================
# Patch: Qwen2_5OmniForConditionalGeneration.forward (NEW)
# Simplified training path: only forward through thinker; talker +
# token2wav are skipped (only used in the TTS generate path).
# ================================================================
@config.override_method(
    "Qwen2_5OmniForConditionalGeneration.forward",
    description="Forward through thinker only (talker / token2wav not trained via this path)",
)
def qwen2_5_omni_for_conditional_generation_forward_patched(
    self,
    **kwargs,
) -> tuple | Qwen2_5OmniThinkerCausalLMOutputWithLogProbs:
    return self.thinker(**kwargs)


# ================================================================
# Patch: Qwen2_5OmniForConditionalGeneration.get_position_id_func (NEW)
# Delegate to the thinker's closure; the data pipeline calls the top-level
# model's get_position_id_func.
# ================================================================
@config.override_method(
    "Qwen2_5OmniForConditionalGeneration.get_position_id_func",
    description="Delegate position-id computation to the thinker submodule",
)
def qwen2_5_omni_top_get_position_id_func_patched(self):
    return self.thinker.get_position_id_func()


# ================================================================
# Patch: Qwen2_5OmniForConditionalGeneration.get_extra_collate_infos (NEW)
# Declare the omni-specific collate rules (audio feature tensors) so the
# trainer doesn't hardcode them by model_type — the model owns its own
# modality-specific collate topology. Tuples are
# (pack_dim, sp_slice, sp_pad_value, sp_pad_scale); MainCollator coerces them.
# ================================================================
@config.override_method(
    "Qwen2_5OmniForConditionalGeneration.get_extra_collate_infos",
    description="Declare omni-specific (audio) collate rules for the VeOmni collator",
)
def qwen2_5_omni_top_get_extra_collate_infos_patched(self):
    return {
        "audio_feature_lengths": (0, False, None, None),
        "input_features": (0, True, 0, 1),
        "audio_mask": (-1, False, 0, 1),
    }


@config.add_helper
def _qwen2_5_vit_metadata(grid_list, sp_padding_size, window_size, spatial_merge_size, patch_size):
    """Host-side port of the qwen2.5-omni window-attention ViT metadata.

    Returns the per-modality `vit_metadata` sub-dict (grid_thw_list,
    cu_seqlens, cu_window_seqlens, window_index, max_seqlen, win_max_seqlen)
    for a list of ``(t, h, w)`` grids. Shared by ``collate_multimodal_metadata``
    (collator path) and ``dummy_forward`` (FSDP path). Pure CPU — the
    window-index loop is a line-for-line port of
    ``Qwen2_5OmniVisionEncoder.get_window_index`` (which otherwise runs in the
    forward and syncs on ``grid_thw.tolist()`` / ``cu_seqlens_tmp.tolist()``).
    Every value it produces is consumed by the ViT forward without a
    host-device sync. ``max_seqlen`` / ``win_max_seqlen`` are emitted for
    contract parity with qwen2.5-VL; the omni ViT doesn't read them (its
    vision attention computes ``.max()`` on-device, no host sync).
    """
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    vit_merger_window_size = window_size // spatial_merge_size // patch_size

    # Plain varlen cu_seqlens — temporal unroll: each (t, h, w) expands to
    # ``t`` cu steps of ``h * w``.
    cu = [0]
    max_hw = 0
    for t, h, w in grid_list:
        hw = h * w
        max_hw = max(max_hw, hw)
        for _ in range(t):
            cu.append(cu[-1] + hw)

    # Window-attention metadata — host-side port of get_window_index.
    window_index_parts = []
    cu_window = [0]
    window_index_id = 0
    for t, h, w in grid_list:
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size
        index = torch.arange(t * llm_grid_h * llm_grid_w).reshape(t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index_parts.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window[-1]
        cu_window.extend(cu_seqlens_tmp.tolist())
        window_index_id += t * llm_grid_h * llm_grid_w
    window_index = torch.cat(window_index_parts, dim=0)

    # unique_consecutive drops zero-length window segments (HF parity); the
    # forward applies it to the device tensor — do it host-side here instead.
    cu_window = torch.unique_consecutive(torch.tensor(cu_window, dtype=torch.int32, device="cpu")).tolist()
    win_max = 0
    for i in range(1, len(cu_window)):
        win_max = max(win_max, cu_window[i] - cu_window[i - 1])

    # SP-pad tail: the collator zero-pads pixel_values to SP-divisible; those
    # patches become one synthetic sequence so both the full-attention and
    # window-attention varlen paths treat them independently. ``pad > 0``
    # always extends strictly, so no unique_consecutive duplicate is created.
    # Discarded after the per-rank slice.
    if sp_padding_size > 0:
        cu.append(cu[-1] + sp_padding_size)
        max_hw = max(max_hw, sp_padding_size)
        cu_window.append(cu_window[-1] + sp_padding_size)
        win_max = max(win_max, sp_padding_size)

    # device='cpu': this runs in CPU dataloader workers — pin to CPU so a
    # global torch.set_default_device('cuda') can't misallocate it.
    return {
        "grid_thw_list": grid_list,
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device="cpu"),
        "max_seqlen": max_hw,
        "cu_window_seqlens": torch.tensor(cu_window, dtype=torch.int32, device="cpu"),
        "win_max_seqlen": win_max,
        "window_index": window_index,
    }


@config.add_helper
def collate_multimodal_metadata(batch, sp_pad, window_size, spatial_merge_size, patch_size):
    """Derive ``multimodal_metadata`` for the Qwen2.5-omni window-attention ViT.

    Module-level so ``get_metadata_collate_func`` can hand it (``partial``-
    closed over the vision-config dims) to VeOmni's collator as a picklable
    callable. Runs purely on CPU inside the collator after SP padding — every
    value it produces is consumed by the ViT forward without a host-device
    sync.

    ``batch`` is the packed (+ SP-padded) batch dict; ``sp_pad`` maps
    ``pixel_values`` / ``pixel_values_videos`` to the number of patch rows the
    SP collator appended. Mutates ``batch`` in place, writing
    ``batch["multimodal_metadata"]``.
    """
    md = {}
    for modality, grid_key, pad_key in (
        ("image", "image_grid_thw", "pixel_values"),
        ("video", "video_grid_thw", "pixel_values_videos"),
    ):
        grid = batch.get(grid_key)
        if grid is None:
            continue
        grid_list = grid.tolist() if torch.is_tensor(grid) else grid
        if not grid_list:
            continue
        sub = _qwen2_5_vit_metadata(  # noqa: F821 defined via add_helper
            grid_list,
            sp_pad.get(pad_key, 0),
            window_size,
            spatial_merge_size,
            patch_size,
        )
        # The omni ViT consumes cu_seqlens / cu_window_seqlens / window_index;
        # `sub` also carries max_seqlen / win_max_seqlen (the shared helper is
        # identical to qwen2.5-VL's) but the omni vision attention computes its
        # varlen `.max()` on-device, so those are not written into the batch.
        md[f"{modality}_grid_thw_list"] = sub["grid_thw_list"]
        md[f"vit_{modality}_cu_seqlens"] = sub["cu_seqlens"]
        md[f"vit_{modality}_cu_window_seqlens"] = sub["cu_window_seqlens"]
        md[f"vit_{modality}_window_index"] = sub["window_index"]

    if md:
        batch["multimodal_metadata"] = md


# ================================================================
# Patch: Qwen2_5OmniThinkerForConditionalGeneration.get_metadata_collate_func
# (NEW) Expose the window-attention ViT metadata derivation (cu_seqlens /
# cu_window_seqlens / window_index) to VeOmni's collator as a picklable
# callable. The collator invokes it after SP padding; deriving the metadata
# CPU-side off the GPU critical path eliminates the host-device syncs the
# ViT.forward would otherwise pay (grid_thw.tolist(), cu_window tolist()).
# `partial` binds the vision-config dims, so the returned callable takes only
# (batch, sp_pad). See .agents/knowledge/multimodal_metadata.md.
# ================================================================
@config.override_method(
    "Qwen2_5OmniThinkerForConditionalGeneration.get_metadata_collate_func",
    description="Expose CPU-side window-attention ViT multimodal-metadata derivation to the VeOmni collator",
)
def qwen2_5_omni_thinker_get_metadata_collate_func_patched(self):
    vc = self.config.vision_config
    # collate_multimodal_metadata is a module-level helper (via add_helper);
    # partial over plain-int config dims keeps the callable picklable for the
    # multiprocessing DataLoader workers.
    return partial(
        collate_multimodal_metadata,  # noqa: F821 defined via add_helper
        window_size=vc.window_size,
        spatial_merge_size=vc.spatial_merge_size,
        patch_size=vc.patch_size,
    )


# ================================================================
# Patch: Qwen2_5OmniForConditionalGeneration.get_metadata_collate_func (NEW)
# Delegate to the thinker's closure; the data pipeline calls the top-level
# model's get_metadata_collate_func (mirrors get_position_id_func).
# ================================================================
@config.override_method(
    "Qwen2_5OmniForConditionalGeneration.get_metadata_collate_func",
    description="Delegate ViT multimodal-metadata derivation to the thinker submodule",
)
def qwen2_5_omni_top_get_metadata_collate_func_patched(self):
    return self.thinker.get_metadata_collate_func()
