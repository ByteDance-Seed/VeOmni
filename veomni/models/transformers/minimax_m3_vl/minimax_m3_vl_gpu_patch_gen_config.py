# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
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
Patch configuration for MiniMax M3 VL transformers>=5.12.0 code generation.

The VeOmni integration keeps the upstream MiniMax M3 VL modeling body from
transformers, then adds VeOmni hooks for parallel plans, VLM collator metadata,
and FSDP-symmetric dummy vision execution. The public MiniMaxAI/MiniMax-M3
checkpoint uses an older language/MoE key layout, so runtime checkpoint
conversion is registered separately in checkpoint_tensor_converter.py. The
public index maps the spatial merge projector through `patch_merge_mlp` into
the generated `multi_modal_projector.merge_linear_{1,2}` parameters; full
public checkpoint loading still requires running the 59-shard load gate.

Regen command:
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config -o veomni/models/transformers/minimax_m3_vl/generated --diff
"""

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.minimax_m3_vl.modeling_minimax_m3_vl",
    target_file="patched_modeling_minimax_m3_vl_gpu.py",
    description="MiniMax M3 VL with VeOmni parallel-plan hooks",
    transformers_version="5.12.0",
)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.ops.dispatch", names=["OpSlot"])
config.add_post_import_block(
    """
veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
"""
)


@config.add_helper
def collate_multimodal_metadata(batch, sp_pad):
    """Derive MiniMax ViT metadata on CPU inside the VeOmni collator.

    MiniMax's 3D RoPE only needs the packed `image_grid_thw` /
    `video_grid_thw` values as Python lists. Computing `.tolist()` here keeps
    the generated vision forward free of a host-device sync.
    """
    md = {}
    for modality, grid_key in (
        ("image", "image_grid_thw"),
        ("video", "video_grid_thw"),
    ):
        grid = batch.get(grid_key)
        if grid is not None:
            md[f"{modality}_grid_thw_list"] = grid.tolist()
    if md:
        batch["multimodal_metadata"] = md


@config.add_helper
def _grid_thw_to_list(grid_thw, grid_thw_list):
    if grid_thw_list is not None:
        return grid_thw_list
    return grid_thw.tolist()


# ================================================================
# Patch: MiniMaxM3VL3DRotaryEmbedding.forward
# 1. accept CPU-precomputed grid_thw_list from the collator so the vision
#    forward does not call `.tolist()` on a CUDA tensor in the hot path
# 2. keep the upstream tensor fallback for external callers that bypass
#    VeOmni's MainCollator
# ================================================================
@config.override_method(
    "MiniMaxM3VL3DRotaryEmbedding.forward",
    description="Consume collator-precomputed MiniMax vision grid lists when available",
)
def minimax_m3_vl_3d_rotary_embedding_forward_patched(self, grid_thw, device, dtype, grid_thw_list=None):
    # --- Patch.1 ---
    m = self.spatial_merge_size
    coords = []
    for t, h, w in _grid_thw_to_list(grid_thw, grid_thw_list):
        hi = torch.arange(h).unsqueeze(1).expand(-1, w)
        hi = hi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wi = torch.arange(w).unsqueeze(0).expand(h, -1)
        wi = wi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        ti = torch.arange(t).repeat_interleave(h * w)
        coords.append(torch.stack([ti, hi.repeat(t), wi.repeat(t)], dim=-1))
    # --- Patch.1 ---
    coords = torch.cat(coords).to(device=device, dtype=torch.float32)

    inv_freq = 1.0 / (
        self.theta ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32, device=device) / self.axis_dim)
    )
    freqs = torch.cat([coords[:, i : i + 1] * inv_freq for i in range(3)], dim=-1)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


# ================================================================
# Patch: MiniMaxM3VLVisionModel.forward
# 1. pop the per-modality vit_metadata sub-dict that Model.forward passes
# 2. route `grid_thw_list` into 3D RoPE to avoid CUDA `.tolist()` syncs
# ================================================================
@config.override_method(
    "MiniMaxM3VLVisionModel.forward",
    description="Consume MiniMax collator-precomputed vision grid metadata",
)
def minimax_m3_vl_vision_model_forward_patched(self, pixel_values, image_grid_thw, **kwargs):
    r"""
    image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
        The temporal, height and width of each image's feature grid, used to build the vision 3D RoPE.
    """
    # --- Patch.1 ---
    vit_metadata = kwargs.pop("vit_metadata", None) or {}
    # --- Patch.1 ---
    embeds = self.embeddings(pixel_values).to(self.pre_layrnorm.weight.dtype)
    # --- Patch.2 ---
    cos, sin = self.rotary_emb(
        image_grid_thw,
        device=embeds.device,
        dtype=embeds.dtype,
        grid_thw_list=vit_metadata.get("grid_thw_list"),
    )
    # --- Patch.2 ---
    hidden_states = self.pre_layrnorm(embeds).unsqueeze(0)
    for layer in self.layers:
        hidden_states = layer(hidden_states, attention_mask=None, position_embeddings=(cos, sin), **kwargs)
    return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=hidden_states[:, 0])


# ================================================================
# Patch: MiniMaxM3VLVisionModel.dummy_forward (new)
# 1. add dummy_forward so ranks without pixel_values can still run the
#    shared vision/projector parameters under FSDP
# 2. derive dummy shape from the vision config and pass host-built
#    vit_metadata so the dummy path is also sync-free
# ================================================================
@config.override_method(
    "MiniMaxM3VLVisionModel.dummy_forward",
    description="Provide MiniMax dummy vision forward for asymmetric FSDP batches",
)
def minimax_m3_vl_vision_dummy_forward_patched(self):
    # --- Patch.1 ---
    patch_size = self.config.patch_size
    temporal_patch_size = self.config.temporal_patch_size
    in_channels = getattr(self.config, "num_channels", getattr(self.config, "in_channels", 3))
    merge_size = self.config.spatial_merge_size
    t = 1
    h = 2 * merge_size
    w = 2 * merge_size
    num_patches = t * h * w
    pixel_row_size = in_channels * temporal_patch_size * patch_size * patch_size

    weight = self.embeddings.proj.weight
    pixel_values = torch.zeros((num_patches, pixel_row_size), dtype=weight.dtype, device=weight.device)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long, device=weight.device)
    vit_metadata = {"grid_thw_list": [[t, h, w]]}
    return self(pixel_values=pixel_values, image_grid_thw=grid_thw, vit_metadata=vit_metadata)
    # --- Patch.1 ---


# ================================================================
# Patch: MiniMaxM3VLModel.forward
# 1. consume MiniMax multimodal_metadata and pass per-modality grid lists
#    to the vision tower
# 2. pop VeOmni data-pipeline helper masks/ids before dispatching to the
#    language model
# 3. run a zero-valued dummy vision/projector path when FSDP has no visual
#    inputs on this rank, preventing asymmetric collectives from hanging
# ================================================================
@config.override_method(
    "MiniMaxM3VLModel.forward",
    description="Add MiniMax VLM metadata fast path and FSDP dummy vision branch",
)
def minimax_m3_vl_model_forward_patched(
    self,
    input_ids=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    **kwargs,
):
    r"""
    image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of each image's feature grid.
    video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of each video's feature grid.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.1 ---
    multimodal_metadata = kwargs.pop("multimodal_metadata", None) or {}
    image_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("image_grid_thw_list"),
        }
    }
    video_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("video_grid_thw_list"),
        }
    }
    # --- Patch.1 ---

    # --- Patch.2 ---
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    kwargs.pop("mm_token_type_ids", None)
    # --- Patch.2 ---

    image_features = None
    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw, **image_vit_kwargs
        ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)

    video_features = None
    if pixel_values_videos is not None:
        video_features = self.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **video_vit_kwargs
        ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)

    # --- Patch.3 ---
    if pixel_values is None and pixel_values_videos is None and get_parallel_state().fsdp_enabled:
        fake_vision = self.vision_tower.dummy_forward()
        fake_features = self.multi_modal_projector(fake_vision.last_hidden_state.squeeze(0))
        fake_features = fake_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_features.mean() * 0.0
    # --- Patch.3 ---

    if image_mask is None or video_mask is None:
        image_mask, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds, image_features=image_features, video_features=video_features
        )
    else:
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    if image_features is not None:
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
    if video_features is not None:
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    return MiniMaxM3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        image_hidden_states=image_features,
        video_hidden_states=video_features,
    )


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.forward",
    description="Unpack VeOmni causal LM loss tuple and route fused loss kernels when selected",
)
def minimax_m3_vl_sparse_for_conditional_generation_forward_patched(
    self,
    input_ids=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    logits_to_keep=0,
    **kwargs,
):
    r"""
    image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of each image's feature grid.
    video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of each video's feature grid.
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1 ---
    loss = None
    logits = None
    if labels is not None:
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, _ = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss_result = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.1 ---

    return MiniMaxM3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        video_hidden_states=outputs.video_hidden_states,
    )


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_parallel_plan",
    description="Register MiniMax M3 VL expert parallel plan for the multimodal training path",
)
def minimax_m3_vl_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_position_id_func",
    description="Use VeOmni's default 1-D packed-sequence position IDs for MiniMax M3 VL SFT data",
)
def minimax_m3_vl_get_position_id_func_patched(self):
    return None


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_metadata_collate_func",
    description="Expose MiniMax CPU-side vision grid metadata derivation to the VeOmni collator",
)
def minimax_m3_vl_get_metadata_collate_func_patched(self):
    return collate_multimodal_metadata  # noqa: F821 defined via add_helper


@config.override_method(
    "MiniMaxM3VLForCausalLM.get_parallel_plan",
    description="Register MiniMax M3 VL expert parallel plan for text-only reduced-layer smoke tests",
)
def minimax_m3_vl_text_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()
