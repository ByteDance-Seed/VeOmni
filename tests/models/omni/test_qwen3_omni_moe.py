import copy
import gc
import types

import torch
from transformers import AutoConfig, set_seed

from veomni import _safe_apply_patches
from veomni.optim import build_optimizer
from veomni.utils.device import empty_cache, get_device_type, synchronize

from ...tools.common_utils import print_device_mem_info
from ..utils import (
    ModelMode,
    build_base_model_optim,
    compare_multi_items,
    prepare_model_modes,
    print_all_values,
    set_environ_param,
)


def _release_device_memory():
    synchronize()
    gc.collect()
    empty_cache()


# ----------------------------------------------------------------
# Helpers: data preparation
# ----------------------------------------------------------------


def _get_feat_extract_output_lengths(input_lengths):
    """Compute the output length of the audio encoder's convolutional layers."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def _compute_vision_tokens(grid_thw, spatial_merge_size):
    """Compute the number of language-model tokens produced by the vision encoder.

    After patch embedding + spatial merging the token count is:
        T * H * W / spatial_merge_size^2
    """
    t, h, w = grid_thw
    return (t * h * w) // (spatial_merge_size**2)


def _prepare_qwen3_omni_moe_data(
    config,
    bsz=1,
    text_len=128,
    audio_mel_len=200,
    num_mel_bins=128,
    include_image=False,
    include_video=False,
    image_grid_thw=(1, 4, 4),
    video_grid_thw=(2, 4, 4),
    seed=42,
):
    """Prepare dummy data with fake audio + text (+ optional image/video) input for Qwen3-Omni-MoE.

    Constructs input_ids with special token placeholders for each modality,
    and matching features / masks for the model's forward pass.

    Data is prepared in VeOmni format:
    - input_features: 2D (num_mel_bins, total_mel_len)
    - audio_feature_lengths: 1D (bsz,)
    - pixel_values: 2D (total_patches, patch_flat_size)  -- if include_image
    - pixel_values_videos: 2D (total_patches, patch_flat_size)  -- if include_video
    - image_mask / video_mask / audio_mask: pre-computed boolean masks
    """
    torch.manual_seed(seed)

    thinker_config = config.thinker_config
    vision_config = thinker_config.vision_config
    audio_start_token_id = thinker_config.audio_start_token_id
    audio_end_token_id = thinker_config.audio_end_token_id
    audio_token_id = thinker_config.audio_token_id
    image_token_id = thinker_config.image_token_id
    video_token_id = thinker_config.video_token_id
    vision_start_token_id = thinker_config.vision_start_token_id
    vision_end_token_id = thinker_config.vision_end_token_id
    vocab_size = thinker_config.text_config.vocab_size

    spatial_merge_size = vision_config.spatial_merge_size
    patch_size = vision_config.patch_size
    temporal_patch_size = vision_config.temporal_patch_size
    in_channels = vision_config.in_channels
    patch_flat_size = in_channels * temporal_patch_size * patch_size * patch_size  # 3*2*16*16 = 1536

    num_audio_tokens = int(_get_feat_extract_output_lengths(torch.tensor(audio_mel_len)).item())

    # Vision token counts
    num_image_tokens = _compute_vision_tokens(image_grid_thw, spatial_merge_size) if include_image else 0
    num_video_tokens = _compute_vision_tokens(video_grid_thw, spatial_merge_size) if include_video else 0

    prefix_len = text_len // 2
    suffix_len = text_len - prefix_len

    input_ids_list = []
    for _ in range(bsz):
        parts = [torch.randint(0, vocab_size, (prefix_len,), dtype=torch.long)]

        # Audio segment: <audio_start> <audio_token>*N <audio_end>
        parts.append(torch.tensor([audio_start_token_id], dtype=torch.long))
        parts.append(torch.full((num_audio_tokens,), audio_token_id, dtype=torch.long))
        parts.append(torch.tensor([audio_end_token_id], dtype=torch.long))

        # Image segment: <vision_start> <image_token>*N <vision_end>
        if include_image:
            parts.append(torch.tensor([vision_start_token_id], dtype=torch.long))
            parts.append(torch.full((num_image_tokens,), image_token_id, dtype=torch.long))
            parts.append(torch.tensor([vision_end_token_id], dtype=torch.long))

        # Video segment: <vision_start> <video_token>*N <vision_end>
        if include_video:
            parts.append(torch.tensor([vision_start_token_id], dtype=torch.long))
            parts.append(torch.full((num_video_tokens,), video_token_id, dtype=torch.long))
            parts.append(torch.tensor([vision_end_token_id], dtype=torch.long))

        parts.append(torch.randint(0, vocab_size, (suffix_len,), dtype=torch.long))
        input_ids_list.append(torch.cat(parts))

    total_seq_len = input_ids_list[0].shape[0]
    input_ids = torch.stack(input_ids_list)

    attention_mask = torch.ones(bsz, total_seq_len, dtype=torch.long)
    labels = input_ids.clone()
    audio_mask = (input_ids == audio_token_id).bool()
    image_mask = (
        (input_ids == image_token_id).bool() if include_image else torch.zeros_like(input_ids, dtype=torch.bool)
    )
    video_mask = (
        (input_ids == video_token_id).bool() if include_video else torch.zeros_like(input_ids, dtype=torch.bool)
    )

    # VeOmni format: 2D concatenated mel features (num_mel_bins, bsz * mel_len)
    total_mel_length = audio_mel_len * bsz
    input_features = torch.randn(num_mel_bins, total_mel_length, dtype=torch.bfloat16)
    audio_feature_lengths = torch.tensor([audio_mel_len] * bsz, dtype=torch.long)

    # 3D position_ids for MROPE: (3, bsz, seq_len)
    position_ids = torch.zeros(3, bsz, total_seq_len, dtype=torch.long)
    for b in range(bsz):
        pos = torch.arange(total_seq_len, dtype=torch.long)
        position_ids[0, b] = pos
        position_ids[1, b] = pos
        position_ids[2, b] = pos

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_features": input_features,
        "audio_feature_lengths": audio_feature_lengths,
        "audio_mask": audio_mask,
        "image_mask": image_mask,
        "video_mask": video_mask,
        "position_ids": position_ids,
    }

    # Image pixel values: (total_patches_all_images, patch_flat_size)
    if include_image:
        t, h, w = image_grid_thw
        patches_per_image = t * h * w
        total_image_patches = patches_per_image * bsz
        result["pixel_values"] = torch.randn(total_image_patches, patch_flat_size, dtype=torch.bfloat16)
        result["image_grid_thw"] = torch.tensor([list(image_grid_thw)] * bsz, dtype=torch.long)

    # Video pixel values: (total_patches_all_videos, patch_flat_size)
    if include_video:
        t, h, w = video_grid_thw
        patches_per_video = t * h * w
        total_video_patches = patches_per_video * bsz
        result["pixel_values_videos"] = torch.randn(total_video_patches, patch_flat_size, dtype=torch.bfloat16)
        result["video_grid_thw"] = torch.tensor([list(video_grid_thw)] * bsz, dtype=torch.long)

    return result


# ----------------------------------------------------------------
# Helpers: HF model building with monkey-patched forward()
# ----------------------------------------------------------------


def _merge_joint_deepstack_embeds(image_mask, video_mask, deepstack_image_embeds, deepstack_video_embeds):
    """Merge image/video deepstack embeddings with 2D token masks.

    HF thinker's native image+video joint path currently builds joint deepstack
    tensors with 3D placeholder masks and can trigger shape mismatch.
    We merge using token-level 2D masks here, then pass the merged deepstack
    tensors into the language model explicitly.
    """
    visual_pos_masks = image_mask | video_mask  # (bsz, seq_len), bool
    n_visual_tokens = int(visual_pos_masks.sum().item())
    image_mask_joint = image_mask[visual_pos_masks]
    video_mask_joint = video_mask[visual_pos_masks]

    deepstack_visual_embeds = []
    for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
        embed_joint = img_embed.new_zeros((n_visual_tokens, img_embed.shape[-1]))
        embed_joint[image_mask_joint, :] = img_embed
        embed_joint[video_mask_joint, :] = vid_embed
        deepstack_visual_embeds.append(embed_joint)

    # HF model._deepstack_process expects a mask where mask[..., 0] becomes (bsz, seq_len).
    return deepstack_visual_embeds, visual_pos_masks.unsqueeze(-1)


def _hf_qwen3_omni_moe_thinker_model_forward(self, *args, **kwargs):
    """Inject manually prepared deepstack tensors into HF thinker's language model.

    This hook is test-only and only active for the image+video workaround path.
    """
    manual_deepstack = kwargs.pop("_manual_deepstack_visual_embeds", None)
    manual_visual_pos_masks = kwargs.pop("_manual_visual_pos_masks", None)
    if manual_deepstack is not None:
        kwargs["deepstack_visual_embeds"] = manual_deepstack
        kwargs["visual_pos_masks"] = manual_visual_pos_masks
    return self._hf_orig_forward(*args, **kwargs)


def _hf_qwen3_omni_moe_forward(self, **kwargs):
    """Test-only forward() monkey-patched onto HF Qwen3OmniMoeForConditionalGeneration.

    Bridges the data format gap between VeOmni and HF:
    - Strips VeOmni-specific kwargs (audio_mask, image_mask, video_mask)
    - Converts 2D input_features to 3D + feature_attention_mask for HF thinker
    - Pre-embeds vision features via the thinker's visual encoder and scatters
      them into inputs_embeds (avoids an HF bug in deepstack joint embedding
      when both image and video are present simultaneously)
    - Delegates to self.thinker(**kwargs)

    This mirrors VeOmni's forward() pattern at
    veomni/models/transformers/qwen3_omni_moe/modeling_qwen3_omni_moe.py.
    """
    image_mask = kwargs.get("image_mask")
    video_mask = kwargs.get("video_mask")

    # Keep audio_mask, image_mask, video_mask in fwd_inputs so they pass through
    # to the thinker's **kwargs (needed when VeOmni patches are applied globally).
    fwd_inputs = dict(kwargs)

    # Convert VeOmni 2D input_features (num_mel_bins, total_mel_len) to
    # HF 3D format (bsz, num_mel_bins, mel_len) + feature_attention_mask
    if "input_features" in fwd_inputs and "feature_attention_mask" not in fwd_inputs:
        audio_feature_lengths = fwd_inputs.pop("audio_feature_lengths")
        bsz = audio_feature_lengths.shape[0]
        mel_len = audio_feature_lengths[0].item()
        input_features_2d = fwd_inputs.pop("input_features")  # (num_mel_bins, total_mel_len)
        num_mel_bins = input_features_2d.shape[0]
        # Reshape: (num_mel_bins, bsz*mel_len) -> (bsz, num_mel_bins, mel_len)
        fwd_inputs["input_features"] = input_features_2d.T.reshape(bsz, mel_len, num_mel_bins).permute(0, 2, 1)
        fwd_inputs["feature_attention_mask"] = torch.ones(
            bsz, mel_len, dtype=torch.long, device=input_features_2d.device
        )

    # Pre-embed vision features and scatter into inputs_embeds.
    # This is ONLY needed when both image and video pixel_values are present in
    # the same call: HF's native joint deepstack path can fail with shape mismatch.
    # For single-modality vision, keep HF native path so deepstack is untouched.
    pixel_values = fwd_inputs.get("pixel_values", None)
    pixel_values_videos = fwd_inputs.get("pixel_values_videos", None)

    if pixel_values is not None and pixel_values_videos is not None:
        # Both modalities present: use workaround + manually merge deepstack embeddings
        # so HF still receives deepstack inputs even though pixel_values are pre-consumed.
        image_grid_thw = fwd_inputs.pop("image_grid_thw", None)
        video_grid_thw = fwd_inputs.pop("video_grid_thw", None)
        fwd_inputs.pop("pixel_values")
        fwd_inputs.pop("pixel_values_videos")

        input_ids = fwd_inputs["input_ids"]
        inputs_embeds = self.thinker.get_input_embeddings()(input_ids)
        image_mask = image_mask.bool().to(inputs_embeds.device)
        video_mask = video_mask.bool().to(inputs_embeds.device)

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.thinker.get_image_features(pixel_values, image_grid_thw)
            n_image_tokens = int(image_mask.sum().item())
            image_embeds = image_embeds[:n_image_tokens]
            deepstack_image_embeds = [embed[:n_image_tokens] for embed in deepstack_image_embeds]
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.thinker.get_video_features(pixel_values_videos, video_grid_thw)
            n_video_tokens = int(video_mask.sum().item())
            video_embeds = video_embeds[:n_video_tokens]
            deepstack_video_embeds = [embed[:n_video_tokens] for embed in deepstack_video_embeds]
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(mask, video_embeds)

        deepstack_visual_embeds, visual_pos_masks = _merge_joint_deepstack_embeds(
            image_mask=image_mask,
            video_mask=video_mask,
            deepstack_image_embeds=deepstack_image_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
        )

        fwd_inputs["inputs_embeds"] = inputs_embeds
        # Pass manual deepstack via kwargs; consumed by patched thinker.model.forward.
        fwd_inputs["_manual_deepstack_visual_embeds"] = deepstack_visual_embeds
        fwd_inputs["_manual_visual_pos_masks"] = visual_pos_masks.to(inputs_embeds.device)

    return self.thinker(**fwd_inputs)


def _build_qwen3_omni_moe_hf_model(config_path, device=None, torch_dtype="bfloat16", seed=42):
    """Build HF Qwen3-Omni-MoE model with monkey-patched forward().

    The HF Qwen3OmniMoeForConditionalGeneration is not registered in AutoModel mappings,
    so we import and instantiate it directly. Then we add a forward() method that delegates
    to self.thinker with format conversion, enabling the same model(**inputs) call path
    as VeOmni.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration as HFQwen3OmniMoeForConditionalGeneration,
    )

    if device is None:
        device = get_device_type()

    set_seed(seed)
    config = AutoConfig.from_pretrained(config_path)
    config._attn_implementation = "eager"
    with torch.device(device):
        model = HFQwen3OmniMoeForConditionalGeneration._from_config(config)
    model = model.to(getattr(torch, torch_dtype))

    # Patch HF thinker's language model forward so test-only manual deepstack
    # kwargs can override the default deepstack inputs in the joint workaround path.
    model.thinker.model._hf_orig_forward = model.thinker.model.forward
    model.thinker.model.forward = types.MethodType(_hf_qwen3_omni_moe_thinker_model_forward, model.thinker.model)

    # Monkey-patch forward() onto the HF model
    model.forward = types.MethodType(_hf_qwen3_omni_moe_forward, model)

    optimizer = build_optimizer(
        model,
        lr=0.0001,
        weight_decay=0,
        fused=True,
        optimizer_type="adamw",
        no_decay_modules=[],
        no_decay_params=[],
    )
    return model, optimizer


# ----------------------------------------------------------------
# Helpers: train step
# ----------------------------------------------------------------


def _qwen3_omni_moe_train_one_step(model, optimizer, inputs):
    """Train one step for Qwen3-Omni-MoE. Works for both HF (monkey-patched) and VeOmni."""
    device = get_device_type()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    optimizer.zero_grad()
    output = model(**inputs, use_cache=False)
    loss = output.loss
    if isinstance(loss, tuple):
        loss = loss[0]
    loss = loss.mean()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    optimizer.step()

    return loss, gnorm


# ----------------------------------------------------------------
# Test: HF vs VeOmni forward/backward comparison
# ----------------------------------------------------------------

_DEFAULT_RTOL = 1e-2
_DEFAULT_ATOL = 1e-2


def _run_fwd_bwd_comparison(test_name, dummy_data, config_path, config):
    """Run HF vs VeOmni forward/backward comparison for a given set of dummy data.

    Shared logic for audio-only, audio+image, and audio+video tests.
    """
    rtol, atol = 0.5, 0.02  # MoE tolerances

    # Prepare VeOmni model modes (MoE: base eager + fused)
    _, all_veomni_modes = prepare_model_modes(is_moe=True)

    print_device_mem_info(f"[Memory Info] start {test_name}:")

    # ---- HF baseline ----
    model_hf, optim_hf = _build_qwen3_omni_moe_hf_model(config_path)
    state_dict = copy.deepcopy(model_hf.state_dict())

    hf_mode = ModelMode(modeling_backend="hf", attn_implementation="eager", attn_case="padded_bsh")
    print(f"{'-' * 10} {config.model_type}_{hf_mode} {'-' * 10}")
    print_device_mem_info("[Memory Info] after building HF model:")

    loss, gnorm = _qwen3_omni_moe_train_one_step(model_hf, optim_hf, dummy_data)
    res = {hf_mode: {"loss": loss.item(), "gnorm": gnorm.item()}}

    del model_hf, optim_hf, loss, gnorm
    _release_device_memory()
    print_device_mem_info("[Memory Info] after HF train_one_step:")

    # ---- VeOmni modes ----
    for idx, mode in enumerate(all_veomni_modes):
        print(f"{'-' * 10} {config.model_type}_{mode} {'-' * 10}")

        set_environ_param(mode)
        _safe_apply_patches()

        model_veomni, optim_veomni = build_base_model_optim(
            config_path,
            attn_implementation=mode.attn_implementation,
            moe_implementation=mode.moe_implementation,
        )
        print_device_mem_info(f"[Memory Info] after building VeOmni model {idx}:")

        model_veomni.load_state_dict(state_dict)

        loss, gnorm = _qwen3_omni_moe_train_one_step(model_veomni, optim_veomni, dummy_data)
        res[mode] = {"loss": loss.item(), "gnorm": gnorm.item()}

        del model_veomni, optim_veomni, loss, gnorm
        _release_device_memory()
        print_device_mem_info(f"[Memory Info] after VeOmni model {idx} train_one_step:")

    assert len(res) == 1 + len(all_veomni_modes)
    print_all_values(res, "loss", config.model_type)
    print_all_values(res, "gnorm", config.model_type)
    compare_multi_items(res, rtol=rtol, atol=atol)

    _release_device_memory()
    print_device_mem_info(f"[Memory Info] after {test_name}:")


def test_qwen3_omni_moe_fwd_bwd():
    """Test Qwen3-Omni-MoE: HF vs VeOmni with fake audio + text input.

    Compares loss and gradient norm between HuggingFace baseline (with monkey-patched
    forward) and VeOmni implementation across multiple attention/MoE/liger modes,
    using dummy audio mel-spectrogram + text input (no image/video).
    """
    config_path = "./tests/toy_config/qwen3_omni_moe_toy/config.json"
    config = AutoConfig.from_pretrained(config_path)
    num_mel_bins = config.thinker_config.audio_config.num_mel_bins

    dummy_data = _prepare_qwen3_omni_moe_data(
        config,
        bsz=2,
        text_len=128,
        audio_mel_len=200,
        num_mel_bins=num_mel_bins,
        seed=42,
    )

    _run_fwd_bwd_comparison("test_qwen3_omni_moe_fwd_bwd", dummy_data, config_path, config)


def test_qwen3_omni_moe_fwd_bwd_image():
    """Test Qwen3-Omni-MoE: HF vs VeOmni with fake audio + image + text input.

    Same as test_qwen3_omni_moe_fwd_bwd but with an additional fake image per sample.
    Image grid_thw=(1, 4, 4) produces 1*4*4/4=4 vision tokens per image.
    """
    config_path = "./tests/toy_config/qwen3_omni_moe_toy/config.json"
    config = AutoConfig.from_pretrained(config_path)
    num_mel_bins = config.thinker_config.audio_config.num_mel_bins

    dummy_data = _prepare_qwen3_omni_moe_data(
        config,
        bsz=2,
        text_len=128,
        audio_mel_len=200,
        num_mel_bins=num_mel_bins,
        include_image=True,
        image_grid_thw=(1, 4, 4),
        seed=42,
    )

    _run_fwd_bwd_comparison("test_qwen3_omni_moe_fwd_bwd_image", dummy_data, config_path, config)


def test_qwen3_omni_moe_fwd_bwd_video():
    """Test Qwen3-Omni-MoE: HF vs VeOmni with fake audio + video + text input.

    Same as test_qwen3_omni_moe_fwd_bwd but with an additional fake video per sample.
    Video grid_thw=(2, 4, 4) produces 2*4*4/4=8 vision tokens per video.
    """
    config_path = "./tests/toy_config/qwen3_omni_moe_toy/config.json"
    config = AutoConfig.from_pretrained(config_path)
    num_mel_bins = config.thinker_config.audio_config.num_mel_bins

    dummy_data = _prepare_qwen3_omni_moe_data(
        config,
        bsz=2,
        text_len=128,
        audio_mel_len=200,
        num_mel_bins=num_mel_bins,
        include_video=True,
        video_grid_thw=(2, 4, 4),
        seed=42,
    )

    _run_fwd_bwd_comparison("test_qwen3_omni_moe_fwd_bwd_video", dummy_data, config_path, config)


def test_qwen3_omni_moe_fwd_bwd_all_modalities():
    """Test Qwen3-Omni-MoE: HF vs VeOmni with fake audio + image + video + text input.

    Exercises all modalities simultaneously: audio mel-spectrogram, static image,
    and video frames alongside text tokens.
    """
    config_path = "./tests/toy_config/qwen3_omni_moe_toy/config.json"
    config = AutoConfig.from_pretrained(config_path)
    num_mel_bins = config.thinker_config.audio_config.num_mel_bins

    dummy_data = _prepare_qwen3_omni_moe_data(
        config,
        bsz=2,
        text_len=128,
        audio_mel_len=200,
        num_mel_bins=num_mel_bins,
        include_image=True,
        include_video=True,
        image_grid_thw=(1, 4, 4),
        video_grid_thw=(2, 4, 4),
        seed=42,
    )

    _run_fwd_bwd_comparison("test_qwen3_omni_moe_fwd_bwd_all_modalities", dummy_data, config_path, config)
