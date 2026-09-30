"""Prepare a negative H3 branch without resampling targets, noise or references."""

import torch
from safetensors.torch import load_file

from .packed_sequence import host_cu_seqlens


def packed_seq_params(pk, device):
    """Main/refiner segment bounds from the prepared sample's host metadata."""
    cu_host = host_cu_seqlens(pk)
    text_len = int(pk["text_len"])
    return {
        "packed_seq_params": {
            "cu_seqlens_q": pk["cu_seqlens"].to(device),
            "cu_seqlens_host": cu_host,
            "max_seqlen_q": cu_host[1],
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.arange(2, dtype=torch.int32, device=device) * text_len,
            "cu_seqlens_host": (0, text_len),
            "max_seqlen_q": text_len,
        },
    }


def load_empty_embedding(path):
    """Read an actual encoder output once; never fabricate an all-zero condition."""
    tensors = load_file(path, device="cpu")
    prompt = tensors.get("prompt_embeds")
    tags = tensors.get("text_token_tags")
    if prompt is None or tags is None:
        raise ValueError("Empty embedding requires prompt_embeds and text_token_tags.")
    _validate_prompt(prompt)
    if tags.shape != (prompt.shape[0],) or tags.dtype not in (torch.int32, torch.int64) or not torch.all(tags == 1):
        raise ValueError("shared_empty requires text-only token tags matching the embedding length.")
    return prompt


def _validate_prompt(prompt):
    if isinstance(prompt, torch.Tensor) and prompt.device.type != "cpu":
        raise ValueError("Validate CFG embeddings on CPU before transferring the batch to the accelerator.")
    if (
        not isinstance(prompt, torch.Tensor)
        or prompt.ndim != 2
        or min(prompt.shape) < 1
        or not prompt.is_floating_point()
        or not torch.isfinite(prompt).all()
    ):
        raise ValueError("unconditional_prompt_embeds must be a nonempty finite floating-point [L, D] tensor.")


def _validate_layout(pk):
    """CFG remapping supports upstream compact [text | cond | audio | video] only."""
    if any(isinstance(value, torch.Tensor) and value.device.type != "cpu" for value in pk.values()):
        raise ValueError("Validate CFG packed layouts on CPU before transferring the batch to the accelerator.")
    length, text_len, cond = int(pk["seq_len"]), int(pk["text_len"]), int(pk["cond_rows"])
    audio_len, image_len = pk["audio_pos"].numel(), pk["img_pos"].numel()
    if text_len < 1 or cond < 0 or cond > image_len or length != text_len + audio_len + image_len:
        raise ValueError("CFG requires a compact unconditional packed layout with nonempty text.")
    device = pk["text_pos"].device
    expected = {
        "text_pos": torch.arange(text_len, device=device),
        "audio_pos": torch.arange(text_len + cond, text_len + cond + audio_len, device=device),
        "img_pos": torch.cat(
            (
                torch.arange(text_len, text_len + cond, device=device),
                torch.arange(text_len + cond + audio_len, length, device=device),
            )
        ),
    }
    if any(not torch.equal(pk[key], value) for key, value in expected.items()):
        raise ValueError("CFG packed positions must follow [text | cond | audio | video].")
    if pk["img_position_ids"].shape != (1, length, 3) or pk["token_tags"].shape != (length,):
        raise ValueError("CFG packed coordinates and token tags must match seq_len.")
    tags = pk["token_tags"]
    if (
        tags.dtype not in (torch.int32, torch.int64)
        or not torch.all((tags[:text_len] == 0) | (tags[:text_len] == 1))
        or not torch.all(tags[pk["img_pos"]] == 0)
        or not torch.all(tags[pk["audio_pos"]] == 2)
    ):
        raise ValueError("CFG packed token tags must match text/visual/audio modalities.")
    if not torch.isfinite(pk["img_position_ids"]).all():
        raise ValueError("CFG packed coordinates must be finite.")
    if not torch.equal(pk["cu_seqlens"], torch.tensor([0, length], device=device, dtype=torch.int32)):
        raise ValueError("CFG requires one compact segment per sample.")


def replace_text_layout(pk, text_len, text_token_tags=None):
    """Replace a validated compact prefix, preserving visual-latent references.

    The caller validates the input/output layouts on CPU before device transfer.
    Paired caches supply the encoder's tags; shared-empty uses text-only tags.
    """
    old_len = pk["text_len"]
    delta = text_len - old_len
    result = dict(pk, text_len=text_len, seq_len=pk["seq_len"] + delta)
    device = pk["text_pos"].device
    result["text_pos"] = torch.arange(text_len, device=device)
    for key in ("img_pos", "audio_pos"):
        result[key] = pk[key] + delta
    grid = pk["img_position_ids"].new_zeros((1, result["seq_len"], 3))
    grid[0, :text_len, 0] = torch.arange(text_len, device=grid.device)
    grid[:, text_len:] = pk["img_position_ids"][:, old_len:]
    grid[:, text_len:, 0] += delta
    result["img_position_ids"] = grid
    tags = pk["token_tags"].new_ones(text_len) if text_token_tags is None else text_token_tags
    if tags.shape != (text_len,) or tags.dtype not in (torch.int32, torch.int64):
        raise ValueError("Negative text_token_tags must be integer [text_len].")
    result["token_tags"] = torch.cat((tags, pk["token_tags"][old_len:]))
    result["cu_seqlens"] = pk["cu_seqlens"].new_tensor([0, result["seq_len"]])
    result["update_mask"] = None
    return result


def validate_unconditional(positive_pk, negative_pk, prompt, positive_prompt):
    """Validate cached CFG inputs on the host, before the trainer's device transfer."""
    _validate_prompt(prompt)
    for pk in (positive_pk, negative_pk):
        _validate_layout(pk)
    geometry = ("latent_t", "latent_h_patched", "latent_w_patched", "audio_t", "audio_channel", "cond_rows")
    if any(positive_pk[key] != negative_pk[key] for key in geometry):
        raise ValueError("Conditional and unconditional target/reference geometry must match.")
    if prompt.shape != (negative_pk["text_len"], positive_prompt.shape[-1]):
        raise ValueError("unconditional embedding length/width does not match the packed layout/model.")
    for key in ("img_pos", "audio_pos"):
        p = positive_pk["img_position_ids"][0, positive_pk[key]].clone()
        n = negative_pk["img_position_ids"][0, negative_pk[key]].clone()
        p[:, 0] -= positive_pk["text_len"]
        n[:, 0] -= negative_pk["text_len"]
        if p.shape != n.shape or not torch.allclose(p, n.to(p), rtol=0, atol=1e-6):
            raise ValueError("Conditional and unconditional reference/target positional geometry differs.")


def prepare_unconditional(sample, positive_pk, negative_pk, prompt):
    """Remap host-validated inputs without device-value checks or a second unique operation."""
    device = sample["x"].device
    result = {
        key: sample[key]
        for key in ("video_latent_shape", "audio_latent_shape", "cond_rows", "skip_mask_out_condition")
    }
    result.update(use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False, update_mask=None)
    length = negative_pk["seq_len"]
    # Text tokens use the video timestep. Reuse the positive branch's unique
    # values and remap indices, including the near-clean reference timesteps.
    inverse = sample["inverse_indices"][0].expand(length).clone()
    for tensor_key, index_key in (("x", "img_pos"), ("audio_x", "audio_pos")):
        source, dest = positive_pk[index_key].to(device), negative_pk[index_key].to(device)
        value = sample[tensor_key].new_zeros((1, length, sample[tensor_key].shape[-1]))
        value[0, dest] = sample[tensor_key][0, source]
        result[tensor_key] = value
        inverse[dest] = sample["inverse_indices"][source]
    result["unique_timesteps"] = sample["unique_timesteps"]
    result["inverse_indices"] = inverse
    for key in ("img_position_ids", "token_tags"):
        result[key] = negative_pk[key].to(device)
    for name, key in (
        ("img_pos_info", "img_pos"),
        ("audio_pos_info", "audio_pos"),
        ("text_pos_info", "text_pos"),
        ("img_pos_for_infer_output_info", "img_pos"),
    ):
        result[name] = {"position_ids": negative_pk[key].to(device)}
    result["prompt_embeds"] = prompt.to(device=device, dtype=sample["prompt_embeds"].dtype)
    result.update(packed_seq_params(negative_pk, device))
    return result
