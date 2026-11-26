"""Concatenate complete H3 samples while preserving local geometry and timesteps."""

import torch


def _validate_fp32_metadata(timesteps: torch.Tensor, position_ids: torch.Tensor) -> None:
    """Reject timestep and position inputs that have already lost FP32 precision."""
    if any(tensor.is_floating_point() and tensor.element_size() < 4 for tensor in (timesteps, position_ids)):
        raise ValueError(
            "H3 floating-point timesteps and positions need FP32 or higher. Set cast_forward_inputs=False."
        )


def pack_samples(samples):
    if not samples:
        raise ValueError("H3 requires a nonempty microbatch.")
    chunks = {
        key: []
        for key in (
            "x",
            "audio_x",
            "img_position_ids",
            "token_tags",
            "prompt_embeds",
            "unique_timesteps",
            "inverse_indices",
        )
    }
    positions = {
        key: [] for key in ("img_pos_info", "audio_pos_info", "text_pos_info", "img_pos_for_infer_output_info")
    }
    cu, refiner_cu, row_counts = [0], [0], []
    time_offset = 0
    for inp in samples:
        # Check each sample before concatenation can promote a rounded input.
        _validate_fp32_metadata(inp["unique_timesteps"], inp["img_position_ids"])
        if not inp["skip_mask_out_condition"]:
            raise ValueError("H3 packing requires explicit condition-row output cropping.")
        length = inp["x"].shape[1]
        text_len = inp["prompt_embeds"].shape[0]
        # Host check: a single valid segment spans the whole sample (a legacy tail makes it shorter).
        if length <= 0 or text_len <= 0 or int(inp["packed_seq_params"]["max_seqlen_q"]) != length:
            raise ValueError("H3 samples must contain one valid segment without tail padding; rebuild cached layouts.")
        if inp["text_pos_info"]["position_ids"].numel() != text_len:
            raise ValueError("H3 text rows must match the sample's text positions.")
        for key in chunks:
            chunks[key].append(inp[key] + time_offset if key == "inverse_indices" else inp[key])
        for key in positions:
            positions[key].append(inp[key]["position_ids"] + cu[-1])
        row_counts.append(
            (
                inp["img_pos_for_infer_output_info"]["position_ids"].numel(),
                inp["audio_pos_info"]["position_ids"].numel(),
            )
        )
        time_offset += inp["unique_timesteps"].numel()
        cu.append(cu[-1] + length)
        refiner_cu.append(refiner_cu[-1] + text_len)
    result = {
        key: torch.cat(values, dim=1 if key in ("x", "audio_x", "img_position_ids") else 0)
        for key, values in chunks.items()
    }
    result.update({key: {"position_ids": torch.cat(values)} for key, values in positions.items()})
    device = result["x"].device
    result.update(
        update_mask=None,
        skip_mask_out_condition=True,
        packed_seq_params={
            "cu_seqlens_q": torch.tensor(cu, dtype=torch.int32, device=device),
            "cu_seqlens_host": tuple(cu),
            "max_seqlen_q": max(b - a for a, b in zip(cu, cu[1:])),
        },
        refiner_packed_seq_params={
            "cu_seqlens_q": torch.tensor(refiner_cu, dtype=torch.int32, device=device),
            "cu_seqlens_host": tuple(refiner_cu),
            "max_seqlen_q": max(b - a for a, b in zip(refiner_cu, refiner_cu[1:])),
        },
    )
    return result, row_counts
