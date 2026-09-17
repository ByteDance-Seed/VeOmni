"""Pack prepared H3 samples without changing their local geometry or timestep tables."""

import torch

from ...packing import validate_diffusion_samples


def pack_samples(samples):
    validate_diffusion_samples(samples, batch_size=len(samples))
    first = samples[0]
    shape = (first["metadata"]["video_latent_shape"], first["metadata"]["audio_latent_shape"])
    checkpointing = first["model_inputs"]["use_gradient_checkpointing"]
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
    for sample in samples:
        inp, meta = sample["model_inputs"], sample["metadata"]
        if inp["unique_timesteps"].dtype != torch.float32 or inp["img_position_ids"].dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("H3 packing must retain timestep/position precision; use cast_forward_inputs=false.")
        if (meta["video_latent_shape"], meta["audio_latent_shape"]) != shape:
            raise ValueError("H3 remove-padding currently requires fixed target video/audio geometry.")
        if inp["use_gradient_checkpointing"] != checkpointing:
            raise ValueError("H3 remove-padding requires one gradient-checkpointing setting per microbatch.")
        if not inp["skip_mask_out_condition"]:
            raise ValueError("H3 packing requires explicit condition-row output cropping.")
        used = int(inp["packed_seq_params"]["cu_seqlens_q"][1])
        text_len = inp["text_pos_info"]["position_ids"].numel()
        if not 0 < used <= inp["x"].shape[1] or not 0 < text_len <= inp["prompt_embeds"].shape[0]:
            raise ValueError("Invalid H3 valid-row or text lengths.")
        for key in ("x", "audio_x", "img_position_ids"):
            chunks[key].append(inp[key][:, :used])
        chunks["token_tags"].append(inp["token_tags"][:used])
        chunks["prompt_embeds"].append(inp["prompt_embeds"][:text_len])
        chunks["unique_timesteps"].append(inp["unique_timesteps"])
        chunks["inverse_indices"].append(inp["inverse_indices"][:used] + time_offset)
        for key in positions:
            positions[key].append(inp[key]["position_ids"] + cu[-1])
        row_counts.append(
            (
                inp["img_pos_for_infer_output_info"]["position_ids"].numel(),
                inp["audio_pos_info"]["position_ids"].numel(),
            )
        )
        time_offset += inp["unique_timesteps"].numel()
        cu.append(cu[-1] + used)
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
        use_gradient_checkpointing=checkpointing,
        packed_seq_params={
            "cu_seqlens_q": torch.tensor(cu, dtype=torch.int32, device=device),
            "max_seqlen_q": max(b - a for a, b in zip(cu, cu[1:])),
        },
        refiner_packed_seq_params={
            "cu_seqlens_q": torch.tensor(refiner_cu, dtype=torch.int32, device=device),
            "max_seqlen_q": max(b - a for a, b in zip(refiner_cu, refiner_cu[1:])),
        },
    )
    return result, row_counts
