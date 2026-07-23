import torch
from torch import Tensor


def _output_scale(statistic: Tensor, output: Tensor) -> Tensor:
    scale = statistic[..., :1]
    if scale.ndim != output.ndim:
        raise ValueError(
            f"Softmax statistic rank ({scale.ndim}) must match attention output rank ({output.ndim})."
        )
    if scale.shape[:-1] != output.shape[:-1]:
        raise ValueError(
            f"Softmax statistic prefix {scale.shape[:-1]} must match attention output prefix {output.shape[:-1]}."
        )
    return scale


def merge_attention_blocks(
    previous_output: Tensor,
    previous_softmax_max: Tensor,
    previous_softmax_sum: Tensor,
    current_output: Tensor,
    current_softmax_max: Tensor,
    current_softmax_sum: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Merge independently normalized attention blocks with online softmax."""
    if previous_output.shape != current_output.shape:
        raise ValueError(
            f"Attention output shapes must match, got {previous_output.shape} and {current_output.shape}."
        )
    if previous_softmax_max.shape != current_softmax_max.shape:
        raise ValueError(
            "Softmax max shapes must match, got "
            f"{previous_softmax_max.shape} and {current_softmax_max.shape}."
        )
    if previous_softmax_sum.shape != current_softmax_sum.shape:
        raise ValueError(
            "Softmax sum shapes must match, got "
            f"{previous_softmax_sum.shape} and {current_softmax_sum.shape}."
        )

    accumulator_dtype = torch.promote_types(previous_output.dtype, current_output.dtype)
    if accumulator_dtype in (torch.float16, torch.bfloat16):
        accumulator_dtype = torch.float32
    previous_max = previous_softmax_max.to(accumulator_dtype)
    current_max = current_softmax_max.to(accumulator_dtype)
    merged_max = torch.maximum(previous_max, current_max)

    previous_finite = torch.isfinite(previous_max)
    current_finite = torch.isfinite(current_max)
    previous_scale = torch.where(
        previous_finite,
        torch.exp(previous_max - torch.where(previous_finite, merged_max, previous_max)),
        torch.zeros_like(previous_max),
    )
    current_scale = torch.where(
        current_finite,
        torch.exp(current_max - torch.where(current_finite, merged_max, current_max)),
        torch.zeros_like(current_max),
    )

    previous_sum_scaled = previous_softmax_sum.to(accumulator_dtype) * previous_scale
    current_sum_scaled = current_softmax_sum.to(accumulator_dtype) * current_scale
    merged_sum = previous_sum_scaled + current_sum_scaled
    nonzero_sum = merged_sum > 0
    previous_weight = torch.where(nonzero_sum, previous_sum_scaled / merged_sum, torch.zeros_like(merged_sum))
    current_weight = torch.where(nonzero_sum, current_sum_scaled / merged_sum, torch.zeros_like(merged_sum))

    previous_output_weight = _output_scale(previous_weight, previous_output)
    current_output_weight = _output_scale(current_weight, current_output)
    output = previous_output.to(accumulator_dtype) * previous_output_weight
    output = output + current_output.to(accumulator_dtype) * current_output_weight

    return (
        output.to(previous_output.dtype),
        merged_max.to(previous_softmax_max.dtype),
        merged_sum.to(previous_softmax_sum.dtype),
    )
