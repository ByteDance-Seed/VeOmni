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

"""Opt-in DiT execution contracts; validity, packing and attention remain model-owned."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import torch
from transformers.utils import ModelOutput


if TYPE_CHECKING:
    from ...trainer.dit_trainer import VeOmniDiTArguments


class DiffusionSample(TypedDict):
    """One prepared sample, using native pytrees for FSDP input traversal.

    Noise and timesteps are sampled before this boundary. Model-specific inputs,
    targets and metadata retain their single-sample shapes and semantics.
    """

    model_inputs: dict[str, Any]
    targets: dict[str, torch.Tensor]
    metadata: dict[str, Any]


@dataclass
class DiffusionBatchOutput(ModelOutput):
    """Ordered per-sample predictions and optional, already modality-weighted losses.

    Each loss has shape [B], with its within-sample reduction owned by the model.
    External objectives may consume predictions without supplying sample_losses.
    """

    sample_predictions: list[dict[str, torch.Tensor]] | None = None
    sample_losses: dict[str, torch.Tensor] | None = None

    def mean_losses(self, *, batch_size: int) -> dict[str, torch.Tensor]:
        """Preserve equal sample weighting without detaching the backward graph."""
        if (
            batch_size < 1
            or not isinstance(self.sample_predictions, list)
            or len(self.sample_predictions) != batch_size
            or any(not isinstance(prediction, dict) for prediction in self.sample_predictions)
        ):
            raise ValueError("sample_predictions must contain one dictionary per input sample.")
        if not isinstance(self.sample_losses, dict) or not self.sample_losses:
            raise ValueError("sample_losses must be a nonempty dictionary for DiT training.")
        for name, loss in self.sample_losses.items():
            if not isinstance(loss, torch.Tensor) or loss.shape != (batch_size,):
                raise ValueError(f"sample_losses[{name!r}] must be a tensor of shape [{batch_size}].")
        return {name: loss.mean() for name, loss in self.sample_losses.items()}


def validate_diffusion_samples(samples: list[DiffusionSample], *, batch_size: int) -> None:
    """Reject incomplete batches; the condition model validates its column lengths."""
    if not isinstance(samples, list) or batch_size < 1 or len(samples) != batch_size:
        raise ValueError(f"sample_inputs must contain exactly {batch_size} prepared samples.")
    for sample in samples:
        for key in ("model_inputs", "targets", "metadata"):
            if not isinstance(sample, dict) or not isinstance(sample.get(key), dict):
                raise ValueError(f"Each sample_inputs entry must contain a {key!r} dictionary.")


def validate_diffusion_remove_padding_config(args: "VeOmniDiTArguments") -> None:
    """Initial supported envelope, checked before distributed setup or weight loading.

    This API does not itself opt any built-in model into remove-padding. Model
    hooks additionally validate backend availability and model-specific settings.
    """
    if not args.model.use_remove_padding:
        return
    train, acc = args.train, args.model.accelerator
    if train.training_task != "offline_training":
        raise ValueError("model.use_remove_padding currently requires offline_training.")
    if train.dyn_bsz:
        raise ValueError("model.use_remove_padding requires train.dyn_bsz=false; dynamic batching is independent.")
    if train.bsz_warmup_ratio != 0:
        raise ValueError("model.use_remove_padding does not support batch-size warmup.")
    if train.micro_batch_size < 1:
        raise ValueError("model.use_remove_padding requires a positive micro_batch_size.")
    if train.global_batch_size is not None and train.global_batch_size % (train.micro_batch_size * acc.dp_size):
        raise ValueError("global_batch_size must be a multiple of micro_batch_size * dp_size.")
    if not args.data.dataloader.drop_last:
        raise ValueError("model.use_remove_padding requires data.dataloader.drop_last=true for equal microbatches.")
    for name in ("ulysses_size", "cp_size", "tp_size", "pp_size"):
        if getattr(acc, name) != 1:
            raise ValueError(f"model.use_remove_padding requires {name}=1.")
    if any(size != 1 for size in acc.extra_parallel_sizes):
        raise ValueError("model.use_remove_padding does not yet support extra_parallel_sizes > 1.")
    if acc.fsdp_config.fsdp_mode != "fsdp2":
        raise ValueError("model.use_remove_padding currently requires fsdp2.")
    if acc.torch_compile.enable:
        raise ValueError("model.use_remove_padding does not yet support torch.compile.")
    if acc.fsdp_config.offload or acc.offload_config.enable_activation or acc.offload_config.enable_async_activation:
        raise ValueError("model.use_remove_padding does not yet support parameter/activation offload.")
    if args.model.lora_config:
        raise ValueError("model.use_remove_padding does not yet support LoRA.")


def _validate_model_support(model) -> None:
    if getattr(model, "supports_remove_padding", False) is not True:
        raise ValueError(f"{getattr(model, '__name__', type(model).__name__)} does not support use_remove_padding.")
    if not callable(getattr(model, "configure_remove_padding", None)):
        raise ValueError("A remove-padding model must implement configure_remove_padding.")


def validate_diffusion_remove_padding_support(model_class, condition_model_class) -> None:
    """Check explicit opt-in on registry-resolved classes before constructing either."""
    _validate_model_support(model_class)
    if getattr(condition_model_class, "supports_sample_inputs", False) is not True or not callable(
        getattr(condition_model_class, "prepare_samples", None)
    ):
        raise ValueError("The condition model must declare supports_sample_inputs=True and implement prepare_samples.")


def configure_diffusion_remove_padding(model, *, enabled: bool, attn_implementation: str) -> None:
    """Configure a newly built model before wrapping/sharding; disabled is a no-op.

    The model hook must validate the requested backend, retain all parameter names,
    and configure only this instance. It must not patch module-global forwards.
    """
    if not enabled:
        return
    _validate_model_support(model)
    model.configure_remove_padding(attn_implementation=attn_implementation)
