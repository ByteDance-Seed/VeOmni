"""Public BAGEL reference data helper surface."""

from .common import first_output_of_type, float_pair, int_pair, normalize_reference_kwargs
from .conversation import conversation_item_image, conversation_item_text, conversation_to_interleaved_reference_inputs
from .generation import inferencer_generation_kwargs
from .train import (
    BagelReferenceTrainLoss,
    build_reference_train_batch_from_stimulus,
    encode_reference_vae_latents,
    latent_position_ids,
    official_train_forward_batch,
    reduce_reference_train_losses,
    reference_train_batch_from_inputs,
    reference_train_forward_context,
    shifted_timesteps,
    train_loss_options,
    train_options_from_inputs,
)


__all__ = [
    "BagelReferenceTrainLoss",
    "build_reference_train_batch_from_stimulus",
    "conversation_item_image",
    "conversation_item_text",
    "conversation_to_interleaved_reference_inputs",
    "encode_reference_vae_latents",
    "first_output_of_type",
    "float_pair",
    "inferencer_generation_kwargs",
    "int_pair",
    "latent_position_ids",
    "normalize_reference_kwargs",
    "official_train_forward_batch",
    "reduce_reference_train_losses",
    "reference_train_batch_from_inputs",
    "reference_train_forward_context",
    "shifted_timesteps",
    "train_loss_options",
    "train_options_from_inputs",
]
