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

"""Reference flow-matching training-path helpers.

The caller owns a single :class:`torch.Generator` (in practice, a per-DP-replica
generator on the model — see the HunyuanImage 3 modeling for the canonical
implementation) and hands it in per call.

:func:`normalize_flow_config` is the single source of truth for which flow
recipes are supported. Model configs call it at build, so an unsupported recipe
fails before any weight load; :func:`prepare_reference_flow_batch` re-validates
whatever mapping it is handed, so the two callers cannot drift apart.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch


DEFAULT_REFERENCE_FLOW_CONFIG = {
    "num_train_timesteps": 1000,
    "timestep_sampling": "uniform",
    "training_shift": 1.0,
    "prediction_type": "velocity",
    "loss_weighting": "uniform",
    # Base seed for the flow generator. Part of the recipe, not of the launcher:
    # it rides on the model config so it is written into the checkpoint's
    # config.json and a resumed / re-run job reproduces the same noise without
    # depending on the CLI. Mirrors ``dit_trainer._build_condition_model``, which
    # likewise passes the noise seed through the model config. Note this is
    # deliberately independent of ``args.train.seed`` (which seeds the global RNG
    # and the dataloader) -- change ``flow.seed`` to change the noise stream.
    "seed": 0,
}


def derive_flow_seed(flow_seed: int, data_replica_rank: int) -> int:
    """Derive the flow generator seed for one DP replica.

    Adding the two (``flow_seed + dp_rank``) collides across runs: seed 41 on
    replica 1 draws exactly what seed 42 on replica 0 draws. Hashing the pair
    keeps neighbouring run seeds independent. blake2b rather than the builtin
    ``hash()`` because the latter is salted per process for bytes, which would
    change the stream on every restart.
    """
    material = f"veomni.flow-matching:{int(flow_seed)}:{int(data_replica_rank)}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")


def prepare_reference_flow_batch(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    *,
    vae_config: Mapping[str, object],
    flow_config: Mapping[str, object] | None,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Sample posterior, timestep, and diffusion noise from ``generator``.

    **Draw order**: posterior noise, then timestep sigma, then diffusion noise.
    All three draws come from the single shared ``generator``. Every SP/EP rank
    inside a DP replica must call this with the same generator state, or the
    ranks desync and the summed gradient stops being the flow-matching gradient.

    ``generator`` is a state machine, so this must be called exactly once per
    micro-batch: keep the call outside any activation-checkpointing boundary, or
    the backward recompute draws fresh noise and silently corrupts the gradient.
    """
    if posterior_mean.ndim != 4 or posterior_logvar.shape != posterior_mean.shape:
        raise ValueError("Cached posterior mean and logvar must have identical [B, C, H, W] shapes.")
    if not posterior_mean.is_floating_point() or not posterior_logvar.is_floating_point():
        raise TypeError("Cached posterior mean and logvar must be floating-point tensors.")
    if posterior_mean.device != posterior_logvar.device:
        raise ValueError("Cached posterior mean and logvar must be on the same device.")
    if not isinstance(vae_config, Mapping):
        raise TypeError("vae_config must be a mapping.")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator instance.")
    gen_device = generator.device
    post_device = posterior_mean.device
    # ``torch.Generator(device="cuda")`` may leave the index unset; treat that as
    # matching the posterior's concrete CUDA index (the actual draw happens on
    # ``posterior_mean.device`` below regardless).
    same_device = gen_device == post_device or (
        gen_device.type == post_device.type
        and gen_device.type == "cuda"
        and (gen_device.index is None or post_device.index is None or gen_device.index == post_device.index)
    )
    if not same_device:
        raise ValueError(f"generator device {gen_device} must match posterior device {post_device}.")
    num_train_timesteps = normalize_flow_config(flow_config)["num_train_timesteps"]

    # --- Draw 1: posterior noise ---
    posterior_noise = torch.randn(
        posterior_mean.shape,
        generator=generator,
        device=posterior_mean.device,
        dtype=posterior_mean.dtype,
    )
    latents = posterior_mean + torch.exp(0.5 * posterior_logvar.clamp(-30.0, 20.0)) * posterior_noise
    shift_factor = vae_config.get("shift_factor")
    scaling_factor = vae_config.get("scaling_factor")
    if shift_factor is not None:
        latents = latents - float(shift_factor)
    if scaling_factor is None or float(scaling_factor) == 0.0:
        raise ValueError("vae.scaling_factor must be non-zero for the reference flow path.")
    latents = latents * float(scaling_factor)

    # --- Draw 2: timestep sigma ---
    sigmas = torch.rand(
        (posterior_mean.shape[0],),
        generator=generator,
        device=posterior_mean.device,
        dtype=torch.float32,
    )
    # --- Draw 3: diffusion noise ---
    diffusion_noise = torch.randn(
        posterior_mean.shape,
        generator=generator,
        device=posterior_mean.device,
        dtype=posterior_mean.dtype,
    )
    broadcast_sigmas = sigmas.to(dtype=latents.dtype).reshape(-1, 1, 1, 1)
    noised_latents = (1.0 - broadcast_sigmas) * latents + broadcast_sigmas * diffusion_noise
    flow_target = diffusion_noise - latents
    timesteps = sigmas * num_train_timesteps
    return {
        "latents": latents,
        "noised_latents": noised_latents,
        "flow_target": flow_target,
        "sigmas": sigmas,
        "timesteps": timesteps,
    }


def flow_matching_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute channel-mean, token-mean velocity MSE in FP32."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Flow prediction and target must have identical [B, C, H, W] shapes.")
    return (prediction.float() - target.float()).square().mean(dim=1).mean()


def normalize_flow_config(flow_config: Mapping[str, object] | None) -> dict:
    """Validate a flow recipe and return the full, defaulted config dict.

    Unlisted keys take their :data:`DEFAULT_REFERENCE_FLOW_CONFIG` value, so a
    model config lists only what deviates from the reference recipe.
    """
    values = dict(DEFAULT_REFERENCE_FLOW_CONFIG)
    if flow_config is not None:
        if not isinstance(flow_config, Mapping):
            raise TypeError("flow_config must be a mapping.")
        unknown = sorted(set(flow_config).difference(values))
        if unknown:
            raise ValueError(f"Unsupported reference flow config fields: {unknown}.")
        values.update(flow_config)

    num_train_timesteps = values["num_train_timesteps"]
    if isinstance(num_train_timesteps, bool) or not isinstance(num_train_timesteps, int) or num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be a positive integer.")
    training_shift = values["training_shift"]
    if isinstance(training_shift, bool) or not isinstance(training_shift, (int, float)) or training_shift <= 0:
        raise ValueError("training_shift must be positive.")
    if values["timestep_sampling"] != "uniform":
        raise ValueError("The reference flow path supports only uniform timestep sampling.")
    if float(training_shift) != 1.0:
        raise ValueError("Uniform reference timestep sampling requires training_shift=1.0.")
    if values["prediction_type"] != "velocity":
        raise ValueError("The reference flow path supports only velocity prediction.")
    if values["loss_weighting"] != "uniform":
        raise ValueError("The reference flow path supports only uniform flow loss weighting.")
    seed = values["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("flow seed must be a non-negative integer.")
    values["training_shift"] = float(training_shift)
    return values


__all__ = [
    "DEFAULT_REFERENCE_FLOW_CONFIG",
    "derive_flow_seed",
    "flow_matching_loss",
    "normalize_flow_config",
    "prepare_reference_flow_batch",
]
