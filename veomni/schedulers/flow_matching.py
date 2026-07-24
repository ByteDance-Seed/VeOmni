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

"""Deterministic reference flow-matching training-path helpers.

The reference path is what a from-scratch DiT-style trainer needs to reproduce
byte-for-byte on resume:

* :func:`derive_seed_v1` -- frozen BLAKE2b seed derivation from
  ``(train_seed, data_replica_rank, optimizer_step, micro_step, stream_name)``
  so each named RNG stream (``posterior``, ``diffusion_noise``, ``timestep``,
  ...) draws independently while the sequence stays reproducible.
* :func:`prepare_reference_flow_batch` -- given cached posterior mean+logvar,
  samples latents, uniform sigmas, diffusion noise, and returns the flow
  target + noised latents + timesteps.
* :func:`flow_matching_loss` -- channel-mean, token-mean velocity MSE in FP32.

The default config is the standard uniform-timestep / velocity-prediction /
uniform-weighting reference; :func:`_normalize_flow_config` is the single
source of truth for accepted keys and rejects unsupported combinations
loudly. Model-specific ``flow`` dicts (from YAML) are opaque at the trainer
layer and validated here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import torch


DEFAULT_REFERENCE_FLOW_CONFIG = {
    "num_train_timesteps": 1000,
    "timestep_sampling": "uniform",
    "training_shift": 1.0,
    "prediction_type": "velocity",
    "loss_weighting": "uniform",
}

# The reference algorithm draws three independent streams per training step.
# Callers may pass any subset of stream names to ``derive_seed_v1``; these are
# the ones this module's own ``prepare_reference_flow_batch`` uses internally.
REFERENCE_RNG_STREAMS = ("posterior", "diffusion_noise", "timestep")


@dataclass(frozen=True)
class _ReferenceFlowConfig:
    num_train_timesteps: int
    training_shift: float


def derive_seed_v1(
    train_seed: int,
    data_replica_rank: int,
    optimizer_step: int,
    micro_step: int,
    stream_name: str,
) -> int:
    """Derive a stable, non-negative 63-bit seed from logical step identity.

    Frozen BLAKE2b of the JSON-serialised ``(train_seed, data_replica_rank,
    optimizer_step, micro_step, stream_name)`` tuple. Same identity → same
    seed on every rank and across DCP resume; different ``stream_name`` →
    independent seed for that stream.
    """
    integer_fields = (train_seed, data_replica_rank, optimizer_step, micro_step)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
        raise TypeError("Flow RNG identity fields must be integers.")
    if data_replica_rank < 0 or optimizer_step < 0 or micro_step < 0:
        raise ValueError("Flow RNG rank and step fields must be non-negative.")
    if not isinstance(stream_name, str) or not stream_name:
        raise TypeError("stream_name must be a non-empty string.")
    payload = json.dumps(
        [train_seed, data_replica_rank, optimizer_step, micro_step, stream_name],
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


def prepare_reference_flow_batch(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    *,
    vae_config: Mapping[str, object],
    flow_config: Mapping[str, object] | None,
    flow_step_context: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Sample cached posteriors, timesteps, and diffusion noise reproducibly."""
    if posterior_mean.ndim != 4 or posterior_logvar.shape != posterior_mean.shape:
        raise ValueError("Cached posterior mean and logvar must have identical [B, C, H, W] shapes.")
    if not posterior_mean.is_floating_point() or not posterior_logvar.is_floating_point():
        raise TypeError("Cached posterior mean and logvar must be floating-point tensors.")
    if posterior_mean.device != posterior_logvar.device:
        raise ValueError("Cached posterior mean and logvar must be on the same device.")
    if not isinstance(vae_config, Mapping):
        raise TypeError("vae_config must be a mapping.")
    normalized_flow = _normalize_flow_config(flow_config)
    identity = _normalize_step_context(flow_step_context)

    generators = {
        stream_name: _make_generator(
            posterior_mean.device,
            derive_seed_v1(*identity, stream_name),
        )
        for stream_name in REFERENCE_RNG_STREAMS
    }
    posterior_noise = torch.randn(
        posterior_mean.shape,
        generator=generators["posterior"],
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

    uniform_sigma = torch.rand(
        (posterior_mean.shape[0],),
        generator=generators["timestep"],
        device=posterior_mean.device,
        dtype=torch.float32,
    )
    sigmas = uniform_sigma
    diffusion_noise = torch.randn(
        posterior_mean.shape,
        generator=generators["diffusion_noise"],
        device=posterior_mean.device,
        dtype=posterior_mean.dtype,
    )
    broadcast_sigmas = sigmas.to(dtype=latents.dtype).reshape(-1, 1, 1, 1)
    noised_latents = (1.0 - broadcast_sigmas) * latents + broadcast_sigmas * diffusion_noise
    flow_target = diffusion_noise - latents
    timesteps = sigmas * normalized_flow.num_train_timesteps
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


def _normalize_flow_config(flow_config: Mapping[str, object] | None) -> _ReferenceFlowConfig:
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
    return _ReferenceFlowConfig(
        num_train_timesteps=num_train_timesteps,
        training_shift=float(training_shift),
    )


def _normalize_step_context(flow_step_context: Mapping[str, object]) -> tuple[int, int, int, int]:
    if not isinstance(flow_step_context, Mapping):
        raise TypeError("flow_step_context must be a mapping.")
    required = ("train_seed", "data_replica_rank", "optimizer_step", "micro_step")
    missing = [name for name in required if name not in flow_step_context]
    unknown = sorted(set(flow_step_context).difference(required))
    if missing or unknown:
        raise ValueError(f"Invalid flow_step_context fields; missing={missing}, unknown={unknown}.")
    identity = tuple(flow_step_context[name] for name in required)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in identity):
        raise TypeError("flow_step_context values must be integers.")
    train_seed, data_replica_rank, optimizer_step, micro_step = identity
    if data_replica_rank < 0 or optimizer_step < 0 or micro_step < 0:
        raise ValueError("Flow RNG rank and step fields must be non-negative.")
    return train_seed, data_replica_rank, optimizer_step, micro_step


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


__all__ = [
    "DEFAULT_REFERENCE_FLOW_CONFIG",
    "REFERENCE_RNG_STREAMS",
    "derive_seed_v1",
    "flow_matching_loss",
    "prepare_reference_flow_batch",
]
