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

"""HunyuanImage 3 data-path CPU tests: resolution policy, processor, online/cache
VAE parity, and MainCollator wiring.

Covers the CPU pieces the 16xH20 E2E training smoke does not otherwise pin
byte-exactly:
  - resolution buckets are VAE- and patch-aligned;
  - image processor normalizes pixel values to [-1, 1] and reports the grid;
  - online-VAE forward is byte-identical to feeding the cached posterior
    (proves ``vae.encode -> posterior`` is a pure preprocessing step);
  - the ``vae_encoder='frozen'`` component-policy guard fires when someone
    tries the online path without a VAE encoder;
  - the runtime collator's packed metadata is bit-equal to what
    ``compile_single_gen_t2i_packed`` produces from the same plans.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from veomni.data.data_collator import MainCollator
from veomni.models.loader import get_model_class, get_model_config
from veomni.models.transformers.hunyuan_image_3.data_transform import process_sample_hunyuan_image_3
from veomni.models.transformers.hunyuan_image_3.processing_hunyuan_image_3 import (
    HunyuanImage3ImageProcessor,
    HunyuanImage3ImageProcessorConfig,
)
from veomni.models.transformers.hunyuan_image_3.resolution_policy import build_resolution_policy
from veomni.models.transformers.hunyuan_image_3.sequence_compiler import (
    build_single_gen_t2i_plan,
    compile_single_gen_t2i_packed,
)


_TOY_CONFIG_PATH = Path(__file__).parents[1] / "toy_config" / "hunyuan_image_3_toy"


def _toy_config():
    return get_model_config(str(_TOY_CONFIG_PATH))


def _build_toy_model_with_vae(device="cpu", dtype=torch.float32):
    """Toy config with ``vae_encoder='frozen'`` so ``model.vae`` is built."""
    config = get_model_config(str(_TOY_CONFIG_PATH))
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    policy = dict(config.component_policy)
    policy["vae_encoder"] = "frozen"
    config.component_policy = policy
    model = get_model_class(config)(config).to(device=device, dtype=dtype)
    return config, model


def _cached_sample(config, *, grid=(2, 2), prompt="a b c"):
    channels = config.vae["latent_channels"]
    return {
        "id": "sample",
        "prompt": prompt,
        "latent_mean": torch.randn(channels, grid[0], grid[1]),
        "latent_logvar": torch.zeros(channels, grid[0], grid[1]),
    }


def _transform_kwargs(config):
    return dict(
        resolution_policy=build_resolution_policy(),
        latent_source="posterior_cache",
        latent_channels=config.vae["latent_channels"],
        im_start_id=config.im_start_id,
        im_end_id=config.im_end_id,
        image_token_id=config.image_token_id,
    )


def test_resolution_policy_buckets_are_vae_and_patch_aligned():
    """Every bucket lands on the VAE downsample grid AND the patch grid."""
    policy = build_resolution_policy()  # all anchors, patch_size=1, factor 16
    assert policy.buckets, "policy must expand at least one bucket"
    for bucket in policy.buckets:
        assert bucket.height % 16 == 0 and bucket.width % 16 == 0
        assert bucket.latent_height == bucket.height // 16
        assert bucket.grid_hw == (bucket.latent_height, bucket.latent_width)


def test_processor_normalizes_to_unit_range_and_reports_grid():
    processor = HunyuanImage3ImageProcessor(HunyuanImage3ImageProcessorConfig())
    image = Image.new("RGB", (1200, 800), color=(255, 0, 0))
    processed = processor.preprocess_target_image(image, base_size=512)

    bucket = processed.bucket
    assert processed.pixel_values.shape == (3, bucket.height, bucket.width)
    assert processed.pixel_values.min() >= -1.0 - 1e-6 and processed.pixel_values.max() <= 1.0 + 1e-6
    # Pure red -> R channel at +1, G/B at -1 after Normalize([0.5], [0.5]).
    torch.testing.assert_close(processed.pixel_values[0].max(), torch.tensor(1.0), rtol=0, atol=1e-6)
    torch.testing.assert_close(processed.pixel_values[1].max(), torch.tensor(-1.0), rtol=0, atol=1e-6)
    assert processed.grid_hw == (bucket.height // 16, bucket.width // 16)


def test_online_pixel_values_match_cached_posterior():
    """Online (pixel_values -> vae.encode -> posterior) must equal fed cached posterior."""
    config, model = _build_toy_model_with_vae()
    from veomni.models.transformers.hunyuan_image_3.sequence_compiler import (
        build_single_gen_t2i_plan as _plan,
    )
    from veomni.models.transformers.hunyuan_image_3.sequence_compiler import (
        compile_single_gen_t2i_plans as _compile,
    )

    plan = _plan(sample_id="sample", text_token_count=2, grid_hw=(2, 2))
    metadata = _compile([plan])
    input_ids = torch.arange(plan["sequence_length"], dtype=torch.long).unsqueeze(0)
    flow_step_context = {"train_seed": 7, "data_replica_rank": 0, "optimizer_step": 1, "micro_step": 0}

    pixel_values = torch.rand(1, config.vae["in_channels"], 2, 2)
    distribution = model.vae.encode(pixel_values)
    cached_mean = distribution.mean.squeeze(2)
    cached_logvar = distribution.logvar.squeeze(2)

    online = model(
        input_ids=input_ids,
        component_inputs={"pixel_values": pixel_values},
        hy3_sequence_metadata=metadata,
        flow_step_context=flow_step_context,
        use_cache=False,
    )
    cached = model(
        input_ids=input_ids,
        component_inputs={"latent_posterior": {"mean": cached_mean, "logvar": cached_logvar}},
        hy3_sequence_metadata=metadata,
        flow_step_context=flow_step_context,
        use_cache=False,
    )
    torch.testing.assert_close(online.latents, cached.latents, rtol=0, atol=0)
    torch.testing.assert_close(online.loss["image_decoder_loss"], cached.loss["image_decoder_loss"], rtol=0, atol=0)


def test_online_pixel_values_require_frozen_vae_encoder():
    """The online path must fail loud when ``vae_encoder`` is absent from the policy."""
    config = get_model_config(str(_TOY_CONFIG_PATH))
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = get_model_class(config)(config)
    assert not hasattr(model, "vae")
    with pytest.raises(RuntimeError, match="vae_encoder='frozen'"):
        model._encode_pixel_values_to_posterior(torch.rand(1, 3, 2, 2))


def test_collator_finalizes_packed_metadata_matches_compiler():
    """``MainCollator`` + model metadata hook must match the sequence compiler bit-exactly.

    This is the wire between the CPU data path (transform + collator) and the
    GPU-side packed varlen forward: if these two ever drift, cross-sample
    attention or per-sample flow RNG identity silently breaks.
    """
    config = _toy_config()
    torch.manual_seed(0)
    model = get_model_class(config)(config)

    sample = _cached_sample(config)
    (feature,) = process_sample_hunyuan_image_3(sample, **_transform_kwargs(config))

    collator = MainCollator(
        data_collate_info=model.get_extra_collate_infos(),
        metadata_collate_func=model.get_metadata_collate_func(),
    )
    batch = collator([feature])

    metadata = batch["hy3_sequence_metadata"]
    assert metadata["layout"] == "packed_varlen"
    assert metadata["num_samples"] == 1

    reference = compile_single_gen_t2i_packed(
        [build_single_gen_t2i_plan(sample_id="s0", text_token_count=3, grid_hw=(2, 2))]
    )
    for key in ("position_ids", "timestep_sample_index", "image_output_mask", "image_payload_indices"):
        torch.testing.assert_close(metadata[key], reference[key], rtol=0, atol=0)
    for key in ("cu_seqlens_q_prefix", "cu_seqlens_k_full", "cu_seqlens_q_image_suffix"):
        torch.testing.assert_close(metadata[key], reference[key], rtol=0, atol=0)
    assert metadata["padded_sequence_length"] == batch["input_ids"].size(-1)

    # Staging keys removed; posterior reassembled into component_inputs as a
    # length-1 list under the P1a smart-stack contract.
    posterior = batch["component_inputs"]["latent_posterior"]
    assert isinstance(posterior["mean"], list) and len(posterior["mean"]) == 1
    assert posterior["mean"][0].shape == (1, config.vae["latent_channels"], 2, 2)
    for staging_key in ("hy3_text_token_count", "hy3_grid_hw", "hy3_latent_mean", "hy3_latent_logvar"):
        assert staging_key not in batch
