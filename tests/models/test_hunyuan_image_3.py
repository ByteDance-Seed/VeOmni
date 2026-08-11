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

"""HunyuanImage 3 single_gen_t2i_v1 tests.

Ordered cheapest-first: CPU invariants, then the GPU packed-varlen fast path,
then the spawned end-to-end training smoke.

The recurring device under test is the two-call varlen GCA decomposition, checked
against a *dense oracle*: the same packed forward with ``dense_reference_attention``
set on the compiled metadata, so attention uses the dense block-diagonal edge mask
that the decomposition encodes. Everything else -- projection, scatters, 2D RoPE,
image head, flow loss -- is bit-identical between the two arms, so a diff isolates
the attention topology.

Everything from ``test_packed_fast_path_matches_dense_oracle`` down requires
flash-attention varlen and runs in BF16 on an SM80+ CUDA GPU.
"""

import io
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from veomni.models.loader import get_model_class, get_model_config
from veomni.models.transformers.hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config
from veomni.models.transformers.hunyuan_image_3.generalized_causal_attention import build_packed_gca_dense_mask
from veomni.models.transformers.hunyuan_image_3.processing_hunyuan_image_3 import HunyuanImage3ImageProcessor
from veomni.models.transformers.hunyuan_image_3.rope_2d import build_2d_rope
from veomni.models.transformers.hunyuan_image_3.sequence_layout import (
    T2ILayout,
    UnsupportedSequenceLayout,
    compile_single_gen_t2i_packed,
)
from veomni.schedulers.flow_matching_loss import (
    DEFAULT_REFERENCE_FLOW_CONFIG,
    derive_flow_seed,
    prepare_reference_flow_batch,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


_REPO_ROOT = Path(__file__).parents[2]
_TOY_CONFIG_PATH = _REPO_ROOT / "tests" / "toy_config" / "hunyuan_image_3_toy"
_TOY_TRAIN_CONFIG = _REPO_ROOT / "configs" / "multimodal" / "hunyuan_image_3" / "hunyuan_image_3_toy.yaml"
_TRAIN_SCRIPT = _REPO_ROOT / "tests" / "train_scripts" / "train_hunyuan_image_3_test.py"
# Source PIL image size for the e2e parquet; the image_processor resizes to
# ``mm_configs.resolution`` ([2, 2] in the toy) before the VAE encode.
_TOY_IMAGE_HW = (32, 32)
_BF16_TOLERANCE = {"rtol": 2e-2, "atol": 2e-2}
# Dense-oracle equivalence runs much tighter than the generic BF16 slack: the two
# arms share every op except the attention call, so the only legitimate spread is
# accumulation order. Measured on H20 over 4 seeds x {1,2,3}-sample packs the worst
# clean deviation is 2.9e-3, while breaking the decomposition (suffix made causal,
# or suffix K/V scoped to the suffix) moves it to 1.1e-2 - 1.7e-2. 6e-3 sits between
# the two, so these tests actually fail when the topology is wrong -- at 2e-2 they
# did not. Re-measure before loosening.
_ORACLE_TOLERANCE = {"rtol": 6e-3, "atol": 6e-3}

_requires_flash_gpu = pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or get_gpu_compute_capability() < 80,
    reason="Hunyuan Image 3 varlen GCA fast path requires an NVIDIA CUDA SM80+ GPU.",
)


def _build_model(*, device, dtype, attn_implementation="flash_attention_2", overrides=None):
    config = get_model_config(str(_TOY_CONFIG_PATH))
    config._attn_implementation = attn_implementation
    config._experts_implementation = "eager"
    for name, value in (overrides or {}).items():
        setattr(config, name, value)
    torch.manual_seed(0)
    model = get_model_class(config)(config).to(device=device, dtype=dtype)
    return config, model


def _cached_posterior(num_samples, config, *, device, dtype, grid=(2, 2), seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (num_samples, config.vae["latent_channels"], grid[0], grid[1])
    mean = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    logvar = torch.zeros(shape, device=device, dtype=dtype)
    return mean, logvar


def _to_device(metadata, device):
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in metadata.items()}


def _dense_oracle(metadata):
    """Same compiled metadata, dense edge mask instead of the two-call split."""
    return {**metadata, "dense_reference_attention": True}


def _pin_flow_generator(model, *, device, seed=0):
    """Pin ``model._flow_generator`` so paired forwards draw identical noise.

    The model lazily instantiates its flow generator on the first forward from
    ``derive_flow_seed(config.flow["seed"], dp_rank)`` and threads it through
    training. Tests that call ``model(...)`` twice expecting bit-identical
    outputs need to reset the stream between calls; that is what this helper
    does. Keep the seed constant across a paired call to compare like-for-like.
    """
    model._flow_generator = torch.Generator(device=device).manual_seed(seed)


# ----------------------------- CPU invariants --------------------------------


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


def _online_sample(prompt="a b c", size=(256, 256), color=(255, 0, 0)):
    """One raw T2I sample for the online-VAE data transform."""
    return {"id": "sample", "prompt": prompt, "image": Image.new("RGB", size, color=color)}


def _fake_processor_bundle(image_processor, *, tokenizer=None):
    """Stand-in for ``HunyuanImage3Processor`` in unit tests.

    ``HunyuanImage3Processor.from_pretrained`` also loads an ``AutoTokenizer``,
    unavailable in the toy config dir. The transform reads only
    ``.image_processor`` and ``.tokenizer`` off the bundle.
    """
    return SimpleNamespace(tokenizer=tokenizer, image_processor=image_processor)


def _transform_kwargs(config, image_processor):
    # Special-token ids are read off ``processor.image_processor.config`` inside
    # the transform, so only the generation-task kwargs are supplied here.
    return dict(
        processor=_fake_processor_bundle(image_processor),
        resolution=(2 * int(config.vae_downsample_factor[0]) * int(config.patch_size),) * 2,
        target_image_key="image",
        prompt_dropout_prob=0.0,
        random_flip=False,
    )


@pytest.mark.parametrize(
    ("grid_hw", "expected_y", "expected_x"),
    [
        ((2, 3), [8, 8, 8, 9, 9, 9], [7, 8, 9, 7, 8, 9]),
        ((3, 2), [7, 7, 8, 8, 9, 9], [8, 9, 8, 9, 8, 9]),
    ],
)
def test_compiler_freezes_mixed_parity_coordinates_and_dense_gca(grid_hw, expected_y, expected_x):
    """Mixed-parity grids centre on half-integer offsets that must truncate the
    same way across every consumer (compiler + RoPE + upstream). The dense oracle
    mask also asserts image-payload tokens attend everywhere while text is causal.
    """
    layout = T2ILayout(text_len=2, grid_h=grid_hw[0], grid_w=grid_hw[1])
    metadata = compile_single_gen_t2i_packed([layout])

    payload_start, payload_stop = 6, 12
    assert (layout.payload_start, layout.payload_stop) == (payload_start, payload_stop)
    assert metadata["position_ids"][0, 0, payload_start:payload_stop].tolist() == expected_y
    assert metadata["position_ids"][0, 1, payload_start:payload_stop].tolist() == expected_x
    assert metadata["position_ids"][0, :, -1].tolist() == [12, 12]
    assert metadata["timestep_positions"].tolist() == [5]
    assert metadata["image_payload_indices"][0].tolist() == list(range(payload_start, payload_stop))

    attention_mask = build_packed_gca_dense_mask(metadata, device="cpu")[0]
    assert not attention_mask[5, 6]  # text is causal (cannot see image)
    assert attention_mask[6].all()  # image payload is full-attention
    assert attention_mask[-1].all()


def test_reference_2d_rope_preserves_official_frequency_interleave():
    """Lock the per-head frequency interleave the fused-QKV forward assumes."""
    position_ids = torch.tensor([[[0, 2], [0, 3]]], dtype=torch.long)
    cos, sin = build_2d_rope(position_ids, head_dim=8)
    expected_angles = torch.tensor([2.0, 0.3, 0.02, 0.003] * 2)

    torch.testing.assert_close(cos[0, 1], expected_angles.cos())
    torch.testing.assert_close(sin[0, 1], expected_angles.sin())


def test_reference_flow_generator_is_deterministic_and_seed_scoped():
    """Model-owned flow generator: same seed -> same noise; different seed -> different noise.

    The forward's ``_ensure_flow_generator`` lazily creates one generator per rank
    from ``derive_flow_seed(config.flow["seed"], dp_rank)`` and draws posterior /
    diffusion noise from it in a fixed order. This test pins the equivalent
    invariants at the scheduler level: a fresh generator with seed S produces
    byte-identical draws twice, and seed S' yields a different noised_latents.
    """
    posterior_mean = torch.zeros(2, 4, 2, 2)
    posterior_logvar = torch.zeros_like(posterior_mean)
    vae_config = {"scaling_factor": 0.5, "shift_factor": None}

    def _draw(seed):
        generator = torch.Generator(device=posterior_mean.device).manual_seed(seed)
        return prepare_reference_flow_batch(
            posterior_mean,
            posterior_logvar,
            vae_config=vae_config,
            flow_config=None,
            generator=generator,
        )

    first = _draw(1234)
    second = _draw(1234)
    changed = _draw(4321)

    for name in first:
        torch.testing.assert_close(first[name], second[name])
    assert not torch.equal(first["noised_latents"], changed["noised_latents"])


def test_derive_flow_seed_separates_neighbouring_run_seeds():
    """``flow_seed + dp_rank`` would alias; the hash must not.

    Plain addition makes (seed=41, dp=1) draw exactly what (seed=42, dp=0) draws,
    so two runs whose seeds differ by one share a noise stream. The derivation
    must also be stable across processes -- Python's builtin ``hash()`` is salted
    for bytes and would change the stream on every restart.
    """
    assert derive_flow_seed(41, 1) != derive_flow_seed(42, 0)
    assert derive_flow_seed(0, 0) != derive_flow_seed(0, 1)
    # Same inputs -> same seed, and inside torch's accepted range.
    assert derive_flow_seed(7, 3) == derive_flow_seed(7, 3)
    assert 0 <= derive_flow_seed(7, 3) < 2**64
    torch.Generator().manual_seed(derive_flow_seed(7, 3))

    # Stable across interpreters (guards against a salted-hash regression).
    # ``import veomni`` logs to stdout, so read the value off the last line.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from veomni.schedulers.flow_matching_loss import derive_flow_seed; print(derive_flow_seed(7, 3))",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=True,
    )
    assert int(completed.stdout.strip().splitlines()[-1]) == derive_flow_seed(7, 3)


def _decode_flow_slots(payload):
    return torch.load(io.BytesIO(payload["flow_generator_by_dp_rank"]), weights_only=True)


def _encode_flow_slots(slots):
    buffer = io.BytesIO()
    torch.save(slots, buffer)
    return buffer.getvalue()


def test_flow_generator_extra_state_round_trip_resumes_the_stream():
    """``get_extra_state``/``set_extra_state`` must resume the stream, not restart it.

    Under frequent preemption a restart-from-seed would replay the same sigma
    prefix after every crash, over-sampling that slice of the flow trajectory.

    Single-process, so this pins the round trip only -- it cannot see the DCP
    dedup that makes the payload rank-invariant. The multi-rank contract (each
    replica gets *its own* slot back) needs a gloo two-rank case; what is covered
    here is the shape of the payload and the two fallback paths.
    """
    model = _build_toy_model_with_vae()[1]

    model._flow_generator = torch.Generator().manual_seed(derive_flow_seed(11, 0))
    torch.randn(8, generator=model._flow_generator)  # advance past the initial state

    payload = model.get_extra_state()
    # One opaque blob under one stable key. The key set must not depend on how many
    # replicas there are or whether a generator exists, because DCP flattens this
    # mapping into the checkpoint key space and mismatched shapes fail the load.
    assert list(payload) == ["flow_generator_by_dp_rank"]
    assert isinstance(payload["flow_generator_by_dp_rank"], bytes)
    # Uninitialized process group / dp_size == 1: a single slot, indexed by dp_rank 0.
    assert len(_decode_flow_slots(payload)) == 1
    expected = torch.randn(8, generator=model._flow_generator)

    resumed = _build_toy_model_with_vae()[1]
    resumed.set_extra_state(payload)
    torch.testing.assert_close(torch.randn(8, generator=resumed._flow_generator), expected)

    # A checkpoint saved before the first forward restores to "uninitialized" so
    # the next forward re-seeds lazily.
    fresh = _build_toy_model_with_vae()[1]
    fresh_payload = fresh.get_extra_state()
    assert list(fresh_payload) == ["flow_generator_by_dp_rank"]  # same key as the live case
    assert _decode_flow_slots(fresh_payload) == [None]
    resumed.set_extra_state(fresh_payload)
    assert resumed._flow_generator is None

    # Legacy single-payload checkpoints cannot be attributed to a DP replica.
    # Adopting one would hand every replica the same stream -- the bug this
    # format replaced -- so the generator is dropped and re-seeded lazily.
    legacy = _build_toy_model_with_vae()[1]
    legacy._flow_generator = torch.Generator().manual_seed(7)
    legacy.set_extra_state({"flow_generator": {"device_type": "cpu", "state": torch.Generator().get_state()}})
    assert legacy._flow_generator is None

    # dp_size mismatch on resume: no meaningful slot, same lazy re-seed.
    mismatched = _build_toy_model_with_vae()[1]
    mismatched._flow_generator = torch.Generator().manual_seed(7)
    mismatched.set_extra_state({"flow_generator_by_dp_rank": _encode_flow_slots([None, None])})
    assert mismatched._flow_generator is None


def test_config_materializes_the_flow_recipe_and_rejects_unsupported_ones():
    """The flow objective is static per run, so it lives on the model config
    rather than riding on every micro-batch. The config must fill the reference
    defaults, survive a to_dict round trip (the DCP manifest reads it back), and
    reject an unsupported recipe at build -- before an 83B weight load, not on
    the first forward."""
    assert HunyuanImage3Config().flow == DEFAULT_REFERENCE_FLOW_CONFIG

    overridden = HunyuanImage3Config(flow={"num_train_timesteps": 500})
    assert overridden.flow == {**DEFAULT_REFERENCE_FLOW_CONFIG, "num_train_timesteps": 500}
    assert HunyuanImage3Config(**overridden.to_dict()).flow == overridden.flow

    with pytest.raises(ValueError, match="Unsupported reference flow config fields"):
        HunyuanImage3Config(flow={"not_a_flow_field": 1})
    with pytest.raises(ValueError, match="only velocity prediction"):
        HunyuanImage3Config(flow={"prediction_type": "epsilon"})

    # The noise seed is part of the recipe, so it survives the config round trip
    # and lands in the checkpoint's config.json rather than living in a runtime
    # singleton -- a re-run reproduces the noise without the original CLI.
    seeded = HunyuanImage3Config(flow={"seed": 1234})
    assert seeded.flow["seed"] == 1234
    assert HunyuanImage3Config(**seeded.to_dict()).flow["seed"] == 1234
    with pytest.raises(ValueError, match="flow seed must be a non-negative integer"):
        HunyuanImage3Config(flow={"seed": -1})


def test_online_pixel_values_match_cached_posterior():
    """Online (pixel_values -> vae.encode -> posterior) must equal a fed cached posterior.

    The model accepts a ``latent_posterior`` component-input dict as a
    direct-injection entry point, so the two paths must stay interchangeable.
    """
    config, model = _build_toy_model_with_vae()
    layout = T2ILayout(text_len=2, grid_h=2, grid_w=2)
    metadata = compile_single_gen_t2i_packed([layout])
    input_ids = torch.arange(layout.seq_len, dtype=torch.long).unsqueeze(0)

    pixel_values = torch.rand(1, config.vae["in_channels"], 2, 2)
    posterior = model.vae.encode(pixel_values)
    cached_mean = posterior.mean.squeeze(2)
    cached_logvar = posterior.logvar.squeeze(2)

    _pin_flow_generator(model, device=pixel_values.device, seed=7)
    online = model(
        input_ids=input_ids,
        component_inputs={"pixel_values": pixel_values},
        hy3_sequence_metadata=metadata,
        use_cache=False,
    )
    _pin_flow_generator(model, device=pixel_values.device, seed=7)
    cached = model(
        input_ids=input_ids,
        component_inputs={"latent_posterior": {"mean": cached_mean, "logvar": cached_logvar}},
        hy3_sequence_metadata=metadata,
        use_cache=False,
    )
    torch.testing.assert_close(online.latents, cached.latents, rtol=0, atol=0)
    torch.testing.assert_close(online.loss["image_decoder_loss"], cached.loss["image_decoder_loss"], rtol=0, atol=0)


def test_collator_finalizes_packed_metadata_matches_compiler():
    """``MainCollator`` + model metadata hook must match the sequence compiler bit-exactly.

    This is the wire between the CPU data path (transform + collator) and the
    GPU-side packed varlen forward: if these two ever drift, cross-sample
    attention or per-sample flow RNG identity silently breaks.
    """
    # Imported here, not at module scope: veomni.data pulls in ``datasets``, and
    # every other test in this file runs without it.
    from veomni.data.data_collator import MainCollator
    from veomni.data.data_transform import process_sample_hunyuan_image_3

    config = _toy_config()
    torch.manual_seed(0)
    model = get_model_class(config)(config)

    # ``_transform_kwargs`` sizes the image to two strides per side, so the toy
    # config's stride of 1 (vae_downsample_factor=1 * patch_size=1) yields
    # grid_hw=(2, 2), matching the reference plan.
    image_processor = HunyuanImage3ImageProcessor(config)
    sample = _online_sample()
    (feature,) = process_sample_hunyuan_image_3(sample, **_transform_kwargs(config, image_processor))

    collator = MainCollator(
        data_collate_info=model.get_extra_collate_infos(),
        metadata_collate_func=model.get_metadata_collate_func(),
    )
    batch = collator([feature])

    metadata = batch["hy3_sequence_metadata"]
    assert metadata["layout"] == "packed_varlen"
    assert metadata["num_samples"] == 1

    reference = compile_single_gen_t2i_packed([T2ILayout(text_len=3, grid_h=2, grid_w=2)])
    for key in ("position_ids", "timestep_positions", "image_payload_indices"):
        torch.testing.assert_close(metadata[key], reference[key], rtol=0, atol=0)
    for key in ("cu_seqlens_prefix", "cu_seqlens_k_full", "cu_seqlens_q_image_suffix"):
        torch.testing.assert_close(metadata[key], reference[key], rtol=0, atol=0)
    # The transform's per-sample loss mask must agree with the same layout.
    torch.testing.assert_close(
        batch["image_output_mask"][0],
        T2ILayout(text_len=3, grid_h=2, grid_w=2).build_image_output_mask(),
        rtol=0,
        atol=0,
    )
    assert metadata["padded_sequence_length"] == batch["input_ids"].size(-1)

    # pixel_values is reassembled into component_inputs as a length-1 list under
    # the smart-stack contract; the staging keys must not survive collation.
    pixel_values = batch["component_inputs"]["pixel_values"]
    assert isinstance(pixel_values, list) and len(pixel_values) == 1
    stride = int(config.vae_downsample_factor[0]) * int(config.patch_size)
    expected_hw = 2 * stride
    assert pixel_values[0].shape == (1, 3, expected_hw, expected_hw)
    for staging_key in ("hy3_text_token_count", "hy3_grid_hw", "hy3_pixel_values"):
        assert staging_key not in batch


def _rejects_degenerate_layout():
    T2ILayout(text_len=0, grid_h=2, grid_w=2)


def _rejects_negative_grid():
    T2ILayout(text_len=2, grid_h=2, grid_w=-1)


def _rejects_empty_plan():
    compile_single_gen_t2i_packed([])


def _rejects_raw_dict_plan():
    compile_single_gen_t2i_packed([{"text_len": 2, "grid_h": 2, "grid_w": 2}])


def _rejects_bad_image_size():
    HunyuanImage3ImageProcessor(_toy_config()).preprocess(Image.new("RGB", (256, 256)), image_size=(0, 512))


def _rejects_image_size_off_stride():
    # The shared toy config has vae_downsample_factor=(1, 1), so its natural
    # stride is 1 and every size is a multiple -- override locally so the
    # divisibility rejection is actually exercised. The check lives in
    # preprocess() (post-resize) to keep the processor stateless in image_size.
    config = _toy_config()
    config.vae_downsample_factor = (16, 16)
    stride = int(config.vae_downsample_factor[0]) * int(config.patch_size)
    HunyuanImage3ImageProcessor(config).preprocess(Image.new("RGB", (256, 256)), image_size=(stride + 1, stride))


def _rejects_online_path_without_vae():
    config = _toy_config()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = get_model_class(config)(config)
    assert not hasattr(model, "vae"), "policy default must leave the VAE unbuilt"
    model._encode_pixel_values_to_posterior(torch.rand(1, 3, 2, 2))


def _rejects_heterogeneous_grids():
    # The packed compiler supports per-sample grids; the guard that matters is the
    # model's, because _get_latent_posterior stacks every sample into one
    # [B, C, H, W] batch and the 2D-conv head needs a single grid.
    config, model = _build_toy_model_with_vae()
    model._validate_reference_grid(((2, 3), (3, 2)), torch.zeros(2, config.vae["latent_channels"], 2, 2))


@pytest.mark.parametrize(
    ("thunk", "exception", "match"),
    [
        (_rejects_degenerate_layout, UnsupportedSequenceLayout, "text_len"),
        (_rejects_negative_grid, UnsupportedSequenceLayout, "grid_w"),
        (_rejects_empty_plan, TypeError, "non-empty sequence"),
        (_rejects_raw_dict_plan, TypeError, "T2ILayout"),
        (_rejects_bad_image_size, ValueError, "image_size"),
        (_rejects_image_size_off_stride, ValueError, "multiple of vae_downsample_factor"),
        (_rejects_online_path_without_vae, RuntimeError, "vae_encoder='frozen'"),
        (_rejects_heterogeneous_grids, ValueError, "shared image grid"),
    ],
    ids=lambda v: v.__name__[9:] if callable(v) else None,
)
def test_rejects_malformed_input(thunk, exception, match):
    """Every one of these is a silent-corruption vector if it were accepted:
    a degenerate layout compiles a corrupt pack, an off-stride image_size
    desyncs grid_hw from the sequence plan, and heterogeneous grids corrupt the
    stacked posterior plus the RoPE broadcast."""
    with pytest.raises(exception, match=match):
        thunk()


# --------------------------- GPU packed fast path ----------------------------


@_requires_flash_gpu
def test_packed_fast_path_matches_dense_oracle():
    device, dtype = "cuda", torch.bfloat16
    config, model = _build_model(device=device, dtype=dtype)
    grid = (2, 2)
    layout = T2ILayout(text_len=3, grid_h=grid[0], grid_w=grid[1])

    mean, logvar = _cached_posterior(1, config, device=device, dtype=dtype, grid=grid)
    component_inputs = {"latent_posterior": {"mean": mean, "logvar": logvar}}
    input_ids = torch.arange(layout.seq_len, device=device, dtype=torch.long).unsqueeze(0)

    packed_metadata = _to_device(compile_single_gen_t2i_packed([layout]), device)
    dense_metadata = _dense_oracle(packed_metadata)

    _pin_flow_generator(model, device=device, seed=5)
    dense = model(
        input_ids=input_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=dense_metadata,
        use_cache=False,
    )
    _pin_flow_generator(model, device=device, seed=5)
    packed = model(
        input_ids=input_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=packed_metadata,
        use_cache=False,
    )
    torch.testing.assert_close(packed.diffusion_prediction, dense.diffusion_prediction, **_ORACLE_TOLERANCE)
    torch.testing.assert_close(
        packed.loss["image_decoder_loss"], dense.loss["image_decoder_loss"], **_ORACLE_TOLERANCE
    )

    # Gradients keep the generic slack: their scale is not the prediction's, and
    # the forward comparison above is what pins the attention topology.
    packed.loss["image_decoder_loss"].backward()
    packed_grad = model.model.embed_tokens.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    dense.loss["image_decoder_loss"].backward()
    dense_grad = model.model.embed_tokens.weight.grad.clone()
    assert torch.isfinite(packed_grad).all() and torch.isfinite(dense_grad).all()
    torch.testing.assert_close(packed_grad, dense_grad, **_BF16_TOLERANCE)


def _packed_input_ids(layouts, device):
    # Per-sample sample-local token ids concatenated, so each packed sample uses
    # the same ids as when run standalone (the flow RNG, however, is batch-shaped,
    # so cross-composition comparisons must keep the batch composition fixed).
    blocks = [torch.arange(layout.seq_len, dtype=torch.long) for layout in layouts]
    return torch.cat(blocks).unsqueeze(0).to(device)


@_requires_flash_gpu
def test_packed_varlen_multi_sample_matches_dense_block_diagonal():
    device, dtype = "cuda", torch.bfloat16
    config, model = _build_model(device=device, dtype=dtype)
    grid = (2, 2)
    # Unequal prefix lengths: the dense oracle derives its per-sample blocks from
    # cu_seqlens, so it is a valid reference for a genuinely varlen pack -- this
    # pins the packed-global prefix/suffix boundaries, not just the topology of a
    # uniform batch.
    layouts = [
        T2ILayout(text_len=2, grid_h=grid[0], grid_w=grid[1]),
        T2ILayout(text_len=5, grid_h=grid[0], grid_w=grid[1]),
    ]
    mean, logvar = _cached_posterior(2, config, device=device, dtype=dtype, grid=grid)
    component_inputs = {"latent_posterior": {"mean": mean, "logvar": logvar}}

    packed_metadata = _to_device(compile_single_gen_t2i_packed(layouts), device)
    packed_ids = _packed_input_ids(layouts, device)
    _pin_flow_generator(model, device=device, seed=9)
    dense = model(
        input_ids=packed_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=_dense_oracle(packed_metadata),
        use_cache=False,
    )
    _pin_flow_generator(model, device=device, seed=9)
    packed = model(
        input_ids=packed_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=packed_metadata,
        use_cache=False,
    )
    torch.testing.assert_close(packed.diffusion_prediction, dense.diffusion_prediction, **_ORACLE_TOLERANCE)


@_requires_flash_gpu
def test_packed_heterogeneous_has_no_cross_sample_attention():
    device, dtype = "cuda", torch.bfloat16
    config, model = _build_model(device=device, dtype=dtype)
    grid = (2, 2)
    # Different prefix lengths (varlen) with a shared grid. Perturbing sample 1
    # must leave sample 0's prediction untouched: the batch-shaped flow RNG for
    # index 0 is independent of sample 1's posterior VALUES, so any change to
    # prediction[0] could only come from cross-sample attention leakage.
    layouts = [
        T2ILayout(text_len=2, grid_h=grid[0], grid_w=grid[1]),
        T2ILayout(text_len=5, grid_h=grid[0], grid_w=grid[1]),
    ]
    packed_metadata = _to_device(compile_single_gen_t2i_packed(layouts), device)
    packed_ids = _packed_input_ids(layouts, device)

    mean, logvar = _cached_posterior(2, config, device=device, dtype=dtype, grid=grid, seed=0)
    _pin_flow_generator(model, device=device, seed=9)
    baseline = model(
        input_ids=packed_ids,
        component_inputs={"latent_posterior": {"mean": mean, "logvar": logvar}},
        hy3_sequence_metadata=packed_metadata,
        use_cache=False,
    )

    perturbed_mean = mean.clone()
    perturbed_mean[1] = perturbed_mean[1] + 3.0
    _pin_flow_generator(model, device=device, seed=9)
    perturbed = model(
        input_ids=packed_ids,
        component_inputs={"latent_posterior": {"mean": perturbed_mean, "logvar": logvar}},
        hy3_sequence_metadata=packed_metadata,
        use_cache=False,
    )
    torch.testing.assert_close(
        perturbed.diffusion_prediction[0:1], baseline.diffusion_prediction[0:1], rtol=0, atol=1e-3
    )
    assert not torch.allclose(perturbed.diffusion_prediction[1:2], baseline.diffusion_prediction[1:2], atol=1e-2)


# ------------------------- Ulysses SP parity (spawned) ------------------------


_SP_HEAD_OVERRIDES = {
    "hidden_size": 64,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "attention_head_dim": 8,
    "head_dim": 8,
}


def _sp_worker(rank, world_size, grid, text_tokens, return_dict):
    import torch.distributed as dist

    from veomni.distributed.sequence_parallel.comm import set_ulysses_sequence_parallel_group

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29513",
        world_size=world_size,
        rank=rank,
    )
    group = dist.new_group(ranks=list(range(world_size)))
    device, dtype = f"cuda:{rank}", torch.bfloat16

    config, model = _build_model(device=device, dtype=dtype, overrides=_SP_HEAD_OVERRIDES)
    layouts = [T2ILayout(text_len=text_tokens + i, grid_h=grid[0], grid_w=grid[1]) for i in range(world_size)]
    mean, logvar = _cached_posterior(len(layouts), config, device=device, dtype=dtype, grid=grid)
    component_inputs = {"latent_posterior": {"mean": mean, "logvar": logvar}}

    packed = compile_single_gen_t2i_packed(layouts, pad_to_multiple_of=world_size)
    packed = _to_device(packed, device)
    input_ids = torch.arange(packed["padded_sequence_length"], device=device, dtype=torch.long).unsqueeze(0)

    # SP reference: same weights, SP disabled, no padding.
    set_ulysses_sequence_parallel_group(None)
    reference_packed = _to_device(compile_single_gen_t2i_packed(layouts), device)
    reference_ids = torch.arange(reference_packed["sequence_length"], device=device, dtype=torch.long).unsqueeze(0)
    _pin_flow_generator(model, device=device, seed=3)
    reference = model(
        input_ids=reference_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=reference_packed,
        use_cache=False,
    )
    reference_loss = float(reference.loss["image_decoder_loss"].detach().float().cpu())

    # SP path: every rank sees the full replicated inputs; the model slices.
    # Pin the same seed on every rank so SP and non-SP arms consume identical
    # noise -- mirrors the runtime invariant that ranks within a DP replica
    # share ``derive_flow_seed(config.flow["seed"], dp_rank)``.
    set_ulysses_sequence_parallel_group(group)
    _pin_flow_generator(model, device=device, seed=3)
    sp_output = model(
        input_ids=input_ids,
        component_inputs=component_inputs,
        hy3_sequence_metadata=packed,
        use_cache=False,
    )
    sp_loss = float(sp_output.loss["image_decoder_loss"].detach().float().cpu())

    if rank == 0:
        return_dict["reference_loss"] = reference_loss
        return_dict["sp_loss"] = sp_loss
    dist.barrier()
    dist.destroy_process_group()


@_requires_flash_gpu
@pytest.mark.parametrize("world_size", [1, 2])
def test_packed_varlen_sp_matches_single_gpu(world_size):
    # Parity for the SP fast path is monotone in ``world_size``: SP=2 hits every
    # code path (A2A + slice roundtrip) that SP=4/8 exercise, while keeping the
    # per-test GPU requirement to 2. The 16xH20 E2E training smoke covers
    # SP=2/4/8 under real workloads.
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Ulysses SP={world_size} needs {world_size} GPUs.")
    import torch.multiprocessing as mp

    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        _sp_worker,
        args=(world_size, (2, 2), 3, return_dict),
        nprocs=world_size,
        join=True,
    )
    assert "sp_loss" in return_dict and "reference_loss" in return_dict
    assert abs(return_dict["sp_loss"] - return_dict["reference_loss"]) <= 2e-2 * (
        1 + abs(return_dict["reference_loss"])
    )


# ------------------------ End-to-end training smoke --------------------------


def _write_toy_parquet(path: Path, num_rows: int = 8) -> None:
    """Write a toy T2I parquet with ``{id, prompt, image}`` rows.

    ``datasets`` serializes PIL images natively (Image feature), which is what
    the transform's ``image_processor.preprocess`` expects on the way into the
    frozen online VAE encoder.
    """
    from datasets import Dataset, Features, Value
    from datasets import Image as ImageFeature

    generator = torch.Generator().manual_seed(0)
    rows = []
    for index in range(num_rows):
        # Deterministic per-row colour, so the smoke's finite-loss check sees
        # byte-identical inputs across runs.
        color = tuple((torch.randint(0, 256, (3,), generator=generator)).tolist())
        rows.append(
            {
                "id": f"sample_{index}",
                "prompt": f"a toy prompt number {index}",
                "image": Image.new("RGB", _TOY_IMAGE_HW, color=color),
            }
        )
    features = Features({"id": Value("string"), "prompt": Value("string"), "image": ImageFeature()})
    Dataset.from_list(rows, features=features).to_parquet(str(path))


def _run_training(tmp_path: Path, *, nproc: int, extra_args: list[str], port: int) -> dict:
    data_path = tmp_path / "toy_t2i.parquet"
    output_dir = tmp_path / "out"
    _write_toy_parquet(data_path)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        str(_TRAIN_SCRIPT),
        str(_TOY_TRAIN_CONFIG),
        f"--data.train_path={data_path}",
        f"--train.checkpoint.output_dir={output_dir}",
        *extra_args,
    ]
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))

    with open(output_dir / "log_dict.json") as handle:
        return json.load(handle)


def _assert_finite_training(log: dict, *, min_steps: int) -> None:
    losses = log["image_decoder_loss"]
    grad_norms = log["grad_norm"]
    assert len(losses) >= min_steps, f"expected >= {min_steps} steps, got {len(losses)}"
    assert all(math.isfinite(v) and v > 0 for v in losses), losses
    assert all(math.isfinite(v) for v in grad_norms), grad_norms


@_requires_flash_gpu
def test_end_to_end_single_gpu_online_vae(tmp_path):
    """Single-GPU (ddp/cuda, bf16) e2e via online VAE: >=5 finite flow-loss steps."""
    log = _run_training(tmp_path, nproc=1, extra_args=[], port=29541)
    _assert_finite_training(log, min_steps=5)


@_requires_flash_gpu
def test_end_to_end_fsdp2_online_vae(tmp_path):
    """2-GPU FSDP2 meta-init + mixed precision e2e via online VAE: >=4 finite flow-loss steps."""
    if torch.cuda.device_count() < 2:
        pytest.skip("FSDP2 e2e needs 2 GPUs.")
    log = _run_training(
        tmp_path,
        nproc=2,
        extra_args=[
            "--train.global_batch_size=2",
            "--train.init_device=meta",
            "--train.accelerator.fsdp_config.fsdp_mode=fsdp2",
            "--train.accelerator.fsdp_config.mixed_precision.enable=true",
        ],
        port=29542,
    )
    _assert_finite_training(log, min_steps=4)
