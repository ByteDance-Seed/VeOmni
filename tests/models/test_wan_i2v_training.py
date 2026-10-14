"""CPU checks for Wan I2V conditioning against the pinned Diffusers pipeline."""

import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline
from diffusers.models import normalization
from diffusers.video_processor import VideoProcessor

from veomni.distributed.async_offload import apply_async_activation_offload
from veomni.models.diffusers.wan_t2v.wan_condition import modeling_wan_condition as condition_module
from veomni.models.diffusers.wan_t2v.wan_condition.configuration_wan_condition import (
    WanTransformer3DConditionModelConfig,
)
from veomni.models.diffusers.wan_t2v.wan_transformer import modeling_wan_transformer as transformer_module
from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import WanTransformer3DModelConfig


@pytest.fixture
def condition_factory(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(condition_module, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(condition_module, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0))
    monkeypatch.setattr(transformer_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(
        transformer_module._WanTransformer3DModel, "forward", transformer_module.WanTransformer3DModel_forward
    )
    monkeypatch.setattr(normalization, "is_torch_npu_available", lambda: False)

    def load_components(self):
        self.vae = AutoencoderKLWan(
            base_dim=4,
            z_dim=4,
            dim_mult=[1, 2, 2],
            num_res_blocks=1,
            temperal_downsample=[True, True],
            scale_factor_spatial=4,
            scale_factor_temporal=4,
            latents_mean=[0.1, -0.2, 0.3, -0.4],
            latents_std=[1.0, 1.5, 2.0, 2.5],
        ).eval()
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.video_processor = VideoProcessor(vae_scale_factor=4)
        self.negative_prompt_embeds = torch.zeros(1, 3, 8)

    monkeypatch.setattr(condition_module.WanTransformer3DConditionModel, "_load_components", load_components)
    # Text encoding is orthogonal to the image-conditioning tensor under test.
    monkeypatch.setattr(
        condition_module.WanPipeline,
        "encode_prompt",
        lambda self, prompt, **kwargs: (torch.zeros(len(prompt), 3, 8), None),
    )

    def build(**kwargs):
        config = WanTransformer3DConditionModelConfig(video_max_size=16, cfg_negative_prob=0, **kwargs)
        return condition_module.WanTransformer3DConditionModel(config).requires_grad_(False).eval()

    yield build
    torch.set_num_threads(previous_threads)


def test_wan_i2v_condition_matches_inference_and_updates(condition_factory, monkeypatch):
    condition = condition_factory(image_conditioning=True)
    video = torch.randint(0, 256, (9, 3, 16, 16), dtype=torch.uint8)
    encoded = condition.get_condition(inputs=["a moving car"], videos=[[video]])
    packed = condition.process_condition(**encoded)
    reference = WanImageToVideoPipeline(
        tokenizer=None, text_encoder=None, transformer=None, vae=condition.vae, scheduler=condition.scheduler
    )
    pixels = condition._preprocess_video(video)
    _, expected = reference.prepare_latents(
        pixels[:, :, 0],
        batch_size=1,
        num_channels_latents=4,
        height=16,
        width=16,
        num_frames=9,
        dtype=torch.float32,
        device=torch.device("cpu"),
        latents=torch.zeros_like(packed["latents"][0]),
    )
    actual = torch.cat([packed["mask"][0], packed["image_latents"][0]], dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(actual[:, 4:, 1:], actual[:, 4:, :1].expand_as(actual[:, 4:, 1:]))
    state = condition.rng_state_dict()
    first = condition.process_condition(**encoded)
    condition.load_rng_state_dict(state)
    # Offline embeddings retain posterior parameters and the full temporal mask.
    second = condition.process_condition(**{key: [value[0].clone()] for key, value in encoded.items()})
    for key in first:
        torch.testing.assert_close(first[key][0], second[key][0], rtol=0, atol=0)

    config = WanTransformer3DModelConfig(
        in_channels=12,
        out_channels=4,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        text_dim=8,
        freq_dim=8,
        ffn_dim=32,
    )
    config._attn_implementation = "eager"
    model = transformer_module.WanTransformer3DModel(config)

    # A host with torch_npu installed must still accept CPU model tensors.
    def reject_npu_transpose(*args, **kwargs):
        pytest.fail("CPU input reached npu_confusion_transpose")

    monkeypatch.setattr(transformer_module, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(transformer_module, "torch_npu", SimpleNamespace(npu_confusion_transpose=reject_npu_transpose))
    before = model.proj_out.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = model(**packed)
    assert result.predictions[0].shape == packed["training_target"][0].shape
    result.loss["mse_loss"].backward()
    assert torch.isfinite(model.proj_out.weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.proj_out.weight)
    with pytest.raises(ValueError, match="full image_latents"):
        model(**{key: value for key, value in packed.items() if key != "image_latents"})
    with pytest.raises(ValueError, match="first-frame repetition"):
        model(**{**packed, "image_latents": [packed["image_latents"][0][:, :, :1]]})


def test_wan_i2v_rejects_missing_or_repeated_conditions(condition_factory):
    condition = condition_factory(image_conditioning=True)
    encoded = condition.get_condition(inputs=["video"], videos=[[torch.zeros(5, 3, 16, 16)]])
    with pytest.raises(ValueError, match="regenerate legacy"):
        condition.process_condition(latents=encoded["latents"], context=encoded["context"])
    with pytest.raises(ValueError, match="full target video grid"):
        condition.process_condition(**{**encoded, "image_latents": [encoded["image_latents"][0][:, :, :1]]})
    with pytest.raises(ValueError, match="video frames"):
        condition.get_condition(inputs=["video"], videos=[[torch.zeros(6, 3, 16, 16)]])


@pytest.mark.parametrize("expanded", [False, True])
def test_wan_non_concat_paths_keep_their_input_contract(condition_factory, expanded):
    condition = condition_factory(expand_timesteps=expanded)
    encoded = condition.get_condition(inputs=["video"], videos=[[torch.zeros(5, 3, 16, 16)]])
    assert set(encoded) == {"latents", "context"}
    packed = condition.process_condition(**encoded)
    assert "image_latents" not in packed and "mask" not in packed
    assert packed["hidden_states"][0].shape == packed["latents"][0].shape
    if expanded:
        torch.testing.assert_close(packed["hidden_states"][0][:, :, 0], packed["latents"][0][:, :, 0])
        assert packed["timestep"][0].ndim == 2
        assert packed["loss_start_frame"] == [1]
    else:
        assert packed["timestep"][0].ndim == 1
        assert "loss_start_frame" not in packed


def test_wan_expanded_loss_excludes_condition_frame(condition_factory, monkeypatch):
    condition = condition_factory(expand_timesteps=True)
    encoded = condition.get_condition(inputs=["video"], videos=[[torch.zeros(9, 3, 16, 16)]])
    packed = condition.process_condition(**encoded)
    config = WanTransformer3DModelConfig(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        text_dim=8,
        freq_dim=8,
        ffn_dim=32,
    )
    config._attn_implementation = "eager"
    model = transformer_module.WanTransformer3DModel(config)
    result = model(**packed)
    prediction = result.predictions[0]
    expected = torch.nn.functional.mse_loss(prediction[:, :, 1:], packed["training_target"][0][:, :, 1:])
    torch.testing.assert_close(result.loss["mse_loss"], expected, atol=1e-6, rtol=1e-6)
    prediction.retain_grad()
    result.loss["mse_loss"].backward()
    assert torch.count_nonzero(prediction.grad[:, :, 0]) == 0
    assert torch.count_nonzero(prediction.grad[:, :, 1:]) > 0
    assert torch.isfinite(model.proj_out.weight.grad).all()

    # Changing only a condition-frame output must not add a training objective.
    perturbed = prediction.detach().clone()
    perturbed[:, :, 0] += 1000
    monkeypatch.setattr(transformer_module._WanTransformer3DModel, "forward", lambda *args, **kwargs: perturbed)
    torch.testing.assert_close(model(**packed).loss["mse_loss"], result.loss["mse_loss"], rtol=0, atol=0)
    full_loss = model(**{**packed, "loss_start_frame": 0}).loss["mse_loss"]
    torch.testing.assert_close(full_loss, torch.nn.functional.mse_loss(perturbed, packed["training_target"][0]))
    with pytest.raises(ValueError, match="leave at least one target frame"):
        model(**{**packed, "loss_start_frame": [perturbed.shape[2]]})


def test_wan_expanded_training_requires_target_frames(condition_factory):
    condition = condition_factory(expand_timesteps=True)
    encoded = condition.get_condition(inputs=["video"], videos=[[torch.zeros(1, 3, 16, 16)]])
    with pytest.raises(ValueError, match="at least one frame after the condition"):
        condition.process_condition(**encoded)


@pytest.mark.parametrize("name", ["wan2.2_I2V_5B", "wan2.2_I2V_14B"])
def test_wan_examples_preserve_checkpoint_recomputation(condition_factory, name):
    settings = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/dit" / f"{name}.yaml").read_text())
    accelerator = settings["model"]["accelerator"]
    concat = name.endswith("14B")
    condition = condition_factory(image_conditioning=concat, expand_timesteps=not concat)
    packed = condition.process_condition(
        **condition.get_condition(inputs=["video"], videos=[[torch.zeros(9, 3, 16, 16)]])
    )
    config = WanTransformer3DModelConfig(
        in_channels=12 if concat else 4,
        out_channels=4,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        text_dim=8,
        freq_dim=8,
        ffn_dim=32,
    )
    config._attn_implementation = "eager"
    model = transformer_module.WanTransformer3DModel(config)
    calls = []
    hook = model.blocks[0].register_forward_pre_hook(lambda *args: calls.append(1))
    if accelerator["offload_config"]["enable_async_activation"]:
        apply_async_activation_offload(model, activation_offload_modules=[])
    assert accelerator["gradient_checkpointing"]["enable"]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        model(**packed).loss["mse_loss"].backward()
    finally:
        hook.remove()
    assert len(calls) == 2, "Wan must run the block once in forward and again during backward recomputation."
    assert torch.isfinite(model.proj_out.weight.grad).all()


@pytest.mark.parametrize("in_channels,out_channels,expected", [(36, 16, True), (48, 48, False), (16, 16, False)])
def test_wan_condition_mode_comes_from_snapshot(tmp_path, in_channels, out_channels, expected):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer/config.json").write_text(
        json.dumps({"in_channels": in_channels, "out_channels": out_channels, "patch_size": [1, 2, 2]})
    )
    config = WanTransformer3DConditionModelConfig.from_pretrained(str(tmp_path))
    assert config.image_conditioning is expected


def _wan_sp_worker(rank, rendezvous):
    from veomni.arguments.arguments_types import AcceleratorConfig, FSDPConfig
    from veomni.distributed import parallel_state

    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(parallel_state, "get_device_type", lambda: "cpu")
        patch.setattr(normalization, "is_torch_npu_available", lambda: False)
        patch.setattr(
            transformer_module._WanTransformer3DModel, "forward", transformer_module.WanTransformer3DModel_forward
        )
        torch.distributed.init_process_group(
            "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60)
        )
        try:
            # The pinned Linux Gloo backend supports all_to_all_single, but not
            # the list API. Newer local builds must exercise the same contract.
            def reject_list_all_to_all(*args, **kwargs):
                raise RuntimeError("Backend gloo does not support alltoall")

            patch.setattr(torch.distributed, "all_to_all", reject_list_all_to_all)
            parallel_state.init_parallel_state_from_config(
                AcceleratorConfig(ulysses_size=2, fsdp_config=FSDPConfig(fsdp_mode="ddp")), name=None
            )
            torch.manual_seed(12)
            config = WanTransformer3DModelConfig(
                in_channels=4,
                out_channels=4,
                num_layers=1,
                num_attention_heads=2,
                attention_head_dim=8,
                text_dim=8,
                freq_dim=8,
                ffn_dim=32,
            )
            config._attn_implementation = "eager"
            model = transformer_module.WanTransformer3DModel(config)
            reference = copy.deepcopy(model)
            for batch_size, frames in ((1, 3), (1, 4), (2, 3), (2, 4)):
                # Three tokens need communication padding, four do not.
                latents = torch.randn(batch_size, 4, frames, 2, 2)
                batch = dict(
                    latents=latents,
                    hidden_states=latents,
                    timestep=torch.arange(frames).expand(batch_size, -1).float(),
                    encoder_hidden_states=torch.randn(batch_size, 3, 8),
                    training_target=torch.zeros_like(latents),
                )
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                result = model(**batch)
                assert result.predictions[0].shape == latents.shape
                result.loss["mse_loss"].backward()
                with parallel_state.use_parallel_state(
                    parallel_state.ParallelState(device_type="cpu", dp_size=2, dp_shard_size=2)
                ):
                    expected = reference(**batch)
                    expected.loss["mse_loss"].backward()
                torch.testing.assert_close(result.predictions[0], expected.predictions[0], atol=2e-6, rtol=1e-5)
                for (name, parameter), (_, ref_parameter) in zip(
                    model.named_parameters(), reference.named_parameters()
                ):
                    assert parameter.grad is not None, name
                    torch.distributed.all_reduce(parameter.grad)
                    parameter.grad.div_(2)
                    torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=2e-6, rtol=1e-5, msg=name)
        finally:
            torch.distributed.destroy_process_group()
            parallel_state.clear_parallel_state()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires CPU Gloo")
def test_wan_sp_unpads_before_output_head(tmp_path):
    torch.multiprocessing.spawn(_wan_sp_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
