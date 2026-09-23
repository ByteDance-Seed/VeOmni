"""Real tiny-component checks; no official weights or accelerator are required."""

import importlib.util
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from diffusers import QwenImageTransformer2DModel as DiffusersQwenImageTransformer
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from veomni.arguments import parse_args
from veomni.arguments.arguments_types import AcceleratorConfig, OpsImplementationConfig
from veomni.data import build_data_transform, build_dataloader, build_dataset
from veomni.data.multimodal.dit.preprocess import qwen_image_preprocess
from veomni.distributed import parallel_state
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.models import build_foundation_model
from veomni.models.auto import build_config
from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.trainer.dit_trainer import DiTDataCollator, DiTTrainer, VeOmniDiTArguments


ROOT = Path(__file__).resolve().parents[2]
DIFFUSERS_FORWARD = DiffusersQwenImageTransformer.forward


@pytest.fixture(scope="module", autouse=True)
def single_process_rank(tmp_path_factory):
    from veomni.models.diffusers.qwen_image.qwen_image_condition import modeling_qwen_image_condition

    # The CPU loader materializes checkpoint tensors on distributed rank zero.
    rendezvous = tmp_path_factory.mktemp("qwen_image_gloo") / "rendezvous"
    torch.distributed.init_process_group("gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1)
    try:
        # Only device selection is controlled; every numerical component is real.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(modeling_qwen_image_condition, "get_device_type", lambda: "cpu")
            patch.setattr(parallel_state, "get_device_type", lambda: "cpu")
            parallel_state.init_parallel_state_from_config(AcceleratorConfig(), name="base")
            yield
    finally:
        torch.distributed.destroy_process_group()
        parallel_state.clear_parallel_state()


def eager_ops():
    config = yaml.safe_load((ROOT / "configs/dit/qwen_image_sft.yaml").read_text())
    return OpsImplementationConfig(**config["model"]["ops_implementation"])


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    """Random test dimensions are fixture choices, not Qwen-Image model metadata."""
    torch.manual_seed(123)
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("qwen_image_snapshot")
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = {token: idx for idx, token in enumerate(special + sorted(ByteLevel.alphabet()))}
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], pad_token=special[0], eos_token=special[2])
    tokenizer.add_special_tokens({"additional_special_tokens": special[1:]})
    tokenizer.save_pretrained(root / "tokenizer")
    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 512,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=len(tokenizer) - 1,
        video_token_id=len(tokenizer) - 2,
        vision_start_token_id=len(tokenizer) - 3,
    )
    Qwen2_5_VLForConditionalGeneration(config).save_pretrained(root / "text_encoder")
    AutoencoderKLQwenImage(
        base_dim=4,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[False],
        latents_mean=[0.1, -0.2, 0.3, -0.4],
        latents_std=[1.0, 1.5, 2.0, 2.5],
    ).save_pretrained(root / "vae")
    FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
        shift_terminal=0.02,
    ).save_pretrained(root / "scheduler")
    return root


def condition_model(snapshot, recipe="diffsynth", offline=False, **kwargs):
    config = MODEL_CONFIG_REGISTRY["QwenImageConditionModel"]().from_pretrained(
        str(snapshot),
        height=16,
        width=16,
        max_sequence_length=256,
        training_recipe=recipe,
        image_resize_mode="center_crop",
        seed=7,
        **kwargs,
    )
    model = MODELING_REGISTRY["QwenImageConditionModel"]()._from_config(config, meta_init=offline)
    return model.requires_grad_(False).eval()


def transformer(dtype="float32"):
    return build_foundation_model(
        str(ROOT / "tests/toy_config/qwen_image_toy/config.json"),
        init_device="cpu",
        torch_dtype=dtype,
        ops_implementation=eager_ops(),
    )


@pytest.fixture(scope="module")
def encoded(snapshot):
    model = condition_model(snapshot)
    images = [[Image.new("RGB", (24, 16), color="red")], [Image.new("RGB", (16, 24), color="blue")]]
    return model.get_condition(inputs=["red", "a blue rectangle on a white wall"], images=images)


def test_yaml_registry_and_dataclasses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dit.py", str(ROOT / "configs/dit/qwen_image_sft.yaml")])
    args = parse_args(VeOmniDiTArguments)
    assert args.train.training_task == "online_training"
    assert args.data.source_name == "Qwen-Image"
    assert args.model.condition_model_cfg["training_recipe"] == "diffsynth"
    assert args.model.optimizer.type == "adamw"
    assert args.model.optimizer.lr == 1e-5
    assert args.model.optimizer.max_grad_norm == 1.0
    assert args.model.accelerator.init_device == "meta"
    assert args.model.accelerator.gradient_checkpointing.enable
    assert args.model.accelerator.fsdp_config.fsdp_mode == "fsdp2"
    assert not args.model.accelerator.fsdp_config.mixed_precision.enable
    assert args.model.accelerator.dp_replicate_size == 1
    cfg = build_config(str(ROOT / "tests/toy_config/qwen_image_toy/config.json"))
    assert cfg.condition_model_type == "QwenImageConditionModel"
    assert MODELING_REGISTRY[cfg.model_type]().__name__ == "QwenImageTransformer2DModel"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_native_data_to_optimizer_and_checkpoint(snapshot, tmp_path, capsys, dtype):
    records = []
    for idx, prompt in enumerate(["red", "a blue rectangle on a white wall"]):
        name = f"{idx}.png"
        Image.new("RGB", (24, 16), color=(200, idx * 100, 40)).save(tmp_path / name)
        records.append({"prompt": prompt, "image_path": name})
    data_path = tmp_path / "train.jsonl"
    data_path.write_text("\n".join(json.dumps(row) for row in records))
    transform = build_data_transform("dit_online", data_dir=str(tmp_path))
    dataset = build_dataset(
        "iterable", train_path=str(data_path), source_name="Qwen-Image", transform=transform, shuffle=False
    )
    loader = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=1,
        global_batch_size=2,
        dataloader_batch_size=2,
        max_seq_len=256,
        train_steps=1,
        dyn_bsz=False,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        shuffle=False,
        collate_fn=DiTDataCollator(),
    )
    condition = condition_model(snapshot)
    model = transformer(dtype).train()
    model.gradient_checkpointing_enable()
    optimizer = build_optimizer(model, lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, train_steps=1, lr=1e-3)
    before = model.proj_out.weight.detach().clone()
    # Use the actual trainer step and single-rank mesh, without accelerator/FSDP wrapping.
    trainer = DiTTrainer.__new__(DiTTrainer)
    trainer.base = SimpleNamespace(
        num_micro_batches=2,
        model=model,
        device=torch.device("cpu"),
        LOG_SAMPLE=False,
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
    )
    trainer.condition_model = condition
    trainer.training_task = "online_training"
    losses = []
    micro_batches = next(iter(loader))
    assert len(micro_batches) == trainer.base.num_micro_batches
    for batch in micro_batches:
        loss, loss_dict = trainer.forward_backward_step(batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        torch.testing.assert_close(loss, loss_dict["mse_loss"])
        losses.append(float(loss.detach()))
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert sum(float(grad.abs().sum()) for grad in grads) > 0
    assert all(param.grad is None for param in condition.parameters())
    grad_norm = veomni_clip_grad_norm(model, max_norm=1.0, error_if_nonfinite=True)
    assert torch.isfinite(grad_norm) and grad_norm > 0
    optimizer.step()
    scheduler.step()
    assert not torch.equal(before, model.proj_out.weight)
    model.eval()
    with torch.no_grad():
        inputs = condition.process_condition(**condition.get_condition(**micro_batches[-1]))
        expected = model(**inputs).predictions[0]
    checkpoint = tmp_path / "transformer"
    model.save_pretrained(checkpoint)
    reloaded = build_foundation_model(
        str(checkpoint),
        weights_path=str(checkpoint),
        init_device="cpu",
        torch_dtype=dtype,
        ops_implementation=eager_ops(),
    ).eval()
    with torch.no_grad():
        actual = reloaded(**inputs).predictions[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    with capsys.disabled():
        print(
            f"QWEN_IMAGE_UPDATE_PASS dtype={dtype}, losses={losses}, grad_norm={float(grad_norm)}, changed_parameters=True, checkpoint_max_abs={(actual - expected).abs().max().item()}"
        )


@pytest.mark.parametrize("recipe", ["veomni", "diffsynth"])
def test_online_offline_conditions(snapshot, encoded, recipe):
    online = condition_model(snapshot, recipe)
    offline = condition_model(snapshot, recipe, offline=True)
    assert offline.vae is None and offline.text_encoder is None
    expected, actual = online.process_condition(**encoded), offline.process_condition(**encoded)
    for key in ("hidden_states", "training_target", "timestep", "latents"):
        for lhs, rhs in zip(expected[key], actual[key]):
            torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
    for noisy, target, latent, timestep in zip(
        actual["hidden_states"], actual["training_target"], actual["latents"], actual["timestep"]
    ):
        torch.testing.assert_close(noisy, latent + timestep.view(-1, 1, 1) * target)
    assert ("loss_weights" in actual) == (recipe == "diffsynth")


def test_diffsynth_schedule_reference(snapshot, encoded):
    root = Path(os.environ.get("DIFFSYNTH_ROOT", ROOT.parent / "DiffSynth-Studio"))
    if not root.is_dir():
        pytest.skip("Set DIFFSYNTH_ROOT to compare against its actual scheduler source.")
    spec = importlib.util.spec_from_file_location(
        "diffsynth_flow_match_reference", Path(root) / "diffsynth/diffusion/flow_match.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.FlowMatchScheduler(template="Qwen-Image")
    reference.set_timesteps(1000, training=True)
    model = condition_model(snapshot, offline=True)
    model.process_condition(**encoded)
    torch.testing.assert_close(model.scheduler.sigmas[:-1], reference.sigmas, atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(model.scheduler.timesteps, reference.timesteps, atol=2e-4, rtol=1e-6)
    torch.testing.assert_close(model._training_weights, reference.linear_timesteps_weights, atol=2e-6, rtol=1e-5)


def test_weighted_loss_padding_and_invalid_batches(snapshot, encoded):
    condition = condition_model(snapshot, offline=True)
    inputs = condition.process_condition(**encoded)
    model = transformer().eval()
    outputs = model(**inputs)
    reference = torch.stack(
        [
            ((prediction.float() - target.float()) ** 2).mean() * weight.squeeze()
            for prediction, target, weight in zip(
                outputs.predictions, inputs["training_target"], inputs["loss_weights"]
            )
        ]
    ).mean()
    torch.testing.assert_close(outputs.loss["mse_loss"], reference)
    mask = inputs["encoder_hidden_states_mask"][0]
    assert mask is not None and not mask.all()
    inputs["encoder_hidden_states"][0] = inputs["encoder_hidden_states"][0].clone()
    inputs["encoder_hidden_states"][0][~mask.bool()] = 1000
    torch.testing.assert_close(model(**inputs).predictions[0], outputs.predictions[0], atol=2e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="sample count"):
        model(**{**inputs, "training_target": inputs["training_target"][:1]})
    with pytest.raises(ValueError, match="shapes must match"):
        model(**{**inputs, "training_target": [target[..., :1] for target in inputs["training_target"]]})
    with pytest.raises(ValueError, match="sample count"):
        condition.process_condition(**{**encoded, "latents": encoded["latents"][:1]})
    with pytest.raises(ValueError, match="packed latent grid"):
        condition.process_condition(**{**encoded, "img_shapes": [[(1, 1, 1)]] * 2})
    with pytest.raises(ValueError, match="equally sized"):
        condition.get_condition(inputs=["one"], images=[])
    with pytest.raises(ValueError, match="divisible"):
        config = condition.config.to_dict()
        config["height"] = 15
        MODELING_REGISTRY["QwenImageConditionModel"]()(condition.config_class(**config), meta_init=True)


def test_preprocessor_contract(tmp_path):
    assert qwen_image_preprocess({"prompt": "", "image_path": "sample.png"}, data_dir=str(tmp_path))[2] == [
        str(tmp_path / "sample.png")
    ]
    assert qwen_image_preprocess({"caption": "image", "image_bytes": b"png"})[2] == [b"png"]
    assert qwen_image_preprocess({"text": "image", "image": "https://example.com/image.png"}, data_dir=str(tmp_path))[
        2
    ] == ["https://example.com/image.png"]
    with pytest.raises(ValueError, match="text prompt"):
        qwen_image_preprocess({"prompt": ["one"], "image": "sample.png"})


def test_backbone_reference_forward_and_gradients(snapshot, encoded):
    inputs = condition_model(snapshot, offline=True).process_condition(**encoded)
    model = transformer()
    reference = DiffusersQwenImageTransformer(**model.config.to_diffuser_dict())
    reference.load_state_dict(model.state_dict())
    sample = {
        key: inputs[key][0]
        for key in ("hidden_states", "timestep", "encoder_hidden_states", "encoder_hidden_states_mask")
    }
    sample["img_shapes"] = [inputs["img_shapes"][0]]
    actual = model.predict_noise(**sample)[0]
    expected = DIFFUSERS_FORWARD(reference, **sample, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (_, param), (_, reference_param) in zip(model.named_parameters(), reference.named_parameters()):
        if reference_param.grad is not None:
            assert param.grad is not None
            torch.testing.assert_close(param.grad, reference_param.grad, atol=2e-6, rtol=1e-5)
