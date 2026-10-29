from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.training_utils import make_eager_ops_config, unbuilt_runtime
from veomni.models import build_foundation_model
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.trainer import vlm_trainer
from veomni.trainer.vlm_trainer import (
    VeOmniVLMArguments,
    VLMMDataArguments,
    VLMMModelArguments,
    VLMModelRuntime,
    _get_vlm_visual_module,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


_FREEZE_VIT_VLM_CASES = [
    pytest.param("./tests/toy_config/qwen2vl_toy/config.json", id="qwen2_vl"),
    pytest.param("./tests/toy_config/qwen3_5_toy/config.json", id="qwen3_5"),
    pytest.param("./tests/toy_config/qwen3_5_moe_toy/config.json", id="qwen3_5_moe"),
    pytest.param("./tests/toy_config/qwen25vl_toy/config.json", id="qwen2_5_vl"),
    pytest.param("./tests/toy_config/qwen3vl_toy/config.json", id="qwen3_vl"),
    pytest.param("./tests/toy_config/qwen3vlmoe_toy/config.json", id="qwen3_vl_moe"),
    pytest.param("./tests/toy_config/qwen4_exp_toy/config.json", id="qwen4_exp"),
    pytest.param("./tests/toy_config/gemma4_toy/config.json", id="gemma4"),
]


class _Gemma3VLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.vision_tower = torch.nn.Linear(4, 4)
        self.model.language_model = torch.nn.Linear(4, 4)


def test_get_vlm_visual_module_finds_gemma3_vision_tower():
    model = _Gemma3VLM()

    assert _get_vlm_visual_module(model) is model.model.vision_tower


def test_freeze_vit_freezes_gemma3_vision_tower(monkeypatch):
    model = _Gemma3VLM()
    runtime = unbuilt_runtime(
        SimpleNamespace(lora_config=None),
        cls=VLMModelRuntime,
        train=SimpleNamespace(freeze_vit=True, freeze_audio_tower=False),
    )
    monkeypatch.setattr(vlm_trainer, "pretty_print_trainable_parameters", lambda _model: None)
    monkeypatch.setattr(vlm_trainer.helper, "print_device_mem_info", lambda _message: None)
    runtime.model = model
    runtime.model_config = SimpleNamespace(model_type="gemma3")

    runtime._freeze_model_module()

    assert all(not param.requires_grad for param in model.model.vision_tower.parameters())
    assert all(param.requires_grad for param in model.model.language_model.parameters())


def test_optimizer_uses_vit_lr_for_gemma3_vision_tower(monkeypatch):
    model = _Gemma3VLM()
    runtime = unbuilt_runtime(
        SimpleNamespace(optimizer=SimpleNamespace(lr=2e-4)),
        cls=VLMModelRuntime,
        train=SimpleNamespace(vit_lr=3e-5),
    )
    captured = {}

    def fake_build_optimizer(_runtime, param_groups=None):
        captured["param_groups"] = param_groups

    monkeypatch.setattr(VeOmniModelRuntime, "_build_optimizer", fake_build_optimizer)
    runtime.model = model
    runtime.model_config = SimpleNamespace(model_type="gemma3")

    runtime._build_optimizer()

    vision_param_ids = {id(param) for param in model.model.vision_tower.parameters()}
    vit_group, other_group = captured["param_groups"]
    assert vit_group["lr"] == runtime.train_args.vit_lr
    assert {id(param) for param in vit_group["params"]} == vision_param_ids
    assert not vision_param_ids.intersection(id(param) for param in other_group["params"])


def test_vlm_data_transform_receives_configured_max_seq_len(monkeypatch):
    args = SimpleNamespace(data=SimpleNamespace(max_seq_len=16384, mm_configs={}))
    model = SimpleNamespace(
        model_config=SimpleNamespace(model_type="gemma3"),
        processor=object(),
        chat_template=object(),
        unwrapped_module=SimpleNamespace(get_position_id_func=lambda: object()),
    )
    trainer = object.__new__(vlm_trainer.VLMTrainer)
    trainer.base = SimpleNamespace(args=args, model=model)
    captured = {}

    def fake_build_data_transform(model_type, **kwargs):
        captured["model_type"] = model_type
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(vlm_trainer, "build_data_transform", fake_build_data_transform)

    trainer._build_data_transform()

    assert captured["model_type"] == "gemma3"
    assert captured["max_seq_len"] == 16384


@pytest.mark.parametrize("freeze_vit", [False, True])
@pytest.mark.parametrize("config_path", _FREEZE_VIT_VLM_CASES)
def test_freeze_vit_on_vlm_model(config_path, freeze_vit):
    # This test only constructs the model on `meta` and verifies freeze
    # behaviour — it never runs forward. Use an all-eager ops config so the
    # build works everywhere: it pins every per-op field (including the
    # Qwen3.5 GatedDeltaNet trio that has no FLA backend on NPU and the
    # GPU-only liger/triton defaults that fail NPU validation). Eager paths
    # that raise only at forward time are fine because this test never
    # forwards.
    ops_implementation = make_eager_ops_config()
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=ops_implementation,
    )
    visual = _get_vlm_visual_module(model)
    assert visual is not None

    args = VeOmniVLMArguments(
        model=VLMMModelArguments(config_path=config_path, ops_implementation=ops_implementation),
        data=VLMMDataArguments(train_path="dummy"),
    )
    args.train.freeze_vit = freeze_vit
    runtime = unbuilt_runtime(args.model, cls=VLMModelRuntime, train=args.train)
    runtime.model = model
    runtime.model_config = model.config

    runtime._freeze_model_module()

    assert all(param.requires_grad is not freeze_vit for param in visual.parameters())


def test_gemma4_vlm_trainer_initializes_raw_data_pipeline(monkeypatch):
    """Run the actual constructor; replace only distributed/runtime and job I/O."""
    from types import SimpleNamespace

    import torch

    from tests.tools.gemma4_data_utils import gemma4_processor
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.trainer.base import BaseTrainer
    from veomni.trainer.vlm_trainer import VLMTrainer

    model = build_foundation_model(
        "tests/toy_config/gemma4_toy",
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=make_eager_ops_config(),
    )
    runtime = SimpleNamespace(
        unwrapped_module=model,
        model_config=model.config,
        processor=gemma4_processor(),
        chat_template=None,
        parallel_state=get_parallel_state(),
    )
    args = VeOmniVLMArguments(
        model=VLMMModelArguments(
            config_path="tests/toy_config/gemma4_toy", ops_implementation=make_eager_ops_config()
        ),
        data=VLMMDataArguments(train_path="dummy"),
    )
    monkeypatch.setattr(BaseTrainer, "_setup", staticmethod(lambda args: torch.device("cpu")))
    monkeypatch.setattr(VLMTrainer, "_build_model_runtime", lambda self: runtime)
    for name in (
        "_build_dataset",
        "_build_dataloader",
        "_build_lr_scheduler",
        "_build_training_context",
        "_init_callbacks",
    ):
        monkeypatch.setattr(BaseTrainer, name, lambda *args, **kwargs: None)
    trainer = VLMTrainer(args)
    feature = trainer.base.data_transform(
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
    )[0]
    batch = trainer.base.collate_fn([feature])
    assert batch["input_ids"].ndim == 2
    assert batch["cu_seq_lens_q"].tolist() == [0, len(feature["input_ids"])]
    output = model(**batch, use_cache=False)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    args.data.max_seq_len = 4
    trainer._build_data_transform()
    with pytest.raises(ValueError, match="max_seq_len"):
        trainer.base.data_transform(
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
        )


def test_gemma4_runtime_honors_audio_freezing_and_vision_learning_rate(monkeypatch):
    from veomni.models.model_runtime import VeOmniModelRuntime

    args = VeOmniVLMArguments(
        model=VLMMModelArguments(
            config_path="tests/toy_config/gemma4_toy", ops_implementation=make_eager_ops_config()
        ),
        data=VLMMDataArguments(train_path="dummy"),
    )
    args.train.freeze_audio_tower = True
    runtime = unbuilt_runtime(args.model, cls=VLMModelRuntime, train=args.train)
    runtime.model = build_foundation_model(
        args.model.config_path, init_device="meta", torch_dtype="float32", ops_implementation=make_eager_ops_config()
    )
    runtime.model_config = runtime.model.config
    runtime._freeze_model_module()
    assert not any(p.requires_grad for p in runtime.model.model.audio_tower.parameters())
    assert all(p.requires_grad for p in runtime.model.model.embed_audio.parameters())
    monkeypatch.setattr(VeOmniModelRuntime, "_build_optimizer", lambda self, param_groups: param_groups)
    groups = runtime._build_optimizer()
    visual = set(map(id, runtime.model.model.vision_tower.parameters()))
    visual.update(map(id, runtime.model.model.embed_vision.parameters()))
    assert {id(p) for group in groups if group["lr"] == args.train.vit_lr for p in group["params"]} == visual
