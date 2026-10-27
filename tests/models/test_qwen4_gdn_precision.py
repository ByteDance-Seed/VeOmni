"""Regression tests for Qwen4 GDN scalar precision and checkpoint export."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from torch import nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from scripts.merge_dcp_to_hf import save_model_weights as export_dcp
from veomni.checkpoint.dcp_checkpointer import (
    _QWEN4_GDN_FP32_LEGACY_MARKER,
    _QWEN4_GDN_FP32_MARKER,
    ModelState,
    _ModelStrictLoadPlanner,
)
from veomni.models.checkpoint.convert import parse_fqn_to_index_mapping_from_json
from veomni.models.checkpoint.weights import _dispatch_parameter, post_process_after_weight_loading, save_model_weights
from veomni.models.loader import CustomizedModelingLoader
from veomni.models.qwen4_gdn_precision import (
    is_qwen4_exp_config,
    prepare_qwen4_gdn_precision,
    validate_qwen4_gdn_pairs,
)
from veomni.utils.save_safetensor_utils import _save_hf_safetensor_legacy, get_model_save_state


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_hf_index_allows_named_auxiliary_shards(tmp_path):
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00131.safetensors",
                    "b": "model-00131-of-00131.safetensors",
                    "visual.weight": "model.visual.safetensors",
                    "visual.bias": "model.visual.safetensors",
                    "mtp.weight": "model.mtp.safetensors",
                }
            }
        )
    )
    mapping = parse_fqn_to_index_mapping_from_json(str(path))
    assert mapping["a"] == 1 and mapping["b"] == 131
    assert mapping["visual.weight"] == mapping["visual.bias"]
    assert len({mapping["a"], mapping["b"], mapping["visual.weight"], mapping["mtp.weight"]}) == 4
    assert min(mapping["visual.weight"], mapping["mtp.weight"]) > 131


def test_hf_index_does_not_merge_differently_named_numbered_files(tmp_path):
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "extra-00001-of-00002.safetensors",
                    "c": "model-00002-of-00002.safetensors",
                    "d": "model-00001-extra.safetensors",
                }
            }
        )
    )
    first = parse_fqn_to_index_mapping_from_json(str(path))
    assert len(set(first.values())) == 4
    assert first == {"a": 1, "b": 3, "c": 2, "d": 4}
    assert first == parse_fqn_to_index_mapping_from_json(str(path))


class Qwen4ExpTextGatedDeltaNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.A_log = nn.Parameter(torch.empty(4, device=device, dtype=torch.bfloat16))
        self.dt_bias = nn.Parameter(torch.empty(4, device=device, dtype=torch.bfloat16))
        self.proj = nn.Linear(4, 4, device=device, dtype=torch.bfloat16)


class ToyQwen4(nn.Module):
    def __init__(self, device="meta", model_type="qwen4_exp"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.linear_attn = Qwen4ExpTextGatedDeltaNet(device)


def test_distributed_export_rejects_gdn_parameters_missing_from_index():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = Qwen4ExpTextGatedDeltaNet("meta")

    class TwoLayerQwen4(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="qwen4_exp", text_config=SimpleNamespace(layer_types=["linear_attention"] * 2)
            )
            self.layers = nn.ModuleList([Layer(), Layer()])

    model = TwoLayerQwen4()
    prepare_qwen4_gdn_precision(model)
    model.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    state = get_model_save_state(model, None)
    complete = dict.fromkeys(state, 1)
    assert set(get_model_save_state(model, complete)) == set(state)
    incomplete_pair = {name: index for name, index in complete.items() if name != "layers.0.linear_attn.A_log"}
    with pytest.raises(RuntimeError, match="complete A_log/dt_bias pairs"):
        get_model_save_state(model, incomplete_pair)
    incomplete_layer = {name: index for name, index in complete.items() if not name.startswith("layers.0.")}
    with pytest.raises(RuntimeError, match="layer coverage differs"):
        get_model_save_state(model, incomplete_layer)
    model.config.text_config.layer_types = None
    with pytest.raises(RuntimeError, match="weight_map dropped Qwen4 GDN parameters"):
        get_model_save_state(model, incomplete_layer)


def _source_values():
    return torch.tensor([0.1234567, -0.456789, 0.00123456, 1.234567], dtype=torch.float32)


def _populated_model(model_type="qwen4_exp"):
    model = ToyQwen4(model_type=model_type)
    prepare_qwen4_gdn_precision(model)
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.linear_attn.A_log.copy_(_source_values())
        model.linear_attn.dt_bias.copy_(_source_values() + 0.01)
        model.linear_attn.proj.weight.fill_(0.125)
        model.linear_attn.proj.bias.zero_()
    return model


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_meta_preparation_preserves_loaded_fp32_values(model_type):
    model = ToyQwen4(model_type=model_type)
    names = dict(model.named_parameters())
    prepare_qwen4_gdn_precision(model)
    assert names["linear_attn.A_log"] is model.linear_attn.A_log
    assert model.linear_attn.A_log.dtype == model.linear_attn.dt_bias.dtype == torch.float32
    assert model.linear_attn.proj.weight.dtype == torch.bfloat16
    model.to_empty(device="cpu")
    source = _source_values()
    with torch.no_grad():
        _dispatch_parameter(model, "linear_attn.A_log", source)
        _dispatch_parameter(model, "linear_attn.dt_bias", source)
    assert torch.equal(model.linear_attn.A_log, source)
    assert not torch.equal(source.bfloat16().float(), source)


def test_late_promotion_rejected():
    with pytest.raises(RuntimeError, match="BF16 rounding has already occurred"):
        prepare_qwen4_gdn_precision(ToyQwen4(device="cpu"))


def test_materialized_scratch_model_reinitializes_scalars_in_fp32():
    model = ToyQwen4(device="cpu")
    prepare_qwen4_gdn_precision(model, initialize_materialized=True)
    assert model.linear_attn.A_log.dtype == model.linear_attn.dt_bias.dtype == torch.float32
    assert model.linear_attn.proj.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.linear_attn.dt_bias, torch.ones(4))
    assert torch.isfinite(model.linear_attn.A_log).all()


def test_customized_loader_scratch_initializes_fp32():
    class Factory:
        @staticmethod
        def _from_config(**kwargs):
            return ToyQwen4(device="cpu")

    model = CustomizedModelingLoader(Factory).load_model({}, init_device="cpu", weights_path=None)
    assert model.linear_attn.A_log.dtype == model.linear_attn.dt_bias.dtype == torch.float32
    assert model.linear_attn.proj.weight.dtype == torch.bfloat16


def test_customized_loader_prepares_meta_before_pretrained_load(monkeypatch):
    class Factory:
        @staticmethod
        def _from_config(**kwargs):
            return ToyQwen4(device="cpu")

    observed = []

    def fake_load(model, path, device):
        observed.append((model.linear_attn.A_log.device.type, model.linear_attn.A_log.dtype, path, device))

    monkeypatch.setattr("veomni.models.loader.load_model_weights", fake_load)
    CustomizedModelingLoader(Factory).load_model({}, init_device="cpu", weights_path="fp32-source")
    assert observed == [("meta", torch.float32, "fp32-source", "cpu")]


def test_hf_bf16_scalar_source_rejected():
    model = ToyQwen4()
    prepare_qwen4_gdn_precision(model)
    model.to_empty(device="cpu")
    with pytest.raises(RuntimeError, match="requires FP32 source and target"):
        _dispatch_parameter(model, "linear_attn.A_log", _source_values().bfloat16())


def test_hf_scalar_shape_mismatch_rejected():
    model = ToyQwen4()
    prepare_qwen4_gdn_precision(model)
    model.to_empty(device="cpu")
    with pytest.raises(RuntimeError, match="source and target shapes differ"):
        _dispatch_parameter(model, "linear_attn.A_log", torch.ones(1, dtype=torch.float32))


def test_hf_missing_scalar_source_rejected():
    model = ToyQwen4()
    prepare_qwen4_gdn_precision(model)
    model.to_empty(device="cpu")
    with pytest.raises(RuntimeError, match="missing required FP32 GDN parameters"):
        post_process_after_weight_loading(model, {}, {"linear_attn.dt_bias"})


@pytest.mark.parametrize("assets", [[SimpleNamespace(model_type="qwen4_exp")], None])
def test_hf_export_preserves_scalar_values_and_casts_other_weights(tmp_path, assets):
    source = _source_values()
    state = {
        "model.layers.0.linear_attn.A_log": source,
        "model.layers.0.linear_attn.dt_bias": source + 0.01,
        "model.layers.0.linear_attn.proj.weight": torch.ones(4, 4, dtype=torch.float32),
    }
    out = tmp_path / "hf"
    save_model_weights(str(out), state, model_assets=assets, qwen4_gdn_fp32=assets is None)
    exported = load_file(str(out / "model.safetensors"))
    for name in ("A_log", "dt_bias"):
        key = f"model.layers.0.linear_attn.{name}"
        assert exported[key].dtype == torch.float32
        assert torch.equal(exported[key], state[key])
    assert exported["model.layers.0.linear_attn.proj.weight"].dtype == torch.bfloat16


def test_explicit_other_model_assets_keep_original_export_policy(tmp_path):
    state = {"model.layers.0.linear_attn.A_log": _source_values()}
    save_model_weights(str(tmp_path), state, model_assets=[SimpleNamespace(model_type="qwen3_5")])
    assert load_file(str(tmp_path / "model.safetensors"))[next(iter(state))].dtype == torch.bfloat16


def test_qwen4_hf_export_rejects_old_bf16_scalars(tmp_path):
    state = {f"model.layers.0.linear_attn.{name}": _source_values().bfloat16() for name in ("A_log", "dt_bias")}
    with pytest.raises(RuntimeError, match="Cannot export non-FP32 Qwen4 GDN"):
        save_model_weights(
            str(tmp_path), state, save_dtype="float32", model_assets=[SimpleNamespace(model_type="qwen4_exp")]
        )


def test_unknown_model_keeps_original_export_policy(tmp_path):
    state = {"model.layers.0.linear_attn.A_log": _source_values().bfloat16()}
    save_model_weights(str(tmp_path), state, save_dtype="float32")
    assert load_file(str(tmp_path / "model.safetensors"))[next(iter(state))].dtype == torch.float32


@pytest.mark.parametrize("assets", [[SimpleNamespace(model_type="qwen4_exp")], None])
def test_dcp_to_hf_export_preserves_fp32(tmp_path, assets):
    state = {
        "model.layers.0.linear_attn.A_log": _source_values(),
        "model.layers.0.linear_attn.dt_bias": _source_values() + 0.01,
        "model.layers.0.linear_attn.proj.weight": torch.ones(4, 4, dtype=torch.float32),
    }
    manifest = json.dumps(
        {
            "version": 1,
            "parameter_names": sorted(
                name for name in state if ".linear_attn." in name and name.endswith(("A_log", "dt_bias"))
            ),
        }
    )
    dcp.save({"model": {**state, _QWEN4_GDN_FP32_MARKER: manifest}}, checkpoint_id=tmp_path / "dcp")
    export_dcp(str(tmp_path / "hf"), str(tmp_path / "dcp"), model_assets=assets)
    exported = load_file(str(tmp_path / "hf" / "model.safetensors"))
    for name in ("A_log", "dt_bias"):
        key = f"model.layers.0.linear_attn.{name}"
        assert exported[key].dtype == torch.float32
        assert torch.equal(exported[key], state[key])
    assert exported["model.layers.0.linear_attn.proj.weight"].dtype == torch.bfloat16


def test_old_bf16_dcp_rejected_before_restore(tmp_path):
    key = "model.layers.0.linear_attn.A_log"
    dcp.save({"model": {key: _source_values().bfloat16()}}, checkpoint_id=tmp_path / "dcp")
    with pytest.raises(BaseException, match="GDN parameter.*not FP32"):
        dcp.load(
            {"model": {key: torch.empty(4, dtype=torch.float32)}},
            checkpoint_id=tmp_path / "dcp",
            planner=_ModelStrictLoadPlanner(strict_model=True, strict_gdn_fp32=True),
        )


def test_fp32_dcp_rejects_bf16_target(tmp_path):
    key = "model.layers.0.linear_attn.A_log"
    dcp.save({"model": {key: _source_values()}}, checkpoint_id=tmp_path / "dcp")
    with pytest.raises(BaseException, match="target.*must be FP32"):
        dcp.load(
            {"model": {key: torch.empty(4, dtype=torch.bfloat16)}},
            checkpoint_id=tmp_path / "dcp",
            planner=_ModelStrictLoadPlanner(strict_model=True, strict_gdn_fp32=True),
        )


def test_dcp_to_hf_rejects_old_bf16_scalars(tmp_path):
    state = {f"model.layers.0.linear_attn.{name}": _source_values().bfloat16() for name in ("A_log", "dt_bias")}
    dcp.save({"model": state}, checkpoint_id=tmp_path / "dcp")
    with pytest.raises(RuntimeError, match="Cannot export non-FP32 Qwen4 GDN"):
        export_dcp(
            str(tmp_path / "hf"),
            str(tmp_path / "dcp"),
            save_dtype="float32",
            model_assets=[SimpleNamespace(model_type="qwen4_exp")],
        )


def test_legacy_fp32_dcp_can_export_with_explicit_model_policy(tmp_path):
    source = _source_values()
    state = {f"model.layers.0.linear_attn.{name}": source for name in ("A_log", "dt_bias")}
    dcp.save({"model": state}, checkpoint_id=tmp_path / "dcp")
    export_dcp(str(tmp_path / "hf"), str(tmp_path / "dcp"), qwen4_gdn_fp32=True)
    actual = load_file(str(tmp_path / "hf" / "model.safetensors"))
    for key in state:
        assert actual[key].dtype == torch.float32
        assert torch.equal(actual[key], source)


def test_qwen4_hf_export_rejects_missing_scalar(tmp_path):
    state = {"model.layers.0.linear_attn.A_log": _source_values()}
    with pytest.raises(RuntimeError, match="complete A_log/dt_bias pairs"):
        save_model_weights(str(tmp_path), state, model_assets=[SimpleNamespace(model_type="qwen4_exp")])


def test_qwen4_dcp_export_rejects_missing_scalar(tmp_path):
    state = {"model.layers.0.linear_attn.A_log": _source_values()}
    dcp.save({"model": state}, checkpoint_id=tmp_path / "dcp")
    with pytest.raises(RuntimeError, match="complete A_log/dt_bias pairs"):
        export_dcp(str(tmp_path / "hf"), str(tmp_path / "dcp"), qwen4_gdn_fp32=True)


def test_qwen4_export_checks_config_layer_coverage(tmp_path):
    state = {f"model.layers.0.linear_attn.{name}": _source_values() for name in ("A_log", "dt_bias")}
    config = SimpleNamespace(
        model_type="qwen4_exp", text_config=SimpleNamespace(layer_types=["linear_attention", "linear_attention"])
    )
    with pytest.raises(RuntimeError, match="GDN layer coverage differs"):
        save_model_weights(str(tmp_path), state, model_assets=[config])


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_modelstate_marker_preserves_both_hf_export_paths(tmp_path, model_type):
    model = _populated_model(model_type)
    wrapper = ModelState(model, parallel_state=SimpleNamespace(dp_mode="fsdp2"))
    manifest = json.loads(wrapper.state_dict()[_QWEN4_GDN_FP32_MARKER])
    assert set(manifest["parameter_names"]) == {"linear_attn.A_log", "linear_attn.dt_bias"}
    dcp.save({"model": wrapper}, checkpoint_id=tmp_path / "dcp")
    export_dcp(str(tmp_path / "hf-offline"), str(tmp_path / "dcp"))
    _save_hf_safetensor_legacy(str(tmp_path / "dcp"), str(tmp_path / "hf-legacy"), None, "dcp", None)
    for directory in ("hf-offline", "hf-legacy"):
        actual = load_file(str(tmp_path / directory / "model.safetensors"))
        for name in ("A_log", "dt_bias"):
            key = f"linear_attn.{name}"
            assert actual[key].dtype == torch.float32
            assert torch.equal(actual[key], getattr(model.linear_attn, name))
        assert _QWEN4_GDN_FP32_MARKER not in actual


def test_qwen4_sparse_only_text_has_no_gdn_marker_and_exports(tmp_path):
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp_text", layer_types=["qwen_sparse_attention"])
    model.proj = nn.Linear(4, 4, dtype=torch.bfloat16)
    prepare_qwen4_gdn_precision(model)
    state = ModelState(model, parallel_state=SimpleNamespace(dp_mode="ddp")).state_dict()
    assert _QWEN4_GDN_FP32_MARKER not in state
    assert set(get_model_save_state(model, None)) == set(state)
    dcp.save({"model": state}, checkpoint_id=tmp_path / "dcp")
    _save_hf_safetensor_legacy(str(tmp_path / "dcp"), str(tmp_path / "hf"), None, "dcp", None, model=model)
    assert set(load_file(str(tmp_path / "hf" / "model.safetensors"))) == set(state)
    validate_qwen4_gdn_pairs([], model.config)
    with pytest.raises(RuntimeError, match="complete A_log/dt_bias"):
        validate_qwen4_gdn_pairs([])


def test_real_qwen4_text_config_is_recognized():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json").text_config
    assert config.model_type == "qwen4_exp_text"
    assert is_qwen4_exp_config(config)


def test_zero_gdn_model_rejects_legacy_checkpoint_with_gdn(tmp_path):
    state = {f"model.layers.0.linear_attn.{name}": _source_values() for name in ("A_log", "dt_bias")}
    dcp.save({"model": state}, checkpoint_id=tmp_path / "dcp")
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp_text", layer_types=["qwen_sparse_attention"])
    with pytest.raises(RuntimeError, match="declares no GDN"):
        _save_hf_safetensor_legacy(str(tmp_path / "dcp"), str(tmp_path / "hf"), None, "dcp", None, model=model)


def test_legacy_v1_gdn_export_accepts_text_config_assets(tmp_path):
    state = {f"model.layers.0.linear_attn.{name}": _source_values() for name in ("A_log", "dt_bias")}
    dcp.save({"model": {**state, _QWEN4_GDN_FP32_LEGACY_MARKER: True}}, checkpoint_id=tmp_path / "dcp")
    assets = [SimpleNamespace(model_type="qwen4_exp_text")]
    _save_hf_safetensor_legacy(str(tmp_path / "dcp"), str(tmp_path / "hf"), assets, "dcp", None)
    actual = load_file(str(tmp_path / "hf" / "model.safetensors"))
    assert all(
        actual[name].dtype == torch.float32 and torch.equal(actual[name], value) for name, value in state.items()
    )


def test_legacy_unmarked_fp32_dcp_restores_via_modelstate(tmp_path):
    source = _populated_model()
    dcp.save({"model": source.state_dict()}, checkpoint_id=tmp_path / "dcp")
    target = _populated_model()
    with torch.no_grad():
        target.linear_attn.A_log.zero_()
        target.linear_attn.dt_bias.zero_()
    wrapper = ModelState(target, parallel_state=SimpleNamespace(dp_mode="fsdp2"))
    dcp.load(
        {"model": wrapper},
        checkpoint_id=tmp_path / "dcp",
        planner=_ModelStrictLoadPlanner(strict_model=True, strict_gdn_fp32=True),
    )
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], expected, rtol=0, atol=0)


def test_modelstate_recognizes_ddp_wrapped_qwen4(tmp_path):
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'process-group'}", rank=0, world_size=1)
    try:
        model = torch.nn.parallel.DistributedDataParallel(_populated_model())
        wrapper = ModelState(model, parallel_state=SimpleNamespace(dp_mode="ddp"))
        assert json.loads(wrapper.state_dict()[_QWEN4_GDN_FP32_MARKER])["version"] == 1
    finally:
        dist.destroy_process_group()


def test_dcp_inventory_rejects_whole_missing_gdn_layer(tmp_path):
    state = {f"model.layers.0.linear_attn.{name}": _source_values() for name in ("A_log", "dt_bias")}
    names = list(state) + [f"model.layers.1.linear_attn.{name}" for name in ("A_log", "dt_bias")]
    dcp.save(
        {"model": {**state, _QWEN4_GDN_FP32_MARKER: json.dumps({"version": 1, "parameter_names": names})}},
        checkpoint_id=tmp_path / "dcp",
    )
    with pytest.raises(RuntimeError, match="inventory does not match"):
        export_dcp(str(tmp_path / "hf"), str(tmp_path / "dcp"))


def test_legacy_hf_export_uses_model_config_coverage(tmp_path):
    state = {f"model.layers.0.linear_attn.{name}": _source_values() for name in ("A_log", "dt_bias")}
    dcp.save({"model": state}, checkpoint_id=tmp_path / "dcp")
    model = ToyQwen4(device="cpu")
    model.config = SimpleNamespace(
        model_type="qwen4_exp", text_config=SimpleNamespace(layer_types=["linear_attention", "linear_attention"])
    )
    with pytest.raises(RuntimeError, match="GDN layer coverage differs"):
        _save_hf_safetensor_legacy(str(tmp_path / "dcp"), str(tmp_path / "hf"), None, "dcp", None, model=model)


def test_old_boolean_marker_requires_explicit_policy(tmp_path):
    state = {f"model.layers.0.linear_attn.{name}": _source_values() for name in ("A_log", "dt_bias")}
    dcp.save({"model": {**state, _QWEN4_GDN_FP32_LEGACY_MARKER: True}}, checkpoint_id=tmp_path / "dcp")
    with pytest.raises(RuntimeError, match="Legacy Qwen4 GDN marker"):
        export_dcp(str(tmp_path / "hf-rejected"), str(tmp_path / "dcp"))
    export_dcp(str(tmp_path / "hf"), str(tmp_path / "dcp"), qwen4_gdn_fp32=True)
    actual = load_file(str(tmp_path / "hf" / "model.safetensors"))
    assert all(torch.equal(actual[key], tensor) for key, tensor in state.items())
