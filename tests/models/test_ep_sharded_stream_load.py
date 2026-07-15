from types import SimpleNamespace

import torch
import torch.nn as nn

from veomni.distributed import parallel_plan as parallel_plan_module
from veomni.lora import weight_loading
from veomni.models import module_utils


class _WrappedProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})


class _LoraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.proj = _WrappedProjection()

    def get_parallel_plan(self):
        raise AssertionError("the runtime parallel-plan helper should be used")

    def get_base_model(self):
        return self.base_model.model


class _DensePlan:
    extra_parallel_plan = {"ep": {}}

    def __init__(self):
        self.dispatched = []

    def _get_shard_parameter_groupname(self, name):
        return None

    def shard_tensor(self, tensor, name, shape):
        self.dispatched.append(name)
        return tensor


def test_ep_stream_loader_maps_base_weights_and_forwards_lora_adapter(monkeypatch):
    model = _LoraModel()
    plan = _DensePlan()
    checkpoint_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    observed = {}

    class _SafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get_tensor(self, name):
            assert name == "proj.weight"
            return checkpoint_weight

    monkeypatch.setattr(module_utils, "safe_open", lambda *args, **kwargs: _SafeOpen())
    monkeypatch.setattr(
        module_utils,
        "_resolve_safetensors_shards",
        lambda *args, **kwargs: ({"proj.weight": "model.safetensors"}, {"model.safetensors": "unused"}),
    )
    monkeypatch.setattr(module_utils, "get_parallel_state", lambda: SimpleNamespace())
    monkeypatch.setattr(module_utils, "get_checkpoint_tensor_converter", lambda model: None)
    monkeypatch.setattr(module_utils, "empty_cache", lambda: None)
    monkeypatch.setattr(parallel_plan_module, "get_runtime_parallel_plan", lambda model: plan)

    def _load_adapter(model, path, init_device, dtensor_factory, parameter_names_to_load, parallel_plan):
        observed["adapter_path"] = path
        observed["adapter_missing"] = set(parameter_names_to_load)
        observed["adapter_plan"] = parallel_plan
        parameter_names_to_load.clear()

    def _post_process(model, buffers, parameter_names_to_load, dtensor_factory, dtensor_to_cpu=False):
        observed["post_missing"] = set(parameter_names_to_load)

    monkeypatch.setattr(weight_loading, "load_lora_weights", _load_adapter)
    monkeypatch.setattr(module_utils, "post_process_after_weight_loading", _post_process)

    module_utils.load_model_weights_ep_sharded(
        model,
        "unused",
        init_device="cpu",
        is_peft_model=True,
        adapter_path="adapter",
    )

    torch.testing.assert_close(model.base_model.model.proj.base_layer.weight, checkpoint_weight)
    assert plan.dispatched == ["base_model.model.proj.base_layer.weight"]
    assert observed["adapter_path"] == "adapter"
    assert observed["adapter_plan"] is plan
    assert observed["adapter_missing"] == {
        "base_model.model.proj.lora_A.default.weight",
        "base_model.model.proj.lora_B.default.weight",
    }
    assert observed["post_missing"] == set()
