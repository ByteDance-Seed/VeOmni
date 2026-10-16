"""DeepSeek-V4.1 straight off the release, through its ``OmniHFLayout``.

The offline splitter is the reference: a module read from the release in place
must hold exactly what the same module holds when read from the split
checkpoint, in both the bf16 training build and the quantized inference build.
And an export from that module must write the release's own keys and dtypes,
with values that dequantize to what was loaded.

The release is the miniature from ``test_deepseek_v41_convert``.
"""

from __future__ import annotations

import json
import os
import re
import types

import pytest
import torch
from safetensors import safe_open

import veomni.models.module_utils as module_utils
from veomni.models.module_utils import init_empty_weights, load_model_weights, load_model_weights_ep_sharded
from veomni.models.seed_omni.modules.deepseek_v41.checkpoint_tensor_converter import (
    DeepseekV41LlmConverter,
    dequant_converted_weight,
)
from veomni.models.seed_omni.modules.deepseek_v41.convert_model import convert_deepseek_v41_checkpoint
from veomni.models.seed_omni.modules.deepseek_v41.dequant import dequant_fp4_blockwise, dequant_fp8_blockwise
from veomni.models.seed_omni.modules.deepseek_v41.hf_layout import build_deepseek_v41_hf_layout
from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Llm
from veomni.models.seed_omni.modules.deepseek_v41.released_names import LLM, MODULES, NGRAM
from veomni.models.seed_omni.utils.hf_layout import (
    HFSource,
    attach_hf_source_converter,
    load_module_from_hf_source,
    save_hf_source_checkpoint,
)

from .test_deepseek_v41_convert import (
    ENGRAM_LAYERS,
    N_EXPERTS,
    N_LAYERS,
    _write_released_config,
    build_released_checkpoint,
    llm_config,
)


QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


#: Fewer draft-stage experts than main-layer ones, as in the release (128 vs 384),
#: so a rank's two expert ranges differ.
N_DRAFT_EXPERTS = N_EXPERTS // 2


def _config():
    config = llm_config()
    config.dspark_n_routed_experts = N_DRAFT_EXPERTS
    return config


def _release(path: str, *, quantized: bool) -> dict[str, torch.Tensor]:
    config = _config()
    tensors = build_released_checkpoint(path, config)
    _write_released_config(path)
    with open(os.path.join(path, "config.json")) as f:
        released = json.load(f)
    # The llm half of the engram fields must describe the tensors just built.
    released["text_config"].update(
        dspark_n_routed_experts=N_DRAFT_EXPERTS,
        engram_head_dim=config.engram_head_dim,
        engram_max_ngram_size=config.engram_max_ngram_size,
    )
    if quantized:
        released["quantization_config"] = QUANTIZATION
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(released, f)
    return tensors


def _read(path: str) -> dict[str, torch.Tensor]:
    tensors = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(path, name), framework="pt") as f:
                tensors.update({key: f.get_tensor(key) for key in f.keys()})
    return tensors


def _load_split_llm(release: str, out: str):
    convert_deepseek_v41_checkpoint(release, out, modules=[LLM])
    module_dir = os.path.join(out, LLM)
    config = DeepseekV41Llm.config_class.from_pretrained(module_dir)
    with init_empty_weights():
        model = DeepseekV41Llm._from_config(config)
    load_model_weights(model, module_dir, init_device="cpu")
    return model


def _load_direct_llm(release: str):
    layout = build_deepseek_v41_hf_layout(release)
    config = layout.build_module_configs(release)[LLM]
    source = HFSource(path=os.path.realpath(release), model_type="deepseek_v41")
    return load_module_from_hf_source(DeepseekV41Llm, config, source, LLM, device_map="cpu")


@pytest.fixture(autouse=True)
def _miniature_is_the_release(monkeypatch):
    """Any other depth reads as a shrunk stack, which remaps the release's layer ids."""
    from veomni.models.seed_omni.modules.deepseek_v41 import checkpoint_tensor_converter, export

    for module in (checkpoint_tensor_converter, export):
        monkeypatch.setattr(module, "RELEASED_NUM_HIDDEN_LAYERS", N_LAYERS)
        monkeypatch.setattr(module, "RELEASED_ENGRAM_LAYER_IDS", list(ENGRAM_LAYERS), raising=False)


@pytest.fixture(params=[False, True], ids=["train", "quantized"])
def release(request, tmp_path):
    path = str(tmp_path / "release")
    return path, _release(path, quantized=request.param), request.param


class TestTheLayout:
    def test_every_released_tensor_lands_on_one_module(self, release):
        path, tensors, _ = release
        layout = build_deepseek_v41_hf_layout(path)
        assert set(layout.modules) == set(MODULES)
        for key in tensors:
            routed = layout.route(key)
            assert routed is not None, key
            assert layout.source_key(*routed) == key

    def test_the_prefixes_follow_the_config(self, release):
        """The renames are listed per layer, so they must cover exactly the release's layers."""
        path, _, _ = release
        layout = build_deepseek_v41_hf_layout(path)
        n_layers = llm_config().num_hidden_layers
        assert layout.route(f"layers.{n_layers - 1}.attn_norm.weight") is not None
        assert layout.route(f"layers.{n_layers}.attn_norm.weight") is None

    def test_a_scale_lands_beside_its_weight(self, release):
        path, _, _ = release
        layout = build_deepseek_v41_hf_layout(path)
        assert layout.route("layers.0.attn.wq_a.scale") == (LLM, "model.layers.0.self_attn.wq_a.weight_scale")
        assert layout.route("layers.0.ffn.experts.3.w1.scale") == (LLM, "model.layers.0.mlp.experts.3.w1.scale")


class TestDirectLoadMatchesTheSplit:
    def test_every_llm_tensor_is_identical(self, release, tmp_path):
        path, _, quantized = release
        split = _load_split_llm(path, str(tmp_path / "split")).state_dict()
        direct = _load_direct_llm(path).state_dict()
        assert split.keys() == direct.keys()
        for key in split:
            assert split[key].dtype == direct[key].dtype, key
            assert torch.equal(split[key].view(torch.uint8), direct[key].view(torch.uint8)), key
        experts = direct["model.layers.0.mlp.experts.gate_up_proj"]
        assert experts.dtype == (torch.int8 if quantized else torch.float32)

    def test_the_load_holds_the_released_values(self, release):
        """Equal to the split is only meaningful if neither side fell back to its init."""
        path, tensors, quantized = release
        direct = _load_direct_llm(path).state_dict()
        dense = "model.layers.0.self_attn.wq_a.weight"
        experts = "model.layers.0.mlp.experts.gate_up_proj"
        w1 = [_dequant_release(tensors, f"layers.0.ffn.experts.{e}.w1.weight") for e in range(N_EXPERTS)]
        w3 = [_dequant_release(tensors, f"layers.0.ffn.experts.{e}.w3.weight") for e in range(N_EXPERTS)]
        want_experts = torch.stack([torch.cat(pair) for pair in zip(w1, w3, strict=True)]).bfloat16().float()
        if quantized:
            released = tensors["layers.0.attn.wq_a.weight"]
            assert torch.equal(direct[dense].view(torch.uint8), released.view(torch.uint8))
            loaded_experts = dequant_converted_weight(direct[experts], direct[f"{experts}_scale"]).float()
        else:
            want_dense = _dequant_release(tensors, "layers.0.attn.wq_a.weight").bfloat16().float()
            torch.testing.assert_close(direct[dense], want_dense, rtol=0, atol=0)
            loaded_experts = direct[experts]
        torch.testing.assert_close(loaded_experts, want_experts, rtol=0, atol=0)


class TestExpertRanges:
    def _converter(self, quantized: bool) -> DeepseekV41LlmConverter:
        config = _config()
        if quantized:
            config.weight_quantization = dict(QUANTIZATION)
        return DeepseekV41LlmConverter(config, dequantize=not quantized)

    @pytest.mark.parametrize("quantized", [False, True], ids=["train", "quantized"])
    def test_each_range_stacks_only_its_experts(self, release, quantized):
        path, tensors, _ = release
        layout = build_deepseek_v41_hf_layout(path)
        full = self._converter(quantized)
        prefix = "model.layers.0.mlp.experts"
        module_keys = {layout.route(key)[1]: key for key in tensors if ".ffn.experts." in key}
        layer0 = {k: v for k, v in module_keys.items() if k.startswith(f"{prefix}.")}

        halves = []
        for start in range(0, N_EXPERTS, N_EXPERTS // 2):
            ranged = full.for_expert_range(start, N_EXPERTS // 2)
            emitted = {}
            for module_key, key in sorted(layer0.items()):
                target, expert = full.fused_expert_target(module_key)
                if not start <= expert < start + N_EXPERTS // 2:
                    continue
                result = ranged.convert(module_key, tensors[key])
                if result is not None:
                    emitted[result.name] = result.tensor
            assert ranged.finalize() == []
            halves.append(emitted)

        expect = {}
        for module_key, key in sorted(layer0.items()):
            result = full.convert(module_key, tensors[key])
            if result is not None:
                expect[result.name] = result.tensor
        assert set(halves[0]) == set(expect)
        for name, tensor in expect.items():
            assert torch.equal(
                torch.cat([halves[0][name], halves[1][name]]).view(torch.uint8), tensor.view(torch.uint8)
            )

    def test_a_scale_targets_its_own_slab_only_when_quantized(self):
        name = "model.layers.0.mlp.experts.3.w1.scale"
        assert self._converter(False).fused_expert_target(name) == ("model.layers.0.mlp.experts.gate_up_proj", 3)
        assert self._converter(True).fused_expert_target(name) == ("model.layers.0.mlp.experts.gate_up_proj_scale", 3)

    def test_a_missing_expert_is_refused(self, release):
        path, tensors, _ = release
        layout = build_deepseek_v41_hf_layout(path)
        converter = self._converter(False)
        for key in tensors:
            if key.startswith("layers.0.ffn.experts.0."):
                converter.convert(layout.route(key)[1], tensors[key])
        with pytest.raises(RuntimeError, match="incomplete checkpoint"):
            converter.finalize()


EP_SIZE = 2
_EXPERT_PARAM = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)(_scale)?$")


def _stream_as_rank(path: str, monkeypatch, ep_rank: int) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Stream the llm off the release as EP rank ``ep_rank``; return its state and the keys it read."""
    from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import DeepseekV41LlmAccelerated

    layout = build_deepseek_v41_hf_layout(path)
    config = layout.build_module_configs(path)[LLM]
    with init_empty_weights():
        model = DeepseekV41Llm._from_config(config)
    # The rank-local expert slabs EP wrapping would leave on this rank.
    for name, param in list(model.named_parameters()):
        if _EXPERT_PARAM.search(name):
            owner, attr = model.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]
            local = torch.empty((param.shape[0] // EP_SIZE, *param.shape[1:]), dtype=param.dtype, device="meta")
            setattr(owner, attr, torch.nn.Parameter(local, requires_grad=param.requires_grad))
    model.get_parallel_plan = types.MethodType(DeepseekV41LlmAccelerated.get_parallel_plan, model)
    attach_hf_source_converter(model, HFSource(path=os.path.realpath(path), model_type="deepseek_v41"), LLM)

    monkeypatch.setattr(
        module_utils, "_ep_dim0_slice_meta", lambda _state, _group, target0: (EP_SIZE, ep_rank, target0 * EP_SIZE)
    )
    read = []

    class _Recording:
        def __init__(self, *args, **kwargs):
            self._handle = safe_open(*args, **kwargs)

        def __enter__(self):
            handle = self._handle.__enter__()

            class _Proxy:
                def get_tensor(self, key):
                    read.append(key)
                    return handle.get_tensor(key)

                def __getattr__(self, attr):
                    return getattr(handle, attr)

            return _Proxy()

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    monkeypatch.setattr(module_utils, "safe_open", _Recording)
    load_model_weights_ep_sharded(model, path, "cpu")
    return {name: tensor.detach() for name, tensor in model.state_dict().items()}, read


class TestStreamLoad:
    def test_each_rank_streams_only_its_experts_of_both_ranges(self, release, monkeypatch):
        """Main layers hold 4 experts and draft stages 2, so a rank has two expert ranges."""
        path, _, _ = release
        whole = _load_direct_llm(path).state_dict()
        ranks = [_stream_as_rank(path, monkeypatch, rank) for rank in range(EP_SIZE)]

        for key, value in whole.items():
            if _EXPERT_PARAM.search(key):
                streamed = torch.cat([state[key] for state, _ in ranks])
            else:
                streamed = ranks[1][0][key]
            assert streamed.dtype == value.dtype, key
            assert torch.equal(streamed.view(torch.uint8), value.view(torch.uint8)), key

        for rank, (_, read) in enumerate(ranks):
            for kind, count in (("layers", N_EXPERTS), ("mtp", N_DRAFT_EXPERTS)):
                local = count // EP_SIZE
                experts = {
                    int(key.split(".")[4]) for key in read if key.startswith(f"{kind}.") and ".ffn.experts." in key
                }
                assert experts == set(range(rank * local, (rank + 1) * local)), (rank, kind)


def _add_ngram_tables(path: str, ngram_config) -> dict[str, torch.Tensor]:
    """The release's fp8 tables and row scales, in a shard of their own; the llm miniature has none."""
    from safetensors.torch import save_file

    from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram

    from .test_deepseek_v41_ngram_quantized import quantize_rows

    with init_empty_weights():
        width = DeepseekV41Ngram._from_config(ngram_config).tables[0].weight.shape[1]
    torch.manual_seed(1)
    tables = {}
    for layer_id, rows in zip(ngram_config.layer_ids, ngram_config.num_embeddings):
        packed, scale = quantize_rows(torch.randn(rows, width, dtype=torch.bfloat16))
        tables[f"layers.{layer_id}.engram.embed.weight"] = packed
        tables[f"layers.{layer_id}.engram.embed.scale"] = scale
    save_file(tables, os.path.join(path, "model-ngram.safetensors"))
    index_path = os.path.join(path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    index["weight_map"].update(dict.fromkeys(tables, "model-ngram.safetensors"))
    with open(index_path, "w") as f:
        json.dump(index, f)
    return tables


def _stream_ngram_as_rank(path: str, monkeypatch, rank: int) -> tuple[dict[str, torch.Tensor], list[str]]:
    from veomni.models.seed_omni.modules.deepseek_v41.ngram.accelerated.accelerated import DeepseekV41NgramAccelerated
    from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram

    config = build_deepseek_v41_hf_layout(path).build_module_configs(path)[NGRAM]
    with init_empty_weights():
        model = DeepseekV41Ngram._from_config(config)
    for table in model.tables:
        for attr in ("weight", "weight_scale"):
            param = getattr(table, attr, None)
            if isinstance(param, torch.nn.Parameter):
                local = torch.empty((param.shape[0] // EP_SIZE, *param.shape[1:]), dtype=param.dtype, device="meta")
                setattr(table, attr, torch.nn.Parameter(local, requires_grad=param.requires_grad))
    model.get_parallel_plan = types.MethodType(DeepseekV41NgramAccelerated.get_parallel_plan, model)
    attach_hf_source_converter(model, HFSource(path=os.path.realpath(path), model_type="deepseek_v41"), NGRAM)
    monkeypatch.setattr(
        module_utils, "_ep_dim0_slice_meta", lambda _state, _group, target0: (EP_SIZE, rank, target0 * EP_SIZE)
    )
    read = []
    real_safe_open = module_utils.safe_open

    class _Recording:
        def __init__(self, *args, **kwargs):
            self._handle = real_safe_open(*args, **kwargs)

        def __enter__(self):
            handle = self._handle.__enter__()

            class _Proxy:
                def get_tensor(self, key):
                    read.append(key)
                    return handle.get_tensor(key)

                def __getattr__(self, attr):
                    return getattr(handle, attr)

            return _Proxy()

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    monkeypatch.setattr(module_utils, "safe_open", _Recording)
    load_model_weights_ep_sharded(model, path, "cpu")
    monkeypatch.setattr(module_utils, "safe_open", real_safe_open)
    return {name: tensor.detach() for name, tensor in model.state_dict().items()}, read


class TestNgramStreamLoad:
    def test_each_rank_streams_its_table_rows_and_gets_the_token_map(self, release, monkeypatch):
        """The tables shard by row with their scales read beside them; ``token_map`` is built, not read."""
        from veomni.models.seed_omni.modules.deepseek_v41 import hf_layout as v41_hf_layout
        from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram

        path, _, _ = release
        with open(os.path.join(path, "config.json")) as f:
            released = json.load(f)
        # Rows above the 6 x hash-vocab floor the module checks; two ranks straddle no padding.
        released["text_config"].update(
            engram_vocab_size=64, engram_num_embeddings=[1024] * len(ENGRAM_LAYERS), engram_compressed_vocab_size=40
        )
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(released, f)
        config = build_deepseek_v41_hf_layout(path).build_module_configs(path)[NGRAM]
        tables = _add_ngram_tables(path, config)
        token_map = torch.arange(config.vocab_size, dtype=torch.int64) % config.compressed_vocab_size
        monkeypatch.setattr(v41_hf_layout, "build_ngram_token_map", lambda _path, _config: token_map.clone())

        source = HFSource(path=os.path.realpath(path), model_type="deepseek_v41")
        whole = load_module_from_hf_source(DeepseekV41Ngram, config, source, NGRAM, device_map="cpu").state_dict()
        ranks = [_stream_ngram_as_rank(path, monkeypatch, rank) for rank in range(EP_SIZE)]

        assert torch.equal(whole["token_map"], token_map)
        for key, value in whole.items():
            if key.startswith("tables."):
                streamed = torch.cat([state[key] for state, _ in ranks])
            else:
                streamed = ranks[1][0][key]
            assert streamed.dtype == value.dtype, key
            assert torch.equal(streamed.view(torch.uint8), value.view(torch.uint8)), key
        for _, read in ranks:
            assert not set(read) & set(tables), read


def _dequant_release(tensors: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    weight = tensors[key]
    scale = tensors.get(key[: -len(".weight")] + ".scale") if key.endswith(".weight") else None
    if scale is None:
        return weight.float()
    if weight.dtype == torch.int8:
        return dequant_fp4_blockwise(weight, scale, out_dtype=torch.float32)
    return dequant_fp8_blockwise(weight, scale, out_dtype=torch.float32)


class TestExport:
    def test_a_trained_llm_exports_in_the_release_formats(self, release, tmp_path):
        """Keys and dtypes are the release's; values dequantize to what was loaded."""
        path, tensors, quantized = release
        if quantized:
            pytest.skip("the quantized build is inference-only")
        llm = _load_direct_llm(path)
        source = HFSource(path=os.path.realpath(path), model_type="deepseek_v41")
        out = tmp_path / "export"
        save_hf_source_checkpoint(source, {LLM: (llm, None)}, str(out))

        exported = _read(str(out))
        assert exported.keys() == tensors.keys()
        for key, value in tensors.items():
            assert exported[key].dtype == value.dtype, key
            if key.endswith(".scale"):
                continue
            torch.testing.assert_close(_dequant_release(exported, key), _dequant_release(tensors, key), rtol=0, atol=0)
        with (
            open(out / "model.safetensors.index.json") as f,
            open(os.path.join(path, "model.safetensors.index.json")) as g,
        ):
            assert json.load(f)["weight_map"] == json.load(g)["weight_map"]

    def test_a_shrunk_llm_is_not_exported_as_the_release(self, release, tmp_path):
        path, _, quantized = release
        if quantized:
            pytest.skip("the quantized build is inference-only")
        llm = _load_direct_llm(path)
        llm.config.num_hidden_layers = 1
        source = HFSource(path=os.path.realpath(path), model_type="deepseek_v41")
        with pytest.raises(NotImplementedError, match="layers"):
            save_hf_source_checkpoint(source, {LLM: (llm, None)}, str(tmp_path / "export"))

    def test_experts_split_on_dim0_by_two_mesh_dims_are_refused(self):
        """Muon zero-comm restores (ep_fsdp, ep) both on dim 0, block-transposing the global expert order."""
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard
        from torch.testing._internal.distributed.fake_pg import FakeStore

        from veomni.models.seed_omni.modules.deepseek_v41.export import _local_experts

        dist.init_process_group("fake", rank=0, world_size=4, store=FakeStore())
        try:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
            experts = DTensor.from_local(torch.zeros(2, 4, 4), mesh, [Shard(0), Shard(0)], run_check=False)
            with pytest.raises(NotImplementedError, match="several mesh dims"):
                _local_experts(experts, 0)
        finally:
            dist.destroy_process_group()


def test_dequantized_experts_match_the_split_converter_math(release):
    """Stacking after dequant equals dequant after stacking, which is what the split path does."""
    path, tensors, quantized = release
    if quantized:
        pytest.skip("covers the training build")
    w1 = [tensors[f"layers.0.ffn.experts.{e}.w1.weight"] for e in range(N_EXPERTS)]
    s1 = [tensors[f"layers.0.ffn.experts.{e}.w1.scale"] for e in range(N_EXPERTS)]
    stacked = dequant_converted_weight(torch.stack(w1), torch.stack(s1))
    per_expert = torch.stack([dequant_converted_weight(w, s) for w, s in zip(w1, s1, strict=True)])
    assert torch.equal(stacked, per_expert)
