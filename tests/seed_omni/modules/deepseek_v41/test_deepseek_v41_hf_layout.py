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
from safetensors.torch import save_file

import veomni.models.checkpoint.weights as module_utils
from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.checkpoint.weights import init_empty_weights, load_model_weights, load_model_weights_ep_sharded
from veomni.models.seed_omni.modules.deepseek_v41.checkpoint_tensor_converter import (
    DeepseekV41DequantConverter,
    DeepseekV41LlmConverter,
    dequant_converted_weight,
)
from veomni.models.seed_omni.modules.deepseek_v41.convert_model import convert_deepseek_v41_checkpoint
from veomni.models.seed_omni.modules.deepseek_v41.dequant import (
    dequant_fp4_blockwise,
    dequant_fp8_blockwise,
    quant_fp4_blockwise,
    quant_fp8_blockwise,
)
from veomni.models.seed_omni.modules.deepseek_v41.hf_layout import build_deepseek_v41_hf_layout
from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Llm
from veomni.models.seed_omni.modules.deepseek_v41.released_names import (
    LLM,
    MODULES,
    NGRAM,
    llm_layer_names,
    mtp_layer_names,
)
from veomni.models.seed_omni.utils.hf_layout import (
    HFSource,
    attach_hf_source_converter,
    load_module_from_hf_source,
    save_hf_source_checkpoint,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

HIDDEN = 64
N_LAYERS = 2
N_EXPERTS = 4
MOE_INTER = 32
HC = 4
ENGRAM_LAYERS = [1]

QUANTIZED_SUFFIXES = (
    "wq_a",
    "wq_b",
    "attn.wkv",
    "wo_a",
    "wo_b",
    "indexer.wq_b",
    "shared_experts.w1",
    "shared_experts.w2",
    "shared_experts.w3",
)


def llm_config():
    from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig

    return DeepseekV41LlmConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        sliding_window=6,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=4,
        rms_norm_eps=1e-20,
        hc_mult=HC,
        compress_ratios=[2] * N_LAYERS + [0, 0, 0],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        engram_layer_ids=list(ENGRAM_LAYERS),
        dspark_target_layer_ids=[1],
        dspark_n_routed_experts=N_EXPERTS,
        max_position_embeddings=256,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=MOE_INTER,
        n_shared_experts=1,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        ops_implementation={
            "mhc_implementation": "eager",
            "moe_implementation": "eager",
            "rms_norm_implementation": "eager",
            "gated_mlp_implementation": "eager",
            "linear_implementation": "eager",
            "kv_round_implementation": "eager",
            "dsa_indexer_implementation": "eager",
        },
    )


def _e8m0_scale(*shape):
    return torch.randint(123, 132, shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def _fp8_pair(out_dim, in_dim, block=32):
    weight = torch.randn(out_dim, in_dim).to(torch.float8_e4m3fn)
    return weight, _e8m0_scale(out_dim // block, in_dim // block)


def _fp4_pair(out_dim, in_dim, block=32):
    packed = torch.randint(0, 256, (out_dim, in_dim // 2), dtype=torch.uint8).view(torch.int8)
    return packed, _e8m0_scale(out_dim, in_dim // block)


def build_released_checkpoint(path, config):
    """Build a two-shard synthetic checkpoint in the released DeepSeek-V4.1 layout."""
    from veomni.models.seed_omni.modules.deepseek_v41.mtp.configuration import DeepseekV41MtpConfig
    from veomni.models.seed_omni.modules.deepseek_v41.mtp.modeling import DeepseekV41Mtp

    torch.manual_seed(0)
    llm = DeepseekV41Llm(config)
    mtp_cfg = DeepseekV41MtpConfig(**{k: v for k, v in config.to_dict().items() if k != "model_type"})
    mtp = DeepseekV41Mtp(mtp_cfg)
    module_shapes = {name: tuple(p.shape) for name, p in llm.named_parameters()}
    module_shapes.update({name: tuple(b.shape) for name, b in llm.named_buffers()})
    module_shapes.update({name: tuple(p.shape) for name, p in mtp.named_parameters()})
    module_shapes.update({name: tuple(b.shape) for name, b in mtp.named_buffers()})
    tensors = {}

    for layer in range(config.num_hidden_layers):
        for released, target in llm_layer_names(layer).items():
            shape = module_shapes.get(target)
            if shape is None:
                continue
            if any(released.endswith(f"{suffix}.weight") for suffix in QUANTIZED_SUFFIXES):
                weight, scale = _fp8_pair(*shape)
                tensors[released] = weight
                tensors[f"{released[: -len('.weight')]}.scale"] = scale
            else:
                tensors[released] = torch.randn(*shape) if len(shape) > 1 else torch.ones(*shape)

        fused_shape = module_shapes[f"model.layers.{layer}.mlp.experts.gate_up_proj"]
        down_shape = module_shapes[f"model.layers.{layer}.mlp.experts.down_proj"]
        inter, hidden = fused_shape[1] // 2, fused_shape[2]
        for expert in range(config.n_routed_experts):
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            for part, (out_dim, in_dim) in (
                ("w1", (inter, hidden)),
                ("w3", (inter, hidden)),
                ("w2", (down_shape[1], down_shape[2])),
            ):
                packed, scale = _fp4_pair(out_dim, in_dim)
                tensors[f"{prefix}.{part}.weight"] = packed
                tensors[f"{prefix}.{part}.scale"] = scale

        if layer in ENGRAM_LAYERS:
            engram = f"layers.{layer}.engram"
            for part in ("q_weight", "k_weight"):
                tensors[f"{engram}.{part}"] = torch.randn(*module_shapes[f"model.layers.{layer}.engram.{part}"])
            weight, scale = _fp8_pair(*module_shapes[f"model.layers.{layer}.engram.wkv.weight"])
            tensors[f"{engram}.wkv.weight"] = weight
            tensors[f"{engram}.wkv.scale"] = scale

    n_mtp = config.num_nextn_predict_layers
    n_mtp_experts = config.dspark_n_routed_experts or config.n_routed_experts
    for stage in range(n_mtp):
        for released, target in mtp_layer_names(stage).items():
            shape = module_shapes.get(target)
            if shape is None:
                continue
            if any(released.endswith(f"{suffix}.weight") for suffix in QUANTIZED_SUFFIXES):
                weight, scale = _fp8_pair(*shape)
                tensors[released] = weight
                tensors[f"{released[: -len('.weight')]}.scale"] = scale
            else:
                tensors[released] = torch.randn(*shape) if len(shape) > 1 else torch.ones(*shape)
        fused_shape = module_shapes[f"model.mtp.{stage}.mlp.experts.gate_up_proj"]
        down_shape = module_shapes[f"model.mtp.{stage}.mlp.experts.down_proj"]
        inter, hidden = fused_shape[1] // 2, fused_shape[2]
        for expert in range(n_mtp_experts):
            prefix = f"mtp.{stage}.ffn.experts.{expert}"
            for part, (out_dim, in_dim) in (
                ("w1", (inter, hidden)),
                ("w3", (inter, hidden)),
                ("w2", (down_shape[1], down_shape[2])),
            ):
                packed, scale = _fp4_pair(out_dim, in_dim)
                tensors[f"{prefix}.{part}.weight"] = packed
                tensors[f"{prefix}.{part}.scale"] = scale

    last = n_mtp - 1
    weight, scale = _fp8_pair(*module_shapes["model.mtp.0.main_proj.weight"])
    tensors["mtp.0.main_proj.weight"] = weight
    tensors["mtp.0.main_proj.scale"] = scale
    tensors["mtp.0.main_norm.weight"] = torch.ones(*module_shapes["model.mtp.0.main_norm.weight"])
    tensors[f"mtp.{last}.norm.weight"] = torch.ones(*module_shapes[f"model.mtp.{last}.norm.weight"])
    tensors[f"mtp.{last}.markov_head.embed.weight"] = torch.randn(
        *module_shapes[f"model.mtp.{last}.markov_head.embed.weight"]
    )
    tensors[f"mtp.{last}.markov_head.head.weight"] = torch.randn(
        *module_shapes[f"model.mtp.{last}.markov_head.head.weight"]
    )
    tensors[f"mtp.{last}.confidence_head.proj.weight"] = torch.randn(
        *module_shapes[f"model.mtp.{last}.confidence_head.proj.weight"]
    )

    tensors["norm.weight"] = torch.ones(*module_shapes["model.norm.weight"])
    tensors["vision.norm.weight"] = torch.ones(HIDDEN)
    tensors["aligner.w1.bias"] = torch.zeros(HIDDEN)
    tensors["image_start"] = torch.randn(HIDDEN)

    os.makedirs(path, exist_ok=True)
    names = sorted(tensors)
    half = len(names) // 2
    weight_map = {}
    for shard_index, chunk in enumerate((names[:half], names[half:])):
        file_name = f"model-{shard_index + 1:05d}-of-00002.safetensors"
        save_file({n: tensors[n].contiguous() for n in chunk}, os.path.join(path, file_name))
        weight_map.update(dict.fromkeys(chunk, file_name))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as handle:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, handle)
    return tensors


def _write_released_config(path):
    text = {k: v for k, v in llm_config().to_dict().items() if not k.startswith("_")}
    text.update(
        engram_layer_ids=ENGRAM_LAYERS,
        engram_num_embeddings=[1024],
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=16,
        engram_vocab_size=4096,
        engram_compressed_vocab_size=100,
        engram_pad_token_id=2,
    )
    vision = {"num_hidden_layers": 1, "hidden_size": HIDDEN, "num_attention_heads": 4, "intermediate_size": 128}
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump({"model_type": "deepseek_v41", "text_config": text, "vision_config": vision}, handle)


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


def _load_direct_mtp(path: str):
    from veomni.models.seed_omni.modules.deepseek_v41.mtp.modeling import DeepseekV41Mtp
    from veomni.models.seed_omni.modules.deepseek_v41.released_names import MTP

    layout = build_deepseek_v41_hf_layout(path)
    config = layout.build_module_configs(path)[MTP]
    model = DeepseekV41Mtp._from_config(config)
    attach_hf_source_converter(model, HFSource(path=os.path.realpath(path), model_type="deepseek_v41"), MTP)
    load_model_weights(model, path, "cpu")
    return model


def _stream_as_rank(
    path: str,
    monkeypatch,
    ep_rank: int,
    module_name: str = LLM,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Stream ``llm`` or ``mtp`` off the release as EP rank ``ep_rank``; return its state and the keys it read."""
    from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import DeepseekV41LlmAccelerated
    from veomni.models.seed_omni.modules.deepseek_v41.mtp.accelerated import DeepseekV41MtpAccelerated
    from veomni.models.seed_omni.modules.deepseek_v41.mtp.modeling import DeepseekV41Mtp

    layout = build_deepseek_v41_hf_layout(path)
    config = layout.build_module_configs(path)[module_name]
    model_cls = DeepseekV41Llm if module_name == LLM else DeepseekV41Mtp
    accel_cls = DeepseekV41LlmAccelerated if module_name == LLM else DeepseekV41MtpAccelerated
    with init_empty_weights():
        model = model_cls._from_config(config)
    # The rank-local expert slabs EP wrapping would leave on this rank.
    for name, param in list(model.named_parameters()):
        if _EXPERT_PARAM.search(name):
            owner, attr = model.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]
            local = torch.empty((param.shape[0] // EP_SIZE, *param.shape[1:]), dtype=param.dtype, device="meta")
            setattr(owner, attr, torch.nn.Parameter(local, requires_grad=param.requires_grad))
    model.get_parallel_plan = types.MethodType(accel_cls.get_parallel_plan, model)
    attach_hf_source_converter(model, HFSource(path=os.path.realpath(path), model_type="deepseek_v41"), module_name)

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
        """Main layers hold 4 experts (``deepseek_v41_llm``) and draft stages 2 (``deepseek_v41_mtp``)."""
        from veomni.models.seed_omni.modules.deepseek_v41.released_names import MTP

        path, _, _ = release
        for module_name, loader, kind, count in (
            (LLM, _load_direct_llm, "layers", N_EXPERTS),
            (MTP, _load_direct_mtp, "mtp", N_DRAFT_EXPERTS),
        ):
            whole = loader(path).state_dict()
            ranks = [_stream_as_rank(path, monkeypatch, rank, module_name=module_name) for rank in range(EP_SIZE)]

            for key, value in whole.items():
                if _EXPERT_PARAM.search(key):
                    streamed = torch.cat([state[key] for state, _ in ranks])
                else:
                    streamed = ranks[1][0][key]
                assert streamed.dtype == value.dtype, (module_name, key)
                assert torch.equal(streamed.view(torch.uint8), value.view(torch.uint8)), (module_name, key)

            local = count // EP_SIZE
            for rank, (_, read) in enumerate(ranks):
                experts = {
                    int(key.split(".")[4]) for key in read if key.startswith(f"{kind}.") and ".ffn.experts." in key
                }
                assert experts == set(range(rank * local, (rank + 1) * local)), (module_name, rank, kind)


def _quantize_rows(values: torch.Tensor):
    blocks = values.float().unflatten(-1, (values.shape[-1] // 32, 32))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    packed = (blocks / scale).to(torch.float8_e4m3fn)
    return packed.flatten(-2), scale.squeeze(-1).to(torch.float8_e8m0fnu)


def _add_ngram_tables(path: str, ngram_config) -> dict[str, torch.Tensor]:
    """The release's fp8 tables and row scales, in a shard of their own; the llm miniature has none."""
    from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram

    quantize_rows = _quantize_rows

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
    sliced: dict[str, list[slice]] = {}
    real_safe_open = module_utils.safe_open

    class _SliceProxy:
        def __init__(self, key: str, inner):
            self._key = key
            self._inner = inner

        def __getitem__(self, item):
            row_sel = item[0] if isinstance(item, tuple) else item
            if isinstance(row_sel, slice):
                sliced.setdefault(self._key, []).append(row_sel)
            return self._inner[item]

        def __getattr__(self, attr):
            return getattr(self._inner, attr)

    class _Recording:
        def __init__(self, *args, **kwargs):
            self._handle = real_safe_open(*args, **kwargs)

        def __enter__(self):
            handle = self._handle.__enter__()

            class _Proxy:
                def get_tensor(self, key):
                    read.append(key)
                    return handle.get_tensor(key)

                def get_slice(self, key):
                    return _SliceProxy(key, handle.get_slice(key))

                def __getattr__(self, attr):
                    return getattr(handle, attr)

            return _Proxy()

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    monkeypatch.setattr(module_utils, "safe_open", _Recording)
    load_model_weights_ep_sharded(model, path, "cpu")
    monkeypatch.setattr(module_utils, "safe_open", real_safe_open)
    return {name: tensor.detach() for name, tensor in model.state_dict().items()}, read, sliced


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
                streamed = torch.cat([state[key] for state, _, _ in ranks])
            else:
                streamed = ranks[1][0][key]
            assert streamed.dtype == value.dtype, key
            assert torch.equal(streamed.view(torch.uint8), value.view(torch.uint8)), key
        for rank_idx, (_, read, sliced) in enumerate(ranks):
            assert not set(read) & set(tables), read
            for table_key, tensor in tables.items():
                local_rows = tensor.shape[0] // EP_SIZE
                expected = slice(rank_idx * local_rows, (rank_idx + 1) * local_rows)
                assert sliced.get(table_key) == [expected], (rank_idx, table_key, sliced.get(table_key))


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

    def test_experts_split_on_dim0_by_two_mesh_dims_recover_physical_expert_order(self):
        """Muon zero-comm restores ``(ep_fsdp, ep)`` both on dim 0; ``_local_experts``
        walks ``ep`` outer then ``ep_fsdp`` inner to recover the physical expert offset."""
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard
        from torch.testing._internal.distributed.fake_pg import FakeStore

        from veomni.models.seed_omni.modules.deepseek_v41.export import _local_experts

        # Rank 2 in a 2x2 (ep_fsdp=2, ep=2) mesh has ep_fsdp_rank=1, ep_rank=0 ->
        # physical chunk index = 0 * 2 + 1 = 1 -> start expert = 2.
        dist.init_process_group("fake", rank=2, world_size=4, store=FakeStore())
        try:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
            local_slab = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
            experts = DTensor.from_local(local_slab, mesh, [Shard(0), Shard(0)], run_check=False)
            local, start = _local_experts(experts, 0)
            assert start == 2
            assert torch.equal(local, local_slab)
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


def test_deepgemm_streams_and_dequantizes_quantized_release(tmp_path, monkeypatch):
    """Under ``moe_implementation='deepgemm'``, streaming a quantized release
    automatically dequantizes FP8 dense and packed FP4 expert weights to BF16/FP32
    parameters even when ``config.weight_quantization`` is left on disk."""
    from unittest.mock import Mock

    import veomni.models.seed_omni.modules.deepseek_v41.llm.modeling as quant_mod
    from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import DeepseekV41LlmAccelerated

    path = str(tmp_path / "release_quant")
    tensors = _release(path, quantized=True)
    monkeypatch.setattr(
        quant_mod,
        "VeomniOp",
        lambda _domain, _variant, impl, **_kw: Mock(impl=impl),
    )

    layout = build_deepseek_v41_hf_layout(path)
    ranks = []
    for ep_rank in range(EP_SIZE):
        config = layout.build_module_configs(path)[LLM]
        assert config.weight_quantization is not None
        config._set_ops_implementation({"moe_implementation": "deepgemm"})
        with init_empty_weights():
            model = DeepseekV41Llm._from_config(config, dtype=torch.bfloat16)
        assert model.model.layers[0].mlp.experts.veomni_moe.impl == "deepgemm"
        assert not model.model.layers[0].mlp.experts.experts_are_quantized
        for name, param in list(model.named_parameters()):
            if _EXPERT_PARAM.search(name):
                owner, attr = model.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]
                local = torch.empty((param.shape[0] // EP_SIZE, *param.shape[1:]), dtype=param.dtype, device="meta")
                setattr(owner, attr, torch.nn.Parameter(local, requires_grad=param.requires_grad))
        model.get_parallel_plan = types.MethodType(DeepseekV41LlmAccelerated.get_parallel_plan, model)
        attach_hf_source_converter(model, HFSource(path=os.path.realpath(path), model_type="deepseek_v41"), LLM)
        monkeypatch.setattr(
            module_utils,
            "_ep_dim0_slice_meta",
            lambda _state, _group, target0, r=ep_rank: (EP_SIZE, r, target0 * EP_SIZE),
        )
        load_model_weights_ep_sharded(model, path, "cpu")
        ranks.append({name: tensor.detach() for name, tensor in model.state_dict().items()})

    experts_key = "model.layers.0.mlp.experts.gate_up_proj"
    dense_key = "model.layers.0.self_attn.wq_a.weight"
    streamed_experts = torch.cat([state[experts_key] for state in ranks])
    assert streamed_experts.dtype == torch.bfloat16
    w1 = [_dequant_release(tensors, f"layers.0.ffn.experts.{e}.w1.weight") for e in range(N_EXPERTS)]
    w3 = [_dequant_release(tensors, f"layers.0.ffn.experts.{e}.w3.weight") for e in range(N_EXPERTS)]
    want_experts = torch.stack([torch.cat(pair) for pair in zip(w1, w3, strict=True)]).bfloat16()
    torch.testing.assert_close(streamed_experts, want_experts, rtol=0, atol=0)
    want_dense = _dequant_release(tensors, "layers.0.attn.wq_a.weight").bfloat16()
    torch.testing.assert_close(ranks[0][dense_key], want_dense, rtol=0, atol=0)


def test_fp8_and_fp4_requant_round_trip_is_value_exact():
    torch.manual_seed(0)
    fp8_w = (torch.randn(64, 96) * 100).to(torch.float8_e4m3fn)
    fp8_s = _e8m0_scale(2, 3)
    fp8_vals = dequant_fp8_blockwise(fp8_w, fp8_s, out_dtype=torch.float32)
    q_w, q_s = quant_fp8_blockwise(fp8_vals, (32, 32))
    assert torch.equal(dequant_fp8_blockwise(q_w, q_s, out_dtype=torch.float32), fp8_vals)

    packed = torch.randint(-128, 128, (16, 32), dtype=torch.int8)
    packed[:, 0] = 0x7F
    fp4_s = _e8m0_scale(16, 2)
    fp4_vals = dequant_fp4_blockwise(packed, fp4_s, out_dtype=torch.float32)
    rq_w, rq_s = quant_fp4_blockwise(fp4_vals)
    assert torch.equal(rq_s.view(torch.uint8), fp4_s.view(torch.uint8))
    assert torch.equal(dequant_fp4_blockwise(rq_w, rq_s, out_dtype=torch.float32), fp4_vals)


def test_dequant_converter_export_weights_keeps_unquantized_3d_experts_bf16():
    conv = DeepseekV41DequantConverter(config=None)
    state = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.ones((4, 8, 4), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.down_proj": torch.ones((4, 4, 8), dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.float32),
        "model.layers.0.mlp.gate.e_score_correction_bias": torch.zeros((4,), dtype=torch.bfloat16),
    }
    dummy_model = types.SimpleNamespace(state_dict=lambda: state, named_modules=lambda: [])
    exported = dict(conv.export_weights(dummy_model))
    assert exported["model.layers.0.mlp.experts.gate_up_proj"].dtype == torch.bfloat16
    assert exported["model.layers.0.mlp.experts.down_proj"].dtype == torch.bfloat16
    assert exported["model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16
    assert exported["model.layers.0.mlp.gate.e_score_correction_bias"].dtype == torch.float32


def test_resolve_omni_checkpoint_root_passes_path_to_deepseek_v41_layout(tmp_path, monkeypatch):
    """Public ``resolve_omni_checkpoint_root`` must work on a DeepSeek-V4.1 release via ``model.model_path``."""
    from transformers import AutoTokenizer

    from veomni.models.seed_omni.utils.hf_layout import resolve_omni_checkpoint_root

    class _DummyTokenizer:
        def save_pretrained(self, out_dir: str) -> None:
            os.makedirs(out_dir, exist_ok=True)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", staticmethod(lambda *_a, **_kw: _DummyTokenizer()))
    path = str(tmp_path / "release_root")
    _release(path, quantized=False)
    assert HFSource.from_path(path, "deepseek_v41").path == os.path.realpath(path)
    resolved = resolve_omni_checkpoint_root(path)
    assert os.path.isfile(os.path.join(resolved, "config.json"))
    assert os.path.isfile(os.path.join(resolved, LLM, "config.json"))
    with open(os.path.join(resolved, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["model_type"] == "omni"


def test_convert_llm_rejects_missing_required_weight(tmp_path):
    """Missing required LLM weights (e.g. ``layers.0.attn_norm.weight``) fail conversion instead of being skipped."""
    from veomni.models.seed_omni.modules.deepseek_v41.convert_model import (
        TensorSource,
        _build_configs,
        convert_llm,
    )

    path = str(tmp_path / "release_missing")
    _release(path, quantized=False)
    index_path = os.path.join(path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    index["weight_map"].pop("layers.0.attn_norm.weight")
    with open(index_path, "w") as f:
        json.dump(index, f)
    with open(os.path.join(path, "config.json")) as f:
        hf_cfg = json.load(f)
    _, _, _, llm_cfg, _ = _build_configs(hf_cfg)
    with pytest.raises(KeyError, match=r"layers\.0\.attn_norm\.weight"):
        convert_llm(TensorSource(path), str(tmp_path / "out"), llm_cfg, list(llm_cfg.engram_layer_ids))


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "fp8"])
def test_ngram_stream_load_rejects_truncated_table_below_logical_rows(tmp_path, monkeypatch, quantized):
    """Streaming EP load must reject checkpoint tables with fewer than logical ``num_embeddings`` rows."""
    import copy

    from veomni.models.seed_omni.modules.deepseek_v41 import hf_layout as v41_hf_layout

    path = str(tmp_path / f"release_trunc_{quantized}")
    _release(path, quantized=quantized)
    with open(os.path.join(path, "config.json")) as f:
        released = json.load(f)
    released["text_config"].update(
        engram_vocab_size=64, engram_num_embeddings=[1000] * len(ENGRAM_LAYERS), engram_compressed_vocab_size=40
    )
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(released, f)
    config = build_deepseek_v41_hf_layout(path).build_module_configs(path)[NGRAM]
    # Write truncated tables (512 rows < logical 1000 rows, while model pads to 1024).
    trunc_cfg = copy.deepcopy(config)
    trunc_cfg.num_embeddings = [512] * len(ENGRAM_LAYERS)
    _add_ngram_tables(path, trunc_cfg)
    token_map = torch.arange(config.vocab_size, dtype=torch.int64) % config.compressed_vocab_size
    monkeypatch.setattr(v41_hf_layout, "build_ngram_token_map", lambda _path, _config: token_map.clone())

    with pytest.raises(RuntimeError, match="below required logical dim0=1000"):
        _stream_ngram_as_rank(path, monkeypatch, rank=0)


def test_convert_llm_allows_ratio1_kv_source_without_wgate(tmp_path):
    """A ratio-1 KV source (e.g. default layer 20) has no ``compressor.wgate.weight``, while ratio > 1 requires it."""
    from veomni.models.seed_omni.modules.deepseek_v41.convert_model import (
        TensorSource,
        _build_configs,
        convert_llm,
    )

    path = str(tmp_path / "release_ratio1")
    _release(path, quantized=False)
    index_path = os.path.join(path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    # Layer 0 is the KV source in the miniature release; remove its compressor.wgate.weight.
    index["weight_map"].pop("layers.0.attn.compressor.wgate.weight")
    with open(index_path, "w") as f:
        json.dump(index, f)
    with open(os.path.join(path, "config.json")) as f:
        hf_cfg = json.load(f)
    # With compress_ratio=2, missing wgate must fail conversion.
    _, _, _, llm_cfg_r2, _ = _build_configs(hf_cfg)
    assert 0 in llm_cfg_r2.kv_source_layer_ids and llm_cfg_r2.compress_ratios[0] == 2
    with pytest.raises(KeyError, match=r"layers\.0\.attn\.compressor\.wgate\.weight"):
        convert_llm(TensorSource(path), str(tmp_path / "out_r2"), llm_cfg_r2, list(llm_cfg_r2.engram_layer_ids))

    # With compress_ratio=1, DeepseekV41Compressor has no wgate so conversion must succeed.
    hf_cfg["text_config"]["compress_ratios"] = [1, 1, 0, 0, 0]
    _, _, _, llm_cfg_r1, _ = _build_configs(hf_cfg)
    assert 0 in llm_cfg_r1.kv_source_layer_ids and llm_cfg_r1.compress_ratios[0] == 1
    n_written = convert_llm(
        TensorSource(path), str(tmp_path / "out_r1"), llm_cfg_r1, list(llm_cfg_r1.engram_layer_ids)
    )
    assert n_written > 0


def test_quant_fp4_blockwise_matches_nearest_even_reference_on_midpoints():
    """Midpoint ties must round to nearest-even code, matching ``veomni.ops.kernels.kv_round.fp4.eager``."""
    from veomni.ops.kernels.kv_round.fp4.eager import _round_to_fp4

    # Include all 7 positive & negative midpoints plus 6.0 so block scale is exactly 1.0 (amax=6.0).
    midpoints = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    row = midpoints + [-m for m in midpoints] + [6.0, -6.0] + [0.0] * 16
    weight = torch.tensor([row], dtype=torch.float32)
    packed, scale = quant_fp4_blockwise(weight, block=32)
    deq = dequant_fp4_blockwise(packed, scale, out_dtype=torch.float32)
    ref = _round_to_fp4(weight)
    torch.testing.assert_close(deq, ref, rtol=0, atol=0)
    # Explicitly verify 0.75 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0, 5.0 -> 4.0.
    assert deq[0, :7].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    assert deq[0, 7:14].tolist() == [0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0]
