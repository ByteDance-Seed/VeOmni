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

"""The V4.1 converter, over a synthetic checkpoint in the released layout.

The real checkpoint is 510 GB, so the thing built here is a miniature in the
same shape: the released naming (``layers.N.attn.wq_a``, ``ffn.gate.bias``,
``hc_attn_fn``), fp8 dense weights with block scales, fp4 packed experts, an
index spanning more than one shard.

What that buys is the check worth having -- the converted output *loads into
the modules*, by name, with no missing and no unexpected keys. A rename that is
merely self-consistent passes every test that only reads the converter's own
output; loading it against the real module definitions is what catches one.
"""

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file, save_file

from veomni.models.seed_omni.modules.deepseek_v41.convert_model import (
    ShardWriter,
    TensorSource,
    convert_deepseek_v41_checkpoint,
    convert_llm,
    convert_vision,
)
from veomni.models.seed_omni.modules.deepseek_v41.dequant import dequant_fp8_blockwise
from veomni.models.seed_omni.modules.deepseek_v41.released_names import llm_layer_names, mtp_layer_names


HIDDEN = 64
N_LAYERS = 2
N_EXPERTS = 4
MOE_INTER = 32
HC = 4
ENGRAM_LAYERS = [1]


def llm_config():
    from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig

    return DeepseekV41LlmConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        # Every fp8 dense dimension must be a multiple of the 32-wide block, the
        # way the released shapes are; otherwise the fixture exercises a block
        # the dequantizer is right to refuse.
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
    )


def e8m0_scale(*shape):
    """Block scales in the dtype the release stores them in.

    e8m0 is an exponent and nothing else -- byte ``b`` means ``2**(b - 127)`` --
    so the byte is drawn directly. Drawing a float and casting would collapse:
    every value rounds to the nearest power of two, and the narrow band around
    1.0 that reads naturally becomes all ones, which is a scale that cannot tell
    a correct dequantization from one that ignored it.

    It also matters that this is one byte rather than four. A float32 stand-in
    makes an expert look 18% larger than it is, which is enough
    to move a size ratio past a threshold and send someone looking for a bug in
    the converter.
    """
    return torch.randint(123, 132, shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def fp8_pair(out_dim, in_dim, block=32):
    """A weight and the block scale that dequantizes it."""
    weight = torch.randn(out_dim, in_dim).to(torch.float8_e4m3fn)
    return weight, e8m0_scale(out_dim // block, in_dim // block)


def fp4_pair(out_dim, in_dim, block=32):
    packed = torch.randint(0, 256, (out_dim, in_dim // 2), dtype=torch.uint8).view(torch.int8)
    return packed, e8m0_scale(out_dim, in_dim // block)


#: Released tensors that carry a block scale, as in the release's index. Everything
#: else is a norm gain, a bias, an embedding or a small projection (the compressor's
#: ``wkv`` among them), which the release stores unquantized.
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


def build_released_checkpoint(path, config):
    """A miniature in the released layout, split over two shards.

    Shapes are taken from the module the converter targets, by inverting the
    name map -- the released checkpoint's own shapes cannot be scaled down, and
    hand-written ones drift from the module as it changes. That does make the
    shape agreement in ``TestLlmConversionLoads`` follow by construction; the
    names do not, and they are what this fixture exists to exercise. The real
    checkpoint's shapes are checked against the module separately, from its
    safetensors headers, which needs the 510 GB file and so is not a CI test.
    """
    from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Llm

    torch.manual_seed(0)
    module_shapes = {name: tuple(p.shape) for name, p in DeepseekV41Llm(config).named_parameters()}
    module_shapes.update({name: tuple(b.shape) for name, b in DeepseekV41Llm(config).named_buffers()})
    tensors = {}

    for layer in range(config.num_hidden_layers):
        for released, target in llm_layer_names(layer).items():
            shape = module_shapes.get(target)
            if shape is None:
                continue  # a layer without a compressor / indexer
            if any(released.endswith(f"{suffix}.weight") for suffix in QUANTIZED_SUFFIXES):
                weight, scale = fp8_pair(*shape)
                tensors[released] = weight
                tensors[f"{released[: -len('.weight')]}.scale"] = scale
            else:
                tensors[released] = torch.randn(*shape) if len(shape) > 1 else torch.ones(*shape)

        # Experts arrive per expert and per projection; the fused slab is what
        # the converter must build out of them.
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
                packed, scale = fp4_pair(out_dim, in_dim)
                tensors[f"{prefix}.{part}.weight"] = packed
                tensors[f"{prefix}.{part}.scale"] = scale

        if layer in ENGRAM_LAYERS:
            engram = f"layers.{layer}.engram"
            for part in ("q_weight", "k_weight"):
                tensors[f"{engram}.{part}"] = torch.randn(*module_shapes[f"model.engram.{layer}.{part}"])
            weight, scale = fp8_pair(*module_shapes[f"model.engram.{layer}.wkv.weight"])
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
                weight, scale = fp8_pair(*shape)
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
                packed, scale = fp4_pair(out_dim, in_dim)
                tensors[f"{prefix}.{part}.weight"] = packed
                tensors[f"{prefix}.{part}.scale"] = scale

    last = n_mtp - 1
    weight, scale = fp8_pair(*module_shapes["model.mtp.0.main_proj.weight"])
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


@pytest.fixture
def released(tmp_path):
    config = llm_config()
    tensors = build_released_checkpoint(str(tmp_path / "released"), config)
    return config, tensors, str(tmp_path / "released")


class TestShardWriter:
    def test_a_single_shard_becomes_model_safetensors(self, tmp_path):
        writer = ShardWriter(str(tmp_path / "out"))
        writer.add("a", torch.zeros(4))
        assert writer.finalize() == 1
        assert os.path.exists(tmp_path / "out" / "model.safetensors")
        assert not os.path.exists(tmp_path / "out" / "model.safetensors.index.json")

    def test_multiple_shards_get_an_index(self, tmp_path):
        writer = ShardWriter(str(tmp_path / "out"), shard_bytes=64)
        for i in range(6):
            writer.add(f"t{i}", torch.zeros(32))  # 128 bytes each -> one per shard
        assert writer.finalize() == 6
        index = json.load(open(tmp_path / "out" / "model.safetensors.index.json"))
        assert set(index["weight_map"]) == {f"t{i}" for i in range(6)}
        assert index["metadata"]["total_size"] == 6 * 32 * 4

    def test_shard_names_agree_with_the_index(self, tmp_path):
        """Provisional names are renamed only once the count is known; a file
        the index points at but that does not exist is the failure this guards."""
        writer = ShardWriter(str(tmp_path / "out"), shard_bytes=64)
        for i in range(4):
            writer.add(f"t{i}", torch.zeros(32))
        writer.finalize()
        index = json.load(open(tmp_path / "out" / "model.safetensors.index.json"))
        for file_name in set(index["weight_map"].values()):
            assert os.path.exists(tmp_path / "out" / file_name), file_name
            assert file_name.endswith("-of-00004.safetensors")
        assert not any(p.endswith(".partial") for p in os.listdir(tmp_path / "out"))

    def test_writing_a_name_twice_is_refused(self, tmp_path):
        writer = ShardWriter(str(tmp_path / "out"))
        writer.add("a", torch.zeros(4))
        with pytest.raises(ValueError, match="written twice"):
            writer.add("a", torch.zeros(4))

    def test_an_empty_module_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError, match="no tensors were written"):
            ShardWriter(str(tmp_path / "out")).finalize()


class TestDestinationsWithoutMemoryMapping:
    """``save_file`` serializes through a memory map, which the HDFS FUSE mount
    refuses with ``OSError 38`` -- and that mount is where this model's output
    has to go, since 1.5 TB does not fit locally. The writer probes for that and
    stages each shard through local scratch instead."""

    @staticmethod
    def _reject_mmap_under(monkeypatch, directory):
        """Make ``save_file`` behave like the FUSE mount for one directory."""
        import veomni.models.seed_omni.modules.deepseek_v41.convert_model as convert_module

        real_save_file = convert_module.save_file

        def guarded(tensors, path, **kwargs):
            if os.path.realpath(path).startswith(os.path.realpath(str(directory))):
                raise OSError(38, "Function not implemented")
            return real_save_file(tensors, path, **kwargs)

        monkeypatch.setattr(convert_module, "save_file", guarded)

    def test_a_local_directory_needs_no_staging(self, tmp_path):
        assert ShardWriter(str(tmp_path / "out"))._stage is False

    def test_a_destination_that_refuses_the_map_is_detected(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        self._reject_mmap_under(monkeypatch, out)
        assert ShardWriter(str(out))._stage is True

    def test_staged_shards_arrive_intact(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        self._reject_mmap_under(monkeypatch, out)

        writer = ShardWriter(str(out), shard_bytes=64)
        written = {f"t{i}": torch.full((32,), float(i)) for i in range(4)}
        for name, tensor in written.items():
            writer.add(name, tensor)
        assert writer.finalize() == 4

        index = json.load(open(out / "model.safetensors.index.json"))
        read_back = {}
        for file_name in set(index["weight_map"].values()):
            read_back.update(load_file(str(out / file_name)))
        assert set(read_back) == set(written)
        for name, tensor in written.items():
            assert torch.equal(read_back[name], tensor)

    def test_scratch_is_not_left_behind(self, tmp_path, monkeypatch):
        """The staging copy is the one that could silently fill the local disk
        over a 468 GiB run."""
        out = tmp_path / "out"
        out.mkdir()
        self._reject_mmap_under(monkeypatch, out)
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
        (tmp_path / "scratch").mkdir()

        writer = ShardWriter(str(out), shard_bytes=64)
        for i in range(4):
            writer.add(f"t{i}", torch.zeros(32))
        writer.finalize()
        assert os.listdir(tmp_path / "scratch") == []


class TestTensorSource:
    def test_reads_across_shards(self, released):
        _, tensors, path = released
        source = TensorSource(path)
        assert set(source.names()) == set(tensors)
        for name in list(tensors)[:5]:
            assert source.get(name).shape == tensors[name].shape

    def test_a_quantized_weight_comes_with_its_scale(self, released):
        _, tensors, path = released
        source = TensorSource(path)
        weight, scale = source.get_released("layers.0.attn.wq_a.weight")
        assert weight.dtype == torch.float8_e4m3fn
        assert scale is not None
        assert scale.shape == tensors["layers.0.attn.wq_a.scale"].shape

    def test_an_unquantized_weight_has_no_scale(self, released):
        _, _, path = released
        source = TensorSource(path)
        weight, scale = source.get_released("layers.0.attn.q_norm.weight")
        assert scale is None
        assert weight.dtype != torch.float8_e4m3fn

    def test_a_missing_index_is_refused(self, tmp_path):
        os.makedirs(tmp_path / "bare", exist_ok=True)
        with pytest.raises(FileNotFoundError, match="index-driven"):
            TensorSource(str(tmp_path / "bare"))


class TestNameMapping:
    def test_the_two_norms_are_renamed_to_the_hf_spelling(self):
        names = llm_layer_names(3)
        assert names["layers.3.attn_norm.weight"] == "model.layers.3.input_layernorm.weight"
        assert names["layers.3.ffn_norm.weight"] == "model.layers.3.post_attention_layernorm.weight"

    def test_the_router_bias_becomes_the_correction_bias(self):
        """The release calls it `bias`; this repo's DeepSeek modules hold it as
        e_score_correction_bias, and as a buffer rather than a parameter."""
        names = llm_layer_names(0)
        assert names["layers.0.ffn.gate.bias"] == "model.layers.0.mlp.gate.e_score_correction_bias"
        assert names["layers.0.ffn.gate.bias_vl"] == "model.layers.0.mlp.gate.e_score_correction_bias_vl"

    def test_the_hc_tensors_split_by_site(self):
        names = llm_layer_names(1)
        assert names["layers.1.hc_attn_fn"] == "model.layers.1.attn_hc.fn"
        assert names["layers.1.hc_ffn_scale"] == "model.layers.1.ffn_hc.scale"

    def test_the_shared_expert_is_renamed_to_gate_up_down(self):
        names = llm_layer_names(0)
        assert names["layers.0.ffn.shared_experts.w1.weight"].endswith("shared_experts.gate_proj.weight")
        assert names["layers.0.ffn.shared_experts.w2.weight"].endswith("shared_experts.down_proj.weight")
        assert names["layers.0.ffn.shared_experts.w3.weight"].endswith("shared_experts.up_proj.weight")

    def test_attention_keeps_its_own_names_under_self_attn(self):
        names = llm_layer_names(0)
        assert names["layers.0.attn.wq_a.weight"] == "model.layers.0.self_attn.wq_a.weight"
        assert names["layers.0.attn.attn_sink"] == "model.layers.0.self_attn.attn_sink"

    def test_mtp_layers_map_under_model_mtp(self):
        names = mtp_layer_names(0)
        assert names["mtp.0.attn.wq_a.weight"] == "model.mtp.0.self_attn.wq_a.weight"
        assert names["mtp.0.attn_norm.weight"] == "model.mtp.0.input_layernorm.weight"


def _load_converted_llm(tmp_path, released):
    config, _, path = released
    convert_llm(TensorSource(path), str(tmp_path / "out"), config, list(ENGRAM_LAYERS))
    out_dir = tmp_path / "out" / "deepseek_v41_llm"
    state = {}
    for file_name in os.listdir(out_dir):
        if file_name.endswith(".safetensors"):
            state.update(load_file(str(out_dir / file_name)))
    return config, state


def _dequant_state(state):
    """Spend every ``P_scale`` the way training load does."""
    from veomni.models.checkpoint.convert import maybe_convert_checkpoint_tensor
    from veomni.models.seed_omni.modules.deepseek_v41.checkpoint_tensor_converter import DeepseekV41DequantConverter

    converter = DeepseekV41DequantConverter()
    out = {}
    for name, tensor in state.items():
        result = maybe_convert_checkpoint_tensor(name, tensor, converter)
        if result is not None:
            out[result.name] = result.tensor
    for leftover in converter.finalize():
        out[leftover.name] = leftover.tensor
    return out


class TestLlmConversionLoads:
    """The check that a self-consistent rename cannot pass.

    Convert writes released dtypes. Training load spends the scales through
    ``DeepseekV41DequantConverter``, then the bf16 module accepts the names.
    """

    def test_the_output_loads_into_the_module(self, tmp_path, released):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Llm

        config, state = _load_converted_llm(tmp_path, released)
        dequantized = _dequant_state(state)
        assert not any(name.endswith("_scale") for name in dequantized)

        model = DeepseekV41Llm(config)
        target = model.state_dict()
        aligned = {
            name: tensor.to(dtype=target[name].dtype) if name in target else tensor
            for name, tensor in dequantized.items()
        }
        missing, unexpected = model.load_state_dict(aligned, strict=False)
        assert not unexpected, f"converter produced names the module does not have: {sorted(unexpected)[:5]}"
        allowed = {"model.rotary_emb.main_inv_freq", "model.rotary_emb.compress_inv_freq"}
        assert set(missing) <= allowed, (
            f"module parameters the converter never filled: {sorted(set(missing) - allowed)[:5]}"
        )

    def test_dspark_stages_land_in_the_llm_folder(self, tmp_path, released):
        _, state = _load_converted_llm(tmp_path, released)
        assert "model.mtp.0.main_proj.weight" in state
        assert "model.mtp.0.self_attn.wq_a.weight" in state
        last = llm_config().num_nextn_predict_layers - 1
        assert f"model.mtp.{last}.markov_head.embed.weight" in state
        assert f"model.mtp.{last}.confidence_head.proj.weight" in state
        assert "model.mtp.0.embed.weight" not in state
        assert "model.mtp.0.head.weight" not in state

    def test_the_router_biases_land_on_the_buffers(self, tmp_path, released):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Llm

        config, state = _load_converted_llm(tmp_path, released)
        model = DeepseekV41Llm(config)
        key = "model.layers.0.mlp.gate.e_score_correction_bias"
        model.load_state_dict({key: state[key].to(dtype=model.state_dict()[key].dtype)}, strict=False)
        torch.testing.assert_close(
            model.model.layers[0].mlp.gate.e_score_correction_bias.float(),
            state[key].float(),
        )

    def test_the_experts_are_stacked_in_expert_order(self, tmp_path, released):
        _, state = _load_converted_llm(tmp_path, released)
        fused = state["model.layers.0.mlp.experts.gate_up_proj"]
        assert fused.shape == (N_EXPERTS, 2 * MOE_INTER, HIDDEN // 2)
        assert state["model.layers.0.mlp.experts.down_proj"].shape == (N_EXPERTS, HIDDEN, MOE_INTER // 2)

    def test_each_expert_keeps_its_own_weights(self, tmp_path, released):
        """A stack built from a reused loop variable has the right shape and the
        same expert repeated; distinct random experts make that visible."""
        _, state = _load_converted_llm(tmp_path, released)
        fused = state["model.layers.0.mlp.experts.gate_up_proj"].float()
        for expert in range(1, N_EXPERTS):
            assert not torch.allclose(fused[0], fused[expert])

    def test_the_engram_connection_stays_in_the_backbone(self, tmp_path, released):
        _, state = _load_converted_llm(tmp_path, released)
        assert "model.engram.1.wkv.weight" in state
        assert "model.engram.1.q_weight" in state
        assert not any("engram" in k and "embed" in k for k in state)


class TestConvertedDtypes:
    """Released dtypes kept, scales carried alongside."""

    def test_quantized_weights_keep_the_dtype_they_shipped_in(self, tmp_path, released):
        _, state = _load_converted_llm(tmp_path, released)

        assert state["model.layers.0.self_attn.wq_a.weight"].dtype == torch.float8_e4m3fn
        assert state["model.layers.0.mlp.experts.gate_up_proj"].dtype == torch.int8
        assert state["model.norm.weight"].dtype == torch.float32

    def test_every_quantized_weight_has_its_scale_beside_it(self, tmp_path, released):
        _, state = _load_converted_llm(tmp_path, released)

        quantized = [
            name
            for name, tensor in state.items()
            if tensor.dtype in (torch.float8_e4m3fn, torch.int8) and not name.endswith("_scale")
        ]
        assert quantized
        for name in quantized:
            assert f"{name}_scale" in state, f"{name} was written without its scale"

    def test_the_released_scale_names_do_not_leak_through(self, tmp_path, released):
        """``wq_a.scale`` is the release's name for it; the module's is ``wq_a.weight_scale``."""
        _, state = _load_converted_llm(tmp_path, released)

        assert "model.layers.0.self_attn.wq_a.weight_scale" in state
        assert "model.layers.0.self_attn.wq_a.scale" not in state

    def test_the_expert_scales_are_stacked_to_match_their_slab(self, tmp_path, released):
        _, state = _load_converted_llm(tmp_path, released)

        fused = state["model.layers.0.mlp.experts.gate_up_proj"]
        scale = state["model.layers.0.mlp.experts.gate_up_proj_scale"]
        assert fused.shape == (N_EXPERTS, 2 * MOE_INTER, HIDDEN // 2)
        assert scale.shape == (N_EXPERTS, 2 * MOE_INTER, HIDDEN // 32)

    def test_the_load_converter_spends_the_scales(self, tmp_path, released):
        """A scale separated from its fused slab would leave names and shapes
        correct and every number wrong. The training hook has to undo the
        packing the converter wrote."""
        from veomni.models.seed_omni.modules.deepseek_v41.checkpoint_tensor_converter import dequant_converted_weight

        _, state = _load_converted_llm(tmp_path, released)
        dequantized = _dequant_state(state)

        for name in ("gate_up_proj", "down_proj"):
            key = f"model.layers.0.mlp.experts.{name}"
            torch.testing.assert_close(
                dequantized[key],
                dequant_converted_weight(state[key], state[f"{key}_scale"]),
                rtol=0,
                atol=0,
            )

        key = "model.layers.0.self_attn.wq_a.weight"
        torch.testing.assert_close(
            dequantized[key],
            dequant_fp8_blockwise(state[key], state[f"{key}_scale"], torch.bfloat16),
            rtol=0,
            atol=0,
        )

    def test_the_experts_cost_a_quarter_of_their_dequantized_size(self, tmp_path, released):
        """Keeping them packed is why inference fits on eight H20s."""
        from veomni.models.seed_omni.modules.deepseek_v41.checkpoint_tensor_converter import dequant_converted_weight

        _, state = _load_converted_llm(tmp_path, released)
        key = "model.layers.0.mlp.experts.gate_up_proj"
        packed = state[key].numel() * state[key].element_size() + state[f"{key}_scale"].numel()
        rebuilt = dequant_converted_weight(state[key], state[f"{key}_scale"])
        # Four bits against sixteen, plus one e8m0 byte per 32 values: (1/2 + 1/32)
        # bytes per value against bf16's 2, or 26.6%.
        assert packed == pytest.approx(rebuilt.numel() * rebuilt.element_size() * 0.2656, rel=0.01)

    def test_unquantized_tensors_keep_the_released_dtype(self, tmp_path, released):
        _, state = _load_converted_llm(tmp_path, released)
        assert state["model.norm.weight"].dtype == torch.float32


class TestConversionOptions:
    def test_a_precision_argument_is_refused(self, tmp_path, released):
        _, _, path = released

        with pytest.raises(TypeError, match="no longer takes precision"):
            convert_deepseek_v41_checkpoint(path, str(tmp_path / "out"), precision="bf16")

    def test_an_unknown_module_is_refused(self, tmp_path, released):
        _, _, path = released

        with pytest.raises(ValueError, match="unknown modules"):
            convert_deepseek_v41_checkpoint(path, str(tmp_path / "out"), modules=["deepseek_v41_brain"])


def _write_released_config(path):
    """The released ``config.json``'s shape: ``text_config`` plus ``engram_*``, ``vision_config``."""
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


class TestThroughTheRegistry:
    def test_a_partial_run_into_a_fresh_directory_still_loads(self, tmp_path, released):
        """``--extra modules=deepseek_v41_llm`` writes one folder, yet the root names all four.

        The root config reads each module's config.json, so the folders a
        partial run skipped still need one.
        """
        from veomni.models.seed_omni.configuration_omni import OmniConfig
        from veomni.models.seed_omni.modules.deepseek_v41.convert_model import MODULES
        from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint

        _, _, path = released
        _write_released_config(path)
        output = tmp_path / "omni"

        convert_checkpoint(path, str(output), modules="deepseek_v41_llm")

        assert (output / "deepseek_v41_llm" / "model.safetensors.index.json").is_file() or (
            output / "deepseek_v41_llm" / "model.safetensors"
        ).is_file()
        for name in set(MODULES) - {"deepseek_v41_llm"}:
            assert not any(f.endswith(".safetensors") for f in os.listdir(output / name))
        config = OmniConfig.from_pretrained(output)
        assert list(config.module_names) == list(MODULES)
        assert config.infer_type == "vision_understanding"
        assert config.training_graph

    def test_the_llm_keeps_the_released_engram_layers(self, tmp_path, released):
        """The llm's engram connection sits on the layers the tables feed, not its defaults."""
        from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
        from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint

        _, _, path = released
        _write_released_config(path)
        convert_checkpoint(path, str(tmp_path / "omni"), modules="deepseek_v41_llm")

        llm = DeepseekV41LlmConfig.from_pretrained(tmp_path / "omni" / "deepseek_v41_llm")
        assert llm.engram_layer_ids == ENGRAM_LAYERS
        assert llm.engram_n_heads == 2


class TestVisionPassthrough:
    def _convert(self, tmp_path, path):
        from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY

        VisionConfig = OMNI_CONFIG_REGISTRY["deepseek_v41_vision"]()
        config = VisionConfig(num_hidden_layers=1, hidden_size=HIDDEN, num_attention_heads=4, intermediate_size=128)
        convert_vision(TensorSource(path), str(tmp_path / "out"), config)
        return load_file(str(tmp_path / "out" / "deepseek_v41_vision" / "model.safetensors"))

    def test_names_are_unchanged(self, tmp_path, released):
        """The vision module was written against the released names, so the
        converter has nothing to do here but carry them across."""
        _, _, path = released
        state = self._convert(tmp_path, path)
        assert "vision.norm.weight" in state
        assert "aligner.w1.bias" in state
        assert "image_start" in state

    def test_scales_are_not_emitted_as_tensors(self, tmp_path, released):
        """A `.scale` copied through would be an unexpected key at load time."""
        _, _, path = released
        state = self._convert(tmp_path, path)
        assert not any(k.endswith(".scale") for k in state)
