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

"""HunyuanImage 3 checkpoint export: unit round-trip + full DCP -> HF integration.

Two layers of coverage:

* ``test_expert_round_trip_is_bit_exact`` (CPU, synthetic): the exporter is the
  bit-exact inverse of the import converter for the non-trivial transforms
  (fused-expert stack + [gate,up]->[up,gate] swap). Runs without extras.

* ``test_hunyuan_export_end_to_end`` and ``test_hunyuan_export_casts_trained_dtype``
  (needs ``transformers`` and a visible CUDA device — the DCP shard loader
  calls ``device.empty_cache()``): drive
  ``scripts/merge_dcp_to_hf.save_hunyuan_image_3_weights`` against a synthetic
  runtime DCP + fake official Base and check the exported safetensors + index
  actually load as the official key set, with absent-prefix components
  restored byte-for-byte from the Base and trained tensors cast to the
  target dtype.
"""

import importlib.util
import json
import os

import pytest
import torch

from veomni.models.transformers.hunyuan_image_3.checkpoint_tensor_converter import (
    HunyuanImage3CheckpointTensorConverter,
)
from veomni.models.transformers.hunyuan_image_3.checkpoint_tensor_export import (
    HunyuanImage3CheckpointExporter,
)
from veomni.models.transformers.hunyuan_image_3.component_policy import HunyuanImage3ComponentPolicy


# ---------------------------------------------------------------------------
# CPU-always: unit round-trip of the import + export tensor transforms.
# ---------------------------------------------------------------------------

_NUM_EXPERTS = 4
_HIDDEN = 8
_INTERMEDIATE = 6

_ONLINE_POLICY = {
    "transformer": "trainable",
    "text_embedding": "trainable",
    "image_projector": "trainable",
    "timestep_modules": "trainable",
    "image_head": "trainable",
    "vae_encoder": "frozen",
    "vae_decoder": "absent",
    "vision_model": "absent",
    "vision_aligner": "absent",
    "lm_head": "absent",
}


def _policy() -> HunyuanImage3ComponentPolicy:
    return HunyuanImage3ComponentPolicy.from_dict(_ONLINE_POLICY)


def test_expert_round_trip_is_bit_exact():
    """Official split experts -> runtime fused -> official split, bit-equal.

    Exercises the two hardest transforms in one shot: the per-expert stack
    into ``experts.{gate_up_proj,down_proj}`` and the [gate,up] -> [up,gate]
    half-swap. If either half of the swap regresses, the reconstructed
    ``gate_and_up_proj`` will not match the input tensor.
    """
    generator = torch.Generator().manual_seed(0)
    prefix = "model.layers.3.mlp"
    official = {}
    for expert in range(_NUM_EXPERTS):
        official[f"{prefix}.experts.{expert}.gate_and_up_proj.weight"] = torch.randn(
            2 * _INTERMEDIATE, _HIDDEN, generator=generator
        )
        official[f"{prefix}.experts.{expert}.down_proj.weight"] = torch.randn(
            _HIDDEN, _INTERMEDIATE, generator=generator
        )

    importer = HunyuanImage3CheckpointTensorConverter(_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE, _policy())
    runtime = {}
    for name, tensor in official.items():
        converted = importer.convert(name, tensor)
        if converted is not None:
            runtime[converted.name] = converted.tensor
    assert set(runtime) == {f"{prefix}.experts.gate_up_proj", f"{prefix}.experts.down_proj"}

    exporter = HunyuanImage3CheckpointExporter(_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE)
    reconstructed = {}
    for name, tensor in runtime.items():
        for out_name, out_tensor in exporter.export_tensor(name, tensor):
            reconstructed[out_name] = out_tensor

    assert set(reconstructed) == set(official)
    for name, tensor in official.items():
        torch.testing.assert_close(reconstructed[name], tensor, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Integration tests below need ``transformers`` (via veomni) and a visible
# CUDA device (the DCP shard loader calls ``device.empty_cache()``).
# ---------------------------------------------------------------------------

pytest.importorskip("transformers", reason="scripts.merge_dcp_to_hf imports veomni -> transformers; run on the pod")

try:
    import torch.distributed.checkpoint as dcp
except Exception:  # pragma: no cover - torch always ships dcp in this repo
    pytest.skip("torch.distributed.checkpoint unavailable", allow_module_level=True)


# The export path reuses the real DCP shard loader (``_process_shard``), which calls
# ``veomni.utils.device.empty_cache()`` -> ``torch.cpu.empty_cache()`` on a CPU-only host.
# That symbol does not exist, so the end-to-end tests require a visible CUDA device.
_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DCP shard path calls device.empty_cache() (torch.cpu.empty_cache missing on CPU); run on a GPU host",
)


def _load_merge_module():
    """Load ``scripts/merge_dcp_to_hf.py`` by path (``scripts`` is not a package)."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    module_path = os.path.join(repo_root, "scripts", "merge_dcp_to_hf.py")
    spec = importlib.util.spec_from_file_location("_merge_dcp_to_hf_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INT_NUM_EXPERTS = 3
_INT_HIDDEN = 8
_INT_INTERMEDIATE = 5
_LAYER = "model.layers.0"

_EXPORTED_NON_EXPERT_KEYS = (
    "model.wte.weight",  # <- model.embed_tokens.weight (rename)
    "model.ln_f.weight",  # <- model.norm.weight (rename)
    f"{_LAYER}.self_attn.qkv_proj.weight",  # identity
    "vae.encoder.conv.weight",  # identity (vae_encoder frozen -> present)
    "final_layer.proj.weight",  # identity
)
_ABSENT_KEYS = (
    "lm_head.weight",
    "vae.decoder.conv.weight",
    "vision_model.embeddings.weight",
    "vision_aligner.proj.weight",
)


def _expert_keys():
    keys = []
    for expert in range(_INT_NUM_EXPERTS):
        keys.append(f"{_LAYER}.mlp.experts.{expert}.gate_and_up_proj.weight")
        keys.append(f"{_LAYER}.mlp.experts.{expert}.down_proj.weight")
    return keys


def _write_runtime_dcp(dcp_dir, gate, up, down):
    """Save a tiny runtime state dict as a DCP checkpoint (keys as training emits them)."""
    fused_gate_up = torch.cat([gate, up], dim=1)  # [E, 2I, H] via [gate; up] on dim 1
    state = {
        "model.model.embed_tokens.weight": torch.randn(4, _INT_HIDDEN),
        "model.model.norm.weight": torch.randn(_INT_HIDDEN),
        f"model.{_LAYER}.self_attn.qkv_proj.weight": torch.randn(6, _INT_HIDDEN),
        f"model.{_LAYER}.mlp.experts.gate_up_proj": fused_gate_up,
        f"model.{_LAYER}.mlp.experts.down_proj": down,
        "model.vae.encoder.conv.weight": torch.randn(2, 2),
        "model.final_layer.proj.weight": torch.randn(4, _INT_HIDDEN),
    }
    dcp.save(state, checkpoint_id=dcp_dir, no_dist=True)


def _write_fake_base(base_dir):
    """Write a fake official Base: config.json + one safetensors shard + index."""
    os.makedirs(base_dir, exist_ok=True)
    config = {
        "model_type": "hunyuan_image_3_moe",
        "num_experts": _INT_NUM_EXPERTS,
        "hidden_size": _INT_HIDDEN,
        "intermediate_size": _INT_INTERMEDIATE,
        "component_policy": _ONLINE_POLICY,
    }
    with open(os.path.join(base_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    with open(os.path.join(base_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "FakeTokenizer"}, f)

    from safetensors.torch import save_file

    all_official_keys = list(_EXPORTED_NON_EXPERT_KEYS) + _expert_keys() + list(_ABSENT_KEYS)
    absent_tensors = {}
    base_tensors = {}
    for key in all_official_keys:
        tensor = torch.randn(3, 4)
        base_tensors[key] = tensor
        if any(key.startswith(p) for p in ("lm_head.", "vae.decoder.", "vision_model.", "vision_aligner.")):
            absent_tensors[key] = tensor
    save_file(base_tensors, os.path.join(base_dir, "model.safetensors"), metadata={"format": "pt"})
    index = {"metadata": {"total_size": 0}, "weight_map": dict.fromkeys(all_official_keys, "model.safetensors")}
    with open(os.path.join(base_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return set(all_official_keys), absent_tensors


def _read_output_tensor(out_dir, weight_map, name):
    from safetensors import safe_open

    with safe_open(os.path.join(out_dir, weight_map[name]), framework="pt") as handle:
        return handle.get_tensor(name)


@_requires_cuda
def test_hunyuan_export_end_to_end(tmp_path):
    """Full DCP -> official HF path: index covers the official key set, renames
    were inverted, expert split + half-swap match, and absent-prefix components
    are restored byte-for-byte from the fake Base."""
    merge = _load_merge_module()

    generator = torch.Generator().manual_seed(0)
    gate = torch.randn(_INT_NUM_EXPERTS, _INT_INTERMEDIATE, _INT_HIDDEN, generator=generator)
    up = torch.randn(_INT_NUM_EXPERTS, _INT_INTERMEDIATE, _INT_HIDDEN, generator=generator)
    down = torch.randn(_INT_NUM_EXPERTS, _INT_HIDDEN, _INT_INTERMEDIATE, generator=generator)

    dcp_dir = str(tmp_path / "dcp")
    base_dir = str(tmp_path / "base")
    out_dir = str(tmp_path / "hf")
    _write_runtime_dcp(dcp_dir, gate, up, down)
    base_key_set, absent_tensors = _write_fake_base(base_dir)

    merge.save_hunyuan_image_3_weights(
        output_dir=out_dir,
        checkpoint_path=dcp_dir,
        base_dir=base_dir,
        save_dtype="float32",
        shard_size=10_000_000_000,
        verify=True,
    )

    with open(os.path.join(out_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    assert set(weight_map) == base_key_set

    # Renames were inverted.
    assert "model.wte.weight" in weight_map and "model.embed_tokens.weight" not in weight_map
    assert "model.ln_f.weight" in weight_map and "model.norm.weight" not in weight_map

    # Expert split + [gate,up]->[up,gate] swap: official gate_and_up = [up; gate].
    for expert in range(_INT_NUM_EXPERTS):
        gate_up = _read_output_tensor(out_dir, weight_map, f"{_LAYER}.mlp.experts.{expert}.gate_and_up_proj.weight")
        assert gate_up.shape == (2 * _INT_INTERMEDIATE, _INT_HIDDEN)
        torch.testing.assert_close(gate_up[:_INT_INTERMEDIATE], up[expert], rtol=0, atol=0)
        torch.testing.assert_close(gate_up[_INT_INTERMEDIATE:], gate[expert], rtol=0, atol=0)

        down_out = _read_output_tensor(out_dir, weight_map, f"{_LAYER}.mlp.experts.{expert}.down_proj.weight")
        torch.testing.assert_close(down_out, down[expert], rtol=0, atol=0)

    # Absent components restored byte-for-byte from the fake Base.
    for key, expected in absent_tensors.items():
        restored = _read_output_tensor(out_dir, weight_map, key)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)

    # Auxiliary Base assets copied; no stray temp shard parts left behind.
    assert os.path.isfile(os.path.join(out_dir, "config.json"))
    assert os.path.isfile(os.path.join(out_dir, "tokenizer_config.json"))
    assert not any(name.startswith(".part-") for name in os.listdir(out_dir))


@_requires_cuda
def test_hunyuan_export_casts_trained_dtype(tmp_path):
    """Trained tensors cast to ``save_dtype``; absent-prefix tensors keep the Base dtype."""
    merge = _load_merge_module()

    gate = torch.randn(_INT_NUM_EXPERTS, _INT_INTERMEDIATE, _INT_HIDDEN)
    up = torch.randn(_INT_NUM_EXPERTS, _INT_INTERMEDIATE, _INT_HIDDEN)
    down = torch.randn(_INT_NUM_EXPERTS, _INT_HIDDEN, _INT_INTERMEDIATE)

    dcp_dir = str(tmp_path / "dcp")
    base_dir = str(tmp_path / "base")
    out_dir = str(tmp_path / "hf")
    _write_runtime_dcp(dcp_dir, gate, up, down)
    _write_fake_base(base_dir)

    merge.save_hunyuan_image_3_weights(
        output_dir=out_dir,
        checkpoint_path=dcp_dir,
        base_dir=base_dir,
        save_dtype="bfloat16",
        shard_size=10_000_000_000,
    )

    with open(os.path.join(out_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    exported = _read_output_tensor(out_dir, weight_map, "model.wte.weight")
    assert exported.dtype == torch.bfloat16
    restored = _read_output_tensor(out_dir, weight_map, "lm_head.weight")
    assert restored.dtype == torch.float32
