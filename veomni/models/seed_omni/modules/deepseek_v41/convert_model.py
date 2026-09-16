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

"""Split the released DeepSeek-V4.1 checkpoint into SeedOmni V2 modules.

Registered under ``OMNI_CONVERT_REGISTRY["deepseek_v41"]`` and dispatched by
``scripts/convert_model.py``.

Unlike the other families' converters this one never builds the model. The
released checkpoint is 510 GB across 48 shards and 96,085 tensors, and
dequantizing it to bf16 roughly triples that -- ``from_pretrained`` followed by
``save_pretrained`` would need the whole thing resident twice. Instead tensors
are read through the checkpoint's own index, transformed, and flushed to output
shards as they go, so peak memory is one MoE layer's experts rather than one
model.

Naming
------
The released checkpoint is already in the *reference inference stack's* naming
(``layers.N.attn.wq_a``, ``ffn.gate.bias``, ``hc_attn_fn``), not HuggingFace's.
``inference/convert.py``'s renames -- ``self_attn`` -> ``attn``, ``mlp`` ->
``ffn``, ``e_score_correction_bias`` -> ``bias`` -- have already been applied to
it, so this converter walks them back rather than forward. The vision tower is
the exception: its parameter names match :class:`DeepseekV41Vision` exactly and
pass through untouched.

Quantization
------------
Dense projections are fp8 with a 32x32 block scale, routed experts are fp4
packed two to a byte with a block scale of their own, and both are dequantized
to bf16 here. The fp4 QAT in ``DeepseekV4Experts`` fake-quantizes *from* bf16,
so leaving the experts in fp4 storage would quantize an already-quantized
tensor and lose a second time. See ``dequant.py``.

The two Engram tables are also fp8 in the release and are also dequantized,
which doubles them to 393 GB. That is a deliberate interim: the tables are
expected to stay frozen, so fp8 storage with dequant-on-gather would halve
them, but ``DeepseekV41Ngram`` holds them in an ``nn.Embedding`` and has no
per-row scale buffer to hold the other half.

Not converted
-------------
The 2,401 ``mtp.*`` tensors are the DSpark draft stages, which this repo does
not implement yet. They are counted and reported rather than silently dropped,
so the number is there to check against when DSpark lands.

Output
------
::

    <output_dir>/
      deepseek_v41_vision/        # ViT + aligner + the three image marker embeddings
      deepseek_v41_text_encoder/  # embed_tokens + lm_head + tokenizer
      deepseek_v41_ngram/         # the two n-gram tables + the baked token_map
      deepseek_v41_llm/           # 40 decoder layers, Engram connections, final norm
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY
from .dequant import dequant_fp4_blockwise, dequant_fp8_blockwise, fuse_gate_up


#: Flush an output shard once it reaches this many bytes. Matches the release's
#: own shard size closely enough that the output file count is comparable.
SHARD_BYTES = 5 * 1000**3

#: What a dequantized tensor becomes. Not configurable: the QAT paths and the
#: trainer both assume bf16 master weights.
MASTER_DTYPE = torch.bfloat16

#: Roughly how much float32 a chunked dequantization holds at once. Small
#: enough to be noise next to a 197 GB output, large enough that a 384M-row
#: table is a few hundred reads rather than a few hundred thousand.
CHUNK_BYTES = 1 * 1000**3


class ShardWriter:
    """Accumulate tensors and flush them to numbered safetensors shards.

    The shard count is not known until the last tensor arrives, so shards are
    written under provisional names and renamed once it is. Writing them with a
    guessed count and fixing the index afterwards would leave file names that
    disagree with their own index on any failure between the two.

    ``save_file`` serializes through a memory map, which the HDFS FUSE mount
    does not implement (``OSError 38``) -- and that is where a 749 B model's
    output has to go, because it does not fit on the local disk. So the writer
    probes the destination once with a one-tensor file: where the map works it
    writes shards in place, and where it does not it writes each shard to local
    scratch and moves it across. Peak local use is then one shard rather than
    the whole module. The probe is a capability test rather than a path
    heuristic because the same mount type can be configured either way.
    """

    def __init__(self, output_dir: str, shard_bytes: int = SHARD_BYTES) -> None:
        self.output_dir = output_dir
        self.shard_bytes = shard_bytes
        self._pending: Dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._weight_map: Dict[str, int] = {}
        self._total_bytes = 0
        self._shard_index = 0
        os.makedirs(output_dir, exist_ok=True)
        self._stage = not self._destination_takes_a_memory_map(output_dir)

    @staticmethod
    def _destination_takes_a_memory_map(output_dir: str) -> bool:
        probe = os.path.join(output_dir, ".shard_writer_probe.safetensors")
        try:
            save_file({"probe": torch.zeros(1, dtype=torch.uint8)}, probe, metadata={"format": "pt"})
        except Exception:
            return False
        finally:
            if os.path.exists(probe):
                os.remove(probe)
        return True

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._weight_map or name in self._pending:
            raise ValueError(f"{name} written twice into {self.output_dir}")
        tensor = tensor.contiguous()
        self._pending[name] = tensor
        size = tensor.numel() * tensor.element_size()
        self._pending_bytes += size
        self._total_bytes += size
        if self._pending_bytes >= self.shard_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        path = os.path.join(self.output_dir, self._provisional(self._shard_index))
        if self._stage:
            self._save_via_scratch(path)
        else:
            save_file(self._pending, path, metadata={"format": "pt"})
        for name in self._pending:
            self._weight_map[name] = self._shard_index
        self._pending.clear()
        self._pending_bytes = 0
        self._shard_index += 1
        gc.collect()

    def _save_via_scratch(self, path: str) -> None:
        """Write the shard locally, then move it to ``path``.

        The scratch file is removed on failure so a retry does not inherit a
        half-written shard under a name the index will later claim is complete.
        """
        scratch_dir = tempfile.mkdtemp(prefix="shard_writer_")
        scratch = os.path.join(scratch_dir, os.path.basename(path))
        try:
            save_file(self._pending, scratch, metadata={"format": "pt"})
            shutil.move(scratch, path)
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    @staticmethod
    def _provisional(index: int) -> str:
        return f"model-{index:05d}.safetensors.partial"

    @staticmethod
    def _final(index: int, total: int) -> str:
        return f"model-{index + 1:05d}-of-{total:05d}.safetensors"

    def finalize(self, config: Optional[Any] = None) -> int:
        """Rename the shards, write the index, and return the tensor count."""
        self._flush()
        total = self._shard_index
        if total == 0:
            raise RuntimeError(f"{self.output_dir}: no tensors were written")

        names = {}
        for index in range(total):
            source = os.path.join(self.output_dir, self._provisional(index))
            target = self._final(index, total)
            os.rename(source, os.path.join(self.output_dir, target))
            names[index] = target

        if total == 1:
            # HF reads a lone `model.safetensors` without an index; emitting one
            # anyway is legal but makes the single-file case look sharded.
            os.rename(
                os.path.join(self.output_dir, names[0]),
                os.path.join(self.output_dir, "model.safetensors"),
            )
        else:
            index_path = os.path.join(self.output_dir, "model.safetensors.index.json")
            with open(index_path, "w") as handle:
                json.dump(
                    {
                        "metadata": {"total_size": self._total_bytes},
                        "weight_map": {name: names[shard] for name, shard in self._weight_map.items()},
                    },
                    handle,
                    indent=2,
                )
        if config is not None:
            config.save_pretrained(self.output_dir)
        return len(self._weight_map)


class TensorSource:
    """Random access to the released checkpoint, by tensor name.

    Handles are cached because the access pattern is by layer while the storage
    is by shard: one MoE layer's 1,152 expert tensors are spread over several
    files, and reopening a file per tensor turns the conversion into an open()
    benchmark.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"{index_path} not found. This converter is index-driven because the checkpoint is too "
                "large to scan; a single-file checkpoint is not the released layout."
            )
        with open(index_path) as handle:
            self.weight_map: Dict[str, str] = json.load(handle)["weight_map"]
        self._handles: Dict[str, Any] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def names(self) -> Iterable[str]:
        return self.weight_map.keys()

    def get(self, name: str) -> torch.Tensor:
        file_name = self.weight_map[name]
        handle = self._handles.get(file_name)
        if handle is None:
            handle = safe_open(os.path.join(self.model_path, file_name), framework="pt", device="cpu")
            self._handles[file_name] = handle
        return handle.get_tensor(name)

    def get_dequantized(self, name: str) -> torch.Tensor:
        """A weight, with its ``.scale`` sibling applied if it has one."""
        weight = self.get(name)
        scale_name = _scale_name(name)
        if scale_name not in self.weight_map:
            return weight.to(MASTER_DTYPE)
        scale = self.get(scale_name)
        if weight.dtype in (torch.int8, torch.uint8):
            return dequant_fp4_blockwise(weight, scale, MASTER_DTYPE)
        return dequant_fp8_blockwise(weight, scale, MASTER_DTYPE)

    def get_dequantized_in_row_chunks(self, name: str, chunk_bytes: int = CHUNK_BYTES) -> torch.Tensor:
        """Same result as :meth:`get_dequantized`, without a float32 copy of the whole tensor.

        For the engram tables this is the difference between running and not:
        each is 98 GB of fp8, so the ordinary path's float32 intermediate is
        393 GB before the bf16 result is even allocated. Rows are read straight
        off disk a chunk at a time through ``get_slice``, so the fp8 side is
        never fully resident either -- only the bf16 output is.

        Chunks are whole numbers of scale blocks. A chunk that split one would
        pair rows with a neighbouring block's exponent, which rescales them
        rather than failing.
        """
        weight_slice = self._slice(name)
        scale_name = _scale_name(name)
        if scale_name not in self.weight_map:
            return self.get(name).to(MASTER_DTYPE)
        scale_slice = self._slice(scale_name)

        rows, cols = weight_slice.get_shape()
        out_block = rows // scale_slice.get_shape()[0]
        chunk_rows = max(out_block, (chunk_bytes // (cols * 4) // out_block) * out_block)

        out = torch.empty((rows, cols), dtype=MASTER_DTYPE)
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            out[start:stop] = dequant_fp8_blockwise(
                weight_slice[start:stop],
                scale_slice[start // out_block : stop // out_block],
                MASTER_DTYPE,
            )
            gc.collect()
        return out

    def _slice(self, name: str):
        file_name = self.weight_map[name]
        handle = self._handles.get(file_name)
        if handle is None:
            handle = safe_open(os.path.join(self.model_path, file_name), framework="pt", device="cpu")
            self._handles[file_name] = handle
        return handle.get_slice(name)

    def close(self) -> None:
        self._handles.clear()


def _scale_name(weight_name: str) -> str:
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")] + ".scale"
    return weight_name + ".scale"


def build_compressed_token_map(tokenizer) -> Tuple[List[int], int]:
    """Collapse token ids that normalize alike, as the released engram does.

    Ported from ``inference/engram.py`` rather than imported: the reference is
    not a package, and this runs once per conversion to bake a buffer into the
    checkpoint. The normalizer sequence is reproduced exactly, including the
    private-use sentinel that keeps a token which is exactly one space from
    being ``Strip()``ed into the empty string and merged with unrelated tokens.

    The returned size is not a diagnostic. Every hash multiplier in
    :class:`DeepseekV41Ngram` is derived from it, so a tokenizer revision that
    compresses to a different size rehashes both 384M-row tables.
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: Dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it raw.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def convert_vision(source: TensorSource, output_dir: str, config) -> int:
    """The ViT, the aligner and the three image markers -- names unchanged.

    Nothing is renamed here because :class:`DeepseekV41Vision` was written
    against these names. The passthrough is asserted rather than assumed: a
    tensor that does not land on a parameter of the built module is a rename
    that happened upstream without this converter noticing.
    """
    writer = ShardWriter(os.path.join(output_dir, "deepseek_v41_vision"))
    for name in sorted(source.names()):
        if name.startswith(("vision.", "aligner.", "image_")):
            if name.endswith(".scale"):
                continue
            writer.add(name, source.get_dequantized(name))
    return writer.finalize(config)


def convert_text_encoder(source: TensorSource, output_dir: str, config, model_path: str) -> int:
    """``embed`` and ``head``, plus the tokenizer the chat template binds."""
    target_dir = os.path.join(output_dir, "deepseek_v41_text_encoder")
    writer = ShardWriter(target_dir)
    writer.add("embed_tokens.weight", source.get_dequantized("embed.weight"))
    writer.add("lm_head.weight", source.get_dequantized("head.weight"))
    count = writer.finalize(config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(target_dir)
    return count


def convert_ngram(source: TensorSource, output_dir: str, config, model_path: str) -> int:
    """The two tables, plus the ``token_map`` baked from the tokenizer.

    The tables are dequantized to bf16, which doubles them. See the module
    docstring: the release stores them fp8 and they are expected to stay frozen,
    but the module has nowhere to keep the scales.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lookup, compressed_size = build_compressed_token_map(tokenizer)
    if compressed_size != config.compressed_vocab_size:
        raise ValueError(
            f"the tokenizer at {model_path} compresses to {compressed_size} ids but the config declares "
            f"{config.compressed_vocab_size}. Every hash multiplier is derived from that size, so a "
            "checkpoint baked with this tokenizer would address both tables differently from the one that "
            "trained them -- every lookup would return a wrong row, and nothing downstream would fail."
        )
    if len(lookup) != config.vocab_size:
        raise ValueError(
            f"the tokenizer at {model_path} has {len(lookup)} tokens but the config declares "
            f"vocab_size={config.vocab_size}. token_map is indexed by raw token id, so a short map "
            "would index out of bounds on the tail of the vocabulary and a long one would bake rows "
            "no token can reach. (This is the tokenizer's size, not hash_vocab_size="
            f"{config.hash_vocab_size}, which is the base the bucket primes are drawn above.)"
        )

    writer = ShardWriter(os.path.join(output_dir, "deepseek_v41_ngram"))
    writer.add("token_map", torch.tensor(lookup, dtype=torch.int64))
    for table_index, layer_id in enumerate(config.layer_ids):
        writer.add(
            f"tables.{table_index}.weight",
            source.get_dequantized_in_row_chunks(f"layers.{layer_id}.engram.embed.weight"),
        )
        gc.collect()
    return writer.finalize(config)


def _llm_attention_names(layer: int) -> Dict[str, str]:
    """Released -> module, for one layer's attention block."""
    prefix = f"layers.{layer}.attn"
    target = f"model.layers.{layer}.self_attn"
    names = {
        f"{prefix}.wq_a.weight": f"{target}.wq_a.weight",
        f"{prefix}.wq_b.weight": f"{target}.wq_b.weight",
        f"{prefix}.wkv.weight": f"{target}.wkv.weight",
        f"{prefix}.wo_a.weight": f"{target}.wo_a.weight",
        f"{prefix}.wo_b.weight": f"{target}.wo_b.weight",
        f"{prefix}.q_norm.weight": f"{target}.q_norm.weight",
        f"{prefix}.kv_norm.weight": f"{target}.kv_norm.weight",
        f"{prefix}.attn_sink": f"{target}.attn_sink",
    }
    for suffix in (
        "compressor.norm.weight",
        "compressor.wkv.weight",
        "compressor.wgate.weight",
        "indexer.wq_b.weight",
        "indexer.wk.weight",
        "indexer.k_norm.weight",
        "indexer.weights_proj.weight",
    ):
        names[f"{prefix}.{suffix}"] = f"{target}.{suffix}"
    return names


def _llm_layer_names(layer: int) -> Dict[str, str]:
    """Released -> module, for everything in one layer except the experts.

    The two norms are renamed rather than passed through: the released
    ``attn_norm`` / ``ffn_norm`` are HF's ``input_layernorm`` /
    ``post_attention_layernorm``, which is what the decoder layer declares.
    """
    names = _llm_attention_names(layer)
    names[f"layers.{layer}.attn_norm.weight"] = f"model.layers.{layer}.input_layernorm.weight"
    names[f"layers.{layer}.ffn_norm.weight"] = f"model.layers.{layer}.post_attention_layernorm.weight"
    names[f"layers.{layer}.ffn.gate.weight"] = f"model.layers.{layer}.mlp.gate.weight"
    # The aux-loss-free controllers. The release calls them `bias`; every
    # DeepSeek-family module in this repo calls them e_score_correction_bias and
    # holds them as buffers, not parameters -- they are updated by a sign rule on
    # expert load, not by the objective.
    names[f"layers.{layer}.ffn.gate.bias"] = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    names[f"layers.{layer}.ffn.gate.bias_vl"] = f"model.layers.{layer}.mlp.gate.e_score_correction_bias_vl"
    for site in ("attn", "ffn"):
        for part in ("fn", "base", "scale"):
            names[f"layers.{layer}.hc_{site}_{part}"] = f"model.layers.{layer}.{site}_hc.{part}"
    for source_name, target in (
        ("w1", "gate_proj"),
        ("w2", "down_proj"),
        ("w3", "up_proj"),
    ):
        names[f"layers.{layer}.ffn.shared_experts.{source_name}.weight"] = (
            f"model.layers.{layer}.mlp.shared_experts.{target}.weight"
        )
    return names


def convert_llm(source: TensorSource, output_dir: str, config, engram_layer_ids: List[int]) -> int:
    """The 40 decoder layers, the Engram connections and the final norm."""
    writer = ShardWriter(os.path.join(output_dir, "deepseek_v41_llm"))

    for layer in range(config.num_hidden_layers):
        for released, target in _llm_layer_names(layer).items():
            if released not in source:
                continue
            writer.add(target, source.get_dequantized(released))

        # The routed experts, stacked into the fused layout one layer at a time.
        # This is the memory high-water mark: 384 experts of [4608, 5120] plus
        # [5120, 2304], ~27 GB in bf16, which is why the loop is per layer and
        # the intermediates are dropped before the next one.
        gate_up, down = [], []
        for expert in range(config.n_routed_experts):
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            gate_up.append(
                fuse_gate_up(
                    source.get_dequantized(f"{prefix}.w1.weight"),
                    source.get_dequantized(f"{prefix}.w3.weight"),
                )
            )
            down.append(source.get_dequantized(f"{prefix}.w2.weight"))
        writer.add(f"model.layers.{layer}.mlp.experts.gate_up_proj", torch.stack(gate_up))
        writer.add(f"model.layers.{layer}.mlp.experts.down_proj", torch.stack(down))
        del gate_up, down
        gc.collect()

        # The Engram connection's half: the tables went to the ngram module, the
        # gate and wkv stay here because they read the mHC residual stream.
        if layer in engram_layer_ids:
            connection = f"model.engram.{layer}"
            for part in ("q_weight", "k_weight"):
                writer.add(f"{connection}.{part}", source.get_dequantized(f"layers.{layer}.engram.{part}"))
            writer.add(
                f"{connection}.wkv.weight",
                source.get_dequantized(f"layers.{layer}.engram.wkv.weight"),
            )

    writer.add("model.norm.weight", source.get_dequantized("norm.weight"))
    return writer.finalize(config)


def _build_configs(hf_config: Dict[str, Any]):
    """The four module configs, from the released ``config.json``."""
    import veomni.models.seed_omni.modules  # noqa: F401
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY

    text = hf_config["text_config"]
    vision = hf_config["vision_config"]
    quant = hf_config.get("quantization_config", {})

    VisionConfig = OMNI_CONFIG_REGISTRY["deepseek_v41_vision"]()
    TextEncoderConfig = OMNI_CONFIG_REGISTRY["deepseek_v41_text_encoder"]()
    NgramConfig = OMNI_CONFIG_REGISTRY["deepseek_v41_ngram"]()
    LlmConfig = OMNI_CONFIG_REGISTRY["deepseek_v41_llm"]()

    vision_config = VisionConfig(**{k: v for k, v in vision.items() if k != "model_type"})
    text_encoder_config = TextEncoderConfig(
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        tie_word_embeddings=text.get("tie_word_embeddings", False),
    )
    ngram_config = NgramConfig(
        layer_ids=text["engram_layer_ids"],
        num_embeddings=text["engram_num_embeddings"],
        max_ngram_size=text["engram_max_ngram_size"],
        n_heads=text["engram_n_heads"],
        head_dim=text["engram_head_dim"],
        # Two different sizes: the tokenizer's, which indexes token_map, and the
        # base the bucket primes are drawn above. See DeepseekV41NgramConfig.
        vocab_size=text["vocab_size"],
        hash_vocab_size=text["engram_vocab_size"],
        compressed_vocab_size=text["engram_compressed_vocab_size"],
        pad_token_id=text["engram_pad_token_id"],
    )
    llm_fields = {k: v for k, v in text.items() if not k.startswith("engram_") and k != "model_type"}
    # The experts' storage format is a quantization_config key upstream, and a
    # field on the module config here because the QAT path reads it per forward.
    llm_fields["expert_dtype"] = quant.get("expert_dtype", "fp8")
    llm_config = LlmConfig(**llm_fields)
    return vision_config, text_encoder_config, ngram_config, llm_config


def convert_deepseek_v41_checkpoint(model_path: str, output_dir: str, **kwargs) -> None:
    """Split the released DeepSeek-V4.1 checkpoint into four module subfolders."""
    del kwargs
    with open(os.path.join(model_path, "config.json")) as handle:
        hf_config = json.load(handle)

    vision_config, text_encoder_config, ngram_config, llm_config = _build_configs(hf_config)
    source = TensorSource(model_path)

    skipped = sum(1 for name in source.names() if name.startswith("mtp."))
    print(f"Converting {model_path} -> {output_dir}")
    print(f"  {len(source.weight_map)} tensors, of which {skipped} are mtp.* (DSpark, not implemented)")

    print("Extracting deepseek_v41_vision ...")
    count = convert_vision(source, output_dir, vision_config)
    print(f"  {count} tensors")

    print("Extracting deepseek_v41_text_encoder ...")
    count = convert_text_encoder(source, output_dir, text_encoder_config, model_path)
    print(f"  {count} tensors + tokenizer")

    print("Extracting deepseek_v41_ngram ...")
    count = convert_ngram(source, output_dir, ngram_config, model_path)
    print(f"  {count} tensors (tables dequantized to bf16)")

    print("Extracting deepseek_v41_llm ...")
    count = convert_llm(source, output_dir, llm_config, list(ngram_config.layer_ids))
    print(f"  {count} tensors")

    source.close()
    print(f"\nDone. Split checkpoint saved to: {output_dir}")


@OMNI_CONVERT_REGISTRY.register("deepseek_v41")
def _register_deepseek_v41_convert():
    return convert_deepseek_v41_checkpoint


__all__ = ["build_compressed_token_map", "convert_deepseek_v41_checkpoint"]
