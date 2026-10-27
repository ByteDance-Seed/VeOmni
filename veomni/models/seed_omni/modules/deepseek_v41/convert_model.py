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

"""Split the released DeepSeek-V4.1 checkpoint into SeedOmni modules.

Registered under ``OMNI_CONVERT_REGISTRY["deepseek_v41"]`` and dispatched by
``scripts/seed_omni/convert_model.py``. To rewrite only some subfolders, pass
``--extra modules=deepseek_v41_ngram,deepseek_v41_llm``.

Unlike the other families' converters this one never builds the model. The
released checkpoint is 510 GB across 48 shards and 96,085 tensors. Tensors are
read through the checkpoint's own index, renamed into the four SeedOmni
modules, and flushed to output shards as they go, so peak memory is one MoE
layer's experts rather than one model. Dtypes stay as released: training
dequantizes at load through ``DeepseekV41DequantConverter``.

Naming
------
The released checkpoint is already in the *reference inference stack's* naming
(``layers.N.attn.wq_a``, ``ffn.gate.bias``, ``hc_attn_fn``), not HuggingFace's.
``inference/convert.py``'s renames -- ``self_attn`` -> ``attn``, ``mlp`` ->
``ffn``, ``e_score_correction_bias`` -> ``bias`` -- have already been applied to
it, so this converter walks them back rather than forward. The vision tower is
the exception: its parameter names match :class:`DeepseekV41Vision` exactly and
pass through untouched.

Precision
---------
Dense projections are fp8 with a 32x32 block scale, routed experts are fp4
packed two to a byte with a block scale of their own, and the two Engram tables
are fp8 with a scale every 32 columns. This converter **does not change that**.
Each scale rides with its weight as ``P + "_scale"``
(``...wq_a.weight`` -> ``...wq_a.weight_scale``,
``...experts.gate_up_proj`` -> ``...gate_up_proj_scale``). Training spends the
scale at load; inference keeps the quantized modules. See ``dequant.py`` and
``checkpoint_tensor_converter.py``.

Not converted
-------------
Tied ``mtp.*.embed.weight`` / ``mtp.*.head.weight`` alias the text encoder and
are skipped. Everything else under ``mtp.*`` lands in ``deepseek_v41_llm``.

Output
------
::

    <output_dir>/
      deepseek_v41_vision/        # ViT + aligner + the three image marker embeddings
      deepseek_v41_text_encoder/  # embed_tokens + lm_head + tokenizer
      deepseek_v41_ngram/         # the two n-gram tables + the baked token_map
      deepseek_v41_llm/           # 40 decoder layers, Engram connections, DSpark, final norm
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY, load_family_graphs
from .dequant import fuse_gate_up
from .released_names import (
    MODULES,
    TEXT_ENCODER_NAMES,
    VISION_PREFIXES,
    engram_connection_names,
    llm_layer_names,
    mtp_extra_names,
    mtp_layer_names,
    ngram_table_names,
    released_scale_name,
)


#: Flush an output shard once it reaches this many bytes. Matches the release's
#: own shard size closely enough that the output file count is comparable.
SHARD_BYTES = 5 * 1000**3


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
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._weight_map: dict[str, int] = {}
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

    def finalize(self, config: Any | None = None) -> int:
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
            self.weight_map: dict[str, str] = json.load(handle)["weight_map"]
        self._handles: dict[str, Any] = {}

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

    def get_released(self, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A weight in the dtype it shipped in, and its scale, unmodified.

        ``None`` means the tensor was never quantized -- norms, biases, the
        sink -- not that its scale is missing. Training spends the scale later,
        in ``DeepseekV41DequantConverter``.
        """
        weight = self.get(name)
        scale_name = released_scale_name(name)
        if scale_name not in self.weight_map:
            return weight, None
        return weight, self.get(scale_name)

    def close(self) -> None:
        self._handles.clear()


class Emitter:
    """Reads a released tensor, renames it, and writes it at released precision.

    Exists so the four ``convert_*`` functions stay a list of renames. They
    describe *what* maps to what, which is the part a reader has to check
    against the module definitions; a scale riding along is a property of the
    source tensor, not of a conversion mode.
    """

    def __init__(self, source: TensorSource, writer: ShardWriter) -> None:
        self.source = source
        self.writer = writer

    def emit(self, target: str, released: str) -> None:
        weight, scale = self.source.get_released(released)
        self.writer.add(target, weight)
        if scale is not None:
            self.writer.add(f"{target}_scale", scale)

    def emit_tensor(self, target: str, tensor: torch.Tensor, scale: torch.Tensor | None = None) -> None:
        """For tensors the converter builds rather than reads -- fused, stacked, baked."""
        self.writer.add(target, tensor)
        if scale is not None:
            self.writer.add(f"{target}_scale", scale)


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
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
    key_to_new: dict[str, int] = {}
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

    The image processor is written out alongside them. The release does not
    ship a ``preprocessor_config.json`` -- its reference runner reads the
    resize policy off the model config instead -- but a split module is loaded
    through ``from_pretrained``, which looks for that file and fails without
    it. The five values are copied from ``vision_config``, so the artifact
    states the policy the weights were trained under rather than relying on
    this class's defaults still agreeing with it.
    """
    target_dir = os.path.join(output_dir, "deepseek_v41_vision")
    writer = ShardWriter(target_dir)
    emitter = Emitter(source, writer)
    for name in sorted(source.names()):
        if name.startswith(VISION_PREFIXES):
            if name.endswith(".scale"):
                continue
            emitter.emit(name, name)
    count = writer.finalize(config)
    build_vision_image_processor(config).save_pretrained(target_dir)
    return count


def build_vision_image_processor(config):
    """The image processor, with the resize policy copied from ``vision_config``."""
    from .vision.processing import DeepseekV41VisionImageProcessor

    return DeepseekV41VisionImageProcessor(
        patch_size=config.patch_size,
        downsample_ratio=config.downsample_ratio,
        max_image_tokens=config.max_image_tokens,
        min_pixels=config.min_pixels,
        max_wh_ratio=config.max_wh_ratio,
    )


def convert_text_encoder(source: TensorSource, output_dir: str, config, model_path: str) -> int:
    """``embed`` and ``head``, plus the tokenizer the chat template binds."""
    target_dir = os.path.join(output_dir, "deepseek_v41_text_encoder")
    writer = ShardWriter(target_dir)
    emitter = Emitter(source, writer)
    for released, target in TEXT_ENCODER_NAMES.items():
        emitter.emit(target, released)
    count = writer.finalize(config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(target_dir)
    return count


def convert_ngram(source: TensorSource, output_dir: str, config, model_path: str) -> int:
    """The two tables, plus the ``token_map`` baked from the tokenizer.

    Each table is a single ``[384M, 256]`` fp8 tensor -- 91.6 GiB that no shard
    boundary can split, since safetensors stores a tensor whole. Copied as it
    lies, with the ``[384M, 8]`` scale beside it. The n-gram module keeps the
    tables frozen in fp8 for both training and inference.
    """
    token_map = build_ngram_token_map(model_path, config)
    writer = ShardWriter(os.path.join(output_dir, "deepseek_v41_ngram"))
    emitter = Emitter(source, writer)
    writer.add("token_map", token_map)
    for released, target in ngram_table_names(config.layer_ids).items():
        emitter.emit(target, released)
        gc.collect()
    return writer.finalize(config)


def build_ngram_token_map(model_path: str, config) -> torch.Tensor:
    """The ``token_map`` the n-gram module hashes through, from the tokenizer at ``model_path``.

    Checked against the config's sizes, which the tables were trained under.
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
    return torch.tensor(lookup, dtype=torch.int64)


def convert_llm(source: TensorSource, output_dir: str, config, engram_layer_ids: list[int]) -> int:
    """The 40 decoder layers, DSpark stages, the Engram connections and the final norm."""
    writer = ShardWriter(os.path.join(output_dir, "deepseek_v41_llm"))
    emitter = Emitter(source, writer)

    for layer in range(config.num_hidden_layers):
        for released, target in llm_layer_names(layer).items():
            if released not in source:
                continue
            emitter.emit(target, released)

        _convert_layer_experts(emitter, layer, config.n_routed_experts)

        # The Engram connection's half: the tables went to the ngram module, the
        # gate and wkv stay here because they read the mHC residual stream.
        if layer in engram_layer_ids:
            for released, target in engram_connection_names(layer).items():
                emitter.emit(target, released)

    _convert_mtp(emitter, source, config)
    emitter.emit("model.norm.weight", "norm.weight")
    return writer.finalize(config)


def _convert_mtp(emitter: Emitter, source: TensorSource, config) -> None:
    """DSpark stages under ``mtp.*``. Tied embed/head are skipped on purpose."""
    n_stages = config.num_nextn_predict_layers
    if n_stages <= 0 or "mtp.0.attn.wq_a.weight" not in source:
        return
    n_experts = config.dspark_n_routed_experts or config.n_routed_experts
    for stage in range(n_stages):
        for released, target in mtp_layer_names(stage).items():
            if released not in source:
                continue
            emitter.emit(target, released)
        _convert_layer_experts(emitter, stage, n_experts, source_prefix="mtp", dest_prefix="model.mtp")

    for released, target in mtp_extra_names(n_stages).items():
        if released in source:
            emitter.emit(target, released)


def _convert_layer_experts(
    emitter: Emitter,
    layer: int,
    n_routed_experts: int,
    source_prefix: str = "layers",
    dest_prefix: str = "model.layers",
) -> None:
    """One layer's routed experts, fused and stacked into the module's layout.

    This is the memory high-water mark, which is why the caller loops per layer
    and these intermediates are dropped before the next one: 384 experts of
    packed fp4, not the dequantized slab.

    The fusion is the same ``cat`` on dim 0 for a weight and for its scale. fp4
    packs two values per byte along the *last* dimension, and a block scale
    covers 32 columns of one row, so both tensors are indexed by output row and
    both concatenate along it -- the gate rows keep their scales and the up rows
    keep theirs.
    """
    prefix = f"{source_prefix}.{layer}.ffn.experts"
    target = f"{dest_prefix}.{layer}.mlp.experts"

    gate_up, gate_up_scale, down, down_scale = [], [], [], []
    for expert in range(n_routed_experts):
        expert_prefix = f"{prefix}.{expert}"
        w1, w1_scale = emitter.source.get_released(f"{expert_prefix}.w1.weight")
        w3, w3_scale = emitter.source.get_released(f"{expert_prefix}.w3.weight")
        w2, w2_scale = emitter.source.get_released(f"{expert_prefix}.w2.weight")
        if w1_scale is None or w3_scale is None or w2_scale is None:
            raise ValueError(
                f"layer {layer} expert {expert} has no scale. Routed experts are quantized in every "
                "released DeepSeek-V4.1 checkpoint, so a missing scale means the source is not the "
                "release this converter targets."
            )
        gate_up.append(fuse_gate_up(w1, w3))
        gate_up_scale.append(fuse_gate_up(w1_scale, w3_scale))
        down.append(w2)
        down_scale.append(w2_scale)
    emitter.emit_tensor(f"{target}.gate_up_proj", torch.stack(gate_up), torch.stack(gate_up_scale))
    emitter.emit_tensor(f"{target}.down_proj", torch.stack(down), torch.stack(down_scale))
    del gate_up, gate_up_scale, down, down_scale
    gc.collect()


def _build_configs(hf_config: dict[str, Any]):
    """The four module configs, from the released ``config.json``."""
    import veomni.models.seed_omni.modules  # noqa: F401
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY

    text = hf_config["text_config"]
    vision = hf_config["vision_config"]
    quant = hf_config.get("quantization_config", {}) or None

    VisionConfig = OMNI_CONFIG_REGISTRY["deepseek_v41_vision"]()
    TextEncoderConfig = OMNI_CONFIG_REGISTRY["deepseek_v41_text_encoder"]()
    NgramConfig = OMNI_CONFIG_REGISTRY["deepseek_v41_ngram"]()
    LlmConfig = OMNI_CONFIG_REGISTRY["deepseek_v41_llm"]()

    vision_config = VisionConfig(**{k: v for k, v in vision.items() if k != "model_type"})
    text_encoder_config = TextEncoderConfig(
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        tie_word_embeddings=text.get("tie_word_embeddings", False),
        dspark_block_size=text.get("dspark_block_size", 5),
        dspark_noise_token_id=text.get("dspark_noise_token_id", 128799),
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
        weight_quantization=dict(quant) if quant else None,
    )
    # The llm declares the engram fields that shape its connection to the
    # tables; the rest describe the tables and belong to the ngram config.
    llm_engram = {"engram_layer_ids", "engram_max_ngram_size", "engram_n_heads", "engram_head_dim"}
    llm_fields = {
        k: v for k, v in text.items() if (not k.startswith("engram_") or k in llm_engram) and k != "model_type"
    }
    # The experts' storage format is a quantization_config key upstream, and a
    # field on the module config here because the QAT path reads it per forward.
    llm_fields["expert_dtype"] = (quant or {}).get("expert_dtype", "fp8")
    # Declares that the files still hold the released dtypes. Training clears
    # this on the in-memory config so the modules allocate bf16 and the load
    # converter spends the scales.
    llm_fields["weight_quantization"] = dict(quant) if quant else None
    llm_config = LlmConfig(**llm_fields)
    return vision_config, text_encoder_config, ngram_config, llm_config


def convert_deepseek_v41_checkpoint(
    model_path: str,
    output_dir: str,
    modules: Iterable[str] | str | None = None,
    training_graph: str | None = None,
    generation_graph: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Stream the released DeepSeek-V4.1 checkpoint into four module subfolders.

    Dtypes stay as released. Training dequantizes at load. Takes ``output_dir``,
    so :func:`~veomni.models.seed_omni.utils.convert_registry.convert_checkpoint`
    leaves the subfolders to this function and writes only the root config and
    graph sidecars from what it returns.

    Args:
        modules: Which subfolders to write, defaulting to all four; a
            comma-separated string as ``--extra modules=...`` passes it. A
            folder is self-contained, so converting one does not invalidate the
            others -- which is the point, since the experts are 269 GiB and a
            rerun to pick up a fix in, say, the n-gram ``token_map`` should not
            repay that. The root config always names all four.
    """
    if "precision" in kwargs:
        raise TypeError(
            "convert_deepseek_v41_checkpoint no longer takes precision=; it always keeps the "
            "released dtypes. Training dequantizes at load via DeepseekV41DequantConverter."
        )
    if kwargs:
        raise TypeError(f"unexpected convert kwargs {sorted(kwargs)}")
    if isinstance(modules, str):
        modules = [name.strip() for name in modules.split(",") if name.strip()]
    selected = set(modules) if modules is not None else set(MODULES)
    unknown = selected - set(MODULES)
    if unknown:
        raise ValueError(f"unknown modules {sorted(unknown)}; expected a subset of {list(MODULES)}")

    training_graphs, generation_graphs = load_family_graphs(
        "configs/seed_omni/DeepSeek/deepseek_v41_flash",
        training="train/graph_train.yaml",
        generation={"vision_understanding": "infer/graph_infer.yaml"},
        training_graph=training_graph,
        generation_graph=generation_graph,
    )

    with open(os.path.join(model_path, "config.json")) as handle:
        hf_config = json.load(handle)

    vision_config, text_encoder_config, ngram_config, llm_config = _build_configs(hf_config)
    source = TensorSource(model_path)

    tied_mtp = sum(
        1
        for name in source.names()
        if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight")
    )
    print(f"Converting {model_path} -> {output_dir}")
    print(f"  modules={sorted(selected)}")
    print(f"  {len(source.weight_map)} tensors ({tied_mtp} tied mtp embed/head skipped)")

    if "deepseek_v41_vision" in selected:
        print("Extracting deepseek_v41_vision ...")
        count = convert_vision(source, output_dir, vision_config)
        print(f"  {count} tensors")

    if "deepseek_v41_text_encoder" in selected:
        print("Extracting deepseek_v41_text_encoder ...")
        count = convert_text_encoder(source, output_dir, text_encoder_config, model_path)
        print(f"  {count} tensors + tokenizer")

    if "deepseek_v41_ngram" in selected:
        print("Extracting deepseek_v41_ngram ...")
        count = convert_ngram(source, output_dir, ngram_config, model_path)
        print(f"  {count} tensors (tables kept fp8)")

    if "deepseek_v41_llm" in selected:
        print("Extracting deepseek_v41_llm ...")
        count = convert_llm(source, output_dir, llm_config, list(ngram_config.layer_ids))
        print(f"  {count} tensors")

    # The root config loads every module's config.json, including the ones a
    # partial run skipped; a folder converted earlier keeps the config it has.
    configs = dict(zip(MODULES, (vision_config, text_encoder_config, ngram_config, llm_config)))
    for name in sorted(set(MODULES) - selected):
        if not os.path.isfile(os.path.join(output_dir, name, "config.json")):
            configs[name].save_pretrained(os.path.join(output_dir, name))

    source.close()
    return {
        "module_names": list(MODULES),
        "training_graphs": training_graphs,
        "generation_graphs": generation_graphs,
        "infer_type": "vision_understanding",
    }


@OMNI_CONVERT_REGISTRY.register("deepseek_v41")
def _register_deepseek_v41_convert():
    return convert_deepseek_v41_checkpoint


__all__ = [
    "build_compressed_token_map",
    "build_ngram_token_map",
    "build_vision_image_processor",
    "convert_deepseek_v41_checkpoint",
]
