"""Teacher-forcing forward over the converted DeepSeek-V4.1-Flash checkpoint.

The backbone is 1.1 TB in bf16 and this box has 765 GB of GPU memory, so no
arrangement holds every layer at once. A single forward pass does not need one:
layers run in order and nothing reads a layer's weights after it has run. Each
layer's parameters are read from the shards just before it runs and released
right after, so the peak is one layer -- about 28 GB -- instead of the whole
backbone. CPU is not an option here because the sparse attention is a TileLang
GPU kernel.

The measurement is cross-entropy on coherent natural text. A correctly converted
trained model puts it low; an error anywhere in the chain -- dequantisation,
tensor naming, n-gram row addressing, mHC wiring, MoE routing -- pushes it back
to ln(vocab_size) = 11.77, because a model that has lost its weights can only
predict the uniform distribution. It is one number, and it is not one that
passes by accident.

Run ``--self-test`` first: it builds a small random model, runs it both streamed
and fully resident, and requires the two to agree bit for bit. That separates a
bug in this harness from a bug in the port.

    python scripts/deepseek_v41/streaming_forward.py --self-test
    python scripts/deepseek_v41/streaming_forward.py \
        --checkpoint /mnt/hdfs/veomni/models/seed_omni/DeepSeek-V4.1-Flash
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors import safe_open

from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Llm
from veomni.models.seed_omni.modules.deepseek_v41.ngram.configuration import DeepseekV41NgramConfig
from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram


class ShardIndex:
    """Tensor name -> the shard holding it, with handles kept open.

    Accepts both layouts the converter emits: a lone ``model.safetensors`` and a
    sharded set with an index. Handles are cached because a streamed forward
    reopens the same few files once per layer.
    """

    def __init__(self, module_dir: str) -> None:
        self.module_dir = module_dir
        index_path = os.path.join(module_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            self.weight_map: Dict[str, str] = json.load(open(index_path))["weight_map"]
        else:
            with safe_open(os.path.join(module_dir, "model.safetensors"), framework="pt") as handle:
                self.weight_map = dict.fromkeys(handle.keys(), "model.safetensors")
        self._handles: Dict[str, object] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def names(self) -> Iterable[str]:
        return self.weight_map.keys()

    def get(self, name: str) -> torch.Tensor:
        file_name = self.weight_map[name]
        handle = self._handles.get(file_name)
        if handle is None:
            handle = safe_open(os.path.join(self.module_dir, file_name), framework="pt", device="cpu")
            self._handles[file_name] = handle
        return handle.get_tensor(name)

    def get_rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Gather individual rows without reading the whole tensor.

        The n-gram tables are 197 GB each and a short prompt touches a few
        thousand rows, so reading them whole to use 0.001% would dominate
        everything else this script does.
        """
        file_name = self.weight_map[name]
        handle = self._handles.get(file_name)
        if handle is None:
            handle = safe_open(os.path.join(self.module_dir, file_name), framework="pt", device="cpu")
            self._handles[file_name] = handle
        table = handle.get_slice(name)
        order = torch.argsort(rows)  # ascending reads are kinder to the mount
        gathered = torch.empty((rows.numel(), table.get_shape()[1]), dtype=torch.bfloat16)
        for position in order.tolist():
            row = int(rows[position])
            gathered[position] = table[row : row + 1][0]
        return gathered


def _buffers_to(model: nn.Module, device: torch.device) -> None:
    """Move every buffer to ``device``, leaving meta parameters alone.

    ``Module.to`` would try to move the parameters too and fail on meta.
    """
    for name, buffer in list(model.named_buffers()):
        if buffer is None or buffer.is_meta:
            continue
        owner, attr = _owner_and_attr(model, name)
        setattr(owner, attr, buffer.to(device))


def _owner_and_attr(root: nn.Module, qualified: str) -> Tuple[nn.Module, str]:
    parts = qualified.split(".")
    owner = root
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


def materialize(module: nn.Module, prefix: str, index: ShardIndex, device: torch.device) -> int:
    """Replace ``module``'s meta parameters and buffers with real ones on ``device``."""
    moved = 0
    for name, _ in list(module.named_parameters(recurse=True)):
        tensor = index.get(prefix + name).to(device)
        owner, attr = _owner_and_attr(module, name)
        owner._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
        moved += 1
    for name, buffer in list(module.named_buffers(recurse=True)):
        if prefix + name not in index:
            # Derived rather than stored -- rebuilt at construction, so a meta
            # placeholder here would be a real bug rather than a missing file.
            if buffer.is_meta:
                raise RuntimeError(f"{prefix + name} is on meta and absent from the checkpoint")
            continue
        owner, attr = _owner_and_attr(module, name)
        setattr(owner, attr, index.get(prefix + name).to(device))
        moved += 1
    return moved


def release(module: nn.Module) -> None:
    """Send the module's parameters back to meta, freeing their storage."""
    for name, param in list(module.named_parameters(recurse=True)):
        owner, attr = _owner_and_attr(module, name)
        owner._parameters[attr] = nn.Parameter(
            torch.empty(param.shape, dtype=param.dtype, device="meta"), requires_grad=False
        )


def attach_layer_streaming(model: DeepseekV41Llm, index: ShardIndex, device: torch.device) -> List[object]:
    """Load each decoder layer on entry and free it on exit.

    Correct only because this is a forward pass with no backward: a released
    layer's weights are never read again. Under autograd the released storage
    would be needed for the gradient, which is why this is a verification
    harness and not a training path.
    """
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_idx}."

        def load(module, _args, _prefix=prefix, _idx=layer_idx):
            materialize(module, _prefix, index, device)

        def unload(module, _args, _output):
            release(module)
            torch.cuda.empty_cache()

        handles.append(layer.register_forward_pre_hook(load))
        handles.append(layer.register_forward_hook(unload))
    return handles


def build_ngram_embeds(
    ngram_dir: str,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[List[torch.Tensor], DeepseekV41NgramConfig]:
    """Hash the prompt and gather only the table rows it addresses."""
    raw = {k: v for k, v in json.load(open(os.path.join(ngram_dir, "config.json"))).items() if k != "model_type"}
    config = DeepseekV41NgramConfig(**raw)
    index = ShardIndex(ngram_dir)

    # Built on meta so the two 197 GB tables allocate nothing -- this path never
    # calls `lookup`, it gathers rows itself. The derived buffers would be meta
    # too, so they are rebuilt here with the same functions the module uses.
    # `token_map` is the one buffer that must come from the checkpoint rather
    # than be re-derived, since a different tokenizer revision would rehash both
    # tables while still indexing in range.
    from veomni.models.seed_omni.modules.deepseek_v41.ngram.hashing import (
        build_bucket_offsets,
        build_bucket_primes,
        build_hash_multipliers,
    )

    with torch.device("meta"):
        module = DeepseekV41Ngram(config)
    primes = build_bucket_primes(len(config.layer_ids), config.max_ngram_size, config.n_heads, config.hash_vocab_size)
    module.primes = torch.tensor(primes)
    module.offsets = build_bucket_offsets(primes)
    module.multipliers = build_hash_multipliers(config.layer_ids, config.max_ngram_size, config.compressed_vocab_size)
    module.token_map = index.get("token_map")

    hash_ids = module.hash_ids(input_ids.cpu())
    batch, seq_len, n_tables, cols = hash_ids.shape
    embeds = []
    for table_idx in range(n_tables):
        flat = hash_ids[:, :, table_idx].reshape(-1)
        rows = index.get_rows(f"tables.{table_idx}.weight", flat)
        embeds.append(rows.view(batch, seq_len, cols, config.head_dim).to(device))
    return embeds, config


def _tiny_config(n_layers: int = 4) -> DeepseekV41LlmConfig:
    """Small, but not smaller than the TileLang sparse-attention kernel accepts.

    The kernel derives ``warp_col_tiles`` from ``head_dim`` and asserts it is at
    least 8, so anything under ``head_dim=64`` fails to compile. That is why the
    CPU unit tests stub the layers out instead of shrinking them: there is no
    eager fallback to shrink into.
    """
    return DeepseekV41LlmConfig(
        hidden_size=256,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        head_dim=64,
        qk_rope_head_dim=16,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        sliding_window=8,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=8,
        rms_norm_eps=1e-20,
        hc_mult=4,
        hc_sinkhorn_iters=6,
        # The released schedule lists every layer, the three draft stages included.
        compress_ratios=[2] * n_layers + [0, 0, 0],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        engram_layer_ids=[1],
        dspark_target_layer_ids=[],
        max_position_embeddings=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_shared_experts=1,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=8,
        # Wide enough to contain the released marker ids, so the self-test is not
        # buried under warnings about ids outside a toy vocabulary.
        vocab_size=129280,
    )


def self_test(device: torch.device, tmp_dir: str) -> None:
    """Streamed and fully resident must agree bit for bit on a small model."""
    torch.manual_seed(0)
    config = _tiny_config()
    resident = DeepseekV41Llm(config).to(device=device, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for param in resident.parameters():
            param.normal_(0, 0.02)

    os.makedirs(tmp_dir, exist_ok=True)
    from safetensors.torch import save_file

    save_file(
        {k: v.cpu().contiguous() for k, v in resident.state_dict().items()},
        os.path.join(tmp_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )

    seq_len = 24
    inputs_embeds = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
    ngram = [
        torch.randn(1, seq_len, cols, config.engram_head_dim, device=device, dtype=torch.bfloat16)
        for _ in config.engram_layer_ids
    ]

    with torch.no_grad():
        want = resident(inputs_embeds=inputs_embeds, position_ids=position_ids, ngram_embeds=ngram)["hidden_states"]
    del resident
    torch.cuda.empty_cache()

    # Parameters on meta, buffers real: the rotary tables and the like are
    # derived rather than stored, so they are absent from the shards and would
    # stay on meta under a plain `torch.device("meta")`, failing mid-forward.
    with init_empty_weights(include_buffers=False):
        streamed = DeepseekV41Llm(config)
    streamed = streamed.to(dtype=torch.bfloat16).eval()
    _buffers_to(streamed, device)
    index = ShardIndex(tmp_dir)
    layer_prefixes = tuple(f"model.layers.{i}." for i in range(config.num_hidden_layers))
    for name in index.names():
        if name.startswith(layer_prefixes):
            continue
        owner, attr = _owner_and_attr(streamed, name)
        if attr in owner._parameters:
            owner._parameters[attr] = nn.Parameter(index.get(name).to(device), requires_grad=False)
        else:
            setattr(owner, attr, index.get(name).to(device))
    attach_layer_streaming(streamed, index, device)

    with torch.no_grad():
        got = streamed(inputs_embeds=inputs_embeds, position_ids=position_ids, ngram_embeds=ngram)["hidden_states"]

    identical = torch.equal(want, got)
    print(f"self-test: streamed vs resident bit-identical = {identical}")
    if not identical:
        delta = (want.float() - got.float()).abs().max().item()
        raise SystemExit(f"self-test FAILED, max |delta| = {delta}")


PASSAGE = (
    "The history of computing is often told as a story of hardware, but the "
    "decisive changes were usually changes in how people described problems to "
    "machines. Assembly language let a programmer name an operation instead of "
    "a number. The compiler let a programmer describe a computation instead of "
    "a sequence of operations. Each step traded some control for a great deal "
    "of leverage, and each was resisted by people who had mastered the level "
    "below it."
)


def load_llm(checkpoint: str, device: torch.device) -> Tuple[DeepseekV41Llm, ShardIndex]:
    """Real config, parameters on meta, everything outside the layers resident."""
    llm_dir = os.path.join(checkpoint, "deepseek_v41_llm")
    raw = {k: v for k, v in json.load(open(os.path.join(llm_dir, "config.json"))).items() if k != "model_type"}
    config = DeepseekV41LlmConfig(**raw)
    index = ShardIndex(llm_dir)

    with init_empty_weights(include_buffers=False):
        model = DeepseekV41Llm(config)
    model = model.to(dtype=torch.bfloat16).eval()
    _buffers_to(model, device)

    layer_prefixes = tuple(f"model.layers.{i}." for i in range(config.num_hidden_layers))
    resident = 0
    for name in index.names():
        if name.startswith(layer_prefixes):
            continue
        owner, attr = _owner_and_attr(model, name)
        tensor = index.get(name).to(device)
        if attr in owner._parameters:
            owner._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
        else:
            setattr(owner, attr, tensor)
        resident += 1
    print(f"  resident outside the layers: {resident} tensors, {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    attach_layer_streaming(model, index, device)
    return model, index


@torch.no_grad()
def score(
    checkpoint: str,
    model: DeepseekV41Llm,
    embed: torch.Tensor,
    head: torch.Tensor,
    input_ids: torch.Tensor,
    device: torch.device,
) -> float:
    """Mean next-token cross-entropy, in nats."""
    ngram_embeds, _ = build_ngram_embeds(os.path.join(checkpoint, "deepseek_v41_ngram"), input_ids, device)
    ngram_embeds = [t.to(torch.bfloat16) for t in ngram_embeds]

    inputs_embeds = embed[input_ids.to(device)].to(torch.bfloat16)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    hidden = model(inputs_embeds=inputs_embeds, position_ids=position_ids, ngram_embeds=ngram_embeds)["hidden_states"]
    logits = (hidden[0, :-1].float() @ head.float().T).float()
    targets = input_ids[0, 1:].to(device)
    return torch.nn.functional.cross_entropy(logits, targets).item()


def run_real(checkpoint: str, text: str, max_tokens: int, device: torch.device) -> None:
    from transformers import AutoTokenizer

    text_dir = os.path.join(checkpoint, "deepseek_v41_text_encoder")
    tokenizer = AutoTokenizer.from_pretrained(text_dir, trust_remote_code=True)
    text_index = ShardIndex(text_dir)
    embed = text_index.get("embed_tokens.weight").to(device)
    head = text_index.get("lm_head.weight").to(device)
    print(f"text_encoder: embed {tuple(embed.shape)}, head {tuple(head.shape)}")

    input_ids = torch.tensor([tokenizer.encode(text)[:max_tokens]])
    print(f"prompt: {input_ids.shape[1]} tokens")

    print("building the backbone (parameters stream in per layer) ...")
    model, _ = load_llm(checkpoint, device)

    natural = score(checkpoint, model, embed, head, input_ids, device)
    peak = torch.cuda.max_memory_allocated() / 1e9

    generator = torch.Generator().manual_seed(0)
    random_ids = torch.randint(0, embed.shape[0], input_ids.shape, generator=generator)
    shuffled = score(checkpoint, model, embed, head, random_ids, device)

    uniform = float(torch.log(torch.tensor(float(embed.shape[0]))))
    print()
    print(f"  cross-entropy, natural text   : {natural:.3f} nats  (perplexity {pow(2.718281828, natural):.1f})")
    print(f"  cross-entropy, random token ids: {shuffled:.3f} nats")
    print(f"  uniform over the vocabulary    : {uniform:.3f} nats")
    print(f"  peak GPU memory                : {peak:.1f} GB")
    print()
    if natural < 4.0 and shuffled > natural + 2.0:
        print("PASS: the model predicts natural text far better than noise, so the")
        print("      converted weights carry the trained distribution.")
    else:
        print("FAIL: a correctly converted model scores natural text well below the")
        print("      uniform bound and well below random ids. Suspect the conversion.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="converted SeedOmni checkpoint root")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tmp-dir", default="/tmp/v41_stream_selftest")
    parser.add_argument("--text", default=None, help="text to score; defaults to a built-in passage")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    if args.self_test:
        self_test(device, args.tmp_dir)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --self-test")
    run_real(args.checkpoint, args.text or PASSAGE, args.max_tokens, device)


if __name__ == "__main__":
    main()
