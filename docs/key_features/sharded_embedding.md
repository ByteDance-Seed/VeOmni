# Sharded Embedding

> TL;DR: `veomni.distributed.emb_parallel.ShardedEmbedding` is a drop-in `nn.Embedding` whose vocab rows are split over the `emb` extra-parallel group. Build the text embedding as a `ShardedEmbedding`, list it in `_no_split_modules`, shard its weight on dim-0 under `"emb"` in the module's parallel plan, and set `extra_parallel_names: [emb]` in the accelerator config. A table the plan did not split runs exactly as `nn.Embedding`.

## Motivation

A large vocabulary table is replicated in memory on every FSDP2 rank between the all-gather and the reshard, and its gradient is as large as the table. FSDP2 alone shards the storage but still all-gathers the whole `[V, H]` table for every lookup. Splitting the table by vocab rows makes each rank hold, and all-gather, only `V / emb` rows.

Under data parallelism every rank looks up its own tokens, so a rank's ids can point at rows that another rank owns. That rules out Megatron-style vocab-parallel embedding, which masks out-of-shard ids and all-reduces partial outputs: that only works when every rank in the group sees the same input. `ShardedEmbedding` instead routes each id to the rank that owns its row and returns the looked-up vector, both with an all-to-all.

## Design

### Layout

The table is a `[V, H]` parameter sharded on two dims, in the same way ExtraParallel shards MoE experts (see [extra_parallel.md](./extra_parallel.md)):

| Dim | Mesh dim | Placement |
|---|---|---|
| 0 (vocab) | `emb` | sliced by the parallel plan: each rank keeps `V / emb` contiguous rows |
| 1 (hidden) | `emb_fsdp` | `Shard(1)` by FSDP2 |

Each rank therefore stores `[V / emb, H / emb_fsdp]`. The parallelizer wraps the module that owns the planned weight (`embed_tokens` for `embed_tokens.weight`) as its own FSDP2 unit on the `emb_fsdp` mesh, so by the time `forward` runs, FSDP2 has all-gathered the hidden dim and the module sees its plain `[V / emb, H]` rows. It only does so for a module at or below a wrap target (a class in `_no_split_modules` or `basic_modules`); otherwise a parent unit would shard and gather the rows over the whole FSDP mesh, mixing different ranks' vocab slices, so `ShardedEmbedding` raises if a split table reaches its forward without being its own FSDP2 unit.

### Dispatch

`ShardedEmbedding` decides per call from its own weight, not from a config switch. If the weight holds all `V` rows (`emb` off, or the module not in the plan), it runs `nn.Embedding.forward`. If it holds `V / emb` rows, it runs the sharded lookup over the `emb` group of the parallel state. Any other row count raises, because global ids would silently index the wrong rows.

### Lookup

`ShardedEmbedding.forward` calls `AllToAllEmbedding`, an autograd function in `veomni/distributed/emb_parallel/all_to_all.py`:

1. Bucket the flattened ids by owner rank (`id // (V / emb)`) and all-to-all the per-rank counts, plus a flag for any out-of-range id so that every rank raises together instead of blocking.
2. All-to-all the ids to their owners.
3. Each owner runs a local `F.embedding` on the ids it received, rebased to its own rows.
4. All-to-all the vectors back and restore the original order and shape.

Backward runs the same exchange in reverse. It sums the incoming row gradients in fp32 over the rows that were actually touched, then casts the sum to the table dtype. Called with a group of one rank (or no group), the op aliases its buffers instead of running collectives, and reduces exactly to `F.embedding`.

### Gradient

The weight gradient covers every token of the `emb` group, but only this data-parallel replica's tokens. FSDP2 reduce-scatters it over `emb_fsdp` and divides it by the world size, like an MoE expert gradient (`extra_parallel_gradient_divide_factor`). The padding row gets no gradient, as with `F.embedding(padding_idx=...)`.

### `padding_idx`

`num_embeddings` and the constructor's `padding_idx` stay global. After the table is split, reading `padding_idx` returns the index into the rows this rank holds, or `None` on ranks that do not hold the padding row. That way, initializers that zero `weight[padding_idx]` (such as `nn.Embedding.reset_parameters` and HF `_init_weights`) hit the right row on its owner only.

## Usage

### 1. Module

Build the embedding as a `ShardedEmbedding`, call it in the module's forward like any `nn.Embedding`, and list it in `_no_split_modules` so the parallelizer makes it its own FSDP2 unit:

```python
from veomni.distributed.emb_parallel import ShardedEmbedding


class TextEncoder(PreTrainedModel):
    _no_split_modules = ["ShardedEmbedding"]

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ShardedEmbedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def forward(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids)  # global ids in, [*, H] out
        ...
```

`_no_split_modules` matches class names, so a list that names `Embedding` does not cover `ShardedEmbedding`. Whether to run the sharded lookup is decided from the weight (see [Dispatch](#dispatch)), so there is no model-side switch.

### 2. Parallel plan

Shard the weight on dim-0 under `"emb"` in the module's `parallel_plan.py`:

```python
from torch.distributed._tensor import Shard

from veomni.distributed.parallel_plan import ParallelPlan


def get_parallel_plan():
    return ParallelPlan(extra_parallel_plan={"emb": {"embed_tokens.weight": Shard(0)}})
```

### 3. Accelerator config

```yaml
accelerator:
  fsdp_config:
    fsdp_mode: fsdp2
  extra_parallel_names: [emb]
  extra_parallel_sizes: [4]
  extra_parallel_placement_innermost: [false]
```

`emb` must divide the FSDP shard size (`dp_shard` x sequence-parallel size, which is the world size without HSDP replication). `V` must be divisible by `emb`, and `H` must be divisible by the `emb_fsdp` size, which is that shard size divided by `emb`.

## Constraints

- **No tied output head.** Outside the embedding's own forward, its weight is in general the FSDP2-sharded DTensor, and a vocab-sharded head would need a different collective (its logits are sharded on vocab). Only large tables are worth sharding, and those are never tied. Use an untied `lm_head`, and keep `tie_word_embeddings` false in the config of any module that uses `ShardedEmbedding` with `emb` on.
- **Read the weight only through the module.** Calling `F.embedding(ids, module.weight)` or indexing `module.weight` from outside the module's forward sees the FSDP2-sharded DTensor. `ShardedEmbedding` raises if its forward is reached with a DTensor weight.
- **Every rank of the `emb` group must call the module equally often.** Each call is a set of collectives over the group, and so is each backward. A rank with no tokens must still call the module with empty ids, and must still backpropagate through the result, or the other ranks block.
- **`max_norm`, `scale_grad_by_freq` and `sparse` are not supported** on a split table; the sharded lookup raises `NotImplementedError` for each.
- **One table per `emb` plan.** `ParallelPlan` makes only the parent of the plan's first entry the `emb` FSDP2 unit, so a second planned table is not wrapped, and its forward raises. Several tables need `extra_parallel_fsdp_no_shard_module` set by hand until [#1270](https://github.com/ByteDance-Seed/VeOmni/issues/1270) lands.

## Status and roadmap

Today, `ShardedEmbedding` is the operator only; no model in this repository builds it yet. The SeedOmni V2 text encoder is the first planned user, calling it in its own forward. Two follow-ups are tracked in [#1270](https://github.com/ByteDance-Seed/VeOmni/issues/1270):

- Unify it with the Qwen3.8 (`qwen4_exp`) PLE lookup, which uses the same vocab-row partition but keeps the hidden dim persistently sharded. It gathers activations rather than parameters; see [qwen4_exp_ple_2d_parallelism.md](../design/qwen4_exp_ple_2d_parallelism.md).
- Bind the text embedding, PLE and n-gram tables of every transformers model to this operator through the parallel plan.

Tests: `tests/distributed/test_emb_parallel.py` (CPU gloo, including an `emb=2 x emb_fsdp=2` FSDP2 run compared against a dense reference).
