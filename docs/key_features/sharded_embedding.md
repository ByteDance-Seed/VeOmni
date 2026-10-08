# Sharded Embedding

> TL;DR: `veomni.distributed.emb_parallel.ShardedEmbedding` is a drop-in `nn.Embedding` whose vocab rows are split over the `emb` extra-parallel group. Build the text embedding as a `ShardedEmbedding`, shard its weight on dim-0 under `"emb"` in the module's parallel plan, and set `extra_parallel_names: [emb]` in the accelerator config. With `emb` off it is exactly `nn.Embedding`.

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

Each rank therefore stores `[V / emb, H / emb_fsdp]`. The parallelizer wraps the module that owns the planned weight (`embed_tokens` for `embed_tokens.weight`) as its own FSDP2 unit on the `emb_fsdp` mesh, so by the time `forward` runs, FSDP2 has all-gathered the hidden dim and the module sees its plain `[V / emb, H]` rows.

### Lookup

`ShardedEmbedding.forward` calls `AllToAllEmbedding`, an autograd function in `veomni/distributed/emb_parallel/all_to_all.py`:

1. Bucket the flattened ids by owner rank (`id // (V / emb)`) and all-to-all the per-rank counts, plus a flag for any out-of-range id so that every rank raises together instead of blocking.
2. All-to-all the ids to their owners.
3. Each owner runs a local `F.embedding` on the ids it received, rebased to its own rows.
4. All-to-all the vectors back and restore the original order and shape.

Backward runs the same exchange in reverse. It sums the incoming row gradients in fp32 over the rows that were actually touched, then casts the sum to the table dtype. With one shard (`emb` reported on with size 1), the collectives alias their buffers, and the op reduces exactly to `F.embedding`.

### Gradient

The weight gradient covers every token of the `emb` group, but only this data-parallel replica's tokens. FSDP2 reduce-scatters it over `emb_fsdp` and divides it by the world size, like an MoE expert gradient (`extra_parallel_gradient_divide_factor`). The padding row gets no gradient, as with `F.embedding(padding_idx=...)`.

### `padding_idx`

`num_embeddings` and the constructor's `padding_idx` stay global. After the table is split, reading `padding_idx` returns the index into the rows this rank holds, or `None` on ranks that do not hold the padding row. That way, initializers that zero `weight[padding_idx]` (such as `nn.Embedding.reset_parameters` and HF `_init_weights`) hit the right row on its owner only.

## Usage

### 1. Module

Build the embedding as a `ShardedEmbedding` and call it in the module's forward, like any `nn.Embedding`:

```python
from veomni.distributed.emb_parallel import ShardedEmbedding

self.embed_tokens = ShardedEmbedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

def forward(self, input_ids):
    inputs_embeds = self.embed_tokens(input_ids)  # global ids in, [*, H] out
    ...
```

Whether `emb` is on, and over which group, is read from the parallel state. No model-side switch is needed.

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

`emb` must divide the world size, `V` must be divisible by `emb`, and `H` must be divisible by `world_size / emb` (the `emb_fsdp` size).

## Constraints

- **No tied output head.** Outside the embedding's own forward, its weight is in general the FSDP2-sharded DTensor, and a vocab-sharded head would need a different collective (its logits are sharded on vocab). Only large tables are worth sharding, and those are never tied. Use an untied `lm_head`, and keep `tie_word_embeddings` false in the config of any module that uses `ShardedEmbedding` with `emb` on.
- **Read the weight only through the module.** Calling `F.embedding(ids, module.weight)` or indexing `module.weight` from outside the module's forward sees the FSDP2-sharded DTensor. `ShardedEmbedding` raises if its forward is reached with a DTensor weight.
- **Every rank of the `emb` group must call the module equally often.** Each call is a set of collectives over the group, and so is each backward. A rank with no tokens must still call the module with empty ids, and must still backpropagate through the result, or the other ranks block.
- **`max_norm` is not supported** under `emb`, because it would renormalize rows in place across ranks. `scale_grad_by_freq` and `sparse` are ignored by the sharded path.
- A table whose rows do not match the parallel state is rejected: a split table with `emb` off, or an unsplit one with `emb` on. Without this check, global ids would silently index the wrong rows.

## Status and roadmap

Today, `ShardedEmbedding` is the operator, and the SeedOmni text encoder is its first user. Two follow-ups are tracked in [#1270](https://github.com/ByteDance-Seed/VeOmni/issues/1270):

- Unify it with the Qwen3.8 (`qwen4_exp`) PLE lookup, which uses the same vocab-row partition but keeps the hidden dim persistently sharded. It gathers activations rather than parameters; see [qwen4_exp_ple_2d_parallelism.md](../design/qwen4_exp_ple_2d_parallelism.md).
- Bind the text embedding, PLE and n-gram tables of every transformers model to this operator through the parallel plan.

Tests: `tests/distributed/test_emb_parallel.py` (CPU gloo, including an `emb=2 x emb_fsdp=2` FSDP2 run compared against a dense reference).
