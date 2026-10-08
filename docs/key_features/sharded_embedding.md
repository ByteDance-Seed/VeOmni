# Sharded Embedding

> TL;DR: `veomni.distributed.emb_parallel.ShardedEmbedding` is a drop-in `nn.Embedding` whose vocab rows are split over the `emb` extra-parallel group; a tied output head uses the same split through `ShardedEmbedding.project`. Build the text embedding as a `ShardedEmbedding`, list it in `_no_split_modules`, shard its weight on dim-0 under `"emb"` in the module's parallel plan, and set `extra_parallel_names: [emb]` in the accelerator config. A table the plan did not split runs exactly as `nn.Embedding`.

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

Each rank therefore stores `[V / emb, H / emb_fsdp]`. The parallelizer wraps the module that owns the planned weight (`embed_tokens` for `embed_tokens.weight`) as its own FSDP2 unit on the `emb_fsdp` mesh, so by the time `forward` runs, FSDP2 has all-gathered the hidden dim and the module sees its plain `[V / emb, H]` rows. It only does so for a module at or below a wrap target (a class in `_no_split_modules` or `basic_modules`); otherwise a parent unit would shard and gather the rows over the whole FSDP mesh, mixing different ranks' vocab slices. A unit of its own on the regular FSDP mesh mixes them the same way, so `ShardedEmbedding` raises unless a split table reaches its lookup or projection as the unit the parallelizer wrapped on the `emb_fsdp` mesh.

### Dispatch

`ShardedEmbedding` decides per call from its own weight, not from a config switch. If the weight holds all `V` rows (`emb` off, or the module not in the plan), it runs `nn.Embedding.forward`. If it holds `V / emb` rows, it runs the sharded lookup over the `emb` group of the parallel state. `project` dispatches the same way, between `F.linear` and `VocabParallelLinear`. Any other row count raises, because global ids would silently index the wrong rows.

### Lookup

`ShardedEmbedding.forward` calls `AllToAllEmbedding`, an autograd function in `veomni/distributed/emb_parallel/all_to_all.py`:

1. Bucket the flattened ids by owner rank (`id // (V / emb)`) and all-to-all the per-rank counts, plus a flag for any out-of-range id so that every rank raises together instead of blocking.
2. All-to-all the ids to their owners.
3. Each owner runs a local `F.embedding` on the ids it received, rebased to its own rows.
4. All-to-all the vectors back and restore the original order and shape.

Backward runs the same exchange in reverse. It sums the incoming row gradients in fp32 over the rows that were actually touched, then casts the sum to the table dtype. Called with a group of one rank (or no group), the op aliases its buffers instead of running collectives, and reduces exactly to `F.embedding`.

### Tied output head

A tied head is the same table read a second time, so it has the same `[V / emb, H / emb_fsdp]` split. `ShardedEmbedding.project(hidden)` computes `F.linear(hidden, weight)` over the global vocab:

- On a split table it calls `VocabParallelLinear` (`veomni/distributed/emb_parallel/vocab_parallel_linear.py`). That op all-gathers the vocab rows over `emb` in rank order and projects locally to full-vocab logits `[*, V]`. Its backward gathers the rows again rather than keeping them alive since forward, and reduce-scatters the full weight gradient over `emb`, so each rank keeps the gradient of its own rows.
- On an unsplit table it is `F.linear`.

`project` is registered with FSDP2 (`register_fsdp_forward_method`) on its first call, so like `forward` it runs inside the module's unshard hooks. The lookup and the head then contribute to one weight gradient, which FSDP2 reduces once. This holds even when the two calls are separate forwards, such as an encode node and a decode node of an omni graph.

`VocabParallelLinear` materializes the full `[V, H]` weight in the compute dtype for every projection, and its backward materializes the full weight gradient. That is fine for a text vocabulary, but not for a table too large to gather on one device.

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

        # Tied: no separate head module; decoding goes through the table itself.
        self.lm_head = None if config.tie_word_embeddings else nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids)  # global ids in, [*, H] out
        ...

    def project(self, hidden_states):
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return self.embed_tokens.project(hidden_states)  # [*, V] logits
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

- **Tie through `project`, not by sharing the parameter.** A tied model has no head module of its own and calls `embed_tokens.project`, as in the example above. HF-style tying, where `lm_head.weight` is `embed_tokens.weight`, is not supported with a split table. The loaders re-tie after `fully_shard`, so the head would hold the embedding's `[V / emb, H / emb_fsdp]` shard under its own FSDP2 unit's `[V, H]` metadata, and the first forward fails inside FSDP2.
- **Read the weight only through the module.** Calling `F.linear(hidden, module.weight)`, `F.embedding(ids, module.weight)` or indexing `module.weight` from outside `forward` / `project` sees the FSDP2-sharded DTensor. `ShardedEmbedding` raises if either call reaches it with a DTensor weight.
- **Every rank of the `emb` group must call the module equally often.** Each call is a set of collectives over the group, and so is each backward. A rank with no tokens must still call the module with empty ids, and must still backpropagate through the result, or the other ranks block.
- **`max_norm`, `scale_grad_by_freq` and `sparse` are not supported** on a split table; the sharded lookup raises `NotImplementedError` for each.
- **One table per `emb` plan.** `ParallelPlan` makes only the parent of the plan's first entry the `emb` FSDP2 unit. A second planned table is still split but lands in a unit on the regular FSDP mesh, even when its class is in `_no_split_modules`, so its forward raises. Several tables need `extra_parallel_fsdp_no_shard_module` set by hand until [#1270](https://github.com/ByteDance-Seed/VeOmni/issues/1270) lands.
- **`project` repeats the unit's forward prefetch.** With an extra-parallel group enabled, the parallelizer sets each wrap target to prefetch the next one in forward. `project` runs through the same FSDP2 pre-forward hook, so a tied head called at the end of forward issues that prefetch again, and the next block's gathered parameters stay resident until its backward.

## Status and roadmap

Today, `ShardedEmbedding` is the operator only; no model in this repository builds it yet. The SeedOmni V2 text encoder is the first planned user, calling it in its own forward. Two follow-ups are tracked in [#1270](https://github.com/ByteDance-Seed/VeOmni/issues/1270):

- Unify it with the Qwen3.8 (`qwen4_exp`) PLE lookup, which uses the same vocab-row partition but keeps the hidden dim persistently sharded. It gathers activations rather than parameters; see [qwen4_exp_ple_2d_parallelism.md](../design/qwen4_exp_ple_2d_parallelism.md).
- Bind the text embedding, PLE and n-gram tables of every transformers model to this operator through the parallel plan.

Tests: `tests/distributed/test_emb_parallel.py` (CPU gloo). It includes `emb=2 x emb_fsdp=2` FSDP2 runs with an untied and a tied head, a tied run whose lookup and head are separate calls outside the root forward, and a plain-FSDP2 tied run, all compared against a dense reference. The runs wrap the units by hand rather than through the parallelizer, because the parallelizer's gradient divide factor needs a backend with `PREMUL_SUM`.
