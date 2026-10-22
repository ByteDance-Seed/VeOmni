# Offline Encoding (`OfflineEncodingMixin`)

`OfflineEncodingMixin` lets a SeedOmni module run an expensive, deterministic
encoder, such as a VAE or a ViT, once offline, and train from the cached tensors
afterwards. It is a marker: mixing it in is what makes a module cacheable, and
it declares no method.

- Mixin: [`OfflineEncodingMixin`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L20)
- Run selector: [`train.training_task`](../../../veomni/arguments/omni_arguments_types.py#L701)
- Meta build: [`ModuleRuntime.reads_offline_cache`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L216)

## Why cache a module's encoder

An omni training graph often contains modules that are frozen and
deterministic: a VAE encoder that turns images into latents, or a ViT that turns
images into patch features. Nothing trains them, and for the same input they
produce the same output in every epoch and every run. Running them online costs:

- **Repeated compute.** Every training step pays the frozen encoder's forward
  again, although its output never changes.
- **Memory and load time.** Every rank keeps the encoder's weights and
  activations resident and loads them at startup, only to produce tensors that
  could have been read from disk.
- **Coupled layouts.** Encoding needs no optimizer and no backward, yet it has to
  run inside the training job, on the training layout.

Offline caching splits such a module into two stages: an **encode** stage that
runs once, in a dedicated encoding run, and produces a cache artifact per
sample; and a **process** stage that runs every training step on the cached
artifact and produces the tensors the rest of the graph expects.

The cut is two stages rather than "cache the module's output" because not
everything downstream of the encoder is deterministic. A VAE encoder outputs a
posterior (mean and log-variance), and training samples a fresh latent from it
every step. Caching the sampled latent would freeze one sample forever. So the
module cuts at the boundary between the expensive deterministic part and the
cheap per-step part: for BAGEL's VAE, `offline_encode` stops at the posterior,
and `online_process` samples and scales the latent.

## The contract

A cacheable module splits its online encoding into two methods and names them
itself. This page uses BAGEL's names, `offline_encode` and `online_process`;
any other names work, because only the graph YAML refers to them. The module
guarantees, for every input `x`:

```text
online_method(x) == offline_training_method(offline_embedding_method(x))
```

The offline-training method runs on a module built on meta, so it must read
only the config, never a parameter or buffer.

## What decides caching: the mixin, the graph and `train.training_task`

Three things decide how a module runs. None of them is a config flag:

| What | Where it lives | Decides |
|------|----------------|---------|
| Capability | The module mixes in `OfflineEncodingMixin` | That an `offline_training` run reads this module from the cache and builds it on meta. |
| Method | The graph YAML of each task, as `module.method` | Which of the module's methods this run calls. [`validate_graph_modules`](../../../veomni/models/seed_omni/graphs/base.py#L162) rejects a method the module does not have, at build time. |
| Run | `train.training_task`, one value per run | How the trainer loops, and whether a cacheable module is built on meta. |

`train.training_task` takes one of three values, named after the DiT trainer's
tasks:

| `training_task` | What the run does | What a cached module does |
|-----------------|-------------------|---------------------------|
| `online_training` (default) | Trains from raw data. | Nothing special: the graph calls its online encode method, and it is built like any module. |
| `offline_embedding` | Reads raw `seedomni` data once (`train.num_train_epochs: 1`), runs the graph without autograd and writes each encoded conversation to `train.offline_cache_dir`, which it requires. Every module is frozen: no optimizer, no lr scheduler, no checkpoint manager. | Full load, frozen; the graph calls its offline-embedding method. |
| `offline_training` | Trains from the cache, read back with `data.data_type: seedomni_cached`. | Built on meta, never loaded, and frozen; the graph calls its offline-training method. |

`OmniArguments` ([`_validate_training_task_data`](../../../veomni/arguments/omni_arguments_types.py#L822)) rejects
a run whose `data.data_type` does not match: only `offline_training` reads
`seedomni_cached`.

Which modules exist and which method each graph node calls stay in the
modules / graph YAML: each task has its own pair. `training_task` only decides
how the trainer loops and how a cacheable module is built. Nothing checks that
the graph YAML was written for the task: an `offline_training` run given the
online graph builds the cacheable module on meta and then calls its online
method, which fails at the first step on meta parameters. The other mistake
fails later: an `offline_embedding` run given a graph that never calls the
offline-embedding method writes whatever the graph leaves in the conversation,
so the cache looks complete and the error only shows in `offline_training`.

`offline_embedding` freezes every module, so it never writes or restores module
weights; encoder weights always come from each module's `model_path`. A saved
step is therefore only the cursor, and its manifest
[says so](../../../veomni/trainer/callbacks/global_state_callback.py#L214), so
`train.checkpoint.load_path: auto` finds it like any other step. Give the
encoding run and the training run separate `train.checkpoint.output_dir`s:
`auto` resumes the newest complete step in the directory, and
[`OmniTrainer.load`](../../../veomni/trainer/omni/omni_trainer.py#L394) refuses a
step whose manifest disagrees with the run on carrying model state. A resume of that
task restores the dataloader cursor and the cache writer's position together,
both through `GlobalStateCallback`:

- Saving the cursor first
  [flushes the writer](../../../veomni/models/seed_omni/utils/offline_cache.py#L90),
  so every row the cursor covers is on disk.
- On resume the writer
  [reopens at the saved shard position](../../../veomni/models/seed_omni/utils/offline_cache.py#L132)
  and deletes this rank's shards written after it. The restored cursor replays
  those steps, so keeping them would cache those samples twice.
- Resume fails if the cache was already finalized, because it would rewrite
  part of a complete cache. It also fails if the data-parallel size changed:
  a shard's name says which rank wrote it, and at which position, only under
  the data-parallel size that wrote it. Finalizing therefore keeps those names,
  so shard numbers can have gaps; the cached dataset reads every shard in the
  directory.
- A run without a saved writer position still
  [refuses a directory that holds shards](../../../veomni/models/seed_omni/utils/offline_cache.py#L108)
  or a leftover completion marker, which would otherwise make a later resume
  of the new run refuse the cache as already finalized.

## How a run builds a cached module

This follows the DiT trainer's condition model: either the module is loaded in
full, or it is built on meta.

- **`offline_training`.**
  [`ModuleRuntime.reads_offline_cache`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L216)
  is true when the module's class mixes in `OfflineEncodingMixin`. The class
  is resolved from the module config with `get_model_class`, because the init
  device has to be chosen before the module exists. Then
  [`_build_model`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L342)
  passes `init_device="meta"`, and the
  [training build stops](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L160)
  right after `requires_grad_(False)`: no parallel wrap, no weight load, no
  optimizer, no checkpoint manager. The offline-training method therefore must
  read only the config, never a parameter or buffer: parameters are meta
  tensors, and buffers keep the values the constructor gave them, not the
  checkpoint's. Under `fsdp_scope: model`,
  [`_parallelize_composed_model`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L241)
  detaches cache readers before the composed wrap so they stay on meta and are
  not materialised with the rest.
- **`offline_embedding`.** Every module, cached or not, is built and
  [frozen](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L171),
  then loaded and wrapped like any other: the run trains nothing, so no module
  gets an optimizer or a checkpoint manager, and
  [`OmniTrainer`](../../../veomni/trainer/omni/omni_trainer.py#L276) allows the
  empty optimizer. A `fsdp_mode: ddp` module is
  [loaded but left unwrapped](../../../veomni/distributed/torch_parallelize.py#L963),
  because DDP rejects a module with no parameter that requires grad.
- **`online_training`.** The module is built like any other.

Inference builds have no train arguments, so they always build the module in
full.

### Examples with BAGEL

- **The training run.** The VAE mixes in `OfflineEncodingMixin`, so it is
  built on meta, and its graph calls `bagel_vae.online_process`. The LLM, the
  ViT, the text encoder and the flow connector do not mix it in, so they load
  and train as usual. Its modules YAML is the online training run's; only the
  VAE's edge changes, from `bagel_vae.encode` to `bagel_vae.online_process`.
- **A cacheable module that still needs its weights.** An `offline_training`
  graph that also called `bagel_vae.decode`, for example for a reconstruction
  loss, would fail at its first step: the VAE is on meta, and decoding reads
  its weights. Train such a graph with `online_training`.
- **A graph that names a missing method.** A node naming a method the module
  does not define fails at build time:
  [`validate_graph_modules`](../../../veomni/models/seed_omni/graphs/base.py#L162)
  reports it.

The data side needs no switch. The
[`SeedOmniCollator`](../../../veomni/data/seed_omni/collator.py#L26) always runs
every module's preprocessor, `seedomni_cached` included, so a preprocessor must
leave items that already hold a cache artifact alone. The simplest way is to
mark such items in the offline-embedding method's post-hook with a key in `item.meta` and
skip marked items in the preprocessor: the meta is saved with the item, so the
training run sees the marker.

## Opting in

Offline encoding is opt-in through multiple inheritance. `BaseMixin` does
**not** inherit `OfflineEncodingMixin`, and modules that do not mix it in are
unaffected. Put the two methods on a sibling `*OfflineMixin` that inherits
`OfflineEncodingMixin`, under names of the module's choosing, and write the
module's own `@pre_forward` / `@post_forward` hooks for the two call-sites:

```python
class XxxOfflineMixin(OfflineEncodingMixin):
    def offline_encode(self, pixel_values):
        return {"encoded_cache": ...}

    def online_process(self, encoded_cache):
        return {"latents": ...}  # reads only self.config


class XxxAccelerated(BaseMixin, XxxOfflineMixin, TrainingModuleMixin, XxxModel):
    # Each hook must return a dict: a pre-hook's dict becomes the endpoint's
    # kwargs, a post-hook's dict is merged into the shared batch.
    @pre_forward("offline_encode")
    def offline_encode_pre(self, conversation_list=None, **batch):
        return {"pixel_values": ...}  # this module's items from conversation_list

    @post_forward("offline_encode")
    def offline_encode_post(self, encoded_cache):
        return {"conversation_list": ...}  # attach and mark each sample's cache artifact

    @pre_forward("online_process")
    def online_process_pre(self, conversation_list=None, **batch):
        return {"encoded_cache": ...}  # cached artifacts read back from the batch

    @post_forward("online_process")
    def online_process_post(self, latents):
        return {"latents": latents}  # where downstream nodes read them
```

The HF modeling class always builds every sub-network; it holds no VeOmni run
logic, because which method runs is decided by the graph YAML, and whether
weights are loaded by `ModuleRuntime`.

## The functions

### What a module implements

| Function | Module's job | Called by | Effect |
|----------|--------------|-----------|--------|
| The offline-embedding method, e.g. `offline_encode(**kwargs) -> dict` | **Must implement** on the sibling `*OfflineMixin`, under any name. Turn tensor inputs into deterministic cache tensors. | The `offline_embedding` graph node naming it, through [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L27) | Produces the cache artifacts. |
| The offline-training method, e.g. `online_process(**kwargs) -> dict` | **Must implement** on the sibling `*OfflineMixin`, under any name, without touching weights. Turn cache tensors back into the tensors training needs. | The `offline_training` graph node naming it, through [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L27) | Materializes runtime tensors from the cache. |
| `@pre_forward` / `@post_forward` for both methods | **Must write** on the module. Translate between the graph's conversation payload and the method's tensors. The mixin defines none. | [`TrainingModuleMixin.pre_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L59) / [`post_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L66) | The endpoint receives tensors and the graph receives the hook's output. |

### What each hook does

The graph passes every node the shared batch and merges whatever the node
returns back into it. The four hooks are where a module translates between that
batch and its tensor endpoints:

| Hook | Receives | Returns | Job |
|------|----------|---------|-----|
| `@pre_forward("offline_encode")` | The shared batch | Kwargs for `offline_encode`, e.g. `{"pixel_values": ...}` | Select this module's items from the batch and stack them into input tensors. |
| `@post_forward("offline_encode")` | `offline_encode`'s outputs | A dict merged into the batch | Attach each sample's cache artifact to the batch, e.g. onto its conversation item, where a cache writer can persist it, and mark the item so the preprocessor skips it in the training run. |
| `@pre_forward("online_process")` | The shared batch, carrying cached artifacts | Kwargs for `online_process`, e.g. `{"encoded_cache": ...}` | Read the cached artifacts back out of the batch and move them to the accelerator device. Under `offline_training` the module is on meta, so use `get_device_type()` / the local rank, not `self.device`. |
| `@post_forward("online_process")` | `online_process`'s outputs | A dict merged into the batch | Write the results where downstream nodes read them, i.e. the same place the online encode path writes. |

`offline_encode` takes the same inputs as the module's online encode method, and
`online_process` returns the same outputs, so one hook can usually serve both
call-sites: `@pre_forward("encode", "offline_encode")` and
`@post_forward("encode", "online_process")`. Downstream nodes then cannot tell
whether the module ran online or from the cache.

## Resulting workflow

Offline caching turns one training job into two runs over the same data, each
with its own modules and graph YAML. Each method is named in the graph as
`module.method`:

```yaml
# Encoding run, train.training_task: offline_embedding
- {from: xxx.offline_encode, to: end}
```

```yaml
# Training run, train.training_task: offline_training
- {from: xxx.online_process, to: yyy}   # replaces {from: xxx.encode, to: yyy}
- {from: yyy, to: end}
```

1. **Encoding run.** Only `offline_encode` runs, without autograd. Its post-hook
   attaches the artifacts to the batch, and a cache writer persists them per
   sample. Dummy items a preprocessor appends for FSDP are encoded and saved
   too, so the training run sees the same item layout.
2. **Training run.** The dataset yields the cached artifacts in place of the raw
   media for this module. The module is built on meta, and `online_process`
   turns each artifact into the tensors downstream nodes expect.

The result:

- The encoder runs once per dataset instead of once per step in every run.
- Training ranks hold neither the encoder's weights nor its activations, and do
  not load them at startup.
- The training data pipeline no longer decodes and preprocesses the raw media
  for this module.
- Per-step randomness after the cut, such as VAE latent sampling, is kept,
  because `online_process` still runs every step.

The pieces that complete this workflow are in place:
[`OmniTrainer.offline_cache_step`](../../../veomni/trainer/omni/omni_trainer.py#L568)
runs the encoding run, and
[`SeedOmniOfflineCacheWriter.save_conversation_list`](../../../veomni/models/seed_omni/utils/offline_cache.py#L181)
persists each conversation of the `conversation_list` that
[`OmniModel.forward`](../../../veomni/models/seed_omni/modeling_omni.py#L511)
returns. The training graph must only mutate `ConversationItem` fields in
place (for example `value` and `meta`); it must not replace the list, append
items, or drop items. Structural changes belong in the processor / collator
stage, before the graph runs.

Under mixed precision, the pre-forward of every FSDP2 wrap unit casts its inputs
with `dataclasses.replace`, so a wrapped call sees copies of the items it was
given. Two rules follow:

- The trainer reads the list from the return value, not from the micro-batch it
  passed in. Under `fsdp_scope: model` the root wrap already hands the graph
  copies.
- The offline-embedding node must put its edits where the batch sees them. The
  [`pre_forward` / `post_forward` hooks](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L61)
  run on the batch itself, outside the module's wrap, so edits made there land.
  The method body gets its kwargs after the module wrap's cast, so in-place
  edits there are lost unless it returns `conversation_list`.

Ranks that share a `dp_rank` (SP, CP or
TP peers) hold the same batch, so
[only the first of them writes](../../../veomni/models/seed_omni/utils/offline_cache.py#L36).
The training run reads the cache through the
[`seedomni_cached`](../../../veomni/data/seed_omni/seedomni_transform.py#L246)
data transform. It unpickles each row, so point `data.train_path` only at a
cache you trust.

### The generation graph in a cache run

A cache run's modules YAML usually lacks modules that the checkpoint's
generation graph names, e.g. an encoding run loads only the encoder. So
`OmniModel` checks only the generation graph's structure when it is built, and
checks it against the loaded modules
([`GenerationGraph.validate_modules`](../../../veomni/models/seed_omni/graphs/generation_graph.py#L312))
at the start of
[`OmniModel.generate`](../../../veomni/models/seed_omni/modeling_omni.py#L630).
A missing module or method still fails before the first request runs.

## Current scope

- **Cached modules are not checkpointed.** They are frozen in both cache tasks,
  and [fully frozen modules have no checkpoint manager](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L662),
  so a checkpoint never holds a meta module.
- **Every cacheable module is read from the cache.** `offline_training` builds
  every module that mixes in `OfflineEncodingMixin` on meta; one run cannot
  read one cacheable module from the cache and encode another online.
- **A graph written for another task is not detected at build time.** See
  [What decides caching](#what-decides-caching-the-mixin-the-graph-and-traintraining_task).
- **Weight reads in the offline-training method are not detected at build
  time.** A module whose offline-training method reads a parameter passes the build on meta and
  fails at its first step. Reading a buffer does not fail at all: the meta build
  only puts parameters on meta, so the buffer holds its constructor value on the
  CPU, and a scalar one silently broadcasts into the result.
- **No per-module checkpoint hooks.** Checkpoint I/O stays with
  [`OmniModuleCheckpointManager`](../../../veomni/models/seed_omni/utils/checkpoint.py#L30).

## Tests

- [`tests/seed_omni/mixins/test_offline_encoding_mixin.py`](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py):
  [the mixin is a marker that names no method](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L64),
  and the hook slots dispatch through `TrainingModuleMixin`.
- [`tests/seed_omni/test_omni_offline_cache_args.py`](../../../tests/seed_omni/test_omni_offline_cache_args.py):
  `train.training_task` values, the `offline_cache_dir` requirement,
  [one epoch for `offline_embedding`](../../../tests/seed_omni/test_omni_offline_cache_args.py#L60) and
  [the `data.data_type` match](../../../tests/seed_omni/test_omni_offline_cache_args.py#L76).
- [`tests/seed_omni/runtime/test_module_runtime.py`](../../../tests/seed_omni/runtime/test_module_runtime.py):
  [only a cacheable module under `offline_training` is built on meta](../../../tests/seed_omni/runtime/test_module_runtime.py#L165),
  [the class the omni registries build is the one checked](../../../tests/seed_omni/runtime/test_module_runtime.py#L190),
  [it is frozen and never wrapped, trained or saved](../../../tests/seed_omni/runtime/test_module_runtime.py#L209),
  and [an `offline_embedding` run freezes every module](../../../tests/seed_omni/runtime/test_module_runtime.py#L241).
- [`tests/seed_omni/runtime/test_omni_model_runtime.py`](../../../tests/seed_omni/runtime/test_omni_model_runtime.py):
  [under `fsdp_scope: model`, cache readers stay on meta outside the composed wrap](../../../tests/seed_omni/runtime/test_omni_model_runtime.py#L409), and
  [under a module-level input cast, a node's item edits reach the batch only from its hooks or its return](../../../tests/seed_omni/runtime/test_omni_model_runtime.py#L296).
- [`tests/seed_omni/test_offline_cache_writer.py`](../../../tests/seed_omni/test_offline_cache_writer.py):
  [the `seedomni_cached` transform](../../../tests/seed_omni/test_offline_cache_writer.py#L33), the
  [writer round trip](../../../tests/seed_omni/test_offline_cache_writer.py#L51),
  [one writer per `dp_rank`](../../../tests/seed_omni/test_offline_cache_writer.py#L225),
  [a resumed writer keeps every row its cursor covers and drops the replayed ones](../../../tests/seed_omni/test_offline_cache_writer.py#L132),
  [a finalized cache](../../../tests/seed_omni/test_offline_cache_writer.py#L160) or [another dp size](../../../tests/seed_omni/test_offline_cache_writer.py#L173) cannot be resumed into, and
  [a shard only one rank sees fails every rank](../../../tests/seed_omni/test_offline_cache_writer.py#L311).
- [`tests/seed_omni/trainer/test_omni_trainer.py`](../../../tests/seed_omni/trainer/test_omni_trainer.py):
  [an `offline_embedding` run may build no optimizer](../../../tests/seed_omni/trainer/test_omni_trainer.py#L125),
  [it saves no model state](../../../tests/seed_omni/trainer/test_omni_trainer.py#L137),
  [a step from the other kind of run is refused before any module loads](../../../tests/seed_omni/trainer/test_omni_trainer.py#L151),
  [the writer opens after the trainer state is restored](../../../tests/seed_omni/trainer/test_omni_trainer.py#L174), and
  [`offline_cache_step` writes every micro-batch without autograd](../../../tests/seed_omni/trainer/test_omni_trainer.py#L197).
- [`tests/checkpoints/test_checkpoint_callback.py`](../../../tests/checkpoints/test_checkpoint_callback.py):
  [the writer's position is saved and restored with the cursor, and the cursor-only step is complete](../../../tests/checkpoints/test_checkpoint_callback.py#L632).
- [`tests/checkpoints/test_dcp_checkpointer.py`](../../../tests/checkpoints/test_dcp_checkpointer.py):
  [a step without `model/` is found only when its manifest says it carries no model state](../../../tests/checkpoints/test_dcp_checkpointer.py#L2546).
- [`tests/seed_omni/trainer/test_step_metrics_callback.py`](../../../tests/seed_omni/trainer/test_step_metrics_callback.py):
  [a step without an lr scheduler logs no lr](../../../tests/seed_omni/trainer/test_step_metrics_callback.py#L82).
- [`tests/seed_omni/model/test_graph.py`](../../../tests/seed_omni/model/test_graph.py):
  [a generation graph builds without the modules it names](../../../tests/seed_omni/model/test_graph.py#L141).
