# Offline Encoding (`OfflineEncodingMixin`)

`OfflineEncodingMixin` lets a SeedOmni module run an expensive, deterministic
encoder, such as a VAE or a ViT, once offline, and train from the cached tensors
afterwards.

- Mixin: [`OfflineEncodingMixin`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L23)
- Run selector: [`train.training_task`](../../../veomni/arguments/omni_arguments_types.py#L603)
- Meta build: [`ModuleRuntime.reads_offline_cache`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L208)

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
cheap per-step part: for a VAE, `offline_encode` stops at the posterior, and
`online_process` samples and scales the latent.

## What decides caching: the mixin, the graph and `train.training_task`

Three things decide whether a module is cached. None of them is a config flag:

| What | Where it lives | Decides |
|------|----------------|---------|
| Capability | The module mixes in `OfflineEncodingMixin` | That the module has the two offline endpoints. A graph can call them only on such a module: [`validate_graph_modules`](../../../veomni/models/seed_omni/graphs/base.py#L162) rejects a missing method at build time. |
| Endpoint | The graph YAML, as `module.offline_encode` or `module.online_process` | Which endpoint this run calls on the module. |
| Run | `train.training_task`, one value per run | How the trainer loops, and whether a module is built on meta. |

`train.training_task` takes one of three values, named after the DiT trainer's
tasks:

| `training_task` | What the run does | What a cached module does |
|-----------------|-------------------|---------------------------|
| `online_training` (default) | Trains from raw data. | Nothing special: the graph calls its online encode endpoint, and it is built like any module. |
| `offline_embedding` | Reads raw `seedomni` data once (`train.num_train_epochs: 1`), runs the graph without autograd and writes each encoded conversation to `train.offline_cache_dir`, which it requires. Every module is frozen: no optimizer, no lr scheduler, no checkpoint manager. | Full load, frozen; the graph calls its `offline_encode`. |
| `offline_training` | Trains from the cache, read back with `data.data_type: seedomni_cached`. | The graph calls only its `online_process`, so it is built on meta, never loaded, and frozen. |

`OmniArguments` ([`_validate_training_task_data`](../../../veomni/arguments/omni_arguments_types.py#L731)) rejects
a run whose `data.data_type` does not match: only `offline_training` reads
`seedomni_cached`. It also rejects `train.checkpoint.load_path` under
`offline_embedding`
([`OmniTrainingArguments.__post_init__`](../../../veomni/arguments/omni_arguments_types.py#L645)):
frozen modules have no checkpoint manager, so the encoder's weights would not
be restored while the dataloader position would. Point the encoder's
`model_path` at an HF checkpoint instead.

Which modules exist and which endpoint each graph node calls stay in the
modules / graph YAML: each task has its own pair, e.g. BAGEL's
`offline_cache/` and `with_cache/` directories. `training_task` only decides how
the trainer loops and how a cached module is built, and the two must agree:
[`build_omni_model_runtime`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L623)
[rejects a training graph](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L140)
that never calls the task's endpoint (`offline_encode` for `offline_embedding`,
`online_process` for `offline_training`) or calls the other task's endpoint.

## How a run builds a cached module

This follows the DiT trainer's condition model: either the module is loaded in
full, or it is built on meta.

- **`offline_training`.**
  [`build_omni_model_runtime`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L623)
  collects the methods the training graph calls on each module
  ([`_training_graph_methods`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L127))
  and hands each `ModuleRuntime` its own set.
  [`ModuleRuntime.reads_offline_cache`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L208)
  is true when that set is exactly `{"online_process"}`. Then
  [`_build_model`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L302)
  passes `init_device="meta"`, and the
  [training build stops](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L159)
  right after `requires_grad_(False)`: no parallel wrap, no weight load, no
  optimizer, no checkpoint manager. `online_process` therefore must read only
  the config, never a parameter or buffer: parameters are meta tensors, and
  buffers keep the values the constructor gave them, not the checkpoint's. A
  module the graph calls through any other method as well needs its weights,
  so it is built in full. Under
  `fsdp_scope: model` the composed wrap still loads every module, the cached
  ones included.
- **`offline_embedding`.** Every module, cached or not, is built, loaded and
  wrapped like any other, then
  [frozen](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L164):
  the run trains nothing, so no module gets an optimizer or a checkpoint
  manager, and [`OmniTrainer`](../../../veomni/trainer/omni/omni_trainer.py#L275)
  allows the empty optimizer.
- **`online_training`.** The module is built like any other.

Inference builds have no train arguments, so they always build the module in
full.

### Examples with BAGEL

- **The training run.** Its graph calls `bagel_vae.online_process` and nothing
  else on `bagel_vae`, so the VAE is built on meta. The LLM, the ViT, the text
  encoder and the flow connector are called through their usual methods, so
  they load and train as usual. `with_cache/modules_train.yaml` is the same
  file as the online run's `train/modules_train.yaml`; between the two graphs
  only the VAE's edge changes, from `bagel_vae.encode` to
  `bagel_vae.online_process`.
- **A cached module that still needs its weights.** If a graph also called
  `bagel_vae.decode`, for example for a reconstruction loss, the VAE would be
  built in full, because decoding reads its weights.
- **A graph that names a missing endpoint.** An `online_process` or
  `offline_encode` node on a module that does not mix in
  `OfflineEncodingMixin` fails at build time:
  [`validate_graph_modules`](../../../veomni/models/seed_omni/graphs/base.py#L162)
  reports the missing method.

The data side needs no switch. The
[`SeedOmniCollator`](../../../veomni/data/seed_omni/collator.py#L26) always runs every
module's preprocessor, `seedomni_cached` included, so a preprocessor must leave
items that already hold a cache artifact alone. BAGEL marks such items in
`offline_encode`'s post-hook with
[`BAGEL_VAE_POSTERIOR`](../../../veomni/models/seed_omni/modules/bagel/vae/processing.py#L417),
and the VAE preprocessor
[skips them](../../../veomni/models/seed_omni/modules/bagel/vae/processing.py#L479).
The marker is saved with the item's meta, so the training run sees it.

## Opting in

Offline encoding is opt-in through multiple inheritance. `BaseMixin` does
**not** inherit `OfflineEncodingMixin`, and modules that do not mix it in are
unaffected. Put the endpoint pair on a sibling `*OfflineMixin` that inherits
`OfflineEncodingMixin`, and write the module's own `@pre_forward` /
`@post_forward` hooks for the two call-sites:

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

BAGEL's VAE does exactly this:
[`BagelVAEOfflineMixin`](../../../veomni/models/seed_omni/modules/bagel/vae/accelerated/accelerated.py#L44)
and
[`VeOmniMixin`](../../../veomni/models/seed_omni/modules/bagel/vae/accelerated/accelerated.py#L305).
The HF modeling class always builds the encoder and decoder; it holds no
VeOmni run logic.

## The functions

### What a module implements

| Function | Module's job | Called by | Effect |
|----------|--------------|-----------|--------|
| [`offline_encode(**kwargs) -> dict`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L40) | **Must implement** on the sibling `*OfflineMixin`. Turn tensor inputs into deterministic cache tensors. | The graph node whose method is `offline_encode`, through [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L27) | Produces the cache artifacts. Abstract on the mixin, so a module without it cannot be instantiated. |
| [`online_process(**kwargs) -> dict`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L44) | **Must implement** on the sibling `*OfflineMixin`, without touching weights. Turn cache tensors back into the tensors training needs. | The graph node whose method is `online_process`, through [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L27) | Materializes runtime tensors from the cache. Abstract on the mixin, like `offline_encode`. |
| `@pre_forward("offline_encode")`, `@post_forward("offline_encode")`, and the same for `online_process` | **Must write** on the module. Translate between the graph's conversation payload and the endpoint's tensors. The mixin defines none. | [`TrainingModuleMixin.pre_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L59) / [`post_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L66) | The endpoint receives tensors and the graph receives the hook's output. |

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
with its own modules and graph YAML. The endpoint is named in the graph as
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
   sample. For BAGEL the VAE preprocessor's per-sample dummy rows are encoded
   too, so every cached sample carries a VAE item.
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
[`OmniTrainer.offline_cache_step`](../../../veomni/trainer/omni/omni_trainer.py#L538)
runs the encoding run, and
[`SeedOmniOfflineCacheWriter.save_conversation_list`](../../../veomni/models/seed_omni/utils/offline_cache.py#L104)
persists each conversation. The step writes the `conversation_list` that
[`OmniModel.forward` returns](../../../veomni/models/seed_omni/modeling_omni.py#L504),
not the micro-batch it passed in: under `fsdp_scope: model`, FSDP2's root
pre-forward may rebuild the batch's dicts and lists, so what the post-hooks
merge would not reach the trainer's copy. Ranks that share a `dp_rank` (SP, CP or TP peers)
hold the same batch, so
[only the first of them writes](../../../veomni/models/seed_omni/utils/offline_cache.py#L33).
The training run reads the cache through the
[`seedomni_cached`](../../../veomni/data/seed_omni/seedomni_transform.py#L246)
data transform. It unpickles each row, so point `data.train_path` only at a
cache you trust. See the
[Bagel offline VAE cache](../example_models/bagel.md#32-offline-vae-posterior-cache-two-stages)
for a two-stage example.

### The generation graph in a cache run

A cache run's modules YAML usually lacks modules that the checkpoint's
generation graph names, e.g. BAGEL's encoding run loads only `bagel_vae`. So
`OmniModel` checks only the generation graph's structure when it is built, and
checks it against the loaded modules
([`GenerationGraph.validate_modules`](../../../veomni/models/seed_omni/graphs/generation_graph.py#L312))
at the start of
[`OmniModel.generate`](../../../veomni/models/seed_omni/modeling_omni.py#L607).
A missing module or method still fails before the first request runs.

## Current scope

- **Cached modules are not checkpointed.** They are frozen in both cache tasks,
  and [fully frozen modules have no checkpoint manager](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L586),
  so a checkpoint never holds a meta module.
- **Weight reads in `online_process` are not detected at build time.** A
  module whose `online_process` reads a parameter passes the build on meta and
  fails at its first step. Reading a buffer does not fail at all: the meta build
  only puts parameters on meta, so the buffer holds its constructor value on the
  CPU, and a scalar one silently broadcasts into the result.
- **No per-module checkpoint hooks.** Checkpoint I/O stays with
  [`OmniModuleCheckpointManager`](../../../veomni/models/seed_omni/utils/checkpoint.py#L30).

## Tests

- [`tests/seed_omni/mixins/test_offline_encoding_mixin.py`](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py):
  [a module missing an endpoint cannot be built](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L69),
  and the hook slots dispatch through `TrainingModuleMixin`.
- [`tests/seed_omni/test_omni_offline_cache_args.py`](../../../tests/seed_omni/test_omni_offline_cache_args.py):
  `train.training_task` values, the `offline_cache_dir` requirement,
  [one epoch for `offline_embedding`](../../../tests/seed_omni/test_omni_offline_cache_args.py#L61),
  [no `checkpoint.load_path` for `offline_embedding`](../../../tests/seed_omni/test_omni_offline_cache_args.py#L66) and
  [the `data.data_type` match](../../../tests/seed_omni/test_omni_offline_cache_args.py#L84).
- [`tests/seed_omni/test_offline_cache_writer.py`](../../../tests/seed_omni/test_offline_cache_writer.py):
  [the writer round trip](../../../tests/seed_omni/test_offline_cache_writer.py#L50) and
  [one writer per `dp_rank`](../../../tests/seed_omni/test_offline_cache_writer.py#L134).
- [`tests/seed_omni/trainer/test_step_metrics_callback.py`](../../../tests/seed_omni/trainer/test_step_metrics_callback.py):
  [a step without an lr scheduler logs no lr](../../../tests/seed_omni/trainer/test_step_metrics_callback.py#L82).
- [`tests/seed_omni/runtime/test_module_runtime.py`](../../../tests/seed_omni/runtime/test_module_runtime.py):
  [only a module the `offline_training` graph calls just through `online_process` is built on meta](../../../tests/seed_omni/runtime/test_module_runtime.py#L156),
  [it is frozen and never wrapped, trained or saved](../../../tests/seed_omni/runtime/test_module_runtime.py#L179),
  and [an `offline_embedding` run freezes every module](../../../tests/seed_omni/runtime/test_module_runtime.py#L212).
- [`tests/seed_omni/runtime/test_omni_model_runtime.py`](../../../tests/seed_omni/runtime/test_omni_model_runtime.py):
  [each module gets the methods the training graph calls on it](../../../tests/seed_omni/runtime/test_omni_model_runtime.py#L341)
  and [the training graph must call the offline endpoint of its task](../../../tests/seed_omni/runtime/test_omni_model_runtime.py#L373).
- [`tests/seed_omni/bagel/test_processing.py`](../../../tests/seed_omni/bagel/test_processing.py):
  [`online_process` on a meta-built VAE](../../../tests/seed_omni/bagel/test_processing.py#L264) and the
  [offline cache round trip](../../../tests/seed_omni/bagel/test_processing.py#L305).
- [`tests/seed_omni/test_preprocessor.py`](../../../tests/seed_omni/test_preprocessor.py):
  [the VAE preprocessor leaves cached posteriors alone](../../../tests/seed_omni/test_preprocessor.py#L394).
- [`tests/seed_omni/trainer/test_omni_trainer.py`](../../../tests/seed_omni/trainer/test_omni_trainer.py):
  [an `offline_embedding` run may build no optimizer](../../../tests/seed_omni/trainer/test_omni_trainer.py#L123), and
  [`offline_cache_step` writes every micro-batch without autograd](../../../tests/seed_omni/trainer/test_omni_trainer.py#L134).
- [`tests/seed_omni/model/test_graph.py`](../../../tests/seed_omni/model/test_graph.py):
  [a generation graph builds without the modules it names](../../../tests/seed_omni/model/test_graph.py#L141), and
  [`OmniModel.forward` returns the `conversation_list` the graph wrote](../../../tests/seed_omni/model/test_graph.py#L334).
