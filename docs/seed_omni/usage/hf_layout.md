# Upstream Checkpoint Layout (`OmniHFLayout`)

An `OmniHFLayout` tells SeedOmni how a family's upstream HuggingFace checkpoint
splits into its modules. With one registered, `model.model_path` (and
`OmniModel` / `OmniConfig` / `OmniProcessor.from_pretrained`) can name the
upstream checkpoint itself: no offline convert, and the HF export is written
back in the upstream layout, so it loads wherever the source does.

- Declaration: [`OmniHFLayout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L272) and [`OmniHFModuleLayout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L221)
- Registry: [`OMNI_HF_LAYOUT_REGISTRY`](../../../veomni/models/seed_omni/utils/hf_layout.py#L83), keyed by the upstream `model_type`

The layout is framework code; each family registers its own layout together
with the family's modules.

## Loading an upstream checkpoint

Detection is by the root `config.json`'s `model_type`: `omni` is a split
checkpoint, a type with a registered layout loads through it, and any other
type is an error.

- **Load.** Each process writes a weight-free view of the split checkpoint (module
  configs, tokenizer / processors, root config and graphs) to a temp directory
  and records it as `model.seed_omni_model_path`; `model.model_path` stays the
  upstream root. Configs and preprocessors read the view like any split root.
  Module weights are read from the upstream checkpoint at weight-load time:
  each HF key is renamed onto the module that owns it, and keys owned by other
  modules are skipped without being read. This works for training (FSDP2, DDP,
  rank-0 broadcast, `fsdp_scope: model`) and for eager and distributed
  inference.
- **Save.** The HF export (`save_hf_weights`) is one checkpoint in the upstream
  layout under `global_step_N/hf_ckpt/`: the same keys, dtypes, shard files and
  non-weight files as the source, so it loads wherever the source does
  (`transformers`, or `model.model_path` again). Tensors of frozen modules and
  of LoRA bases are copied from the source; LoRA adapters still export per
  module. DCP checkpoints and resume stay per module.
- **Limits.** Every module in the `modules` YAML must be one the layout cuts
  from the checkpoint and keep its default `model_path` (the module name). A
  model composed from several sources (say a text backbone plus a ViT from
  another checkpoint) still needs split checkpoints. Per-module `model_config`
  overrides apply to the run but not to the exported `config.json`, which is
  the source's (a warning lists them). Expert-parallel streaming load
  (`extra_parallel_stream_load`) does apply the layout: it maps each raw key onto a
  module key through this layout's [`route`](../../../veomni/models/seed_omni/utils/hf_layout.py#L329)
  (`HFSourceKeyConverter.module_key`) before streaming that rank's slice, so a
  module loaded from the upstream root streams directly. That is the loading
  direction; `source_key` is its reverse and is used on export. A module whose
  checkpoint tensor converter renames or reshapes
  tensors, or that adds a parameter with no upstream counterpart (a freshly
  initialized projector or head), loads from the upstream root but cannot
  export to it **unless the module declares `export_state`**: that hook runs
  before the key/shape/dtype checks and returns the source-format tensors, which
  is what makes such a module exportable. Without it the HF export raises rather
  than write the stale source copy or drop the new weight. Export such a run from
  a split checkpoint, declare `export_state`, or disable `save_hf_weights`.
- **Export host memory.** A source tensor no trained module produces (a frozen
  module's weights, an unclaimed key) is *carried*: the export copies it from the
  source instead of gathering it. Carried tensors are spread over the ranks —
  small ones go whole to one rank each, round-robin, and one at or above 64 MiB
  is read as a `Shard(0)` slice per rank when `world_size > 1` — so reading them
  is not a rank-0 cost the size of the frozen checkpoint.
  During DCP consolidation, the patched consolidator
  (`veomni/checkpoint/dcp_consolidation.py`) detects contiguous dim-0 whole-row
  shards — which covers all `Shard(0)` carried tensors as well as whole tensors —
  and streams each shard directly to the output file in bounded 64 MiB chunks
  without allocating a full-tensor host buffer. Full per-tensor host allocation
  is now limited to fallback non-dim-0 shard layouts (such as `Shard(1)` or
  2D-sharded tensors), and single-process exports (`world_size == 1`) where
  carried tensors are read whole during `dcp.save`.
- **Load host memory.** Eager loads take `device_map` as `from_pretrained` does;
  a module spread over several devices (`"auto"`) is loaded on CPU first, so host
  memory must hold it once.

The offline convert of such a family goes through the same layout, so a split
checkpoint and the direct load hold identical module weights.

## When to use a layout

Use a layout when every module is a key-prefix slice of the upstream checkpoint:
the module's weights are the upstream ones under a renamed prefix, possibly
transformed by the module's own checkpoint tensor converter. Keep a family
converter ([`OMNI_CONVERT_REGISTRY`](../../../veomni/models/seed_omni/utils/convert_registry.py#L31))
when the split needs more than that (weights from several checkpoints,
re-packed tensors, generated assets). A family with both uses its converter
for the offline convert.

## Declaring a layout

Put it in `veomni/models/seed_omni/modules/<family>/hf_layout.py` and import
that file from the family's `__init__.py`, so the registry is filled when
`veomni.models.seed_omni.modules` is imported. For a decoder-only LM whose
embedding and head form one module and whose transformer stack forms another:

```python
from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .llm.configuration import MyLlmConfig
from .text_encoder.configuration import MyTextEncoderConfig


def _tokenizer(model_path):
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(model_path)}


MY_HF_LAYOUT = OmniHFLayout(
    modules={
        "my_text_encoder": OmniHFModuleLayout(
            key_prefixes=(("model.embed_tokens.", "embed_tokens."), ("lm_head.", "lm_head.")),
            build_config=lambda hf: MyTextEncoderConfig(vocab_size=hf.vocab_size, hidden_size=hf.hidden_size),
            build_assets=_tokenizer,
        ),
        "my_llm": OmniHFModuleLayout(
            key_prefixes=(("model.", "language_model."),),
            build_config=lambda hf: MyLlmConfig(text_config=hf.to_dict()),
        ),
    },
    tied_source_keys={"lm_head.weight": "model.embed_tokens.weight"},
    graph_dir="configs/seed_omni/<family>/train",
    training_graphs="graph_train.yaml",
    generation_graphs={"infer_text": "graph_infer.yaml"},
    infer_type="infer_text",
)


@OMNI_HF_LAYOUT_REGISTRY.register("<upstream model_type>")
def _register_my_hf_layout(source_path=None):
    # `get_hf_layout` passes the checkpoint directory, so a layout may vary its
    # module set with what the file stores. A factory that declares no parameter
    # is called without one and keeps working.
    del source_path
    return MY_HF_LAYOUT
```

A `build_config` that has to read the checkpoint's stored tensors declares a
second parameter; there is no other way to ask for it:

```python
def _text_encoder_config(hf, source_path):
    # ``source_path`` is the checkpoint directory. Reachable only here, because
    # this is the only point that sees the config and the weights together.
    tied = resolve_tie_from_stored_tensors(hf, source_path)
    return MyTextEncoderConfig(vocab_size=hf.vocab_size, hidden_size=hf.hidden_size, tie_word_embeddings=tied)


OmniHFModuleLayout(
    key_prefixes=(("model.embed_tokens.", "embed_tokens."),),
    build_config=_text_encoder_config,
)
```

[`config_for`](../../../veomni/models/seed_omni/utils/hf_layout.py#L260) inspects
the signature and passes `source_path` only to a builder that declares it, so
every other family keeps the one-argument form and never opens the weight files.
Janus is the in-tree example: its byte-level tokenizer folds a trailing space
into a word or not depending on the next token, so whether its head ties has to
be resolved against the checkpoint's actual `lm_head.weight` rather than its
config alone.

A runnable layout over the in-tree `fake_module_a` / `fake_module_b` stand-ins
is `FAKE_HF_LAYOUT` in
[`tests/seed_omni/model/test_hf_layout.py`](../../../tests/seed_omni/model/test_hf_layout.py).

## Contract

The family declares the fields below. Everything else is framework-provided.

| Field | Owner | What it does | Where the framework reads it | End effect |
|-------|-------|--------------|------------------------------|------------|
| `modules[name].key_prefixes` | must declare | `(source_prefix, module_prefix)` pairs. A source key belongs to the module whose source prefix is the longest match across the whole layout, so a backbone can claim `model.` while a sibling claims `model.embed_tokens.`. A prefix claimed twice is an error. Prefixes match source keys after the module class's `_checkpoint_conversion_mapping`, if it has one. | [`OmniHFLayout.route`](../../../veomni/models/seed_omni/utils/hf_layout.py#L329) when loading, and its reverse [`source_key`](../../../veomni/models/seed_omni/utils/hf_layout.py#L338) when saving | Each upstream weight lands on exactly one module, and exports under its upstream name. Unclaimed keys are never read, and are copied verbatim into an export. |
| `modules[name].build_config` | must declare | Upstream HF config to the module's `OmniModuleConfig`. Declare a second parameter (`def build_config(hf, source_path)`) when the answer depends on what the checkpoint *stores* rather than only on its config — Janus resolves its tokenizer tie against the weight files. The path is passed only to a signature that asks for it. | [`build_module_configs`](../../../veomni/models/seed_omni/utils/hf_layout.py#L359) → [`config_for`](../../../veomni/models/seed_omni/utils/hf_layout.py#L260), called by [`_write_hf_view`](../../../veomni/models/seed_omni/utils/hf_layout.py#L555) and [`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L819) | The module config every load path builds the module from. |
| `modules[name].build_assets` | optional | Loads the module's assets from the checkpoint directory, keyed `tokenizer` / `processor` / `image_processor` / `video_processor`. | [`build_module_assets`](../../../veomni/models/seed_omni/utils/hf_layout.py#L363), same callers | The assets are saved into the module's view subfolder, so the module and the collator bind them as from a split root. |
| `modules[name].generated_tensors` | optional | `(source_path, module_config) -> {module_key: tensor}`: module tensors the checkpoint does not store, computed from the checkpoint directory and the config (a lookup table derived from the tokenizer, say). They load after the stored weights, under module keys. | [`HFSourceKeyConverter.generated_tensors`](../../../veomni/models/seed_omni/utils/hf_layout.py#L646), reached from both weight loaders and from `rank0_load_and_broadcast_weights` — not from `build_module_assets`, which reads only the declared assets. The two loaders call it differently: the whole-tensor loader (`load_model_weights`) takes it indirectly through `HFSourceKeyConverter.finalize` ([`weights.py`](../../../veomni/models/checkpoint/weights.py#L490)), while the streaming loader (`load_model_weights_ep_sharded`) calls `generated_tensors()` directly ([`weights.py`](../../../veomni/models/checkpoint/weights.py#L1187)) because a range-restricted converter emits none | The module owns a tensor the source lacks. Such a key has no source key by construction, so an export neither writes it back nor reports it as unmapped — which also means **a trained value of a generated tensor is lost**: reloading recomputes it from the source. Keep these tensors non-trainable and reproducible from the checkpoint plus the config; a module that has to train one needs a checkpoint format that stores it. |
| `modules[name].export_state` | optional | `(model, state) -> {module_key: tensor}`: turns the module's state dict -- **after** its weight conversions are reverted -- into the form the checkpoint stores (requantized weights and their scales, per-expert splits). | [`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L1032) | The model keeps transformers' fused layout while the export writes the release layout. Without it a module whose converter renames or reshapes tensors cannot export. A key may be returned by **one** rank or by several: the rank(s) holding it write their own local value, their key sets are unioned (`_all_ranks_union`), and **values are never compared across ranks** — so a hook returning a rank-local value writes whichever replica the writer reads, silently. Return a value that is rank-invariant (a full tensor, or a shard of one), not something computed from this rank's slice. |
| `tied_source_keys` | optional | Source key → the source key it duplicates when the model ties them (a checkpoint may store `lm_head.weight` although the config ties it). | [`attach_hf_source_converter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L674) and [`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L1032) | Loading skips the duplicate when the module does not hold it; an export writes the original's trained value under it, not the stale copy. |
| `graph_dir`, `training_graphs`, `generation_graphs`, `infer_type`, `generation_kwargs` | optional | The family's default graph YAML (repo-relative, see `load_family_graphs`), default generation scenario, and default `generation_kwargs`. | [`load_graphs`](../../../veomni/models/seed_omni/utils/hf_layout.py#L369), [`_write_hf_view`](../../../veomni/models/seed_omni/utils/hf_layout.py#L555), [`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L819) | The view and split root carry these graphs and generation defaults; a launcher graph overrides them. Run outside a VeOmni checkout, the view carries none and the launcher has to supply them. |
| `load_hf_config` | optional | Reads the upstream config; `AutoConfig.from_pretrained` when unset. | [`read_hf_config`](../../../veomni/models/seed_omni/utils/hf_layout.py#L352) | For checkpoints `AutoConfig` cannot read. |
| Subclass hooks (`source_weight_files`, `resolve_weights_path`, `extract_module_save_state`, `build_fqn_to_index_mapping`, `finalize_exported_shards`) | optional | Methods on `OmniHFLayout` that default to standard HF single-file / numbered-shard layout and `ModelState` extraction. A family subclass (such as [`BagelHFLayout`](../../../veomni/models/seed_omni/modules/bagel/hf_layout.py)) can override them for named safetensors shards (`ae.safetensors` + `ema.safetensors`) or converter-driven weight splitting on export. | [`_check_hf_source_exportable`](../../../veomni/arguments/omni_arguments_types.py#L121), [`HFSource.weights_path`](../../../veomni/models/seed_omni/utils/hf_layout.py#L463), and [`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L1032) | Non-standard upstream shard layouts and reversible runtime fusions stay encapsulated in the family's subclass. |
| registration | must call | `OMNI_HF_LAYOUT_REGISTRY.register("<upstream model_type>")` on a function returning the layout. | [`resolve_omni_checkpoint_root`](../../../veomni/models/seed_omni/utils/hf_layout.py#L525) | A root whose `config.json` has that `model_type` loads through the layout; an unregistered non-`omni` type is an error. |

Framework-provided, not to be overridden:
[`resolve_omni_checkpoint_root`](../../../veomni/models/seed_omni/utils/hf_layout.py#L525),
[`HFSourceKeyConverter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L585),
[`attach_hf_source_converter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L674),
[`load_module_from_hf_source`](../../../veomni/models/seed_omni/utils/hf_layout.py#L767),
[`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L819) and
[`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L1032).

## Call flow

Training from an upstream root:

- [`build_omni_model_runtime_args`](../../../veomni/arguments/omni_arguments_types.py#L183) calls
  [`_resolve_hf_checkpoint_view`](../../../veomni/arguments/omni_arguments_types.py#L97):
  - [`resolve_omni_checkpoint_root`](../../../veomni/models/seed_omni/utils/hf_layout.py#L525) reads the root `model_type`, and
    [`_write_hf_view`](../../../veomni/models/seed_omni/utils/hf_layout.py#L555) writes the weight-free view once per process (module configs, assets, root config, graphs, `hf_source.json`);
  - `model.seed_omni_model_path` is set to the view; `model.model_path` is unchanged.
- [`_validate_hf_view_modules`](../../../veomni/arguments/omni_arguments_types.py#L151) checks every module comes from the layout; with `save_hf_weights`,
  [`_check_hf_source_exportable`](../../../veomni/arguments/omni_arguments_types.py#L121) checks the source has safetensors
  (skipped for `train.training_task: offline_embedding`, which exports nothing).
- [`OmniConfig.load_checkpoint_sidecars`](../../../veomni/models/seed_omni/configuration_omni.py#L426) records the `HFSource`, which each `ModuleRuntime` receives.
- `ModuleRuntime._build_model` builds the module on meta and
  [attaches the key converter](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L361);
  [`weights_path`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L244) resolves through `HFSource.weights_path`.
- The FSDP2 / DDP wrap (per module, or [composed](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L267)) loads weights:
  - [`load_model_weights`](../../../veomni/models/checkpoint/weights.py#L422) and the [rank-0 broadcast loader](../../../veomni/models/checkpoint/weights.py#L1339) skip keys the converter declares, without reading them;
  - [`HFSourceKeyConverter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L585) renames the module's own keys and hands them to the module's converter.

HF export (`CheckpointCallback` → [`OmniModelRuntime.save_hf_or_lora`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L649)):

- [`_save_hf_source_layout`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L662) exports LoRA modules' adapters per module, skips frozen modules, and makes sure each trained module's DCP exists;
- [`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L1032) extracts each module's save state (via `layout.extract_module_save_state`), renames keys back (raising on a key with no source key or a different shape), casts to the source dtypes, fills tied duplicates, copies every other source tensor (spread over the ranks, not on rank 0 alone — see `_read_carried_tensors`), and writes `global_step_N/hf_ckpt/` with the source's shard files and non-weight files.

Bare `OmniModel.from_pretrained` and eager inference:

- [`OmniModel.from_pretrained`](../../../veomni/models/seed_omni/modeling_omni.py#L208) resolves the root first;
- [`_load_module_from_hf_source`](../../../veomni/models/seed_omni/modeling_omni.py#L335) loads each module with [`load_module_from_hf_source`](../../../veomni/models/seed_omni/utils/hf_layout.py#L767) (preferring the module config's `dtype` under `torch_dtype="auto"`) and binds its assets from the view;
- [`resolve_hf_source_device_map`](../../../veomni/models/seed_omni/utils/hf_layout.py#L733) gives `device_map` its `from_pretrained` meaning (`"auto"` is planned by accelerate over `_no_split_modules`). A multi-device map loads on CPU, then `accelerate.dispatch_model` places the module.

Offline convert: a family with a layout and no converter splits through
[`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L818),
called from [`_run_converter`](../../../veomni/models/seed_omni/utils/convert_registry.py#L213).
It returns `{modules, training_graphs, generation_graphs, infer_type,
generation_kwargs}`, which `_save_converted_omni` consumes: `modules` maps a
module name to an already-loaded in-memory module, so the split writer
`save_pretrained`s each one into its own subfolder under `output_dir`. The other
form a converter may return — `module_names`, for a streaming converter that
wrote those subfolders itself — is *not* what the layout path produces: a layout
convert builds every module on CPU, so it always returns `modules`. Likewise
`output_dir` never reaches `convert_with_hf_layout`: `convert_checkpoint` passes
it separately to `_save_converted_omni` and only forwards `output_dir` to a
registered family converter whose signature declares it, so a layout-backed
family cannot take one. Both are the contract of the *converter*, not of the
layout.

Its `attn_implementation` argument is forwarded to every module build, so an
explicit selection (for example `eager` on a host without flash-attn) is recorded
in the exported module config; leave it unset to keep each config's default.

## Extension hooks and family detection

`OmniHFLayout` provides two lifecycle hooks for checkpoint-specific behavior:

- `resolve_source_path(source_path)` runs when the source is canonicalized by
  `HFSource.from_path`. It returns the root used for config, assets, weights,
  and export; the default returns the input unchanged. A layout can override
  it when the user-facing path is an alias or points at a nested release root.
- `validate_module_state(source_path, name, model)` runs when a module
  is attached to an HF source, before either loader reads its weights. The
  module may still be on `meta`, so the default no-op hook is for validating
  source/layout compatibility, not for inspecting loaded tensor values or
  post-load normalization. It should raise a descriptive exception when the
  source state is incompatible.

The generic resolver does not import a concrete family. For releases whose
root `config.json` does not identify the model, a family registers a detector
in `OMNI_HF_LAYOUT_DETECTORS` next to its layout. A detector receives the
candidate checkpoint path and returns the registered `model_type` string when
it recognizes the directory, or `None` otherwise. It must only inspect the
layout's identifying files and must not load model weights. The resolver uses
that result to select the registered `OmniHFLayout`; an unrecognized path is
left as a regular split checkpoint or rejected later by the normal validation.

The hooks are family-owned extensions: keep the generic behavior in
`veomni/models/seed_omni/utils/hf_layout.py`, and register a detector/layout
from the family's module package so importing SeedOmni modules makes it
available to loading, offline conversion, and HF export.
## Rules

- Prefixes rename only. A module may still reshape, fuse or split tensors in
  its checkpoint tensor converter, which receives the renamed keys. Such a
  module loads from the upstream root, and exports only if it declares
  `export_state`: that hook maps its live state back onto source keys before the
  prefix reversal, so the write is not limited to what the prefixes can express.
  Without the hook the export writes live keys back through the prefixes alone
  and raises when that cannot reproduce the source.
- Each module claims disjoint weights: a source key may route to one module
  only, and two modules must not export the same source key.
- `build_config` must give the config the module had when split by the
  family's converter, so a split checkpoint and the direct load hold the same
  module.
- Write `build_assets` to load from the directory it is given; the view saves
  what it returns with `save_pretrained`.

## Tests

[`tests/seed_omni/model/test_hf_layout.py`](../../../tests/seed_omni/model/test_hf_layout.py)
covers routing, the view, loading, tied keys, the export round trip (single
file and sharded) and its refusals, the offline convert, `device_map`
handling, and the argument checks.
[`tests/seed_omni/runtime/test_omni_model_runtime.py`](../../../tests/seed_omni/runtime/test_omni_model_runtime.py)
covers the export orchestration.
