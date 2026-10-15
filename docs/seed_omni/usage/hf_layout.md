# Upstream Checkpoint Layout (`OmniHFLayout`)

An `OmniHFLayout` tells SeedOmni how a family's upstream HuggingFace checkpoint
splits into its modules. With one registered, `model.model_path` (and
`OmniModel` / `OmniConfig` / `OmniProcessor.from_pretrained`) can name the
upstream checkpoint itself: no offline convert, and the HF export is written
back in the upstream layout, so it loads wherever the source does.

- Declaration: [`OmniHFLayout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L126) and [`OmniHFModuleLayout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L104)
- Registry: [`OMNI_HF_LAYOUT_REGISTRY`](../../../veomni/models/seed_omni/utils/hf_layout.py#L79), keyed by the upstream `model_type`

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
  (`ep_sharded_stream_load`) does not apply the renames and is rejected at
  startup. A module whose checkpoint tensor converter renames or reshapes
  tensors, or that adds a parameter with no upstream counterpart (a freshly
  initialized projector or head), loads from the upstream root but cannot
  export to it: the HF export raises rather than write the stale source copy
  or drop the new weight. Export such a run from a split checkpoint, or
  disable `save_hf_weights`. Each HF export reads every
  source tensor no trained module produces (frozen modules, unclaimed keys)
  into rank 0's host memory, so a small trained module beside a large frozen
  one costs the frozen one's size in rank-0 RAM per export. Eager loads take
  `device_map` as `from_pretrained` does; a module spread over several devices
  (`"auto"`) is loaded on CPU first, so host memory must hold it once.

The offline convert of such a family goes through the same layout, so a split
checkpoint and the direct load hold identical module weights.

## When to use a layout

Use a layout when every module is a key-prefix slice of the upstream checkpoint:
the module's weights are the upstream ones under a renamed prefix, possibly
transformed by the module's own checkpoint tensor converter. Keep a family
converter ([`OMNI_CONVERT_REGISTRY`](../../../veomni/models/seed_omni/utils/convert_registry.py#L27))
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
def _register_my_hf_layout():
    return MY_HF_LAYOUT
```

A runnable layout over the in-tree `fake_module_a` / `fake_module_b` stand-ins
is `FAKE_HF_LAYOUT` in
[`tests/seed_omni/model/test_hf_layout.py`](../../../tests/seed_omni/model/test_hf_layout.py).

## Contract

The family declares the fields below. Everything else is framework-provided.

| Field | Owner | What it does | Where the framework reads it | End effect |
|-------|-------|--------------|------------------------------|------------|
| `modules[name].key_prefixes` | must declare | `(source_prefix, module_prefix)` pairs. A source key belongs to the module whose source prefix is the longest match across the whole layout, so a backbone can claim `model.` while a sibling claims `model.embed_tokens.`. A prefix claimed twice is an error. Prefixes match source keys after the module class's `_checkpoint_conversion_mapping`, if it has one. | [`OmniHFLayout.route`](../../../veomni/models/seed_omni/utils/hf_layout.py#L160) when loading, and its reverse [`source_key`](../../../veomni/models/seed_omni/utils/hf_layout.py#L167) when saving | Each upstream weight lands on exactly one module, and exports under its upstream name. Unclaimed keys are never read, and are copied verbatim into an export. |
| `modules[name].build_config` | must declare | Upstream HF config to the module's `OmniModuleConfig`. | [`build_module_configs`](../../../veomni/models/seed_omni/utils/hf_layout.py#L188), called by [`_write_hf_view`](../../../veomni/models/seed_omni/utils/hf_layout.py#L301) and [`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L535) | The module config every load path builds the module from. |
| `modules[name].build_assets` | optional | Loads the module's assets from the checkpoint directory, keyed `tokenizer` / `processor` / `image_processor` / `video_processor`. | [`build_module_assets`](../../../veomni/models/seed_omni/utils/hf_layout.py#L192), same callers | The assets are saved into the module's view subfolder, so the module and the collator bind them as from a split root. |
| `tied_source_keys` | optional | Source key → the source key it duplicates when the model ties them (a checkpoint may store `lm_head.weight` although the config ties it). | [`attach_hf_source_converter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L403) and [`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L673) | Loading skips the duplicate when the module does not hold it; an export writes the original's trained value under it, not the stale copy. |
| `graph_dir`, `training_graphs`, `generation_graphs`, `infer_type` | optional | The family's default graph YAML (repo-relative, see `load_family_graphs`) and default generation scenario. | [`load_graphs`](../../../veomni/models/seed_omni/utils/hf_layout.py#L198) | The view carries these graphs; a launcher graph overrides them. Run outside a VeOmni checkout, the view carries none and the launcher has to supply them. |
| `load_hf_config` | optional | Reads the upstream config; `AutoConfig.from_pretrained` when unset. | [`read_hf_config`](../../../veomni/models/seed_omni/utils/hf_layout.py#L181) | For checkpoints `AutoConfig` cannot read. |
| registration | must call | `OMNI_HF_LAYOUT_REGISTRY.register("<upstream model_type>")` on a function returning the layout. | [`resolve_omni_checkpoint_root`](../../../veomni/models/seed_omni/utils/hf_layout.py#L271) | A root whose `config.json` has that `model_type` loads through the layout; an unregistered non-`omni` type is an error. |

Framework-provided, not to be overridden:
[`resolve_omni_checkpoint_root`](../../../veomni/models/seed_omni/utils/hf_layout.py#L271),
[`HFSourceKeyConverter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L331),
[`attach_hf_source_converter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L403),
[`load_module_from_hf_source`](../../../veomni/models/seed_omni/utils/hf_layout.py#L483),
[`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L535) and
[`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L673).

## Call flow

Training from an upstream root:

- [`build_omni_model_runtime_args`](../../../veomni/arguments/omni_arguments_types.py#L188) calls
  [`_resolve_hf_checkpoint_view`](../../../veomni/arguments/omni_arguments_types.py#L97):
  - [`resolve_omni_checkpoint_root`](../../../veomni/models/seed_omni/utils/hf_layout.py#L271) reads the root `model_type`, and
    [`_write_hf_view`](../../../veomni/models/seed_omni/utils/hf_layout.py#L301) writes the weight-free view once per process (module configs, assets, root config, graphs, `hf_source.json`);
  - `model.seed_omni_model_path` is set to the view; `model.model_path` is unchanged.
- [`_validate_hf_view_modules`](../../../veomni/arguments/omni_arguments_types.py#L141) checks every module comes from the layout; with `save_hf_weights`,
  [`_check_hf_source_exportable`](../../../veomni/arguments/omni_arguments_types.py#L121) checks the source has safetensors
  (skipped for `train.training_task: offline_embedding`, which exports nothing).
- [`OmniConfig.load_checkpoint_sidecars`](../../../veomni/models/seed_omni/configuration_omni.py#L421) records the `HFSource`, which each `ModuleRuntime` receives.
- `ModuleRuntime._build_model` builds the module on meta and
  [attaches the key converter](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L347);
  [`weights_path`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L233) is the upstream root.
- The FSDP2 / DDP wrap (per module, or [composed](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L296)) loads weights:
  - [`load_model_weights`](../../../veomni/models/checkpoint/weights.py#L410) and the [rank-0 broadcast loader](../../../veomni/models/checkpoint/weights.py#L1207) skip keys the converter declares, without reading them;
  - [`HFSourceKeyConverter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L331) renames the module's own keys and hands them to the module's converter.

HF export (`CheckpointCallback` → [`OmniModelRuntime.save_hf_or_lora`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L611)):

- [`_save_hf_source_layout`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L624) exports LoRA modules' adapters per module, skips frozen modules, and makes sure each trained module's DCP exists;
- [`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L673) renames keys back (raising on a key with no source key or a different shape), casts to the source dtypes, fills tied duplicates, copies every other source tensor on rank 0, and writes `global_step_N/hf_ckpt/` with the source's shard files and non-weight files.

Bare `OmniModel.from_pretrained` and eager inference:

- [`OmniModel.from_pretrained`](../../../veomni/models/seed_omni/modeling_omni.py#L203) resolves the root first;
- [`_load_module_from_hf_source`](../../../veomni/models/seed_omni/modeling_omni.py#L343) loads each module with [`load_module_from_hf_source`](../../../veomni/models/seed_omni/utils/hf_layout.py#L483) and binds its assets from the view;
- [`resolve_hf_source_device_map`](../../../veomni/models/seed_omni/utils/hf_layout.py#L449) gives `device_map` its `from_pretrained` meaning (`"auto"` is planned by accelerate over `_no_split_modules`). A multi-device map loads on CPU, then `accelerate.dispatch_model` places the module.

Offline convert: a family with a layout and no converter splits through
[`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L535),
called from [`_run_converter`](../../../veomni/models/seed_omni/utils/convert_registry.py#L200).

## Rules

- Prefixes rename only. A module may still reshape, fuse or split tensors in
  its checkpoint tensor converter, which receives the renamed keys, but such a
  module only loads from the upstream root: the export writes live keys back
  through the prefixes alone, and raises when that cannot reproduce the
  source.
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
