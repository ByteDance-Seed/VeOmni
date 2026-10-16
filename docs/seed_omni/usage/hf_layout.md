# Upstream Checkpoint Layout (`OmniHFLayout`)

An `OmniHFLayout` tells SeedOmni how a family's upstream HuggingFace checkpoint
splits into its modules. With one registered, `model.model_path` (and
`OmniModel` / `OmniConfig` / `OmniProcessor.from_pretrained`) can name the
upstream checkpoint itself: no offline convert, and the HF export is written
back in the upstream layout, so it loads wherever the source does.

- Declaration: [`OmniHFLayout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L143) and [`OmniHFModuleLayout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L105)
- Registry: [`OMNI_HF_LAYOUT_REGISTRY`](../../../veomni/models/seed_omni/utils/hf_layout.py#L80), keyed by the upstream `model_type`
- Reference family: [`QWEN3_HF_LAYOUT`](../../../veomni/models/seed_omni/modules/qwen3/hf_layout.py#L42)

User-facing behavior and limits are in
[Load an upstream checkpoint directly](training_and_inference.md#21-load-an-upstream-checkpoint-directly).

## When to use a layout

Use a layout when every module is a key-prefix slice of the upstream checkpoint:
the module's weights are the upstream ones under a renamed prefix, possibly
transformed by the module's own checkpoint tensor converter. Keep a
[family converter](adding_a_model.md#1-split-the-checkpoint) when the split
needs more than that (weights from several checkpoints, re-packed tensors,
generated assets). A family with both uses its converter for the offline
convert.

## Declaring a layout

Put it in `veomni/models/seed_omni/modules/<family>/hf_layout.py` and import
that file from the family's `__init__.py`, so the registry is filled when
`veomni.models.seed_omni.modules` is imported:

```python
from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .llm.configuration import Qwen3LlmConfig
from .text_encoder.configuration import Qwen3TextEncoderConfig


def _tokenizer(model_path):
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)}


QWEN3_HF_LAYOUT = OmniHFLayout(
    modules={
        "qwen3_text_encoder": OmniHFModuleLayout(
            key_prefixes=(("model.embed_tokens.", "embed_tokens."), ("lm_head.", "lm_head.")),
            build_config=lambda hf: Qwen3TextEncoderConfig(
                vocab_size=hf.vocab_size, hidden_size=hf.hidden_size, tie_word_embeddings=hf.tie_word_embeddings
            ),
            build_assets=_tokenizer,
        ),
        "qwen3_llm": OmniHFModuleLayout(
            key_prefixes=(("model.", "language_model."),),
            build_config=lambda hf: Qwen3LlmConfig(text_config=hf.to_dict()),
        ),
    },
    tied_source_keys={"lm_head.weight": "model.embed_tokens.weight"},
    graph_dir="configs/seed_omni/Qwen/qwen3_0.6b/train",
    training_graphs="graph_train.yaml",
    generation_graphs={"infer_text": "graph_infer.yaml"},
    infer_type="infer_text",
)


@OMNI_HF_LAYOUT_REGISTRY.register("qwen3")
def _register_qwen3_hf_layout(source_path=None):
    return QWEN3_HF_LAYOUT
```

A runnable layout over the in-tree `fake_module_a` / `fake_module_b` stand-ins
is `FAKE_HF_LAYOUT` in
[`tests/seed_omni/model/test_hf_layout.py`](../../../tests/seed_omni/model/test_hf_layout.py).

## Contract

The family declares the fields below. Everything else is framework-provided.

| Field | Owner | What it does | Where the framework reads it | End effect |
|-------|-------|--------------|------------------------------|------------|
| `modules[name].key_prefixes` | must declare | `(source_prefix, module_prefix)` pairs. A source key belongs to the module whose source prefix is the longest match across the whole layout, so a backbone can claim `model.` while a sibling claims `model.embed_tokens.`. A prefix claimed twice is an error. Prefixes match source keys after the module class's `_checkpoint_conversion_mapping`, if it has one. | [`OmniHFLayout.route`](../../../veomni/models/seed_omni/utils/hf_layout.py#L199) when loading, and its reverse [`source_key`](../../../veomni/models/seed_omni/utils/hf_layout.py#L208) when saving | Each upstream weight lands on exactly one module, and exports under its upstream name. Unclaimed keys are never read, and are copied verbatim into an export. |
| `modules[name].build_config` | must declare | Upstream HF config to the module's `OmniModuleConfig`. | [`build_module_configs`](../../../veomni/models/seed_omni/utils/hf_layout.py#L229), called by [`_write_hf_view`](../../../veomni/models/seed_omni/utils/hf_layout.py#L350) and [`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L613) | The module config every load path builds the module from. |
| `modules[name].build_assets` | optional | Loads the module's assets from the checkpoint directory, keyed `tokenizer` / `processor` / `image_processor` / `video_processor`. | [`build_module_assets`](../../../veomni/models/seed_omni/utils/hf_layout.py#L233), same callers | The assets are saved into the module's view subfolder, so the module and the collator bind them as from a split root. |
| `modules[name].generated_tensors` | optional | `(checkpoint_dir, module_config) -> {module_key: tensor}` for module tensors the checkpoint does not store (e.g. a lookup table built from the tokenizer). | [`HFSourceKeyConverter.finalize`](../../../veomni/models/seed_omni/utils/hf_layout.py#L447) | Every loader dispatches them after the stored weights. They have no source key, so an export drops them. |
| `modules[name].export_state` | optional | `(module, state_dict) -> state_dict`: the module's reverted state in the form the checkpoint stores, under module keys `source_key` maps back (requantized weights with their scales, per-expert tensors split out of fused ones). A plain tensor is written by the rank that returns it. | [`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L780) | Needed whenever the source stores a quantized dtype: without it the export refuses to cast a float weight into fp8 / int8. |
| `tied_source_keys` | optional | Source key → the source key it duplicates when the model ties them (a checkpoint may store `lm_head.weight` although the config ties it). | [`attach_hf_source_converter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L469) and [`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L780) | Loading skips the duplicate when the module does not hold it; an export writes the original's trained value under it, not the stale copy. |
| `graph_dir`, `training_graphs`, `generation_graphs`, `infer_type` | optional | The family's default graph YAML (repo-relative, see `load_family_graphs`) and default generation scenario. | [`load_graphs`](../../../veomni/models/seed_omni/utils/hf_layout.py#L239) | The view carries these graphs; a launcher graph overrides them. Run outside a VeOmni checkout, the view carries none and the launcher has to supply them. |
| `load_hf_config` | optional | Reads the upstream config; `AutoConfig.from_pretrained` when unset. | [`read_hf_config`](../../../veomni/models/seed_omni/utils/hf_layout.py#L222) | For checkpoints `AutoConfig` cannot read. |
| registration | must call | `OMNI_HF_LAYOUT_REGISTRY.register("<upstream model_type>")` on a function taking the checkpoint directory (`None` when there is none) and returning the layout. A layout whose key map depends on the checkpoint (a per-layer rename needs the layer count) reads its config there; the result is cached per directory. | [`resolve_omni_checkpoint_root`](../../../veomni/models/seed_omni/utils/hf_layout.py#L320) | A root whose `config.json` has that `model_type` loads through the layout; an unregistered non-`omni` type is an error. |

Framework-provided, not to be overridden:
[`resolve_omni_checkpoint_root`](../../../veomni/models/seed_omni/utils/hf_layout.py#L320),
[`HFSourceKeyConverter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L380),
[`attach_hf_source_converter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L469),
[`load_module_from_hf_source`](../../../veomni/models/seed_omni/utils/hf_layout.py#L561),
[`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L613) and
[`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L780).

## Call flow

Training from an upstream root:

- [`build_omni_model_runtime_args`](../../../veomni/arguments/omni_arguments_types.py#L173) calls
  [`_resolve_hf_checkpoint_view`](../../../veomni/arguments/omni_arguments_types.py#L97):
  - [`resolve_omni_checkpoint_root`](../../../veomni/models/seed_omni/utils/hf_layout.py#L320) reads the root `model_type`, and
    [`_write_hf_view`](../../../veomni/models/seed_omni/utils/hf_layout.py#L350) writes the weight-free view once per process (module configs, assets, root config, graphs, `hf_source.json`);
  - `model.seed_omni_model_path` is set to the view; `model.model_path` is unchanged.
- [`_validate_hf_view_modules`](../../../veomni/arguments/omni_arguments_types.py#L141) checks every module comes from the layout; with `save_hf_weights`,
  [`_check_hf_source_exportable`](../../../veomni/arguments/omni_arguments_types.py#L121) checks the source has safetensors.
- [`OmniConfig.load_checkpoint_sidecars`](../../../veomni/models/seed_omni/configuration_omni.py#L421) records the `HFSource`, which each `ModuleRuntime` receives.
- `ModuleRuntime._build_model` builds the module on meta and
  [attaches the key converter](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L371);
  [`weights_path`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L233) is the upstream root.
- The FSDP2 / DDP wrap (per module, or [composed](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L306)) loads weights:
  - [`load_model_weights`](../../../veomni/models/module_utils.py#L406) and the [rank-0 broadcast loader](../../../veomni/models/module_utils.py#L1299) skip keys the converter declares, without reading them;
  - [`load_model_weights_ep_sharded`](../../../veomni/models/module_utils.py#L656) (`ep_sharded_stream_load`) renames each source key with the converter before planning its rank's slice, so per-expert source keys are read only by the rank that owns the expert, and dispatches the layout's `generated_tensors` last;
  - [`HFSourceKeyConverter`](../../../veomni/models/seed_omni/utils/hf_layout.py#L380) renames the module's own keys and hands them to the module's converter.

HF export (`CheckpointCallback` → [`OmniModelRuntime.save_hf_or_lora`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L611)):

- [`_save_hf_source_layout`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L624) exports LoRA modules' adapters per module, skips frozen modules, and makes sure each trained module's DCP exists;
- [`save_hf_source_checkpoint`](../../../veomni/models/seed_omni/utils/hf_layout.py#L780) applies each module's `export_state`, renames keys back, refuses a key with no source key or a different shape, casts to the source dtypes (float to float only), fills tied duplicates, copies every other source tensor (spread over the ranks), and writes `global_step_N/hf_ckpt/` with the source's shard files and non-weight files.

Bare `OmniModel.from_pretrained` and eager inference:

- [`OmniModel.from_pretrained`](../../../veomni/models/seed_omni/modeling_omni.py#L203) resolves the root first;
- [`_load_module_from_hf_source`](../../../veomni/models/seed_omni/modeling_omni.py#L348) loads each module with [`load_module_from_hf_source`](../../../veomni/models/seed_omni/utils/hf_layout.py#L561) and binds its assets from the view;
- [`resolve_hf_source_device_map`](../../../veomni/models/seed_omni/utils/hf_layout.py#L527) gives `device_map` its `from_pretrained` meaning (`"auto"` is planned by accelerate over `_no_split_modules`). A multi-device map loads on CPU, then `accelerate.dispatch_model` places the module.

Offline convert: a family with a layout and no converter splits through
[`convert_with_hf_layout`](../../../veomni/models/seed_omni/utils/hf_layout.py#L613),
called from [`_run_converter`](../../../veomni/models/seed_omni/utils/convert_registry.py#L200).

## Rules

- Prefixes rename only. Reshape, fuse or split tensors in the module's
  checkpoint tensor converter, which receives the renamed keys, and undo it in
  `export_state` when the module's weight conversions do not revert it.
- Each module claims disjoint weights: a source key may route to one module
  only, and two modules must not export the same source key.
- `build_config` must give the config the module had when split by the
  family's converter, so a split checkpoint and the direct load hold the same
  module.
- Write `build_assets` to load from the directory it is given; the view saves
  what it returns with `save_pretrained`.

## Tests

[`tests/seed_omni/model/test_hf_layout.py`](../../../tests/seed_omni/model/test_hf_layout.py)
covers routing, the view, loading, untied and tied heads, the export round trip
(single file and sharded), the offline convert, and the argument checks.
[`tests/seed_omni/runtime/test_omni_model_runtime.py`](../../../tests/seed_omni/runtime/test_omni_model_runtime.py)
covers the export orchestration.
