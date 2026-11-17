# Module-level configuration

VeOmni model configurations inherit `VeOmniPretrainedConfig`. The sparse
`ops_implementation` dictionary owns the selected implementations;
`ops_config` caches a normally constructed `OpsImplementationConfig` with defaults
and validation. Updating the dictionary refreshes that runtime cache on its next
read; the cache is not serialized.
Neither importing VeOmni nor constructing an adapter changes native HF config
classes. `OmniConfig` and `OmniModuleConfig` use this same base.

## Loading and construction contract

| Interface | Owner and behavior | Framework caller |
|---|---|---|
| [get_veomni_config_class](../../veomni/models/configuration_base.py#L333) | Framework factory for constructing adapted HF schemas. Returns a cached subclass of both the VeOmni base and the requested HF schema. Native validation and `model_type` remain available. Handwritten extensions may inherit the VeOmni base and HF schema directly. | [get_model_config](../../veomni/models/registry.py#L76), [VeOmniPreTrainedModel.__init_subclass__](../../veomni/models/modeling_base.py#L25) |
| [VeOmniPretrainedConfig.__init_subclass__](../../veomni/models/configuration_base.py#L63) | Framework-provided class lifecycle. Recursively adapts declared `sub_configs` on a new class without modifying the native schema's mapping. Applies to factory adapters, handwritten HF extensions and `OmniModuleConfig` subclasses. Overrides must call `super()`. | Python subclass creation, including the cached adapter factory |
| [VeOmniPretrainedConfig.__init__](../../veomni/models/configuration_base.py#L73) | Framework-provided cooperative constructor. Forwards keyword arguments to the next HF schema in the inheritance order, so custom constructors retain its fields and post-init behavior. Custom constructors must call `super().__init__`. | Handwritten model and module config constructors |
| [as_veomni_config](../../veomni/models/configuration_base.py#L361) | Framework helper; callers may explicitly opt an existing HF config tree in. Adapts in place, preserving references and supplied native child instances. Validation can be deferred until load overrides are complete. | [build_foundation_model](../../veomni/models/auto.py#L164), [VeOmniPreTrainedModel.__init__](../../veomni/models/modeling_base.py#L30) |
| [VeOmniPretrainedConfig.__post_init__](../../veomni/models/configuration_base.py#L76) | Framework-provided lifecycle. Custom HF post-init overrides must call `super()`. Concrete HF initialization runs, explicit child selections survive its attention broadcasts, then parent defaults are merged beneath child selections. Empty selections remain sparse. | HF config constructor and [from_dict](../../veomni/models/configuration_base.py#L245) |
| [ops_config](../../veomni/models/configuration_base.py#L100) | Framework-provided cached read projection; models consume its fields while constructing handles. Uses the standard ops dataclass constructor and reuses the instance while selections remain unchanged. | Model and module constructors |
| [_apply_ops_overrides](../../veomni/models/configuration_base.py#L209) | Framework-provided load boundary. Applies sparse or complete ops selections plus explicit attention, validates the resulting tree, and updates it atomically. Callers must apply all load overrides together. | [get_model_config](../../veomni/models/registry.py#L76), [build_foundation_model](../../veomni/models/auto.py#L164), config [from_dict](../../veomni/models/configuration_base.py#L245) |
| [_set_ops_implementation](../../veomni/models/configuration_base.py#L160) | Framework-provided update helper. A mapping changes its supplied keys; `OpsImplementationConfig` replaces the selection. Every affected config gets its own dictionary. Failure leaves the existing tree unchanged. | Post-init and the load boundary; standalone module tests may call it explicitly. |

`get_veomni_config_class` keeps the native schema's class name because HF's lazy
AutoModel mappings use that name. Config instances, including nested adapters,
can be pickled and reconstructed in a fresh process without importing a dynamic
class from the parent's globals. Handwritten subclasses keep their exact type.
When a remote schema registers an HF Auto class, saving uses a native config
copy so HF exports that schema's original source and relative dependencies.
The saved `auto_map` points to the original remote class; loading through
VeOmni adapts it again and restores the saved ops selections.

## Attention and override precedence

`config._attn_implementation` reads the attention selection from the same
dictionary, using the ops dataclass default when it is unset. This read does
not validate intermediate load layers.
Its setter normalizes supported aliases and accepts a scalar or nested mapping.
A scalar targets the entire declared subtree. A mapping targets only named
children; the empty key `""` targets its own node. Untargeted children retain
their choices. Setting `None` removes the explicit selection and exposes the
default again.

HF also writes `_attn_implementation_internal` directly during model
initialization. On VeOmni configs this name is a property alias into the same
dictionary. These private writes affect their own node, preserving child-local
choices. There is no independent private attention value to synchronize or
backfill in model builders.

Initialization merges parent defaults below child and grandchild selections,
for both mapping and typed parent inputs.
At one constructor, an explicit ops attention selection ranks above a scalar
HF `attn_implementation`; explicit nested attention maps target their named
subtrees. At load time, saved selections are merged with caller ops overrides,
then caller `attn_implementation` wins. Sparse mappings preserve unspecified
child choices; a complete `OpsImplementationConfig` replaces them.

Composed `OmniConfig` load kwargs merge into the root or named module entry
before sidecars load. They accept dictionaries and typed
`OpsImplementationConfig` selections. Typed selections contribute every
non-`None` value from `to_dict()`, including dataclass defaults; unspecified
fields remain in that layer. Module entry choices still override root choices.
Standalone `OmniModuleConfig.from_pretrained` applies an explicit
`attn_implementation` after the file, root-default, and entry ops layers, then
validates the final selection. Composed loading passes root defaults separately,
so module entry choices retain their precedence over those defaults.

The handwritten Wan T2V, LTX, and Qwen-Image transformer constructors also accept
`attn_implementation`. They write a non-`None` override to the owning config
before HF initialization checks the backend and before attention handles bind.
An omitted argument or `None` preserves the config's existing selection; other
ops remain unchanged. The override uses the same config state as normal loading.

Reading and merging load overrides use dictionaries without constructing an
intermediate ops dataclass. Composed loaders pass `_validate_ops=False` while
reading a lower layer, then validate the final module selection once all layers
are merged. No ambient validation context is needed. Validation runs after load
overrides, so a checkpoint saved with device-specific ops can be loaded on
another device by replacing those selections. Model bases validate the final
tree before HF initialization, reusing each config's cached ops instance.
The composed `OmniModel` is a graph container: it uses HF initialization directly
because its root ops are inheritance defaults, not kernel selections it executes.
Each module validates its effective selection after root defaults and entry
overrides are merged. A GPU-specific root default may therefore remain in a
checkpoint loaded on NPU when every affected module overrides it; an incompatible
selection left on any module still raises.
Changing
ops dictionaries after model construction does not replace bound `VeomniOp`
handles. Wrappers that execute no HF attention, including condition model
containers and the token embedding/projecting `TextEncoder`, override HF's
`_check_and_adjust_attn_implementation` to return the canonical selection.
This avoids unsupported capability checks and kernel preloading on a container;
components that execute attention retain their own backend validation. Keep
this override on the owning class, not on the shared model base. MiniMax's
custom packed-attention dispatcher also overrides the HF check: it validates
and binds FlashAttention on the first forward through
`_load_packed_attention_kernel`, preserving deferred loading even when HF
reads a canonical flash selection during construction.
Unsupported MiniMax flash selections, including FA4, fail before DiT execution
for scalar, singleton-list, and packed inputs; serial execution does not fall
back to SDPA while the config advertises another backend.
HF mask builders still read the canonical attention selection at
forward time, so keep it consistent with the constructed attention handle.

When a child resets a field to its device default while its parent selects an
implementation, its serialized ops diff records `null` for that field. On reload,
this prevents parent inheritance from replacing the child's reset. The runtime
ops dataclass omits these reset markers when applying its normal defaults.
HF load-time logging can serialize a caller-supplied native child before the
loader adapts it; a child without VeOmni metadata is treated as having no explicit
ops selections during that intermediate serialization.

## Opt in

A handwritten config inherits the base and forwards its ops argument:

```python
from veomni.models.configuration_base import VeOmniPretrainedConfig

class ExampleConfig(VeOmniPretrainedConfig):
    model_type = "example"

    def __init__(self, hidden_size=64, ops_implementation=None, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(ops_implementation=ops_implementation, **kwargs)
```

To extend an HF schema, use normal multiple inheritance:

```python
from transformers import LlamaConfig as HFLlamaConfig
from veomni.models import VeOmniPretrainedConfig

class ExampleLlamaConfig(VeOmniPretrainedConfig, HFLlamaConfig):
    pass
```

Declared `sub_configs` are recursively adapted when the class is defined. This
also applies when the VeOmni base is `OmniModuleConfig`. A custom constructor
forwards schema fields and `ops_implementation` through `super().__init__`.

For an HF schema, construct an adapted config explicitly when testing a leaf
module such as an attention layer or MLP:

```python
from transformers import LlamaConfig
from veomni.models import get_veomni_config_class

Config = get_veomni_config_class(LlamaConfig)
config = Config(ops_implementation={"attn_implementation": "eager"})
assert config.ops_config.attn_implementation == "eager"
```

Generated model specs replace the HF model base import with the VeOmni base:

```python
config.drop_import_names("PreTrainedModel")
config.add_import(
    "veomni.models.modeling_base",
    names=["VeOmniPreTrainedModel as PreTrainedModel"],
)
```

Native HF classes used outside these entry points keep their native behavior
and have no VeOmni `ops_config`. In `MODELING_BACKEND=hf`, the
[HF loader](../../veomni/models/loader.py#L46) copies adapter configs back to
native types before AutoModel dispatch.

One loading request flows through:

1. [build_foundation_model](../../veomni/models/auto.py#L164) receives model ops.
   1. [get_model_config](../../veomni/models/registry.py#L76) selects a registered config or an adapted HF schema.
   2. [from_dict](../../veomni/models/configuration_base.py#L245) initializes the complete tree with validation deferred.
   3. [_apply_ops_overrides](../../veomni/models/configuration_base.py#L209) merges overrides and validates the final selection.
2. [VeOmniPreTrainedModel](../../veomni/models/modeling_base.py#L22) adapts direct model inputs and validates their final ops tree before HF initialization. The composed `OmniModel` initializes its HF container after its modules have independently validated their final selections.
   1. Model constructors bind instance-local handles from `config.ops_config`.
   2. Forward calls use those handles.
3. Config serialization stores sparse ops selections; unchanged inherited child selections are omitted from the diff.

[Config lifecycle tests](../../tests/ops/base/test_ops_config.py) cover native
HF isolation, recursive merging, save/reload, fresh-process pickling, canonical
attention storage, atomic validation, cross-device overrides, and Omni root
configuration. [Registry tests](../../tests/models/base/test_auto_registry.py)
exercise model families and both backends. Wan and Qwen-Image model tests cover
their local attention dispatch and persistence.
