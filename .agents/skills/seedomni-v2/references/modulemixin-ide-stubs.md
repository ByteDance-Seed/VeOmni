# ModuleMixin IDE Type Stubs

Read this when editing `modulemixin.py` hooks that call into the sibling
`modeling.py` class.

## Why

Graph hooks live on `TrainingMixin` / `InferenceMixin`, but weights and
`forward` / `encode` / … live on the concrete model in `modeling.py`:

```python
class BagelFlowConnector(VeOmniMixin, PreTrainedModel): ...
```

At edit time a hook only sees `self`, not the merged MRO of the final model
class. **Class-level annotations and `...` method stubs** tell the IDE which
attributes and modeling methods exist, without copying implementation into the
mixin file.

Live reference: `veomni/models/seed_omni/modules/bagel/flow_connector/modulemixin.py`.

## Mixin file shape

Each module folder splits capabilities progressively:

```python
class TrainingMixin(TrainingModuleMixin):
    ...

class InferenceMixin(InferenceModuleMixin):
    ...

class MeterMixin(MetricMeterMixin):
    ...

class VeOmniMixin(BaseMixin, TrainingMixin, InferenceMixin, MeterMixin):
    ...
```

`modeling.py` only declares:

```python
class Xxx(VeOmniMixin, PreTrainedModel):
    ...
```

Do **not** put modeling logic in the mixin stubs. Stubs are for static analysis
and navigation only.

## What to declare

On each mixin class, at the top of the body (before `__init__` or hooks):

| Kind | When | Example |
|------|------|---------|
| `config: XxxConfig` | always | typed module config |
| `device: torch.device` | hooks touch `self.device` | from `PreTrainedModel` |
| `dtype: torch.dtype` | hooks touch `self.dtype` | from `PreTrainedModel` |
| `training: bool` | FSDP dummy-anchor gating | from `nn.Module.training` |
| modeling-owned attrs | hook reads `self._tokenizer`, `self.model`, … | match `modeling.py` field names |
| method stubs | hook calls `self.encode(...)`, `self.forward(...)`, … | signature copied from `modeling.py` |

**Scope rule:** each mixin declares **only what its own hooks use**. Do not
duplicate the full model surface on every mixin.

## Method stub style

Use a real signature and an ellipsis body. Add a one-line docstring naming the
modeling class:

```python
def embed_latent(
    self,
    latents: torch.Tensor,
    position_ids: torch.LongTensor,
    timesteps: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """IDE stub — implemented on :class:`BagelFlowConnector` in ``modeling.py``."""
    ...
```

Conventions:

- Docstring prefix: `IDE stub — implemented on :class:`…`` in ``modeling.py``.
- Replace `BagelFlowConnector` with the concrete class from that module's
  `modeling.py` (`TextEncoder`, `Qwen3Llm`, `BagelVAE`, …).
- Signatures must match `modeling.py` exactly (args, types, return type).
- Import config / processor types at the top of `modulemixin.py` when used in
  annotations (`from __future__ import annotations` is fine).

## Property stubs

When a hook on `TrainingMixin` reads a property defined on `VeOmniMixin` in
the **same file**, stub the property on `TrainingMixin` for isolated editing:

```python
@property
def _spatial_merge_size(self) -> int:
    """IDE stub — see :class:`VeOmniMixin` below (``config.spatial_merge_size``)."""
    ...
```

Use the `see :class:`VeOmniMixin` below` form when the real `@property` lives
in this file, not in `modeling.py`.

## Checklist for a new / changed hook

1. Grep the mixin for `self.<name>` calls not defined in the mixin file.
2. If `<name>` is implemented in `modeling.py`, add or update a stub on the
   mixin that owns the hook (`TrainingMixin` vs `InferenceMixin`).
3. If `<name>` is a class attribute on the model, add a typed class attribute
   on the mixin.
4. Copy the docstring template and verify jump-to-definition lands in
   `modeling.py`.
5. Do **not** add stubs for helpers already defined in the same mixin file, for
   module-level carrier helpers, or for framework mixins (`metric_meter_set_seqlens`,
   `pre_forward`, …).

## Module-level carrier helpers

Training and inference hooks often share carrier logic (select items, pack
tensors, scatter outputs). When a helper does **not** need hook-local
`self._*` state, define it as a plain function at the top of `modulemixin.py`
and call it from both mixins — do not hang it on `TrainingMixin` and reach it
via MRO from `InferenceMixin`.

Reference: `modules/bagel/flow_connector/modulemixin.py`
(`select_vae_context_latent_items`, `scatter_flow_latent_embeds`).

Standard AR LLM backbones (1-D positions) reuse
:func:`~veomni.models.seed_omni.modules.base.llm_packing.pack_llm_conversations_for_forward`
from ``modules/base/llm_packing.py`` instead of duplicating per family.
All AR families (including Qwen3-VL) reuse
:func:`~veomni.models.seed_omni.modules.base.llm_packing.scatter_llm_hidden_states`
in ``forward_post``.

Pass `device`, `dtype`, and config fields explicitly inside these functions.
Keep training-only SP bookkeeping (`self._sp_*`) in the mixin hook methods.

## Modules with full stub coverage

All `modules/**/modulemixin.py` files follow this pattern. When adding a module,
mirror an adjacent example:

| Pattern | Example module |
|---------|----------------|
| Vision encode helper | `janus/siglip` (`_encode_pixel_values`) |
| AR backbone `forward` | `janus/llama`, `qwen3/llm`, `qwen3vl/llm` |
| Text encoder `encode` / `_project` | `base/text_encoder` + family wrappers |
| VAE codec | `bagel/vae` (`encode`, `decode`) |
| Flow / velocity heads | `bagel/flow_connector` (`embed_latent`, `decode_velocity`) |
| Packed inference | `bagel/qwen2_mot` (`forward_inference`) |
