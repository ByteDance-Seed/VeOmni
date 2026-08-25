# Accelerated Mixin IDE Type Stubs

Read this when editing `accelerated.py` `TrainingMixin` hooks that call into
the sibling `modeling.py` class.

## Why

Training-graph hooks live on `TrainingMixin` (`accelerated.py`), but weights,
`forward` / `encode` / … **and** `generate()` / FSM inference live on the
native model class in `modeling.py`:

```python
# modeling.py
class InferenceMixin:
    """generate() + FSM inference state — analogous to HF's GenerationMixin."""
    ...

class BagelFlowConnector(InferenceMixin, OmniPreTrainedModel):
    ...

# accelerated.py
class BagelFlowConnectorAccelerated(VeOmniMixin, BagelFlowConnector): ...
```

At edit time a `TrainingMixin` hook only sees `self`, not the merged MRO of
the final `Accelerated` model class. **Class-level annotations and `...`
method stubs** tell the IDE which attributes and modeling methods exist,
without copying implementation into the mixin file.

Live reference: `veomni/models/seed_omni/modules/bagel/flow_connector/accelerated.py`.

## File shape

Each module folder splits capabilities across two files:

```python
# modeling.py — pure HF-native.
class InferenceMixin:
    """generate() + FSM state — omit if the module isn't inference-capable."""
    def generate(self, conversation_list=None, **kwargs): ...

class Xxx(InferenceMixin, OmniPreTrainedModel):
    """InferenceMixin listed FIRST: OmniPreTrainedModel ships no-op
    reset_local_inference_state / reset_global_inference_state / finalize
    defaults, and MRO resolves left-to-right — second, those no-ops would
    shadow the real implementations above."""
    def forward(self, ...): ...

# accelerated.py — VeOmni-only training-graph hooks. No InferenceMixin here:
# generate() / reset_* / finalize already reach XxxAccelerated unshadowed via
# normal inheritance from Xxx.
class TrainingMixin(TrainingModuleMixin):
    ...

class MeterMixin(MetricMeterMixin):
    ...

class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    ...

class XxxAccelerated(VeOmniMixin, Xxx):
    ...
```

A handful of backbones (`qwen3/llm`, `qwen3_moe/llm`) share a family-wide
`SimpleArGenerationMixin` (`modules/base/llm_packing.py`) instead of a
per-module `InferenceMixin` — same rule: listed before `OmniPreTrainedModel`.

Do **not** put modeling logic in the `TrainingMixin` stubs. Stubs are for
static analysis and navigation only.

## What to declare

On `TrainingMixin`, at the top of the body (before `__init__` or hooks):

| Kind | When | Example |
|------|------|---------|
| `config: XxxConfig` | always | typed module config |
| `device: torch.device` | hooks touch `self.device` | from `PreTrainedModel` |
| `dtype: torch.dtype` | hooks touch `self.dtype` | from `PreTrainedModel` |
| `training: bool` | FSDP dummy-anchor gating | from `nn.Module.training` |
| modeling-owned attrs | hook reads `self._tokenizer`, `self.model`, … | match `modeling.py` field names |
| method stubs | hook calls `self.encode(...)`, `self.forward(...)`, … | signature copied from `modeling.py` |

**Scope rule:** `TrainingMixin` declares **only what its own hooks use**. Do
not duplicate the full model surface. `generate()` and its FSM helpers live
entirely on the native class's `InferenceMixin` now — `accelerated.py` needs
no IDE stub for `generate` itself unless a training hook calls a
`generate`-only helper.

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
- Import config / processor types at the top of `accelerated.py` when used in
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

1. Grep `accelerated.py` for `self.<name>` calls not defined in the file.
2. If `<name>` is implemented in `modeling.py`, add or update a stub on
   `TrainingMixin`.
3. If `<name>` is a class attribute on the model, add a typed class attribute
   on `TrainingMixin`.
4. Copy the docstring template and verify jump-to-definition lands in
   `modeling.py`.
5. Do **not** add stubs for helpers already defined in the same file, for
   module-level carrier helpers, or for framework mixins (`metric_meter_set_seqlens`,
   `pre_forward`, …).

## Module-level carrier helpers

Training hooks and the native `InferenceMixin`'s `generate()` often share
carrier logic (select items, pack tensors, scatter outputs). When a helper
does **not** need hook-local `self._*` state, define it as a plain function at
the top of `modeling.py` and call it from both `modeling.py` and
`accelerated.py` — do not hang it on `TrainingMixin` and reach it via MRO from
the native class.

Reference: `modules/bagel/flow_connector/modeling.py`
(`select_vae_context_latent_items`, `scatter_flow_latent_embeds`).

Standard AR LLM backbones (1-D positions) reuse
:func:`~veomni.models.seed_omni.modules.base.llm_packing.pack_llm_conversations_for_forward`
from ``modules/base/llm_packing.py`` instead of duplicating per family.
All AR families (including Qwen3-VL) reuse
:func:`~veomni.models.seed_omni.modules.base.llm_packing.scatter_llm_hidden_states`
in ``forward_post``.

Pass `device`, `dtype`, and config fields explicitly inside these functions.
Keep training-only SP bookkeeping (`self._sp_*`) in the `TrainingMixin` hook
methods.

## Modules with full stub coverage

All `modules/**/accelerated.py` files follow this pattern. When adding a
module, mirror an adjacent example:

| Pattern | Example module |
|---------|----------------|
| Vision encode helper | `janus/siglip` (`_encode_pixel_values`) |
| AR backbone `forward` | `janus/llama`, `qwen3/llm`, `qwen3vl/llm` |
| Text encoder `encode` / `_project` | `base/text_encoder` + family wrappers |
| VAE codec | `bagel/vae` (`encode`, `decode`) |
| Flow / velocity heads | `bagel/flow_connector` (`embed_latent`, `decode_velocity`) |
| Packed inference | `bagel/qwen2_mot` (`forward_inference`) |
