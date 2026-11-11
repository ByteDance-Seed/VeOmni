# Adding a Model to SeedOmni

This page is the checklist for bringing a new model family into SeedOmni. Read
[Architecture](../design/architecture.md) first for the module split, the
conversation carrier and the graph views. Agents should use the
`/veomni-seed-omni` skill, which carries the same checklist plus review
references.

## 1. Split the checkpoint

Register a converter in `veomni/models/seed_omni/modules/<family>/convert_model.py`
(dispatched by `scripts/seed_omni/convert_model.py` on the upstream
`model_type`). It returns the upstream HF checkpoint as one module instance per
module, with processors and tokenizers attached through `attach_module_assets`,
plus the family's default graphs. `convert_checkpoint` saves the result through
`OmniModel.save_pretrained`: a root `config.json`, graph sidecars, and one
self-contained subfolder per module (`config.json`, `model.safetensors`, and any
processor or tokenizer files).

When every module is a key-prefix slice of the upstream checkpoint, register an
`OmniHFLayout` in `modules/<family>/hf_layout.py` under the upstream `model_type`
instead (`modules/qwen3/hf_layout.py` is the reference). Per module it declares
the `key_prefixes` map (source prefix → module prefix; the longest match wins),
`build_config` (HF config → module config) and `build_assets` (tokenizer /
processors); `tied_source_keys` names stored duplicates of tied weights. A
layout gives both the offline convert and the
[direct load and HF-layout export](training_and_inference.md#21-load-an-upstream-checkpoint-directly);
the contract is in [Upstream Checkpoint Layout](hf_layout.md).

## 2. Write each module

Under `veomni/models/seed_omni/modules/<family>/<sub>/`:

- `configuration.py`: a `PretrainedConfig` with a unique `model_type`.
- `modeling.py`: `class X(InferenceMixin, PretrainedOmniModule)`, pure
  HF-native. It holds `__init__`, the submodule layout and `forward`, and, if the
  module takes part in inference, an in-file `InferenceMixin` with the FSM
  `generate()` and its state. `InferenceMixin` must come **before**
  `PretrainedOmniModule` in the bases. The class must load and run under plain
  `from_pretrained` with no VeOmni import. Declare the HF attention capability
  flags the module accepts on the class itself — `PretrainedOmniModule` leaves
  `_supports_sdpa` / `_supports_flash_attn` / `_supports_flex_attn` at the
  transformers default of `False`, so a module whose inner backbone runs
  through VeOmni's patched attention has to opt in or HF rejects `sdpa` (the
  default) at construction. `modules/qwen3/llm/modeling.py` copies them off the
  inner model; `modules/qwen3vl/llm/modeling.py` sets `_supports_sdpa = True`
  because the wrapper only delegates.
- `accelerated/accelerated.py`: `TrainingMixin` and `VeOmniMixin` plus IDE type
  stubs (below), and `class XAccelerated(VeOmniMixin, X)`. The Janus modules
  under `modules/janus/*/` are the reference pattern.
- `processing.py` (optional): when the module consumes raw images, audio or
  video.

Reuse the shared `TextEncoder` module in `modules/text_encoder/` where possible.

### IDE type stubs

Because `TrainingMixin` hooks call native modeling APIs through `self` while the
implementations live on the `modeling.py` class mixed in at
`XAccelerated(VeOmniMixin, X)` (`VeOmniMixin` precedes `X` in the MRO), declare
class-level attribute annotations directly on `TrainingMixin` and guard any
typing-only method or property stubs with `if TYPE_CHECKING:`:

```python
from typing import TYPE_CHECKING


class TrainingMixin(TrainingModuleMixin):
    config: BagelFlowConnectorConfig
    device: torch.device
    dtype: torch.dtype

    if TYPE_CHECKING:

        def embed_latent(
            self,
            latents: torch.Tensor,
            position_ids: torch.LongTensor,
            timesteps: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            """IDE stub — implemented on :class:`BagelFlowConnector` in ``modeling.py``."""
            ...
```

- Match signatures in `modeling.py` exactly and keep method/property stubs under
  `if TYPE_CHECKING:` so runtime dispatch resolves `X`'s implementation.
- For properties defined on `VeOmniMixin` in the same file, use the docstring
  ``IDE stub — see :class:`VeOmniMixin` below (``config.field``).``

`modules/bagel/flow_connector/accelerated/accelerated.py` shows the full
convention.

## 3. Register

Register the classes in `veomni/models/seed_omni/modules/__init__.py`, keyed by
`model_type`:

| Registry | Holds |
|----------|-------|
| `OMNI_CONFIG_REGISTRY` | `configuration.py` classes |
| `OMNI_MODEL_REGISTRY` | native `modeling.py` classes |
| `OMNI_ACCELERATED_MODEL_REGISTRY` | `accelerated/accelerated.py` classes |
| `OMNI_PROCESSOR_REGISTRY` | `processing.py` classes |

The runtime resolves a module by reading its `config.json`, then `model_type`,
then the registry.

## 4. Write the YAML

Create `configs/seed_omni/<model>/<task>/` following
[the config layout](training_and_inference.md#1-config-layout): `base.yaml`,
`modules_train.yaml`, `graph_train.yaml`, plus shared `infer/` graphs and
`data.yaml`. Edges only declare order; conversation graphs move data through the
conversation list.

If the model needs a new data source, add a preprocessor as described in
[Data Format](data_format.md#custom-datasets).

### 4.1 Packing (Optional)

A second graph shape exists: packed graphs (Qwen3-VL's
`packed/graph_train.yaml` is the in-tree example) call `pack_*` methods and move
packed tensors on the batch dict instead of a conversation carrier. The
text-encoder preprocessor builds one flat token sequence per sample on the CPU —
tokens, M-RoPE positions, visual masks — and every later node only mask-selects
its own embeddings and `masked_scatter`s them back.

**Adding a model does not require it.** Ship the conversation graph first: it is
the reference path, it is what inference uses, and it is what a packed graph must
keep matching. Packing is a throughput optimization, not a capability.

It is also not a padding win, despite how it reads: the conversation path does not
pad either — it concatenates every sample's embed segments into one varlen
sequence too. What packing changes is *where* the per-item work runs. The packed
path has the dataloader worker emit one ready-made tensor triple, so
`batch_to_device` moves it in a single transfer. The conversation path's carrier
tensors are built per item and `batch_to_device` deliberately leaves them on the
host ("placing those is the module's own `pre_forward` job"), so the step pays one
`to(device)` per `ConversationItem` plus a `torch.cat` over all of them.

So the gain is bounded by the step's Python and H2D bookkeeping, not by the
model's FLOPs, and it scales with **item count** rather than token count. Measured
against the real Qwen3-VL-2B checkpoint on one A100 (4 images + two ~400-token
turns per sample):

| Samples / micro-batch | items | per-item placement | one-shot | backbone forward | saved |
|---|---|---|---|---|---|
| 3 | 18 | 1.26 ms | 0.05 ms | 223 ms | 0.54% |
| 8 | 48 | 4.55 ms | 0.08 ms | 511 ms | 0.88% |

On Qwen3-Omni-30B-A3B the same fixed cost measured −1.01% step time at a
VRAM-saturating batch and −0.44% at mbs=2, against `fsdp_scope: model` alone
recovering 0.47% / 0.21% of that same gap.

| Round | `packed` vs `split` | `packed` vs `mfsdp` |
|-------|--------------------|---------------------|
| Qwen3-VL-2B, 3 samples, 4.1k rows | n/a | −0.54% |
| Qwen3-VL-2B, 8 samples, 11k rows | n/a | −0.88% |
| Qwen3-Omni-30B, thinker mbs=12 (VRAM-saturating) | −1.01% | −0.54% |
| Qwen3-Omni-30B, thinker mbs=2 | −0.44% | −0.23% |
| Qwen3-Omni-30B, speech (6 modules), mbs=32 | −3.45% | −0.18% |

Two readings matter here:

- **The wrap scope is the bigger lever.** `split` → `mfsdp` (`fsdp_scope: model`)
  recovers 0.47% / 0.21% / 3.28% on the Omni rounds, leaving the data path
  0.18–0.54% on top — inside packed's own 27 ms jitter at the saturated batch.
- **It buys step-side overhead, nothing else.** Where a module runs one utterance
  at a time regardless (the talker), or the micro-batch holds few samples, there
  is little left to win.

#### Where packing does not apply

Packing needs **one shared token axis**. The packer emits a single
`packed_features` tensor of shape `(1, T, D)`, and each module in the graph writes
its own embeddings into it with `masked_scatter_embeds` over a boolean mask:

```python
# Two different masks, both (T,):
#   `owned` — which encoded rows belong to this module (derived from the flag)
#   `mask`  — where those rows sit in the packed sequence (`visual_pos_mask`)
owned = visual_row_mask(grid, merge, is_real)
real_embeds = image_embeds[owned]
packed = masked_scatter_embeds(packed, mask, real_embeds)
```

`masked_scatter_embeds(packed, mask, embeds)` writes `embeds` — one vector per
`mask`-selected position — into `packed` wherever `mask` is true. The third
argument is therefore the module's encoded output, not the selector.


Every participant therefore has to agree on **what it writes**: `T` (what one row
of the shared tensor means) and `D` (the width it is written at), with each row
owned by exactly one module. A module's *own* internals are unconstrained — it may
run at any width or length as long as it adapts to the packed layout before the
scatter. That holds for a family whose modules are different *views of the same
sequence* — the text encoder owns tokens, the vision tower owns visual rows, the
backbone reads the whole thing — which is what Qwen3-VL's packed graph is.

It does **not** hold for an arbitrary assembly of modules:

- **A row set that cannot be known before the modules run.** Packing decides the
  row layout on the CPU, so a module whose rows depend on another module's *output*
  has no static split to pack. (A module that merely *reads* a shorter sequence
  internally is fine — the vision tower sees patch rows, not the LLM sequence — as
  long as what it writes still covers exactly the rows it owns.) Two modules
  claiming the same rows is the same defect seen from the other side.
- **A row written at the wrong width.** The constraint is on what a module
  *writes into* `packed_features`, not on its internal width: a tower may run at
  any hidden size as long as it projects before the scatter — Qwen3-VL's ViT is
  1024-wide and its merger emits 2048. A module that cannot produce the shared
  width, or that needs to hand its output to another module at a different width,
  is not a `masked_scatter` participant.
- **A graph composed across families** (say a ViT from one checkpoint with a
  backbone from another). The packer's token layout is family-specific — Qwen3-VL
  builds M-RoPE and a `visual_pos_mask`; another family would build different
  metadata — so a mixed graph has no single layout to pack.

When the composition does not fit, keep the conversation graph: it moves each
modality's own tensors on the carrier, and each module reads what it needs at its
own length. A heterogeneous graph is a legitimate design; it just cannot be
collapsed onto one row axis.

So add a packed graph when profiling shows `pre_forward` bookkeeping is a
measurable slice of the step **and** the modules are one family over one sequence.
It is extra surface: a second train graph, `pack_*` hooks on every module in it,
its own collation contract in `SeedOmniCollator`, and a parity test against the
conversation path (`tests/seed_omni/modules/<family>/test_*_accel_align.py`
compares loss, per-parameter gradients and the AdamW step). None of that is on the
critical path for "support model X".

### 4.2 Composing modules across families

The other direction from §4.1: a graph whose modules come from **different**
families, i.e. different checkpoints, composed by YAML alone. This is the strongest
test of the module contract — if it needs Python, the contract has a gap.

The in-tree example is
[Qwen3-0.6B visual instruction tuning](../models/qwen3.md#7-visual-instruction-tuning-composing-the-qwen3-vl-vit):
`qwen3vl_vision` (from the Qwen3-VL family) beside `qwen3_text_encoder` and
`qwen3_llm` (from the Qwen3 family). No new class; the recipe is three
`model_config:` overrides in `modules_train.yaml`.

What a cross-family composition may need — and what each knob is for:

| Need | Mechanism | Example |
|------|-----------|---------|
| Match the source module's output width to the destination's hidden size | a config field the source module already reads at build time | `qwen3vl_vision` `out_hidden_size: 1024` retargets its own patch merger |
| Drop features the destination cannot consume | likewise a build-time config field | `qwen3vl_vision` `disable_deepstack: true`, because the plain Qwen3 backbone has no DeepStack injection |
| Freeze most of a borrowed module, keep part trainable | the module's own `freeze_model()` reading a config flag | `qwen3vl_vision` `freeze: true` keeps only the merger |
| Make a module accept another family's item types | the module's own config flag | `qwen3_text_encoder` `enable_image: true` switches it to the Qwen3-VL chat template and keeps only the vision special-token rows trainable |
| Point the modules at different checkpoints | per-module `model_path` | `qwen3vl_vision: model_path: /abs/path/to/Qwen3-VL-2B/qwen3vl_vision` |

Three rules make this work:

- **One `dp_size`.** Every distributed module and `model.accelerator` must resolve
  to the same `dp_size`; see
  [Per-Module Parallelism](../distributed/per_module_parallelism.md). A borrowed
  module does not get its own data stream.
- **One carrier.** A cross-family graph uses the conversation graph, never a packed
  one (§4.1). Each module reads the items it owns by `type` / `role` / `meta` and
  writes its output back onto the same item, which is what lets a tower from one
  family feed a backbone from another.
- **A split checkpoint per source.** `model.model_path` is a HuggingFace checkpoint
  only when every module comes from that one layout
  ([Upstream Checkpoint Layout](hf_layout.md) rejects a module outside it), so a
  multi-source composition starts from split checkpoints and exports one
  `hf_ckpt/` per module rather than a single flat checkpoint.

## 5. Honour the contracts

- Return at most one scalar `_loss` per node, token-mean reduced over the
  module's own micro-batch.
- Read inputs in `pre_forward`, write results in `post_forward`. Conversation
  nodes return `{"conversation_list": ...}` so the carrier flows on.
- Implement `dummy_inputs()` and `ModulePreprocessorBase.bind_dummy_inputs()` for
  any encoder whose modality can be absent from a sample, and inject `is_dummy=True`
  placeholders **per sample** in `preprocess_conversations(..., inference=False)` so
  FSDP stays aligned when `OmniProcessor` runs per sample in `seedomni_transform`.
- Populate `item.meta[_OMNI_TOKEN_LEN]` (`from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN`)
  on every `ConversationItem` (real and `is_dummy=True`) in your module's CPU
  preprocessor / chat template so dynamic batching (`train.dyn_bsz: true`) can
  compute both physical (`total`, including dummy items) and supervised
  (`effective`, excluding dummy items) token lengths at the Dataset stage:
  - `TextEncoderChatTemplate` sets `_OMNI_TOKEN_LEN` and `meta["labels"]` automatically
    (preserve both if overriding `tokenize` / `merge_text_embeds`).
  - Vision / audio / codec preprocessors set `item.meta[_OMNI_TOKEN_LEN]` to the
    token count the item contributes to the module / backbone sequence.
- Under Ulysses, slice in `pre_forward` and gather in `post_forward`; see
  [Sequence Parallelism](../distributed/sequence_parallel.md).
- For inference, emit `module_signal` strings to drive FSM transitions and clear
  private buffers in `reset_local_inference_state()` /
  `reset_global_inference_state()` / `finalize()`.
- Optional: mix in [Metric Meter](../mixins/metric_meter.md) for per-module
  FLOPs, and [Offline Encoding](../mixins/offline_encoding.md) for a cacheable
  encoder.

## 6. Validate

1. Inspect the graph trace and optionally profile per-node timing / memory
   ([Debugging the graph](training_and_inference.md#6-debugging-the-graph)).
2. Run the unit tests: `pytest tests/seed_omni/`.
3. Run a short train and inference end to end, then add a page under
   `docs/seed_omni/models/` and register it in `docs/index.md`.

Worked examples:

- **Janus**: multimodal understanding and generation (SigLIP / VQVAE / LLaMA).
- **Qwen3**: minimal text-only split, plus visual instruction tuning.
- **Qwen3 MoE**: MoE backbone with expert parallelism.
- **Qwen3-VL**: vision-language split with per-module ViT.
- **BAGEL**: MoT backbone with flow-matching image generation.
