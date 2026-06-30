# SeedOmni V2 Refactor — Merge / Migration Guide

> **Audience:** an agent (or human) whose branch adds or modifies a SeedOmni V2
> model and now needs to merge the `szl.refact_omni_v2` refactor (target branch
> `szl.omni_v2`). This document is self-contained: it lists every structural move,
> the import rewrites, the behavioural/API changes, and a mechanical conflict-resolution recipe.

## 0. TL;DR

This branch is a **pure structural + API refactor** of `veomni/models/seed_omni/`
and `veomni/trainer/` — **no model capability was added or removed**. It:

1. **Reorganises** `seed_omni/` into `graphs/`, `mixins/`, `utils/` subpackages
   (flat files moved, not rewritten in behaviour).
2. **Splits** the monolithic `veomni/trainer/omni_trainer.py` / `omni_inferencer.py`
   into a `veomni/trainer/omni/` package of four units.
3. **Removes the executor indirection**: per-node execution now lives in
   `TrainingGraph.step`, making the training loop symmetric with the inference FSM.
4. **Unifies CPU input preprocessing** across training and inference, and unifies
   dummy handling around `item.source`.

`szl.refact_omni_v2` is a **clean fast-forward** over `szl.omni_v2` (no divergent
commits on the target). Conflicts only arise for **other branches that forked the
old layout** — resolve them with the path/import maps below.

---

## 1. File moves (path map)

| Old path | New path | Notes |
|----------|----------|-------|
| `veomni/models/seed_omni/module.py` | `veomni/models/seed_omni/mixins/modulemixin.py` | class is still `ModuleMixin`; adds `CPUPreprocessor` |
| `veomni/models/seed_omni/tracemixin.py` | `veomni/models/seed_omni/mixins/tracemixin.py` | |
| `veomni/models/seed_omni/conversation.py` | `veomni/models/seed_omni/utils/conversation.py` | |
| `veomni/models/seed_omni/graph.py` | `veomni/models/seed_omni/graphs/graph.py` | `NodeDef` / `EdgeDef` / `END` |
| `veomni/models/seed_omni/generation_graph.py` | `veomni/models/seed_omni/graphs/generation_graph.py` | `GenerationGraph`, `FSM_SIGNAL_KEY` |
| `veomni/models/seed_omni/training_graph.py` | `veomni/models/seed_omni/graphs/training_graph.py` | now an FSM-style stepper |
| `veomni/models/seed_omni/convert_registry.py` | `veomni/models/seed_omni/utils/convert_registry.py` | `OMNI_CONVERT_REGISTRY`, `convert_checkpoint` |
| `veomni/trainer/omni_trainer.py` | `veomni/trainer/omni/omni_trainer.py` | orchestrator `OmniTrainer` only |
| `veomni/trainer/omni_inferencer.py` | `veomni/trainer/omni/omni_inferencer.py` | driver `OmniInferencer` only |
| — (new) | `veomni/trainer/omni/omni_module_trainer.py` | per-module `OmniModuleTrainer` + ckpt callbacks (split out of old `omni_trainer.py`) |
| — (new) | `veomni/trainer/omni/omni_module_inferencer.py` | per-module `OmniModuleInferencer` (split out of old `omni_inferencer.py`) |
| — (new) | `veomni/models/seed_omni/modules/base/text_encoder/chat_template.py` | base `TextEncoderChatTemplate` |

New package `__init__.py` files: `seed_omni/graphs/`, `seed_omni/mixins/`,
`seed_omni/utils/`, `trainer/omni/`.

## 2. Import rewrite cheatsheet

Absolute imports (search-and-replace across your branch):

```text
veomni.models.seed_omni.module            → veomni.models.seed_omni.mixins.modulemixin
veomni.models.seed_omni.tracemixin        → veomni.models.seed_omni.mixins.tracemixin
veomni.models.seed_omni.conversation      → veomni.models.seed_omni.utils.conversation
veomni.models.seed_omni.graph             → veomni.models.seed_omni.graphs.graph
veomni.models.seed_omni.generation_graph  → veomni.models.seed_omni.graphs.generation_graph
veomni.models.seed_omni.training_graph    → veomni.models.seed_omni.graphs.training_graph
veomni.models.seed_omni.convert_registry  → veomni.models.seed_omni.utils.convert_registry
veomni.trainer.omni_trainer               → veomni.trainer.omni.omni_trainer      (or: veomni.trainer.omni)
veomni.trainer.omni_inferencer            → veomni.trainer.omni.omni_inferencer   (or: veomni.trainer.omni)
```

Relative imports **inside** `modules/<family>/<sub>/*.py` (4 dots reach `seed_omni/`):

```text
from ....module import ModuleMixin, pre_forward, post_forward
    → from ....mixins.modulemixin import ModuleMixin, pre_forward, post_forward, CPUPreprocessor
from ....tracemixin import TraceMixin
    → from ....mixins.tracemixin import TraceMixin
from ....conversation import ConversationItem, iter_desired_items, ...
    → from ....utils.conversation import ConversationItem, iter_desired_items, is_dummy, ...
from ....generation_graph import FSM_SIGNAL_KEY
    → from ....graphs.generation_graph import FSM_SIGNAL_KEY
```

Prefer the re-export hubs where possible (stable across future moves):

```python
from veomni.models.seed_omni import OmniModel, OmniConfig, ModuleMixin, build_conversation
from veomni.models.seed_omni.mixins import ModuleMixin, CPUPreprocessor, pre_forward, post_forward, TraceMixin
from veomni.models.seed_omni.utils import ConversationItem, iter_desired_items, is_dummy
from veomni.models.seed_omni.graphs import TrainingGraph, GenerationGraph, NodeDef, EdgeDef, END
from veomni.trainer.omni import OmniTrainer, OmniInferencer, OmniModuleTrainer, OmniModuleInferencer
```

Entry points (already updated on this branch; mirror in yours if you forked them):

```python
# tasks/omni/train_omni.py
from veomni.trainer.omni import OmniTrainer
# tasks/omni/infer_omni.py
from veomni.trainer.omni import OmniInferencer
# scripts/convert_model.py
from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint
```

## 3. Behavioural / API changes (require code edits, not just import moves)

### 3.1 Executor removed → `TrainingGraph.step`
- **Gone:** `OmniModel.set_node_executors`, `OmniModel._node_executors`,
  `OmniModel._run_node`, and `OmniModuleTrainer.forward` (the old executor callable).
- **Now:** `OmniModel.forward` loops the FSM exactly like `OmniModel.generate`:
  ```python
  training_graph.reset()
  profiler = GraphProfiler()
  while not training_graph.is_done():
      batch = training_graph.step(modules, batch, profiler=profiler, scope_fn=scope_fn)
      self._collect_training_loss(batch, profiler)  # pop _loss → self._losses
      training_graph.maybe_transition(profiler=profiler)
  ```
  `TrainingGraph` gained `reset()` / `is_done()` / `current_node_name` /
  `maybe_transition()` (mirrors `GenerationGraph`).
- **If your code** called `set_node_executors` or relied on `OmniModuleTrainer.forward`,
  delete that wiring — the orchestrator no longer injects an executor.

### 3.2 Removed graph helpers
- `TrainingGraph.collect_inputs` removed (it was a no-op; the `conversation_list`
  carrier flows through `batch`). The `"outputs"` key of `OmniModel.forward`'s
  return is gone.
- `OmniTrainer.collect_module_trace` renamed → `OmniTrainer.collect_trace`.

### 3.3 Chat templates: per-model subclasses of a base
- New base `TextEncoderChatTemplate` (`modules/base/text_encoder/chat_template.py`)
  provides `tokenize_conversation()`, `tokenize()`, `merge_text_embeds()`,
  `pack_input_ids()`, and abstract `apply_chat_template()` / `apply_generation_prompt()`.
- A new text-encoder model implements **only** its `apply_chat_template` /
  `apply_generation_prompt` (and `ChatMarkers`); reuse the base for the rest.
- `tokenize_conversation(sample, *, add_generation_prompt=False)` — inference passes
  `True`.

### 3.4 CPU preprocessor: one path for training AND inference
- Each module may return a `CPUPreprocessor` from `build_cpu_preprocessor()`.
  Signature: `__call__(self, conversation_list, inference=False, **kwargs)`,
  mutating items **in place**.
- **Training:** collected by `OmniTrainer._build_collate_fn`, run inside
  `SeedOmniCollator` (DataLoader worker).
- **Inference:** run by `OmniInferencer._preprocess_request` over the request once,
  before the FSM — **module `generate` no longer processes raw input** (only
  packs → encodes → scatters; mid-FSM-generated items are the one exception).
- `inference=True` flips train/infer-only bits: image modules **skip dummy
  injection**; text encoders **append the generation prompt**.
- **Order is fixed + serial** = config `modules:` declaration order (see
  `OmniConfig.module_names`). Declare an order-dependent module (e.g. text encoder
  after a vision tower that patchifies its image items) accordingly.

### 3.5 Dummy handling unified on `item.source`
- **Gone:** `worker_dummy_items` / `has_worker_dummy` (from `utils/conversation.py`).
- Dummies are appended by the module's `CPUPreprocessor` (training only), tagged
  with `item.source == _SOURCE` and real-shaped zero `value`; real items are tagged
  the same way. Hooks filter with a single `iter_desired_items(sources=[_SOURCE])`
  — **no `None` / role branching**.
- FSDP gating lives in `modeling`: it runs the real forward only when
  `self.training and fsdp_enabled`, otherwise fabricates **real-shaped zeros**
  (never `None`). Use the `is_dummy(item)` helper.
- Source lives on `item.source`, **not** `meta["source"]`.

## 4. Mechanical merge recipe (for an agent)

1. **Branch off / rebase target.** Bring `szl.omni_v2` (post-refactor) into your
   feature branch:
   ```bash
   git fetch origin
   git checkout <your-model-branch>
   git merge origin/szl.omni_v2        # or: git rebase origin/szl.omni_v2
   ```
2. **Resolve conflicts by relocation, not by reverting.** Most conflicts are
   "file added on both sides" or "modified/deleted" because your model's files sat
   in the old tree. For each conflict:
   - If it's one of the **moved files** in §1, apply your changes to the **new
     path** and `git rm` the old one.
   - If it's a `modules/<family>/...` model file you added (no move), keep it, then
     **apply the §2 import rewrites** and the §3 API edits.
3. **Run the import rewrite** over your added files (sed/grep the §2 table).
4. **Fix API breaks** from §3 (executor removal, `collect_trace`, chat-template
   base, CPU preprocessor signature, dummy/source).
5. **Register your model** (unchanged location): `modules/__init__.py`
   (`OMNI_*_REGISTRY`), plus YAML graphs under `configs/seed_omni/<model>/`.
6. **Verify** (§5).

If your model adds a new module mixin, the canonical reference implementations
post-refactor are `modules/janus/siglip` (vision encoder), `modules/janus/vqvae`
(codec), `modules/qwen3vl/vision` (image+video), and the text-encoder trio
(`janus` / `qwen3` / `qwen3vl`). The `/seedomni-v2` skill is updated to match.

## 5. Verification

```bash
source .venv/bin/activate
make quality                          # ruff check + format
pytest tests/seed_omni/               # 110 tests, must be green
python scripts/visualize_omni_graph.py configs/seed_omni/<model>/base.yaml   # graph topo + FSM
```

A successful merge: no import errors, `pytest tests/seed_omni/` green, and your
model's training + inference launchers run as before.
