---
name: veomni-patchgen-model
description: "Author or refresh a VeOmni model's patchgen-generated modeling under generated/ — GPU and/or NPU config, dense or MoE, text / VLM / Omni. Covers the patchgen decorators, sharing patches across sibling models via name_map, MoE fused-expert weight loading, Ulysses SP in multimodal forwards, __init__.py registration, running codegen, and the test cases. This is the modeling step of adding a new model, not only of refreshing an existing one. Trigger: 'add patchgen for a model', 'write a patch_gen_config', 'regenerate the generated modeling', 'add NPU patchgen', 'port a model to patchgen', 'transformers v5 migration'. Never hand-edit anything under generated/."
---

# VeOmni Patchgen Modeling Protocol

Purpose: add or refresh a model's patchgen-generated modeling under
`veomni/models/transformers/<model>/generated/`. VeOmni pins
`transformers==5.9.0` and ships patchgen-generated modeling for every
supported transformers-family model. The non-transformers architectures
(`flux`, `movqgan`, `wan`) have no `generated/` directory and are out of scope.

**References (read first, load on demand):**

- `docs/design/patchgen.md` — patchgen DSL, CLI, CI drift check
- `docs/transformers_v5/transformers_v5_moe_weight_loading.md` — MoE fused-expert layout + runtime converter
- `docs/transformers_v5/veomni_flash_attention_kernel_adapter.md` — FA custom-name adapter
- `docs/transformers_v5/testing_new_model.md` — test case SOP for a new model

## What to read for your model

This file is the spine: it applies to every model. The category-specific
material lives in `references/` — load only what your model needs.

| Your model | Also read |
|---|---|
| Any model, before Phase 1 | `references/model-examples.md` — pick the closest existing model and mirror it |
| Has routed experts (MoE) | `references/moe.md` — Phase 2 expert patches, Phase 3 checkpoint converter, MoE pitfalls |
| Has a vision / audio / speech tower (VLM or Omni) | `references/multimodal.md` — SP-aware multimodal forward, metadata precompute, `dummy_forward`, subtree pruning, VLM/Omni pitfalls |
| Text-only and dense | neither — the spine plus the examples file is the whole protocol |

A text-only dense GPU model therefore reads this file plus the examples, and
skips about 380 lines of MoE and multimodal material. A VLM+MoE model reads
everything. Read the spine first either way; the reference files add to it and
never replace a phase.

---

## Phase 0: Environment + Reference Setup

### 0.1 Verify transformers venv

Patchgen runs against `transformers==5.9.0`. Before touching code:

```bash
source .venv/bin/activate
python -c "import transformers; print(transformers.__version__)"
```

If not `5.9.0`, re-sync the default env:

```bash
uv sync --frozen --extra gpu --group dev
source .venv/bin/activate
```

### 0.2 (Strongly recommended) Drop HF reference source into `.agents_workspace/`

`.agents_workspace/` is gitignored. Keeping the upstream HF source next to your
patchgen config is the single biggest accelerator for catching subtle
signature/contract drift while iterating.

Use the pinned version as the directory name so several pins can coexist:

```bash
PIN=$(python -c "import transformers; print(transformers.__version__)")
mkdir -p ".agents_workspace/hf_reference/<m>/v${PIN}"

curl -fsSL -o ".agents_workspace/hf_reference/<m>/v${PIN}/modeling_<m>.py" \
  "https://github.com/huggingface/transformers/raw/v${PIN}/src/transformers/models/<m>/modeling_<m>.py"
```

`-f` matters: without it a missing tag or renamed module returns 404 and curl
writes the error page into `modeling_<m>.py` with exit status 0, so you would
diff against an HTML page and not notice.

For VLMs also grab `processing_<m>.py` / `image_processing_<m>.py` /
`configuration_<m>.py` if you expect processor-side or config-shape work.

If you are **refreshing** an existing patchgen-generated file across a
transformers minor bump (the pin the generated file was produced against → the
new pin), pull both versions side-by-side and diff to spot contract drift —
substitute the
`<old_ver>` / `<new_ver>` tags with the actual versions you are migrating
between:

```bash
mkdir -p .agents_workspace/hf_reference/<m>/{old,new}
curl -fsSL -o .agents_workspace/hf_reference/<m>/old/modeling_<m>.py \
  "https://github.com/huggingface/transformers/raw/<old_ver>/src/transformers/models/<m>/modeling_<m>.py"
curl -fsSL -o .agents_workspace/hf_reference/<m>/new/modeling_<m>.py \
  "https://github.com/huggingface/transformers/raw/<new_ver>/src/transformers/models/<m>/modeling_<m>.py"
diff -u .agents_workspace/hf_reference/<m>/{old,new}/modeling_<m>.py | less
```

Things to watch for in upstream contracts:

- `@can_return_tuple`, `@capture_outputs`, `@merge_with_config_defaults`,
  `@auto_docstring` decorators → affect behavior of your `override_method`.
  When you `override_method` on a `@auto_docstring`-decorated method, **every
  parameter you declare in the new signature must also appear in the patched
  docstring's `Args:` block** — otherwise `auto_docstring` will emit warnings
  at import time about "undocumented parameter". For Omni-style overrides that
  add params like `audio_feature_lengths`, `feature_lens`, `aftercnn_lens`,
  `rope_deltas`, `image_grid_thw`, `video_grid_thw`, etc., copy the upstream
  docstring and append minimal one-line entries for every new param.
- **`attention_mask` may be a dict** — HF v5 routinely passes
  `attention_mask={"full_attention": <tensor>, ...}` keyed by attention type.
  Any patched forward that forwards `attention_mask` to
  `compute_3d_position_ids` / `get_rope_index` / other tensor-expecting
  helpers must defensively unwrap `attention_mask.get("full_attention", None)`
  when it's a dict.

VLM and Omni models have four more upstream contracts to check before writing
any patch — placeholder masks, `get_{image,video}_features` return shapes, the
packed position-ids layout and mrope shape collapse. See
`references/multimodal.md`, "Phase 0: upstream contracts".

Keep this directory around through commit; delete it after the PR merges (it's
already gitignored so it won't leak into the repo).

---

## Before You Start: Create a Plan

Track the phases with whatever todo/plan tool the running agent provides.
Suggested plan:

```
Phase 0: Verify venv + drop HF reference files       -> in_progress
Phase 1: Scope & audit upstream surface              -> pending
Phase 2: Draft <model>_gpu_patch_gen_config.py       -> pending
Phase 3: (MoE only) Add checkpoint converter         -> pending
Phase 4: Wire __init__.py to expose generated classes -> pending
Phase 5: Run patchgen + verify diff                   -> pending
Phase 6: Add test cases                               -> pending
Phase 7: Run tests (single-GPU + e2e)                 -> pending
Phase 8: Docs + /veomni-review + commit               -> pending
```

Drop phases that don't apply (e.g. Phase 3 for non-MoE models).

---

## Phase 1: Scope & Audit

**Input**: model name `<M>` (e.g. `qwen3_5`, `glm_moe_dsa`).

**Operations:**

1. Locate `veomni/models/transformers/<M>/`. If the directory does not exist yet
   you are being called as the modeling step of `/veomni-new-model`: create it,
   and read that skill's Phase 1 first so the category (text / VLM / Omni,
   dense / MoE, GPU-only or GPU+NPU) is already decided when you get here.
2. If a patchgen-generated file already exists under
   `veomni/models/transformers/<M>/generated/` you are **refreshing** an
   existing config (e.g. picking up upstream changes, adding NPU sibling,
   fixing a bug). Otherwise you are writing the first config for this model.
   Either way, the rest of this protocol applies identically.
3. Decide backend coverage:
   - GPU only → one `<m>_gpu_patch_gen_config.py` + one
     `generated/patched_modeling_<m>_gpu.py`.
   - GPU + NPU → add sibling `<m>_npu_patch_gen_config.py` that writes
     `generated/patched_modeling_<m>_npu.py`; mirror the `glm_moe_dsa` or
     `qwen3_vl` layout.
4. Check model category. Each entry below names the closest existing model;
   `references/model-examples.md` says what to copy out of it, file by file.
   - Text-only LLM → reference `qwen3/` (or `llama/` for the minimal example)
   - MoE → reference `qwen3_moe/` (plus converter work in Phase 3)
   - VLM (non-MoE) → reference `qwen3_vl/`
   - VLM + MoE → reference `qwen3_vl_moe/` (multimodal forward + SP scatter,
     ViT dummy forward, Flash-attn kwargs popping, `get_position_id_func`)
   - Omni (non-MoE thinker + speech subtree to exclude) → reference
     `qwen2_5_omni/` (audio/vision SP + dummy_forward, talker/token2wav/BigVGAN
     exclusion, `log_probs`/`entropy` output dataclass, no parallel_plan/converter)
   - Omni MoE → reference `qwen3_omni_moe/`
5. Check upstream source (`from transformers.models.<m> import modeling_<m>`).
   Confirm class/function names still exist; MoE expert layouts especially
   diverge between sibling models — see
   `docs/transformers_v5/transformers_v5_moe_weight_loading.md`.
6. Note related configs/loaders to preserve: `MODELING_REGISTRY`,
   `MODEL_CONFIG_REGISTRY` in `veomni/models/loader.py`; any auto-config
   registrations.
7. Look for a **sibling model** you can borrow patches from: e.g. qwen3_5_moe
   reuses GatedDeltaNet/ViT patches from `qwen3_5` via direct import +
   `name_map={"Qwen3_5": "Qwen3_5Moe"}`. Prefer reuse over copy-paste when the
   upstream classes are structural duplicates with only a name-prefix
   difference.

**Validation**: you have a concrete list of patches to apply, the reference
model directory to mirror, and the backend/category decision pinned down.

---

## Phase 2: Draft `<M>_gpu_patch_gen_config.py`

Create `veomni/models/transformers/<M>/<M>_gpu_patch_gen_config.py` at the model root.

**Skeleton (mirror `qwen3_gpu_patch_gen_config.py`):**

```python
from veomni.patchgen.patch_spec import PatchConfig, create_patch_from_external

config = PatchConfig(
    source_module="transformers.models.<m>.modeling_<m>",
    target_file="patched_modeling_<m>_gpu.py",
    description="<M> with LigerKernel GPU replacements + VeOmni SP/fused-loss patches",
)
```

**Patch primitives:**

| Effect                                        | patchgen decorator / API                               |
| --------------------------------------------- | ------------------------------------------------------ |
| Replace whole class (RMSNorm, MLP, Experts)   | `@config.replace_class("<Class>")` or `create_patch_from_external(...)` for liger |
| Replace module-level function (rotary, loss)  | `@config.replace_function("<name>")`                   |
| Override a single method (Attention.forward, Model.forward, ForCausalLM.forward) | `@config.override_method("<Class>.<method>")`         |
| Add attribute / extra `super().__init__()` wiring | `@config.modify_init("<Class>")`                   |
| Reuse patch from a sibling config (name-prefix difference) | `config.override_method("<NewClass>.<m>", replacement=<imported_fn>, name_map={"OldPrefix": "NewPrefix"})` — non-decorator form. **Caveat**: name_map only rewrites symbol *names* at the AST level; it does NOT align field sets between sibling output dataclasses (e.g. dense `ModelOutputWithPast` vs MoE `ModelOutputWithPast` with extra `router_logits`). Any `<OldClass>Output(...)` constructor call in the body gets its name rewritten but keeps the original arg list, silently dropping MoE-only fields. Clone the body when return dataclasses differ. |
| Supporting import needed in generated file    | `config.add_import("<module>", names=[...])` (or `alias=..., is_from_import=False`) |
| Remove an upstream import the generated file should NOT keep | `config.drop_import_names("<symbol>", ...)`     |
| Inject raw code (try/except import fallback, helper fn used by patched code) near top of generated file | `config.add_post_import_block("""...""")` |
| Remove unused class from output               | `config.exclude_from_output("<Class>")`                |
| Inherit an entire sibling GPU config into an NPU config (reuse helpers / imports / post-import blocks; only override device-specific kernels) | `config.helpers.extend(gpu_config.helpers)` + `config.post_import_blocks.extend(gpu_config.post_import_blocks)` + `config.additional_imports.extend(gpu_config.additional_imports)` + import each `<fn>_patched` and re-register via `config.override_method(...)`. See `qwen3_vl_npu_patch_gen_config.py` |


**Cross-config reuse pattern** (qwen3_5_moe reusing qwen3_5):

```python
from veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config import (
    qwen3_5_gated_deltanet_forward_patched,
    qwen3_5_vision_model_forward,
    # ...
)

_NAME_MAP = {"Qwen3_5": "Qwen3_5Moe"}
config.override_method(
    "Qwen3_5MoeGatedDeltaNet.forward",
    replacement=qwen3_5_gated_deltanet_forward_patched,
    name_map=_NAME_MAP,
    description="...",
)
```

`name_map` rewrites symbol references *inside* the replacement body so the shared
function transparently targets the correct class namespace. Use it to avoid
duplicating ~hundreds of lines per sibling model.

**Common v5 patch set** (steal from qwen3):

- `create_patch_from_external` → `LigerRMSNorm` replacing `<M>RMSNorm` (for models
  with a "1 + weight" centered RMSNorm formulation — e.g. Qwen3Next variants —
  use `LigerRMSNormForQwen3Next` instead; check the upstream RMSNorm definition).
- `create_patch_from_external` → `LigerSwiGLUMLP` replacing `<M>MLP`.
- `@config.replace_function("apply_rotary_pos_emb")` → `liger_rotary_pos_emb`.
  **Exception**: do NOT replace rotary when the model uses partial rotary
  (`partial_rotary_factor < 1.0`) or `mrope_interleaved=True` — liger applies RoPE
  to the full head_dim and produces NaN. Qwen3_5Moe explicitly skips this; leave
  an inline comment in the patchgen config when you do.
- `@config.override_method("<M>Model.forward")` → keep SP-friendly shape handling.
- `@config.override_method("<M>ForCausalLM.forward")` (or `ForConditionalGeneration.forward`
  for VLM) → fused cross-entropy path via `self.loss_function(logits=logits,
  labels=labels, vocab_size=..., hidden_states=..., weights=self.lm_head.weight, **kwargs)`.
  Note VLM top-level models use `config.text_config.vocab_size`, not `config.vocab_size`.
- **DecoderLayer varlen metadata** — if the model has linear-attention / Mamba /
  GatedDeltaNet layers, override `<M>DecoderLayer.forward` to pass `cu_seq_lens_q`
  through (see qwen3_5_moe), and import cu-free FLA impls via
  `add_post_import_block` with a try/except fallback.

**MoE models** add three more patches here — expert replacement, `_moe_implementation`
propagation and the expert parallel plan. See `references/moe.md`, "Phase 2 additions".

**VLM / Omni models** add the SP-aware multimodal forward and the metadata
precompute contract, and Omni models also prune the speech subtree. See
`references/multimodal.md`.

**Flash attention**: VeOmni custom names
(`veomni_flash_attention_{2,3,4}_with_sp`) are handled globally by
`transformers.integrations.hub_kernels.load_and_register_attn_kernel` adapter —
**no per-model patching needed**. Just keep `attn_implementation` names unchanged
in configs. See
`docs/transformers_v5/veomni_flash_attention_kernel_adapter.md`.

**Patch comment style:**

Every decorated patch function / replaced class must be preceded by a
numbered header block enumerating what changed and why, and every modified
region inside the body must be bracketed by inline `# --- Patch.N ---`
markers that correspond to the header numbers. The comments survive into the
generated `patched_modeling_*.py`, giving reviewers a self-documenting diff
against the upstream HF source.

```python
# ================================================================
# Patch: <Class>.<method>
# 1. <what changed> — <why>
# 2. <next change>  — <why>
# ================================================================
@config.override_method("<Class>.<method>", description="...")
def <name>_patched(self, ...):
    ...
    # --- Patch.1 ---
    <modified region>
    # --- Patch.1 ---
    ...
    # --- Patch.2 ---
    <other modified region>
    # --- Patch.2 ---
```

Guidelines:

- Header numbering is local to the function; reuse the same number for
  all inline markers that belong to the same logical change.
- For removed/replaced upstream lines, keep the original as a commented
  line inside the `# --- Patch.N ---` block (see
  `qwen2_5_vl_gpu_patch_gen_config.py`'s vision-attention `max_seqlen`
  patch) so the diff against HF is self-documenting.
- Mention upstream-contract subtleties explicitly (e.g.
  `BaseModelOutputWithPooling` return type, `pooler_output` tuple-of-tensors)
  — these are the most common source of regressions when HF bumps minor
  versions.

**Regen command** (put at top of file as docstring, mirror qwen3):

```
patchgen \
    veomni.models.transformers.<m>.<m>_gpu_patch_gen_config \
    -o veomni/models/transformers/<m>/generated --diff
```

**Validation**: file is syntactically valid (import it: `python -c "import
veomni.models.transformers.<m>.<m>_gpu_patch_gen_config"`) and every behaviour
identified in Phase 1 has a corresponding decorator here.

---

## Phase 3: MoE Checkpoint Tensor Converter (MoE models only)

**Skip this phase entirely for dense models.**

MoE models need a runtime converter because v5 stores fused expert tensors
while HF checkpoints ship one of three different layouts. Picking the wrong
one loads zero expert weights, silently. The full procedure — how to verify the
HF layout empirically, which template to copy, and the round-trip safety rule —
is in `references/moe.md`, "Phase 3: checkpoint tensor converter".

---

## Phase 4: Wire `__init__.py`

Pick one of three patterns based on Phase 1's backend + capability decision.

**Pattern A — text LLM / dense (qwen3 style):**

```python
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("<m>")
def register_<m>_modeling(architecture: str):
    from .generated.patched_modeling_<m>_gpu import (
        <M>ForCausalLM,
        <M>Model,
    )

    if "ForCausalLM" in architecture:
        return <M>ForCausalLM
    return <M>Model
```

**Pattern B — MoE (qwen3_moe style):** same as A, plus register the converter
on each generated model class:

```python
from .checkpoint_tensor_converter import create_<m>_checkpoint_tensor_converter

for model_cls in (<M>ForCausalLM, <M>Model, ...):
    model_cls._create_checkpoint_tensor_converter = staticmethod(
        create_<m>_checkpoint_tensor_converter
    )
```

`staticmethod(...)` is required — the loader calls it as
`model._create_checkpoint_tensor_converter(model)`.

**Pattern C — GPU + NPU sibling (glm_moe_dsa / qwen3_vl style):** branch on
`IS_NPU_AVAILABLE` between the two generated modules:

```python
from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("<m>")
def register_<m>_modeling(architecture: str):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_<m>_npu import <M>ForCausalLM, <M>Model
    else:
        from .generated.patched_modeling_<m>_gpu import <M>ForCausalLM, <M>Model

    if "ForCausalLM" in architecture:
        return <M>ForCausalLM
    return <M>Model
```

**Rules:**

- All logic lives in the patchgen config + generated file. Do **not** create
  hand-written `modeling_<m>.py` / `gpu_patch.py` / `npu_patch.py` — those
  files have been retired across the codebase.
- For NPU (Pattern C): write a separate `<m>_npu_patch_gen_config.py` — do
  not toggle GPU vs NPU kernels inside a single config via runtime `if`s.

---

## Phase 5: Run Patchgen + Verify Diff

1. Regenerate. `make patchgen` (`patchgen --all --diff`) rebuilds every model's
   generated file, which is the safe default because it cannot leave a GPU/NPU
   sibling behind. Target a single module only when you want a fast loop:
   ```bash
   patchgen \
       veomni.models.transformers.<m>.<m>_gpu_patch_gen_config \
       -o veomni/models/transformers/<m>/generated --diff -v
   ```
2. Inspect `generated/patched_modeling_<m>_gpu.py`:
   - Header lists every patch you defined under "Patches applied".
   - Patched classes/methods carry the `# [PATCHED ...]` markers.
   - Relative imports (`from ...activations`) rewritten to absolute
     (`from transformers.activations`).
3. Inspect `generated/patched_modeling_<m>_gpu.diff` — every hunk must correspond
   to an intentional patch. Unexpected hunks (e.g. whitespace, unrelated classes)
   indicate a misconfigured patchgen config.
4. `make quality` / `ruff format` on the generated file (patchgen pipeline runs
   ruff, but double-check).
5. Check CI drift guard:
   ```bash
   patchgen --check
   ```
   Must exit 0. `--fix` overwrites checked-in files if drift is intentional.
6. If `make style` / `ruff --fix` auto-removed unused imports from the generated
   `*.py` (this happens when patchgen pulls an import from HF source that the
   patched version doesn't use, e.g. `torch_compilable_check` in transformers
   v5.2), the sibling `*.diff` file becomes stale against the post-fix `*.py`.
   Re-sync with:
   ```bash
   patchgen --check --fix
   ```
   Do NOT manually re-run `patchgen` (without `--check`) to "fix" it — that
   would re-introduce the unused imports and you'd ping-pong between ruff and
   patchgen. `patchgen --check --fix` writes the diff against the
   post-style-fix `.py`, which is what CI expects.

**Never edit `generated/*.py` by hand** — always go back to the patchgen config
and regenerate. This is a hard rule called out in `AGENTS.md`.

---

## Phase 6: Add Test Cases

Follow `docs/transformers_v5/testing_new_model.md`. Every file below is already
enumerated in a CI workflow — the unit-test ones, or `gpu_e2e_test.yml` /
`npu_e2e_test.yml` for the e2e tables — so appending a case needs no workflow
change. That is exactly why this phase extends tables instead of adding files.
`tests/models/test_model_registry.py` and
`tests/models/test_models_logits_equal_v5.py` are part of the minimum too:
the first proves the registry returns the generated class, the second that it
is numerically equal to upstream.
If you think you need a new test file, read `.agents/knowledge/testing.md` first.
Minimum coverage:

1. **Toy config**: create `tests/toy_config/<m>_toy/config.json` (few layers,
   small hidden/intermediate, tiny vocab). Add a `README.md` next to it noting
   source config + changes.
2. **`tests/models/test_models_patch.py`**: append an entry to the test cases
   list with `id="<m>"` and `is_moe=<bool>`. If the model lacks certain
   attention/MoE backends, add a `case_id == "<m>"` filter block in
   `test_models_patch_fwd_bwd`.
3. **`tests/e2e/test_e2e_parallel.py`**: append a `pytest.param(...)`. Use
   `max_sp_size=1` if SP not yet supported, else `None`.
4. **VLM only** — `tests/models/test_vlm_trainer.py`: add to the freeze-ViT
   VLM cases list.
5. **VLM / Omni only** — `tests/distributed/test_dummy_forward.py`: add a
   `pytest.param(...)` in `_vlm_cases` (or `_omni_cases`). Required because
   patchgen-generated VLMs override
   `<M>VisionTransformerPretrainedModel.dummy_forward` (or equivalent) and
   this test is the only place the FSDP2 asymmetric-forward + `dummy_forward`
   hook is exercised on multi-GPU.
6. **Text LLM equivalence (optional)** — `tests/distributed/test_fsdp_equivalence.py`
   covers single-GPU vs FSDP2 `grad_norm` for *text* models only. If the model
   is text-only, append to the text test cases list. VLM/Omni models are out
   of scope for this suite (no VLM scaffolding exists).
7. **MoE only** — `tests/models/test_checkpoint_tensor_converter.py`: add a
   test group mirroring the existing `qwen3_moe` / `qwen3_vl_moe` blocks.
   Minimum coverage:
   - `can_handle` — matches the expected key regex, rejects non-expert keys.
   - `convert` — HF-layout input produces correct v5-layout output (shape +
     value-preserving transpose for fused-key converters); for fused-key
     converters also test **v5-layout passthrough** (same tensor object / values)
     and **hard-error on unrecognized shape**.
   - `finalize` — returns `[]` (or raises on unflushed per-expert buffers for
     the qwen3_moe-style stacking converter).
   - Factory — works with both nested `config.text_config` (top-level VLM-MoE
     config) *and* flat `config` (standalone `<M>TextModel` with `<M>TextConfig`).
   - Integration — run one layer end-to-end through `maybe_convert_checkpoint_tensor`.
   Use constants where the shape dims are pairwise-distinct (e.g.
   `hidden=8`, `intermediate=6` so `2*intermediate=12 ≠ hidden`) — overlapping
   dims silently hide dispatch bugs.

---

## Phase 7: Run Tests

Activate the project venv:

```bash
source .venv/bin/activate
# If not already synced:
# uv sync --extra gpu --dev
```

Run:

```bash
pytest tests/models/test_models_patch.py -k <m> -v
pytest tests/e2e/test_e2e_parallel.py::<test_fn> -k <model_name> -v   # see note below; needs multi-GPU worker
# VLM only:
pytest tests/models/test_vlm_trainer.py -k <m> -v
```

**`-k` keyword rules — the three suites use *different* id conventions, and
getting this wrong silently produces `0 selected / N deselected`:**

| Suite | id source | keyword to pass to `-k` |
|---|---|---|
| `test_models_patch.py` | explicit `pytest.param(..., id="<m>")` | model id as registered (e.g. `qwen2_5_vl`, `qwen3_5_moe`) |
| `test_vlm_trainer.py` | explicit `id="<m>"` | same as above |
| `test_e2e_parallel.py` | **first positional arg (`model_name`)**, *no explicit id* | the HF-style short name (e.g. `qwen25vl`, `qwen2vl`, `qwen3vl`, `qwen3vlmoe`) — **no underscores for VL series** |

Extra e2e gotchas:
- VL-family params piggyback on shared functions (`test_qwen2vl_parallel_align`
  hosts both `qwen2vl` and `qwen25vl`; `test_qwen3vl_parallel_align` hosts
  `qwen3vl`, `qwen3vlmoe`, `qwen3_5`, `qwen3_5_moe`). Qualify with
  `::<test_fn>` to avoid sweeping unrelated siblings.
- When in doubt, list actual ids before running:
  ```bash
  pytest tests/e2e/test_e2e_parallel.py --collect-only -q | grep -i <m>
  ```
- If `pytest -k <m>` reports `0 selected`, the id almost certainly disagrees
  with `<m>` — do NOT assume the test doesn't exist; re-check with
  `--collect-only`.

**Acceptance:**

- `test_models_patch` passes for every `(hf_mode, veomni_mode, moe_backend)`
  combo the filter allows — loss and grad norm match within `(_DEFAULT_RTOL,
  _DEFAULT_ATOL)`.
- `test_e2e_parallel` passes across all `(sp_size, ep_size)` combos.
- `make quality` is clean.

---

## Phase 8: Documentation + Review + Commit

1. **Docs:**
   - If the model required a non-trivial quirk (e.g. new MoE layout variant,
     unusual loss-function signature), add a short note under
     `docs/transformers_v5/` or extend an existing page.
   - Update supported-models / transformers-v5 coverage tables if present.
2. **.agents knowledge**: if the work surfaced a new hard constraint
   (e.g. "model X requires `logits_to_keep` handled in ForCausalLM.forward"),
   add it to `.agents/knowledge/constraints.md`.
3. **Run `/veomni-review`** (mandatory pre-commit gate).
   - `safe` → commit.
   - `risky` → report, wait for user.
4. **Commit**:
   - Title: `[BREAKING]` only if the change alters checkpoint format
     expectations or public APIs. Follow `[{modules}] {type}: {description}`.
     Example: `[veomni] feat: add patchgen-generated modeling for <m>`.
   - Commit message describes the change, not the tool that produced it — do
     not name the assistant or agent that wrote it, and no `Co-Authored-By`
     trailers. Naming the *model being added* is of course expected; that is
     the change.

---

## Common Pitfalls

These apply to every model. MoE and VLM/Omni have their own lists in
`references/moe.md` and `references/multimodal.md` — read the one for your
category too, since most of the expensive, silent failures live there.

- **Editing `generated/`** → any manual edit is wiped on next regen and CI drift
  check fails. Always go back to `<m>_gpu_patch_gen_config.py`.
- **Forgetting `config.add_import(...)`** → generated file will import-fail when
  replacement code references symbols absent from the original modeling file.
- **Forgetting `config.drop_import_names(...)`** → generated file inherits an
  upstream import (e.g. Dao-AILab `causal_conv1d_fn`) that you replaced with a
  try/except FLA fallback via `add_post_import_block`; the two collide at runtime.
- **Hand-writing `modeling_<m>.py` / `gpu_patch.py`** → don't. The
  patchgen-generated file under `generated/` is the single source of truth;
  legacy monkey-patch modules have been retired.
- **Replacing `apply_rotary_pos_emb` with liger on partial-rotary models** —
  liger applies RoPE to full head_dim; partial-rotary models (e.g. qwen3_5_moe
  with `partial_rotary_factor=0.25`, `mrope_interleaved=True`) will NaN.
  Leave the upstream function alone; add a comment in the patchgen config.
- **Flash attention per-model patch** → don't. The hub-kernel adapter handles
  all three VeOmni custom FA names globally.
- **Loss function signature** — `self.loss_function(...)` returns
  `(loss, logits)` and expects `hidden_states` + `weights` kwargs (see qwen3
  ForCausalLM.forward). Calling it the old pre-v5 way will silently compute
  nothing or double-compute logits.
- **`logits_to_keep` handling** — `ForCausalLM.forward` takes
  `logits_to_keep: int | torch.Tensor = 0` and slices `hidden_states` before the
  `lm_head` path. Omitting it breaks generation-time compatibility.
- **Duplicating patches across sibling models** — if qwen3_5 and qwen3_5_moe share
  a GatedDeltaNet / ViT, import the replacement functions from the sibling
  patchgen config and use `name_map={"OldPrefix": "NewPrefix"}` — don't copy.
- **Don't override a public HF method just to change its return shape** — if the
  v5 upstream contract says `get_{image,video}_features(...).pooler_output` is a
  `tuple[per-item tensor]` after `torch.split`, don't `override_method` to return
  a flat tensor: external callers (including the unpatched
  `ForConditionalGeneration.get_{image,video}_features` which delegates to
  `self.model...`) break silently. Keep the upstream shape and do the
  post-processing (e.g. `torch.cat(..., dim=0)`) inside your patched
  `<M>Model.forward` instead. Qwen2_5_VL migration learned this the hard way.
- **Preserve full method signature when overriding** — `override_method` keeps
  the original decorators; if you also trim the parameter list (e.g. drop
  `inputs_embeds` + `image_features` from v5's `get_placeholder_mask`), any
  HF-internal caller that still passes those kwargs silently breaks. Keep the
  parameters as no-ops (just unused) unless you are 100% sure no internal path
  calls the method.
- **`logits_to_keep` must slice `hidden_states` before the labels branch** — in
  `<M>ForConditionalGeneration.forward`, slice `hidden_states = hidden_states[:,
  slice_indices, :]` *before* dispatching to `self.loss_function(...)` vs
  `self.lm_head(...)`. Slicing only in the `else` (no-labels) branch silently
  computes loss on the wrong positions when labels + `logits_to_keep>0` are
  both set.
- **Forgetting `hidden_states` / `attentions` on custom return objects** — when
  your patched `Model.forward` or `ForConditionalGeneration.forward` manually
  constructs a `<M>ModelOutputWithPast` / `<M>CausalLMOutputWithPast` (instead
  of relying on the upstream `@can_return_tuple`-decorated path), always pass
  through `hidden_states=outputs.hidden_states` and
  `attentions=outputs.attentions`. Otherwise callers using
  `output_hidden_states=True` / `output_attentions=True` silently get `None`.
- **Skipping `check_patchgen`** → CI will fail on PR. Always run it locally.
- **Empty class body written as `: ...` instead of `: pass`** — when the upstream
  HF source defines an empty class via inline Ellipsis (e.g.
  `class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...`)
  rather than the multi-line `pass` form, `_replace_method_body_with_preserved`
  in `veomni/patchgen/codegen.py` is responsible for both stripping the inline
  `: ...` tail and re-opening the class header so the injected `forward` indents
  correctly. This is wired up since the Llama migration. If a future HF refactor
  introduces a *new* empty-body syntax the helper doesn't recognize, the
  generated file will emit
  `class Foo(...): ...\n    def forward(...): ...` — invalid Python — and
  `import` will fail with `IndentationError: unexpected indent`. In transformers
  4.57.3, 8 modeling files use this inline form: llama, mistral, nemotron,
  persimmon, phimoe, qwen2_moe, stablelm, jetmoe. When migrating any of these
  via `override_method` on a synthetic class (e.g.
  `LlamaForSequenceClassification`), verify the generated file imports cleanly
  before declaring victory.
- **Text/MoE models silently fail on NPU CI with `KeyError: "Unknown kernel
  'npu' for op='rotary_pos_emb'/'rms_norm'"`** — the `KERNEL_REGISTRY` (used
  by the OpSlot path in patchgen-generated modeling) currently registers only
  the `liger_kernel` GPU backend for `rotary_pos_emb/full` and
  `rms_norm/standard`. Until matching NPU `KernelSpec`s are added, every
  patchgen-generated text/MoE model that runs on NPU CI must be pinned to
  eager via `_NPU_PER_MODEL_OVERRIDES` in `tests/tools/training_utils.py`:
  ```python
  "<model_name>": {
      "rms_norm_implementation": "eager",
      "rotary_pos_emb_implementation": "eager",
  },
  ```
  Match the `model_name` exactly to the key used in `test_e2e_parallel.py`'s
  parametrize (e.g. `"qwen2"`, `"qwen3_moe"`, `"llama3.1"`, `"qwen2_5_omni"`).
  Skipping this step is the canonical "GPU CI is green but NPU CI explodes at
  model build" symptom. Multimodal/Omni models often need the override on
  **both** `rms_norm_implementation` and `rotary_pos_emb_implementation`
  because the audio/vision encoders pull the same OpSlots as the text tower.
- **`pytest -k` mismatch on e2e** — `test_e2e_parallel.py` uses the first
  positional arg (`model_name`) as id, not the registry `<m>` id. For VL
  models that's the HF short name (`qwen25vl`, `qwen3vl`, `qwen3vlmoe`, …),
  which has no underscores and does NOT match `-k qwen2_5_vl`. See Phase 7
  keyword-rules table.
- **Only regenerating GPU when NPU config exists** — if the model has a sibling
  `<m>_npu_patch_gen_config.py`, run codegen for **both** (or use `--all`) before
  committing. CI checks both generated files for drift.
- **`LigerSwiGLUMLP` incompatible with MLPs that accept `intermediate_size` kwarg** —
  e.g. DeepseekV3 reuses `DeepseekV3MLP` for `shared_experts` passing an explicit
  `intermediate_size`; `LigerSwiGLUMLP.__init__` rejects that kwarg and raises
  `TypeError`. Don't blindly copy the qwen3 Liger MLP swap — if the model uses the
  same MLP class for routed + shared experts with different `intermediate_size`,
  skip the Liger replacement.

---

## Scope Guard

This skill owns everything that produces `generated/patched_modeling_<m>_*.py`
for a model under `veomni/models/transformers/` — for a brand-new model
directory as much as for an existing one. For:

- The rest of onboarding a new model — deciding the model category, the
  training config, trainer and data-pipeline integration, docs: use
  `/veomni-new-model`, which hands the modeling step back here.
- A diffusion or other non-transformers architecture (`veomni/models/diffusers/`,
  or `flux` / `movqgan` / `wan`): patchgen does not apply — use
  `/veomni-new-model`.
- New op / kernel: use `/veomni-new-op`.
- uv / dependency bumps (e.g. upgrading the `transformers-stable` pin): use
  `/veomni-uv-update`.
- Bugs uncovered during this work: use `/veomni-debug`.
