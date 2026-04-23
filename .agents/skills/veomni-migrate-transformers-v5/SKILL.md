---
name: veomni-migrate-transformers-v5
description: "Use this skill when migrating an existing VeOmni model under veomni/models/transformers/<model>/ to the transformers v5 (patchgen + generated modeling) path — whether coexisting with v4 (qwen3/qwen3_moe style) or v5-only (qwen3_5/qwen3_5_moe/glm_moe_dsa style), GPU-only or GPU+NPU. Covers: creating <model>_{gpu,npu}_patch_gen_config.py, porting v4 patches to patchgen decorators (replace_class/override_method/replace_function/modify_init/add_post_import_block/drop_import_names), reusing sibling-model patches via name_map, handling MoE weight-loading (CheckpointTensorConverter + fused gate_up_proj layout), multimodal/VLM forward with Ulysses SP, selecting the right __init__.py pattern and min-transformers-version gate, running codegen, and adding v5 test cases. Trigger: 'migrate model to transformers v5', 'port <model> to v5', 'add v5 patchgen for <model>', 'transformers v5 migration', 'convert monkey patch to patchgen', 'add NPU patchgen'. Do NOT edit files under generated/ manually — always regenerate via patchgen."
---

# VeOmni Transformers v5 Migration Protocol

Purpose: migrate an existing model in `veomni/models/transformers/<model>/` from the
transformers v4 runtime monkey-patch path to the transformers v5 patchgen +
self-contained generated modeling path.

**References (read first, load on demand):**

- `docs/transformers_v5/index.md` — overview of what v5 migration covers
- `docs/transformers_v5/patchgen.md` — patchgen DSL, CLI, CI drift check
- `docs/transformers_v5/transformers_v5_moe_weight_loading.md` — MoE fused-expert layout + runtime converter
- `docs/transformers_v5/veomni_flash_attention_kernel_adapter.md` — FA custom-name adapter
- `docs/transformers_v5/testing_new_model.md` — v5 test case SOP

**Working examples (copy the structure, do not edit `generated/`):**

Scenarios differ by *v4-coexistence* vs *v5-only* — pick the closest example:

- **v4↔v5 coexist, text LLM** — `veomni/models/transformers/qwen3/`
  - `__init__.py` — registry dispatch splits on `is_transformers_version_greater_or_equal_to("5.0.0")`; v4 branch imports `modeling_<m>.py` and calls `apply_veomni_<m>_patch()`.
  - `qwen3_gpu_patch_gen_config.py` — Liger + SP + fused-CE patches.
- **v4↔v5 coexist, MoE** — `veomni/models/transformers/qwen3_moe/`
  - `__init__.py` — additionally attaches `_create_checkpoint_tensor_converter` as a `staticmethod` on every v5 model class.
  - `qwen3_moe_gpu_patch_gen_config.py` — replaces `Qwen3MoeExperts` with fused-MoE layout + overrides `get_parallel_plan`.
  - `checkpoint_tensor_converter.py` — HF per-expert → v5 fused runtime converter.
  - `parallel_plan.py` — `get_parallel_plan(use_gate_up_proj: bool = True)` switch: v5 (default) shards fused `gate_up_proj`, v4 monkey patch passes `False` to shard split `gate_proj`/`up_proj`. Required whenever v4 experts use a different param layout than v5 (qwen3_moe, qwen3_omni_moe). Not needed when v4 inherits an already-fused HF base (qwen3_vl_moe).
- **v4↔v5 coexist, VLM (non-MoE) + GPU+NPU** — `veomni/models/transformers/qwen3_vl/`
  - `__init__.py` — registry dispatch on transformers version; v4 branch keeps the monkey patch, v5 branch branches on `IS_NPU_AVAILABLE` between `patched_modeling_qwen3_vl_{gpu,npu}`.
  - `qwen3_vl_gpu_patch_gen_config.py` — full VLM forward with Ulysses SP, async Ulysses text attention, deepstack, precomputed mrope via `get_position_id_func`, and a SP-aware `dummy_forward`.
  - `qwen3_vl_npu_patch_gen_config.py` — demonstrates the **NPU-inherits-GPU** pattern: a thin NPU config that extends `gpu_config.helpers` / `gpu_config.post_import_blocks` / `gpu_config.additional_imports` and only overrides RMSNorm / rotary with `torch_npu.npu_rms_norm` / `torch_npu.npu_rotary_mul`. Avoids duplicating ~1K lines of shared VLM SP/deepstack patches.
- **v4↔v5 coexist, VLM + MoE + GPU+NPU** — `veomni/models/transformers/qwen3_vl_moe/`
  - `__init__.py` — Pattern B with three classes: `_create_checkpoint_tensor_converter` attached as `staticmethod` on `Qwen3VLMoeForConditionalGeneration`, `Qwen3VLMoeModel`, **and** `Qwen3VLMoeTextModel` (the inner text submodel is also loadable standalone and must carry the converter).
  - `qwen3_vl_moe_gpu_patch_gen_config.py` — minimal config that imports *most* VLM SP / deepstack / async-Ulysses / dummy_forward patches from `qwen3_vl` via `name_map={"Qwen3VL": "Qwen3VLMoe"}`, and only writes MoE-specific deltas: `replace_class("Qwen3VLMoeExperts")` with fused layout, `override_method("Qwen3VLMoeModel.__init__")` to propagate `_moe_implementation` into `config.text_config`, a hand-cloned `Qwen3VLMoeModel.forward` (see below), `Qwen3VLMoeForConditionalGeneration.forward` with fused loss + aux_loss, and `get_parallel_plan`. This is the canonical template for any new VLM+MoE migration. **Exception — do NOT reuse `Model.forward` via name_map**: `Qwen3VLMoeModelOutputWithPast` carries an extra `router_logits` field absent from the dense `Qwen3VLModelOutputWithPast`; rewriting class names at the AST level keeps the dense constructor's argument list, silently dropping `router_logits` and collapsing MoE routing. Clone the forward body and hand-author the return.
  - `checkpoint_tensor_converter.py` — HF ships *fused* expert tensors under the *same key names* as v5 but in transposed layout (`[E, H, 2*I]` vs `[E, 2*I, H]`). Uses dim-1 shape dispatch to recognize HF vs v5 layout, passes v5-native tensors through untouched, and hard-errors on unrecognized shapes — see Phase 3 "round-trip safety".
- **v5-only, text/VLM+MoE** — `veomni/models/transformers/qwen3_5/`, `qwen3_5_moe/`
  - `__init__.py` — module-level `if is_transformers_version_greater_or_equal_to("5.2.0"):` gate wraps the whole `@MODELING_REGISTRY.register(...)`; there is **no v4 branch, no `modeling_<m>.py`, no `gpu_patch.py`/`npu_patch.py`**.
  - `qwen3_5_moe_gpu_patch_gen_config.py` — demonstrates `config.drop_import_names(...)`, `config.add_post_import_block(...)`, cross-config reuse via `from ...qwen3_5.qwen3_5_gpu_patch_gen_config import <fn>`, and `name_map={"Qwen3_5": "Qwen3_5Moe"}` on `override_method` to share patches between sibling configs.
- **v5-only with NPU patchgen** — `veomni/models/transformers/glm_moe_dsa/`
  - `__init__.py` — branches on `IS_NPU_AVAILABLE` to import `patched_modeling_glm_moe_dsa_{npu,gpu}`, both under the same v5 gate; raises `RuntimeError` on v4.
  - `glm_moe_dsa_gpu_patch_gen_config.py` + `glm_moe_dsa_npu_patch_gen_config.py` — sibling configs produce separate `generated/*_{gpu,npu}.py` outputs.

---

## Phase 0: Environment + Reference Setup

### 0.1 Verify transformers venv

Migration runs against the v5 experimental extra. Before touching code:

```bash
source .venv/bin/activate
python -c "import transformers; print(transformers.__version__)"
```

If not `5.2.0`, switch envs:

```bash
uv sync --frozen --no-group transformers-stable --extra transformers5-exp --extra gpu --extra audio --group dev
source .venv/bin/activate
```

Running the skill against `transformers==4.57.3` will silently succeed for
patchgen (it reads v5 upstream via `importlib`) but every smoke `import` and
test will fail — **check the version first, always**.

### 0.2 (Strongly recommended) Drop HF reference source into `.agents_workspace/`

`.agents_workspace/` is gitignored. Putting both the v4 and v5 HF originals
side-by-side next to your patchgen config is the single biggest accelerator for
catching subtle signature/contract drift (method arg removal, return-type
changes, decorator additions, split-tuple vs flat-tensor conventions).

```bash
mkdir -p .agents_workspace/hf_reference/<m>/{v4_57_3,v5_2_0}

# v5.2.0 (new target)
curl -sL -o .agents_workspace/hf_reference/<m>/v5_2_0/modeling_<m>.py \
  "https://github.com/huggingface/transformers/raw/v5.2.0/src/transformers/models/<m>/modeling_<m>.py"

# v4.57.3 (old VeOmni baseline — skip for v5-only models)
curl -sL -o .agents_workspace/hf_reference/<m>/v4_57_3/modeling_<m>.py \
  "https://github.com/huggingface/transformers/raw/v4.57.3/src/transformers/models/<m>/modeling_<m>.py"
```

For VLMs also grab `processing_<m>.py` / `image_processing_<m>.py` /
`configuration_<m>.py` if you expect processor-side or config-shape changes.

Diff the two copies before drafting the patchgen config:

```bash
diff -u .agents_workspace/hf_reference/<m>/v4_57_3/modeling_<m>.py \
        .agents_workspace/hf_reference/<m>/v5_2_0/modeling_<m>.py | less
```

Things to watch for in that diff:

- New/removed `@can_return_tuple`, `@capture_outputs`, `@merge_with_config_defaults`,
  `@auto_docstring` decorators → affects behavior of your `override_method`.
- Method signature churn: e.g. `get_placeholder_mask` in v5 takes
  `inputs_embeds` + `image_features` / `video_features`; v4 did not.
- Return-shape churn: e.g. v5 `get_{image,video}_features` `.pooler_output` is
  `tuple[per-image tensor]` after `torch.split`, v4 returned a flat tensor.
- New helper methods the patched forward should delegate to (e.g. v5 added
  `compute_3d_position_ids`, `get_rope_index` moved).
- Packed position-ids contract (`[4, bs, seq-len]` with prepended `text_position_ids`).
- **RoPE shape collapse** — v5 VLMs introduced `apply_interleaved_mrope` (and
  similar helpers) that collapse the leading 3-axis of mrope before layers see
  cos/sin, so the shape goes from `(3, bs, seq_len, head_dim)` in v4 to
  `(bs, seq_len, head_dim)` in v5. Any SP path that gathers cos/sin across the
  sequence dim (async Ulysses, ring attention) must update its `gather_dim`
  accordingly. Grep upstream for `interleaved_mrope`, `mrope_section`, or any
  pre-attention RoPE reshape before writing the patch.
- **`attention_mask` may be a dict** — HF v5 routinely passes
  `attention_mask={"full_attention": <tensor>, ...}` keyed by attention type.
  Any patched forward that forwards `attention_mask` to `compute_3d_position_ids` /
  `get_rope_index` / other tensor-expecting helpers must defensively unwrap
  `attention_mask.get("full_attention", None)` when it's a dict.

Keep this directory around through commit; delete it after the PR merges (it's
already gitignored so it won't leak into the repo).

---

## Before You Start: Create Todos

Use TodoWrite to track phases. Suggested plan:

```
Phase 0: Verify venv + drop HF reference files  -> in_progress
Phase 1: Scope & audit existing v4 patches      -> pending
Phase 2: Draft <model>_gpu_patch_gen_config.py  -> pending
Phase 3: (MoE only) Add checkpoint converter    -> pending
Phase 4: Wire __init__.py v5/v4 split           -> pending
Phase 5: Run patchgen + verify diff              -> pending
Phase 6: Add v5 test cases                       -> pending
Phase 7: Run tests (single-GPU + e2e)            -> pending
Phase 8: Docs + /veomni-review + commit          -> pending
```

Drop phases that don't apply (e.g. Phase 3 for non-MoE models).

---

## Phase 1: Scope & Audit

**Input**: model name `<M>` (e.g. `qwen3_5`, `glm4_moe`).

**Operations:**

1. Confirm model exists at `veomni/models/transformers/<M>/`. If not, the task is
   "add new model", not migration — use `/veomni-new-model` instead.
2. Decide the **coexistence mode** — this drives everything downstream:
   - **v4↔v5 coexist** — model has legacy `modeling_<m>.py` + `apply_veomni_<m>_patch`
     and must keep working on `transformers==4.57.3`. Mirror `qwen3` / `qwen3_moe`.
   - **v5-only** — model was introduced in transformers v5 (e.g. `qwen3_5*`) or we
     are explicitly dropping v4 for this model. Mirror `qwen3_5` / `qwen3_5_moe` /
     `glm_moe_dsa`. There will be no `modeling_<m>.py` / `gpu_patch.py`.
3. **Transformers version gate** — use `"5.2.0"` uniformly across all v5
   migrations. Do not pin to other v5 minor versions (e.g. `5.0.0`) even if the
   model technically exists earlier upstream. Existing models that still use
   `"5.0.0"` (e.g. qwen3, qwen3_moe) will be migrated to `"5.2.0"` separately —
   do not introduce new uses of other v5 pins.
4. Enumerate current v4 patch surface (skip for v5-only models):
   - `modeling_<M>.py` → list every monkey-patched function/class/method.
   - `gpu_patch.py` / `npu_patch.py` → note backend-specific swaps.
   - `parallel_plan.py` → inventory FSDP/EP plan hooks (e.g. `get_parallel_plan`).
5. Decide backend coverage:
   - GPU only → one `<m>_gpu_patch_gen_config.py` + one `generated/patched_modeling_<m>_gpu.py`.
   - GPU + NPU → add sibling `<m>_npu_patch_gen_config.py` that writes
     `generated/patched_modeling_<m>_npu.py`; mirror the `glm_moe_dsa` layout.
6. Check model category:
   - Text-only LLM → reference `qwen3/`
   - MoE → reference `qwen3_moe/` (plus converter work in Phase 3)
   - VLM / Omni MoE → reference `qwen3_5_moe/` (multimodal forward + SP scatter, ViT dummy forward, Flash-attn kwargs popping, `get_position_id_func`)
7. Check transformers v5 upstream source (`from transformers.models.<m> import modeling_<m>`).
   Confirm class/function names still exist; MoE expert layouts especially diverge
   between sibling models — see `transformers_v5_moe_weight_loading.md`.
8. Note related configs/loaders to preserve: `MODELING_REGISTRY`,
   `MODEL_CONFIG_REGISTRY` in `veomni/models/loader.py`; any auto-config registrations.
9. Look for a **sibling model** you can borrow patches from: e.g. qwen3_5_moe
   reuses GatedDeltaNet/ViT patches from `qwen3_5` via direct import +
   `name_map={"Qwen3_5": "Qwen3_5Moe"}`. Prefer reuse over copy-paste when the
   upstream classes are structural duplicates with only a name-prefix difference.

**Validation**: you have a concrete list of patches to port, the reference model
directory to mirror, and the coexistence mode + min transformers version pinned.

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

**Map v4 patches → patchgen decorators:**

| v4 monkey patch                               | patchgen decorator / API                               |
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

**Pruning inactive subtrees** (e.g. talker / code2wav in an omni model where
training only uses the thinker): use `config.exclude_from_output(<Class>, ...)`
to drop classes entirely from the generated file. This has three downstream
ripples you must clean up in the same patch config — otherwise `make quality`
or `import` will fail on the regenerated output:

- **`_init_weights` `isinstance(...)` branches** — upstream's
  `<M>PreTrainedModel._init_weights` typically has one `elif isinstance(module,
  <ExcludedClass>)` branch per leaf init. Override it
  (`@config.override_method("<M>PreTrainedModel._init_weights")`) and drop
  every branch that references an excluded class.
- **Public methods whose bodies reference excluded classes** — e.g.
  `enable_talker` constructs the talker. Override it to
  `raise NotImplementedError("<what>. Use upstream transformers for <purpose>.")`
  so callers get a clear message instead of an F821/NameError at import.
- **`__all__` is auto-filtered** by `veomni/patchgen/codegen.py` — any excluded
  class name is removed from the generated `__all__` list automatically, so
  you don't need a manual `drop_import_names` dance for it.
- **Transitively-dead helper classes** — activations / small utility modules
  used *only* by classes you just excluded will still land in the generated
  file as dead code. Grep the generated output for each excluded class's
  private helpers and add them to `exclude_from_output` too. Example:
  `SnakeBeta` is only referenced by `Qwen3OmniMoeCode2WavDecoderResidualUnit`;
  excluding Code2Wav without also excluding `SnakeBeta` leaves ~40 lines of
  dead code in `generated/`.

See `qwen3_omni_moe_gpu_patch_gen_config.py` for the canonical template
(excludes the whole talker + code2wav subtree plus the dead-after-exclusion
`SnakeBeta` activation, overrides `_init_weights` and `enable_talker`).

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
- **MoE expert replacement** — `@config.replace_class("<M>Experts")` with
  `gate_up_proj [E, 2*I, H]` + `down_proj [E, H, I]` + `fused_moe_forward(...)`
  branching on `_moe_implementation in {"eager", "fused"}`. See qwen3_moe and
  qwen3_5_moe (the latter also removes the upstream `@use_experts_implementation`
  decorator which would otherwise re-route around our fused path).
- **MoE top-level init propagation** — v5 often wraps a text_config under a top
  model. You must propagate `_moe_implementation` from `config` to
  `config.text_config` *before* `super().__init__(config)`, via a
  `@config.override_method("<M>Model.__init__")` patch (see qwen3_5_moe).
- **MoE expert parallel plan** — `@config.override_method("<M>ForCausalLM.get_parallel_plan")`
  (or `ForConditionalGeneration.get_parallel_plan`) returning
  `parallel_plan.get_parallel_plan()`. **If v4 reimplements `<M>Experts` with
  split `gate_proj`/`up_proj` while v5 uses fused `gate_up_proj`** (qwen3_moe,
  qwen3_omni_moe pattern), `parallel_plan.py` must take a
  `use_gate_up_proj: bool = True` switch — v4 monkey patch calls with `False`
  (split keys), v5 patchgen calls with default `True` (fused key). See
  `qwen3_moe/parallel_plan.py` for the canonical template. Models whose v4
  inherits from an already-fused HF base (qwen3_vl_moe pattern) don't need the
  switch — a single fused-only plan matches both paths.
- **VLM/multimodal forward** — replicate qwen3_5_moe's pattern (VLM+MoE) or
  qwen3_vl's (VLM, non-MoE): pop LM-level flash-attn kwargs before ViT call,
  transpose seq↔head layout for Ulysses SP, shard image/video embeds, shard
  placeholder masks, and transpose back. Add
  `@config.override_method("<M>ForConditionalGeneration.get_position_id_func")`
  via an `add_post_import_block` that defines the helper `get_position_id` in
  generated scope (module-level, so multiprocessing can pickle it).
  When SP is enabled and you need to all-gather `input_ids` (or any tensor that
  went through `MainCollator`'s `pack_dim=-1` path) back to full seq on each
  rank, use `torch.cat(list, dim=1)` — the collator's `PackingCollator.__call__`
  does `torch.cat(..., dim=pack_dim).unsqueeze(0)` (see
  `veomni/data/data_collator.py:246-248`), so the shape at model forward is
  `[1, seq_per_rank]`, not flat `[seq_per_rank]`. Using `dim=0` would wrongly
  produce `[sp_size, seq_per_rank]` and silently break downstream mask slicing.
- **DecoderLayer varlen metadata** — if the model has linear-attention / Mamba /
  GatedDeltaNet layers, override `<M>DecoderLayer.forward` to pass `cu_seq_lens_q`
  through (see qwen3_5_moe), and import cu-free FLA impls via
  `add_post_import_block` with a try/except fallback.

**Flash attention**: VeOmni custom names
(`veomni_flash_attention_{2,3,4}_with_sp`) are handled globally by
`transformers.integrations.hub_kernels.load_and_register_attn_kernel` adapter —
**no per-model patching needed**. Just keep `attn_implementation` names unchanged
in configs. See `veomni_flash_attention_kernel_adapter.md`.

**Patch comment style** (mirror `veomni/models/transformers/qwen3_omni_moe/modeling_qwen3_omni_moe.py`):

Every decorated patch function / replaced class must be preceded by a
numbered header block enumerating what changed and why, and every modified
region inside the body must be bracketed by inline `# --- Patch.N ---`
markers that correspond to the header numbers. This mirrors the v4
monkey-patch convention so reviewers can diff v4↔v5 patches line-by-line,
and the comments survive into the generated `patched_modeling_*.py`.

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
- Mention v5-contract changes explicitly (e.g. `BaseModelOutputWithPooling`
  return type, `pooler_output` tuple-of-tensors) — these are the most
  common source of regressions when HF bumps minor versions.

**Regen command** (put at top of file as docstring, mirror qwen3):

```
python -m veomni.patchgen.run_codegen \
    veomni.models.transformers.<m>.<m>_gpu_patch_gen_config \
    -o veomni/models/transformers/<m>/generated --diff
```

**Validation**: file is syntactically valid (import it: `python -c "import
veomni.models.transformers.<m>.<m>_gpu_patch_gen_config"`) and every v4 patch
from Phase 1 has a corresponding decorator here.

---

## Phase 3: MoE Checkpoint Tensor Converter (MoE models only)

Skip for text-only LLMs.

V5 MoE uses fused expert tensors `gate_up_proj [E, 2*I, H]` + `down_proj [E, H, I]`,
but HF safetensor checkpoints may ship either **per-expert split** keys *or*
**pre-fused** keys (sometimes transposed) depending on the model. A runtime
converter avoids the old `scripts/moe_ckpt_merge/moe_merge.py` offline step.

**Verify the HF source layout empirically BEFORE picking a template** — do not
infer it from model family / sibling converter docstrings, because those have
been copy-pasted across unrelated layout families in the past (e.g. the initial
qwen3_omni_moe converter shipped a qwen3_vl_moe-style transposer while the real
checkpoint had per-expert split keys — silent load failure).

Two authoritative sources:

1. **HF's own mapping** — `transformers/conversion_mapping.py::_MODEL_TO_CONVERSION_PATTERN`
   points the model_type at a WeightConverter recipe:
   - `"qwen2_moe"` recipe = `MergeModulelist(dim=0) + Concatenate(dim=1)` →
     source is **per-expert split** → qwen3_moe-style template.
   - `"qwen3_vl_moe"` recipe = `Transpose(1, 2)` →
     source is **pre-fused, transposed** → qwen3_vl_moe-style template.
   - No entry or pass-through → source is **pre-fused, direct v5 layout** →
     no converter needed (qwen3_5_moe-style).
   Cross-family aliases are common: `qwen3_omni_moe → qwen2_moe`,
   `deepseek_v3 → qwen2_moe`, etc. Always resolve the alias before choosing.
2. **A real checkpoint's index** — sanity-check by grepping
   `<ckpt>/model.safetensors.index.json`:
   ```bash
   python3 -c "
   import json, sys
   idx = json.load(open(sys.argv[1]))
   per_expert = sum(1 for k in idx['weight_map'] if '.experts.' in k and k.endswith('gate_proj.weight'))
   fused      = sum(1 for k in idx['weight_map'] if k.endswith('.experts.gate_up_proj'))
   print(f'per-expert keys: {per_expert}, fused keys: {fused}')
   " <ckpt_path>/model.safetensors.index.json
   ```
   If per-expert > 0 → qwen3_moe-style. If fused > 0 → inspect one tensor's
   shape to distinguish transposed (qwen3_vl_moe-style) from direct v5 (no
   converter).

**Pick the template by the verified HF layout, not by model family:**

- **HF ships per-expert split keys** (`*.mlp.experts.{j}.{gate|up|down}_proj.weight`)
  → template = `veomni/models/transformers/qwen3_moe/checkpoint_tensor_converter.py`.
  The regex only matches *HF-side* keys, so a v5-saved fused-key checkpoint
  passes through the converter untouched — no round-trip hazard.
- **HF ships fused expert keys with same names as v5** (`*.mlp.experts.{gate_up_proj|down_proj}`
  at the module level, not per-expert) → template =
  `veomni/models/transformers/qwen3_vl_moe/checkpoint_tensor_converter.py`.
  Key names collide with v5 output, so you **must** use shape-based dispatch
  (see "Round-trip safety" below); blindly transposing corrupts v5-saved ckpts.

**Steps:**

1. Copy the matching template above.
2. Update the regex `_EXPERT_PATTERN` to match your upstream key layout.
3. Update merge order / transpose for the HF-side layout. Three layouts exist
   — see table in `transformers_v5_moe_weight_loading.md`:
   - qwen3_moe: per-expert split → stack on dim 0.
   - qwen3_vl_moe: fused, transposed (`[E, H, 2*I]` / `[E, I, H]`) → `transpose(1, 2)`.
   - qwen3_5_moe: fused, direct (`[E, 2*I, H]` / `[E, H, I]`) → no-op (no converter needed).
4. Export a factory `create_<m>_checkpoint_tensor_converter(model)`:
   - Keyed on `num_experts` + (for fused-key converters) `hidden_size` + `intermediate_size`.
   - Resolve the text config defensively: `text_config = getattr(model.config, "text_config", model.config)`.
     VLM-MoE submodels (e.g. `Qwen3VLMoeTextModel`) are loaded standalone with a
     *flat* `<M>TextConfig` that has no `text_config` attribute; top-level
     `<M>Model` / `<M>ForConditionalGeneration` have a nested one. Both paths
     must work because Pattern B registers the converter on all three classes.
5. Implement `can_handle`, `convert`, and `finalize` — `finalize` must raise on
   any unflushed per-expert or stacked buffer (indicates corrupt/partial ckpt).

**Round-trip safety (fused-key converters only):**

When HF and v5 use identical expert key names but different axis orders
(qwen3_vl_moe pattern), the converter will be invoked on both HF-original
checkpoints *and* v5-saved checkpoints (VeOmni's save path can emit either
format). Dispatch on the `dim-1` shape:

- `gate_up_proj`: HF has `dim-1 == hidden_size`, v5 has `dim-1 == 2 * intermediate_size`.
- `down_proj`:    HF has `dim-1 == intermediate_size`, v5 has `dim-1 == hidden_size`.

For any realistic config, these four numbers are pairwise distinct, so the
dispatch is unambiguous. Transpose only when dim-1 matches the HF expectation;
pass through when it matches v5; **raise on anything else** rather than
silently corrupting weights. See `qwen3_vl_moe/checkpoint_tensor_converter.py`
for the canonical implementation.

**Validation**: on a toy checkpoint with per-expert keys, the converter emits
exactly one `experts.gate_up_proj` and one `experts.down_proj` per layer and
`finalize()` returns `[]` without raising. For fused-key converters, also
validate that a v5-saved checkpoint round-trips: feed `[E, 2*I, H]` / `[E, H, I]`
tensors through and confirm they come out identical (no transpose applied).

---

## Phase 4: Wire `__init__.py`

Pick one of four patterns based on Phase 1's coexistence + backend decision.

**Pattern A — v4↔v5 coexist, text LLM (qwen3 style):**

```python
from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("<m>")
def register_<m>_modeling(architecture: str):
    if is_transformers_version_greater_or_equal_to("<min_v5>"):
        from .generated.patched_modeling_<m>_gpu import (
            <M>ForCausalLM,
            <M>Model,
        )
    else:
        from transformers import <M>ForCausalLM, <M>Model
        from .modeling_<m> import apply_veomni_<m>_patch
        apply_veomni_<m>_patch()

    if "ForCausalLM" in architecture:
        return <M>ForCausalLM
    return <M>Model
```

**Pattern B — v4↔v5 coexist, MoE (qwen3_moe style):** same as A, plus register
the converter on each v5 model class *inside* the v5 branch:

```python
from .checkpoint_tensor_converter import create_<m>_checkpoint_tensor_converter
for model_cls in (<M>ForCausalLM, <M>Model, ...):
    model_cls._create_checkpoint_tensor_converter = staticmethod(
        create_<m>_checkpoint_tensor_converter
    )
```

`staticmethod(...)` is required — the loader calls it as `model._create_checkpoint_tensor_converter(model)`.

**Pattern C — v5-only (qwen3_5 / qwen3_5_moe style):** module-level gate, no
registry decorator on v4:

```python
from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODELING_REGISTRY


if is_transformers_version_greater_or_equal_to("<min_v5>"):

    @MODELING_REGISTRY.register("<m>")
    def register_<m>_modeling(architecture: str):
        from .generated.patched_modeling_<m>_gpu import <M>ForCausalLM, <M>Model
        if "ForCausalLM" in architecture:
            return <M>ForCausalLM
        return <M>Model
```

**Pattern D — v5-only + NPU (glm_moe_dsa style):** single v5 gate, device branch
inside the registry function. Raise on v4 instead of silently falling back:

```python
from ....utils.device import IS_NPU_AVAILABLE
from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("<m>")
def register_<m>_modeling(architecture: str):
    if is_transformers_version_greater_or_equal_to("<min_v5>"):
        if IS_NPU_AVAILABLE:
            from .generated.patched_modeling_<m>_npu import <M>ForCausalLM, <M>Model
        else:
            from .generated.patched_modeling_<m>_gpu import <M>ForCausalLM, <M>Model
    else:
        raise RuntimeError("<m> not available. Please make sure transformers version >= <min_v5>")

    if "ForCausalLM" in architecture:
        return <M>ForCausalLM
    return <M>Model
```

**Rules:**

- **Coexist patterns (A, B)** — never delete `modeling_<m>.py` / `gpu_patch.py` /
  `npu_patch.py`; the v4 branch must keep working on `transformers==4.57.3`.
- **v5-only patterns (C, D)** — do NOT create `modeling_<m>.py` or
  `gpu_patch.py`; all logic lives in the patchgen config + generated file.
  This is cleaner than an empty v4 stub.
- Use `<min_v5> = "5.2.0"` for all new v5 gates. Do not introduce `5.0.0` or
  other v5 pins; standardized per Phase 1 step 3.
- If the model exists on v4 but you're intentionally dropping v4 support, prefer
  Pattern D's explicit `raise RuntimeError(...)` over a silent skip.
- For NPU (Pattern D): write a separate `<m>_npu_patch_gen_config.py` — do not
  try to toggle GPU vs NPU kernels inside one config via runtime ifs.

---

## Phase 5: Run Patchgen + Verify Diff

1. Regenerate:
   ```bash
   python -m veomni.patchgen.run_codegen \
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
   python -m veomni.patchgen.check_patchgen
   ```
   Must exit 0. `--fix` overwrites checked-in files if drift is intentional.
6. If `make style` / `ruff --fix` auto-removed unused imports from the generated
   `*.py` (this happens when patchgen pulls an import from HF source that the
   patched version doesn't use, e.g. `torch_compilable_check` in transformers
   v5.2), the sibling `*.diff` file becomes stale against the post-fix `*.py`.
   Re-sync with:
   ```bash
   python -m veomni.patchgen.check_patchgen --fix
   ```
   Do NOT manually re-run `run_codegen` to "fix" it — that would re-introduce
   the unused imports and you'd ping-pong between ruff and patchgen.
   `check_patchgen --fix` writes the diff against the post-style-fix `.py`,
   which is what CI expects.

**Never edit `generated/*.py` by hand** — always go back to the patchgen config
and regenerate. This is a hard rule called out in `AGENTS.md`.

---

## Phase 6: Add v5 Test Cases

Follow `docs/transformers_v5/testing_new_model.md`. Minimum coverage:

1. **Toy config**: create `tests/toy_config/<m>_toy/config.json` (few layers,
   small hidden/intermediate, tiny vocab). Add a `README.md` next to it noting
   source config + changes.
2. **`tests/models/test_models_patch.py`**: append an entry to
   `_TEST_CASES_TRANSFORMERS_V5` with `id="<m>"` and `is_moe=<bool>`. If the
   model lacks certain attention/MoE backends, add a `case_id == "<m>"` filter
   block in `test_models_patch_fwd_bwd`.
3. **`tests/e2e/test_e2e_parallel_{text,vlm,omni,dit}.py`**: append a
   `pytest.param(...)` to the list in the file matching the model's
   modality (import `DEFAULT_RTOL` / `DEFAULT_ATOL` / `v5_only` from
   `tests/e2e/_harness.py`). Use `marks=v5_only`, and `max_sp_size=1`
   if SP not yet supported, else `None`.
4. **VLM only** — `tests/models/test_vlm_trainer.py`: add to
   `_FREEZE_VIT_VLM_CASES_TRANSFORMERS_V5`.
5. **VLM / Omni only** — `tests/distributed/test_dummy_forward.py`: add a
   `_v5_only` sibling of the existing `_v4_only` case in `_vlm_cases` (or
   `_omni_cases`). Required because v5 migrations override
   `<M>VisionTransformerPretrainedModel.dummy_forward` (or equivalent) and this
   test is the only place the FSDP2 asymmetric-forward + `dummy_forward` hook
   is exercised on multi-GPU. Give the v5 entry an `id="<m>_v5"` so pytest `-k`
   can disambiguate.
6. **Text LLM equivalence (optional)** — `tests/distributed/test_fsdp_equivalence.py`
   covers single-GPU vs FSDP2 `grad_norm` for *text* models only. If the model
   is text-only, append to `_text_test_cases_v5`. VLM/Omni models are out of
   scope for this suite (no VLM scaffolding exists).
7. **MoE only** — `tests/models/test_checkpoint_tensor_converter.py`: add a test
   group mirroring the existing `qwen3_moe` / `qwen3_vl_moe` blocks. Minimum coverage:
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

Activate venv with the v5 extra:

```bash
source .venv/bin/activate
# If not already synced with v5:
# uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu --extra audio --dev
```

Run (v5 presence is auto-detected by the test suite):

```bash
pytest tests/models/test_models_patch.py -k <m> -v
pytest tests/e2e/test_e2e_parallel_<modality>.py::<test_fn> -k <model_name> -v   # see note below; needs multi-GPU worker
# VLM only:
pytest tests/models/test_vlm_trainer.py -k <m> -v
```

**`-k` keyword rules — the three suites use *different* id conventions, and
getting this wrong silently produces `0 selected / N deselected`:**

| Suite | id source | keyword to pass to `-k` |
|---|---|---|
| `test_models_patch.py` | explicit `pytest.param(..., id="<m>")` | model id as registered (e.g. `qwen2_5_vl`, `qwen3_5_moe`) |
| `test_vlm_trainer.py` | explicit `id="<m>"` | same as above |
| `test_e2e_parallel_{text,vlm,omni,dit}.py` | **first positional arg (`model_name`)**, *no explicit id* | the HF-style short name (e.g. `qwen25vl`, `qwen2vl`, `qwen3vl`, `qwen3vlmoe`) — **no underscores for VL series** |

Extra e2e gotchas:
- VL-family params piggyback on shared functions (`test_qwen2vl_parallel_align`
  in `test_e2e_parallel_vlm.py` hosts both `qwen2vl` and `qwen25vl`;
  `test_qwen3vl_parallel_align` in the same file hosts `qwen3vl`,
  `qwen3vlmoe`, `qwen3_5`, `qwen3_5_moe`). Qualify with `::<test_fn>`
  to avoid sweeping unrelated siblings.
- When in doubt, list actual ids before running:
  ```bash
  pytest tests/e2e/ --collect-only -q | grep -i <m>
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
   - If the model required a non-trivial v5 quirk (e.g. new MoE layout variant,
     unusual loss-function signature), add a short note under
     `docs/transformers_v5/` or extend an existing page.
   - Update supported-models / transformers-v5 coverage tables if present.
2. **.agents knowledge**: if the migration surfaced a new hard constraint
   (e.g. "model X requires `logits_to_keep` handled in ForCausalLM.forward"),
   add it to `.agents/knowledge/constraints.md`.
3. **Run `/veomni-review`** (mandatory pre-commit gate).
   - `safe` → commit.
   - `risky` → report, wait for user.
4. **Commit**:
   - Title: `[BREAKING]` only if the migration changes checkpoint format
     expectations or public APIs. Follow `[{modules}] {type}: {description}`.
     Example: `[veomni] feat: migrate <m> to transformers v5 patchgen path`.
   - Commit message **must not** mention Claude / AI / Co-Authored-By.

---

## Common Pitfalls

- **Editing `generated/`** → any manual edit is wiped on next regen and CI drift
  check fails. Always go back to `<m>_gpu_patch_gen_config.py`.
- **Forgetting `config.add_import(...)`** → generated file will import-fail when
  replacement code references symbols absent from the original modeling file.
- **Forgetting `config.drop_import_names(...)`** → generated file inherits an
  upstream import (e.g. Dao-AILab `causal_conv1d_fn`) that you replaced with a
  try/except FLA fallback via `add_post_import_block`; the two collide at runtime.
- **v4 branch broken (coexist patterns)** → always keep `modeling_<m>.py` +
  `apply_veomni_<m>_patch` intact for the v4 path until transformers v4 is dropped.
- **Creating a v4 stub for v5-only models** → don't. Use Pattern C / D
  module-level version gate; a stubbed `modeling_<m>.py` adds drift with no benefit.
- **Wrong min transformers version** — always use `"5.2.0"` for new v5 gates.
  Older pins like `"5.0.0"` are legacy and being phased out.
- **MoE expert layout mismatch** → three distinct upstream layouts exist
  (qwen3_moe per-expert, qwen3_vl_moe transposed, qwen3_5_moe direct). Confirm
  which one applies before writing the converter.
- **`parallel_plan.py` EP keys must match the live param names on both v4 and
  v5** — when v4 reimplements `<M>Experts` with split `gate_proj`/`up_proj`
  (qwen3_moe, qwen3_omni_moe pattern) but v5 uses fused `gate_up_proj`, a
  single fused-only EP plan silently leaves v4's split params unsharded. Group
  GEMM then sees full-expert tensors and `assert len(cumsum_M) == b.shape[0]`
  fires inside `group_gemm_same_nk`. Fix: add a
  `use_gate_up_proj: bool = True` switch in `parallel_plan.py`, pass `False`
  from the v4 monkey patch, default `True` from patchgen — see
  `qwen3_moe/parallel_plan.py`. Audit by checking the live param names on the
  v4 expert class (`grep -n 'self\.\(gate\|up\|down\|gate_up\)_proj' modeling_<m>.py`)
  vs the EP keys in `parallel_plan.py`. qwen3_vl_moe is exempt because its v4
  inherits HF's already-fused `_Qwen3VLMoeTextExperts`.
- **Copy-pasting a sibling converter's docstring** — the `__doc__` on a
  neighboring `checkpoint_tensor_converter.py` is an unreliable source of truth
  for the HF layout; it was written for *that* model, not yours, and survives
  unchanged through copy-paste. Always cross-check against
  `conversion_mapping._MODEL_TO_CONVERSION_PATTERN[<model_type>]` and a real
  checkpoint's index file (Phase 3). This is exactly the trap the qwen3_omni_moe
  migration hit — docstring claimed "HF ships fused, transposed" (copied from
  qwen3_vl_moe) but HF actually ships per-expert split for qwen3_omni_moe
  (via the `qwen2_moe` alias). Direct `from_pretrained(...)` silently loaded
  zero expert weights until the converter was rewritten.
- **Blind-transpose fused-key converter corrupts v5-save round-trip** — when HF
  and v5 use *identical* fused expert key names but different axis orders
  (qwen3_vl_moe pattern), a converter that transposes every matching key will
  silently corrupt a v5-saved checkpoint on reload (VeOmni's training save path
  can emit the v5 layout directly). Dispatch on `tensor.shape[1]`: transpose
  only when it matches the HF layout, pass through when it matches v5, hard-error
  otherwise. The qwen3_moe-style per-expert converter is immune because its
  regex only matches HF-side keys (the v5 fused keys have different names).
- **Converter factory assumes nested `config.text_config`** → VLM-MoE submodels
  like `<M>TextModel` are loaded standalone with a flat `<M>TextConfig` that
  has no `text_config` attribute. Use
  `text_config = getattr(model.config, "text_config", model.config)` so the
  factory works for all three classes Pattern B registers the converter on.
- **Leaving `@use_experts_implementation` on the MoE experts class** — upstream
  v5 may decorate `<M>Experts` with this, which routes to `grouped_mm` and
  bypasses our fused path. Use `@config.replace_class("<M>Experts")` (not
  `override_method`) so the decorator is dropped in the generated file.
- **Forgetting to propagate `_moe_implementation` to `config.text_config`** in
  VLM-MoE models — the submodel reads `config.text_config._moe_implementation`,
  so override the top-level `__init__` to copy it down before `super().__init__(config)`.
- **Replacing `apply_rotary_pos_emb` with liger on partial-rotary models** —
  liger applies RoPE to full head_dim; partial-rotary models (e.g. qwen3_5_moe
  with `partial_rotary_factor=0.25`, `mrope_interleaved=True`) will NaN.
  Leave the upstream function alone; add a comment in the patchgen config.
- **Flash attention per-model patch** → don't. The hub-kernel adapter handles
  all three VeOmni custom FA names globally.
- **Loss function signature drift** — v5 `self.loss_function(...)` returns
  `(loss, logits)` and expects `hidden_states` + `weights` kwargs (see qwen3
  ForCausalLM.forward). Reusing a v4 loss call will silently compute nothing or
  double-compute logits.
- **VLM `vocab_size` lookup** — top-level VLM configs use
  `config.text_config.vocab_size`, not `config.vocab_size`. Same for
  `num_experts`, `num_experts_per_tok`, `router_aux_loss_coef` on VLM-MoE.
- **`logits_to_keep` handling** — v5 `ForCausalLM.forward` takes
  `logits_to_keep: int | torch.Tensor = 0` and slices `hidden_states` before the
  `lm_head` path. Omitting it breaks generation-time compatibility.
- **Registering converter on the wrong class tuple** — make sure `_create_checkpoint_tensor_converter`
  is attached to every concrete model class you import from `generated/`, not
  just `ForCausalLM`. Must use `staticmethod(...)`.
- **Duplicating patches across sibling models** — if qwen3_5 and qwen3_5_moe share
  a GatedDeltaNet / ViT, import the replacement functions from the sibling
  patchgen config and use `name_map={"OldPrefix": "NewPrefix"}` — don't copy.
- **Reusing a dense `Model.forward` on an MoE sibling via `name_map`** — name_map
  rewrites `<DensePrefix>*` → `<MoePrefix>*` at the AST level, but the
  constructed `<DensePrefix>ModelOutputWithPast(...)` return call is rewritten
  to `<MoePrefix>ModelOutputWithPast(...)` **with the same argument list as the
  dense version**, silently dropping MoE-only fields (`router_logits`).
  Downstream `ForConditionalGeneration.forward` then sees
  `outputs.router_logits = None`; `load_balancing_loss_func(None, ...)` returns
  int `0`, and either (a) aux_loss stays at 0 → router collapse, or
  (b) `0.to(loss.device)` crashes with `AttributeError`. Clone the forward body
  and hand-author the return whenever the sibling output dataclass has extra
  fields. `qwen3_vl_moe` hit this — see `qwen3_vl_moe_gpu_patch_gen_config.py`
  for the clone pattern.
- **`load_balancing_loss_func` can return a Python `int`, not a tensor** — when
  `router_logits` is `None` or an empty tuple, `load_balancing_loss_func(...)`
  returns scalar `0` (int), not `torch.tensor(0.0)`. Any later
  `loss += coef * aux_loss.to(loss.device)` will then raise
  `AttributeError: 'int' object has no attribute 'to'`. Guard with
  `isinstance(aux_loss, torch.Tensor)` before composing into `loss`, and
  prefer out-of-place `loss = loss + ...` over `+=` to avoid mutating a tensor
  that may be used elsewhere.
- **Non-picklable helpers inside override bodies** — VLM `get_position_id_func`
  returns a `partial` over a helper; that helper must be at module scope in the
  generated file (injected via `add_post_import_block`), not a local closure,
  or DataLoader worker processes will fail to pickle it.
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
  `self.lm_head(...)`. Slicing only in the `else` (no-labels) branch is a v4→v5
  regression — labels + `logits_to_keep>0` silently computes loss on the wrong
  positions.
- **SP + `compute_3d_position_ids` on-the-fly is incorrect** — under Ulysses SP
  the `input_ids` / `inputs_embeds` arriving at `<VLM>Model.forward` are per-rank
  slices; computing mrope positions on them produces positions that drift across
  ranks. VeOmni training expects precomputed position_ids via `get_position_id_func`
  in the data transform. If your patched `Model.forward` has a fallback branch
  that calls `compute_3d_position_ids` (or equivalent) when `position_ids is
  None`, raise a clear `RuntimeError` under `get_parallel_state().sp_enabled`
  rather than silently returning wrong positions. This keeps inference /
  generation (single-rank, SP off) working while fail-fast-ing under SP.
- **Forgetting `hidden_states` / `attentions` on custom return objects** — when
  your patched `Model.forward` or `ForConditionalGeneration.forward` manually
  constructs a `<M>ModelOutputWithPast` / `<M>CausalLMOutputWithPast` (instead
  of relying on the upstream `@can_return_tuple`-decorated path), always pass
  through `hidden_states=outputs.hidden_states` and
  `attentions=outputs.attentions`. Otherwise callers using
  `output_hidden_states=True` / `output_attentions=True` silently get `None`.
  This is a recurring v4→v5 regression because v4 models often returned bare
  tuples and dropped these fields implicitly.
- **Hardcoded shapes in `<M>VisionModel.dummy_forward`** — compute pixel row
  size and `grid_thw` from `self.config.patch_size` / `temporal_patch_size` /
  `in_channels` and `self.spatial_merge_size`, not from the model variant you
  first tested. Grids must be multiples of `spatial_merge_size` (merger
  requirement); under SP, scale one spatial dim by `sp_size` so the post-slice
  seq length stays a multiple of `sp_size`.
- **`self.dtype` / cached `_dummy_data` in `dummy_forward` is wrong under
  FSDP2 + MixedPrecisionConfig** — `self.dtype` returns the *first parameter's*
  dtype, which under FSDP2+MixedPrecision is the stored dtype (fp32), not the
  per-call compute dtype (bf16) the framework casts weights to at forward time.
  If `dummy_forward` allocates inputs via `torch.zeros(..., dtype=self.dtype)`
  or caches a `_dummy_data` buffer at `__init__`, the first conv/linear on a
  text-only rank crashes with "Input type (float) and bias type
  (c10::BFloat16) should be the same", while the multimodal rank hangs on the
  collective — masquerading as an NCCL hang. Always look up dtype from a live
  parameter at call time (e.g. `dtype = self.conv2d1.weight.dtype`,
  `dtype = self.patch_embed.proj.weight.dtype`) and don't cache dummy tensors
  across calls. See `qwen3_omni_moe_gpu_patch_gen_config.py`'s audio / vision
  `dummy_forward` patches.
- **FSDP2 "hang" may be a rank-asymmetric crash** — when one rank crashes
  inside a collective-spanning forward (dtype mismatch, shape mismatch,
  unexpected `None`), the surviving ranks block on the never-completing
  collective and the test wall-clocks to SIGTERM. Re-run with
  `TORCH_DISTRIBUTED_DEBUG=DETAIL` to force the per-rank exception to surface;
  once you see the real traceback on the crashing rank, fix *that* rather than
  hunting for deadlocks in the happy-path code.
- **`gather_dim` for cos/sin in async Ulysses attention paths** — the correct
  seq dim depends on whether a pre-attention RoPE reshape has happened. In
  Qwen3-VL v5, `apply_interleaved_mrope` runs before attention and collapses
  the leading 3-axis, so cos/sin arriving at async Ulysses is
  `(bs, seq_len, head_dim)` → `gather_dim=1`. Don't blindly copy `gather_dim`
  from a sibling model; read the upstream RoPE path first.
- **Skipping `check_patchgen`** → CI will fail on PR. Always run it locally.
- **`pytest -k` mismatch on e2e** — the per-modality `test_e2e_parallel_*.py`
  files use the first positional arg (`model_name`) as id, not the registry
  `<m>` id. For VL
  models that's the HF short name (`qwen25vl`, `qwen3vl`, `qwen3vlmoe`, …),
  which has no underscores and does NOT match `-k qwen2_5_vl`. See Phase 7
  keyword-rules table.
- **Only regenerating GPU when NPU config exists** — if the model has a sibling
  `<m>_npu_patch_gen_config.py`, run codegen for **both** (or use `--all`) before
  committing. CI checks both generated files for drift.

---

## Scope Guard

This skill migrates an **existing** model directory to v5. For:

- New model (does not yet exist under `veomni/models/transformers/`): use
  `/veomni-new-model`.
- New op / kernel: use `/veomni-new-op`.
- uv / dependency bumps (e.g. upgrading to a new `transformers5-exp` version):
  use `/veomni-uv-update`.
- Bugs uncovered during migration: use `/veomni-debug`.
