---
name: veomni-migrate-transformers-v5
description: "Use this skill when adding or refreshing a patchgen-generated modeling file for a VeOmni model under veomni/models/transformers/<model>/generated/ — GPU-only or GPU+NPU, dense or MoE, text-only / VLM / Omni-thinker+talker. Covers: creating <model>_{gpu,npu}_patch_gen_config.py, using patchgen decorators (replace_class/override_method/replace_function/modify_init/add_post_import_block/drop_import_names), reusing sibling-model patches via name_map, handling MoE weight-loading (CheckpointTensorConverter + fused gate_up_proj layout), multimodal/VLM forward with Ulysses SP, excluding speech/vocoder subtrees in Omni models (talker/token2wav/DiT/BigVGAN), wiring __init__.py for the patchgen-generated classes, running codegen, and adding test cases. Trigger: 'port <model> to patchgen', 'add patchgen for <model>', 'transformers v5 migration', 'add NPU patchgen'. Do NOT edit files under generated/ manually — always regenerate via patchgen."
---

# VeOmni Transformers v5 Patchgen Protocol

Purpose: add or refresh a model's patchgen-generated modeling under
`veomni/models/transformers/<model>/generated/`. VeOmni pins
`transformers==5.2.0` and ships patchgen-generated modeling for every
supported model; legacy v4 monkey-patches have been retired.

**References (read first, load on demand):**

- `docs/transformers_v5/index.md` — overview of what v5 migration covers
- `docs/transformers_v5/patchgen.md` — patchgen DSL, CLI, CI drift check
- `docs/transformers_v5/transformers_v5_moe_weight_loading.md` — MoE fused-expert layout + runtime converter
- `docs/transformers_v5/veomni_flash_attention_kernel_adapter.md` — FA custom-name adapter
- `docs/transformers_v5/testing_new_model.md` — v5 test case SOP

**Working examples (copy the structure, do not edit `generated/`):**

Examples grouped by complexity / capability — pick the closest one and adapt:

- **Text LLM (dense)** — `veomni/models/transformers/qwen3/`, `veomni/models/transformers/llama/`, `veomni/models/transformers/qwen2/`, `veomni/models/transformers/seed_oss/`
  - `__init__.py` — registers a patchgen-generated `<Model>ForCausalLM` / `<Model>Model` / `<Model>ForSequenceClassification` via `MODELING_REGISTRY`.
  - `<m>_gpu_patch_gen_config.py` — Liger + SP + fused-CE patches. Llama is the minimal reference (5 OpSlot patches: RMSNorm, MLP, RoPE, ForCausalLM, ForSequenceClassification — no SP or MoE specifics).
- **Text LLM with NPU patchgen** — `veomni/models/transformers/seed_oss/`
  - `__init__.py` — branches on `IS_NPU_AVAILABLE` between `patched_modeling_seed_oss_{gpu,npu}`.
  - Sibling configs produce separate `generated/*_{gpu,npu}.py` outputs.
- **MoE** — `veomni/models/transformers/qwen3_moe/`
  - `__init__.py` — attaches `_create_checkpoint_tensor_converter` as a `staticmethod` on every patchgen-generated class.
  - `qwen3_moe_gpu_patch_gen_config.py` — replaces `Qwen3MoeExperts` with the fused-MoE layout and overrides `get_parallel_plan`.
  - `checkpoint_tensor_converter.py` — HF per-expert → fused runtime converter.
  - `parallel_plan.py` — single `get_parallel_plan()` sharding the fused `gate_up_proj`.
- **MoE + NPU patchgen** — `veomni/models/transformers/deepseek_v3/`
  - Sibling `deepseek_v3_{gpu,npu}_patch_gen_config.py`; both generated files committed.
  - Runtime kernel choice (deterministic Triton RoPE + batch-invariant RMSNorm) is wired in `__init__.py` via `apply_veomni_deepseek_v3_device_patch(gen_module)` for actor/rollout numerical parity. No Liger kernels in the generated file itself.
- **VLM (non-MoE) + GPU+NPU patchgen** — `veomni/models/transformers/qwen3_vl/`
  - `__init__.py` — registers the patchgen-generated classes, branching on `IS_NPU_AVAILABLE` between `patched_modeling_qwen3_vl_{gpu,npu}`.
  - `qwen3_vl_gpu_patch_gen_config.py` — full VLM forward with Ulysses SP, async Ulysses text attention, deepstack, precomputed mrope via `get_position_id_func`, and a SP-aware `dummy_forward`.
  - `qwen3_vl_npu_patch_gen_config.py` — demonstrates the **NPU-inherits-GPU** pattern: a thin NPU config that extends `gpu_config.helpers` / `gpu_config.post_import_blocks` / `gpu_config.additional_imports` and only overrides RMSNorm / rotary with `torch_npu.npu_rms_norm` / `torch_npu.npu_rotary_mul`. Avoids duplicating ~1K lines of shared VLM SP/deepstack patches.
- **Omni (thinker+talker subtree, non-MoE)** — `veomni/models/transformers/qwen2_5_omni/`
  - `__init__.py` — imports `Qwen2_5OmniForConditionalGeneration` / `Qwen2_5OmniThinkerForConditionalGeneration` from the patchgen-generated module **and** `Qwen2_5OmniTalkerModel` / `Qwen2_5OmniTalkerForConditionalGeneration` directly from `transformers.models.qwen2_5_omni.modeling_qwen2_5_omni` (talker classes are excluded from the generated file but the registry still needs to return them when `architecture` mentions `Talker...`). `MODEL_CONFIG_REGISTRY` applies the `tie_word_embeddings=False` config patch.
  - `qwen2_5_omni_gpu_patch_gen_config.py` — the canonical **non-MoE Omni** template: excludes talker + token2wav + DiT + BigVGAN subtrees, overrides `_init_weights` to drop excluded `UpSample1d`/`DownSample1d` branches, overrides `ForConditionalGeneration.__init__` to force `has_talker=False` and pin `_no_split_modules=[DecoderLayer, VisionBlock, AudioEncoderLayer]` (use a `list[str]` to match the upstream HF convention — `modeling_utils.py` converts it to a set internally, so either works at runtime, but staying with `list[str]` keeps the patched class isomorphic with the upstream base class attr), registers a load-state-dict pre-hook to strip `talker.*`/`token2wav.*` keys, overrides `enable_talker`/`generate` to raise `NotImplementedError`, and forwards `ForConditionalGeneration.forward` to thinker only — **minus** all MoE/EP machinery (no `replace_class("…Experts")`, no `parallel_plan.py`, no `checkpoint_tensor_converter.py`). Thinker uses `Qwen2_5OmniThinkerCausalLMOutputWithLogProbs` from `veomni.utils.model_outputs` to carry `log_probs`/`entropy` as constructor fields (same FSDP2 unshard-hook rationale as qwen3_omni_moe). Audio encoder uses 1D convs (`conv1`/`conv2`) — pull dummy-forward dtype from `self.conv1.weight.dtype`, not `self.conv2d1` (that's qwen3_omni_moe-specific).
  - **No `parallel_plan.py` / no `checkpoint_tensor_converter.py`** — qwen2.5-Omni's thinker text model is dense (Qwen2-class MLP, not MoE), so neither EP nor fused-expert weight conversion applies. If you start from the qwen3_omni_moe template and forget to delete these, you'll get import errors from dangling references.
- **VLM + MoE + GPU+NPU patchgen** — `veomni/models/transformers/qwen3_vl_moe/`
  - `__init__.py` — registers three classes (`Qwen3VLMoeForConditionalGeneration`, `Qwen3VLMoeModel`, `Qwen3VLMoeTextModel`) and attaches `_create_checkpoint_tensor_converter` as a `staticmethod` on each (the inner text submodel is also loadable standalone and must carry the converter).
  - `qwen3_vl_moe_gpu_patch_gen_config.py` — minimal config that imports *most* VLM SP / deepstack / async-Ulysses / dummy_forward patches from `qwen3_vl` via `name_map={"Qwen3VL": "Qwen3VLMoe"}`, and only writes MoE-specific deltas: `replace_class("Qwen3VLMoeExperts")` with fused layout, `override_method("Qwen3VLMoeModel.__init__")` to propagate `_moe_implementation` into `config.text_config`, a hand-cloned `Qwen3VLMoeModel.forward` (see below), `Qwen3VLMoeForConditionalGeneration.forward` with fused loss + aux_loss, and `get_parallel_plan`. This is the canonical template for any new VLM+MoE migration. **Exception — do NOT reuse `Model.forward` via name_map**: `Qwen3VLMoeModelOutputWithPast` carries an extra `router_logits` field absent from the dense `Qwen3VLModelOutputWithPast`; rewriting class names at the AST level keeps the dense constructor's argument list, silently dropping `router_logits` and collapsing MoE routing. Clone the forward body and hand-author the return.
  - `checkpoint_tensor_converter.py` — HF ships *fused* expert tensors under the *same key names* as VeOmni but in transposed layout (`[E, H, 2*I]` vs `[E, 2*I, H]`). Uses dim-1 shape dispatch to recognize HF vs VeOmni layout, passes VeOmni-native tensors through untouched, and hard-errors on unrecognized shapes — see Phase 3 "round-trip safety".
- **Text + linear attention (`qwen3_5`) / VLM + MoE (`qwen3_5_moe`)** — `veomni/models/transformers/qwen3_5/`, `qwen3_5_moe/`
  - `qwen3_5_moe_gpu_patch_gen_config.py` — demonstrates `config.drop_import_names(...)`, `config.add_post_import_block(...)`, cross-config reuse via `from ...qwen3_5.qwen3_5_gpu_patch_gen_config import <fn>`, and `name_map={"Qwen3_5": "Qwen3_5Moe"}` on `override_method` to share patches between sibling configs.
- **MLA + MoE (GLM)** — `veomni/models/transformers/glm_moe_dsa/`
  - Sibling `glm_moe_dsa_{gpu,npu}_patch_gen_config.py` produces separate `generated/*_{gpu,npu}.py` outputs.

---

## Phase 0: Environment + Reference Setup

### 0.1 Verify transformers venv

Patchgen runs against `transformers==5.2.0`. Before touching code:

```bash
source .venv/bin/activate
python -c "import transformers; print(transformers.__version__)"
```

If not `5.2.0`, re-sync the default env:

```bash
uv sync --frozen --extra gpu --extra audio --group dev
source .venv/bin/activate
```

### 0.2 (Strongly recommended) Drop HF reference source into `.agents_workspace/`

`.agents_workspace/` is gitignored. Keeping the upstream HF source next to your
patchgen config is the single biggest accelerator for catching subtle
signature/contract drift while iterating.

```bash
mkdir -p .agents_workspace/hf_reference/<m>/v5_2_0

curl -sL -o .agents_workspace/hf_reference/<m>/v5_2_0/modeling_<m>.py \
  "https://github.com/huggingface/transformers/raw/v5.2.0/src/transformers/models/<m>/modeling_<m>.py"
```

For VLMs also grab `processing_<m>.py` / `image_processing_<m>.py` /
`configuration_<m>.py` if you expect processor-side or config-shape work.

If you are **refreshing** an existing patchgen-generated file across a
transformers minor bump (e.g. `5.2.0 → 5.3.0`), pull both versions side-by-side
and diff to spot contract drift:

```bash
mkdir -p .agents_workspace/hf_reference/<m>/{old,new}
curl -sL -o .agents_workspace/hf_reference/<m>/old/modeling_<m>.py \
  "https://github.com/huggingface/transformers/raw/v5.2.0/src/transformers/models/<m>/modeling_<m>.py"
curl -sL -o .agents_workspace/hf_reference/<m>/new/modeling_<m>.py \
  "https://github.com/huggingface/transformers/raw/v5.3.0/src/transformers/models/<m>/modeling_<m>.py"
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
- Helper-method signatures (e.g. `get_placeholder_mask` takes `inputs_embeds`
  + `image_features` / `video_features`).
- Return-shape conventions: e.g. `get_{image,video}_features.pooler_output`
  is a `tuple[per-image tensor]` after `torch.split`, not a flat tensor.
- Packed position-ids contract (`[4, bs, seq-len]` with prepended
  `text_position_ids`).
- **RoPE shape collapse** — VLMs use `apply_interleaved_mrope` (and similar
  helpers) that collapse the leading 3-axis of mrope before layers see
  cos/sin, so the shape is `(bs, seq_len, head_dim)`. Any SP path that gathers
  cos/sin across the sequence dim (async Ulysses, ring attention) must use
  the correct `gather_dim`. Grep upstream for `interleaved_mrope`,
  `mrope_section`, or any pre-attention RoPE reshape before writing the patch.
- **`attention_mask` may be a dict** — HF v5 routinely passes
  `attention_mask={"full_attention": <tensor>, ...}` keyed by attention type.
  Any patched forward that forwards `attention_mask` to
  `compute_3d_position_ids` / `get_rope_index` / other tensor-expecting
  helpers must defensively unwrap `attention_mask.get("full_attention", None)`
  when it's a dict.

Keep this directory around through commit; delete it after the PR merges (it's
already gitignored so it won't leak into the repo).

---

## Before You Start: Create Todos

Use TodoWrite to track phases. Suggested plan:

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

**Input**: model name `<M>` (e.g. `qwen3_5`, `glm4_moe`).

**Operations:**

1. Confirm model exists at `veomni/models/transformers/<M>/`. If not, the task is
   "add new model" — use `/veomni-new-model` instead.
2. If a patchgen-generated file already exists under
   `veomni/models/transformers/<M>/generated/` you are **refreshing** an
   existing config (e.g. picking up upstream changes, adding NPU sibling,
   fixing a bug). Otherwise you are adding patchgen support to a model whose
   `__init__.py` previously imported HF classes directly. Either way, the rest
   of this protocol applies identically.
3. Decide backend coverage:
   - GPU only → one `<m>_gpu_patch_gen_config.py` + one
     `generated/patched_modeling_<m>_gpu.py`.
   - GPU + NPU → add sibling `<m>_npu_patch_gen_config.py` that writes
     `generated/patched_modeling_<m>_npu.py`; mirror the `glm_moe_dsa` or
     `qwen3_vl` layout.
4. Check model category:
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
   diverge between sibling models — see `transformers_v5_moe_weight_loading.md`.
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
  dead code in `generated/`. For qwen2_5_omni's BigVGAN vocoder,
  `UpSample1d`/`DownSample1d` are referenced **both** by Token2Wav residual
  blocks (caught by exclusion) **and** by the base `_init_weights` method
  via `isinstance` checks (NOT caught — `ast.walk` doesn't trace
  `isinstance` strings). After excluding the speech subtree, always
  `rg "isinstance\(.*<excluded_class>" generated/` and override the methods
  that still reference excluded names.
- **`_init_weights` referencing excluded classes** — base `PreTrainedModel._init_weights`
  often has `isinstance(module, <SpeechHeadClass>)` / `<UpSample1d>` /
  `<SnakeBeta>` branches that init excluded modules. These do not generate a
  patchgen warning but explode at first model build with `NameError: name 'X'
  is not defined` (ruff also flags as `F821`). Always override `_init_weights`
  to drop branches that touch excluded classes — see qwen2_5_omni's override
  that strips `UpSample1d`/`DownSample1d` branches.
- **Upstream `generate()` with mutable default arg** — Omni models like
  qwen2_5_omni define `generate(..., talker_eos_token_id: list[int] = [8292, 8294], ...)`
  which `ruff B006` rejects when copied verbatim into the generated file.
  Since the speech path is excluded anyway, override `<M>ForConditionalGeneration.generate`
  to raise `NotImplementedError("...generate is disabled in the VeOmni
  training modeling (talker / token2wav are excluded). Use upstream
  transformers for TTS generation.")`. This double-serves to kill the lint
  and make the contract explicit.

See `qwen3_omni_moe_gpu_patch_gen_config.py` (MoE thinker) and
`qwen2_5_omni_gpu_patch_gen_config.py` (dense thinker) for the canonical
templates. Both exclude the whole speech subtree plus the dead-after-exclusion
activations (`SnakeBeta` for qwen3_omni_moe; `UpSample1d`/`DownSample1d` for
qwen2_5_omni's BigVGAN), override `_init_weights` to drop the excluded-module
branches, override `enable_talker` to raise, and (for qwen2_5_omni) also
override `ForConditionalGeneration.generate` to raise `NotImplementedError`
— upstream's `generate(...)` signature has a mutable default arg
(`talker_eos_token_id: list[int] = [...]`) that trips `ruff B006` in the
generated file, and the TTS path is excluded anyway.

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
  `parallel_plan.get_parallel_plan()`. `parallel_plan.py` shards the fused
  `model.layers.*.mlp.experts.gate_up_proj` (Shard(0)) — see
  `qwen3_moe/parallel_plan.py` for the canonical template.
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
python -m veomni.patchgen.run_codegen \
    veomni.models.transformers.<m>.<m>_gpu_patch_gen_config \
    -o veomni/models/transformers/<m>/generated --diff
```

**Validation**: file is syntactically valid (import it: `python -c "import
veomni.models.transformers.<m>.<m>_gpu_patch_gen_config"`) and every behaviour
identified in Phase 1 has a corresponding decorator here.

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

## Phase 6: Add Test Cases

Follow `docs/transformers_v5/testing_new_model.md`. Minimum coverage:

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
# uv sync --extra gpu --extra audio --dev
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
- **Hand-writing `modeling_<m>.py` / `gpu_patch.py`** → don't. The
  patchgen-generated file under `generated/` is the single source of truth;
  legacy monkey-patch modules have been retired.
- **MoE expert layout mismatch** → three distinct upstream layouts exist
  (qwen3_moe per-expert, qwen3_vl_moe transposed, qwen3_5_moe direct). Confirm
  which one applies before writing the converter.
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
- **Loss function signature** — `self.loss_function(...)` returns
  `(loss, logits)` and expects `hidden_states` + `weights` kwargs (see qwen3
  ForCausalLM.forward). Calling it the old pre-v5 way will silently compute
  nothing or double-compute logits.
- **VLM `vocab_size` lookup** — top-level VLM configs use
  `config.text_config.vocab_size`, not `config.vocab_size`. Same for
  `num_experts`, `num_experts_per_tok`, `router_aux_loss_coef` on VLM-MoE.
- **`logits_to_keep` handling** — `ForCausalLM.forward` takes
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
  `self.lm_head(...)`. Slicing only in the `else` (no-labels) branch silently
  computes loss on the wrong positions when labels + `logits_to_keep>0` are
  both set.
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
  parameter at call time and don't cache dummy tensors across calls. The
  exact attribute is **model-specific** and copy-pasting the wrong one is a
  classic-silently-broken bug:
  - qwen3_omni_moe audio: `dtype = self.conv2d1.weight.dtype` (2D conv front-end)
  - qwen2_5_omni audio: `dtype = self.conv1.weight.dtype` (1D conv front-end —
    qwen3_omni_moe-style `conv2d1` does not exist on this model)
  - qwen2_5_omni / qwen3_omni_moe vision: `dtype = self.patch_embed.proj.weight.dtype`
  See the audio / vision `dummy_forward` patches in
  `qwen2_5_omni_gpu_patch_gen_config.py` and
  `qwen3_omni_moe_gpu_patch_gen_config.py`.
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
- **`TypeError: expected string or buffer` when manually exercising
  `MODEL_CONFIG_REGISTRY` before `MODELING_REGISTRY` (Omni models with patched
  configs)** — calling `MODEL_CONFIG_REGISTRY.get("<m>")()` *before*
  `MODELING_REGISTRY.get("<m>")()` causes the config-registration monkey patch
  to fire first; transformers' `@auto_docstring` then tries to read the patched
  config class's source via `CONFIG_MAPPING` and gets a live Python object
  instead of a source string. This blows up inside upstream
  `transformers/utils/auto_docstring.py`. **Not a real bug** — the natural model
  build order (`build_foundation_model_from_config(...)` → `MODELING_REGISTRY`
  first, which imports modeling and triggers the config import transitively)
  hits the right order and the error never fires. Only matters if your smoke
  test calls the registries directly in the wrong order. Confirmed on
  qwen2_5_omni / qwen3_omni_moe.
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
- **Parallel plan keys must track the fused expert layout** — `parallel_plan.py`
  shards `model.layers.*.mlp.experts.gate_up_proj` (Shard(0)) and
  `model.layers.*.mlp.experts.down_proj` (Shard(0)). Stale split-key plans
  leave `gate_up_proj` un-sharded and EP training hits
  `AssertionError: len(cumsum_M) == b.shape[0]` inside `group_gemm_same_nk`
  (cumsum length = `E_local`, but the weight has all `E` experts). See
  `veomni/models/transformers/deepseek_v3/parallel_plan.py`.
- **Sync-weight adapters must detect the fused layout** — HF checkpoints may
  already ship `experts.gate_up_proj` / `experts.down_proj`. Test adapters in
  `tests/models/weight_sync_adapters.py::sync_weight_<m>` that unconditionally
  stack per-expert `gate_proj`/`up_proj`/`down_proj` will raise
  `KeyError: '...experts.0.gate_proj.weight'`. Guard with a key-existence
  check and skip stacking when the fused keys are already present.

---

## Scope Guard

This skill adds or refreshes patchgen-generated modeling for an **existing**
model directory under `veomni/models/transformers/`. For:

- New model (does not yet exist under `veomni/models/transformers/`): use
  `/veomni-new-model`.
- New op / kernel: use `/veomni-new-op`.
- uv / dependency bumps (e.g. upgrading the `transformers-stable` pin): use
  `/veomni-uv-update`.
- Bugs uncovered during this work: use `/veomni-debug`.
