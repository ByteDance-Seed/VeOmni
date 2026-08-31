# MoE specifics

Read this when the model has routed experts. It covers the patches Phase 2
adds for MoE, the Phase 3 checkpoint converter, and the MoE-only pitfalls.
This is *in addition to* the SKILL.md spine, not a replacement for it.

## Phase 2 additions

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

## Phase 3: checkpoint tensor converter

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
   — see table in
   `docs/transformers_v5/transformers_v5_moe_weight_loading.md`:
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

## Pitfalls

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
- **Registering converter on the wrong class tuple** — make sure `_create_checkpoint_tensor_converter`
  is attached to every concrete model class you import from `generated/`, not
  just `ForCausalLM`. Must use `staticmethod(...)`.
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
- **Parallel plan keys must track the fused expert layout** — `parallel_plan.py`
  shards `model.layers.*.mlp.experts.gate_up_proj` (Shard(0)) and
  `model.layers.*.mlp.experts.down_proj` (Shard(0)). Stale split-key plans
  leave `gate_up_proj` un-sharded and EP training hits
  `AssertionError: len(cumsum_M) == b.shape[0]` inside `group_gemm_same_nk`
  (cumsum length = `E_local`, but the weight has all `E` experts). See
  `veomni/models/transformers/deepseek_v3/parallel_plan.py`.
- **Checkpoint converters must detect the fused layout** — HF checkpoints may
  already ship `experts.gate_up_proj` / `experts.down_proj`. A
  `CheckpointTensorConverter` that unconditionally stacks per-expert
  `gate_proj`/`up_proj`/`down_proj` will raise
  `KeyError: '...experts.0.gate_proj.weight'`. Guard with a key-existence check,
  skip stacking when fused keys are already present, and cover both layouts in
  `tests/models/test_checkpoint_tensor_converter.py`.
