---
name: seedomni-v2
description: "Use this skill when adding or modifying anything inside `veomni/models/seed_omni/` (the SeedOmni V2 graph-based multi-modal model). Covers: writing a new `XxxModuleMixin` (subclass `ModuleMixin` + HF/diffusers model multi-inheritance), wiring it into the YAML graph (`modules` / `nodes` / `edges` / `training_graph` / `generation_graph`), writing `configuration.py` + `modulemixin.py` + `modeling.py` (+ optional `processing.py`), updating the split-checkpoint script, registering with `OMNI_*_REGISTRY`, and validating with the visualization script + tests. Trigger: 'add seedomni module', 'new omnimodule', 'extend seed_omni', 'add encode/decode module', 'wire into omni graph', 'split <model> checkpoint into omni modules', 'modify training_graph / generation_graph', 'add SeedOmni V2 backbone'."
---

## Required Reading (before any code change)

1. `design.md` — full SeedOmni V2 design rationale. The skill assumes you've internalized:
   - Two-pool model (`nodes` = call-sites, `edges` = data routes; independent namespaces).
   - `to: end` reserved keyword (no orphans, no cycles).
   - `ModuleMixin` is a **mixin**, not a base class; per-module hooks live in `*ModuleMixin`.
   - `_loss` suffix collection (single key): each module loops all micro-batches inside one `forward`, `post_forward` does the token-level mean, and emits a scalar `<name>_loss`. OmniModel just sums.
   - **Data is 100% model-agnostic**: `raw_batch` starts with a single key `conversation_list` (`list[list[dict]]` of `{type, value, role, loss_mask}` items). All chat templating, tokenization, image processing, audio feature extraction, and boundary-marker injection happen **inside model modules** during forward. The same dataset can feed any ug model — Janus / Qwen-Omni / Bagel — without changes.
   - **Per-module assets including tokenizer**: every processor (vision / image / audio) AND the tokenizer live inside their owning module's subfolder. There is **no top-level `tokenizer_path`** field — the tokenizer is owned by the family-specific `text_encoder` module (`modules/<family>/text_encoder/tokenizer/`). Pure DiT models without a text encoder have no tokenizer at all.
   - **Module forward → mutates raw_batch**: every module's `forward(**kwargs) -> Dict` return dict is **immediately written back into raw_batch** by OmniModel (keyed by `edge.output`). Data does **not** flow through edge channels to downstream modules — downstream reads the same `raw_batch` by its own input keys. Edges are dependency / topology contracts, not data conduits.
   - **No global collator final-step, no global SP slice node**: each module calls collator helpers and applies SP slicing **inside its own `pre_forward`** for the fields it owns. ViT slices the image batch dimension; text encoder slices the sequence dimension; nothing gets sliced twice.
   - **Status**: this is the target contract; the current code still runs the V1-compatible path (`multimodal_chat_template.py` does chat templating + N pre-expanded placeholders; `JanusLlama.pre_forward` uses `masked_scatter`). Migration happens feature-by-feature (D1: lighter `multimodal_transform`; D2: lighter collator; D3: vision modules take over image processing + boundary markers; D4: `text_encoder` takes over chat template + tokenize; D5: backbone splice). Don't try to land it all in one PR.
2. `veomni/models/seed_omni/module.py` — the `ModuleMixin` base contract.
3. `veomni/models/seed_omni/graph.py` + `training_graph.py` + `generation_graph.py` — `NodeDef` / `EdgeDef` schema and the DAG / FSM views.
4. An existing module that matches your shape:
   - **Generic / cross-family** → `modules/base/text_encoder/modeling.py` (and the planned `modules/base/mlp_adapter/modeling.py`).
   - **Vision encoder** → `modules/janus/siglip/modeling.py`.
   - **VQ codec** (encode + decode) → `modules/janus/vqvae/modeling.py`.
   - **AR backbone (no vocab layers)** → `modules/janus/llama/modeling.py`.

   File layout convention: `modules/<family>/<sub_module>/(configuration.py, modeling.py[, processing.py])`. The folder name carries the namespace, so the inner files use the short `configuration.py` / `modeling.py` / `processing.py` names rather than re-spelling the family + sub-module in every filename. Cross-family lightweight modules live under `modules/base/<sub_module>/` with the same shape.

## Data flow (read this once)

> **Status**: the pipeline below is the **target contract** for V2. The current code under `veomni/data/multimodal/multimodal_chat_template.py` + `JanusLlama.pre_forward` still runs the V1-compatible "chat-template helper + N pre-expanded placeholders + masked_scatter" form. Migration happens feature-by-feature (D1-D5; see `design.md`).

V2 has **no framework-level chat template, no modality slot router, no top-level tokenizer**. The pipeline is six serial layers; the data is `model-agnostic` end-to-end up until layer 4.

```
Layer 1: jsonl on disk
         ──────────────────────────────────────────
         each line = list[dict]  (type / value / role / loss_mask)

Layer 2: multimodal_transform.py (lightweight tool layer)
         ──────────────────────────────────────────
         basic IO + resize per modality:
           type=image: read + resize → Tensor(C,H,W) into item["value"]
           type=video: read + resize_frames → Tensor(...) into item["value"]
           type=audio: load_audio → Tensor(...) into item["value"]
           type=text:  keep string
         ❌ no chat template, no tokenize, no image processor
         ❌ no boundary-marker injection
         output is still conversation_list (only item.value upgraded path → tensor)

Layer 3: dataloader / collator (basic)
         ──────────────────────────────────────────
         batch N samples into raw_batch = {"conversation_list": [conv_0, ..., conv_{N-1}]}
         ❌ no sequence-domain padding (input_ids does not exist yet)
         ❌ no SP slicing

         [OmniModel.forward / generate entry; raw_batch holds only conversation_list]

Layer 4: vision / audio encoder modules (forward stage)
         ──────────────────────────────────────────
         ViT / VAE / audio_encoder each:
           1. pre_forward: extract own modality items from conversation_list,
              stack into batch tensor, apply SP slice on its OWN dim
              (image batch / patch dim, NOT sequence)
           2. forward: image_processor (patch / normalize) → encoder
              → image_embeds / vq_token_ids / audio_embeds
           3. mutate conversation_list: insert {type:"boi"} / {type:"eoi"}
              before/after each image item; audio modules insert audio_bos/eos;
              boundary markers inherit role / loss_mask from the
              original modality item
           4. return dict written back to raw_batch by framework

Layer 5: text_encoder module (base provides default; family overrides chat-template)
         ──────────────────────────────────────────
         base module:    modules/base/text_encoder/  (default impl: split-conversation
                                                      + tokenize + wte + segment-split output)
         family module:  modules/<family>/text_encoder/  (inherits base; tokenizer asset
                                                          lives here; only overrides
                                                          chat-template details)
         pre_forward: collator helper to gather batch
         forward.encode (single pass):
           1. for each sample, walk conversation_list (now containing boundary
              markers from layer 4) and apply this family's chat template:
              - system / user / assistant role prefixes
              - per item:
                  type=text:        tokenizer.encode(item.value)
                  type=boi/eoi/...: tokenizer.convert_tokens_to_ids("<boi>")
                  type=image/video/audio/vq_image: 1 placeholder token id
                                                   (backbone splice expands it)
              - append EOS
           2. compute labels (image/audio segments → -100; text segments use
              role + loss_mask)
           3. compute attention_mask
           4. wte → flat inputs_embeds (each modality item still 1 placeholder slot)
           5. SPLIT: emit a NEW conversation_list, sliced by the original item
              boundaries. Each segment carries:
                {type, value: <Tensor (L_seg, D)>, role, loss_mask}
              text segment value = wte embeddings of all its tokens (multi-token)
              modality segment value = 1 placeholder embedding (single-token tensor)
              boundary marker segment value = 1 marker token embedding
           6. SP slice (input_ids / inputs_embeds / labels / attention_mask)
         return: {
           conversation_list (split, value=embeds segments),
           input_ids, inputs_embeds (flat), labels, attention_mask,
         }

Layer 6: backbone (JanusLlama / QwenOmniThinker / ...)
         ──────────────────────────────────────────
         pre_forward:
           1. segment-driven splice: walk the SPLIT conversation_list:
                - text/boundary segment: keep segment.value as-is
                - image/audio/vq_image segment: replace its 1-token placeholder
                  embedding with the upstream encoder's N patch tokens
                  (matched by appearance order)
              Then concat all segments → final inputs_embeds.
           2. regenerate labels (-100 for image segments) and attention_mask (1)
              from segment metadata
           3. compute_position_ids on post-splice length
           4. final SP pad_and_slice
         forward → hidden_states
         post_forward → SP gather
```

Key invariants:
- **conversation_list is the ONLY raw_batch field at entry**. All other fields (input_ids, image_embeds, attention_mask, labels, position_ids, hidden_states) are produced by modules during forward and written back into raw_batch via return dicts. **(Exception: offline-embedding training — see "Offline Embedding" section in design.md — has a different raw_batch entry shape because the dataset itself is a pickle of pre-computed tensors.)**
- **Each module owns its token concatenation + boundary injection**. text_encoder assembles system prompt + EOS + role markers + emits the split-conversation output; ViT writes `boi`/`eoi` into conversation_list; audio writes `audio_bos`/`audio_eos`. There is no chat-template helper, no global image_pattern registry.
- **Two transformations of conversation_list in forward**: (a) Layer 4 ViT/VAE INSERT marker items; (b) Layer 5 text_encoder SPLITS the list and replaces item.value with embedding-segment tensors; (c) Layer 6 backbone collapses the split list into flat inputs_embeds via segment-driven splice. labels / attention_mask / position_ids are computed at Layers 5 and 6.
- **Backbone splice is segment-order driven, NOT token-id-position driven**: don't scan input_ids for placeholder positions; iterate segments and replace each modality segment's placeholder embedding with the corresponding ViT/VAE output (by appearance order). No `image_pos` / `und_image_pos` index fields needed.
- **Graph topology auto-orders modules**: ViT/VAE write `conversation_list`, text_encoder reads `conversation_list`, so topological sort places ViT/VAE before text_encoder; text_encoder writes the split `conversation_list` (and `inputs_embeds`), backbone reads the split `conversation_list`, so text_encoder lands before backbone. **No explicit ordering edges needed.**

When adding a new modality:
- Create a new encoder OmniModule with its processor (module-private asset under `modules/<family>/<encoder>/`).
- In its forward, extract own-type items from `conversation_list`, encode, and inject `<modality>_bos` / `<modality>_eos` markers around each item.
- The family's `text_encoder` module must know the new boundary token strings (or you add a hook on `text_encoder` that the new encoder calls during init to register its markers).
- Add an edge `<new_encoder> → <backbone>` so the backbone receives the new embeds for splice.
- Extend the backbone's splice to handle the new modality's embeds and update `compute_position_ids` if the modality affects RoPE layout.

You **do not** add a chat-template variant, a modality registry, or any other framework-level switch.

## Training vs. inference — "no input" semantics

The two runtimes treat a missing optional input differently — by design. This is the **only** asymmetry between them and it's the source of most bugs when porting code:

| Phase | What happens when `module.forward(foo=None, ...)` is called | Why |
|---|---|---|
| **Training (FSDP)** | The trainer fills missing kwargs from `module.dummy_inputs(...)` *before* dispatch. The `if foo is None: return {}` short-circuit is **never reached** during training. | Every active node MUST forward on every micro-batch or DP/SP all-reduce hangs. Dummy zero tensors keep the FSDP graph aligned across DP ranks; backbone scatter on a placeholder mask matched zero times is a no-op (gradients are zero but the graph traverses). |
| **Inference (no FSDP)** | The model `return {}` immediately. The `GenerationGraph` then **permissively skips** any outgoing edge whose `output:` key isn't in ctx — the destination still executes when its other inputs land. | No FSDP, no DP alignment. Skipping an empty forward saves real compute. Permissive routing handles text-only prompts in a multimodal FSM (e.g. `infer_understanding.yaml#prompt_to_text` with no `pixel_values`). |

What you must implement:

1. **Inference fast-skip** in every model's `forward` / `encode` / `decode`:
   ```python
   def forward(self, foo=None, **_):
       if foo is None:
           return {}        # inference fast path; trainer never sees this branch
       ...
   ```

2. **Per-module `dummy_inputs`** (optional, but required for any module whose primary input is loaded from raw_batch — e.g. vision encoders that consume `pixel_values`):
   ```python
   def dummy_inputs(self, *, batch_size, device, dtype):
       return {"foo": torch.zeros(batch_size, ..., device=device, dtype=dtype)}
   ```
   Modules whose inputs always come from upstream nodes (e.g. an AR backbone consuming `inputs_embeds` from sibling encoders) can leave the default `{}` — the upstream's own `dummy_inputs` populates the kwargs that eventually reach this module.

3. **FSDP grad-sync anchor** in any backbone that scatters dummy upstream outputs through `masked_scatter` (or any equivalent op that drops gradients when its mask is empty):
   ```python
   def pre_forward(self, inputs_embeds, und_image_embeds=None, ...):
       if und_image_embeds is not None and self.training:
           inputs_embeds = inputs_embeds + und_image_embeds.sum() * 0.0   # zero-valued autograd anchor
   ```
   Without this, `masked_scatter` on an all-False mask produces an output that's algebraically independent of `und_image_embeds` — autograd drops the gradient back to the upstream encoder, FSDP grad-reduce mismatches across DP ranks, and training silently diverges or hangs. The anchor adds a zero contribution that *is* part of the autograd graph, forcing a (zero) gradient through the upstream module so DP ranks stay synced.

4. **Permissive edge routing** is implemented in `GenerationGraph.step`; you don't need to do anything in YAML to opt in. An edge whose `output:` key isn't in ctx (or is `None`) silently skips. The destination's fan-in counter still decrements regardless of whether the route succeeded — so multi-source nodes (e.g. `janus_llama` in `prompt_to_text` consuming both `und_image_embeds` and `inputs_embeds`) execute exactly once after the LAST in-body fan-in edge has been processed.

Common pitfall: `if foo is None: return {}` works during inference but breaks training under FSDP unless the trainer fills dummies. The `dummy_inputs` method is the contract by which the trainer knows what zero tensors to fill — if you forget it, the trainer falls back to passing `None` and your `return {}` triggers the FSDP hang.

## SeedOmni V2 — 18 Hard Invariants

These rules are enforced by the framework. Violate them and you'll fight the design.

1. **`module` ≠ `node` ≠ `edge`.** Three layers: instance, call-site, data flow. Never conflate.
2. **`ModuleMixin` is a mixin.** Each module adds `XxxModuleMixin(ModuleMixin)` in `modulemixin.py` and inherits `(XxxModuleMixin, HFModelClass)` in `modeling.py`. Hooks (`forward` / `generate` / `pre_forward` / `post_forward` / `init_omni_state`) are **optional** except training-graph `forward`.
3. **`model_type` lives in `configuration_xxx.py`, not in YAML.** YAML `modules:` only declares `weights_path` (or `config_path`). HF `AutoConfig.from_pretrained(<weights_path>)` reads `model_type` from `<weights_path>/config.json`, then `MODULE_MIXIN_REGISTRY[model_type]` maps it to the combined class.
4. **Default node method**: training defaults to `forward`, inference defaults to `generate_step`. Specify `method:` only when both training and inference share a non-default method (e.g. `vae_encode` uses `encode` for both).
5. **Single forward, multi-role via kwargs**: a method may serve multiple roles by input-driven dispatch. Present-input → run, absent-input → return dummy or empty dict (never raise). Examples:
   - `vqvae.decode(hidden, gt_token_ids=...)` → `gen_loss` (train).
   - `vqvae.decode(hidden_only)` → sample `vq_token_id` + lookup `embed` (inference).
   - `vqvae.decode(token_id=...)` → just lookup `embed` (inference feedback).
6. **Backbone modules don't own vocab layers.** `embed_tokens` and `lm_head` belong to a separate `text_encoder` module (e.g. `janus_text_encoder`). The AR backbone consumes `inputs_embeds` and returns `hidden_states` only. Replace the HF model's internal `embed_tokens` with `nn.Identity()` after `from_config`. Filter `embed_tokens.*` and `lm_head.*` out of the backbone state dict in the split script.
7. **No orphan, no cycle.** Every node has at least one outgoing edge. If the node is a sink (e.g. produces only `*_loss`), add an explicit `{from: X, output: <loss>, to: end}` edge. Self-loops or any cycle are rejected by `TrainingGraph` construction (cycles in inference go in `generation_graph` only).
8. **`training_graph` / `generation_graph` only list `edges`** (subset of `edges:` pool). Active nodes are the union of `from` / `to` endpoints. The execution order is derived by topo sort — never write it manually.
9. **Loss collection is by `_loss` suffix (single key).** Each module's `forward` internally iterates all micro-batches; `post_forward` computes the token-level mean (sum_loss / sum_tokens) and emits a single scalar `<name>_loss`. `OmniModel.forward()` collects all `_loss`-suffixed scalars and sums them. **No `*_loss_token_count` companion key, no per-batch mean-then-mean** (which would batch-weight the loss when token counts vary). Loss is NOT routed via edges — `to: end` is just a topology marker.
10. **Dummy forward in training; permissive skip in inference.** Any node listed in `training_graph.edges` MUST run a forward pass on every micro-batch (FSDP backward hangs otherwise). The trainer fills missing kwargs from each module's `dummy_inputs(...)` before dispatch, so the `if x is None: return {}` short-circuit is never reached during training. Inference is the opposite: the model `return {}`, edge routing permissively skips, the destination still executes. See "Training vs. inference — no input semantics" for the full contract including the FSDP grad-sync anchor.
11. **One module instance, possibly many nodes.** Same module under multiple node names is the canonical pattern for "encode + decode" or "round-trip" use cases. Parameters are shared; gradients accumulate naturally.
12. **One module type, one instance per OmniModel.** RL scenarios with reference + actor models build two OmniModels; within a single OmniModel a model_type maps to exactly one `nn.Module`.
13. **SP slicing is a backbone-internal concern.** SP `pad_and_slice` happens in the backbone's `pre_forward`; SP `gather` happens in its `post_forward`. Pre-LLM nodes (e.g. `tok_encode`) and post-LLM nodes (e.g. `tok_decode`, `vae_decode`) operate on full-length tensors — they are SP-agnostic.
14. **Per-module checkpoint callback, per-module subfolder; tokenizer is per-module too.** Every module has its own `CheckpointCallback` writing to `<output_ckpt_dir>/<module_name>/{config.json, model.safetensors, [optional asset]}`. **All processors AND tokenizers stay per-module** (vision processor, image processor, audio feature extractor, AND the tokenizer → live inside their owning module's subfolder). The tokenizer specifically lives in the family's `text_encoder` module (`<output_ckpt_dir>/<family>_text_encoder/tokenizer/`). **OmniConfig has no top-level `tokenizer_path` field.** Pure DiT models without a text encoder have no tokenizer in their config at all.

15. **raw_batch = single field `conversation_list` at entry; module-driven processing thereafter.** raw_batch is a mutable dict whose only initial key is `conversation_list` (`list[list[dict]]` of `{type, value, role, loss_mask}` items). All derived fields (input_ids, image_embeds, attention_mask, labels, position_ids, hidden_states, ...) are produced by modules during forward and written back into raw_batch via return dicts. `multimodal_transform.py` only does basic IO + resize (path → tensor into item.value) — no chat templating, no tokenization, no image processing. The same dataset can feed any ug model — chat template / tokenize / image processor / boundary marker injection are all owned by the relevant module.

16. **Module forward = kwargs + Dict return; data flows 100% through raw_batch.** Each module's `forward(**kwargs) -> Dict[str, Any]` keeps the HF-compatible signature, but OmniModel **immediately writes the return dict back into raw_batch** (keyed by `edge.output`). Edges do **not** carry data to downstream modules — downstream reads the same `raw_batch` by its own input keys. Edges are dependency contracts and topology markers. Collator helpers and SP slicing are called **inside each module's `pre_forward`** for the fields it owns: ViT slices the image batch dim; text_encoder slices the sequence dim. There is no global collator final-step and no global SP slice node.

17. **Sampling state (incl. CFG) is per-request runtime ctx, not graph state.** `temperature` / `top_p` / `cfg_weight` and friends are passed via `OmniModel.generate(request, *, sampling: dict)` and written into `ctx` (which **is** the mutable raw_batch during inference), alongside `past_key_values`. Backbones consume them locally:
    - **CFG** is a backbone-private batch-axis mechanism. When `cfg_weight != 1.0`, the backbone calls its own `build_cfg_uncond_inputs` hook, replicates `inputs_embeds` / `attention_mask` / `position_ids` to 2× along the batch dim (even rows = cond, odd rows = uncond), runs forward, splits logits, merges via `uncond + cfg_weight * (cond - uncond)`, samples, and feeds the next token back as 2× along batch. FSM / graph see batch=1× (= `parallel_size`). When the FSM exits `image_vq` state, the backbone discards the 2× KV cache (consistent with #13 "KV cache is module-managed").
    - **`build_cfg_uncond_inputs`** is an optional `OmniModule` hook. Default raises `NotImplementedError` — backbones that don't implement it are not allowed to receive `cfg_weight != 1.0` (generate-time validation throws `ValueError`). Pad / boundary token ids needed for uncond construction are taken from the backbone's tokenizer reference (typically borrowed from sibling `text_encoder` module via `set_tokenizer`).
    - **`parallel_size`** is a backbone module's `PretrainedConfig` field (NOT a sampling parameter), and is **only supported in T2I mode** (`infer_t2i.yaml`). Interleave / understanding inference must run with `parallel_size = 1` because batch-dim expansion is incompatible with switching back to text states. The same `parallel_size` must be configured on the AR backbone and the VQ codec (e.g. `JanusLlama` and `JanusVQVAE`); `OmniModel` validates this at build time.

    **Status**: target contract; current code does not yet implement inference CFG or `parallel_size`. To be implemented feature-by-feature.

18. **Token concatenation / boundary markers / chat template all live in the relevant module.** text_encoder (base in `modules/base/text_encoder/`, family-specific in `modules/<family>/text_encoder/` inherits base) owns the family's chat template implementation, tokenizer asset, and produces `input_ids` / `labels` / `attention_mask` plus a **split conversation_list whose item.value is the embedding segment** (text segment = multi-token wte embeddings; modality / boundary segment = single-token placeholder embedding). ViT / VAE inject `boi`/`eoi` items into `conversation_list` during their forward. Audio encoders inject `audio_bos`/`audio_eos`. **There is no `chat_template` module, no top-level chat-template helper, no global image_pattern registry.** The backbone collapses the split conversation_list into flat `inputs_embeds` via segment-order-driven splice (replace each modality segment's placeholder embedding with N patch tokens from upstream ViT/VAE; concat). labels / attention_mask / position_ids are realigned at this final splice step.

## Phase 1 — Decide module shape and location

Pick the closest existing pattern; copy its file structure and signatures.

| Shape | Methods | Reference | Location |
|---|---|---|---|
| Pure encoder | `forward = encode` | `JanusSiglip` | `modules/<family>/` |
| Codec (encode + decode) | `encode`, `decode`, `forward = encode` | `JanusVQVAE` | `modules/<family>/` |
| Generic text encoder (chat-template + tokenize + wte + lm_head) | `encode`, `decode`, `forward = encode` | `TextEncoder` (base; family inherits) | `modules/base/text_encoder/` |
| AR backbone (no vocab) | `forward`, `generate_step`, `pre_forward`/`post_forward` for SP | `JanusLlama` | `modules/<family>/` |
| Diffusion | `forward = denoise_step`, `generate_step` (full denoise) | TBD | `modules/<family>/` |
| Generic adapter (e.g. dim projection) | `forward` | `MLPAdapter` | `modules/base/` |

Decide:

- **Generic vs. family-specific**: cross-family reuse → `modules/base/`. Family-specific → `modules/<family>/`.
- **Single method or multiple call-sites?** Multiple methods → expose `encode` + `decode` and let YAML reference them via `module: <name>, method: <m>`. Don't proliferate small modules just because they have two roles.
- **Tie / untie / freeze options?** Add to the config class so YAML can flip them without code changes.

## Phase 2 — Write the (configuration, modeling, [processing]) triplet

For a model named `foo` in family `bar`:

### `modules/<family>/configuration_bar_foo.py`

```python
from transformers import PretrainedConfig

class BarFooConfig(PretrainedConfig):
    model_type = "bar_foo"   # MUST be globally unique; matches MODULE_MIXIN_REGISTRY key

    def __init__(self, hidden_size=2048, vocab_size=151936, ..., **kwargs):
        # Store every knob BEFORE super().__init__
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        super().__init__(**kwargs)
```

### `modules/<family>/modeling_bar_foo.py`

```python
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_bar_foo import BarFooConfig
from .modulemixin import BarFooModuleMixin

class BarFoo(BarFooModuleMixin, PreTrainedModel):
    """HF weights / forward in modeling.py; graph hooks in modulemixin.py."""
    config_class = BarFooConfig

    def __init__(self, config: BarFooConfig):
        super().__init__(config)
        # build sub-modules from config
        self.post_init()

    # ── Call-site methods ───────────────────────────────────────────
    def forward(self, **kwargs) -> Dict[str, Any]:
        """Default training entry. For multi-method modules, alias to primary."""
        return self.encode(**kwargs)

    def encode(self, foo_input: Optional[torch.Tensor] = None, **_) -> Dict[str, Any]:
        if foo_input is None:
            # dummy path — see invariant 10
            dummy = torch.zeros(1, 1, self.config.hidden_size, device=self.device)
            return {"foo_embeds": dummy}
        return {"foo_embeds": self._encode(foo_input)}

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,                 # may be a list[Tensor] across micro-batches
        gt_token_ids: Optional[torch.Tensor] = None,
        token_id: Optional[torch.Tensor] = None,
        **_,
    ) -> Dict[str, Any]:
        """input-driven dispatch — see invariant 5.
        Loop over micro-batches inside the module: accumulate (sum_loss, total_tokens),
        then mean once at the end and emit a single scalar `foo_loss` (invariant 9).
        """
        out: Dict[str, Any] = {}
        if hidden_states is not None and gt_token_ids is not None:
            # training path — module owns the micro-batch loop
            sum_loss = hidden_states.new_zeros(())
            total_tok = 0
            for mb_hidden, mb_gt in zip(hidden_states, gt_token_ids):
                logits = self.head(mb_hidden)
                sum_loss = sum_loss + nn.functional.cross_entropy(logits, mb_gt, reduction="sum")
                total_tok += mb_gt.numel()
            out["foo_loss"] = sum_loss / max(total_tok, 1)            # scalar, token-level mean
            return out
        if hidden_states is not None:
            # inference sample path
            sampled = torch.multinomial(torch.softmax(self.head(hidden_states[:, -1:]), -1), 1)
            out["token_id"] = sampled.squeeze(-1)
            out["embed"] = self.codebook(sampled)
            return out
        if token_id is not None:
            out["embed"] = self.codebook(token_id)
            return out
        return out  # no inputs → empty (will be skipped downstream gracefully)

    # ── Optional hooks ──────────────────────────────────────────────
    def pre_forward(self, **kwargs):
        # SP slice / packing / dummy-fill / pack micro-batches into a list
        return kwargs

    def post_forward(self, outputs):
        # SP gather; if you didn't already mean inside `forward`, do it here:
        # outputs["foo_loss"] = sum_loss / total_tok  (scalar, see invariant 9)
        return outputs

    def generate_step(self, **kwargs):
        return self.forward(**kwargs)  # if forward already serves inference

    def get_parallel_plan(self):
        return None  # or a ParallelPlan with module-local fqn

    # ── Extraction helper (called by split script) ──────────────────
    @classmethod
    def from_<base_model>(cls, base) -> "BarFoo":
        # Extract weights from a loaded base HF model.
        # Must filter out keys that belong to sibling modules.
        ...
```

### Optional `modules/<family>/processing_bar_foo.py`

Only for modules that own a processor (vision processor, image processor, audio feature extractor). Modules that only own tokenizer (LLM backbones, text_encoder) skip this.

### Patterns checklist

- [ ] Every kwarg in `forward` / `encode` / `decode` has `Optional` typing and a sensible default.
- [ ] All methods return `dict[str, Any]`. Loss keys end in `_loss` and are **scalar tensors after token-level mean** (invariant 9).
- [ ] The module loops all micro-batches itself (one `forward` call processes the whole step), so the mean denominator is the global token count, not the batch count.
- [ ] `model_type` string is unique across the whole repo.
- [ ] No cross-module references (no holding a sibling OmniModule by attribute). Sibling data flows through edges only.
- [ ] No `inputs_embeds` fallback to internal `embed_tokens` if you've delegated wte to a sibling — replace with `nn.Identity()` and raise a clear error if absent.

## Phase 3 — Register the module

1. Add the model_type → class mapping in `veomni/models/seed_omni/modules/__init__.py`:

   ```python
   from .<family>.modeling_bar_foo import BarFoo
   ...
   MODULE_MIXIN_REGISTRY = {
       ...
       "bar_foo": BarFoo,
   }
   __all__ = [..., "BarFoo", ...]
   ```

2. Re-export from `veomni/models/seed_omni/__init__.py`:

   ```python
   from .modules import ..., BarFoo
   __all__ = [..., "BarFoo"]
   ```

3. If the family folder is new, also create `<family>/__init__.py` re-exporting the class.

## Phase 4 — Update / write the split-checkpoint script

If the new module's weights come from extracting a subset of an existing HuggingFace checkpoint, update `scripts/convert_model/split_<family>.py`:

1. Add a new output directory `<output_dir>/<sub_name>/`.
2. Walk the source state dict and write only the keys that belong to this module (filter by prefix, e.g. `embed_tokens.*` for `text_encoder`).
3. Build a `BarFooConfig(...)` from the source HF config; call `config.save_pretrained(<output_dir>/<sub_name>/)`. This automatically writes `model_type` into `config.json`.
4. **Remove those keys from any sibling module's state dict** to avoid duplication and strict-load failures (e.g. drop `embed_tokens.*` and `lm_head.*` from the AR backbone's state dict when extracting `text_encoder`).
5. Save the asset (if any) to the same subfolder via the appropriate `save_pretrained` (e.g. `image_processor.save_pretrained(<output_dir>/<sub_name>/)`).
6. Print the new `weights_path` in the final summary so the YAML can be updated by copy-paste.

The `text_encoder` extraction in `veomni/models/seed_omni/modules/janus/convert_model.py` (especially the `tie_word_embeddings`-aware lm_head save) is the reference.

## Phase 5 — Wire into the YAML graph

### YAML organisation: training file vs. inference file(s)

V2 splits configuration into a master **training YAML** and one or more **inference YAMLs** under `configs/seed_omni/<model>/`. The training YAML carries the master vocabulary (`modules`, `nodes`, `edges`, `training_graph` — including any inference-only nodes / edges); each inference YAML carries **only** a `generation_graph` block for one inference scenario. **There is no top-level `tokenizer_path`** — the tokenizer asset lives in the family's `text_encoder` module subfolder.

```
configs/seed_omni/janus_1.3b/
├── train_joint.yaml          # master vocabulary; the only file that defines modules / nodes / edges / training_graph
├── infer_interleave.yaml     # generation_graph only — T2T+T2I mid-stream image generation
├── infer_t2i.yaml            # generation_graph only — T2I-only (no preceding text)
└── infer_understanding.yaml  # generation_graph only — I2T / VQA (uni-directional)
```

Loading rules:

- **Training only**: load just `train_joint.yaml`.
- **Inference**: load `train.yaml` + the chosen `infer_<scenario>.yaml` via `OmniConfig._init(global_args, model_path, train_yaml_path, infer_yaml_path)` — train and infer `modules:` blocks are deep-merged per module name.

```python
cfg = OmniConfig.from_paths(
    model_path="/tmp/janus_1.3b_split",
    tokenizer_path="/tmp/janus_1.3b_split",
    train_yaml_path="configs/seed_omni/janus_1.3b/train_joint.yaml",
    infer_yaml_path="configs/seed_omni/janus_1.3b/infer_interleave.yaml",
)
```

Naming conventions:

| Pattern | Example | Purpose |
|---|---|---|
| `train_<scope>.yaml` | `train_joint.yaml`, `train_und_only.yaml` | Master file — defines vocabulary AND a specific `training_graph.edges` subset |
| `infer_<scenario>.yaml` | `infer_interleave.yaml`, `infer_t2i.yaml`, `infer_understanding.yaml` | Scenario-specific FSM — `generation_graph` only |

**Don't hide inference-only nodes / edges in the inference YAML.** They live in the master training YAML's `nodes` / `edges` pool (the inference YAML cannot partially extend pool dicts via the flat overlay, and having all nodes / edges in one place is a readability win). The training YAML simply omits them from `training_graph.edges`.

### What goes in each YAML

Editing `train_<scope>.yaml`:

1. **`modules:`** — add an entry with `weights_path` (or `config_path`) and any module-specific knobs (`freeze`, `gradient_checkpointing`, etc). **Don't write `model_type`** — it's read from the config.json at the path. The tokenizer (if needed) sits inside one of the modules' subfolders, typically `<family>_text_encoder/tokenizer/`.

   ```yaml
   modules:
     bar_foo: {weights_path: /path/to/bar_foo}
   ```

2. **`nodes:`** — add one entry per call-site, including inference-only ones (e.g. `emit_image_start`, `emit_image_end`):

   ```yaml
   nodes:
     foo_encode: {module: bar_foo, method: encode}   # explicit method
     foo_decode: {module: bar_foo, method: decode}
     # or just `{module: bar_foo}` for default forward/generate_step
   ```

3. **`edges:`** — add data routes, including inference-only ones (feedback edges for VQ loops, boundary-token bridge edges, body-terminal sinks like `ar_run_sink`). Conventions:
   - `from`/`to` reference **node** names (or `end` for sinks).
   - `output:` is the dict key returned by the source node.
   - `as:` is the kwarg name on the destination node.
   - Every node MUST have at least one outgoing edge. Sinks → `to: end`.
   - **Body terminal edges** (`<node>_run_sink: {from: <node>, to: end}`) are required for FSM bodies where a node executes but has no same-body consumer (e.g. `janus_llama` in `image_vq_start` updates its KV cache; its output is consumed by the *next* state). The runtime triggers `from_` execution via the body walk; no body sink → the node never runs.
   - Loss values DON'T need to flow via edges (collected by suffix), but you SHOULD still add a `to: end` sink edge to keep topology complete.

4. **`training_graph:`** — list the active edges only (excluding inference-only ones):

   ```yaml
   training_graph:
     edges: [..., foo_to_bar, foo_dec_to_end]
   ```

   Active nodes are derived from edge endpoints. Execution order is topo-sorted from edges.

5. **Comment the DAG layout** at the top of the YAML (ASCII diagram or short description) — this is the canonical reference for readers.

Editing `infer_<scenario>.yaml`:

6. **`generation_graph:`** — only this block. Each `state.body` is an ordered list of edge names from the master pool; `from` nodes are executed on first encounter (default method → `generate_step`; explicit method → direct dispatch); edges route ctx (permissively — see below).

7. Reference the master YAML for vocabulary; never redeclare `modules` / `nodes` / `edges` / `training_graph` in an `infer_*.yaml` (deep-merge would let it work, but it makes the file a fragile partial copy).

8. **`done` is framework-injected** — never declare a `done:` state block, never set `done_state`. `GenerationGraph` auto-injects a `done` state with empty body, no transitions, zero token budget; transitions whose `next_state: done` land on it. Authoring either knob is a hard error (raised at FSM build time). When the FSM enters `done`, `OmniModel.generate` walks every active module and calls its **`finalize(ctx, request)` hook**, merging non-empty returns into `ctx['finalize'][<module_name>]`. Override `finalize` on a module when you need to dump accumulated outputs — tokenizer-decode all generated `input_ids`, save accumulated VQ patches as images, write audio waveforms. The framework imposes no accumulation scheme: modules that need cross-step history append into a running list inside `ctx` during their own `generate_step`, then read it back in `finalize`.

9. **FSM transition conditions** — states carry **no iteration-count budget**. A state body runs once and then keeps iterating until one of its transitions fires; *modules* decide when a state ends. Two primary types for inference:

   | Type | Who decides | YAML shape | Example |
   |------|-------------|------------|---------|
   | `module_signal` | Module writes a one-shot flag in its return dict | `{ type: module_signal, key: K }` | VQ decoder emits `image_complete`; text decoder emits `start_image_gen` / `text_done` |
   | `default` | Catch-all fallback — matches unconditionally, **must be listed last** | `{ type: default }` | Single-pass bridge/leaf states (prompt encode, `<boi>` / `<eoi>` emit) or the else-branch after `module_signal` checks |

   **`module_signal`** — when a state's exit condition is "the module decided it's finished" (e.g. VQ decoder hit the last patch of a 24×24 grid; text decoder sampled `<begin_of_image>` or `</s>`), the state simply loops on a `module_signal` transition:

    ```yaml
    image_vq:
      body: [ar_to_vae_decode, vae_decode_to_ar]
      transitions:
        - { condition: { type: module_signal, key: image_complete }, next_state: image_vq_end }

    text_ar:
      body: [tok_encode, janus_llama, tok_decode]
      transitions:
        - { condition: { type: module_signal, key: start_image_gen }, next_state: image_vq_start }
        - { condition: { type: module_signal, key: text_done }, next_state: done }
    ```

    The module writes the signal key from inside its `generate_step` (or `decode`/`encode`/etc). The FSM never inspects raw token ids — resolve ids via `set_tokenizer()` on the module and emit `module_signal` keys instead (see `JanusTextEncoder.decode`).

   **`default`** — the switch-`default` catch-all: it matches unconditionally, so with first-match ordering it is the lowest-priority **fallback**. Use it for a deterministic single-pass state (run the body once, then advance) or as the else-branch after `module_signal` checks. It **must be the last** transition in a state — the FSM raises at build time otherwise, since any transition after a `default` is dead code:

    ```yaml
    image_vq_start:           # leaf — append <begin_of_image>, then advance
      body: [emit_start_sink]
      transitions:
        - { condition: { type: default }, next_state: image_vq }

    prompt_encode:            # branch on the sampled token, else fall through
      body: [vision_to_llm, tokenc_to_llm, llm_to_tokdec, tok_decode_sink]
      transitions:
        - { condition: { type: module_signal, key: text_done }, next_state: done }
        - { condition: { type: default }, next_state: text_ar }
    ```

    The framework checks `ctx[<key>]` after each step and **automatically pops the key** once a `module_signal` transition fires — so a one-shot signal never stale-fires the next state. Hard rules:

    - **Don't** try to express a fixed patch budget (e.g. 576) in YAML — there is no `token_length` knob anymore. The patch budget belongs to the *model* (real VQ decoders carry their own grid arithmetic); the FSM loops until the module raises `image_complete`.
    - **Naming**: pick a signal key that is unique across the FSM lifetime (`image_complete`, `start_image_gen`, `text_done`, …).
    - Only two condition types exist: `module_signal` and `default`. Build-time validation rejects unknown types, a missing `key` on `module_signal`, and a `default` that isn't listed last.

## Phase 6 — Validate

Run all four checks in order. Failing any one means stop and fix before continuing.

```bash
source .venv/bin/activate

# 1. Topo sort + FSM body resolution (no torch needed for this).  Pass the master
#    training YAML first; subsequent files deep-merge over it.
python scripts/visualize_omni_graph.py configs/seed_omni/<model>/train_<scope>.yaml --only train
python scripts/visualize_omni_graph.py \
    configs/seed_omni/<model>/train_<scope>.yaml \
    configs/seed_omni/<model>/infer_<scenario>.yaml --only fsm

# 2. Functional smoke test of the new module (training + inference paths, dummy fallback).
python -c "from veomni.models.seed_omni import BarFoo, BarFooConfig; ..."

# 3. Existing graph tests must still pass.
pytest tests/seed_omni/ -q

# 4. Lint + format gate.
make quality
```

Visual sanity:

- **Sources** include every node fed only by `raw_batch` (e.g. `tok_encode`, `vit_encode`, `vae_encode`).
- **Sinks** all flow to `end` and the modules they live on emit `*_loss` + `*_loss_token_count`.
- **Backbone (e.g. `janus_llama`) is `middle`** in joint training (consumes inputs_embeds + image embeds, produces hidden_states for both decoders).
- FSM `text_ar.body` flows `tok_enc → llama → tok_dec → input_ids loop`.
- FSM `image_vq.body` — sampling lives in `vae_decode` (hidden_states alone path), NOT in the AR backbone.

## Phase 7 — Update documentation

- **`design.md`**: only update if you introduced a new shape (e.g. a new abstract pattern beyond encode/decode). Family-specific modules that fit existing patterns don't need design-doc changes — design.md is shape-level.
- **Module docstring**: must describe (a) which YAML node forms invoke which methods, (b) what each method returns by key, (c) what kwargs each method dispatches on. The `JanusVQDecoder` docstring is the gold standard.
- **YAML config**: top-of-file comment block describes the DAG and FSM at a glance. Don't skip this.

## Common Pitfalls

- **Returning a tensor instead of a dict**: every OmniModule method must return `dict[str, Any]`. Tensors don't fan-in / fan-out.
- **Forgetting `forward` alias**: `OmniModule.forward` is the FSDP wrapper entrypoint. For multi-method modules, alias it to the primary call-site (usually `encode`).
- **Edge `output:` mismatch**: if the source node returns `{"embeds": ...}` but the edge says `output: embed`, the route silently drops to `None`. Always grep the source node's return dict for the exact key.
- **Cycle in `training_graph.edges`**: any feedback edge (e.g. `vae_decode_to_llama`) accidentally listed in `training_graph` will fail topo sort. Feedback edges belong only in `generation_graph` state bodies.
- **Missing `to: end` for a sink**: produces a graph orphan. The framework will reject it; add the sink edge even if loss is collected by suffix.
- **Backbone holding vocab layers**: AR LLM modules must NOT own `embed_tokens` / `lm_head` after migration. Replace internal `embed_tokens` with `nn.Identity()` after `from_config`, route `inputs_embeds` from a sibling `text_encoder` node. Filter `embed_tokens.*` and `lm_head.*` out of the backbone's state dict in the split script.
- **Per-batch mean then outer mean**: this batch-weights the loss when token counts vary, causing silent quality regressions. Invariant 9 — loop micro-batches inside the module, sum loss + sum tokens, divide once. Emit a single scalar `<name>_loss`. OmniModel just sums; never expect a `*_loss_token_count` companion key.
- **Forgetting to update `MODULE_MIXIN_REGISTRY`**: instantiation will fail with "unknown model_type" — register before running any config that references the new module.
- **Mixing `nodes` and `edges` fields in a single YAML entry**: the parser rejects entries with both `module:` and `from:` keys. Each entry belongs to exactly one pool.
- **Skipping `pre_forward`/`post_forward` for SP-capable modules**: backbones that use SP must slice in their `pre_forward` and gather in their `post_forward`. Sibling pre/post-LLM modules are SP-agnostic and stay full-length.
- **Writing `model_type` in YAML modules**: invariant 3 — YAML only declares paths. `model_type` lives in `configuration_xxx.py` and is read automatically.
- **Adding a top-level `tokenizer_path`**: V2 has no such field. Tokenizer is per-module (lives inside `text_encoder` module's subfolder). Pure DiT models without text encoder have no tokenizer in their config.
- **Returning `{}` for missing inputs instead of dummy forward**: invariant 10 — empty dict breaks FSDP backward; build dummy tensors and run the full path.
- **Looking up `raw_batch["input_ids"]` at OmniModel entry**: V2 entry-time raw_batch contains only `conversation_list`. `input_ids` is produced by `text_encoder.forward` later in the pipeline — modules consuming `input_ids` (e.g. backbone splice) must depend on edges that route `text_encoder`'s output, not raw_batch directly.
- **Backbone scanning `input_ids` for placeholder positions**: V2 backbone splice is segment-order driven — iterate the split conversation_list (output of text_encoder) and replace each modality segment's placeholder embedding with N patch tokens. Don't add `image_pos` / `und_image_pos` index fields (V1 leftover).
- **Putting chat templating / tokenization into `multimodal_transform.py`**: in V2, that file does basic IO + resize only. Chat template + tokenize live in `text_encoder.forward` (per family). Image processing lives in vision encoder modules. Putting them in the data layer breaks model-agnostic data.

## When to use this skill vs. siblings

- **`/seedomni-v2`** — anything inside `veomni/models/seed_omni/`: new OmniModule, modify YAML graph, update split scripts, add call-site methods.
- **`/veomni-new-model`** — adding a new HuggingFace model under `veomni/models/transformers/<model>/` (patchgen + parallel_plan + foundation model). NOT for SeedOmni V2.
- **`/veomni-migrate-transformers-v5`** — porting a transformers-v4 patch directory to v5 patchgen. NOT for SeedOmni V2.
- **`/veomni-new-op`** — adding a new kernel under `veomni/ops/`. Orthogonal.
- **`/veomni-develop`** — generic feature/refactoring outside SeedOmni V2.

If the task is _both_ "add a new HF model" AND "make it usable as a SeedOmni V2 module": run `/veomni-new-model` first to land the patchgen + parallel_plan + foundation model, then use this skill to wrap it as an OmniModule (multi-inherit) and wire it into the omni graph.
