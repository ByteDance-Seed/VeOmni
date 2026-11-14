# Qwen3-MoE (SeedOmni)

End-to-end recipe for training and inferring a **Qwen3 Mixture-of-Experts** model
as a SeedOmni graph model. It reuses the dense Qwen3 split skeleton, swapping the
backbone for the MoE one and adding **Expert Parallel (EP)** on the experts.

The recipe is not tied to one checkpoint: the config dir is `Qwen/moe/` and the
parameter count comes from whichever Qwen3-MoE checkpoint `model_path` names —
Qwen3-MoE (128 experts, top-8) is only the example used below.

| Module | Holds | Role |
|--------|-------|------|
| `qwen3_text_encoder` | `embed_tokens` (+ `lm_head` if untied) + tokenizer | token ↔ embedding, CE-loss head, ChatML template (MoE-agnostic — reused from dense Qwen3) |
| `qwen3_moe_llm` | MoE decoder backbone (fused experts; no embed / no head) | `inputs_embeds → hidden_states`; experts sharded over the `ep` group |

Graph: `qwen3_text_encoder.encode → qwen3_moe_llm → qwen3_text_encoder.decode`.

`model_path` and the `data.yaml` sources are placeholders: point them at your own
storage. The commands below use the upstream checkpoint root `Qwen3-MoE` and, for
the offline split, `Qwen3-MoE-split` as stand-ins — substitute the checkpoint you
actually have.

Config dir: `configs/seed_omni/Qwen/moe/`. As with dense Qwen3, both
training and inference take the **same** `train/base.yaml`; the data list is its
own copy of the dense-Qwen3 mixture at `train/data.yaml`.

| File | Role |
|------|------|
| `train/base.yaml` | Top-level omni launcher: model paths, top-level `accelerator`, data, train, `infer` block. |
| `train/modules_train.yaml` | Per-module training overrides — `qwen3_moe_llm` carries the `ep` extra-parallel block. |
| `train/graph_train.yaml` | Training DAG. |
| `infer/modules_infer_eager.yaml` | Inference overrides — all eager (single-process). |
| `infer/modules_infer_fsdp.yaml` | Inference overrides — distributed FSDP2 + EP (mirrors train). |
| `train/graph_infer.yaml` | Text chat generation graph (`model.model_config.infer_graph.infer_text`). |
| `train/data.yaml` | Weighted multisource data list — a local copy of the dense-Qwen3 mixture. |

---

## 1. Checkpoint

Qwen3-MoE registers an HF layout (`modules/qwen3_moe/hf_layout.py`), so
`model.model_path` can be the **upstream checkpoint itself** — no offline split
needed. The layout routes `model.embed_tokens.*` (and `lm_head.*`) plus the
tokenizer to `qwen3_text_encoder`, and the rest of `model.*` to `qwen3_moe_llm`;
the graphs default to the YAML in this config dir. (See
[Load an upstream checkpoint directly](../usage/training_and_inference.md#21-load-an-upstream-checkpoint-directly)
for the general contract — but read the two MoE caveats below before relying on
it for training.)

The upstream file stores experts **per expert**
(`model.layers.N.mlp.experts.{j}.{gate,up,down}_proj.weight`), while the module
holds them fused (`experts.gate_up_proj` / `experts.down_proj`). The backbone's
per-expert converter bridges that on load.

One thing does **not** work against an upstream root:

- **`ep_sharded_stream_load: false` is required.** The shipped
  `train/modules_train.yaml` enables streaming, and the HF-view validator rejects
  it (`...does not apply the qwen3_moe HF layout's key renames`) because streaming
  reads raw checkpoint slices and so cannot go through the layout's renaming. Point
  `--model.model_config.modules` at an overlay that sets
  `ep_sharded_stream_load: false` and `broadcast_model_weights_from_rank0: true`.

That combination is rejected at startup rather than mid-run — the argument parser
fails before a 30B checkpoint is loaded, so a misconfigured job does not waste the
setup.

**HF export does work against an upstream root.** The layout splits the fused
expert tensors back into per-expert keys on the way out
(`Qwen3MoeHFLayout.extract_module_save_state`), so a run trained from the upstream
checkpoint writes an `hf_ckpt/` in the source's own layout — the same shape
`bagel/hf_layout.py` uses for its fused QKV. Verified round-trip: the exported key
set and every value match the source, and a mutated `experts.gate_up_proj` comes
back out as the matching `gate_proj` / `up_proj` pair.

`base.yaml` still points at the split root, because that is the combination the
shipped streaming overlay wants: its experts are already fused, so streaming
applies to them directly.

The split is also what a cross-family composed model needs. It goes through the
same layout: `convert_checkpoint` finds no family converter for `qwen3_moe` and a
registered `OmniHFLayout`, so it calls
:func:`~veomni.models.seed_omni.utils.hf_layout.convert_with_hf_layout` — the same
key map a direct HF load uses, just materialised to disk. The result is
`qwen3_text_encoder/` (embeddings + tokenizer) and `qwen3_moe_llm/` (MoE backbone),
with the experts already fused.

```bash
python scripts/seed_omni/convert_model.py \
  --model_path Qwen3-MoE \
  --output_dir Qwen3-MoE-split
```

Either root works as `model.model_path`.

Notes:
- Needs ample CPU RAM (the whole checkpoint is loaded on CPU to split). The conversion host
  may have no GPU: pass `--extra attn_implementation=eager` if it lacks
  flash-attn, because building the modules instantiates the patched attention and
  the default backend is `veomni_flash_attention_2`. The choice is recorded in
  the split config, but the launcher's `model.ops_implementation` wins at run
  time — `base.yaml` sets `flash_attention_2`, so add
  `--model.ops_implementation.attn_implementation eager` to a training or
  inference command that must stay eager.

  ```bash
  python scripts/seed_omni/convert_model.py \
    --model_path Qwen3-MoE \
    --output_dir Qwen3-MoE-split \
    --extra attn_implementation=eager
  ```
- The split subfolder is saved in the v5 **fused** expert layout
  (`experts.gate_up_proj` / `experts.down_proj`), so the backbone's per-expert
  converter is a no-op when loading it (it fires only on an upstream per-expert
  root, as §1 describes). The fused layout is also what makes
  `ep_sharded_stream_load` valid: each rank slices its own rows on dim 0 instead
  of every rank reading the full expert set.
- `output_dir` becomes `model.model_path`, for training and inference alike.

---

## 2. Prepare data

Same weighted multisource mix as dense Qwen3, in this model's own
`configs/seed_omni/Qwen/moe/train/data.yaml`. Each `names` entry must match a
preprocessor key in `veomni/data/seed_omni/preprocess.py`
(`SEED_OMNI_PREPROCESSOR_REGISTRY`); `tulu-3-sft-mixture` maps each row's
`messages` list to a `[role, ("text", content)]` conversation. `sources` are
paths (`names` are keys, not paths) — the shipped value is a placeholder.

```yaml
# configs/seed_omni/Qwen/moe/train/data.yaml
sources:
  - tulu-3-sft-mixture        # placeholder: replace with a readable path / hdfs:// URL
names:
  - tulu-3-sft-mixture
schedule:
  - { schedule_type: const, weights: [1.0] }
level: token
stopping_strategy: all_exhausted
upstream_sharded: true
```

The on-disk row schema is documented in
[Data Format](../usage/data_format.md).

---

## 3. Train

Expert Parallel is configured per-module in `train/modules_train.yaml`:

```yaml
# configs/seed_omni/Qwen/moe/train/modules_train.yaml — qwen3_moe_llm
accelerator:
  extra_parallel_names: ["ep"]
  extra_parallel_sizes: [4]                    # EP degree; divides world size AND num_experts
  extra_parallel_placement_innermost: [false]
```

Spell the dimension out with the `extra_parallel_*` lists rather than `ep_size`.
A per-module entry is merged onto `asdict(ModelArguments)`, which already carries
`extra_parallel_names: ["ep"]`, and `AcceleratorConfig.__post_init__` only
appends `ep_size` when `"ep"` is absent — so `ep_size: 4` here is inert and the
run silently has no Expert Parallel. Write all three lists together: they are
paired by position, so overriding only `sizes` desynchronises them whenever the
launcher's global accelerator contributes another dim (e.g. a text encoder's
`emb`). See §6.

The backbone builds its **own** `ParallelState` (fsdp2 + ep) while the text
encoder reuses the global fsdp2 mesh; the per-module grad-clip
(`veomni_omni_module_clip_grad_norm`, called once per module-trainer) handles both
topologies. EP requires `world_size % ep == 0`, so the shipped `ep=4` needs at
least 4 GPUs (`train.sh` auto-detects the count and launches torchrun).

Batch size is the tighter limit. EP does not shrink the top-level `dp_size`, so
`dp_size == world_size` and argument parsing requires `global_batch_size` to be a
multiple of `micro_batch_size * dp_size`. With the shipped
`micro_batch_size: 1` / `global_batch_size: 8`, only 4 or 8 GPUs parse. At 16
GPUs the run stops with `` `global_batch_size` should be a multiple of 16 ``, so
raise it (e.g. `--train.global_batch_size 16`).

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/moe/train/base.yaml \
  --model.model_path Qwen3-MoE-split
```

Key knobs:

- `--model.model_path` — split-checkpoint root from step 1.
- `accelerator.extra_parallel_sizes` (in `train/modules_train.yaml`, `qwen3_moe_llm`) — Expert Parallel degree; must divide both the world size and the checkpoint's `num_experts` (128 for
  Qwen3-30B-A3B). Not `ep_size`: that field is inert on a per-module entry (§6).
- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--data.max_seq_len` — the per-micro-batch token budget dynamic batching sizes
  against. This config ships `dyn_bsz: false`, so the flag is inert here; nothing
  truncates samples to it (the SeedOmni preprocessors do not take it).
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.wandb.enable false` — disable wandb for quick smoke runs.


```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/moe/train/base.yaml \
  --model.model_path Qwen3-MoE-split \
  --train.max_steps 20 --train.global_batch_size 8 --train.micro_batch_size 1 \
  --data.max_seq_len 2048 --train.checkpoint.save_steps 10 --train.wandb.enable false
```

---

## 4. Resume

Each save writes per-module DCP shards plus the per-rank job state (global step,
dataloader position, RNG state) under `<output_dir>/checkpoints/global_step_N/`.
Resume by pointing `load_path` at that directory:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/moe/train/base.yaml \
  --model.model_path Qwen3-MoE-split \
  --train.checkpoint.load_path outputs/moe_omni_sft/checkpoints/global_step_500
```

---

## 5. Inference

Two paths, selected by the `--model.model_config.modules` file (with `base.yaml`'s
own, inference loads every module eager):

**Eager — single process**: every module loads via
`from_pretrained(device_map='auto')`, so FSDP2/EP are off and every expert's
weights sit on one device. The experts kernel is unchanged by this file
(`moe_implementation` is unset in `base.yaml`, so each host takes its own
default: `fused_triton` on CUDA/MLU, `fused_npu` on Ascend); pass
`--model.ops_implementation.moe_implementation eager` to run the plain PyTorch
loop instead. `infer/modules_infer_eager.yaml` only switches the *wrap* mode, not
the experts kernel. No torchrun / EP.

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/moe/train/base.yaml \
  --model.model_path Qwen3-MoE-split \
  --model.model_config.infer_type infer_text \
  --infer.prompt "Give me a short introduction to large language models." \
  --infer.output_dir qwen3moe_out \
  --infer.generation_kwargs.max_new_tokens 40
```

**Distributed FSDP2 + EP** (mirrors train): point
`--model.model_config.modules` at the fsdp file; `OmniInferencer` auto-detects the non-eager
modules, inits the process group, and runs each module's forward under its own
`ParallelState` (fused MoE takes the EP all-to-all path). Needs ≥ ep GPUs.

```bash
bash train.sh tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/moe/train/base.yaml \
  --model.model_path Qwen3-MoE-split \
  --model.model_config.modules configs/seed_omni/Qwen/moe/infer/modules_infer_fsdp.yaml \
  --model.model_config.infer_type infer_text \
  --infer.prompt "Give me a short introduction to large language models." \
  --infer.output_dir qwen3moe_out \
  --infer.generation_kwargs.max_new_tokens 40
```

To infer from a **trained** checkpoint, assemble a flat root from the per-module
`hf_ckpt/` subfolders (same pattern as dense Qwen3):

```bash
STEP=outputs/moe_omni_sft/checkpoints/global_step_20
ASM=outputs/moe_omni_sft/infer_ckpt/global_step_20
mkdir -p "$ASM"
for m in qwen3_text_encoder qwen3_moe_llm; do
  ln -sfn "$(realpath "$STEP/hf_ckpt/$m")" "$ASM/$m"
done
# then: --model.model_path "$ASM"
```

---

## 6. MoE / Expert Parallel notes

This is the first omni backbone using Expert Parallel + a fused MoE kernel, which
surfaced (and fixed) a few omni-specific interactions:

- **Router load-balancing loss.** With `output_router_logits: true` on the module
  config, the backbone returns `router_aux_loss_coef * load_balancing_loss` as its
  own `_loss`. The graph sums every node's `_loss`, so this *adds to* the decode
  head's cross-entropy rather than replacing it, and the trainer logs each
  module's term separately. Off by default (`Qwen3MoeConfig` ships
  `output_router_logits=False`), in which case the node contributes nothing.
- **LoRA on the fused experts.** `lora_modules: [gate_proj, up_proj, down_proj]`
  maps onto `experts.gate_up_proj` / `experts.down_proj` through the same
  `_convert_lora_targets_to_parameters` hook the foundation-model path registers
  on `Qwen3MoeModel`. SeedOmni builds the patched class directly rather than
  through `MODELING_REGISTRY`, so `Qwen3MoeLlm` carries its own copy with
  `language_model.`-prefixed patterns. Without it those names would resolve to
  nothing and the experts would silently get no adapters while attention did.
- **Express EP via the explicit `extra_parallel_*` lists, not `ep_size`.**
  `AcceleratorConfig.__post_init__` appends `ep_size` as the `ep` dimension only
  when `"ep"` is not already in `extra_parallel_names` — a guard that exists so
  re-instantiating a saved config (an `asdict` round-trip) does not add a second
  `ep` dim. A per-module entry is built from `asdict(ModelArguments)`, which
  already carries `["ep"]` / `[1]`, so that guard always fires and `ep_size` in a
  module YAML is inert: the effective dimension stays 1 and the run has no Expert
  Parallel while still looking configured. Write all three lists instead —
  `extra_parallel_names: ["ep"]`, `extra_parallel_sizes: [4]`,
  `extra_parallel_placement_innermost: [false]` — because they are paired by
  position: a module entry inherits the launcher accelerator's list, so replacing
  only `sizes` leaves a stale entry behind when the launcher adds another dim
  (e.g. a text encoder's `emb`) and the lengths stop matching. The top-level
  `model.accelerator` is built from the YAML directly, so `ep_size` does work
  there.
- **Fused MoE kernel** is bound as an instance-local `VeomniOp("moe_experts", ...)`
  on each `Qwen3MoeExperts` (one class below `Qwen3MoeSparseMoeBlock`, in the
  generated modeling) from that layer's `config.ops_config.moe_implementation`,
  which `PretrainedOmniModule` scoped to this module's own
  `OmniModuleConfig.ops_implementation`; the EP-aware kernel reads
  `get_parallel_state().ep_group`, resolved per-node via `module_context`.
- **Gradient checkpointing** recompute is re-scoped to the module's `ParallelState`
  (`ModuleRuntime._recompute_context`, carried by the recompute policy) so EP stays enabled
  during backward recompute.
- **Checkpoint load**: a split checkpoint stores fused experts, so every loader
  takes them whole (or, with `ep_sharded_stream_load`, slices this rank's rows on
  dim 0) and the attached per-expert converter is a no-op there. Pointing
  `model.model_path` at an upstream per-expert root instead goes through the HF
  layout, and the converter fuses in place (§1). The stacked `*-merge` checkpoint
  format is **not** loadable (neither HF nor the veomni converter handles it) —
  use the standard checkpoint.
