# Qwen3-30B-A3B (MoE) (SeedOmni V2)

End-to-end recipe for training and inferring **Qwen3-30B-A3B** (a 30B-total /
3B-active Mixture-of-Experts model: 128 experts, top-8) as a SeedOmni V2 graph
model. It reuses the dense Qwen3 split skeleton, swapping the backbone for the
MoE one and adding **Expert Parallel (EP)** on the experts.

| Module | Holds | Role |
|--------|-------|------|
| `qwen3_text_encoder` | `embed_tokens` (+ `lm_head` if untied) + tokenizer | token ↔ embedding, CE-loss head, ChatML template (MoE-agnostic — reused from dense Qwen3) |
| `qwen3_moe_llm` | MoE decoder backbone (fused experts; no embed / no head) | `inputs_embeds → hidden_states`; experts sharded over the `ep` group |

Graph: `qwen3_text_encoder.encode → qwen3_moe_llm → qwen3_text_encoder.decode`.

All paths below assume the upstream HuggingFace checkpoint lives at
`/mnt/hdfs/veomni/models/Qwen3-30B-A3B`. Adjust to your own storage.

Config dir: `configs/seed_omni/Qwen/qwen3_30b_a3b/`. As with dense Qwen3, both
training and inference take the **same** `base.yaml`; the data list is shared
from `configs/seed_omni/Qwen/qwen3_0.6b/data.yaml`.

| File | Role |
|------|------|
| `base.yaml` | Top-level omni launcher: model paths, top-level `accelerator`, data, train, `infer` block. |
| `modules_train.yaml` | Per-module training overrides — `qwen3_moe_llm` carries the `ep` extra-parallel block. |
| `graph_train.yaml` | Training DAG. |
| `modules_infer_eager.yaml` | Inference overrides — all eager (single-process). |
| `modules_infer_fsdp.yaml` | Inference overrides — distributed FSDP2 + EP (mirrors train). |
| `graph_infer.yaml` | Text chat generation graph (`infer.infer_graph.infer_text`). |
| `../qwen3_0.6b/data.yaml` | Weighted multisource data list (shared with dense Qwen3). |

---

## 1. Convert the checkpoint

The converter reads `model_type` (`qwen3_moe`) from the HF `config.json` and
dispatches to `modules/qwen3_moe/convert_model.py`, splitting the weights into
`qwen3_text_encoder/` (embeddings + tokenizer) and `qwen3_moe_llm/` (MoE backbone).

```bash
python scripts/convert_model.py \
  --model_path /mnt/hdfs/veomni/models/Qwen3-30B-A3B \
  --output_dir /mnt/hdfs/user_dir/omni_v2/ckpt/Qwen3-30B-A3B
```

Notes:
- Needs ample CPU RAM (loads the 30B model on CPU to split).
- The backbone subfolder is saved in HF-canonical **per-expert** layout
  (`experts.{j}.{gate,up,down}_proj`); both eager (HF `from_pretrained`) and the
  veomni FSDP loader fuse it to the v5 `experts.gate_up_proj` layout at load time
  (the omni `Qwen3MoeLlm` carries the per-expert→fused checkpoint converter).
- `output_dir` becomes `model.model_path` (training) / `infer.model_path` (inference).

---

## 2. Prepare data

Same weighted multisource mix as dense Qwen3 — the shared
`configs/seed_omni/Qwen/qwen3_0.6b/data.yaml`. Each `names` entry must match a
preprocessor key in `veomni/data/seed_omni/preprocess.py`
(`SEED_OMNI_PREPROCESSOR_REGISTRY`); `tulu-3-sft-mixture` maps each row's
`messages` list to a `[role, ("text", content)]` conversation.

```yaml
# configs/seed_omni/Qwen/qwen3_0.6b/data.yaml (shared)
sources:
  - /mnt/hdfs/veomni/datasets/tulu-3-sft-mixture/mini_data
names:
  - tulu-3-sft-mixture
schedule:
  - { schedule_type: const, weights: [1.0] }
level: token
stopping_strategy: all_exhausted
upstream_sharded: true
```

The on-disk row schema is documented in
[`docs/seed_omni/data_format.md`](../data_format.md).

---

## 3. Train

Expert Parallel is configured per-module in `modules_train.yaml`:

```yaml
# configs/seed_omni/Qwen/qwen3_30b_a3b/modules_train.yaml — qwen3_moe_llm
accelerator:
  ep_size: 4            # EP degree; must divide world size AND num_experts (128)
```

The backbone builds its **own** `ParallelState` (fsdp2 + ep) while the text
encoder reuses the global fsdp2 mesh; the per-module grad-clip
(`veomni_omni_module_clip_grad_norm`, called once per module-trainer) handles both
topologies. EP requires `world_size % ep == 0`, so the
shipped `ep=4` needs ≥ 4 GPUs (`train.sh` auto-detects the count and launches torchrun).

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_30b_a3b/base.yaml \
  --model.model_path /mnt/hdfs/user_dir/omni_v2/ckpt/Qwen3-30B-A3B
```

Key knobs:

- `--model.model_path` — split-checkpoint root from step 1.
- `accelerator.ep_size` (in `modules_train.yaml`, `qwen3_moe_llm`) — Expert Parallel degree; must divide both the world size and `num_experts` (128).
- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--data.max_seq_len` — packed sequence length.
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.wandb.enable false` — disable wandb for quick smoke runs.


```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_30b_a3b/base.yaml \
  --model.model_path /mnt/hdfs/user_dir/omni_v2/ckpt/Qwen3-30B-A3B \
  --train.max_steps 20 --train.global_batch_size 8 --train.micro_batch_size 1 \
  --data.max_seq_len 2048 --train.checkpoint.save_steps 10 --train.wandb.enable false
```

---

## 4. Resume

Each save writes per-module DCP shards plus a `trainer_state.pt` (global step,
dataloader position, RNG state) under `<output_dir>/checkpoints/global_step_N/`.
Resume by pointing `load_path` at that directory:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_30b_a3b/base.yaml \
  --model.model_path /mnt/hdfs/user_dir/omni_v2/ckpt/Qwen3-30B-A3B \
  --train.checkpoint.load_path outputs/qwen3_30b_a3b_omni_sft/checkpoints/global_step_500
```

---

## 5. Inference

Two paths, selected by `infer.modules` (default = eager in `base.yaml`):

**Eager — single process**: every module loads via
`from_pretrained(device_map='auto')` and the MoE runs the eager experts loop over
the full 128-expert weights. No torchrun / EP.

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3_30b_a3b/base.yaml \
  --infer.model_path /mnt/hdfs/user_dir/omni_v2/ckpt/Qwen3-30B-A3B \
  --infer.infer_type infer_text \
  --infer.prompt "Give me a short introduction to large language models." \
  --infer.output_dir qwen3moe_out \
  --infer.generation_kwargs.max_new_tokens 40
```

**Distributed FSDP2 + EP** (mirrors train): override
`infer.modules` to the fsdp file; `OmniInferencer` auto-detects the non-eager
modules, inits the process group, and runs each module's forward under its own
`ParallelState` (fused MoE takes the EP all-to-all path). Needs ≥ ep GPUs.

```bash
bash train.sh tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3_30b_a3b/base.yaml \
  --infer.model_path /mnt/hdfs/user_dir/omni_v2/ckpt/Qwen3-30B-A3B \
  --infer.modules configs/seed_omni/Qwen/qwen3_30b_a3b/modules_infer_fsdp.yaml \
  --infer.infer_type infer_text \
  --infer.prompt "Give me a short introduction to large language models." \
  --infer.output_dir qwen3moe_out \
  --infer.generation_kwargs.max_new_tokens 40
```

To infer from a **trained** checkpoint, assemble a flat root from the per-module
`hf_ckpt/` subfolders (same pattern as Janus / dense Qwen3):

```bash
STEP=outputs/qwen3_30b_a3b_omni_sft/checkpoints/global_step_20
ASM=outputs/qwen3_30b_a3b_omni_sft/infer_ckpt/global_step_20
mkdir -p "$ASM"
for m in qwen3_text_encoder qwen3_moe_llm; do
  ln -sfn "$(realpath "$STEP/$m/hf_ckpt")" "$ASM/$m"
done
# then: --infer.model_path "$ASM"
```

---

## 6. MoE / Expert Parallel notes

This is the first omni backbone using Expert Parallel + a fused MoE kernel, which
surfaced (and fixed) a few omni-specific interactions. Full design notes:
[`docs/seed_omni/omni_v2_per_module_parallel.md`](../omni_v2_per_module_parallel.md) §15.

- **Express EP via `ep_size`.** `AcceleratorConfig.__post_init__` appends `ep_size`
  as the `ep` extra-parallel dim (any duplicate `ep` from per-module re-instantiation
  is collapsed by `_dedup_extra_parallel`), leaving a single `ep` dim of size `ep_size`.
  Equivalent to writing `extra_parallel_sizes: [N] / extra_parallel_names: ["ep"]`
  directly, but don't set both or you get a doubled `ep` dim. (The non-`ep` extra-parallel
  dims such as the text encoder's `emb` still use the explicit `extra_parallel_*` lists,
  since `__post_init__` only appends `ep`.)
- **Fused MoE kernel** is bound because `qwen3_moe/llm/modeling.py` re-exports the
  patched module's OpSlots into the omni module namespace (so
  `build_foundation_model`'s `_bind_veomni_ops` finds them); the EP-aware kernel
  reads `get_parallel_state().ep_group`, resolved per-node via `_module_scope`.
- **Gradient checkpointing** recompute is re-scoped to the module's `ParallelState`
  (`OmniModuleTrainer._scope_recompute_to_parallel_state`) so EP stays enabled
  during backward recompute.
- **FSDP inference** routes the backbone forward through `self(...)` (not
  `self.forward(...)`) so the FSDP root `__call__` runs lazy_init + unshards
  root-owned params.
- **Checkpoint load**: the split is per-expert; eager (HF) fuses natively, the
  veomni FSDP loader fuses via the converter attached to `Qwen3MoeLlm`. The
  stacked `*-merge` checkpoint format is **not** loadable (neither HF nor the
  veomni converter handles it) — use the standard per-expert checkpoint.
