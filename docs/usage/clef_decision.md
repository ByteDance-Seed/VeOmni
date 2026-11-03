# CLEF decision training

VeOmni supports CLEF with a dense Qwen3.5 backbone, the released
`JointSchemaHead`, and the pinned `LocalLLaMA/typed-decisions` dataset.

## Run

Launch one process per GPU. For example, to train on one GPU:

```bash
uv run torchrun --standalone --nproc-per-node=<count> \
  tasks/train_decision.py configs/decision/clef_qwen3_5.yaml \
  --model_path /path/to/Cloudflare/clef-flash
```

Set `--nproc-per-node` to the number of devices used on the node. For Ascend
NPU, use the same command with `configs/decision/clef_qwen3_5_npu.yaml`.

`model.model_path` must contain the backbone checkpoint and these CLEF files:

- `joint_schema_model.py`
- `joint_head_config.json`
- `joint_head.safetensors`

The decision architecture and optional head overrides belong under `decision`:

```yaml
model:
  model_path: /path/to/Cloudflare/clef-flash
decision:
  decision_model: clef
  decision_model_config:
    head_path: /path/to/joint_head.safetensors
    head_config: /path/to/joint_head_config.json
```

`head_config` may also be an inline dictionary. `model.model_config` remains the
backbone config override.

## Data and batching

The adapter loads `LocalLLaMA/typed-decisions`. It converts `state`, `questions`,
and `gold` into typed requests and aligns target probabilities by option ID.
Structured states are serialized to stable JSON before CLEF encoding.

The training split is shuffled with `train.seed` and divided by
`data.validation_fraction`. `limit_train_cases` and `limit_eval_cases` restrict
local runs. The upstream test split is loaded only when both
`decision.mode: evaluate` and `decision.eval_split: test` are set.

CLEF encodes a complete request before batching. The collator can pack multiple
requests into one forward while preserving request boundaries and local
question spans. Fixed batching pads the final global batch with zero-weight
requests. Dynamic batching uses the shared token budget:

```yaml
train:
  dyn_bsz: true
  micro_batch_size: 1
  global_batch_size: 8
data:
  max_request_tokens: 16384
  max_seq_len: 16384
```

Choice shuffling is deterministic by seed and epoch when
`data.shuffle_choices: true`. Loader checkpoints restore the consumed cursor,
batching buffers, and augmentation epoch.

## Loss, evaluation, and checkpoints

Supported objectives are:

- `soft_ce`
- `smooth_brier`
- `distill_rps`

Each optimizer step is normalized by the global sum of effective question
weights. Single-option, unsupervised, and padding questions contribute zero
weight. Evaluation writes per-rank prediction JSONL files and aggregate
cross-entropy, KL, Brier, accuracy, and ordinal RPS metrics.

DCP checkpoints provide exact training resume. Hugging Face exports contain the
backbone, `joint_head.safetensors`, `joint_head_config.json`,
`joint_schema_model.py`, and `decision_config.json`. CLEF scalar head parameters
are adapted internally for FSDP2 and exported in the official shape.

Current Decision training does not support MoE backbones, SP/CP, LoRA,
`torch.compile`, persistent data-loader workers, or EP-sharded split-head
loading.

## Validation

The core offline comparison uses the vendored, unmodified
`joint_schema_model.py` from `Cloudflare/clef-flash`, a deterministic tokenizer, and a
tiny random Qwen3.5 model:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/models/decision/test_clef_official_parity.py -q
```

It compares serialization, supervision, forward/backward results, all supported
objectives, optimizer updates, packed requests, and exported head compatibility.
Set `CLEF_REFERENCE_DIR` to a local snapshot of the pinned release to also use
the released tokenizer; the test uses `local_files_only=True`.

The accelerator test validates split-head loading and weighted FSDP2 gradients:

```bash
.venv/bin/python -m pytest tests/models/decision/test_decision_fsdp.py -q
```

See the [Decision model contract](../design/decision_models.md) for the shared
batch and model interfaces and [trainer loss reduction](trainer_loss_reduction.md)
for the step-wide normalization hooks.
