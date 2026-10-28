# Qwen3 DPO training guide

DPO training with **Qwen3-0.6B** on the
[trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) dataset.

Config: [`configs/text/qwen3_dpo.yaml`](../../configs/text/qwen3_dpo.yaml)

---

## Step 1 — Prepare the dataset

```shell
python scripts/download_hf_data.py \
    --repo_id trl-lib/ultrafeedback_binarized \
    --local_dir ./ultrafeedback_binarized \
    --allow_patterns "*.parquet"
```

This downloads all train and test Parquet files from the Hub into `./ultrafeedback_binarized/`.

---

## Step 2 — Prepare the model

```shell
python scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-0.6B \
    --local_dir .
```

The script appends the model name to `--local_dir`, saving to `./Qwen3-0.6B`.

---

## Step 3 — Start DPO training

Before launching, audit the downloaded tokenizer's backend, special-token
postprocessor, and native chat template to ensure they produce unpadded single
sequences for this dataset. Compute its fingerprint as shown in
[explicit segment provenance](#experimental-explicit-segment-provenance) and
set `dpo_config.audited_unpadded_tokenizer_sha256` in the YAML. The example
intentionally leaves it unset and fails closed until that audit is performed.

```shell
bash train.sh tasks/train_text_dpo.py configs/text/qwen3_dpo.yaml
```

Key config values (full DPO argument reference: [arguments.md — DPOConfig](../usage/arguments.md#dpoconfig)):

| Parameter | Value |
|---|---|
| `model.model_path` | `./Qwen3-0.6B` |
| `data.train_path` | `./ultrafeedback_binarized` |
| `data.max_seq_len` | `2048` |
| `train.global_batch_size` | `8` |
| `train.micro_batch_size` | `1` |
| `train.num_train_epochs` | `1` |
| `dpo_config.beta` | `0.1` |
| `dpo_config.loss_type` | `sigmoid` |
| frozen reference | copies `model`; custom `reference_model` config is not supported |
| `train.checkpoint.output_dir` | `Qwen3-0.6B-dpo-ultrafeedback` |
| `train.wandb.project` | `VeOmni` |
| `train.wandb.name` | `Qwen3-0.6B-dpo-ultrafeedback` |

---

## Step 4 — Monitor training

Training outputs (DPO loss, chosen/rejected rewards, reward accuracy, grad norm) are printed every
step and logged to Weights & Biases.

---

## Checkpoints

Checkpoints are saved under `train.checkpoint.output_dir` every `save_steps` steps.
With `save_hf_weights: true`, a HuggingFace-compatible checkpoint is also written:

```
Qwen3-0.6B-dpo-ultrafeedback/
└── checkpoints/
    └── global_step_200/
        ├── checkpoint_manifest.json ← written last; marks the step resumable
        ├── model/
        │   ├── ckpt/                ← DCP shards: weights
        │   ├── optimizer/           ← DCP shards: optimizer state
        │   └── lr_scheduler.pt
        ├── loader/rank_{R}.pt       ← dataloader cursor
        ├── extra_state/rank_{R}.pt  ← step, rng, meters
        └── hf_ckpt/                 ← HuggingFace safetensors (when save_hf_weights)
```

File-by-file contract: [Checkpoint layout](../usage/checkpoint.md).

Each optimizer step uses the global mean over preference pairs, including when
microbatches or data-parallel ranks contain different numbers of pairs. The
reported loss and reward metrics use the same pair weights. Sequence-parallel
ranks must have the same segment boundaries, and all data-parallel ranks must
execute the same number of microbatches for an optimizer step. The required
experimental provenance mode also checks sequence-parallel input identity.
Validate expert and dense gradients separately
on the target EP/HSDP runtime; their distributed reduction implementations
can differ. Callback-internal collective failures still rely on the launcher's
failure handling.
Legacy tail-padding inference can mistake a real final one-token segment for
padding, or treat pre-existing padding as another pair. DPO training requires
producer-provided provenance and fails before training when it is disabled.
For conversation pairs, the shared history is masked and only the divergent
assistant completion is supervised. Chosen and rejected must share their
history, templates must preserve that token prefix and honor the mask, and
left truncation must leave the same nonempty prompt context in both branches.
Samples that cannot satisfy these conditions are rejected before forward.
Plaintext pairs likewise require the tokenized prompt context to match in both
branches; tokenization that changes the shared context is rejected. Tool, user,
and other non-assistant messages after the branch point are never supervised.

Custom DPO trainers overriding `forward_backward_step` must explicitly declare
the `pair_normalization` keyword and apply it to the sum of per-pair losses before
backward. A catch-all `**kwargs` alone is insufficient because it can silently
discard the scale. An old override without that argument is rejected before model forward;
using its unweighted microbatch mean would silently change the objective.

The standard DPO transform carries chosen/rejected lengths before packing; the
collator verifies them and accounts for padding it adds. It does not infer that
the producer supplied unpadded values from an all-one attention mask. DPO
preflight validates integer token IDs against the policy model's configured text
vocabulary before running the model.

Checkpoints record the objective normalization and producer contract version.
Earlier trainer-state checkpoints cannot establish this source contract and are
rejected; start from model weights for a clean comparison or use an independently
validated migration. The legacy resume flags cannot bypass this gate.

## Experimental: explicit segment provenance

Per-source channel-loss metrics are optional. DPO training with
`train.channel_loss.enable: true` additionally requires the per-source observer
integration that accepts expanded DPO log-probabilities; without that
integration the trainer rejects the configuration before training.

`dpo_config.explicit_segment_provenance: true` is the default and is required
for DPO training. It retains the physical
chosen/rejected lengths produced by the built-in DPO transform. This distinguishes
a real one-token response from padding appended by the standard collator and
attributes packed source-token counts per original DPO example. It does not
infer or remove padding already present in input examples.
The DPO objective parameters are held fixed for a run and must match on every
rank; rank-local changes and step-begin callbacks that change them fail before
model forward. To change the objective, start a new run from weights.

This proof assumes the input producer honors its unpadded-output contract. The
built-in text templates declare that contract. A tokenizer's Python class alone
does not prove the behavior of its model-supplied backend or postprocessor. For
the exact `PreTrainedTokenizerFast` (`TokenizersBackend` in Transformers 5) and
`Qwen2Tokenizer`, `LlamaTokenizer`, and `GPT2Tokenizer` classes, audit the tokenizer backend and native chat template,
then set `dpo_config.audited_unpadded_tokenizer_sha256` to its fingerprint:

```python
from transformers.processing_utils import ProcessorMixin
from veomni.models.auto import build_processor
from veomni.utils.dpo_provenance import dpo_tokenizer_fingerprint

loaded = build_processor("<same-model.tokenizer_path>", **{})  # use the training processor_config
tokenizer = loaded.tokenizer if isinstance(loaded, ProcessorMixin) else loaded
print(dpo_tokenizer_fingerprint(tokenizer))
```

The fingerprint binds the class, serialized backend, special tokens, padding
settings, and native chat template. Transient backend padding/truncation state
that Transformers resets on each call is normalized before hashing; DPO uses
single-sequence calls without padding. It is an operator attestation, not an
automatic audit: do not set it until the actual tokenizer and template are
verified to emit unpadded single sequences. It is checked during transformation
and recorded in the checkpoint protocol, so a changed tokenizer is rejected.
The serialization and hashing add CPU preflight cost; measure it for the target
workload. Other tokenizer or template classes, including subclasses, must explicitly declare
`dpo_unpadded_output_contract = True` on the concrete class and provide a stable
`dpo_unpadded_output_contract_id` string (or zero-argument method). The identifier
must change when the producer's behavior or configuration changes, and the
producer must actually return unpadded sequences. The declaration is the producer's responsibility, not a
mathematical proof derived from tensor values; an opaque custom producer is
rejected. Already padded input must be unpadded by a separately validated
producer before entering this path.

The plaintext path uses tokenizer character
offsets to mask prompt-only tokens, including when BPE merges cross the boundary.
If offsets are unavailable, an exact token-prefix boundary is required.
When a fast tokenizer returns a zero-width token at the prompt boundary,
the boundary cannot be attributed safely; use untrimmed offsets or a
tokenizer that provides an unambiguous tokenization.
A plaintext pair with a non-empty prompt is rejected when its prompt leaves
either completion with no scoreable tokens after truncation; shorten the prompt
or increase `max_seq_len`. A whole one-token segment has no causal prediction
target and is rejected before training. A one-token completion with a preceding
prompt or BOS remains valid.

Exact DPO trainer-state resume requires unchanged batching, worker, and parallel
geometry. In-order multi-worker loading is required; worker-mode dynamic batching
and background prefetch combined with dynamic flush padding are not supported for
this recovery contract. Trainer-state recovery currently requires a mapping
dataset: the built-in iterable and multisource sources do not yet provide a
verified cursor. These sources can train but their trainer-state checkpoints
cannot be resumed through this path. Epoch and intra-epoch positions are stored
separately to handle an early end of a finite dataset. Duplicated padding
microbatches are rejected rather than counted as extra preference pairs.
Main-process dynamic batching rejects unexpected early exhaustion instead of
silently restarting the dataset and repeating samples.
The mapping distributed sampler may repeat individual samples to make shards
even when dataset size is not divisible by the data-parallel size; the pair mean
weights each presented pair, including those sampler repetitions. For strict
once-per-epoch sampling, use a dataset size divisible by that size.
Tensor and pipeline parallelism are currently rejected for this DPO path;
their cross-group failure coordination has not been validated.
The checkpoint protocol does not fingerprint dataset or reference-model
contents; pin and record those separately.
Use the option with the same value on every rank, using the built-in DPO transform and standard
`MainCollator`. Each step validates policy/reference SP membership and a small
fingerprint of each SP group's transformed tensors and source IDs before model
execution. Identical transformed inputs may originate from different raw
records; this check does not certify raw dataset identity or cursor alignment. Unproven or
inconsistent samples are rejected on all ranks before model forward; with
multisource loading, every example must have one ordered integer `ds_idx`.
Source names are checked against their IDs only when the loader supplies them.
Token and position tensors must use 32- or 64-bit integers, the packed attention
mask must contain only ones, and input/label token IDs must fit the policy model's
configured text vocabulary. Before packing, non-ignored labels must equal their
corresponding unshifted input IDs, and both chosen and rejected branches must
contain at least one supervised token. Multi-token branches must have a target
after the first position. A one-token branch can be represented as data, but
has no within-branch causal prediction target and is rejected for training.
Pre-padded attention masks containing zeros require a
separately validated packed unpadding implementation and are rejected here.
This experimental path accepts only the built-in text tensors, `cur_token_num`,
and configured source metadata on input features; unknown extra feature keys are
rejected. Source and bookkeeping metadata are never sent to the model in this
mode. Callbacks may remove these fields but may not add or mutate them.
The actual tokenizer and template identities, transform maximum sequence length,
and DPO objective parameters are recorded in checkpoints and rechecked before
any buffered loader data are restored. Each rank must have a loader cursor;
local cursor restoration failures are synchronized. A custom loader's restore
method must not contain its own unmatched collectives. This protocol does not hash reference
weights or dataset contents: an exact resume still requires an immutable external
contract for the reference model and input data. Feature tensors are
fingerprinted at production and checked again at collation;
the digest detects mutation but cannot prove a producer's contract is truthful.
Local packed tensors and channel-loss metadata are fingerprinted at preflight
to detect mutation after collation. Model inputs are checked again after
step-begin callbacks; callbacks should transform data before collation instead.
Input validation and ordinary rank-local callback exceptions after callbacks
return are coordinated before model execution. Failures inside a callback's own
collective still rely on the launcher's failure handler or timeout.
Use the standard trainer loop: direct `concatenated_forward` calls require a
microbatch already approved by both rank-agreement phases. The provenance path
uses four WORLD-level and one FSDP-group object collective per step; measure
their overhead and validate the target hardware before using the option for production. CPU/Gloo tests do not establish
Ascend HCCL behavior, full-model numerical equivalence, or checkpoint recovery.

Resume only with the same data-semantics version and provenance option recorded
in trainer state. An old or missing trainer state is rejected: plaintext prompt-masking semantics changed, so a legacy loader buffer can
mix old labels with newly transformed labels. Start a fresh run from model weights
without restoring optimizer, scheduler, or loader state for a consistent objective.
Unversioned legacy checkpoints cannot resume through this required provenance
path; use model weights for a fresh run or an independently validated migration.
