# Usage of dyn_bsz

## Background of VeOmni data packing

In LLM or VLM training, we usually adopt batch training, and the sequence lengths of different samples often vary, for example:

* S1 = [A B C D] (len 4)
* S2 = [E F] (len 2)
* S3 = [G H I] (len 3)
* S4 = [J] (len 1)

To form a batch, a common practice is to pad all sequences to the maximum length within the batch:

* S1_padding = [A B C D]
* S2_padding = [E F PAD PAD]
* S3_padding = [G H I PAD]
* S4_padding = [J PAD PAD PAD]

This approach introduces several serious problems:
1. **A large amount of meaningless computation**. Transformer attention and FFN layers still perform computations on PAD tokens.
2. **Unnecessary memory consumption**. Padding tokens also occupy GPU memory.

The core idea of VeOmni data packing is:

Before computation, remove padding tokens from the tensors and perform computation only on real tokens.

The specific procedure is as follows:

1. Within a batch, flatten all real tokens into a single contiguous sequence.
2. Record the token boundaries of each sample using cu_seq_lens_q and cu_seq_lens_k.
3. In Attention (using FlashAttention or FlexAttention) and FFN, compute only on the real tokens.

## Background of dyn_bsz

In distributed training, data packing may cause the sequence lengths to be different across ranks. Suppose there are two ranks: S1 and S2 are on rank0, and S3 and S4 are on rank1:

* Rank0: [A B C D E F]
* Rank1: [G H I J]

Rank1 will finish the forward and backward computation earlier than rank0, and will wait for rank0 to complete before performing gradient communication. This introduces unnecessary idle time. A common solution is to introduce a dynamic batchsize (dyn_bsz), where different numbers of samples are concatenated according to a target budget so that the total sequence length of a batch is around the target budget. This approach is very common in pretraining.

However, for SFT tasks, we care more about how many epochs are completed. In this case, it is recommended to disable dyn_bsz.

So the difference is:

**dyn_bsz ON: batch size varies to hit a token budget.**

**dyn_bsz OFF: batch size is fixed, but still uses position_ids packing (no padding).**

## Usage

### dyn_bsz = True (Recommended for Pretraining)

Goal: pack multiple samples into one long sequence and choose how many samples to fit per batch based on token budget.

Example packed batch (one micro‑batch):

* input_ids = [A B C D E F G H I J]
* position_ids = [0 1 2 3 0 1 0 1 2 0]
* (No padding added.)

dyn_bsz decides how many samples to pack so total tokens ~ target budget. If your target is, say, 10 tokens, it packs all 4 here.

`train.dyn_bsz_count_mode` controls how that budget is measured:

* `total` (default): count all packed tokens via `attention_mask.sum()`. This matches the physical sequence budget and preserves legacy behavior.
* `effective`: count only tokens with `labels != IGNORE_INDEX`. This is useful for prompt-heavy SFT data where DP ranks can have similar packed lengths but very different amounts of loss-contributing tokens.

When `dyn_bsz_count_mode=effective`, sample selection targets the effective-token budget while also enforcing a hard physical-token cap of `ceil(micro_batch_size * max_seq_len * train.dyn_bsz_physical_overflow_ratio)` for each micro batch. The ratio defaults to `1.5`, so effective-token batches can be physically longer than the legacy total-token budget while still bounding prompt-heavy examples with very few labels. Set the ratio closer to `1.0` for a tighter OOM guard, or higher for more effective-token balancing headroom. A single sample can still be longer than this cap by itself, so keep preprocessing/truncation aligned with `data.max_seq_len`.

### dyn_bsz = False (Recommended for SFT)

Goal: still pack with position_ids, but batch size is fixed in number of samples, not by tokens. The num of samples in a batch is determined by the `micro_batch_size`.

Example fixed batch size = 2 samples:

* Batch 1: S1 + S2
* input_ids = [A B C D E F]
* position_ids = [0 1 2 3 0 1]
* Batch 2: S3 + S4
* input_ids = [G H I J]
* position_ids = [0 1 2 0]

## Padding packed inputs (pad_packed_input)
`pad_to_length` can pad the packed sequence to a fixed length. This is useful to avoid uneven lengths that can trigger kernel recompilation.
- Related GitHub issue: [#402](https://github.com/ByteDance-Seed/VeOmni/issues/402)
`pad_to_length` is useful only when `dyn_bsz` = True, all the packed sequences in a batch will be padded to the `max_seq_len`.
It is also required when `model.accelerator.torch_compile.enable=True`, because per-block
`torch.compile` needs stable packed shapes to avoid repeated recompilation. If
CUDA Graphs are explicitly enabled, stable shapes are also required for useful
graph replay.

Important details:

* Padding is applied only after packing; labels are padded with `IGNORE_INDEX`.
* FlashAttention kwargs (`cu_seq_lens_q/k` and `max_length_q/k`) are recomputed after padding so they match
  the padded length and avoid numerical instability.

Examples (data level):

1) SFT-style packing with fixed-length padding:

* Packed tokens (two samples): input_ids = [A B C D E F], position_ids = [0 1 2 3 0 1]
* cu_seqlens (from position_ids): [0 4 6]
* With `pad_packed_to_length=8`:
  * input_ids = [A B C D E F 0 0]
  * position_ids = [0 1 2 3 0 1 0 1]
  * cu_seqlens (padded): [0 4 6 8]
  * labels padded with `IGNORE_INDEX` for the last 2 positions

2) dyn_bsz packing with fixed-length padding:

* Packed tokens (three samples): input_ids = [A B C D E F G], position_ids = [0 1 2 0 1 0 1]
* cu_seqlens (from position_ids): [0 3 5 7]
* With `pad_packed_to_length=10`:
  * input_ids = [A B C D E F G 0 0 0]
  * position_ids = [0 1 2 0 1 0 1 0 1 2]
  * cu_seqlens (padded): [0 3 5 7 10]
  * labels padded with `IGNORE_INDEX` for the last 3 positions

3) Inferred pad length (micro_batch_size * max_seq_len):

* micro_batch_size=2, max_seq_len=4 -> pad_packed_to_length=8
* Packed tokens (two samples): input_ids = [A B C D E], position_ids = [0 1 2 3 0]
* cu_seqlens (from position_ids): [0 4 5]
* After padding:
  * input_ids = [A B C D E 0 0 0]
  * position_ids = [0 1 2 3 0 1 2 3]
  * cu_seqlens (padded): [0 4 5 8]

## Cyclic packing for deterministic map-style SFT

`cyclic_packed` is an opt-in loader for a finite, immutable map-style SFT
dataset. It greedily packs samples up to `max_seq_len`, emits up to the global
batch size in micro-batches per optimizer step, and ends a short step at every
dataset boundary. A subsequent step starts a new pass in the same source order.
Each source row must deterministically produce exactly one feature; plaintext
rows that the standard transform splits into multiple features are rejected.
If a step has no supervised tokens with positive weight across the SP group,
it fails rather than applying an optimizer update or joining the next pass.
Trainer epochs and data passes are independent. `data.train_size` determines
the scheduler's nominal capacity-based step estimate, not the number of tokens
actually consumed: short dataset-tail steps can contain far fewer tokens than
`global_batch_size * max_seq_len`. Set the step schedule with that distinction
in mind and measure real consumed tokens separately.

The loader commits its data cursor after the optimizer and scheduler complete,
before checkpoint callbacks. Synchronous checkpoints save a small cursor and
hashes rather than decoded image tensors. Recovery replays the current pass,
checks the previously emitted packing and collator outputs, and reconstructs
the pending buffer. Resume is rejected if the saved identity or position does
not match. A checkpoint from a different loader, including a legacy custom
cyclic packer, is not automatically migrated.

The caller must pin an immutable snapshot and provide a SHA-256 input contract
covering all source data, external media contents, tokenizer, chat template,
preprocessing and collator versions and settings. Both transforms and the
metadata hook must be deterministic and free of hidden state or independent
random generators. The framework checks its dataset fingerprint, collator
identity, buffered contents and process-global RNGs, but cannot independently
prove this attestation or hash unread external media. All SP ranks must use the
same source snapshot and deterministic preprocessing. Their declared input
identity is compared at startup and restore; sample content is not compared
between ranks on every step. An incorrect attestation can silently change the
training data, and inconsistent execution paths can also cause a distributed
hang even if the checked sequence geometry happens to match.

The built-in integration is restricted to the standard VeOmni SFT/VLM trainer,
one sequence-parallel group with DP=1 and TP=1, a native mapping dataset and
`MainCollator`, `micro_batch_size=1`, in-order single-process data loading,
no shuffle, prefetch or drop-last, and synchronous, generation-aware
MultiOptimizer checkpoints. SP must be enabled. Multimodal hooks must use the
certified pure collator hook. The generation-aware checkpoint and pure hook are
provided by their respective upstream changes; without them the loader rejects
the configuration. DPO, streaming and custom training loops are not supported.

For example, after installing the required checkpoint and metadata changes:

```yaml
data:
  data_type: conversation
  datasets_type: mapping
  max_seq_len: 262144
  train_size: <explicit token budget for the scheduler>
  dyn_bsz_buffer_size: 32
  dataloader:
    type: cyclic_packed
    num_workers: 0
    persistent_workers: false
    in_order: true
    shuffle: false
    drop_last: false
    use_background_prefetcher: false
train:
  cyclic_input_contract: sha256:<immutable-input-contract-digest>
  cyclic_input_deterministic: true
  cyclic_max_buffer_bytes: 536870912
  micro_batch_size: 1
  global_batch_size: 64
  dyn_bsz: true
  dyn_bsz_runtime: main
  dyn_bsz_count_mode: total
  bsz_warmup_ratio: 0
  checkpoint:
    save_async: false
```

The buffer limit is a soft budget for logical tensor bytes and string lengths, so
packing does not depend on rank-local storage capacity or aliasing. Tensor views
can retain larger backing storage, while shared tensors may be counted more
than once. One lookahead, an oversized sample, emitted micro-batches, object
overhead, collator/hash temporaries, pinned copies and allocator caching are
outside it. It is not an RSS or accelerator-memory limit. Replay rereads the
current data pass, and incremental feature/output hashing for recovery has a
CPU cost. A per-step reduction coordinates data errors, supervised-token
presence, micro-batch count, label lengths and image/video presence; it does
not establish content equality. Optimizer failures are also coordinated before
committing the cursor. Measure these costs with the actual data. Checkpoints
from the earlier storage-byte budgeting algorithm are rejected because their
subsequent packing order may differ.
Same-topology accelerator recovery and throughput must be verified separately.
