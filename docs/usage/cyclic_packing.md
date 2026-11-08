# Cyclic packing for deterministic map-style SFT

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
prove this attestation or hash unread external media.

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

The buffer limit is a soft budget for decoded feature tensor storage and string
bytes. One lookahead, an oversized sample, emitted micro-batches, object
overhead, collator/hash temporaries, pinned copies and allocator caching are
outside it. It is not an RSS or accelerator-memory limit. Replay rereads the
current data pass, and full feature/output hashing and per-step rank agreement
have CPU and communication costs. Measure these costs with the actual data.
Same-topology accelerator recovery and throughput must be verified separately.
