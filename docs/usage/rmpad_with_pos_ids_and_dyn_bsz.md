# Usage of rmpad_with_pos_ids and dyn_bsz

## Background of rmpad_with_pos_ids

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

The core idea of rmpad_with_pos_ids is:

Before computation, remove padding tokens from the tensors and perform computation only on real tokens.

The specific procedure is as follows:

1. Within a batch, flatten all real tokens into a single contiguous sequence.
2. Record the token boundaries of each sample using cu_seq_lens_q and cu_seq_lens_k.
3. In Attention (using FlashAttention or FlexAttention) and FFN, compute only on the real tokens.

## Background of dyn_bsz

In distributed training, rmpad_with_pos_ids may cause the sequence lengths to be different across ranks. Suppose there are two ranks: S1 and S2 are on rank0, and S3 and S4 are on rank1:

* Rank0: [A B C D E F]
* Rank1: [G H I J]

Rank1 will finish the forward and backward computation earlier than rank0, and will wait for rank0 to complete before performing gradient communication. This introduces unnecessary idle time. A common solution is to introduce a dynamic batchsize (dyn_bsz), where different numbers of samples are concatenated according to a target budget so that the total sequence length of a batch is around the target budget. This approach is very common in pretraining.

However, for SFT tasks, we care more about how many epochs are completed. In this case, it is recommended to disable dyn_bsz.

So the difference is:

**dyn_bsz ON: batch size varies to hit a token budget.**

**dyn_bsz OFF: batch size is fixed, but still uses position_ids packing (no padding) when rmpad_with_pos_ids.**

## Usage

### rmpad_with_pos_ids = True, dyn_bsz = ON (Recommended for Pretraining)

Goal: pack multiple samples into one long sequence and choose how many samples to fit per batch based on token budget.

Example packed batch (one micro‑batch):

* input_ids = [A B C D E F G H I J]
* position_ids = [0 1 2 3 0 1 0 1 2 0]
* (No padding added.)

dyn_bsz decides how many samples to pack so total tokens ~ target budget. If your target is, say, 10 tokens, it packs all 4 here.

### rmpad_with_pos_ids = True, dyn_bsz = OFF (Recommended for SFT)

Goal: still pack with position_ids, but batch size is fixed in number of samples, not by tokens.

Example fixed batch size = 2 samples:

* Batch 1: S1 + S2
* input_ids = [A B C D E F]
* position_ids = [0 1 2 3 0 1]
* Batch 2: S3 + S4
* input_ids = [G H I J]
* position_ids = [0 1 2 0]

## Padding packed inputs (pad_packed_input)

When `rmpad_with_pos_ids` is enabled, `pad_packed_input` can pad the packed sequence to a fixed length
(`pad_packed_to_length`). This is useful to avoid uneven lengths that can trigger kernel recompilation.
- Related GitHub issue: [#402](https://github.com/ByteDance-Seed/VeOmni/issues/402)

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
