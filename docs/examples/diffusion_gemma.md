# DiffusionGemma training guide

[DiffusionGemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
is an encoder-decoder **block-diffusion** MoE, not an autoregressive LM: the encoder
prefills the prompt (and optional images) and exports a KV cache; the decoder denoises
a fixed-length canvas that is bidirectional over `[encoder cache | canvas]`.

Launch with `tasks/train_diffusion_gemma.py`. Do not reuse `tasks/train_text.py` or
`tasks/train_vlm.py` — those collators and losses are for causal LMs.

Configs:

- Text-only: [`configs/text/diffusion_gemma.yaml`](../../configs/text/diffusion_gemma.yaml)
- Text+image: [`configs/multimodal/diffusion_gemma/diffusion_gemma_mm.yaml`](../../configs/multimodal/diffusion_gemma/diffusion_gemma_mm.yaml)

The default recipe uses FSDP2 + expert parallelism (`ep_size: 8`). Run on **8 GPUs**.

---

## Download dataset

### Text-only

Download the [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset.

```python
import pyarrow.parquet as pq
input_path = "tulu-3-sft-mixture/data/train-00000-of-00006.parquet"
output_path = "tulu-first2000.parquet"
# Read parquet file and extract the first 2000 rows
table = pq.read_table(input_path)
table_first_2000 = table.slice(0, 2000)
pq.write_table(table_first_2000, output_path)
```

The transform reads OpenAI-style `messages` (the parquet column name). Keep
`model.chat_template: tokenizer_generation_prompt` so the checkpoint's own
`<|turn>` / `<turn|>` markers are used and only the generation-prompt suffix is
supervised. Plain `tokenizer` would also supervise the `<|turn>model` header.
`chatml` is not in this vocabulary.

### Vision-Language

Download the [COCO2017](https://images.cocodataset.org/zips/train2017.zip) dataset
and the annotation JSON
[sharegpt4v_instruct_gpt4-vision_cap100k.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main).

Modify the ShareGPT4V JSON so each row has an `images` list. The multimodal
transform accepts ShareGPT `from`/`value` conversations, so pass
`--data.text_keys conversations` at launch:

```python
import json
with open('sharegpt4v_instruct_gpt4-vision_cap100k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered_data = []
for item in data:
    if item.get('image', '').startswith('coco'):
        new_item = item.copy()
        image_path = new_item.pop('image')
        new_item['images'] = [image_path]
        filtered_data.append(new_item)
with open('sharegpt4v_instruct_gpt4-vision_cap100k_coco.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
```

Images are encoder-only conditioning. The canvas the decoder denoises stays text.

---

## Download DiffusionGemma model

```shell
python3 scripts/download_hf_model.py \
    --repo_id google/diffusiongemma-26B-A4B-it \
    --local_dir .
```

The script appends the model name to `--local_dir`, saving to `./diffusiongemma-26B-A4B-it`.

> **Note.** VeOmni's runtime `CheckpointTensorConverter` folds the per-expert
> HF safetensor keys into VeOmni's fused expert layout at load time, so the
> stock HF checkpoint can be passed directly to training — no offline merge
> step is required. See
> `docs/transformers_v5/transformers_v5_moe_weight_loading.md` for details.

---

## Start training on GPU

Testing on 8×80GB GPUs. Export expandable CUDA allocator segments before launch
(recommended; the 26B encoder context otherwise fragments):

```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Text-only

```shell
bash train.sh tasks/train_diffusion_gemma.py configs/text/diffusion_gemma.yaml \
    --model.model_path ./diffusiongemma-26B-A4B-it \
    --data.train_path ./tulu-first2000.parquet \
    --model.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --model.accelerator.init_device meta
```

### Text+image

```shell
bash train.sh tasks/train_diffusion_gemma.py configs/multimodal/diffusion_gemma/diffusion_gemma_mm.yaml \
    --model.model_path ./diffusiongemma-26B-A4B-it \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --data.text_keys conversations \
    --model.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --model.accelerator.init_device meta
```

Key config values:

| Parameter | Value |
|---|---|
| `model.ops_implementation.attn_implementation` | `sdpa` |
| `model.ops_implementation.moe_implementation` | `fused_triton` |
| `model.model_config.text_config.attn_dg_prefix` | `true` (decoder Triton kernel) |
| `model.model_config.text_config.attn_ffpa` | `true` (optional full-attn speedup) |
| `model.chat_template` | `tokenizer_generation_prompt` (text-only; omit on the mm recipe) |
| `data.max_seq_len` | `4096` |
| `model.accelerator.ep_size` | `8` |
| `model.accelerator.ulysses_size` | `1` |
| `train.global_batch_size` | `8` |
| `train.micro_batch_size` | `1` |
| `model.accelerator.gradient_checkpointing.enable_reentrant` | `false` |

---

## Attention backends

Do **not** set `attn_implementation` to any flavour of FlashAttention. The decoder
attends bidirectionally over `[encoder cache | canvas]`, which flash cannot
represent — it treats any mask as padding over the query length. FA2/FA3 also cap
the head dimension at 256, while the five full-attention layers here are 512 wide.
The model raises rather than failing inside a kernel, and it refuses the whole
family: FA2, FA3, FA4, the `veomni_flash_attention_N_with_sp` names the recipe's
own backend rewrites those to, and the `kernels-community/flash-attn*` names
transformers rewrites them to when flash-attn is unavailable but `kernels` is.

`flex_attention` trains correctly now. It used to return zero `dK` and `dV` on
this model, which left the loss curve looking healthy while `k_proj` and `v_proj`
did not move; see `uncompiled_block_masks` in `modeling_diffusion_gemma.py`. At
the real decoder geometry, compiled flex sits closer to sdpa's math backend than
sdpa's own memory-efficient kernel does — bf16 5.87e-3 against 6.47e-3, by
`scripts/bench_diffusion_gemma_decoder_backends.py` — and its backward repeats
itself without needing `enable_full_determinism` to force sdpa onto math.

`sdpa` stays the default because flex is not yet cheaper here; at this geometry it
is more expensive. Per-step over four steps of the 26B recipe: flex 87/30/34/26 s,
sdpa on math 29/20/26/26 s, sdpa on its memory-efficient kernel 21/17/21/21 s. A
production run does not set `enable_full_determinism`, so the last row is what flex
has to beat, and sdpa is the one ahead: 21 s against flex's 26 s at step 4, and 80 s
against 177 s over all four. Peak memory was 59.41 GB on all three. The bench
that motivated flex — prompt 16384, forward 481 ms and 5029 MiB against sdpa's
1462 ms and 8951 MiB — was an isolated call, and nothing has reproduced that win
inside the recipe; flex was still recompiling at step 4, so 26 s may not be its
steady state.

Do not read a `grad_norm` gap on this recipe as a backend fault — the difference
between backends is far smaller than the one sdpa's own two kernels show against
each other. See constraint 31 (source numbering) in `.agents/knowledge/constraints.md`.

With the defaults:

- Encoder full-attention layers can use FFPA instead of materializing the score
  matrix (`attn_ffpa: true`). Install it with `uv pip install --no-deps ffpa-attn`
  (not in the lockfile; reinstall after `uv sync`). If it is missing, the model
  refuses to build instead of downgrading to sdpa, so that step time and peak memory
  are never reported under the wrong kernel's name. To use sdpa on purpose, set
  `attn_ffpa: false` — including when loading a checkpoint that was trained with the
  flag on, since it round-trips.
- The decoder packs the visible encoder cache plus the noisy canvas and runs the
  `dllm_attention` Triton kernel (`attn_dg_prefix: true`, CUDA only).

Ulysses sequence parallelism needs `attn_dg_prefix: true` — it is the decoder's
only shardable path — and `canvas_batching: flat_seq_pack`. See
[Sequence parallelism](#sequence-parallelism-ulysses_size) below.

---

## Denoising every canvas: `canvas_batching`

With `compute_all_canvas_df_loss: true` an example carries K noisy canvases against
one encoder cache, and `canvas_batching` decides how the decoder consumes them. The
three modes below compute the same thing; they differ in how many passes they issue
and in what attention has to build first. (There is a fourth, `flat_seq_pack`, which
changes what a row *holds* rather than how it is consumed — see the section after
this one.)

| Mode | Decoder passes | Attention KV per layer |
|---|---|---|
| `serial` | K, one per canvas | `K*P + K^2*C/2`, built K times |
| `flat_batch` | 1, over `B*K` rows | `K*P + K^2*C/2`, gathered once |
| `flat_seq` | 1, over `B` rows of `K*C` | `P + K*C`, read in place |

`flat_batch` collapses the per-canvas FSDP all-gathers and EP all-to-alls into one
launch, but attention still gathers each canvas's visible prefix into a private copy.
`flat_seq` packs the canvases onto the sequence and hands `dllm_hole_attention` each
canvas's visible run as an interval instead, so the encoder cache is read where it
already lies.

Measured on one A800 at the 26B shapes (prompt 2048, canvas 256, bf16,
forward+backward — `scripts/bench_diffusion_gemma_hole.py`), `flat_seq` against
`flat_batch`:

| Layer | K=4 | K=8 | K=16 |
|---|---|---|---|
| sliding (`head_dim` 256, 25 layers) | 3.1x time, 1.2x peak | 2.8x, 1.3x | 2.6x, 1.4x |
| global (`head_dim` 512, 5 layers) | 1.1x time, 2.4x peak | 1.1x, 2.9x | 1.2x, 3.6x |

The global layers run the same arithmetic either way, so what the interval saves
there is the copy. The sliding layers are the opposite: their prefixes are already
capped at `W-1`, and the time goes to building the gather rather than to attention.

`flat_seq` requires the untruncated encoder cache — training passes
`past_key_values=None` and gets it — and each canvas's visible keys must form one
contiguous run, which the standard layout always produces. Both are checked and
refused rather than worked around. Without `attn_dg_prefix` the mode still runs, on
sdpa through a dense hole mask, but then it pays for the holes the kernel exists to
skip. The kernel does not support attention dropout and raises on it. Under
sequence parallelism the kernel is the only decoder path there is, so the sdpa
fallback is refused instead.

The shipped recipes set `flat_seq`, paired with `attn_dg_prefix` so the interval reaches
the kernel. `DiffusionGemmaConfig`'s own default is still `serial`, which is the path
that has trained at 26B and the fallback if `flat_seq` OOMs or diverges there — the
numbers above are an attention-layer microbenchmark, not a measured step time at scale.

---

## Several examples in one row: `flat_seq_pack`

The three modes above pack *canvases*. `flat_seq_pack` also packs *examples*: the
collator concatenates their encoder rows, marks the boundaries by restarting
`position_ids`, and stacks every example's canvases on the canvas axis. The encoder
then runs varlen attention over the segments — `flash_attn_varlen_func` on the 25
sliding layers, `wide_varlen_attention` on the 5 global ones, whose `head_dim` of 512
is past flash's ceiling of 256.

What this buys is a micro-batch of several samples whose cost is the tokens it holds.
The padded path has to pad prompts to the longest in the batch and canvas counts to
the largest `K`, so ragged data pays for the batch's worst case; `micro_batch_size: 1`
avoids that padding by giving up batching altogether and accumulating instead.

`scripts/bench_flat_seq_pack.py` measures the whole model through the real collator on
one A800, three arms over the same samples and the same accumulated gradient: **A** the
shipped `flat_seq` at `micro_batch_size: 1`, **B** this mode with all N in one row, **D**
`flat_seq` at `micro_batch_size: N` padded to the batch's worst case. A padded batch pays
on two axes — the encoder row `L = P + K*C` to the longest in the batch, and the canvas
stack to the largest `K` — so both ratios are quoted.

| padding the batch would pay | **B** packed, vs A | **D** padded, vs A | **B vs D** |
|---|---|---|---|
| 1.95x encoder row, 1.00x canvas | 2.4x faster | 1.4x faster | **1.69x** |
| 1.80x encoder row, 1.00x canvas | 2.9x faster | 1.7x faster | **1.73x** |
| 1.25x encoder row, 1.74x canvas | 2.6x faster | 1.8x faster | **1.43x** |
| nothing to pad | 3.0x faster | 2.8x faster | **1.10x** |

Read the table before choosing the mode, because the headline column is not packing's:
most of the win is simply having a micro-batch wider than one, which **D gets too** by
raising `micro_batch_size` with no mode change at all. Packing's own contribution is the
last column, and it rises with the padding — up to 1.7x on ragged rows. The 1.10x left
when nothing needs padding is *not* packing being a better shape: packing the encoder
means the varlen kernels, so B runs flash and wide varlen attention where A and D go
through sdpa against a materialized mask, and that gap is about the size of the kernel
difference measured on its own. So the case for this mode is ragged data, and length
bucketing is the competing answer.

It is also **not** a memory optimisation against the shipped recipe: a packed row holds
every sample's checkpointed activations at once where accumulation frees each before the
next, which measures at 0.96–1.79x A's transient allocation, i.e. more in three of the
four cases and never appreciably less. What packing does beat on memory is D, at
0.50–0.89x. So it is how you widen the micro-batch *without* the padded batch's memory,
not how you use less than today; note that D itself costs 1.9–3.1x A's transient, which is
what raising `micro_batch_size` alone would have to fit.

Doubling the depth from 6 to 12 layers moves two of these cases by under 1.5%, though
repeating two others at fixed depth moves them by up to 4.9% — so read it as the ratios
not collapsing with depth rather than as a measured invariance. The machine also carries
another job at about a third of its GPUs throughout.

The single-GPU setting cuts both ways. Production's per-micro-step parameter all-gather
would make A relatively worse, so B's lead is understated. Against that, production's
wider MoE GEMMs are better occupied at `micro_batch_size: 1` than this benchmark's toy
expert width, and the benchmark runs with `attn_ffpa` off although the recipe turns it on
— which only A and D could have benefited from, since the packed encoder bypasses the
attention-implementation dispatch entirely. Both of those overstate B's lead.

Unlike the other three modes this is not the same computation as `serial` on the same
collated batch. It is the same computation as running those examples one at a time,
which is what `tests/models/test_diffusion_gemma_packed_collator.py` pins end to end
through the real collator: bit-identical canvas logits at float64, both terms of
Eq. 13 summing, gradients agreeing to float32 reduction noise, and neither direction
of cross-sample leakage.

Two things it refuses rather than approximates:

- **Images.** The varlen kernels take `cu_seqlens` and nothing else, so there is
  nowhere to put an image block boundary and an image span would be attended as text.
  The trainer refuses the multimodal data types at startup.
- **A row it cannot read the boundaries of.** Packing needs `position_ids` with the
  restarts in them, an all-ones `attention_mask`, and an encoder row to derive the
  segments from. A missing one is an error, not a default: the default positions run
  straight across the row and describe one long sample, which nothing downstream can
  tell from a row that really is one sample.

The encoder AR term of Eq. 13 is measured per segment under this mode, from the same
boundaries the attention used. Three of its steps are otherwise per row — the weight
normalization, dropping position 0, and cutting the trailing EOS run — and leaving any
of them per row divides a sample's share by the number of samples packed with it,
which is a changed objective with no changed shape. `encoder_ar_row_weight`, if
supplied, must therefore arrive per sample.

The recipes leave this off. It is unproven at 26B scale, and the benchmark above is a
single-GPU measurement without FSDP or expert parallelism.

### A token budget on top: `dyn_bsz`

Packing is what makes `dyn_bsz: true` mean anything here, and the trainer refuses the
combination without it. Under any other mode the collator left-pads prompts to the
batch maximum, so a batch the buffer selected to hold `N` tokens reaches the model as
up to `micro_batch_size * max(L)` — the budget stops predicting the step, in the
direction that OOMs, and nothing in between would notice.

What a DiffusionGemma sample costs the budget is `L + K*C`, not `L`. The default
measure is the `attention_mask` sum, which is the encoder row `L = P + K*C` and misses
nothing about the encoder; what it misses is the *decoder*, which passes over the `K*C`
canvas tokens a second time against that cache. So a canvas token is paid for twice
and a prompt token once, and two samples with the same `L` and different `K` cost
differently while the default calls them equal.

Measured at production attention geometry (`scripts/bench_dg_step_cost.py`): holding
`L` fixed at 2048 and raising `K` from 1 to 7 moves the step time by **1.3–1.6x**, and
charging the canvas tokens twice is what flattens it. Least squares puts a canvas token
higher — 2.6–2.7x a prompt token, since the decoder's attention over the cache grows
faster than its token count — but 2x is where the spread of time-per-budget-unit is
smallest at the production expert count, and it needs no fitted constant. Memory is a
separate question that the default measure already answers: across the same sweep the
transient allocation within a step moved 4%. The benchmark runs at reduced depth on
sdpa rather than the recipe's FFPA, so read the ratios and not the absolutes;
self-conditioning is the one term you would expect to matter here, being a third
decoder pass, and sweeping it moved the fit by under 1%.

Two things have to be set by hand. First, `micro_batch_size` — the budget per
micro-batch is `micro_batch_size * max_seq_len`, and the recipe's `micro_batch_size: 1`
gives 4096 where a sample costs up to `4096 + 8*256 = 6144`. Under packing that factor
scales the budget instead of the number of rows, so raising it is the fix; the trainer
warns below twice the worst-case sample. What happens to a sample that does not fit at
all depends on `dyn_bsz_runtime`, in opposite directions: `main` passes it through, so
the budget stops bounding the row; `worker` **discards** it, and cannot be told not to,
so at `micro_batch_size: 1` every sample with canvases would be dropped and an epoch
could produce nothing. The trainer refuses that combination rather than emptying the
dataset quietly.

Second, `train_steps` is estimated as
`data.train_size / (global_batch_size * max_seq_len)`, so `train_size` now has to be
declared in the same unit the budget spends — the dataset's token count times its mean
`1 + K*C/L`, between 1x and 2x. Declared in plain tokens, the run spends its budget
faster than the estimate, stops early, and leaves the learning-rate schedule unfinished
with nothing in the logs to say so. The trainer warns and does not correct it: the
ratio is a property of the data.

---

## Sequence parallelism: `ulysses_size`

`ulysses_size > 1` needs `canvas_batching: flat_seq_pack` and `attn_dg_prefix: true`,
and is refused otherwise at model build. **EP stays on**: keep the recipe's
`ep_size: 8`. Expert modules `fully_shard` on `ep_fsdp` (whose members do not include
the SP peers), then `extra_parallel_gradient_divide_factor` overrides the FSDP divide
factor to `world_size`, which already folds the SP dimension in — the same path
Qwen3-MoE / DeepSeek-V3 use. The SP-only refusals are not a missing feature: SP
splits the sequence axis, the padded modes pad prompts to the batch maximum, and sdpa
needs its mask over the full length — the `[B, 1, L, L]` sliding-window mask cannot be
sharded and grows as `L²`. Packing is already the varlen path, and a varlen kernel
takes `cu_seqlens` instead of a mask, which is what makes SP possible at all. Enabling
it in a mode that cannot use it used to run: every rank computed the same batch for the
same gradient, at `sp_size` times the cost, with nothing in the logs.

What SP shards is the **encoder** row. The decoder's canvases stay replicated on every
rank; its attention gathers the sharded encoder cache back to full length through an
all-to-all, so a canvas sees the whole context it would have seen unsharded.

Measured on 2xA800 at sp=2, 6 layers, with

```bash
torchrun --standalone --nproc_per_node=2 scripts/bench_encoder_sequence_parallel.py \
    --layers 6 --lengths 2048 4096 8192 16384            # encoder-only column
torchrun --standalone --nproc_per_node=2 scripts/bench_encoder_sequence_parallel.py \
    --layers 6 --lengths 2048 4096 8192 16384 --whole-model
```

The script's defaults are `--layers 12` and lengths from 4096 up, so running it bare
gives a different table from this one:

| encoder tokens | encoder only | its memory | whole model | its memory |
|---|---|---|---|---|
| 2048 | 1.13x faster | 0.43x | **0.92x — slower** | 0.53x |
| 4096 | 1.36x | 0.45x | 1.07x | 0.51x |
| 8192 | 1.58x | 0.46x | 1.26x | 0.50x |
| 16384 | 1.63x | 0.49x | 1.37x | 0.50x |

(Memory is transient allocation within a step, at sp=2 against sp=1.)

Read the whole-model column before turning this on. The replicated decoder is why it
lags the encoder one — the `K*C` canvas tokens cost every rank what they cost one — and
at short lengths the collectives are not earned back at all. At the recipe's
`max_seq_len: 4096` the case for SP is the **halved memory**, not the 1.07x; the time
argument starts around 8k. Memory is the more dependable half, sitting near 0.5x at
every length because it is activations that halve rather than arithmetic. (Caveat on the
time column: that arm holds 1, 2, 4 and 4 samples across the four rows, so canvas work
is not constant down it. The 8192/16384 pair is the controlled one.)

Constraints worth knowing before choosing a topology:

- **`sp_size` is bounded by the smaller of the two KV head counts.** Production has
  8 sliding and only 2 global, so `ulysses_size > 2` walks the repeat path on every
  global layer and pays back some of the memory GQA was saving; past the repeated
  count `sp_size` must divide it.
- The recipe keeps `ep_size: 8` under SP. Expert grads divide by `world_size`
  (`extra_parallel_gradient_divide_factor`), so the SP peers do not need to sit in
  `ep_fsdp`.
- Cross-node SP puts the all-to-all on the slower fabric. The 0.3–0.4%-of-step figure
  measured here is single-node NVLink.
- The decoder shards only the `attn_dg_prefix` + `flat_seq_pack` path. Others are
  refused rather than approximated, because their shapes stay valid while holding one
  rank's slice of the context — they would train, on a fraction of the conditioning.

---

## Generation

Block-autoregressive sampling against a checkpoint (HF weights or a training
`save_hf_weights` dump):

```shell
python -m scripts.generate_diffusion_gemma ./diffusiongemma-26B-A4B-it \
    --prompt "What is the capital of France? Answer in one sentence."
```

Optional image conditioning:

```shell
python -m scripts.generate_diffusion_gemma ./diffusiongemma-26B-A4B-it \
    --prompt "Describe this image." \
    --image cat.jpg
```
