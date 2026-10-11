# LLaDA family training guide

The LLaDA models are **masked-diffusion** LMs: a bidirectional mask predictor trained
to recover masked answer tokens in place. They train with the generic `DllmTrainer`
through `tasks/train_dllm.py`; do not use `tasks/train_text.py`, whose collator shifts
labels for next-token prediction.

| Model | `model_type` | Chat template | `[MASK]` id | Config |
|-------|--------------|---------------|-------------|--------|
| LLaDA-8B-Instruct | `llada` | `llada` | 126336 | [`configs/dllm/llada_8b.yaml`](../../configs/dllm/llada_8b.yaml) |
| LLaDA-MoE-7B-A1B-Instruct | `llada_moe` | `llada_moe` | 156895 | [`configs/dllm/llada_moe_7b.yaml`](../../configs/dllm/llada_moe_7b.yaml) |
| LLaDA-V | `llada_v` | `llada` | 126336 | [`configs/dllm/llada_v_8b.yaml`](../../configs/dllm/llada_v_8b.yaml) |
| LLaDA2.0-Uni (I2T) | `llada2_uni` | `llada2_uni` | 156895 | [`configs/dllm/llada2_uni_16b.yaml`](../../configs/dllm/llada2_uni_16b.yaml) |

The text models use `data_type: llada`, and `model.chat_template` is what differs
between them. LLaDA-V uses `data_type: llada_v`. LLaDA2.0-Uni I2T uses
`data_type: llada2_uni`. Image generation (`decoder/`, T2I) is not part of this path.

## Objective

`data_type: llada` applies the LLaDA forward process to each sample
(`veomni/data/data_transform.py::apply_dllm_masking`):

1. Draw one mask ratio `t ~ U(1e-3, 1)` per sample.
2. Replace each answer token with `[MASK]` with probability `t`. Prompt tokens are never masked.
3. Keep `labels` only at masked positions (no shift) and give them `loss_weight = 1 / (t * N)`,
   `N` being the sample's answer length.

The model returns the local `sum_i w_i * CE_i`, and `DllmTrainer` divides it by the
number of samples with an answer across the step and all data-parallel ranks, including
the ones whose draw masked nothing. The `[MASK]` id is read from the model config's
`mask_token_id`. Each sample's draw comes from a generator keyed on `train.seed`, the
epoch and the sample's tokens, so ranks do not share masks and a resume replays them.

The answer is supervised up to its own end-of-turn token. LLaDA's SFT recipe also pads
short answers with `|EOS|` and supervises that padding, which is what its fixed-length
sampler relies on to stop; this transform does not, so fine-tune data that is short
relative to the generation length can weaken that behaviour.

The chat templates are reimplemented so the answer span can be labelled, and match the
checkpoint's own `apply_chat_template` token for token. `get_jinja_template` hands the
checkpoint's template back unchanged, so a saved tokenizer keeps the original template.

## Packing

`MainCollator` packs samples into one row, with `position_ids` restarting at 0 at each
sample. Attention stays inside a sample: the varlen flash backends read the collator's
`cu_seq_lens`, and sdpa / eager get a block-diagonal mask built from `position_ids`.
`DllmTrainer` counts one sample per segment rather than per row, so packing
(`dyn_bsz: true`) and `micro_batch_size` change nothing in the loss.

## Sequence parallelism

All four models train under Ulysses sequence parallelism: set
`model.accelerator.ulysses_size` (it combines with `ep_size` for the MoE models). It needs a
flash backend (`attn_implementation: flash_attention_2` or 3/4, which VeOmni maps to
its `_with_sp` variants), and `ulysses_size` must divide `num_attention_heads`; KV
heads are repeated when there are fewer of them than ranks.

- Samples are counted before the sequence is sliced, so every rank divides by the
  whole row's sample count.
- Each rank's masked loss covers its shard, and `DllmTrainer` sums it over the group.
- LLaDA-MoE's load-balancing loss sums its expert statistics over the group, so it
  equals the unsharded loss. The exception is the few tokens of SP padding
  (fewer than `ulysses_size` per micro-batch), which are routed like real tokens.
- LLaDA-V and LLaDA2.0-Uni keep the pixels and `image_mask` whole on every rank: each
  rank encodes every image and fills only the image slots in its own shard, so an
  image cut by a shard boundary is split correctly. The vision encoding is therefore
  repeated on each of the `ulysses_size` ranks rather than divided among them.
- The vision towers always use sdpa, so they never pick up the backbone's
  all-to-all attention. This holds without SP too, so LLaDA-V's SigLIP no longer runs
  the backbone's flash kernel at `ulysses_size: 1`.

## Limitations

- **No context parallelism.** `cp_size > 1` is refused; use `ulysses_size`.
- **Training only.** Diffusion sampling (`generate`) is not part of this path.

## Checkpoint conversion

LLaDA-8B ships OLMo-style keys (`transformer.blocks.*`). The block is a Llama block, so
conversion is a pure rename:

```shell
python scripts/convert_llada_to_veomni.py --src LLaDA-8B-Instruct --dst LLaDA-8B-Instruct-veomni
python scripts/convert_llada_to_veomni.py --self-test   # key mapping only, no weights
```

Checkpoints saved by training (`save_hf_weights: true`) stay in the Llama layout with
`model_type: llada`. Upstream LLaDA code (`trust_remote_code`) cannot load them without
renaming the keys back.

LLaDA-MoE's weights are already in the Qwen3-MoE key layout, so its conversion only
rewrites `config.json` (`model_type: llada_moe`) and links the shards; the per-expert
tensors are fused into `gate_up_proj` / `down_proj` at load time:

```shell
python scripts/convert_lladamoe_to_veomni.py --src LLaDA-MoE-7B-A1B-Instruct --dst LLaDA-MoE-7B-A1B-Instruct-veomni
```

LLaDA-V's weights already use the LLaVA names the model mirrors. Its conversion rewrites
`config.json`, links the shards, and writes the SigLIP `preprocessor_config.json`
(384 px, mean/std 0.5) that the upstream repo lacks:

```shell
python scripts/convert_lladav_to_veomni.py --src LLaDA-V --dst LLaDA-V-veomni
```

The checkpoint keeps only the SigLIP layers up to the one LLaDA-V reads, so the
converted config records the 26 stored layers with `mm_vision_select_layer: -1`, which
is the same layer the upstream `-2` selects on the full 27-layer tower. The BOS id comes
from the tokenizer, because the upstream config's `bos_token_id: 128000` is outside the
vocabulary.

LLaDA2.0-Uni's backbone already uses the keys the model mirrors (`model.word_embeddings`,
fused-qkv attention, per-expert MoE). Conversion rewrites `config.json`
(`model_type: llada2_uni`), links the backbone shards and `image_tokenizer/model.safetensors`
as one more numbered shard (`model-00014-of-00014.safetensors`; VeOmni reads shard numbers
from these names), and writes one index covering both. Per-expert tensors
are fused into `gate_up_proj` / `down_proj` at load time. `decoder/` and `vae/` are unused.

```shell
python scripts/convert_llada2uni_to_veomni.py --src LLaDA2.0-Uni --dst LLaDA2.0-Uni-veomni
```

## LLaDA-MoE

The experts are sharded over `model.accelerator.ep_size` and run on the fused MoE kernel
(`moe_implementation: fused_triton`). The load-balancing loss is off by default
(`output_router_logits: false`), as in upstream LLaDA-MoE training. When it is on, the
model returns it apart from the masked-diffusion loss and `DllmTrainer` adds
`router_aux_loss_coef * aux_loss` after the per-sample normalization, averaged over the
step's micro-batches and logged as `aux_loss`.

## LLaDA-V

`LLaDAVForConditionalGeneration` is the LLaDA backbone plus a SigLIP tower, an
`mlp2x_gelu` projector and an `image_newline` embedding. Each image is resized to one
384 px tile, and its 27x27 patch features go through the projector with a newline
closing every row, which gives 756 embeddings. The upstream AnyRes multi-tile path
(`image_aspect_ratio: anyres_max_4`) is not implemented, as in the Open-VeOmni port, so
high-resolution or very non-square images lose detail.

The data is ShareGPT or OpenAI-style `messages` with `<image>` placeholders, or with
typed `image` / `text` content parts, plus the `images` in order of appearance. As in
LLaVA's preprocessing, a turn whose string content holds one `<image>` becomes
`<image>\n{text.strip()}`, so the image directly follows the user header wherever the
placeholder was. Typed parts and turns with several placeholders keep their order, and
text after an image starts on a new line. The image's 756 slots are marked by
`image_mask`, and the model scatters the image embeddings into them. They are never
masked and never supervised. An over-long sample is cut from the answer side, unlike
`data_type: llada`, which keeps the end of the conversation, so a long multi-turn sample
can lose its final answer entirely. A sample whose image would not fit in
`max_seq_len` is kept as an unsupervised placeholder, so it adds nothing to the loss or
to the sample count. Text-only samples can be mixed in: when a micro-batch has no image,
the vision tower runs on a dummy tile with zero weight, so every FSDP rank reduces the
same gradients.

## LLaDA2.0-Uni (I2T)

`LLaDA2UniForConditionalGeneration` is the LLaDA2 MoE mask predictor plus a frozen
GlmImage vision tower and VQ codec. Each image is smart-resized inside a 512²–2048²
pixel budget to 16 px patches (`merge_size` 1). The VQ codes are offset by
`image_token_offset` (157184) into the unified vocabulary and fill `image_mask` slots
as ordinary token ids. The image span is
`<|image|><|reserved_token_{h}|><|reserved_token_{w}|><boi>` + `h*w` codes + `<|/image|>`,
matching the checkpoint's `understand_image` prompt. Role tags have no `<|role_end|>`,
and the default system line is space-joined to the first HUMAN header.

The vocabulary has no `<|reserved_token_n|>` for odd `n` up to 35. When the processor's
grid would use one of those sides, the transform resizes to the next spellable side
instead of dropping the sample. Image tokens are never masked and never supervised.
An over-long sample is cut from the answer side, as in LLaDA-V. A sample whose image
would not fit in `max_seq_len` (or that the processor refuses) is kept as an
unsupervised placeholder, with a single warning. At the checkpoint's pixel budget an
image takes 1024 to 16384 tokens, so with `max_seq_len: 4096` every image above ~1M
pixels is dropped this way: lower `size.longest_edge` in the converted
`preprocessor_config.json`, or raise `max_seq_len`, to match the data. The resize to
that budget is this port's choice (the checkpoint ships `do_resize: false`, and the
upstream `understand_image` takes images already encoded).

The tower runs under the training mixed precision, so the VQ nearest-code search is in
bf16 under FSDP and a near-tie can pick a different code than an fp32 encode would.
The backbone's `expert_bias` is a frozen buffer and no load-balancing loss is wired, so
nothing balances the routing during SFT.

The vision/VQ weights stay frozen: the codebook argmax is not differentiable, so the
backbone loss cannot reach them. A rank whose micro-batch has no image still runs a
dummy 2x2-patch encode and discards the codes, so FSDP all-gathers stay symmetric.
The experts are sharded over `model.accelerator.ep_size` like LLaDA-MoE.

## Launch

```shell
bash train.sh tasks/train_dllm.py configs/dllm/llada_8b.yaml \
    --model.model_path LLaDA-8B-Instruct-veomni \
    --data.train_path tulu-3-sft-mixture \
    --model.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --model.accelerator.init_device meta
```

The data reads OpenAI-style `messages`; assistant turns are the answer unless a message
sets `loss_mask`.
