# Multi-Token Prediction (MTP) Training

## Table of Contents

- [Multi-Token Prediction (MTP) Training](#multi-token-prediction-mtp-training)
  - [Table of Contents](#table-of-contents)
  - [📚 Overview](#-overview)
  - [🚀 Quick Start](#-quick-start)
  - [🔍 How the MTP head works](#-how-the-mtp-head-works)
  - [⚙️ Plumbing](#️-plumbing)
    - [The MTP label row](#the-mtp-label-row)
    - [Why `loss_dict` and not `loss`](#why-loss_dict-and-not-loss)
    - [`mtp_context`](#mtp_context)
  - [💾 Checkpoints](#-checkpoints)
  - [📉 Cost](#-cost)
  - [🚧 Limitations](#-limitations)
  - [🛠️ Supporting MTP for a new model](#️-supporting-mtp-for-a-new-model)

## 📚 Overview

Multi-token prediction trains an auxiliary head to predict token `i+2` alongside the
main head's `i+1`, which is what an inference engine later uses as the draft model
for speculative decoding. Qwen3.5 checkpoints ship pretrained MTP weights (15
tensors under the `mtp.` prefix), but upstream `transformers` 5.9.0 has no MTP
module — only `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`. Before this
feature those tensors were loaded and discarded, so continued training silently
degraded the MTP head relative to the trunk.

Currently supported through `tasks/train_text.py`: **Qwen3.5 dense**
(`Qwen3_5ForConditionalGeneration`) and **Qwen3.5 MoE**
(`Qwen3_5MoeForConditionalGeneration`) on GPU and Ascend NPU. DPO, RL, and VLM
trainers do not support MTP. The NPU path uses FLA's Ascend dispatch for the
GatedDeltaNet kernels.

## 🚀 Quick Start

MTP has a single knob, on the *text* config:

```yaml
model:
  model_path: /path/to/Qwen3.5-2B
  ops_implementation:
    attn_implementation: flash_attention_2
    # Strongly recommended with MTP: the head runs a second full-vocab projection,
    # and the eager loss would materialize a second [tokens, vocab_size] fp32
    # logits tensor.
    cross_entropy_loss_implementation: liger_kernel
  model_config:
    text_config:
      mtp_loss_weight: 0.3
```

On Ascend, install `flash-linear-attention>=0.5.2` and a CANN-compatible
`triton-ascend`, then run the provided 27B configuration:

```shell
bash train.sh tasks/train_text.py configs/text/qwen3_5_27b_mtp_npu.yaml
```

Its three GatedDeltaNet implementation fields select FLA's native Ascend dispatch.

MTP is disabled when `mtp_loss_weight` is unset, `null`, or non-positive, or when
`mtp_num_hidden_layers` is unset, `null`, or non-positive. In all of these cases,
the module is never constructed, no `mtp.*` parameters exist, and load / DCP /
export behave exactly as they did before MTP support existed. There is deliberately
no second boolean flag: `mtp_num_hidden_layers` describes the checkpoint architecture,
and MTP is enabled only when both the layer count and loss weight are positive.

The weight is applied inside the model, so **`training/mtp_loss` in the logs is the
weighted contribution**. `BaseTrainer.postforward` sums the loss dict, so whatever is
in it is the optimized objective. Set `mtp_loss_weight: 1.0` to read the raw MTP
cross-entropy.

Expected log line:

```text
total_loss: 2.03, foundation_loss: 1.52, mtp_loss: 0.51, grad_norm: 50.92, lr: 0.00
```

## 🔍 How the MTP head works

The module layout mirrors the checkpoint exactly, and the forward mirrors vLLM's
inference implementation (`vllm/model_executor/models/qwen3_5_mtp.py`), which is the
deployment contract:

```text
e_shift[i] = inputs_embeds[i+1]                       # tail slot zero-filled
h = fc(cat([pre_fc_norm_embedding(e_shift),
            pre_fc_norm_hidden(trunk_hidden)], dim=-1))
h = mtp.layers(h, ...)                                # full_attention decoder layers
mtp_hidden = mtp.norm(h)
logits = lm_head(mtp_hidden)                          # shared with the main head
loss   = CE(logits, labels[i+2])
```

Three details are load-bearing and easy to get backwards. Each was verified
numerically against the pretrained weights (see the table under [Cost](#-cost) for
the method):

| Detail | Correct | If you get it wrong |
|---|---|---|
| `fc` input order | **embedding first**, `cat([emb, hidden])` | Multiplies the hidden state by `fc.weight`'s embedding half. Loss jumps to ~9x — above `ln(vocab_size)`, i.e. worse than uniform. |
| Which trunk hidden state | **post**-final-norm (the trunk's returned `last_hidden_state`) | `pre_fc_norm_hidden` was trained on the normed distribution; loss rises ~1.3x. |
| positions / rotary | **not shifted** — reuse the trunk's `position_embeddings` | Megatron-style MTP does roll `position_ids`; vLLM's Qwen3.5 path does not. Copying that would desynchronize from inference. |

The head shares `embed_tokens` and `lm_head` with the main model: the released
checkpoints carry neither `mtp.embed_tokens` nor `mtp.lm_head`, and
`mtp_use_dedicated_embeddings` is `False` (asserted at construction).

## ⚙️ Plumbing

### The MTP label row

`ForCausalLMLoss` shifts labels by one internally when SP is disabled
(`veomni/ops/kernels/cross_entropy/__init__.py`), so position `i` of the main head
predicts `labels[i+1]`. The MTP head needs `labels[i+2]`, supplied as an explicit
`shift_labels=` argument that bypasses the internal shift.

The row is built by a **per-sample** collator hook
(`Qwen3_5ForConditionalGeneration.get_sample_collate_func` →
`SampleFieldsCollator`), which runs after `PrecomputePositionIDsCollator` and
*before* `PackingCollator`. That ordering is the whole point: shifting by two inside
an already-packed row would pull the next sample's first tokens into the tail of the
current one. Doing it per sample makes that impossible by construction, so no
`cu_seq_lens` boundary arithmetic is needed.

`mtp_labels` is registered via `get_extra_collate_infos()` as
`(-1, True, IGNORE_INDEX, 1)`. Registration matters beyond SP: `pad_to_length` pads
every `pack_dim == -1` key with its `sp_pad_value`, so an unregistered row would be
left short. Registration also makes `count_loss_token` emit `mtp_tokens` for free
(it derives `{prefix}_tokens` from any `*_labels` key), which is what
`mean_global_loss` normalizes the MTP loss by.

`TextTrainer._build_collate_fn` resolves `get_extra_collate_infos` and
`get_sample_collate_func` for text training. Other trainer-specific collators are
unchanged.

### Why `loss_dict` and not `loss`

Per-head losses travel in a dedicated `loss_dict` field. They **cannot** go in
`loss`: it is field 0 of the output dataclass, and on the fused-loss path every
other field is `None`, so `ModelOutput.__post_init__` scatters a dict first field
into standalone attributes and deletes `loss` — leaving `outputs.loss is None`.

Nor can the dict be wrapped in an opaque object: `ModelOutput`'s pytree flattening
walks `output.values()`, and FSDP2's pre-backward unshard hook depends on the loss
tensors being reachable there. Hiding them reintroduces
`setStorage … storage of size 0` in the fused-linear backward.

`tests/trainer/test_multi_head_loss.py` pins both behaviours.

### `mtp_context`

The MTP head needs four tensors that only exist inside `Qwen3_5TextModel.forward`:
the post-vision-scatter `inputs_embeds`, the normalized `position_ids`, the trunk's
`position_embeddings`, and the `causal_mask`. `Qwen3_5ModelOutputWithPast` carries
only `last_hidden_state` / `past_key_values` / `rope_deltas`, so they are threaded
out through an added `mtp_context` field (`Qwen3_5MTPContextOutput`).

Recomputing them one level up would re-embed `input_ids` *without* the scattered
vision features — silently wrong for multimodal batches. `mtp_context` is populated
only while training and only when MTP is enabled, so inference and MTP-off runs
allocate nothing extra.

## 💾 Checkpoints

The module is registered at the top level as `ForConditionalGeneration.mtp`, so live
parameter FQNs are exactly `mtp.*` — identical to the released checkpoint. For dense
models, that alignment makes all three paths work with no MTP-specific code:

- **HF weight load** — `load_model_weights` matches by name; no conversion mapping.
- **DCP save/resume** — `get_model_state_dict` picks the params up automatically.
  A saved checkpoint contains 15 `model.mtp.*` parameters plus 45
  `optimizer.state.mtp.*` AdamW entries.
- **HF safetensors export** — `get_model_save_state` hard-drops any FQN missing from
  the index-derived `fqn_to_index_mapping`, and the shipped
  `model.safetensors.index.json` already lists all 15 `mtp.*` keys. Registering the
  module under `model.language_model.mtp.*` instead would have it **silently
  discarded** on export.

Qwen3.5-MoE checkpoints keep trunk experts fused but store MTP experts as per-expert
`gate_proj` / `up_proj` / `down_proj` tensors. The model-specific checkpoint converter
merges only those MTP keys into the generated model's fused expert layout at load time.

⚠️ **Do not toggle MTP across a resume.** `set_model_state_dict` is strict, so
resuming a pre-MTP checkpoint with `mtp_loss_weight` set fails on missing keys, and
the optimizer param groups change shape too. Enable MTP from step 0 off HF weights.

## 📉 Cost

MTP adds one decoder layer and a second projection through the shared `lm_head`.
Lowering `mtp_loss_weight` changes the objective but does not reduce this compute.

On Qwen3.5-35B-A3B with 16 Ascend ranks, EP16, sequence length 1024 and one MTP
layer, a warm-cache 20-step comparison measured 3.36s without MTP and 3.49s with
MTP per step (median, +3.9%). Peak memory increased from 43.95GB to 44.89GB (+2.1%).

## 🚧 Limitations

- **Sequence parallel is rejected**, with an assert at construction. Two independent
  reasons: the `inputs_embeds` left-shift crosses rank boundaries, and
  `ForCausalLMLoss` *silently ignores* `shift_labels` when `sp_enabled`, which would
  train the head on 1-shifted labels behind nothing louder than a `warning_once`.
- **Training only.** Speculative decoding runs in the inference engine; the forward
  asserts `past_key_values is None`.
- **Text SFT only.** Use `tasks/train_text.py`; DPO, RL, and VLM trainers are not
  wired to construct MTP labels.
- **Multimodal shift semantics differ slightly from vLLM.** vLLM rotates `input_ids`
  then embeds; training shifts the already-scattered `inputs_embeds`. Equivalent for
  text, and only different at multimodal placeholder boundaries.
- `output_hidden_states=True` gains one entry: `_can_record_outputs` captures by
  `Qwen3_5DecoderLayer` class, so `mtp.layers.0`'s output is collected too. Code
  indexing `hidden_states[-2]` is affected.
- `EnvironMeter`'s MFU/FLOPs accounting does not include the MTP head, so reported
  MFU is an underestimate when MTP is on.
- **Ascend NPU is supported.** Use `configs/text/qwen3_5_27b_mtp_npu.yaml` as the
  dense-model reference; MoE uses the same MTP knob and FLA Ascend operator settings.

## 🛠️ Supporting MTP for a new model

The trainer-side plumbing (`SampleFieldsCollator`, the `TextTrainer` hook lookup,
`loss_dict` in `postforward`, `count_loss_token`'s `{prefix}_tokens`) is
model-agnostic. Per model you need, in its patch config:

1. An MTP `nn.Module` whose submodule names match the checkpoint's `mtp.*` FQNs,
   added with `config.add_helper_after(...)`.
2. A way to reach the trunk's `inputs_embeds` / `position_embeddings` / masks —
   for Qwen3.5 that meant overriding the text model's `forward` to emit
   `mtp_context`.
3. `__init__` overridden to construct the module under the FQN the checkpoint uses,
   plus the SP/EP asserts.
4. `get_extra_collate_infos` / `get_sample_collate_func` returning the label rule
   and the per-sample hook (module-level functions only — they are pickled to
   DataLoader workers).
5. `forward` extended with an explicit `mtp_labels` parameter (never left in
   `**kwargs`, which would leak it into the attention and CE kernels) returning
   `loss_dict`.

Note `config.modify_init()` looks like the natural fit for step 3 but is **dead
API**: `PatchType.INIT_MODIFICATION` has no implementation in patchgen's codegen, so
the patch is silently dropped. Use `override_method("<Class>.__init__")`.
