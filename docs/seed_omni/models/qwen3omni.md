# Qwen3-Omni-30B-A3B-Instruct (SeedOmni)

End-to-end recipe for running **Qwen3-Omni-30B-A3B-Instruct** as a SeedOmni
graph model. This is a **speech-capable omni** model: the monolithic
`Qwen3OmniMoeForConditionalGeneration` is split into six OmniModules. The
thinker half (audio + vision + text + backbone) produces the text reply; the
speech half (talker + codec) turns that reply into a waveform.

```
qwen3omni_audio  ─┐
qwen3omni_vision ─┼─→ qwen3omni_llm → qwen3omni_text_encoder.decode → end   (text reply)
qwen3omni_text_encoder.encode ─┘        │
                                        └─→ qwen3omni_talker → qwen3omni_codec → end   (audio reply)
```

| Module | Holds | Role |
|--------|-------|------|
| `qwen3omni_audio` | audio tower + `WhisperFeatureExtractor` | waveform → audio embeds |
| `qwen3omni_vision` | ViT + patch merger + image/video processors | image/video → merged patch tokens + DeepStack features |
| `qwen3omni_text_encoder` | `embed_tokens` + untied `lm_head` + tokenizer | token ↔ embedding, CE loss head, ChatML (text + image + video + audio) template |
| `qwen3omni_llm` | thinker backbone (no embed / no head) | `inputs_embeds → hidden_states`, TMRoPE, exposes the intermediate layer the talker consumes |
| `qwen3omni_talker` | talker + residual code predictor | thinker states → RVQ codes |
| `qwen3omni_codec` | Qwen `code2wav` **+** [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) encoder | codes ↔ waveform |

Three Qwen3-Omni specifics worth knowing before you touch the configs:

- **TMRoPE** — 3-row (t/h/w) position ids covering text, image, video *and*
  audio. Every modality shares one wall clock: the temporal row is
  `position_id_per_seconds` (13) times elapsed seconds. Audio gets there by
  emitting 13 tokens per second and spending one position id per token; video by
  advancing `second_per_grid * 13` per grid step. Two inputs decide whether that
  holds, and both fail silently if wrong, so both are explicit:
  - Video segments need `second_per_grid`, which the vision preprocessor computes
    from the clip's own kept frames (`VideoMetadata.seconds_per_patch`). It
    raises rather than defaulting.
  - Audio must reach the tower at **16 kHz**, since "13 tokens per second" is
    really "13 tokens per 100 mel frames" and a mel frame is 10 ms at 16 kHz
    only. The data layer declares the clip's rate in `meta["audio_metadata"]`
    (`AudioMetadata.sampling_rate`) and the audio preprocessor resamples a local
    copy; a clip with no metadata is taken to be 16 kHz already, which is what
    `audio_utils.load_audio` produces for an array. Feeding 22.05 kHz undeclared
    makes a 4 s clip occupy 5.54 s of the clock and drift against video by 38%.
    Note `WhisperFeatureExtractor` never resamples on its own.

    The preprocessor also stamps `_omni_token_len` with the encoder's *output*
    row count (13 per 100 mel frames), which is what `dyn_bsz` counts — a
    sample's backbone length has to be known before any model exists, and an
    audio row left unmarked contributes zero.

  A video's *own* soundtrack (`use_audio_in_video`) is a separate matter:
  upstream merge-sorts the two streams' tokens by timestamp on that shared clock.
  VeOmni does not implement that interleave yet, so the chat template and the
  backbone place such a clip by its frames and skip the track: training and
  answers see the pictures only. Decode with `use_audio_in_video: false` until it
  lands. See [media](../design/media.md). Separate `type="audio"` and
  `type="video"` items in one conversation are fully supported.
- **The talker reads an interior layer.** It consumes hidden states from
  `accept_hidden_layer` (24) of the thinker, not the final layer, plus the
  thinker's input embeddings. Keeping those means keeping the whole layer stack,
  so `collect_thinker_states` gates it and each scenario's `modules_*.yaml` opts
  in — it is a property of the graph, not of the weights. The two paths hand the
  traces over differently, because generation sees the thinker one token at a
  time: it accumulates them and passes them through the FSM's `ctx`
  (`qwen3omni_llm.flush_thinker_states`, `qwen3omni_text_encoder.flush_talker_prompt`),
  whereas training runs one pass over the packed sequence and leaves each item's
  slice on the carrier in `meta["thinker_embed"]` / `meta["thinker_hidden"]` —
  the same per-item split that returns hidden states to `item.value`.
- **The codec is two checkpoints.** Qwen ships a vocoder but no audio
  tokenizer, so the encoder half comes from
  [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) (CC-BY-4.0, on the Hub —
  the converter downloads it if you pass the id). See §1.

Config dir: `configs/seed_omni/Qwen/qwen3omni_30b_a3b/`

| File | Role |
|------|------|
| `train/base.yaml` | Launcher: `model` / `accelerator` / `data` / `train` + `infer` block. |
| `train/modules_train.yaml` | Per-module training overrides — thinker only (`audio` / `vision` / `text_encoder` / `llm`). |
| `train/graph_train.yaml` | Training DAG for the thinker. |
| `train/data.yaml` | Weighted multisource data list. |
| `train_talker/base.yaml` | Launcher for the speech round. |
| `train_talker/modules_train.yaml` | All six modules — thinker frozen, talker the only trainee. |
| `train_talker/graph_train.yaml` | Training DAG for the talker. |
| `train_talker/data.yaml` | Points at the generated assistant-speech corpus (see §4). |
| `train_joint/base.yaml` | Launcher for the joint round — thinker and talker in one backward. |
| `train_joint/modules_train.yaml` | All six modules, nothing frozen. |
| `train_joint/graph_train.yaml` | Training DAG carrying both the text CE and the talker CE. |
| `train_joint/data.yaml` | The talker corpus — the one shape carrying both targets. |
| `infer/modules_infer_eager.yaml` | Single-process eager module list (no FSDP wrap). |
| `infer/modules_infer_eager_audio.yaml` | The same list plus the checkpoint's speech system turn. |
| `infer/graph_infer_audio.yaml` | Generation FSM: thinker text → talker codes → waveform. |
| `infer/graph_infer_thinker_only.yaml` | Generation FSM for text only (no speech half). |

---

## 1. Convert the checkpoint

`model_type=qwen3_omni_moe` is served by two entry points, and only this family
needs both:

* `modules/qwen3omni/hf_layout.py` — the key map, so "which upstream tensor goes
  to which module" is stated once. Pointing `model.model_path` at the upstream
  root uses it directly.
* `modules/qwen3omni/convert_model.py` — the offline converter, which must run
  for this family because the codec needs a **second** checkpoint: the
  [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) encoder that pairs with
  Qwen's decoder. Its converter takes that
  as a `mimi_path` kwarg, which the shared entry point forwards through
  `--extra` rather than carrying a flag no other family uses. It accepts a local
  directory or a hub id, and defaults to the Hub id `kyutai/mimi`
  (`hf download kyutai/mimi --local-dir <dir>` to fetch it explicitly).

```bash
python scripts/seed_omni/convert_model.py \
  --model_path /mnt/hdfs/veomni/models/Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --output_dir /mnt/hdfs/veomni/models/seed_omni/Qwen3-Omni-30B-A3B-Instruct \
  --extra mimi_path=kyutai/mimi
```

The split fails loudly if the Mimi half is missing rather than emitting a codec
that can only decode — talker training needs the encoder, and the failure would
otherwise surface far from here.

The MoE experts are fused on the way out (per-expert 2D tensors → one 3D
tensor) for both the thinker backbone and the talker, matching what the patched
modeling expects.

The `output_dir` becomes `model.model_path` in `base.yaml`.

### Module boundary map

| Module | Upstream prefixes |
|--------|-------------------|
| `qwen3omni_audio` | `thinker.audio_tower.*` |
| `qwen3omni_vision` | `thinker.visual.*` |
| `qwen3omni_text_encoder` | `thinker.model.embed_tokens.*`, `thinker.lm_head.*` |
| `qwen3omni_llm` | the rest of `thinker.model.*` |
| `qwen3omni_talker` | `talker.*` |
| `qwen3omni_codec` | `code2wav.*` (encoder half from Mimi) |

A source key belongs to the module whose prefix is the **longest** match, which
is what keeps `embed_tokens` out of the backbone without an ordering rule.

---

## 2. Inference

Single-process eager, one GPU. The 30B-A3B weights fit on one H20-class card,
and the talker's sampling loop mixes `pad_token_id` with `next_tokens` without a
device cast, so an `auto` split across GPUs dies on a `cuda:0`/`cuda:N` mismatch.

```bash
python tasks/omni/infer_omni.py configs/seed_omni/Qwen/qwen3omni_30b_a3b/train/base.yaml \
  --model.model_config.modules configs/seed_omni/Qwen/qwen3omni_30b_a3b/infer/modules_infer_eager_audio.yaml \
  --model.model_config.infer_type audio_output \
  --infer.prompt "Hello! Please say this sentence out loud." \
  --infer.output_dir qwen3omni_out
```

* `--model.model_config.infer_type thinker_only` for a text reply with the
  speech half left out — any text-out evaluation (transcription, VQA, knowledge
  questions) scores that, and the 3.6B talker plus a vocoder pass per sample is
  wasted there.
* `thinker_only` also isolates faults: if it is right and `audio_output` is not,
  the fault is in the bridge or the talker, not the thinker.
* The audio scenario needs `add_speech_system_prompt: true` on
  `qwen3omni_text_encoder` (set in `modules_infer_eager_audio.yaml`). Without it
  the thinker answers in numbered prose that never reaches `<|im_end|>` inside the
  token budget, so the FSM never hands over to the talker and no audio comes out.
  Text-only rounds leave it off: it caps replies at 50 words and imposes a voice
  persona, which would skew a text evaluation.

### Which ops it runs

`base.yaml` selects `attn_implementation: flash_attention_2`. On a host without
flash-attn, override to `eager` — the audio tower, ViT and thinker all take the
selected backend through their nested configs, so one flag covers all three:

```bash
--model.ops_implementation.attn_implementation eager
```

The Mimi half of the codec is pinned to `sdpa` in `codec/modeling.py` regardless:
Mimi dispatches through its own attention table and does not know VeOmni's
sequence-parallel variants. That costs nothing — the encoder is frozen, tiny, and
runs once per clip.

### Reproducibility of the audio

The talker's residual code predictor is hardcoded to sample in transformers, and
its draws re-enter the AR context, so even the semantic group drifts between runs
and the frame count varies. Codes are therefore reproducible only under a pinned
RNG (a seed and a prompt reproduce upstream bit-for-bit, 688/688 codes on the
released checkpoint); **compare audio, not codes**, when no seed is pinned.

The codec halves are numerically identical to upstream to within floating-point
reassociation (~1e-10 on CPU), which
`tests/seed_omni/modules/qwen3omni/test_qwen3omni_codec.py` asserts with a stated
tolerance rather than bit equality.

---

## 3. Training

```bash
bash train.sh tasks/omni/train_omni.py configs/seed_omni/Qwen/qwen3omni_30b_a3b/train/base.yaml \
  --model.model_path /mnt/hdfs/veomni/models/seed_omni/Qwen3-Omni-30B-A3B-Instruct
```

The backbone is MoE, so `qwen3omni_llm` declares an expert parallel plan
(`mlp.experts.gate_up_proj` / `down_proj`); set `--model.accelerator.ep_size N`
to use it. Accelerator settings nest under `model`, so a bare
`--accelerator.ep_size` is rejected as an unknown root key.

`qwen3omni_talker` is MoE too but ships **no** EP plan yet: its experts are
sharded by FSDP only, and the talker round is not EP-tested. Adding one is a
`get_parallel_plan` on its accelerated mixin plus coverage.

`graph_train.yaml` trains the thinker only — the talker and codec are not
declared, so they are neither built nor sharded.

### `cast_forward_inputs: false`

The launcher turns FSDP2's forward-input cast off, and the reason is specific to
this graph. What FSDP2 would cast depends on where the wrap boundaries fall.
Under `fsdp_scope: module` (the default) the boundary is each module, so the
backbone's TMRoPE `position_ids` — float, running up to `max_seq_len` — arrive as
bf16, which at 4096 resolves only 769 distinct values (up to 8 token positions of
error) and at 16384 only 1025 (32 positions). Every module casts inside its own
`forward` off the real weight dtype instead, which is also the only place that
reads bf16 rather than the fp32 master shard.

### What the talker reads, and from where

| Key | Published by | Carried on |
|-----|--------------|------------|
| `thinker_embed` / `thinker_hidden` | `qwen3omni_llm.forward` post-hook | `item.meta`, one slice per **backbone-input** row (`is_backbone_input`) |
| `qwen3omni_talker_prompt` | `qwen3omni_text_encoder.encode` post-hook | `item.meta` on the sample's first backbone row |
| `qwen3omni_talker_labels` | `qwen3omni_codec.encode` post-hook | `item.meta` on the assistant audio row |

The first two are gated by `collect_thinker_states` / `publish_talker_prompt` in
the scenario's `modules_*.yaml`. Setting one and forgetting the other cannot fail
silently: the talker names the flag it is missing.

### Audio on the carrier

`audio_utils.load_audio` returns a clip at whatever rate the file holds and
`AudioMetadata` records that rate; each module then converts to its own. The
split exists because the two modules reading a conversation disagree — the
thinker's tower wants 16 kHz and the codec that tokenises assistant speech wants
24 kHz — so any rate the data layer picked would be wrong for one of them, and a
second conversion cannot restore what the first removed.

A module that only needs a different rate to *read* the clip (the audio tower,
building mels) resamples a local copy and leaves `meta["audio_metadata"]` alone.
A module that converts the payload **in place** (the codec, which leaves a 24 kHz
tensor on `value`) must move the metadata with it, via `AudioMetadata.resampled_to`
— which refuses a conversion that retimed the clip, since the talker's targets are
scored against text.

### The target audio is a label, not a turn

`role="assistant"` audio is the waveform the talker is taught to say. The thinker
has no speech output and never consumes its own voice, so that row stays on the
carrier — that is how the codec finds it — but takes **no position** in the
backbone's packed sequence.

Every packer, every hidden-state scatter, the text encoder's decode walk, the
talker's segment view and the rebuilt token ids skip it by one shared predicate,
`veomni.models.seed_omni.utils.conversation.is_backbone_input`. They index the
same sequence by position, so a row skipped by one and kept by another would
shift every row after it. The row says so itself: the chat template sets
`meta["_not_backbone_input"]` when it defers the speech target out of the
sequence. Deriving the rule from `role`/`type` instead would not work — the same
`type="audio"` row is an *input* on the user's side and a *target* on the
assistant's, and only the row itself can say which it is.

That asymmetry is also why the talker's `prompt_length` is not simply the sum of
the segments. At inference the prompt ends at the generation prompt and the trace
continues over text the thinker generated; in training that text is already on
the carrier, teacher-forced, so the sequence runs past the prompt and taking all
of it as "the prompt" would make the speech-span prefix swallow the response the
talker is meant to speak. The boundary is read per *token* from `meta["labels"]`,
and relative to the final assistant turn. Not `meta["loss_mask"]`: `tokenize`
pops that key once it has expanded it into labels, and `merge_text_embeds` then
fuses the masked `<|im_start|>assistant\n` prefix into the supervised response as
one item, so an item-granularity rule cannot see the boundary at all. And not the
sample's first supervised token either: ordinary multi-turn SFT supervises every
assistant turn, so a global search would land behind the final turn and give a
negative prefix. With nothing supervised at or after that turn — the inference
shape — the whole given sequence is the prompt, which is what inference wants.

The span has a far end too, and the mask cannot find it: the `<|im_end|>\n` that
closes the assistant turn is supervised as well, since the thinker is scored on
emitting it. So the end is read off the token id instead — the first `<|im_end|>`
at or after the turn's start. Those two rows would otherwise land in
`trailing_text_hidden`, pushing `tts_eos` (the "the text is finished" cue the
talker times its `codec_eos` against) two codec frames later than it will fall at
inference, and the loss would look perfectly healthy while the model learned to
end late. Inference needs no trimming and gets none: it has generated no closing
marker yet, so the search finds nothing and the span runs to the end of the
trace.

---

## 4. The talker round

```bash
bash train.sh tasks/omni/train_omni.py configs/seed_omni/Qwen/qwen3omni_30b_a3b/train_talker/base.yaml \
  --model.model_path /mnt/hdfs/veomni/models/seed_omni/Qwen3-Omni-30B-A3B-Instruct
```

`train_talker/graph_train.yaml` has two roots, because the talker needs two
unrelated things: a prefill derived from the frozen thinker's pass, and the RVQ
codes to score it against. The codec is a root — it reads the assistant's
waveform straight off the carrier and owes nothing to the thinker.
`qwen3omni_text_encoder.decode` is absent, unlike the thinker graph: the thinker
is frozen here, so its text CE would build a loss with no parameter behind it.

Four things to keep in mind when editing `train_talker/modules_train.yaml`:

* Every sample must carry an assistant audio turn. The talker all-gathers its
  sharded layers once per utterance, so a partly-speech mix would give the ranks
  different call counts — it raises rather than hangs, but the run stops.
  `drop_last: true` and `dyn_bsz: false` are there for the same reason.
* `freeze_merger: true` on the vision module is load-bearing, not an optimizer
  saving: one trainable parameter in that tower makes the thinker's
  `inputs_embeds` require grad (the FSDP dummy anchors are added in as `x * 0`,
  which preserves it), and the frozen 30B backbone would then build and retain a
  full un-checkpointed activation graph for a backward that never comes.
* The codec is pinned to fp32 (`mixed_precision: param_dtype: float32`). It
  produces the talker's labels, and a bf16 Mimi encoder changes ~32% of the code
  indices it emits. `enable: false` would *not* do this —
  `ModuleRuntime.build_model` reads that flag to pick the build dtype and would
  hand back bfloat16 weights.
* The codec's `encode` hook all-gathers every FSDP2 unit under Mimi for the
  duration of one encode and reshards after. `encode_wav` reaches the encoder
  through methods (`self.mimi.encode(...)`, then the quantizers' own `encode`),
  so FSDP's `__call__` hooks never fire for them.

### The assistant-speech corpus

Talker training needs each sample to carry the assistant's *text* and the
waveform that says it. The shipped `voice_assistant` source is the opposite shape
— a spoken question with a text answer — so its clip is a tower input, and
nothing else shipped here has assistant speech at all.

Producing that corpus means running the speech graph over real prompts and
recording both halves of each reply. The `voice_assistant_talker` preprocessor
(`veomni/data/seed_omni/preprocess.py`) consumes that shape: it requires both
parts of the assistant turn and pairs clips to turns by position, refusing a
count mismatch in either direction.

Being the model's own speech makes this self-distillation: enough to make the
round runnable and to overfit a sample end to end, not enough to teach the talker
anything it does not already say. A TTS corpus of the same row shape is the
substitute for a run whose output is meant to be a better talker.

A generator's own metadata filters cannot hear that a clip is fluent noise, so a
generated corpus needs a listening pass: transcribe each clip back through the
`thinker_only` graph (see §2) and drop the rows whose speech does not say their
text. Skipping it trains the talker on clips that are not utterances.

### What the codec sharing was verified to mean

Measuring both directions on real checkpoints is the only way to settle this.
Audio-level agreement is good; code-level agreement is **not** identity and must
not be expected to be. One run on the released checkpoint, for orientation:

```
log-STFT L1 vs the talker's own audio (lower is better)
  mimi.encode -> mimi.decode       0.0653
  mimi.encode -> qwen.code2wav     0.1235   <- encode direction works
  talker codes -> mimi.decode      0.1256   <- decode direction (ref)
  shuffled codes -> mimi.decode    0.2344   <- control
```

The decisive control is that Mimi against *itself* recovers only 85% of its own
semantic group and 12.7% of the deep residual groups, so the loss is intrinsic
RVQ residual sensitivity rather than an encoder/decoder mismatch. Validate this
codec — and any talker trained on it — on audio, never on exact code match
beyond group 0.

**Reproduced on `kyutai/mimi`.** Those numbers need the Qwen checkpoint too. What
was checked without it, on a 1 s 220 Hz tone against the real Mimi weights:

| Check | Result |
|-------|--------|
| Mimi loads, fp32 | 79.3 M params, 350 tensors |
| config halves agree | 24 kHz, 16 quantizers, 1920 samples/frame on **both** sides |
| `encode` → `decode` round trip | `(1, 16, 13)` codes → 1.04 s audio; 13 frames/s = 12.5 Hz ✓ |
| `Qwen3OmniCodec.encode_wav` on the same tone | `(16, 13)` — the talker's target contract |

That pins the *plumbing*: the encoder half the converter fetches is the one this
module's config was written against, the per-frame timing matches on both sides,
and the target-tensor contract holds through this module's own code rather than a
reimplementation. It does not measure the index-space agreement above, which
still needs both checkpoints.

---

## 5. Tests

`tests/seed_omni/modules/qwen3omni/` covers each module on random weights plus
the carrier contracts the graph depends on. Cases that need the shapes and
special-token ids of the shipped model read them from
`tests/toy_config/qwen3omni_toy/`, which carries the real identity (token ids,
speaker table, codec ids, patch/merge sizes) in a CPU-sized configuration, so
the suite runs whole without weights.

```bash
pytest tests/seed_omni/modules/qwen3omni/ tests/seed_omni/test_shipped_training_graphs.py
```

One case decodes the prompt back to text to read the speech-span boundaries, so
it does need the real tokenizer; it resolves through `CI_HF_MODELS_DIR` and skips
when no mirror is staged.

The codec is the one module that can be partly exercised against real weights
without the Qwen checkpoint, and was — see §4. Loading `kyutai/mimi`, checking its
config against what `Qwen3OmniCodecConfig` expects (rate, quantizers,
samples/frame on both sides) and round-tripping a tone reproduces those numbers;
`Qwen3OmniCodec.encode_wav` returning `(num_quantizers, T)` is the contract the
talker is written against. The two-codec comparison in §4 needs **both**
checkpoints, because it has to run the talker to obtain codes at all.

```{note}
Still unverified here: training and generation end to end, and the index-space
agreement between Mimi's codes and Qwen's vocoder. Both need the full Qwen3-Omni
checkpoint, not only the encoder half.
```

