# H3 Ref2VA and CFG-aware training

## Scope

Extend the public H3 implementation rather than porting an internal trainer.
The baseline already supports prepared visual Ref2VA samples, model-owned packing,
FSDP2, checkpointing and sequence parallelism. Those implementations remain intact.
This change adds a Ref2VA offline recipe and one opt-in CFG-calibrated FM objective with two
unconditional data contracts. Upstream noise sampling is unchanged.

## Ownership

| Component | Responsibility |
| --- | --- |
| H3 condition config | Validate model-specific objective, conditioning and loss-weight settings |
| H3 condition model | Sample noise once; prepare conditional/unconditional layouts using identical target latents, anchors and timesteps |
| H3 conditioning helpers | Load a local shared embedding once; rebuild or validate the unconditional layout |
| H3 objective helpers | Resolve modality scales, loss coordinates and sigma-bin loss weights |
| H3 transformer wrapper | Run detached unconditional prediction, then differentiable conditional prediction; retain upstream loss reduction |
| Existing DiT trainer/runtime | LoRA, optimization, accumulation, checkpointing and device topology; unchanged |

## CFG-calibrated flow-matching objective

Let `p` be the current model's conditional velocity prediction, `u` its detached
unconditional prediction, and `y = noise - clean` the flow-matching target.

One objective is exposed, with a curvature exponent `k` in `[0, 2]`:

```text
p_calibrated = (p + (s - 1) * stopgrad(u)) / s
loss = s ** (2 - k) * MSE(p_calibrated, y)
```

`training_cfg_curvature_power=2` is the default and preserves the original
inverse-CFG formula. `k=0` is algebraically equivalent to the previous
target-guidance form; neither is presented as a separate algorithm. The name
CFG-calibrated FM is descriptive, not a claim of a novel objective. Relative to
ordinary MSE, conditional curvature is `1/s**k`. With fixed `u` and `s`, changing
`k` changes the strength but not the optimum. With dynamic scales it also changes
the relative weighting of samples and modalities, so it is not generally a
single learning-rate adjustment. No new teacher is introduced.
`constant` uses the configured scale; `sigma` uses `1 + (scale - 1) * sigma`
separately for audio and video. The scale-one default retains the existing single
forward, loss and RNG consumption. Raw predictions remain raw in model outputs.
Existing timestep weights and no-audio masking apply after the objective mapping.
Do not place diagnostic scalars in `outputs.loss`: the trainer sums every entry.

## Conditioning contracts

- `shared_empty`: load a local safetensors file with `prompt_embeds [L,D]` and
  `text_token_tags [L]` once per condition model. Require finite floating-point
  embeddings, text-only tags and matching embedding width. No synthesized zeros,
  HDFS downloader or internal environment variables.
- `per_sample`: decoded offline samples contain `unconditional_prompt_embeds` and
  `unconditional_packed`. These can encode empty text with reference visual tokens.
  Validate matching target geometry and reference-row counts before a model forward.

Both modes retain the original noisy video/audio and visual latent anchors; shared
empty replaces the Qwen condition only, not the independent visual latent anchors.
The unconditional layout has its own text length, positions and segment bounds.
It must not reuse the conditional text length or resample target noise. Prepared
per-sample layouts must use the upstream compact layout builders.

## Noise policy

Upstream uniform timestep-index sampling is unchanged, including RNG consumption.
Video and audio use the same selected index with their respective existing shifts.
Optional five-bin video sigma weights change video loss only, not sampling.
Use the existing global RNG so upstream checkpoint RNG capture remains applicable.

## Compatibility and limits

No new shared trainer, checkpoint format, hardware kernel, service, Magnus schema,
training-adapter loader, online reference encoder, or audio-reference support.
Existing single-sample SP and multi-sample packing constraints still apply.
Probability gates, step-dependent CFG warmup and frozen-base drift are separate
follow-ups: they require optimizer-step and/or model-runtime ownership and should
not be smuggled in as H3 model-global mutable state.

## Validation

CPU tiny-model tests cover FL2VA/Ref2VA, unequal text lengths, shared/per-sample
conditioning, missing or mismatched data, loss/gradient curvature, branch detachment,
packing/serial equivalence and default-path regression. Existing H3 packing tests
remain mandatory. Real distributed SP/FSDP2, pretrained quality and throughput
require accelerator validation; local CPU tests do not establish those claims.

Local verification after unifying the curvature interface: 82 H3 tests and 163
native-LoRA/checkpoint-converter tests pass. The full pytest run stops during
collection because the E2E suite cannot import `exec_scripts`; it is not reported
as passing. Independent review of the preceding implementation found no
confirmed correctness defect. Additional two-rank CPU/Gloo FSDP2 tests matched
unsharded losses/gradients across repeated single/packed forwards and checkpoint
recomputation. GPU/NPU SP, mixed precision and distributed-resume checks remain
required. Generic DiT microbatch tests hit the upstream Triton dependency gate on
this Mac before model execution; they are not reported as passing.
