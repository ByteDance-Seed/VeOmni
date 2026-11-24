# DiffusionGemma cross-stack parity

Checks VeOmni's DiffusionGemma against NeMo AutoModel's independent implementation of
the same model.

**Not bitwise.** The two stacks cannot be made bitwise identical and it is not worth
trying: NeMo does not run SDPA (it has its own eager matmul with an fp32 softmax),
VeOmni keeps `per_expert_scale` as a parameter where NeMo folds it into the expert
down-projection, the two normalize the loss differently, use different mixed-precision
policies and different RNG derivations, and VeOmni is not bitwise reproducible against
*itself* (`enable_full_determinism: false`, Triton autotuning). What is checkable, and
what these scripts check, is that the two compute the same function and the same
gradients.

## Result

| what | where | outcome |
|---|---|---|
| Mask semantics | `tests/models/test_diffusion_gemma_mask_matches_nemo.py` | identical sets, including the sliding window |
| Forward, tiny (CPU/fp32) | `dump_forward.py` + `compare_dumps.py` | `encoder_hidden` 1.5e-8, `decoder_logits` 1.9e-9 |
| Gradients, tiny (CPU/fp32) | same, with `--grads` | 104/104 parameters within 1e-5 |
| Gradients, tiny, self-conditioning on, non-unit scale | same, `--self-conditioning --align-nemo` | 126/126 within 1e-5; **15 fail without the switches** |
| Forward, 26B + real Sudoku | same, with `--inputs` | **8.5e-2 as shipped** — one loader bug; 3.8e-6 once fixed |
| Gradients, 26B + real Sudoku | same, with `--grads` | **719/723 wrong as shipped**; 90/723 once fixed, all of them `per_expert_scale` |
| Gradients, 26B, expert scale folded | same, `--align-nemo` | those 90 go to 0; 25 left at 1.0e-3 – 2.4e-3, from one tie-break (below) |
| bf16 GPU floor | `measure_hole_kernel_bf16_floor.py` | kernel vs sdpa 2.8e-3 – 7.7e-3 |

The stacked expert weights' storage layout differs (NeMo transposes them) and is
translated rather than tolerated. It does not change the function; the forward agrees
either way.

### Where the two stacks train differently, and the switches that close it

Two differences produce the same forward and a different *first step*, because they
hand the optimizer a different set of parameters. Neither is visible in a forward
comparison, and both survive turning the AR loss off.

| | VeOmni | NeMo | switch |
|---|---|---|---|
| expert scale | trains `s` and `W` separately | folds `W' = sW` at load, trains `W'` | `VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD` |
| self-conditioning | freezes all of `u = p @ E` | freezes `p`, recomputes `p @ E` differentiably | `VEOMNI_ALIGN_NEMO_SELF_COND_EMBED_GRAD` |

The expert scale is a reparameterization: `s(Wh)` and `(sW)h` agree, but AdamW's
per-coordinate step, weight decay and clipping all read the parameter organization, so
one SGD step already separates them. VeOmni's `freeze_router` does not cover
`per_expert_scale`, so it is live in training.

Self-conditioning: neither stack lets the loss reach pass 1, but they cut at different
points. VeOmni's embedding table is not frozen — it still learns from the input and
tied-head paths — it just cannot learn from the hint it supplies to itself. The
published formulation does not fix this boundary, so this is a choice between two
readings, not a correction of one.

Both switches are off by default and live in
`veomni/models/transformers/diffusion_gemma/align_nemo.py`; `dump_forward.py
--align-nemo` sets both. They are attribution knobs, and the fold currently refuses a
sharded model — it rewrites the parameter list, so it has to run after the weights land
and before the optimizer is built, which under EP is a different place than this. Wire
that only if the switches are wanted in a real training run.

With both on, the tiny comparison closes completely at a *non-unit* expert scale, which
the earlier runs could only reach by forcing the scale to 1 with `--unit-expert-scale`:
126/126 gradients within 1e-5 and a worst relative norm of 4.8e-7, against 15 failures
at up to 9.9e-1 with the switches off. The self-conditioning path is the whole of the
embedding table's disagreement there — VeOmni without the switch differs from NeMo by
6.162e-6, the path itself is worth 6.161e-6, and turning it on leaves 9.8e-8.

At 26B the fold removes all 90 `experts.down_proj` gradient mismatches. What is left is
**one near-exact routing tie**, not a semantic difference, and the evidence is direct:
folding changes 3 of 15360 decoder token routings and none at all of the encoder's
28620, and the first of the three sits on a top-k margin of 7.8e-8 at decoder layer 26
— inside fp32's last bit, so any perturbation decides it. The other two are that same
token downstream. It lands as 82% of the squared `decoder_logits` difference on a single
canvas position out of 512, and (through the backward, which carries it into every
earlier layer) as the 25 leftover gradients, all between 1.0e-3 and 2.4e-3. Measured
where nothing flips — the encoder, 30 layers deep — the fold's own numerical cost is
2.7e-6, below the 3.1e-6 the two stacks already disagree by.

So a `--rel-tol 1e-3` forward comparison at 26B is partly a coin toss: on this input the
checkpoint carries 18 of 15360 decoder routing margins below 1e-6 (and 20 of 28620 in
the encoder), and whether the two stacks resolve them the same way is not something
either implementation controls. Only the tightest few actually flip — the fold moves
`down_proj` by ~1e-7 — which is also why the second and third flips sit at margins of
2.4e-4 and 1.3e-6 rather than at a tie: they are the first token's change arriving
downstream, not independent coin tosses.

### The loader bug the real model found

NeMo's `DiffusionGemmaStateDictAdapter` folds `per_expert_scale` into the expert
down-projection *before* upcasting: `_from_hf_w_merged_qkv_and_experts` multiplies the
raw state-dict values and applies `.to(self.dtype)` afterwards. Both operands come from
a bf16 checkpoint, so the product is rounded to 8 mantissa bits and the widening keeps
the rounding — an fp32 run gets an expert down-projection carrying ~1.5e-3 of relative
error, on ~90% of the model's parameters. That is the whole 8% forward gap and all of
the gradient divergence. **The fix belongs in the adapter**, upcasting the operands
before the multiply.

`check_expert_scale_fold.py` calls the real adapter on the real checkpoint and exits
non-zero while the fold is lossy, so it reports what the loader does rather than what
it is read to do. `dump_forward.py --fp32-expert-fold` is the diagnostic arm that
measures the consequence end to end.

The tiny checkpoint cannot see this: it is written in fp32, and the gradient artifact
additionally uses `--unit-expert-scale`, so the fold is exact either way. Re-run step 5
whenever the loader path changes — tiny is a regression net, not evidence about
production.

## Running it

The two stacks pin transformers versions that cannot share a process (VeOmni 5.16, NeMo
5.15+), so this is dump-and-diff across two venvs.

```bash
VE=/opt/tiger/diffusion-gemma-veomni
AM=/opt/tiger/diffusion-gemma-automodel

# 1. Mask semantics. No model, no weights, no GPU. The closed-form half of this runs
#    in CI on its own; NEMO_AUTOMODEL_REPO enables the comparison against NeMo's code.
NEMO_AUTOMODEL_REPO=$AM $VE/.venv/bin/python -m pytest \
    $VE/tests/models/test_diffusion_gemma_mask_matches_nemo.py

# 2. A tiny checkpoint in HF layout, written by transformers' own reference model so
#    that neither stack's idea of the layout defines the artifact. Run under the NeMo
#    venv: the reference model ships in transformers >= 5.15.
$AM/.venv/bin/python $VE/scripts/parity/make_tiny_diffusion_gemma.py \
    --out /tmp/dg_parity/tiny_ckpt
# For gradients, a second one whose per_expert_scale is 1 (see above).
$AM/.venv/bin/python $VE/scripts/parity/make_tiny_diffusion_gemma.py \
    --out /tmp/dg_parity/tiny_unit --unit-expert-scale

# 3. Forward, then gradients. Same script, once per venv.
for stack in veomni nemo; do
  py=$([ $stack = veomni ] && echo $VE/.venv/bin/python || echo $AM/.venv/bin/python)
  $py $VE/scripts/parity/dump_forward.py --stack $stack \
      --ckpt /tmp/dg_parity/tiny_unit --out /tmp/dg_parity/$stack.pt --grads
done
$VE/.venv/bin/python $VE/scripts/parity/compare_dumps.py \
    /tmp/dg_parity/veomni.pt /tmp/dg_parity/nemo.pt --quiet

# 3b. The same at a non-unit expert scale and with self-conditioning live, which is
#     where the two training parameterizations differ. Drop --align-nemo to see the
#     15 tensors that disagree without it.
for stack in veomni nemo; do
  py=$([ $stack = veomni ] && echo $VE/.venv/bin/python || echo $AM/.venv/bin/python)
  align=$([ $stack = veomni ] && echo --align-nemo || echo "")
  $py $VE/scripts/parity/dump_forward.py --stack $stack --ckpt /tmp/dg_parity/tiny_ckpt \
      --out /tmp/dg_parity/sc_$stack.pt --grads --self-conditioning $align
done
$VE/.venv/bin/python $VE/scripts/parity/compare_dumps.py \
    /tmp/dg_parity/sc_veomni.pt /tmp/dg_parity/sc_nemo.pt --quiet

# 4. What bf16 costs on a GPU, which the fp32 steps above say nothing about.
$VE/.venv/bin/python $VE/scripts/parity/measure_hole_kernel_bf16_floor.py

# 5. The same comparison at the shipped 26B checkpoint on real Sudoku rows. CPU/fp32,
#    ~30 min and ~10 GiB of dump per stack. Freeze the inputs first so both stacks are
#    provably fed the same tokens; --check-templates asserts the inlined prompt
#    boundary against VeOmni's chat template and NeMo's jinja file.
CKPT=/mnt/hdfs/user_dir/omni_v2/ckpt/diffusiongemma-26B-A4B-it
$VE/.venv/bin/python $VE/scripts/parity/make_real_inputs.py \
    --jsonl /mnt/hdfs/user_dir/text_diff/data/sudoku/train.jsonl \
    --tokenizer $CKPT --rows 2 --out /tmp/dg_parity/real_inputs.pt --check-templates
for stack in veomni nemo; do
  py=$([ $stack = veomni ] && echo $VE/.venv/bin/python || echo $AM/.venv/bin/python)
  # Drop --fp32-expert-fold to reproduce the 8% gap the shipped loader gives.
  CUDA_VISIBLE_DEVICES="" $py $VE/scripts/parity/dump_forward.py --stack $stack \
      --ckpt $CKPT --inputs /tmp/dg_parity/real_inputs.pt --grads --fp32-expert-fold \
      --out /tmp/dg_parity/real/$stack.pt
done
$VE/.venv/bin/python $VE/scripts/parity/compare_dumps.py \
    /tmp/dg_parity/real/veomni.pt /tmp/dg_parity/real/nemo.pt --rel-tol 1e-3

# 6. The same with VeOmni on NeMo's parameterization, which is what removes the 90
#    `down_proj` gradient mismatches step 5 leaves behind.
CUDA_VISIBLE_DEVICES="" $VE/.venv/bin/python $VE/scripts/parity/dump_forward.py --stack veomni \
    --ckpt $CKPT --inputs /tmp/dg_parity/real_inputs.pt --grads --align-nemo \
    --out /tmp/dg_parity/real/veomni_align.pt
$VE/.venv/bin/python $VE/scripts/parity/compare_dumps.py \
    /tmp/dg_parity/real/veomni_align.pt /tmp/dg_parity/real/nemo.pt --rel-tol 1e-3
```

### Kept baselines

A 26B run costs ~30 min per stack, so the forward-only dumps are kept rather than
re-made. Without `--grads` a dump holds just `encoder_hidden` and `decoder_logits` —
about 550 MiB, against the ~10 GiB the gradients add — which is cheap enough to leave
on disk:

```
/opt/tiger/dg_parity_baselines/
    26b_nemo.pt           # --fp32-expert-fold, i.e. NeMo with the loader defect fixed
    26b_veomni_base.pt    # VeOmni's own parameterization
    26b_veomni_align.pt   # --align-nemo
    real_inputs.pt        # the frozen Sudoku rows all three were fed
```

Every dump carries a `from` key recording the checkpoint and its config hash, the input
file and its hash, the alignment switches, the torch version and the git revision of
whichever stack produced it. Without that a kept dump is a number whose meaning has to
be reconstructed from memory — and these files outlive the shell that made them by
weeks. `compare_dumps.py` prints both sides' command lines for the same reason.

`--elementwise` reports mean and max of the per-element absolute and relative
difference instead of the norm ratio. The two answer different questions, and the
relative columns come in pairs because near-zero elements dominate an unrestricted
mean: on the tiny model `mean rel` is 7.6e-06 over every element of `encoder_hidden`
but 1.7e-07 over the 96% of elements carrying at least 1% of the tensor's RMS.

`--canvases` and `--prefix` on `dump_forward.py` vary the geometry; `--prefix 5 11`
gives a batch with uneven prompts, which is the case where the two stacks describe the
encoder's tail padding differently. `--layer-trace` and `--trace-layer N` dump per-layer
and per-submodule hidden states, which is how the 26B divergence was localised to one
module in layer 0. At 26B the `(experts, ., .)` gradients are dumped as row/column norm
profiles rather than in full — `--full-grads` overrides that, and costs ~91 GiB.

## Reading the numbers

Steps 1–3 and 5 run on CPU in fp32 because bf16 puts a ~1e-2 floor under a comparison
trying to resolve 1e-5 — a disagreement that large is indistinguishable from a real
semantic difference. That makes them a statement about *semantics* and not about
training.

At 26B, quote the fp32 accumulation profile rather than a single tolerance: the
per-layer error rises monotonically from 1.9e-7 at layer 0 to 3.1e-6 at layer 29. That
is the floor for a 30-layer model, and it is what "the two agree" means here — not the
1e-8 the 5-layer tiny artifact reaches.

The leftover `experts.down_proj` gradient gap in step 5 has to be divided out before it
can be read. The two stacks differentiate with respect to different parameters — `W`
with a live scale on VeOmni's side, `W' = sW` on NeMo's — so the gradients differ by `s`
per expert by construction. On this checkpoint `s` is in [0.9805, 1.0234], which makes
the raw gap a deceptively small 1-2%; rescaling by `s` takes the worst of the 30 layers
from 1.5e-2 to 5.1e-5.

Step 4 is the bridge to training, and its last row is the one to quote: the two decoder
attention paths sit 2.8e-3 – 7.7e-3 from each other in bf16 at the shipped geometry.
Any difference observed between two bf16 training runs has to be read against that,
not against the fp32 tolerances above. Note also that rounding the *inputs* to bf16
costs 3.5e-2 – 7.4e-2 before either path runs — an order of magnitude more than either
path's own arithmetic error, and attributable to neither.
