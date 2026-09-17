---
name: veomni-gpu-to-npu
description: "Migrate an existing VeOmni GPU model to Ascend NPU. Use when the GPU patchgen path already works and the task needs dependency scanning, NPU operator selection, patchgen, runtime configuration, tests, E2E accuracy comparison, or MFU evidence. Gemma 3 is the worked example."
---

# VeOmni GPU-to-NPU Migration

Produce the smallest maintainable NPU implementation for a model that already
trains through VeOmni's GPU path. Keep reusable procedure and evidence templates
in this skill. Commit only artifacts required by the repository or issue; runtime
configs, launch scripts, raw logs, plots, and reports may instead be attached to
the PR when maintainers request a minimal model diff.

Read these before changing code:

- `.agents/knowledge/constraints.md`
- `.agents/skills/veomni-patchgen-model/SKILL.md`
- `docs/design/patchgen.md`
- `docs/design/kernel_selection.md`
- `.agents/knowledge/testing.md`

Use `veomni/models/transformers/qwen3/`, `seed_oss/`, `qwen3_vl/`, and
`qwen3_5/` as reference implementations. Gemma 3 under
`veomni/models/transformers/gemma3/` is the dense text/VLM worked example.

## 1. Define The Deliverables

Start by printing a file manifest split into two groups:

1. Repository artifacts: NPU patch spec, generated `.py` and `.diff`, registry
   dispatch, and any existing CI test that must be extended.
2. Validation artifacts: exact commands, temporary config overrides, raw logs,
   loss comparison, MFU calculation, plots, and environment metadata.

Do not assume every validation artifact belongs in git. Follow the issue and
maintainer direction. Never commit checkpoints, datasets, raw logs, local paths,
one-off launch scripts, or generated plots unless explicitly requested.

Confirm the GPU path and pinned environment:

```bash
python -c "import transformers; print(transformers.__version__)"
rg -n "transformers==" pyproject.toml
ls veomni/models/transformers/<model>/{__init__.py,*_gpu_patch_gen_config.py}
ls veomni/models/transformers/<model>/generated/patched_modeling_*_gpu.py
```

The installed Transformers version must equal the repository pin before
running patchgen.

## 2. Audit GPU Assumptions

Scan the patch spec, generated model, upstream Transformers model, registry,
and training config. Record each match and its NPU decision.

```bash
rg -n "cuda|triton|flash_attention|flex_attention|liger|is_cuda|\.cuda\(" \
  veomni/models/transformers/<model> \
  configs

rg -n "OpSlot|ops_implementation|attn_implementation|RMSNorm|apply_rotary" \
  veomni/models/transformers/<model> \
  veomni/ops/kernels
```

For every operator, verify the live registry rather than relying on this table:

| Operation | Typical GPU backend | Typical NPU choice |
|---|---|---|
| attention | `flex_attention` / flash attention | `sdpa` or `eager` |
| RMSNorm | `liger_kernel` | `npu` when the formula matches |
| rotary embedding | `liger_kernel` | `npu` |
| causal cross entropy | `liger_kernel` | `npu`, `chunk_loss`, or `eager` |
| SwiGLU | fused GPU backend | registered NPU backend or `eager` |
| MoE | `fused_triton` | `fused_npu` |
| load-balancing loss | `triton` | registered NPU backend or `eager` |

Also check device-neutral behavior: allocations must derive their device from
inputs/parameters, no `.cuda()` calls may remain, dtype casts must preserve the
model contract, and distributed collectives must use VeOmni's current parallel
state APIs.

Model-specific normalization semantics are correctness-critical. Standard
RMSNorm initializes weights to ones and passes `weight`; Gemma 3 initializes
weights to zeros and computes with `1 + weight`, so its NPU kernel call must
also receive `1 + self.weight`.

## 3. Implement The NPU Patch

Prefer an NPU sibling patch spec that reuses the GPU patch bodies and changes
only device-specific operators:

```python
from veomni.models.transformers.<model>.<model>_gpu_patch_gen_config import (
    config as gpu_config,
    <shared_patch_function>,
)
from veomni.patchgen.patch_spec import PatchConfig

config = PatchConfig(
    source_module="transformers.models.<model>.modeling_<model>",
    target_file="patched_modeling_<model>_npu.py",
    description="<Model> with VeOmni NPU operator replacements",
)
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)
```

Use an `OpSlot` guard when eager fallback is meaningful on CPU/GPU tests:

```python
@config.override_method("<Model>RMSNorm.forward", description="Use NPU RMSNorm through OpSlot")
def rmsnorm_forward_npu(self, x):
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(x, self.weight, self.variance_epsilon)
    return <upstream eager implementation>
```

Use a direct NPU kernel only when the surrounding model family already uses
that pattern and CPU fallback is not required. Do not duplicate a GPU patch
body merely to change its target file.

Update `__init__.py` at import time:

```python
from ....utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    from .generated.patched_modeling_<model>_npu import <classes>
else:
    from .generated.patched_modeling_<model>_gpu import <classes>
```

Register every supported `model_type`. A VLM commonly needs a second registry
entry for the multimodal wrapper in addition to its text backbone. Preserve the
existing architecture fallback behavior.

For multimodal models, separately audit the vision/audio tower, placeholder
masks, packed metadata, sequence-parallel boundaries, and the outer
`ForConditionalGeneration.forward`. Read
`.agents/skills/veomni-patchgen-model/references/multimodal.md` before changing
those paths.

Generate files; never edit `generated/` by hand:

```bash
patchgen veomni.models.transformers.<model>.<model>_npu_patch_gen_config \
  -o veomni/models/transformers/<model>/generated --diff -v
git diff --check
```

Then rerun the same command and assert it leaves the generated files unchanged.

## 4. Configure Without Committing A Config

Start from the existing GPU YAML and override only hardware-specific values at
launch. Adapt field paths to the current argument schema:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/train_text.py configs/text/<model>.yaml \
  --model.model_path /path/to/model \
  --data.train_path /path/to/data \
  --model.ops_implementation.attn_implementation sdpa \
  --model.ops_implementation.rms_norm_implementation npu \
  --model.ops_implementation.rotary_pos_emb_implementation npu \
  --model.ops_implementation.cross_entropy_loss_implementation npu \
  --model.accelerator.ulysses_size 1 \
  --train.max_steps 20 \
  --train.checkpoint.save_steps 0 \
  --train.checkpoint.save_hf_weights false \
  --train.checkpoint.output_dir /tmp/<model>-npu-smoke
```

Validate backend availability with the current operator registry. Use `eager`
for an unsupported fused operator. If the model enables logit soft-capping,
confirm whether the selected fused loss implements it; otherwise select
`chunk_loss` or `eager`.

## 5. Validate In Layers

Run the narrowest useful checks first:

```bash
python -m py_compile \
  veomni/models/transformers/<model>/<model>_npu_patch_gen_config.py \
  veomni/models/transformers/<model>/generated/patched_modeling_<model>_npu.py

python -c "import veomni.models.transformers.<model>.generated.patched_modeling_<model>_npu"
make quality
```

Extend an existing CI-enumerated test when practical. Test at least:

- generated module import and expected classes;
- NPU/GPU registry dispatch;
- fused-op guard and eager fallback parity;
- forward/backward with finite loss and gradients;
- patchgen reproducibility.

If no repository test is added because the requested diff is model-and-skill
only, say so explicitly in the PR and provide the exact commands and observed
results. Hardware E2E evidence does not replace import and quality checks.

## 6. E2E Accuracy Protocol

Run GPU and NPU with the same model checkpoint, ordered dataset, seed, batch
sizes, sequence length, optimizer, learning-rate schedule, precision, and step
count. Only hardware operator choices should differ. Disable checkpoint saves
for measurement runs.

Record per step:

- loss and gradient norm;
- effective tokens and step time;
- allocated/reserved memory;
- any NaN, Inf, OOM, hang, or retry.

Report the first loss, final loss, mean loss, and a clearly defined alignment
metric. Symmetric relative error is robust near zero:

```text
relative_error(a, b) = 2 * abs(a - b) / (abs(a) + abs(b) + 1e-12)
```

State the acceptance threshold before interpreting the result. Plot both loss
series against optimizer step using Python/matplotlib, upload the image to the
PR, and keep the raw logs outside git.

## 7. MFU Protocol

Measure after compilation/warmup and exclude initialization, checkpointing,
evaluation, and cache-cleanup outliers. Report median step time and effective
tokens/s over the stated window.

Use a model-appropriate training FLOP formula. For a dense decoder, document
linear-layer forward/backward FLOPs plus attention-score FLOPs; for sliding
window attention, use the actual window for those layers. For VLMs include the
vision tower only when it is trained. Then calculate:

```text
achieved_FLOP/s = training_FLOPs_per_step / median_step_time
MFU = achieved_FLOP/s / (device_count * peak_BF16_FLOP/s_per_device)
```

The PR must name the accelerator, device count, peak value source, model,
precision, sequence length, global/micro batch size, warmup/exclusion window,
formula, achieved throughput, and MFU. Do not add a shared FLOPs estimator only
to display one acceptance result; change shared infrastructure only when the
runtime feature itself requires it and add focused tests.

## 8. PR Evidence Template

Put this material in the PR body (or a requested report), aligned one-to-one
with the issue acceptance criteria:

```markdown
### Migration scope
- Dependency scan: <findings and decisions>
- Repository files: <minimal file list>
- Reproduction commands: <GPU and NPU commands>

### Accuracy
| Metric | GPU | NPU | Difference |
|---|---:|---:|---:|
| First loss | | | |
| Final loss | | | |
| Mean loss | | | |

<uploaded loss curve>

### Performance
| Item | Value |
|---|---:|
| Hardware / devices | |
| Model / precision | |
| Sequence / global / micro batch | |
| Measurement window | |
| Median step time / tokens per second | |
| FLOPs formula / peak per device | |
| MFU | |

### Verification
- `patchgen ...`: PASS
- `python -m py_compile ...`: PASS
- `make quality`: PASS
- NPU E2E: PASS, <steps>, no runtime errors
```

## Completion Checklist

- [ ] GPU baseline and pinned Transformers version confirmed
- [ ] CUDA/GPU-only dependencies classified
- [ ] Device-neutral APIs and distributed assumptions checked
- [ ] NPU backends verified in the live registry
- [ ] Model-specific normalization/loss semantics preserved
- [ ] Patch spec, generated files, and import-time dispatch implemented
- [ ] Text and multimodal model types covered as applicable
- [ ] Patchgen rerun is reproducible
- [ ] Import, quality, and focused correctness checks pass
- [ ] GPU/NPU E2E uses matched inputs and hyperparameters
- [ ] Loss curve and numerical comparison reported
- [ ] MFU setup, formula, peak hardware value, and result reported
- [ ] PR diff contains no configs, scripts, logs, datasets, plots, or checkpoints unless requested
