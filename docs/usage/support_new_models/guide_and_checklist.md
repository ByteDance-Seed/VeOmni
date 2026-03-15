# Support New Models — Guide and Checklist

**TLDR:** VeOmni patches HuggingFace models to add FSDP, Sequence Parallelism (SP), Expert Parallelism (EP), and fused kernels for distributed post-training. This guide walks you through the integration steps with checklists per model type. For worked examples, see:
- [qwen3_vl_example.md](./qwen3_vl_example.md) — VLM + MoE (image/video, deepstack, EP)
- [qwen3_omni_moe_example.md](./qwen3_omni_moe_example.md) — Omni-modal MoE (image/video/audio, talker)

> **Version note:** This guide covers both **transformers v4** (runtime monkey-patching) and **transformers v5** (patchgen code generation) workflows. The v5 approach is recommended for new models. See [patchgen.md](../../transformers_v5/patchgen.md) for details on the code generation framework.

---

## Integration Complexity by Model Type

| Model Type | Files Required | Key Additions |
|---|---|---|
| Dense text-only LLM | `__init__.py` | SP position embedding slicing |
| VLM (image/video) | `__init__.py` + `modeling_*.py` | FSDP dummy forward, SP in ViT + LM, position ID func |
| Omni-modal MoE | `__init__.py` + 4 more files | All of the above + audio encoder, fused MoE, EP plan, processor patch |

---

## Staged Onboarding Workflow

New model onboarding follows 6 stages, each with a test gate that must pass before moving on. This ensures incremental validation and catches issues early.

### Stage 1: Model Understanding and Setup

**Goal:** Understand the target model and create the minimal test infrastructure.

Before writing any VeOmni code, answer:

1. `model_type` in `config.json`? — your registry key
2. `architectures[0]` in `config.json`? — selects the model class
3. Processor class in `processor_config.json`? — `MODEL_PROCESSOR_REGISTRY` key
4. MoE? — needs `parallel_plan.py`
5. Multimodal (image/video/audio)? — needs processor patch and data transform
6. Multimodal RoPE? — needs `get_position_id_func`
7. `tie_word_embeddings`? — needs special FSDP handling

**Checklist:**

- [ ] Identify `model_type`, `architectures`, and special features (MoE, multimodal, tied embeddings)
- [ ] Create toy config in `tests/toy_config/<model>_toy/config.json` with minimal layers/hidden dims
- [ ] Add a `README.md` in the toy config folder noting the source config and changes
- [ ] Verify toy config loads: `AutoModel.from_config(AutoConfig.from_pretrained("tests/toy_config/<model>_toy/"))`

### Stage 2: Data Pipeline

**Goal:** Implement and validate data preprocessing for the model.

- [ ] Implement/verify data transform (for multimodal: `process_sample_*` in `veomni/data/multimodal/data_transform.py`)
- [ ] Add `DummyDataset` class in `veomni/data/dummy_dataset.py` (for multimodal models)
- [ ] Add `MODEL_TO_DATASET` entry in `tests/models/utils.py`

**Test gate — run Level 0 tests:**

```bash
pytest -m L0 -k "<model_name>"
```

Level 0 validates that data processing, tokenization, and multimodal input handling produce correct outputs. Tests are CPU-only and fast.

### Stage 3: Model Patch Implementation

**Goal:** Implement the model patches for custom attention, MoE, and multimodal support.

#### For transformers v4 (runtime monkey-patching)

Create the model directory and registration files:

```bash
mkdir veomni/models/transformers/your_model_name/
touch veomni/models/transformers/your_model_name/__init__.py
# For complex models:
touch veomni/models/transformers/your_model_name/modeling_your_model_name.py
touch veomni/models/transformers/your_model_name/processing_your_model_name.py    # if multimodal
touch veomni/models/transformers/your_model_name/parallel_plan.py                 # if MoE
```

Register using the `@REGISTRY.register` decorators (see registry examples below).

Patch the model by importing the HF module and overriding methods:

```python
import transformers.models.your_model.modeling_your_model as hf_your_model

def apply_veomni_patch():
    hf_your_model.YourClass.method = patched_method
```

#### For transformers v5 (patchgen code generation)

Create a patch configuration file:

```bash
touch veomni/models/transformers/your_model_name/your_model_gpu_patch_gen_config.py
mkdir veomni/models/transformers/your_model_name/patches/
touch veomni/models/transformers/your_model_name/patches/your_model_gpu_patches.py
```

Define patches declaratively using `PatchConfig`:

```python
from veomni.patchgen.patch_spec import PatchConfig

config = PatchConfig(
    source_module="transformers.models.your_model.modeling_your_model",
    target_file="patched_modeling_your_model_gpu.py",
    description="Your model GPU patches",
)

@config.replace_class("YourModelRMSNorm", description="Use fused kernel")
class OptimizedRMSNorm(nn.Module):
    ...

@config.override_method("YourModelAttention.forward", description="Add Ulysses SP")
def ulysses_attention_forward(self, hidden_states, ...):
    ...
```

Generate the patched modeling code:

```bash
python -m veomni.patchgen.run_codegen veomni.models.transformers.your_model.your_model_gpu_patch_gen_config --diff -v
```

See [patchgen.md](../../transformers_v5/patchgen.md) for the full reference.

**Checklist (both v4 and v5):**

- [ ] Implement attention patch (support all backends: eager, flash_attention_2, flash_attention_3)
- [ ] Implement MoE patch (if applicable): stacked expert weights + `fused_moe_forward`
- [ ] Implement multimodal processor patch (if applicable)
- [ ] Register in model registry (`MODELING_REGISTRY`, `MODEL_CONFIG_REGISTRY`, `MODEL_PROCESSOR_REGISTRY`)
- [ ] Add module to `veomni/models/transformers/__init__.py`

**Test gate — run Level 1 tests:**

```bash
pytest -m L1 -k "<model_name>"
```

Level 1 validates forward/backward consistency across all attention and MoE backends on a single GPU.

### Stage 4: Distributed Training Support

**Goal:** Enable FSDP, Sequence Parallelism, and Expert Parallelism.

- [ ] Define FSDP parallel plan (for MoE: `parallel_plan.py` with expert weight FQN patterns)
- [ ] Handle FSDP dummy forward (for multimodal/uneven batches — prevents NCCL hangs)
- [ ] Handle tied embeddings (if `tie_word_embeddings=True`: patch config or add FSDP handling)
- [ ] Implement SP modifications in attention forward (see [Ulysses guide](../../key_features/ulysses.md))
- [ ] Implement EP support (for MoE models, see [EP+FSDP2 guide](../../key_features/ep_fsdp2.md))

**Test gate — run Level 2 tests:**

```bash
pytest -m L2 -k "<model_name>"
```

Level 2 validates that FSDP-wrapped training matches single-GPU numerics (loss, gradients).

**Test gate — run Level 3 tests:**

```bash
pytest -m L3 -k "<model_name>"
```

Level 3 validates that SP/EP combinations on top of FSDP produce numerically equivalent results. Combinations tested:
- FSDP only (baseline)
- FSDP + SP (`sp_size=2`)
- FSDP + EP (`ep_size=2`, MoE only)
- FSDP + SP + EP (MoE only)

### Stage 5: Robustness and Edge Cases

**Goal:** Validate corner cases that commonly cause production failures.

- [ ] Dummy forward: ranks with no data (e.g., no images in a VLM batch) must still produce valid gradients
- [ ] Checkpoint save/load roundtrip: save mid-training, reload, verify loss continuity
- [ ] Variable-length sequences: packing multiple sequences, padding edge cases
- [ ] Mixed-precision edge cases (if applicable)

**Test gate — run Level 4 tests:**

```bash
pytest -m L4 -k "<model_name>"
```

### Stage 6: Integration

**Goal:** End-to-end validation and CI integration.

- [ ] Add training config in `configs/` (YAML with model path, attention/MoE implementation, SP/EP sizes)
- [ ] Hook into trainer (`vlm_trainer.py`: `build_model_assets`, `build_data_collate_info`, `build_data_transform`)

**Test gate — run Level 5 tests:**

```bash
pytest -m L5 -k "<model_name>"
```

Level 5 runs a short end-to-end training loop (2 epochs, few steps) and verifies no crashes, finite loss, and reasonable metrics.

- [ ] Add to CI nightly suite

---

## Model Registration Reference

### Minimal registration (text-only):

```python
from ...loader import MODELING_REGISTRY

@MODELING_REGISTRY.register("your_model_type")
def register_modeling(architecture: str):
    from transformers.models.your_model import YourModelForCausalLM
    return YourModelForCausalLM
```

### Full registration (multimodal MoE):

```python
from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY

@MODEL_CONFIG_REGISTRY.register("your_model_type")
def register_config():
    from .configuration_your_model import YourModelConfig, apply_veomni_patch
    apply_veomni_patch()
    return YourModelConfig

@MODELING_REGISTRY.register("your_model_type")
def register_modeling(architecture: str):
    from .modeling_your_model import YourModelForCausalLM, apply_veomni_patch
    apply_veomni_patch()
    return YourModelForCausalLM

@MODEL_PROCESSOR_REGISTRY.register("YourModelProcessor")  # exact class name from processor_config.json
def register_processor():
    from .processing_your_model import YourModelProcessor, apply_veomni_patch
    apply_veomni_patch()
    return YourModelProcessor
```

> **Registry key rules:**
> - `MODELING_REGISTRY` and `MODEL_CONFIG_REGISTRY`: use `model_type` from `config.json`
> - `MODEL_PROCESSOR_REGISTRY`: use the Python class name string from `processor_config.json`

---

## Patch Reference (Quick Table)

| Patch | Text LLM | VLM | Omni MoE |
|---|:---:|:---:|:---:|
| `tie_word_embeddings` config fix | sometimes | sometimes | yes |
| FSDP dummy forward | — | yes | yes (ViT + Audio) |
| SP: LM position embedding slicing | yes | yes | yes |
| SP: ViT pad+slice | — | yes | yes |
| SP: `cu_seqlens` padding entry | — | yes | yes |
| SP: ViT-to-LM fill-back | — | yes | yes |
| SP: deepstack all-gather | — | if deepstack | yes |
| Fused MoE + stacked weights | — | if MoE | yes |
| Flash-attn kwargs pop/restore | — | yes | yes |
| Pre-compute `max_seqlen` | — | yes | yes |
| Position ID transposition | — | yes | yes |
| `ForCausalLMLoss` | yes | yes | yes |
| `get_position_id_func` | — | yes | yes |

For implementation details of each patch, refer to the example docs.

---

## Checklists by Model Type

### Any New Model

- [ ] `veomni/models/transformers/your_model/__init__.py` with `@MODELING_REGISTRY.register`
- [ ] `veomni/models/transformers/__init__.py` updated
- [ ] Toy config in `tests/toy_config/your_model_toy/`

### VLMs (image/video)

- [ ] FSDP `dummy_forward` in ViT encoder
- [ ] SP `sp_pad_and_slice` in ViT (correct `pad_scale`)
- [ ] SP `cu_seqlens` padding entry
- [ ] SP ViT-to-LM fill-back (`gather_seq_scatter_heads` / `gather_heads_scatter_seq`)
- [ ] `get_position_id_func` using VeOmni token ID constants
- [ ] `process_sample_*` in `data_transform.py`; `build_data_transform` in `VLMTrainer`

### MoE Models

- [ ] `parallel_plan.py` with correct expert weight paths
- [ ] `get_parallel_plan` wired on the pretrained model base class
- [ ] Stacked-weight `YourModelExperts` module + `fused_moe_forward`
- [ ] `_moe_implementation` propagated from top-level config to text sub-config
- [ ] `_init_weights` patched for stacked expert params

### Omni-modal (audio)

- [ ] FSDP `dummy_forward` in audio encoder
- [ ] SP gather/slice in audio encoder (`gather_outputs` + `slice_input_tensor`)
- [ ] `audio_mask` in data transform; `audio_feature_lengths` in `build_data_collate_info`
- [ ] Processor patched: `if audios:` truthy check

### Testing (all models)

- [ ] Toy config in `tests/toy_config/your_model_toy/`
- [ ] `DummyDataset` entry in `veomni/data/dummy_dataset.py` (if multimodal)
- [ ] `MODEL_TO_DATASET` entry in `tests/models/utils.py`
- [ ] **L0**: Data transform alignment test (if custom processor)
- [ ] **L1**: `pytest.param` entry in `test_models_patch.py` (forward/backward across backends)
- [ ] **L2**: FSDP equivalence test (single-GPU vs FSDP numerics)
- [ ] **L3**: Test case in `test_e2e_parallel.py` (SP/EP combinations)
- [ ] **L4**: Dummy forward test (if VLM/omni)
- [ ] **L4**: Checkpoint save/load roundtrip test (if new weight layout)
- [ ] **L5**: End-to-end smoke test config

---

## Common Pitfalls

| Symptom | Likely Cause | Fix |
|---|---|---|
| NCCL hang during backward | Missing `dummy_forward` on ViT/AudioEncoder | Add and call on `fsdp_enabled` ranks when input is `None` |
| Shape mismatch in ViT attention | `cu_seqlens` missing padding entry for SP | Append `cu_seqlens[-1] + pad_seq_len` when SP is active |
| `masked_scatter` size error | Fill-back attempted in SP-sliced layout | Call `gather_seq_scatter_heads` before fill-back |
| Crash: `tie_word_embeddings` | Config default `True` but no `get_output_embeddings` | Patch config to `tie_word_embeddings=False` |
| Wrong position IDs in multi-sample batch | `(bs, 3, L)` not transposed to `(3, bs, L)` | Add transpose check in model forward |
| Audio inputs silently skipped | `if audio is not None:` passes for empty list `[]` | Change to `if audio:` in processor |
| EP has no effect | Expert weight paths in `parallel_plan` don't match | Run `named_parameters()` on model to verify exact paths |
| Fused MoE produces wrong outputs | Weight shape/transpose mismatch | Verify `(num_experts, out, in)` convention; check `.contiguous()` |
| Patchgen drift in CI | Generated code out of date with HF or patches | Run `python -m veomni.patchgen.check_patchgen --fix` |

---

## Key Imports

```python
from veomni.distributed.parallel_state import get_parallel_state

from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,   # (bs, seq, h//sp) -> (bs, seq//sp, h)
    gather_outputs,             # all-gather along a dim (no autograd)
    gather_seq_scatter_heads,   # (bs, seq//sp, h) -> (bs, seq, h//sp)
    slice_input_tensor,         # slice along a dim for this SP rank
    sp_pad_and_slice,           # pad to multiple of pad_scale, then slice
    unpad_tensor,               # remove padding from a tensor
)
from veomni.distributed.sequence_parallel.ulysses import _Gather  # all-gather with autograd

from veomni.ops import fused_moe_forward
from veomni.ops.fused_cross_entropy import ForCausalLMLoss

from veomni.utils.constants import (
    AUDIO_INPUT_INDEX,   # placeholder token ID for audio in input_ids
    IGNORE_INDEX,        # -100, label mask value
    IMAGE_INPUT_INDEX,   # placeholder token ID for images in input_ids
    VIDEO_INPUT_INDEX,   # placeholder token ID for videos in input_ids
)
```
