# Support New Models — Guide and Checklist

**TLDR:** VeOmni patches HuggingFace models at runtime to add FSDP, Sequence Parallelism (SP), Expert Parallelism (EP), and fused kernels. This guide walks you through the integration steps with checklists per model type. For worked examples, see:
- [qwen3_vl_example.md](./qwen3_vl_example.md) — VLM + MoE (image/video, deepstack, EP)
- [qwen3_omni_moe_example.md](./qwen3_omni_moe_example.md) — Omni-modal MoE (image/video/audio, talker)

> **Scope note:** This guide currently targets the **transformers v4** integration/patchgen flow.
> **TODO:** Add a dedicated **transformers v5** section, since modeling code patchgen requires a slightly different approach.

---

## Integration Complexity by Model Type

| Model Type | Files Required | Key Additions |
|---|---|---|
| Dense text-only LLM | `__init__.py` | SP position embedding slicing |
| VLM (image/video) | `__init__.py` + `modeling_*.py` | FSDP dummy forward, SP in ViT + LM, position ID func |
| Omni-modal MoE | `__init__.py` + 4 more files | All of the above + audio encoder, fused MoE, EP plan, processor patch |

---

## Step-by-Step Integration

### Step 0: Understand the Target Model

Before writing any VeOmni code, answer:

1. `model_type` in `config.json`? → your registry key
2. `architectures[0]` in `config.json`? → selects the model class
3. Processor class in `processor_config.json`? → `MODEL_PROCESSOR_REGISTRY` key
4. MoE? → needs `parallel_plan.py`
5. Multimodal (image/video/audio)? → needs processor patch and data transform
6. Multimodal RoPE? → needs `get_position_id_func`

### Step 1: Create the Model Directory

```bash
mkdir veomni/models/transformers/your_model_name/
touch veomni/models/transformers/your_model_name/__init__.py
# For complex models, also add:
touch veomni/models/transformers/your_model_name/modeling_your_model_name.py
touch veomni/models/transformers/your_model_name/configuration_your_model_name.py  # if config fix needed
touch veomni/models/transformers/your_model_name/processing_your_model_name.py    # if multimodal
touch veomni/models/transformers/your_model_name/parallel_plan.py                 # if MoE
```

### Step 2: Register Your Model (`__init__.py`)

**Minimal (text-only):**
```python
from ...loader import MODELING_REGISTRY

@MODELING_REGISTRY.register("your_model_type")
def register_modeling(architecture: str):
    from transformers.models.your_model import YourModelForCausalLM
    return YourModelForCausalLM
```

**Full (multimodal MoE):**
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

### Step 3: Add to Package `__init__.py`

Add your module to [veomni/models/transformers/__init__.py](../../../veomni/models/transformers/__init__.py):

```python
from . import (
    # ... existing models ...
    your_model_name,  # ADD THIS
)
```

### Step 4: Patch the Model (`modeling_*.py`)

Standard pattern — import HF module as alias, define patches, apply at end:

```python
import transformers.models.your_model.modeling_your_model as hf_your_model

# ... define patches ...

def apply_veomni_patch():
    hf_your_model.YourClass.method = patched_method
```

Which patches to apply depends on model type (see checklist below). For implementation details of each patch, see the example docs.

### Step 5: Define Expert Parallelism Plan (`parallel_plan.py`, MoE only)

```python
from torch.distributed._tensor import Shard
from ....distributed.parallel_plan import ParallelPlan

def get_parallel_plan():
    ep_plan = {
        "model.layers.*.mlp.experts.gate_proj": Shard(0),
        "model.layers.*.mlp.experts.up_proj":   Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    return ParallelPlan(extra_parallel_plan={"ep": ep_plan})
```

> **Finding correct paths:** run `for name, _ in model.named_parameters(): print(name)` on the unpatched HF model.

### Step 6: Patch the Processor (`processing_*.py`, multimodal only)

Two common issues:
1. HF checks `if audio is not None:` — VeOmni passes `[]` for absent inputs → override with `if audio:`
2. Keyword argument mismatch (`audios=` vs `audio=`) — match what `data_transform.py` passes

### Step 7: Write the Data Transform Function

Add `process_sample_your_model()` to [veomni/data/multimodal/data_transform.py](../../../veomni/data/multimodal/data_transform.py). See the example docs for the full function signature and steps.

### Step 8: Hook into the Trainer

Edit [veomni/trainer/vlm_trainer.py](../../../veomni/trainer/vlm_trainer.py). Add your model type to `build_model_assets`, `build_data_collate_info`, `build_data_transform`, and optionally `freeze_module` / `build_param_groups`.

### Step 9: Add a Config File

Create `configs/multimodal/your_model/your_model.yaml` with `model.config_path`, `model.attn_implementation`, `model.moe_implementation`, `train.sp_size`, `train.ep_size`.

### Step 10: Test

See the testing checklist below for what to add.

---

## Patch Reference (Quick Table)

| Patch | Text LLM | VLM | Omni MoE |
|---|:---:|:---:|:---:|
| `tie_word_embeddings` config fix | sometimes | sometimes | ✓ |
| FSDP dummy forward | — | ✓ | ✓ (ViT + Audio) |
| SP: LM position embedding slicing | ✓ | ✓ | ✓ |
| SP: ViT pad+slice | — | ✓ | ✓ |
| SP: `cu_seqlens` padding entry | — | ✓ | ✓ |
| SP: ViT-to-LM fill-back | — | ✓ | ✓ |
| SP: deepstack all-gather | — | if deepstack | ✓ |
| Fused MoE + stacked weights | — | if MoE | ✓ |
| Flash-attn kwargs pop/restore | — | ✓ | ✓ |
| Pre-compute `max_seqlen` | — | ✓ | ✓ |
| Position ID transposition | — | ✓ | ✓ |
| `ForCausalLMLoss` | ✓ | ✓ | ✓ |
| `get_position_id_func` | — | ✓ | ✓ |

For implementation details of each patch, refer to the example docs.

---

## Checklists

### Any New Model

- [ ] `veomni/models/transformers/your_model/__init__.py` with `@MODELING_REGISTRY.register`
- [ ] `veomni/models/transformers/__init__.py` updated

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
- [ ] `DummyYourModelDataset` in `veomni/data/dummy_dataset.py` (multimodal)
- [ ] `MODEL_TO_DATASET` entry in `tests/models/utils.py`
- [ ] `pytest.param` in `test_cases` in `tests/models/test_models_patch.py` (Level 1)
- [ ] Test case + fixture + test function in `tests/e2e/test_e2e_parallel.py` (Level 2)
- [ ] For VLM models, add the toy config to the `freeze_vit` smoke test list in `tests/models/test_vlm_trainer.py`

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

---

## Key Imports

```python
from veomni.distributed.parallel_state import get_parallel_state

from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,   # (bs, seq, h//sp) → (bs, seq//sp, h)
    gather_outputs,             # all-gather along a dim (no autograd)
    gather_seq_scatter_heads,   # (bs, seq//sp, h) → (bs, seq, h//sp)
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
