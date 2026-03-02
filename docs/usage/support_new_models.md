# Support New Models

**Authors**: Juntian Liu, Ting Yang

**TLDR:** This guide walks you through integrating any HuggingFace model into VeOmni's distributed training framework — from file creation to data pipeline hookup. We use Qwen3-Omni-MoE as the primary reference and Qwen3-VL MoE for additional examples.

---

## Overview

VeOmni does **not** rewrite HuggingFace models from scratch. Instead, it **patches** them at runtime to add:

- **FSDP** — shard parameters, gradients, and optimizer states across GPUs; fix reduce-scatter hangs in multimodal models
- **SP (Sequence Parallelism)** — distribute input along the sequence dimension across GPUs
- **EP (Expert Parallelism)** — shard MoE expert weights across GPUs
- **Fused kernels** — replace eager MoE with fast triton kernels; Liger cross-entropy
- **VeOmni data format** — pre-computed masks, position IDs, packed sequences

The integration complexity scales with the model:

| Model Type | Files Required | Key Additions |
|---|---|---|
| Dense text-only LLM | `__init__.py` | Minimal SP patches to attention/rotary emb |
| Vision-Language (VLM) | `__init__.py` + `modeling_*.py` | SP in ViT + LM, FSDP dummy forwards, position ID func |
| Omni-modal MoE | `__init__.py` + 4 more files | All of the above + audio encoder, fused MoE, EP plan, processor patch |

---

## Part 1: Step-by-Step Integration

### Step 0: Understand the Model

Before touching any VeOmni code, answer these questions about the target HuggingFace model:

1. **`model_type` in `config.json`?** (e.g. `"qwen3_omni_moe"`) — your registration key.
2. **`architectures[0]` in `config.json`?** (e.g. `"Qwen3OmniMoeForConditionalGeneration"`) — selects the model class.
3. **Processor class name?** Check `processor_config.json` → `processor_class` field.
4. **MoE model?** → needs `parallel_plan.py`
5. **Multimodal inputs (image/video/audio)?** → needs processor patch and data transform
6. **Multimodal RoPE?** (Qwen-VL family) → needs `get_position_id_func`

*Example — Qwen3-Omni-MoE:* `model_type="qwen3_omni_moe"`, architecture=`"Qwen3OmniMoeForConditionalGeneration"`, processor=`"Qwen3OmniMoeProcessor"`, MoE+image+video+audio, multimodal 3D RoPE.

---

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

---

### Step 2: Register Your Model (`__init__.py`)

This is the **only mandatory file**. It wires your model into VeOmni's registry so the framework auto-discovers the right classes.

**Minimal template** (text-only model):
```python
from ...loader import MODELING_REGISTRY

@MODELING_REGISTRY.register("your_model_type")
def register_modeling(architecture: str):
    from transformers.models.your_model import YourModelForCausalLM
    return YourModelForCausalLM
```

**Full template** (multimodal MoE, e.g. Qwen3-Omni-MoE):
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
    if "ThinkerForConditionalGeneration" in architecture:
        from .modeling_your_model import YourThinkerModel
        return YourThinkerModel
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

### Step 3: Add to the Package `__init__.py`

Add your module to [veomni/models/transformers/__init__.py](../../veomni/models/transformers/__init__.py):

```python
from . import (
    # ... existing models ...
    your_model_name,  # ADD THIS
)

__all__ = [
    # ... existing ...
    "your_model_name",  # ADD THIS
]
```

This ensures the `@register` decorators in your `__init__.py` execute at import time.

---

### Step 4: Patch the Model (`modeling_*.py`)

The standard pattern: **import the HF module as an alias**, **define patched classes/functions**, **apply at the end**.

```python
import transformers.models.your_model.modeling_your_model as hf_your_model

# ... define patches ...

def apply_veomni_your_model_patch():
    hf_your_model.YourClass.method = patched_method
    hf_your_model.YourOtherClass = PatchedClass
```

The specific patches to apply are covered in **Part 2** below. A quick reference of which patches are needed per model type:

| Patch | Text LLM | VLM | Omni MoE |
|---|:---:|:---:|:---:|
| `tie_word_embeddings` config fix | sometimes | sometimes | ✓ |
| FSDP dummy forward | — | ✓ | ✓ (ViT + Audio) |
| SP: LM position embedding slicing | ✓ | ✓ | ✓ |
| SP: ViT pad+slice | — | ✓ | ✓ |
| SP: ViT-to-LM fill-back | — | ✓ | ✓ |
| SP: deepstack all-gather | — | if deepstack | ✓ |
| Fused MoE + stacked weights | — | if MoE | ✓ |
| Flash-attn kwargs pop/restore | — | ✓ | ✓ |
| Pre-compute `max_seqlen` | — | ✓ | ✓ |
| Position ID transposition | — | ✓ | ✓ |
| `ForCausalLMLoss` | ✓ | ✓ | ✓ |
| `get_position_id_func` | — | ✓ | ✓ |

---

### Step 5: Define Expert Parallelism Plan (`parallel_plan.py`, MoE only)

```python
from torch.distributed._tensor import Shard
from ....distributed.parallel_plan import ParallelPlan

def get_parallel_plan():
    ep_plan = {
        # Use glob-style * to match layer indices
        # Paths must match the stacked weight names from Step 4 (§ Fused MoE)
        "model.layers.*.mlp.experts.gate_proj": Shard(0),
        "model.layers.*.mlp.experts.up_proj":   Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    return ParallelPlan(ep_plan=ep_plan)
```

Expose it from the model base class in `apply_veomni_patch()`:
```python
def _get_parallel_plan(_self):
    from .parallel_plan import get_parallel_plan
    return get_parallel_plan()

hf_your_model.YourPreTrainedModel.get_parallel_plan = _get_parallel_plan
```

> **Finding the right paths:** run `for name, _ in model.named_parameters(): print(name)` on the unpatched HF model. The paths must match the **stacked** weight parameter names you defined.

---

### Step 6: Patch the Processor (`processing_*.py`, multimodal only)

Two common issues with HF processors in VeOmni's pipeline:

1. HF checks `if audio is not None:` — VeOmni passes `[]` for absent inputs, so override with `if audio:` (truthy check).
2. Keyword argument mismatch (`audios=` vs `audio=`) — match what `data_transform.py` passes.

```python
import transformers.models.your_model.processing_your_model as hf_processing

class YourModelProcessor(_HFYourModelProcessor):
    def __call__(self, text=None, images=None, videos=None, audios=None, **kwargs):
        if audios:   # truthy check, not `is not None`
            ...
        if images:
            ...
        if videos:
            ...

def apply_veomni_your_model_patch():
    hf_processing.YourModelProcessor = YourModelProcessor
```

---

### Step 7: Write the Data Transform Function

Add a `process_sample_your_model()` to [veomni/data/multimodal/data_transform.py](../../veomni/data/multimodal/data_transform.py).

**For VLMs** (image/video), follow `process_sample_qwen3_vl`:
1. Fetch media with `fetch_images` / `fetch_videos_metadata`
2. Run processor image/video processor → get `pixel_values`, `grid_thw`
3. Compute token counts → `chat_template.encode_messages`
4. Call `position_id_func` → 3D position IDs
5. Build `image_mask` / `video_mask` from special token indices
6. Zero out special tokens: `input_ids[mask] = 0`

**For Omni-modal** (image + video + audio), follow `_process_sample_omni`:
1. Build conversation with `processor.apply_chat_template`
2. Fetch images, videos, audios; align audio to conversations
3. Call `processor(text, audios, images, videos)` → `model_inputs`
4. Post-process `input_features` (permute, filter zero-length audio)
5. Build `image_mask`, `video_mask`, `audio_mask`; replace token IDs with VeOmni constants
6. Call `position_id_func` with all modality inputs
7. Build `labels` by locating assistant spans in `input_ids`

Function signature:
```python
def process_sample_your_model(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",  # pass None for omni-style models
    position_id_func: "Callable",
    **kwargs,
) -> list[Dict[str, Any]]:
```

---

### Step 8: Hook into the Trainer

Edit [veomni/trainer/vlm_trainer.py](../../veomni/trainer/vlm_trainer.py). You need to handle up to 5 methods:

**`build_model_assets`** — load processor and chat template:
```python
def build_model_assets(self):
    self.processor = build_processor(self.args.model.tokenizer_path, max_pixels=MAX_PIXELS)
    if self.model_config.model_type in ("qwen3_omni_moe", "your_omni_model"):
        self.chat_template = None  # omni models use processor.apply_chat_template directly
        return [self.processor]
    else:
        self.chat_template = build_multimodal_chat_template(...)
        return [self.processor, self.chat_template]
```

**`build_data_collate_info`** — extra collation config for special fields:
```python
def build_data_collate_info(self):
    if self.model_config.model_type in ("qwen3_omni_moe", "your_omni_model"):
        return {
            # (pack_dim, sp_slice, pad_value, pad_scale)
            "audio_feature_lengths": (0, False, None, None),
            "input_features":        (0, True,  0,    1),
            "audio_mask":            (-1, False, 0,    1),
        }
    return {}
```

**`build_data_transform`** — select the process_sample function:
```python
def build_data_transform(self):
    if self.model_config.model_type in ("your_model_type",):
        process_function = process_sample_your_model
        position_id_func = self.model.get_position_id_func()
    elif self.model_config.model_type in ("your_omni_type",):
        process_function = process_sample_your_model
        position_id_func = self.model.thinker.get_position_id_func()  # omni: access thinker
    ...
    return partial(process_function, processor=self.processor,
                   chat_template=self.chat_template,
                   position_id_func=position_id_func,
                   **self.args.data.mm_configs)
```

**`freeze_module`** and **`build_param_groups`** — add your model type to the relevant branches if it needs ViT/audio tower freezing or separate learning rates.

---

### Step 9: Add a Config File

Create `configs/multimodal/your_model/your_model.yaml`. Key fields:

```yaml
model:
  config_path: /path/to/model
  model_path: /path/to/model
  tokenizer_path: /path/to/model
  attn_implementation: flash_attention_2
  moe_implementation: fused   # or "eager" if no MoE

train:
  sp_size: 1   # >1 for Sequence Parallelism
  ep_size: 1   # >1 for Expert Parallelism
  freeze_vit: false
```

---

### Step 10: Test Your Integration

Testing for a new model integration happens at three levels, each catching a different class of bugs. See **Part 4: Testing** for detailed guidance on each level and what to add for your model.

---

## Part 2: Patch Reference

This section covers each patch in depth — what it does, why it's needed, and how to implement it. Referenced from the table in Step 4.

---

### P1. Fix `tie_word_embeddings` (Config)

Many models set `tie_word_embeddings=True` by default but don't implement `get_output_embeddings()`. VeOmni's `CustomizedModelingLoader` tries to tie embeddings after weight loading and will crash. Fix in `configuration_*.py`:

```python
class YourModelConfig(_HFYourModelConfig):
    def __init__(self, **kwargs):
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=False, **kwargs)

def apply_veomni_your_model_patch():
    hf_config_module.YourModelConfig = YourModelConfig
```

---

### P2. FSDP Dummy Forward (VLMs and Omni-modal)

When using FSDP, if some ranks receive `None` for `pixel_values` (or `input_features`) while others receive valid tensors, the backward reduce-scatter will **hang**. Every encoder that may receive `None` on some ranks needs a `dummy_forward()`:

```python
class YourVisionEncoder(hf_your_model.YourVisionEncoder):
    def dummy_forward(self):
        if get_parallel_state().sp_enabled:
            sp_size = get_parallel_state().sp_size
            # grid_thw height scaled by sp_size to produce enough tokens for all ranks
            pixel_values = torch.zeros((16, input_dim), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
        else:
            pixel_values = torch.zeros((16, input_dim), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
        return self(hidden_states=pixel_values, grid_thw=grid_thw)
```

In the main model forward, call it when the input is `None`:

```python
if pixel_values is not None:
    image_embeds, deepstack_embeds = self.get_image_features(pixel_values, image_grid_thw)
elif get_parallel_state().fsdp_enabled:
    fake_embeds, fake_deepstack = self.visual.dummy_forward()
    fake_embeds = fake_embeds.mean() * 0.0  # zero-out: no gradient contribution
    inputs_embeds = inputs_embeds + fake_embeds
```

The same applies to the audio encoder for omni-modal models.

---

### P3. SP: Language Model Position Embedding Slicing

VeOmni's data collator pre-slices `input_ids` and `inputs_embeds` along the sequence dimension (`seq // sp_size` per rank). The LM's rotary position embeddings are computed from the **full** position IDs but must be sliced to match:

```python
from ....distributed.sequence_parallel import slice_position_embedding

position_embeddings = self.rotary_emb(hidden_states, position_ids)

sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
```

VeOmni automatically registers a wrapped FlashAttention implementation, so attention layers require no further changes as long as they use `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`.

---

### P4. SP: Vision Transformer Padding and Slicing

The ViT has a fundamental mismatch under SP:

| Tensor | State entering ViT |
|---|---|
| `hidden_states` | **Padded** to a multiple of `pad_scale`, then **sequence-sliced** by the collator |
| `grid_thw` | **Unpadded and unsliced** — always the original full grid |
| `cu_seqlens` | Computed from raw `grid_thw` — **does not know about padding** |

All three must be reconciled before the block loop.

**Pad and slice position embeddings** to match the padded hidden states:

```python
from ....distributed.sequence_parallel import sp_pad_and_slice

pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
if sp_group is not None:
    pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=MERGE_RATIO)

hidden_states = hidden_states + pos_embeds
```

Apply the **same padding and slicing to rotary position embeddings**:
```python
rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)  # total_seq_len = cu_seqlens[-1]
emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
position_embeddings = (emb.cos(), emb.sin())

if sp_group is not None:
    cos, sin = position_embeddings
    cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=MERGE_RATIO)
    sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=MERGE_RATIO)
    position_embeddings = (cos, sin)
```

**Extend `cu_seqlens`** with a padding entry to cover the padded tail on the last rank:
```python
total_seq_len = cu_seqlens[-1]
seq_len = hidden_states.size(0)  # after collator padding+slicing

if sp_group is not None:
    sp_size = get_parallel_state().sp_size
    pad_seq_len = seq_len * sp_size - total_seq_len.item()
    if pad_seq_len > 0:
        cu_seqlens = torch.cat([cu_seqlens, (cu_seqlens[-1] + pad_seq_len).unsqueeze(0)])
```

> **What is `MERGE_RATIO` / `pad_scale`?** It equals the number of ViT tokens that get merged into one LM token. Qwen-VL uses a 2×2 spatial merge → `pad_scale=4`. A 3×3 merge → `pad_scale=9`. The collator pads vision sequences to a multiple of this value so the merge operation sees complete groups; position embeddings must match.

---

### P5. SP: ViT-to-LM Fill-Back (3-Step Dance)

After the ViT, image embeddings must be scattered back into the correct positions in `inputs_embeds`. The `image_mask` is pre-computed in `process_sample` and covers the **full** sequence — but under SP, `inputs_embeds` is only `seq // sp_size` long. This requires a temporary layout change:

```python
# Step 1: Gather sequence, scatter heads → full-seq layout
# (bs, seq//sp, hidden) → (bs, seq, hidden//sp)
if self.training and get_parallel_state().sp_enabled:
    inputs_embeds = gather_seq_scatter_heads(
        inputs_embeds, seq_dim=1, head_dim=2, group=sp_group
    )

# Step 2: Same transform on image/video/audio embeddings, then fill back
if pixel_values is not None:
    image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    if self.training and get_parallel_state().sp_enabled:
        # (seq//sp, hidden) → (seq, hidden//sp)
        image_embeds = gather_seq_scatter_heads(
            image_embeds, seq_dim=0, head_dim=-1, group=sp_group
        )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
# repeat for video, audio...

# Step 3: Restore SP layout
# (bs, seq, hidden//sp) → (bs, seq//sp, hidden)
if self.training and get_parallel_state().sp_enabled:
    inputs_embeds = gather_heads_scatter_seq(
        inputs_embeds, head_dim=2, seq_dim=1, group=sp_group
    )
```

> **Why this works:** `masked_scatter` places image tokens exactly at positions where `image_mask` is True. When both `inputs_embeds` and `image_embeds` are in `(seq, hidden//sp)` layout, every rank covers the entire sequence (scattered along the hidden dimension), so the fill-back is position-correct. Restoring to `(seq//sp, hidden)` layout afterwards pays back the communication cost.

---

### P6. SP: Deepstack / Cross-Layer Visual Embeddings

If your model injects visual features into **multiple decoder layers** (DeepStack pattern), each layer would need an All2All if you kept deepstack embeddings distributed. Instead, do one all-gather after ViT and slice per rank:

```python
from ....distributed.sequence_parallel.ulysses import _Gather

if sp_enabled and pixel_values is not None:
    # All-gather: (seq//sp, hidden) → (seq, hidden)
    deepstack_embeds = [
        _Gather.apply(sp_group, embed, 0, False) for embed in deepstack_embeds
    ]

    image_mask_1d = image_mask[..., 0]     # (bs, seq)
    seq_per_rank = seq_len // sp_size
    rank_start   = sp_rank * seq_per_rank
    rank_mask    = image_mask_1d[:, rank_start : rank_start + seq_per_rank]
    offset       = image_mask_1d[:, :rank_start].sum().item()
    n_tokens     = rank_mask.sum().item()

    deepstack_embeds = [e[offset : offset + n_tokens] for e in deepstack_embeds]
```

Each rank then holds only the deepstack tokens that correspond to its sequence partition. The decoder layers can inject them without any further communication.

---

### P7. MoE: Fused Forward + Stacked Expert Weights

The standard HuggingFace MoE uses an `nn.ModuleList` of individual expert MLPs. VeOmni replaces this with a single module holding **stacked 3D weight tensors** — the shape required by both the fused triton kernel and EP sharding (`Shard(0)` along the expert dimension).

```python
class YourModelExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Shape: (num_experts, out_dim, in_dim)
        self.gate_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.up_proj   = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

    def forward(self, hidden_states, routing_weights, selected_experts, num_experts):
        return fused_moe_forward(
            module=self,
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=self.gate_proj,
            fc1_2_weight=self.up_proj,
            fc2_weight=self.down_proj,
        )
```

Use this in the SparseMoeBlock; keep the original `nn.ModuleList` path for `moe_implementation="eager"` (which does not support EP):

```python
if self._moe_implementation == "fused":
    self.experts = YourModelExperts(config)
elif self._moe_implementation == "eager":
    self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(num_experts)])
```

> **If the model uses a fused `gate_up_proj`** (shape `(num_experts, hidden, 2 * expert_dim)`, e.g. Qwen3-VL MoE), split it before calling `fused_moe_forward`:
> ```python
> gate_proj_t = self.gate_up_proj[..., :expert_dim].transpose(1, 2).contiguous()
> up_proj_t   = self.gate_up_proj[..., expert_dim:].transpose(1, 2).contiguous()
> down_proj_t = self.down_proj.transpose(1, 2).contiguous()
> ```
> The transpose is needed because the checkpoint stores `(num_experts, hidden, expert_dim)` while `fused_moe_forward` expects `(num_experts, expert_dim, hidden)`.

Also patch `_init_weights` to handle the stacked parameter:
```python
@torch.no_grad()
def custom_init_weights(self, module):
    super(HFPreTrainedModel, self)._init_weights(module)
    if isinstance(module, YourModelExperts):
        nn.init.normal_(module.gate_proj, std=self.config.initializer_range)
        nn.init.normal_(module.up_proj,   std=self.config.initializer_range)
        nn.init.normal_(module.down_proj, std=self.config.initializer_range)
```

---

### P8. Pop Flash-Attention kwargs Before ViT Forward

The LM-level flash-attention kwargs (`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`) are injected by the data collator for the LM's packed-sequence attention. They must **not** reach the ViT, which computes its own `cu_seqlens` from `grid_thw`.

```python
# At the start of forward(), before ViT:
flash_attn_kwargs = {}
for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
    if key in kwargs:
        flash_attn_kwargs[key] = kwargs.pop(key)

# ... all encoder (ViT, audio) forwards here ...

# Restore before LM forward:
kwargs.update(flash_attn_kwargs)
outputs = self.language_model(..., **kwargs)
```

---

### P9. Pre-compute `max_seqlen` (Performance)

`(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()` triggers a CPU-GPU sync. Inside a layer loop, this fires once per layer. Move it outside:

```python
max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()

for blk in self.blocks:
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings, max_seqlen=max_seqlen)
```

---

### P10. Position ID Transposition

VeOmni collates per-sample position IDs as `(bs, dim, L)` for convenience. The HuggingFace model API expects `(dim, bs, L)`. Add a transpose at the start of the top-level forward:

```python
if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[1] == 3:
    position_ids = position_ids.transpose(0, 1).contiguous()  # (bs, 3, L) → (3, bs, L)
```

---

### P11. VeOmni Loss Utility

Replace the model's built-in CE loss with `ForCausalLMLoss` to get Liger/fused kernel selection and correct SP loss reduction automatically:

```python
from ....ops.fused_cross_entropy import ForCausalLMLoss

if labels is not None:
    loss, logits = ForCausalLMLoss(
        labels=labels,
        vocab_size=self.config.vocab_size,
        hidden_states=hidden_states,
        weights=self.lm_head.weight,
        ignore_index=IGNORE_INDEX,
    )
```

---

### P12. `get_position_id_func` (Multimodal RoPE)

VeOmni pre-computes position IDs per sample **during data preprocessing** (before training, in worker processes). The model must expose a `get_position_id_func()` that returns a **picklable** callable:

```python
def get_position_id(main_func, self, **kwargs):
    """Must be a module-level function (not a method) for multiprocessing pickle."""
    position_ids, rope_deltas = main_func(self, **kwargs)  # (dim, 1, L), (1, 1)
    assert position_ids.shape[1] == 1
    return {"position_ids": position_ids.squeeze(1), "rope_deltas": rope_deltas.squeeze(0)}

class YourModel(hf_your_model.YourModel):
    def get_position_id_func(self):
        fake_config = copy.copy(self.config)
        # Use VeOmni constants so get_rope_index sees the same token IDs as at train time
        fake_config.image_token_id = IMAGE_INPUT_INDEX
        fake_config.video_token_id = VIDEO_INPUT_INDEX
        fake_model = SimpleNamespace(
            config=fake_config,
            spatial_merge_size=self.spatial_merge_size,
            get_llm_pos_ids_for_vision=partial(
                hf_your_model.YourClass.get_llm_pos_ids_for_vision, None
            ),
        )
        return partial(get_position_id, hf_your_model.YourClass.get_rope_index, fake_model)
```

> **Why `IMAGE_INPUT_INDEX` instead of the model's own token ID?** In `process_sample`, multimodal token IDs are replaced with VeOmni constants (`IMAGE_INPUT_INDEX`, `VIDEO_INPUT_INDEX`, `AUDIO_INPUT_INDEX`) and then zeroed out before storage. `get_rope_index` must see these same constants when called during preprocessing.

---

## Part 3: Checklists

### Any new model

- [ ] `veomni/models/transformers/your_model/__init__.py` with `@MODELING_REGISTRY.register`
- [ ] `veomni/models/transformers/__init__.py` updated
- [ ] `get_parallel_plan` wired on the pretrained model base class

### VLMs (image/video)

- [ ] FSDP `dummy_forward` in ViT encoder
- [ ] SP `sp_pad_and_slice` in ViT (correct `pad_scale`)
- [ ] SP `cu_seqlens` padding entry
- [ ] SP ViT-to-LM fill-back (`gather_seq_scatter_heads` / `gather_heads_scatter_seq`)
- [ ] `get_position_id_func` using VeOmni token ID constants
- [ ] `process_sample_*` in `data_transform.py`; `build_data_transform` in `VLMTrainer`

### MoE models

- [ ] `parallel_plan.py` with correct expert weight paths
- [ ] Stacked-weight `YourModelExperts` module + `fused_moe_forward`
- [ ] `_moe_implementation` propagated from top-level config to text sub-config
- [ ] `_init_weights` patched for stacked expert params

### Omni-modal (audio)

- [ ] FSDP `dummy_forward` in audio encoder
- [ ] SP gather/slice in audio encoder (`gather_outputs` + `slice_input_tensor`)
- [ ] `audio_mask` in data transform; `audio_feature_lengths` in `build_data_collate_info`
- [ ] Processor patched: `if audios:` truthy check

### Testing (all models — see Part 4 for details)

- [ ] Toy config added to `tests/toy_config/your_model_toy/`
- [ ] `DummyYourModelDataset` added to `veomni/data/dummy_dataset.py` (multimodal)
- [ ] Entry added to `test_cases` in `tests/models/test_models_patch.py` (Level 1)
- [ ] Test case + fixture + test function added to `tests/e2e/test_e2e_parallel.py` (Level 2)

---

## Part 4: Testing

VeOmni has a structured test suite under `tests/`. For a new model integration, testing spans three levels:

```
Level 1 — Unit (single GPU, no real weights)   → tests/models/
Level 2 — Parallel alignment (multi-GPU)        → tests/e2e/test_e2e_parallel.py
Level 3 — End-to-end training (real data/ckpt)  → tests/e2e/test_e2e_training.py
```

Each level is additive: pass Level 1 before running Level 2, and Level 2 before Level 3.

---

### Level 1 — Unit Tests (`tests/models/`)

**What it tests:** model registration, forward/backward pass correctness, and HF↔VeOmni patch alignment — all on a single GPU using tiny random weights.

#### 1a. Toy Config

Every model under test needs a toy `config.json` (and `preprocessor_config.json` for multimodal models) in `tests/toy_config/your_model_toy/`. The toy config should be structurally identical to the real config but with drastically reduced sizes:

| Field | Real Qwen3-Omni-MoE | Toy version |
|---|---|---|
| `num_hidden_layers` | 28 | 2 |
| `hidden_size` | 2048 | 2048 (keep; shapes matter) |
| `num_experts` | 128 | 128 (keep for routing logic) |
| `encoder_layers` | 32 | 2 |

For omni-modal models the toy config also needs `preprocessor_config.json` — copy from the real model and keep as-is, since the feature extractor parameters (mel bins, sample rate, patch size) are not reducible without changing the data pipeline.

Reference: [tests/toy_config/qwen3omni_toy/config.json](../../tests/toy_config/qwen3omni_toy/config.json), [tests/toy_config/qwen3omni_toy/preprocessor_config.json](../../tests/toy_config/qwen3omni_toy/preprocessor_config.json)

#### 1b. Dummy Dataset

For multimodal models, add a `DummyXxxDataset` class to [veomni/data/dummy_dataset.py](../../veomni/data/dummy_dataset.py) and register it in `build_dummy_dataset()`. The class must produce the exact tensor keys and shapes your model's collator expects.

Key design notes:
- Compute `image_seqlen`, `audio_seq_length`, `video_seqlen` from the same formulas used in the real data pipeline (patch size, merge size, audio feature extractor downsampling) so the dummy shapes are realistic.
- Provide `image_mask`, `audio_mask`, `video_mask` as boolean tensors over the full sequence.
- For Qwen3-Omni-MoE the audio output length formula matches the convolutional downsampler:

```python
# DummyQwen3OmniMoeDataset._get_feat_extract_output_lengths
input_lengths_leave = input_lengths % 100
feat_lengths = (input_lengths_leave - 1) // 2 + 1
output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
```

Then register the new dataset type in `build_dummy_dataset()`:

```python
elif task_type == "your_model":
    return DummyYourModelDataset(size=size, seq_length=max_seq_len, patch_size=16)
```

Reference: [veomni/data/dummy_dataset.py](../../veomni/data/dummy_dataset.py)

#### 1c. Forward/Backward Patch Test

Add a `pytest.param` entry to the `test_cases` list in [tests/models/test_models_patch.py](../../tests/models/test_models_patch.py):

```python
pytest.param(
    "./tests/toy_config/your_model_toy",
    is_moe,          # True if model has expert parallelism
    _DEFAULT_RTOL,
    _DEFAULT_ATOL,
    id="your_model_type",   # must match model_type in config.json
),
```

This single entry exercises:
- Model loading via registry (`MODELING_REGISTRY`, `MODEL_CONFIG_REGISTRY`)
- Forward pass under all attention implementations (`eager`, `flash_attention_2`, `flash_attention_3`)
- Forward pass under all MoE implementations (`eager`, `fused`) if `is_moe=True`
- Numerical alignment between HF baseline and VeOmni-patched forward
- Backward pass (gradient flow through all patched components)

If the HF and VeOmni models have different state-dict layouts (e.g. stacked expert weights vs `nn.ModuleList`), add a weight sync function to [tests/models/weight_sync_adapters.py](../../tests/models/weight_sync_adapters.py) and register it under your model's `id`.

Run:
```bash
source .venv/bin/activate
pytest -s tests/models/test_models_patch.py -k your_model_type
```

Also add a token-ID mapping entry to [tests/models/utils.py](../../tests/models/utils.py) — `MODEL_TO_DATASET` maps `model_type` → dummy dataset key, and omni-modal models need special handling in `parse_token_id_from_config()` to read token IDs from the nested `thinker_config`:

```python
# MODEL_TO_DATASET mapping
"your_model_type": "your_dataset_key",

# parse_token_id_from_config — for omni models with nested thinker_config
if model_config.model_type in ["qwen2_5_omni", "qwen3_omni_moe", "your_omni_model"]:
    token_ids_dict = {
        "image_token_id": model_config.thinker_config.image_token_id,
        "video_token_id": model_config.thinker_config.video_token_id,
        "audio_token_id": model_config.thinker_config.audio_token_id,
    }
```

---

### Level 2 — Parallel Alignment Test (`tests/e2e/test_e2e_parallel.py`)

**What it tests:** that training loss and gradient norms stay numerically aligned across different parallelism configurations — specifically SP×1 vs SP×2, and EP×1 vs EP×2 for MoE models. This catches bugs in SP gather/scatter, FSDP dummy forwards, and EP sharding that only surface under distributed execution.

**How it works:**
1. Builds a random-weight model from the toy config via `build_foundation_model()`.
2. Runs `torchrun` with 8 GPUs in each parallel mode (SP=1, SP=2; EP=1, EP=2 if MoE).
3. Each run writes metrics to `log_dict.json`.
4. Compares all runs with `compare_multi_items(rtol, atol)`.

Add a test case list and test function:

```python
# In test_e2e_parallel.py

your_model_test_cases = [
    pytest.param(
        "your_model_type",
        "./tests/toy_config/your_model_toy",
        is_moe,          # True enables EP=2 variant
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
    ),
]

@pytest.fixture(scope="session")
def dummy_your_model_dataset():
    dummy_dataset = DummyDataset(seq_len=2048, dataset_type="your_dataset_key")
    train_path = dummy_dataset.save_path
    yield train_path
    del dummy_dataset

@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", your_model_test_cases)
def test_your_model_parallel_align(
    model_name, config_path, is_moe, rtol, atol, dummy_your_model_dataset
):
    main(
        task_name="train_vlm_test",   # or "train_text_test" for text-only
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        train_path=dummy_your_model_dataset,
    )
```

For Qwen3-Omni-MoE the task is `"train_vlm_test"` (uses `TestVLMTrainer` which extends `VLMTrainer`). Text-only models use `"train_text_test"`.

Run:
```bash
source .venv/bin/activate
pytest -s tests/e2e/test_e2e_parallel.py -k your_model_type
```

Reference implementation: `qwen3omni_test_cases` and `test_qwen3omni_parallel_align` in [tests/e2e/test_e2e_parallel.py](../../tests/e2e/test_e2e_parallel.py).

---

### Level 3 — End-to-End Training Test (`tests/e2e/test_e2e_training.py`)

**What it tests:** a full training run with real model weights, real data, checkpointing, and resumption. This is the final gate before declaring a model production-ready.

This level currently covers text-only models (Qwen3 0.6B with Tulu SFT data). Extending it to a new multimodal model requires:
1. A real model checkpoint accessible at `CI_MODEL_DIR`.
2. A real dataset accessible at `CI_DATASET_DIR`.
3. An entry in the `E2E_TEST_SCRIPT` dictionary in [tests/e2e/exec_scripts.py](../../tests/e2e/exec_scripts.py) with the full `torchrun` command.
4. A new `pytest.param` in `test_e2e_training.py` pointing to it.

Run:
```bash
source .venv/bin/activate
CI_MODEL_DIR=/path/to/models CI_DATASET_DIR=/path/to/data \
pytest -s tests/e2e/test_e2e_training.py -k your_model
```

---

### Quick Reference: What to Add Per Test Level

| What to add | Location | Required for |
|---|---|---|
| Toy `config.json` | `tests/toy_config/your_model_toy/` | All levels |
| `preprocessor_config.json` | `tests/toy_config/your_model_toy/` | Multimodal |
| `DummyYourModelDataset` | `veomni/data/dummy_dataset.py` | Multimodal |
| `build_dummy_dataset` entry | `veomni/data/dummy_dataset.py` | Multimodal |
| `MODEL_TO_DATASET` entry | `tests/models/utils.py` | Level 1 |
| `parse_token_id_from_config` branch | `tests/models/utils.py` | Omni-modal |
| `pytest.param` in `test_cases` | `tests/models/test_models_patch.py` | Level 1 |
| Weight sync adapter | `tests/models/weight_sync_adapters.py` | Level 1 (MoE only) |
| `pytest.param` in `*_test_cases` | `tests/e2e/test_e2e_parallel.py` | Level 2 |
| Dataset fixture | `tests/e2e/test_e2e_parallel.py` | Level 2 |
| Test function | `tests/e2e/test_e2e_parallel.py` | Level 2 |
| Entry in `E2E_TEST_SCRIPT` | `tests/e2e/exec_scripts.py` | Level 3 |
| `pytest.param` in training test | `tests/e2e/test_e2e_training.py` | Level 3 |

---

### Future Work

The following testing gaps exist and are candidates for future improvement:

- **Checkpoint round-trip for multimodal/omni models.** `tests/checkpoints/test_trainer_saveload.py` currently only covers text MoE models (`qwen3_moe`, `deepseek_v3`). VLM and omni-modal models with heterogeneous sub-components (ViT, audio encoder, talker, code2wav) need dedicated save/load/conversion tests to catch dtype and layout bugs in their individual sub-module checkpoints.

- **Data collator tests for omni-modal keys.** `tests/data/test_collators.py` tests the base collator with sequence parallelism but does not cover the extra collation fields for omni-modal models (`input_features`, `audio_feature_lengths`, `audio_mask`). These fields have non-trivial padding and SP-slicing behavior that warrants explicit coverage.

- **Processor patch tests.** There are no unit tests specifically for the patched processors (e.g. truthy `if audios:` vs `if audio is not None:` behavior). A lightweight test that calls the patched processor with empty lists vs `None` vs populated inputs would catch regressions early.

- **`get_position_id_func` pickling test.** The position ID function must be picklable for multiprocessing data loaders. A test that pickles and unpickles the function with `pickle.dumps` / `pickle.loads` would guard against accidental closure captures.

- **Expert Parallel checkpoint for omni models.** MoE checkpoint tests only test EP=[1,4,8] for text MoE; omni-modal MoE models (which have MoE in the thinker LLM and the talker) are not yet covered.

- **NPU coverage for multimodal models.** The NPU CI workflow mirrors the GPU unit test structure but multimodal and omni-modal models are not yet included. Some ops (e.g. `torch.kaiser_window` in BigVGAN) are known to be unsupported on NPU — these should be explicitly skipped and documented.

---

## Part 5: Reference

### Key Imports

```python
from veomni.distributed.parallel_state import get_parallel_state

from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,   # (bs, seq, h//sp) → (bs, seq//sp, h)
    gather_outputs,             # all-gather along a dim (no autograd)
    gather_seq_scatter_heads,   # (bs, seq//sp, h) → (bs, seq, h//sp)
    slice_input_tensor,         # slice along a dim for this SP rank
    slice_position_embedding,   # slice position embeddings for SP
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

from veomni.models.transformers.attention_utils import VARLEN_ATTENTION_TYPES
```

### Common Pitfalls

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

## Acknowledgements

Thanks to ByteDance Seed and AML team: Qianli Ma, Zhelun Shi, Yifan Pi, Tianle Zhong, Xiao Yu.
