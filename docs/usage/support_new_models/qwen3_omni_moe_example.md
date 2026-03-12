# Qwen3-Omni-MoE Integration Example

**Authors**: Ting Yang

This document provides in-depth implementation details for each patch applied in the **Qwen3-Omni-MoE** integration — VeOmni's most complex model type, covering image, video, and audio modalities with MoE and Expert Parallelism. Use this alongside [guide_and_checklist.md](./guide_and_checklist.md).

---

## P1. Fix `tie_word_embeddings` (Config)

Many models set `tie_word_embeddings=True` by default but don't implement `get_output_embeddings()`. VeOmni's `CustomizedModelingLoader` tries to tie embeddings after weight loading and will crash. Fix in `configuration_*.py`:

```python
class YourModelConfig(_HFYourModelConfig):
    def __init__(self, **kwargs):
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=False, **kwargs)

def apply_veomni_patch():
    hf_config_module.YourModelConfig = YourModelConfig
```

---

## P2. FSDP Dummy Forward (VLMs and Omni-modal)

When using FSDP, ranks that receive `None` for `pixel_values` (or `input_features`) while others receive valid tensors cause backward reduce-scatter hangs. Every encoder that may receive `None` on some ranks needs a `dummy_forward()`:

```python
class YourVisionEncoder(hf_your_model.YourVisionEncoder):
    def dummy_forward(self):
        # Replace `input_dim` with the actual flat input dimension for your encoder.
        # For Qwen3-Omni-MoE ViT: 3 + 2 * 16 + 16
        input_dim = ...  # e.g. self.config.vision_config.patch_size ** 2 * channels
        if get_parallel_state().sp_enabled:
            sp_size = get_parallel_state().sp_size
            pixel_values = torch.zeros((16, input_dim), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
        else:
            pixel_values = torch.zeros((16, input_dim), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
        return self(hidden_states=pixel_values, grid_thw=grid_thw)
```

Call it in the main forward when the input is `None`:

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

## P3. SP: Language Model Position Embeddings

Since `position_ids` is already sliced by `SequenceParallelCollator`, calling `rotary_emb(hidden_states, position_ids)` directly produces local-length position embeddings — no additional slicing is needed.

VeOmni automatically registers a wrapped FlashAttention implementation, so attention layers require no further changes.

---

## P4. SP: Vision Transformer Padding and Slicing

The ViT has a fundamental mismatch under SP:

| Tensor | State entering ViT |
|---|---|
| `hidden_states` | Padded to a multiple of `pad_scale`, then sequence-sliced by the collator |
| `grid_thw` | Unpadded and unsliced — always the original full grid |
| `cu_seqlens` | Computed from raw `grid_thw` — does not know about padding |

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
rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)
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

> **What is `MERGE_RATIO` / `pad_scale`?** It equals the number of ViT tokens merged into one LM token. Qwen-VL uses a 2×2 spatial merge → `pad_scale=4`.

---

## P5. SP: ViT-to-LM Fill-Back (3-Step Dance)

After the ViT, image embeddings must be scattered into the correct positions in `inputs_embeds`. The `image_mask` covers the full sequence, but under SP `inputs_embeds` is only `seq // sp_size` long:

```python
# Step 1: Gather sequence, scatter heads → full-seq layout
# (bs, seq//sp, hidden) → (bs, seq, hidden//sp)
sp_enabled = self.training and get_parallel_state().sp_enabled
sp_group = get_parallel_state().sp_group if sp_enabled else None

if sp_enabled:
    inputs_embeds = gather_seq_scatter_heads(
        inputs_embeds, seq_dim=1, head_dim=2, group=sp_group
    )

# Step 2: Same transform on image/video/audio embeddings, then fill back
if pixel_values is not None:
    image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    if sp_enabled:
        # (seq//sp, hidden) → (seq, hidden//sp)
        image_embeds = gather_seq_scatter_heads(
            image_embeds, seq_dim=0, head_dim=-1, group=sp_group
        )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
# repeat for video, audio...

# Step 3: Restore SP layout
# (bs, seq, hidden//sp) → (bs, seq//sp, hidden)
if sp_enabled:
    inputs_embeds = gather_heads_scatter_seq(
        inputs_embeds, head_dim=2, seq_dim=1, group=sp_group
    )
```

> **Why this works:** `masked_scatter` places image tokens exactly at positions where `image_mask` is True. When both `inputs_embeds` and `image_embeds` are in `(seq, hidden//sp)` layout, every rank covers the entire sequence (scattered along the hidden dimension), so the fill-back is position-correct.

---

## P6. SP: Deepstack / Cross-Layer Visual Embeddings

If your model injects visual features into multiple decoder layers (DeepStack pattern), all-gather once after ViT and slice per rank — avoiding repeated All2All in every decoder layer:

```python
from ....distributed.sequence_parallel.ulysses import _Gather
from ....distributed.parallel_state import get_parallel_state

sp_enabled = get_parallel_state().sp_enabled
if sp_enabled and pixel_values is not None:
    sp_group = get_parallel_state().sp_group
    sp_size = get_parallel_state().sp_size
    sp_rank = get_parallel_state().sp_rank
    seq_len = image_mask.shape[1]  # image_mask is (bs, seq, ...)

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

---

## P7. MoE: Fused Forward + Stacked Expert Weights

The standard HuggingFace MoE uses `nn.ModuleList` of individual expert MLPs. VeOmni replaces this with a single module holding stacked 3D weight tensors — required by both the fused triton kernel and EP sharding:

```python
class YourModelExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_experts = config.num_experts
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
        # Shape: (num_experts, out_dim, in_dim)
        self.gate_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.up_proj   = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

    def forward(self, hidden_states, routing_weights, selected_experts, num_experts):
        return fused_moe_forward(
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=self.gate_proj,
            fc1_2_weight=self.up_proj,
            fc2_weight=self.down_proj,
        )
```

Keep the original `nn.ModuleList` path for `moe_implementation="eager"` (which does not support EP):

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

Also patch `_init_weights` for the stacked parameter:
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

## P8. Pop Flash-Attention kwargs Before ViT Forward

The LM-level flash-attention kwargs (`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`) are injected for packed-sequence attention. They must not reach the ViT, which computes its own `cu_seqlens`:

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

## P9. Pre-compute `max_seqlen` (Performance)

`(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()` triggers a CPU-GPU sync. Inside a layer loop this fires once per layer. Move it outside:

```python
max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()

for blk in self.blocks:
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings, max_seqlen=max_seqlen)
```

---

## P10. Position ID Transposition

VeOmni collates per-sample position IDs as `(bs, dim, L)`. The HuggingFace model API expects `(dim, bs, L)`. Add a transpose at the start of the top-level forward:

```python
if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[1] == 3:
    position_ids = position_ids.transpose(0, 1).contiguous()  # (bs, 3, L) → (3, bs, L)
```

---

## P11. VeOmni Loss Utility

Replace the model's built-in CE loss with `ForCausalLMLoss` to get Liger/fused kernel selection and correct SP loss reduction:

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

## P12. `get_position_id_func` (Multimodal RoPE)

VeOmni pre-computes position IDs per sample during data preprocessing in worker processes. The model must expose a `get_position_id_func()` that returns a **picklable** callable:

```python
import copy
from functools import partial
from types import SimpleNamespace

from ....utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX

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

> **Why `IMAGE_INPUT_INDEX` instead of the model's own token ID?** In `process_sample`, multimodal token IDs are replaced with VeOmni constants and then zeroed out before storage. `get_rope_index` must see these same constants when called during preprocessing.

---

## Testing

### Three-Level Strategy

```
Level 1 — Unit (single GPU, no real weights)   → tests/models/
Level 2 — Parallel alignment (multi-GPU)        → tests/e2e/test_e2e_parallel.py
Level 3 — End-to-end training (real data/ckpt)  → tests/e2e/test_e2e_training.py
```

Pass Level 1 before running Level 2, and Level 2 before Level 3.

### Level 1 — Unit Tests

#### Toy Config

Add a toy `config.json` (and `preprocessor_config.json` for multimodal) to `tests/toy_config/your_model_toy/` with drastically reduced sizes:

| Field | Real Qwen3-Omni-MoE | Toy version |
|---|---|---|
| `num_hidden_layers` | 28 | 2 |
| `hidden_size` | 2048 | 2048 (keep; shapes matter) |
| `num_experts` | 128 | 128 (keep for routing logic) |
| `encoder_layers` | 32 | 2 |

For omni-modal models, copy `preprocessor_config.json` from the real model as-is — feature extractor parameters (mel bins, sample rate, patch size) are not reducible.

Reference: [tests/toy_config/qwen3omni_toy/](../../../tests/toy_config/qwen3omni_toy/)

#### Dummy Dataset

Add a `DummyXxxDataset` class to [veomni/data/dummy_dataset.py](../../../veomni/data/dummy_dataset.py) and register it in `build_dummy_dataset()`. For Qwen3-Omni-MoE, the audio output length formula matches the convolutional downsampler:

```python
# DummyQwen3OmniMoeDataset._get_feat_extract_output_lengths
input_lengths_leave = input_lengths % 100
feat_lengths = (input_lengths_leave - 1) // 2 + 1
output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
```

Register:
```python
elif task_type == "your_model":
    return DummyYourModelDataset(size=size, seq_length=max_seq_len, patch_size=16)
```

#### Forward/Backward Patch Test

Add to `test_cases` in [tests/models/test_models_patch.py](../../../tests/models/test_models_patch.py):

```python
pytest.param(
    "./tests/toy_config/your_model_toy",
    is_moe,
    _DEFAULT_RTOL,
    _DEFAULT_ATOL,
    id="your_model_type",   # must match model_type in config.json
),
```

Also add `MODEL_TO_DATASET` entry and (for omni models) `parse_token_id_from_config` branch to [tests/models/utils.py](../../../tests/models/utils.py):

```python
# MODEL_TO_DATASET
"your_model_type": "your_dataset_key",

# parse_token_id_from_config — omni models with nested thinker_config
if model_config.model_type in ["qwen2_5_omni", "qwen3_omni_moe", "your_omni_model"]:
    token_ids_dict = {
        "image_token_id": model_config.thinker_config.image_token_id,
        "video_token_id": model_config.thinker_config.video_token_id,
        "audio_token_id": model_config.thinker_config.audio_token_id,
    }
```

Run:
```bash
source .venv/bin/activate
pytest -s tests/models/test_models_patch.py -k your_model_type
```

### Level 2 — Parallel Alignment Test

Add to [tests/e2e/test_e2e_parallel.py](../../../tests/e2e/test_e2e_parallel.py):

```python
your_model_test_cases = [
    pytest.param(
        "your_model_type",
        "./tests/toy_config/your_model_toy",
        is_moe,
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

Run:
```bash
source .venv/bin/activate
pytest -s tests/e2e/test_e2e_parallel.py -k your_model_type
```

Reference: `qwen3omni_test_cases` and `test_qwen3omni_parallel_align` in [tests/e2e/test_e2e_parallel.py](../../../tests/e2e/test_e2e_parallel.py).

### Level 3 — End-to-End Training Test

Requires a real checkpoint and dataset. Add an entry to `E2E_TEST_SCRIPT` in [tests/e2e/exec_scripts.py](../../../tests/e2e/exec_scripts.py) and a `pytest.param` in `test_e2e_training.py`.

Run:
```bash
source .venv/bin/activate
CI_MODEL_DIR=/path/to/models CI_DATASET_DIR=/path/to/data \
pytest -s tests/e2e/test_e2e_training.py -k your_model
```

### What to Add Per Test Level

| What to add | Location | Required for |
|---|---|---|
| Toy `config.json` | `tests/toy_config/your_model_toy/` | All levels |
| `preprocessor_config.json` | `tests/toy_config/your_model_toy/` | Multimodal |
| `DummyYourModelDataset` | `veomni/data/dummy_dataset.py` | Multimodal |
| `build_dummy_dataset` entry | `veomni/data/dummy_dataset.py` | Multimodal |
| `MODEL_TO_DATASET` entry | `tests/models/utils.py` | Level 1 |
| `parse_token_id_from_config` branch | `tests/models/utils.py` | Omni-modal |
| `pytest.param` in `test_cases` | `tests/models/test_models_patch.py` | Level 1 |
| `pytest.param` in `*_test_cases` | `tests/e2e/test_e2e_parallel.py` | Level 2 |
| Dataset fixture | `tests/e2e/test_e2e_parallel.py` | Level 2 |
| Test function | `tests/e2e/test_e2e_parallel.py` | Level 2 |
| Entry in `E2E_TEST_SCRIPT` | `tests/e2e/exec_scripts.py` | Level 3 |
| `pytest.param` in training test | `tests/e2e/test_e2e_training.py` | Level 3 |

---

## Future Testing Gaps

- **Checkpoint round-trip for multimodal/omni models** — `tests/checkpoints/test_trainer_saveload.py` currently only covers text MoE models.
- **Data collator tests for omni-modal keys** — `input_features`, `audio_feature_lengths`, `audio_mask` have non-trivial padding/SP-slicing behavior.
- **Processor patch tests** — no unit tests for truthy `if audios:` vs `if audio is not None:` behavior.
- **`get_position_id_func` pickling test** — the function must be picklable for multiprocessing data loaders.
- **Expert Parallel checkpoint for omni models** — MoE checkpoint tests only cover text MoE.
- **NPU coverage for multimodal models** — some ops (e.g. `torch.kaiser_window` in BigVGAN) are known to be unsupported on NPU.

---

## Acknowledgements

Thanks to ByteDance Seed and AML team: Zhelun Shi, Jia Bin, Yifan Pi, Tianle Zhong, Xiao Yu.
