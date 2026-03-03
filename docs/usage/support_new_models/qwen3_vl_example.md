# Qwen3-VL MoE Integration Example

**Author**: Juntian Liu

This document walks through the specific patches applied to integrate **Qwen3-VL MoE** into VeOmni. It is a concreate example of the patterns described in [guide_and_checklist.md](./guide_and_checklist.md), covering FSDP, Sequence Parallelism, Expert Parallelism, and model registration.

---

## 1. FSDP: Dummy ViT Forward

When using FSDP, ranks that receive `None` for `pixel_values` or `pixel_values_videos` while other ranks receive valid tensors will cause a backward reduce-scatter hang. Add a `dummy_forward` to the ViT:

```python
def dummy_forward(self):
    """
    Dummy forward to avoid FSDP reduce-scatter hang when some ranks get None pixel_values.
    Needed for both image and video inputs.
    """
    pixel_values = torch.zeros([16, 3 + 2 * 16 + 16], dtype=self.dtype, device=self.device)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
    return self(hidden_states=pixel_values, grid_thw=grid_thw)
```

Call it in the main forward when inputs are `None`:

```python
if pixel_values is not None:
    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    # ...
elif get_parallel_state().fsdp_enabled:
    fake_embeds, fake_deepstack = self.visual.dummy_forward()
    fake_embeds = fake_embeds.mean() * 0.0  # no gradient contribution
    inputs_embeds = inputs_embeds + fake_embeds
```

---

## 2. Sequence Parallelism

### 2.1 Language Model — Position Embedding Slicing

```python
position_embeddings = self.rotary_emb(hidden_states, position_ids)

sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
```

VeOmni automatically registers a wrapped FlashAttention, so attention layers need no further changes.

### 2.2 Vision Transformer — Padding and Slicing

The data collator pads and sequence-slices `hidden_states`, but `grid_thw` and `cu_seqlens` remain unpadded and unsliced. Use `sp_pad_and_slice` to align position embeddings:

```python
pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
if sp_group is not None:
    pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)

hidden_states = hidden_states + pos_embeds

cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
    dim=0, dtype=torch.int32,
)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

rotary_pos_emb = self.rot_pos_emb(grid_thw)
total_seq_len = cu_seqlens[-1]
seq_len, _ = hidden_states.size()
rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)
emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
position_embeddings = (emb.cos(), emb.sin())

if sp_group is not None:
    cos, sin = position_embeddings
    cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
    sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
    position_embeddings = (cos, sin)
```

> **Why `pad_scale=4`?** Qwen3-VL performs a 4-to-1 spatial merge at the end of the ViT, so the collator pads vision sequences to multiples of 4. Position embeddings must match.

Also extend `cu_seqlens` with a padding entry to cover the padded tail on the last rank:

```python
total_seq_len = cu_seqlens[-1]
seq_len = hidden_states.size(0)

if sp_group is not None:
    sp_size = get_parallel_state().sp_size
    pad_seq_len = seq_len * sp_size - total_seq_len.item()
    if pad_seq_len > 0:
        cu_seqlens = torch.cat([cu_seqlens, (cu_seqlens[-1] + pad_seq_len).unsqueeze(0)])
```

### 2.3 ViT-to-LM Fill-Back (3-Step Dance)

After ViT processing, image embeddings must be scattered into the correct positions in `inputs_embeds`. Under SP, `inputs_embeds` is sequence-sliced, so a temporary layout change is needed:

**Step 1:** Gather sequence, scatter heads:
```python
if self.training and get_parallel_state().sp_enabled:
    # (batch, seq//sp, hidden) → (batch, seq, hidden//sp)
    inputs_embeds = gather_seq_scatter_heads(
        inputs_embeds, seq_dim=1, head_dim=-1, group=get_parallel_state().sp_group
    )
```

**Step 2a:** Apply the same transform to image embeddings:
```python
if self.training and get_parallel_state().sp_enabled:
    # (seq//sp, hidden) → (seq, hidden//sp)
    image_embeds = gather_seq_scatter_heads(
        image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
    )
```

**Step 2b:** Fill back using `image_mask` (pre-computed in `process_sample`, kept unsliced):
```python
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

**Step 3:** Restore SP layout:
```python
if self.training and get_parallel_state().sp_enabled:
    # (batch, seq, hidden//sp) → (batch, seq//sp, hidden)
    inputs_embeds = gather_heads_scatter_seq(
        inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
    )
```

The same logic applies to video embeddings.

### 2.4 Deepstack Visual Embeddings

Deepstack embeddings are injected into multiple decoder layers. Instead of running All2All at every layer, all-gather once after ViT and slice per rank:

```python
if pixel_values is not None and get_parallel_state().sp_enabled:
    sp_group = get_parallel_state().sp_group
    sp_size = get_parallel_state().sp_size
    sp_rank = get_parallel_state().sp_rank

    # (seq//sp, hidden) → (seq, hidden)
    deepstack_image_embeds = [
        _Gather.apply(sp_group, embed, 0, False) for embed in deepstack_image_embeds
    ]

    image_mask_1d = image_mask[..., 0]  # (batch, seq)
    seq_len = image_mask_1d.shape[1]
    seq_per_rank = seq_len // sp_size
    rank_start = sp_rank * seq_per_rank

    rank_mask = image_mask_1d[:, rank_start : rank_start + seq_per_rank]
    offset = image_mask_1d[:, :rank_start].sum().item()
    n_tokens = rank_mask.sum().item()

    deepstack_image_embeds = [e[offset : offset + n_tokens] for e in deepstack_image_embeds]
```

Each rank holds only the deepstack tokens for its sequence partition. No further communication is needed in the LM deepstack layers.

---

## 3. Expert Parallelism

### 3.1 Parallel Plan

```python
from torch.distributed._tensor import Shard
from ....distributed.parallel_plan import ParallelPlan

def get_parallel_plan():
    ep_plan = {
        "model.language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.language_model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    return ParallelPlan(ep_plan=ep_plan)
```

### 3.2 Fused MoE Forward

Qwen3-VL MoE uses a fused `gate_up_proj` tensor `(num_experts, hidden, 2 * expert_dim)`. Split and transpose before calling `fused_moe_forward`:

```python
if self.training and self.moe_implementation == "fused":
    gate_proj = self.gate_up_proj[..., : self.expert_dim]
    up_proj   = self.gate_up_proj[..., self.expert_dim :]

    gate_proj_t = gate_proj.transpose(1, 2).contiguous()
    up_proj_t   = up_proj.transpose(1, 2).contiguous()
    down_proj_t = self.down_proj.transpose(1, 2).contiguous()

    next_states = fused_moe_forward(
        module=self,
        num_experts=self.num_experts,
        routing_weights=routing_weights_topk,
        selected_experts=router_indices,
        hidden_states=hidden_states,
        fc1_1_weight=gate_proj_t,
        fc1_2_weight=up_proj_t,
        fc2_weight=down_proj_t,
    )
elif self.training and self.moe_implementation == "eager":
    assert not get_parallel_state().ep_enabled, "_moe_implementation='eager' does not support EP"
    # ... standard nn.ModuleList path ...
```

---

## 4. Performance: Pre-compute `max_seqlen`

`(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()` causes a CPU-GPU sync. Move it outside the block loop:

```python
max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()

for blk in self.blocks:
    hidden_states = blk(
        hidden_states=hidden_states,
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        max_seqlen=max_seqlen,
    )
```

Pop LM flash-attention kwargs before ViT forward, restore before LM forward:

```python
flash_attn_kwargs = {}
for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
    if key in kwargs:
        flash_attn_kwargs[key] = kwargs.pop(key)

# ... ViT and audio encoder forwards ...

kwargs.update(flash_attn_kwargs)
outputs = self.language_model(..., **kwargs)
```

---

## 5. Model Registration

In [veomni/models/transformers/__init__.py](../../../veomni/models/transformers/__init__.py):
```python
from . import qwen3_vl_moe
```

In your model's `__init__.py`:
```python
from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY

@MODEL_CONFIG_REGISTRY.register("qwen3_vl_moe")
def register_qwen3_vl_moe_config():
    from .configuration_qwen3_vl_moe import Qwen3VLMoeConfig
    return Qwen3VLMoeConfig

@MODELING_REGISTRY.register("qwen3_vl_moe")
def register_qwen3_vl_moe_modeling(architecture: str):
    from .modeling_qwen3_vl_moe import Qwen3VLMoeForCausalLM
    return Qwen3VLMoeForCausalLM

@MODEL_PROCESSOR_REGISTRY.register("Qwen3VLMoeProcessor")
def register_qwen3_vl_moe_processor():
    from .processing_qwen3_vl_moe import Qwen3VLMoeProcessor
    return Qwen3VLMoeProcessor
```

Expose `get_position_id_func` from the model:

```python
def get_position_id(main_func, self, **kwargs):
    # Must be a module-level function for multiprocessing pickle
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}

def get_position_id_func(self):
    fake_model = SimpleNamespace(config=self.config)
    return partial(get_position_id, Qwen3VLMoeModel.get_rope_index, fake_model)
```

---

## Acknowledgements

Thanks to ByteDance Seed and AML team: Qianli Ma, Zhelun Shi, Yifan Pi, Tianle Zhong, Xiao Yu.
