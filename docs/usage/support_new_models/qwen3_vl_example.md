# Qwen3-VL MoE Integration Example

**Author**: Juntian Liu

This document walks through the specific patches applied to integrate **Qwen3-VL MoE** into VeOmni. It is a concrete example of the patterns described in [guide_and_checklist.md](./guide_and_checklist.md), covering FSDP, Sequence Parallelism, Expert Parallelism, and model registration.

---

## 1. FSDP: Dummy ViT Forward

When using FSDP, ranks that receive `None` for `pixel_values` or `pixel_values_videos` while other ranks receive valid tensors will cause a backward reduce-scatter hang. Add a `dummy_forward` to the ViT:

```python
def dummy_forward(self, encoder_data_balance=None):
    """
    Dummy forward to avoid FSDP reduce-scatter hang when some ranks get None pixel_values.
    Also handles encoder_data_balance communication hang.
    Needed for both image and video inputs.
    """
    if get_parallel_state().sp_enabled:
        sp_size = get_parallel_state().sp_size
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        # If using SP, pixel_values is sliced but grid_thw is not
        grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
    else:
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)

        if encoder_data_balance is not None:
            # add dummy data to avoid encoder data balance communication hang
            pixel_values, grid_thw = encoder_data_balance.balance_data(pixel_values, grid_thw)
            dummy_image_embeds, dummy_deepstack_image_embeds = self(hidden_states=pixel_values, grid_thw=grid_thw)
            dummy_image_embeds, dummy_deepstack_image_embeds = encoder_data_balance.data_bridge(
                dummy_image_embeds, dummy_deepstack_image_embeds
            )
            return dummy_image_embeds, dummy_deepstack_image_embeds

    return self(hidden_states=pixel_values, grid_thw=grid_thw)
```

> **SP handling:** Under SP, the collator slices `pixel_values` per rank but leaves `grid_thw` unsliced. The dummy forward must match this: scale `grid_thw` height by `sp_size` so the ViT sees a consistent full-sequence grid.

Call it in the main forward when inputs are `None`. Note the condition also covers `encoder_data_balance` to avoid communication hangs:

```python
if pixel_values is not None:
    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    # ...
elif get_parallel_state().fsdp_enabled or self.encoder_data_balance is not None:
    fake_embeds, fake_deepstack = self.visual.dummy_forward(encoder_data_balance=self.encoder_data_balance)
    fake_embeds = fake_embeds.mean() * 0.0  # no gradient contribution
    fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds + fake_embeds
```

The same condition and pattern applies to `pixel_values_videos`.

---

## 2. Sequence Parallelism

### 2.1 Language Model — Position Embeddings

Since `position_ids` is already sliced by `SequenceParallelCollator`, calling `rotary_emb(hidden_states, position_ids)` directly produces local-length position embeddings — no additional slicing is needed.

VeOmni automatically registers a wrapped FlashAttention, so attention layers need no further changes.

### 2.2 Vision Transformer — Padding and Slicing

The data collator pads and sequence-slices `hidden_states`, but `grid_thw` and `cu_seqlens` remain unpadded and unsliced. Use `sp_pad_and_slice` to align position embeddings:

```python
hidden_states = self.patch_embed(hidden_states)

pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

# Patch.1: slice pos embeddings to match SP-sliced hidden_states
if get_parallel_state().sp_enabled:
    pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)

hidden_states = hidden_states + pos_embeds

cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
    dim=0, dtype=torch.int32,
)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

rotary_pos_emb = self.rot_pos_emb(grid_thw)
seq_len, _ = hidden_states.size()
# Patch.2: reshape using cu_seqlens[-1] (the actual total seq len before SP padding)
rotary_pos_emb = rotary_pos_emb.reshape(cu_seqlens[-1], -1)
emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
position_embeddings = (emb.cos(), emb.sin())

# Patch.3: slice rotary embeddings to match SP-sliced hidden_states
if get_parallel_state().sp_enabled:
    cos, sin = position_embeddings
    cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
    sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
    position_embeddings = (cos, sin)
```

> **Why `pad_scale=4`?** Qwen3-VL performs a 4-to-1 spatial merge at the end of the ViT, so the collator pads vision sequences to multiples of 4. Position embeddings must match.

Also extend `cu_seqlens` with a padding entry to cover the padded tail on the last rank:

```python
# Patch.4: pad cu_seqlens to match the padded hidden_states under SP
if get_parallel_state().sp_enabled:
    sp_size = get_parallel_state().sp_size
    pad_seq_len = seq_len * sp_size - cu_seqlens[-1].item()
    if pad_seq_len > 0:
        new_cumsum = cu_seqlens[-1] + pad_seq_len
        cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
```

### 2.3 ViT-to-LM Fill-Back (3-Step Dance)

After ViT processing, image embeddings must be scattered into the correct positions in `inputs_embeds`. Under SP, `inputs_embeds` is sequence-sliced, so a temporary layout change is needed:

**Step 1:** Gather sequence, scatter heads:
```python
if get_parallel_state().sp_enabled:
    # (batch, seq//sp, hidden) → (batch, seq, hidden//sp)
    inputs_embeds = gather_seq_scatter_heads(
        inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
    )
```

**Step 2a:** Apply the same transform to image embeddings:
```python
if get_parallel_state().sp_enabled:
    # (seq//sp, hidden) → (seq, hidden//sp)
    image_embeds = gather_seq_scatter_heads(
        image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
    )
```

**Step 2b:** Fill back using `image_mask` (pre-computed in `process_sample`, kept unsliced):
```python
embeds_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
image_embeds = image_embeds[:n_image_tokens].to(inputs_embeds.device, inputs_embeds.dtype)
inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)
```

**Step 3:** Restore SP layout:
```python
if get_parallel_state().sp_enabled:
    # (batch, seq, hidden//sp) → (batch, seq//sp, hidden)
    inputs_embeds = gather_heads_scatter_seq(
        inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
    )
```

The same logic applies to video embeddings.

### 2.4 Deepstack Visual Embeddings

Deepstack embeddings are injected into multiple decoder layers. Instead of running All2All at every layer, all-gather once after ViT and slice per rank. The actual implementation uses `gather_outputs` (no autograd) rather than `_Gather.apply`:

```python
if pixel_values is not None:
    # ...image fill-back...

    # sequence parallel patch for image_mask & deepstack_image_embeds
    if get_parallel_state().sp_enabled:
        # All-gather deepstack: (seq//sp, hidden) → (seq, hidden)
        deepstack_image_embeds = [
            gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
            for embed in deepstack_image_embeds
        ]

        seq_len = image_mask.shape[1]
        seq_per_rank = seq_len // get_parallel_state().sp_size
        rank_start = get_parallel_state().sp_rank * seq_per_rank
        rank_end = rank_start + seq_per_rank

        deepstack_offset = image_mask[:, :rank_start].sum().item()
        image_mask = image_mask[:, rank_start:rank_end]
        deepstack_len = image_mask.sum().item()

        deepstack_image_embeds = [
            embed[deepstack_offset : deepstack_offset + deepstack_len] for embed in deepstack_image_embeds
        ]
```

Each rank holds only the deepstack tokens for its sequence partition. No further communication is needed in the LM deepstack layers.

> **`gather_outputs` vs `_Gather.apply`:** `gather_outputs` is a no-autograd all-gather, safe to use here because deepstack embeddings are already detached from the ViT backward graph at this point. Use `_Gather.apply` only when gradient flow through the all-gather is needed.

---

## 3. Expert Parallelism

### 3.1 Parallel Plan

The HF model stores expert weights as a fused `gate_up_proj` of shape `(num_experts, hidden_size, 2 * expert_dim)`. The EP plan shards both it and `down_proj` along the experts dimension:

```python
from torch.distributed._tensor import Shard
from ....distributed.parallel_plan import ParallelPlan

def get_parallel_plan():
    ep_plan = {
        "model.language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.language_model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    return ParallelPlan(extra_parallel_plan={"ep": ep_plan})
```

### 3.2 Fused MoE Forward

Qwen3-VL MoE uses a fused `gate_up_proj` tensor of shape `(num_experts, hidden_size, 2 * expert_dim)`. The `fused_moe_forward` kernel expects `(num_experts, expert_dim, hidden_size)`, so split and transpose before calling:

```python
def fused_moe_forward(self, hidden_states, router_weights, router_indices, routing_weights):
    hidden_states = hidden_states.reshape(-1, self.hidden_size)

    # Split the fused gate_up_proj along the last dim
    gate_proj = self.gate_up_proj[..., : self.expert_dim]   # (num_experts, hidden_size, expert_dim)
    up_proj   = self.gate_up_proj[..., self.expert_dim :]   # (num_experts, hidden_size, expert_dim)

    # Transpose to (num_experts, expert_dim, hidden_size) as expected by fused_moe_forward
    gate_proj_t = gate_proj.transpose(1, 2).contiguous()
    up_proj_t   = up_proj.transpose(1, 2).contiguous()
    down_proj_t = self.down_proj.transpose(1, 2).contiguous()  # (num_experts, hidden_size, expert_dim)

    next_states = fused_moe_forward(
        module=self,
        num_experts=self.num_experts,
        routing_weights=routing_weights,   # compact top-k weights, not the full scatter tensor
        selected_experts=router_indices,
        hidden_states=hidden_states,
        fc1_1_weight=gate_proj_t,
        fc1_2_weight=up_proj_t,
        fc2_weight=down_proj_t,
    )
    next_states = next_states.view(batch_size, -1, self.hidden_size)
    return next_states
```

The `SparseMoeBlock` must pass the compact top-k `routing_weights` (not the full scatter tensor) to `Experts.forward`:

```python
def Qwen3VLMoeTextSparseMoeBlock_forward(self, hidden_states):
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    router_logits = self.gate(hidden_states)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
    hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
    # Pass compact routing_weights (top-k) as 4th arg for fused path
    routed_out = self.experts(hidden_states, router_weights, router_indices, routing_weights)
    return routed_out, router_logits
```

In `Qwen3VLMoeTextExperts.forward`, dispatch based on `moe_implementation`:

```python
def forward(self, hidden_states, router_weights, router_indices, routing_weights=None):
    if self.training and self.moe_implementation == "fused":
        return self.fused_moe_forward(hidden_states, router_weights, router_indices, routing_weights)
    else:
        assert not get_parallel_state().ep_enabled or not self.training, \
            "_moe_implementation='eager' does not support EP"
        return super().forward(hidden_states, router_weights, router_indices)
```

---

## 4. Performance: Pre-compute `max_seqlen`

`(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()` causes a CPU-GPU sync. Move it outside the block loop:

```python
# Patch.5: pre-compute max_seqlen to avoid per-layer CPU-GPU sync
max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()

# Patch.6: move cu_seqlens to CPU when using NPU
if is_torch_npu_available():
    cu_seqlens = cu_seqlens.cpu()

for blk in self.blocks:
    hidden_states = blk(
        hidden_states,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        position_embeddings=position_embeddings,
    )
```

Pop LM flash-attention kwargs before ViT forward, restore before LM forward:

```python
# Patch.7: pop flash-attn kwargs so they don't reach the ViT
text_flash_attn_kwargs = {}
for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
    if key in kwargs:
        text_flash_attn_kwargs[key] = kwargs.pop(key)

# ... ViT and video encoder forwards ...

kwargs.update(text_flash_attn_kwargs)
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
    from .modeling_qwen3_vl_moe import Qwen3VLMoeForCausalLM, apply_veomni_qwen3vlmoe_patch
    apply_veomni_qwen3vlmoe_patch()
    return Qwen3VLMoeForCausalLM

@MODEL_PROCESSOR_REGISTRY.register("Qwen3VLMoeProcessor")
def register_qwen3_vl_moe_processor():
    from .processing_qwen3_vl_moe import Qwen3VLMoeProcessor
    return Qwen3VLMoeProcessor
```

Expose `get_position_id_func` from the model. Note the use of `copy.copy` and VeOmni's token ID constants so `get_rope_index` sees the same IDs as at train time:

```python
import copy
from functools import partial
from types import SimpleNamespace

from ....utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX

def get_position_id(main_func, self, **kwargs):
    # Must be a module-level function for multiprocessing pickle
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}

class Qwen3VLMoeForConditionalGeneration(_Qwen3VLMoeForConditionalGeneration):
    def get_position_id_func(self):
        fake_config = copy.copy(self.config)
        # Use VeOmni constants — process_sample replaces model-specific token IDs with these
        fake_config.image_token_id = IMAGE_INPUT_INDEX
        fake_config.video_token_id = VIDEO_INPUT_INDEX
        fake_model = SimpleNamespace(config=fake_config)
        return partial(get_position_id, Qwen3VLMoeModel.get_rope_index, fake_model)
```

Also handle the position ID transposition for multi-sample batches (VeOmni collates as `(bs, 3, L)`, HF expects `(3, bs, L)`):

```python
# Patch.6: transpose position_ids if VeOmni-collated shape (bs, 3, L)
if position_ids is not None and position_ids.dim() == 3 and position_ids.shape[1] == 3:
    position_ids = position_ids.transpose(0, 1).contiguous()
```

---

## Acknowledgements

Thanks to ByteDance Seed and AML team: Qianli Ma, Zhelun Shi, Yifan Pi, Tianle Zhong, Xiao Yu.
