# How to Enable New Models in VeOmni
**Author**: Juntian Liu

**TLDR:** This tutorial demonstrates how to enable new models in VeOmni using a combination of FSDP, Expert Parallelism (EP), and Sequence Parallelism (SP/Ulysses), using Qwen3-VL MoE as a practical example. We'll modify the modeling code from HuggingFace directly to support distributed training at scale.

---

## Overview

Enabling a new multimodal model in VeOmni requires careful integration of three parallelism strategies:
- **FSDP**: Native PyTorch support for sharding the model's parameters, gradients, and optimizer states across multiple GPUs, enabling efficient training of large models that exceed single-GPU memory capacity.
- **SP**: Distributing input along the sequence dimension across GPUs
- **EP**: Sharding MoE experts across GPUs

This guide uses Qwen3-VL MoE as a reference implementation, showing the specific code modifications needed for each parallelism type.

---

## 1. FSDP Support - Native Integration

FSDP support in VeOmni is straightforward thanks to PyTorch's native FSDP implementation. Most models work out-of-the-box with minimal modifications.

### Dummy ViT Forward for VLMs

This is **only required for VLM (Vision-Language Models)** and **only for the ViT (Vision Transformer) component**. The dummy forward is needed for **both image and video inputs** to prevent FSDP reduce-scatter hangs.

When using FSDP, if some ranks in the same FSDP group receive `None` for `pixel_values` (images) or `pixel_values_videos` (videos) while others receive valid inputs, the backward reduce-scatter operation will hang.

#### Reference Implementation in Vision Model
```python
def dummy_forward(self):
    """
    Dummy forward to avoid FSDP reduce-scatter hang when some ranks get None pixel_values
    This is only needed for VLM's ViT component, for both image and video inputs
    """
    if get_parallel_state().fsdp_enabled:
        sp_size = get_parallel_state().sp_size
        pixel_values = torch.zeros([16, 3 + 2 * 16 + 16], dtype=self.dtype, device=self.device)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
        dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        return self(**dummy_data)
```

#### Add into the Main Forward Pass
```python
# For image inputs as the example
if pixel_values is not None:
    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    # ...
elif get_parallel_state().fsdp_enabled:
    # Dummy ViT forward for image path
    fake_embeds, fake_deepstack = self.visual.dummy_forward()
    fake_embeds = fake_embeds.mean() * 0.0
    fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds + fake_embeds
```

This ensures all ranks participate in collective operations without affecting the actual result (multiplying by 0.0 ensures no contribution to gradients).

---

## 2. Sequence Parallelism Support

VeOmni automatically registers the wrapped FlashAttention from veomni/ops/attention.py, and we should use causallm_loss_function from veomni/ops/loss.py to support Ulysses within the unified interfaces. You also need to handle sequence-dimension slicing for embeddings carefully and manage the tensor transformations between the ViT and language model components.

### 2.1 Language Model Part - Simple Position Embedding Slicing

For the language model, the modification is straightforward - just a few extra lines to slice position embeddings to match each rank's hidden states:
```python
# create complete position embeddings to be shared across the decoder layers
position_embeddings = self.rotary_emb(hidden_states, position_ids)

# slice position embedding if using sp
sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
```

The `slice_position_embedding` function handles distributing the position embeddings across ranks to match the sequence-sliced hidden states.

### 2.2 Vision Transformer (ViT) Part - Padding and Slicing

The ViT component needs to be handled carefully because of padding considerations and the need to process the unpadded and unsliced grid_thw

#### Key Challenge: Padding and Grid Handling

The vision hidden states are **padded and sliced** in the data collator, but `grid_thw` (temporal, height, width grid information) remains **unpadded and unsliced**. This creates a mismatch that must be handled carefully:

1. Position embeddings are computed from the **raw grid_thw**
2. Hidden states have been **padded and sequence-sliced**
3. `cu_seqlens` (cumulative sequence lengths) are also computed from **raw grid_thw**

We need to use `sp_pad_and_slice` to pad and slice position embeddings to match the padded hidden states.


#### Reference Implementation
```python
# Compute position embeddings from raw grid
pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

# slice pos embedding if using sp so that the sharded hidden_states receive their corresponding position embeddings
sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
if sp_group is not None:
    # We need to do padding here because hidden_states was padded with pad_scale=4
    pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)

hidden_states = hidden_states + pos_embeds

# Compute cumulative sequence lengths from raw grid
cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
    dim=0,
    dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

# Compute rotary position embeddings
rotary_pos_emb = self.rot_pos_emb(grid_thw)

# Get before-sliced full seq from cu_seqlens
# total_seq_len should equal to seq_len when not using SP
total_seq_len = cu_seqlens[-1]
seq_len, _ = hidden_states.size()
hidden_states = hidden_states.reshape(seq_len, -1)
rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)
emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
position_embeddings = (emb.cos(), emb.sin())

# slice pos embedding if using sp to let sharded hidden_states get its corresponding pos embedding
if sp_group is not None:
    cos, sin = position_embeddings
    # Apply same padding and slicing as hidden_states
    cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
    sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
    position_embeddings = (cos, sin)
```

#### Why `pad_scale=4`?

Qwen3-VL performs a **4-to-1 spatial merge** at the end of the ViT. The data collator pads vision sequences to multiples of 4 to ensure this merge operation works correctly. Since `grid_thw` is unpadded but hidden states are padded with pad_scale=4, `sp_pad_and_slice` with `pad_scale=4` ensures position embeddings are padded and sliced to match.

### 2.3 ViT-LM Connection Part
Connecting the ViT to the LM requires carefully orchestrating gather_seq_scatter_heads (All2All) operations to merge the vision embeddings back into the full LM input; after ViT processing, this merge back into the original sequence proceeds through three sub-steps (using image fill-back as an example).

**Step 1: Process the Input Embeddings**

Apply the All2All operation to the sp-sliced `inputs_embeds` (including both vision placeholder and text):
```python
if self.training and get_parallel_state().sp_enabled:
    # Input:  (batch_size, seq_len // sp_size, hidden_size)
    # Output: (batch_size, seq_len, hidden_size // sp_size)
    inputs_embeds = gather_seq_scatter_heads(
        inputs_embeds, seq_dim=1, head_dim=-1, group=get_parallel_state().sp_group
    )
```

**Step 2.3b: Gather Complete Image Sequence Length**

Also use `gather_seq_scatter_heads` on ViT-processed image embeddings to also gather along the `seq_len` dimension and scatter along the `hidden_size` dimension:
```python
 if self.training and get_parallel_state().sp_enabled:
    # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
    image_embeds = gather_seq_scatter_heads(
        image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
    )
```


**Step 2.3b: Fill Back to Correct Positions**

- The `image_mask` marks the positions of image tokens. This mask is precomputed in process_sample and kept unsliced during data preprocessing to avoid unnecessary communication during training and to ensure correct handling under sequence parallelism.

- `image_embeds` are ViT-processed and after All2All image features, with shape `(seq_length, hidden_size // sp_size)`

Use `masked_scatter` to fill image features back into the corresponding positions in input embeddings:
```python
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

Each rank only fills only a portion of `input_embeds`, but this portion is **partitioned at hidden_size dimension**, not at the sequence dimension.

**Final Transformation**

Finally, restore to normal SP partitioning before sending to the LM part:
```python
if self.training and get_parallel_state().sp_enabled:
    # Restore: (batch_size, seq_len, hidden_size // sp_size)
    #       -> (batch_size, seq_len // sp_size, hidden_size)
    inputs_embeds = gather_heads_scatter_seq(
        inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
    )
```

The same logic applies to video embeddings processing.

### 2.4 Special Case: Deepstack Visual Embeddings

Deepstack embeddings require a different approach: **All-Gather followed by rank-specific slicing**. This is necessary because we want to avoid multiple extra All2All operations later in `_deepstack_process`.

#### Why This Approach?

Instead of keeping deepstack embeddings distributed and requiring All2All communication in every LM deepstack layer, we
1. All-Gather deepstack embeddings once after ViT
2. Slice them according to each rank's sequence partition using masks
3. Each rank gets its local deepstack embeddings and masks
4. In later LM parts, **no extra communication is needed**

#### Reference SP Processing for Image Deepstack
```python
if pixel_values is not None:
    # Do all_gather on deepstack_image_embeds
    # (seq_len // sp_size, hidden_size) -> (seq_len, hidden_size)
    deepstack_image_embeds = [
        _Gather.apply(get_parallel_state().sp_group, embed, 0, False)
        for embed in deepstack_image_embeds
    ]

    # Now use image_mask to select visual tokens for this rank's sequence slice
    image_mask_1d = image_mask[..., 0]  # (batch_size, seq_len)

    # Determine which sequence positions belong to this rank
    seq_len = image_mask_1d.shape[1]
    seq_per_rank = seq_len // sp_size
    rank_start = sp_rank * seq_per_rank
    rank_end = rank_start + seq_per_rank

    # Get the mask for this rank's sequence slice and save it for later
    rank_image_mask = image_mask_1d[:, rank_start:rank_end]  # (batch_size, seq_len // sp_size)

    # Count how many visual tokens are before this rank's slice
    before_rank_mask = image_mask_1d[:, :rank_start]
    offset = before_rank_mask.sum().item()

    # Count how many visual tokens are in this rank's slice
    num_visual_tokens = rank_image_mask.sum().item()

    # Slice the all-gathered deepstack embeddings
    deepstack_image_embeds = [
        embed[offset : offset + num_visual_tokens]
        for embed in deepstack_image_embeds
    ]
```

The same logic applies to video deepstack embeddings. Performing this step once ensures that each rank has the exact deepstack embeddings required for its sequence partition, removing the need for further communication in the LM deepstack layers.

---

## 3. Expert Parallelism (EP) Support

EP requires two main components: define EP parallel plan and enable fused MoE forward implementation.

### 3.1 Define Parallel Plan

Create a `parallel_plan.py` file to specify which expert layers should be sharded:
```python
from torch.distributed._tensor import Shard
from ....distributed.parallel_plan import ParallelPlan

def get_parallel_plan():
    ep_plan = {
        "model.language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.language_model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    parallel_plan = ParallelPlan(
        ep_plan=ep_plan,
    )
    return parallel_plan
```

### 3.2 Enable Fused MoE Forward

Add the fused MoE MLP implementation to improve performance and enable EP support.

#### Special Handling for Qwen3-VL MoE

**Important**: Qwen3-VL MoE **combines gate and up projections into a single `gate_up_proj` tensor**. We need additional handling to match our `fused_moe_forward` interface, which expects separate gate and up projections.
```python
if self.training and self.moe_implementation == "eager":
    assert not get_parallel_state().ep_enabled, "_moe_implementation='eager' does not support EP"
    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
    # ... standard implementation

elif self.training and self.moe_implementation == "fused":


    # Qwen3-VL MoE combines gate and up into gate_up_proj
    # Split gate_up_proj into gate_proj and up_proj to match fused_moe_forward interface
    # Current: gate_up_proj shape is (num_experts, hidden_size, 2 * expert_dim)
    gate_proj = self.gate_up_proj[..., : self.expert_dim]
    up_proj = self.gate_up_proj[..., self.expert_dim :]

    # Transpose weights to match expected shape for fused_moe_forward
    gate_proj_t = gate_proj.transpose(1, 2).contiguous()
    up_proj_t = up_proj.transpose(1, 2).contiguous()
    down_proj_t = self.down_proj.transpose(1, 2).contiguous()

    next_states = fused_moe_forward(
        module=self,
        num_experts=self.num_experts,
        routing_weights=routing_weights_topk,
        selected_experts=router_indices,
        hidden_states=hidden_states,
        fc1_1_weight=gate_proj_t,  # Separated gate projection
        fc1_2_weight=up_proj_t,    # Separated up projection
        fc2_weight=down_proj_t,
    )

    # ...
```

The key modifications are:
1. **Split** `gate_up_proj` into separate `gate_proj` and `up_proj`
2. **Transpose** all weight matrices to match `fused_moe_forward` expected shapes
3. Pass separated projections to `fused_moe_forward`

---

## 4. Remove CPU-GPU Synchronization for Better Performance

CPU-GPU synchronization in attention layers can create significant performance bottlenecks in some cases. We eliminate this issue by precomputing the attention parameters before entering the blocks.

### 4.1 Vision Attention Optimization

Compute max_seqlen to perform the CPU-GPU synchronization once, then pass the value to be shared for all layers:

```python
# Calculate max_seqlen from cu_seqlens here to avoid per layer CPU-GPU sync
max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()

for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(
        hidden_states=hidden_states,
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        max_seqlen=max_seqlen,  # Pre-computed, no per-layer sync needed
    )
```

### 4.2 Language Model Attention Optimization

Pre-compute FlashAttention kwargs from 3D position IDs:
```python
# Pre-compute flash_attn_kwargs in the data_collator to avoid per-layer CPU-GPU sync
# Store in kwargs before pass to language model

# Pop these keys before the ViT parts to prevent the ViT attention from using them
flash_attn_kwargs = {}
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        if key in kwargs:
            flash_attn_kwargs[key] = kwargs.pop(key)

# Later, restore these kwargs before calling language_model
kwargs.update(flash_attn_kwargs)

outputs = self.language_model(
    input_ids=None,
    position_ids=position_ids,
    inputs_embeds=inputs_embeds,
    # ... other args
)
```

---

## 5. Register Your Model

Finally, register your model class so VeOmni can automatically match it with the config.

### In `veomni/models/transformers/__init__.py`

Add your model to the lists:
```python
from .qwen3_vl_moe import (
    qwen3_vl_moe,
)

__all__ = [
    "qwen3_vl_moe",
]
```

### In your model file

Export the model class:
```python
# Register the cutomized model in __init__.py so that VeOmni uses our custom modeling/config/processor code instead of the Hugging Face version
from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY, MODEL_PROCESSOR_REGISTRY

@MODEL_CONFIG_REGISTRY.register("qwen3_vl_moe")
def register_qwen3_vl_moe_config():
    from .configuration_qwen3_vl_moe import Qwen3VLMoeConfig
    return Qwen3VLMoeConfig

@MODELING_REGISTRY.register("qwen3_vl_moe")
def register_qwen3_vl_moe_modeling(architecture: str):
    from . import modeling_qwen3_vl_moe
    from .modeling_qwen3_vl_moe import Qwen3VLMoeForCausalLM, Qwen3VLMoeForSequenceClassification, Qwen3VLMoeForTokenClassification
    if "ForCausalLM" in architecture:
        return Qwen3VLMoeForCausalLM
    elif "ForSequenceClassification" in architecture:
        return Qwen3VLMoeForSequenceClassification
    elif "ForTokenClassification" in architecture:
        return Qwen3VLMoeForTokenClassification
    else: # None
        return Qwen3VLMoeForCausalLM

@MODEL_PROCESSOR_REGISTRY.register("Qwen3VLMoeProcessor")
def register_qwen3_vl_moe_processor():
    from . import processing_qwen3_vl_moe
    from .processing_qwen3_vl_moe import Qwen3VLMoeProcessor
    return Qwen3VLMoeProcessor
```

### Additional Helper Functions

Add helper functions for position ID computation:
```python
# Add the get_position_id_func to be used in data_transform
def get_position_id_func(self):
    fake_model = SimpleNamespace(config=self.config)
    return partial(get_position_id, Qwen3VLMoeModel.get_rope_index, fake_model)
```

And wrap the position ID function:
```python
# Wrapped Qwen3VLMoeModel.get_rope_index to use in process_sample for obtaining position_ids in advance in process_sample
def get_position_id(main_func, self, **kwargs):
    # must be a global func for multiprocessing serialize
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}
```

---

## Summary

Enabling a new model in VeOmni with full parallelism support involves:

1. **FSDP**: Add dummy forward paths to ViT components (for VLMs only) to handle both image and video inputs and prevent collective operation hangs
2. **Sequence Parallelism**:
   - LM: Simple position embedding slicing with `slice_position_embedding`
   - ViT: Careful padding and slicing with `sp_pad_and_slice` to handle data collator padding
   - ViT-LM connection: Multi-step gather/scatter operations to merge vision features
   - Deepstack: All-Gather once and slice per-rank to eliminate later communication
3. **Expert Parallelism**: Define parallel plans, implement fused MoE forward, handle combined weight tensors (like Qwen3-VL's `gate_up_proj`)
4. **Performance**: Pre-compute attention parameters to eliminate CPU-GPU synchronization
5. **Registration**: Export model classes and helper functions

By following these patterns from the Qwen3-VL MoE implementation, you can adapt most Hugging Face models to work efficiently with VeOmni’s distributed training infrastructure

## Acknowledgements
Thanks to ByteDance Seed and AML team: Qianli Ma, Zhelun shi, Yifan Pi, Tianle Zhong and Xiao Yu
