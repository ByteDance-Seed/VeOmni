# MiniMax M3 VL

MiniMax M3 VL is registered as `minimax_m3_vl` under VeOmni's transformers backend. The generated modeling files are based on `transformers==5.16.0`: `transformers.models.minimax_m3_vl` first shipped in 5.12, but 5.16 changed the indexer to select key blocks per KV group and renamed the vision tower argument to `grid_thw`, and the VeOmni patches target that version.

GPU:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml
```

NPU:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/minimax_m3_vl/minimax_m3_vl_npu.yaml
```

The public checkpoint can be referenced through either Hugging Face or ModelScope:

- Hugging Face: `MiniMaxAI/MiniMax-M3`
- ModelScope: `MiniMax/MiniMax-M3`

## Files

- `configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml` (GPU)
- `configs/multimodal/minimax_m3_vl/minimax_m3_vl_npu.yaml` (NPU)
- `veomni/models/transformers/minimax_m3_vl/processing_minimax_m3_vl.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py`
- `veomni/models/transformers/minimax_m3_vl/parallel_plan.py`
- `veomni/models/transformers/minimax_m3_vl/checkpoint_tensor_converter.py`

## Data Path

The `minimax_m3_vl` data transform reuses VeOmni's multimodal fetch and collate pipeline, then delegates image/video tensorization and conversation rendering to the MiniMax Hugging Face processor:

- `processor.image_processor(..., return_tensors="pt")` emits `pixel_values` and `image_grid_thw`.
- `processor.video_processor(..., return_metadata=True)` emits `pixel_values_videos`, `video_grid_thw`, and metadata used to expand MiniMax video timestamp tokens.
- `processor.apply_chat_template(..., tokenize=False)` applies the checkpoint's native MiniMax conversation protocol; VeOmni tokenizes the rendered string and masks the generation header, non-assistant turns, and visual placeholders in the labels.
- `MODEL_PROCESSOR_REGISTRY` maps both `MiniMaxM3VLProcessor` (upstream) and `MiniMaxVLProcessor` (the class in the checkpoint's bundled `processing_minimax.py`) to `veomni/models/transformers/minimax_m3_vl/processing_minimax_m3_vl.py`, so training always runs the transformers>=5.16 implementation the generated modeling is written against, whatever the checkpoint's `auto_map` resolves to. Training never uses the checkpoint's bundled processor: the loader's initial `AutoProcessor` call (with `trust_remote_code=True`) still imports it once to read its class name, then VeOmni's processor replaces it, and that processor does not go through `AutoProcessor` / `ProcessorMixin.from_pretrained` at all. Instead it builds the tokenizer and the transformers-native `MiniMaxM3VLImageProcessor` / `MiniMaxM3VLVideoProcessor` by concrete class (each reads all its parameters from the checkpoint's own config files) and wires the chat template in explicitly. This sidesteps two ways the public `MiniMaxAI/MiniMax-M3` layout is unusable:
  - Its `preprocessor_config.json` declares the image/video processors *only* through a `trust_remote_code` `auto_map` (no `image_processor_type` / `video_processor_type`). Going through `AutoProcessor` would demand remote code — and once `get_model_processor` strips `trust_remote_code`, the loader's fallback masks the failure as a misleading *"does not appear to have a file named processor_config.json"*. Building the native classes by name needs no remote code.
  - The bundled `MiniMaxVLProcessor.__init__` drops `**kwargs` and so loses the `chat_template.jinja` that `from_pretrained` passes through it (checkpoints that keep the template in `tokenizer_config.json` land in the same state). VeOmni takes the template straight from the tokenizer, which parses whichever file carries it, so the template is never lost — which otherwise surfaces as *"this processor does not have a chat template"* on the first sample.
- `MainCollator` packs `pixel_values`, `pixel_values_videos`, `image_grid_thw`, and `video_grid_thw` through the existing VLM collate rules.
- The MiniMax generated model exposes `get_metadata_collate_func()`, which converts packed `image_grid_thw` / `video_grid_thw` into `multimodal_metadata` grid lists on CPU. The vision tower consumes those lists to avoid calling `grid_thw.tolist()` inside the CUDA/NPU forward path.

MiniMax placeholder ids are preserved in `input_ids` so the upstream forward can scatter vision features by `config.image_token_id` and `config.video_token_id`. Labels for placeholder tokens are masked with `IGNORE_INDEX`.

## Current Scope

This recipe covers config loading, generated modeling import, MiniMax processor-shaped VLM samples, FSDP2 training, checkpoint conversion, multimodal metadata wiring, expert parallelism, and packed Ulysses sequence parallelism. Context parallelism is not supported, so keep `cp_size: 1`.

`model.ops_implementation.attn_implementation` controls the vision tower only. Keep it on a FlashAttention variant: ViT uses the varlen metadata to isolate multiple images/videos, and VeOmni's `*_with_sp` wrapper performs the ViT Ulysses all-to-all when `ulysses_size > 1`. The language tower deliberately ignores this generic setting for now. Each decoder layer keeps its input and output in local packed `[1, T_local, D]` form and applies RoPE to local Q/K before communication. Under Ulysses, the main Q/K/V use the sequence/head all-to-all, the per-KV-group index Q is split across ranks exactly like K, and the single-head index K is all-gathered. Block selection and attention then run through the packed TND operators in `veomni/ops/kernels/minimax_sparse_attention/`, which take the standard `cu_seq_lens_q` / `max_length_q` varlen metadata. The current eager implementation right-pads internally for MiniMax's reference math; that padded layout never leaves the operator. The language model accepts only packed input: VeOmni's collator (or verl's remove-padding path) supplies the varlen kwargs, and a caller without them may pass one unpadded sequence with batch size 1, which is run as a single-segment pack. Padded `[B, S]` batches (e.g. verl `use_remove_padding=False`), KV cache / `generate`, and context parallelism are rejected with an explicit error.

MiniMax's Gemma-style RMSNorm is wired to VeOmni's `rms_norm/qwen3_5` operator variant, selecting Liger on GPU and `torch_npu.npu_rms_norm` on NPU according to `model.ops_implementation.rms_norm_implementation`.

MiniMax routed experts use the `moe_experts/swiglu_oai` operator variant. It preserves the model's clamped `(up + 1) * gate * sigmoid(alpha * gate)` activation while reusing VeOmni's standard EP token dispatch. Enable EP with `fused_triton` on GPU or `fused_npu` on NPU:

```yaml
model:
  ops_implementation:
    moe_implementation: fused_triton  # use fused_npu on NPU
  accelerator:
    ep_size: 8
```

Router auxiliary load balancing is intentionally disabled for MiniMax M3. The upstream Switch-style loss recomputes routing with `softmax(router_logits)`, while MiniMax selects experts with `sigmoid(router_logits) + e_score_correction_bias`; these can select different experts. Keep `output_router_logits=false`. VeOmni raises an error if router-logit capture is requested until a MiniMax-specific balancing recipe is available.

The public MiniMax checkpoint stores separate per-expert `w1`/`w2`/`w3` tensors. Its runtime converter must first assemble the complete fused expert tensor, so this checkpoint layout is incompatible with `model.ep_sharded_stream_load=true`. Keep that option disabled unless the checkpoint has first been exported in the fused VeOmni/Hugging Face v5 layout.

Exported Hugging Face weights (`save_hf_weights`) use the transformers-native layout: `model.language_model.*`, fused `mlp.experts.gate_up_proj` / `down_proj`, `self_attn.indexer.*`, and a `config.json` with `layer_types` / `index_*` fields. Transformers loads them directly, but vLLM and SGLang read the original MiniMax layout (`language_model.model.*`, per-expert `block_sparse_moe.experts.{j}.w{1,2,3}`, `index_{q,k}_proj`) and the legacy `sparse_attention_config` / `moe_layer_freq` config fields, so an export cannot be served by them without converting back. For the same reason, RL frameworks that sync actor weights into a vLLM/SGLang rollout (e.g. verl's VeOmni engine) are not supported yet: the checkpoint converter does not implement `export_weights`, and the rollout loader silently skips the unmatched names.

To regenerate generated modeling files:

```bash
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config \
  -o veomni/models/transformers/minimax_m3_vl/generated --diff
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_npu_patch_gen_config \
  -o veomni/models/transformers/minimax_m3_vl/generated --diff
```
