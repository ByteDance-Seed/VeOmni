# MiniMax M3 VL

MiniMax M3 VL is registered as `minimax_m3_vl` under VeOmni's transformers backend. The generated modeling files are based on `transformers==5.12.0`, because earlier transformers releases do not include `transformers.models.minimax_m3_vl`.

VeOmni's global `transformers-stable` dependency remains unchanged. Use a local MiniMax environment when training or regenerating this model path:

```bash
uv run --no-default-groups --with transformers==5.12.0 --with torch==2.7.1 \
  torchrun --nproc_per_node=8 examples/train_vlm.py \
  --config configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml
```

Equivalent helper:

```bash
NUM_GPUS=8 scripts/multimodal/train_minimax_m3_vl.sh
```

The public checkpoint can be referenced through either Hugging Face or ModelScope:

- Hugging Face: `MiniMaxAI/MiniMax-M3`
- ModelScope: `MiniMax/MiniMax-M3`

## Files

- `configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml`
- `veomni/models/transformers/minimax_m3_vl/configuration_minimax_m3_vl.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py`
- `veomni/models/transformers/minimax_m3_vl/parallel_plan.py`
- `veomni/models/transformers/minimax_m3_vl/checkpoint_tensor_converter.py`
- `scripts/multimodal/train_minimax_m3_vl.sh`

## Data Path

The `minimax_m3_vl` data transform reuses VeOmni's multimodal fetch and collate pipeline, then delegates image and video tensorization to the MiniMax Hugging Face processors:

- `processor.image_processor(..., return_tensors="pt")` emits `pixel_values` and `image_grid_thw`.
- `processor.video_processor(..., return_metadata=True)` emits `pixel_values_videos`, `video_grid_thw`, and metadata used to expand MiniMax video timestamp tokens.
- `MainCollator` packs `pixel_values`, `pixel_values_videos`, `image_grid_thw`, and `video_grid_thw` through the existing VLM collate rules.
- The MiniMax generated model exposes `get_metadata_collate_func()`, which converts packed `image_grid_thw` / `video_grid_thw` into `multimodal_metadata` grid lists on CPU. The vision tower consumes those lists to avoid calling `grid_thw.tolist()` inside the CUDA/NPU forward path.

MiniMax placeholder ids are preserved in `input_ids` so the upstream forward can scatter vision features by `config.image_token_id` and `config.video_token_id`. Labels for placeholder tokens are masked with `IGNORE_INDEX`.

## Current Scope

This recipe validates config loading, generated modeling import, MiniMax processor-shaped VLM samples, toy forward/backward, expert parallel-plan registration, MiniMax multimodal metadata wiring, public checkpoint index coverage, public safetensors shard-header shape coverage, and FSDP asymmetric-rank dummy vision execution. The NPU generated file is present for runtime selection, but this slice does not claim Ascend-specific RMSNorm, RoPE, attention, or MSA kernel replacements. Add those only after collecting NPU runtime evidence.

To regenerate generated modeling files:

```bash
PYTHONPATH=$PWD uv run --no-project --with-editable ./patchgen-pkg --with transformers==5.12.0 \
  --with torch==2.7.1 --with packaging --with psutil --with einops \
  patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config \
  -o veomni/models/transformers/minimax_m3_vl/generated --diff

PYTHONPATH=$PWD uv run --no-project --with-editable ./patchgen-pkg --with transformers==5.12.0 \
  --with torch==2.7.1 --with packaging --with psutil --with einops \
  patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_npu_patch_gen_config \
  -o veomni/models/transformers/minimax_m3_vl/generated --diff
```

To verify public checkpoint index and shard-header metadata coverage without downloading tensor payloads:

```bash
curl -fsSL --retry 5 --retry-all-errors \
  -o /tmp/minimax_m3_model.safetensors.index.json \
  https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/model.safetensors.index.json

PYTHONPATH=$PWD uv run --no-project --python 3.11 \
  --with torch==2.7.1 --with transformers==5.12.0 \
  --with packaging --with psutil --with rich --with numpy --with pillow \
  --with requests --with safetensors --with tqdm \
  scripts/multimodal/verify_minimax_m3_vl_checkpoint_index.py \
  --config-path MiniMaxAI/MiniMax-M3 \
  --index-json /tmp/minimax_m3_model.safetensors.index.json \
  --verify-shard-metadata \
  --metadata-cache-dir /tmp/minimax_m3_safetensors_headers \
  --fail-on-unexpected
```

This check compares converted public index keys with the generated model `state_dict()` keys, then reads only safetensors headers from the 59 public shards through `/resolve/main` Range requests. It proves converted key and shape coverage without downloading the 869 GB tensor payloads. Dtype differences are reported by default; use `--fail-on-dtype-mismatch` only when you want strict metadata dtype equality instead of VeOmni's normal load-time cast into the target parameter or buffer dtype.
