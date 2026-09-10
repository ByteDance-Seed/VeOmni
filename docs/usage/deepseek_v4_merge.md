# Export DeepSeek V4 checkpoints

Convert a full VeOmni DCP checkpoint to DeepSeek's native inference safetensors:

```bash
python scripts/deepseek_v4/merge_dcp_to_deepseek.py \
    --load-dir /path/to/checkpoints/global_step_2585 \
    --save-dir /path/to/checkpoints/global_step_2585/hf_ckpt \
    --format v4-flash \
    --ep-size 4 \
    --workers 8
```

Use `--format v4-flash-base` for the base release's format. Both options require
an unquantized training checkpoint with the official Flash architecture (43 layers,
256 routed experts). The format selects serialization and release assets; it does
not turn base-model weights into instruction-tuned weights or vice versa.

| Format | Routed experts | Attention / shared experts | Quantization scales |
| --- | --- | --- | --- |
| `v4-flash` | Packed E2M1 FP4, block size 32 | E4M3 FP8, 128 × 128 blocks | UE8M0 |
| `v4-flash-base` | E4M3 FP8, 128 × 128 blocks | E4M3 FP8, 128 × 128 blocks | FP32 (power-of-two values) |

Unquantized tensors retain the release's dtype: BF16 weights, FP32 mHC parameters
and attention sinks, and integer hash-router tables. Output keys, shapes, dtypes,
and shard assignment are validated against bundled official release schemas.
`--reference-dir` is not needed or accepted. No original model weights are downloaded.

Tokenizer files (and Flash's generation config) are fetched from the pinned
Hugging Face release using the standard HF cache. Pre-cache these files for offline
use with `HF_HUB_OFFLINE=1`. `--skip-assets` skips downloads and config/tokenizer
writing, but still writes the complete weight index. Such output needs matching
assets before use by an inference engine.

VeOmni does not train MTP: the exporter omits MTP, including any MTP entries in the
source, and writes `num_nextn_predict_layers: 0`. It never fills missing tensors
from another checkpoint. Missing non-MTP tensors or incompatible shapes fail validation.

For fine-tuned models, pass the training-time template with
`--custom-template /path/to/chat_template.jinja` (inline Jinja is also accepted).
It replaces any release template and is saved as `chat_template.jinja`.
Base assets have no chat template by default.

`--ep-size` is specifically for checkpoints saved with `muon_expert_zero_comm`,
where both EP and FSDP shard the expert dimension. The script derives the FSDP
factor from DCP metadata and restores expert order. Leave this option at its default
of 1 for the usual FSDP `Shard(1)` expert layout, even when training used EP.

Run with the GPU extra installed. Each worker holds one layer in host memory;
workers rotate across visible CUDA devices. Ensure sufficient host and GPU memory
before increasing `--workers`. No distributed launch is required.

Complete output shards are skipped on reruns after checking their keys, shapes,
dtypes, and file length. Use a separate output directory for each input checkpoint;
`--overwrite` rewrites all selected shards. Shards and the final index are published
with atomic renames. `--layers 0,2-3` produces only selected layers for smoke tests,
without assets or a complete model index; use a separate directory for smoke tests.

## Schema provenance

The JSON files under `scripts/deepseek_v4/formats/` contain config and safetensors
header metadata, with repeated layers/expert entries represented as templates.
They include no model weights. Schemas were checked against every non-MTP shard
and the official weight index at these immutable revisions:

- [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1)
- [DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/tree/8855555deef230a27a21a8d6f294b7b7497759b6)

The upstream MIT license is included in `scripts/deepseek_v4/formats/LICENSE`.
To refresh a preset, read the new release's config, weight index and safetensors
headers, remove MTP entries, deduplicate identical layer/expert schemas, then verify
the expanded key/dtype/shape/shard map against every source header before updating
the pinned revision. Do not change the revision independently of the schema.
