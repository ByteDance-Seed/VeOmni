# Quick Start

Run five supervised fine-tuning steps on Qwen3-0.6B, then locate the training
checkpoint and exported model. This is a pipeline smoke test using a small
synthetic conversation dataset; it is not a model-quality or throughput benchmark.

## Prerequisites

- A Linux host with two NVIDIA GPUs supported by the [GPU environment](installation/install.md).
- Python 3.11 or 3.12, the locked GPU dependencies, and network access to
  Hugging Face for the model download.
- Space for the environment, model snapshot, optimizer checkpoint, and export.
  Keep these on local disk for the first run.

The command uses two GPUs with FSDP2 and short sequences. This keeps the recipe
on the sharded loading path required by its `init_device: meta` setting; do not
reduce it to one process without adapting model initialization. Memory usage depends on the
accelerator and kernel versions; the model's parameter count alone is not a
memory requirement. For Ascend, ROCm, or MLU, install the platform environment
and follow the [hardware-specific guidance](../hardware_support/index.md).

## 1. Install and check the environment

Follow [NVIDIA installation](installation/install.md), then run everything below
from the repository root with the environment activated:

```bash
source .venv/bin/activate
python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
assert torch.cuda.device_count() >= 2, "Two CUDA GPUs are required for this example"
assert transformers.__version__ == "5.16.1"
print("GPU:", torch.cuda.get_device_name(0))
PY
```

## 2. Download the model and prepare data

Download the model into a known local directory:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3-0.6B", local_dir="downloads/Qwen3-0.6B")
PY
```

Create 64 conversation records. The `messages` column matches
`data.text_keys` in the existing Qwen3 configuration:

```bash
python - <<'PY'
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

Path("downloads").mkdir(exist_ok=True)
rows = [
    {"messages": [
        {"role": "user", "content": "What is 2 plus 2?"},
        {"role": "assistant", "content": "2 plus 2 is 4."},
    ]},
    {"messages": [
        {"role": "user", "content": "Name a primary color."},
        {"role": "assistant", "content": "Red is a primary color."},
    ]},
] * 32
pq.write_table(pa.Table.from_pylist(rows), "downloads/quick-start.parquet")
print("Wrote", len(rows), "records")
PY
```

## 3. Launch five training steps

Reuse the [Qwen3 configuration](../../configs/text/qwen3.yaml) and override only
the inputs and smoke-test settings. Fixed-size batches make the number of
examples per step explicit: a global batch of two and one micro-batch example
per GPU means one micro-step on each of the two GPUs. Dynamic packing can be
enabled later.

Use a fresh output directory when repeating the example. `train.sh` writes
`log.txt` in the current directory and replaces it on the next invocation.

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
    --model.model_path downloads/Qwen3-0.6B \
    --data.train_path downloads/quick-start.parquet \
    --data.max_seq_len 512 \
    --data.dataloader.num_workers 0 \
    --train.dyn_bsz false \
    --train.global_batch_size 2 \
    --train.micro_batch_size 1 \
    --train.bsz_warmup_ratio 0 \
    --train.num_train_epochs 1 \
    --train.max_steps 5 \
    --train.wandb.enable false \
    --train.checkpoint.output_dir outputs/quick-start \
    --train.checkpoint.save_steps 5 \
    --train.checkpoint.save_epochs 0 \
    --train.checkpoint.save_hf_weights true
```

## 4. Check the result

The process should exit successfully after five optimizer steps. Inspect the
training loss and gradient-norm logs for finite values; five steps on repeated
synthetic records do not establish convergence.

The completed run writes:

| Location | Purpose |
| --- | --- |
| `log.txt` | Launcher and training output |
| `outputs/quick-start/model_assets/` | Model configuration and tokenizer assets |
| `outputs/quick-start/checkpoints/global_step_5/` | Training state, including the completion manifest |
| `outputs/quick-start/checkpoints/global_step_5/hf_ckpt/` | Hugging Face model export |

Check that `checkpoint_manifest.json` exists in the step directory and that
`model/ckpt/` and `model/optimizer/` contain DCP metadata. Use the step directory
for `train.checkpoint.load_path` when resuming a longer run; the HF export is
an inference artifact. See [checkpoint layout](../usage/checkpoint.md) for the
full completion and resume contract.

If startup fails, first check the selected GPU, environment versions, and input
paths. For an out-of-memory error, reduce the sequence length or use the
[LoRA guide](../key_features/lora.md). Do not treat a successful model download
or a checkpoint directory without its completion metadata as a completed run.

## Validation scope

This procedure completed five steps on two NVIDIA H20 GPUs with Python 3.11,
`torch==2.11.0+cu130`, and `transformers==5.16.1`, using runtime code from
`77cf73e69`. Both DCP metadata files, per-rank trainer/loader state, the step-5
manifest, and the HF safetensors export were checked. This establishes a smoke
run for that environment, not a minimum-memory or convergence claim.

## Next steps

- Replace the synthetic records with a real dataset using the
  [Qwen3 recipe](../models/qwen/qwen3.md).
- Enable [packing and dynamic batching](../usage/data_packing_and_dyn_bsz.md).
- Choose a different modality from [Models and Recipes](../models/index.md).
- Look up overrides in the [arguments reference](../usage/arguments.md).
