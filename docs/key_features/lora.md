# LoRA Fine-Tuning

VeOmni supports [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) as a first-class
feature of `BaseTrainer`. LoRA injects trainable low-rank matrices into selected linear layers
while freezing the rest of the base model, enabling parameter-efficient fine-tuning with
significantly reduced GPU memory.

---

## Installation

`peft` ships under both `gpu` and `npu` extras — a standard `uv sync --extra
gpu --group dev` (or `--extra npu`) is sufficient. For pip users:

```shell
pip install peft==0.18.1
```

---

## 1. LoRA Config Definition

LoRA is configured through the `model.lora_config` field in your YAML:

```yaml
model:
  lora_config:
    rank: 64          # LoRA rank r
    alpha: 32         # Scaling factor α (effective scale = α / r)
    lora_modules:     # Target linear layer names (module name substrings)
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
```

| Field | Type | Description |
|---|---|---|
| `rank` | int | Rank of the decomposition matrices A and B |
| `alpha` | int | LoRA scaling factor; effective scale = `alpha / rank` |
| `lora_modules` | list[str] | Target linear layer name substrings (matched against module FQNs) |
| `lora_adapter` | str (optional) | Path to a saved adapter directory to resume from |

For **resume training**, add the `lora_adapter` key pointing to the saved adapter directory:

```yaml
model:
  lora_config:
    rank: 64
    alpha: 32
    lora_modules: [q_proj, k_proj, v_proj, o_proj]
    lora_adapter: ./exp/my_run/global_step_500   # HF adapter dir to resume from
```

---

## 2. LoRA Initialization in BaseTrainer

LoRA wrapping happens in `BaseTrainer._setup_lora()`, called from `_freeze_model_module()`:

```python
# veomni/trainer/base.py

def _setup_lora(self):
    lora_config = self.args.model.lora_config
    if not bool(lora_config):
        return

    lora_adapter_path = lora_config.get("lora_adapter", None)

    if lora_adapter_path is not None:
        # Resume: read PEFT config from disk; weights loaded later during parallelization
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.model, lora_adapter_path, is_trainable=True
        )
    else:
        # From scratch: wrap with LoraConfig
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(
            r=lora_config["rank"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["lora_modules"],
        )
        self.model = get_peft_model(self.model, peft_cfg)
```

After wrapping, `BaseTrainer._init_callbacks()` automatically selects `HFLoraCkptCallback`
instead of `HuggingfaceCkptCallback` when `lora_config` is set:

```python
if self.args.model.lora_config:
    self.hf_ckpt_callback = HFLoraCkptCallback(self)
else:
    self.hf_ckpt_callback = HuggingfaceCkptCallback(self)
```

---

## 3. Weight Loading with LoRA

VeOmni LoRA training uses FSDP2 with `init_device: meta`. Weight loading goes through
`build_parallelize_model` and then `post_process_after_weight_loading` in
`torch_parallelize.py`. The LoRA-specific path:

1. **Base-model weights**: loaded via `rank0_load_and_broadcast_weights` or
   `load_model_weights` — the standard FSDP2 path, unchanged for LoRA.

2. **Adapter weights** (resume only): `_build_parallelized_model` passes `adapter_path`
   to `build_parallelize_model`, which calls `load_lora_model_weights` (all-ranks read)
   or `rank0_load_and_broadcast_adapter_weights` (rank-0 reads then broadcasts).
   Both functions remap PEFT keys to model FQNs before dispatching into DTensors.

3. **Adapter weight initialisation from scratch**: `post_process_after_weight_loading`
   calls `_init_lora_parameter` for any LoRA parameter not yet filled, invoking
   `reset_lora_parameters` to apply kaiming/zero init.

**Key difference from base model loading:** PEFT saves adapter keys without the adapter-name
infix (e.g. `lora_A.weight`), whereas the live model stores them as
`lora_A.<adapter_name>.weight`. The `_remap_adapter_key` utility handles this translation.

---

## 4. Checkpoint Saving

### DCP checkpoint (training state)

`CheckpointerCallback._save_checkpoint` saves the full distributed state (model + optimizer +
extra state) via PyTorch DCP. For LoRA training this includes both base-model parameters
**and** adapter parameters; the optimizer state only covers the trainable adapter parameters.

### HF LoRA adapter (inference artifact)

`HFLoraCkptCallback._save_checkpoint` calls `save_lora_adapter_with_dcp`
(`veomni/utils/save_safetensor_utils.py`), which:

1. Extracts adapter-only tensors via `get_peft_model_state_dict`.
2. Saves them with `dcp.save` in parallel to a temporary DCP directory.
3. Consolidates on rank 0 into `adapter_model.bin` and `adapter_config.json`.
4. Removes the temporary DCP directory.

Output structure for each checkpoint:

```
<output_dir>/
├── checkpoints/
│   └── global_step_N/          ← DCP checkpoint (resume training)
│       ├── __0_0.distcp
│       └── .metadata
└── global_step_N/              ← HF adapter (inference / resume)
    ├── adapter_config.json
    └── adapter_model.bin
```

---

## 5. Training Examples

### 5.1 Wan2.1-I2V-1.3B LoRA (DiT, FSDP2)

Config: [`configs/dit/wan2.1_I2V_1.3B_lora.yaml`](../../configs/dit/wan2.1_I2V_1.3B_lora.yaml)

```yaml
model:
  lora_config:
    rank: 128
    alpha: 64
    lora_modules:
      - to_q
      - to_k
      - to_v
      - to_out.0
      - ffn.net.0.proj
      - ffn.net.2

train:
  init_device: meta
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
```

Launch (8 GPUs, SP=2):

```shell
SP_SIZE=2
NPROC_PER_NODE=8

bash train.sh tasks/train_dit.py configs/dit/wan2.1_I2V_1.3B_lora.yaml \
    --model.model_path           ./Wan2.1-T2V-1.3B-Diffusers/transformer \
    --model.condition_model_path ./Wan2.1-T2V-1.3B-Diffusers \
    --data.train_path            ./my_dataset_offline \
    --data.source_name           my_dataset \
    --train.training_task        offline_training \
    --train.global_batch_size    8 \
    --train.micro_batch_size     1 \
    --train.accelerator.ulysses_size ${SP_SIZE} \
    --train.checkpoint.output_dir ./exp/wan_lora \
    --train.checkpoint.save_hf_weights true \
    --train.checkpoint.save_epochs 5 \
    --train.checkpoint.load_path auto \
    --train.num_train_epochs 30
```

See [Wan2.1-I2V-1.3B Training Guide](../examples/wan2.1_I2V_1.3B.md) for the complete
dataset preparation and inference workflow.

### 5.2 Qwen3-0.6B LoRA (LLM, FSDP2)

Config: [`configs/text/qwen3_lora.yaml`](../../configs/text/qwen3_lora.yaml)

```yaml
model:
  model_path: Qwen3-0.6B-Base
  ops_implementation:
    attn_implementation: flash_attention_2
    cross_entropy_loss_implementation: eager
    rms_norm_implementation: eager
    swiglu_mlp_implementation: eager
    rotary_pos_emb_implementation: eager
  lora_config:
    rank: 64
    alpha: 32
    lora_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj

train:
  init_device: meta          # required for FSDP2
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
```

Launch (8 GPUs, SP=2):

```shell
SP_SIZE=2
NPROC_PER_NODE=8

bash train.sh tasks/train_text.py configs/text/qwen3_lora.yaml \
    --model.model_path /path/to/Qwen3-0.6B-Base \
    --data.train_path  /path/to/tulu-3-sft-mixture/data \
    --train.global_batch_size 8 \
    --train.micro_batch_size  1 \
    --train.num_train_epochs  3 \
    --train.checkpoint.output_dir ./exp/qwen3_lora \
    --train.checkpoint.save_hf_weights true \
    --train.checkpoint.load_path auto \
    --train.wandb.enable true
```

To resume from a saved adapter:

```shell
bash train.sh tasks/train_text.py configs/text/qwen3_lora.yaml \
    --model.model_path /path/to/Qwen3-0.6B-Base \
    --data.train_path  /path/to/tulu-3-sft-mixture/data \
    --train.checkpoint.output_dir ./exp/qwen3_lora \
    --train.checkpoint.load_path auto          # auto-picks latest DCP checkpoint
```

### 5.3 Qwen-Image LoRA (DiT, FSDP2)

Config: [`configs/dit/qwen_image_lora.yaml`](../../configs/dit/qwen_image_lora.yaml)

The Qwen-Image transformer uses a **dual-stream** MMDiT block: each `QwenImageTransformerBlock` carries a separate image-stream attention projection (`to_q`, `to_k`, `to_v`, `to_out.0`) and a text-stream one (`add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`), plus paired `img_mlp` / `txt_mlp` FeedForward sub-modules. The recommended target set covers both streams:

```yaml
model:
  lora_config:
    rank: 128
    alpha: 64
    lora_modules:
      - to_q
      - to_k
      - to_v
      - to_out.0
      - add_q_proj
      - add_k_proj
      - add_v_proj
      - to_add_out
      - net.0.proj
      - net.2

train:
  init_device: meta
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
```

`net.0.proj` and `net.2` are substring-matched, so they hit the Linear layers inside both `img_mlp` and `txt_mlp` (each is a `diffusers.models.attention.FeedForward`).

Launch (e.g. 8 GPUs, single-node):

```shell
NPROC_PER_NODE=8

bash train.sh tasks/train_dit.py configs/dit/qwen_image_lora.yaml \
    --model.model_path           /path/to/Qwen-Image/transformer \
    --model.condition_model_path /path/to/Qwen-Image \
    --data.train_path            /path/to/dataset \
    --train.global_batch_size    8 \
    --train.micro_batch_size     1 \
    --train.checkpoint.output_dir ./exp/qwen_image_lora \
    --train.checkpoint.save_hf_weights true \
    --train.num_train_epochs 3
```

`HFLoraCkptCallback` writes the trained adapter to `${output_dir}/global_step_${step}/{adapter_config.json, adapter_model.{bin,safetensors}}`, which is the standard PEFT format consumable by `PeftModel.from_pretrained` and `diffusers`' `pipeline.transformer.load_lora_adapter` (the adapter keys carry the `base_model.model.` prefix expected by `peft`).

---

## 6. Testing

The test suite is under `tests/lora/` and verifies save/load correctness using a
two-layer toy Qwen3 model:

```shell
torchrun --nproc_per_node=4 tests/lora/test_lora_trainer_saveload.py \
    tests/lora/qwen3_toy_lora.yaml
```

**What the test verifies:**
1. Train 3 steps with LoRA on a dummy dataset (FSDP2, meta device).
2. After step 1: snapshot LoRA weights and save DCP checkpoint.
3. Continue training (steps 2–3 mutate adapter weights).
4. Reload the step-1 checkpoint; assert every LoRA tensor is bit-identical to the snapshot.
