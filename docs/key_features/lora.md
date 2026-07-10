# LoRA Fine-Tuning

VeOmni supports [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) as a first-class
feature of `BaseTrainer`. LoRA injects trainable low-rank matrices into selected linear layers
while freezing the rest of the base model, enabling parameter-efficient fine-tuning with
significantly reduced GPU memory.

LoRA is implemented by VeOmni's **own** stack, `veomni.lora` — a PEFT-free
[`peft.PeftModel`](https://huggingface.co/docs/peft) replacement (`VeOmniLoraModel` /
`VeOmniLoraConfig`). It supports PEFT's main features, adds MoE-LoRA and EP-LoRA, and is
FSDP2-native. Crucially it stays **format-compatible** with PEFT: it reads and writes
standard `adapter_config.json` + `adapter_model.{safetensors,bin}` files, so adapters
train-here / load-there interchangeably with stock `peft`.

---

## Installation

No extra dependency is required to *train* LoRA — `veomni.lora` has no `peft` dependency:

```shell
uv sync --extra gpu --dev
```

`peft` is only needed to run the cross-compatibility interop test
(`tests/lora/test_veomni_lora_native.py::test_peft_bidirectional_interop`, which
`pytest.importorskip`s it). It ships inside the hardware extras (`peft==0.18.1` in
`gpu` / `npu` / `npu_aarch64`), so the standard `uv sync --extra gpu --dev` above already
installs it — no separate extra is required.

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
| `use_rslora` | bool (optional, default `false`) | Rank-stabilised LoRA scaling (`alpha / sqrt(r)`) |
| `lora_modules` | list[str] or str | `nn.Linear` LoRA targets. List → PEFT substring/suffix match on module FQNs; a single `str` → regex (full match). Alias: `target_modules`. |
| `exclude_modules` | list[str] or str (optional) | Modules to skip even if matched by `lora_modules`. |
| `target_parameters` | list[str] (optional) | MoE-LoRA target — glob patterns matching the 3-D `nn.Parameter`s on a v5 MoE experts module (e.g. `model.layers.*.mlp.experts.gate_up_proj`). Usually unnecessary: on fused-MoE models, listing `gate_proj` / `up_proj` / `down_proj` in `lora_modules` maps to these automatically. See §5 below. |
| `share_expert_lora` | bool (optional, default `false`) | When `target_parameters` is set: `false` → per-expert independent LoRA; `true` → a single LoRA pair shared across all experts of a layer. See §5. |
| `lora_dropout` | float (optional, default `0.0`) | Dropout applied to the LoRA input. |
| `bias` | str (optional, default `none`) | Which biases stay trainable: `none` / `all` / `lora_only` (PEFT semantics). |
| `rank_pattern` | dict[str,int] (optional) | Per-module rank overrides, `{regex: r}` (first match wins). |
| `alpha_pattern` | dict[str,int] (optional) | Per-module alpha overrides, `{regex: alpha}`. |
| `lora_adapter` | str (optional) | Path to a saved adapter directory to resume from. |
| `is_trainable` | bool (optional, default `true`) | When resuming, set `false` to load adapters frozen for inference only. |

`lora_modules` and `target_parameters` may be set together — the former installs
`LoraLinear` on `nn.Linear` layers (q/k/v/o, MLP, shared_expert, etc.), the latter installs
VeOmni's MoE-LoRA wrappers on the fused experts `nn.Parameter`s. At least one must be
non-empty. On fused-MoE models the expert names `gate_proj` / `up_proj` / `down_proj` in
`lora_modules` are auto-mapped to the corresponding `target_parameters` (see §5), so you
rarely need to write the glob patterns by hand. (Field names mirror `VeOmniLoraConfig`; the
historical `rank`/`alpha`/`lora_modules` YAML names and the PEFT-style
`r`/`lora_alpha`/`target_modules` names are both accepted.)

> `modules_to_save` is parsed but not yet supported by the native stack (it raises a clear
> `NotImplementedError`); correctly seeding the extra trainable modules under FSDP2
> meta-device init is a planned follow-up.

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

LoRA wrapping happens in `BaseTrainer._setup_lora()`, called from `_freeze_model_module()`.
A single native path wraps the model with `VeOmniLoraModel`, handling dense `nn.Linear`
LoRA, MoE expert LoRA, and the two combined:

```python
# veomni/trainer/base.py
def _setup_lora(self):
    lora_config = self.args.model.lora_config
    if not bool(lora_config):
        return

    from ..lora import VeOmniLoraConfig, VeOmniLoraModel

    cfg = VeOmniLoraConfig.from_yaml(lora_config)
    lora_adapter_path = lora_config.get("lora_adapter", None)
    if lora_adapter_path is not None:
        # Resume: rebuild dense + MoE wrappers from the on-disk adapter_config.json
        # (MoE mode lives in its `veomni_lora` block). Weights load later.
        self.model = VeOmniLoraModel.from_pretrained(
            self.model,
            lora_adapter_path,
            is_trainable=lora_config.get("is_trainable", True),
        )
    else:
        self.model = VeOmniLoraModel(self.model, cfg)
```

`VeOmniLoraModel` mirrors `peft.PeftModel` structurally: `model.base_model.model` is the
original model, so every LoRA parameter FQN and every saved adapter key carries the
`base_model.model.` prefix — byte-identical to a PEFT checkpoint. Attribute access and
`forward` are forwarded to the wrapped model, so the trainer, loss computation, and
`generate` are unchanged. After wrapping, the base model is fully frozen and only the LoRA
parameters (dense `LoraLinear` and MoE-LoRA, if any) have `requires_grad=True`.

`BaseTrainer._init_callbacks()` automatically selects `HFLoraCkptCallback`
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
   to `build_parallelize_model`, which — for a `VeOmniLoraModel` — calls the native
   `veomni.lora.weight_loading.load_lora_weights` (all-ranks read) or
   `rank0_load_and_broadcast_lora_weights` (rank-0 reads then broadcasts). Both read the
   PEFT-format adapter file natively (safetensors / torch, **no `peft` import**) and remap
   on-disk keys to model FQNs before dispatching into DTensors. EP-sharded LoRA tensors are
   sliced from the disk-side `[E, ...]` shape to the local `[E_local, ...]` via the runtime
   parallel plan, exactly as on the base-weight path.

3. **Adapter weight initialisation from scratch**: `post_process_after_weight_loading`
   calls `_init_lora_parameter` for any LoRA parameter not yet filled, invoking
   `reset_lora_parameters` (kaiming `A` / zero `B`). For MoE wrappers the reset only fires
   when every wrapper parameter is still on meta device, so a partially-loaded wrapper is
   never clobbered.

**Key difference from base model loading:** the on-disk adapter keys omit the adapter-name
infix (PEFT convention — e.g. `lora_A.weight`), whereas the live model stores them as
`lora_A.<adapter_name>.weight`. `veomni.lora.state_dict.insert_adapter_name` /
`strip_adapter_name` handle the translation in both directions.

---

## 4. Checkpoint Saving

### DCP checkpoint (training state)

`CheckpointerCallback._save_checkpoint` saves the full distributed state (model + optimizer +
extra state) via PyTorch DCP. For LoRA training this includes both base-model parameters
**and** adapter parameters; the optimizer state only covers the trainable adapter parameters.

### HF LoRA adapter (inference artifact)

`HFLoraCkptCallback._save_checkpoint` calls `save_lora_adapter_with_dcp`
(`veomni/utils/save_safetensor_utils.py`), which:

1. Extracts adapter-only tensors via `veomni.lora.state_dict.get_lora_state_dict`
   (PEFT on-disk key format).
2. Restores the EP shard dim on `Shard()`-placed LoRA tensors so DCP gathers the full
   `[E, ...]` shape, then saves with `dcp.save` in parallel to a temporary DCP directory.
3. Consolidates on rank 0 into `adapter_model.bin` and writes `adapter_config.json` via
   `VeOmniLoraModel.get_lora_config().save_pretrained` — which embeds any MoE metadata
   (`target_parameters` + `veomni_lora.moe_mode`) directly in the config. **No separate
   sidecar.**
4. Removes the temporary DCP directory.

Output structure for each checkpoint:

```
<output_dir>/
├── checkpoints/
│   └── global_step_N/          ← DCP checkpoint (resume training)
│       ├── __0_0.distcp
│       └── .metadata
└── global_step_N/              ← HF adapter (inference / resume)
    ├── adapter_config.json     ← PEFT-format; MoE mode in its `veomni_lora` block
    └── adapter_model.bin
```

---

## 5. MoE-LoRA

For MoE models on transformers v5, expert weights live as fused 3-D `nn.Parameter`s on
the experts module — not `nn.Linear`:

```text
experts.gate_up_proj  : nn.Parameter[E, 2*I, H]   # gate / up concatenated along dim-1
experts.down_proj     : nn.Parameter[E, H,   I]
```

PEFT's `target_modules` only matches `nn.Module` subclasses (typically `nn.Linear`) and
therefore cannot reach these parameters. VeOmni's MoE-LoRA wrappers
(`veomni/lora/moe_layers.py`) target them via `target_parameters`, in two flavours:

| Mode | Wrapper class | Per-expert LoRA shape | When to use |
|---|---|---|---|
| **Independent** (default) | `LoraIndependentExperts` | `[E, r, H]` / `[E, O, r]` | Each expert gets its own LoRA pair; functional drop-in for PEFT 0.19's `target_parameters` 3-D path. |
| **Shared** | `LoraSharedExperts` | `[r, H]` / `[O, r]` | A single LoRA pair shared across all experts of a layer; ~`E×` fewer LoRA parameters. PEFT does not support this natively. |

### Configuration

**Recommended — semantic module names.** For the supported fused-MoE models
(see §5.2) just list the expert MLP module names `gate_proj` / `up_proj` /
`down_proj` in `lora_modules`, exactly as you would for a dense model. VeOmni
maps them onto the model's fused expert parameters automatically
(`gate_proj` / `up_proj` → `experts.gate_up_proj`, `down_proj` →
`experts.down_proj`):

```yaml
model:
  lora_config:
    rank: 16
    alpha: 32
    lora_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    share_expert_lora: false                           # false = Independent, true = Shared
```

The mapping is driven by a per-model `_convert_lora_targets_to_parameters` hook
(registered in the model's `__init__.py`) plus
`veomni.lora.resolve_fused_moe_lora_targets`, invoked by
`BaseTrainer._setup_lora` before the adapter is built. It is a **no-op on dense
models and on models without the hook**, so `gate_proj` / `up_proj` /
`down_proj` there stay ordinary `nn.Linear` LoRA targets.

**Advanced — explicit `target_parameters`.** You can still point directly at the
fused expert parameters with glob patterns (mixed freely with `lora_modules`).
This is equivalent to the recommended form above and is what gets serialized
into `adapter_config.json`:

```yaml
model:
  lora_config:
    rank: 16
    alpha: 32
    lora_modules: [q_proj, k_proj, v_proj, o_proj]    # linear LoRA
    target_parameters:                                 # VeOmni MoE-LoRA
      - model.layers.*.mlp.experts.gate_up_proj
      - model.layers.*.mlp.experts.down_proj
    share_expert_lora: false                           # false = Independent, true = Shared
```

For VLMs the FQN includes the wrapping `language_model.` prefix — see the
multimodal example configs listed below.

### 5.1 Two LoRAs on the fused `gate_up_proj`

The wrapper does **not** allocate one rank-`r` LoRA over the fused `[E, 2I, H]` parameter.
Because `SiLU(gate) * up` is non-linear, LoRA must be added **before activation**, not on
the merged `gate_up`. The wrapper therefore allocates **two independent rank-`r` LoRAs** —
one over the gate half, one over the up half:

| Logical spec | Shared mode (2-D) | Independent mode (3-D) |
|---|---|---|
| `gate_proj` | `A:[r,H]` / `B:[I,r]` | `A:[E,r,H]` / `B:[E,I,r]` |
| `up_proj`   | `A:[r,H]` / `B:[I,r]` | `A:[E,r,H]` / `B:[E,I,r]` |
| `down_proj` | `A:[r,I]` / `B:[H,r]` | `A:[E,r,I]` / `B:[E,H,r]` |

This costs `2r(H+I)` extra parameters vs `r(H+2I)` for a single LoRA, but doubles the
effective joint output rank from `r` to `2r`. Users only write `gate_up_proj` in
`target_parameters`; the wrapper auto-expands to the three logical specs.

### 5.2 Supported models

VeOmni MoE-LoRA wrappers require the **v5 fused experts layout** (`gate_up_proj` +
`down_proj` as 3-D `nn.Parameter`s) and will raise on any other layout.

| Model | Runtime expert layout | Notes |
|---|---|---|
| `qwen3_moe`        | fused `gate_up_proj` | Reference; per-expert disk → fused via on-load converter. |
| `qwen3_5_moe`      | fused `gate_up_proj` | Disk already v5 layout. |
| `qwen3_vl_moe`     | fused `gate_up_proj` | VLM; FQN includes `model.language_model.`. |
| `qwen3_omni_moe`   | fused `gate_up_proj` | Thinker tower. |
| `deepseek_v3` (v5) | fused `gate_up_proj` | Per-expert disk → fused via on-load converter. |

DeepSeek-V3 also exposes **shared experts** (`mlp.shared_experts.{gate,up,down}_proj`)
implemented as plain `nn.Linear`. Add them to `lora_modules` if you want PEFT to LoRA them
too — they are orthogonal to the MoE-LoRA on the routed experts.

### 5.3 Forward path: fused triton vs eager

Each MoE-LoRA wrapper dispatches based on `model.ops_implementation.moe_implementation`:

| `moe_implementation` | non-EP | EP | Notes |
|---|---|---|---|
| `fused_triton` | fused triton kernel | fused triton kernel | Recommended for training. |
| `eager` | eager loop (reference) | not supported (raises) | Reference / fallback for NPU and Quack. |

The fused path lives in `veomni/lora/ops/moe_group_gemm.py` and reuses the same
`group_gemm_same_nk` / `group_gemm_same_mn` primitives (from `veomni/ops/kernels/moe/_kernels/`)
as the non-LoRA MoE forward, so it inherits the same EP `all-to-all` dispatch pipeline. The
LoRA fused pointers are bound by `veomni.ops.kernels.moe.apply_veomni_fused_moe_patch` via
`veomni.lora.ops.bind_lora_moe_kernels`.

### 5.4 Expert Parallelism (EP)

When `train.accelerator.ep_size > 1`, base experts are sharded along the expert dim by
`ParallelPlan` (`Shard(0)` on `gate_up_proj` / `down_proj`). MoE-LoRA tracks this layout:

- **Independent mode**: LoRA tensors are 3-D `[E, ...]` and are EP-sharded along the
  expert dim, mirroring base experts.
- **Shared mode**: LoRA tensors are 2-D and remain **replicated** across the EP group
  (the "shared" semantics is one LoRA per layer, not per expert). The wrapper installs a
  post-accumulate-grad hook that runs `all_reduce(SUM)` on the shared LoRA grads across
  the EP group, restoring the EP=1 + DP=N equivalence. The grad-norm path
  (`extra_parallel_fsdp2_clip_grad_norm`) is also taught — via
  `_collect_ep_replicated_lora_param_ids` in `veomni/optim/optimizer.py` — to skip the
  EP all-reduce for these replicated params so the global grad-norm matches EP=1.

Both modes work with FSDP2 + EP. EP is only supported on the `fused_triton` forward path.

### 5.5 Save / load artefacts

A MoE-LoRA run writes only the two standard PEFT artefacts — **no sidecar**:

```
<output_dir>/global_step_N/
├── adapter_config.json     # PEFT-format; MoE mode/rank/alpha in its `veomni_lora` block
└── adapter_model.bin       # PEFT-format — both linear LoRA and MoE-LoRA tensors
```

At resume, `VeOmniLoraModel.from_pretrained` reads `adapter_config.json`: the
`veomni_lora.moe_mode` field decides which wrapper class to install
(`LoraIndependentExperts` vs `LoraSharedExperts`) **before** weights are loaded. If a
stock-PEFT adapter (no `veomni_lora` block) is loaded, the MoE mode is inferred from the
on-disk LoRA tensor shapes (3-D per-expert → independent, 2-D → shared).

The MoE-LoRA tensors in `adapter_model.bin` use PEFT-aligned FQNs
(`base_model.model.<...>.<spec>.lora_A.<adapter>.weight`), so third-party PEFT tooling
(HuggingFace Hub, `peft.PeftModel.from_pretrained`) can also load the file — though without
VeOmni's wrappers re-installed first, the MoE-LoRA half of the keys will be dropped as
`unexpected_keys`. The `veomni_lora` block is an unknown top-level key to `peft` and is
ignored on its side, keeping the file loadable by both stacks.

### 5.6 Ready-to-use configs

Text-only:
- [`configs/text/qwen3_moe_lora.yaml`](../../configs/text/qwen3_moe_lora.yaml) — Qwen3-MoE
- [`configs/text/deepseek_v3_lora.yaml`](../../configs/text/deepseek_v3_lora.yaml) — DeepSeek-V3 (v5), with EP=8

Multimodal:
- [`configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_lora.yaml`](../../configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_lora.yaml)
- [`configs/multimodal/qwen3_vl/qwen3_vl_moe_lora.yaml`](../../configs/multimodal/qwen3_vl/qwen3_vl_moe_lora.yaml)
- [`configs/multimodal/qwen3_omni/qwen3_omni_lora.yaml`](../../configs/multimodal/qwen3_omni/qwen3_omni_lora.yaml)

---

## 6. Training Examples

### 6.1 Wan2.1-I2V-1.3B LoRA (DiT, FSDP2)

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

### 6.2 Qwen3-0.6B LoRA (LLM, FSDP2)

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

### 6.3 Qwen3-MoE LoRA (FSDP2, optional EP)

Config: [`configs/text/qwen3_moe_lora.yaml`](../../configs/text/qwen3_moe_lora.yaml)

```yaml
model:
  model_path: Qwen3-30B-A3B-merge
  ops_implementation:
    attn_implementation: flash_attention_2
    moe_implementation: eager           # EP (ep_size > 1) REQUIRES fused_triton
  lora_config:
    rank: 16
    alpha: 32
    # gate_proj/up_proj/down_proj auto-map to the fused expert parameters.
    lora_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    share_expert_lora: true                            # one LoRA per layer (Mode 2)

train:
  init_device: meta
  accelerator:
    ulysses_size: 1
    ep_size: 1                          # set >1 to enable EP; also set moe_implementation: fused_triton
    fsdp_config:
      fsdp_mode: fsdp2
```

Launch (8 GPUs):

```shell
bash train.sh tasks/train_text.py configs/text/qwen3_moe_lora.yaml \
    --model.model_path /path/to/Qwen3-30B-A3B \
    --data.train_path  /path/to/tulu-3-sft-mixture/data \
    --train.checkpoint.output_dir ./exp/qwen3_moe_lora \
    --train.checkpoint.save_hf_weights true \
    --train.checkpoint.load_path auto
```

Resume from a saved adapter (DCP path is auto-detected via `load_path: auto`; an HF
adapter resume goes through `model.lora_config.lora_adapter`):

```shell
bash train.sh tasks/train_text.py configs/text/qwen3_moe_lora.yaml \
    --model.model_path                 /path/to/Qwen3-30B-A3B \
    --model.lora_config.lora_adapter   ./exp/qwen3_moe_lora/global_step_500
```

DeepSeek-V3 (v5) follows the same shape — see
[`configs/text/deepseek_v3_lora.yaml`](../../configs/text/deepseek_v3_lora.yaml), which
sets `ep_size: 8` and additionally LoRA-wraps DeepSeek's MLA projections
(`q_a_proj` / `q_b_proj` / `kv_a_proj_with_mqa` / `kv_b_proj` / `o_proj`) via
`lora_modules`.

---

### 6.4 Qwen-Image LoRA (DiT, FSDP2)

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

## 7. Testing

The test suite is under `tests/lora/` and uses small MoE toy models.
Suite layout:

| Test | Coverage |
|------|----------|
| `test_veomni_lora_native.py` | CPU, single-process: `VeOmniLoraConfig` round-trips, dense injection structure / no-op-at-init, `get_lora_state_dict` PEFT key format, save→`from_pretrained`→load parity, `merge_and_unload`, rank/alpha patterns, exclude_modules, rslora scaling, and **bidirectional PEFT interop** (gated on `peft`). |
| `test_veomni_lora_moe_native.py` | CPU, single-process: native MoE injection (independent/shared), MoE metadata embedded in `adapter_config.json` (no sidecar), state-dict key format/rank, save→reload param round-trip, and MoE-mode inference when the `veomni_lora` block is absent. |
| `test_moe_lora_eager.py` | Wrapper layout + autograd parity for both Mode 1 (independent) and Mode 2 (shared); covers all v5 MoE configs (qwen3_moe / qwen3_5_moe / qwen3_vl_moe / qwen3_omni_moe / deepseek_v3). |
| `test_moe_lora_fused.py` | Triton fused MoE-LoRA kernel parity vs eager (forward + backward) and EP autograd-class parity vs non-EP under controlled inputs. |
| `test_moe_lora_trainer.py` | Production save/load/resume round-trip: writer (DCP shard + HF adapter) → DCP-resume subprocess → adapter-resume subprocess; bit-exact LoRA reload assertion. The yaml enables both `lora_modules` (linear LoRA on `q_proj`/`v_proj`) and `target_parameters` (MoE-LoRA wrappers), so both LoRA flavors round-trip end-to-end. |
| `test_moe_lora_ep2.py` | Trainer-driven EP=2 coverage: (a) integration assertions that the EP plumbing engages (plan-bridge fires, EP slicing happens at the right ratio, DCP consolidates EP shards before HF save); (b) `test_moe_lora_ep_save_load_parallel_align` -- one EP=2 seeder writes both an HF adapter (full `[E, r, H]`) and DCP shards, then three resumer subprocesses validate cross-EP adapter parity (EP=1 vs EP=2 adapter-load trajectories match) and EP=2 DCP round-trip parity (DCP-resumer trajectory matches the seeder's tail). |

Quick run (all four; ~10 min total on 4 GPUs):

```shell
pytest -s tests/lora/
```

Run just the production-shaped save/load/resume round-trip:

```shell
pytest -s -v tests/lora/test_moe_lora_trainer.py::test_save_load_resume_round_trip
```

Or as a manual torchrun against either `independent` or `shared`:

```shell
torchrun --nproc_per_node=4 tests/lora/test_moe_lora_trainer.py \
    tests/lora/qwen3_moe_toy_lora_independent.yaml \
    --train.checkpoint.output_dir /tmp/test_moe_lora_run
```

**What the round-trip test verifies:**
1. Writer trains `max_steps=4`, writes DCP shards at step 2 / step 4 and the HF LoRA adapter (`adapter_config.json` with the `veomni_lora` block + `adapter_model.bin`, no sidecar) at the same cadence; snapshots the full LoRA tensors at `on_train_begin` and `on_train_end`.
2. DCP-resume subprocess loads the step-2 DCP shard, continues to step 4, and the resulting LoRA tensors must be **bit-exact** vs the writer's `on_train_end` snapshot (DCP ships model + optimizer + RNG + dataloader state).
3. LoRA-adapter-resume subprocess loads the step-4 HF adapter via `model.lora_config.lora_adapter`, and the resulting `on_train_begin` snapshot (= post-load, pre-train) must be bit-exact vs the writer's `on_train_end` snapshot in bf16 (the on-disk adapter dtype).
