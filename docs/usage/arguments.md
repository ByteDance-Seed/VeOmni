# Arguments API Reference

Training arguments use nested dataclasses defined in `veomni.arguments.arguments_types`.
The root config `VeOmniArguments` assembles three top-level groups — **model**, **data**, and **train** —
each of which contains further nested sub-configs.

Example YAML structure:

```yaml
train:
  wandb:
    enable: true
    project: VeOmni
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp1
      full_shard: true
  checkpoint:
    manager: dcp
```

---

## Configuration

Top-level configuration that assembles all argument groups.

* `VeOmniArguments` — Root config: `model` + `data` + `train`
* `VeOmniVLMArguments` — VLM extension of `VeOmniArguments`

---

## Model

Model architecture, paths, and multimodal encoder / decoder setup.

* `ModelArguments` — `model.*`
* `OpsImplementationConfig` — `model.ops_implementation.*`

### VLM Extensions

* `VLMMModelArguments` — extends `ModelArguments` with encoder data-balancing options

---

## Data

Dataset paths, tokenization, and batching configuration.

* `DataArguments` — `data.*`
* `DataloaderConfig` — `data.dataloader.*`

### VLM Extensions

* `VLMMDataArguments` — extends `DataArguments` with multimodal configs (`mm_configs`)

---

## Training

Training loop, optimizer, parallelism, checkpointing, profiling, and logging.

* `TrainingArguments` — `train.*`
    * `OptimizerConfig` — `train.optimizer.*`
    * `WandbConfig` — `train.wandb.*`
    * `ProfileConfig` — `train.profile.*`
    * `GradientCheckpointingConfig` — `train.gradient_checkpointing.*`
    * `AcceleratorConfig` — `train.accelerator.*`
        * `FSDPConfig` — `train.accelerator.fsdp_config.*`
          * `MixedPrecisionConfig` — `train.accelerator.fsdp_config.mixed_precision`
        * `OffloadConfig` — `train.accelerator.offload_config.*`
    * `CheckpointConfig` — `train.checkpoint.*`

### VLM Extensions

* `VLMTrainingArguments` — extends `TrainingArguments` with ViT / audio freeze & learning-rate options

---

## DPO

DPO-specific hyperparameters, accessed via `dpo_config.*`.  
Root config: `VeOmniDPOArguments` (extends `VeOmniArguments`).

* `DPOConfig` — `dpo_config.*`

---

## Inference

Standalone inference configuration.

* `InferArguments`

---

## Detailed Reference

### VeOmniArguments

Root config — assembles `model`, `data`, and `train`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| model | `ModelArguments` | — | Model configuration |
| data | `DataArguments` | — | Data configuration |
| train | `TrainingArguments` | — | Training configuration |

### ModelArguments

`model.*` — Model architecture, paths, and multimodal encoder / decoder setup.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| config_path | `Optional[str]` | `None` | Path to the model HuggingFace config (e.g. `config.json`). Defaults to `model_path`. |
| model_path | `Optional[str]` | `None` | Path to the pre-trained model weights. If unset, random init is used. |
| tokenizer_path | `Optional[str]` | `None` | Path to the tokenizer. Defaults to `config_path`. |
| safetensor_idx_path | `Optional[str]` | `None` | Path to `model.safetensors.index.json`. |
| foundation | `Dict[str, str]` | `{}` | Foundation model extra config. |
| encoders | `Dict` | `{}` | Multimodal encoder configs keyed by modality (`image`, `video`, `audio`). |
| decoders | `Dict` | `{}` | Multimodal decoder configs keyed by modality (`image`). |
| input_encoder | `Literal["encoder", "decoder"]` | `"encoder"` | Whether to use the encoder or decoder to encode input images. |
| output_encoder | `Literal["encoder", "decoder"]` | `"decoder"` | Whether to use the encoder or decoder to encode output images. |
| encode_target | `bool` | `False` | Whether to encode training targets with decoder (diffusion only). |
| basic_modules | `Optional[List[str]]` | `[]` | Additional modules beyond `_no_split_modules` to shard in FSDP. |
| ops_implementation | `OpsImplementationConfig` | — | Attention / MoE kernel configuration. |

### OpsImplementationConfig

`model.ops_implementation.*` — Attention, MoE, and fused kernel implementation.

Each `*_implementation` field selects the kernel backend for that operation.
The type is `str` (not `Literal`) so third-party backends can be registered
without modifying the config class.

**Defaults are GPU-optimal** (Liger / Triton / fused_triton). On Ascend NPU
these defaults raise; NPU users must set every field explicitly to an
NPU-supported value (`"npu"`, `"chunk_loss"`, `"fused_npu"`, `"triton"` for
load-balancing loss via `triton-ascend`) or to `"eager"` when the op has no
NPU backend (e.g. `swiglu_mlp_implementation`, DeepSeek-V3 / Qwen2-VL
multimodal RoPE).

NPU validation runs at two times:

- **Config-parse time** (`OpsImplementationConfig.__post_init__`) for the
  six general-purpose ops (`moe`, `cross_entropy_loss`, `rms_norm`,
  `swiglu_mlp`, `rotary_pos_emb`, `load_balancing_loss`). Errors fire
  immediately with a model-agnostic allow-list.
- **Model-build time** for Qwen3.5-only ops (`rms_norm_gated`,
  `causal_conv1d`, `chunk_gated_delta_rule`). Validating these at config
  parse would force every NPU user to override them even when training
  non-Qwen3.5 models. Instead, `qwen3_5_moe`'s NPU registration raises
  upfront if any of the three is set to a non-eager value, and the OpSlot
  bind layer is the safety net behind it. **Qwen3.5 GatedDeltaNet has no
  NPU kernel today** — varlen training (`dyn_bsz=True`, the default) is
  not supported on NPU; non-varlen training works only with all three
  fields pinned to `"eager"`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| attn_implementation | `Optional[Literal[...]]` | `"flash_attention_2"` | Attention implementation to use. |
| moe_implementation | `str` | `"fused_triton"` | MoE experts forward implementation. `fused_triton` uses Triton group-gemm (GPU, SM70+); `fused_quack` uses Quack CUTLASS/CuTe (GPU, SM90+); `fused_npu` uses the NPU group-gemm kernel; `eager` is the reference loop. Mismatches (e.g. `fused_triton` on NPU) raise at config validation time — no silent fallback. |
| cross_entropy_loss_implementation | `str` | `"liger_kernel"` | Cross-entropy loss. `liger_kernel` (default, GPU only) fuses `lm_head` linear + CE; requires VeOmni-patched modeling files that pass `hidden_states=`/`weights=` to `self.loss_function(...)` — unpatched HF models that pass logits will RuntimeError. `chunk_loss` is the hardware-agnostic chunked F.linear+CE (CUDA + NPU). `npu` is a back-compat alias for `chunk_loss`. `eager` is `F.cross_entropy`. |
| rms_norm_implementation | `str` | `"liger_kernel"` | RMSNorm. Known values: `liger_kernel` (default, GPU only), `npu`, `triton` (DeepSeek-V3 only; GPU only), `eager`. |
| swiglu_mlp_implementation | `str` | `"liger_kernel"` | SwiGLU MLP. Known values: `liger_kernel` (default, GPU only), `eager`. There is no NPU backend — NPU users must set this to `"eager"`. |
| rotary_pos_emb_implementation | `str` | `"liger_kernel"` | Rotary pos emb. Known values: `liger_kernel` (default, GPU only), `npu`, `triton` (DeepSeek-V3 only; GPU only), `eager`. |
| load_balancing_loss_implementation | `str` | `"triton"` | MoE load-balancing loss. `triton` (default) requires the `triton` Python package (or `triton-ascend` on NPU); raises at config validation time if the package is missing. `eager` is the pure-PyTorch reference. |
| rms_norm_gated_implementation | `str` | `"fla"` | Gated RMSNorm (Qwen3.5 GatedDeltaNet `self.norm`). Known values: `eager`, `fla` (FLA `FusedRMSNormGated`, requires `flash-linear-attention`, GPU). No NPU backend — selecting any non-eager value on NPU raises at OpSlot bind time. |
| causal_conv1d_implementation | `str` | `"fla"` | Varlen depthwise causal conv1d (Qwen3.5 GatedDeltaNet pre-mixer). Known values: `eager`, `fla` (FLA `causal_conv1d`, requires `flash-linear-attention`, GPU). `eager` raises at forward time for varlen training (no torch fallback handles `cu_seqlens`). No NPU backend. |
| chunk_gated_delta_rule_implementation | `str` | `"fla"` | Chunk gated delta-rule kernel for Qwen3.5 linear attention. Known values: `eager`, `fla` (FLA `chunk_gated_delta_rule`, GPU), `flash_qla` (QwenLM FlashQLA, requires the optional `flash-qla` extra, GPU). `eager` falls back to transformers' `torch_chunk_gated_delta_rule`, which raises at forward time for varlen training. No NPU backend. |

### DataArguments

`data.*` — Dataset paths, tokenization, and batching.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| train_path | `str` | **Required** | Path of the training dataset. Use comma to separate multiple datasets. |
| eval_path | `Optional[str]` | `None` | Path of the evaluation dataset. |
| train_size | `int` | `10_000_000` | Number of tokens for training (used to compute steps under dynamic batch). |
| train_sample | `int` | `10_000` | Number of samples for training (used to compute steps under non-dynamic batch). |
| data_type | `Literal["plaintext", "conversation", "diffusion", "classification"]` | `"conversation"` | Type of the training data. |
| datasets_type | `str` | `"mapping"` | `IterableDataset` or `MappingDataset` (or custom). |
| multisource_datasets_type | `str` | `"interleave"` | Dataset type for multisource training. |
| source_name | `str` | `None` | Dataset name. Loaded from multisource YAML if multisource is enabled. |
| dyn_bsz_buffer_size | `int` | `200` | Buffer size for dynamic batch size. |
| text_keys | `str` | `None` | Key to retrieve text from data. Auto-resolved: `"content_split"` for plaintext, `"messages"` for conversation, `"text"` for classification. |
| chat_template | `str` | `"default"` | Chat template name. |
| max_seq_len | `int` | `2048` | Maximum sequence length. |
| silent_exception | `bool` | `False` | Whether to ignore exceptions when loading data. |
| dataloader | `DataloaderConfig` | — | DataLoader construction parameters. |

### DataloaderConfig

`data.dataloader.*` — DataLoader construction parameters.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| type | `str` | `"native"` | Type of the dataloader. |
| num_workers | `int` | `2` | Number of workers for data loading. |
| prefetch_factor | `int` | `2` | Number of batches loaded in advance per worker. |
| drop_last | `bool` | `True` | Whether to drop the last incomplete batch. |
| pin_memory | `bool` | `True` | Whether to pin memory for the dataloader. |

### TrainingArguments

`train.*` — Top-level training configuration.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| train_architecture | `Literal["full", "lora"]` | `"full"` | Parameter update strategy: `"full"` for standard SFT, `"lora"` for LoRA. |
| dyn_bsz | `bool` | `True` | Enable dynamic batch size for padding-free training. |
| micro_batch_size | `int` | `1` | Number of samples per iteration on each device. |
| global_batch_size | `Optional[int]` | `None` | Global batch size. If `None`, uses `micro_batch_size × dp_size`. |
| num_train_epochs | `int` | `1` | Number of training epochs. |
| pad_to_length | `bool` | `False` | Pad packed sequences to a fixed length (requires `dyn_bsz`). |
| bsz_warmup_ratio | `float` | `0` | Ratio of batch size warmup steps. |
| bsz_warmup_init_mbtoken | `int` | `200` | Initial number of tokens in a batch during warmup. |
| init_device | `Literal["cpu", "cuda", "meta", "npu"]` | `"cuda"` | Device for model weight initialization. Use `"meta"` or `"cpu"` for large models (>30B). |
| broadcast_model_weights_from_rank0 | `bool` | `True` | Only rank 0 reads weights from disk; other ranks receive via broadcast. |
| enable_full_determinism | `bool` | `False` | Enable full determinism (bitwise alignment). |
| enable_batch_invariant_mode | `bool` | `False` | Enable batch invariant mode. |
| empty_cache_steps | `int` | `500` | Steps between two `torch.cuda.empty_cache()` calls. |
| gc_steps | `int` | `500` | Steps between two `gc.collect()` calls. Disabled if positive. |
| eval_steps | `int` | `0` | Steps between evaluations. `0` to disable. |
| eval_epochs | `int` | `1` | Epochs between evaluations. `0` to disable. |
| seed | `int` | `42` | Random seed. |
| enable_compile | `bool` | `False` | Enable `torch.compile`. |
| max_steps | `Optional[int]` | `None` | Max training steps per epoch (debug only). |
| optimizer | `OptimizerConfig` | — | Optimizer and learning-rate schedule. |
| wandb | `WandbConfig` | — | Weights & Biases logging. |
| profile | `ProfileConfig` | — | Torch profiler settings. |
| gradient_checkpointing | `GradientCheckpointingConfig` | — | Gradient checkpointing settings. |
| accelerator | `AcceleratorConfig` | — | Parallelism and distributed-training topology. |
| checkpoint | `CheckpointConfig` | — | Checkpoint saving and loading. |

### OptimizerConfig

`train.optimizer.*` — Optimizer and learning-rate schedule.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| type | `Literal["adamw", "anyprecision_adamw"]` | `"adamw"` | Optimizer type. |
| lr | `float` | `5e-5` | Maximum / default learning rate. |
| lr_min | `float` | `1e-7` | Minimum learning rate. |
| lr_start | `float` | `0.0` | Starting learning rate for warmup. |
| lr_warmup_ratio | `float` | `0` | Ratio of learning rate warmup steps. |
| lr_decay_style | `str` | `"constant"` | Learning rate scheduler (`"constant"`, `"linear"`, `"cosine"`). |
| lr_decay_ratio | `float` | `1.0` | Ratio of learning rate decay steps. |
| weight_decay | `float` | `0` | L2 regularization strength. |
| no_decay_modules | `List[str]` | `[]` | Modules excluded from weight decay (e.g. `RMSNorm`). |
| no_decay_params | `List[str]` | `[]` | Parameters excluded from weight decay (e.g. `bias`). |
| max_grad_norm | `float` | `1.0` | Gradient clipping norm. |

### WandbConfig

`train.wandb.*` — Weights & Biases logging.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable | `bool` | `False` | Enable W&B logging. |
| project | `str` | `"VeOmni"` | W&B project name. |
| name | `Optional[str]` | `None` | W&B experiment name. |
| id | `Optional[str]` | `None` | W&B run ID for resuming a previous run. |

### ProfileConfig

`train.profile.*` — Torch profiler settings.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable | `bool` | `False` | Enable profiling. |
| start_step | `int` | `1` | Start step for profiling. |
| end_step | `int` | `2` | End step for profiling. |
| trace_dir | `str` | `"./trace"` | Directory to save profiling traces. |
| record_shapes | `bool` | `True` | Record input tensor shapes. |
| profile_memory | `bool` | `True` | Record memory usage. |
| with_stack | `bool` | `True` | Record stack traces. |
| rank0_only | `bool` | `True` | Profile rank 0 only. |

### GradientCheckpointingConfig

`train.gradient_checkpointing.*` — Activation recomputation settings.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable | `bool` | `True` | Enable gradient checkpointing. |
| debug | `bool` | `False` | Enable [checkpoint debugging](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.set_checkpoint_debug_enabled). |
| enable_reentrant | `bool` | `False` | Use reentrant gradient checkpointing. |

### AcceleratorConfig

`train.accelerator.*` — Parallelism and distributed-training topology.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| dp_replicate_size | `int` | `-1` | Data parallel replicate size. |
| dp_shard_size | `int` | `-1` | Data parallel shard degree. |
| tp_size | `int` | `1` | Tensor parallel size. |
| ep_size | `int` | `1` | Expert parallel size. |
| ep_outside | `bool` | `False` | Expert parallelism outside in EP-FSDP. |
| pp_size | `int` | `1` | Pipeline parallel size. |
| ulysses_size | `int` | `1` | Ulysses sequence parallel size. |
| enable_async | `bool` | `False` | Enable async Ulysses. |
| cp_size | `int` | `1` | Ring-attention context parallel size. |
| fsdp_config | `FSDPConfig` | — | FSDP sharding configuration. |
| offload_config | `OffloadConfig` | — | Activation offload settings. |

### FSDPConfig

`train.accelerator.fsdp_config.*` — FSDP sharding configuration.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| fsdp_mode | `Literal["ddp", "fsdp1", "fsdp2"]` | `"ddp"` | Data parallel mode. |
| reshard_after_forward | `bool` | `True` | Reshard after forward (FSDP2). |
| reshard_after_backward | `bool` | `True` | Reshard after backward (FSDP2). |
| full_shard | `bool` | `True` | Enable full sharding — equivalent to ZeRO-3. |
| forward_prefetch | `bool` | `True` | Enable forward prefetch. |
| offload | `bool` | `False` | Enable CPU offload (FSDP1 only). |
| max_load_broadcast_size | `float` | `20.0` | Maximum size (in GB) of parameters broadcasted from rank 0 during loading weights (FSDP2). Parameters exceeding this threshold will be chunked according to the parallel plan before broadcasting. |
| mixed_precision | `MixedPrecisionConfig` | — | Mixed precision configuration. |

### MixedPrecisionConfig

`train.accelerator.fsdp_config.mixed_precision.*` — Mixed precision configuration.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable | `bool` | `True` | Enable mixed precision training. |
| param_dtype | `str` | `"bfloat16"` | Dtype for the unsharded parameter (DDP, FSDP1). |
| reduce_dtype | `str` | `"float32"` | Dtype for gradient reduction (i.e. reduce-scatter or all-reduce) (DDP, FSDP1). |
| buffer_dtype | `str` | `None` | Dtype for the buffer (DDP, FSDP1). |
| output_dtype | `str` | `None` | Dtype for casting floating-point forward outputs (DDP, FSDP1). |
| cast_forward_inputs | `bool` | `True` | Enable mixed precision cast forward inputs (FSDP2). |


### OffloadConfig

`train.accelerator.offload_config.*` — Activation offload settings.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable_activation | `bool` | `False` | Enable activation offload to CPU. |
| activation_gpu_limit | `float` | `0.0` | GB of activations allowed to remain on GPU. |

### CheckpointConfig

`train.checkpoint.*` — Checkpoint saving and loading.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| output_dir | `str` | `"output"` | Path to save model checkpoints. |
| manager | `str` | `"dcp"` | Checkpoint manager. |
| save_async | `bool` | `False` | Save checkpoints asynchronously. |
| load_path | `Optional[str]` | `None` | Path to checkpoint for resuming training. Use `"auto"` for auto-detection. |
| save_steps | `int` | `0` | Steps between checkpoint saves. `0` to disable. |
| save_epochs | `int` | `1` | Epochs between checkpoint saves. `0` to disable. |
| hf_save_steps | `int` | `0` | Steps between HuggingFace weight saves. `0` to disable. |
| hf_save_epochs | `int` | `0` | Epochs between HuggingFace weight saves. `0` to disable. |
| save_hf_weights | `bool` | `True` | Save HuggingFace-format weights to the last checkpoint directory. |

### InferArguments

Standalone inference configuration.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| model_path | `str` | **Required** | Path to the pre-trained model. |
| tokenizer_path | `Optional[str]` | `None` | Path to the tokenizer. Defaults to `model_path`. |
| seed | `int` | `42` | Random seed. |
| do_sample | `bool` | `True` | Enable sampling in decoding. |
| temperature | `float` | `1.0` | Sampling temperature. |
| top_p | `float` | `1.0` | Nucleus sampling top-p value. |
| max_tokens | `int` | `1024` | Maximum tokens to generate. |

---

## VLM Extensions

Additional fields for Vision-Language Model training, defined in `veomni.trainer.vlm_trainer`.

### VLMTrainingArguments

Extends `TrainingArguments` with ViT / audio tower controls.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| freeze_vit | `bool` | `False` | Freeze ViT parameters. |
| freeze_audio_tower | `bool` | `False` | Freeze audio tower parameters. |
| vit_lr | `float` | `1e-6` | Maximum learning rate for ViT parameters. |

### VLMMModelArguments

Extends `ModelArguments` with encoder data-balancing options.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| encoder_data_balance | `Optional[bool]` | `False` | Enable encoder data balancing (e.g. for Qwen3-VL). |
| encoder_data_balance_sorting_algo | `Optional[str]` | `"post_mbs_balancing_greedy_without_pad"` | Sorting algorithm for encoder data balancing. |

### VLMMDataArguments

Extends `DataArguments` with multimodal input configs.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| mm_configs | `Optional[Dict]` | `{}` | Multimodal input configuration. |

---

## DPO Reference

(dpo-arguments)=
### DPOConfig

`dpo_config.*` — Direct Preference Optimization hyperparameters.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| beta | `float` | `0.1` | KL penalty coefficient. Controls deviation from the reference model. |
| label_smoothing | `float` | `0.0` | Label smoothing for DPO loss. Non-zero values assume noisy preference labels. |
| reference_free | `bool` | `False` | If `True`, ignore the reference model and use an implicit uniform reference. |
| loss_type | `"sigmoid" \| "ipo"` | `"sigmoid"` | DPO loss variant: `sigmoid` for standard DPO, `ipo` for Identity Preference Optimization. |
| average_log_prob | `bool` | `False` | If `True`, average log probs per token instead of summing. |
| refer_model_precision | `"float32" \| "bfloat16"` | `"bfloat16"` | dtype used to load the frozen reference model. |
