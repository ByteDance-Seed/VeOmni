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
      fsdp_mode: fsdp2
  init_device: meta
  checkpoint:
    manager: dcp
```

---

## Configuration

Top-level configuration that assembles all argument groups.

* `VeOmniArguments` — Root config: `model` + `data` + `train`
* `VeOmniVLMArguments` — VLM extension of `VeOmniArguments`
* `VeOmniDiTArguments` — diffusion-transformer extension of `VeOmniArguments`

---

## Model

Model architecture, paths, and multimodal encoder / decoder setup.

* `ModelArguments` — `model.*`
* `OpsImplementationConfig` — `model.ops_implementation.*`

### VLM Extensions

* `VLMMModelArguments` — extends `ModelArguments` with encoder data-balancing options

### DiT Extensions

* `DiTModelArguments` — extends `ModelArguments` with condition-model settings

---

## Data

Dataset paths, tokenization, and batching configuration.

* `DataArguments` — `data.*`
* `DataloaderConfig` — `data.dataloader.*`

### VLM Extensions

* `VLMMDataArguments` — extends `DataArguments` with multimodal configs (`mm_configs`)

### DiT Extensions

* `DiTDataArguments` — extends `DataArguments` with diffusion input and offline-embedding settings

---

## Training

Training loop, optimizer, parallelism, checkpointing, profiling, and logging.

* `TrainingArguments` — `train.*`
    * `OptimizerConfig` — `train.optimizer.*`
    * `WandbConfig` — `train.wandb.*`
    * `ProfileConfig` — `train.profile.*`
    * `ChannelLossConfig` — `train.channel_loss.*`
    * `GradientCheckpointingConfig` — `train.gradient_checkpointing.*`
    * `TorchCompileConfig` — `train.torch_compile.*`
    * `ChunkMBSConfig` — `train.chunk_mbs_config.*`
    * `AcceleratorConfig` — `train.accelerator.*`
        * `FSDPConfig` — `train.accelerator.fsdp_config.*`
          * `MixedPrecisionConfig` — `train.accelerator.fsdp_config.mixed_precision`
        * `OffloadConfig` — `train.accelerator.offload_config.*`
    * `CheckpointConfig` — `train.checkpoint.*`

### VLM Extensions

* `VLMTrainingArguments` — extends `TrainingArguments` with ViT / audio freeze & learning-rate options

### DiT Extensions

* `DiTTrainingArguments` — extends `TrainingArguments` with the diffusion training workflow

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
| model_config | `Optional[Dict]` | `{}` | Values used to override the loaded foundation-model config. |
| tokenizer_path | `Optional[str]` | `None` | Path to the tokenizer. Defaults to `config_path`. |
| safetensor_idx_path | `Optional[str]` | `None` | Path to `model.safetensors.index.json`. |
| foundation | `Dict[str, str]` | `{}` | Foundation model extra config. |
| encoders | `Dict` | `{}` | Multimodal encoder configs keyed by modality (`image`, `video`, `audio`). |
| decoders | `Dict` | `{}` | Multimodal decoder configs keyed by modality (`image`). |
| input_encoder | `Literal["encoder", "decoder"]` | `"encoder"` | Whether to use the encoder or decoder to encode input images. |
| output_encoder | `Literal["encoder", "decoder"]` | `"decoder"` | Whether to use the encoder or decoder to encode output images. |
| encode_target | `bool` | `False` | Whether to encode training targets with decoder (diffusion only). |
| basic_modules | `Optional[List[str]]` | `[]` | Additional modules beyond `_no_split_modules` to shard in FSDP. |
| lora_config | `Optional[Dict]` | `{}` | Native VeOmni LoRA configuration. See the LoRA feature guide. |
| ops_implementation | `OpsImplementationConfig` | — | Attention / MoE kernel configuration. |

### OpsImplementationConfig

`model.ops_implementation.*` — Attention, MoE, and fused kernel implementation.

Each `*_implementation` field selects the kernel backend for that operation.
The type is `str` (not `Literal`) so third-party backends can be registered
without modifying the config class.

**Defaults are GPU-optimal** (Liger / Triton / fused_triton). On Ascend NPU,
values that are still equal to the dataclass defaults automatically resolve as
follows:

| GPU default field | NPU fallback |
|---|---|
| `rms_norm_implementation` | `npu` |
| `rotary_pos_emb_implementation` | `npu` |
| `rotary_pos_emb_vision_implementation` | `npu` |
| `swiglu_mlp_implementation` | `eager` |
| `load_balancing_loss_implementation` | `eager` |
| `cross_entropy_loss_implementation` | `npu` |
| `moe_implementation` | `fused_npu` |

Explicit non-default overrides are not rewritten; unsupported NPU values raise
during validation. Qwen3.5's model-specific GatedDeltaNet fields are not in
this global fallback table and must be set to `npu` explicitly on NPU.

NPU validation runs at two times:

- **Config-parse time** (`OpsImplementationConfig.__post_init__`) for the
  seven general-purpose ops (`moe`, `cross_entropy_loss`, `rms_norm`,
  `swiglu_mlp`, `rotary_pos_emb`, `rotary_pos_emb_vision`,
  `load_balancing_loss`). Errors fire
  immediately with a model-agnostic allow-list.
- **OpSlot-bind time** (`KERNEL_REGISTRY.resolve` via the kernel's
  `HardwareRequirement`) for Qwen3.5-only ops (`rms_norm_gated`,
  `causal_conv1d`, `chunk_gated_delta_rule`). Validating these at config
  parse would force every NPU user to override them even when training
  non-Qwen3.5 models, so the check fires only when Qwen3.5's patched
  modeling is actually loaded. Qwen3.5 on NPU should select the `"npu"`
  backend for these three operations.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| attn_implementation | `Optional[Literal[...]]` | `"flash_attention_2"` | Attention implementation to use. |
| moe_implementation | `str` | `"fused_triton"` | MoE experts forward implementation. `fused_triton` uses Triton group-gemm (GPU, SM70+); `fused_quack` uses Quack CUTLASS/CuTe (GPU, SM90+); `fused_npu` uses the NPU group-gemm kernel; `eager` is the reference loop. A value still equal to the GPU default auto-resolves to `fused_npu` on NPU; explicit incompatible non-default overrides raise. |
| cross_entropy_loss_implementation | `str` | `"liger_kernel"` | Cross-entropy loss. `liger_kernel` (default, GPU only) fuses `lm_head` linear + CE; requires VeOmni-patched modeling files that pass `hidden_states=`/`weights=` to `self.loss_function(...)` — unpatched HF models that pass logits will RuntimeError. `chunk_loss` is the hardware-agnostic chunked F.linear+CE (CUDA + NPU). `npu` is a back-compat alias for `chunk_loss`. `eager` is `F.cross_entropy`. |
| rms_norm_implementation | `str` | `"liger_kernel"` | RMSNorm. Known values: `liger_kernel` (default, GPU only), `npu`, `triton` (DeepSeek-V3 only; GPU only), `eager`. |
| swiglu_mlp_implementation | `str` | `"liger_kernel"` | SwiGLU MLP. Known values: `liger_kernel` (default, GPU only), `eager`. There is no NPU backend, so a value still equal to the default auto-resolves to `eager` on NPU. |
| rotary_pos_emb_implementation | `str` | `"liger_kernel"` | Rotary pos emb. Known values: `liger_kernel` (default, GPU only), `npu`, `triton` (DeepSeek-V3 only; GPU only), `eager`. |
| rotary_pos_emb_vision_implementation | `str` | `"eager"` | Vision rotary positional embedding. Known values: `eager`, `npu`. |
| load_balancing_loss_implementation | `str` | `"triton"` | MoE load-balancing loss. `triton` uses the fused CUDA kernel; `eager` is the pure-PyTorch reference. On NPU, config normalization maps every value equal to the default `triton` (including an explicit YAML value) to `eager`. |
| rms_norm_gated_implementation | `str` | `"fla"` | Gated RMSNorm (Qwen3.5 GatedDeltaNet `self.norm`). Known values: `eager`, `fla` (FLA `FusedRMSNormGated`, GPU), `npu`. |
| causal_conv1d_implementation | `str` | `"fla"` | Varlen depthwise causal conv1d (Qwen3.5 GatedDeltaNet pre-mixer). Known values: `eager`, `fla` (GPU), `npu` (requires `triton-ascend`). `eager` does not support the varlen path. |
| chunk_gated_delta_rule_implementation | `str` | `"fla"` | Chunk gated delta-rule kernel for Qwen3.5 linear attention. Known values: `eager`, `fla` (GPU), `flash_qla` (Hopper SM90), `npu` (requires `triton-ascend`). `eager` does not support varlen training. |
| dsa_indexer_implementation | `Literal["eager", "cudnn", "tilelang"]` | `"eager"` | DeepSeek sparse-attention top-k indexer implementation. `tilelang` selects the DeepSeek-V4 Lightning Indexer kernel and requires an SM90+ CUDA GPU. |
| dsa_attention_implementation | `Literal["eager", "flashmla_cudnn", "tilelang"]` | `"eager"` | DeepSeek sparse-attention implementation. `tilelang` selects the DeepSeek-V4 sparse MQA kernel and requires an SM90+ CUDA GPU. |
| mhc_implementation | `Literal["eager", "tilelang"]` | `"eager"` | DeepSeek V4 manifold-constrained Hyper-Connection implementation. `tilelang` enables the forward/backward path provided by the `tile-kernels` package and requires an SM90+ CUDA GPU. |

### DataArguments

`data.*` — Dataset paths, tokenization, and batching.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| train_path | `str` | **Required** | Path of the training dataset. Use comma to separate multiple datasets. |
| eval_path | `Optional[str]` | `None` | Path of the evaluation dataset. |
| train_size | `int` | `10_000_000` | Number of tokens for training (used to compute steps under dynamic batch). |
| train_sample | `int` | `10_000` | Number of samples for training (used to compute steps under non-dynamic batch). |
| data_type | `Literal["plaintext", "conversation", "diffusion", "classification", "dpo"]` | `"conversation"` | Type of the training data. |
| datasets_type | `str` | `"mapping"` | `IterableDataset` or `MappingDataset` (or custom). |
| multisource_datasets_type | `str` | `"interleave"` | Dataset type for multisource training. |
| source_name | `str` | `None` | Dataset name. Loaded from multisource YAML if multisource is enabled. |
| dyn_bsz_buffer_size | `int` | `200` | Buffer size for dynamic batch size. |
| text_keys | `str` | `None` | Key to retrieve text from data. Auto-resolved: `"content_split"` for plaintext, `"messages"` for conversation, `"text"` for classification, `"chosen"` for DPO. |
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
| worker_num_threads | `Optional[int]` | `None` | Number of PyTorch threads used by each DataLoader worker. |
| use_background_prefetcher | `bool` | `False` | Enable background prefetching around the DataLoader. |

### TrainingArguments

`train.*` — Top-level training configuration.

| Field | Type | Default | Description |
| --- | --- | --- | --- |

| dyn_bsz | `bool` | `True` | Enable dynamic batch size for padding-free training. |
| dyn_bsz_runtime | `Literal["main", "worker"]` | `"main"` | Where dynamic batching runs. `"main"` keeps the legacy main-process batching path; `"worker"` batches inside DataLoader workers to support exact `StatefulDataLoader` resume. |
| dyn_bsz_count_mode | `Literal["total", "effective"]` | `"total"` | How dynamic batching counts tokens. `"total"` uses `attention_mask.sum()` (legacy behavior); `"effective"` counts only `labels != IGNORE_INDEX` for balancing while still applying a physical-token cap. |
| dyn_bsz_physical_overflow_ratio | `float` | `1.5` | Physical-token cap multiplier used with `dyn_bsz_count_mode="effective"`: `ceil(micro_batch_size * max_seq_len * ratio)`. Values above `1.0` allow controlled physical overflow so effective-token batching does not degenerate into total-token batching. |
| micro_batch_size | `int` | `1` | Number of samples per iteration on each device. |
| global_batch_size | `Optional[int]` | `None` | Global batch size. If `None`, uses `micro_batch_size × dp_size`. |
| num_train_epochs | `int` | `1` | Number of training epochs. |
| pad_to_length | `bool` | `False` | Pad packed sequences to a fixed length (requires `dyn_bsz`). |
| bsz_warmup_ratio | `float` | `0` | Ratio of batch size warmup steps. |
| bsz_warmup_init_mbtoken | `int` | `200` | Initial number of tokens in a batch during warmup. |
| init_device | `Literal["cpu", "cuda", "meta", "npu"]` | `"meta"` | Device for model weight initialization. `"meta"` is required for FSDP2. |
| broadcast_model_weights_from_rank0 | `bool` | `True` | Only rank 0 reads weights from disk; other ranks receive via broadcast. |
| ep_sharded_stream_load | `bool` | `False` | Opt-in fast/low-memory MoE loader: each rank reads only its ExtraParallel dim-0 slice from the checkpoint. Requires `broadcast_model_weights_from_rank0=False` and a model with an ExtraParallel parallel_plan. |
| enable_full_determinism | `bool` | `False` | Enable full determinism (bitwise alignment). |
| enable_batch_invariant_mode | `bool` | `False` | Enable batch invariant mode. |
| empty_cache_steps | `int` | `500` | Steps between device-cache cleanup calls. A non-positive value disables scheduled cleanup. |
| gc_steps | `int` | `500` | When positive, disable automatic Python GC and run `gc.collect()` every N steps. A non-positive value leaves automatic GC enabled and disables scheduled collection. |
| eval_steps | `int` | `0` | Steps between evaluations. `0` to disable. |
| eval_epochs | `int` | `1` | Epochs between evaluations. `0` to disable. |
| seed | `int` | `42` | Random seed. |
| max_steps | `Optional[int]` | `None` | Max training steps per epoch (debug only). |
| moe_load_balance_monitor_interval | `int` | `0` | Log a globally reduced MoE expert-load heatmap every N steps. `0` disables monitoring. |
| optimizer | `OptimizerConfig` | — | Optimizer and learning-rate schedule. |
| wandb | `WandbConfig` | — | Weights & Biases logging. |
| profile | `ProfileConfig` | — | Torch profiler settings. |
| channel_loss | `ChannelLossConfig` | — | Detached per-channel causal-LM loss logging. |
| gradient_checkpointing | `GradientCheckpointingConfig` | — | Gradient checkpointing settings. |
| torch_compile | `TorchCompileConfig` | — | Per-block `torch.compile` settings. |
| chunk_mbs_config | `ChunkMBSConfig` | — | Packed-sequence layer micro-batching settings. |
| accelerator | `AcceleratorConfig` | — | Parallelism and distributed-training topology. |
| checkpoint | `CheckpointConfig` | — | Checkpoint saving and loading. |

### TorchCompileConfig

`train.torch_compile.*` — Per-block `torch.compile` options for text training. This path currently supports text trainers only and requires `train.dyn_bsz=True` plus `train.pad_to_length=True`, so packed inputs have stable shapes. The default `mode=None` follows TorchTitan's main path by using the `inductor` backend without CUDA Graph replay. Setting `mode="reduce-overhead"` explicitly enables CUDA Graphs on the `inductor` backend and requires `train.accelerator.fsdp_config.reshard_after_forward=False`. When CUDA Graphs are enabled, each micro-batch calls `torch.compiler.cudagraph_mark_step_begin()` when available so CUDA Graph Trees can separate iterations.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable | `bool` | `False` | Enable per-block `torch.compile` on FSDP2 decoder blocks. |
| backend | `Optional[str]` | `"inductor"` | Backend passed to `torch.compile`. |
| mode | `Optional[str]` | `None` | Mode passed to `torch.compile`. `None` uses the `inductor` backend default. `"reduce-overhead"` enables CUDA Graphs on the `inductor` backend, requires `train.accelerator.fsdp_config.reshard_after_forward=False`, and must be `None` when `backend="cudagraphs"`. |
| fullgraph | `bool` | `True` | Whether to pass `fullgraph=True` to `torch.compile`. |
| dynamic | `bool` | `False` | Whether to pass `dynamic=True` to `torch.compile`. |

### OptimizerConfig

`train.optimizer.*` — Optimizer and learning-rate schedule.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| type | `Literal["adamw", "anyprecision_adamw", "muon"]` | `"adamw"` | Optimizer type. `muon` builds Muon and AdamW parameter groups. |
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
| muon_lr | `Optional[float]` | `None` | Learning rate for Muon-managed 2-D/3-D weights. Unset: inherits `lr` under `match_rms_adamw`, else `25×lr` under `original`. |
| muon_momentum | `float` | `0.95` | Momentum factor for Muon. |
| muon_nesterov | `bool` | `True` | Enable Nesterov momentum for Muon. |
| muon_weight_decay | `float` | `0.0` | Decoupled weight decay for Muon parameter groups. |
| muon_ns_steps | `int` | `5` | Number of Newton–Schulz iterations. |
| muon_ns_coefficients | `List[float]` | `[3.4445, -4.7750, 2.0315]` | Quintic Newton–Schulz polynomial coefficients. |
| muon_eps | `float` | `1e-7` | Numerical-stability epsilon used in spectral-norm normalization. |
| muon_adjust_lr_fn | `Literal["original", "match_rms_adamw"]` | `"match_rms_adamw"` | Per-matrix learning-rate adjustment strategy. |
| muon_expert_zero_comm | `bool` | `False` | Use whole-expert `Shard(0)` when the FSDP+ExtraParallel topology permits zero-communication expert Muon updates. |
| muon_ns_implementation | `Literal["std", "gram", "gram_quack"]` | `"gram_quack"` | Newton–Schulz backend: standard, pure-PyTorch Gram-NS, or Gram-NS with quack kernels (default; falls back to `gram` if unavailable). |
| muon_gram_ns_reset_iterations | `List[int]` | `[2]` | Restart indices for Gram Newton–Schulz (ignored by `std`). |

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
| with_modules | `bool` | `False` | Record module hierarchy in profiler traces. |
| rank0_only | `bool` | `True` | Profile rank 0 only. |

### ChannelLossConfig

`train.channel_loss.*` — Detached per-channel causal-LM loss logging.

This is an observability-only side channel. It computes detached per-token CE
from the model loss inputs, aggregates by packed-sequence source metadata, and
adds metrics such as `channel_loss/<source-id>__<source>` to the normal step metrics. It does
not change the returned training loss or gradients. Fused-loss backends may
recompute the LM-head projection on sampled steps, so the default interval is
10 steps; set `interval=1` for per-step metrics. DiT trainers and
`data.data_type="classification"` are not supported because they do not optimize
a causal-LM objective. SeedOmni's `Qwen3MoeFoundationModel` is also unsupported
because its legacy forward bypasses the observable loss dispatch. `BaseRLTrainer`
is unsupported because it packs source alignment metadata after the common step
lifecycle. In DPO training, only the policy-model forward is observed; the
reference-model forward is excluded, and the chosen/rejected segments both use
their preference pair's source metadata. If distinct source names sanitize to
the same metric key, the stable source-ID prefix keeps their time series
distinct from the first emission.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable | `bool` | `False` | Enable channel loss logging. |
| interval | `int` | `10` | Compute and log channel loss every N optimizer steps. |
| source_id_keys | `List[str]` | `["channel_id", "source_id", "dataset_id", "ds_idx"]` | Batch metadata keys to read as channel/source IDs. |
| source_name_keys | `List[str]` | `["channel_name", "source_name", "dataset_name", "data_name"]` | Batch metadata keys to read as display names. |
| extra_strip_keys | `List[str]` | `["cur_token_num"]` | Extra metadata keys removed before model forward. |
| loss_metric_prefix | `str` | `"channel_loss"` | Prefix for average CE metrics. |
| weighted_loss_metric_prefix | `str` | `"channel_loss_weighted"` | Prefix for loss-sum divided by all logged step tokens. |
| token_count_metric_prefix | `str` | `"channel_tokens"` | Prefix for supervised token-count metrics. |
| log_weighted_loss | `bool` | `True` | Log weighted loss metrics. |
| log_token_count | `bool` | `True` | Log token-count metrics. |
| strict | `bool` | `False` | Raise when source metadata is missing or cannot be aligned with packed segments; otherwise skip invalid batches. |

### GradientCheckpointingConfig

`train.gradient_checkpointing.*` — Activation recomputation settings.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable | `bool` | `True` | Enable gradient checkpointing. |
| debug | `bool` | `False` | Enable [checkpoint debugging](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.set_checkpoint_debug_enabled). |
| enable_reentrant | `bool` | `False` | Use reentrant gradient checkpointing. |
| early_stop | `bool` | `True` | Stop non-reentrant checkpoint recomputation as soon as all needed tensors are computed. PyTorch ignores this option when `enable_reentrant=True`. |

### ChunkMBSConfig

`train.chunk_mbs_config.*` — Packed-sequence layer micro-batching settings.

`chunk_mbs` is the number of packed samples per layer chunk. With dynamic batching, the runtime sample
count is inferred from `cu_seq_lens_q`, so it is independent of `train.micro_batch_size`. Chunks are cut
only on packed sample boundaries. The current implementation supports trainer-based SFT with packed-sequence
FlashAttention kwargs using `torch.int32` cumulative lengths, identical query/key metadata, exactly one
`*DecoderLayer` class and one matching decoder stack, decoder layers derived from Transformers'
`GradientCheckpointingLayer`, and decoder states with shape `[1, sequence, hidden]`. Gradient checkpointing may be
enabled or disabled; when enabled, it must use the non-reentrant implementation. CPU model-level numerical coverage
currently includes Qwen3-VL and dense Qwen3.5; accelerator-specific kernels require separate hardware validation. Sequence parallelism,
tensor parallelism, pipeline parallelism, ExtraParallel/MoE, DiT trainers, RL trainers, DPO, the custom Omni training loop,
`pad_to_length`, and `torch.compile` are not supported. Chunk boundaries must also align with linear-attention
cumulative sequence boundaries when that metadata is present. Models with ambiguous decoder classes or stacks fail
validation instead of applying ChunkMBS to multiple stacks.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable | `bool` | `False` | Enable ChunkMBS for packed-sequence decoder layers listed in `model._no_split_modules`. |
| chunk_mbs | `int` | `1` | Number of packed samples per layer chunk. |

### AcceleratorConfig

`train.accelerator.*` — Parallelism and distributed-training topology.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| dp_replicate_size | `int` | `-1` | Data parallel replicate size for both dense and moe parameters. |
| dp_shard_size | `int` | `-1` | Data parallel shard degree. |
| tp_size | `int` | `1` | Tensor parallel size. |
| ep_size | `int` | `1` | Expert parallel size, should be fit into dp_shard group if HSDP enabled |
| ep_outside | `bool` | `False` | Expert parallelism outside in EP-FSDP. |
| extra_parallel_sizes | `List[int]` | `[]` | Sizes of additional parallel dimensions; EP is appended automatically. |
| extra_parallel_placement_innermost | `List[bool]` | `[]` | Whether each additional parallel dimension is placed innermost relative to FSDP. |
| extra_parallel_names | `List[str]` | `[]` | Names of additional parallel dimensions; `ep` is appended automatically. |
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
| fsdp_mode | `Literal["ddp", "fsdp2"]` | `"fsdp2"` | Data parallel mode. |
| reshard_after_forward | `bool` | `True` | Reshard after forward (FSDP2). |
| reshard_after_backward | `bool` | `True` | Reshard after backward (FSDP2). |
| forward_prefetch | `bool` | `True` | Enable forward prefetch. |
| offload | `bool` | `False` | Enable CPU offload. |
| max_load_broadcast_size | `float` | `20.0` | Maximum size (in GB) of parameters broadcasted from rank 0 during loading weights (FSDP2). Parameters exceeding this threshold will be chunked according to the parallel plan before broadcasting. |
| mixed_precision | `MixedPrecisionConfig` | — | Mixed precision configuration. |

### MixedPrecisionConfig

`train.accelerator.fsdp_config.mixed_precision.*` — Mixed precision configuration.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| enable | `bool` | `True` | Enable mixed precision training. |
| param_dtype | `str` | `"bfloat16"` | Dtype for the unsharded parameter. |
| reduce_dtype | `str` | `"float32"` | Dtype for gradient reduction (i.e. reduce-scatter or all-reduce). |
| output_dtype | `str` | `None` | Dtype for casting floating-point forward outputs (FSDP2). |
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
| dcp_save_to_lowest_rank | `bool` | `False` | Write each replicated DCP shard from the lowest global rank that holds it instead of load-balancing across replicas. On a non-shared filesystem this concentrates the deduplicated copy onto the lowest-ranked replica group rather than scattering it across replicas; in the standard HSDP layout (shard within a node, replicate across nodes) that group is one node, which then holds a complete checkpoint. Only affects replicated data — unique expert/tensor/pipeline-parallel shards stay distributed. Leave `False` when `output_dir` is shared. |
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

## DiT Extensions

Additional fields for diffusion-transformer training, defined in
`veomni.trainer.dit_trainer`. The root `VeOmniDiTArguments` combines the three
derived argument groups below.

### DiTModelArguments

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| condition_model_path | `Optional[str]` | `None` | Path to the condition model. |
| condition_model_cfg | `Optional[Dict]` | `{}` | Condition-model configuration. |

### DiTDataArguments

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| mm_configs | `Optional[Dict]` | `{}` | Multimodal input configuration. |
| offline_embedding_save_dir | `Optional[str]` | `None` | Directory used to save offline embeddings. |
| shuffle | `bool` | `True` | Shuffle the training dataset. |

### DiTTrainingArguments

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| training_task | `Literal["offline_training", "online_training", "offline_embedding"]` | `"online_training"` | Select offline training, online training, or offline embedding generation. |

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
