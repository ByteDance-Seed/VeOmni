# Arguments API Reference

## Model configuration arguments
| Name | Type | Description | Default Value |
| --- | --- | --- | --- |
| model.config_path | str | Path to the model huggingface configuration, like `config.json` | model.model_path |
| model.model_path | str | Path to the model parameter file. If empty, random initialization will be performed | None |
| model.tokenizer_path | str | Path to the tokenizer | model.model_path |
| model.safetensor_idx_path | str | Path to the safetensor index file | None |
| model.encoders | dict | Configuration file for multi-modal encoders | {} |
| model.decoders | dict | Configuration file for multi-modal decoders | {} |
| model.input_encoder | str: {"encoder", "decoder"} | Use the encoder or decoder to encode the input image | encoder |
| model.output_encoder | str: {"encoder", "decoder"} | Use the encoder or decoder to encode the output image | decoder |
| model.encode_target | bool | Whether to encode the training data for the diffusion model | False |
| model.attn_implementation | str: {"eager", "sdpa", "flash_attention_2", "flash_attention_3", "native-sparse"} | The attention implementation to use. | flash_attention_2 |
| model.moe_implementation | str: {"eager", "fused"} | The MoE implementation to use. | None |
| model.basic_modules | List of str | Basic modules beyond model._no_split_modules to be sharded in FSDP. | [] |


## Data configuration arguments

| Name | Type | Description | Default Value |
| --- | --- | --- | --- |
| data.train_path | str | Path of training dataset | Required |
| data.eval_path | str | Path of evaluation dataset | None |
| data.train_size | int | Number of tokens for training to compute training steps for dynamic batch dataloader | 10,000,000 |
| data.train_sample | int | Number of samples for training to compute training steps for non-dynamic batch dataloader | 10,000 |
| data.data_type | str: {"plaintext", "conversation", "classification"} | Dataset type.  | conversation |
| data.dataloader_type | str: {"native"} | Type of the dataloader | native |
| data.datasets_type | str: {"mapping", "iterable"} | Dataset type. `IterativeDataset` or `MappingDataset`, or your custom datsets | mapping |
| data.multisource_datasets_type | str: {"interleave"} | Type of multisource datasets. | interleave |
| data.source_name | str | Name of the data source. Load from multisource yaml if multisource enabled | None |
| data.dyn_bsz_buffer_size | int | Buffer size for dynamic batch size. | 200 |
| data.text_keys | str: {"content_split", "messages"} | The key corresponding to the text samples in the data dictionary. Generally, it is "content_split" for pretraining and "messages" for SFT. | content_split |
| data.chat_template | str | Name of the chat template. | default |
| data.max_seq_len | int | Maximum training length. | 2048 |
| data.num_workers | int | Number of multi-process loaders for the dataloader. | 4 |
| data.prefetch_factor | int | Number of samples preprocessed by the dataloader. | 2 |
| data.drop_last | bool | Whether to discard the remaining data at the end. | True |
| data.pin_memory | bool | Whether to pin the data in the CPU memory. | True |
| data.silent_exception | bool | Whether to ignore exceptions in the dataloader. | False (TODO) |

### Training configuration arguments
| Name | Type | Description | Default Value |
| --- | --- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| train.output_dir | str | Path to save the model. | Required |
| train.architecture | Literal["full", "lora"] | Whether to train the full model or LoRA. | full |
| train.dyn_bsz | bool | Whether to use dynamic batch size. | False |
| train.lr | float | Maximum learning rate.   | 5e - 5 |
| train.lr_min | float | Minimum learning rate.  | 1e - 7 |
| train.lr_start | float | Starting learning rate for warmup. | 0.0 |
| train.weight_decay | float | Weight decay coefficient.      | 0 |
| train.no_decay_modules | List of str | Modules to exclude from weight decay. | [] |
| train.no_decay_params | List of str | Parameters to exclude from weight decay. | [] |
| train.optimizer | str: {"adamw", "anyprecision_adamw"} | Name of the optimizer. | adamw |
| train.max_grad_norm | float | Gradient clipping norm. | 1.0 |
| train.micro_batch_size | int | Number of samples processed simultaneously on each GPU. | 1 |
| train.global_batch_size | int | Global batch size, which must be a multiple of the number of GPUs. | train.micro_batch_size * n_gpus |
| train.num_train_epochs | int | Number of training epochs. | 1 |
| train.pad_to_length | bool | Whether to pad the input to the maximum sequence length when using dyn_bsz. | False |
| train.bsz_warmup_ratio | float | Proportion of batch size warmup in the total number of steps. | 0 |
| train.bsz_warmup_init_mbtoken | int | Initial micro batch size for warmup. | 200 |
| train.lr_warmup_ratio | float | Proportion of learning rate warmup in the total number of steps. | 0 |
| train.lr_decay_style | str: {"constant", "linear", "cosine"} | Name of the learning rate scheduler. | cosine |
| train.lr_decay_ratio | float | Proportion of learning rate decay in the total number of steps. | 1.0 |
| train.enable_reshard_after_forward | bool | Enable reshard after forward for FSDP2. | False |
| train.enable_reshard_after_backward | bool | Enable reshard after backward for FSDP2. | False |
| train.enable_mixed_precision | bool | Whether to enable mixed precision training (higher memory usage but more stable). | True |
| train.enable_gradient_checkpointing | bool | Whether to enable gradient checkpointing to reduce memory usage. | True |
| train.enable_reentrant | bool | Whether to enable reentrant in gradient checkpointing. | True |
| train.enable_full_shard | bool | Whether to use full sharding FSDP (equivalent to ZeRO3). | True |
| train.enable_fsdp_offload | bool | Whether to enable FSDP CPU offloading (only supported for FSDP1). | False |
| train.enable_activation_offload | bool | Whether to enable activation value CPU offloading. | False |
| train.activation_gpu_limit | float | Size of the activation values retained on the GPU (in GB). | 0.0 |
| train.init_device | str | "cpu", "cuda", "meta", "npu", init device for model initialization. use "meta" or cpu for large model(>30B) | cuda |
| train.broadcast_model_weights_from_rank0 | bool | Whether to broadcast model weights from rank 0 to all other ranks. | True |
| train.enable_full_determinism | bool | Whether to enable deterministic mode (for bitwise alignment). | False |
| train.enable_batch_invariant_mode | bool | Whether to enable batch invariant mode. | False |
| train.empty_cache_steps | int | Number of steps between two cache clearings. | 500 |
| train.gc_steps | int | Number of steps between two gc.collect. | 500 |
| train.data_parallel_mode | str: {"ddp", "fsdp1", "fsdp2"} | Data parallel algorithm.  | ddp |
| train.data_parallel_replicate_size | int | Number of replicas for data parallel. | 1 |
| train.data_parallel_shard_size | int | Number of shards for data parallel. | 1 |
| train.tensor_parallel_size | int | Tensor parallel size (currently not supported). | 1 |
| train.expert_parallel_size | int | Expert parallel size (currently only supported DeepseekMOE) | 1 |
| train.ep_outside | bool | Whether to use expert parallel outside in ep-fsdp. | False |
| train.pipeline_parallel_size | int | Pipeline parallel size (currently not supported). | 1 |
| train.ulysses_parallel_size | int | Ulysses sequence parallel size. | 1 |
| train.async_enabled | bool | Whether to enable async ulysses. | False |
| train.context_parallel_size | int | Ring sequence parallel size (currently not supported). | 1 |
| train.ckpt_manager | str: {"dcp"} | Checkpoint manager. | dcp |
| train.save_async | bool | Whether to save checkpoint asynchronously. | False |
| train.load_checkpoint_path | str | Path to the omnistore checkpoint for resuming training.  | None |
| train.save_steps | int | Number of steps between two checkpoint saves. 0 means invalid. | 0 |
| train.save_epochs | int | Number of epochs between two checkpoint saves. 0 means invalid. | 1 |
| train.hf_save_steps | int | Number of steps between two huggingface model weights saves. 0 means invalid. | 0 |
| train.hf_save_epochs | int | Number of epochs between two huggingface model weights saves. 0 means invalid. | 0 |
| train.eval_steps | int | Number of steps between two evaluation. 0 means invalid. | 0 |
| train.eval_epochs | int | Number of epochs between two evaluation. 0 means invalid. | 0 |
| train.save_hf_weights | bool | Whether to save the model weights in the huggingface format at the end of training. It is recommended to set it to False for models > 30B to prevent NCCL timeout. You can convert it after training. | True |
| train.seed | int | Random seed. | 42 |
| train.enable_compile | bool | Whether to enable torch compile. | False |
| train.use_wandb | bool | Whether to enable byted wandb experiment logging. | False |
| train.wandb_project | str | Name of the wandb experiment project. | VeOmni |
| train.wandb_name | str | Name of the wandb experiment. | None |
| train.wandb_id | str | Wandb run ID for resuming a previous run. When specified, training logs will continue in the existing wandb run. | None |
| train.enable_profiling | bool | Whether to use torch profiling. | False |
| train.profile_start_step | int | Starting step of profiling. | 1 |
| train.profile_end_step | int | Ending step of profiling. | 2 |
| train.profile_trace_dir | str | Path to save the profiling results. | ./trace |
| train.profile_record_shapes | bool | Whether to record the shapes of the input tensors. | True |
| train.profile_profile_memory | bool | Whether to record the memory usage. | True |
| train.profile_with_stack | bool | Whether to record the stack information. | True |
| train.profile_rank0_only | bool | Whether to profile only on rank 0. | True |
| train.max_steps | int | Number of steps per training epoch (only used for debugging). | None |

## Inference configuration arguments
| Name | Type | Description | Default Value |
| --- | --- | --- | --- |
| infer.model_path | str | Path to the model parameter file. | Required |
| infer.tokenizer_path | str | Path to the tokenizer. | model.model_path |
| infer.seed | int | Random seed. | 42 |
| infer.do_sample | bool | Whether to enable sampling. | True |
| infer.temperature | float | Sampling temperature. | 1.0 |
| infer.top_p | float | Sampling Top P value. | 1.0 |
| infer.max_tokens | int | Maximum number of tokens generated each time. | 1024 |
