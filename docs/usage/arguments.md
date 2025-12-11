## Config arguments Explanation
### Model configuration arguments
| Name | Type | Description | Default Value |
| --- | --- | --- | --- |
| model.config_path | str | Path to the model huggingface configuration, like `config.json` | model.model_path |
| model.model_path | str | Path to the model parameter file. If empty, random initialization will be performed | None |
| model.tokenizer_path | str | Path to the tokenizer | model.model_path |
| model.encoders | dict | Configuration file for multi-modal encoders | {} |
| model.decoders | dict | Configuration file for multi-modal decoders | {} |
| model.input_encoder | str: {"encoder", "decoder"} | Use the encoder of the encoder or decoder to encode the input image | encoder |
| model.output_encoder | str: {"encoder", "decoder"} | Use the encoder of the encoder or decoder to encode the output image | decoder |
| model.encode_target | bool | Used to encode the training data for the diffusion model | False |

### Data configuration arguments

| Name | Type | Description | Default Value |
| --- | --- | --- | --- |
| data.train_path | str | Path of training dataset | Required |
| data.train_size | int | Total number of tokens in the training set | 10,000,000 |
| data.data_type | str: {"plaintext", "conversation"} | Dataset type.  | conversation |
| data.dataloader_type | str: {"native"} | Use the pytorch dataloader or  | native |
| data.datasets_type | str: {"mapping", "iterable"} | Dataset type. `IterativeDataset` or `MappingDataset`, or your custom datsets | mapping |
| data.text_keys | str: {"content_split", "messages"} | The key corresponding to the text samples in the data dictionary. Generally, it is "content_split" for pretraining and "messages" for SFT. | content_split |
| data.image_keys | str | The key corresponding to the image samples in the data dictionary. Generally, it is "images". | images |
| data.chat_template | str | Name of the chat template. | default |
| data.max_seq_len | int | Maximum training length. | 2048 |
| data.num_workers | int | Number of multi-process loaders for the dataloader. | 4 |
| data.drop_last | bool | Whether to discard the remaining data at the end. | True |
| data.pin_memory | bool | Whether to pin the data in the CPU memory. | True |
| data.prefetch_factor | int | Number of samples preprocessed by the dataloader. | 2 |

#### Training configuration arguments
| Name | Type | Description                                                                                                                                                                    | Default Value |
| --- | --- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| train.output_dir | str | Path to save the model.                                                                                                                                                        | Required |
| train.lr | float | Maximum learning rate.                                                                                                                                                         | 5e - 5 |
| train.lr_min | float | Minimum learning rate.                                                                                                                                                         | 1e - 7 |
| train.weight_decay | float | Weight decay coefficient.                                                                                                                                                      | 0 |
| train.optimizer | str: {"adamw", "anyprecision_adamw"} | Name of the optimizer.                                                                                                                                                         | adamw |
| train.max_grad_norm | float | Gradient clipping norm.                                                                                                                                                        | 1.0 |
| train.micro_batch_size | int | Number of samples processed simultaneously on each GPU.                                                                                                                        | 1 |
| train.global_batch_size | int | Global batch size, which must be a multiple of the number of GPUs.                                                                                                             | train.micro_batch_size * n_gpus |
| train.num_train_epochs | int | Number of training epochs.                                                                                                                                                     | 1 |
| train.rmpad | bool | Whether to use rmpad training based on cu_seqlens.                                                                                                                             | False |
| train.rmpad_with_pos_ids | bool | Whether to use rmpad training based on position_ids.                                                                                                                           | False |
| train.dyn_bsz_margin | int | Number of pad tokens in the dynamic batch.                                                                                                                                     | 0 |
| train.dyn_bsz_runtime | str: {"main", "worker"} | Running process of the dynamic batch.                                                                                                                                          | worker |
| train.bsz_warmup_ratio | float | Proportion of batch size warmup in the total number of steps.                                                                                                                  | 0 |
| train.lr_warmup_ratio | float | Proportion of learning rate warmup in the total number of steps.                                                                                                               | 0 |
| train.lr_decay_style | str: {"constant", "linear", "cosine"} | Name of the learning rate scheduler.                                                                                                                                           | cosine |
| train.lr_decay_ratio | float | Proportion of learning rate decay in the total number of steps                                                                                                                 | 1.0 |
| train.use_doptim | bool | Whether to use the distributed optimizer during Vescale training(no use for torch fsdp)                                                                                        | False |
| train.enable_mixed_precision | bool | Whether to enable mixed precision training (higher memory usage but more stable)                                                                                               | True |
| train.enable_gradient_checkpointing | bool | Whether to enable gradient checkpointing to reduce memory usage.                                                                                                               | True |
| train.enable_reentrant | bool | Whether to enable reentrant in gradient checkpointing.                                                                                                                         | True |
| train.enable_full_shard | bool | Whether to use full sharding FSDP (equivalent to ZeRO3).                                                                                                                       | True |
| train.enable_fsdp_offload | bool | Whether to enable FSDP CPU offloading (only supported for FSDP1).                                                                                                              | False |
| train.enable_activation_offload | bool | Whether to enable activation value CPU offloading.                                                                                                                             | False |
| train.activation_gpu_limit | float | Size of the activation values retained on the GPU (in GB).                                                                                                                     | 0.0 |
| train.enable_manual_eager | bool | Whether to use manual eager during Vescale training.                                                                                                                           | False |
| train.init_device: meta | str | "cpu", "cuda", "meta", "npu", init device for model initialization. use "meta" or cpu for large model(>30B)                                                                    | cuda |
| train.enable_full_determinism | bool | Whether to enable deterministic mode (for bitwise alignment).                                                                                                                  | False |
| train.empty_cache_steps | int | Number of steps between two cache clearings. -1 means not enabled.                                                                                                             | 500 |
| train.data_parallel_mode | str: {"ddp", "fsdp1", "fsdp2"} | Data parallel algorithm.                                                                                                                                                       | ddp |
| train.tensor_parallel_size | int | Tensor parallel size (currently only supported for vescale training).                                                                                                          | 1 |
| train.pipeline_parallel_size | int | Pipeline parallel size (currently not supported).                                                                                                                              | 1 |
| train.ulysses_parallel_size | int | Ulysses sequence parallel size (currently only supported for P6dense and Qwen2VL).                                                                                             | 1 |
| train.context_parallel_size | int | Ring sequence parallel size (currently not supported)                                                                                                                          | 1 |
| train.expert_parallel_size | int | Expert parallel size (currently only supported DeepseekMOE)                                                                                                                    | 1 |
| train.load_checkpoint_path | str | Path to the omnistore checkpoint for resuming training.                                                                                                                        | None |
| train.save_steps | int | Number of steps between two checkpoint saves. 0 means invalid.                                                                                                                 | 0 |
| train.save_epochs | int | Number of epochs between two checkpoint saves. 0 means invalid.                                                                                                                | 1 |
| train.save_hf_weights | bool | Whether to save the model weights in the huggingface format. It is recommended to set it to False for models > 30B to prevent NCCL timeout. You can convert it after training. | True |
| train.seed | int | Random seed.                                                                                                                                                                   | 42 |
| train.use_wandb | bool | Whether to enable byted wandb experiment logging.                                                                                                                              | True |
| train.wandb_project | str | Name of the wandb experiment project.                                                                                                                                          | VeOmni |
| train.wandb_name | str | Name of the wandb experiment.                                                                                                                                                  | None |
| train.enable_profiling | bool | Whether to use torch profiling.                                                                                                                                                | False |
| train.profile_start_step | int | Starting step of profiling.                                                                                                                                                    | 1 |
| train.profile_end_step | int | Ending step of profiling.                                                                                                                                                      | 2 |
| train.profile_trace_dir | str | Path to save the profiling results.                                                                                                                                            | ./trace |
| train.profile_record_shapes | bool | Whether to record the shapes of the input tensors.                                                                                                                             | True |
| train.profile_profile_memory | bool | Whether to record the memory usage.                                                                                                                                            | True |
| train.profile_with_stack | bool | Whether to record the stack information.                                                                                                                                       | True |
| train.max_steps | int | Number of steps per training epoch (only used for debugging).                                                                                                                  | None |

### Inference configuration arguments
| Name | Type | Description | Default Value |
| --- | --- | --- | --- |
| infer.model_path | str | Path to the model parameter file. | Required |
| infer.tokenizer_path | str | Path to the tokenizer. | model.model_path |
| infer.seed | int | Random seed. | 42 |
| infer.do_sample | bool | Whether to enable sampling. | True |
| infer.temperature | float | Sampling temperature. | 1.0 |
| infer.top_p | float | Sampling Top P value. | 1.0 |
| infer.max_tokens | int | Maximum number of tokens generated each time. | 1024 |
