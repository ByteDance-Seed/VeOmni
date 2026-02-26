# Basic Modules

## Usage
1. **Install VeOmni**  
    Please refer to [Install](../get_started/installation/install.md) for detailed instructions.

2. **Run Example Script**  
   Verify training startup: (need download the dataset first)

    - Use plain python scripts:
        ```bash
        bash train.sh tasks/deprecated_task/train_torch.py configs/text/qwen2_5.yaml
        ```
    - Use trainer:
        ```bash
        bash train.sh tasks/train_text.py configs/text/qwen2_5.yaml
        ```

3. **Create Custom Task Directory**  
    [`train_text.py`](https://github.com/ByteDance-Seed/VeOmni/blob/main/tasks/train_text.py) can be used for most of task pre-training and post-training tasks, you can just modify the train config to complete your task. However, if you want to create a new task, you can copy the `train_text.py` file from the `tasks` directory and modify it. like [`tasks/train_vlm.py`](https://github.com/ByteDance-Seed/VeOmni/blob/main/tasks/train_vlm.py)

4. **Launch Custom Training**  
    You can overwrite the default arguments in train yaml by passing them to the script.
    ```bash
    bash train.sh tasks/your_train_script.py \
        $CONFIG.yaml \
        --model.model_path your_path_to_model \
        --data.train_path your_path_to_dataset \
        --train.output_dir your_path_to_save_checkpoints \
        --train.wandb_project your_project_name \
        --train.wandb_name your_experiment_name
    ```

## Arguments
**Default Parameter Access**:  
VeOmni offers a unified argument management system, which can be easily extended to support custom arguments. About the default arguments explanation, you can refer to the [Config arguments Explanation](arguments.md). A basic argument example is defined in [`arguments_types.py`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/arguments/arguments_types.py).

```python
from dataclasses import dataclass, field
from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, VeOmniArguments

@dataclass
class Arguments(VeOmniArguments):
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)

if __name__ == "__main__":
    args = parse_args(Arguments)
    print(args.train.lr)  # Access default arguments
```

**Custom Parameter Extension**:  
you can extend the default arguments by creating a new class that inherits from the existing class.
```python
@dataclass
class CustomTrainingArguments(TrainingArguments):
    enable_xxx: bool = field(
        default=False,
        metadata={"help": "Enable me if necessary."},
    )

@dataclass
class Arguments(VeOmniArguments):
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "CustomTrainingArguments" = field(default_factory=CustomTrainingArguments)
```

## Parallel State
VeOmni use torch device mesh to manage all the parallel state, which is useful and concise when working with multi-dimensional parallelism (i.e. 3-D parallel) where parallelism composability is required. You can create the parallel state by calling the `init_parallel_state` function. and get the parallel state by calling the `get_parallel_state` function.

More details about torch device mesh, you can refer to the [Getting Started with DeviceMesh](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html).

- source code [veomni/distributed/parallel_state.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/distributed/parallel_state.py).

```python
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state

init_parallel_state(
    dp_size=args.train.data_parallel_size, # data parallel size
    dp_replicate_size=args.train.data_parallel_replicate_size, # data parallel replicate size
    dp_shard_size=args.train.data_parallel_shard_size, # data parallel shard degree
    tp_size=args.train.tensor_parallel_size, # tensor parallel size
    ep_size=args.train.expert_parallel_size, # expert parallel size
    pp_size=args.train.pipeline_parallel_size, # pipeline parallel size, not support now
    cp_size=args.train.context_parallel_size, # context parallel size, not support now
    ulysses_size=args.train.ulysses_parallel_size, # ulysses parallel size
    dp_mode=args.train.data_parallel_mode, # data parallel mode, can be "ddp", "fsdp1", "fsdp2"
    async_enabled=self.args.train.async_enabled, # async ulysses
)

parallel_state = get_parallel_state()

# Access dp state
dp_mesh = parallel_state.dp_mesh
dp_group = parallel_state.dp_group

# Access sp state
sp_group = parallel_state.sp_group
sp_rank = parallel_state.sp_rank

# Access tp state
tp_group = parallel_state.tp_group
tp_mesh = parallel_state.tp_mesh
```

## Dataset
VeOmni default supports three types of datasets(source code: [veomni/data/dataset.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/dataset.py)):
1. **IterativeDataset** (recommended for large datasets)  
2. **MappingDataset** (default for small datasets)
3. **InterleaveDataset** (InterleavedMappingDataset | InterleavedIterableDataset)

```python
from veomni.data import build_dataset
train_dataset = build_dataset(
    dataset_name=args.data.dataset_name,
    transform=transform,
    seed=args.train.seed,
    **asdict(args.data)
)
```

### Interleave Dataset
To use interleave dataset, pass a `dataset.yaml` file path to the `data.train_path` argument. The argument management system will parse the file and build the interleave dataset based on `data.datasets_type`.

An example of `dataset.yaml` file: [configs/multimodal/data/tulu_sharegpt4v_llavavideo_voiceassistant.yaml](https://github.com/ByteDance-Seed/VeOmni/blob/main/configs/multimodal/data/tulu_sharegpt4v_llavavideo_voiceassistant.yaml).

Example usage:

1. InterleavedMappingDataset
```bash
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml \
    --data.train_path configs/multimodal/data/tulu_sharegpt4v_llavavideo_voiceassistant.yaml \
    --data.datasets_type mapping \
```

2. InterleavedIterableDataset
```bash
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml \
    --data.train_path configs/multimodal/data/tulu_sharegpt4v_llavavideo_voiceassistant.yaml \
    --data.datasets_type iterable \
```

### Train Steps Computation

`args.compute_train_steps` is used to compute the number of training steps. without this, the train steps will be computed incorrectly.

If your dataset is iterable, you are recommended to add data.train_size (the token you want to consume) or data.train_sample (the sample you want to consume) to the config file, the `train_steps` will approximate to

1. when dyn_bsz enabled:
`train_size / (global_batch_size * max_seq_len)` (without any warm strategy).

2. when dyn_bsz disabled:
`train_sample / dataloader_batch_size` (without any warm strategy).

If your dataset is mapping, you are recommended to pass `len(train_dataset)` to the `train_steps` to compute the correct train steps.

```python
dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
if args.data.datasets_type == "mapping":
    dataset_length = dataset_length / args.train.data_parallel_size
args.compute_train_steps(dataset_length)
train_steps = args.train_steps
```

### Custom Datasets
VeOmni is a flexible framework that supports custom datasets. You can implement your own dataset function and use it with VeOmni by registering it to the dataset registry.

Examples in [veomni/data/dataset.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/dataset.py).
```python
@DATASET_REGISTRY.register("custom")
def build_custom_dataset(
    train_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
    source_name: Optional[str] = None,
    **kwargs,
)-> Dataset:
    # Implement your custom dataset logic
    pass
```

### Data Transform (Preprocess)

#### Text Transform
VeOmni default supports two types of transform(source code: [veomni/data/data_transform.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/data_transform.py)):
1. **process_pretrain_example** (recommended for pretrain task)
2. **process_sft_example** (recommended for sft task)

**Pretrain Example**:  
```python
from functools import partial
from veomni.data.data_transform import process_pretrain_example
from veomni.models import build_tokenizer

tokenizer = build_tokenizer(args.model.tokenizer_path)
# Can replace with the following code if you want to use the AutoTokenizer from transformers.
# tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer_path)

transform = partial(
    process_pretrain_example,
    tokenizer=tokenizer,
    max_seq_len=args.data.max_seq_len,
    text_keys=args.data.text_keys,
)
```

**SFT Example**:  
```python
from veomni.data.chat_template import build_chat_template

chat_template = build_chat_template(args.data.chat_template, tokenizer)
transform = partial(
    process_sft_example,
    chat_template=chat_template,
    max_seq_len=args.data.max_seq_len,
    text_keys=args.data.text_keys,
)
```

#### Multimodal Transform
VeOmni offers several multimodal transform functions (source code: [veomni/data/multimodal/data_transform.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/multimodal/data_transform.py)):
1. **process_sample_qwen2_5_vl** (process function for Qwen2VL & Qwen2.5VL)
2. **process_sample_qwen3_vl** (process function for Qwen3VL-MoE & Qwen3VL-dense)
3. **process_sample_qwen_omni** (process function for Qwen2.5Omni & Qwen3Omni-MoE)

Example usage in `def build_data_transform` in [veomni/trainer/vlm_trainer.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/vlm_trainer.py).
```python
from veomni.models import build_processor
from veomni.data import build_multimodal_chat_template
processor = build_processor(args.model.tokenizer_path)
chat_template = build_multimodal_chat_template(args.data.chat_template, processor.tokenizer)
position_id_func = model.get_position_id_func()
transform = partial(
    process_function,
    processor=processor,
    chat_template=chat_template,
    position_id_func=position_id_func,
    **args.data.mm_configs,
)
```

The position_id_func is used to generate the position ids for the input sequence. For example, `get_rope_index` in qwen-vl series.

Multimodal dataset transform follows the similar pipeline:
1. conversation preprocess (build the same data structure from different data format) [preprocess.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/multimodal/preprocess.py)
2. build conversation with multimodal sequence (pad special tokens manually or using chat_template | processor)
3. tokenize the input sequence
4. add position_ids and multimodal mask


### Chat Template
VeOmni default supports several chat template(source code: [veomni/data/chat_template.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/chat_template.py) for text-only model and [veomni/data/multimodal/multimodal_chat_template.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/multimodal/multimodal_chat_template.py) for multimodal model):
you can add your custom chat template by implementing the `ChatTemplate` class.
**Custom Template Implementation**:  
```python
from veomni.data.chat_template import ChatTemplate

class CustomTemplate(ChatTemplate):
    def encode_messages(self, messages: Sequence[Dict[str, str]], max_seq_len: int = 8192) -> Dict[str, List[int]]:
        # Implement encoding logic
        pass

    def get_jinja_template(self) -> str:
        return ""  # Jinja template string
```


## DataLoader
VeOmni offered a flexible and powerful dataloader implementation, which supports
- remove padding (packing) strategy
- dynamic batching strategy

(source code: [veomni/data/data_loader.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/data_loader.py)):

```python
from veomni.data import build_dataloader
train_dataloader = build_dataloader(
    dataloader_type=args.data.dataloader_type,
    dataset=train_dataset,
    micro_batch_size=args.train.micro_batch_size, # micro batch size
    global_batch_size=args.train.global_batch_size, # global batch size
    dataloader_batch_size=args.train.dataloader_batch_size, # dataloader batch size, how many micro batches to get with next(train_dataloader), automatically calculate
    max_seq_len=args.data.max_seq_len, # max sequence length
    train_steps=args.train.train_steps, # train steps, calculate by args.train.compute_train_steps
    dyn_bsz=args.train.dyn_bsz, # enable dynamic batching
    bsz_warmup_ratio=args.train.bsz_warmup_ratio, # bsz warmup ratio
    bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken, # bsz warmup init micro batch token
    dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size, # dynamic batching buffer size
    num_workers=args.data.num_workers, # dataloader num workers
    drop_last=args.data.drop_last,  # dataloader drop last
    pin_memory=args.data.pin_memory,  # dataloader pin memory
    prefetch_factor=args.data.prefetch_factor, # dataloader prefetch factor
    seed=args.train.seed, # random seed
    build_collate_fn=True,
    collate_fn_kwargs=collate_fn_kwargs, # kwargs for collate_fn
)
```

### Collate Function
VeOmni default supports a unified collate function for all tasks (text task, multimodal task, omni task, etc.) (source code: [veomni/data/data_collator.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/data_collator.py)). The `MainCollator` handles: packing sequences, precompute position_ids & cu_seqlens & max_seqlens, and sequence parallel slice.

Users can pass `collate_fn_kwargs` to control the behavior of the collate function.

1. DataCollateInfo
The `DataCollateInfo` is defined as:
- pack_dim: Dim to pack in batch. Default is 0. If -1, pack in last dim and unsqueeze(0)
> `input_ids` is -1, and `pixel_values` is 0.
- sp_slice: Whether to do sp slice when sp_enabled. Default is False.
> `input_ids` is true, and `image_mask` is false, as we need the full sequence of `image_mask` on each sp rank.
- sp_pad_value: sp_pad value of a sequence in batch. Not pad if None. Default is None.
> `labels` is -100, and `image_mask` is 0.
- sp_pad_scale: sp_pad scale of a sequence in batch. Default is 1.
> `pixel_values` is merge_size ** 2 for qwen-vl-series.
2. seq_classification
If seq_classification is True, the collate function will not shift and mask the labels during packing and sp_slice.
3. pad_to_length
If pad_to_length is True, the collate function will pad the desired sequences to the max_seq_len. Default is False. More details in [data_packing_and_dyn_bsz.md](./data_packing_and_dyn_bsz.md).

An example of usage in `def build_data_collate_info` in [veomni/trainer/vlm_trainer.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/vlm_trainer.py) for training qwen-omni-models.


## Model and Optimizer
### Model Initialization
`build_foundation_model` implement the model initialization with the config and weights path.
- meta device init
- init model from model config or weights path

- source code [veomni/models/auto.py](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/models/auto.py)

```python
from veomni.models import build_foundation_model

model = build_foundation_model(
    config_path=args.model.config_path, # model config path, can be None if weights_path is not None
    weights_path=args.model.model_path, # model weights path, can be None if config_path is not None
    init_device=args.train.init_device, # model init device
    torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
    attn_implementation=args.model.attn_implementation,
    moe_implementation=args.model.moe_implementation,
    config_kwargs=config_kwargs,
)

# You can replace the following code if you want to use the AutoModelForCausalLM from transformers.
# model = AutoModelForCausalLM.from_pretrained(args.model.model_path)
```

### Parallelization your model
```python
from veomni.distributed.torch_parallelize import build_parallelize_model
model = build_parallelize_model(
    model,
    init_device=args.train.init_device, # model init device
    weights_path=args.model.model_path,
    enable_full_shard=args.train.enable_full_shard, # enable full shard, same to Zero3
    enable_reshard_after_forward=args.train.enable_reshard_after_forward, # enable reshard after forward for FSDP2
    enable_mixed_precision=args.train.enable_mixed_precision, # enable mixed precision
    enable_gradient_checkpointing=args.train.enable_gradient_checkpointing, # enable gradient checkpointing
    enable_fsdp_offload=args.train.enable_fsdp_offload, # enable fsdp offload
    basic_modules=list(set(getattr(model, "_no_split_modules", None) or []) | set(args.model.basic_modules)), # FSDP basic modules
    enable_reentrant=args.train.enable_reentrant,
    enable_forward_prefetch=args.train.enable_forward_prefetch,
)
```

### Optimizer and LR Scheduler
```python
from veomni.optim import build_lr_scheduler, build_optimizer

optimizer = build_optimizer(
    model,
    lr=args.train.lr,
    weight_decay=args.train.weight_decay,
    # ... other parameters
)

lr_scheduler = build_lr_scheduler(
    optimizer,
    train_steps=args.train.train_steps * args.train.num_train_epochs,
    # ... other parameters
)
```


## Train Loop
After the parallel_state, model, optimizer, and dataloader are initialized, you can start the training loop.

```python
for epoch in range(args.train.num_train_epochs):
    data_iterator = iter(train_dataloader)
    for _ in range(args.train.train_steps):
        micro_batches = next(data_iterator)
        for micro_batch in micro_batches:
            loss = model(**micro_batch).loss / len(micro_batches)
            loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```


### Custom Loss Function
```python
import torch

loss_fct = torch.nn.CrossEntropyLoss()

def loss_func(logits, labels):
    return loss_fct(logits, labels)

# In train loop:
output = model(**micro_batch)
logits = output.logits
loss = loss_func(logits, labels) / len(micro_batches)
```
