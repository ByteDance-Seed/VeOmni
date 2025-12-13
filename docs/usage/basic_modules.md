
# Basic Modules

## Usage
1. **Install VeOmni**  
    Please refer to [Install](quick_start.md) for detailed instructions.

2. **Run Example Script**  
   Verify training startup: (need download the dataset first)

   ```bash
   bash train.sh tasks/train_torch.py configs/pretrain/qwen2_5.yaml
   ```

3. **Create Custom Task Directory**  
    [`train_torch.py`](../../tasks/train_torch.py) can be used for most of task pre-training and post-training tasks, youcan just modify the train config to complete your task. However, if you want to create a new task, you can copy the `train_torch.py` file from the `tasks` directory and modify it. like [`tasks/omni/train_qwen2_vl.py`](../../tasks/omni/train_qwen2_vl.py)
    ```bash
    mkdir tasks/your_task
    cp tasks/train_torch.py tasks/your_task/train.py
    ```

4. **Launch Custom Training**  
    you  can overwrite the default arguments in train yaml by passing them to the script.
    ```bash
    bash train.sh tasks/your_task/train.py \
        $CONFIG.yaml \
        --model.model_path your_path_to_model \
        --data.train_path your_path_to_dataset \
        --train.output_dir your_path_to_save_checkpoints \
        --train.wandb_project your_project_name \
        --train.wandb_name your_experiment_name
    ```

## Arguments
**Default Parameter Access**:  
veomni offers a unified argument management system, which can be easily extended to support custom arguments. About the default arguments explanation, you can refer to the [Config arguments Explanation](../config/config.md)
- source code [veomni/utils/arguments.py](../../veomni/utils/arguments.py).

```python
from dataclasses import dataclass, field
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args

@dataclass
class Arguments:
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
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "CustomTrainingArguments" = field(default_factory=CustomTrainingArguments)
```

## Parallel State
VeOmni use torch device mesh to manage all the parallel state, which is useful and concise when working with multi-dimensional parallelism (i.e. 3-D parallel) where parallelism composability is required. You can create the parallel state by calling the `init_parallel_state` function. and get the parallel state by calling the `get_parallel_state` function.

More details about torch device mesh, you can refer to the [Getting Started with DeviceMesh](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html).

- source code [veomni/distributed/parallel_state.py](../../veomni/distributed/parallel_state.py).

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
VeOmni default supports two types of datasets(source code: [veomni/data/dataset.py](../../veomni/data/dataset.py)):
1. **IterativeDataset** (recommended for large datasets)  
2. **MappingDataset** (default for small datasets)

```python
from veomni.data import build_dataset
train_dataset = build_dataset(
    dataset_name=args.data.dataset_name,
    transform=transform,
    dataloader_batch_size=args.train.dataloader_batch_size,
    seed=args.train.seed,
    **asdict(args.data)
)
```

> **Note**:
>
> args.train.compute_train_steps is used to compute the number of training steps. without this, the train steps will be computed incorrectly.
>
> if you dataset is iterable, you are recommended to add data.train_size(the token you want to comsume) to the config file, the `train_steps` will approximate to `train_size / (global_batch_size * max_seq_len)`(without any warm strategy).
>
> if you dataset is mapping, you are recommended to add pass the len(train_dataset) to the `train_steps` to compute the correct train steps.

### Custom Datasets
VeOmni is a flexible framework that supports custom datasets. You can implement your own dataset function and use it with VeOmni.

```python
def build_custom_dataset(data_path, transform)-> Dataset:
    # Implement your custom dataset logic
    pass

elif args.data.datasets_type == "custom":
    logger.info_rank0("Start building custom dataset")
    train_dataset = build_custom_dataset(args.data.train_path, transform=transform)
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset)) # compute train steps, remove the len(train_dataset) if you dataset is iterable
```

### Data Transform (Preprocess)
VeOmni default supports two types of transform(source code: [veomni/data/data_transform.py](../../veomni/data/data_transform.py)):
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
)
```

### Chat Template
VeOmni default supports few chat template(source code: [veomni/data/chat_template.py](../../veomni/data/chat_template.py)):
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
- both padding and remove padding(packing) strategy
- dynamic batching strategy
-
(source code: [veomni/data/data_loader.py](../../veomni/data/data_loader.py)):

```python
from veomni.data import build_dataloader, build_mapping_dataset

transform = YOUR_TRANSFORM_FUNCTION

train_dataset = build_mapping_dataset(
    data_path=args.data.train_path,
    transform=transform,
)

args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

train_dataloader = build_dataloader(
    dataset=train_dataset,
    micro_batch_size=args.train.micro_batch_size, # micro batch size
    global_batch_size=args.train.global_batch_size, # global batch size
    dataloader_batch_size=args.train.dataloader_batch_size, # dataloader batch size, how many micro batches to get with next(train_dataloader), automatically calculate
    seed=args.train.seed, # random seed
    max_seq_len=args.data.max_seq_len, # max sequence length
    collate_fn=None, # you can pass your custom collate_fn
    train_steps=args.train.train_steps, # train steps, calculate by args.train.compute_train_steps
    rmpad=args.train.rmpad, # remove padding
    rmpad_with_pos_ids=args.train.rmpad_with_pos_ids, # remove padding with position ids
    bsz_warmup_ratio=args.train.bsz_warmup_ratio, # bsz warmup ratio
    bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken, # bsz warmup init micro batch token
    dyn_bsz_margin=args.train.dyn_bsz_margin, # dynamic batching margin
    dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size, # dynamic batching buffer size
    num_workers=args.data.num_workers, # dataloader num workers
    drop_last=args.data.drop_last,  # dataloader drop last
    pin_memory=args.data.pin_memory,  # dataloader pin memory
    prefetch_factor=args.data.prefetch_factor, # dataloader prefetch factor
)
```

### Collate Function
VeOmni default supports three types of collate function for text task(source code: [veomni/data/data_collator.py](../../veomni/data/data_collator.py):
1. `DataCollatorWithPadding` (enable when `rmpad` is False and `rmpad_with_pos_ids` is False)
2. `DataCollatorWithPacking` (enable when `rmpad` is True and `rmpad_with_pos_ids` is False)
3. `DataCollatorWithPositionIDs` (enable when `rmpad` is False and `rmpad_with_pos_ids` is True)

For Omni model task:
1. `OmniDataCollatorWithPacking` (for when `rmpad_with_pos_ids` is True)
2. `OmniDataCollatorWithPadding` (for `rmpad` is False and `rmpad_with_pos_ids` is False)

See detail in source code: [veomni/data/multimodal/data_collator.py](../../veomni/data/multimodal/data_collator.py)) and how to use it in the [train_omni_model.py](../../tasks/omni/train_omni_model.py)


## Model and Optimizer
### Model Initialization
`build_foundation_model` implement the model initialization with the config and weights path.
- meta device init
- init model from model config or weights path

- source code [veomni/models/auto.py](../../veomni/models/auto.py)

```python
from veomni.models import build_foundation_model

model = build_foundation_model(
    config_path=args.model.config_path, # model config path, can be None if weights_path is not None
    weights_path=args.model.model_path, # model weights path, can be None if config_path is not None
    init_device=args.train.init_device, # model init device
)

# You can replace the following code if you want to use the AutoModelForCausalLM from transformers.
# model = AutoModelForCausalLM.from_pretrained(args.model.model_path)
```

### Parallelization your model
```python
from veomni.distributed.torch_parallelize import build_parallelize_model

model = build_foundation_model(...)

model = build_parallelize_model(
    model,
    enable_full_shard=args.train.enable_full_shard, # enable full shard, same to Zero3
    enable_mixed_precision=args.train.enable_mixed_precision, # enable mixed precision
    enable_gradient_checkpointing=args.train.enable_gradient_checkpointing, # enable gradient checkpointing
    init_device=args.train.init_device, # model init device
    enable_fsdp_offload=args.train.enable_fsdp_offload, # enable fsdp offload
    basic_modules=model._no_split_modules + args.model.basic_modules, # FSDP basic modules
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


## Profiler
VeOmni offers a profiler function for users to trace training, use like that.

```python
from veomni.utils import helper

# before train loop, create your profiler
if args.train.global_rank == 0:
    if args.train.enable_profiling:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
        )
        profiler.start()

for epoch in range(args.train.num_train_epochs):
    data_iterator = iter(train_dataloader)
    for _ in range(args.train.train_steps):

        ## train code

        profiler.step()
        if global_step == args.train.profile_end_step:
            profiler.stop()
            # upload file to merlin
            helper.upload_trace(args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir)
```
