# Trainer

This document details the Trainer system in VeOmni. While [Basic Modules](./basic_modules.md) introduces the individual components (Dataset, Model, Parallel State, etc.), the `BaseTrainer` orchestrates these components to execute the training loop, handle distributed training complexities, and manage the training lifecycle through callbacks.

## Base Trainer

The [`BaseTrainer`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/base.py) class is the foundation for all training tasks in VeOmni. It handles:

- **Distributed Setup**: Initializes process groups and parallel states (DP, TP, EP, etc.).
- **Component Construction**: Builds a [`VeOmniModelRuntime`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/models/model_runtime.py) (model, freeze/LoRA, parallelize, optimizer, preprocessor) and the dataloaders.
- **Training Loop**: Implements the standard training loop with gradient accumulation.
- **State Management**: Handles checkpointing and resuming training.
- **Extensibility**: Provides hooks and a callback system for customization.

### Core Attributes

- `args`: Global arguments containing model, data, and training configurations.
- `model`: A `VeOmniModelRuntime`. The wrapped `nn.Module` is `trainer.model.model`; module APIs (`config`, `parameters()`, `trainer.model(**batch)`) still forward. Optimizer, lr-scheduler, tokenizer, processor, and `model_assets` live on the runtime (`trainer.model.optimizer`, …).
- `train_dataloader`: The distributed dataloader.
- `state`: The `TrainerState` cursor dispatched to callbacks.

## Training Loop

The core training logic is encapsulated in the `train()` and `train_step()` methods of `BaseTrainer`.

### The `train` Method

The `train()` method is the entry point for training. It:
1. Calls `on_train_begin` callback.
2. Iterates through epochs.
3. Calls `on_epoch_begin`.
4. Iterates through the dataloader.
5. Calls `train_step()` for each batch.
6. Calls `on_epoch_end`.
7. Calls `on_train_end` after the loop finishes.

```python
def train(self):
    self.on_train_begin()

    for epoch in range(self.start_epoch, args.train.num_train_epochs):
        self.on_epoch_begin()

        data_iterator = VeOmniIter(self.train_dataloader, ...)
        for _ in range(self.start_step, args.train_steps):
            self.train_step(data_iterator)

        self.on_epoch_end()

    self.on_train_end()
```

### The `train_step` Method

The `train_step()` method handles a single global training step, including gradient accumulation:

1. **Micro-Batch Iteration**: Iterates over micro-batches (accumulated gradients).
2. **Forward & Backward**: Calls `forward_backward_step()` for each micro-batch.
3. **Gradient Synchronization**: Synchronizes gradients across data parallel ranks.
4. **Gradient Clipping**: Clips gradients to ensure stability.
5. **Optimizer Step**: Updates model parameters.
6. **Scheduler Step**: Updates the learning rate.
7. **Zero Grad**: Clears gradients for the next step.

```python
def train_step(self, data_iterator):
    micro_batches: List[Dict[str, Any]] = next(data_iterator)
    self.on_step_begin(micro_batches=micro_batches)

    for micro_step, micro_batch in enumerate(micro_batches):
        loss, loss_dict = self.forward_backward_step(micro_batch)
        # ... accumulation ...

    grad_norm = self.model.clip_grad_norm()
    self.model.optimizer.step()
    self.model.lr_scheduler.step()
    self.model.optimizer.zero_grad()

    self.on_step_end(loss=..., loss_dict=..., grad_norm=grad_norm)
```

### Forward and Backward

The `forward_backward_step` allows for customization of the forward and backward passes. It includes hooks for pre-processing (`preforward`) and post-processing (`postforward`).

- `preforward`: Moves data to the correct device.
- `postforward`: Computes the final loss from model outputs.

## Callbacks

The Trainer uses a callback system to decouple logging, checkpointing, and evaluation from the core training loop.

### Built-in Callbacks

VeOmni includes several built-in callbacks:

- **[EnvironMeterCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/trace_callback.py)**: Logs system metrics (MFU, FLOPs, memory usage).
- **[TqdmCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/trace_callback.py)**: Displays a progress bar.
- **[WandbTraceCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/trace_callback.py)**: Logs metrics to wandb.
- **[ProfileTraceCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/trace_callback.py)**: Handles profiling.
- **[ChannelLossCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/channel_loss_callback.py)**: Logs detached per-channel causal-LM loss metrics when `train.channel_loss.enable=true`.
- **[ModelDcpCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/checkpoint_callback.py)**: Saves resumable model checkpoints.
- **[ModelHfCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/checkpoint_callback.py)**: Exports HuggingFace / LoRA weights.
- **[GlobalStateCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/global_state_callback.py)**: Saves job-level state (dataloader cursor, rng, meters).
- **[RootAssetsCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/global_state_callback.py)**: Exports the config / tokenizer sidecars.
- **[EvaluateCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/evaluate_callback.py)**: Runs evaluation on the validation set.

### Custom Callbacks

You can create custom callbacks by inheriting from [`Callback`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/base.py) and appending them to `trainer._callbacks` after `_init_callbacks()`.

```python
from veomni.trainer.callbacks import Callback

class MyCustomCallback(Callback):
    def on_step_end(self, state, **kwargs):
        if state.global_step % 100 == 0:
            print(f"Step {state.global_step}: Custom action executed.")

# After BaseTrainer._init_callbacks()
trainer._callbacks.append(MyCustomCallback(trainer))
```

## Customizing the Trainer

To implement a specific training task (like VLM training), compose a `BaseTrainer` and override the step that differs. Model-bound differences belong on a `VeOmniModelRuntime` subclass, not on the trainer. [`VLMTrainer`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/vlm_trainer.py) is the in-tree example.

### Key Methods to Override

1. **`build_model_runtime(self)`**:
   Return the runtime that owns this job's model. Auxiliary components — tokenizer, processor, chat template — are built there rather than on the trainer, so override `VeOmniModelRuntime.build_model_assets` on a runtime subclass if a model needs different ones.
   ```python
   def build_model_runtime(self) -> MyModelRuntime:
       return MyModelRuntime(
           self.args.model, "base", train=self.args.train, chat_template_name=self.args.data.chat_template
       )
   ```

2. **`VeOmniModelRuntime.freeze_model` / `build_optimizer`**:
   Freeze towers or split parameter groups on the runtime. `VLMModelRuntime` freezes ViT / audio and gives visual params a separate `vit_lr`.

3. **`_build_data_transform(self)`**:
   Define how raw data samples are processed into model inputs. Read the preprocessor off the runtime.
   ```python
   def _build_data_transform(self):
       self.data_transform = build_data_transform(
           model_type, processor=self.model.processor, chat_template=self.model.chat_template, ...
       )
   ```

4. **`_build_collate_fn(self)`**:
   Wire extra collate rules. Prefer model hooks (`get_extra_collate_infos`, `get_metadata_collate_func`) over a trainer-side `model_type` switch.

### Extending Arguments

You can also extend the configuration arguments to support your custom trainer settings.

```python
@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(default=False, metadata={"help": "Freeze ViT"})

@dataclass
class Arguments(VeOmniArguments):
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)
    # ...
```

By following this pattern, you can leverage the robust infrastructure of `BaseTrainer` while tailoring the training process to your specific model and data requirements.
