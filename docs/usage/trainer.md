# Trainer

This document details the Trainer system in VeOmni. While [Basic Modules](./basic_modules.md) introduces the individual components (Dataset, Model, Parallel State, etc.), the `BaseTrainer` orchestrates these components to execute the training loop, handle distributed training complexities, and manage the training lifecycle through callbacks.

## Base Trainer

The [`BaseTrainer`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/base.py) class is the foundation for all training tasks in VeOmni. It handles:

- **Distributed Setup**: Initializes process groups and parallel states (DP, TP, EP, etc.).
- **Component Construction**: Builds the model, optimizer, scheduler, and dataloaders using the configuration.
- **Training Loop**: Implements the standard training loop with gradient accumulation.
- **State Management**: Handles checkpointing and resuming training.
- **Extensibility**: Provides hooks and a callback system for customization.

### Core Attributes

- `args`: Global arguments containing model, data, and training configurations.
- `model`: The parallelized model (wrapped with FSDP/DDP).
- `optimizer` & `lr_scheduler`: The optimizer and learning rate scheduler.
- `train_dataloader`: The distributed dataloader.
- `callbacks`: A handler for managing registered callbacks.

## Training Loop

The core training logic is encapsulated in the `fit()` and `train_step()` methods of `BaseTrainer`.

### The `fit` Method

The `fit()` method is the entry point for training. It:
1. Calls `on_train_begin` callback.
2. Iterates through epochs.
3. Calls `on_epoch_begin`.
4. Iterates through the dataloader.
5. Calls `train_step()` for each batch.
6. Calls `on_epoch_end`.
7. Calls `on_train_end` after the loop finishes.

```python
def fit(self):
    # ... setup ...
    self.callbacks.call("on_train_begin", self.state)

    for epoch in range(self.start_epoch, args.train.num_train_epochs):
        self.callbacks.call("on_epoch_begin", self.state)

        data_iterator = iter(self.train_dataloader)
        for _ in range(self.start_step, args.train_steps):
            self.train_step(data_iterator)

        self.callbacks.call("on_epoch_end", self.state)

    self.callbacks.call("on_train_end", self.state)
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
    # ...
    micro_batches: List[Dict[str, Any]] = next(data_iterator)
    self.callbacks.call("on_step_begin", self.state, micro_batches=micro_batches)

    # Gradient Accumulation Loop
    for micro_step, micro_batch in enumerate(micro_batches):
        loss, loss_dict = self.forward_backward_step(micro_batch)
        # ... accumulation ...

    # Optimization
    grad_norm = veomni_clip_grad_norm(self.model, self.args.train.max_grad_norm)
    self.optimizer.step()
    self.lr_scheduler.step()
    self.optimizer.zero_grad()

    self.callbacks.call("on_step_end", self.state, ...)
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
- **[CheckpointerCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/checkpoint_callback.py)**: Saves training checkpoints.
- **[HuggingfaceCkptCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/checkpoint_callback.py)**: Saves HuggingFace checkpoints.
- **[EvaluateCallback](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/evaluate_callback.py)**: Runs evaluation on the validation set.

### Custom Callbacks

You can create custom callbacks by inheriting from [`Callback`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/base.py) and registering them with `trainer.add_callback`.

```python
from veomni.trainer.callbacks import Callback

class MyCustomCallback(Callback):
    def on_step_end(self, state, **kwargs):
        if state.global_step % 100 == 0:
            print(f"Step {state.global_step}: Custom action executed.")

# In your trainer
trainer.add_callback(MyCustomCallback(trainer))
```

## Customizing the Trainer

To implement a specific training task (like VLM training), you should subclass `BaseTrainer` and override specific methods. The [`VLMTrainer`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/vlm_trainer.py) serves as an excellent example.

### Key Methods to Override

1. **`post_init(self)`**:
   Perform any additional initialization after the base setup.

2. **`build_model_assets(self)`**:
   Initialize and return auxiliary model components like tokenizers, processors, or chat templates.
   ```python
   def build_model_assets(self):
       self.processor = build_processor(self.args.model.tokenizer_path)
       return [self.processor]
   ```

3. **`build_data_transform(self)`**:
   Define how raw data samples are processed into model inputs. This is crucial for multimodal tasks where image/video processing is required.
   ```python
   def build_data_transform(self):
       return partial(process_sample_function, processor=self.processor, ...)
   ```

4. **`build_data_collate_info(self)`**:
   Provide configuration for the data collator, such as which dimensions to pack or pad.
   ```python
   def build_data_collate_info(self):
       return {"input_features": (0, True, 0, 1)} # Example for VLM
   ```

5. **`freeze_module(self)`**:
   Freeze specific parts of the model (e.g., the vision encoder in a VLM).
   ```python
   def freeze_module(self):
       if self.args.train.freeze_vit:
           self.model.visual.requires_grad_(False)
   ```

6. **`build_param_groups(self)`**:
   Define parameter groups for the optimizer, useful for setting different learning rates for different components.
   ```python
   def build_param_groups(self):
       return [
           {"params": vit_params, "lr": self.args.train.vit_lr},
           {"params": other_params, "lr": self.args.train.lr}
       ]
   ```

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
