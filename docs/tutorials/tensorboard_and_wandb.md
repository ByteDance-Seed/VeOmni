# TensorBoard and Wandb

This guide explains how to use TensorBoard and Wandb to log and monitor your training process. Both tools can be enabled simultaneously or used independently.

## TensorBoard

TensorBoard provides local visualization of training metrics and can be enabled by adding the following configuration to your training config file:

```yaml
train:
  output_dir: output
  use_tensorboard: true 
  tensorboard_dir: "./output/tensorboard"
```

After training starts, launch TensorBoard with:

```bash
tensorboard --logdir ./output/tensorboard
```

Then open your browser and navigate to `http://localhost:6006` to view the training metrics.

## Wandb

Wandb provides cloud-based experiment tracking and visualization. To use Wandb, configure it in your training config file:

```yaml
train:
  output_dir: output
  use_wandb: true
  wandb_project: VeOmni
  wandb_name: exp_train_qwen3_vl_moe
```

**Configuration parameters:**
- `wandb_project`: The project name (outer-level path) that groups related experiments together in Wandb. This is like a folder that contains multiple runs.
- `wandb_name`: The experiment name (run name) that identifies a specific training run within the project. Each run in the same project should have a unique name.

Before training, authenticate with Wandb using one of the following methods:

```bash
# Method 1: Interactive login
wandb login

# Method 2: Set API key as environment variable
export WANDB_API_KEY=your_api_key
```

After training starts, you can view the training process in the [Wandb dashboard](https://wandb.ai).

## Using Both TensorBoard and Wandb

You can enable both TensorBoard and Wandb simultaneously to leverage the benefits of both tools:

```yaml
train:
  output_dir: output
  use_tensorboard: true
  tensorboard_dir: "./output/tensorboard"
  use_wandb: true
  wandb_project: VeOmni
  wandb_name: exp_train_qwen3_vl_moe
```

You can also override these settings via command-line arguments:

```bash
bash train.sh tasks/omni/train_qwen3_vl.py configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml \
  --train.use_tensorboard true \
  --train.use_wandb true
```

This allows you to:
- View real-time metrics locally with TensorBoard
- Track experiments in the cloud with Wandb for collaboration and comparison