# HyperParallel

[HyperParallel](https://github.com/mindspore-ai/hyper-parallel) is an optional
acceleration backend for Ascend SuperPoD training, providing FSDP/HSDP and distributed Muon.

## Install dependencies

Set up VeOmni using the Ascend installation guide for your machine:
[ARM](../get_started/installation/install_ascend_arm.md) or
[x86](../get_started/installation/install_ascend_x86.md).

The `npu` and `npu_aarch64` extras include HyperParallel. From the VeOmni
repository root, use the extra for your machine:

```bash
# Ascend x86_64
uv sync --locked --extra npu

# Ascend ARM64
uv sync --locked --extra npu_aarch64
```

For pip, use `pip install -e '.[npu]'` or `pip install -e '.[npu_aarch64]'`
and complete the remaining steps in the installation guide above.
Installing HyperParallel does not enable it; select the backend in your training YAML.

## Configure the backend

Add the following settings to your training YAML to enable Hyper FSDP/HSDP and Hyper Muon:

```yaml
model:
  accelerator:
    fsdp_config:
      fsdp_mode: hyper_fsdp2
      forward_prefetch: true
      hyper_parallel:
        comm_fusion: false  # packing parameter/gradient into buffer
        comm_fusion_vit: null # inherit comm_fusion; set true/false to override the visual tower
        forward_prefetch_depth: 1 # number of following blocks to prefetch
        backward_prefetch_depth: 1 # number of blocks to prefetch in backward execution order
        reduce_comm_interval: 1 # gradient-reduce comm wait interval.
  optimizer:
    type: hyper_muon
```

Hyper Muon is optional and can be enabled independently of the FSDP backend.

Set `comm_fusion_vit: true` to enable communication fusion for the visual tower
independently. The default `null` inherits `comm_fusion`.

## Start training

Use VeOmni's standard training commands. See the [Qwen3 training guide](../examples/qwen3.md).
