# HyperParallel

[HyperParallel](https://github.com/mindspore-ai/hyper-parallel) is an optional
acceleration backend for Ascend SuperPoD training, providing FSDP/HSDP and distributed Muon.

## Install dependencies

Set up VeOmni using the Ascend installation guide for your machine:
[ARM](../get_started/installation/install_ascend_arm.md) or
[x86](../get_started/installation/install_ascend_x86.md).

Clone HyperParallel and install it in the same Python environment as VeOmni:

```bash
git clone https://github.com/mindspore-ai/hyper-parallel.git
cd hyper-parallel
python -m pip install -e .
```

## Configure the backend

Add the following settings to your training YAML to enable Hyper FSDP/HSDP and Hyper Muon:

```yaml
model:
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
      fsdp_backend: hyper_parallel
      forward_prefetch: true
      hyper_parallel:
        comm_fusion: false  # packing parameter/gradient into buffer
        comm_fusion_zero_copy: null # copy parameters into flat-buffer, saving allgather copy-in cost when comm_fusion
        wrap_policy: block # fully_shard wrap granularity
        forward_prefetch_depth: 2 # prefetch deptch
        reduce_comm_interval: 1 # gradient-reduce comm wait interval.
  optimizer:
    type: muon
    use_hyper_optimizer: true
```

Hyper Muon is optional and can be enabled independently of the FSDP backend.

## Start training

Use VeOmni's standard training commands. See the [Qwen3 training guide](../examples/qwen3.md).
