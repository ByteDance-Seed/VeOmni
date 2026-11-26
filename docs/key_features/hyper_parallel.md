# HyperParallel

[HyperParallel](https://github.com/mindspore-ai/hyper-parallel) is an optional
acceleration backend for Ascend SuperPoD training, providing FSDP/HSDP and distributed Muon.
This integration supports the Text and VLM trainers.

## Install dependencies

Set up VeOmni using the Ascend installation guide for your machine:
[ARM](../get_started/installation/install_ascend_arm.md) or
[x86](../get_started/installation/install_ascend_x86.md).

HyperParallel is installed through the optional `hyper` extra. From the VeOmni
repository root, combine it with the Ascend extra for your machine:

```bash
# Ascend x86_64
uv sync --locked --extra npu --extra hyper

# Ascend ARM64
uv sync --locked --extra npu_aarch64 --extra hyper
```

pip does not read `tool.uv.sources`, so install the pinned HyperParallel revision
explicitly alongside VeOmni:

```bash
pip install -e '.[npu_aarch64,hyper]' 'hyper-parallel @ git+https://github.com/mindspore-ai/hyper-parallel.git@f570046629fa7f43428861cba0e54996e1ce6a3d'
```

Use `.[npu,hyper]` instead on x86_64, and complete the remaining steps in the
installation guide above.
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

FSDP and optimizer selection are independent:

- `fsdp_mode: hyper_fsdp2` supports AdamW (`adamw`), VeOmni Muon (`muon`), and Hyper Muon (`hyper_muon`), but not `anyprecision_adamw`.
- `optimizer.type: hyper_muon` works with either Torch FSDP2 or Hyper FSDP.

`comm_fusion_vit` controls visual-tower communication fusion. The default `null`
uses the value of `comm_fusion`; set `true` or `false` to override it.

Normal trainer teardown removes the Hyper patch.

## Adapter patch targets

`hyper_parallel.integration.veomni` uses the following VeOmni bindings and helpers:

| Binding / helper | Purpose |
| --- | --- |
| `veomni.distributed.torch_parallelize.{parallelize_model_fsdp2, fully_shard, FSDPModule}` | Hyper FSDP construction. |
| `veomni.optim.optimizer.build_optimizer` | Optimizer construction. |
| `veomni.distributed.fsdp2.clip_grad_norm.clip_grad_norm` | VeOmni gradient clipping. |
| `veomni.models.checkpoint.weights.{_dispatch_parameter, _find_submodule}` | Pretrained weight loading. |
| `veomni.checkpoint.dcp_checkpointer.{OptimizerState, get_model_state_dict, set_model_state_dict}` | Model and optimizer checkpoints. |
| `veomni.lora.state_dict.get_lora_state_dict` | LoRA adapter export. |
| `veomni.models.checkpoint.convert.export_weights` | HF weight export. |
| `veomni.distributed.emb_parallel.sharded_embedding.register_fsdp_forward_method` | Embedding forward hooks. |
| `veomni.trainer.vlm_trainer._get_vlm_visual_module` | Resolve the visual tower (called, not replaced). |
| `veomni.optim.optimizer.{_collect_muon_kwargs, _make_param_groups_for_subset, _should_build_extra_parallel_aware, _is_extra_parallel_param}` | Optimizer options and parameter groups (called, not replaced). |
| `veomni.distributed.torch_parallelize.sort_fqn_by_submodule_first` | Parameter ordering (called, not replaced). |
| `veomni.lora.state_dict.strip_adapter_name` | LoRA export names (called, not replaced). |
| `veomni.utils.dtensor_utils.rewrap_dtensor_local` | Native Muon tensor views (called, not replaced). |

## Start training

Use VeOmni's standard training commands. See the [Qwen3 training guide](../examples/qwen3.md).
