# Byted Loader Backend

VeOmni can use `bytedance.dataloader` as an alternative training dataloader backend through
`data.dataloader.type=byted_loader`.

This backend keeps VeOmni's transform and `MainCollator` boundary, but lets byted loader own storage reads, source
scheduling, shuffle, global-batch construction, and dataloader progress checkpoints.

## Requirements

- Install `bytedance.dataloader>=0.1.42` in the training image.
- Use `mode=local`. Cluster/replay/remote CPU modes are not wired through this backend yet.
- Prefer Lance tables for training data. JSONL is accepted by the config path; Parquet is a restricted compatibility path.
- Use `pp_size=1` and `tp_size=1`. Sequence-parallel slicing remains owned by VeOmni `MainCollator`.

## Example

```bash
torchrun ... tasks/train_text.py config.yaml \
  --data.dataloader.type byted_loader \
  --data.dataloader.file_type lance \
  --data.dataloader.shuffle_algo no_shuffle \
  --data.dataloader.ckpt_dir /path/to/byted_loader_progress \
  --data.dataloader.enable_batch_db_save false \
  --data.dataloader.do_sp_split_in_loader false \
  --data.dataloader.gpu_prefetch false
```

When `train_path` is a multi-source YAML, VeOmni disables its legacy multisource meter in byted-loader mode and preserves
the original YAML path in `BytedLoaderDatasetSpec.source_config_path`. Source-mixing metrics should be read from byted
loader's own metrics.

## Supported V1 Boundary

- Supported: local mode, Lance, JSONL, VeOmni transform, VeOmni `MainCollator`, progress checkpoint/snapshot, CPU tensor
  path, FSDP/SP with SP slicing in `MainCollator`.
- Restricted: Parquet requires `worker_subprocess_num=1` and `worker_parallel_read_num=1` and should be used only for
  compatibility smoke tests.
- Disabled by default: batch DB persistence, GPU prefetch, byted loader SP split, sample/microbatch balance.
- Not supported by this integration yet: PP/TP, remote CPU, cluster mode, replay mode, and default-on balance.
- Local mode can have higher first-batch latency than native dataloaders because it starts byted loader local roles and
  service-discovery/RPC state. This integration starts local roles at client construction by default, but it does not
  merge or change byted loader's internal role processes.

## Resume

The adapter implements `state_dict()` / `load_state_dict()` so VeOmni's checkpoint callback can save and restore the
byted loader client state alongside trainer state. For production validation, compare batch fingerprints between an
uninterrupted run and a save/resume run; matching step counters alone are not enough.

## Notes

The real `bytedance.dataloader` package is imported lazily only after `data.dataloader.type=byted_loader` is selected.
Native/streaming paths do not require the wheel.
