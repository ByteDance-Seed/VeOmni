# Qwen4-Exp GDN scalar precision

`A_log` and `dt_bias` in every Qwen4-Exp GDN layer use FP32 storage,
computation and gradients. Other model parameters follow the configured
precision policy. This avoids rounding the source scalar values before the
first forward or before constructing the FP32 optimizer master weights.

The supported pretrained path constructs the model on `meta`, promotes these
parameters before any HF tensors are loaded, and verifies the source dtype,
shape and presence. A BF16 source is rejected: widening a rounded value cannot
recover the original FP32 value. A materialized scratch model is initialized
anew in FP32 before training. Parameter names remain unchanged.

FSDP2 manages each layer's GDN scalar pair separately on the same FSDP mesh as
the rest of that layer. Other parameters retain their existing FSDP, expert,
sequence- and context-parallel layouts. The small FP32 group uses native FP32
gradient communication and does not cast the BF16 inputs to FP32. The current
implementation does not support `torch.compile` for Qwen4-Exp; other model
families are unaffected by that restriction. Collective overhead must be
measured on the intended training geometry.

Full Qwen4 DCP model saves record an inventory of FP32 GDN parameter names.
Both online and offline HF exports verify the complete inventory and retain
these tensors in FP32. Older BF16 model checkpoints require an explicit
migration; restoring them as FP32 would not recover discarded bits. For a
legacy FP32 DCP without a precision inventory, provide model assets or use
`--qwen4-gdn-fp32` with `scripts/merge_dcp_to_hf.py`. Without model assets the
explicit flag checks present pairs but cannot infer a completely missing
layer; pass the matching model config to check layer coverage.

Both `qwen4_exp` and `qwen4_exp_text` configurations are covered. A config
that explicitly declares only QSA layers has no GDN tensors and needs no
special precision group. The current Qwen4 converter does not implement online
`export_weights`; a custom converter exposing it is rejected for GDN models
until its scalar precision and distributed coverage have been validated.

The extra unsharded parameter payload for the 36-by-48 Qwen4-Exp GDN scalar
pairs is 6,912 bytes relative to BF16. This excludes FSDP communication and
allocator overhead, which require hardware measurement.

## Optional HF read-cache advice

Set `VEOMNI_HF_DROP_READ_CACHE=1` to request release of clean source-weight
pages after a successful HF checkpoint load. The default is `0` (disabled).
The ordinary and EP-streaming loaders request advice on each reader; the
rank-0 broadcast loader requests it on rank 0. The request uses read-only
file descriptors and does not change weights, optimizer state or precision.

This is a best-effort OS/filesystem hint: it may be ignored, and rereading
evicted weights can increase I/O. Advice failures are logged and training
continues. It covers the base HF checkpoint shards, not LoRA adapter files
or distributed checkpoint output.
