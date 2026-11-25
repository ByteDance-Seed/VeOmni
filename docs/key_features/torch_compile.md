# Per-Block torch.compile

> TL;DR: `torch.compile` wraps individual decoder blocks, not the whole model, so
> FSDP2 keeps its all-gather/reshard hooks outside the compiled region. Turn it on
> with `model.accelerator.torch_compile.enable: true`; under SeedOmni it resolves
> per module, so one `accelerator:` overlay can compile one module and not another.

## Background

Compiling a whole model is usually the wrong granularity. FSDP2 wraps decoder blocks
one at a time, so the compilation unit that preserves module identity — and therefore
keeps all-gather/reshard hooks outside the compiled region — is the block. Wrapping
`module.forward` per block is also what lets a SeedOmni run mix compiled and eager
modules.

## What gets compiled

Every block named in `_no_split_modules` — the same list FSDP wraps. Compiling it
verbatim keeps the compiled set and the sharded set identical, so turning
`torch_compile.enable` on compiles the whole model rather than a hand-picked part
of it. For a multimodal model that includes the vision tower and any audio
encoder, because those are wrap units too.

Under SeedOmni the list is scoped per child (`janus_llama.LlamaDecoderLayer`, see
`_scoped_no_split_modules`), so each entry matches only inside its own module; the
unit of compilation there is still whatever that module declares.

## Two trainers, two ways to enable it

### Base trainer (single model)

Set the accelerator field and run:

```yaml
model:
  accelerator:
    torch_compile:
      enable: true
```

Every decoder block in the model is compiled. Requirements: FSDP2 on CUDA,
`train.dyn_bsz=true`, and `train.pad_to_length=true`, so packed token tensors have
stable shapes — which is also why `dynamic` is not offered here: with every batch
padded to the same length there is no shape variation to absorb.

### SeedOmni trainer (composed model)

The same field, plus the per-module resolution, because a composed run builds each
module separately and applies each one's own `accelerator:` overlay:

```yaml
model:
  accelerator:
    torch_compile:
      enable: true          # the default every module starts from
```

Two conditions decide per module:

1. its own resolved `accelerator.torch_compile.enable` — a per-module overlay can
   flip it either way, including turning it on when the launcher default is off;
2. `fsdp_mode: fsdp2` for that module — compiling does not run under DDP.

A module failing either trains eagerly while its siblings compile.

Compilation happens per module before the FSDP wrap, under either `fsdp_scope`. With
`fsdp_scope='model'` the wrap belongs to the composed `OmniModel`, so the launcher's
`fsdp_mode` is consulted as well — a `fsdp2` overlay cannot make a `ddp` wrap able to
compile. Under `fsdp_scope='module'` the module's own overlay *is* its wrap.

## `dynamic=True`

`dynamic` lets one Inductor graph cover a range of shapes. Whether it is needed
depends on whether the packed length is stable — not on which trainer you use:

- **The base trainer forces `pad_to_length`**, so every micro-batch is padded to
  the same length and `dynamic` gains nothing. Leave it off.
- **SeedOmni's collator does not pad** (`SeedOmniCollator`: "no stacking, no
  padding"), so a text backbone's packed length changes every step and a static
  compile would recompile per new length. Both trainers expose `dyn_bsz`, but in
  SeedOmni it only groups samples by token count — it does not pad.

Turn it on for the modules whose input shape actually moves, and leave it off where
the shape is fixed by a processor (an image tower, a codec): Inductor emits
shape-polymorphic kernels — a runtime stride instead of a constant, a predicate on
every load, some fusions broken, one autotune choice for the whole symbolic range —
which costs a fixed amount per call and can exceed the compile win on a kernel-heavy
module.

## What stays refused

CUDA Graph replay and sequence parallelism, under SeedOmni: replay needs one
`cudagraph_mark_step_begin()` per micro-batch across the composed graph, which
`OmniTrainer` does not make, and Ulysses slices activations the graph edges reorder.
Those are `backend='cudagraphs'`, `mode='reduce-overhead'`, `mode='max-autotune'`,
`ulysses_size>1`, `cp_size>1`, `enable_async`, and they are rejected when the
launcher config is parsed, before any model is built.

Compiling also requires FSDP2 and CUDA; that is checked once per build in
`validate_compile_runtime`, not per model. A selection Inductor cannot lower — the
`torchao_mxfp4` rows emit `float8_e8m0fnu` scale tensors, which have no Inductor
lowering — is rejected up front by config field rather than failing at runtime.

## The default `mode=None`

`mode=None` follows TorchTitan's main path: the `inductor` backend without CUDA Graph
replay. Setting `mode="reduce-overhead"` enables CUDA Graphs on `inductor` and
requires `model.accelerator.fsdp_config.reshard_after_forward=False`. When CUDA Graphs
are enabled, each micro-batch calls `torch.compiler.cudagraph_mark_step_begin()` when
available so CUDA Graph Trees can separate iterations.
