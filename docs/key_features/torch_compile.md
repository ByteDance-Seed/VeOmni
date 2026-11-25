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

Compilation happens per module before the FSDP wrap, under `fsdp_scope='module'`. With
`fsdp_scope='model'` the wrap belongs to the composed `OmniModel` instead, so no module
can compile: a module cannot wrap itself inside a parent that owns the wrap, and the
parent's `_no_split_modules` are its children. That scope is rejected at parse time.

## `dynamic=True` (the default)

`dynamic` lets one Inductor graph cover a range of shapes, and it defaults to `True`.
That is not a performance preference — with `fullgraph=True`, a *cache miss* is a hard
error rather than a silent recompile. A static graph that meets one new shape per step
walks into `torch._dynamo.config.cache_size_limit` (8) and raises
`FailOnRecompileLimitHit`:

```
dynamic=False, vision block, bf16, fullgraph=True
  token counts 96, 128, ..., 320  -> 8 graphs, OK
  token count  352                -> FailOnRecompileLimitHit
```

`dynamic=True` costs one compile up front — about 2x the first-call latency — and
nothing steady state: it emits one symbolic graph and reuses it for the rest of the run.

`pad_to_length` does not make this unnecessary. It only pads the keys whose
`pack_dim` is `-1` (`input_ids`, `labels`, `position_ids`, `attention_mask`,
`image_mask`, `video_mask`), which fixes the *text* token count. `pixel_values` is
concatenated along dim 0, so a vision tower's patch count follows the images in the
batch and moves every step. The base trainer forcing `pad_to_length` therefore keeps
text stable while leaving the towers variable.

Set `dynamic=False` only when every micro-batch really does have the same shape. On a
shape that moves it will exhaust the cache as above.

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
