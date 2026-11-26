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
`train.dyn_bsz=true`, and `train.pad_to_length=true`, so the packed token tensors
have stable shapes. `dynamic` is still offered and still defaults to unset — see
[below](#dynamic--leave-it-unset), because a vision tower's patch count is not
padded even when the text side is.

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

## `dynamic` — leave it unset

`dynamic` defaults to `None`, which is torch's own default and the only setting that
works on both counts below. It means: the first call compiles a **static** graph, and
a second distinct shape promotes it to a **symbolic** one that covers the rest of the
run. Measured on a toy block, `[512, 512, 512, 640, 512, 640, 768, 512, 768]`:

| `dynamic` | graphs after each call | meaning |
|---|---|---|
| `False` | 1,1,1,**2**,2,2,**3**,3,3 | one per shape, walks into the cache limit |
| `True` | 1,1,1,1,1,1,1,1,1 | symbolic from the first call |
| `None` | 1,1,1,**2**,2,2,2,2,2 | static, then promoted once |

**Why not `False`.** With `fullgraph=True` a cache miss is a hard error, not a silent
recompile. One new shape per step exhausts `torch._dynamo.config.cache_size_limit`
(8) and raises `FailOnRecompileLimitHit`:

```
dynamic=False, vision block, bf16, fullgraph=True
  token counts 96, 128, ..., 320  -> 8 graphs, OK
  token count  352                -> FailOnRecompileLimitHit
```

`pad_to_length` does not prevent this. It pads only the keys whose `pack_dim` is `-1`
(`input_ids`, `labels`, `position_ids`, `attention_mask`, `image_mask`,
`video_mask`), which fixes the *text* token count. `pixel_values` is concatenated
along dim 0, so a vision tower's patch count follows the images in the batch and moves
every step. The base trainer forcing `pad_to_length` therefore keeps text stable while
leaving the towers variable.

**Why not `True`.** It survives a moving shape, but it pays symbolic tracing on the
first call even when the shape never moves — about 2x the first-call latency, and
nothing steady state. More importantly, symbolic tracing of a custom autograd
`Function` fails when the `Function` chains two Triton kernels:

```python
class Fn(torch.autograd.Function):
    def forward(ctx, x, idx):
        return moe_gather(moe_scatter(x, idx), idx)   # kernel output -> kernel input
```

Under `dynamic=True` Dynamo raises `AssertionError` in
`torch/_dynamo/variables/higher_order_ops.py::overwrite_tensor_vt_proxy`, because the
first kernel's launch scalars (`N`, `num_elts`, `num_elts - 1`) come back as constants
where the outer graph still holds symbols for them. What matters is not the shapes but
the chaining: the same `moe_gather` succeeds under `dynamic=True` when its input is a
graph input or a fresh `torch.empty`, and fails only when that input came from another
kernel in the same `Function`. The fused-Triton MoE experts
(`moe_implementation='fused_triton'`, the deepseek-v4 default) are built exactly that
way, so `deepseek_v4` compiles under `None` and under `False`, and raises under `True`.

**The same limit applies after a `None` promotion.** `None` only postpones it: the
first call compiles a static graph, and the promotion to a symbolic one on a new shape
enters the same path and fails the same way. `deepseek_v4` is unaffected in practice
only because `pad_to_length` keeps its token count constant, so no promotion ever
happens. A run whose shapes move needs a model without such a `Function`.

Set `None` and do not spell it. `False` is only safe when every micro-batch really does
have the same shape; `True` is only safe for a model whose compiled blocks contain no
custom autograd `Function` that chains kernels.

## What stays refused

Two groups, refused in two different ways.

**CUDA Graph replay** — `backend='cudagraphs'`, `mode='reduce-overhead'`,
`mode='max-autotune'`. Banned outright **only under SeedOmni**: replay needs one
`cudagraph_mark_step_begin()` per micro-batch across the composed graph, which
`OmniTrainer` does not make. The single-model trainers allow replay, subject to the
requirement every path has — `fsdp_config.reshard_after_forward` must be `False`,
since replay records the all-gather/reshard sequence. That is
`validate_compile_config_for_fsdp2`; see [the default `mode=None`](#the-default-modenone)
for how to enable it.

**Sequence parallelism** — `ulysses_size>1`, `cp_size>1`, `enable_async`. Ulysses and
context parallelism slice activations the graph edges of a compiled block reorder, and
async Ulysses pipelines the same exchanges across steps. Refused for every trainer:

- **Under SeedOmni** the launcher config is checked when it is parsed, before any
  model is built (`_validate_omni_accelerator`), so a per-module overlay cannot slip
  past it either.
- **Single-model trainers** (text and VLM) check it at build time in
  `validate_compile_runtime`, before the first `fully_shard`. This applies to text
  models as well as VLM ones — no `torch_compile` + sequence-parallel path exists in
  this repo yet, for either, and refusing it uniformly is deliberate rather than a
  claim that one of them works.

Compiling also requires FSDP2 and CUDA; that is checked in the same
`validate_compile_runtime`, not per model. A selection Inductor cannot lower — the
`torchao_mxfp4` rows emit `float8_e8m0fnu` scale tensors, which have no Inductor
lowering — is rejected up front by config field rather than failing at runtime.

## The default `mode=None`

`mode=None` follows TorchTitan's main path: the `inductor` backend without CUDA Graph
replay. Setting `mode="reduce-overhead"` enables CUDA Graphs on `inductor` and
requires `model.accelerator.fsdp_config.reshard_after_forward=False`. When CUDA Graphs
are enabled, each micro-batch calls `torch.compiler.cudagraph_mark_step_begin()` when
available so CUDA Graph Trees can separate iterations.
