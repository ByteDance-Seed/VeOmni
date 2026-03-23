# Support New DiT Models — Guide and Reference

> This guide covers integrating a **diffusion transformer (DiT)** model into VeOmni.
> DiT models (e.g. Wan, Flux) come from **diffusers**, not transformers, so they
> follow a different integration pattern from the LLM/VLM guide.

For reference, the complete Wan2.1 I2V integration lives in:
- `veomni/models/diffusers/wan_t2v/wan_condition/` — condition model
- `veomni/models/diffusers/wan_t2v/wan_transformer/` — transformer model

---

## Architecture Overview

DiT training in VeOmni is split into **two independent models**, each registered
separately in VeOmni's model registry:

```
┌─────────────────────────────────────────────────────────────────┐
│  Condition Model  (frozen, not parallelized)                     │
│  ─ encodes raw video/image into latents                          │
│  ─ encodes text prompts into embeddings                          │
│  ─ samples noise, timesteps, and builds the training targets     │
│  ─ loaded from the same checkpoint directory as the DiT          │
└─────────────────────┬───────────────────────────────────────────┘
                      │  get_condition() + process_condition()
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Transformer Model  (trainable, FSDP + SP-parallelized)          │
│  ─ the core DiT backbone loaded from diffusers                   │
│  ─ wrapped in a PreTrainedModel shell for FSDP/checkpoint        │
│  ─ forward() computes loss from conditioned inputs               │
└─────────────────────────────────────────────────────────────────┘
```

The **trainer** (`DiTTrainer`) builds both models independently.
Only the transformer model goes through FSDP / Ulysses SP parallelization.

The two models are linked by the transformer config's `condition_model_type`
class attribute (a string registry key), which tells `DiTTrainer` which
condition model class to instantiate.

---

## Diffusers ↔ Transformers Compatibility

DiT models live in diffusers but VeOmni's infrastructure (registry, checkpointing,
FSDP, SP) is built for `transformers.PreTrainedModel`.
The compatibility goal is strict: **load any standard diffusers checkpoint without
conversion, and save checkpoints that a diffusers pipeline can load directly for
inference without any VeOmni code**.

### The Two Representations

At runtime VeOmni uses a **VeOmni config** (a `PretrainedConfig` subclass) that
holds both the diffusers model parameters and VeOmni-specific fields such as
`model_type`, `condition_model_type`, and `_attn_implementation`.  
On disk, only a **pure diffusers config** is ever written — those VeOmni fields
are never serialized to `config.json`.

```
On disk (diffusers format)          In memory (VeOmni)
─────────────────────────           ──────────────────────────────────────────
config.json                         WanTransformer3DModelConfig
  _class_name: WanTransformer3DModel   ├── all diffusers params (num_layers, ...)
  _diffusers_version: 0.x.x            ├── model_type: "WanTransformer3DModel"
  num_layers: 40                        ├── condition_model_type: "WanTransformer3DConditionModel"
  ...                                   └── _attn_implementation: "flash_attention_2"
model.safetensors
  (identical key names)
```

### Loading a Diffusers Checkpoint

`from_pretrained` delegates unconditionally to the **base diffusers class**:

```python
@classmethod
def from_pretrained(cls, path, **kwargs):
    return _WanTransformer3DModel.from_pretrained(path, **kwargs)
```

This means VeOmni reads the checkpoint exactly as diffusers would — standard
`config.json` format, standard `model.safetensors` weight keys. No adapter,
no conversion, no VeOmni-specific JSON fields required in the checkpoint.

After loading, `_WanTransformer3DModel.__init__` has already populated all
layer weights. VeOmni's `__init__` then wraps the result in a `PreTrainedModel`
shell and, if the SP attention implementation is configured, installs the
`WanSPAttnProcessor`. The weights are not touched.

### Saving a Checkpoint Compatible with Diffusers Inference

`save_pretrained` works in three steps:

```python
def save_pretrained(self, path, **kwargs):
    hf_config = copy.deepcopy(self.config)      # 1. stash VeOmni config
    self.config = self.config.to_diffuser_dict() # 2. swap in pure diffusers dict
    _WanTransformer3DModel.save_pretrained(self, path, **kwargs)  # 3. delegate
    self.config = hf_config                      # 4. restore VeOmni config
```

Step 2 calls `to_diffuser_dict()`, which returns **only the parameters that
appear in the diffusers model's `__init__` signature** (inspected at import
time via `inspect.signature`). VeOmni-only fields (`model_type`,
`condition_model_type`, `_attn_implementation`, etc.) are automatically
excluded because they are not in that signature.

Step 3 delegates to the diffusers `save_pretrained`, which:
- Writes `config.json` using `to_dict()` (overridden to inject `_class_name`
  and `_diffusers_version` and strip `dtype`).
- Writes `model.safetensors` using the model's own `state_dict()` — the weight
  key names are identical to the original diffusers model because VeOmni
  inherits all layers without renaming them.

The resulting checkpoint directory is indistinguishable from one produced by
diffusers directly. A downstream pipeline can load it without any knowledge of
VeOmni:

```python
from diffusers import WanPipeline, WanTransformer3DModel

# Load the original pipeline
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")

# Swap in the fine-tuned transformer (trained with VeOmni)
pipe.transformer = WanTransformer3DModel.from_pretrained(
    "veomni_output/checkpoint-500/transformer"
)
pipe.to("cuda")
output = pipe(prompt="...", ...)
```

### Config Bridge — `to_diffuser_dict()` and `to_dict()`

`to_diffuser_dict()` uses Python's `inspect` module to extract exactly the
parameters the diffusers `__init__` accepts, making it robust to diffusers
version changes:

```python
WAN_INIT_SIGNATURE = inspect.signature(WanTransformer3DModel.__init__)

def to_diffuser_dict(self):
    return {
        key: getattr(self, key)
        for key in WAN_INIT_SIGNATURE.parameters
        if key != "self"
    }
```

`to_dict()` overrides the transformers default to produce a diffusers-format
`config.json` rather than a transformers-format one:

```python
def to_dict(self):
    d = super().to_dict()
    d["_class_name"] = "WanTransformer3DModel"   # diffusers loader key
    d["_diffusers_version"] = diffusers.__version__
    del d["dtype"]   # transformers adds this; diffusers configs don't have it
    return d
```

### Why Dual MRO (`PreTrainedModel` + diffusers model)?

Multiple inheritance gives VeOmni access to both ecosystems from a single
object:

| Inherited from `PreTrainedModel` | Inherited from diffusers model |
|---|---|
| `_from_config()` — construct without checkpoint | All layer definitions (`blocks`, `rope`, etc.) |
| `supports_gradient_checkpointing` | `_gradient_checkpointing_func` |
| FSDP-compatible parameter iteration | `save_pretrained` (diffusers format) |
| `requires_grad_()`, `named_parameters()` | `from_pretrained` (diffusers format) |

The only conflict is `_internal_dict`, which both classes use for different
purposes. This is resolved by deleting `_internal_dict` after
`PreTrainedModel.__init__` and re-creating it through a property setter after
the diffusers `__init__` runs.

### Summary Table

| Aspect | How it works |
|---|---|
| Loading diffusers ckpt | `from_pretrained` fully delegates to `_DiffusersModel.from_pretrained`; reads standard `config.json` + `model.safetensors` |
| Config in memory | `PretrainedConfig` subclass holding both diffusers params and VeOmni fields |
| Weight key names | Unchanged — no layer renaming, weights are identical to diffusers |
| Saving | Temporarily swaps config to `to_diffuser_dict()`, delegates to `_DiffusersModel.save_pretrained`, then restores |
| Saved `config.json` | Pure diffusers format (`_class_name`, `_diffusers_version`, no VeOmni fields) |
| Saved `model.safetensors` | Identical format to original diffusers checkpoint |
| Diffusers inference | Load saved ckpt directly with `WanTransformer3DModel.from_pretrained` or swap into any diffusers pipeline — no VeOmni required |
| SP forward patch | Applied to the **base diffusers class** (not VeOmni subclass) so both code paths benefit |

---

## Step 1: Create the Directory Structure

```bash
mkdir -p veomni/models/diffusers/your_dit/your_condition/
mkdir -p veomni/models/diffusers/your_dit/your_transformer/

touch veomni/models/diffusers/your_dit/__init__.py
touch veomni/models/diffusers/your_dit/your_condition/__init__.py
touch veomni/models/diffusers/your_dit/your_condition/configuration_your_condition.py
touch veomni/models/diffusers/your_dit/your_condition/modeling_your_condition.py
touch veomni/models/diffusers/your_dit/your_transformer/__init__.py
touch veomni/models/diffusers/your_dit/your_transformer/configuration_your_transformer.py
touch veomni/models/diffusers/your_dit/your_transformer/modeling_your_transformer.py
```

Then add your module to `veomni/models/diffusers/__init__.py`:

```python
from . import your_dit
```

---

## Step 2: Condition Model Config

Subclass `PretrainedConfig`. The config stores **paths to the checkpoint components**
(text encoder, VAE, scheduler, etc.) that the condition model loads at init time.
Override `get_config_dict` to inject the checkpoint directory as `base_model_path`
so `from_pretrained(path)` works automatically.

```python
from transformers import PretrainedConfig

class YourConditionModelConfig(PretrainedConfig):
    model_type = "YourConditionModel"  # must be unique; used as registry key

    def __init__(
        self,
        base_model_path: str = "",
        text_encoder_subfolder: str = "text_encoder",
        vae_subfolder: str = "vae",
        scheduler_subfolder: str = "scheduler",
        num_train_timesteps: int = 1000,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.num_train_timesteps = num_train_timesteps
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        # Inject the root checkpoint path so subfolders resolve correctly.
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
```

---

## Step 3: Condition Model — Required Methods

Subclass `transformers.PreTrainedModel`. Implement the two methods that
`DiTTrainer` calls at every training step:

### `get_condition(inputs, videos, **kwargs) -> dict`

**Called once per batch on raw inputs** (prompts, raw video frames).

- Encodes text prompts into embeddings (e.g. using a T5/UMT5 encoder).
- Encodes raw video frames into latent parameters (e.g. using a VAE encoder).
- Returns a dict whose values are lists-of-tensors (one element per sample), ready
  to be stored or passed to `process_condition`.

```python
@torch.no_grad()
def get_condition(self, inputs, videos, **kwargs) -> dict:
    # inputs: list[str]  — text prompts, one per sample
    # videos: list[list[Tensor]]  — raw video frames, one list per sample
    prompt_embeds = self._encode_text(inputs)        # list of (1, seq, dim)
    latents_list = [self._encode_video(v) for v in videos]  # list of (1, C, F, H, W)
    return {"latents": latents_list, "context": prompt_embeds}
```

### `process_condition(latents, context, **kwargs) -> dict`

**Called once per batch** to add noise and build training targets.

- Samples timesteps and noise.
- Builds `noisy_latents` (the model input) and `training_target` (the supervision signal).
- Returns a dict whose values are lists of tensors, ready to be passed as
  keyword arguments to `YourTransformerModel.forward`.

```python
def process_condition(self, latents, context) -> dict:
    packed = {"hidden_states": [], "timestep": [], "encoder_hidden_states": [], "training_target": []}
    for sample_latents, sample_context in zip(latents, context):
        latents = self._decode_latent_params(sample_latents)
        noise = torch.randn_like(latents)
        timestep = self._sample_timestep(latents)
        noisy_latents = self.scheduler.scale_noise(latents, timestep, noise)
        packed["hidden_states"].append(noisy_latents)
        packed["timestep"].append(timestep)
        packed["encoder_hidden_states"].append(sample_context)
        packed["training_target"].append(noise - latents)  # flow matching target
    return packed
```

The keys in the returned dict **must exactly match the parameter names** of
`YourTransformerModel.forward` (except `latents`, which the trainer passes
separately for logging/visualization).

---

## Step 4: Register the Condition Model

```python
# veomni/models/diffusers/your_dit/your_condition/__init__.py
from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY

@MODEL_CONFIG_REGISTRY.register("YourConditionModel")
def register_config():
    from .configuration_your_condition import YourConditionModelConfig
    return YourConditionModelConfig

@MODELING_REGISTRY.register("YourConditionModel")
def register_modeling(architecture: str = None):
    from .modeling_your_condition import YourConditionModel
    return YourConditionModel
```

---

## Step 5: Transformer Model Config

Subclass `PretrainedConfig` and mirror every parameter of the diffusers
model's `__init__`. Add `to_diffuser_dict()` to project back to the kwargs
that the diffusers model expects, and override `to_dict()` so the saved
`config.json` is compatible with `diffusers.from_pretrained`.

The `condition_model_type` class attribute links the transformer config to its
companion condition model — `DiTTrainer` reads it to decide which condition
model class to instantiate.

```python
import inspect
from diffusers import YourDiffusersTransformerModel as _YourDiffusersModel
from transformers import PretrainedConfig

_DIFFUSERS_INIT_SIGNATURE = inspect.signature(_YourDiffusersModel.__init__)

class YourTransformerConfig(PretrainedConfig):
    model_type = "YourTransformerModel"             # registry key for this model
    condition_model_type = "YourConditionModel"     # registry key for companion

    def __init__(self, num_layers=28, hidden_size=1024, **kwargs):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # ... all diffusers init params ...
        super().__init__(**kwargs)

    def to_diffuser_dict(self) -> dict:
        """Return only the kwargs accepted by the diffusers model __init__."""
        return {
            key: getattr(self, key)
            for key in _DIFFUSERS_INIT_SIGNATURE.parameters
            if key != "self"
        }

    def to_dict(self) -> dict:
        d = super().to_dict()
        # Ensure the saved config.json looks like a diffusers checkpoint.
        d["_class_name"] = "YourDiffusersTransformerModel"
        d["_diffusers_version"] = diffusers.__version__
        d.pop("dtype", None)
        return d
```

---

## Step 6: Transformer Model — Class and Init

Inherit from **both** `PreTrainedModel` (transformers) **and** the diffusers
model. Call them in the right order and resolve the config-dict conflict.

```python
from diffusers import YourDiffusersTransformerModel as _YourDiffusersModel
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

class YourTransformerModel(PreTrainedModel, _YourDiffusersModel):
    config_class = YourTransformerConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: YourTransformerConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        # diffusers stores model state in _internal_dict; remove the transformers one.
        del self._internal_dict
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)
        _YourDiffusersModel.__init__(self, **config.to_diffuser_dict())
        self.config: YourTransformerConfig = config
        self.config.tie_word_embeddings = False

    # ── config property: resolves the two-class conflict ────────────────────
    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value
```

> **Why delete `_internal_dict` then reassign?**
> `PreTrainedModel.__init__` creates `self._internal_dict` to store the config.
> The diffusers `__init__` also uses `_internal_dict` for its own state dict.
> By deleting it after `PreTrainedModel.__init__` and re-creating it through
> the property setter after `_YourDiffusersModel.__init__`, VeOmni's
> `YourTransformerConfig` ends up as the sole occupant of `_internal_dict`.

---

## Step 7: Transformer Model — Required Methods

### `forward(self, hidden_states, timestep, encoder_hidden_states, training_target, ...) -> ModelOutput`

**This is the main training forward.** It is called once per global batch by
`DiTTrainer` with the outputs of `process_condition()` unpacked as keyword args.
It must:

1. Iterate over samples in the batch (DiT batches are lists of per-sample tensors,
   because video resolutions differ).
2. Call the diffusers backbone forward for each sample.
3. Compute a per-sample loss (typically MSE between prediction and target).
4. Return a `ModelOutput` subclass with a `loss` dict and optional `predictions`.

```python
@dataclass
class YourModelOutput(ModelOutput):
    loss: dict[str, torch.FloatTensor] | None = None
    predictions: list[torch.FloatTensor] | None = None

def forward(self, hidden_states, timestep, encoder_hidden_states, training_target, **kwargs):
    per_sample_losses = []
    predictions = []
    for hs, ts, enc_hs, target in zip(hidden_states, timestep, encoder_hidden_states, training_target):
        prediction = _YourDiffusersModel.forward(
            self, hidden_states=hs, timestep=ts, encoder_hidden_states=enc_hs
        )
        predictions.append(prediction)
        loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
        per_sample_losses.append(loss.view(loss.shape[0], -1).mean(dim=1))
    loss = torch.stack(per_sample_losses).mean()
    return YourModelOutput(loss={"mse_loss": loss}, predictions=predictions)
```

> **Why call `_YourDiffusersModel.forward` explicitly?**
> Because `self.forward` is overridden by VeOmni. The diffusers backbone
> forward (with the SP patch applied) lives on `_YourDiffusersModel.forward`.
> Calling it explicitly ensures the SP-patched version is used.

### `save_pretrained(path, **kwargs)`

Convert the VeOmni config back to a diffusers dict before saving so that the
checkpoint can be reloaded by diffusers directly:

```python
def save_pretrained(self, path, **kwargs):
    hf_config = copy.deepcopy(self.config)
    self.config = self.config.to_diffuser_dict()
    _YourDiffusersModel.save_pretrained(self, path, **kwargs)
    self.config = hf_config
```

### `from_pretrained(path, **kwargs)` (classmethod)

Delegate to the diffusers loader to read the original checkpoint format:

```python
@classmethod
def from_pretrained(cls, path, **kwargs):
    return _YourDiffusersModel.from_pretrained(path, **kwargs)
```

---

## Step 8: SP Forward Patch (Ulysses Sequence Parallelism)

For DiT models, SP works by slicing the **patchified token sequence** across
Ulysses ranks before the transformer blocks and gathering it back before the
output projection. This patch is applied to the **base diffusers class**
(not the VeOmni subclass) so that calling `_YourDiffusersModel.forward(self, ...)`
from inside `YourTransformerModel.forward` also runs the patched version.

### Pattern

Define the patched forward as a **standalone module-level function** with a
descriptive comment block (same convention as `Qwen3VLVisionAttention_forward`):

```python
# ================================================================
# Patch: YourDiffusersTransformerModel.forward
# 1. Slice patchified token sequence across Ulysses SP ranks.
# 2. Gather back before the output projection head.
# ================================================================
def YourDiffusersTransformerModel_forward(
    self: _YourDiffusersModel,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    **kwargs,
):
    # ... patchify ...
    # SP slice
    if _sp_active:
        hidden_states = slice_input_tensor_scale_grad(hidden_states, dim=1)
    # ... transformer blocks ...
    # SP gather
    if _sp_active:
        hidden_states = gather_outputs(hidden_states, gather_dim=1)
    # ... output projection + unpatchify ...
    return output


def apply_veomni_sp_patch() -> None:
    _YourDiffusersModel.forward = YourDiffusersTransformerModel_forward
    logger.info_rank0("Applied VeOmni SP patch to YourDiffusersTransformerModel.forward.")
```

**Key rules:**

- Only slice/gather when both `ulysses_enabled` **and** an SP-aware attention
  implementation is configured. Without the attention-level AllToAll the
  sequence slice would be incorrect.
- Also slice the rotary embeddings to the local rank's positions after the
  sequence is sliced.
- The patch function is registered in `__init__.py` at model load time (see Step 9).

---

## Step 9: SP Attention Processor

Ulysses SP requires an AllToAll before and after the attention kernel.
For diffusers models this is done via an **attention processor** installed with
`attn.set_processor(...)`.

### Pattern (mirrors `veomni/ops/flash_attn` registration)

1. Implement a `your_eager_attention_forward` function — the non-flash fallback.
   It must follow the `ALL_ATTENTION_FUNCTIONS` calling convention:
   - Inputs: `(module, query, key, value, attention_mask, ...)` where Q/K/V are
     in `(B, heads, seq, head_dim)` format.
   - Output: `((B, seq, heads, head_dim), None)`.

2. In `YourSPAttnProcessor.__init__`, store the implementation name and expose
   the attributes that `flash_attention_forward` (from `veomni/ops/flash_attn`)
   reads from `module`:

   ```python
   from types import SimpleNamespace
   from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

   class YourSPAttnProcessor:
       def __init__(self, attn_implementation: str):
           self.attn_implementation = attn_implementation
           self.config = SimpleNamespace(_attn_implementation=attn_implementation)
           self.is_causal = False
           self.layer_idx = None
   ```

3. In `__call__`, resolve the attention function via `ALL_ATTENTION_FUNCTIONS`:

   ```python
   attention_interface: Callable = your_eager_attention_forward
   if self.attn_implementation != "eager":
       attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
   ```

4. Transpose Q/K/V to `(B, heads, seq, head_dim)` before calling and pass
   `skip_ulysses=True` (SP is already handled by the manual AllToAll above):

   ```python
   hidden_states_out = attention_interface(
       self,
       query.transpose(1, 2),
       key.transpose(1, 2),
       value.transpose(1, 2),
       attention_mask=None,
       dropout=0.0,
       is_causal=False,
       skip_ulysses=True,
   )[0]  # returns (B, seq, heads, head_dim)
   ```

   The `veomni_flash_attention_*_with_sp` functions registered in
   `ALL_ATTENTION_FUNCTIONS` (see `veomni/ops/flash_attn/__init__.py`) will
   select the correct FA2/FA3/FA4 kernel automatically.

5. Install the processor on every attention block:

   ```python
   def _setup_sp_attention(model, attn_implementation):
       sp_processor = YourSPAttnProcessor(attn_implementation)
       for block in model.blocks:
           block.attn.set_processor(sp_processor)
   ```

   Call `_setup_sp_attention` from `YourTransformerModel.__init__` when
   `attn_implementation in _VEOMNI_SP_ATTN_IMPLS`.

---

## Step 10: Register the Transformer Model

```python
# veomni/models/diffusers/your_dit/your_transformer/__init__.py
from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY

@MODEL_CONFIG_REGISTRY.register("YourTransformerModel")
def register_config():
    from .configuration_your_transformer import YourTransformerConfig
    return YourTransformerConfig

@MODELING_REGISTRY.register("YourTransformerModel")
def register_modeling(architecture: str):
    from .modeling_your_transformer import YourTransformerModel, apply_veomni_sp_patch
    apply_veomni_sp_patch()   # patch base diffusers class at load time
    return YourTransformerModel
```

> `apply_veomni_sp_patch()` is called in the register function (not at import
> time) to match the lazy-loading pattern used by `MODELING_REGISTRY`.

---

## Step 11: Add the Config File

Create `configs/dit/your_model_sft.yaml`. The trainer reads
`model.config_path` to determine which transformer and condition model to build:

```yaml
model:
  model_path: YourOrg/YourModel-Diffusers/transformer
  config_path: ./configs/model_configs/your_model/your_model.json
  condition_model_path: YourOrg/YourModel-Diffusers
  ops_implementation:
    attn_implementation: veomni_flash_attention_2_with_sp

train:
  accelerator:
    ulysses_size: 4
    fsdp_config:
      fsdp_mode: fsdp1
      full_shard: true
  global_batch_size: 8
  micro_batch_size: 1
```

And create `configs/model_configs/your_model/your_model.json` mirroring the
diffusers `transformer/config.json` with the VeOmni additions:

```json
{
  "_class_name": "YourDiffusersTransformerModel",
  "model_type": "YourTransformerModel",
  "condition_model_type": "YourConditionModel",
  "num_layers": 28,
  ...
}
```

---

## Required Functions — Summary

### Condition Model

| Function | Required | Purpose |
|---|:---:|---|
| `__init__` | ✓ | Load VAE, text encoder, scheduler from `config.base_model_path` |
| `get_condition(inputs, videos)` | ✓ | Encode raw data → latent params + text embeddings |
| `process_condition(latents, context)` | ✓ | Sample noise/timesteps, build training inputs and targets |
| `_load_components()` | — | Helper; load model components (call from `__init__`) |

### Transformer Config

| Method | Required | Purpose |
|---|:---:|---|
| `to_diffuser_dict()` | ✓ | Project VeOmni config → diffusers `__init__` kwargs |
| `to_dict()` | ✓ | Override so saved `config.json` is diffusers-compatible |
| `condition_model_type` (class attr) | ✓ | String registry key of the companion condition model |

### Transformer Model

| Function | Required | Purpose |
|---|:---:|---|
| `__init__` | ✓ | Dual-MRO init; install SP attention processor if needed |
| `config` property + setter | ✓ | Resolve `_internal_dict` conflict between transformers and diffusers |
| `forward(hidden_states, timestep, encoder_hidden_states, training_target, ...)` | ✓ | Per-sample iteration, call diffusers backbone, compute loss |
| `save_pretrained` | ✓ | Convert config back to diffusers format before delegating |
| `from_pretrained` (classmethod) | ✓ | Delegate to diffusers loader |
| `YourDiffusersModel_forward` (module-level) | ✓ | SP-patched forward for the diffusers base class |
| `apply_veomni_sp_patch()` | ✓ | Monkey-patch the diffusers base class at load time |
| `wan_eager_attention_forward` (or equivalent) | ✓ | SDPA fallback following `ALL_ATTENTION_FUNCTIONS` convention |
| `YourSPAttnProcessor` | ✓ (for SP) | Ulysses AllToAll + routes to `flash_attention_forward` |
| `_setup_sp_attention(model, impl)` | ✓ (for SP) | Install the SP processor on every attention block |

---

## Checklist

### Any New DiT Model

- [ ] `veomni/models/diffusers/your_dit/__init__.py` (imports sub-packages)
- [ ] `veomni/models/diffusers/__init__.py` updated with `from . import your_dit`
- [ ] Condition model config: `PretrainedConfig` subclass with `get_config_dict` override
- [ ] Condition model: `get_condition()` and `process_condition()` implemented
- [ ] Condition model registered in `MODEL_CONFIG_REGISTRY` and `MODELING_REGISTRY`
- [ ] Transformer config: `to_diffuser_dict()`, `to_dict()`, and `condition_model_type` defined
- [ ] Transformer model: dual-MRO `__init__` with `_internal_dict` cleanup
- [ ] Transformer model: `config` property + setter
- [ ] Transformer model: `forward()` iterating per-sample and returning `ModelOutput`
- [ ] Transformer model: `save_pretrained` + `from_pretrained` overrides
- [ ] SP forward patch: standalone module-level function, assigned in `apply_veomni_sp_patch()`
- [ ] Transformer model registered with `apply_veomni_sp_patch()` call in register function
- [ ] `configs/dit/your_model_sft.yaml` and `configs/model_configs/your_model/your_model.json`

### If Supporting Ulysses SP

- [ ] `_VEOMNI_SP_ATTN_IMPLS` frozenset defined (the three `veomni_flash_attention_*_with_sp` names)
- [ ] `YourEagerAttentionForward` function with `ALL_ATTENTION_FUNCTIONS` convention
- [ ] `YourSPAttnProcessor` with `SimpleNamespace` config, `is_causal=False`, `layer_idx=None`
- [ ] SP guard in forward patch: slice/gather only when `ulysses_enabled and attn_impl in _VEOMNI_SP_ATTN_IMPLS`
- [ ] RoPE embeddings sliced to the local rank's positions when SP is active
- [ ] `_setup_sp_attention` installs the processor on every attention block

---

## Common Pitfalls

| Symptom | Likely Cause | Fix |
|---|---|---|
| `AttributeError: _internal_dict` on init | `del self._internal_dict` called before diffusers init sets it | Call `del self._internal_dict` after `PreTrainedModel.__init__`, before `_DiffusersModel.__init__` |
| Config saved without `_class_name` / diffusers can't reload | `to_dict()` not overridden | Override `to_dict()` to inject `_class_name` and `_diffusers_version` |
| NCCL deadlock on SP with gradient checkpointing | SP gather not inside the checkpoint boundary | Ensure `gather_outputs` is called outside the block loop, not inside `_gradient_checkpointing_func` |
| Wrong shape after AllToAll in attention | Q/K/V not transposed before `flash_attention_forward` | `flash_attention_forward` expects `(B, heads, seq, head_dim)`; transpose before calling |
| Double SP gather (output is wrong) | `flash_attention_forward` called without `skip_ulysses=True` while AllToAll already done manually | Pass `skip_ulysses=True` to all `attention_interface(...)` calls in the SP processor |
| `KeyError` in `ALL_ATTENTION_FUNCTIONS` | `apply_veomni_attention_patch()` not called before model load | `ops` patch is applied via `apply_ops_patch()` at startup; ensure it runs before the register function |
| `process_condition` key names mismatch | Returned dict keys don't match `forward()` parameter names | Check that every key in `process_condition` output corresponds to a `forward()` parameter |
