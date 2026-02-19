# Modeling Code Generation

A code generation framework for creating patched HuggingFace modeling files. Instead of runtime monkey patches that are hard to debug, this tool generates self-contained, readable modeling code with all patches applied at code-generation time.

## Quick Start

```bash
# Generate patched Qwen3 GPU modeling code (writes to veomni/models/transformers/qwen3/generated/)
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config

# With verbose output
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config -v

# Dry run (preview without writing)
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config --dry-run

# Custom output directory
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config -o /path/to/output

# List available patch configurations
python -m veomni.patchgen.run_codegen --list

# Save unified diff alongside generated modeling code
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config --diff
```

## Project Structure

```
veomni/
├── patchgen/
│   ├── patch_spec.py              # Patch specification DSL
│   ├── codegen.py                 # AST-based code generator
│   └── run_codegen.py             # CLI runner script
└── models/
    └── transformers/
        └── qwen3/
            ├── patches/
            │   └── qwen3_gpu_patch_gen_config.py       # Qwen3 GPU patches
            └── generated/
                └── patched_modeling_qwen3_gpu.py  # Generated output
```

## Core Design

### The Problem

When adapting HuggingFace models for training frameworks (VeOmni, veRL, etc.), we need to apply various modifications:

- **Attention replacement**: Custom flash attention, Ulysses sequence parallelism
- **Kernel fusion**: LigerKernel RMSNorm, fused rotary embeddings
- **MoE optimizations**: Fused MoE with expert parallelism
- **Framework-specific code**: Gradient checkpointing, loss computation

Current approaches have significant drawbacks:

```python
# BAD: Runtime monkey patching - hard to debug, order-dependent
apply_ops_patch()
apply_logprobs_patch()
apply_xpu_patch()

ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = my_impl

class OptimizedQwen3Model(Qwen3Model):
    ...
Qwen3Model = OptimizedQwen3Model  # Who knows what this is now?
```

**Problems with monkey patching:**

- Cannot see the final patched code
- Import order affects behavior
- Difficult to debug
- Multiple patches can conflict
- Hard to maintain across HF version upgrades

### The Solution

Generate a single, self-contained modeling file with all patches applied:

```python
# GOOD: Generated file - everything is visible
# ======================================================================
# [PATCHED CLASS] Qwen3RMSNorm
# Reason: Use fused RMSNorm kernel for better performance
# ======================================================================
class Qwen3RMSNorm(nn.Module):
    def forward(self, hidden_states):
        # Patched implementation - fully visible
        ...
```

**Benefits:**

- Complete visibility of final code
- Easy to debug and understand
- Clear documentation of what changed and why
- No runtime surprises
- Can diff against original HF code
- Comments in patch code are preserved in output

### Design Principles

1. **AST-based transformation**: Uses Python AST for robust code manipulation, not fragile regex
1. **Declarative patches**: Define what to patch using decorators, not imperative monkey patches
1. **Source preservation**: Extracts source from installed transformers at generation time
1. **Comment preservation**: Comments in replacement code are preserved in the generated output
1. **Self-contained output**: Generated file has no hidden dependencies on patch code

## How to Use

### 1. Create a Patch Configuration

Create a new file under `veomni/models/transformers/qwen3/` (for now we only ship Qwen3):

```python
from veomni.patchgen.patch_spec import PatchConfig

# Define the configuration
config = PatchConfig(
    source_module="transformers.models.qwen3.modeling_qwen3",
    target_file="patched_modeling_qwen3_gpu.py",
    description="Qwen3 GPU patches",
)
```

### 2. Define Patches

#### Class Replacement

Replace an entire class with a custom implementation:

```python
@config.replace_class("Qwen3RMSNorm", description="Use fused kernel")
class OptimizedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Your optimized implementation
        ...
```

#### Function Replacement

Replace a module-level function:

```python
@config.replace_function("apply_rotary_pos_emb", description="Use fused RoPE")
def optimized_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # Your optimized implementation
    ...
```

#### Method Override

Replace a specific method within a class (keeps the rest of the class unchanged):

```python
@config.override_method("Qwen3Attention.forward", description="Add Ulysses SP")
def ulysses_attention_forward(self, hidden_states, position_embeddings, ...):
    # Your modified forward pass
    # Comments here will be preserved in the generated output!

    # ==========================================================================
    # BEGIN ULYSSES SEQUENCE PARALLEL MODIFICATIONS
    # ==========================================================================
    # These comments will appear in the generated file
    ...
```

### 3. Add Supporting Imports

```python
# Add imports needed by your patches
config.add_import("torch.distributed", names=["all_gather", "all_reduce"])

# Exclude classes you don't need
config.exclude_from_output("Qwen3ForTokenClassification")
```

### 4. Generate Code

```bash
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config -v
```

## Patch Types Reference

| Type                 | Decorator                                 | Use Case                                             |
| -------------------- | ----------------------------------------- | ---------------------------------------------------- |
| Class Replacement    | `@config.replace_class("ClassName")`      | Replace entire class (e.g., RMSNorm -> LigerRMSNorm) |
| Function Replacement | `@config.replace_function("func_name")`   | Replace module-level function                        |
| Method Override      | `@config.override_method("Class.method")` | Replace single method, keep rest of class            |

## Generated Output Format

The generated file includes:

1. **Header** with metadata:

    ```python
    # ==============================================================================
    #  AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY
    # ==============================================================================
    #  Source: transformers.models.qwen3.modeling_qwen3
    #  Based on: transformers==4.57.3
    #  Generated: 2026-01-25T10:30:00
    #
    #  Patches applied:
    #    - class_replacement: Qwen3RMSNorm
    #    - function_replacement: apply_rotary_pos_emb
    #    - method_override: Qwen3Attention.forward
    # ==============================================================================
    ```

1. **Converted imports** (relative -> absolute):

    ```python
    # Original relative import: from ...activations import ACT2FN
    from transformers.activations import ACT2FN
    ```

1. **Patch markers** for modified code:

    ```python
    # ======================================================================
    # [PATCHED CLASS] Qwen3RMSNorm
    # Reason: Use fused RMSNorm kernel for better performance
    # ======================================================================
    class Qwen3RMSNorm(nn.Module):
        ...
    ```

1. **Preserved comments** in patched methods:

    ```python
    # ======================================================================
    # [MODIFIED CLASS] Qwen3Attention
    # Methods patched: forward
    # ======================================================================
    class Qwen3Attention(nn.Module):
        def forward(self, hidden_states, ...):
            # ==========================================================================
            # BEGIN ULYSSES SEQUENCE PARALLEL MODIFICATIONS
            # ==========================================================================
            # All your inline comments are preserved!
            ...
    ```

## Example: Qwen3 GPU Patches

See `veomni/models/transformers/qwen3/qwen3_gpu_patch_gen_config.py` for a complete example that includes:

- **LigerRMSNorm**: Fused kernel replacement for `Qwen3RMSNorm`
- **LigerSwiGLUMLP**: Fused SwiGLU MLP replacement for `Qwen3MLP`
- **apply_rotary_pos_emb**: LigerKernel rotary position embedding

Run it:

```bash
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config -v
```

Output: `veomni/models/transformers/qwen3/generated/patched_modeling_qwen3_gpu.py` (~600 lines of self-contained code)

## Comparing Generated vs Original Code

Use the `--diff` flag to save a unified diff file next to the generated modeling file:

```bash
# Generate patched modeling code and save a .diff file in output directory
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config --diff
```

With `--diff`, `run_codegen` writes:

- Compares the generated file against the original HuggingFace source
- Saves unified diff as `<generated_modeling_name>.diff` in the output directory
- Shows exactly which classes, methods, and functions were modified

Example output:

```diff
-class Qwen3RMSNorm(nn.Module):
-    def forward(self, hidden_states):
-        input_dtype = hidden_states.dtype
-        hidden_states = hidden_states.to(torch.float32)
-        ...
+# ======================================================================
+# [PATCHED CLASS] Qwen3RMSNorm
+# Reason: Use fused RMSNorm kernel for better performance
+# ======================================================================
+class Qwen3RMSNorm(nn.Module):
+    def forward(self, hidden_states):
+        # Optimized implementation with comments preserved
+        ...
```

## Advanced Usage

### External Class References

For large classes from external libraries, reference them without copying source:

```python
from veomni.patchgen.patch_spec import create_patch_from_external

patch = create_patch_from_external(
    target="Qwen3RMSNorm",
    replacement_module="liger_kernel.transformers.rms_norm",
    replacement_name="LigerRMSNorm",
)
config.patches.append(patch)
```

### Programmatic Generation

```python
from codegen import ModelingCodeGenerator
from veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config import config

generator = ModelingCodeGenerator(config)
generator.load_source()
output = generator.generate(
    Path("veomni/models/transformers/qwen3/generated/patched_modeling_qwen3_gpu.py")
)
```

### Init Modification

Modify `__init__` methods without replacing the entire class:

```python
@config.modify_init("Qwen3Attention")
def modified_init(original_init, self, config, layer_idx):
    original_init(self, config, layer_idx)
    self.custom_attr = some_value
```

## CLI Reference

```
usage: python -m veomni.patchgen.run_codegen [-h] [-o OUTPUT_DIR] [-c CONFIG_NAME] [--list]
                      [--dry-run] [--diff] [-v]
                      [patch_module]

positional arguments:
  patch_module          Patch module to use (e.g., 'veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config')

options:
  -h, --help            Show help message
  -o, --output-dir      Output directory (default: sibling generated/ next to patch module)
  -c, --config-name     Config variable name in the patch module (default: config)
  --list                List available patch configurations
  --dry-run             Show what would be generated without writing files
  --diff                Save a unified .diff file alongside generated modeling code
  -v, --verbose         Print detailed progress
```

## Background: Why Not Monkey Patching?

### Existing Approaches and Their Trade-offs

| Approach            | Example | Pros                 | Cons                              |
| ------------------- | ------- | -------------------- | --------------------------------- |
| **Monkey Patching** | veRL    | Reuses HF code       | Hard to debug, order-dependent    |
| **Copy + Modify**   | VeOmni  | Fully visible code   | Manual maintenance, drift from HF |
| **Inheritance**     | Various | Code reuse           | Deep inheritance chains           |
| **Custom Backend**  | vLLM    | Maximum optimization | Accuracy black hole               |

### This Tool's Approach

Inspired by HuggingFace's own `modular_model_converter.py`, we:

1. **Define patches declaratively** in Python files
1. **Generate code at build time**, not runtime
1. **Produce readable output** with clear patch markers
1. **Preserve comments** from patch definitions
1. **Support easy regeneration** when HF updates

## Common Patch Scenarios

| Scenario                    | Patch Type           | Example              |
| --------------------------- | -------------------- | -------------------- |
| Fused kernels (LigerKernel) | Class replacement    | RMSNorm, SwiGLU      |
| Optimized RoPE              | Function replacement | apply_rotary_pos_emb |
| Sequence parallelism        | Method override      | Attention.forward    |
| Expert parallelism          | Method override      | MoE.forward          |
| Custom loss                 | Function replacement | cross_entropy        |
| VLM modifications           | Method override      | Model.forward        |

## Limitations

- **Python 3.9+** required (uses `ast.unparse`)
- Generated code may need manual adjustment for complex patches
- Some HF decorators (e.g., `@use_kernel_forward_from_hub`) may need special handling
- Does not handle dynamic/conditional patches (use config flags in patches instead)

## Contributing

To add support for a new model:

1. Create `veomni/models/transformers/<model>/patches/<model>_patches.py`
1. Define your `PatchConfig` and patches
1. Test with `python -m veomni.patchgen.run_codegen veomni.models.transformers.<model>.patches.<model>_patches --dry-run`
1. Generate and verify the output in `veomni/models/transformers/<model>/generated/`
1. Use `--diff` to review changes against original HF code
