# Custom Preprocessor Registry

## Overview

The Custom Preprocessor Registry provides a simple and extensible system for registering data source-specific preprocessor functions in VeOmni. All preprocessors (built-in and custom) are registered using the `@register_preprocessor` decorator and automatically available throughout the framework.

**Terminology Clarification:**
- **Dataset**: Classes that handle data loading (e.g., `MultiSourceDataset`, `MappingDataset`, `IterableDataset`)
- **Preprocessor**: Functions that convert raw data samples from a specific source into model-ready format

This registry manages preprocessor functions, not dataset classes.

## Features

- **Decorator-based API**: Simple `@register_preprocessor` decorator for registration
- **Auto-registration**: Preprocessors are automatically registered when the module is imported
- **Multiple Names**: Register the same preprocessor under multiple data source names
- **Clear Terminology**: Distinguishes between dataset classes (data loading) and preprocessor functions (format conversion)

## Quick Start

### 1. Define Your Custom Preprocessor

Add your preprocessor to [`veomni/data/multimodal/preprocess.py`](../../veomni/data/multimodal/preprocess.py):

```python
from .preprocessor_registry import register_preprocessor

@register_preprocessor("my_custom_source")
def my_custom_source_preprocessor(conversations, **kwargs):
    """
    Preprocessor for a custom data source.

    Args:
        conversations: Raw conversation data from your source
        **kwargs: Additional arguments (e.g., generation_ratio, max_image_nums)

    Returns:
        constructed_conversation: List of [role, (modality, content), ...]

    Expected format:
        [
            ["user", ("image", None), ("text", "What is this?")],
            ["assistant", ("text", "This is a cat.")]
        ]
    """
    constructed_conversation = []

    # Your preprocessing logic here
    # Convert your data source format to VeOmni's format

    return constructed_conversation
```

### 2. Use Your Preprocessor

Once registered, your preprocessor is immediately available:

```python
# Use the convenience function from preprocessor_registry.py
from veomni.data.multimodal.preprocessor_registry import conv_preprocess

# Or get the preprocessor function directly from the registry
from veomni.data.multimodal import get_preprocessor

# Option 1: Using conv_preprocess (convenience function)
result = conv_preprocess("my_custom_source", conversations)

# Option 2: Using get_preprocessor (direct access to preprocessor)
preprocessor = get_preprocessor("my_custom_source")
result = preprocessor(conversations)
```

### 3. Use in Your Config

```yaml
data:
  datasets:
    - name: my_data
      source_name: my_custom_source  # Matches @register_preprocessor name
      data_path: /path/to/my/dataset
      weight: 1.0
```

## Architecture

### Registration Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Define preprocessor with @register_preprocessor decorator│
│    └─> Immediately adds to _PREPROCESSOR_REGISTRY          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Import veomni.data.multimodal module                     │
│    └─> Automatically triggers all @register_preprocessor    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Preprocessor is now available via get_preprocessor() or  │
│    conv_preprocess()                                        │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
veomni/data/multimodal/
├── __init__.py                 # Exports registry functions
├── preprocessor_registry.py    # Core registration system
└── preprocess.py               # All preprocessors (built-in + custom)
```

## Preprocessor Format

Your preprocessor must follow VeOmni's conversation format:

```python
# Input: Your data source's raw format (flexible)
conversations = [
    {"from": "human", "value": "<image> What is this?"},
    {"from": "gpt", "value": "A cat."}
]

# Output: VeOmni's standardized format (strict)
constructed_conversation = [
    ["user", ("image", None), ("text", "What is this?")],
    ["assistant", ("text", "A cat.")]
]
```

### Supported Modalities

| Modality | Format | Example |
|----------|--------|---------|
| Text | `("text", str)` | `("text", "Hello world")` |
| Image | `("image", None)` | `("image", None)` |
| Video | `("video", None)` | `("video", None)` |
| Audio | `("audio", None)` | `("audio", None)` |

## Examples

### Example 1: VQA Preprocessor

For an example of a VQA preprocessor, see [`custom_vqa_preprocess` in `preprocess.py`](../../veomni/data/multimodal/preprocess.py#L225).

### Example 2: Multi-turn Conversation Preprocessor

```python
For an example of a multi-turn conversation preprocessor, see [`sharegpt4v_sft_preprocess` in `preprocess.py`](../../veomni/data/multimodal/preprocess.py#L58).
```

### Example 3: OCR Preprocessor

For an example of a preprocessor that handles image and text, similar to an OCR preprocessor, see [`doom_preprocess` in `preprocess.py`](../../veomni/data/multimodal/preprocess.py#L76).

### Example 4: Registering Multiple Names

For an example of a preprocessor registered under multiple names, see [`sharegpt4v_pretrain_preprocess` in `preprocess.py`](../../veomni/data/multimodal/preprocess.py#L38).

## Advanced Usage

### Conditional Preprocessing

```python
@register_preprocessor("adaptive_source")
def adaptive_preprocessor(conversations, mode="caption", **kwargs):
    """Preprocessor with different modes"""
    if mode == "caption":
        return [
            ["user", ("image", None), ("text", "Describe this image.")],
            ["assistant", ("text", conversations)]
        ]
    elif mode == "generation":
        return [
            ["user", ("text", conversations)],
            ["assistant", ("image", None)]
        ]
```

Use in config:
```yaml
data:
  datasets:
    - name: adaptive_caption
      source_name: adaptive_source
      data_path: /path/to/data
      source_config:
        mode: caption
```

### Random Sampling

```python
import random

@register_preprocessor("random_prompt_source")
def random_prompt_preprocessor(conversations, **kwargs):
    """Preprocessor with randomized prompts"""
    prompts = [
        "Describe this image in detail.",
        "What do you see in this image?",
        "Please analyze this image."
    ]
    prompt = random.choice(prompts)

    return [
        ["user", ("image", None), ("text", prompt)],
        ["assistant", ("text", conversations)]
    ]
```

### Handling Multiple Formats

```python
@register_preprocessor("flexible_source")
def flexible_format_preprocessor(conversations, **kwargs):
    """Handle different input formats"""
    if isinstance(conversations, str):
        # Simple caption format
        return [
            ["user", ("image", None)],
            ["assistant", ("text", conversations)]
        ]
    elif isinstance(conversations, dict):
        # Structured format
        return [
            ["user", ("image", None), ("text", conversations["question"])],
            ["assistant", ("text", conversations["answer"])]
        ]
    elif isinstance(conversations, list):
        # Standard ShareGPT format
        role_mapping = {"human": "user", "gpt": "assistant"}
        constructed = []
        for msg in conversations:
            role = role_mapping[msg["from"]]
            value = msg["value"]
            if "<image>" in value:
                value = value.replace("<image>", "").strip()
                constructed.append([role, ("image", None), ("text", value)])
            else:
                constructed.append([role, ("text", value)])
        return constructed
```

## API Reference

### Registry Functions

The following functions are available directly from the `veomni.data.multimodal` package.

```python
from veomni.data.multimodal import (
    register_preprocessor,          # Decorator to register a preprocessor
    get_preprocessor,               # Get a preprocessor by name
    get_all_preprocessors,          # Get all registered preprocessors
    list_preprocessors,             # List all preprocessor names
    is_preprocessor_registered,     # Check if a preprocessor is registered
)
```

#### `register_preprocessor(name: str)`

Decorator to register a preprocessor for a specific data source.

```python
@register_preprocessor("my_source")
def my_preprocessor(conversations, **kwargs):
    return [["user", ("text", "hello")]]
```

#### `get_preprocessor(name: str) -> Callable`

Get a specific preprocessor by data source name.

```python
preprocessor = get_preprocessor("sharegpt4v_pretrain")
result = preprocessor(conversations)
```

Raises `ValueError` if the preprocessor is not registered.

#### `get_all_preprocessors() -> Dict[str, Callable]`

Get all registered preprocessors as a dictionary.

```python
all_preprocessors = get_all_preprocessors()
print(all_preprocessors.keys())
# dict_keys(['sharegpt4v_pretrain', 'sharegpt4v_captioner', 'doom', ...])
```

#### `list_preprocessors() -> List[str]`

List all registered preprocessor names (sorted).

```python
preprocessor_names = list_preprocessors()
print(preprocessor_names)
# ['ArxivQA', 'CHartQA', 'DenseFusion-1M', ...]
```

#### `is_preprocessor_registered(name: str) -> bool`

Check if a preprocessor is registered for the given data source name.

```python
if is_preprocessor_registered("my_source"):
    preprocessor = get_preprocessor("my_source")
```

### Convenience Functions

#### `conv_preprocess(source: str, conversations, **kwargs)`

This convenience function, located in `veomni.data.multimodal.preprocessor_registry`, wraps `get_preprocessor()` for easier use.

```python
from veomni.data.multimodal.preprocessor_registry import conv_preprocess

result = conv_preprocess("sharegpt4v_pretrain", conversations)
```

This is equivalent to:
```python
from veomni.data.multimodal import get_preprocessor

preprocessor = get_preprocessor("sharegpt4v_pretrain")
result = preprocessor(conversations)
```

## Testing

Example test for your custom preprocessor:

```python
def test_custom_source_preprocessor():
    from veomni.data.multimodal import get_preprocessor, is_preprocessor_registered

    # Check if registered
    assert is_preprocessor_registered("my_custom_source")

    # Get preprocessor
    preprocessor = get_preprocessor("my_custom_source")

    # Test your preprocessor
    test_conversations = [
        {"from": "human", "value": "<image> What is this?"},
        {"from": "gpt", "value": "A cat."}
    ]
    # Assuming my_custom_source_preprocessor is defined as in the Quick Start
    result = preprocessor(test_conversations)

    assert result == [
        ["user", ("image", None), ("text", "What is this?")],
        ["assistant", ("text", "A cat.")]
    ]
```

## Troubleshooting

### Preprocessor Not Found Error

```
ValueError: Unknown dataset name: my_source. No preprocessor registered for this source.
```

**Solution**:
1. Ensure your preprocessor is decorated with `@register_preprocessor("my_source")`.
2. Check that the `source_name` in your config matches the registered name exactly.
3. Verify the module containing your preprocessor is imported. If you add it to `veomni/data/multimodal/preprocess.py`, this is handled automatically.

### Duplicate Registration Warning

```
UserWarning: Preprocessor for 'my_source' is already registered. Overwriting with new preprocessor.
```

**Solution**: This warning means you have registered the same name more than once. Make sure each preprocessor name is unique, or confirm that you intend to overwrite the existing function.

### Wrong Output Format

```
TypeError: 'NoneType' object is not iterable
```

**Solution**: Ensure your preprocessor always returns a list of lists, even if it's empty.
```python
# ❌ Wrong
return None

# ✅ Correct
return [["user", ("text", "hello")], ["assistant", ("text", "hi")]]
```

### Import Error

```
ImportError: cannot import name 'get_preprocessor'
```

**Solution**: Make sure you're importing from the correct public API module:
```python
# ✅ Correct
from veomni.data.multimodal import get_preprocessor

# ❌ Wrong (accessing internal module)
from veomni.data.multimodal.preprocessor_registry import get_preprocessor
```

## Best Practices

1. **Naming Convention**: Use descriptive, lowercase names for preprocessors (e.g., `internal_vqa`, `custom_ocr`).
2. **Documentation**: Add docstrings to your preprocessor explaining its expected input format and what it does.
3. **Error Handling**: Add validation for the input format if it's complex, and provide clear error messages.
4. **Testing**: Write unit tests for your preprocessors.
5. **Reusability**: Extract common logic into helper functions that are not decorated.
6. **Multiple Aliases**: Use multiple `@register_preprocessor` decorators if a preprocessor can be used for different but compatible data sources.

## Usage in Training Scripts

Once you've defined your preprocessor (e.g., in `preprocess.py`), it's automatically available throughout the framework just by importing `veomni.data.multimodal`.

```python
# In your training script
from veomni.data import build_multisource_dataset

# The preprocessor is automatically registered and available
# as long as the config specifies the correct `source_name`.
dataset = build_multisource_dataset(config)
```

The preprocessor becomes available as soon as `veomni.data.multimodal` is imported anywhere in your project—no manual registration calls are needed!

### End-to-End Example: Qwen2-VL Training Pipeline

For a complete working example of how preprocessors integrate into the training pipeline, see the Qwen2-VL training setup:

**1. Training Entry Point**: [train.sh](../../train.sh)
   - Launches distributed training with torchrun

**2. Training Script**: [tasks/omni/train_qwen2_vl.py](../../tasks/omni/train_qwen2_vl.py)
   - **Line 27**: Imports `conv_preprocess` from the preprocessor registry
   - **Lines 60-103**: Defines `process_sample()` function that:
     - Calls `conv_preprocess()` at line 72 to apply the registered preprocessor
     - Handles image processing and tokenization
     - Returns the processed example ready for training

**3. Configuration**: [configs/multimodal/qwen2_vl/qwen2_vl.yaml](../../configs/multimodal/qwen2_vl/qwen2_vl.yaml)
   - **Line 11**: Specifies `source_name: sharegpt4v_pretrain` which matches the preprocessor name

**4. Preprocessor Definition**: [veomni/data/multimodal/preprocess.py](../../veomni/data/multimodal/preprocess.py)
   - **Lines 41-61**: Defines `sharegpt4v_pretrain_preprocess()` decorated with `@register_preprocessor("sharegpt4v_pretrain")`
   - This preprocessor converts ShareGPT4V data format into VeOmni's standardized conversation format

**5. Registry System**: [veomni/data/multimodal/preprocessor_registry.py](../../veomni/data/multimodal/preprocessor_registry.py)
   - Provides the registration decorator and lookup functions

**Flow Summary**:
```
Config (qwen2_vl.yaml)
  └─> source_name: sharegpt4v_pretrain
       └─> Training Script (train_qwen2_vl.py)
            └─> process_sample() calls conv_preprocess("sharegpt4v_pretrain", ...)
                 └─> Registry looks up sharegpt4v_pretrain_preprocess()
                      └─> Preprocessor (preprocess.py) transforms raw data
                           └─> Returns standardized conversation format
```

## See Also

- [VeOmni preprocess.py](../../veomni/data/multimodal/preprocess.py) - All built-in dataset preprocessors
- [Preprocessor Registry](../../veomni/data/multimodal/preprocessor_registry.py) - The registration system implementation
- [Enabling New Models](./enable_new_models.md) - Tutorial on adding new models
- [Model Loader](./model_loader.md) - Understanding VeOmni's model loading system
