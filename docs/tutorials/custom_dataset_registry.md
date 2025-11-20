# Custom Dataset Registry

## Overview

The Custom Dataset Registry provides a plugin-style system for registering custom dataset preprocessors **without modifying `preprocess.py`**. This allows you to add custom datasets while keeping the core code clean.

## Features

- **No Core Code Modification**: Add datasets without touching `preprocess.py`
- **Decorator-based API**: Simple `@register_dataset` decorator for registration
- **Auto-integration**: Automatically adds to VeOmni's `DATASETS` dictionary
- **Backward Compatible**: All existing datasets continue to work

## Quick Start

### 1. Define Your Custom Dataset Preprocessor

Add your preprocessor to [`veomni/data/multimodal/custom_datasets.py`](../../veomni/data/multimodal/custom_datasets.py):

```python
from veomni.data.multimodal.dataset_registry import register_dataset

@register_dataset("my_custom_dataset")
def my_custom_dataset_preprocess(conversations, **kwargs):
    """
    Preprocessor for your custom dataset

    Args:
        conversations: Raw conversation data from your dataset
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
    # Convert your dataset format to VeOmni's format

    return constructed_conversation
```

### 2. Register Custom Datasets

In your training script, register custom datasets before creating the dataset:

```python
from veomni.data.multimodal.dataset_registry import register_all_datasets
import veomni.data.multimodal.custom_datasets  # Import to trigger decorators

# Register all custom datasets
register_all_datasets()

# Now create your dataset
dataset = build_multisource_dataset(...)
```

### 3. Use in Your Config

```yaml
data:
  datasets:
    - name: my_data
      source_name: my_custom_dataset  # Matches @register_dataset name
      data_path: /path/to/my/dataset
      weight: 1.0
```

## Architecture

### Registration Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Define preprocessor with @register_dataset decorator     │
│    └─> Adds to _CUSTOM_DATASETS registry                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Import custom_datasets.py in training script             │
│    └─> Triggers all @register_dataset decorators            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Call register_all_datasets()                             │
│    └─> Adds custom datasets to DATASETS dict                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. VeOmni's dataset loader can now find your dataset        │
│    └─> conv_preprocess(source_name, conversations)          │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
veomni/data/multimodal/
├── preprocess.py           # Built-in datasets (DO NOT MODIFY)
├── dataset_registry.py     # Core registration system
└── custom_datasets.py      # Your custom preprocessors (ADD HERE!)
```

## Preprocessor Format

Your preprocessor must follow VeOmni's conversation format:

```python
# Input: Your dataset's raw format (flexible)
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

### Example 1: VQA Dataset

```python
@register_dataset("custom_vqa")
def custom_vqa_preprocess(conversations, **kwargs):
    """Visual Question Answering dataset"""
    question = conversations[0]["value"].replace("<image>", "").strip()
    answer = conversations[1]["value"]

    return [
        ["user", ("image", None), ("text", question)],
        ["assistant", ("text", answer)]
    ]
```

### Example 2: Multi-turn Conversation

```python
@register_dataset("multi_turn_chat")
def multi_turn_preprocess(conversations, **kwargs):
    """Multi-turn conversation dataset"""
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

### Example 3: OCR Dataset

```python
@register_dataset("custom_ocr")
def custom_ocr_preprocess(conversations, **kwargs):
    """OCR dataset - extract text from images"""
    if isinstance(conversations, str):
        text = conversations
    else:
        text = conversations[-1]["value"]

    return [
        ["user", ("image", None), ("text", "Extract all text from this image.")],
        ["assistant", ("text", text)]
    ]
```

### Example 4: Chart Understanding

```python
@register_dataset("chart_analysis")
def chart_analysis_preprocess(conversations, **kwargs):
    """Chart understanding dataset"""
    role_mapping = {"human": "user", "gpt": "assistant"}
    constructed = []

    for msg in conversations:
        role = role_mapping[msg["from"]]
        value = msg["value"]

        if "<image>" in value:
            value = value.replace("<image>", "").strip()
            if value:
                constructed.append([role, ("image", None), ("text", value)])
            else:
                constructed.append([role, ("image", None)])
        else:
            constructed.append([role, ("text", value)])

    return constructed
```

## Advanced Usage

### Conditional Preprocessing

```python
@register_dataset("adaptive_dataset")
def adaptive_preprocess(conversations, mode="caption", **kwargs):
    """Dataset with different modes"""
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
      source_name: adaptive_dataset
      data_path: /path/to/data
      source_config:
        mode: caption
```

### Random Sampling

```python
import random

@register_dataset("random_prompt")
def random_prompt_preprocess(conversations, **kwargs):
    """Dataset with randomized prompts"""
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
@register_dataset("flexible_format")
def flexible_format_preprocess(conversations, **kwargs):
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

## Testing

Example test for your custom dataset:

```python
def test_custom_dataset():
    from veomni.data.multimodal.dataset_registry import get_custom_datasets
    import veomni.data.multimodal.custom_datasets

    custom_datasets = get_custom_datasets()

    # Test your preprocessor
    test_conversations = [
        {"from": "human", "value": "<image> What is this?"},
        {"from": "gpt", "value": "A cat."}
    ]
    result = custom_datasets["my_custom_dataset"](test_conversations)

    assert result == [
        ["user", ("image", None), ("text", "What is this?")],
        ["assistant", ("text", "A cat.")]
    ]
```

## Troubleshooting

### Dataset Not Found Error

```
ValueError: Unknown dataset name: my_dataset
```

**Solution**:
1. Ensure your dataset is registered in `custom_datasets.py`
2. Import `custom_datasets` before calling `register_all_datasets()`
3. Check that `source_name` in config matches the registered name

### Import Error

```
ImportError: cannot import name 'register_dataset'
```

**Solution**: Make sure VeOmni is properly installed

### Wrong Output Format

```
TypeError: 'NoneType' object is not iterable
```

**Solution**: Ensure your preprocessor returns a list:
```python
# ❌ Wrong
return None

# ✅ Correct
return [["user", ("text", "hello")], ["assistant", ("text", "hi")]]
```

## Best Practices

1. **Naming Convention**: Use descriptive names like `internal_vqa`, `custom_ocr`
2. **Documentation**: Add docstrings explaining expected input/output formats
3. **Error Handling**: Validate input format and provide clear error messages
4. **Testing**: Write tests for your preprocessors before using in production
5. **Reusability**: Extract common logic into helper functions

## See Also

- [VeOmni preprocess.py](../../veomni/data/multimodal/preprocess.py) - Built-in dataset preprocessors
- [Enabling New Models](./enable_new_models.md) - Tutorial on adding new models
- [Model Loader](./model_loader.md) - Understanding VeOmni's model loading system
