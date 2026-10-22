# Module CPU Preprocessors (`ModulePreprocessorBase`)

`ModulePreprocessorBase` is the per-module CPU worker contract for SeedOmni. Each
module that needs CPU-side input preparation (chat-template application,
tokenization, image normalization, or FSDP dummy-anchor injection) subclasses
[`ModulePreprocessorBase`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L55)
in its `processing.py` and points its native model class's `preprocessor_class`
at that subclass.

- Base class: [`ModulePreprocessorBase`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L55)
- Asset binding helper: [`bind_module_assets`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L145)
- Composed orchestrator: [`OmniProcessor`](../../../veomni/models/seed_omni/processing_omni.py#L81)
- Collator caller: [`SeedOmniCollator`](../../../veomni/data/seed_omni/collator.py#L26)

## Why preprocessing is per-module and CPU-only

The global data transform ([`process_seedomni_example`](../../../veomni/data/seed_omni/seedomni_transform.py#L183))
decodes raw media and produces a model-agnostic `conversation_list` of
[`ConversationItem`](../../../veomni/models/seed_omni/utils/conversation.py#L78)
objects. Only each module knows which items it consumes, how its chat template or
image processor transforms them, and what dummy placeholder shape its FSDP
wrapper needs when a micro-batch has no real items for that modality.

Running that preparation inside [`ModulePreprocessorBase`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L55)
keeps heavy CPU work (tokenization, image resize/normalize) inside DataLoader
worker processes during training while sharing the exact same preprocessing chain
with single-request inference via [`OmniProcessor`](../../../veomni/models/seed_omni/processing_omni.py#L81).

## Opting in

Define a `ModulePreprocessorBase` subclass in `modules/<family>/<sub>/processing.py`
and attach it as `preprocessor_class` on the native model class in `modeling.py`:

```python
from typing import Any

import torch

from veomni.models.seed_omni.modules.module_processing_base import ModulePreprocessorBase
from veomni.models.seed_omni.utils.conversation import (
    _IMG_TAG_KEY,
    ConversationItem,
    iter_desired_items,
)


class VisionPreprocessor(ModulePreprocessorBase):
    def __init__(self, image_processor: Any, dtype: torch.dtype = torch.bfloat16) -> None:
        self._image_processor = image_processor
        self.dtype = dtype
        self._dummy_pixel_values: torch.Tensor | None = None

    @classmethod
    def from_pretrained(
        cls, module_path: str, *, config_overrides: dict[str, Any] | None = None, **kwargs: Any
    ) -> "VisionPreprocessor":
        del kwargs
        config = VisionConfig.from_pretrained(module_path, model_config=config_overrides)
        image_processor = ...  # build CPU-safe processor from module_path / config
        return cls(image_processor=image_processor)

    def bind_dummy_inputs(self, config: Any, dtype: Any = None) -> None:
        self._dummy_pixel_values = torch.zeros(3, config.image_size, config.image_size, dtype=dtype or self.dtype)

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs
        user_images = [
            it
            for it in iter_desired_items(conversation_list, types=["image"], roles=["user"])
            if not it.is_dummy
        ]
        for item in user_images:
            item.value = self._image_processor(item.value)
        if not user_images and not inference and self._dummy_pixel_values is not None:
            self.append_batch_anchor(
                conversation_list,
                ConversationItem(
                    type="image",
                    value=self._dummy_pixel_values.clone(),
                    role="user",
                    is_dummy=True,
                    meta={_IMG_TAG_KEY: "und"},
                ),
            )
```

## The functions

### What a module implements or overrides

| Function | Module's job | Called by | Effect |
|----------|--------------|-----------|--------|
| [`preprocess_conversations(conversation_list, inference=False, **kwargs)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L89) | **Must implement** (unless overriding `__call__`). Mutate matching `ConversationItem.value` / `meta` in place across `conversation_list` (`list[list[ConversationItem]]`). When `inference=False` and no real item matches this module, append a dummy placeholder via [`append_batch_anchor`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L128). | [`ModulePreprocessorBase.__call__`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L85) | Prepares CPU tensors for the module's `pre_forward` / `generate`. Not implementing it raises `NotImplementedError` when the preprocessor runs. |
| [`from_pretrained(module_path, *, config_overrides=None, **kwargs)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L99) | **May override.** Build and return the CPU preprocessor from the module's checkpoint subfolder without instantiating the `nn.Module`. Default returns `None`. | [`OmniProcessor.from_config`](../../../veomni/models/seed_omni/processing_omni.py#L112) and [`bind_module_assets`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L145) | Returning `None` means the module contributes no CPU preprocessor (for example a pure AR backbone). |
| [`bind_dummy_inputs(config, dtype=None)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L119) | **May override.** Pre-compute dummy tensor(s) from the module's resolved `config` and `dtype` for the `inference=False` branch. Default is a no-op. | [`OmniProcessor.bind_dummy_inputs`](../../../veomni/models/seed_omni/processing_omni.py#L91) during [`OmniTrainer._build_collate_fn`](../../../veomni/trainer/omni/omni_trainer.py#L306) | Attaches FSDP-anchor dummy shapes before DataLoader workers fork/spawn. |
| [`__call__(batch, inference=False, **kwargs)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L85) | **May override** for packed preprocessors that write extra batch-level tensors onto `batch` in addition to or instead of mutating `batch["conversation_list"]`. Default delegates `batch["conversation_list"]` to `preprocess_conversations`. | [`OmniProcessor.preprocess_batch`](../../../veomni/models/seed_omni/processing_omni.py#L231) | Runs this module's CPU preparation step in `config.module_names` order. |

### What the framework provides (do not override)

| Function | Called by | Effect |
|----------|-----------|--------|
| [`ModulePreprocessorBase.append_batch_anchor(conversation_list, item)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L128) | Module's `preprocess_conversations` when `inference=False` and no real input for this module exists in the micro-batch | Appends `item` (`is_dummy=True`) to `conversation_list[0]` — a single anchor row per micro-batch rather than one per sample, keeping FSDP2 collectives aligned without running `N` dummy forward passes. |
| [`bind_module_assets(model, *, checkpoint_path=None, preprocessor=None, config_overrides=None)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L145) | [`PretrainedOmniModule.from_pretrained`](../../../veomni/models/seed_omni/modules/module_modeling_base.py#L36) and [`ModuleRuntime._build_model_assets`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L356) | Copies CPU asset attributes (`_processor`, `_image_processor`, `_video_processor`, `_tokenizer`, `_chat_template`) from the preprocessor onto `model` if not already set, so `generate()` and HF checkpoint export can access them. |

## Call flow

- **Training micro-batch**:
  - [`OmniTrainer._build_collate_fn`](../../../veomni/trainer/omni/omni_trainer.py#L306) builds [`OmniProcessor.from_config`](../../../veomni/models/seed_omni/processing_omni.py#L112), calls [`OmniProcessor.bind_dummy_inputs`](../../../veomni/models/seed_omni/processing_omni.py#L91), and wraps it in [`SeedOmniCollator`](../../../veomni/data/seed_omni/collator.py#L26).
  - [`SeedOmniCollator.__call__`](../../../veomni/data/seed_omni/collator.py#L74) verifies key alignment across samples, groups `conversation_list`, and calls [`OmniProcessor.preprocess_batch(..., inference=False)`](../../../veomni/models/seed_omni/processing_omni.py#L231).
  - Each module's [`ModulePreprocessorBase.__call__`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L85) calls [`preprocess_conversations`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L89), mutating items in place and appending at most one batch anchor via [`append_batch_anchor`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L128).
- **Inference request**:
  - [`OmniInferencer._build_model`](../../../veomni/trainer/omni/omni_inferencer.py#L120) builds [`OmniProcessor.from_config`](../../../veomni/models/seed_omni/processing_omni.py#L112).
  - [`OmniInferencer._run`](../../../veomni/trainer/omni/omni_inferencer.py#L237) calls [`OmniProcessor.__call__(..., inference=True)`](../../../veomni/models/seed_omni/processing_omni.py#L170) → [`OmniProcessor.preprocess`](../../../veomni/models/seed_omni/processing_omni.py#L219) → [`OmniProcessor.preprocess_batch(..., inference=True)`](../../../veomni/models/seed_omni/processing_omni.py#L231).

## Rules an implementation must follow

1. **Keep preprocessors picklable, weight-free, and CPU-only.** DataLoader workers fork or spawn with a pickled copy of `OmniProcessor`; never store an `nn.Module` or allocate CUDA tensors inside a preprocessor.
2. **Skip existing dummy rows when selecting real inputs.** Filter with `not item.is_dummy` so a dummy item injected earlier does not get re-processed as real data.
3. **Use `append_batch_anchor` for missing-modality FSDP anchors.** Only inject dummy items when `not inference`, and attach a single micro-batch anchor rather than one per sample.
4. **Honour `config_overrides` in `from_pretrained`.** Apply any `model_config` overrides passed via `config_overrides` on top of the on-disk `config.json` so the worker preprocessor matches the live model's configuration.

## Tests

- [`tests/seed_omni/data/test_preprocessor.py`](../../../tests/seed_omni/data/test_preprocessor.py):
  - [strict key alignment across collated samples](../../../tests/seed_omni/data/test_preprocessor.py#L52),
  - [mixed-resolution images preserved unstacked and normalized per item](../../../tests/seed_omni/data/test_preprocessor.py#L62), and
  - [`append_batch_anchor` attaches a single dummy anchor row when a modality is absent](../../../tests/seed_omni/data/test_preprocessor.py#L101).
- [`tests/seed_omni/model/test_processor.py`](../../../tests/seed_omni/model/test_processor.py):
  - [preprocessors run in module order](../../../tests/seed_omni/model/test_processor.py#L36),
  - [`model_config` and `processor_config` overrides reach `from_pretrained`](../../../tests/seed_omni/model/test_processor.py#L162), and
  - [`bind_module_assets` copies missing assets idempotently](../../../tests/seed_omni/model/test_processor.py#L230).
