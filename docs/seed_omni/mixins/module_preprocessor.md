# Module CPU Preprocessors (`ModulePreprocessorBase`)

`ModulePreprocessorBase` is the per-module CPU worker contract for SeedOmni. Each
module that needs CPU-side input preparation (chat-template application,
tokenization, image normalization, `_omni_token_len` tagging, or per-sample FSDP
dummy-anchor injection) subclasses
[`ModulePreprocessorBase`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L55)
in its `processing.py` and points its native model class's `preprocessor_class`
at that subclass.

- Base class: [`ModulePreprocessorBase`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L55)
- Asset binding helper: [`bind_module_assets`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L147)
- Composed orchestrator: [`OmniProcessor`](../../../veomni/models/seed_omni/processing_omni.py#L78)
- Dataset transform caller: [`process_seedomni_example`](../../../veomni/data/seed_omni/seedomni_transform.py#L183) / [`process_seedomni_cached_example`](../../../veomni/data/seed_omni/seedomni_transform.py#L254)
- Collator: [`SeedOmniCollator`](../../../veomni/data/seed_omni/collator.py#L22)

## Why preprocessing is per-module, per-sample, and CPU-only

The global data transform ([`process_seedomni_example`](../../../veomni/data/seed_omni/seedomni_transform.py#L183))
decodes raw media, constructs a `conversation_list` of
[`ConversationItem`](../../../veomni/models/seed_omni/utils/conversation.py#L83)
objects, and immediately runs the bound [`OmniProcessor`](../../../veomni/models/seed_omni/processing_omni.py#L78)
per sample at the Dataset stage before dynamic batching (`dyn_bsz`) and collation.
Only each module knows which items it consumes, how its chat template or image
processor transforms them, how many sequence tokens each item contributes
([`_OMNI_TOKEN_LEN`](../../../veomni/models/seed_omni/utils/conversation.py#L79)),
and what dummy placeholder shape its FSDP wrapper needs when a sample has no
real items for that modality.

Running that preparation inside [`ModulePreprocessorBase`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L55)
at the Dataset stage keeps heavy CPU work (tokenization, image resize/normalize)
inside DataLoader worker processes, populates per-item `_omni_token_len` and
`labels` metadata so `dyn_bsz` (`count_mode: total` or `effective`) can size each
sample before batching, and shares the exact same preprocessing chain with
single-request inference via [`OmniProcessor`](../../../veomni/models/seed_omni/processing_omni.py#L78).

## Opting in

Define a `ModulePreprocessorBase` subclass in `modules/<family>/<sub>/processing.py`
and attach it as `preprocessor_class` on the native model class in `modeling.py`:

```python
from typing import Any

import torch

from veomni.models.seed_omni.modules.module_processing_base import ModulePreprocessorBase
from veomni.models.seed_omni.utils.conversation import (
    _IMG_TAG_KEY,
    _OMNI_TOKEN_LEN,
    ConversationItem,
    iter_desired_items,
)


class VisionPreprocessor(ModulePreprocessorBase):
    def __init__(
        self,
        image_processor: Any,
        dtype: torch.dtype = torch.bfloat16,
        num_image_tokens: int = 576,
    ) -> None:
        self._image_processor = image_processor
        self.dtype = dtype
        self._dummy_pixel_values: torch.Tensor | None = None
        self._num_image_tokens = int(num_image_tokens)

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
        self._num_image_tokens = (config.image_size // config.patch_size) ** 2

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs
        for sample in conversation_list:
            user_images = [
                it
                for it in iter_desired_items([sample], types=["image"], roles=["user"])
                if not it.is_dummy
            ]
            if user_images:
                for item in user_images:
                    item.value = self._image_processor(item.value)
                    item.meta[_OMNI_TOKEN_LEN] = self._num_image_tokens
            elif not inference and self._dummy_pixel_values is not None:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=self._dummy_pixel_values.clone(),
                        role="user",
                        is_dummy=True,
                        meta={_IMG_TAG_KEY: "und", _OMNI_TOKEN_LEN: self._num_image_tokens},
                    )
                )
```

## The functions

### What a module implements or overrides

| Function | Module's job | Called by | Effect |
|----------|--------------|-----------|--------|
| [`preprocess_conversations(conversation_list, inference=False, **kwargs)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L91) | **Must implement.** Mutate matching `ConversationItem.value` / `meta` in place across `conversation_list` (`list[list[ConversationItem]]`), set `item.meta[_OMNI_TOKEN_LEN]`, and append a per-sample `is_dummy=True` placeholder when `not inference` and a sample lacks this module's modality. | [`ModulePreprocessorBase.__call__`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L87) | Prepares CPU tensors and `dyn_bsz` length metadata for `DynamicBatchingSizeDataset` / `TextBatchingStrategy` and the module's `pre_forward` / `generate`. Not implementing it raises `NotImplementedError` when the preprocessor runs. |
| [`from_pretrained(module_path, *, config_overrides=None, **kwargs)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L101) | **May override.** Build and return the CPU preprocessor from the module's checkpoint subfolder without instantiating the `nn.Module`. Default returns `None`. | [`OmniProcessor.from_config`](../../../veomni/models/seed_omni/processing_omni.py#L109) and [`bind_module_assets`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L147) | Returning `None` means the module contributes no CPU preprocessor (for example a pure AR backbone). |
| [`bind_dummy_inputs(config, dtype=None)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L121) | **May override.** Pre-compute dummy tensor(s) and token lengths from the module's resolved `config` and `dtype` for the `inference=False` branch. Default is a no-op. | [`OmniProcessor.bind_dummy_inputs`](../../../veomni/models/seed_omni/processing_omni.py#L88) during [`OmniTrainer._build_data_transform`](../../../veomni/trainer/omni/omni_trainer.py#L288) | Attaches FSDP-anchor dummy shapes before DataLoader workers fork/spawn. |

### What the framework provides (do not override)

| Function | Called by | Effect |
|----------|-----------|--------|
| [`ModulePreprocessorBase.__call__(batch, inference=False, **kwargs)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L87) | [`OmniProcessor.preprocess_batch`](../../../veomni/models/seed_omni/processing_omni.py#L228) | Delegates `batch["conversation_list"]` to `preprocess_conversations` in `config.module_names` order. |
| [`ModulePreprocessorBase.append_batch_anchor(conversation_list, item)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L130) | Module's `preprocess_conversations` when `inference=False` and no real input for this module exists in the micro-batch | Appends `item` (`is_dummy=True`) to `conversation_list[0]` — a single anchor row per micro-batch (or per sample during `seedomni_transform`), keeping FSDP2 collectives aligned. |
| [`bind_module_assets(model, *, checkpoint_path=None, preprocessor=None, config_overrides=None)`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L147) | [`PretrainedOmniModule.from_pretrained`](../../../veomni/models/seed_omni/modules/module_modeling_base.py#L36) and [`ModuleRuntime._build_model_assets`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L354) | Copies CPU asset attributes (`_processor`, `_image_processor`, `_video_processor`, `_tokenizer`, `_chat_template`) from the preprocessor onto `model` if not already set, so `generate()` and HF checkpoint export can access them. |

## Call flow

- **Training sample & micro-batch**:
  - [`OmniTrainer._build_data_transform`](../../../veomni/trainer/omni/omni_trainer.py#L288) builds [`OmniProcessor.from_config`](../../../veomni/models/seed_omni/processing_omni.py#L109), calls [`OmniProcessor.bind_dummy_inputs`](../../../veomni/models/seed_omni/processing_omni.py#L88), and passes `processor` into [`process_seedomni_example`](../../../veomni/data/seed_omni/seedomni_transform.py#L183) / [`process_seedomni_cached_example`](../../../veomni/data/seed_omni/seedomni_transform.py#L254).
  - For each dataset sample, `seedomni_transform` calls [`OmniProcessor.preprocess(conversation_list)`](../../../veomni/models/seed_omni/processing_omni.py#L216) → [`OmniProcessor.preprocess_batch(..., inference=False)`](../../../veomni/models/seed_omni/processing_omni.py#L228), running each module's [`ModulePreprocessorBase.__call__`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L87) → [`preprocess_conversations`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L91) to mutate items in place, populate `_omni_token_len` / `labels`, and append per-sample `is_dummy=True` placeholders.
  - When `train.dyn_bsz: true`, [`get_length_fn`](../../../veomni/data/dataset.py#L444) (`count_mode="total"` and `physical_token_cap`, including `is_dummy=True` items) and [`get_length_by_labels_fn`](../../../veomni/data/dataset.py#L476) (`count_mode="effective"`, excluding `is_dummy=True` items) read those metadata fields to batch samples by token budget, and [`SeedOmniCollator.__call__`](../../../veomni/data/seed_omni/collator.py#L63) groups the selected sample dicts into a micro-batch.
- **Inference request**:
  - [`OmniInferencer._build_model`](../../../veomni/trainer/omni/omni_inferencer.py#L120) builds [`OmniProcessor.from_config`](../../../veomni/models/seed_omni/processing_omni.py#L109).
  - [`OmniInferencer._run`](../../../veomni/trainer/omni/omni_inferencer.py#L237) calls [`OmniProcessor.__call__(..., inference=True)`](../../../veomni/models/seed_omni/processing_omni.py#L167) → [`OmniProcessor.preprocess`](../../../veomni/models/seed_omni/processing_omni.py#L216) → [`OmniProcessor.preprocess_batch(..., inference=True)`](../../../veomni/models/seed_omni/processing_omni.py#L228).

## Rules an implementation must follow

1. **Keep preprocessors picklable, weight-free, and CPU-only.** DataLoader workers fork or spawn with a pickled copy of `OmniProcessor`; never store an `nn.Module` or allocate CUDA tensors inside a preprocessor.
2. **Skip existing dummy rows when selecting real inputs.** Filter with `not item.is_dummy` so a dummy item injected earlier does not get re-processed as real data.
3. **Populate `item.meta[_OMNI_TOKEN_LEN]` and inject `is_dummy=True` anchors per sample.** Because `OmniProcessor.preprocess` runs on each sample individually in `seedomni_transform` before `dyn_bsz`, iterate `for sample in conversation_list:`, set `item.meta[_OMNI_TOKEN_LEN]` on every real and dummy item, and append missing-modality `is_dummy=True` placeholders to `sample` when `not inference`.
4. **Honour `config_overrides` in `from_pretrained`.** Apply any `model_config` overrides passed via `config_overrides` on top of the on-disk `config.json` so the worker preprocessor matches the live model's configuration.

## Tests

- [`tests/seed_omni/data/test_preprocessor.py`](../../../tests/seed_omni/data/test_preprocessor.py):
  - [strict key alignment across collated samples](../../../tests/seed_omni/data/test_preprocessor.py#L91),
  - [mixed-resolution images preserved unstacked and normalized per item](../../../tests/seed_omni/data/test_preprocessor.py#L101),
  - [per-sample dummy anchor injection when a modality is absent](../../../tests/seed_omni/data/test_preprocessor.py#L145), and
  - [`seedomni_transform` runs `OmniProcessor` and populates `dyn_bsz` total/effective lengths](../../../tests/seed_omni/data/test_preprocessor.py#L164).
- [`tests/seed_omni/model/test_processor.py`](../../../tests/seed_omni/model/test_processor.py):
  - [preprocessors run in module order](../../../tests/seed_omni/model/test_processor.py#L36),
  - [`model_config` and `processor_config` overrides reach `from_pretrained`](../../../tests/seed_omni/model/test_processor.py#L162), and
  - [`bind_module_assets` copies missing assets idempotently](../../../tests/seed_omni/model/test_processor.py#L230).
