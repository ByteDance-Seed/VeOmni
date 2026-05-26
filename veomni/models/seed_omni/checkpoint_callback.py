"""Per-module checkpoint callback for SeedOmni V2.

Each :class:`OmniModule` participating in the V2 graph owns a dedicated
:class:`OmniModuleCheckpointCallback` instance.  The callback is
trainer-agnostic by design — Step 2 of the migration wires it into
:class:`OmniTrainer.on_step_end` / ``on_train_end``.

Layout produced
---------------
For a model with three modules (``janus_siglip``, ``janus_vqvae``,
``janus_llama``, ``text_encoder``)::

    <save_root>/global_step_{N}/hf_ckpt/
    ├── janus_siglip/
    │   ├── config.json
    │   ├── model.safetensors
    │   └── preprocessor_config.json     # per-module asset (vision processor)
    ├── janus_vqvae/
    │   ├── config.json
    │   ├── model.safetensors
    │   └── preprocessor_config.json
    ├── janus_llama/
    │   ├── config.json
    │   └── model.safetensors            # no per-module asset
    ├── text_encoder/
    │   ├── config.json
    │   └── model.safetensors
    ├── tokenizer/                        # global; written by OmniTrainer top-level callback
    │   ├── tokenizer.json
    │   └── special_tokens_map.json
    └── omni_config.yaml                  # OmniConfig snapshot (top-level)

This matches the design in ``design.md`` §11.

FSDP consolidation contract
---------------------------
The callback writes via :meth:`PreTrainedModel.save_pretrained`, which
expects materialised parameters.  Under FSDP2 the trainer must call
:func:`gather_full_state_dict` (or equivalent) **before** invoking the
callback so the raw module sees fully-rematerialised weights.  Step 2
plumbs this; step 1 only validates the layout / asset wiring against
non-FSDP modules.
"""

from __future__ import annotations

import os
from typing import Optional

import torch.distributed as dist
from transformers import PreTrainedModel
from transformers.processing_utils import ProcessorMixin

from ...utils import helper


logger = helper.create_logger(__name__)


class OmniModuleCheckpointCallback:
    """Save one OmniModule's weights + per-module asset to its subfolder.

    Parameters
    ----------
    module:
        The unwrapped raw :class:`PreTrainedModel` (must inherit
        :class:`OmniModule` mixin).  The trainer is responsible for
        full-state-dict consolidation under FSDP before calling
        :meth:`save`.
    module_name:
        The user-facing name from YAML ``modules.<name>``.  Used as the
        subfolder name in the checkpoint layout.
    processor:
        Optional per-module asset — e.g. the
        :class:`JanusSiglipProcessor` for ``janus_siglip``.  When given,
        ``processor.save_pretrained`` is called alongside the module so
        the next ``from_pretrained`` finds it next to ``config.json``.
        Modules without a per-module asset (``janus_llama``,
        ``text_encoder``) leave this ``None``.
    is_rank_0:
        Whether this process is the global rank-0 writer.  Only rank 0
        actually writes — other ranks no-op (matches the convention of
        :class:`HuggingfaceCkptCallback`).
    """

    def __init__(
        self,
        module: PreTrainedModel,
        module_name: str,
        processor: Optional[ProcessorMixin] = None,
        is_rank_0: bool = True,
    ) -> None:
        self.module = module
        self.module_name = module_name
        self.processor = processor
        self.is_rank_0 = is_rank_0

    def save(self, save_root: str) -> None:
        """Persist the module + asset under ``<save_root>/<module_name>/``.

        ``save_root`` is the global ``hf_ckpt`` directory for the current
        step (e.g. ``<output_dir>/global_step_100/hf_ckpt``).  The
        callback creates ``<save_root>/<module_name>/`` and writes:

          * ``config.json``                 — via ``module.save_pretrained``
          * ``model.safetensors``           — via ``module.save_pretrained``
          * ``preprocessor_config.json``    — via ``processor.save_pretrained``
                                              (when ``self.processor is not None``)

        All ranks call this method but only rank 0 performs IO.  A
        ``dist.barrier`` is invoked at the end (when distributed is
        initialised) to keep the callers in lock-step.
        """
        target = os.path.join(save_root, self.module_name)

        if self.is_rank_0:
            os.makedirs(target, exist_ok=True)
            logger.info_rank0(f"[OmniModuleCheckpointCallback] saving '{self.module_name}' → {target}")
            self.module.save_pretrained(target, safe_serialization=True)
            if self.processor is not None:
                self.processor.save_pretrained(target)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
