"""
OmniModule: abstract base class for all composable modules in OmniModel.

Each OmniModule:
  - Implements `forward(**kwargs) -> dict` for training (DAG execution).
  - Optionally overrides `generate_step(**kwargs) -> dict` for inference (FSM execution).
  - Optionally overrides `get_parallel_plan()` for per-module FSDP/EP/SP config.

Return convention:
  - Any key ending with ``_loss`` in the return dict is treated as a training loss
    and collected by OmniModel.forward().
  - Keys used as outputs in connections must be present in the return dict.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch.nn as nn


class OmniModule(nn.Module, ABC):
    """Abstract base class for all OmniModel sub-modules.

    Subclasses must implement :meth:`forward`.  The ``generate_step`` method
    defaults to delegating to ``forward``, which is correct for encoder-style
    modules and for the AR-LLM in teacher-forcing mode.  Modules that need
    different sampling logic (e.g. a DiT denoising step) should override
    ``generate_step``.

    Loss convention
    ---------------
    ``forward`` should return a :class:`dict`.  Any key whose name ends with
    ``_loss`` (e.g. ``"lm_loss"``, ``"diffusion_loss"``) is automatically
    aggregated by :class:`~veomni.models.seed_omni.modeling_omni.OmniModel`.

    Parallel plan
    -------------
    Override :meth:`get_parallel_plan` to return a VeOmni ``ParallelPlan``
    (or equivalent) if this module requires non-default FSDP / SP / EP
    sharding.  Returning ``None`` means default sharding inherited from the
    OmniModel level.
    """

    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, Any]:
        """Training forward pass.

        Args:
            **kwargs: Union of raw-batch fields (globally transparent) and
                connection-routed tensors injected by OmniGraph.

        Returns:
            dict with arbitrary keys.  Keys ending in ``_loss`` are treated
            as scalar loss terms and summed into the total training loss.
        """

    def generate_step(self, **kwargs) -> Dict[str, Any]:
        """Single auto-regressive or diffusion generation step.

        Defaults to calling :meth:`forward`.  Override for modules where
        inference and training behave differently (e.g. a DiT that runs a
        full denoising loop during generation but computes diffusion loss
        during training, or a sampling-based next-token predictor).

        Args:
            **kwargs: Generation context accumulated by the FSM, including
                the raw request dict and all outputs produced so far in the
                current state body.

        Returns:
            dict that is merged back into the FSM context for the next step.
        """
        return self.forward(**kwargs)

    def get_parallel_plan(self) -> Optional[Any]:
        """Return a per-module VeOmni parallel plan, or ``None`` for default."""
        return None
