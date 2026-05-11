"""
OmniModel V2 — composable multi-modal training + inference model.

Architecture
------------

  OmniModel
  ├── modules: nn.ModuleDict[str, OmniModule]   (one entry per named module)
  ├── graph:   OmniGraph                         (training DAG, topo-sorted)
  └── fsm:     GenerationStateMachine            (inference FSM, optional)

Training (forward)
------------------
  1. OmniGraph.execution_order gives the topological ordering.
  2. For each module, collect_inputs() merges raw-batch + connection-routed
     tensors from earlier modules.
  3. Call module.forward(**inputs).
  4. Any key ending in ``_loss`` in the return dict is summed into total loss.
  5. Return {"loss": total_loss, **all_module_outputs_flat}.

Inference (generate)
--------------------
  1. GenerationStateMachine.reset(request) initialises the FSM.
  2. Loop: fsm.step(modules, context) → execute current state's body.
  3. fsm.maybe_transition(context) → check & apply state transitions.
  4. Stop when fsm.is_done() or a caller-supplied stop condition fires.

Per-module FSDP
---------------
OmniModel itself is NOT FSDP-wrapped as a monolith.  Instead, each
OmniModule inside self.modules is individually wrapped by the trainer
(or by OmniModel._wrap_modules_fsdp2 if called explicitly).  This lets
each module have its own sharding / mixed-precision / EP plan.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from .configuration_seed_omni import OmniConfig
from .generation import GenerationStateMachine
from .graph import OmniGraph
from .module import OmniModule


class OmniModel(nn.Module):
    """Composable multi-modal model driven by config-specified graphs.

    Parameters
    ----------
    config:
        :class:`OmniConfig` instance.
    module_factory:
        Callable ``(module_name: str, module_cfg: dict) -> OmniModule``.
        Receives the raw module config dict (including ``model_type``).
        Used to instantiate each sub-module.  If ``None``, sub-modules are
        *not* built (useful when loading a pre-saved OmniModel checkpoint
        that already contains ``self.modules``).
    """

    config_class = OmniConfig

    def __init__(
        self,
        config: OmniConfig,
        module_factory: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config

        # ── Build sub-modules ────────────────────────────────────────
        if module_factory is not None:
            built = {}
            for name in config.module_names:
                mod_cfg = config.module_config(name)
                built[name] = module_factory(name, mod_cfg)
            self.modules_dict = nn.ModuleDict(built)
        else:
            self.modules_dict = nn.ModuleDict()

        # ── Build training graph ─────────────────────────────────────
        self.graph = OmniGraph(
            connections=config.connections,
            training_connections=config.training_connections,
        )

        # ── Build inference FSM (optional) ───────────────────────────
        if config.has_generation_states():
            self.fsm = GenerationStateMachine(
                fsm_config=config.generation_states,
                connections=config.connections,
            )
        else:
            self.fsm = None

    # ── Training ──────────────────────────────────────────────────────────────

    def forward(self, **batch) -> Dict[str, Any]:
        """Execute the training DAG and return the aggregated loss.

        The raw batch is globally visible to every module.  Connection-routed
        tensors from earlier modules are merged on top (override raw-batch
        keys if names collide).

        Returns
        -------
        dict with at least ``"loss"`` (scalar tensor) plus every module's
        output dict under key ``"{module_name}_out"``.
        """
        module_outputs: Dict[str, Dict[str, Any]] = {}
        total_loss: Optional[torch.Tensor] = None
        loss_dict: Dict[str, torch.Tensor] = {}

        for module_name in self.graph.execution_order:
            module = self.modules_dict[module_name]
            inputs = self.graph.collect_inputs(module_name, module_outputs, batch)
            out = module(**inputs)
            module_outputs[module_name] = out

            # Collect losses
            for key, val in out.items():
                if key.endswith("_loss") and isinstance(val, torch.Tensor):
                    loss_dict[f"{module_name}/{key}"] = val
                    total_loss = val if total_loss is None else total_loss + val

        result: Dict[str, Any] = {"loss": total_loss, "loss_dict": loss_dict}
        for name, out in module_outputs.items():
            result[f"{name}_out"] = out
        return result

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 512,
        stop_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run inference using the FSM.

        Parameters
        ----------
        request:
            Generation request dict (e.g. ``{"input_ids": ..., "max_new_tokens": 512}``).
            Passed to the FSM for ``token_length.from_request`` resolution.
        context:
            Initial generation context.  If ``None``, starts from ``request``.
        max_new_tokens:
            Hard upper bound on total generation steps across all states.
        stop_token_ids:
            Token IDs that unconditionally stop generation.

        Returns
        -------
        Final context dict containing all generated outputs.
        """
        if self.fsm is None:
            raise RuntimeError(
                "OmniModel has no GenerationStateMachine. "
                "Set generation_states in OmniConfig to enable inference."
            )

        self.fsm.reset(request=request)
        ctx: Dict[str, Any] = dict(context or request)

        modules = dict(self.modules_dict)
        total_steps = 0

        while not self.fsm.is_done() and total_steps < max_new_tokens:
            ctx = self.fsm.step(modules, ctx)
            total_steps += 1

            # Hard stop on special tokens
            if stop_token_ids:
                last_id = ctx.get("last_token_id")
                if last_id is not None and last_id in stop_token_ids:
                    break

            self.fsm.maybe_transition(ctx)

        return ctx

    # ── Utilities ─────────────────────────────────────────────────────────────

    def named_omni_modules(self) -> Iterator[Tuple[str, OmniModule]]:
        """Yield (name, module) for every OmniModule in modules_dict."""
        for name, mod in self.modules_dict.items():
            if isinstance(mod, OmniModule):
                yield name, mod

    def get_module(self, name: str) -> OmniModule:
        mod = self.modules_dict.get(name)
        if mod is None:
            raise KeyError(f"Module '{name}' not found in OmniModel")
        return mod

    def wrap_modules_fsdp2(
        self,
        mesh=None,
        mixed_precision_policy=None,
    ) -> "OmniModel":
        """Apply FSDP2 ``fully_shard`` to each sub-module independently.

        This is called by the OmniTrainer after model construction.  It wraps
        each OmniModule's internal layers (``_no_split_modules``) first, then
        the OmniModule root, so FSDP shards at the right granularity.

        Returns self for chaining.
        """
        from torch.distributed._composable.fsdp import fully_shard

        fsdp_kwargs: Dict[str, Any] = {}
        if mesh is not None:
            fsdp_kwargs["mesh"] = mesh
        if mixed_precision_policy is not None:
            fsdp_kwargs["mp_policy"] = mixed_precision_policy

        for _name, mod in self.modules_dict.items():
            no_split = getattr(mod, "_no_split_modules", [])
            # Shard inner layers first (bottom-up)
            for _subname, submod in mod.named_modules():
                if submod is mod:
                    continue
                if type(submod).__name__ in no_split:
                    # use per-module plan if available
                    plan = mod.get_parallel_plan()
                    kwargs = dict(fsdp_kwargs)
                    if plan is not None and hasattr(plan, "fsdp_kwargs"):
                        kwargs.update(plan.fsdp_kwargs)
                    fully_shard(submod, **kwargs)
            # Shard the OmniModule root
            plan = mod.get_parallel_plan()
            kwargs = dict(fsdp_kwargs)
            if plan is not None and hasattr(plan, "fsdp_kwargs"):
                kwargs.update(plan.fsdp_kwargs)
            fully_shard(mod, **kwargs)

        return self

    # ── nn.Module overrides ───────────────────────────────────────────────────

    def parameters(self, recurse: bool = True):
        return self.modules_dict.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self.modules_dict.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )
