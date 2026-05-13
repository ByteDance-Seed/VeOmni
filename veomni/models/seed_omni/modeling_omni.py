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
  3. The global batch is split into micro-batches of size
     ``module_config["micro_batch_size"]`` (default: full batch).
  4. For each micro-batch:
       a. module.pre_forward(**inputs)   — packing + SP slice
       b. module(**processed_inputs)     — computation
       c. module.post_forward(outputs)   — SP gather
  5. Micro-batch outputs are aggregated (tensors concatenated along dim 0,
     ``*_loss`` keys averaged).
  6. Any key ending in ``_loss`` in the aggregated output is summed into
     total loss.
  7. Return {"loss": total_loss, **all_module_outputs_flat}.

Inference (generate)
--------------------
  1. GenerationStateMachine.reset(request) initialises the FSM.
  2. Loop: fsm.step(modules, context) → execute current state's body.
  3. fsm.maybe_transition(context) → check & apply state transitions.
  4. Stop when fsm.is_done() or a caller-supplied stop condition fires.

Per-module parallelism
----------------------
Each module's config dict (in OmniConfig.modules) may include:

  ``micro_batch_size``: int, how many samples per forward call (default: full batch)
  ``weights_path``: optional per-module weight path (loaded by ``build_parallelize_model``).

OmniModel itself is NOT wrapped as a monolith.  Use
:meth:`OmniModel.build_from_args`, which calls :meth:`OmniModule.build` per
sub-module and delegates to :func:`veomni.distributed.torch_parallelize.build_parallelize_model`
for DDP / FSDP1 / FSDP2 wrapping (driven by the global ``parallel_state.dp_mode``).
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from .configuration_seed_omni import OmniConfig
from .generation import GenerationStateMachine
from .graph import OmniGraph
from .module import OmniBuildArgs, OmniModule


_DEFAULT_MICRO_BS = 2**31  # effectively "full batch" when not configured


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
        """Execute the training DAG with per-module micro-batch loops.

        For each module in topological order:
          1. ``collect_inputs`` assembles the full-batch inputs.
          2. The batch is chunked into micro-batches of ``micro_batch_size``.
          3. Each chunk runs ``pre_forward → forward → post_forward``.
          4. Chunk outputs are aggregated (concat tensors, mean losses).
          5. The aggregated output is stored for downstream connection routing.

        Returns
        -------
        dict with at least ``"loss"`` (scalar) and ``"loss_dict"``
        (per-module breakdown), plus ``"{module_name}_out"`` for each module.
        """
        module_outputs: Dict[str, Dict[str, Any]] = {}
        total_loss: Optional[torch.Tensor] = None
        loss_dict: Dict[str, torch.Tensor] = {}

        for module_name in self.graph.execution_order:
            module = self.modules_dict[module_name]
            # Unwrap DDP to access OmniModule hooks; forward() still goes through
            # the wrapper so DDP's gradient all-reduce fires correctly.
            omni_mod = _unwrap_module(module)
            full_inputs = self.graph.collect_inputs(module_name, module_outputs, batch)

            micro_bs = self._get_micro_batch_size(module_name)
            batch_size = _infer_batch_size(full_inputs)

            chunk_outputs: List[Dict[str, Any]] = []
            for start in range(0, batch_size, micro_bs):
                end = min(start + micro_bs, batch_size)
                chunk_inputs = _slice_batch(full_inputs, start, end)

                chunk_inputs = omni_mod.pre_forward(**chunk_inputs)
                out = module(**chunk_inputs)  # DDP wrapper → triggers all-reduce
                out = omni_mod.post_forward(out)
                chunk_outputs.append(out)

            agg_out = _aggregate_outputs(chunk_outputs)
            module_outputs[module_name] = agg_out

            # Collect losses (already averaged across micro-batches)
            for key, val in agg_out.items():
                if key.endswith("_loss") and isinstance(val, torch.Tensor):
                    loss_key = f"{module_name}/{key}"
                    loss_dict[loss_key] = val
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
                "OmniModel has no GenerationStateMachine. Set generation_states in OmniConfig to enable inference."
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

    # ── Factory: build from trainer args ──────────────────────────────────────

    @classmethod
    def build_from_args(cls, config: OmniConfig, build_args: OmniBuildArgs) -> "OmniModel":
        """Build OmniModel with all sub-modules constructed and parallelized.

        This is the primary entry point for trainer-side construction.  For each
        module named in ``config``:

        1. Look up the ``OmniModule`` subclass from the global
           :data:`~veomni.models.seed_omni.modules.MODULE_REGISTRY` using
           ``config.modules[name]["model_type"]``.
        2. Call :meth:`OmniModule.build` which handles weight loading and
           parallelisation via :func:`build_parallelize_model`.

        The resulting ``OmniModel`` has fully parallelized modules in
        ``modules_dict`` — no separate parallelisation step is needed.

        Parameters
        ----------
        config:
            Parsed :class:`OmniConfig` (e.g. from ``janus_1.3b.yaml``).
        build_args:
            Parallelisation settings from :class:`OmniBuildArgs`.

        Returns
        -------
        A fully built and parallelized :class:`OmniModel`.
        """
        from veomni.utils import logging as ve_logging

        from .modules import MODULE_REGISTRY

        logger = ve_logging.get_logger(__name__)

        model = cls.__new__(cls)
        super(OmniModel, model).__init__()
        model.config = config

        built = {}
        for name in config.module_names:
            mod_cfg = config.module_config(name)
            model_type = mod_cfg.get("model_type", "")
            module_cls = MODULE_REGISTRY.get(model_type)
            if module_cls is None:
                raise ValueError(
                    f"Unknown model_type {model_type!r} for module '{name}'. Available: {list(MODULE_REGISTRY)}"
                )
            logger.info_rank0(f"Building module [{name}] (model_type={model_type})")
            built[name] = module_cls.build(mod_cfg, build_args)

        model.modules_dict = nn.ModuleDict(built)
        model.graph = OmniGraph(
            connections=config.connections,
            training_connections=config.training_connections,
        )
        model.fsm = (
            GenerationStateMachine(
                fsm_config=config.generation_states,
                connections=config.connections,
            )
            if config.has_generation_states()
            else None
        )
        return model

    def collect_assets(self) -> List[Any]:
        """Collect model assets from all sub-modules.

        Iterates over ``modules_dict``, unwrapping DDP/FSDP wrappers, and
        calls :meth:`OmniModule.get_assets` on each raw ``OmniModule``.
        Returns the concatenated list of all assets (tokenizers, processors).
        """
        assets = []
        for mod in self.modules_dict.values():
            raw = _unwrap_module(mod)
            if isinstance(raw, OmniModule):
                assets.extend(raw.get_assets())
        return assets

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

    def _get_micro_batch_size(self, module_name: str) -> int:
        """Read ``micro_batch_size`` from module config, default to full batch."""
        cfg = self.config.modules.get(module_name, {})
        return int(cfg.get("micro_batch_size", _DEFAULT_MICRO_BS))

    # ── nn.Module overrides ───────────────────────────────────────────────────

    def parameters(self, recurse: bool = True):
        return self.modules_dict.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self.modules_dict.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)


# ── Module-level batch helpers ────────────────────────────────────────────────


def _unwrap_module(mod: nn.Module) -> nn.Module:
    """Return the underlying OmniModule, unwrapping DDP if necessary.

    ``pre_forward`` / ``post_forward`` must be called on the raw OmniModule,
    not on the DDP wrapper.  The DDP wrapper is still used for the actual
    ``forward()`` call so that gradient all-reduce fires correctly.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    if isinstance(mod, DDP):
        return mod.module
    return mod


def _infer_batch_size(inputs: Dict[str, Any]) -> int:
    """Return the batch dimension (dim 0) from the first tensor found in inputs."""
    for v in inputs.values():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return v.size(0)
    return 1


def _slice_batch(inputs: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Slice all tensors in *inputs* along dim 0 to ``[start:end]``.

    Each slice is cloned so it does not share storage with the original tensor.
    This prevents version-counter conflicts when in-place operations inside a
    module's forward (e.g. LayerNorm, dropout) would otherwise increment the
    version of the original tensor and break backward through earlier chunks.

    Non-tensor values (scalars, strings, None, lists) are passed through as-is.
    """
    sliced: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            sliced[k] = v[start:end].clone()
        else:
            sliced[k] = v
    return sliced


def _aggregate_outputs(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate micro-batch output chunks.

    * Tensors → concatenated along dim 0 (batch dimension).
    * ``*_loss`` scalars → averaged (mean reduction across micro-batches).
    * Other scalars / non-tensors → taken from the last chunk.
    """
    if not chunks:
        return {}
    if len(chunks) == 1:
        return chunks[0]

    keys = chunks[0].keys()
    agg: Dict[str, Any] = {}
    for k in keys:
        vals = [c[k] for c in chunks if k in c]
        if not vals:
            continue
        sample = vals[0]
        if isinstance(sample, torch.Tensor):
            if sample.ndim == 0 or k.endswith("_loss"):
                # Scalar losses: mean
                agg[k] = torch.stack(vals).mean()
            else:
                # Batched tensors: concat on dim 0
                agg[k] = torch.cat(vals, dim=0)
        else:
            agg[k] = sample  # non-tensor: last/first chunk value
    return agg
