# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-entropy loss wrappers and dispatch.

``ForCausalLMLoss`` and ``ForSequenceClassificationLoss`` own the outer policy
(label shifting for causal LM, SP-aware reduction) and delegate the actual
cross-entropy computation to ``cross_entropy_fn`` — a single required-style
keyword argument. The wrapper never decides which inner kernel to use: that
decision is made once, at ``install_loss_mapping`` / ``KERNEL_REGISTRY.resolve``
time, and baked in via ``functools.partial``.

Two dispatch paths reach these wrappers:

1. ``LOSS_MAPPING``: ``install_loss_mapping(impl)`` binds
   ``partial(ForCausalLMLoss, cross_entropy_fn=<impl>)`` into
   ``LOSS_MAPPING["ForCausalLM"]`` etc. Models that call
   ``self.loss_function(...)`` go through this path.

2. ``KERNEL_REGISTRY`` / ``OpSlot``: the registered factories below return the
   same ``partial(...)`` shape, bound to ``veomni_causal_lm_loss`` /
   ``veomni_seq_cls_loss`` at model-build time. Generated modeling code that
   already knows it wants a fused kernel calls the ``OpSlot`` directly.

Contract: ``apply_ops_config(ops_config)`` must run before any model is built,
otherwise ``LOSS_MAPPING`` contains HuggingFace's stock wrapper which does not
understand ``hidden_states=``/``weights=`` kwargs. ``build_foundation_model``
owns this — pass ``ops_implementation=...`` (trainers do this) and it will
install the config; otherwise it falls back to ``OpsImplementationConfig()``
defaults.
"""

from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import reduce_sequence_parallel_loss
from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available, is_torch_npu_available
from .chunk_loss import chunk_loss_function  # noqa: F401 re-export for legacy callers
from .eager import eager_cross_entropy


logger = logging.get_logger(__name__)


def ForCausalLMLoss(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    vocab_size: int = None,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    # `*,` marks everything below as keyword-only. HF calls this wrapper with
    # positional args (logits, labels, vocab_size, ...); keeping `cross_entropy_fn`
    # keyword-only guarantees the pre-bound kernel from `install_loss_mapping` /
    # `KERNEL_REGISTRY` (via `functools.partial`) cannot be silently overwritten
    # by a positional arg overflowing into this slot.
    *,
    cross_entropy_fn: Callable = eager_cross_entropy,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)

    assert hidden_states is not None or logits is not None, "hidden_states or logits must be provided."

    device = logits.device if logits is not None else hidden_states.device
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    if logits is not None:
        logits = logits.float()

    sp_enabled = get_parallel_state().sp_enabled

    # veomni sp patch
    if not sp_enabled:
        # Shift so that tokens < n predict n
        if shift_labels is None:
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
    else:
        if shift_labels is not None:
            logger.warning_once("labels have been shifted in dataloader when `sp_enabeld=True`, ignore shift_labels.")
        shift_labels = labels

    # Flatten the tokens
    shift_labels = shift_labels.view(-1)
    if hidden_states is not None:
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    if logits is not None:
        logits = logits.view(-1, vocab_size)
    # Enable model parallelism
    shift_labels = shift_labels.to(device)

    loss, logits = cross_entropy_fn(
        logits,
        shift_labels,
        vocab_size,
        num_items_in_batch,
        ignore_index,
        hidden_states=hidden_states,
        weights=weights,
        **kwargs,
    )

    # Reduce loss when using sp
    if sp_enabled:
        num_valid_tokens = (labels != ignore_index).sum()
        loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
    return loss, logits


def ForSequenceClassificationLoss(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    num_labels: int = None,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
    # `*,` marks `cross_entropy_fn` keyword-only — same reason as in
    # `ForCausalLMLoss`: the inner kernel is bound once at install time via
    # `partial(..., cross_entropy_fn=...)` and must not be reachable via positional args.
    *,
    cross_entropy_fn: Callable = eager_cross_entropy,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Token-level loss for sequence classification.

    This loss follows the "token-level labels" convention:
    `labels` has the same layout as the token sequence,
    with all positions set to `ignore_index` except the supervised tokens (the last valid token of each sample).
    No shifting is applied.
    When SP is enabled, the loss is reduced across SP ranks using the number of non-ignored tokens.

    Args:
        logits (`torch.Tensor`):
            Classification logits.
        labels (`torch.Tensor`):
            Token-level labels with `ignore_index` marking non-supervised positions.
        num_labels (`int`):
            Number of classes.
        num_items_in_batch (`int`):
            Used to accurately calculate the average loss for each sample.
        ignore_index (`int`, defaults to `-100`):
            Label value to ignore when computing the loss.
        cross_entropy_fn (`Callable`):
            Inner CE kernel, pre-bound by ``install_loss_mapping`` /
            ``KERNEL_REGISTRY``. Defaults to eager for direct in-process calls
            (e.g. tests); production dispatch always provides an explicit value.
        hidden_states (`torch.Tensor`):
            Hidden states, used for fused linear cross-entropy.
        weights (`torch.Tensor`):
            Classification head weights, used for fused linear cross-entropy.

    Returns:
        loss (`torch.Tensor`):
            Scalar classification loss.
        logits (`torch.Tensor`):
            Flattened logits.
    """

    # pop fused loss kwargs
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)

    if hidden_states is None and logits is None:
        raise ValueError("Either hidden_states or logits must be provided.")

    if labels is None:
        raise ValueError("labels must be provided for sequence classification loss.")

    if num_labels is None:
        raise ValueError("num_labels must be provided.")

    device = logits.device if logits is not None else hidden_states.device
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    if logits is not None:
        logits = logits.float()

    sp_enabled = get_parallel_state().sp_enabled
    target = labels

    # Flatten the tokens
    target = target.view(-1)
    if hidden_states is not None:
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    if logits is not None:
        logits = logits.view(-1, num_labels)
    # Enable model parallelism
    target = target.to(device)

    loss, logits = cross_entropy_fn(
        logits,
        target,
        num_labels,
        num_items_in_batch,
        ignore_index,
        hidden_states=hidden_states,
        weights=weights,
        **kwargs,
    )

    # Reduce loss when using sp
    if sp_enabled:
        num_valid_tokens = (target != ignore_index).sum()
        loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
    return loss, logits


# ── LOSS_MAPPING installation ────────────────────────────────────────────────


def _resolve_cross_entropy_fn(impl: str) -> Callable:
    """Return the inner CE kernel callable for ``impl`` (one of
    ``"eager"`` / ``"liger_kernel"``). The NPU path does not go through this
    helper — see ``install_loss_mapping``."""
    if impl == "eager":
        return eager_cross_entropy
    if impl == "liger_kernel":
        if not is_liger_kernel_available():
            raise RuntimeError(
                "cross_entropy_loss_implementation='liger_kernel' but liger-kernel "
                "is not installed. Install liger-kernel, or set the field to 'eager'."
            )
        from .liger import fused_liger_kernel_cross_entropy

        return fused_liger_kernel_cross_entropy
    raise ValueError(
        f"Unknown cross_entropy_loss_implementation: {impl!r}. Valid options: 'eager', 'liger_kernel', 'npu'."
    )


def install_loss_mapping(impl: str = "eager") -> str:
    """Install VeOmni's loss wrappers into HuggingFace's ``LOSS_MAPPING``,
    pre-bound to the cross-entropy kernel selected by *impl*.

    This is the single entry point for loss dispatch and is called by
    ``apply_ops_config``, which in turn is invoked from
    ``build_foundation_model`` before the model is constructed (so VeOmni
    modeling code that calls ``self.loss_function(hidden_states=...,
    logits=None, ...)`` finds the wrapper installed and not HF's stock
    ``ForCausalLMLoss``).

    Contract — return type: **VeOmni's wrappers return ``(loss, logits)``**,
    not a bare ``torch.Tensor``. The tuple is load-bearing: fused kernels
    (Liger fused linear+CE, NPU ``chunk_loss_function``) fold the
    ``lm_head`` projection into the loss, so the kernel — not the caller —
    is where logits come out. Every VeOmni-patched v5 modeling file in-tree
    unpacks as ``loss, _ = self.loss_function(...)`` (or
    ``loss, logits = ...`` when the caller needs the fused logits).

    This diverges from upstream ``transformers.loss.loss_utils.ForCausalLMLoss``
    which returns a bare ``Tensor``. Mixing ``install_loss_mapping`` with
    an unpatched HF model's ``forward`` (which still does ``loss =
    self.loss_function(...)``) is therefore unsupported — you're expected
    to run through ``BaseTrainer`` so every model in the process is patched
    coherently. See ``docs/design/kernel_selection.md`` ("BaseTrainer
    contract" and the v4/v5 impact table) for the full contract.

    Returns the human-readable label (e.g. ``"CrossEntropy (liger_kernel)"``)
    for logging.
    """
    from transformers.loss.loss_utils import LOSS_MAPPING

    if impl == "npu":
        if not is_torch_npu_available():
            raise RuntimeError("cross_entropy_loss_implementation='npu' requires torch_npu to be installed.")
        # NPU chunk-loss is a standalone LOSS_MAPPING entry with its own
        # chunked autograd function; it handles hidden_states/weights directly
        # and now also applies the SP reduction internally (see chunk_loss.py),
        # so both ForCausalLM and ForConditionalGeneration can route through
        # it safely. ForSequenceClassification stays on the eager wrapper:
        # chunk_loss hard-codes the causal ``labels[..., 1:]`` shift, which is
        # incompatible with the token-level (no-shift) labels that
        # ``ForSequenceClassificationLoss`` expects.
        #
        # TODO(unify): chunk_loss_function still breaks the
        # ``partial(ForCausalLMLoss, cross_entropy_fn=...)`` pattern because it
        # (a) drives the outer chunk loop via a custom autograd.Function,
        # (b) operates on ``hidden_states`` not pre-flattened logits, and
        # (c) does its own label shifting. Unifying it as a standard
        # ``cross_entropy_fn`` would require letting the wrapper skip its
        # shift/flatten path when the inner kernel advertises ``owns_chunking=True``.
        LOSS_MAPPING["ForCausalLM"] = chunk_loss_function
        LOSS_MAPPING["ForConditionalGeneration"] = chunk_loss_function
        LOSS_MAPPING["ForSequenceClassification"] = partial(
            ForSequenceClassificationLoss, cross_entropy_fn=eager_cross_entropy
        )
        logger.warning_rank0(
            "cross_entropy_loss_implementation='npu' routes ForCausalLM and "
            "ForConditionalGeneration through chunk_loss; ForSequenceClassification "
            "falls back to the eager wrapper because chunk_loss hard-codes the causal "
            "label shift."
        )
        return "CrossEntropy (npu chunk_loss)"

    ce_fn = _resolve_cross_entropy_fn(impl)
    LOSS_MAPPING["ForCausalLM"] = partial(ForCausalLMLoss, cross_entropy_fn=ce_fn)
    LOSS_MAPPING["ForConditionalGeneration"] = partial(ForCausalLMLoss, cross_entropy_fn=ce_fn)
    LOSS_MAPPING["ForSequenceClassification"] = partial(ForSequenceClassificationLoss, cross_entropy_fn=ce_fn)
    return f"CrossEntropy ({impl})"


# ── OpSlot kernel registration ───────────────────────────────────────────────

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


def _liger_fused_ce_causal_factory():
    """ForCausalLMLoss bound to the Liger fused CE kernel.

    Used for causal-LM heads (label shifting + SP reduction).
    """
    from .liger import fused_liger_kernel_cross_entropy

    return partial(ForCausalLMLoss, cross_entropy_fn=fused_liger_kernel_cross_entropy)


def _liger_fused_ce_seq_cls_factory():
    """ForSequenceClassificationLoss bound to the Liger fused CE kernel.

    Used for sequence-classification heads (no label shifting; token-level labels).
    """
    from .liger import fused_liger_kernel_cross_entropy

    return partial(ForSequenceClassificationLoss, cross_entropy_fn=fused_liger_kernel_cross_entropy)


KERNEL_REGISTRY.register(
    KernelSpec(
        name="liger_kernel",
        op_name="cross_entropy_loss",
        variant="causal",
        factory=_liger_fused_ce_causal_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="Liger fused linear cross-entropy loss for causal LM (shifts labels, SP reduction)",
    )
)

KERNEL_REGISTRY.register(
    KernelSpec(
        name="liger_kernel",
        op_name="cross_entropy_loss",
        variant="seq_cls",
        factory=_liger_fused_ce_seq_cls_factory,
        hardware=HardwareRequirement(device_type="gpu"),
        description="Liger fused linear cross-entropy loss for sequence classification (no shift)",
    )
)


def _npu_chunk_loss_causal_factory():
    """NPU chunked cross-entropy for causal LM.

    Unlike the Liger factory above, ``chunk_loss_function`` is itself the
    full loss wrapper: it drives its own chunked autograd ``Function``,
    does its own label shift, and projects ``hidden_states`` through
    ``weights`` internally. No ``partial(ForCausalLMLoss, ...)`` wrapping
    is needed — returning it bare matches how ``install_loss_mapping("npu")``
    installs it into ``LOSS_MAPPING["ForCausalLM"]``.
    """
    from .chunk_loss import chunk_loss_function

    return chunk_loss_function


KERNEL_REGISTRY.register(
    KernelSpec(
        name="npu",
        op_name="cross_entropy_loss",
        variant="causal",
        factory=_npu_chunk_loss_causal_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="NPU chunked cross-entropy loss for causal LM (no SP reduction)",
    )
)
# No ``seq_cls`` variant: ``chunk_loss_function`` hard-codes causal label
# shifting. Sequence-classification heads on NPU stay on eager via the
# LOSS_MAPPING branch in ``install_loss_mapping("npu")``.
