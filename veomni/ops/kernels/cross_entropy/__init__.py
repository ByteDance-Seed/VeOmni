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
keyword argument. The wrapper decides between two paths per call: the loss
path (delegated to the bound ``cross_entropy_fn``) or — when the caller
forwards ``return_log_probs=True`` through ``forward`` kwargs — the per-token
log-probs + entropy path (delegated to ``chunk_logprobs_function``). The
choice of which loss kernel to use is still made once, at
``install_loss_mapping`` / ``KERNEL_REGISTRY.resolve`` time, and baked in via
``functools.partial``.

Wrapper return shape: ``(loss, logits, log_probs, entropy)``. The third and
fourth slots are non-``None`` only on the log-probs path; on the loss path
they are ``None`` so every model `forward` can do
``loss, _, log_probs, entropy = self.loss_function(...)`` followed by
``outputs.log_probs = log_probs; outputs.entropy = entropy`` without a
per-model branch on ``return_log_probs``.

Two dispatch paths reach these wrappers:

1. ``LOSS_MAPPING``: ``install_loss_mapping(impl)`` binds
   ``partial(ForCausalLMLoss, cross_entropy_fn=<impl>)`` (or the
   ``_chunk_loss_dispatch`` shim for the chunk_loss/npu impls) into
   ``LOSS_MAPPING["ForCausalLM"]`` etc. Models that call
   ``self.loss_function(...)`` go through this path.

2. ``KERNEL_REGISTRY`` / ``OpSlot``: the registered factories below return the
   same callable shape, bound to ``veomni_causal_lm_loss`` /
   ``veomni_seq_cls_loss`` at model-build time. Generated modeling code that
   already knows it wants a fused kernel calls the ``OpSlot`` directly.

Contract: ``apply_ops_config(ops_config)`` must run before any model is built,
otherwise ``LOSS_MAPPING`` contains HF's stock wrapper which doesn't understand
``hidden_states=``/``weights=`` kwargs. ``build_foundation_model`` owns this:
pass ``ops_implementation=...`` (trainers do) and it installs the config;
otherwise it falls back to ``OpsImplementationConfig.all_eager()`` so
standalone scripts work on any accelerator without requiring liger / triton.
"""

from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import reduce_sequence_parallel_loss
from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available, is_torch_npu_available
from .chunk_logprobs import chunk_logprobs_function  # noqa: F401 re-export
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
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)
    # Per-call log-probs dispatch: when the caller passes `return_log_probs=True`
    # (the model `forward` simply forwards it through `**kwargs`), skip the loss
    # path and route hidden_states+weights+labels through the chunked log-probs
    # kernel. The loss/logits slots are vacated; the third and fourth tuple
    # slots carry the per-token log-probabilities and softmax entropy back to
    # the caller.
    return_log_probs = kwargs.pop("return_log_probs", False)
    # ``temperature`` is part of the PPO actor contract — verl divides logits
    # by it before log_softmax. Pop here so it does not propagate into the
    # plain loss path (where it would trip the inner CE kernel) and so the
    # ``return_log_probs`` branch can forward it explicitly to the kernel.
    temperature = kwargs.pop("temperature", 1.0)

    assert hidden_states is not None or logits is not None, "hidden_states or logits must be provided."

    if return_log_probs:
        # chunk_logprobs handles SP, label-shift, and ignore_index masking
        # internally; it needs the *unflattened* hidden_states to apply the
        # causal shift along the seq dim, so call it before the flatten block.
        if hidden_states is None:
            raise ValueError("return_log_probs=True requires hidden_states (fused-linear path).")
        if weights is None:
            raise ValueError("return_log_probs=True requires weights (lm_head weight).")
        log_probs, entropy = chunk_logprobs_function(
            hidden_states,
            weights,
            labels,
            ignore_index=ignore_index,
            shift_labels=shift_labels,
            temperature=temperature,
        )
        return None, None, log_probs, entropy

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
    return loss, logits, None, None


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
) -> tuple[torch.Tensor, torch.Tensor | None, None, None]:
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
        log_probs (`None`):
            Always ``None`` for sequence classification — the third tuple slot
            exists to keep the wrapper return shape uniform with
            ``ForCausalLMLoss`` so model `forward` call sites can unpack
            ``(loss, logits, log_probs, entropy)`` regardless of head type.
        entropy (`None`):
            Always ``None`` for sequence classification — fourth slot exists
            for the same uniform-unpack reason as ``log_probs``.
    """

    # pop fused loss kwargs
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)
    # Seq-cls heads have no log-probs path. Pop these kwargs defensively so
    # that a caller that always forwards them doesn't trip the inner
    # ``cross_entropy_fn`` with unexpected kwargs.
    kwargs.pop("return_log_probs", None)
    kwargs.pop("temperature", None)

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
    return loss, logits, None, None


# ── LOSS_MAPPING installation ────────────────────────────────────────────────


def _resolve_cross_entropy_fn(impl: str) -> Callable:
    """Return the inner CE kernel callable for ``impl`` (one of
    ``"eager"`` / ``"liger_kernel"``). The ``chunk_loss`` and legacy ``npu``
    paths do not go through this helper — see ``install_loss_mapping``."""
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
        f"Unknown cross_entropy_loss_implementation: {impl!r}. "
        "Valid options: 'eager', 'liger_kernel', 'chunk_loss', 'npu'."
    )


def _chunk_loss_dispatch(
    *args, **kwargs
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """4-tuple shim for the ``chunk_loss`` LOSS_MAPPING entry.

    ``chunk_loss_function`` historically bypasses the ``ForCausalLMLoss``
    wrapper (it owns its own label shift + SP reduction) and is installed
    bare. To keep the wrapper return contract uniform — every model
    forward unpacks ``loss, _, log_probs, entropy = self.loss_function(...)``
    — we wrap it in this shim:

    - ``return_log_probs=True``: route ``hidden_states`` + ``weights`` +
      ``labels`` through ``chunk_logprobs_function``; return
      ``(None, None, log_probs, entropy)``. ``temperature`` (defaults to
      1.0) is forwarded so the PPO actor path can scale logits inside
      the kernel.
    - otherwise: forward to ``chunk_loss_function`` and append two
      ``None`` slots (log_probs, entropy) to its 2-tuple return. Pop
      ``temperature`` defensively so a caller that always forwards it
      doesn't trip ``chunk_loss_function``'s signature.

    Args/kwargs are forwarded as-is — model `forward` only ever calls this
    via keyword (``self.loss_function(logits=..., labels=..., hidden_states=...,
    weights=..., **kwargs)``), so a single ``**kwargs`` parameter is enough.
    """
    return_log_probs = kwargs.pop("return_log_probs", False)
    temperature = kwargs.pop("temperature", 1.0)

    if return_log_probs:
        hidden_states = kwargs.get("hidden_states")
        weights = kwargs.get("weights")
        labels = kwargs.get("labels")
        if hidden_states is None or weights is None:
            raise ValueError("return_log_probs=True requires hidden_states and weights (fused-linear path).")
        log_probs, entropy = chunk_logprobs_function(
            hidden_states,
            weights,
            labels,
            ignore_index=kwargs.get("ignore_index", -100),
            shift_labels=kwargs.get("shift_labels"),
            temperature=temperature,
        )
        return None, None, log_probs, entropy

    loss, logits_out = chunk_loss_function(*args, **kwargs)
    return loss, logits_out, None, None


def install_loss_mapping(impl: str = "eager") -> str:
    """Install VeOmni's loss wrappers into HuggingFace's ``LOSS_MAPPING``,
    pre-bound to the cross-entropy kernel selected by *impl*.

    This is the single entry point for loss dispatch and is called by
    ``apply_ops_config``, which in turn is invoked from
    ``build_foundation_model`` before the model is constructed (so VeOmni
    modeling code that calls ``self.loss_function(hidden_states=...,
    logits=None, ...)`` finds the wrapper installed and not HF's stock
    ``ForCausalLMLoss``).

    Contract — return type: **VeOmni's wrappers return ``(loss, logits,
    log_probs, entropy)``**, not a bare ``torch.Tensor``. The 4-tuple is
    load-bearing: fused kernels (Liger fused linear+CE, NPU
    ``chunk_loss_function``) fold the ``lm_head`` projection into the loss,
    so the kernel — not the caller — is where logits come out. The third
    and fourth slots, ``log_probs`` and ``entropy``, are non-``None`` only
    on the per-token log-probs path — when the caller passes
    ``return_log_probs=True`` (and optionally ``temperature=...``) through
    ``forward`` kwargs, the wrapper short-circuits to
    ``chunk_logprobs_function`` and returns
    ``(None, None, log_probs, entropy)``. Every VeOmni-patched v5 modeling
    file in-tree unpacks as
    ``loss, _, log_probs, entropy = self.loss_function(...)`` (or
    ``loss, logits, log_probs, entropy = ...`` when the caller needs the
    fused logits). The model `forward` then assigns
    ``outputs.log_probs = log_probs; outputs.entropy = entropy`` so
    per-token log-probs and entropy surface on the standard ``ModelOutput``
    without a per-model early-exit branch.

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

    if impl in ("chunk_loss", "npu"):
        # ``chunk_loss`` is a standalone LOSS_MAPPING entry with its own chunked
        # autograd function; it handles ``hidden_states`` / ``weights`` directly
        # and applies the SP reduction internally (see chunk_loss.py), so both
        # ForCausalLM and ForConditionalGeneration can route through it safely.
        # ForSequenceClassification stays on the eager wrapper because chunk_loss
        # hard-codes the causal ``labels[..., 1:]`` shift, which is incompatible
        # with the token-level (no-shift) labels that
        # ``ForSequenceClassificationLoss`` expects.
        #
        # The kernel is hardware-agnostic (pure ``F.linear`` + eager CE) and
        # works on both CUDA and NPU. ``"npu"`` is kept as a back-compat alias
        # for the same kernel — both names install the identical mapping.
        if impl == "npu" and not is_torch_npu_available():
            raise RuntimeError(
                "cross_entropy_loss_implementation='npu' requires torch_npu to be installed; "
                "use 'chunk_loss' for the same kernel without the NPU gate."
            )
        LOSS_MAPPING["ForCausalLM"] = _chunk_loss_dispatch
        LOSS_MAPPING["ForConditionalGeneration"] = _chunk_loss_dispatch
        LOSS_MAPPING["ForSequenceClassification"] = partial(
            ForSequenceClassificationLoss, cross_entropy_fn=eager_cross_entropy
        )
        return f"CrossEntropy ({impl})"

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


def _chunk_loss_causal_factory():
    """Hardware-agnostic chunked cross-entropy for causal LM.

    Unlike the Liger factory above, ``chunk_loss_function`` is itself the
    full loss wrapper: it drives its own chunked autograd ``Function``,
    does its own label shift, and projects ``hidden_states`` through
    ``weights`` internally. The ``_chunk_loss_dispatch`` shim wraps it to
    return the 3-tuple ``(loss, logits, log_probs)`` that the unified
    wrapper contract requires, and to route ``return_log_probs=True``
    through ``chunk_logprobs_function``. This matches how
    ``install_loss_mapping("chunk_loss")`` populates ``LOSS_MAPPING``.
    """
    return _chunk_loss_dispatch


def _chunk_loss_seq_cls_factory():
    """Sequence-classification fallback for ``chunk_loss``.

    ``chunk_loss_function`` hard-codes the causal ``labels[..., 1:]`` shift,
    so it cannot back token-level seq-cls labels. Register the eager seq-cls
    wrapper under the ``chunk_loss`` name so OpSlot binding (which iterates
    every slot in the modeling module, including ``veomni_seq_cls_loss``)
    succeeds — the user picked ``chunk_loss`` for the causal path; seq-cls
    transparently uses eager CE.
    """
    return partial(ForSequenceClassificationLoss, cross_entropy_fn=eager_cross_entropy)


# Register under both the canonical name ("chunk_loss") and the legacy alias
# ("npu") so KERNEL_REGISTRY.resolve succeeds for either spelling. The kernel
# itself is hardware-agnostic; the device_type="any" gate just confirms a real
# accelerator is present.
for _name, _hw in (
    ("chunk_loss", HardwareRequirement(device_type="any")),
    ("npu", HardwareRequirement(device_type="npu")),
):
    KERNEL_REGISTRY.register(
        KernelSpec(
            name=_name,
            op_name="cross_entropy_loss",
            variant="causal",
            factory=_chunk_loss_causal_factory,
            hardware=_hw,
            description="Chunked cross-entropy loss for causal LM (SP-aware reduction)",
        )
    )
    KERNEL_REGISTRY.register(
        KernelSpec(
            name=_name,
            op_name="cross_entropy_loss",
            variant="seq_cls",
            factory=_chunk_loss_seq_cls_factory,
            hardware=_hw,
            description="Eager seq-cls fallback for chunk_loss (chunk_loss is causal-only)",
        )
    )
