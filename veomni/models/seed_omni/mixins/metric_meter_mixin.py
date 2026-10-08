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

"""MetricMeterMixin — optional per-module training meter (tokens + theoretical FLOPs).

Why per-module
--------------
A single-model trainer counts tokens and estimates FLOPs from one ``model_type``
(see :class:`veomni.utils.helper.EnvironMeter`).  ``OmniModel`` is instead a
*composition* of independent sub-modules with no single config to dispatch a
FLOPs formula on — so each module must count its **own** tokens and estimate its
**own** theoretical FLOPs (with its own config), and the orchestrator rolls them
up into the overall throughput / MFU.

What a module computes vs. what the trainer computes
----------------------------------------------------
A module only ever produces **time-independent** quantities:

* A module reports its tokens by calling :meth:`metric_meter_set_seqlens` inside
  its ``pre_forward`` **before any SP slice**, so metering is identical with or
  without SP (measuring the post-``pre_forward`` shard would under-count by
  ~``sp``). Token domains differ — text seq len vs image patches vs VQ tokens —
  and some call-sites aren't counted, e.g. a VQ codec counts on ``encode`` and
  stashes nothing on ``decode``.
* :meth:`metric_meter_add` moves the lengths stashed for the running call-site
  into the step buffer — the per-module analogue of ``EnvironMeter.add``. The
  training node executor calls it right after ``pre_forward`` with the node's
  ``method``.
* :meth:`metric_meter_collect` returns ``(theoretical_flops, seqlens)`` — the total
  theoretical TFLOPs for this module's compute over the step plus its raw token
  lengths.  **No timing, no MFU, no cross-rank reduction here.**

MFU / achieved-FLOPs / tokens-per-second are computed once, globally, by the
orchestrator: a per-module wall-clock is meaningless because modules share one
backward and their compute interleaves within the step. The trainer therefore
times the whole graph once and divides the summed theoretical FLOPs by that
single delta (see :class:`veomni.utils.omni_helper.OmniEnvironMeter`).

A module has exactly **one** notion of sequence length — a token is a token,
whatever its modality. Each module implements its own :meth:`estimate_flops`
(its own FLOPs formula — there is no shared whole-model counter, which would
mis-count at module granularity).

Opt-in is by **multiple inheritance**, NOT by ``BaseMixin`` (``BaseMixin``
does *not* inherit ``MetricMeterMixin``). A module that wants metering defines its own
``XxxMetricMeterMixin(MetricMeterMixin)`` implementing just ``estimate_flops`` (token
lengths come from :meth:`metric_meter_set_seqlens`), and its concrete model
multi-inherits it, e.g.::

    class XxxModel(XxxMetricMeterMixin, TrainingModuleMixin, BaseMixin, PreTrainedModel): ...

:meth:`OmniModelRuntime.metric_meter_collect` decides whether a module contributes
metrics with ``isinstance(model, MetricMeterMixin)``. Modules without a metric meter contribute
nothing.
"""

from typing import Dict, List, Tuple


# What a metered module hands back each step: its total theoretical TFLOPs +
# the raw (this-rank) per-sample token lengths it processed.
MetricMeterResult = Tuple[float, List[int]]


class MetricMeterMixin:
    """Optional per-module token + theoretical-FLOPs meter for SeedOmni modules."""

    def metric_meter_set_seqlens(self, method: str, seqlens: List[int]) -> None:
        """Stash the FULL (pre-SP-slice) per-sample token lengths for call-site ``method``.

        **Call this inside a ``pre_forward`` hook, BEFORE any SP gather/slice.**
        Only modules that mix in ``MetricMeterMixin`` should call this;
        :meth:`metric_meter_add` drains the stash right after ``pre_forward``.

        Why pre-slice / full-sample: under uniform SP the ``pre_forward`` hook
        slices to this rank's ``1/sp_size`` shard, so a length read *after* the
        slice would under-count by ~``sp``. Stash the FULL (pre-slice) per-sample
        lengths here — before the slice branch — instead: ``OmniEnvironMeter`` sums
        tokens+FLOPs over the ``dp_group`` (which excludes the SP ranks that all
        hold the same replicated sample), so the full-sample value counted once per
        DP shard reconstructs the true global total, matching the non-SP run.
        """
        if not hasattr(self, "_metric_full_seqlens"):
            self._metric_full_seqlens: Dict[str, List[int]] = {}
        self._metric_full_seqlens[method] = [int(s) for s in seqlens]

    def _metric_meter_seqlen_buffer(self) -> List[int]:
        # Lazily initialised so an implementing module never has to touch its own
        # ``__init__`` / ``pre_forward``.
        if not hasattr(self, "_metric_meter_seqlens"):
            self._metric_meter_seqlens: List[int] = []
        return self._metric_meter_seqlens

    def estimate_flops(self, seqlens: List[int]) -> float:
        """Total theoretical TFLOPs for this module's compute over ``seqlens``.

        **Each metered module implements its own** — :class:`VeomniFlopsCounter`
        is a *whole-model* estimator and is wrong at module granularity (e.g. an
        AR backbone owns no ``wte`` / ``lm_head`` — those FLOPs belong to the
        ``text_encoder`` module).  Return the total FLOPs in TFLOPs (forward +
        backward), **time-independent**; the orchestrator divides the summed
        FLOPs across modules + ranks by the single whole-graph wall-clock to get
        achieved FLOPs / MFU.
        """
        raise NotImplementedError(
            f"{type(self).__name__} mixes in MetricMeterMixin but does not implement estimate_flops(seqlens)."
        )

    def metric_meter_add(self, method: str) -> None:
        """Move the lengths stashed for call-site ``method`` into this step's buffer.

        Called once per training node by the executor right after ``pre_forward``,
        so it sums over a whole gradient-accumulation step. The stash is keyed by
        ``method`` because one ``pre_forward`` hook may serve several call-sites
        (``@pre_forward("encode", "offline_encode")``) without knowing which one
        invoked it: such a hook stashes under each, and only the running call-site
        is counted. A call-site that stashed nothing (e.g. ``decode``) adds nothing.
        """
        stash = getattr(self, "_metric_full_seqlens", None)
        if stash:
            self._metric_meter_seqlen_buffer().extend(stash.pop(method, []))

    def metric_meter_collect(self) -> MetricMeterResult:
        """Return ``(theoretical_flops, seqlens)`` for the step, then reset.

        ``theoretical_flops`` is the module's own total theoretical TFLOPs
        (time-independent) via :meth:`estimate_flops`; ``seqlens`` is this rank's
        raw per-sample token lengths.  The orchestrator sums these across modules
        + ranks and divides by the single whole-graph time for achieved FLOPs /
        MFU.
        """
        seqlens = self._metric_meter_seqlen_buffer()
        self._metric_meter_seqlens = []
        return self.estimate_flops(seqlens), seqlens


__all__ = ["MetricMeterMixin", "MetricMeterResult"]
