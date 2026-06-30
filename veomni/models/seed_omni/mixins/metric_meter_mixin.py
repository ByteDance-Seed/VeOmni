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

* :meth:`metric_meter_add` accumulates this module's token lengths — the per-module
  analogue of ``EnvironMeter.add``.  The
  :class:`~veomni.trainer.omni.omni_module_trainer.OmniModuleTrainer` calls it right after
  ``pre_forward`` (when the real input tensors are in hand), passing the node's
  ``method`` + the forward ``data``.  **Each module implements its own**
  :meth:`metric_meter_token_lengths` — there is no generic default, because token
  domains differ and some call-sites shouldn't be counted (e.g. a VQ codec counts
  on ``encode``, returns ``[]`` on ``decode``).
* :meth:`metric_meter_collect` returns ``(theoretical_flops, seqlens)`` — the total
  theoretical TFLOPs for this module's compute over the step plus its raw token
  lengths.  **No timing, no MFU, no cross-rank reduction here.**

MFU / achieved-FLOPs / tokens-per-second are computed once, globally, by the
orchestrator: a per-module wall-clock is meaningless because a module's
``on_step_end`` only fires after the *whole* graph's forward+backward finishes,
so the elapsed time it would see is the whole-step time, not its own. The trainer
therefore times the whole graph once and divides the summed theoretical FLOPs by
that single delta (see :class:`veomni.utils.omni_helper.OmniEnvironMeter`).

A module has exactly **one** notion of sequence length — a token is a token,
whatever its modality. Each module implements its own :meth:`estimate_flops`
(its own FLOPs formula — there is no shared whole-model counter, which would
mis-count at module granularity).

Opt-in is by **multiple inheritance**, NOT by ``ModuleMixin`` (``ModuleMixin``
does *not* inherit ``MetricMeterMixin``). A module that wants metering defines its own
``XxxMetricMeterMixin(MetricMeterMixin)`` implementing ``estimate_flops`` +
``metric_meter_token_lengths``, and its concrete model multi-inherits it, e.g.::

    class TextEncoder(TextEncoderModuleMixin, TextEncoderMetricMeterMixin, PreTrainedModel): ...

The orchestrator decides whether a module contributes metrics with
``isinstance(model, MetricMeterMixin)``. Modules without a metric meter contribute
nothing.
"""

from typing import Any, Dict, List, Tuple


# What a metered module hands back each step: its total theoretical TFLOPs +
# the raw (this-rank) per-sample token lengths it processed.
MetricMeterResult = Tuple[float, List[int]]


class MetricMeterMixin:
    """Optional per-module token + theoretical-FLOPs meter for SeedOmni V2."""

    def _metric_meter_seqlen_buffer(self) -> List[int]:
        # Lazily initialised so an implementing module never has to touch its own
        # ``init_omni_state`` / ``pre_forward``.
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

    def metric_meter_token_lengths(self, method: str, data: Dict[str, Any]) -> List[int]:
        """Per-sample token lengths this module processes for call-site ``method``.

        **Each metered module implements its own** — there is no generic default,
        because token domains differ (text seq len vs image patches vs VQ tokens)
        and some call-sites shouldn't be counted at all.  ``method`` is the graph
        node's call-site (``"forward"`` / ``"encode"`` / ``"decode"`` / …) and
        ``data`` is the post-``pre_forward`` kwargs the module is about to run on
        (the real forward inputs).

        Return ``[]`` to skip a call — e.g. a VQ codec counts its tokens on
        ``encode`` but returns ``[]`` on ``decode``; empty contributions never
        enter the meter.
        """
        raise NotImplementedError(
            f"{type(self).__name__} mixes in MetricMeterMixin but does not implement metric_meter_token_lengths(method, data)."
        )

    def metric_meter_add(self, method: str, data: Dict[str, Any]) -> None:
        """Accumulate this micro-batch's token lengths (per-module ``meter.add``).

        Called once per micro-batch by the module-trainer right after
        ``pre_forward`` (so ``data`` holds the real input tensors), with the
        node's ``method``.  Sums correctly over a whole gradient-accumulation
        step (a call that returns ``[]`` adds nothing).
        """
        self._metric_meter_seqlen_buffer().extend(self.metric_meter_token_lengths(method, data))

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
