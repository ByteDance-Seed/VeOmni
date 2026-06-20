"""V2 request dispatch from reference kind to driver hook methods."""

from __future__ import annotations

from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityCase, effective_reference_kind
from tests.seed_omni.parity_suite.core.stimulus import conversation_stimulus_to_batched_specs
from tests.seed_omni.parity_suite.reference.contract import ReferenceRunResult
from tests.seed_omni.parity_suite.v2.request import (
    V2RequestContext,
    conversation_request_from_conversation_list,
)


class RequestDispatchMixin:
    """Adapt reference canonical data or recipe stimulus into a V2 request."""

    case: ParityCase

    # Public request entry ---------------------------------------------------------

    def v2_request_kwargs(self, reference_output: Any, *, device: torch.device) -> dict[str, Any]:
        """Adapt reference canonical data or recipe stimulus into a V2 request."""
        canonical = {} if reference_output is None else _canonical_from_reference_run(reference_output)
        ctx = V2RequestContext(
            case=self.case,
            kind=effective_reference_kind(self.case),
            canonical=canonical,
            stimulus=self.case.recipe.stimulus,
            reference_output=reference_output,
            device=device,
        )

        if conversation_stimulus_to_batched_specs(self.case.recipe.stimulus) is not None:
            return self._v2_request_from_conversation_list(ctx)

        method_name = self._v2_request_method_name(effective_reference_kind(self.case))
        handler = getattr(self, method_name, None)
        if handler is not None:
            return handler(ctx)

        raise NotImplementedError(
            f"{type(self).__name__} has no method {method_name!r} for reference.kind {effective_reference_kind(self.case)!r}."
        )

    # Driver request hooks --------------------------------------------------------

    def _v2_request_from_conversation_list(self, ctx: V2RequestContext) -> dict[str, Any]:
        return conversation_request_from_conversation_list(ctx)

    # Internal helpers ------------------------------------------------------------

    def _v2_request_method_name(self, kind: str) -> str:
        return f"build_{kind}_request"


def _canonical_from_reference_run(reference_output: Any) -> dict[str, Any]:
    if not isinstance(reference_output, ReferenceRunResult):
        raise TypeError(
            "V2 request dispatch expects ReferenceRunResult from ReferenceOracle.capture; "
            f"got {type(reference_output).__name__}."
        )
    return dict(reference_output.canonical)
