"""V2 request dispatch from reference kind to driver hook methods."""

from __future__ import annotations

from typing import Any

from tests.seed_omni.parity_suite.core import ParityCase, effective_reference_kind
from tests.seed_omni.parity_suite.core.stimulus import conversation_stimulus_to_batched_specs
from tests.seed_omni.parity_suite.driver.v2_run import V2RunContext
from tests.seed_omni.parity_suite.v2.request import (
    V2RequestContext,
    conversation_request_from_conversation_list,
)


class RequestDispatchMixin:
    """Adapt reference canonical data or recipe stimulus into a V2 request."""

    case: ParityCase

    def build_v2_request(self, ctx: V2RunContext) -> dict[str, Any]:
        """Adapt reference canonical data or recipe stimulus into a V2 request."""
        request_ctx = V2RequestContext(
            case=self.case,
            kind=effective_reference_kind(self.case),
            canonical=ctx.canonical,
            stimulus=self.case.recipe.stimulus,
            reference_output=ctx.reference_output,
            device=ctx.device,
        )

        if conversation_stimulus_to_batched_specs(self.case.recipe.stimulus) is not None:
            return self._v2_request_from_conversation_list(request_ctx)

        method_name = self._v2_request_method_name(effective_reference_kind(self.case))
        handler = getattr(self, method_name, None)
        if handler is not None:
            return handler(request_ctx)

        raise NotImplementedError(
            f"{type(self).__name__} has no method {method_name!r} for reference.kind {effective_reference_kind(self.case)!r}."
        )

    # Driver request hooks --------------------------------------------------------

    def _v2_request_from_conversation_list(self, ctx: V2RequestContext) -> dict[str, Any]:
        return conversation_request_from_conversation_list(ctx)

    # Internal helpers ------------------------------------------------------------

    def _v2_request_method_name(self, kind: str) -> str:
        return f"build_{kind}_request"
