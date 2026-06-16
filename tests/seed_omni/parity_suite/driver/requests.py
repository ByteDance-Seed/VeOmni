"""V2 request dispatch from reference kind to driver hook methods."""

from __future__ import annotations

from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityCase
from tests.seed_omni.parity_suite.reference.contract import canonical_from_reference_output
from tests.seed_omni.parity_suite.v2.request import V2RequestContext


class RequestDispatchMixin:
    """Adapt reference canonical data or recipe stimulus into a V2 request."""

    case: ParityCase

    def _v2_request_kind(self) -> str:
        reference = self.case.recipe.reference
        kind = reference.get("kind")
        if kind is None:
            raise ValueError(f"Recipe {self.case.recipe.id!r} must declare reference.kind for V2 request dispatch.")
        return str(kind)

    def _v2_request_method_name(self, kind: str) -> str:
        return f"build_{kind}_request"

    def v2_request_kwargs(self, reference_output: Any, *, device: torch.device) -> dict[str, Any]:
        """Adapt reference canonical data or recipe stimulus into a V2 request."""

        kind = self._v2_request_kind()
        method_name = self._v2_request_method_name(kind)
        handler = getattr(self, method_name, None)
        if handler is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no method {method_name!r} for reference.kind {kind!r}."
            )
        canonical = canonical_from_reference_output(reference_output)
        ctx = V2RequestContext(
            case=self.case,
            kind=kind,
            canonical=canonical,
            stimulus=self.case.recipe.stimulus,
            reference_output=reference_output,
            device=device,
        )
        return handler(ctx)
