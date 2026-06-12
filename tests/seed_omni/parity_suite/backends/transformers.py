"""Transformers reference backend with model-local implementation discovery."""

from __future__ import annotations

import importlib
from typing import Any

from transformers import AutoModel, AutoProcessor, AutoTokenizer

from tests.seed_omni.parity_suite.backends.base import LoadedBackend, ReferenceBackend


class TransformersBackend(ReferenceBackend):
    """Load a reference model through model-local or standard transformers APIs."""

    def register_local_transformers(self) -> Any | None:
        mode = self.spec.local_transformers
        if mode in {False, "false", "disabled", "none"}:
            return None
        module_name = f"tests.seed_omni.{self.case.model_name}.transformers"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                return None
            raise
        register = getattr(module, "register_for_auto_classes", None)
        if register is not None:
            register()
        return module

    def load(self) -> LoadedBackend:
        self.register_local_transformers()
        if not self.spec.model:
            raise ValueError(f"{self.case.node_id} requires reference_backend.model to load transformers backend.")

        kwargs = {"trust_remote_code": self.spec.trust_remote_code, **self.spec.extra}
        model = AutoModel.from_pretrained(self.spec.model, **kwargs)
        model.to(self.device)
        model.eval()

        tokenizer = _try_from_pretrained(AutoTokenizer, self.spec.model, kwargs)
        processor = _try_from_pretrained(AutoProcessor, self.spec.model, kwargs)
        return LoadedBackend(model=model, tokenizer=tokenizer, processor=processor)


def _try_from_pretrained(auto_cls: Any, model_path: str, kwargs: dict[str, Any]) -> Any | None:
    try:
        return auto_cls.from_pretrained(model_path, **kwargs)
    except Exception:
        return None
