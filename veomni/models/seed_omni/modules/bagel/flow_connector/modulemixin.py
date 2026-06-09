"""SeedOmni graph hooks for BAGEL connector call sites."""

from typing import Any, Dict

from ....module import ModuleMixin


class BagelFlowConnectorModuleMixin(ModuleMixin):
    def pre_forward(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        assert method in ("encode_vision", "embed_latent", "decode_velocity", "forward")
        return kwargs

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        assert method in ("encode_vision", "embed_latent", "decode_velocity", "forward")
        return outputs


__all__ = ["BagelFlowConnectorModuleMixin"]
