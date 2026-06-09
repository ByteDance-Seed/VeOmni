"""SeedOmni graph hooks for BAGEL's latent VAE module."""

from typing import Any, Dict

from ....module import ModuleMixin


class BagelVAEModuleMixin(ModuleMixin):
    def pre_forward(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        assert method in ("encode", "decode", "forward")
        return kwargs

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        assert method in ("encode", "decode", "forward")
        return outputs


__all__ = ["BagelVAEModuleMixin"]
