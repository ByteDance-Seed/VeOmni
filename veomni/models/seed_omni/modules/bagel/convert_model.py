"""Split a BAGEL checkpoint into SeedOmni V2 module subfolders."""

from __future__ import annotations

from veomni.models.seed_omni.convert_registry import OMNI_CONVERT_REGISTRY


def convert_bagel_checkpoint(
    model_path: str,
    output_dir: str,
    *,
    force: bool = False,
    max_latent_size: int = 64,
    **kwargs,
) -> None:
    """Split an upstream BAGEL checkpoint into five V2 module subfolders."""
    del kwargs
    from scripts.multimodal.convert_model.split_bagel import split_bagel

    split_bagel(
        model_path=model_path,
        output_dir=output_dir,
        force=force,
        max_latent_size=max_latent_size,
    )


@OMNI_CONVERT_REGISTRY.register("bagel")
def _register_bagel_convert():
    return convert_bagel_checkpoint


__all__ = ["convert_bagel_checkpoint"]
