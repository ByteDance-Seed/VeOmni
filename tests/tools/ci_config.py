"""Explicit CI-sized alternatives for expensive synthetic integration fixtures."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1] / "toy_config"


def ci_config_path(path: str | Path) -> str:
    """Use a checked-in small fixture when available, leaving other configs unchanged."""
    source = Path(path)
    directory = source.parent if source.suffix == ".json" else source
    name = directory.name.removesuffix("_toy")
    name = {"qwen3_5_fsdp_equiv": "qwen3_5_text", "qwen3_5_moe_fsdp_equiv": "qwen3_5_moe_text"}.get(name, name)
    target = _ROOT / "ci" / f"{name}_tiny"
    if directory.parent.resolve() == _ROOT and target.is_dir():
        return str(target / "config.json") if source.suffix == ".json" else str(target)
    return str(path)
