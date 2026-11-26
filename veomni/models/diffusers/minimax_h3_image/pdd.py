"""PDD release metadata shared by H3-Image inference and training."""

import json
from pathlib import Path

from safetensors.torch import load_file


DEFAULT_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)


def validate_pdd_sigmas(sigmas, num_pdd_heads: int | None = None) -> tuple[float, ...]:
    """Validate the schedule, including the extra terminal sigma."""
    sigmas = tuple(sigmas)
    if num_pdd_heads is not None and len(sigmas) - 1 != num_pdd_heads:
        raise ValueError(
            f"Sigma schedule has {len(sigmas) - 1} transitions but the checkpoint has {num_pdd_heads} PDD heads."
        )
    if (
        len(sigmas) < 2
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or not all(end < start for start, end in zip(sigmas, sigmas[1:]))
    ):
        raise ValueError("PDD sigmas must strictly decrease from 1 to 0.")
    return sigmas


def load_pdd_release(model_dir: str | Path, *, require_sigmas: bool = False):
    """Read fused heads and their schedule, optionally requiring explicit release metadata."""
    model_dir = Path(model_dir)
    state = load_file(model_dir / "pdd_fused_heads.safetensors", device="cpu")
    if state.keys() != {"video_out.weight", "video_out.bias"}:
        raise ValueError("Expected only video_out.weight and video_out.bias in pdd_fused_heads.safetensors.")
    weight, bias = state["video_out.weight"], state["video_out.bias"]
    if weight.ndim != 3 or bias.shape != weight.shape[:2]:
        raise ValueError("PDD video_out.weight and video_out.bias must have matching head and output dimensions.")
    pdd_config_path = model_dir / "pdd_config.json"
    sigmas = DEFAULT_SIGMAS
    if require_sigmas or pdd_config_path.is_file():
        pdd_config = json.loads(pdd_config_path.read_text())
        if require_sigmas and "sigmas" not in pdd_config:
            raise ValueError(f"{pdd_config_path} must define 'sigmas' to prepare a training checkpoint.")
        sigmas = pdd_config.get("sigmas", DEFAULT_SIGMAS)
    return state, validate_pdd_sigmas(sigmas, weight.shape[0])
